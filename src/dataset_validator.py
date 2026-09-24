"""
NetworkIDS - Dataset Validator

Purpose:
    Validate the final flow-level ML dataset before model training.

Input:
    data/processed/final_ml_dataset.csv

Checks:
    1. Dataset shape and required columns
    2. ML feature count
    3. Label validity and distribution
    4. Missing values
    5. Infinite values
    6. Duplicate rows
    7. Numeric sanity checks
    8. Flow statistics
    9. Protocol distribution
    10. Original CIC attack-label distribution
    11. Unmatched-label rows

Important:
    - This script does NOT modify the dataset.
    - Unmatched rows with label = NaN are retained in the dataset.
    - Unmatched rows are excluded only from supervised training later.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Project configuration
sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

import src.config as config

# Constants
EXPECTED_TOTAL_COLUMNS = 55
EXPECTED_ML_FEATURE_COUNT = 42

EXPECTED_METADATA_COLUMNS = [
    "flow_id",
    "timestamp",
    "src_ip",
    "dst_ip",
    "src_mac",
    "dst_mac",
]

EXPECTED_LABEL_COLUMNS = [
    "label",
    "original_label",
]

# Helper functions
def print_section(title):
    """Print a formatted validation section."""

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def check_required_columns(df):
    """Check required metadata, ML features and label columns."""

    print_section("1. Column Validation")

    required_columns = (
        EXPECTED_METADATA_COLUMNS
        + list(config.ML_FEATURES)
        + EXPECTED_LABEL_COLUMNS
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        print("[ERROR] Missing required columns:")
        for column in missing_columns:
            print(f"  - {column}")

        return False

    print(
        f"[OK] All required columns are present."
    )

    print(
        f"  ML features : {len(config.ML_FEATURES)}"
    )

    print(
        f"  Metadata    : {len(EXPECTED_METADATA_COLUMNS)}"
    )

    print(
        f"  Label cols  : {len(EXPECTED_LABEL_COLUMNS)}"
    )

    print(
        f"  Total cols  : {len(df.columns)}"
    )

    if len(config.ML_FEATURES) != EXPECTED_ML_FEATURE_COUNT:

        print(
            f"[ERROR] Expected "
            f"{EXPECTED_ML_FEATURE_COUNT} ML features, "
            f"found {len(config.ML_FEATURES)}."
        )

        return False

    if len(df.columns) != EXPECTED_TOTAL_COLUMNS:

        print(
            f"[WARNING] Expected "
            f"{EXPECTED_TOTAL_COLUMNS} total columns, "
            f"found {len(df.columns)}."
        )

    else:

        print(
            f"[OK] Dataset contains exactly "
            f"{EXPECTED_TOTAL_COLUMNS} columns."
        )

    return True


def check_dataset_shape(df):
    """Check dataset row count and basic shape."""

    print_section("2. Dataset Shape")

    rows, columns = df.shape

    print(f"Rows    : {rows:,}")
    print(f"Columns : {columns}")

    if rows == 0:

        print(
            "[ERROR] Dataset is empty."
        )

        return False

    print("[OK] Dataset contains data.")

    return True

def check_labels(df):
    """Validate binary labels and report class distribution."""

    print_section("3. Label Validation")

    label_counts = df["label"].value_counts(
        dropna=False
    )

    print("Label distribution:")
    print(label_counts.to_string())

    invalid_labels = df[
        df["label"].notna()
        & ~df["label"].isin([0, 1])
    ]

    if len(invalid_labels) > 0:

        print()
        print(
            f"[ERROR] Found "
            f"{len(invalid_labels):,} rows "
            "with invalid binary labels."
        )

        print(
            invalid_labels[
                ["flow_id", "label"]
            ]
            .head(10)
            .to_string(index=False)
        )

        return False

    print()
    print(
        "[OK] All non-null labels are "
        "0 or 1."
    )

    unmatched = df["label"].isna().sum()

    print(
        f"Unmatched rows (label = NaN): "
        f"{unmatched:,}"
    )

    if unmatched > 0:

        print(
            "[INFO] Unmatched rows are retained "
            "and should be excluded explicitly "
            "before supervised training."
        )

    matched = df["label"].notna().sum()

    if matched > 0:

        benign = (
            df.loc[
                df["label"].notna(),
                "label"
            ]
            == 0
        ).sum()

        attack = (
            df.loc[
                df["label"].notna(),
                "label"
            ]
            == 1
        ).sum()

        print()
        print("Matched binary distribution:")
        print(
            f"  Benign : {benign:,} "
            f"({benign / matched * 100:.2f}%)"
        )

        print(
            f"  Attack : {attack:,} "
            f"({attack / matched * 100:.2f}%)"
        )

    return True


def check_original_labels(df):
    """Report original CIC labels for traceability."""

    print_section("4. Original CIC Label Distribution")

    if "original_label" not in df.columns:

        print(
            "[WARNING] original_label column "
            "not found."
        )

        return True

    counts = df[
        "original_label"
    ].value_counts(
        dropna=False
    )

    print(
        counts.to_string()
    )

    print()
    print(
        "[OK] Original CIC labels retained "
        "for traceability."
    )

    return True


def check_missing_values(df):
    """Report NaN values in ML features."""

    print_section("5. Missing-Value Validation")

    ml_df = df[
        config.ML_FEATURES
    ]

    nan_counts = (
        ml_df
        .isna()
        .sum()
    )

    nan_columns = nan_counts[
        nan_counts > 0
    ]

    if len(nan_columns) == 0:

        print(
            "[OK] No NaN values found "
            "in ML features."
        )

        return True

    print(
        "ML features containing NaN:"
    )

    for column, count in nan_columns.items():

        percentage = (
            count
            / len(df)
            * 100
        )

        print(
            f"  {column:<30} "
            f"{count:>10,} "
            f"({percentage:>6.2f}%)"
        )

    print()
    print(
        "[INFO] NaN values are allowed when "
        "a feature is not applicable or unavailable."
    )

    return True


def check_infinite_values(df):
    """Check for positive and negative infinity."""

    print_section("6. Infinite-Value Validation")

    numeric_ml = (
        df[
            config.ML_FEATURES
        ]
        .select_dtypes(
            include=[np.number]
        )
    )

    positive_inf = np.isposinf(
        numeric_ml.to_numpy(
            dtype=float
        )
    ).sum()

    negative_inf = np.isneginf(
        numeric_ml.to_numpy(
            dtype=float
        )
    ).sum()

    total_inf = (
        positive_inf
        + negative_inf
    )

    print(
        f"Positive infinity : "
        f"{positive_inf:,}"
    )

    print(
        f"Negative infinity : "
        f"{negative_inf:,}"
    )

    if total_inf > 0:

        print(
            f"[ERROR] Found "
            f"{total_inf:,} infinite values."
        )

        return False

    print(
        "[OK] No infinite ML feature values."
    )

    return True

def check_duplicates(df):
    print_section("7. Duplicate Validation")

    compare_cols = [c for c in df.columns if c not in ["flow_id", "timestamp"]]

    duplicate_count = df.duplicated(subset=compare_cols, keep=False).sum()
    duplicate_rows = df.duplicated(subset=compare_cols, keep="first").sum()

    print(f"Rows involved in exact duplicates : {duplicate_count:,}")
    print(f"Additional duplicate rows          : {duplicate_rows:,}")

    if duplicate_rows > 0:
        print("[WARNING] Exact duplicate feature rows exist (same 5-tuple + identical stats).")
    else:
        print("[OK] No exact duplicate rows found.")

    return True

def check_numeric_sanity(df):
    """Check important flow-level numeric values."""

    print_section("8. Numeric Sanity Checks")

    checks_passed = True

    # Flow duration
    duration_negative = (
        df["flow_duration"] < 0
    ).sum()

    print(
        f"Negative flow duration : "
        f"{duration_negative:,}"
    )

    if duration_negative > 0:
        checks_passed = False

    # Packet count
    packet_invalid = (
        df["flow_packet_count"] <= 0
    ).sum()

    print(
        f"Non-positive packet count : "
        f"{packet_invalid:,}"
    )

    if packet_invalid > 0:
        checks_passed = False

    # Byte count
    byte_invalid = (
        df["flow_byte_count"] < 0
    ).sum()

    print(
        f"Negative byte count : "
        f"{byte_invalid:,}"
    )

    if byte_invalid > 0:
        checks_passed = False

    # Forward / reverse packet counts
    direction_packet_invalid = (
        (
            df["forward_packet_count"]
            + df["reverse_packet_count"]
        )
        != df["flow_packet_count"]
    ).sum()

    print(
        "Packet direction count mismatch : "
        f"{direction_packet_invalid:,}"
    )

    if direction_packet_invalid > 0:
        checks_passed = False

    # Forward / reverse byte counts
    direction_byte_invalid = (
        (
            df["forward_byte_count"]
            + df["reverse_byte_count"]
        )
        != df["flow_byte_count"]
    ).sum()

    print(
        "Byte direction count mismatch : "
        f"{direction_byte_invalid:,}"
    )

    if direction_byte_invalid > 0:
        checks_passed = False

    # Port range
    invalid_src_ports = (
        (df["src_port"] < 0)
        | (df["src_port"] > 65535)
    ).sum()

    invalid_dst_ports = (
        (df["dst_port"] < 0)
        | (df["dst_port"] > 65535)
    ).sum()

    print(
        f"Invalid source ports : "
        f"{invalid_src_ports:,}"
    )

    print(
        f"Invalid destination ports : "
        f"{invalid_dst_ports:,}"
    )

    if (
        invalid_src_ports > 0
        or invalid_dst_ports > 0
    ):
        checks_passed = False

    if checks_passed:

        print()
        print(
            "[OK] Numeric sanity checks passed."
        )

    else:

        print()
        print(
            "[ERROR] One or more numeric "
            "sanity checks failed."
        )

    return checks_passed

def check_flow_statistics(df):
    """Display useful flow-level statistics."""

    print_section("9. Flow Statistics")

    columns = [
        "flow_duration",
        "flow_packet_count",
        "flow_byte_count",
        "forward_packet_count",
        "reverse_packet_count",
        "forward_byte_count",
        "reverse_byte_count",
        "byte_rate",
    ]

    available = [
        column
        for column in columns
        if column in df.columns
    ]

    print(
        df[available]
        .describe()
        .to_string()
    )

    return True

def check_protocol_distribution(df):
    """Report TCP / UDP flow distribution."""

    print_section("10. Protocol Distribution")

    if "ip_protocol" not in df.columns:

        print(
            "[WARNING] ip_protocol not found."
        )

        return True

    counts = (
        df["ip_protocol"]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    print(
        counts.to_string()
    )

    print()
    print(
        "Protocol mapping:"
    )
    print(
        "  6  = TCP"
    )
    print(
        "  17 = UDP"
    )

    return True


def check_flow_direction(df):
    """Report flow direction distribution."""

    print_section("11. Flow Direction Distribution")

    if "flow_direction" not in df.columns:

        print(
            "[WARNING] flow_direction not found."
        )

        return True

    counts = (
        df["flow_direction"]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    print(
        counts.to_string()
    )

    print()
    print(
        "Current project definition:"
    )
    print(
        "  0 = no reverse packet observed"
    )
    print(
        "  1 = at least one reverse packet observed"
    )

    return True


def check_attack_labels(df):
    """Report attack-only label distribution."""

    print_section("12. Attack Distribution")

    if "original_label" not in df.columns:

        print(
            "[WARNING] original_label not found."
        )

        return True

    attack_df = df[
        df["label"] == 1
    ]

    print(
        f"Matched attack flows: "
        f"{len(attack_df):,}"
    )

    if attack_df.empty:

        print(
            "[WARNING] No attack flows found."
        )

        return True

    counts = (
        attack_df[
            "original_label"
        ]
        .value_counts(
            dropna=False
        )
    )

    print(
        counts.to_string()
    )

    return True

def check_unmatched_rows(df):
    """Inspect unmatched rows without modifying them."""

    print_section("13. Unmatched Flow Validation")

    unmatched = df[
        df["label"].isna()
    ]

    count = len(unmatched)

    print(
        f"Unmatched flows : {count:,}"
    )

    if count == 0:

        print(
            "[OK] No unmatched flows."
        )

        return True

    print()
    print(
        "Protocol distribution of unmatched flows:"
    )

    print(
        unmatched[
            "ip_protocol"
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
        .to_string()
    )

    print()
    print(
        "First 10 unmatched flows:"
    )

    preview_columns = [
        "flow_id",
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "ip_protocol",
        "flow_packet_count",
        "flow_duration",
    ]

    preview_columns = [
        column
        for column in preview_columns
        if column in unmatched.columns
    ]

    print(
        unmatched[
            preview_columns
        ]
        .head(10)
        .to_string(index=False)
    )

    print()
    print(
        "[INFO] Unmatched flows remain label=NaN."
    )

    print(
        "[INFO] They will be excluded explicitly "
        "before supervised model training."
    )

    return True

# Main validation
def main():

    print("=" * 60)
    print("NetworkIDS - Dataset Validator")
    print("=" * 60)

    input_path = Path(
        config.FINAL_ML_DATASET_FILE
        if hasattr(
            config,
            "FINAL_ML_DATASET_FILE"
        )
        else "data/processed/final_ml_dataset.csv"
    )

    print(
        f"Input: {input_path}"
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Dataset not found: {input_path}"
        )

    print()
    print(
        "Loading dataset..."
    )

    df = pd.read_csv(
        input_path
    )

    print(
        f"Loaded {len(df):,} rows."
    )

    # Run validation checks
    results = []

    results.append(
        check_dataset_shape(df)
    )

    results.append(
        check_required_columns(df)
    )

    results.append(
        check_labels(df)
    )

    results.append(
        check_original_labels(df)
    )

    results.append(
        check_missing_values(df)
    )

    results.append(
        check_infinite_values(df)
    )

    results.append(
        check_duplicates(df)
    )

    results.append(
        check_numeric_sanity(df)
    )

    results.append(
        check_flow_statistics(df)
    )

    results.append(
        check_protocol_distribution(df)
    )

    results.append(
        check_flow_direction(df)
    )

    results.append(
        check_attack_labels(df)
    )

    results.append(
        check_unmatched_rows(df)
    )

    # Final result
    print_section("Validation Summary")

    failed_checks = sum(
        result is False
        for result in results
    )

    total_checks = len(results)

    print(
        f"Checks completed : "
        f"{total_checks}"
    )

    print(
        f"Checks failed    : "
        f"{failed_checks}"
    )

    print()

    if failed_checks == 0:

        print(
            "[PASS] Dataset validation completed "
            "without critical errors."
        )

        print()
        print(
            "The dataset is ready for the next "
            "stage: train/test split and model training."
        )

    else:

        print(
            "[FAIL] Dataset validation found "
            "one or more critical issues."
        )

        print(
            "Review the sections above before "
            "starting model training."
        )


if __name__ == "__main__":
    main()