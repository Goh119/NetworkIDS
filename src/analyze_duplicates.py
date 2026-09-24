"""
analyze_duplicates.py
---------------------
NetworkIDS - Investigate rows with identical ML feature values in
training_ready_dataset.csv.

Question we want to answer:
    Are the rows involved in exact feature duplicates
    (a) real distinct flows that happen to share the same 42 feature
        values, or
    (b) fragments produced by flow_builder mis-splitting, or
    (c) a data bug?

This script does NOT modify the dataset.
It only reports findings.

Usage:
    python src/analyze_duplicates.py
"""

import sys
from pathlib import Path

import pandas as pd


# ============================================================
# Project import
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


# ============================================================
# Files
# ============================================================

TRAINING_FILE = (
    config.PROCESSED_DIR / "training_ready_dataset.csv"
)

DUPLICATE_GROUPS_FILE = (
    config.PROCESSED_DIR / "duplicate_feature_groups.csv"
)


# ============================================================
# Helper
# ============================================================

def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("NetworkIDS - Duplicate Feature Analysis")
    print("=" * 70)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not TRAINING_FILE.exists():

        print()
        print(f"[ERROR] Not found:")
        print(TRAINING_FILE)

        return

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        TRAINING_FILE,
        low_memory=False
    )

    print()
    print(
        f"Loaded: {len(df):,} rows x {len(df.columns)} cols"
    )

    print(
        f"ML features configured: "
        f"{len(config.ML_FEATURES)}"
    )

    # --------------------------------------------------------
    # Validate ML features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in config.ML_FEATURES
        if feature not in df.columns
    ]

    if missing_features:

        print()
        print("[ERROR] Missing ML features:")

        for feature in missing_features:
            print(f"  - {feature}")

        return

    # --------------------------------------------------------
    # Group by ML features
    # --------------------------------------------------------

    print_section(
        "1. Duplicate groups by ML features only"
    )

    feature_groups = df.groupby(
        config.ML_FEATURES,
        dropna=False,
    )

    group_sizes = feature_groups.size()

    n_groups_total = len(group_sizes)

    duplicated_groups = group_sizes[
        group_sizes > 1
    ]

    n_groups_duplicated = len(
        duplicated_groups
    )

    n_rows_in_dup = int(
        duplicated_groups.sum()
    )

    n_rows_extra = (
        n_rows_in_dup
        - n_groups_duplicated
    )

    print(
        f"Total unique feature combinations : "
        f"{n_groups_total:,}"
    )

    print(
        f"Groups with >1 row                 : "
        f"{n_groups_duplicated:,}"
    )

    print(
        f"Total rows involved                : "
        f"{n_rows_in_dup:,}"
    )

    print(
        f"Extra rows beyond one/group        : "
        f"{n_rows_extra:,}"
    )

    print()
    print("Group size distribution:")

    size_counts = (
        group_sizes
        .value_counts()
        .sort_index()
    )

    for size, count in size_counts.head(15).items():

        print(
            f"  size = {size:<4} : "
            f"{count:,} groups"
        )

    # --------------------------------------------------------
    # Extract duplicated rows
    # --------------------------------------------------------

    duplicate_mask = df.duplicated(
        subset=config.ML_FEATURES,
        keep=False
    )

    dup_df = df[
        duplicate_mask
    ].copy()

    print()
    print(
        f"Duplicated rows extracted: "
        f"{len(dup_df):,}"
    )

    # --------------------------------------------------------
    # Label consistency
    # --------------------------------------------------------

    print_section(
        "2. Label consistency within duplicate feature groups"
    )

    label_consistency = (
        dup_df
        .groupby(
            config.ML_FEATURES,
            dropna=False
        )["label"]
        .nunique()
    )

    n_consistent = int(
        (label_consistency == 1).sum()
    )

    n_conflicting = int(
        (label_consistency > 1).sum()
    )

    print(
        f"Groups with one consistent label : "
        f"{n_consistent:,}"
    )

    print(
        f"Groups with conflicting labels    : "
        f"{n_conflicting:,}"
    )

    if n_conflicting > 0:

        print()
        print(
            "[WARN] Some identical feature groups "
            "contain different labels."
        )

        print(
            "These groups require further investigation "
            "before model training."
        )

    # --------------------------------------------------------
    # Metadata comparison
    # --------------------------------------------------------

    print_section(
        "3. Metadata comparison within duplicate groups"
    )

    metadata_cols = [
        column
        for column in config.METADATA_COLUMNS
        if column in dup_df.columns
    ]

    print(
        "Metadata columns checked:"
    )

    for column in metadata_cols:
        print(f"  - {column}")

    meta_unique = (
        dup_df
        .groupby(
            config.ML_FEATURES,
            dropna=False
        )[metadata_cols]
        .nunique(dropna=False)
    )

    # A group is considered metadata-identical only if
    # every available metadata column has exactly one value.
    all_same = (
        meta_unique == 1
    ).all(axis=1)

    n_metadata_identical = int(
        all_same.sum()
    )

    n_metadata_different = int(
        (~all_same).sum()
    )

    print()
    print(
        f"Groups with identical metadata : "
        f"{n_metadata_identical:,}"
    )

    print(
        f"Groups with different metadata : "
        f"{n_metadata_different:,}"
    )

    # --------------------------------------------------------
    # Timestamp analysis
    # --------------------------------------------------------

    print_section(
        "4. Timestamp spread within duplicate groups"
    )

    if "timestamp" in dup_df.columns:

        # Convert timestamp to numeric.
        # The current dataset stores timestamp as Unix time.
        timestamp_numeric = pd.to_numeric(
            dup_df["timestamp"],
            errors="coerce"
        )

        dup_df["_timestamp_numeric"] = (
            timestamp_numeric
        )

        ts_stats = (
            dup_df
            .groupby(
                config.ML_FEATURES,
                dropna=False
            )["_timestamp_numeric"]
            .agg(["min", "max", "count"])
        )

        ts_stats["spread_sec"] = (
            ts_stats["max"]
            - ts_stats["min"]
        )

        print(
            "Timestamp spread (seconds):"
        )

        print(
            ts_stats["spread_sec"]
            .describe()
            .to_string()
        )

        print()

        near_0 = int(
            (
                ts_stats["spread_sec"] < 1
            ).sum()
        )

        near_60 = int(
            (
                (ts_stats["spread_sec"] >= 1)
                &
                (ts_stats["spread_sec"] < 60)
            ).sum()
        )

        near_hour = int(
            (
                (ts_stats["spread_sec"] >= 60)
                &
                (ts_stats["spread_sec"] < 3600)
            ).sum()
        )

        far = int(
            (
                ts_stats["spread_sec"] >= 3600
            ).sum()
        )

        print(
            f"  < 1 second       : {near_0:,}"
        )

        print(
            f"  1 - 60 seconds   : {near_60:,}"
        )

        print(
            f"  1 min - 1 hour   : {near_hour:,}"
        )

        print(
            f"  > 1 hour         : {far:,}"
        )

    # --------------------------------------------------------
    # Sample duplicate groups
    # --------------------------------------------------------

    print_section(
        "5. Largest duplicate feature groups"
    )

    biggest = (
        duplicated_groups
        .sort_values(ascending=False)
        .head(10)
    )

    for i, (key, size) in enumerate(
        biggest.items(),
        start=1
    ):

        print(
            f"\n--- Group {i} "
            f"(size = {size}) ---"
        )

        if not isinstance(key, tuple):
            key = (key,)

        mask = pd.Series(
            True,
            index=df.index
        )

        for column, value in zip(
            config.ML_FEATURES,
            key
        ):

            if pd.isna(value):

                mask = (
                    mask
                    &
                    df[column].isna()
                )

            else:

                mask = (
                    mask
                    &
                    (df[column] == value)
                )

        sub = df[
            mask
        ]

        columns_to_show = [
            "flow_id",
            "timestamp",
            "src_ip",
            "dst_ip",
            "src_mac",
            "dst_mac",
            "src_port",
            "dst_port",
            "flow_packet_count",
            "flow_duration",
            "label",
            "original_label",
        ]

        columns_to_show = [
            column
            for column in columns_to_show
            if column in sub.columns
        ]

        print(
            sub[
                columns_to_show
            ]
            .head(20)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # Save duplicated rows for further inspection
    # --------------------------------------------------------

    dup_df = dup_df.drop(
        columns=["_timestamp_numeric"],
        errors="ignore"
    )

    dup_df.to_csv(
        DUPLICATE_GROUPS_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print_section(
        "6. Summary"
    )

    print(
        f"Duplicate feature groups : "
        f"{n_groups_duplicated:,}"
    )

    print(
        f"Rows involved             : "
        f"{n_rows_in_dup:,}"
    )

    print(
        f"Extra rows                : "
        f"{n_rows_extra:,}"
    )

    print(
        f"Consistent-label groups   : "
        f"{n_consistent:,}"
    )

    print(
        f"Conflicting-label groups  : "
        f"{n_conflicting:,}"
    )

    print(
        f"Metadata-identical groups: "
        f"{n_metadata_identical:,}"
    )

    print(
        f"Metadata-different groups: "
        f"{n_metadata_different:,}"
    )

    print()
    print(
        "Duplicated rows saved to:"
    )

    print(
        DUPLICATE_GROUPS_FILE
    )

    print()
    print("=" * 70)
    print("Duplicate analysis completed.")
    print("No rows were modified or deleted.")
    print("=" * 70)

if __name__ == "__main__":
    main()