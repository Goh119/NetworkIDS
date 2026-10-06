"""
merge_and_split_multiday.py
---------------------------
Merge 5 days of labeled flow datasets and create a leakage-controlled
train/test split using identical 42-feature vectors as groups.

Strategy:
    1. Load all 5 day-level labeled flow CSVs.
    2. Add a 'day' metadata column.
    3. Merge all five days.
    4. Remove rows without a valid binary label or source IP.
    5. Keep source IP as metadata only; it is NOT an ML feature.
    6. Create a grouping key from the exact 42 ML-feature vector.
    7. Analyze duplicate feature groups and mixed-label groups.
    8. Use StratifiedGroupKFold(n_splits=5).
    9. Use the first fold as a fixed approximately 80/20 holdout.
   10. Verify:
          - no exact feature-group overlap
          - no source-IP requirement for grouping
          - no NaN labels
          - class distribution
          - day distribution
   11. Save train/test datasets and a detailed summary.

Important:
    This is NOT 5-fold cross-validation.
    Only the first fold is used as the fixed test set.

The source IP is retained in the output CSV for traceability, but it is
never passed to the ML model.

Usage:
    python src/merge_and_split_multiday.py
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.append(str(Path(__file__).resolve().parents[1]))
import src.config as config


# ============================================================
# Paths / settings
# ============================================================

PROCESSED = config.PROCESSED_DIR

MULTIDAY = PROCESSED / "multiday_packets"

OUTPUT_DIR = PROCESSED / "multiday_split"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
]

RANDOM_STATE = 42
N_SPLITS = 5


# ============================================================
# Helpers
# ============================================================

def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def label_name(value):
    if value == 0:
        return "Benign"
    if value == 1:
        return "Attack"
    return str(value)


def print_label_distribution(df, name):
    print(f"  {name}:")

    counts = (
        df[config.LABEL_COLUMN]
        .value_counts()
        .sort_index()
    )

    for label, count in counts.items():
        pct = 100.0 * count / len(df)

        print(
            f"    {label} ({label_name(label):<6}): "
            f"{count:>10,}  ({pct:5.2f}%)"
        )


def print_day_distribution(train_df, test_df):
    train_days = Counter(train_df["day"])
    test_days = Counter(test_df["day"])

    print("  Day distribution:")
    print(
        f"  {'Day':<12}"
        f"{'Train':>14}"
        f"{'Test':>14}"
        f"{'Total':>14}"
    )

    for day in DAYS:
        train_count = train_days.get(day, 0)
        test_count = test_days.get(day, 0)
        total_count = train_count + test_count

        print(
            f"  {day:<12}"
            f"{train_count:>14,}"
            f"{test_count:>14,}"
            f"{total_count:>14,}"
        )

    return train_days, test_days


def create_feature_group_ids(df):
    """
    Create deterministic group IDs based on the exact 42-feature row.

    All rows with the same 42 ML feature values receive the same group ID.

    pd.util.hash_pandas_object is used because directly converting millions
    of 42-column rows into Python tuples would be unnecessarily memory-heavy.

    A collision check is performed later using the resulting hashes.
    """

    feature_frame = df[config.ML_FEATURES]

    feature_hash = pd.util.hash_pandas_object(
        feature_frame,
        index=False,
    )

    return feature_hash.astype("uint64")


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Merge 5 days + identical-feature-vector grouped split")
    print("=" * 70)

    # --------------------------------------------------------
    # 0. Validate ML feature configuration
    # --------------------------------------------------------
    print_section("[0/7] Validate ML feature configuration")

    print(f"  Number of ML features: {len(config.ML_FEATURES)}")

    print(
        f"  Number of ML features: "
        f"{len(config.ML_FEATURES)}"
    )
    config.validate_ml_features()
    print("  ML feature schema: OK")

    if "src_ip" in config.ML_FEATURES:
        raise RuntimeError(
            "src_ip must NOT be included in config.ML_FEATURES."
        )

    if "dst_ip" in config.ML_FEATURES:
        raise RuntimeError(
            "dst_ip must NOT be included in config.ML_FEATURES."
        )

    if "day" in config.ML_FEATURES:
        raise RuntimeError(
            "day must NOT be included in config.ML_FEATURES."
        )

    print("  42-feature schema: OK")
    print("  src_ip used as ML feature: NO")
    print("  dst_ip used as ML feature: NO")
    print("  day used as ML feature:    NO")

    # --------------------------------------------------------
    # 1. Load all five days
    # --------------------------------------------------------
    print_section("[1/7] Loading 5 days")

    frames = []

    for day in DAYS:

        path = config.MULTIDAY_LABEL_FILES[day]

        if not path.exists():
            print(f"  [WARN] Missing: {path}")
            continue

        df_day = pd.read_csv(
            path,
            low_memory=False,
        )

        df_day["day"] = day

        frames.append(df_day)

        print(
            f"  [{day:<10}] "
            f"{len(df_day):>10,} flows loaded"
        )

    if not frames:
        raise RuntimeError(
            f"No labeled datasets found in: {MULTIDAY}"
        )

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    print(
        f"\n  Total loaded: "
        f"{len(df):,} flows"
    )

    # --------------------------------------------------------
    # 2. Check required columns
    # --------------------------------------------------------
    print_section("[2/7] Validate dataset columns")

    required_columns = set(config.ML_FEATURES)

    required_columns.add(config.LABEL_COLUMN)
    required_columns.add("src_ip")
    required_columns.add("dst_ip")
    required_columns.add("day")

    missing_columns = sorted(
        required_columns - set(df.columns)
    )

    if missing_columns:
        print("  Missing columns:")

        for col in missing_columns:
            print(f"    - {col}")

        raise RuntimeError(
            "Required columns are missing."
        )

    print(
        f"  Required columns present: "
        f"{len(required_columns)}"
    )

    # --------------------------------------------------------
    # 3. Cleanup
    # --------------------------------------------------------
    print_section("[3/7] Cleanup")

    before = len(df)

    df = df.dropna(
        subset=[
            config.LABEL_COLUMN,
            "src_ip",
        ]
    ).copy()

    dropped = before - len(df)

    print(
        f"  Dropped unmatched / missing-src_ip: "
        f"{dropped:,}"
    )

    df[config.LABEL_COLUMN] = (
        df[config.LABEL_COLUMN]
        .astype(int)
    )

    # Verify only binary labels remain.
    unique_labels = sorted(
        df[config.LABEL_COLUMN]
        .dropna()
        .unique()
        .tolist()
    )

    print(
        f"  Remaining: "
        f"{len(df):,}"
    )

    print(
        f"  Labels found: "
        f"{unique_labels}"
    )

    if unique_labels != [0, 1]:
        raise RuntimeError(
            "Expected binary labels {0, 1}."
        )

    # --------------------------------------------------------
    # 4. Overall dataset analysis
    # --------------------------------------------------------
    print_section("[4/7] Dataset analysis")

    print_label_distribution(
        df,
        "All days combined",
    )

    print()

    print(
        f"  Unique source IPs: "
        f"{df['src_ip'].nunique():,}"
    )

    print(
        f"  Number of days: "
        f"{df['day'].nunique():,}"
    )

    # --------------------------------------------------------
    # 5. Create feature groups
    # --------------------------------------------------------
    print_section(
        "[5/7] Identical 42-feature group analysis"
    )

    print(
        "  Creating group ID from the exact "
        "42-feature vector..."
    )

    df["_feature_group_id"] = create_feature_group_ids(df)

    unique_groups = (
        df["_feature_group_id"]
        .nunique()
    )

    print(
        f"  Total flows: "
        f"{len(df):,}"
    )

    print(
        f"  Unique feature groups: "
        f"{unique_groups:,}"
    )

    duplicate_rows = (
        len(df) - unique_groups
    )

    print(
        f"  Additional rows belonging to "
        f"duplicate groups: {duplicate_rows:,}"
    )

    # --------------------------------------------------------
    # Feature-group sizes
    # --------------------------------------------------------
    group_sizes = (
        df["_feature_group_id"]
        .value_counts()
    )

    duplicate_groups = group_sizes[
        group_sizes > 1
    ]

    print(
        f"  Feature groups with >1 row: "
        f"{len(duplicate_groups):,}"
    )

    if len(duplicate_groups) > 0:

        print(
            f"  Largest feature group: "
            f"{int(group_sizes.max()):,} rows"
        )

        print(
            f"  Median duplicated group size: "
            f"{int(duplicate_groups.median()):,}"
        )

    # --------------------------------------------------------
    # Mixed-label feature groups
    # --------------------------------------------------------
    print(
        "\n  Checking feature groups containing "
        "both Benign and Attack..."
    )

    mixed_label_groups = (
        df.groupby("_feature_group_id")[
            config.LABEL_COLUMN
        ]
        .nunique()
    )

    mixed_label_groups = mixed_label_groups[
        mixed_label_groups > 1
    ]

    print(
        f"  Mixed-label feature groups: "
        f"{len(mixed_label_groups):,}"
    )

    if len(mixed_label_groups) > 0:

        print(
            "  NOTE: These groups contain identical "
            "42-feature vectors but different labels."
        )

        print(
            "  They will remain intact and be kept "
            "on one side of the split."
        )

    if len(mixed_label_groups) > 0:
        print("\n  Mixed-label feature group details:")

        for group_id in mixed_label_groups.index:
            group_rows = df[
                df["_feature_group_id"] == group_id
            ].copy()

            print("\n  Group ID:", int(group_id))
            print(
                group_rows[
                    [
                        "flow_id",
                        "timestamp",
                        "src_ip",
                        "dst_ip",
                        config.LABEL_COLUMN,
                    ]
                ].to_string(index=False)
            )

    # --------------------------------------------------------
    # Source-IP composition
    # --------------------------------------------------------
    print(
        "\n  Source-IP composition "
        "(diagnostic only; NOT used for splitting)..."
    )

    source_stats = (
        df.groupby("src_ip")[
            config.LABEL_COLUMN
        ]
        .agg(
            n_flows="count",
            attack_fraction="mean",
            n_label_values="nunique",
        )
        .sort_values(
            "n_flows",
            ascending=False,
        )
    )

    print(
        f"  Unique source IPs: "
        f"{len(source_stats):,}"
    )

    print(
        f"  Median flows/source IP: "
        f"{int(source_stats['n_flows'].median()):,}"
    )

    print(
        f"  Max flows/source IP: "
        f"{int(source_stats['n_flows'].max()):,}"
    )

    print("\n  Top 10 source IPs by flow count:")

    top_source = source_stats.head(10).copy()

    top_source["pct_of_total"] = (
        100.0
        * top_source["n_flows"]
        / len(df)
    )

    print(top_source.to_string())

    # --------------------------------------------------------
    # Stratified Group Split
    # --------------------------------------------------------
    print_section(
        f"[6/7] Feature-grouped split (n_splits={N_SPLITS})"
    )

    X = df[config.ML_FEATURES]

    y = df[config.LABEL_COLUMN]

    groups = df["_feature_group_id"]

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(
        splitter.split(
            X,
            y,
            groups=groups,
        )
    )

    train_df = df.iloc[
        train_idx
    ].copy()

    test_df = df.iloc[
        test_idx
    ].copy()

    # --------------------------------------------------------
    # Remove internal grouping column
    # --------------------------------------------------------
    train_group_ids = train_df[
        "_feature_group_id"
    ].copy()

    test_group_ids = test_df[
        "_feature_group_id"
    ].copy()

    # --------------------------------------------------------
    # Group overlap check
    # --------------------------------------------------------
    train_groups = set(
        train_group_ids.astype("uint64")
    )

    test_groups = set(
        test_group_ids.astype("uint64")
    )

    group_overlap = (
        train_groups & test_groups
    )

    print(
        f"  Train flows: "
        f"{len(train_df):,}"
    )

    print(
        f"  Test flows : "
        f"{len(test_df):,}"
    )

    print(
        f"  Train feature groups: "
        f"{len(train_groups):,}"
    )

    print(
        f"  Test feature groups : "
        f"{len(test_groups):,}"
    )

    print(
        f"  Feature-group overlap: "
        f"{len(group_overlap):,}  (must be 0)"
    )

    if group_overlap:
        raise RuntimeError(
            "Feature-group leakage detected!"
        )

    # --------------------------------------------------------
    # Source-IP overlap - diagnostic only
    # --------------------------------------------------------
    train_ips = set(
        train_df["src_ip"]
        .astype(str)
    )

    test_ips = set(
        test_df["src_ip"]
        .astype(str)
    )

    ip_overlap = (
        train_ips & test_ips
    )

    print(
        f"  Source-IP overlap: "
        f"{len(ip_overlap):,}"
        "  (diagnostic; allowed)"
    )

    # --------------------------------------------------------
    # Label distributions
    # --------------------------------------------------------
    print()

    print_label_distribution(
        train_df,
        "Train label distribution",
    )

    print_label_distribution(
        test_df,
        "Test label distribution",
    )

    # --------------------------------------------------------
    # Day distributions
    # --------------------------------------------------------
    print()

    train_days, test_days = print_day_distribution(
        train_df,
        test_df,
    )

    # --------------------------------------------------------
    # Exact feature overlap verification
    # --------------------------------------------------------
    print()

    print(
        "  Re-checking exact ML-feature-vector overlap..."
    )

    train_feature_hash = pd.util.hash_pandas_object(
        train_df[config.ML_FEATURES],
        index=False,
    )

    test_feature_hash = pd.util.hash_pandas_object(
        test_df[config.ML_FEATURES],
        index=False,
    )

    feature_hash_overlap = (
        set(
            train_feature_hash.astype("uint64")
        )
        &
        set(
            test_feature_hash.astype("uint64")
        )
    )

    print(
        f"  Exact ML-feature-vector overlap: "
        f"{len(feature_hash_overlap):,}"
    )

    if feature_hash_overlap:
        raise RuntimeError(
            "Exact ML-feature-vector leakage detected!"
        )

    # --------------------------------------------------------
    # Check class ratio difference
    # --------------------------------------------------------
    overall_attack_rate = (
        df[config.LABEL_COLUMN]
        .mean()
    )

    train_attack_rate = (
        train_df[config.LABEL_COLUMN]
        .mean()
    )

    test_attack_rate = (
        test_df[config.LABEL_COLUMN]
        .mean()
    )

    print()
    print("  Attack-rate comparison:")
    print(
        f"    Overall: "
        f"{overall_attack_rate * 100:.4f}%"
    )
    print(
        f"    Train  : "
        f"{train_attack_rate * 100:.4f}%"
    )
    print(
        f"    Test   : "
        f"{test_attack_rate * 100:.4f}%"
    )

    # --------------------------------------------------------
    # Remove internal grouping column before saving
    # --------------------------------------------------------
    train_df.drop(
        columns=["_feature_group_id"],
        inplace=True,
    )

    test_df.drop(
        columns=["_feature_group_id"],
        inplace=True,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    print_section("[7/7] Saving")

    train_path = (
        OUTPUT_DIR
        / "train_multiday.csv"
    )

    test_path = (
        OUTPUT_DIR
        / "test_multiday.csv"
    )

    train_df.to_csv(
        train_path,
        index=False,
    )

    test_df.to_csv(
        test_path,
        index=False,
    )

    print(
        f"  Train: {train_path}"
        f"  ({len(train_df):,} flows)"
    )

    print(
        f"  Test : {test_path}"
        f"  ({len(test_df):,} flows)"
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------
    summary_path = (
        OUTPUT_DIR
        / "split_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "NetworkIDS - Multiday Split Summary\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            "Split strategy\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            "Identical 42-feature-vector-grouped "
            "StratifiedGroupKFold (n_splits=5)\n"
        )

        f.write(
            "First fold used as a fixed approximately "
            "80/20 holdout.\n"
        )

        f.write(
            "This is NOT 5-fold cross-validation.\n\n"
        )

        f.write(
            "ML feature count\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            f"ML features: "
            f"{len(config.ML_FEATURES)}\n"
        )

        f.write(
            "src_ip used as ML feature: NO\n"
        )

        f.write(
            "day used as ML feature: NO\n\n"
        )

        f.write(
            "Dataset sizes\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            f"Total merged flows: "
            f"{len(df):,}\n"
        )

        f.write(
            f"Train flows: "
            f"{len(train_df):,}\n"
        )

        f.write(
            f"Test flows:  "
            f"{len(test_df):,}\n\n"
        )

        f.write(
            "Feature grouping\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            f"Unique feature groups: "
            f"{unique_groups:,}\n"
        )

        f.write(
            f"Feature groups with >1 row: "
            f"{len(duplicate_groups):,}\n"
        )

        f.write(
            f"Mixed-label feature groups: "
            f"{len(mixed_label_groups):,}\n\n"
        )

        f.write(
            "Leakage checks\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            f"Feature-group overlap: "
            f"{len(group_overlap)}\n"
        )

        f.write(
            f"Exact feature-vector overlap: "
            f"{len(feature_hash_overlap)}\n"
        )

        f.write(
            f"Source-IP overlap: "
            f"{len(ip_overlap)} "
            f"(diagnostic only)\n\n"
        )

        f.write(
            "Train labels\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            train_df[
                config.LABEL_COLUMN
            ]
            .value_counts()
            .sort_index()
            .to_string()
        )

        f.write("\n\n")

        f.write(
            "Test labels\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            test_df[
                config.LABEL_COLUMN
            ]
            .value_counts()
            .sort_index()
            .to_string()
        )

        f.write("\n\n")

        f.write(
            "Day distribution (train)\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        for day in DAYS:
            f.write(
                f"  {day:<12}: "
                f"{train_days.get(day, 0):,}\n"
            )

        f.write("\n")

        f.write(
            "Day distribution (test)\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        for day in DAYS:
            f.write(
                f"  {day:<12}: "
                f"{test_days.get(day, 0):,}\n"
            )

        f.write("\n")

        f.write(
            "Top 10 source IPs by flow count\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            top_source.to_string()
        )

        f.write("\n")

    print(
        f"  Summary: {summary_path}"
    )

    print()
    print("Done.")


if __name__ == "__main__":
    main()