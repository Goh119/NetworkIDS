"""
Split the training-ready dataset into training and testing sets.

Strategy:
- Input: training_ready_dataset.csv
- 80% training / 20% testing using 5-fold StratifiedGroupKFold
- Groups are defined by identical 42 ML feature values
- Keeps identical feature vectors within the same split
- Preserves the Benign/Attack class distribution as closely as possible
- No rows are removed or modified
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import config


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = config.PROCESSED_DIR / "training_ready_dataset.csv"

OUTPUT_DIR = config.PROCESSED_DIR / "split"
TRAIN_FILE = OUTPUT_DIR / "train_dataset.csv"
TEST_FILE = OUTPUT_DIR / "test_dataset.csv"
SUMMARY_FILE = OUTPUT_DIR / "split_summary.txt"

RANDOM_STATE = 42
N_SPLITS = 5


# ============================================================
# Helper functions
# ============================================================

def print_label_distribution(df, name):
    """Print label counts and percentages."""

    counts = df[config.LABEL_COLUMN].value_counts().sort_index()

    print(f"\n{name} label distribution:")

    total = len(df)

    for label, count in counts.items():
        percentage = count / total * 100
        label_name = (
            "Benign" if label == config.LABEL_BENIGN
            else "Attack" if label == config.LABEL_ATTACK
            else str(label)
        )

        print(
            f"  {label_name}: "
            f"{count:,} ({percentage:.2f}%)"
        )


def main():

    print("=" * 70)
    print("DATASET TRAIN/TEST SPLIT")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Check input file
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Training-ready dataset not found:\n{INPUT_FILE}"
        )

    # --------------------------------------------------------
    # 2. Load dataset
    # --------------------------------------------------------

    print("\n[1/7] Loading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # --------------------------------------------------------
    # 3. Validate required columns
    # --------------------------------------------------------

    print("\n[2/7] Validating dataset...")

    required_columns = (
        config.ML_FEATURES
        + config.METADATA_COLUMNS
        + [
            config.LABEL_COLUMN,
            "original_label"
        ]
    )

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns:\n{missing_columns}"
        )

    # Check label
    if df[config.LABEL_COLUMN].isna().any():
        raise ValueError(
            "Label column contains NaN values."
        )

    valid_labels = {
        config.LABEL_BENIGN,
        config.LABEL_ATTACK
    }

    actual_labels = set(
        df[config.LABEL_COLUMN].unique()
    )

    if not actual_labels.issubset(valid_labels):
        raise ValueError(
            f"Unexpected labels found: {actual_labels}"
        )

    print("Dataset validation: PASS")

    # --------------------------------------------------------
    # 4. Create feature groups
    # --------------------------------------------------------

    print("\n[3/7] Creating feature groups...")

    print(
        "Grouping rows by identical values across "
        "all 42 ML features..."
    )

    # Each identical 42-feature vector receives the same group ID.
    #
    # dropna=False is important because some legitimate feature
    # values are NaN due to protocol-specific missing values.

    df["_feature_group"] = (
        df.groupby(
            config.ML_FEATURES,
            dropna=False,
            sort=False
        ).ngroup()
    )

    number_of_groups = df["_feature_group"].nunique()

    print(
        f"Total rows: {len(df):,}"
    )

    print(
        f"Unique feature groups: {number_of_groups:,}"
    )

    print(
        f"Rows per feature group: "
        f"{len(df) / number_of_groups:.3f}"
    )

    # --------------------------------------------------------
    # 5. Stratified group split
    # --------------------------------------------------------

    print("\n[4/7] Creating 80/20 train-test split...")

    print(
        "Using StratifiedGroupKFold "
        f"(n_splits={N_SPLITS}, random_state={RANDOM_STATE})..."
    )

    X = df[config.ML_FEATURES]
    y = df[config.LABEL_COLUMN]
    groups = df["_feature_group"]

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # Five folds = approximately 20% per fold.
    # The first fold is used as the test set.
    train_indices, test_indices = next(
        splitter.split(
            X,
            y,
            groups
        )
    )

    train_df = df.iloc[train_indices].copy()
    test_df = df.iloc[test_indices].copy()

    # --------------------------------------------------------
    # 6. Remove internal group ID
    # --------------------------------------------------------

    train_df.drop(
        columns=["_feature_group"],
        inplace=True
    )

    test_df.drop(
        columns=["_feature_group"],
        inplace=True
    )

    # --------------------------------------------------------
    # 7. Validation
    # --------------------------------------------------------

    print("\n[5/7] Validating split...")

    total_rows = len(train_df) + len(test_df)

    if total_rows != len(df):
        raise ValueError(
            "Train + test row count does not equal original dataset."
        )

    # Check row overlap using flow_id
    train_flow_ids = set(train_df["flow_id"])
    test_flow_ids = set(test_df["flow_id"])

    flow_overlap = train_flow_ids.intersection(test_flow_ids)

    if flow_overlap:
        raise ValueError(
            f"Flow ID leakage detected: "
            f"{len(flow_overlap):,} overlapping flow IDs."
        )

    # Re-create feature groups separately for leakage check.
    #
    # We use a merge on the 42 features instead of relying on
    # row order or flow_id.

    train_features = train_df[config.ML_FEATURES].copy()
    test_features = test_df[config.ML_FEATURES].copy()

    train_features["_group_marker"] = 1

    test_features["_group_marker"] = 1

    overlapping_features = train_features.merge(
        test_features,
        on=config.ML_FEATURES,
        how="inner"
    )

    if len(overlapping_features) > 0:
        raise ValueError(
            "Feature-vector leakage detected: "
            "identical 42-feature vectors exist in both "
            "train and test sets."
        )

    print("Flow ID overlap: 0")
    print("42-feature vector overlap: 0")
    print("Data leakage check: PASS")

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print_label_distribution(
        df,
        "Original dataset"
    )

    print_label_distribution(
        train_df,
        "Training set"
    )

    print_label_distribution(
        test_df,
        "Testing set"
    )

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    print("\n[6/7] Saving datasets...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_csv(
        TRAIN_FILE,
        index=False
    )

    test_df.to_csv(
        TEST_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    train_percentage = len(train_df) / len(df) * 100
    test_percentage = len(test_df) / len(df) * 100

    train_benign = (
        train_df[config.LABEL_COLUMN]
        == config.LABEL_BENIGN
    ).sum()

    train_attack = (
        train_df[config.LABEL_COLUMN]
        == config.LABEL_ATTACK
    ).sum()

    test_benign = (
        test_df[config.LABEL_COLUMN]
        == config.LABEL_BENIGN
    ).sum()

    test_attack = (
        test_df[config.LABEL_COLUMN]
        == config.LABEL_ATTACK
    ).sum()

    summary = f"""
Dataset Split Summary
=====================

Input dataset:
{INPUT_FILE}

Original dataset:
Rows: {len(df):,}

Training set:
Rows: {len(train_df):,}
Percentage: {train_percentage:.2f}%
Benign: {train_benign:,}
Attack: {train_attack:,}

Testing set:
Rows: {len(test_df):,}
Percentage: {test_percentage:.2f}%
Benign: {test_benign:,}
Attack: {test_attack:,}

Split method:
StratifiedGroupKFold

Number of folds:
{N_SPLITS}

Random state:
{RANDOM_STATE}

Grouping:
42 ML features

Feature-vector leakage:
0 overlapping groups

Flow ID overlap:
0

Total rows after split:
{len(train_df) + len(test_df):,}

Validation:
PASS
"""

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(summary.strip())

    print("\n[7/7] Split completed.")

    print("\nOutput files:")
    print(f"  Train:   {TRAIN_FILE}")
    print(f"  Test:    {TEST_FILE}")
    print(f"  Summary: {SUMMARY_FILE}")

    print("\n" + "=" * 70)
    print("SPLIT COMPLETE - PASS")
    print("=" * 70)

if __name__ == "__main__":
    main()