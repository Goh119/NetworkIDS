from pathlib import Path
import sys

import pandas as pd


# ============================================================
# Allow importing config.py when running this file directly
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


# ============================================================
# Input / Output
# ============================================================

INPUT_FILE = config.FINAL_ML_DATASET_FILE

OUTPUT_FILE = (
    config.PROCESSED_DIR
    / "training_ready_dataset.csv"
)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("NetworkIDS - Prepare Training Dataset")
    print("=" * 70)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    print()
    print("Input file:")
    print(INPUT_FILE)

    if not INPUT_FILE.exists():

        print()
        print("ERROR: final_ml_dataset.csv was not found.")
        return

    # --------------------------------------------------------
    # Load final ML dataset
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    print()
    print(f"Loaded dataset: {len(df):,} rows")
    print(f"Columns: {len(df.columns)}")

    # --------------------------------------------------------
    # Check label column
    # --------------------------------------------------------

    if config.LABEL_COLUMN not in df.columns:

        print()
        print(
            f"ERROR: Label column '{config.LABEL_COLUMN}' "
            "was not found."
        )

        return

    # --------------------------------------------------------
    # Show original label distribution
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BEFORE FILTERING")
    print("=" * 70)

    print()
    print("Label distribution:")

    print(
        df[config.LABEL_COLUMN]
        .value_counts(dropna=False)
        .sort_index()
    )

    unmatched_count = (
        df[config.LABEL_COLUMN]
        .isna()
        .sum()
    )

    matched_count = (
        df[config.LABEL_COLUMN]
        .notna()
        .sum()
    )

    print()
    print(f"Matched rows:   {matched_count:,}")
    print(f"Unmatched rows: {unmatched_count:,}")

    # --------------------------------------------------------
    # Keep only rows with official labels
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # We do NOT assign labels here.
    #
    # Rows with NaN label are simply excluded from supervised
    # training because no reliable official ground-truth label
    # was obtained during the matching stage.
    #
    # --------------------------------------------------------

    training_df = df[
        df[config.LABEL_COLUMN].notna()
    ].copy()

    # --------------------------------------------------------
    # Validate binary labels
    # --------------------------------------------------------

    valid_labels = {
        config.LABEL_BENIGN,
        config.LABEL_ATTACK,
    }

    actual_labels = set(
        training_df[config.LABEL_COLUMN]
        .dropna()
        .unique()
    )

    unexpected_labels = (
        actual_labels - valid_labels
    )

    if unexpected_labels:

        print()
        print(
            "ERROR: Unexpected labels detected:"
        )

        print(unexpected_labels)

        return

    # --------------------------------------------------------
    # Convert label to integer
    # --------------------------------------------------------
    #
    # The final training dataset should contain:
    #   0 = Benign
    #   1 = Attack
    #
    # --------------------------------------------------------

    training_df[config.LABEL_COLUMN] = (
        training_df[config.LABEL_COLUMN]
        .astype(int)
    )

    # --------------------------------------------------------
    # Check ML features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in config.ML_FEATURES
        if feature not in training_df.columns
    ]

    if missing_features:

        print()
        print("ERROR: Missing ML features:")

        for feature in missing_features:
            print(f"  - {feature}")

        return

    # --------------------------------------------------------
    # Check feature count
    # --------------------------------------------------------

    if len(config.ML_FEATURES) != 42:

        print()
        print(
            "ERROR: Expected exactly 42 ML features, "
            f"but config contains {len(config.ML_FEATURES)}."
        )

        return

    # --------------------------------------------------------
    # Verify output directory
    # --------------------------------------------------------

    config.PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save training-ready dataset
    # --------------------------------------------------------

    training_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING DATASET")
    print("=" * 70)

    print()
    print(f"Training rows: {len(training_df):,}")
    print(f"Training columns: {len(training_df.columns)}")

    print()
    print("Label distribution:")

    label_counts = (
        training_df[config.LABEL_COLUMN]
        .value_counts()
        .sort_index()
    )

    print(label_counts)

    print()
    print("Label percentages:")

    label_percentages = (
        training_df[config.LABEL_COLUMN]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    for label, percentage in label_percentages.items():

        if label == config.LABEL_BENIGN:
            name = "Benign"

        elif label == config.LABEL_ATTACK:
            name = "Attack"

        else:
            name = str(label)

        print(
            f"  {name}: {percentage:.2f}%"
        )

    # --------------------------------------------------------
    # Check remaining NaN labels
    # --------------------------------------------------------

    remaining_nan_labels = (
        training_df[config.LABEL_COLUMN]
        .isna()
        .sum()
    )

    print()
    print(
        f"Remaining NaN labels: {remaining_nan_labels}"
    )

    # --------------------------------------------------------
    # Check exact ML feature availability
    # --------------------------------------------------------

    print()
    print(
        f"ML features available: "
        f"{len(config.ML_FEATURES)}/42"
    )

    # --------------------------------------------------------
    # Output path
    # --------------------------------------------------------

    print()
    print("Output file:")
    print(OUTPUT_FILE)

    # --------------------------------------------------------
    # Important integrity checks
    # --------------------------------------------------------

    if remaining_nan_labels != 0:
        print()
        print(
            "WARNING: NaN labels remain in the training dataset."
        )

    if len(training_df) != matched_count:
        print()
        print(
            "WARNING: Training row count does not match "
            "the number of matched rows."
        )

    if (
        len(training_df) == matched_count
        and remaining_nan_labels == 0
    ):
        print()
        print("Training dataset validation: PASS")

    print()
    print("=" * 70)
    print("Preparation completed.")
    print("=" * 70)

    print()
    print(
        "Important:"
    )
    print(
        "No labels were manually assigned."
    )
    print(
        "No unmatched rows were modified."
    )
    print(
        "No duplicate rows were removed."
    )
    print(
        "No outlier filtering was performed."
    )

if __name__ == "__main__":
    main()