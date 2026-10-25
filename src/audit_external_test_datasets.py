
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.config as config

TEST_DIR = config.TEST_DIR
FEATURES = list(config.ML_FEATURES)

LABEL_LIKE_NAMES = {
    "label",
    "class",
    "target",
    "attack_label",
    "binary_label",
    "y",
}


def main():
    if not TEST_DIR.is_dir():
        raise FileNotFoundError(
            f"Test directory not found: {TEST_DIR}"
        )

    # Only inspect flow CSVs. Packet CSVs are excluded.
    files = sorted(
        path
        for path in TEST_DIR.rglob("*.csv")
        if path.name.lower().endswith("flow.csv")
    )

    if not files:
        print(f"No flow CSV files found under {TEST_DIR}")
        return

    print("=" * 75)
    print("EXTERNAL TEST FLOW DATASET AUDIT")
    print("=" * 75)
    print(f"Directory: {TEST_DIR}")
    print(f"Flow CSV files found: {len(files)}")
    print(f"Required ML features: {len(FEATURES)}")
    print("This script reads headers and small label samples only.")
    print("No dataset files will be modified.")

    feature_complete = 0
    files_with_label = 0
    files_missing_features = []
    files_without_label = []

    for index, path in enumerate(files, start=1):
        print(f"\n[{index}/{len(files)}] {path.name}")

        try:
            header = list(
                pd.read_csv(path, nrows=0).columns
            )

            missing = [
                feature
                for feature in FEATURES
                if feature not in header
            ]

            label_like = [
                column
                for column in header
                if (
                    "label" in column.lower()
                    or column.lower() in LABEL_LIKE_NAMES
                )
            ]

            has_label = "label" in header

            print(f"  Columns: {len(header)}")
            print(f"  All ML features present: {'YES' if not missing else 'NO'}")
            print(f"  Has exact 'label' column: {'YES' if has_label else 'NO'}")

            if missing:
                print(f"  Missing features: {missing}")
                files_missing_features.append(path.name)
            else:
                feature_complete += 1

            if has_label:
                files_with_label += 1
            else:
                files_without_label.append(path.name)

            print(
                "  Label-related columns:",
                label_like if label_like else "None",
            )

            # Inspect only a few label values; do not load the full dataset.
            if label_like:
                sample = pd.read_csv(
                    path,
                    usecols=label_like,
                    nrows=20,
                    low_memory=False,
                )

                for column in label_like:
                    values = (
                        sample[column]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )
                    print(
                        f"  Sample {column} values: {values[:10]}"
                    )

        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")

    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    print(f"Total flow CSV files: {len(files)}")
    print(f"Files with all 42 required features: {feature_complete}")
    print(f"Files with exact label column: {files_with_label}")

    print("\nFiles missing ML features:")
    print(files_missing_features if files_missing_features else "None")

    print("\nFiles without exact 'label' column:")
    print(files_without_label if files_without_label else "None")

    print("\nAudit complete. No files were modified.")


if __name__ == "__main__":
    main()
