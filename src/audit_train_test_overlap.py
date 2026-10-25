
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.config as config

TRAIN = config.TRAIN_DATASET_FILE
TEST = config.TEST_DATASET_FILE

CHUNK_SIZE = 100_000
FEATURES = list(config.ML_FEATURES)
EXPECTED_TRAIN_ROWS = 3_484_498
EXPECTED_TEST_ROWS = 745_724


def check_header(path):
    header = list(pd.read_csv(path, nrows=0).columns)
    missing = [col for col in FEATURES if col not in header]

    if missing:
        raise ValueError(
            f"{path.name} is missing features: {missing}"
        )


def make_hashes(chunk):
    # Parse both datasets using float64 so feature values
    # and NaN representations are handled consistently.
    values = chunk[FEATURES].to_numpy(dtype=np.float64)

    if np.isinf(values).any():
        raise ValueError("Infinite feature values detected.")

    # NaN is retained; no imputation or placeholder is applied.
    return pd.util.hash_pandas_object(
        chunk[FEATURES],
        index=False,
    ).to_numpy(dtype=np.uint64)


def main():
    for path in (TRAIN, TEST):
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")
        check_header(path)

    print("=" * 65)
    print("TRAIN / TEST FEATURE-VECTOR OVERLAP AUDIT")
    print("=" * 65)
    print(f"Features checked: {len(FEATURES)}")
    print("NaN policy: retain NaN")
    print("Reading held-out test feature vectors...")

    test_hashes = set()
    test_rows = 0

    for chunk in pd.read_csv(
        TEST,
        usecols=FEATURES,
        dtype={feature: np.float64 for feature in FEATURES},
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        hashes = make_hashes(chunk)
        test_hashes.update(int(value) for value in hashes)
        test_rows += len(chunk)

        if test_rows % 500_000 < len(chunk):
            print(f"Test scanned: {test_rows:,}", flush=True)

    if test_rows != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_TEST_ROWS:,} test rows, "
            f"read {test_rows:,}."
        )

    print(f"Test rows: {test_rows:,}")
    print(f"Unique test feature-vector hashes: {len(test_hashes):,}")
    print("\nChecking train feature vectors against test...")

    train_rows = 0
    overlap_rows = 0
    overlap_hashes = set()

    for chunk in pd.read_csv(
        TRAIN,
        usecols=FEATURES,
        dtype={feature: np.float64 for feature in FEATURES},
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        hashes = make_hashes(chunk)
        train_rows += len(chunk)

        for value in hashes:
            key = int(value)

            if key in test_hashes:
                overlap_rows += 1
                overlap_hashes.add(key)

        if train_rows % 500_000 < len(chunk):
            print(f"Train scanned: {train_rows:,}", flush=True)

    if train_rows != EXPECTED_TRAIN_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_TRAIN_ROWS:,} train rows, "
            f"read {train_rows:,}."
        )

    print("\n" + "=" * 65)
    print("AUDIT RESULT")
    print("=" * 65)
    print(f"Train rows scanned: {train_rows:,}")
    print(f"Test rows scanned: {test_rows:,}")
    print(f"Train rows matching a test hash: {overlap_rows:,}")
    print(f"Distinct matching hashes: {len(overlap_hashes):,}")

    if overlap_rows == 0:
        print("\nRESULT: PASS - No matching feature-vector hashes found.")
    else:
        print(
            "\nRESULT: REVIEW REQUIRED - Matching hashes found. "
            "These are candidates; verify actual feature vectors "
            "before deciding whether to remove rows."
        )

    print(
        "\nNote: this is a 64-bit hash-based overlap screen. "
        "Matching hashes are not, by themselves, proof of "
        "identical feature vectors."
    )
    print("Audit only: no datasets were modified.")


if __name__ == "__main__":
    main()
