
from pathlib import Path

import numpy as np
import pandas as pd
import src.config as config

ROOT = Path(__file__).resolve().parents[1]
DATA = config.TRAIN_DATASET_FILE

EXPECTED_ROWS = 3_484_498
CHUNK_SIZE = 100_000

FEATURES = [
    "frame_length_mean", "eth_type", "vlan_id", "vlan_priority",
    "ip_version", "ip_dscp", "ip_ecn", "ip_total_length_mean",
    "ip_id_mean", "ip_df", "ip_mf", "fragment_offset", "ttl_mean",
    "ip_protocol", "ip_header_length", "ip_checksum_valid",
    "src_port", "dst_port", "tcp_window_mean", "tcp_header_length_mean",
    "l4_checksum_valid", "flow_duration", "flow_packet_count",
    "flow_byte_count", "forward_packet_count", "reverse_packet_count",
    "forward_byte_count", "reverse_byte_count", "iat_min", "iat_mean",
    "iat_max", "iat_std", "byte_rate", "flow_direction", "syn_count",
    "ack_count", "fin_count", "rst_count", "psh_count", "syn_seen",
    "syn_ack_seen", "ack_after_syn_ack"
]

COLS = FEATURES + ["label"]


def main():
    if not DATA.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA}")

    header = list(pd.read_csv(DATA, nrows=0).columns)

    print("=" * 60)
    print("TRAIN V1 DATA QUALITY AUDIT")
    print("=" * 60)
    print("File:", DATA)
    print("Column count:", len(header))
    print("Expected columns:", len(COLS))
    print("Missing columns:", [c for c in COLS if c not in header])
    print("Unexpected columns:", [c for c in header if c not in COLS])

    if len(header) != len(COLS) or set(header) != set(COLS):
        raise ValueError("Column validation failed. Stop before training.")

    total_rows = 0
    label_counts = {0: 0, 1: 0}
    missing_labels = 0
    unexpected_labels = 0

    missing_by_feature = {c: 0 for c in FEATURES}
    inf_by_feature = {c: 0 for c in FEATURES}

    rows_with_missing_features = 0
    rows_with_inf_features = 0
    rows_with_nonfinite_features = 0

    for chunk in pd.read_csv(
        DATA,
        usecols=COLS,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):
        total_rows += len(chunk)

        labels = pd.to_numeric(chunk["label"], errors="raise")
        missing_labels += int(labels.isna().sum())

        valid_labels = labels.dropna()
        label_counts[0] += int((valid_labels == 0).sum())
        label_counts[1] += int((valid_labels == 1).sum())

        unexpected_labels += int(
            (~valid_labels.isin([0, 1])).sum()
        )

        numeric = chunk[FEATURES].apply(
            pd.to_numeric, errors="raise"
        )

        missing_mask = numeric.isna().any(axis=1)
        missing_by_feature_chunk = numeric.isna().sum()

        values = numeric.to_numpy(dtype=np.float64)
        inf_mask_array = np.isinf(values)
        inf_mask = inf_mask_array.any(axis=1)

        for feature in FEATURES:
            missing_by_feature[feature] += int(
                missing_by_feature_chunk[feature]
            )
            inf_by_feature[feature] += int(
                inf_mask_array[:, FEATURES.index(feature)].sum()
            )

        rows_with_missing_features += int(missing_mask.sum())
        rows_with_inf_features += int(inf_mask.sum())
        rows_with_nonfinite_features += int(
            (missing_mask | inf_mask).sum()
        )

        if total_rows % 500_000 < len(chunk):
            print(f"Scanned {total_rows:,} rows...", flush=True)

    print("\n" + "=" * 60)
    print("ROW AND LABEL SUMMARY")
    print("=" * 60)
    print(f"Total rows: {total_rows:,}")
    print(f"Expected rows: {EXPECTED_ROWS:,}")
    print(f"Benign (0): {label_counts[0]:,}")
    print(f"Attack (1): {label_counts[1]:,}")
    print(f"Missing labels: {missing_labels:,}")
    print(f"Unexpected labels: {unexpected_labels:,}")

    print("\n" + "=" * 60)
    print("FEATURE QUALITY")
    print("=" * 60)
    print(f"Rows with any missing feature: {rows_with_missing_features:,}")
    print(f"Rows with any infinite feature: {rows_with_inf_features:,}")
    print(f"Rows with missing or infinite features: {rows_with_nonfinite_features:,}")

    print("\nFeatures with missing values:")
    missing_found = False
    for feature, count in missing_by_feature.items():
        if count:
            missing_found = True
            print(f"  {feature}: {count:,}")
    if not missing_found:
        print("  None")

    print("\nFeatures with infinite values:")
    inf_found = False
    for feature, count in inf_by_feature.items():
        if count:
            inf_found = True
            print(f"  {feature}: {count:,}")
    if not inf_found:
        print("  None")

    print("\n" + "=" * 60)
    if (
        total_rows == EXPECTED_ROWS
        and missing_labels == 0
        and unexpected_labels == 0
    ):
        print("STRUCTURAL CHECK: PASS")
    else:
        print("STRUCTURAL CHECK: REVIEW REQUIRED")

    if rows_with_nonfinite_features == 0:
        print("FEATURE CHECK: PASS")
    else:
        print("FEATURE CHECK: REVIEW REQUIRED")

    print("Audit only: source dataset was not modified.")


if __name__ == "__main__":
    main()
