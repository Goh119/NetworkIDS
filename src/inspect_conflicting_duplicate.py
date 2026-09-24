"""
inspect_conflicting_duplicate.py
--------------------------------
NetworkIDS - Inspect duplicate feature groups with conflicting labels.

Purpose:
    Find ML feature groups where all 42 ML features are identical,
    but the official binary labels are different.

This script does NOT modify any dataset.
It only reports the conflicting group(s) for investigation.

Usage:
    python src/inspect_conflicting_duplicate.py
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
# Input / Output
# ============================================================

TRAINING_FILE = (
    config.PROCESSED_DIR
    / "training_ready_dataset.csv"
)

OUTPUT_FILE = (
    config.PROCESSED_DIR
    / "conflicting_duplicate_groups.csv"
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
    print("NetworkIDS - Conflicting Duplicate Inspection")
    print("=" * 70)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not TRAINING_FILE.exists():

        print()
        print("[ERROR] Training dataset not found:")
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
        f"Loaded: {len(df):,} rows x "
        f"{len(df.columns)} columns"
    )

    print(
        f"ML features: {len(config.ML_FEATURES)}"
    )

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    required_columns = (
        config.ML_FEATURES
        + [
            config.LABEL_COLUMN,
            "original_label",
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        print()
        print("[ERROR] Missing required columns:")

        for column in missing_columns:
            print(f"  - {column}")

        return

    # --------------------------------------------------------
    # Group by all 42 ML features
    # --------------------------------------------------------

    print_section(
        "1. Finding conflicting feature groups"
    )

    grouped = df.groupby(
        config.ML_FEATURES,
        dropna=False
    )

    # Number of unique binary labels in each feature group
    label_counts = (
        grouped[config.LABEL_COLUMN]
        .nunique()
    )

    conflicting_keys = label_counts[
        label_counts > 1
    ].index

    n_conflicting = len(
        conflicting_keys
    )

    print(
        f"Conflicting feature groups found: "
        f"{n_conflicting}"
    )

    if n_conflicting == 0:

        print()
        print(
            "[OK] No conflicting-label duplicate groups found."
        )

        return

    # --------------------------------------------------------
    # Extract conflicting rows
    # --------------------------------------------------------

    conflict_mask = pd.Series(
        False,
        index=df.index
    )

    for key in conflicting_keys:

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

        conflict_mask = (
            conflict_mask
            | mask
        )

    conflict_df = df[
        conflict_mask
    ].copy()

    print(
        f"Rows involved: "
        f"{len(conflict_df):,}"
    )

    # --------------------------------------------------------
    # Summary of labels
    # --------------------------------------------------------

    print_section(
        "2. Conflicting label details"
    )

    print()
    print("Binary labels:")

    print(
        conflict_df[
            config.LABEL_COLUMN
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    print()
    print("Original labels:")

    print(
        conflict_df[
            "original_label"
        ]
        .value_counts(
            dropna=False
        )
    )

    # --------------------------------------------------------
    # Display identifying information
    # --------------------------------------------------------

    print_section(
        "3. Flow details"
    )

    display_columns = [
        "flow_id",
        "timestamp",
        "src_ip",
        "dst_ip",
        "src_mac",
        "dst_mac",
        "src_port",
        "dst_port",
        "ip_protocol",
        "flow_duration",
        "flow_packet_count",
        "flow_byte_count",
        "forward_packet_count",
        "reverse_packet_count",
        "forward_byte_count",
        "reverse_byte_count",
        "flow_direction",
        "syn_count",
        "ack_count",
        "fin_count",
        "rst_count",
        "psh_count",
        "syn_seen",
        "syn_ack_seen",
        "ack_after_syn_ack",
        config.LABEL_COLUMN,
        "original_label",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in conflict_df.columns
    ]

    print()

    print(
        conflict_df[
            display_columns
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Timestamp analysis
    # --------------------------------------------------------

    print_section(
        "4. Timestamp analysis"
    )

    timestamps = pd.to_numeric(
        conflict_df["timestamp"],
        errors="coerce"
    )

    if timestamps.notna().all():

        timestamp_min = timestamps.min()
        timestamp_max = timestamps.max()

        print(
            f"Minimum timestamp: "
            f"{timestamp_min}"
        )

        print(
            f"Maximum timestamp: "
            f"{timestamp_max}"
        )

        print(
            f"Timestamp spread: "
            f"{timestamp_max - timestamp_min:.6f} seconds"
        )

        print()

        print(
            "Timestamps:"
        )

        for value in timestamps.sort_values():

            dt = pd.to_datetime(
                value,
                unit="s",
                errors="coerce"
            )

            print(
                f"  {value:.6f} -> {dt}"
            )

    # --------------------------------------------------------
    # Feature values
    # --------------------------------------------------------

    print_section(
        "5. The 42 identical ML feature values"
    )

    # Since all rows in a conflicting group have
    # identical feature values, display the first row.
    first_row = conflict_df.iloc[0]

    for feature in config.ML_FEATURES:

        value = first_row[feature]

        print(
            f"{feature:<30} = {value}"
        )

    # --------------------------------------------------------
    # Compare metadata
    # --------------------------------------------------------

    print_section(
        "6. Metadata comparison"
    )

    metadata_columns = [
        column
        for column in config.METADATA_COLUMNS
        if column in conflict_df.columns
    ]

    for column in metadata_columns:

        unique_values = (
            conflict_df[column]
            .drop_duplicates()
            .tolist()
        )

        print()
        print(
            f"{column}:"
        )

        for value in unique_values:

            print(
                f"  {value}"
            )

    # --------------------------------------------------------
    # Check whether different labels correspond to
    # different original CIC labels
    # --------------------------------------------------------

    print_section(
        "7. Label mapping"
    )

    label_mapping = (
        conflict_df[
            [
                config.LABEL_COLUMN,
                "original_label",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            by=[
                config.LABEL_COLUMN,
                "original_label",
            ]
        )
    )

    print(
        label_mapping.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save for further investigation
    # --------------------------------------------------------

    conflict_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print_section(
        "8. Output"
    )

    print(
        "Conflicting rows saved to:"
    )

    print(
        OUTPUT_FILE
    )

    # --------------------------------------------------------
    # Final reminder
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Inspection completed.")
    print("No labels were modified.")
    print("No rows were deleted.")
    print("=" * 70)

if __name__ == "__main__":
    main()