"""
Quality check for newly extracted NetworkIDS datasets.

Checks top-level *_flow.csv files in:

    data/processed/
    data/test/

It intentionally does NOT recurse into existing CIC-IDS2017
processed subdirectories such as multiday_split/.

The script does NOT modify any dataset.

Checks:
    - Flow CSV exists and is readable
    - Row count
    - Column count
    - All configured ML_FEATURES are present
    - Missing / extra columns
    - NaN counts
    - Features that are completely NaN
    - Positive / negative infinity counts
    - TCP / UDP flow counts
    - Duplicate flow_id values
    - Matching packet CSV existence / size

Outputs:
    data/quality_check_report.csv
    data/quality_check_nan_features.csv
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from src import config


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TEST_DIR = PROJECT_ROOT / "data" / "test"

REPORT_PATH = PROJECT_ROOT / "data" / "quality_check_report.csv"
NAN_REPORT_PATH = PROJECT_ROOT / "data" / "quality_check_nan_features.csv"

CHUNK_SIZE = 100_000


# ============================================================
# EXPECTED FEATURES
# ============================================================

ML_FEATURES = list(config.ML_FEATURES)

EXPECTED_METADATA = [
    "flow_id",
    "timestamp",
    "src_ip",
    "dst_ip",
    "src_mac",
    "dst_mac",
]

EXPECTED_AUXILIARY = [
    "flow_mac_ip_inconsistent_count",
    "flow_mac_ip_spoof_suspect",
    "src_ip_mac_seen_mac_count_max",
    "src_mac_ip_seen_ip_count_max",
]

EXPECTED_FLOW_COLUMNS = (
    EXPECTED_METADATA
    + ML_FEATURES
    + EXPECTED_AUXILIARY
)


# ============================================================
# HELPERS
# ============================================================

def find_flow_files() -> list[tuple[str, Path]]:
    """
    Find only top-level *_flow.csv files.

    This intentionally does NOT recurse into CIC-IDS2017
    processed folders.
    """

    files: list[tuple[str, Path]] = []

    if PROCESSED_DIR.exists():
        for path in sorted(PROCESSED_DIR.glob("*_flow.csv")):
            files.append(("processed", path))

    if TEST_DIR.exists():
        for path in sorted(TEST_DIR.glob("*_flow.csv")):
            files.append(("test", path))

    return files


def matching_packet_csv(flow_path: Path) -> Path:
    """
    Convert:

        abc_flow.csv

    to:

        abc_packets.csv
    """

    name = flow_path.name

    if not name.endswith("_flow.csv"):
        raise ValueError(
            f"Unexpected flow filename: {flow_path}"
        )

    packet_name = name[:-len("_flow.csv")] + "_packets.csv"

    return flow_path.parent / packet_name


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0:
            return f"{value:.2f} {unit}"

        value /= 1024.0

    return f"{value:.2f} PB"


# ============================================================
# QUALITY CHECK
# ============================================================

def check_flow_file(
    location: str,
    flow_path: Path,
) -> tuple[dict, list[dict]]:

    packet_path = matching_packet_csv(flow_path)

    print()
    print("=" * 80)
    print(f"Checking : {flow_path}")
    print(f"Location : {location}")
    print("=" * 80)

    # --------------------------------------------------------
    # Packet CSV status
    # --------------------------------------------------------

    packet_exists = packet_path.is_file()

    if packet_exists:
        packet_size_bytes = packet_path.stat().st_size
        packet_size = human_size(packet_size_bytes)

        if packet_size_bytes > 0:
            packet_status = "OK"
        else:
            packet_status = "EMPTY"

    else:
        packet_size_bytes = 0
        packet_size = "-"
        packet_status = "MISSING"

    # --------------------------------------------------------
    # Read header first
    # --------------------------------------------------------

    try:
        header_df = pd.read_csv(
            flow_path,
            nrows=0,
        )
    except Exception as exc:
        result = {
            "location": location,
            "flow_file": flow_path.name,
            "status": "READ ERROR",
            "flow_rows": 0,
            "column_count": 0,
            "expected_column_count": len(EXPECTED_FLOW_COLUMNS),
            "missing_ml_features": "",
            "missing_expected_columns": "",
            "unexpected_columns": "",
            "total_nan_cells": 0,
            "ml_nan_cells": 0,
            "all_nan_ml_features": "",
            "positive_inf": 0,
            "negative_inf": 0,
            "tcp_flows": 0,
            "udp_flows": 0,
            "other_protocol_flows": 0,
            "duplicate_flow_ids": 0,
            "packet_csv": packet_path.name,
            "packet_status": packet_status,
            "packet_size": packet_size,
            "error": str(exc),
        }

        return result, []

    columns = list(header_df.columns)

    missing_ml = [
        feature
        for feature in ML_FEATURES
        if feature not in columns
    ]

    missing_expected = [
        column
        for column in EXPECTED_FLOW_COLUMNS
        if column not in columns
    ]

    unexpected_columns = [
        column
        for column in columns
        if column not in EXPECTED_FLOW_COLUMNS
    ]

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    total_rows = 0

    total_nan_cells = 0
    ml_nan_cells = 0

    positive_inf = 0
    negative_inf = 0

    tcp_flows = 0
    udp_flows = 0
    other_protocol_flows = 0

    duplicate_flow_ids = 0
    seen_flow_ids: set[str] = set()

    feature_nan_counts = {
        feature: 0
        for feature in ML_FEATURES
        if feature in columns
    }

    # --------------------------------------------------------
    # Chunked scan
    # --------------------------------------------------------

    try:
        reader = pd.read_csv(
            flow_path,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        )

        for chunk_number, chunk in enumerate(reader, start=1):

            rows = len(chunk)
            total_rows += rows

            # ----------------------------------------------
            # NaN counts
            # ----------------------------------------------

            total_nan_cells += int(
                chunk.isna().sum().sum()
            )

            existing_ml = [
                feature
                for feature in ML_FEATURES
                if feature in chunk.columns
            ]

            if existing_ml:

                ml_nan = chunk[
                    existing_ml
                ].isna().sum()

                ml_nan_cells += int(
                    ml_nan.sum()
                )

                for feature, count in ml_nan.items():
                    feature_nan_counts[feature] += int(count)

            # ----------------------------------------------
            # Infinity counts
            # ----------------------------------------------

            for feature in existing_ml:

                numeric = pd.to_numeric(
                    chunk[feature],
                    errors="coerce",
                )

                values = numeric.to_numpy(
                    dtype=np.float64,
                    na_value=np.nan,
                )

                positive_inf += int(
                    np.isposinf(values).sum()
                )

                negative_inf += int(
                    np.isneginf(values).sum()
                )

            # ----------------------------------------------
            # Protocol counts
            #
            # IPv4/IPv6:
            # TCP = 6
            # UDP = 17
            # ----------------------------------------------

            if "ip_protocol" in chunk.columns:

                protocol = pd.to_numeric(
                    chunk["ip_protocol"],
                    errors="coerce",
                )

                tcp_flows += int(
                    (protocol == 6).sum()
                )

                udp_flows += int(
                    (protocol == 17).sum()
                )

                other_protocol_flows += int(
                    (
                        protocol.notna()
                        & (protocol != 6)
                        & (protocol != 17)
                    ).sum()
                )

            # ----------------------------------------------
            # Duplicate flow IDs
            # ----------------------------------------------

            if "flow_id" in chunk.columns:

                ids = (
                    chunk["flow_id"]
                    .dropna()
                    .astype(str)
                )

                # Duplicates already seen in earlier chunks
                duplicate_flow_ids += sum(
                    flow_id in seen_flow_ids
                    for flow_id in ids
                )

                # Duplicates within this chunk
                duplicate_flow_ids += int(
                    ids.duplicated().sum()
                )

                seen_flow_ids.update(
                    ids.unique()
                )

            print(
                f"  chunk {chunk_number}: "
                f"{total_rows:,} flows checked",
                end="\r",
            )

    except Exception as exc:

        print()

        result = {
            "location": location,
            "flow_file": flow_path.name,
            "status": "READ ERROR",
            "flow_rows": total_rows,
            "column_count": len(columns),
            "expected_column_count": len(EXPECTED_FLOW_COLUMNS),
            "missing_ml_features": ";".join(missing_ml),
            "missing_expected_columns": ";".join(missing_expected),
            "unexpected_columns": ";".join(unexpected_columns),
            "total_nan_cells": total_nan_cells,
            "ml_nan_cells": ml_nan_cells,
            "all_nan_ml_features": "",
            "positive_inf": positive_inf,
            "negative_inf": negative_inf,
            "tcp_flows": tcp_flows,
            "udp_flows": udp_flows,
            "other_protocol_flows": other_protocol_flows,
            "duplicate_flow_ids": duplicate_flow_ids,
            "packet_csv": packet_path.name,
            "packet_status": packet_status,
            "packet_size": packet_size,
            "error": str(exc),
        }

        return result, []

    print()

    # --------------------------------------------------------
    # NaN feature report
    # --------------------------------------------------------

    nan_feature_rows = []

    all_nan_ml_features = []

    for feature in ML_FEATURES:

        if feature not in feature_nan_counts:
            continue

        nan_count = feature_nan_counts[feature]

        if total_rows > 0:
            nan_percent = (
                nan_count / total_rows
            ) * 100.0
        else:
            nan_percent = 0.0

        if (
            total_rows > 0
            and nan_count == total_rows
        ):
            all_nan_ml_features.append(feature)

        if nan_count > 0:

            nan_feature_rows.append({
                "location": location,
                "flow_file": flow_path.name,
                "feature": feature,
                "nan_count": nan_count,
                "total_rows": total_rows,
                "nan_percent": round(
                    nan_percent,
                    6,
                ),
                "all_nan": (
                    total_rows > 0
                    and nan_count == total_rows
                ),
            })

    # --------------------------------------------------------
    # Determine overall status
    # --------------------------------------------------------

    critical_issues = []

    if total_rows == 0:
        critical_issues.append("NO FLOW ROWS")

    if missing_ml:
        critical_issues.append("MISSING ML FEATURES")

    if positive_inf > 0 or negative_inf > 0:
        critical_issues.append("INFINITY FOUND")

    if duplicate_flow_ids > 0:
        critical_issues.append("DUPLICATE FLOW IDS")

    if packet_status != "OK":
        critical_issues.append(
            f"PACKET CSV {packet_status}"
        )

    if critical_issues:
        status = "CHECK"
    else:
        status = "OK"

    # --------------------------------------------------------
    # Summary result
    # --------------------------------------------------------

    result = {
        "location": location,
        "flow_file": flow_path.name,
        "status": status,
        "flow_rows": total_rows,
        "column_count": len(columns),
        "expected_column_count": len(EXPECTED_FLOW_COLUMNS),
        "missing_ml_features": ";".join(missing_ml),
        "missing_expected_columns": ";".join(missing_expected),
        "unexpected_columns": ";".join(unexpected_columns),
        "total_nan_cells": total_nan_cells,
        "ml_nan_cells": ml_nan_cells,
        "all_nan_ml_features": ";".join(
            all_nan_ml_features
        ),
        "positive_inf": positive_inf,
        "negative_inf": negative_inf,
        "tcp_flows": tcp_flows,
        "udp_flows": udp_flows,
        "other_protocol_flows": other_protocol_flows,
        "duplicate_flow_ids": duplicate_flow_ids,
        "packet_csv": packet_path.name,
        "packet_status": packet_status,
        "packet_size": packet_size,
        "error": "",
    }

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print(f"  Status             : {status}")
    print(f"  Flow rows          : {total_rows:,}")
    print(
        f"  Columns            : "
        f"{len(columns)} "
        f"(expected {len(EXPECTED_FLOW_COLUMNS)})"
    )

    print(
        f"  ML features        : "
        f"{len(ML_FEATURES) - len(missing_ml)}"
        f"/{len(ML_FEATURES)}"
    )

    print(
        f"  TCP / UDP          : "
        f"{tcp_flows:,} / {udp_flows:,}"
    )

    print(
        f"  Other protocols    : "
        f"{other_protocol_flows:,}"
    )

    print(
        f"  Infinity           : "
        f"+{positive_inf:,} / -{negative_inf:,}"
    )

    print(
        f"  Duplicate flow IDs : "
        f"{duplicate_flow_ids:,}"
    )

    print(
        f"  Packet CSV         : "
        f"{packet_status} ({packet_size})"
    )

    if missing_ml:
        print(
            "  Missing ML features: "
            + ", ".join(missing_ml)
        )

    if all_nan_ml_features:
        print(
            "  100% NaN features  : "
            + ", ".join(all_nan_ml_features)
        )

    return result, nan_feature_rows


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    flow_files = find_flow_files()

    if not flow_files:

        print(
            "No top-level *_flow.csv files found in:"
        )

        print(f"  {PROCESSED_DIR}")
        print(f"  {TEST_DIR}")

        return 1

    print("=" * 80)
    print("NetworkIDS - New Dataset Quality Check")
    print("=" * 80)

    print(
        f"Configured ML features : "
        f"{len(ML_FEATURES)}"
    )

    print(
        f"Expected flow columns  : "
        f"{len(EXPECTED_FLOW_COLUMNS)}"
    )

    print(
        f"Flow files found       : "
        f"{len(flow_files)}"
    )

    results = []
    nan_results = []

    for location, flow_path in flow_files:

        result, nan_rows = check_flow_file(
            location,
            flow_path,
        )

        results.append(result)
        nan_results.extend(nan_rows)

    # --------------------------------------------------------
    # Save reports
    # --------------------------------------------------------

    report_df = pd.DataFrame(results)

    report_df.to_csv(
        REPORT_PATH,
        index=False,
    )

    nan_df = pd.DataFrame(nan_results)

    if not nan_df.empty:

        nan_df.to_csv(
            NAN_REPORT_PATH,
            index=False,
        )

    else:

        pd.DataFrame(
            columns=[
                "location",
                "flow_file",
                "feature",
                "nan_count",
                "total_rows",
                "nan_percent",
                "all_nan",
            ]
        ).to_csv(
            NAN_REPORT_PATH,
            index=False,
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("QUALITY CHECK SUMMARY")
    print("=" * 80)

    summary_columns = [
        "location",
        "flow_file",
        "status",
        "flow_rows",
        "column_count",
        "tcp_flows",
        "udp_flows",
        "positive_inf",
        "negative_inf",
        "duplicate_flow_ids",
        "packet_status",
    ]

    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_colwidth",
        55,
        "display.width",
        220,
    ):
        print(
            report_df[
                summary_columns
            ].to_string(index=False)
        )

    ok_count = int(
        (report_df["status"] == "OK").sum()
    )

    check_count = int(
        (report_df["status"] != "OK").sum()
    )

    total_flows = int(
        report_df["flow_rows"].sum()
    )

    print()
    print(f"Files checked : {len(report_df)}")
    print(f"OK            : {ok_count}")
    print(f"Need review   : {check_count}")
    print(f"Total flows   : {total_flows:,}")

    print()
    print("Reports saved:")
    print(f"  {REPORT_PATH}")
    print(f"  {NAN_REPORT_PATH}")

    print()

    if check_count > 0:

        print(
            "Some files require review. "
            "Check quality_check_report.csv."
        )

        return 2

    print(
        "All checked files passed "
        "the critical quality checks."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())