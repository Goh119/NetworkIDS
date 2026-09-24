"""
NetworkIDS - Integrity Rule Baseline Analysis.

Purpose:
    Evaluate candidate packet-header integrity rules against the
    complete packet_features.csv dataset before finalizing the
    Integrity Checker.

This script DOES NOT assign final Intact/Suspicious/Tampered states.

It only:
    1. Processes packet_features.csv in chunks.
    2. Counts how often each candidate rule is triggered.
    3. Counts overlaps between rules.
    4. Saves representative examples for each rule.
    5. Reports the overall rule-trigger frequency.

The final severity of each rule will be decided after inspecting
these results.

Candidate rules
---------------

TAMPERED CANDIDATES:
    T1_invalid_ip_checksum
    T2_fragmentation_contradiction
    T3_invalid_ip_header_length

SUSPICIOUS CANDIDATES:
    S1_ttl_anomaly
    S2_mac_ip_inconsistency
    S3_syn_fin
    S4_syn_rst
    S5_fin_rst
    S6_null_tcp
    S7_invalid_l4_checksum

Usage:
    python src/analyze_integrity_rules.py

Optional:
    python src/analyze_integrity_rules.py --limit 1000000
"""

import argparse
import sys
import time
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import src.config as config


# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = config.PROCESSED_DIR / "integrity_analysis"

SUMMARY_FILE = OUTPUT_DIR / "integrity_rule_summary.txt"
RULE_COUNTS_FILE = OUTPUT_DIR / "integrity_rule_counts.csv"
OVERLAP_FILE = OUTPUT_DIR / "integrity_rule_overlaps.csv"
EXAMPLES_FILE = OUTPUT_DIR / "integrity_rule_examples.csv"

DEFAULT_CHUNK_SIZE = 1_000_000
EXAMPLES_PER_RULE = 10


# Candidate rule definitions.
#
# Severity here is ONLY the current hypothesis.
# It is NOT the final Integrity Checker decision.
RULE_METADATA = {
    "T1_invalid_ip_checksum": {
        "candidate_severity": "Tampered",
        "description": "IPv4 header checksum is invalid",
    },
    "T2_fragmentation_contradiction": {
        "candidate_severity": "Tampered",
        "description": "DF/MF or DF/fragment-offset fields contradict",
    },
    "T3_invalid_ip_header_length": {
        "candidate_severity": "Tampered",
        "description": "IPv4 header length outside valid range [20, 60]",
    },

    "S1_ttl_anomaly": {
        "candidate_severity": "Suspicious",
        "description": "TTL does not fit common initial-TTL heuristic windows",
    },
    "S2_mac_ip_inconsistency": {
        "candidate_severity": "Suspicious",
        "description": "Same source IP is observed with multiple source MACs",
    },
    "S3_syn_fin": {
        "candidate_severity": "Suspicious",
        "description": "TCP SYN and FIN are simultaneously set",
    },
    "S4_syn_rst": {
        "candidate_severity": "Suspicious",
        "description": "TCP SYN and RST are simultaneously set",
    },
    "S5_fin_rst": {
        "candidate_severity": "Suspicious",
        "description": "TCP FIN and RST are simultaneously set",
    },
    "S6_null_tcp": {
        "candidate_severity": "Suspicious",
        "description": "TCP packet has all six monitored flags cleared",
    },
    "S7_invalid_l4_checksum": {
        "candidate_severity": "Suspicious",
        "description": "TCP/UDP L4 checksum is invalid",
    },
}


ALL_RULES = list(RULE_METADATA.keys())


# Common initial TTL assumptions.
TTL_DEFAULTS = [64, 128, 255]
MAX_HOPS = 30


NEEDED_COLS = [
    "packet_id",
    "timestamp",
    "src_ip",
    "dst_ip",
    "src_mac",
    "dst_mac",

    "ip_version",
    "ip_protocol",
    "l4_protocol",

    "ttl",
    "ip_header_length",
    "ip_checksum_valid",
    "l4_checksum_valid",

    "ip_df",
    "ip_mf",
    "fragment_offset",

    "src_port",
    "dst_port",

    "tcp_syn",
    "tcp_ack",
    "tcp_fin",
    "tcp_rst",
    "tcp_psh",
    "tcp_urg",
]


# ============================================================
# TTL rule
# ============================================================

def ttl_anomaly_mask(ttl):
    """
    Flag TTL values that do not fit one of the common initial-TTL
    heuristic windows:

        64  -> 34..64
        128 -> 98..128
        255 -> 225..255

    This is a heuristic anomaly detector, NOT proof of tampering.
    """

    ttl = pd.to_numeric(ttl, errors="coerce")

    valid_ttl = ttl.notna()

    in_expected_range = pd.Series(
        False,
        index=ttl.index,
    )

    for default in TTL_DEFAULTS:
        in_expected_range |= (
            (ttl <= default)
            & (ttl >= default - MAX_HOPS)
        )

    return valid_ttl & ~in_expected_range


# ============================================================
# Rule application
# ============================================================

def apply_rules(chunk, mac_ip_map):
    """
    Apply all candidate rules to one packet chunk.

    Returns:
        dict[str, np.ndarray]
            Boolean mask for each rule.
    """

    # --------------------------------------------------------
    # Basic protocol masks
    # --------------------------------------------------------

    ip_version = chunk["ip_version"]

    ip_protocol = chunk["ip_protocol"]

    is_ipv4 = ip_version == 4
    is_tcp = ip_protocol == 6
    is_udp = ip_protocol == 17

    is_tcp_or_udp = is_tcp | is_udp

    # --------------------------------------------------------
    # TCP flags
    # --------------------------------------------------------

    syn = chunk["tcp_syn"].fillna(0) == 1
    ack = chunk["tcp_ack"].fillna(0) == 1
    fin = chunk["tcp_fin"].fillna(0) == 1
    rst = chunk["tcp_rst"].fillna(0) == 1
    psh = chunk["tcp_psh"].fillna(0) == 1
    urg = chunk["tcp_urg"].fillna(0) == 1

    # --------------------------------------------------------
    # T1: Invalid IPv4 checksum
    # --------------------------------------------------------

    ip_checksum = chunk["ip_checksum_valid"]

    t1 = (
        is_ipv4
        & ip_checksum.notna()
        & (ip_checksum == 0)
    )

    # --------------------------------------------------------
    # T2: Fragmentation contradiction
    #
    # Case A:
    #   DF = 1 and MF = 1
    #
    # Case B:
    #   DF = 1 and fragment_offset > 0
    # --------------------------------------------------------

    df_flag = chunk["ip_df"]
    mf_flag = chunk["ip_mf"]
    fragment_offset = chunk["fragment_offset"]

    t2 = (
        is_ipv4
        & (
            (
                (df_flag == 1)
                & (mf_flag == 1)
            )
            |
            (
                (df_flag == 1)
                & fragment_offset.notna()
                & (fragment_offset > 0)
            )
        )
    )

    # --------------------------------------------------------
    # T3: Invalid IPv4 header length
    #
    # Valid IPv4 header length:
    #   20 to 60 bytes
    # --------------------------------------------------------

    ip_header_length = chunk["ip_header_length"]

    t3 = (
        is_ipv4
        & ip_header_length.notna()
        & (
            (ip_header_length < 20)
            | (ip_header_length > 60)
        )
    )

    # --------------------------------------------------------
    # S1: TTL anomaly
    # --------------------------------------------------------

    s1 = ttl_anomaly_mask(chunk["ttl"])

    # --------------------------------------------------------
    # S2: MAC-IP inconsistency
    #
    # The same source IP appears with multiple source MACs.
    #
    # mac_ip_map is built across the COMPLETE dataset before
    # rule evaluation so this remains a dataset-wide rule.
    # --------------------------------------------------------

    source_ip = chunk["src_ip"]

    s2 = source_ip.map(mac_ip_map).fillna(False)

    # --------------------------------------------------------
    # S3: SYN + FIN
    # --------------------------------------------------------

    s3 = is_tcp & syn & fin

    # --------------------------------------------------------
    # S4: SYN + RST
    # --------------------------------------------------------

    s4 = is_tcp & syn & rst

    # --------------------------------------------------------
    # S5: FIN + RST
    # --------------------------------------------------------

    s5 = is_tcp & fin & rst

    # --------------------------------------------------------
    # S6: NULL TCP
    #
    # All six monitored TCP flags are zero.
    # --------------------------------------------------------

    no_tcp_flags = (
        (~syn)
        & (~ack)
        & (~fin)
        & (~rst)
        & (~psh)
        & (~urg)
    )

    s6 = is_tcp & no_tcp_flags

    # --------------------------------------------------------
    # S7: Invalid TCP/UDP checksum
    # --------------------------------------------------------

    l4_checksum = chunk["l4_checksum_valid"]

    s7 = (
        is_tcp_or_udp
        & l4_checksum.notna()
        & (l4_checksum == 0)
    )

    return {
        "T1_invalid_ip_checksum": t1.to_numpy(dtype=bool),
        "T2_fragmentation_contradiction": t2.to_numpy(dtype=bool),
        "T3_invalid_ip_header_length": t3.to_numpy(dtype=bool),

        "S1_ttl_anomaly": s1.to_numpy(dtype=bool),
        "S2_mac_ip_inconsistency": s2.to_numpy(dtype=bool),
        "S3_syn_fin": s3.to_numpy(dtype=bool),
        "S4_syn_rst": s4.to_numpy(dtype=bool),
        "S5_fin_rst": s5.to_numpy(dtype=bool),
        "S6_null_tcp": s6.to_numpy(dtype=bool),
        "S7_invalid_l4_checksum": s7.to_numpy(dtype=bool),
    }


# ============================================================
# Build dataset-wide MAC/IP map
# ============================================================

def build_mac_ip_map(input_path, chunksize):
    """
    Build:

        source IP -> number of distinct source MACs

    across the complete packet capture.

    Returns:
        dict:
            src_ip -> True/False

    True means the source IP was observed with >1 source MAC.
    """

    print("\n[1/4] Building source IP -> source MAC consistency map...")
    print("  This requires one pass through the packet dataset.")

    ip_to_macs = {}

    reader = pd.read_csv(
        input_path,
        usecols=["src_ip", "src_mac"],
        chunksize=chunksize,
        low_memory=False,
    )

    total = 0

    for chunk_idx, chunk in enumerate(reader, start=1):

        grouped = (
            chunk.dropna(subset=["src_ip", "src_mac"])
            .groupby("src_ip")["src_mac"]
            .unique()
        )

        for src_ip, macs in grouped.items():

            if src_ip not in ip_to_macs:
                ip_to_macs[src_ip] = set()

            ip_to_macs[src_ip].update(macs)

        total += len(chunk)

        print(
            f"  chunk {chunk_idx:>3} | "
            f"processed {total:>12,} rows",
            end="\r",
        )

    print()

    inconsistent = {
        ip: len(macs) > 1
        for ip, macs in ip_to_macs.items()
    }

    inconsistent_count = sum(inconsistent.values())

    print(
        f"  Unique source IPs        : "
        f"{len(ip_to_macs):,}"
    )

    print(
        f"  IPs with multiple MACs   : "
        f"{inconsistent_count:,}"
    )

    return inconsistent


# ============================================================
# Main analysis
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Analyze candidate integrity rules."
    )

    parser.add_argument(
        "--input",
        default=str(config.PACKET_FEATURES_FILE),
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional packet limit for testing.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("NetworkIDS - Integrity Rule Baseline Analysis")
    print("=" * 70)

    print(f"Input     : {input_path}")
    print(f"Chunk size: {args.chunksize:,}")
    print(
        f"Limit     : "
        f"{args.limit:,}"
        if args.limit
        else "Limit     : whole dataset"
    )

    start_time = time.time()

    # --------------------------------------------------------
    # Pass 1:
    # Build MAC/IP consistency map
    # --------------------------------------------------------

    mac_ip_map = build_mac_ip_map(
        input_path,
        args.chunksize,
    )

    # --------------------------------------------------------
    # Pass 2:
    # Apply all rules
    # --------------------------------------------------------

    print("\n[2/4] Applying candidate integrity rules...")

    rule_counts = {
        rule: 0
        for rule in ALL_RULES
    }

    overlap_counts = {}

    examples = {
        rule: []
        for rule in ALL_RULES
    }

    total_packets = 0

    reader = pd.read_csv(
        input_path,
        usecols=NEEDED_COLS,
        chunksize=args.chunksize,
        low_memory=False,
    )

    for chunk_idx, chunk in enumerate(
        reader,
        start=1,
    ):

        if args.limit is not None:

            remaining = args.limit - total_packets

            if remaining <= 0:
                break

            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]

        masks = apply_rules(
            chunk,
            mac_ip_map,
        )

        # ----------------------------------------------------
        # Count individual rules
        # ----------------------------------------------------

        for rule, mask in masks.items():

            count = int(mask.sum())

            rule_counts[rule] += count

            # Save up to N representative examples.
            if (
                count > 0
                and len(examples[rule])
                < EXAMPLES_PER_RULE
            ):

                positions = np.flatnonzero(mask)

                slots = (
                    EXAMPLES_PER_RULE
                    - len(examples[rule])
                )

                for pos in positions[:slots]:

                    row = chunk.iloc[pos]

                    examples[rule].append({
                        "packet_id": row["packet_id"],
                        "timestamp": row["timestamp"],
                        "src_ip": row["src_ip"],
                        "dst_ip": row["dst_ip"],
                        "src_mac": row["src_mac"],
                        "dst_mac": row["dst_mac"],
                        "ip_version": row["ip_version"],
                        "ip_protocol": row["ip_protocol"],
                        "ttl": row["ttl"],
                        "ip_header_length": row[
                            "ip_header_length"
                        ],
                        "ip_checksum_valid": row[
                            "ip_checksum_valid"
                        ],
                        "l4_checksum_valid": row[
                            "l4_checksum_valid"
                        ],
                        "ip_df": row["ip_df"],
                        "ip_mf": row["ip_mf"],
                        "fragment_offset": row[
                            "fragment_offset"
                        ],
                        "src_port": row["src_port"],
                        "dst_port": row["dst_port"],
                        "tcp_syn": row["tcp_syn"],
                        "tcp_ack": row["tcp_ack"],
                        "tcp_fin": row["tcp_fin"],
                        "tcp_rst": row["tcp_rst"],
                        "tcp_psh": row["tcp_psh"],
                        "tcp_urg": row["tcp_urg"],
                        "rule": rule,
                    })

        # ----------------------------------------------------
        # Count rule overlaps
        # ----------------------------------------------------

        fired_rules = np.column_stack(
            [
                masks[rule]
                for rule in ALL_RULES
            ]
        )

        # Only rows triggering >= 2 rules.
        multi_rule_mask = (
            fired_rules.sum(axis=1) >= 2
        )

        if multi_rule_mask.any():

            multi_positions = np.flatnonzero(
                multi_rule_mask
            )

            for pos in multi_positions:

                fired = [
                    ALL_RULES[i]
                    for i, fired_flag
                    in enumerate(
                        fired_rules[pos]
                    )
                    if fired_flag
                ]

                for pair in combinations(
                    sorted(fired),
                    2,
                ):

                    overlap_counts[pair] = (
                        overlap_counts.get(pair, 0)
                        + 1
                    )

        total_packets += len(chunk)

        elapsed = time.time() - start_time

        rate = (
            total_packets / elapsed
            if elapsed > 0
            else 0
        )

        print(
            f"  chunk {chunk_idx:>3} | "
            f"total {total_packets:>12,} | "
            f"{rate:>10,.0f} rows/s"
        )

        if (
            args.limit is not None
            and total_packets >= args.limit
        ):
            break

    # --------------------------------------------------------
    # Rule count table
    # --------------------------------------------------------

    print("\n[3/4] Building summary tables...")

    rule_rows = []

    for rule in ALL_RULES:

        count = rule_counts[rule]

        percentage = (
            count / total_packets * 100
            if total_packets > 0
            else 0
        )

        metadata = RULE_METADATA[rule]

        rule_rows.append({
            "rule": rule,
            "candidate_severity":
                metadata["candidate_severity"],
            "count": count,
            "percentage": percentage,
            "description":
                metadata["description"],
        })

    rule_df = pd.DataFrame(rule_rows)

    rule_df = rule_df.sort_values(
        "count",
        ascending=False,
    )

    rule_df.to_csv(
        RULE_COUNTS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Overlap table
    # --------------------------------------------------------

    overlap_rows = []

    for pair, count in sorted(
        overlap_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):

        percentage = (
            count / total_packets * 100
            if total_packets > 0
            else 0
        )

        overlap_rows.append({
            "rule_a": pair[0],
            "rule_b": pair[1],
            "overlap_count": count,
            "percentage_of_packets": percentage,
        })

    overlap_df = pd.DataFrame(
        overlap_rows,
        columns=[
            "rule_a",
            "rule_b",
            "overlap_count",
            "percentage_of_packets",
        ],
    )

    overlap_df.to_csv(
        OVERLAP_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Examples
    # --------------------------------------------------------

    example_rows = []

    for rule in ALL_RULES:

        example_rows.extend(
            examples[rule]
        )

    examples_df = pd.DataFrame(
        example_rows
    )

    examples_df.to_csv(
        EXAMPLES_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    elapsed = time.time() - start_time

    print("\n[4/4] Writing summary...")

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "NetworkIDS - Integrity Rule "
            "Baseline Analysis\n"
        )

        f.write("=" * 70 + "\n\n")

        f.write(
            f"Input file       : {input_path}\n"
        )

        f.write(
            f"Packets analyzed : "
            f"{total_packets:,}\n"
        )

        f.write(
            f"Elapsed seconds  : "
            f"{elapsed:.1f}\n"
        )

        f.write(
            f"Chunk size       : "
            f"{args.chunksize:,}\n\n"
        )

        f.write(
            "RULE FREQUENCY\n"
        )

        f.write("-" * 70 + "\n")

        for _, row in rule_df.iterrows():

            f.write(
                f"{row['rule']:<32} "
                f"{int(row['count']):>12,} "
                f"({row['percentage']:>8.4f}%) "
                f"[{row['candidate_severity']}]\n"
            )

        f.write("\n")

        f.write(
            "TOP RULE OVERLAPS\n"
        )

        f.write("-" * 70 + "\n")

        if len(overlap_df) == 0:

            f.write(
                "No multi-rule overlaps detected.\n"
            )

        else:

            for _, row in overlap_df.head(30).iterrows():

                f.write(
                    f"{row['rule_a']} + "
                    f"{row['rule_b']} : "
                    f"{int(row['overlap_count']):,} "
                    f"({row['percentage_of_packets']:.4f}%)\n"
                )

    # --------------------------------------------------------
    # Terminal summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("INTEGRITY RULE ANALYSIS - COMPLETE")
    print("=" * 70)

    print(
        f"Packets analyzed: "
        f"{total_packets:,}"
    )

    print("\nRule frequency:")

    for _, row in rule_df.iterrows():

        print(
            f"  {row['rule']:<32} "
            f"{int(row['count']):>12,} "
            f"({row['percentage']:>7.3f}%)"
        )

    print("\nTop rule overlaps:")

    if len(overlap_df) == 0:

        print("  None")

    else:

        for _, row in overlap_df.head(10).iterrows():

            print(
                f"  {row['rule_a']} + "
                f"{row['rule_b']} : "
                f"{int(row['overlap_count']):,}"
            )

    print("\nOutput files:")

    print(
        f"  Rule counts : "
        f"{RULE_COUNTS_FILE}"
    )

    print(
        f"  Overlaps    : "
        f"{OVERLAP_FILE}"
    )

    print(
        f"  Examples    : "
        f"{EXAMPLES_FILE}"
    )

    print(
        f"  Summary     : "
        f"{SUMMARY_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()