"""
NetworkIDS - Packet Integrity Checker

Processes one or more packet-level feature CSV files sequentially.
No intermediate combined packet-feature CSV is created.

Output:
    <project root>/integrity_checker/
        packet_integrity.csv
        integrity_metadata.json
        integrity_summary.txt
        integrity_rule_counts.csv
        integrity_examples.csv

Rules:
    T1-T3: Strong protocol/header integrity violations
    S1-S6: Suspicious packet/header patterns
    S7:    Invalid L4 checksum indicator only

S1 definition:
    Detect IPv4 packets with TTL between 1 and 5 inclusive,
    excluding multicast destinations. This is a heuristic indicator,
    not proof of packet tampering.

S2 exceptions:
    Exclude source address 0.0.0.0 for IPv4 and :: for IPv6
    from MAC-IP consistency evaluation.

State priority:
    Tampered > Suspicious > Intact
"""

from pathlib import Path
import argparse
import json
import sys

import pandas as pd


# ============================================================
# Project import
# ============================================================

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

import src.config as config


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = config.PACKET_FEATURES_FILE

INTEGRITY_DIR = (
    Path(__file__).resolve().parents[1]
    / "integrity_checker"
)

OUTPUT_FILE = INTEGRITY_DIR / "packet_integrity.csv"
METADATA_FILE = INTEGRITY_DIR / "integrity_metadata.json"
SUMMARY_FILE = INTEGRITY_DIR / "integrity_summary.txt"
RULE_COUNTS_FILE = INTEGRITY_DIR / "integrity_rule_counts.csv"
EXAMPLES_FILE = INTEGRITY_DIR / "integrity_examples.csv"

CHUNK_SIZE = 1_000_000
MAX_EXAMPLES_PER_RULE = 5


# ============================================================
# Required packet-level columns
# ============================================================

NEEDED_COLUMNS = [
    "packet_id",
    "timestamp",
    "src_ip",
    "dst_ip",
    "src_mac",
    "dst_mac",

    # Layer 3
    "ip_version",
    "ip_protocol",
    "l4_protocol",
    "ttl",
    "ip_header_length",
    "ip_checksum_valid",
    "ip_df",
    "ip_mf",
    "fragment_offset",

    # MAC-IP consistency fields from packet_parser.py
    "src_ip_mac_consistent",
    "src_ip_mac_seen_mac_count",
    "src_mac_ip_seen_ip_count",
    "src_mac_ip_spoof_suspect",

    # Layer 4
    "src_port",
    "dst_port",
    "l4_checksum_valid",

    # TCP flags
    "tcp_syn",
    "tcp_ack",
    "tcp_fin",
    "tcp_rst",
    "tcp_psh",
    "tcp_urg",
]


# ============================================================
# Rule definitions
# ============================================================

TAMPERED_RULES = [
    "T1_invalid_ip_checksum",
    "T2_fragmentation_contradiction",
    "T3_invalid_ip_header_length",
]

SUSPICIOUS_RULES = [
    "S1_ttl_anomaly",
    "S2_mac_ip_inconsistency",
    "S3_syn_fin",
    "S4_syn_rst",
    "S5_fin_rst",
    "S6_null_tcp",
]

INDICATOR_RULES = [
    "S7_invalid_l4_checksum",
]

ALL_RULES = (
    TAMPERED_RULES
    + SUSPICIOUS_RULES
    + INDICATOR_RULES
)

STATES = [
    "Intact",
    "Suspicious",
    "Tampered",
]


# ============================================================
# Utility functions
# ============================================================

def ensure_output_directory():
    """Create the integrity checker output directory."""
    INTEGRITY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def new_state_counts():
    """Create zeroed integrity-state counters."""
    return {
        state: 0
        for state in STATES
    }


def new_rule_counts():
    """Create zeroed rule counters."""
    return {
        rule: 0
        for rule in ALL_RULES
    }


def new_example_store():
    """Create per-rule example storage."""
    return {
        rule: []
        for rule in ALL_RULES
    }


def validate_input(path):
    """
    Check that an input file exists and includes every required
    packet-level feature column.
    """

    if not path.exists():
        raise FileNotFoundError(
            "Packet feature file not found:\n"
            f"{path}"
        )

    header = pd.read_csv(
        path,
        nrows=0,
    )

    missing_columns = [
        column
        for column in NEEDED_COLUMNS
        if column not in header.columns
    ]

    if missing_columns:
        raise ValueError(
            "Packet feature file is missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_columns
            )
            + f"\n\nInput file:\n{path}"
        )

    return len(header.columns)


# ============================================================
# Integrity rules
# ============================================================

def apply_rules(chunk):
    """
    Apply packet/header integrity rules to one DataFrame chunk.

    State priority:
        Tampered > Suspicious > Intact

    S7 is an indicator only and does not independently
    change integrity_state.
    """

    # --------------------------------------------------------
    # Normalize protocol and address fields
    # --------------------------------------------------------

    ip_version = pd.to_numeric(
        chunk["ip_version"],
        errors="coerce",
    )

    is_ipv4 = (
        ip_version.eq(4)
    ).fillna(False)

    is_ipv6 = (
        ip_version.eq(6)
    ).fillna(False)

    l4_protocol = (
        chunk["l4_protocol"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    is_tcp = (
        chunk["ip_protocol"].eq(6)
        | l4_protocol.eq("TCP")
    ).fillna(False)

    is_tcp_udp = (
        l4_protocol.isin(["TCP", "UDP"])
    ).fillna(False)

    src_ip_text = (
        chunk["src_ip"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    dst_ip_text = (
        chunk["dst_ip"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------------
    # Traffic scope exceptions
    # --------------------------------------------------------

    # IPv4 multicast destinations: 224.0.0.0/4.
    # These addresses should not trigger S1.
    is_ipv4_multicast = (
        is_ipv4
        & dst_ip_text.str.match(
            r"^(22[4-9]|23[0-9])\.",
            na=False,
        )
    ).fillna(False)

    # IPv6 multicast destinations: ff00::/8.
    is_ipv6_multicast = (
        is_ipv6
        & dst_ip_text.str.startswith(
            "ff",
            na=False,
        )
    ).fillna(False)

    is_multicast_destination = (
        is_ipv4_multicast
        | is_ipv6_multicast
    ).fillna(False)

    # IPv4 unspecified source address: 0.0.0.0.
    is_ipv4_unspecified_source = (
        is_ipv4
        & src_ip_text.eq("0.0.0.0")
    ).fillna(False)

    # IPv6 unspecified source address: ::.
    is_ipv6_unspecified_source = (
        is_ipv6
        & src_ip_text.eq("::")
    ).fillna(False)

    is_unspecified_source = (
        is_ipv4_unspecified_source
        | is_ipv6_unspecified_source
    ).fillna(False)

    # ========================================================
    # T1: Invalid IPv4 checksum
    # ========================================================

    t1 = (
        is_ipv4
        & chunk["ip_checksum_valid"].eq(0)
    ).fillna(False)

    # ========================================================
    # T2: Fragmentation contradiction
    #
    # DF + MF is contradictory.
    # DF + non-zero fragment offset is contradictory.
    # ========================================================

    t2 = (
        is_ipv4
        & (
            (
                chunk["ip_df"].eq(1)
                & chunk["ip_mf"].eq(1)
            )
            |
            (
                chunk["ip_df"].eq(1)
                & chunk["fragment_offset"].gt(0)
            )
        )
    ).fillna(False)

    # ========================================================
    # T3: Invalid IPv4 header length
    #
    # Valid IPv4 header length: 20-60 bytes.
    # ========================================================

    ip_header_length = pd.to_numeric(
        chunk["ip_header_length"],
        errors="coerce",
    )

    t3 = (
        is_ipv4
        & (
            ip_header_length.lt(20)
            | ip_header_length.gt(60)
        )
    ).fillna(False)

    # ========================================================
    # S1: Extremely low IPv4 TTL
    #
    # Trigger only when:
    #   1. Packet is IPv4.
    #   2. TTL is between 1 and 5 inclusive.
    #   3. Destination is not multicast.
    #
    # This is a conservative heuristic indicator. It does not
    # establish that a packet was tampered with or is malicious.
    # ========================================================

    ttl = pd.to_numeric(
        chunk["ttl"],
        errors="coerce",
    )

    s1 = (
        is_ipv4
        & ttl.notna()
        & ttl.between(1, 5)
        & ~is_multicast_destination
    ).fillna(False)

    # ========================================================
    # S2: MAC-IP inconsistency
    #
    # Use packet-level consistency fields generated by
    # packet_parser.py.
    #
    # Exclude:
    #   IPv4 source 0.0.0.0
    #   IPv6 source ::
    #
    # These source addresses are not treated as normal host
    # identities for this MAC-IP consistency rule.
    # ========================================================

    src_ip_mac_consistent = pd.to_numeric(
        chunk["src_ip_mac_consistent"],
        errors="coerce",
    )

    src_mac_ip_spoof_suspect = pd.to_numeric(
        chunk["src_mac_ip_spoof_suspect"],
        errors="coerce",
    )

    raw_s2 = (
        src_ip_mac_consistent.eq(0)
        | src_mac_ip_spoof_suspect.eq(1)
    ).fillna(False)

    s2 = (
        raw_s2
        & ~is_unspecified_source
    ).fillna(False)

    # --------------------------------------------------------
    # TCP flag masks
    # --------------------------------------------------------

    syn = (
        chunk["tcp_syn"].eq(1)
    ).fillna(False)

    ack = (
        chunk["tcp_ack"].eq(1)
    ).fillna(False)

    fin = (
        chunk["tcp_fin"].eq(1)
    ).fillna(False)

    rst = (
        chunk["tcp_rst"].eq(1)
    ).fillna(False)

    psh = (
        chunk["tcp_psh"].eq(1)
    ).fillna(False)

    urg = (
        chunk["tcp_urg"].eq(1)
    ).fillna(False)

    # ========================================================
    # S3: SYN + FIN
    # ========================================================

    s3 = (
        is_tcp
        & syn
        & fin
    ).fillna(False)

    # ========================================================
    # S4: SYN + RST
    # ========================================================

    s4 = (
        is_tcp
        & syn
        & rst
    ).fillna(False)

    # ========================================================
    # S5: FIN + RST
    # ========================================================

    s5 = (
        is_tcp
        & fin
        & rst
    ).fillna(False)

    # ========================================================
    # S6: TCP NULL flag pattern
    #
    # All six monitored TCP flags are zero.
    # ========================================================

    s6 = (
        is_tcp
        & ~syn
        & ~ack
        & ~fin
        & ~rst
        & ~psh
        & ~urg
    ).fillna(False)

    # ========================================================
    # S7: Invalid L4 checksum
    #
    # Indicator only. Does not independently change state.
    # ========================================================

    s7 = (
        is_tcp_udp
        & chunk["l4_checksum_valid"].eq(0)
    ).fillna(False)

    # --------------------------------------------------------
    # Store rule columns
    # --------------------------------------------------------

    rule_values = {
        "T1_invalid_ip_checksum": t1,
        "T2_fragmentation_contradiction": t2,
        "T3_invalid_ip_header_length": t3,
        "S1_ttl_anomaly": s1,
        "S2_mac_ip_inconsistency": s2,
        "S3_syn_fin": s3,
        "S4_syn_rst": s4,
        "S5_fin_rst": s5,
        "S6_null_tcp": s6,
        "S7_invalid_l4_checksum": s7,
    }

    for column, values in rule_values.items():
        chunk[column] = values.astype("int8")

    # --------------------------------------------------------
    # Per-packet rule counts
    # --------------------------------------------------------

    tampered_count = (
        t1.astype("int8")
        + t2.astype("int8")
        + t3.astype("int8")
    )

    suspicious_count = (
        s1.astype("int8")
        + s2.astype("int8")
        + s3.astype("int8")
        + s4.astype("int8")
        + s5.astype("int8")
        + s6.astype("int8")
    )

    chunk["tampered_rule_count"] = (
        tampered_count.astype("int8")
    )

    chunk["suspicious_rule_count"] = (
        suspicious_count.astype("int8")
    )

    # --------------------------------------------------------
    # Integrity state
    #
    # Priority:
    #   Tampered > Suspicious > Intact
    #
    # S7 does not affect this state.
    # --------------------------------------------------------

    chunk["integrity_state"] = "Intact"

    chunk.loc[
        suspicious_count.gt(0),
        "integrity_state",
    ] = "Suspicious"

    chunk.loc[
        tampered_count.gt(0),
        "integrity_state",
    ] = "Tampered"

    # --------------------------------------------------------
    # Integrity reasons
    #
    # S7 is excluded because it is an indicator only.
    # --------------------------------------------------------

    reason_columns = [
        (
            "T1_invalid_ip_checksum",
            "Invalid IPv4 checksum",
        ),
        (
            "T2_fragmentation_contradiction",
            "Fragmentation contradiction",
        ),
        (
            "T3_invalid_ip_header_length",
            "Invalid IPv4 header length",
        ),
        (
            "S1_ttl_anomaly",
            "Extremely low IPv4 TTL (1-5)",
        ),
        (
            "S2_mac_ip_inconsistency",
            "MAC-IP consistency violation",
        ),
        (
            "S3_syn_fin",
            "TCP SYN and FIN flags set together",
        ),
        (
            "S4_syn_rst",
            "TCP SYN and RST flags set together",
        ),
        (
            "S5_fin_rst",
            "TCP FIN and RST flags set together",
        ),
        (
            "S6_null_tcp",
            "TCP NULL flag pattern",
        ),
    ]

    reasons = pd.Series(
        "",
        index=chunk.index,
        dtype="object",
    )

    for column, description in reason_columns:
        mask = chunk[column].eq(1)

        reasons.loc[mask] = (
            reasons.loc[mask]
            + description
            + "; "
        )

    chunk["integrity_reasons"] = (
        reasons
        .str.rstrip("; ")
        .replace("", "None")
    )

    # --------------------------------------------------------
    # Weak indicators
    # --------------------------------------------------------

    indicators = pd.Series(
        "",
        index=chunk.index,
        dtype="object",
    )

    indicators.loc[
        chunk["S7_invalid_l4_checksum"].eq(1)
    ] = "Invalid L4 checksum"

    chunk["integrity_indicators"] = (
        indicators.replace("", "None")
    )

    return chunk


# ============================================================
# Example collection
# ============================================================

def collect_rule_examples(
    chunk,
    examples_per_rule,
):
    """
    Collect up to MAX_EXAMPLES_PER_RULE examples for every rule.

    A packet may appear more than once if it triggers multiple
    rules. The example_rule column identifies the rule being shown.
    """

    example_columns = [
        "source_file",
        "packet_id",
        "timestamp",
        "src_ip",
        "dst_ip",
        "src_mac",
        "dst_mac",
        "integrity_state",
        "tampered_rule_count",
        "suspicious_rule_count",
        "integrity_reasons",
        "integrity_indicators",
    ] + ALL_RULES

    for rule in ALL_RULES:
        current_count = len(
            examples_per_rule[rule]
        )

        if current_count >= MAX_EXAMPLES_PER_RULE:
            continue

        mask = chunk[rule].eq(1)

        if not mask.any():
            continue

        remaining = (
            MAX_EXAMPLES_PER_RULE
            - current_count
        )

        selected = (
            chunk.loc[
                mask,
                example_columns,
            ]
            .head(remaining)
            .copy()
        )

        if selected.empty:
            continue

        selected.insert(
            0,
            "example_rule",
            rule,
        )

        examples_per_rule[rule].extend(
            selected.to_dict("records")
        )


def flatten_examples(examples_per_rule):
    """Flatten the per-rule sample store into one list."""
    examples = []

    for rule in ALL_RULES:
        examples.extend(
            examples_per_rule[rule]
        )

    return examples


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run the NetworkIDS packet integrity checker "
            "on one or more packet feature CSV files."
        )
    )

    parser.add_argument(
        "--input",
        nargs="+",
        default=None,
        help=(
            "One or more packet feature CSV files. "
            "Defaults to config.PACKET_FEATURES_FILE."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum total number of packets to process "
            "across all input files."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Resolve input paths
    # --------------------------------------------------------

    input_files = (
        [Path(path) for path in args.input]
        if args.input
        else [Path(INPUT_FILE)]
    )

    if args.limit is not None and args.limit <= 0:
        raise ValueError(
            "--limit must be greater than 0."
        )

    ensure_output_directory()

    print("=" * 70)
    print("NetworkIDS - Packet Integrity Checker")
    print("=" * 70)

    print(
        f"Input files: {len(input_files)}"
    )

    for index, path in enumerate(
        input_files,
        start=1,
    ):
        print(
            f"  [{index}] {path}"
        )

    print(
        f"Chunk size: {CHUNK_SIZE:,}"
    )

    print(
        f"Output    : {INTEGRITY_DIR}"
    )

    print(
        "Examples  : "
        f"{MAX_EXAMPLES_PER_RULE} per rule "
        f"({len(ALL_RULES) * MAX_EXAMPLES_PER_RULE} maximum)"
    )

    print(
        "S1 rule   : IPv4 TTL 1-5; multicast destinations excluded"
    )

    print(
        "S2 rule   : unspecified source addresses excluded"
    )

    if args.limit is not None:
        print(
            f"Limit     : {args.limit:,} total packets"
        )
    else:
        print(
            "Limit     : whole input set"
        )

    print()

    # --------------------------------------------------------
    # Step 1: Validate every input before replacing outputs
    # --------------------------------------------------------

    print(
        "[1/3] Validating packet feature inputs..."
    )

    for index, path in enumerate(
        input_files,
        start=1,
    ):
        column_count = validate_input(path)

        print(
            f"  [{index}/{len(input_files)}] "
            f"{path.name} | "
            f"columns: {column_count} | "
            f"schema: OK"
        )

    # --------------------------------------------------------
    # Initialize aggregate counters
    # --------------------------------------------------------

    rule_counts = new_rule_counts()
    state_counts = new_state_counts()

    processed = 0
    chunk_number = 0

    files_processed = []
    per_file_stats = {}

    examples_per_rule = new_example_store()

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    first_write = True
    stop_processing = False

    # --------------------------------------------------------
    # Step 2: Process files sequentially in chunks
    # --------------------------------------------------------

    print()
    print(
        "[2/3] Applying integrity rules..."
    )

    for file_index, input_path in enumerate(
        input_files,
        start=1,
    ):

        if stop_processing:
            break

        file_processed = 0
        file_rule_counts = new_rule_counts()
        file_state_counts = new_state_counts()

        print()
        print(
            f"--- File {file_index}/{len(input_files)}: "
            f"{input_path.name} ---"
        )

        reader = pd.read_csv(
            input_path,
            usecols=NEEDED_COLUMNS,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        )

        for chunk in reader:

            # -----------------------------------------------
            # Respect the total packet limit
            # -----------------------------------------------

            if args.limit is not None:
                remaining = args.limit - processed

                if remaining <= 0:
                    stop_processing = True
                    break

                if len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining]

            if chunk.empty:
                break

            # Keep input provenance because packet_id can repeat.
            chunk.insert(
                0,
                "source_file",
                input_path.name,
            )

            # Apply packet-level integrity rules.
            chunk = apply_rules(chunk)

            rows = len(chunk)

            processed += rows
            file_processed += rows
            chunk_number += 1

            # -----------------------------------------------
            # Update rule counters
            # -----------------------------------------------

            for rule in ALL_RULES:
                count = int(
                    chunk[rule].sum()
                )

                rule_counts[rule] += count
                file_rule_counts[rule] += count

            # -----------------------------------------------
            # Update state counters
            # -----------------------------------------------

            for state in STATES:
                count = int(
                    (
                        chunk["integrity_state"]
                        == state
                    ).sum()
                )

                state_counts[state] += count
                file_state_counts[state] += count

            # -----------------------------------------------
            # Collect rule-specific examples
            # -----------------------------------------------

            collect_rule_examples(
                chunk,
                examples_per_rule,
            )

            # -----------------------------------------------
            # Append output chunk
            # -----------------------------------------------

            chunk.to_csv(
                OUTPUT_FILE,
                mode=(
                    "w"
                    if first_write
                    else "a"
                ),
                header=first_write,
                index=False,
            )

            first_write = False

            print(
                f"  chunk {chunk_number:3d} | "
                f"file packets {file_processed:>12,} | "
                f"total {processed:>12,}"
            )

            if (
                args.limit is not None
                and processed >= args.limit
            ):
                stop_processing = True
                break

        # -----------------------------------------------
        # Per-file statistics
        # -----------------------------------------------

        files_processed.append(
            str(input_path.resolve())
        )

        per_file_stats[input_path.name] = {
            "input_file": str(
                input_path.resolve()
            ),
            "packets_processed": int(
                file_processed
            ),
            "state_counts": {
                state: int(
                    file_state_counts[state]
                )
                for state in STATES
            },
            "rule_counts": {
                rule: int(
                    file_rule_counts[rule]
                )
                for rule in ALL_RULES
            },
        }

        if (
            args.limit is not None
            and processed >= args.limit
        ):
            break

    # --------------------------------------------------------
    # Check processing result
    # --------------------------------------------------------

    if processed == 0:
        raise RuntimeError(
            "No packets were processed."
        )

    examples = flatten_examples(
        examples_per_rule
    )

    # --------------------------------------------------------
    # Aggregate rule counts
    # --------------------------------------------------------

    rule_rows = []

    for rule in ALL_RULES:

        if rule in TAMPERED_RULES:
            severity = "Tampered"

        elif rule in SUSPICIOUS_RULES:
            severity = "Suspicious"

        else:
            severity = "Indicator"

        count = rule_counts[rule]

        percentage = (
            count / processed * 100
            if processed
            else 0.0
        )

        rule_rows.append({
            "rule": rule,
            "severity": severity,
            "count": count,
            "percentage": percentage,
            "examples_stored": len(
                examples_per_rule[rule]
            ),
        })

    pd.DataFrame(
        rule_rows
    ).to_csv(
        RULE_COUNTS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Write per-rule examples
    # --------------------------------------------------------

    example_columns = [
        "example_rule",
        "source_file",
        "packet_id",
        "timestamp",
        "src_ip",
        "dst_ip",
        "src_mac",
        "dst_mac",
        "integrity_state",
        "tampered_rule_count",
        "suspicious_rule_count",
        "integrity_reasons",
        "integrity_indicators",
    ] + ALL_RULES

    pd.DataFrame(
        examples,
        columns=example_columns,
    ).to_csv(
        EXAMPLES_FILE,
        index=False,
    )

    # ========================================================
    # Summary report
    # ========================================================

    summary_lines = [
        "NetworkIDS - Packet Integrity Checker",
        "=" * 70,
        "",
        f"Input files       : {len(files_processed)}",
        f"Packets processed : {processed:,}",
        f"Chunk size        : {CHUNK_SIZE:,}",
        f"Examples per rule : {MAX_EXAMPLES_PER_RULE}",
        "",
        "INPUT FILES",
        "-" * 70,
    ]

    for index, path in enumerate(
        files_processed,
        start=1,
    ):
        summary_lines.append(
            f"[{index}] {path}"
        )

    # -----------------------------------------------
    # Per-file summary
    # -----------------------------------------------

    summary_lines.extend([
        "",
        "PER-FILE SUMMARY",
        "-" * 70,
    ])

    for filename, stats in per_file_stats.items():

        total = stats["packets_processed"]

        summary_lines.append(
            f"{filename}: {total:,} packets"
        )

        for state in STATES:
            count = stats["state_counts"][state]

            percentage = (
                count / total * 100
                if total
                else 0.0
            )

            summary_lines.append(
                f"    {state:<12}: "
                f"{count:,} "
                f"({percentage:.4f}%)"
            )

    # -----------------------------------------------
    # Aggregate state summary
    # -----------------------------------------------

    summary_lines.extend([
        "",
        "INTEGRITY STATE - AGGREGATE",
        "-" * 70,
    ])

    for state in STATES:

        count = state_counts[state]

        percentage = (
            count / processed * 100
            if processed
            else 0.0
        )

        summary_lines.append(
            f"{state:<12}: "
            f"{count:,} "
            f"({percentage:.4f}%)"
        )

    # -----------------------------------------------
    # Aggregate rule frequency
    # -----------------------------------------------

    summary_lines.extend([
        "",
        "RULE FREQUENCY - AGGREGATE",
        "-" * 70,
    ])

    for rule in ALL_RULES:

        count = rule_counts[rule]

        percentage = (
            count / processed * 100
            if processed
            else 0.0
        )

        if rule in TAMPERED_RULES:
            severity = "Tampered"

        elif rule in SUSPICIOUS_RULES:
            severity = "Suspicious"

        else:
            severity = "Indicator"

        summary_lines.append(
            f"{rule:<38} "
            f"{count:>10,} "
            f"({percentage:>8.4f}%) "
            f"[{severity}] "
            f"examples={len(examples_per_rule[rule])}"
        )

    # -----------------------------------------------
    # Example sampling
    # -----------------------------------------------

    summary_lines.extend([
        "",
        "EXAMPLE SAMPLING",
        "-" * 70,
        (
            f"Up to {MAX_EXAMPLES_PER_RULE} examples "
            "are retained for every rule."
        ),
        (
            "Examples are rule-specific rather than "
            "unique-packet samples."
        ),
        (
            "One packet may appear more than once if it "
            "triggers multiple rules."
        ),
        (
            "The example_rule column identifies the rule "
            "being illustrated."
        ),
        "",
    ])

    # -----------------------------------------------
    # S1 handling
    # -----------------------------------------------

    summary_lines.extend([
        "S1 TTL HANDLING",
        "-" * 70,
        "S1 detects IPv4 packets with TTL between 1 and 5 inclusive.",
        "IPv4 multicast destinations (224.0.0.0/4) are excluded.",
        "IPv6 multicast destinations (ff00::/8) are excluded.",
        "S1 is a heuristic indicator, not proof of packet tampering.",
        "",
    ])

    # -----------------------------------------------
    # S2 handling
    # -----------------------------------------------

    summary_lines.extend([
        "S2 MAC-IP HANDLING",
        "-" * 70,
        (
            "S2 uses packet-level consistency fields generated "
            "by packet_parser.py."
        ),
        "IPv4 source address 0.0.0.0 is excluded from S2.",
        "IPv6 source address :: is excluded from S2.",
        (
            "MAC-IP tracking is maintained within each "
            "capture-processing session."
        ),
        "Cross-day MAC changes are not evaluated.",
        "",
    ])

    # -----------------------------------------------
    # S7 handling
    # -----------------------------------------------

    summary_lines.extend([
        "S7 HANDLING",
        "-" * 70,
        "Invalid L4 checksum is an indicator only.",
        "S7 does not independently change integrity_state.",
        "",
    ])

    # -----------------------------------------------
    # State logic and interpretation
    # -----------------------------------------------

    summary_lines.extend([
        "STATE LOGIC",
        "-" * 70,
        "Tampered: at least one T1-T3 rule triggered.",
        (
            "Suspicious: no Tampered rule, but at least "
            "one S1-S6 rule triggered."
        ),
        "Intact: no T1-T3 or S1-S6 violations.",
        "S7 alone does not change the state.",
        "",
        "IMPORTANT INTERPRETATION",
        "-" * 70,
        (
            "Tampered represents a strong protocol/header "
            "integrity violation according to the defined rules."
        ),
        (
            "It is not definitive proof of malicious packet "
            "modification."
        ),
        (
            "S1 is a heuristic for extremely low IPv4 TTL values "
            "and is not a general-purpose TTL tampering detector."
        ),
        "",
    ])

    # -----------------------------------------------
    # Multi-file processing
    # -----------------------------------------------

    summary_lines.extend([
        "MULTI-FILE PROCESSING",
        "-" * 70,
        "Input files are processed sequentially in chunks.",
        "Aggregate statistics represent all processed files.",
        "No intermediate combined packet-feature CSV is created.",
        "",
    ])

    SUMMARY_FILE.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    # ========================================================
    # Metadata
    # ========================================================

    metadata = {
        "component": "Packet Integrity Checker",
        "input_mode": "multi_file_sequential",
        "input_files": files_processed,
        "output_directory": str(
            INTEGRITY_DIR.resolve()
        ),
        "packets_processed": int(processed),
        "files_processed": int(len(files_processed)),
        "chunk_size": int(CHUNK_SIZE),
        "limit": (
            int(args.limit)
            if args.limit is not None
            else None
        ),

        "max_examples_per_rule": int(
            MAX_EXAMPLES_PER_RULE
        ),
        "maximum_possible_examples": int(
            len(ALL_RULES) * MAX_EXAMPLES_PER_RULE
        ),

        "tampered_rules": TAMPERED_RULES,
        "suspicious_rules": SUSPICIOUS_RULES,
        "indicator_rules": INDICATOR_RULES,

        "state_priority": [
            "Tampered",
            "Suspicious",
            "Intact",
        ],

        "state_counts": {
            state: int(state_counts[state])
            for state in STATES
        },

        "rule_counts": {
            rule: int(rule_counts[rule])
            for rule in ALL_RULES
        },

        "examples_stored_per_rule": {
            rule: len(examples_per_rule[rule])
            for rule in ALL_RULES
        },

        "total_examples_stored": int(len(examples)),
        "per_file_stats": per_file_stats,

        "state_logic": {
            "Tampered": "At least one T1-T3 rule.",
            "Suspicious": (
                "No T1-T3 rule, but at least one S1-S6 rule."
            ),
            "Intact": "No T1-T3 or S1-S6 violations.",
            "S7": (
                "Invalid L4 checksum is an indicator only "
                "and does not independently change integrity_state."
            ),
        },

        "integrity_rule_exceptions": {
            "S1": [
                (
                    "Only IPv4 packets with TTL between "
                    "1 and 5 inclusive are flagged."
                ),
                (
                    "IPv4 multicast destinations "
                    "(224.0.0.0/4) are excluded."
                ),
                (
                    "IPv6 multicast destinations "
                    "(ff00::/8) are excluded."
                ),
            ],
            "S2": [
                (
                    "IPv4 source address 0.0.0.0 is excluded."
                ),
                (
                    "IPv6 source address :: is excluded."
                ),
            ],
        },

        "mac_ip_integrity_source": (
            "Packet-level MAC-IP consistency fields "
            "generated by packet_parser.py."
        ),

        "mac_ip_tracking_scope": (
            "Each capture-processing session is tracked independently; "
            "cross-day MAC-IP history is not maintained."
        ),

        "machine_learning_dataset_dependency": False,
        "random_forest_dependency": False,
        "generated_label_dependency": False,
        "combined_intermediate_file_created": False,

        "outputs": {
            "packet_integrity": str(
                OUTPUT_FILE.resolve()
            ),
            "metadata": str(
                METADATA_FILE.resolve()
            ),
            "summary": str(
                SUMMARY_FILE.resolve()
            ),
            "rule_counts": str(
                RULE_COUNTS_FILE.resolve()
            ),
            "examples": str(
                EXAMPLES_FILE.resolve()
            ),
        },
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    # ========================================================
    # Console summary
    # ========================================================

    print()
    print("[3/3] Writing reports and metadata...")
    print()
    print("=" * 70)
    print("INTEGRITY CHECKER - COMPLETE")
    print("=" * 70)

    print(
        f"Files processed   : {len(files_processed)}"
    )

    print(
        f"Packets processed : {processed:,}"
    )

    print()
    print("Integrity state:")

    for state in STATES:
        count = state_counts[state]

        percentage = (
            count / processed * 100
            if processed
            else 0.0
        )

        print(
            f"  {state:<12} "
            f"{count:>12,} "
            f"({percentage:>7.3f}%)"
        )

    print()
    print("Rule frequency:")

    for rule in ALL_RULES:
        print(
            f"  {rule:<38} "
            f"{rule_counts[rule]:>12,}"
        )

    print()
    print("Examples stored:")

    for rule in ALL_RULES:
        print(
            f"  {rule:<38} "
            f"{len(examples_per_rule[rule]):>3}"
        )

    print()
    print("Output files:")
    print(f"  Integrity CSV : {OUTPUT_FILE}")
    print(f"  Metadata      : {METADATA_FILE}")
    print(f"  Summary       : {SUMMARY_FILE}")
    print(f"  Rule counts   : {RULE_COUNTS_FILE}")
    print(f"  Examples      : {EXAMPLES_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()