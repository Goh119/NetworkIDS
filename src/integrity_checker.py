from pathlib import Path
import argparse
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import src.config as config

INPUT_FILE = config.PACKET_FEATURES_FILE
OUTPUT_DIR = config.PROCESSED_DIR / "integrity"

OUTPUT_FILE = OUTPUT_DIR / "packet_integrity.csv"
SUMMARY_FILE = OUTPUT_DIR / "integrity_summary.txt"
RULE_COUNTS_FILE = OUTPUT_DIR / "integrity_rule_counts.csv"
EXAMPLES_FILE = OUTPUT_DIR / "integrity_examples.csv"

CHUNK_SIZE = 1_000_000

NEEDED_COLUMNS = [
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


def build_source_ip_mac_map():
    """
    First pass:
    Build source IP -> set of observed source MAC addresses.
    """

    ip_to_macs = {}

    reader = pd.read_csv(
        INPUT_FILE,
        usecols=["src_ip", "src_mac"],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    processed = 0
    chunk_number = 0

    print("[1/4] Building source IP -> source MAC consistency map...")

    for chunk in reader:
        chunk_number += 1
        processed += len(chunk)

        valid = chunk.dropna(subset=["src_ip", "src_mac"])

        for src_ip, group in valid.groupby("src_ip")["src_mac"]:
            macs = group.dropna().astype(str).unique()

            if src_ip not in ip_to_macs:
                ip_to_macs[src_ip] = set()

            ip_to_macs[src_ip].update(macs)

        print(
            f"  chunk {chunk_number:3d} | "
            f"processed {processed:>12,}"
        )

    inconsistent = {
        ip: macs
        for ip, macs in ip_to_macs.items()
        if len(macs) > 1
    }

    print(
        f"  Unique source IPs        : {len(ip_to_macs):,}"
    )
    print(
        f"  IPs with multiple MACs   : {len(inconsistent):,}"
    )

    return {
        ip: True
        for ip in inconsistent
    }


def apply_rules(chunk, inconsistent_source_ips):
    """
    Apply all integrity rules to one dataframe chunk.
    """

    # --------------------------------------------------------------
    # Basic protocol masks
    # --------------------------------------------------------------

    is_ipv4 = chunk["ip_version"].eq(4)

    is_tcp = (
        chunk["ip_protocol"].eq(6)
        | chunk["l4_protocol"].eq("TCP")
    )

    is_tcp = is_tcp.fillna(False)

    is_tcp_udp = (
        chunk["l4_protocol"].isin(["TCP", "UDP"])
    ).fillna(False)

    # --------------------------------------------------------------
    # T1 - Invalid IPv4 checksum
    # --------------------------------------------------------------

    t1 = (
        is_ipv4
        & chunk["ip_checksum_valid"].eq(0)
    ).fillna(False)

    # --------------------------------------------------------------
    # T2 - Fragmentation contradiction
    #
    # DF + MF is contradictory.
    # DF + non-zero fragment offset is contradictory.
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # T3 - Invalid IPv4 header length
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # S1 - TTL anomaly
    #
    # Common initial TTL windows:
    # 64  -> 34..63
    # 128 -> 98..127
    # 255 -> 225..254
    #
    # TTL exactly equal to the common initial value is not anomalous.
    # --------------------------------------------------------------

    ttl = pd.to_numeric(
        chunk["ttl"],
        errors="coerce",
    )

    s1 = (
        ~(
            ((ttl >= 34) & (ttl <= 64))
            | ((ttl >= 98) & (ttl <= 128))
            | ((ttl >= 225) & (ttl <= 255))
        )
        & ttl.notna()
    ).fillna(False)

    # --------------------------------------------------------------
    # S2 - Source IP observed with multiple source MACs
    # --------------------------------------------------------------

    s2 = (
        chunk["src_ip"]
        .map(inconsistent_source_ips)
        .fillna(False)
    )

    # --------------------------------------------------------------
    # TCP flag masks
    # --------------------------------------------------------------

    syn = chunk["tcp_syn"].eq(1).fillna(False)
    ack = chunk["tcp_ack"].eq(1).fillna(False)
    fin = chunk["tcp_fin"].eq(1).fillna(False)
    rst = chunk["tcp_rst"].eq(1).fillna(False)
    psh = chunk["tcp_psh"].eq(1).fillna(False)
    urg = chunk["tcp_urg"].eq(1).fillna(False)

    # --------------------------------------------------------------
    # S3 - SYN + FIN
    # --------------------------------------------------------------

    s3 = (
        is_tcp
        & syn
        & fin
    )

    # --------------------------------------------------------------
    # S4 - SYN + RST
    # --------------------------------------------------------------

    s4 = (
        is_tcp
        & syn
        & rst
    )

    # --------------------------------------------------------------
    # S5 - FIN + RST
    # --------------------------------------------------------------

    s5 = (
        is_tcp
        & fin
        & rst
    )

    # --------------------------------------------------------------
    # S6 - TCP NULL flags
    #
    # All six monitored TCP flags are zero.
    # --------------------------------------------------------------

    s6 = (
        is_tcp
        & ~syn
        & ~ack
        & ~fin
        & ~rst
        & ~psh
        & ~urg
    )

    # --------------------------------------------------------------
    # S7 - Invalid L4 checksum
    #
    # IMPORTANT:
    # This is an indicator only.
    # It does NOT independently change integrity_state.
    # --------------------------------------------------------------

    s7 = (
        is_tcp_udp
        & chunk["l4_checksum_valid"].eq(0)
    ).fillna(False)

    # --------------------------------------------------------------
    # Store rule columns
    # --------------------------------------------------------------

    chunk["T1_invalid_ip_checksum"] = t1.astype("int8")
    chunk["T2_fragmentation_contradiction"] = t2.astype("int8")
    chunk["T3_invalid_ip_header_length"] = t3.astype("int8")

    chunk["S1_ttl_anomaly"] = s1.astype("int8")
    chunk["S2_mac_ip_inconsistency"] = s2.astype("int8")
    chunk["S3_syn_fin"] = s3.astype("int8")
    chunk["S4_syn_rst"] = s4.astype("int8")
    chunk["S5_fin_rst"] = s5.astype("int8")
    chunk["S6_null_tcp"] = s6.astype("int8")

    chunk["S7_invalid_l4_checksum"] = s7.astype("int8")

    # --------------------------------------------------------------
    # Rule counts
    # --------------------------------------------------------------

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

    chunk["tampered_rule_count"] = tampered_count.astype("int8")
    chunk["suspicious_rule_count"] = suspicious_count.astype("int8")

    # --------------------------------------------------------------
    # Integrity state
    #
    # Tampered has priority over Suspicious.
    # S7 alone does not change the state.
    # --------------------------------------------------------------

    chunk["integrity_state"] = "Intact"

    chunk.loc[
        suspicious_count.gt(0),
        "integrity_state",
    ] = "Suspicious"

    chunk.loc[
        tampered_count.gt(0),
        "integrity_state",
    ] = "Tampered"

    # --------------------------------------------------------------
    # Reasons
    # --------------------------------------------------------------

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
            "TTL anomaly",
        ),
        (
            "S2_mac_ip_inconsistency",
            "Source IP observed with multiple source MACs",
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

    reason_series = pd.Series(
        "",
        index=chunk.index,
        dtype="object",
    )

    for column, description in reason_columns:
        mask = chunk[column].eq(1)

        reason_series.loc[mask] = (
            reason_series.loc[mask]
            + description
            + "; "
        )

    chunk["integrity_reasons"] = (
        reason_series
        .str.rstrip("; ")
        .replace("", "None")
    )

    # --------------------------------------------------------------
    # Weak indicators
    # --------------------------------------------------------------

    indicator_series = pd.Series(
        "",
        index=chunk.index,
        dtype="object",
    )

    indicator_mask = chunk["S7_invalid_l4_checksum"].eq(1)

    indicator_series.loc[indicator_mask] = (
        "Invalid L4 checksum"
    )

    chunk["integrity_indicators"] = (
        indicator_series
        .replace("", "None")
    )

    return chunk


def main():
    parser = argparse.ArgumentParser(
        description="Run the NetworkIDS packet integrity checker."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of packets to process.",
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("NetworkIDS - Packet Integrity Checker")
    print("=" * 70)
    print(f"Input     : {INPUT_FILE}")
    print(f"Chunk size: {CHUNK_SIZE:,}")

    if args.limit:
        print(f"Limit     : {args.limit:,}")
    else:
        print("Limit     : whole dataset")

    print()

    # --------------------------------------------------------------
    # PASS 1
    # --------------------------------------------------------------

    inconsistent_source_ips = (
        build_source_ip_mac_map()
    )

    # --------------------------------------------------------------
    # PASS 2
    # --------------------------------------------------------------

    print()
    print("[2/4] Applying integrity rules...")

    reader = pd.read_csv(
        INPUT_FILE,
        usecols=NEEDED_COLUMNS,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    processed = 0
    chunk_number = 0

    rule_counts = {
        rule: 0
        for rule in (
            TAMPERED_RULES
            + SUSPICIOUS_RULES
            + INDICATOR_RULES
        )
    }

    state_counts = {
        "Intact": 0,
        "Suspicious": 0,
        "Tampered": 0,
    }

    examples = []

    first_write = True

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    for chunk in reader:
        chunk_number += 1

        if args.limit is not None:
            remaining = args.limit - processed

            if remaining <= 0:
                break

            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]

        if chunk.empty:
            break

        processed += len(chunk)

        chunk = apply_rules(
            chunk,
            inconsistent_source_ips,
        )

        # ----------------------------------------------------------
        # Update counts
        # ----------------------------------------------------------

        for rule in rule_counts:
            rule_counts[rule] += int(
                chunk[rule].sum()
            )

        for state in state_counts:
            state_counts[state] += int(
                (chunk["integrity_state"] == state).sum()
            )

        # ----------------------------------------------------------
        # Save examples
        # ----------------------------------------------------------

        if len(examples) < 100:
            example_mask = (
                chunk["integrity_state"] != "Intact"
            ) | (
                chunk["S7_invalid_l4_checksum"] == 1
            )

            example_columns = [
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
            ] + (
                TAMPERED_RULES
                + SUSPICIOUS_RULES
                + INDICATOR_RULES
            )

            selected = chunk.loc[
                example_mask,
                example_columns,
            ].head(100 - len(examples))

            examples.extend(
                selected.to_dict("records")
            )

        # ----------------------------------------------------------
        # Write output incrementally
        # ----------------------------------------------------------

        chunk.to_csv(
            OUTPUT_FILE,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
        )

        first_write = False

        print(
            f"  chunk {chunk_number:3d} | "
            f"processed {processed:>12,}"
        )

    # --------------------------------------------------------------
    # Write rule count table
    # --------------------------------------------------------------

    rule_rows = []

    for rule in (
        TAMPERED_RULES
        + SUSPICIOUS_RULES
        + INDICATOR_RULES
    ):
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

        rule_rows.append(
            {
                "rule": rule,
                "severity": severity,
                "count": count,
                "percentage": percentage,
            }
        )

    pd.DataFrame(rule_rows).to_csv(
        RULE_COUNTS_FILE,
        index=False,
    )

    # --------------------------------------------------------------
    # Examples
    # --------------------------------------------------------------

    pd.DataFrame(examples).to_csv(
        EXAMPLES_FILE,
        index=False,
    )

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    summary_lines = [
        "NetworkIDS - Packet Integrity Checker",
        "=" * 70,
        "",
        f"Input file       : {INPUT_FILE}",
        f"Packets processed: {processed:,}",
        "",
        "INTEGRITY STATE",
        "-" * 70,
    ]

    for state in [
        "Intact",
        "Suspicious",
        "Tampered",
    ]:
        count = state_counts[state]

        percentage = (
            count / processed * 100
            if processed
            else 0.0
        )

        summary_lines.append(
            f"{state:<12}: "
            f"{count:,} ({percentage:.4f}%)"
        )

    summary_lines.extend(
        [
            "",
            "RULE FREQUENCY",
            "-" * 70,
        ]
    )

    for rule in (
        TAMPERED_RULES
        + SUSPICIOUS_RULES
        + INDICATOR_RULES
    ):
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
            f"[{severity}]"
        )

    summary_lines.extend(
        [
            "",
            "S7 HANDLING",
            "-" * 70,
            "Invalid L4 checksum is treated as a weak",
            "integrity indicator and does not independently",
            "change the packet integrity state.",
            "",
            "STATE LOGIC",
            "-" * 70,
            "Tampered: at least one strong integrity rule.",
            "Suspicious: no Tampered rule, but at least one",
            "            protocol-anomaly rule S1-S6.",
            "Intact:    no T1-T3 or S1-S6 violations.",
            "S7 alone does not change the state.",
            "",
            "IMPORTANT INTERPRETATION",
            "-" * 70,
            "Tampered represents a strong protocol/header",
            "integrity violation according to the defined rules.",
            "It does not constitute definitive proof of malicious",
            "packet modification.",
            "",
        ]
    )

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        f.write("\n".join(summary_lines))

    # --------------------------------------------------------------
    # Console summary
    # --------------------------------------------------------------

    print()
    print("[3/4] Writing summary tables...")
    print()
    print("[4/4] Writing summary...")

    print()
    print("=" * 70)
    print("INTEGRITY CHECKER - COMPLETE")
    print("=" * 70)

    print(
        f"Packets processed: {processed:,}"
    )

    print()
    print("Integrity state:")

    for state in [
        "Intact",
        "Suspicious",
        "Tampered",
    ]:
        count = state_counts[state]

        percentage = (
            count / processed * 100
            if processed
            else 0.0
        )

        print(
            f"  {state:<12} "
            f"{count:>10,} "
            f"({percentage:>7.3f}%)"
        )

    print()
    print("Rule frequency:")

    for rule in (
        TAMPERED_RULES
        + SUSPICIOUS_RULES
        + INDICATOR_RULES
    ):
        print(
            f"  {rule:<38} "
            f"{rule_counts[rule]:>10,}"
        )

    print()
    print("Output files:")
    print(f"  Integrity CSV : {OUTPUT_FILE}")
    print(f"  Summary       : {SUMMARY_FILE}")
    print(f"  Rule counts   : {RULE_COUNTS_FILE}")
    print(f"  Examples      : {EXAMPLES_FILE}")

    print("=" * 70)


if __name__ == "__main__":
    main()