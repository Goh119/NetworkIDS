from pathlib import Path
import argparse

import pandas as pd

import config


OUTPUT_DIR = config.PROCESSED_DIR / "integrity_analysis"

INPUT_FILE = config.PACKET_FEATURES_FILE

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
    "ip_checksum_valid",
    "l4_checksum_valid",
    "src_port",
    "dst_port",
]


def classify_ip_version(value):
    if pd.isna(value):
        return "Unknown"

    try:
        value = int(value)
    except (TypeError, ValueError):
        return str(value)

    if value == 4:
        return "IPv4"
    if value == 6:
        return "IPv6"

    return f"Other({value})"


def main():
    parser = argparse.ArgumentParser(
        description="Analyze packets with invalid L4 checksums."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of packet rows to analyze.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NetworkIDS - L4 Checksum Analysis")
    print("=" * 70)
    print(f"Input     : {INPUT_FILE}")
    print(f"Chunk size: {CHUNK_SIZE:,}")

    if args.limit:
        print(f"Limit     : {args.limit:,}")
    else:
        print("Limit     : whole dataset")

    print()

    # ------------------------------------------------------------------
    # Storage for aggregate statistics
    # ------------------------------------------------------------------

    total_packets = 0
    invalid_l4_packets = 0

    protocol_counts = {}
    ip_version_counts = {}

    source_port_counts = {}
    destination_port_counts = {}

    source_ip_counts = {}
    destination_ip_counts = {}

    ttl_counts = {}

    invalid_tcp = 0
    invalid_udp = 0
    invalid_other_l4 = 0

    invalid_l4_with_ttl_anomaly = 0

    examples = []

    # ------------------------------------------------------------------
    # Read packet_features.csv in chunks
    # ------------------------------------------------------------------

    reader = pd.read_csv(
        INPUT_FILE,
        usecols=NEEDED_COLUMNS,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    processed = 0
    chunk_number = 0

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
        total_packets += len(chunk)

        # --------------------------------------------------------------
        # Identify invalid L4 checksum packets
        # --------------------------------------------------------------

        invalid_mask = (
            chunk["l4_checksum_valid"].notna()
            & (chunk["l4_checksum_valid"] == 0)
        )

        invalid = chunk.loc[invalid_mask].copy()

        invalid_count = len(invalid)
        invalid_l4_packets += invalid_count

        if invalid_count == 0:
            print(
                f"  chunk {chunk_number:3d} | "
                f"total {processed:>12,}"
            )
            continue

        # --------------------------------------------------------------
        # L4 protocol
        # --------------------------------------------------------------

        protocol_series = (
            invalid["l4_protocol"]
            .fillna("Unknown")
            .astype(str)
        )

        for protocol, count in protocol_series.value_counts().items():
            protocol_counts[protocol] = (
                protocol_counts.get(protocol, 0) + int(count)
            )

        invalid_tcp += int((protocol_series == "TCP").sum())
        invalid_udp += int((protocol_series == "UDP").sum())

        invalid_other_l4 += int(
            (~protocol_series.isin(["TCP", "UDP"])).sum()
        )

        # --------------------------------------------------------------
        # IP version
        # --------------------------------------------------------------

        ip_versions = (
            invalid["ip_version"]
            .map(classify_ip_version)
        )

        for version, count in ip_versions.value_counts().items():
            ip_version_counts[version] = (
                ip_version_counts.get(version, 0) + int(count)
            )

        # --------------------------------------------------------------
        # Source / destination ports
        # --------------------------------------------------------------

        src_ports = invalid["src_port"].dropna()

        for port, count in src_ports.value_counts().items():
            try:
                port_key = int(port)
            except (TypeError, ValueError):
                port_key = str(port)

            source_port_counts[port_key] = (
                source_port_counts.get(port_key, 0) + int(count)
            )

        dst_ports = invalid["dst_port"].dropna()

        for port, count in dst_ports.value_counts().items():
            try:
                port_key = int(port)
            except (TypeError, ValueError):
                port_key = str(port)

            destination_port_counts[port_key] = (
                destination_port_counts.get(port_key, 0) + int(count)
            )

        # --------------------------------------------------------------
        # Source / destination IPs
        # --------------------------------------------------------------

        for ip, count in (
            invalid["src_ip"]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .items()
        ):
            source_ip_counts[ip] = (
                source_ip_counts.get(ip, 0) + int(count)
            )

        for ip, count in (
            invalid["dst_ip"]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .items()
        ):
            destination_ip_counts[ip] = (
                destination_ip_counts.get(ip, 0) + int(count)
            )

        # --------------------------------------------------------------
        # TTL distribution
        # --------------------------------------------------------------

        for ttl, count in invalid["ttl"].dropna().value_counts().items():
            try:
                ttl_key = int(ttl)
            except (TypeError, ValueError):
                ttl_key = str(ttl)

            ttl_counts[ttl_key] = (
                ttl_counts.get(ttl_key, 0) + int(count)
            )

        # --------------------------------------------------------------
        # Check overlap with TTL anomaly rule S1
        #
        # Same definition used by analyze_integrity_rules.py
        # --------------------------------------------------------------

        ttl = pd.to_numeric(invalid["ttl"], errors="coerce")

        ttl_anomaly = (
            ((ttl >= 34) & (ttl < 64))
            | ((ttl >= 98) & (ttl < 128))
            | ((ttl >= 225) & (ttl < 255))
        )

        invalid_l4_with_ttl_anomaly += int(ttl_anomaly.sum())

        # --------------------------------------------------------------
        # Save representative examples
        # --------------------------------------------------------------

        if len(examples) < 50:
            example_columns = [
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
                "ip_checksum_valid",
                "l4_checksum_valid",
                "src_port",
                "dst_port",
            ]

            example_rows = invalid.loc[
                :,
                example_columns,
            ].head(50 - len(examples))

            examples.extend(
                example_rows.to_dict("records")
            )

        print(
            f"  chunk {chunk_number:3d} | "
            f"total {processed:>12,} | "
            f"invalid L4 {invalid_l4_packets:>10,}"
        )

    # ------------------------------------------------------------------
    # Build output tables
    # ------------------------------------------------------------------

    invalid_percentage = (
        invalid_l4_packets / total_packets * 100
        if total_packets
        else 0.0
    )

    tcp_percentage = (
        invalid_tcp / invalid_l4_packets * 100
        if invalid_l4_packets
        else 0.0
    )

    udp_percentage = (
        invalid_udp / invalid_l4_packets * 100
        if invalid_l4_packets
        else 0.0
    )

    other_percentage = (
        invalid_other_l4 / invalid_l4_packets * 100
        if invalid_l4_packets
        else 0.0
    )

    ttl_overlap_percentage = (
        invalid_l4_with_ttl_anomaly / invalid_l4_packets * 100
        if invalid_l4_packets
        else 0.0
    )

    # ------------------------------------------------------------------
    # Protocol table
    # ------------------------------------------------------------------

    protocol_rows = []

    for protocol, count in sorted(
        protocol_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        protocol_rows.append(
            {
                "l4_protocol": protocol,
                "invalid_l4_count": count,
                "percentage_of_invalid_l4": (
                    count / invalid_l4_packets * 100
                    if invalid_l4_packets
                    else 0.0
                ),
                "percentage_of_all_packets": (
                    count / total_packets * 100
                    if total_packets
                    else 0.0
                ),
            }
        )

    pd.DataFrame(protocol_rows).to_csv(
        OUTPUT_DIR / "l4_checksum_protocol.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Port table
    # ------------------------------------------------------------------

    port_rows = []

    all_ports = set(source_port_counts) | set(destination_port_counts)

    for port in all_ports:
        src_count = source_port_counts.get(port, 0)
        dst_count = destination_port_counts.get(port, 0)

        port_rows.append(
            {
                "port": port,
                "invalid_as_source_port": src_count,
                "invalid_as_destination_port": dst_count,
                "invalid_total": src_count + dst_count,
            }
        )

    port_df = pd.DataFrame(port_rows)

    if not port_df.empty:
        port_df = port_df.sort_values(
            "invalid_total",
            ascending=False,
        )

    port_df.to_csv(
        OUTPUT_DIR / "l4_checksum_ports.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Examples
    # ------------------------------------------------------------------

    pd.DataFrame(examples).to_csv(
        OUTPUT_DIR / "l4_checksum_examples.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # TTL table
    # ------------------------------------------------------------------

    ttl_rows = []

    for ttl_value, count in sorted(
        ttl_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        ttl_rows.append(
            {
                "ttl": ttl_value,
                "invalid_l4_count": count,
                "percentage_of_invalid_l4": (
                    count / invalid_l4_packets * 100
                    if invalid_l4_packets
                    else 0.0
                ),
            }
        )

    pd.DataFrame(ttl_rows).to_csv(
        OUTPUT_DIR / "l4_checksum_ttl_overlap.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Summary text
    # ------------------------------------------------------------------

    summary_lines = [
        "NetworkIDS - L4 Checksum Analysis",
        "=" * 70,
        "",
        f"Input file       : {INPUT_FILE}",
        f"Packets analyzed : {total_packets:,}",
        "",
        "L4 CHECKSUM SUMMARY",
        "-" * 70,
        (
            f"Invalid L4 checksum packets : "
            f"{invalid_l4_packets:,} "
            f"({invalid_percentage:.4f}%)"
        ),
        "",
        "PROTOCOL BREAKDOWN",
        "-" * 70,
        (
            f"TCP   : {invalid_tcp:,} "
            f"({tcp_percentage:.4f}% of invalid L4)"
        ),
        (
            f"UDP   : {invalid_udp:,} "
            f"({udp_percentage:.4f}% of invalid L4)"
        ),
        (
            f"Other : {invalid_other_l4:,} "
            f"({other_percentage:.4f}% of invalid L4)"
        ),
        "",
        "IP VERSION BREAKDOWN",
        "-" * 70,
    ]

    for version, count in sorted(
        ip_version_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        percentage = (
            count / invalid_l4_packets * 100
            if invalid_l4_packets
            else 0.0
        )

        summary_lines.append(
            f"{version:<10} : {count:,} ({percentage:.4f}%)"
        )

    summary_lines.extend(
        [
            "",
            "TTL ANOMALY OVERLAP",
            "-" * 70,
            (
                f"Invalid L4 + TTL anomaly : "
                f"{invalid_l4_with_ttl_anomaly:,} "
                f"({ttl_overlap_percentage:.4f}% of invalid L4)"
            ),
            "",
            "TOP SOURCE PORTS",
            "-" * 70,
        ]
    )

    for port, count in sorted(
        source_port_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:20]:
        summary_lines.append(
            f"{str(port):<10} : {count:,}"
        )

    summary_lines.extend(
        [
            "",
            "TOP DESTINATION PORTS",
            "-" * 70,
        ]
    )

    for port, count in sorted(
        destination_port_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:20]:
        summary_lines.append(
            f"{str(port):<10} : {count:,}"
        )

    summary_lines.extend(
        [
            "",
            "TOP SOURCE IPs",
            "-" * 70,
        ]
    )

    for ip, count in sorted(
        source_ip_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:20]:
        summary_lines.append(
            f"{ip:<40} : {count:,}"
        )

    summary_lines.extend(
        [
            "",
            "TOP DESTINATION IPs",
            "-" * 70,
        ]
    )

    for ip, count in sorted(
        destination_ip_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:20]:
        summary_lines.append(
            f"{ip:<40} : {count:,}"
        )

    summary_lines.extend(
        [
            "",
            "INTERPRETATION NOTE",
            "-" * 70,
            "This analysis does not classify packets as malicious.",
            "It only characterizes packets with invalid L4 checksums.",
            "Invalid L4 checksums may be affected by packet capture",
            "conditions or checksum offloading and therefore should",
            "not automatically be interpreted as evidence of tampering.",
            "",
        ]
    )

    with open(
        OUTPUT_DIR / "l4_checksum_summary.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write("\n".join(summary_lines))

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("L4 CHECKSUM ANALYSIS - COMPLETE")
    print("=" * 70)

    print(
        f"Packets analyzed          : {total_packets:,}"
    )
    print(
        f"Invalid L4 checksum       : "
        f"{invalid_l4_packets:,} "
        f"({invalid_percentage:.4f}%)"
    )

    print()
    print("Protocol:")
    print(
        f"  TCP   : {invalid_tcp:,} "
        f"({tcp_percentage:.2f}%)"
    )
    print(
        f"  UDP   : {invalid_udp:,} "
        f"({udp_percentage:.2f}%)"
    )
    print(
        f"  Other : {invalid_other_l4:,} "
        f"({other_percentage:.2f}%)"
    )

    print()
    print("TTL overlap:")
    print(
        f"  Invalid L4 + TTL anomaly : "
        f"{invalid_l4_with_ttl_anomaly:,} "
        f"({ttl_overlap_percentage:.2f}%)"
    )

    print()
    print("Output files:")
    print(
        f"  Summary  : "
        f"{OUTPUT_DIR / 'l4_checksum_summary.txt'}"
    )
    print(
        f"  Protocol : "
        f"{OUTPUT_DIR / 'l4_checksum_protocol.csv'}"
    )
    print(
        f"  Ports    : "
        f"{OUTPUT_DIR / 'l4_checksum_ports.csv'}"
    )
    print(
        f"  Examples : "
        f"{OUTPUT_DIR / 'l4_checksum_examples.csv'}"
    )
    print(
        f"  TTL      : "
        f"{OUTPUT_DIR / 'l4_checksum_ttl_overlap.csv'}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()