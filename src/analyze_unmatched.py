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
# Output
# ============================================================

OUTPUT_DIR = config.PROCESSED_DIR

UNMATCHED_FILE = OUTPUT_DIR / "unmatched_flows.csv"
SUMMARY_FILE = OUTPUT_DIR / "unmatched_analysis_summary.txt"


# ============================================================
# Helper Functions
# ============================================================

def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def classify_ip(ip):
    """
    Basic IP classification for investigation purposes.
    """
    if pd.isna(ip):
        return "Missing"

    ip = str(ip).strip()

    # IPv6
    if ":" in ip:

        if ip.lower().startswith("fe80:"):
            return "IPv6 Link-Local"

        if ip.lower().startswith("ff"):
            return "IPv6 Multicast"

        return "IPv6 Other"

    # IPv4
    parts = ip.split(".")

    if len(parts) == 4:

        try:
            first = int(parts[0])
            second = int(parts[1])

            if first == 10:
                return "IPv4 Private"

            if first == 172 and 16 <= second <= 31:
                return "IPv4 Private"

            if first == 192 and second == 168:
                return "IPv4 Private"

            if first == 127:
                return "IPv4 Loopback"

            if first >= 224:
                return "IPv4 Multicast/Reserved"

            return "IPv4 Public"

        except ValueError:
            return "Invalid IPv4"

    return "Other"


def safe_value_counts(df, column, top_n=20):

    if column not in df.columns:
        return pd.Series(dtype="int64")

    return df[column].value_counts(dropna=False).head(top_n)


# ============================================================
# Main Analysis
# ============================================================

def main():

    print("=" * 70)
    print("NetworkIDS - Unmatched Flow Investigation")
    print("=" * 70)

    # --------------------------------------------------------
    # Load final dataset
    # --------------------------------------------------------

    input_file = config.FINAL_ML_DATASET_FILE

    print()
    print("Input file:")
    print(input_file)

    if not input_file.exists():

        print()
        print("ERROR: final_ml_dataset.csv was not found.")

        return

    df = pd.read_csv(input_file)

    print()
    print(f"Loaded dataset: {len(df):,} rows")
    print(f"Columns: {len(df.columns)}")

    # --------------------------------------------------------
    # Find unmatched rows
    # --------------------------------------------------------

    if config.LABEL_COLUMN not in df.columns:

        print()
        print("ERROR: label column not found.")

        return

    unmatched = df[df[config.LABEL_COLUMN].isna()].copy()

    print_section("UNMATCHED FLOW COUNT")

    print(f"Total flows:      {len(df):,}")
    print(f"Matched flows:    {df[config.LABEL_COLUMN].notna().sum():,}")
    print(f"Unmatched flows:  {len(unmatched):,}")

    if len(unmatched) == 0:

        print()
        print("No unmatched flows found.")

        return

    # --------------------------------------------------------
    # Basic protocol analysis
    # --------------------------------------------------------

    print_section("PROTOCOL DISTRIBUTION")

    protocol_counts = safe_value_counts(
        unmatched,
        "ip_protocol"
    )

    print(protocol_counts)

    # --------------------------------------------------------
    # IP version
    # --------------------------------------------------------

    print_section("IP VERSION")

    if "ip_version" in unmatched.columns:

        print(
            unmatched["ip_version"]
            .value_counts(dropna=False)
        )

    # --------------------------------------------------------
    # Source IP classification
    # --------------------------------------------------------

    print_section("SOURCE IP CLASSIFICATION")

    unmatched["src_ip_type"] = unmatched["src_ip"].apply(
        classify_ip
    )

    print(
        unmatched["src_ip_type"]
        .value_counts(dropna=False)
    )

    # --------------------------------------------------------
    # Destination IP classification
    # --------------------------------------------------------

    print_section("DESTINATION IP CLASSIFICATION")

    unmatched["dst_ip_type"] = unmatched["dst_ip"].apply(
        classify_ip
    )

    print(
        unmatched["dst_ip_type"]
        .value_counts(dropna=False)
    )

    # --------------------------------------------------------
    # Source IP
    # --------------------------------------------------------

    print_section("TOP SOURCE IPs")

    print(
        safe_value_counts(
            unmatched,
            "src_ip",
            20
        )
    )

    # --------------------------------------------------------
    # Destination IP
    # --------------------------------------------------------

    print_section("TOP DESTINATION IPs")

    print(
        safe_value_counts(
            unmatched,
            "dst_ip",
            20
        )
    )

    # --------------------------------------------------------
    # Source ports
    # --------------------------------------------------------

    print_section("TOP SOURCE PORTS")

    print(
        safe_value_counts(
            unmatched,
            "src_port",
            20
        )
    )

    # --------------------------------------------------------
    # Destination ports
    # --------------------------------------------------------

    print_section("TOP DESTINATION PORTS")

    print(
        safe_value_counts(
            unmatched,
            "dst_port",
            20
        )
    )

    # --------------------------------------------------------
    # Packet count
    # --------------------------------------------------------

    print_section("FLOW PACKET COUNT")

    if "flow_packet_count" in unmatched.columns:

        print(
            unmatched["flow_packet_count"]
            .describe()
        )

        print()
        print(
            "Single-packet flows:",
            (
                unmatched["flow_packet_count"] == 1
            ).sum()
        )

        print(
            "Flows with >100 packets:",
            (
                unmatched["flow_packet_count"] > 100
            ).sum()
        )

        print(
            "Flows with >1000 packets:",
            (
                unmatched["flow_packet_count"] > 1000
            ).sum()
        )

    # --------------------------------------------------------
    # Flow duration
    # --------------------------------------------------------

    print_section("FLOW DURATION")

    if "flow_duration" in unmatched.columns:

        print(
            unmatched["flow_duration"]
            .describe()
        )

        print()

        print(
            "Zero-duration flows:",
            (
                unmatched["flow_duration"] == 0
            ).sum()
        )

        print(
            "Flows >120 seconds:",
            (
                unmatched["flow_duration"] > 120
            ).sum()
        )

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    print_section("FLOW DIRECTION")

    if "flow_direction" in unmatched.columns:

        print(
            unmatched["flow_direction"]
            .value_counts(dropna=False)
        )

    if {
        "forward_packet_count",
        "reverse_packet_count"
    }.issubset(unmatched.columns):

        unidirectional = (
            unmatched["reverse_packet_count"] == 0
        )

        print()
        print(
            "Unidirectional flows:",
            unidirectional.sum()
        )

        print(
            "Bidirectional flows:",
            (~unidirectional).sum()
        )

    # --------------------------------------------------------
    # TCP flags
    # --------------------------------------------------------

    print_section("TCP FLAG SUMMARY")

    tcp_columns = [
        "syn_count",
        "ack_count",
        "fin_count",
        "rst_count",
        "psh_count",
        "syn_seen",
        "syn_ack_seen",
        "ack_after_syn_ack",
    ]

    available_tcp_columns = [
        column
        for column in tcp_columns
        if column in unmatched.columns
    ]

    if available_tcp_columns:

        print(
            unmatched[available_tcp_columns]
            .sum(numeric_only=True)
        )

    # --------------------------------------------------------
    # TCP-specific analysis
    # --------------------------------------------------------

    print_section("TCP FLAG PATTERNS")

    if "ip_protocol" in unmatched.columns:

        tcp = unmatched[
            unmatched["ip_protocol"] == 6
        ].copy()

        print(f"TCP unmatched flows: {len(tcp):,}")

        if len(tcp) > 0:

            if "syn_seen" in tcp.columns:
                print(
                    "TCP flows with SYN:",
                    tcp["syn_seen"].fillna(False).sum()
                )

            if "syn_ack_seen" in tcp.columns:
                print(
                    "TCP flows with SYN-ACK:",
                    tcp["syn_ack_seen"].fillna(False).sum()
                )

            if "fin_count" in tcp.columns:
                print(
                    "TCP flows with FIN:",
                    (tcp["fin_count"] > 0).sum()
                )

            if "rst_count" in tcp.columns:
                print(
                    "TCP flows with RST:",
                    (tcp["rst_count"] > 0).sum()
                )

    # --------------------------------------------------------
    # IPv6 multicast investigation
    # --------------------------------------------------------

    print_section("IPv6 MULTICAST / LINK-LOCAL COMBINATIONS")

    ipv6_multicast = unmatched[
        (
            unmatched["src_ip_type"] == "IPv6 Link-Local"
        )
        &
        (
            unmatched["dst_ip_type"] == "IPv6 Multicast"
        )
    ]

    print(
        "IPv6 link-local -> IPv6 multicast:",
        len(ipv6_multicast)
    )

    if len(ipv6_multicast) > 0:

        print()
        print("Top destination ports:")

        print(
            ipv6_multicast[
                "dst_port"
            ]
            .value_counts(dropna=False)
            .head(20)
        )

    # --------------------------------------------------------
    # Zero-duration + single-packet combinations
    # --------------------------------------------------------

    print_section("SINGLE-PACKET / ZERO-DURATION COMBINATIONS")

    single_packet = (
        unmatched["flow_packet_count"] == 1
    )

    zero_duration = (
        unmatched["flow_duration"] == 0
    )

    print(
        "Single packet:",
        single_packet.sum()
    )

    print(
        "Zero duration:",
        zero_duration.sum()
    )

    print(
        "Single packet AND zero duration:",
        (single_packet & zero_duration).sum()
    )

    # --------------------------------------------------------
    # Potential large unidirectional flows
    # --------------------------------------------------------

    print_section("LARGE UNIDIRECTIONAL FLOWS")

    if {
        "reverse_packet_count",
        "flow_packet_count"
    }.issubset(unmatched.columns):

        large_uni = unmatched[
            (unmatched["reverse_packet_count"] == 0)
            &
            (unmatched["flow_packet_count"] > 1000)
        ].copy()

        print(
            "Unidirectional flows with >1000 packets:",
            len(large_uni)
        )

        if len(large_uni) > 0:

            columns_to_show = [
                "flow_id",
                "timestamp",
                "src_ip",
                "dst_ip",
                "src_port",
                "dst_port",
                "ip_protocol",
                "flow_duration",
                "flow_packet_count",
                "forward_packet_count",
                "reverse_packet_count",
            ]

            columns_to_show = [
                c
                for c in columns_to_show
                if c in large_uni.columns
            ]

            print()
            print(
                large_uni[
                    columns_to_show
                ].head(20).to_string(index=False)
            )

    # --------------------------------------------------------
    # Save complete unmatched dataset
    # --------------------------------------------------------

    save_columns = [
        column
        for column in unmatched.columns
        if column not in [
            "src_ip_type",
            "dst_ip_type",
        ]
    ]

    unmatched[
        save_columns
    ].to_csv(
        UNMATCHED_FILE,
        index=False
    )

    print_section("OUTPUT")

    print(
        "Unmatched flows saved to:"
    )

    print(UNMATCHED_FILE)

    # --------------------------------------------------------
    # Create text summary
    # --------------------------------------------------------

    summary_lines = []

    summary_lines.append(
        "NetworkIDS - Unmatched Flow Investigation"
    )

    summary_lines.append(
        "=" * 60
    )

    summary_lines.append(
        f"Total dataset rows: {len(df):,}"
    )

    summary_lines.append(
        f"Matched rows: {df[config.LABEL_COLUMN].notna().sum():,}"
    )

    summary_lines.append(
        f"Unmatched rows: {len(unmatched):,}"
    )

    summary_lines.append("")

    summary_lines.append(
        "Protocol distribution:"
    )

    summary_lines.append(
        protocol_counts.to_string()
    )

    summary_lines.append("")

    summary_lines.append(
        "Source IP classification:"
    )

    summary_lines.append(
        unmatched["src_ip_type"]
        .value_counts(dropna=False)
        .to_string()
    )

    summary_lines.append("")

    summary_lines.append(
        "Destination IP classification:"
    )

    summary_lines.append(
        unmatched["dst_ip_type"]
        .value_counts(dropna=False)
        .to_string()
    )

    summary_lines.append("")

    summary_lines.append(
        f"Single-packet flows: {single_packet.sum():,}"
    )

    summary_lines.append(
        f"Zero-duration flows: {zero_duration.sum():,}"
    )

    summary_lines.append(
        "Single-packet AND zero-duration: "
        f"{(single_packet & zero_duration).sum():,}"
    )

    summary_lines.append("")

    summary_lines.append(
        "IPv6 link-local -> IPv6 multicast: "
        f"{len(ipv6_multicast):,}"
    )

    SUMMARY_FILE.write_text(
        "\n".join(summary_lines),
        encoding="utf-8"
    )

    print()
    print(
        "Summary saved to:"
    )

    print(SUMMARY_FILE)

    print()
    print("=" * 70)
    print("Investigation completed.")
    print("No labels were modified.")
    print("No rows were deleted from final_ml_dataset.csv.")
    print("=" * 70)

if __name__ == "__main__":
    main()