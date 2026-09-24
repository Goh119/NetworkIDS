"""
NetworkIDS - Aggregate packet-level rows into bidirectional flows.

Input: data/processed/packet_features.csv

Output: data/processed/flow_features.csv

Design:
    - One row = one bidirectional flow.
    - Only TCP and UDP packets participate in standard 5-tuple flow
      construction.
    - A bidirectional flow is identified using a canonical, unordered
      endpoint pair + protocol.
    - The first observed packet defines the forward direction.
    - A flow ends when:
        1. The idle gap exceeds FLOW_IDLE_TIMEOUT, OR
        2. A TCP FIN or RST packet is observed.
        (A previous version also split on repeated SYN, but investigation
        showed ~84% of those were TCP retransmissions following the
        standard 1/2/4/8/16/32s exponential backoff timer, not genuine new
        connections -- see inspect_syn_restart_context.py. That rule was
        removed.)
    - Remaining active flows are flushed at end of file.

Missing-value policy:
    - Missing / non-applicable values remain NaN.
    - Never use -1 as a missing-value marker.
    - Never use 0 to represent a missing protocol-specific value.

Final ML features:
    - Controlled by config.ML_FEATURES.
    - Current design expects 42 ML features.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Project configuration
sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

import src.config as config

# Constants

TCP_PROTOCOL = "TCP"
UDP_PROTOCOL = "UDP"

SUPPORTED_PROTOCOLS = {
    TCP_PROTOCOL,
    UDP_PROTOCOL,
}

# Packet-level fields that are aggregated into flow-level means.
NUMERIC_MEAN_FIELDS = [
    "frame_length",
    "ip_total_length",
    "ip_id",
    "fragment_offset",
    "ttl",
    "tcp_window",
    "tcp_header_length",
]

# Utility helpers
def safe_mean(series):
    """
    Return the mean of non-null values.
    If the field does not apply to any packet in the flow,
    return NaN rather than using 0 or -1.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric.notna().any():
        return float(numeric.mean())

    return np.nan


def first_non_null(series):
    """
    Return the first non-null value.
    If no value exists, return NaN.
    """
    non_null = series.dropna()

    if len(non_null) > 0:
        return non_null.iloc[0]

    return np.nan


def safe_int(value):
    """
    Convert a value to int if possible.
    Missing values remain NaN.
    """
    if pd.isna(value):
        return np.nan

    return int(value)

# TCP flag helpers
def is_tcp_syn_without_ack(row):
    """
    Return True when a packet is a TCP SYN that does not
    carry ACK.

    This represents a new TCP connection attempt.

    SYN + ACK is NOT treated as a new connection because
    it normally belongs to the response side of the
    existing TCP handshake.
    """

    if row["l4_protocol"] != TCP_PROTOCOL:
        return False

    syn = row.get("tcp_syn")
    ack = row.get("tcp_ack")

    syn_seen = (
        pd.notna(syn)
        and int(syn) == 1
    )

    ack_seen = (
        pd.notna(ack)
        and int(ack) == 1
    )

    return syn_seen and not ack_seen

# Flow key
def make_flow_key(
    src_ip,
    dst_ip,
    src_port,
    dst_port,
    protocol,
):
    """
    Build a canonical bidirectional flow key.
    Direction is deliberately ignored here.
    Example:
        A:12345 -> B:443
        B:443   -> A:12345
    Both packets produce the same flow key.
    The first observed packet is used later to determine
    forward/reverse direction.
    Returns
    -------
    tuple
        (
            endpoint1_ip,
            endpoint1_port,
            endpoint2_ip,
            endpoint2_port,
            protocol
        )
    """

    endpoint_a = (
        str(src_ip),
        int(src_port),
    )

    endpoint_b = (
        str(dst_ip),
        int(dst_port),
    )

    if endpoint_a <= endpoint_b:
        first = endpoint_a
        second = endpoint_b

    else:
        first = endpoint_b
        second = endpoint_a

    return (
        first[0],
        first[1],
        second[0],
        second[1],
        str(protocol),
    )

# Flow state
class FlowState:
    """
    Store packets belonging to one active bidirectional flow.

    The first packet defines the forward direction.
    """

    def __init__(
        self,
        flow_id,
        first_row,
    ):

        self.flow_id = flow_id

        self.protocol = first_row[
            "l4_protocol"
        ]

        # ----------------------------------------------------
        # Forward direction
        # ----------------------------------------------------

        self.forward_ip = first_row[
            "src_ip"
        ]

        self.forward_port = int(
            first_row["src_port"]
        )

        self.forward_dst_ip = first_row[
            "dst_ip"
        ]

        self.forward_dst_port = int(
            first_row["dst_port"]
        )

        # ----------------------------------------------------
        # Timing
        # ----------------------------------------------------

        self.start_time = float(
            first_row["timestamp"]
        )

        self.last_time = float(
            first_row["timestamp"]
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        self.src_ip = first_row[
            "src_ip"
        ]

        self.dst_ip = first_row[
            "dst_ip"
        ]

        self.src_port = int(
            first_row["src_port"]
        )

        self.dst_port = int(
            first_row["dst_port"]
        )

        self.src_mac = first_row.get(
            "src_mac"
        )

        self.dst_mac = first_row.get(
            "dst_mac"
        )

        # Packet counters
        self.packet_count = 0
        self.byte_count = 0

        self.forward_packet_count = 0
        self.reverse_packet_count = 0

        self.forward_byte_count = 0
        self.reverse_byte_count = 0

        # TCP flag counters
        self.syn_count = 0
        self.ack_count = 0
        self.fin_count = 0
        self.rst_count = 0
        self.psh_count = 0

        self.syn_seen = 0
        self.syn_ack_seen = 0
        self.ack_after_syn_ack = 0

        # Used to determine TCP handshake ordering.
        self._saw_syn_ack = False

        # Checksum validity
        # ----------------------------------------------------
        # Rule:
        #   all applicable packets valid -> 1
        #   at least one invalid packet -> 0
        #   no applicable checksum -> NaN

        self.ip_checksum_valid_all = True
        self.l4_checksum_valid_all = True

        self._seen_ip_checksum = False
        self._seen_l4_checksum = False

        # Timestamps
        self.timestamps = []

        # Numeric aggregation
        self.numeric_values = {
            field: []
            for field in NUMERIC_MEAN_FIELDS
        }

        # Static / first-packet fields
        self.eth_type = first_row.get(
            "eth_type"
        )

        self.vlan_id = first_row.get(
            "vlan_id"
        )

        self.vlan_priority = first_row.get(
            "vlan_priority"
        )

        self.ip_version = first_row.get(
            "ip_version"
        )

        self.ip_dscp = first_row.get(
            "ip_dscp"
        )

        self.ip_ecn = first_row.get(
            "ip_ecn"
        )

        self.ip_df = first_row.get(
            "ip_df"
        )

        self.ip_mf = first_row.get(
            "ip_mf"
        )

        self.fragment_offset = first_row.get(
            "fragment_offset"
        )

        self.ip_protocol = first_row.get(
            "ip_protocol"
        )

        self.ip_header_length = first_row.get(
            "ip_header_length"
        )

    # Direction
    def is_forward(self, row):
        """
        Determine whether a packet belongs to the forward
        direction.
        Forward direction is defined by the first observed
        packet, not by IP address ordering and not by
        port-number heuristics.
        """

        return (
            row["src_ip"]
            == self.forward_ip
            and int(row["src_port"])
            == self.forward_port
            and row["dst_ip"]
            == self.forward_dst_ip
            and int(row["dst_port"])
            == self.forward_dst_port
        )

    # Add packet
    def add_packet(self, row):
        """
        Add one packet to this flow.
        """

        timestamp = float(
            row["timestamp"]
        )

        self.last_time = timestamp

        self.timestamps.append(
            timestamp
        )

        # Basic packet / byte statistics
        frame_length = float(
            row["frame_length"]
        )

        self.packet_count += 1

        self.byte_count += int(
            frame_length
        )

        # Forward / reverse statistics
        if self.is_forward(row):

            self.forward_packet_count += 1

            self.forward_byte_count += int(
                frame_length
            )

        else:

            self.reverse_packet_count += 1

            self.reverse_byte_count += int(
                frame_length
            )

        # Numeric aggregation
        for field in NUMERIC_MEAN_FIELDS:

            value = row.get(field)

            if pd.notna(value):

                self.numeric_values[
                    field
                ].append(
                    float(value)
                )

        # TCP flags
        if row["l4_protocol"] == TCP_PROTOCOL:

            syn = row.get("tcp_syn")
            ack = row.get("tcp_ack")
            fin = row.get("tcp_fin")
            rst = row.get("tcp_rst")
            psh = row.get("tcp_psh")

            syn = (
                pd.notna(syn)
                and int(syn) == 1
            )

            ack = (
                pd.notna(ack)
                and int(ack) == 1
            )

            fin = (
                pd.notna(fin)
                and int(fin) == 1
            )

            rst = (
                pd.notna(rst)
                and int(rst) == 1
            )

            psh = (
                pd.notna(psh)
                and int(psh) == 1
            )

            self.syn_count += int(
                syn
            )

            self.ack_count += int(
                ack
            )

            self.fin_count += int(
                fin
            )

            self.rst_count += int(
                rst
            )

            self.psh_count += int(
                psh
            )

            # Any SYN without ACK means
            # a SYN was observed.
            if syn and not ack:

                self.syn_seen = 1

            # SYN + ACK means a SYN-ACK
            # was observed.
            if syn and ack:

                self.syn_ack_seen = 1

                self._saw_syn_ack = True

            # ACK observed after SYN-ACK.
            elif ack and self._saw_syn_ack:

                self.ack_after_syn_ack = 1

        # IP checksum validity
        ip_valid = row.get(
            "ip_checksum_valid"
        )

        if pd.notna(ip_valid):

            self._seen_ip_checksum = True

            if int(ip_valid) == 0:

                self.ip_checksum_valid_all = False

        # L4 checksum validity
        l4_valid = row.get(
            "l4_checksum_valid"
        )

        if pd.notna(l4_valid):

            self._seen_l4_checksum = True

            if int(l4_valid) == 0:

                self.l4_checksum_valid_all = False

    # Flow termination
    def is_tcp_terminator(self, row):
        """
        Return True when the packet carries TCP FIN or RST.
        """

        if row["l4_protocol"] != TCP_PROTOCOL:

            return False

        fin = row.get("tcp_fin")
        rst = row.get("tcp_rst")

        fin_seen = (
            pd.notna(fin)
            and int(fin) == 1
        )

        rst_seen = (
            pd.notna(rst)
            and int(rst) == 1
        )

        return (
            fin_seen
            or rst_seen
        )

    # Final feature calculation
    def finalize(self):
        """
        Convert the accumulated flow state into one ML sample.
        """

        duration = (
            self.last_time
            - self.start_time
        )

        if duration < 0:

            duration = 0.0

        # Inter-arrival time
        if len(self.timestamps) >= 2:

            timestamps = np.asarray(
                self.timestamps,
                dtype=float,
            )

            iats = np.diff(
                timestamps
            )

            iat_min = float(
                iats.min()
            )

            iat_mean = float(
                iats.mean()
            )

            iat_max = float(
                iats.max()
            )

            iat_std = float(
                iats.std(ddof=0)
            )

        else:

            iat_min = np.nan
            iat_mean = np.nan
            iat_max = np.nan
            iat_std = np.nan

        # Byte rate
        if duration > 0:

            byte_rate = (
                self.byte_count
                / duration
            )

        else:

            byte_rate = np.nan

        # Flow direction
        # ----------------------------------------------------
        # Current project definition:
        #   1 = at least one reverse packet exists
        #   0 = no reverse packet exists
        # Forward direction itself is still defined
        # by the first observed packet.

        flow_direction = int(
            self.reverse_packet_count > 0
        )

        # Checksum flow-level values
        if self._seen_ip_checksum:

            ip_checksum_valid = int(
                self.ip_checksum_valid_all
            )

        else:

            ip_checksum_valid = np.nan

        if self._seen_l4_checksum:

            l4_checksum_valid = int(
                self.l4_checksum_valid_all
            )

        else:

            l4_checksum_valid = np.nan

        # Build final row
        row = {
            # Metadata
            "flow_id": self.flow_id,
            "timestamp": self.start_time,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_mac": self.src_mac,
            "dst_mac": self.dst_mac,

            # L2
            "frame_length_mean": safe_mean(
                pd.Series(
                    self.numeric_values[
                        "frame_length"
                    ]
                )
            ),
            "eth_type": first_non_null(
                pd.Series(
                    [self.eth_type]
                )
            ),
            "vlan_id": first_non_null(
                pd.Series(
                    [self.vlan_id]
                )
            ),
            "vlan_priority": first_non_null(
                pd.Series(
                    [self.vlan_priority]
                )
            ),

            # L3
            "ip_version": first_non_null(
                pd.Series(
                    [self.ip_version]
                )
            ),
            "ip_dscp": first_non_null(
                pd.Series(
                    [self.ip_dscp]
                )
            ),
            "ip_ecn": first_non_null(
                pd.Series(
                    [self.ip_ecn]
                )
            ),
            "ip_total_length_mean": safe_mean(
                pd.Series(
                    self.numeric_values[
                        "ip_total_length"
                    ]
                )
            ),
            "ip_id_mean": safe_mean(
                pd.Series(
                    self.numeric_values[
                        "ip_id"
                    ]
                )
            ),
            "ip_df": first_non_null(
                pd.Series(
                    [self.ip_df]
                )
            ),
            "ip_mf": first_non_null(
                pd.Series(
                    [self.ip_mf]
                )
            ),
            "fragment_offset": safe_mean(
                pd.Series(
                    self.numeric_values[
                        "fragment_offset"
                    ]
                )
            ),
            "ttl_mean": safe_mean(
                pd.Series(
                    self.numeric_values[
                        "ttl"
                    ]
                )
            ),
            "ip_protocol": first_non_null(
                pd.Series(
                    [self.ip_protocol]
                )
            ),
            "ip_header_length": first_non_null(
                pd.Series(
                    [self.ip_header_length]
                )
            ),
            "ip_checksum_valid":
                ip_checksum_valid,

            # L4
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "tcp_window_mean": safe_mean(
                pd.Series(
                    self.numeric_values[
                        "tcp_window"
                    ]
                )
            ),
            "tcp_header_length_mean": safe_mean(
                pd.Series(
                    self.numeric_values[
                        "tcp_header_length"
                    ]
                )
            ),
            "l4_checksum_valid":
                l4_checksum_valid,
                
            # Flow
            "flow_duration": duration,
            "flow_packet_count":
                self.packet_count,
            "flow_byte_count":
                self.byte_count,
            "forward_packet_count":
                self.forward_packet_count,
            "reverse_packet_count":
                self.reverse_packet_count,
            "forward_byte_count":
                self.forward_byte_count,
            "reverse_byte_count":
                self.reverse_byte_count,
            "iat_min": iat_min,
            "iat_mean": iat_mean,
            "iat_max": iat_max,
            "iat_std": iat_std,
            "byte_rate": byte_rate,
            "flow_direction":
                flow_direction,
            "syn_count":
                self.syn_count,
            "ack_count":
                self.ack_count,
            "fin_count":
                self.fin_count,
            "rst_count":
                self.rst_count,
            "psh_count":
                self.psh_count,
            "syn_seen":
                self.syn_seen,
            "syn_ack_seen":
                self.syn_ack_seen,
            "ack_after_syn_ack":
                self.ack_after_syn_ack,
        }

        return row

# Build flows
def build_flows(
    packet_df,
    idle_timeout=None,
):
    """
    Build bidirectional TCP/UDP flows.

    Flow termination rules:

        TCP:
            - FIN terminates the current flow.
            - RST terminates the current flow.
            - A new SYN without ACK on the same 5-tuple
              terminates the previous active flow and starts
              a new one.

        TCP / UDP:
            - Idle gap greater than idle_timeout terminates
              the current flow.

    Parameters
    ----------
    packet_df : pandas.DataFrame
        Packet-level feature table.

    idle_timeout : float, optional
        Maximum allowed idle gap before starting a new flow.

    Returns
    -------
    list[dict]
        One dictionary per completed flow.
    """

    if idle_timeout is None:

        idle_timeout = (
            config.FLOW_IDLE_TIMEOUT
        )

    # Select TCP / UDP packets only
    df = packet_df[
        packet_df["l4_protocol"].isin(
            SUPPORTED_PROTOCOLS
        )
    ].copy()

    # Standard 5-tuple requires usable
    # IP addresses and ports.
    df = df.dropna(
        subset=[
            "src_ip",
            "dst_ip",
            "src_port",
            "dst_port",
            "ip_protocol",
        ]
    )

    # Sort chronologically
    df = df.sort_values(
        "timestamp"
    ).reset_index(
        drop=True
    )

    print(
        f"  {len(df):,} TCP/UDP packets "
        "eligible for flow construction."
    )

    # Active flow dictionary
    active = {}

    completed = []

    flow_counter = 0

    # Statistics for debugging / verification
    tcp_fin_rst_terminations = 0
    idle_timeout_terminations = 0

    # Process packets
    for row in df.to_dict("records"):

        key = make_flow_key(
            row["src_ip"],
            row["dst_ip"],
            row["src_port"],
            row["dst_port"],
            row["l4_protocol"],
        )

        flow = active.get(key)

        # Existing flow
        if flow is not None:

            current_timestamp = float(
                row["timestamp"]
            )

            gap = (
                current_timestamp
                - flow.last_time
            )

            # Rule 1:
            # Idle timeout
            if gap > idle_timeout:

                completed.append(
                    flow.finalize()
                )

                del active[key]

                flow = None

                idle_timeout_terminations += 1

        # Start a new flow
        if flow is None:

            flow_counter += 1

            flow = FlowState(
                f"FLOW_{flow_counter:06d}",
                row,
            )

            active[key] = flow

        # Add current packet
        flow.add_packet(row)

        # Rule 2:
        # TCP FIN / RST termination
        if flow.is_tcp_terminator(row):

            completed.append(
                flow.finalize()
            )

            del active[key]

            flow = None

            tcp_fin_rst_terminations += 1

    # Flush remaining active flows
    for flow in active.values():

        completed.append(
            flow.finalize()
        )

    # Segmentation statistics
    print()
    print(
        "Flow termination statistics:"
    )

    print(
        f"  TCP FIN/RST terminations    : "
        f"{tcp_fin_rst_terminations:,}"
    )

    print(
        f"  Idle-timeout terminations   : "
        f"{idle_timeout_terminations:,}"
    )

    print(
        f"  Active flows flushed at EOF : "
        f"{len(active):,}"
    )

    return completed

# Main
def main():

    parser = argparse.ArgumentParser(
        description=(
            "Aggregate packet-level features "
            "into bidirectional TCP/UDP flows."
        )
    )

    parser.add_argument(
        "--in",
        dest="input_csv",
        default=str(
            config.PACKET_FEATURES_FILE
        ),
        help="Input packet-level CSV.",
    )

    parser.add_argument(
        "--out",
        dest="output_csv",
        default=str(
            config.FLOW_FEATURES_FILE
        ),
        help="Output flow-level CSV.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Only process the first N packet rows. "
            "Useful for testing."
        ),
    )

    args = parser.parse_args()

    # Header
    print("=" * 60)
    print("NetworkIDS - Flow Builder")
    print("=" * 60)

    print(
        f"Input : {args.input_csv}"
    )

    print(
        f"Output: {args.output_csv}"
    )

    print(
        f"Idle timeout: "
        f"{config.FLOW_IDLE_TIMEOUT} seconds"
    )

    print()

    # Load packet CSV
    input_path = Path(
        args.input_csv
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Packet feature CSV not found: "
            f"{input_path}"
        )

    packet_df = pd.read_csv(
        input_path
    )

    if args.limit is not None:

        packet_df = packet_df.iloc[
            :args.limit
        ].copy()

    print(
        f"Loaded {len(packet_df):,} packet rows."
    )

    # Build flows
    print(
        "Building bidirectional flows..."
    )

    flow_rows = build_flows(
        packet_df,
        idle_timeout=(
            config.FLOW_IDLE_TIMEOUT
        ),
    )

    print(
        f"Built {len(flow_rows):,} flows."
    )

    # Convert to DataFrame
    flow_df = pd.DataFrame(
        flow_rows
    )

    # Validate ML feature list
    config.validate_ml_features()

    missing_features = [
        feature
        for feature in config.ML_FEATURES
        if feature not in flow_df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing ML features:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing_features
            )
        )

    # Column ordering
    metadata_columns = [
        column
        for column in config.METADATA_COLUMNS
        if column in flow_df.columns
    ]

    ml_columns = [
        column
        for column in config.ML_FEATURES
        if column in flow_df.columns
    ]

    remaining_columns = [
        column
        for column in flow_df.columns
        if column not in (
            metadata_columns
            + ml_columns
        )
    ]

    flow_df = flow_df[
        metadata_columns
        + ml_columns
        + remaining_columns
    ]

    # Output directory
    output_path = Path(
        args.output_csv
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    flow_df.to_csv(
        output_path,
        index=False,
    )

    # Results
    print()

    print(
        f"Written: {output_path}"
    )

    print(
        f"Shape  : "
        f"{flow_df.shape[0]:,} flows x "
        f"{flow_df.shape[1]} columns"
    )

    # Sanity checks
    print()
    print("Sanity checks:")

    print(
        f"  [OK] "
        f"All {len(config.ML_FEATURES)} "
        "ML features are present."
    )

    # NaN report
    nan_counts = (
        flow_df[
            config.ML_FEATURES
        ]
        .isna()
        .sum()
    )

    nan_columns = (
        nan_counts[
            nan_counts > 0
        ]
    )

    if len(nan_columns) > 0:

        print(
            "  [INFO] ML features containing "
            "NaN (expected for some "
            "protocol-specific fields):"
        )

        for column, count in (
            nan_columns.items()
        ):

            print(
                f"         {column}: "
                f"{count:,} rows"
            )

    else:

        print(
            "  [OK] No NaN values "
            "in ML features."
        )

    # Infinite values
    numeric_ml = flow_df[
        config.ML_FEATURES
    ].select_dtypes(
        include=[np.number]
    )

    infinite_count = np.isinf(
        numeric_ml.to_numpy(
            dtype=float
        )
    ).sum()

    if infinite_count > 0:

        print(
            f"  [ERROR] "
            f"{infinite_count} infinite "
            "ML feature values detected."
        )

    else:

        print(
            "  [OK] No infinite ML feature values."
        )

    # Flow packet-count statistics
    if not flow_df.empty:

        print()
        print(
            "Flow packet-count statistics:"
        )

        print(
            flow_df[
                "flow_packet_count"
            ]
            .describe()
            .to_string()
        )

    # Protocol breakdown
    if (
        "ip_protocol" in flow_df.columns
        and not flow_df.empty
    ):

        print()
        print(
            "Flow protocol breakdown:"
        )

        print(
            flow_df[
                "ip_protocol"
            ]
            .value_counts()
            .sort_index()
            .to_string()
        )

    # Direction breakdown
    if (
        "flow_direction" in flow_df.columns
        and not flow_df.empty
    ):

        print()
        print(
            "Flow direction breakdown:"
        )

        print(
            flow_df[
                "flow_direction"
            ]
            .value_counts()
            .sort_index()
            .to_string()
        )

    # Preview
    if not flow_df.empty:

        print()
        print(
            "First 5 flows:"
        )

        preview_columns = [
            "flow_id",
            "src_ip",
            "dst_ip",
            "src_port",
            "dst_port",
            "ip_protocol",
            "flow_packet_count",
            "flow_byte_count",
            "forward_packet_count",
            "reverse_packet_count",
            "flow_duration",
            "flow_direction",
            "syn_count",
            "ack_count",
            "fin_count",
            "rst_count",
        ]

        preview_columns = [
            column
            for column in preview_columns
            if column in flow_df.columns
        ]

        print(
            flow_df[
                preview_columns
            ]
            .head(5)
            .to_string(index=False)
        )

if __name__ == "__main__":
    main()