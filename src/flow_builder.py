"""
NetworkIDS - Aggregate packet-level rows into bidirectional flows.

Input:
    data/processed/packet_features.csv

Output:
    data/processed/flow_features.csv

Design:
    - One row = one bidirectional flow.
    - Only TCP and UDP packets participate in standard 5-tuple
      flow construction.
    - A bidirectional flow is identified using a canonical,
      unordered endpoint pair + protocol.
    - The first observed packet defines the forward direction.

    - A flow ends when:
        1. The idle gap exceeds FLOW_IDLE_TIMEOUT, OR
        2. A TCP FIN or RST packet is observed.

    - Repeated SYN packets do NOT terminate an active flow.

      A previous version used repeated SYN without ACK as a flow
      boundary. Investigation showed that most of those packets were
      TCP SYN retransmissions rather than genuine new connections.

    - Remaining active flows are flushed at end of file.

Memory design:
    - packet_features.csv is read in chunks.
    - The whole packet CSV is never loaded into RAM.
    - to_dict("records") is NOT used.
    - Each packet row is converted to one temporary dictionary only.
    - Active flows persist across CSV chunk boundaries.
    - Completed flows are written directly to output CSV.
    - Completed flows are NOT stored in one giant Python list.
    - Per-flow packet timestamps are NOT stored individually.
    - Numeric flow means and IAT statistics use running aggregates.

Missing-value policy:
    - Missing / non-applicable values remain NaN.
    - Never use -1 as a missing-value marker.
    - Never use 0 to represent a missing protocol-specific value.

Final ML features:
    - Controlled by config.ML_FEATURES.
"""

import argparse
import csv
import math
import sys

from collections import Counter
from pathlib import Path

import numpy as np 
import pandas as pd 

# ============================================================
# Project configuration
# ============================================================

sys.path.append(
    str(
        Path(__file__).resolve().parents[1]
    )
)

import src.config as config


# ============================================================
# Constants
# ============================================================

TCP_PROTOCOL = "TCP"
UDP_PROTOCOL = "UDP"

SUPPORTED_PROTOCOLS = {
    TCP_PROTOCOL,
    UDP_PROTOCOL,
}


# ============================================================
# ==================== ADDED: CHUNK SETTINGS ==================
#
# Input packets:
#     Number of packet CSV rows loaded by pandas at one time.
#
# Unlike the previous version, this is the maximum packet
# DataFrame size rather than the entire dataset.
# ============================================================

PACKET_CHUNK_SIZE = 200_000


# ============================================================
# ==================== ADDED: IDLE CLEANUP ====================
#
# Active flows that have received no packet for longer than
# FLOW_IDLE_TIMEOUT should not remain in RAM until EOF.
#
# We periodically scan active flows and finalize expired ones.
#
# A per-key timeout check also exists, so flow correctness does
# not depend on this periodic cleanup interval.
# ============================================================

IDLE_CLEANUP_INTERVAL = 30.0


# ============================================================
# ==================== CHANGED: TIMESTAMP TOLERANCE ===========
#
# Very small backward timestamp differences can occur because
# PCAP timestamps are converted to floating-point values and
# later written/read through CSV.
#
# Example observed in CIC-IDS2017:
#
#   previous = 1499084039.982496
#   current  = 1499084039.982485
#
# Difference ≈ 10.967 microseconds.
#
# The previous 10-microsecond tolerance was slightly too strict.
#
# We now tolerate up to 1 millisecond of small capture / floating
# point timestamp jitter.
#
# Larger backwards jumps still trigger an error.
# ============================================================

TIMESTAMP_TOLERANCE = 1e-3  # 1 millisecond


# ============================================================
# Packet fields used by the flow builder
# ============================================================

BASE_INPUT_COLUMNS = [

    "timestamp",

    "frame_length",

    "src_mac",
    "dst_mac",

    "eth_type",
    "vlan_id",
    "vlan_priority",

    "ip_version",
    "src_ip",
    "dst_ip",

    "ip_dscp",
    "ip_ecn",

    "ip_total_length",
    "ip_id",

    "ip_df",
    "ip_mf",

    "fragment_offset",

    "ttl",

    "ip_protocol",
    "ip_header_length",

    "ip_checksum_valid",

    "l4_protocol",

    "src_port",
    "dst_port",

    "tcp_window",
    "tcp_header_length",

    "l4_checksum_valid",

    "tcp_syn",
    "tcp_ack",
    "tcp_fin",
    "tcp_rst",
    "tcp_psh",
]


# ============================================================
# MAC-IP packet features
# ============================================================

MAC_IP_COLUMNS = [

    "src_ip_mac_consistent",

    "src_ip_mac_seen_mac_count",

    "src_mac_ip_seen_ip_count",

    "src_mac_ip_spoof_suspect",
]


# ============================================================
# Final flow columns
# ============================================================

FLOW_COLUMNS = [

    # Metadata
    "flow_id",
    "timestamp",

    "src_ip",
    "dst_ip",

    "src_mac",
    "dst_mac",

    # L2
    "frame_length_mean",

    "eth_type",
    "vlan_id",
    "vlan_priority",

    # L3
    "ip_version",

    "ip_dscp",
    "ip_ecn",

    "ip_total_length_mean",
    "ip_id_mean",

    "ip_df",
    "ip_mf",

    "fragment_offset",

    "ttl_mean",

    "ip_protocol",
    "ip_header_length",

    "ip_checksum_valid",

    # MAC-IP integrity
    "flow_mac_ip_inconsistent_count",

    "flow_mac_ip_spoof_suspect",

    "src_ip_mac_seen_mac_count_max",

    "src_mac_ip_seen_ip_count_max",

    # L4
    "src_port",
    "dst_port",

    "tcp_window_mean",
    "tcp_header_length_mean",

    "l4_checksum_valid",

    # Flow
    "flow_duration",

    "flow_packet_count",
    "flow_byte_count",

    "forward_packet_count",
    "reverse_packet_count",

    "forward_byte_count",
    "reverse_byte_count",

    "iat_min",
    "iat_mean",
    "iat_max",
    "iat_std",

    "byte_rate",

    "flow_direction",

    "syn_count",
    "ack_count",
    "fin_count",
    "rst_count",
    "psh_count",

    "syn_seen",
    "syn_ack_seen",
    "ack_after_syn_ack",
]


# ============================================================
# Utility helpers
# ============================================================

def is_missing(value):
    """
    Return True when a value is missing / NaN.
    """

    return pd.isna(value)


def first_value_or_nan(value):
    """
    Preserve the first packet value.

    Missing values remain NaN.
    """

    if is_missing(value):
        return np.nan

    return value


# ============================================================
# Flow key
# ============================================================

def make_flow_key(
    src_ip,
    dst_ip,
    src_port,
    dst_port,
    protocol,
):
    """
    Build a canonical bidirectional 5-tuple flow key.

    Example:

        A:12345 -> B:443
        B:443   -> A:12345

    produce the same key.

    Direction is NOT determined here.

    The first observed packet later defines forward direction.
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


# ============================================================
# Flow state
# ============================================================

class FlowState:
    """
    Store the running state for one active bidirectional flow.

    This implementation does NOT keep per-packet timestamp or
    numeric-value lists.

    Instead it maintains running aggregate values so memory usage
    remains approximately constant per active flow.
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

        # ====================================================
        # Forward direction
        # ====================================================

        self.forward_ip = first_row[
            "src_ip"
        ]

        self.forward_port = int(
            first_row[
                "src_port"
            ]
        )

        self.forward_dst_ip = first_row[
            "dst_ip"
        ]

        self.forward_dst_port = int(
            first_row[
                "dst_port"
            ]
        )

        # ====================================================
        # Timing
        # ====================================================

        self.start_time = float(
            first_row[
                "timestamp"
            ]
        )

        self.last_time = float(
            first_row[
                "timestamp"
            ]
        )

        self.previous_packet_time = None

        # ====================================================
        # Metadata
        # ====================================================

        self.src_ip = first_row[
            "src_ip"
        ]

        self.dst_ip = first_row[
            "dst_ip"
        ]

        self.src_port = int(
            first_row[
                "src_port"
            ]
        )

        self.dst_port = int(
            first_row[
                "dst_port"
            ]
        )

        self.src_mac = first_row.get(
            "src_mac"
        )

        self.dst_mac = first_row.get(
            "dst_mac"
        )

        # ====================================================
        # Packet counters
        # ====================================================

        self.packet_count = 0

        self.byte_count = 0

        self.forward_packet_count = 0
        self.reverse_packet_count = 0

        self.forward_byte_count = 0
        self.reverse_byte_count = 0

        # ====================================================
        # TCP flags
        # ====================================================

        self.syn_count = 0
        self.ack_count = 0

        self.fin_count = 0
        self.rst_count = 0

        self.psh_count = 0

        self.syn_seen = 0

        self.syn_ack_seen = 0

        self.ack_after_syn_ack = 0

        self._saw_syn_ack = False

        # ====================================================
        # Checksum validity
        # ====================================================

        self.ip_checksum_valid_all = True
        self.l4_checksum_valid_all = True

        self._seen_ip_checksum = False
        self._seen_l4_checksum = False

        # ====================================================
        # MAC-IP evidence
        # ====================================================

        self.mac_ip_inconsistent_count = 0

        self.mac_ip_spoof_suspect_any = 0

        self.src_ip_mac_seen_mac_count_max = None

        self.src_mac_ip_seen_ip_count_max = None

        self._seen_mac_ip_evidence = False

        # ====================================================
        # Running numeric means
        # ====================================================

        self.numeric_sum = {
            "frame_length": 0.0,
            "ip_total_length": 0.0,
            "ip_id": 0.0,
            "fragment_offset": 0.0,
            "ttl": 0.0,
            "tcp_window": 0.0,
            "tcp_header_length": 0.0,
        }

        self.numeric_count = {
            field: 0
            for field in self.numeric_sum
        }

        # ====================================================
        # Running IAT statistics
        # ====================================================

        self.iat_count = 0

        self.iat_sum = 0.0

        self.iat_sum_sq = 0.0

        self.iat_min = None

        self.iat_max = None

        # ====================================================
        # Static / first-packet fields
        # ====================================================

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

        self.ip_protocol = first_row.get(
            "ip_protocol"
        )

        self.ip_header_length = first_row.get(
            "ip_header_length"
        )

    # ========================================================
    # Direction
    # ========================================================

    def is_forward(
        self,
        row,
    ):

        return (
            row[
                "src_ip"
            ] == self.forward_ip

            and int(
                row[
                    "src_port"
                ]
            ) == self.forward_port

            and row[
                "dst_ip"
            ] == self.forward_dst_ip

            and int(
                row[
                    "dst_port"
                ]
            ) == self.forward_dst_port
        )

    # ========================================================
    # Numeric aggregation
    # ========================================================

    def add_numeric(
        self,
        field,
        value,
    ):

        if is_missing(
            value
        ):

            return

        value = float(
            value
        )

        self.numeric_sum[
            field
        ] += value

        self.numeric_count[
            field
        ] += 1

    def numeric_mean(
        self,
        field,
    ):

        count = self.numeric_count[
            field
        ]

        if count == 0:

            return np.nan

        return (
            self.numeric_sum[
                field
            ]
            / count
        )

    # ========================================================
    # Add packet
    # ========================================================

    def add_packet(
        self,
        row,
    ):

        timestamp = float(
            row[
                "timestamp"
            ]
        )

        # ====================================================
        # Running IAT
        # ====================================================

        if self.previous_packet_time is not None:

            iat = (
                timestamp
                - self.previous_packet_time
            )

            self.iat_count += 1

            self.iat_sum += iat

            self.iat_sum_sq += (
                iat * iat
            )

            if self.iat_min is None:

                self.iat_min = iat

            else:

                self.iat_min = min(
                    self.iat_min,
                    iat,
                )

            if self.iat_max is None:

                self.iat_max = iat

            else:

                self.iat_max = max(
                    self.iat_max,
                    iat,
                )

        self.previous_packet_time = timestamp

        self.last_time = timestamp

        # ====================================================
        # Packet / byte statistics
        # ====================================================

        frame_length = float(
            row[
                "frame_length"
            ]
        )

        self.packet_count += 1

        self.byte_count += int(
            frame_length
        )

        # ====================================================
        # Forward / reverse
        # ====================================================

        if self.is_forward(
            row
        ):

            self.forward_packet_count += 1

            self.forward_byte_count += int(
                frame_length
            )

        else:

            self.reverse_packet_count += 1

            self.reverse_byte_count += int(
                frame_length
            )

        # ====================================================
        # Running numeric means
        # ====================================================

        self.add_numeric(
            "frame_length",
            row.get(
                "frame_length"
            ),
        )

        self.add_numeric(
            "ip_total_length",
            row.get(
                "ip_total_length"
            ),
        )

        self.add_numeric(
            "ip_id",
            row.get(
                "ip_id"
            ),
        )

        self.add_numeric(
            "fragment_offset",
            row.get(
                "fragment_offset"
            ),
        )

        self.add_numeric(
            "ttl",
            row.get(
                "ttl"
            ),
        )

        self.add_numeric(
            "tcp_window",
            row.get(
                "tcp_window"
            ),
        )

        self.add_numeric(
            "tcp_header_length",
            row.get(
                "tcp_header_length"
            ),
        )

        # ====================================================
        # TCP flags
        # ====================================================

        if row[
            "l4_protocol"
        ] == TCP_PROTOCOL:

            syn_value = row.get(
                "tcp_syn"
            )

            ack_value = row.get(
                "tcp_ack"
            )

            fin_value = row.get(
                "tcp_fin"
            )

            rst_value = row.get(
                "tcp_rst"
            )

            psh_value = row.get(
                "tcp_psh"
            )

            syn = (
                pd.notna(
                    syn_value
                )
                and int(
                    syn_value
                ) == 1
            )

            ack = (
                pd.notna(
                    ack_value
                )
                and int(
                    ack_value
                ) == 1
            )

            fin = (
                pd.notna(
                    fin_value
                )
                and int(
                    fin_value
                ) == 1
            )

            rst = (
                pd.notna(
                    rst_value
                )
                and int(
                    rst_value
                ) == 1
            )

            psh = (
                pd.notna(
                    psh_value
                )
                and int(
                    psh_value
                ) == 1
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

            if syn and not ack:

                self.syn_seen = 1

            if syn and ack:

                self.syn_ack_seen = 1

                self._saw_syn_ack = True

            elif (
                ack
                and self._saw_syn_ack
            ):

                self.ack_after_syn_ack = 1

        # ====================================================
        # IP checksum
        # ====================================================

        ip_valid = row.get(
            "ip_checksum_valid"
        )

        if pd.notna(
            ip_valid
        ):

            self._seen_ip_checksum = True

            if int(
                ip_valid
            ) == 0:

                self.ip_checksum_valid_all = False

        # ====================================================
        # L4 checksum
        # ====================================================

        l4_valid = row.get(
            "l4_checksum_valid"
        )

        if pd.notna(
            l4_valid
        ):

            self._seen_l4_checksum = True

            if int(
                l4_valid
            ) == 0:

                self.l4_checksum_valid_all = False

        # ====================================================
        # MAC-IP
        # ====================================================

        mac_ip_consistent = row.get(
            "src_ip_mac_consistent"
        )

        if pd.notna(
            mac_ip_consistent
        ):

            self._seen_mac_ip_evidence = True

            if int(
                mac_ip_consistent
            ) == 0:

                self.mac_ip_inconsistent_count += 1

            spoof_suspect = row.get(
                "src_mac_ip_spoof_suspect"
            )

            if (
                pd.notna(
                    spoof_suspect
                )
                and int(
                    spoof_suspect
                ) == 1
            ):

                self.mac_ip_spoof_suspect_any = 1

            mac_count = row.get(
                "src_ip_mac_seen_mac_count"
            )

            if pd.notna(
                mac_count
            ):

                mac_count = int(
                    mac_count
                )

                if (
                    self.src_ip_mac_seen_mac_count_max
                    is None
                ):

                    self.src_ip_mac_seen_mac_count_max = (
                        mac_count
                    )

                else:

                    self.src_ip_mac_seen_mac_count_max = max(
                        self.src_ip_mac_seen_mac_count_max,
                        mac_count,
                    )

            ip_count = row.get(
                "src_mac_ip_seen_ip_count"
            )

            if pd.notna(
                ip_count
            ):

                ip_count = int(
                    ip_count
                )

                if (
                    self.src_mac_ip_seen_ip_count_max
                    is None
                ):

                    self.src_mac_ip_seen_ip_count_max = (
                        ip_count
                    )

                else:

                    self.src_mac_ip_seen_ip_count_max = max(
                        self.src_mac_ip_seen_ip_count_max,
                        ip_count,
                    )

    # ========================================================
    # Flow termination
    # ========================================================

    def is_tcp_terminator(
        self,
        row,
    ):

        if row[
            "l4_protocol"
        ] != TCP_PROTOCOL:

            return False

        fin = row.get(
            "tcp_fin"
        )

        rst = row.get(
            "tcp_rst"
        )

        fin_seen = (
            pd.notna(
                fin
            )
            and int(
                fin
            ) == 1
        )

        rst_seen = (
            pd.notna(
                rst
            )
            and int(
                rst
            ) == 1
        )

        return (
            fin_seen
            or rst_seen
        )

    # ========================================================
    # Finalize
    # ========================================================

    def finalize(
        self,
    ):

        duration = (
            self.last_time
            - self.start_time
        )

        if duration < 0:

            duration = 0.0

        # ====================================================
        # IAT
        # ====================================================

        if self.iat_count > 0:

            iat_mean = (
                self.iat_sum
                / self.iat_count
            )

            variance = (
                self.iat_sum_sq
                / self.iat_count
                - iat_mean * iat_mean
            )

            variance = max(
                0.0,
                variance,
            )

            iat_std = math.sqrt(
                variance
            )

            iat_min = float(
                self.iat_min
            )

            iat_max = float(
                self.iat_max
            )

        else:

            iat_min = np.nan
            iat_mean = np.nan
            iat_max = np.nan
            iat_std = np.nan

        # ====================================================
        # Byte rate
        # ====================================================

        if duration > 0:

            byte_rate = (
                self.byte_count
                / duration
            )

        else:

            byte_rate = np.nan

        # ====================================================
        # Flow direction
        # ====================================================

        flow_direction = int(
            self.reverse_packet_count > 0
        )

        # ====================================================
        # Checksums
        # ====================================================

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

        # ====================================================
        # MAC-IP
        # ====================================================

        if self._seen_mac_ip_evidence:

            flow_mac_ip_inconsistent_count = int(
                self.mac_ip_inconsistent_count
            )

            flow_mac_ip_spoof_suspect = int(
                self.mac_ip_spoof_suspect_any
            )

            if (
                self.src_ip_mac_seen_mac_count_max
                is None
            ):

                src_ip_mac_seen_mac_count_max = np.nan

            else:

                src_ip_mac_seen_mac_count_max = int(
                    self.src_ip_mac_seen_mac_count_max
                )

            if (
                self.src_mac_ip_seen_ip_count_max
                is None
            ):

                src_mac_ip_seen_ip_count_max = np.nan

            else:

                src_mac_ip_seen_ip_count_max = int(
                    self.src_mac_ip_seen_ip_count_max
                )

        else:

            flow_mac_ip_inconsistent_count = np.nan

            flow_mac_ip_spoof_suspect = np.nan

            src_ip_mac_seen_mac_count_max = np.nan

            src_mac_ip_seen_ip_count_max = np.nan

        # ====================================================
        # Final row
        # ====================================================

        return {

            "flow_id":
                self.flow_id,

            "timestamp":
                self.start_time,

            "src_ip":
                self.src_ip,

            "dst_ip":
                self.dst_ip,

            "src_mac":
                self.src_mac,

            "dst_mac":
                self.dst_mac,

            "frame_length_mean":
                self.numeric_mean(
                    "frame_length"
                ),

            "eth_type":
                first_value_or_nan(
                    self.eth_type
                ),

            "vlan_id":
                first_value_or_nan(
                    self.vlan_id
                ),

            "vlan_priority":
                first_value_or_nan(
                    self.vlan_priority
                ),

            "ip_version":
                first_value_or_nan(
                    self.ip_version
                ),

            "ip_dscp":
                first_value_or_nan(
                    self.ip_dscp
                ),

            "ip_ecn":
                first_value_or_nan(
                    self.ip_ecn
                ),

            "ip_total_length_mean":
                self.numeric_mean(
                    "ip_total_length"
                ),

            "ip_id_mean":
                self.numeric_mean(
                    "ip_id"
                ),

            "ip_df":
                first_value_or_nan(
                    self.ip_df
                ),

            "ip_mf":
                first_value_or_nan(
                    self.ip_mf
                ),

            "fragment_offset":
                self.numeric_mean(
                    "fragment_offset"
                ),

            "ttl_mean":
                self.numeric_mean(
                    "ttl"
                ),

            "ip_protocol":
                first_value_or_nan(
                    self.ip_protocol
                ),

            "ip_header_length":
                first_value_or_nan(
                    self.ip_header_length
                ),

            "ip_checksum_valid":
                ip_checksum_valid,

            "flow_mac_ip_inconsistent_count":
                flow_mac_ip_inconsistent_count,

            "flow_mac_ip_spoof_suspect":
                flow_mac_ip_spoof_suspect,

            "src_ip_mac_seen_mac_count_max":
                src_ip_mac_seen_mac_count_max,

            "src_mac_ip_seen_ip_count_max":
                src_mac_ip_seen_ip_count_max,

            "src_port":
                self.src_port,

            "dst_port":
                self.dst_port,

            "tcp_window_mean":
                self.numeric_mean(
                    "tcp_window"
                ),

            "tcp_header_length_mean":
                self.numeric_mean(
                    "tcp_header_length"
                ),

            "l4_checksum_valid":
                l4_checksum_valid,

            "flow_duration":
                duration,

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

            "iat_min":
                iat_min,

            "iat_mean":
                iat_mean,

            "iat_max":
                iat_max,

            "iat_std":
                iat_std,

            "byte_rate":
                byte_rate,

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


# ============================================================
# Streaming output
# ============================================================

class FlowOutputWriter:

    def __init__(
        self,
        output_path,
    ):

        self.output_path = Path(
            output_path
        )

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        config.validate_ml_features()

        missing_features = [

            feature

            for feature in config.ML_FEATURES

            if feature not in FLOW_COLUMNS
        ]

        if missing_features:

            raise ValueError(
                "Missing ML features from flow output:\n"
                + "\n".join(
                    f"  - {feature}"
                    for feature in missing_features
                )
            )

        metadata_columns = [

            column

            for column in config.METADATA_COLUMNS

            if column in FLOW_COLUMNS
        ]

        ml_columns = [

            column

            for column in config.ML_FEATURES

            if column in FLOW_COLUMNS
        ]

        already_used = set(
            metadata_columns
            + ml_columns
        )

        remaining_columns = [

            column

            for column in FLOW_COLUMNS

            if column not in already_used
        ]

        self.columns = (
            metadata_columns
            + ml_columns
            + remaining_columns
        )

        self.file = self.output_path.open(
            mode="w",
            newline="",
            encoding="utf-8",
        )

        self.writer = csv.DictWriter(
            self.file,
            fieldnames=self.columns,
            extrasaction="ignore",
        )

        self.writer.writeheader()

        self.total_flows = 0

        self.ml_nan_counts = {
            feature: 0
            for feature in config.ML_FEATURES
        }

        self.infinite_count = 0

        self.mac_ip_evidence_count = 0

        self.mac_ip_suspicious_count = 0

        self.protocol_counts = Counter()

        self.direction_counts = Counter()

        self.preview_rows = []

        self.packet_stat_n = 0
        self.packet_stat_mean = 0.0
        self.packet_stat_m2 = 0.0

        self.packet_stat_min = None
        self.packet_stat_max = None

    def write(
        self,
        row,
    ):

        self.writer.writerow(
            row
        )

        self.total_flows += 1

        if len(
            self.preview_rows
        ) < 5:

            self.preview_rows.append(
                row.copy()
            )

        for feature in config.ML_FEATURES:

            value = row.get(
                feature
            )

            if pd.isna(
                value
            ):

                self.ml_nan_counts[
                    feature
                ] += 1

            elif isinstance(
                value,
                (
                    int,
                    float,
                    np.integer,
                    np.floating,
                ),
            ):

                if np.isinf(
                    float(
                        value
                    )
                ):

                    self.infinite_count += 1

        mac_ip = row.get(
            "flow_mac_ip_spoof_suspect"
        )

        if pd.notna(
            mac_ip
        ):

            self.mac_ip_evidence_count += 1

            if int(
                mac_ip
            ) == 1:

                self.mac_ip_suspicious_count += 1

        protocol = row.get(
            "ip_protocol"
        )

        if pd.notna(
            protocol
        ):

            self.protocol_counts[
                int(
                    protocol
                )
            ] += 1

        direction = row.get(
            "flow_direction"
        )

        if pd.notna(
            direction
        ):

            self.direction_counts[
                int(
                    direction
                )
            ] += 1

        packet_count = float(
            row[
                "flow_packet_count"
            ]
        )

        self.packet_stat_n += 1

        delta = (
            packet_count
            - self.packet_stat_mean
        )

        self.packet_stat_mean += (
            delta
            / self.packet_stat_n
        )

        delta2 = (
            packet_count
            - self.packet_stat_mean
        )

        self.packet_stat_m2 += (
            delta
            * delta2
        )

        if self.packet_stat_min is None:

            self.packet_stat_min = packet_count

        else:

            self.packet_stat_min = min(
                self.packet_stat_min,
                packet_count,
            )

        if self.packet_stat_max is None:

            self.packet_stat_max = packet_count

        else:

            self.packet_stat_max = max(
                self.packet_stat_max,
                packet_count,
            )

    def close(
        self,
    ):

        self.file.close()

    def print_statistics(
        self,
    ):

        print()

        print(
            f"Written: {self.output_path}"
        )

        print(
            f"Total flows: {self.total_flows:,}"
        )

        print()

        print(
            "Sanity checks:"
        )

        print(
            f"  [OK] All "
            f"{len(config.ML_FEATURES)} "
            f"ML features are present."
        )

        nan_features = {

            feature: count

            for feature, count
            in self.ml_nan_counts.items()

            if count > 0
        }

        if nan_features:

            print(
                "  [INFO] ML features containing NaN:"
            )

            for feature, count in nan_features.items():

                print(
                    f"         {feature}: "
                    f"{count:,} flows"
                )

        else:

            print(
                "  [OK] No NaN values in ML features."
            )

        if self.infinite_count > 0:

            print(
                f"  [ERROR] "
                f"{self.infinite_count:,} infinite "
                f"ML values detected."
            )

        else:

            print(
                "  [OK] No infinite ML feature values."
            )

        print()

        print(
            "MAC-IP integrity statistics:"
        )

        print(
            f"  Flows with MAC-IP evidence : "
            f"{self.mac_ip_evidence_count:,}"
        )

        print(
            f"  Flows flagged suspicious   : "
            f"{self.mac_ip_suspicious_count:,}"
        )

        if self.packet_stat_n > 0:

            if self.packet_stat_n > 1:

                packet_std = math.sqrt(
                    self.packet_stat_m2
                    / (
                        self.packet_stat_n - 1
                    )
                )

            else:

                packet_std = np.nan

            print()

            print(
                "Flow packet-count statistics:"
            )

            print(
                f"  count : "
                f"{self.packet_stat_n:,}"
            )

            print(
                f"  mean  : "
                f"{self.packet_stat_mean:.6f}"
            )

            print(
                f"  std   : "
                f"{packet_std:.6f}"
            )

            print(
                f"  min   : "
                f"{self.packet_stat_min:.0f}"
            )

            print(
                f"  max   : "
                f"{self.packet_stat_max:.0f}"
            )

        if self.protocol_counts:

            print()

            print(
                "Flow protocol breakdown:"
            )

            for protocol in sorted(
                self.protocol_counts
            ):

                print(
                    f"  {protocol}: "
                    f"{self.protocol_counts[protocol]:,}"
                )

        if self.direction_counts:

            print()

            print(
                "Flow direction breakdown:"
            )

            for direction in sorted(
                self.direction_counts
            ):

                print(
                    f"  {direction}: "
                    f"{self.direction_counts[direction]:,}"
                )

        if self.preview_rows:

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

                "flow_mac_ip_inconsistent_count",

                "flow_mac_ip_spoof_suspect",

                "src_ip_mac_seen_mac_count_max",

                "src_mac_ip_seen_ip_count_max",
            ]

            preview_df = pd.DataFrame(
                self.preview_rows
            )

            preview_columns = [

                column

                for column in preview_columns

                if column in preview_df.columns
            ]

            print(
                preview_df[
                    preview_columns
                ]
                .to_string(
                    index=False
                )
            )


# ============================================================
# Idle flow cleanup
# ============================================================

def cleanup_expired_flows(
    active,
    current_timestamp,
    idle_timeout,
    output_writer,
):

    expired_keys = []

    for key, flow in active.items():

        gap = (
            current_timestamp
            - flow.last_time
        )

        if gap > idle_timeout:

            expired_keys.append(
                key
            )

    for key in expired_keys:

        flow = active.pop(
            key
        )

        output_writer.write(
            flow.finalize()
        )

    return len(
        expired_keys
    )


# ============================================================
# Streaming builder
# ============================================================

def build_flows_streaming(
    input_path,
    output_path,
    idle_timeout=None,
    limit=None,
    chunk_size=PACKET_CHUNK_SIZE,
):

    if idle_timeout is None:

        idle_timeout = (
            config.FLOW_IDLE_TIMEOUT
        )

    input_path = Path(
        input_path
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Packet feature CSV not found: "
            f"{input_path}"
        )

    header_columns = list(
        pd.read_csv(
            input_path,
            nrows=0,
        ).columns
    )

    missing_required = [

        column

        for column in BASE_INPUT_COLUMNS

        if column not in header_columns
    ]

    if missing_required:

        raise ValueError(
            "Packet CSV is missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_required
            )
        )

    missing_mac_ip_columns = [

        column

        for column in MAC_IP_COLUMNS

        if column not in header_columns
    ]

    if missing_mac_ip_columns:

        print()

        print(
            "  [WARN] MAC-IP packet features are missing."
        )

        print(
            "         The flow builder will use NaN "
            "for those fields."
        )

        for column in missing_mac_ip_columns:

            print(
                f"         Missing: {column}"
            )

    use_columns = (
        BASE_INPUT_COLUMNS
        + [
            column

            for column in MAC_IP_COLUMNS

            if column in header_columns
        ]
    )

    active = {}

    flow_counter = 0

    tcp_fin_rst_terminations = 0

    idle_timeout_terminations = 0

    total_packet_rows = 0

    eligible_packet_rows = 0

    last_input_timestamp = None

    last_cleanup_timestamp = None

    output_writer = FlowOutputWriter(
        output_path
    )

    print()

    print(
        "Building bidirectional flows in streaming mode..."
    )

    print(
        f"Packet chunk size: "
        f"{chunk_size:,}"
    )

    try:

        packet_chunks = pd.read_csv(
            input_path,
            usecols=use_columns,
            chunksize=chunk_size,
            nrows=limit,
        )

        for chunk_number, chunk in enumerate(
            packet_chunks,
            start=1,
        ):

            for column in missing_mac_ip_columns:

                chunk[
                    column
                ] = np.nan

            print(
                f"  Processing chunk "
                f"{chunk_number:,} "
                f"({len(chunk):,} rows)..."
            )

            for record in chunk.itertuples(
                index=False,
            ):

                row = record._asdict()

                total_packet_rows += 1

                timestamp_value = row.get(
                    "timestamp"
                )

                if pd.isna(
                    timestamp_value
                ):

                    raise ValueError(
                        "Encountered packet with missing timestamp."
                    )

                current_timestamp = float(
                    timestamp_value
                )

                # ====================================================
                # ==================== CHANGED ========================
                # TIMESTAMP JITTER HANDLING
                #
                # Small backwards differences <= 1 millisecond are
                # treated as timestamp jitter.
                #
                # Large backwards jumps still raise an exception.
                #
                # IMPORTANT:
                # When a small backwards timestamp is corrected,
                # row["timestamp"] must also be corrected because
                # FlowState.add_packet() reads that value later.
                # ====================================================

                if last_input_timestamp is not None:

                    backward_gap = (
                        last_input_timestamp
                        - current_timestamp
                    )

                    if (
                        backward_gap
                        > TIMESTAMP_TOLERANCE
                    ):

                        raise ValueError(
                            "packet_features.csv is not in "
                            "chronological timestamp order.\n"
                            f"Previous timestamp: "
                            f"{last_input_timestamp}\n"
                            f"Current timestamp : "
                            f"{current_timestamp}\n"
                            f"Backward gap      : "
                            f"{backward_gap:.9f} seconds\n"
                            "\n"
                            "The streaming flow builder does not "
                            "perform an in-memory full-file sort."
                        )

                    if (
                        current_timestamp
                        < last_input_timestamp
                    ):

                        current_timestamp = (
                            last_input_timestamp
                        )

                        # ============================================
                        # ==================== ADDED ==================
                        #
                        # Keep the row timestamp consistent with the
                        # corrected streaming timestamp.
                        #
                        # Otherwise FlowState.add_packet() could still
                        # calculate a negative IAT from the original
                        # slightly-backward timestamp.
                        # ============================================

                        row[
                            "timestamp"
                        ] = current_timestamp

                last_input_timestamp = (
                    current_timestamp
                )

                # ====================================================
                # Periodic idle cleanup
                # ====================================================

                if last_cleanup_timestamp is None:

                    last_cleanup_timestamp = (
                        current_timestamp
                    )

                elif (
                    current_timestamp
                    - last_cleanup_timestamp
                    >= IDLE_CLEANUP_INTERVAL
                ):

                    expired_count = cleanup_expired_flows(
                        active=active,
                        current_timestamp=current_timestamp,
                        idle_timeout=idle_timeout,
                        output_writer=output_writer,
                    )

                    idle_timeout_terminations += (
                        expired_count
                    )

                    last_cleanup_timestamp = (
                        current_timestamp
                    )

                # ====================================================
                # Protocol filter
                # ====================================================

                if row.get(
                    "l4_protocol"
                ) not in SUPPORTED_PROTOCOLS:

                    continue

                required_values = [

                    row.get(
                        "src_ip"
                    ),

                    row.get(
                        "dst_ip"
                    ),

                    row.get(
                        "src_port"
                    ),

                    row.get(
                        "dst_port"
                    ),

                    row.get(
                        "ip_protocol"
                    ),
                ]

                if any(
                    pd.isna(
                        value
                    )
                    for value in required_values
                ):

                    continue

                eligible_packet_rows += 1

                # ====================================================
                # Flow key
                # ====================================================

                key = make_flow_key(

                    row[
                        "src_ip"
                    ],

                    row[
                        "dst_ip"
                    ],

                    row[
                        "src_port"
                    ],

                    row[
                        "dst_port"
                    ],

                    row[
                        "l4_protocol"
                    ],
                )

                flow = active.get(
                    key
                )

                # ====================================================
                # Per-key idle timeout
                # ====================================================

                if flow is not None:

                    gap = (
                        current_timestamp
                        - flow.last_time
                    )

                    if gap > idle_timeout:

                        output_writer.write(
                            flow.finalize()
                        )

                        del active[
                            key
                        ]

                        flow = None

                        idle_timeout_terminations += 1

                # ====================================================
                # Start flow
                # ====================================================

                if flow is None:

                    flow_counter += 1

                    flow = FlowState(

                        f"FLOW_{flow_counter:09d}",

                        row,
                    )

                    active[
                        key
                    ] = flow

                # ====================================================
                # Add packet
                # ====================================================

                flow.add_packet(
                    row
                )

                # ====================================================
                # TCP FIN / RST
                # ====================================================

                if flow.is_tcp_terminator(
                    row
                ):

                    output_writer.write(
                        flow.finalize()
                    )

                    del active[
                        key
                    ]

                    tcp_fin_rst_terminations += 1

            del chunk

        # ====================================================
        # EOF
        # ====================================================

        active_at_eof = len(
            active
        )

        for flow in active.values():

            output_writer.write(
                flow.finalize()
            )

        active.clear()

    finally:

        output_writer.close()

    # ========================================================
    # Results
    # ========================================================

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
        f"{active_at_eof:,}"
    )

    print()

    print(
        f"Packet rows read              : "
        f"{total_packet_rows:,}"
    )

    print(
        f"TCP/UDP packets eligible      : "
        f"{eligible_packet_rows:,}"
    )

    print(
        f"Flows built                   : "
        f"{output_writer.total_flows:,}"
    )

    output_writer.print_statistics()

    return output_writer.total_flows


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Aggregate packet-level features into "
            "bidirectional TCP/UDP flows using streaming "
            "CSV processing."
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

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=PACKET_CHUNK_SIZE,
        help=(
            "Number of packet rows to load into RAM "
            "per CSV chunk."
        ),
    )

    args = parser.parse_args()

    if args.chunk_size <= 0:

        raise ValueError(
            "--chunk-size must be greater than 0."
        )

    if (
        args.limit is not None
        and args.limit <= 0
    ):

        raise ValueError(
            "--limit must be greater than 0 "
            "when specified."
        )

    print(
        "=" * 60
    )

    print(
        "NetworkIDS - Streaming Flow Builder"
    )

    print(
        "=" * 60
    )

    print(
        f"Input        : {args.input_csv}"
    )

    print(
        f"Output       : {args.output_csv}"
    )

    print(
        f"Idle timeout : "
        f"{config.FLOW_IDLE_TIMEOUT} seconds"
    )

    print(
        f"Chunk size   : "
        f"{args.chunk_size:,} packet rows"
    )

    if args.limit is not None:

        print(
            f"Packet limit : "
            f"{args.limit:,}"
        )

    else:

        print(
            "Packet limit : whole file"
        )

    build_flows_streaming(

        input_path=args.input_csv,

        output_path=args.output_csv,

        idle_timeout=(
            config.FLOW_IDLE_TIMEOUT
        ),

        limit=args.limit,

        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()