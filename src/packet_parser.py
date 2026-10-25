"""
packet_parser.py
----------------
NetworkIDS - Packet-level L2-L4 feature extraction.

One row = one packet.
Flow construction happens later in flow_builder.py.

Missing-value policy:
    A field that does not exist for this packet -> None (becomes NaN in pandas).
    Never 0, never -1.

Usage:
    python src/packet_parser.py --preview 20
    python src/packet_parser.py --limit 10000 --out data/processed/packet_features.csv
"""

import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict

import logging
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

from scapy.all import (
    PcapReader,
    Ether,
    Dot1Q,
    IP,
    IPv6,
    TCP,
    UDP,
    ICMP,
    raw,
)

from scapy.layers.inet import in4_chksum 


sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

import src.config as config


# ============================================================
# Column order for the packet-level CSV

PACKET_COLUMNS = [

    # metadata
    "packet_id",
    "timestamp",

    # L2
    "frame_length",
    "eth_type",
    "vlan_id",
    "vlan_priority",
    "src_mac",
    "dst_mac",

    # L3
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
    "ip_checksum",
    "ip_checksum_valid",

    # ========================================================
    # MAC-IP consistency
    # ========================================================

    # 1 = this source IP has only been observed with one MAC so far
    # 0 = this source IP has been observed with multiple MACs
    # None = source IP or source MAC is unavailable
    "src_ip_mac_consistent",

    # Number of distinct MAC addresses observed for this source IP.
    "src_ip_mac_seen_mac_count",

    # Number of distinct IP addresses observed for this source MAC.
    #
    # This is contextual information only.
    # A MAC associated with many IPs is NOT independently considered
    # proof of spoofing.
    "src_mac_ip_seen_ip_count",

    # Lightweight suspicion indicator.
    # 1 does NOT prove spoofing.
    "src_mac_ip_spoof_suspect",

    # L4
    "l4_protocol",
    "src_port",
    "dst_port",
    "tcp_window",
    "tcp_header_length",
    "l4_checksum",
    "l4_checksum_valid",
    "tcp_syn",
    "tcp_ack",
    "tcp_fin",
    "tcp_rst",
    "tcp_psh",
    "tcp_urg",
]


# ============================================================
# MAC-IP tracking state
#
# These dictionaries contain the observations for ONE PCAP.
#
# Example:
#
#   192.168.1.10 -> AA:AA:AA:AA:AA:AA
#   192.168.1.10 -> BB:BB:BB:BB:BB:BB
#
# means that the same IP has appeared with two different MAC
# addresses during the current capture.
#
# Important:
#
# Multiple MAC addresses do NOT automatically prove spoofing.
# DHCP changes, VM migration, failover, proxy ARP and other
# legitimate situations can also cause this behaviour.
# ============================================================

ip_to_macs = defaultdict(set)
mac_to_ips = defaultdict(set)


# ============================================================
# ==================== ADDED: RESET MAC-IP STATE ==============
#
# Each PCAP must start with a fresh MAC-IP observation history.
#
# Without this reset:
#
# PCAP A:
#     10.0.0.5 -> MAC AA
#
# PCAP B:
#     10.0.0.5 -> MAC BB
#
# could incorrectly look like one IP changed MAC inside the same
# capture.
# ============================================================

def reset_mac_ip_tracking():
    """
    Clear all MAC-IP observations before analysing a new PCAP.
    """

    ip_to_macs.clear()
    mac_to_ips.clear()


# ============================================================
# MAC-IP consistency

def update_mac_ip_consistency(src_mac, src_ip):
    """
    Update and return source MAC-IP consistency information.

    This is packet-level integrity evidence only.

    It does NOT prove MAC spoofing or IP spoofing.

    Current rule:
        - If the same source IP has been observed using more than
          one source MAC address, the mapping becomes inconsistent.

    The reverse relationship:

        one MAC -> many IP addresses

    is recorded as contextual information but does NOT independently
    trigger the spoof-suspicion flag because routers, gateways,
    virtualisation environments and other legitimate systems may
    represent multiple IP addresses.

    Returns
    -------
    dict
        src_ip_mac_consistent
        src_ip_mac_seen_mac_count
        src_mac_ip_seen_ip_count
        src_mac_ip_spoof_suspect
    """

    # MAC-IP comparison is impossible when either value is missing.
    if not src_mac or not src_ip:

        return {
            "src_ip_mac_consistent": None,
            "src_ip_mac_seen_mac_count": 0,
            "src_mac_ip_seen_ip_count": 0,
            "src_mac_ip_spoof_suspect": 0,
        }

    src_ip = str(
        src_ip
    ).strip()

    src_mac = str(
        src_mac
    ).lower().strip()

    # --------------------------------------------------------
    # Record observations
    # --------------------------------------------------------

    ip_to_macs[
        src_ip
    ].add(
        src_mac
    )

    mac_to_ips[
        src_mac
    ].add(
        src_ip
    )

    mac_count_for_ip = len(
        ip_to_macs[
            src_ip
        ]
    )

    ip_count_for_mac = len(
        mac_to_ips[
            src_mac
        ]
    )

    # --------------------------------------------------------
    # Consistency result
    #
    # One IP -> one observed MAC
    #     consistent = 1
    #
    # One IP -> multiple observed MACs
    #     consistent = 0
    # --------------------------------------------------------

    if mac_count_for_ip <= 1:
        consistent = 1

    else:
        consistent = 0

    # --------------------------------------------------------
    # Suspicion indicator
    #
    # CHANGED:
    #
    # Only:
    #
    #     same IP -> multiple MAC addresses
    #
    # directly raises this indicator.
    #
    # One MAC associated with many IP addresses is still counted
    # in src_mac_ip_seen_ip_count but does NOT independently raise
    # the suspicion flag.
    # --------------------------------------------------------

    spoof_suspect = int(
        mac_count_for_ip > 1
    )

    return {
        "src_ip_mac_consistent":
            consistent,

        "src_ip_mac_seen_mac_count":
            mac_count_for_ip,

        "src_mac_ip_seen_ip_count":
            ip_count_for_mac,

        "src_mac_ip_spoof_suspect":
            spoof_suspect,
    }


# ============================================================
# Checksum validation

def ipv4_checksum_valid(ip_layer):
    """
    Recompute the IPv4 header checksum and compare it with the value
    carried in the packet.

    Returns True / False, or None if it cannot be determined.
    """

    try:

        stored = ip_layer.chksum

        if stored is None:
            return None

        # Rebuild the header with the checksum field cleared so that
        # Scapy recomputes it from scratch.
        #
        # 这里可能后续会有问题，没问题就别动了

        header = ip_layer.copy()

        header.chksum = None

        del header.payload

        recomputed = IP(
            raw(
                header
            )
        ).chksum

        return bool(
            stored == recomputed
        )

    except Exception:

        return None


def l4_checksum_valid(pkt):
    """
    Recompute the TCP/UDP checksum.

    Returns True / False, or None when the checksum is absent or the
    protocol is not TCP/UDP.

    NOTE:
    Checksum offloading on the capturing host can produce false
    "invalid" results for locally generated traffic.

    For imported captures such as CIC-IDS2017 this is usually not an
    issue, but it is worth remembering when testing live traffic later.
    """

    try:

        if TCP in pkt:

            layer = pkt[
                TCP
            ]

        elif UDP in pkt:

            layer = pkt[
                UDP
            ]

        else:

            return None

        stored = layer.chksum

        if stored is None:

            return None

        # UDP checksum 0 means:
        #
        # "checksum not computed"
        #
        # which is legal for IPv4 UDP.

        if (
            UDP in pkt
            and stored == 0
        ):

            return None

        # Current implementation validates IPv4 TCP/UDP only.

        if IP not in pkt:

            return None

        probe = layer.copy()

        probe.chksum = 0

        recomputed = in4_chksum(
            pkt[IP].proto,
            pkt[IP],
            raw(
                probe
            ),
        )

        return bool(
            stored == recomputed
        )

    except Exception:

        return None


# ============================================================
# Per-packet extraction

def parse_packet(pkt, packet_id):
    """
    Extract L2-L4 header fields from a single Scapy packet.

    Every field that does not apply to this packet is left as None.
    """

    row = {
        name: None
        for name in PACKET_COLUMNS
    }

    row[
        "packet_id"
    ] = packet_id

    row[
        "timestamp"
    ] = float(
        pkt.time
    )

    row[
        "frame_length"
    ] = len(
        pkt
    )

    # ========================================================
    # L2
    # ========================================================

    if Ether in pkt:

        eth = pkt[
            Ether
        ]

        row[
            "src_mac"
        ] = eth.src

        row[
            "dst_mac"
        ] = eth.dst

        row[
            "eth_type"
        ] = int(
            eth.type
        )

    if Dot1Q in pkt:

        vlan = pkt[
            Dot1Q
        ]

        row[
            "vlan_id"
        ] = int(
            vlan.vlan
        )

        row[
            "vlan_priority"
        ] = int(
            vlan.prio
        )

        # With a VLAN tag the real upper-layer type
        # sits inside the Dot1Q header.

        row[
            "eth_type"
        ] = int(
            vlan.type
        )

    # ========================================================
    # L3
    # ========================================================

    if IP in pkt:

        ip = pkt[
            IP
        ]

        row[
            "ip_version"
        ] = 4

        row[
            "src_ip"
        ] = ip.src

        row[
            "dst_ip"
        ] = ip.dst

        tos = int(
            ip.tos
        )

        row[
            "ip_dscp"
        ] = (
            tos >> 2
        )

        row[
            "ip_ecn"
        ] = (
            tos & 0b11
        )

        row[
            "ip_total_length"
        ] = int(
            ip.len
        )

        row[
            "ip_id"
        ] = int(
            ip.id
        )

        flags = int(
            ip.flags
        )

        row[
            "ip_df"
        ] = int(
            bool(
                flags & 0b010
            )
        )

        row[
            "ip_mf"
        ] = int(
            bool(
                flags & 0b001
            )
        )

        row[
            "fragment_offset"
        ] = int(
            ip.frag
        )

        row[
            "ttl"
        ] = int(
            ip.ttl
        )

        row[
            "ip_protocol"
        ] = int(
            ip.proto
        )

        row[
            "ip_header_length"
        ] = (
            int(
                ip.ihl
            )
            * 4
        )

        row[
            "ip_checksum"
        ] = (
            int(
                ip.chksum
            )
            if ip.chksum is not None
            else None
        )

        valid = ipv4_checksum_valid(
            ip
        )

        row[
            "ip_checksum_valid"
        ] = (
            None
            if valid is None
            else int(
                valid
            )
        )

    elif IPv6 in pkt:

        ip6 = pkt[
            IPv6
        ]

        row[
            "ip_version"
        ] = 6

        row[
            "src_ip"
        ] = ip6.src

        row[
            "dst_ip"
        ] = ip6.dst

        tc = int(
            ip6.tc
        )

        row[
            "ip_dscp"
        ] = (
            tc >> 2
        )

        row[
            "ip_ecn"
        ] = (
            tc & 0b11
        )

        # IPv6 payload length excludes the 40-byte
        # base IPv6 header.

        row[
            "ip_total_length"
        ] = (
            40
            + int(
                ip6.plen
            )
        )

        # IPv6 Hop Limit is the TTL equivalent.

        row[
            "ttl"
        ] = int(
            ip6.hlim
        )

        row[
            "ip_protocol"
        ] = int(
            ip6.nh
        )

        # IPv6 base header has fixed length.

        row[
            "ip_header_length"
        ] = 40

        # IPv6 has no IPv4-style:
        #
        #     header checksum
        #     IP ID field
        #     DF/MF fields
        #
        # Those remain None intentionally.

    # ========================================================
    # MAC-IP consistency
    #
    # This must happen AFTER:
    #
    #     src_mac
    #     src_ip
    #
    # have been extracted.
    # ========================================================

    mac_ip_info = update_mac_ip_consistency(
        row[
            "src_mac"
        ],
        row[
            "src_ip"
        ],
    )

    row[
        "src_ip_mac_consistent"
    ] = mac_ip_info[
        "src_ip_mac_consistent"
    ]

    row[
        "src_ip_mac_seen_mac_count"
    ] = mac_ip_info[
        "src_ip_mac_seen_mac_count"
    ]

    row[
        "src_mac_ip_seen_ip_count"
    ] = mac_ip_info[
        "src_mac_ip_seen_ip_count"
    ]

    row[
        "src_mac_ip_spoof_suspect"
    ] = mac_ip_info[
        "src_mac_ip_spoof_suspect"
    ]

    # ========================================================
    # L4
    # ========================================================

    if TCP in pkt:

        tcp = pkt[
            TCP
        ]

        row[
            "l4_protocol"
        ] = "TCP"

        row[
            "src_port"
        ] = int(
            tcp.sport
        )

        row[
            "dst_port"
        ] = int(
            tcp.dport
        )

        row[
            "tcp_window"
        ] = int(
            tcp.window
        )

        row[
            "tcp_header_length"
        ] = (
            int(
                tcp.dataofs
            )
            * 4
        )

        row[
            "l4_checksum"
        ] = (
            int(
                tcp.chksum
            )
            if tcp.chksum is not None
            else None
        )

        flags = int(
            tcp.flags
        )

        row[
            "tcp_fin"
        ] = int(
            bool(
                flags & 0x01
            )
        )

        row[
            "tcp_syn"
        ] = int(
            bool(
                flags & 0x02
            )
        )

        row[
            "tcp_rst"
        ] = int(
            bool(
                flags & 0x04
            )
        )

        row[
            "tcp_psh"
        ] = int(
            bool(
                flags & 0x08
            )
        )

        row[
            "tcp_ack"
        ] = int(
            bool(
                flags & 0x10
            )
        )

        row[
            "tcp_urg"
        ] = int(
            bool(
                flags & 0x20
            )
        )

    elif UDP in pkt:

        udp = pkt[
            UDP
        ]

        row[
            "l4_protocol"
        ] = "UDP"

        row[
            "src_port"
        ] = int(
            udp.sport
        )

        row[
            "dst_port"
        ] = int(
            udp.dport
        )

        row[
            "l4_checksum"
        ] = (
            int(
                udp.chksum
            )
            if udp.chksum is not None
            else None
        )

        # TCP-only fields remain None.

    elif ICMP in pkt:

        row[
            "l4_protocol"
        ] = "ICMP"

        # ICMP has no TCP/UDP ports.

    elif (
        IPv6 in pkt
        and int(
            pkt[
                IPv6
            ].nh
        ) == 58
    ):

        row[
            "l4_protocol"
        ] = "ICMPv6"

        # ICMPv6 also has no TCP/UDP ports.

    elif row[
        "ip_version"
    ] is not None:

        row[
            "l4_protocol"
        ] = "OTHER"

    valid = l4_checksum_valid(
        pkt
    )

    row[
        "l4_checksum_valid"
    ] = (
        None
        if valid is None
        else int(
            valid
        )
    )

    return row


# ============================================================
# PCAP reading

def iter_packets(
    pcap_path,
    limit=None,
    progress_every=100_000,
):
    """
    Stream a PCAP file packet by packet.

    PcapReader is used instead of rdpcap because large captures
    should not be loaded fully into RAM.

    Current streaming behaviour:

        PCAP
          ↓
        read one packet
          ↓
        parse_packet()
          ↓
        yield one row
          ↓
        caller processes it immediately
    """

    pcap_path = Path(
        pcap_path
    )

    if not pcap_path.exists():

        raise FileNotFoundError(
            f"PCAP not found: {pcap_path}"
        )

    # ========================================================
    # ==================== ADDED: RESET =======================
    #
    # Every call to iter_packets() represents a fresh PCAP
    # analysis session.
    #
    # Therefore MAC-IP history from a previous capture must not
    # leak into this one.
    # ========================================================

    reset_mac_ip_tracking()

    packet_id = 0
    skipped = 0

    with PcapReader(
        str(
            pcap_path
        )
    ) as reader:

        for pkt in reader:

            packet_id += 1

            try:

                # Yield immediately.
                #
                # No giant rows[] list is created.

                yield parse_packet(
                    pkt,
                    packet_id,
                )

            except Exception as exc:

                skipped += 1

                if skipped <= 5:

                    print(
                        f"  [warn] packet "
                        f"{packet_id} skipped: {exc}"
                    )

            if (
                progress_every
                and packet_id
                % progress_every
                == 0
            ):

                print(
                    f"  parsed "
                    f"{packet_id:,} packets..."
                )

            if (
                limit is not None
                and packet_id >= limit
            ):

                break

    if skipped:

        print(
            f"  [warn] "
            f"{skipped} packet(s) "
            f"could not be parsed."
        )


# ============================================================
# Preview helper

def preview(rows, count):
    """
    Print a small readable sample of parsed packets.
    """

    show = [

        "packet_id",
        "timestamp",
        "frame_length",

        "src_mac",
        "dst_mac",
        "eth_type",

        "src_ip",
        "dst_ip",
        "ttl",
        "ip_protocol",
        "ip_checksum_valid",

        # MAC-IP evidence
        "src_ip_mac_consistent",
        "src_ip_mac_seen_mac_count",
        "src_mac_ip_seen_ip_count",
        "src_mac_ip_spoof_suspect",

        "l4_protocol",
        "src_port",
        "dst_port",

        "tcp_window",
        "tcp_syn",
        "tcp_ack",
        "tcp_fin",
        "tcp_rst",

        "l4_checksum_valid",
    ]

    for row in rows[
        :count
    ]:

        print(
            "-" * 60
        )

        for key in show:

            value = row.get(
                key
            )

            value = (
                "NaN"
                if value is None
                else value
            )

            print(
                f"  {key:<32} {value}"
            )


# ============================================================
# Main

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extract L2-L4 header features "
            "from a PCAP file."
        )
    )

    parser.add_argument(
        "--pcap",
        required=True,
        help="Path to the PCAP file.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10_000,
        help=(
            "Stop after this many packets. "
            "Use 0 for the whole file."
        ),
    )

    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help=(
            "Print this many parsed packets "
            "instead of writing a CSV."
        ),
    )

    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Where to write the packet-level CSV."
        ),
    )

    args = parser.parse_args()

    limit = (
        None
        if args.limit == 0
        else args.limit
    )

    print(
        "=" * 60
    )

    print(
        "NetworkIDS - Packet Parser"
    )

    print(
        "=" * 60
    )

    print(
        f"PCAP  : {args.pcap}"
    )

    if args.preview:

        print(
            f"Mode  : preview "
            f"({args.preview} packets)"
        )

    else:

        print(
            f"Limit : "
            f"{'whole file' if limit is None else f'{limit:,} packets'}"
        )

    print()

    # ========================================================
    # Streaming input
    # ========================================================

    packet_stream = iter_packets(
        args.pcap,
        limit=limit,
    )

    # ========================================================
    # Preview mode
    # ========================================================

    if args.preview:

        import itertools

        rows = list(
            itertools.islice(
                packet_stream,
                args.preview,
            )
        )

        print()

        print(
            f"Parsed "
            f"{len(rows):,} packets."
        )

        print()

        preview(
            rows,
            args.preview,
        )

        print(
            "-" * 60
        )

        return

    # ========================================================
    # CSV output mode
    # ========================================================

    out_path = (
        Path(
            args.out
        )
        if args.out
        else config.PACKET_FEATURES_FILE
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Chunked writing
    #
    # Only this number of parsed rows is held in memory at once.
    # ========================================================

    CHUNK_SIZE = 200_000

    buffer = []

    total_written = 0

    first_write = True

    # ========================================================
    # Chunk writer
    # ========================================================

    def flush(buffer):
        """
        Write one packet-feature chunk to CSV.
        """

        nonlocal total_written
        nonlocal first_write

        if not buffer:
            return

        with out_path.open(
            mode=(
                "w"
                if first_write
                else "a"
            ),
            newline="",
            encoding="utf-8",
        ) as output:

            writer = csv.DictWriter(
                output,
                fieldnames=PACKET_COLUMNS,
                extrasaction="ignore",
            )

            if first_write:

                writer.writeheader()

            writer.writerows(
                buffer
            )

        first_write = False

        total_written += len(
            buffer
        )

        print(
            f"  written "
            f"{total_written:,} rows so far..."
        )

    # ========================================================
    # Streaming pipeline
    #
    # PCAP
    #   ↓
    # PcapReader
    #   ↓
    # parse_packet
    #   ↓
    # yield row
    #   ↓
    # buffer <= 200,000
    #   ↓
    # CSV
    # ========================================================

    for row in packet_stream:

        buffer.append(
            row
        )

        if len(
            buffer
        ) >= CHUNK_SIZE:

            flush(
                buffer
            )

            # Start a completely new list so the previous chunk
            # can be released from memory.

            buffer = []

    # Write final partial chunk.

    flush(
        buffer
    )

    # ========================================================
    # ==================== ADDED: EMPTY PCAP CSV =============
    #
    # If the PCAP produced zero rows, flush() was never able
    # to create the file.
    #
    # Create a valid CSV containing only the column names.
    # ========================================================

    if first_write:

        with out_path.open(
            mode="w",
            newline="",
            encoding="utf-8",
        ) as output:

            writer = csv.DictWriter(
                output,
                fieldnames=PACKET_COLUMNS,
            )

            writer.writeheader()

        print(
            "  [INFO] No packet rows were produced. "
            "Created an empty CSV with headers."
        )

    print()

    print(
        f"Written: {out_path}"
    )

    print(
        f"Total rows: {total_written:,}"
    )


if __name__ == "__main__":
    main()