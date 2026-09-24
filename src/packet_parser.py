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
import sys
from pathlib import Path

# Silence the "No libpcap provider" warning: we only read files, never sniff.
import logging
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

from scapy.all import PcapReader, Ether, Dot1Q, IP, IPv6, TCP, UDP, ICMP, raw
from scapy.layers.inet import in4_chksum

sys.path.append(str(Path(__file__).resolve().parents[1]))
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
        # Scapy recomputes it from scratch. 这里可能后续会有问题，没问题就别动了
        header = ip_layer.copy()
        header.chksum = None
        del header.payload

        recomputed = IP(raw(header)).chksum

        return bool(stored == recomputed)

    except Exception:
        return None


def l4_checksum_valid(pkt):
    """
    Recompute the TCP/UDP checksum.

    Returns True / False, or None when the checksum is absent or the
    protocol is not TCP/UDP.

    NOTE: checksum offloading on the capturing host can produce
    false "invalid" results for locally generated traffic. For an
    imported capture such as CIC-IDS2017 this is usually not an issue,
    but it is worth remembering when testing with live traffic later.
    """
    try:
        if TCP in pkt:
            layer = pkt[TCP]
        elif UDP in pkt:
            layer = pkt[UDP]
        else:
            return None

        stored = layer.chksum

        if stored is None:
            return None

        # UDP checksum 0 means "no checksum computed" (legal in IPv4).
        if UDP in pkt and stored == 0:
            return None

        if IP not in pkt:
            return None

        probe = layer.copy()
        probe.chksum = 0

        recomputed = in4_chksum(pkt[IP].proto, pkt[IP], raw(probe))

        return bool(stored == recomputed)

    except Exception:
        return None


# ============================================================
# Per-packet extraction
def parse_packet(pkt, packet_id):
    """
    Extract L2-L4 header fields from a single Scapy packet.

    Every field that does not apply to this packet is left as None.
    """

    row = {name: None for name in PACKET_COLUMNS}

    row["packet_id"] = packet_id
    row["timestamp"] = float(pkt.time)
    row["frame_length"] = len(pkt)

    # ---------------- L2 ----------------
    if Ether in pkt:
        eth = pkt[Ether]
        row["src_mac"] = eth.src
        row["dst_mac"] = eth.dst
        row["eth_type"] = int(eth.type)

    if Dot1Q in pkt:
        vlan = pkt[Dot1Q]
        row["vlan_id"] = int(vlan.vlan)
        row["vlan_priority"] = int(vlan.prio)
        # With a VLAN tag the real upper-layer type sits in the Dot1Q header.
        row["eth_type"] = int(vlan.type)

    # ---------------- L3 ----------------
    if IP in pkt:
        ip = pkt[IP]

        row["ip_version"] = 4
        row["src_ip"] = ip.src
        row["dst_ip"] = ip.dst

        tos = int(ip.tos)
        row["ip_dscp"] = tos >> 2
        row["ip_ecn"] = tos & 0b11

        row["ip_total_length"] = int(ip.len)
        row["ip_id"] = int(ip.id)

        flags = int(ip.flags)
        row["ip_df"] = int(bool(flags & 0b010))
        row["ip_mf"] = int(bool(flags & 0b001))
        row["fragment_offset"] = int(ip.frag)

        row["ttl"] = int(ip.ttl)
        row["ip_protocol"] = int(ip.proto)
        row["ip_header_length"] = int(ip.ihl) * 4

        row["ip_checksum"] = int(ip.chksum) if ip.chksum is not None else None

        valid = ipv4_checksum_valid(ip)
        row["ip_checksum_valid"] = None if valid is None else int(valid)

    elif IPv6 in pkt:
        ip6 = pkt[IPv6]

        row["ip_version"] = 6
        row["src_ip"] = ip6.src
        row["dst_ip"] = ip6.dst

        tc = int(ip6.tc)
        row["ip_dscp"] = tc >> 2
        row["ip_ecn"] = tc & 0b11

        #row["ip_total_length"] = int(ip6.plen)
        row["ip_total_length"] = 40 + int(ip6.plen)
        row["ttl"] = int(ip6.hlim)          # hop limit == TTL equivalent
        row["ip_protocol"] = int(ip6.nh)
        row["ip_header_length"] = 40        # fixed IPv6 base header

        # IPv6 has no header checksum, no ID field, no DF/MF flags.
        # Those stay None on purpose.

    # ---------------- L4 ----------------
    if TCP in pkt:
        tcp = pkt[TCP]

        row["l4_protocol"] = "TCP"
        row["src_port"] = int(tcp.sport)
        row["dst_port"] = int(tcp.dport)
        row["tcp_window"] = int(tcp.window)
        row["tcp_header_length"] = int(tcp.dataofs) * 4
        row["tcp_checksum_raw"] = None  # placeholder, real value set below 这里可能之后删掉

        row["l4_checksum"] = int(tcp.chksum) if tcp.chksum is not None else None
        row.pop("tcp_checksum_raw", None) # 这里可能之后删掉

        flags = int(tcp.flags)
        row["tcp_fin"] = int(bool(flags & 0x01))
        row["tcp_syn"] = int(bool(flags & 0x02))
        row["tcp_rst"] = int(bool(flags & 0x04))
        row["tcp_psh"] = int(bool(flags & 0x08))
        row["tcp_ack"] = int(bool(flags & 0x10))
        row["tcp_urg"] = int(bool(flags & 0x20))

    elif UDP in pkt:
        udp = pkt[UDP]

        row["l4_protocol"] = "UDP"
        row["src_port"] = int(udp.sport)
        row["dst_port"] = int(udp.dport)
        row["l4_checksum"] = int(udp.chksum) if udp.chksum is not None else None
        # TCP-only fields stay None.

    elif ICMP in pkt:
        row["l4_protocol"] = "ICMP"
        # ICMP has no ports. src_port / dst_port stay None.

    elif IPv6 in pkt and int(pkt[IPv6].nh) == 58:
        row["l4_protocol"] = "ICMPv6"
        # ICMPv6 has no ports either.

    elif row["ip_version"] is not None:
        row["l4_protocol"] = "OTHER"

    valid = l4_checksum_valid(pkt)
    row["l4_checksum_valid"] = None if valid is None else int(valid)

    return row

# ============================================================
# PCAP reading
def iter_packets(pcap_path, limit=None, progress_every=100_000):
    """
    Stream a PCAP file packet by packet.

    PcapReader is used instead of rdpcap because the CIC-IDS2017 captures
    are far too large to load into memory at once.
    """
    pcap_path = Path(pcap_path)

    if not pcap_path.exists():
        raise FileNotFoundError(f"PCAP not found: {pcap_path}")

    rows = []
    packet_id = 0
    skipped = 0

    with PcapReader(str(pcap_path)) as reader:

        for pkt in reader:

            packet_id += 1

            try:
                rows.append(parse_packet(pkt, packet_id))
            except Exception as exc:
                skipped += 1
                if skipped <= 5:
                    print(f"  [warn] packet {packet_id} skipped: {exc}")

            if progress_every and packet_id % progress_every == 0:
                print(f"  parsed {packet_id:,} packets...")

            if limit is not None and packet_id >= limit:
                break

    if skipped:
        print(f"  [warn] {skipped} packet(s) could not be parsed.")

    return rows


# ============================================================
# Preview helper
def preview(rows, count):
    """Print a small, readable sample so the extraction can be eyeballed."""

    show = [
        "packet_id", "timestamp", "frame_length",
        "src_mac", "dst_mac", "eth_type",
        "src_ip", "dst_ip", "ttl", "ip_protocol",
        "ip_checksum_valid",
        "l4_protocol", "src_port", "dst_port",
        "tcp_window", "tcp_syn", "tcp_ack", "tcp_fin", "tcp_rst",
        "l4_checksum_valid",
    ]

    for row in rows[:count]:
        print("-" * 60)
        for key in show:
            value = row.get(key)
            value = "NaN" if value is None else value
            print(f"  {key:<20} {value}")


# ============================================================
# Main
def main():

    parser = argparse.ArgumentParser(
        description="Extract L2-L4 header features from a PCAP file."
    )

    parser.add_argument(
        "--pcap",
        default=str(config.PCAP_FILE),
        help="Path to the PCAP file.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10_000,
        help="Stop after this many packets. Use 0 for the whole file.",
    )

    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Print this many parsed packets instead of writing a CSV.",
    )

    parser.add_argument(
        "--out",
        default=None,
        help="Where to write the packet-level CSV.",
    )

    args = parser.parse_args()

    limit = None if args.limit == 0 else args.limit

    print("=" * 60)
    print("NetworkIDS - Packet Parser")
    print("=" * 60)
    print(f"PCAP  : {args.pcap}")
    #print(f"Limit : {'whole file' if limit is None else f'{limit:,} packets'}")
    if args.preview:
        print(f"Mode  : preview ({args.preview} packets)")
    else:
        print(f"Limit : {'whole file' if limit is None else f'{limit:,} packets'}")
    print()

    packet_stream = iter_packets(args.pcap, limit=limit)
 
    if args.preview:
        import itertools
        rows = list(itertools.islice(packet_stream, args.preview))
        print()
        print(f"Parsed {len(rows):,} packets.")
        print()
        preview(rows, args.preview)
        print("-" * 60)
        return

    import pandas as pd

    out_path = Path(args.out) if args.out else config.PACKET_FEATURES_FILE
    out_path.parent.mkdir(parents=True, exist_ok=True)
 
    # Flush to disk every CHUNK_SIZE packets instead of holding the whole
    # capture in memory. For a full ~10M-packet day this keeps peak RAM
    # usage roughly constant instead of growing for the entire run.
    CHUNK_SIZE = 200_000
 
    buffer = []
    total_written = 0
    first_write = True
 
    def flush(buffer):
        nonlocal total_written, first_write
        if not buffer:
            return
        chunk_df = pd.DataFrame(buffer, columns=PACKET_COLUMNS)
        chunk_df.to_csv(
            out_path,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
        )
        first_write = False
        total_written += len(buffer)
        print(f"  written {total_written:,} rows so far...")
 
    for row in packet_stream:
        buffer.append(row)
        if len(buffer) >= CHUNK_SIZE:
            flush(buffer)
            buffer = []
 
    flush(buffer)  # final partial chunk
 
    print()
    print(f"Written: {out_path}")
    print(f"Total rows: {total_written:,}")

if __name__ == "__main__":
    main()