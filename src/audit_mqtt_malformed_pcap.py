#!/usr/bin/env python3
"""Read-only MQTT-over-TCP audit for one PCAP. Creates no output files."""
import argparse
from collections import Counter
from datetime import datetime
from dataclasses import dataclass, field

try:
    from scapy.all import PcapReader, IP, TCP
    try:
        from scapy.layers.inet6 import IPv6
    except Exception:
        IPv6 = None
except Exception as exc:
    raise SystemExit(f"Cannot import Scapy: {exc}. Run with the project's .venv Python.")

NAMES = {
    0: "TYPE_0_INVALID", 1: "CONNECT", 2: "CONNACK", 3: "PUBLISH",
    4: "PUBACK", 5: "PUBREC", 6: "PUBREL", 7: "PUBCOMP",
    8: "SUBSCRIBE", 9: "SUBACK", 10: "UNSUBSCRIBE", 11: "UNSUBACK",
    12: "PINGREQ", 13: "PINGRESP", 14: "DISCONNECT", 15: "AUTH_OR_RESERVED"
}
FLAGS = {1: 0, 2: 0, 4: 0, 5: 0, 6: 2, 7: 0, 8: 2, 9: 0, 10: 2,
         11: 0, 12: 0, 13: 0, 14: 0, 15: 0}

def clock(ts):
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def project_timeout():
    try:
        import src.config as cfg
        for name in ("FLOW_IDLE_TIMEOUT", "FLOW_IDLE_TIMEOUT_SECONDS", "IDLE_TIMEOUT"):
            v = getattr(cfg, name, None)
            if isinstance(v, (int, float)) and v > 0:
                return float(v), "project config"
    except Exception:
        pass
    return 120.0, "fallback (not confirmed to match project config)"

def parse_messages(data):
    messages, errors, pos = [], [], 0
    while pos < len(data):
        start = pos
        if len(data) - pos < 2:
            break
        first = data[pos]
        typ, flags = first >> 4, first & 15
        value, multiplier, p = 0, 1, pos + 1
        encoded = []
        finished = False
        for _ in range(4):
            if p >= len(data):
                break
            d = data[p]; encoded.append(d); p += 1
            value += (d & 127) * multiplier
            if not d & 128:
                finished = True
                break
            multiplier *= 128
        if not finished:
            if len(encoded) == 4 and encoded[-1] & 128:
                errors.append((start, NAMES[typ], "remaining_length_exceeds_4_bytes", data[start:start+20].hex(" ")))
            break
        if len(encoded) > 1 and value < 128 ** (len(encoded)-1):
            errors.append((start, NAMES[typ], "non_minimal_remaining_length", data[start:start+20].hex(" ")))
            break
        end = p + value
        if end > len(data):
            break
        body = data[p:end]
        probs = []
        if typ == 0:
            probs.append("reserved_packet_type_0")
        if typ in FLAGS and flags != FLAGS[typ]:
            probs.append(f"invalid_fixed_header_flags_{flags:x}")
        if typ in (12, 13, 14) and len(body) != 0:
            probs.append(f"{NAMES[typ].lower()}_remaining_length_not_zero")
        if typ == 1:
            if len(body) < 10:
                probs.append("connect_body_too_short")
            elif len(body) >= 3:
                ln = int.from_bytes(body[:2], "big")
                if ln == 0 or 2 + ln + 1 > len(body):
                    probs.append("connect_protocol_name_length_invalid")
                elif body[2:2+ln] != b"MQTT":
                    probs.append("connect_protocol_name_not_MQTT")
        if typ == 3:
            if len(body) < 2:
                probs.append("publish_missing_topic_length")
            else:
                ln = int.from_bytes(body[:2], "big")
                if ln == 0:
                    probs.append("publish_empty_topic")
                elif 2 + ln > len(body):
                    probs.append("publish_topic_length_exceeds_packet")
                else:
                    topic = body[2:2+ln]
                    if b"+" in topic or b"#" in topic:
                        probs.append("publish_topic_contains_wildcard")
                    try:
                        topic.decode("utf-8")
                    except UnicodeDecodeError:
                        probs.append("publish_topic_invalid_utf8")
                qos = (flags >> 1) & 3
                if qos == 3:
                    probs.append("publish_reserved_qos_3")
                elif qos in (1, 2) and ln + 4 > len(body):
                    probs.append("publish_missing_packet_identifier")
        if typ in (4, 5, 6, 7) and len(body) < 2:
            probs.append(f"{NAMES[typ].lower()}_body_too_short")
        if typ in (8, 10) and len(body) < 5:
            probs.append(f"{NAMES[typ].lower()}_body_too_short")
        if typ == 9 and len(body) < 3:
            probs.append("suback_body_too_short")
        if typ == 11 and len(body) < 2:
            probs.append("unsuback_body_too_short")
        if typ == 2 and len(body) < 2:
            probs.append("connack_body_too_short")
        messages.append(NAMES[typ])
        for problem in probs:
            errors.append((start, NAMES[typ], problem, data[start:min(end, start+24)].hex(" ")))
        pos = end
    return messages, errors

def chunks_for(segments):
    segments = sorted((seq, payload) for seq, payload in segments if payload)
    chunks, buf, end_seq, gaps = [], bytearray(), None, 0
    for seq, payload in segments:
        finish = seq + len(payload)
        if end_seq is None:
            buf.extend(payload); end_seq = finish
        elif seq > end_seq:
            if buf: chunks.append(bytes(buf))
            gaps += 1
            buf = bytearray(payload); end_seq = finish
        elif finish > end_seq:
            buf.extend(payload[end_seq-seq:]); end_seq = finish
    if buf: chunks.append(bytes(buf))
    return chunks, gaps

@dataclass
class Sess:
    idx: int
    key: tuple
    start: float
    last: float
    end: float
    packets: int = 0
    payload_packets: int = 0
    payload_bytes: int = 0
    dirs: dict = field(default_factory=dict)
    closed: bool = False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap", required=True)
    ap.add_argument("--idle-timeout", type=float, default=None)
    ap.add_argument("--max-anomalies", type=int, default=25)
    args = ap.parse_args()
    auto_timeout, source = project_timeout()
    timeout = args.idle_timeout if args.idle_timeout is not None else auto_timeout
    print(f"PCAP: {args.pcap}")
    print(f"Session idle timeout: {timeout:g}s ({'manual override' if args.idle_timeout is not None else source})")
    print("Read-only: no input files modified; no report files written.\n")
    sessions, active = [], {}
    frames = tcp_total = port_packets = payload_pkts = 0
    with PcapReader(args.pcap) as reader:
        for pkt in reader:
            frames += 1
            if TCP not in pkt: continue
            tcp_total += 1
            tcp = pkt[TCP]
            sport, dport = int(tcp.sport), int(tcp.dport)
            if sport != 1883 and dport != 1883: continue
            port_packets += 1
            ip = pkt[IP] if IP in pkt else (pkt[IPv6] if IPv6 and IPv6 in pkt else None)
            if ip is None: continue
            src, dst, ts = str(ip.src), str(ip.dst), float(pkt.time)
            key = ("TCP", tuple(sorted(((src, sport), (dst, dport)))))
            sess = active.get(key)
            if sess is None or sess.closed or ts - sess.last > timeout:
                sess = Sess(len(sessions)+1, key, ts, ts, ts)
                sessions.append(sess); active[key] = sess
            sess.last = sess.end = ts
            sess.packets += 1
            direction = (src, sport, dst, dport)
            if direction not in sess.dirs: sess.dirs[direction] = []
            payload = bytes(tcp.payload)
            if payload:
                payload_pkts += 1
                sess.payload_packets += 1
                sess.payload_bytes += len(payload)
                seq = int(tcp.seq) + (1 if int(tcp.flags) & 2 else 0)
                sess.dirs[direction].append((seq, payload))
            if int(tcp.flags) & (1 | 4):
                sess.closed = True
                active.pop(key, None)

    print(f"PCAP frames: {frames:,}")
    print(f"TCP packets: {tcp_total:,}")
    print(f"TCP packets involving port 1883: {port_packets:,}")
    print(f"Port-1883 TCP payload packets: {payload_pkts:,}")
    print(f"Sessions grouped: {len(sessions):,}\n")

    total_types, total_errors = Counter(), Counter()
    results, examples, gap_total = [], [], 0
    for s in sessions:
        types, errors, complete, chunk_count = Counter(), [], 0, 0
        for direction, segs in s.dirs.items():
            chunks, gaps = chunks_for(segs)
            gap_total += gaps; chunk_count += len(chunks)
            for chunk in chunks:
                msgs, errs = parse_messages(chunk)
                types.update(msgs); complete += len(msgs)
                for off, typ, problem, raw in errs:
                    errors.append((off, typ, problem, raw, direction))
                    examples.append((s, off, typ, problem, raw, direction))
        total_types.update(types)
        total_errors.update(e[2] for e in errors)
        if types or errors:
            results.append((s, types, complete, chunk_count, errors))
    print("Complete MQTT messages parsed:")
    if total_types:
        for k,v in total_types.most_common(): print(f"  {k}: {v:,}")
    else: print("  None confidently parsed (could be fragmented, truncated, non-MQTT, or another port).")
    print("\nPotential protocol-structure anomalies:")
    if total_errors:
        for k,v in total_errors.most_common(): print(f"  {k}: {v:,}")
    else: print("  None detected by these basic checks (does not prove benign traffic).")
    print(f"TCP sequence gaps between captured payload chunks: {gap_total:,}\n")

    print("Sessions with parsed messages/anomalies (prioritize anomalies; up to 25):")
    results.sort(key=lambda x: (len(x[4]), x[2], x[0].packets), reverse=True)
    for s,types,complete,chunks,errors in results[:25]:
        ends = " <-> ".join(f"{ip}:{port}" for ip,port in s.key[1])
        type_text = ", ".join(f"{k}={v}" for k,v in types.most_common(6)) or "none"
        err_text = ", ".join(f"{k}={v}" for k,v in Counter(e[2] for e in errors).most_common(3)) or "none detected"
        print(f"  S{s.idx:04d} | {clock(s.start)}–{clock(s.end)} | packets={s.packets} | payload_packets={s.payload_packets} | bytes={s.payload_bytes} | messages={complete} | chunks={chunks}")
        print(f"    {ends}")
        print(f"    packet types: {type_text}")
        print(f"    anomalies: {err_text}")
    if examples:
        print(f"\nAnomaly examples (up to {max(0,args.max_anomalies)}; candidates only):")
        for s,off,typ,problem,raw,direction in examples[:max(0,args.max_anomalies)]:
            print(f"  S{s.idx:04d} | {clock(s.start)} | {direction[0]}:{direction[1]} -> {direction[2]}:{direction[3]} | offset={off} | {typ} | {problem} | hex={raw}")
    print("\nCaution: this is a heuristic protocol audit, not ground truth. It does not automatically label any flow Attack/Benign.")

if __name__ == "__main__":
    main()
