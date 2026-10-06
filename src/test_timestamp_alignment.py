from pathlib import Path
from datetime import timedelta
import random

import pandas as pd  # type: ignore[reportMissingModuleSource]
from scapy.all import RawPcapReader  # type: ignore[reportMissingImports]
from scapy.layers.inet import IP, TCP, UDP  # type: ignore[reportMissingImports]
from scapy.layers.inet6 import IPv6  # type: ignore[reportMissingImports]


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

PCAP_FILES = {
    "Monday": RAW_DIR / "Monday-WorkingHours.pcap",
    "Tuesday": RAW_DIR / "Tuesday-WorkingHours.pcap",
    "Wednesday": RAW_DIR / "Wednesday-workingHours.pcap",
    "Thursday": RAW_DIR / "Thursday-WorkingHours.pcap",
    "Friday": RAW_DIR / "Friday-WorkingHours.pcap",
}

LABEL_FILES = {
    "Monday": [
        RAW_DIR / "GeneratedLabelledFlows" / "TrafficLabelling"
        / "Monday-WorkingHours.pcap_ISCX.csv"
    ],
    "Tuesday": [
        RAW_DIR / "GeneratedLabelledFlows" / "TrafficLabelling"
        / "Tuesday-WorkingHours.pcap_ISCX.csv"
    ],
    "Wednesday": [
        RAW_DIR / "GeneratedLabelledFlows" / "TrafficLabelling"
        / "Wednesday-workingHours.pcap_ISCX.csv"
    ],
    "Thursday": [
        RAW_DIR / "GeneratedLabelledFlows" / "TrafficLabelling"
        / "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        RAW_DIR / "GeneratedLabelledFlows" / "TrafficLabelling"
        / "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    ],
    "Friday": [
        RAW_DIR / "GeneratedLabelledFlows" / "TrafficLabelling"
        / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        RAW_DIR / "GeneratedLabelledFlows" / "TrafficLabelling"
        / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        RAW_DIR / "GeneratedLabelledFlows" / "TrafficLabelling"
        / "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    ],
}

PCAP_LOCAL_OFFSET_HOURS = -3

# Number of afternoon label rows to sample per day.
# Increase if needed, but 2000 is already enough for this test.
SAMPLE_PER_DAY = 2000

# Matching tolerance around flow start timestamp.
TIME_TOLERANCE_SECONDS = 60.0

RANDOM_SEED = 42


# ============================================================
# HELPERS
# ============================================================

def canonical_flow_key(src_ip, src_port, dst_ip, dst_port, protocol):
    """
    Direction-independent 5-tuple.
    """
    a = (str(src_ip), int(src_port), int(protocol))
    b = (str(dst_ip), int(dst_port), int(protocol))

    if a <= b:
        return (
            str(src_ip),
            int(src_port),
            str(dst_ip),
            int(dst_port),
            int(protocol),
        )

    return (
        str(dst_ip),
        int(dst_port),
        str(src_ip),
        int(src_port),
        int(protocol),
    )


def get_packet_info(packet):
    """
    Return:
        timestamp_epoch,
        canonical_flow_key
    """

    metadata = packet[1]

    raw_timestamp = (
        (int(metadata.tshigh) << 32)
        | int(metadata.tslow)
    )

    tsresol = int(metadata.tsresol)
    timestamp_epoch = raw_timestamp / tsresol

    raw_packet = packet[0]

    try:
        from scapy.all import Ether  # type: ignore[reportMissingImports]

        p = Ether(raw_packet)

        if IP in p:
            ip = p[IP]
            src_ip = ip.src
            dst_ip = ip.dst
            protocol = int(ip.proto)

        elif IPv6 in p:
            ip = p[IPv6]
            src_ip = ip.src
            dst_ip = ip.dst
            protocol = int(ip.nh)

        else:
            return timestamp_epoch, None

        if protocol == 6 and TCP in p:
            l4 = p[TCP]
            src_port = int(l4.sport)
            dst_port = int(l4.dport)

        elif protocol == 17 and UDP in p:
            l4 = p[UDP]
            src_port = int(l4.sport)
            dst_port = int(l4.dport)

        else:
            return timestamp_epoch, None

        key = canonical_flow_key(
            src_ip,
            src_port,
            dst_ip,
            dst_port,
            protocol,
        )

        return timestamp_epoch, key

    except Exception:
        return timestamp_epoch, None


def epoch_to_local_datetime(epoch):
    """
    Convert PCAP UTC epoch to UTC-3 local time.
    """
    return pd.to_datetime(
        epoch + PCAP_LOCAL_OFFSET_HOURS * 3600,
        unit="s",
    )


def load_label_rows(day):
    """
    Load label CSVs and return only rows whose displayed CSV hour
    is 01:00-05:59.

    These are the rows whose interpretation we want to test.
    """

    frames = []

    for file in LABEL_FILES[day]:
        print(f"  Loading: {file.name}")

        df = pd.read_csv(
            file,
            encoding="latin1",
            low_memory=False,
        )

        df.columns = df.columns.str.strip()

        required = [
            "Timestamp",
            "Source IP",
            "Source Port",
            "Destination IP",
            "Destination Port",
            "Protocol",
            "Label",
        ]

        missing = [c for c in required if c not in df.columns]

        if missing:
            print(f"    Missing columns: {missing}")
            continue

        df = df[required].copy()

        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"],
            dayfirst=True,
            errors="coerce",
        )

        df["Source Port"] = pd.to_numeric(
            df["Source Port"],
            errors="coerce",
        )

        df["Destination Port"] = pd.to_numeric(
            df["Destination Port"],
            errors="coerce",
        )

        df["Protocol"] = pd.to_numeric(
            df["Protocol"],
            errors="coerce",
        )

        df = df.dropna(
            subset=[
                "Timestamp",
                "Source IP",
                "Source Port",
                "Destination IP",
                "Destination Port",
                "Protocol",
            ]
        )

        # ONLY displayed CSV hour 1-5
        df = df[
            df["Timestamp"].dt.hour.between(1, 5)
        ].copy()

        if len(df) == 0:
            continue

        df["flow_key"] = [
            canonical_flow_key(
                row["Source IP"],
                row["Source Port"],
                row["Destination IP"],
                row["Destination Port"],
                row["Protocol"],
            )
            for _, row in df.iterrows()
        ]

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    return result


def prepare_samples(df, day):
    """
    Randomly sample afternoon-label candidates.

    Two interpretations:

    A. AS-IS:
       01:00-05:00 stays 01:00-05:00

    B. +12 HOURS:
       01:00-05:00 becomes 13:00-17:00
    """

    if len(df) > SAMPLE_PER_DAY:
        df = df.sample(
            n=SAMPLE_PER_DAY,
            random_state=RANDOM_SEED,
        ).copy()
    else:
        df = df.copy()

    # CSV timestamp interpreted as local UTC-3
    #
    # Convert the naive CSV time to an epoch that corresponds
    # to UTC-3 local time.
    #
    # local time + 3h = UTC epoch
    df["target_as_is_epoch"] = (
        pd.to_datetime(df["Timestamp"])
        - pd.Timestamp("1970-01-01")
    ).dt.total_seconds() + 3 * 3600

    # +12 hour interpretation
    df["timestamp_plus12"] = (
        df["Timestamp"] + pd.Timedelta(hours=12)
    )

    df["target_plus12_epoch"] = (
        df["timestamp_plus12"]
        - pd.Timestamp("1970-01-01")
    ).dt.total_seconds() + 3 * 3600

    return df


# ============================================================
# PCAP SCAN
# ============================================================

def scan_pcap(day, samples):
    """
    Scan the PCAP once.

    We only care about the flow keys and timestamps contained
    in the selected afternoon label rows.
    """

    pcap_file = PCAP_FILES[day]

    # All candidate keys
    target_keys = set(samples["flow_key"])

    # For each key we store packet timestamps.
    observed = {}

    first_epoch = None
    last_epoch = None

    # Maximum target under +12h interpretation.
    max_target = samples["target_plus12_epoch"].max()

    # Minimum target under +12h interpretation.
    min_target = samples["target_plus12_epoch"].min()

    # We need a little tolerance around the target range.
    scan_start = min_target - TIME_TOLERANCE_SECONDS
    scan_end = max_target + TIME_TOLERANCE_SECONDS

    print()
    print(f"  PCAP: {pcap_file.name}")
    print(
        "  Target afternoon local range:",
        epoch_to_local_datetime(min_target),
        "to",
        epoch_to_local_datetime(max_target),
    )

    print(
        "  Scanning PCAP until:",
        epoch_to_local_datetime(scan_end),
    )

    packet_count = 0
    matched_packet_count = 0

    with RawPcapReader(str(pcap_file)) as reader:

        for packet in reader:

            packet_count += 1

            epoch, flow_key = get_packet_info(packet)

            if first_epoch is None:
                first_epoch = epoch

            last_epoch = epoch

            # Before afternoon target range
            if epoch < scan_start:
                continue

            # After all relevant afternoon targets
            if epoch > scan_end:
                break

            if flow_key not in target_keys:
                continue

            matched_packet_count += 1

            if flow_key not in observed:
                observed[flow_key] = []

            observed[flow_key].append(epoch)

    print(f"  Packets scanned: {packet_count:,}")
    print(f"  Matching packets found: {matched_packet_count:,}")

    if first_epoch is not None:
        print(
            "  PCAP first timestamp:",
            epoch_to_local_datetime(first_epoch),
        )

    if last_epoch is not None:
        print(
            "  Last scanned timestamp:",
            epoch_to_local_datetime(last_epoch),
        )

    return observed


# ============================================================
# MATCHING
# ============================================================

def evaluate_strategy(samples, observed, target_column):
    """
    Determine whether each sampled label flow has at least one
    packet in the PCAP within ±60 seconds of the target timestamp.
    """

    matched = 0
    unmatched = 0
    time_diffs = []

    for _, row in samples.iterrows():

        key = row["flow_key"]

        target = float(row[target_column])

        packet_times = observed.get(key, [])

        if not packet_times:
            unmatched += 1
            continue

        diffs = [
            abs(float(packet_time) - target)
            for packet_time in packet_times
        ]

        best_diff = min(diffs)

        if best_diff <= TIME_TOLERANCE_SECONDS:
            matched += 1
            time_diffs.append(best_diff)
        else:
            unmatched += 1

    if time_diffs:
        mean_diff = sum(time_diffs) / len(time_diffs)
        median_diff = sorted(time_diffs)[
            len(time_diffs) // 2
        ]
    else:
        mean_diff = None
        median_diff = None

    return {
        "matched": matched,
        "unmatched": unmatched,
        "total": len(samples),
        "match_rate": (
            matched / len(samples)
            if len(samples) > 0
            else 0
        ),
        "mean_diff": mean_diff,
        "median_diff": median_diff,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("CIC-IDS2017 AFTERNOON TIMESTAMP ALIGNMENT TEST")
    print("=" * 80)

    print()
    print("Testing ONLY CSV rows with displayed hours 01:00-05:59.")
    print()
    print("Strategy A: CSV timestamp AS-IS")
    print("            01:00-05:59 -> 01:00-05:59")
    print()
    print("Strategy B: CSV timestamp +12 hours")
    print("            01:00-05:59 -> 13:00-17:59")
    print()
    print("PCAP timezone interpretation: UTC-3")
    print(f"Tolerance: ±{TIME_TOLERANCE_SECONDS:.0f} seconds")
    print()

    all_results = []

    for day in PCAP_FILES:

        print()
        print("=" * 80)
        print(day.upper())
        print("=" * 80)

        labels = load_label_rows(day)

        print(
            f"  Afternoon label rows available: {len(labels):,}"
        )

        if len(labels) == 0:
            print("  No 01:00-05:59 rows found.")
            continue

        samples = prepare_samples(
            labels,
            day,
        )

        print(
            f"  Sampled rows: {len(samples):,}"
        )

        # Show sample time range
        print(
            "  CSV displayed range:",
            samples["Timestamp"].min(),
            "to",
            samples["Timestamp"].max(),
        )

        print(
            "  +12h interpreted range:",
            samples["timestamp_plus12"].min(),
            "to",
            samples["timestamp_plus12"].max(),
        )

        # Scan PCAP
        observed = scan_pcap(
            day,
            samples,
        )

        # Strategy A
        result_as_is = evaluate_strategy(
            samples,
            observed,
            "target_as_is_epoch",
        )

        # Strategy B
        result_plus12 = evaluate_strategy(
            samples,
            observed,
            "target_plus12_epoch",
        )

        print()
        print("  RESULTS")
        print("  " + "-" * 60)

        print(
            "  AS-IS:"
        )
        print(
            f"    matched     = {result_as_is['matched']:,}"
        )
        print(
            f"    unmatched   = {result_as_is['unmatched']:,}"
        )
        print(
            f"    match rate  = {result_as_is['match_rate']:.4%}"
        )

        if result_as_is["mean_diff"] is not None:
            print(
                f"    mean diff   = {result_as_is['mean_diff']:.3f}s"
            )
            print(
                f"    median diff = {result_as_is['median_diff']:.3f}s"
            )

        print()
        print(
            "  +12 HOURS:"
        )
        print(
            f"    matched     = {result_plus12['matched']:,}"
        )
        print(
            f"    unmatched   = {result_plus12['unmatched']:,}"
        )
        print(
            f"    match rate  = {result_plus12['match_rate']:.4%}"
        )

        if result_plus12["mean_diff"] is not None:
            print(
                f"    mean diff   = {result_plus12['mean_diff']:.3f}s"
            )
            print(
                f"    median diff = {result_plus12['median_diff']:.3f}s"
            )

        all_results.append({
            "day": day,
            "sampled": len(samples),
            "as_is_rate": result_as_is["match_rate"],
            "plus12_rate": result_plus12["match_rate"],
            "as_is_matched": result_as_is["matched"],
            "plus12_matched": result_plus12["matched"],
            "as_is_mean_diff": result_as_is["mean_diff"],
            "plus12_mean_diff": result_plus12["mean_diff"],
        })

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    if not all_results:
        print("No results.")
        return

    summary = pd.DataFrame(all_results)

    print()
    print(
        summary[
            [
                "day",
                "sampled",
                "as_is_matched",
                "as_is_rate",
                "plus12_matched",
                "plus12_rate",
            ]
        ].to_string(index=False)
    )

    total_sampled = summary["sampled"].sum()
    total_as_is = summary["as_is_matched"].sum()
    total_plus12 = summary["plus12_matched"].sum()

    print()
    print(
        f"Total sampled: {total_sampled:,}"
    )

    print(
        f"AS-IS total:   {total_as_is:,} "
        f"({total_as_is / total_sampled:.4%})"
    )

    print(
        f"+12h total:    {total_plus12:,} "
        f"({total_plus12 / total_sampled:.4%})"
    )

    print()
    print("=" * 80)

    if total_plus12 > total_as_is:
        print(
            "RESULT: +12 HOURS gives stronger afternoon alignment."
        )
    elif total_as_is > total_plus12:
        print(
            "RESULT: AS-IS gives stronger afternoon alignment."
        )
    else:
        print(
            "RESULT: Both strategies produced the same match count."
        )

    print("=" * 80)


if __name__ == "__main__":
    main()