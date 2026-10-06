from pathlib import Path
import datetime
import sys

import pandas as pd
from scapy.utils import RawPcapReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
LABEL_DIR = (
    RAW_DIR
    / "GeneratedLabelledFlows"
    / "TrafficLabelling"
)


DATASETS = {
    "Monday": {
        "pcap": RAW_DIR / "Monday-WorkingHours.pcap",
        "labels": [
            LABEL_DIR / "Monday-WorkingHours.pcap_ISCX.csv",
        ],
    },

    "Tuesday": {
        "pcap": RAW_DIR / "Tuesday-WorkingHours.pcap",
        "labels": [
            LABEL_DIR / "Tuesday-WorkingHours.pcap_ISCX.csv",
        ],
    },

    "Wednesday": {
        "pcap": RAW_DIR / "Wednesday-workingHours.pcap",
        "labels": [
            LABEL_DIR / "Wednesday-workingHours.pcap_ISCX.csv",
        ],
    },

    "Thursday": {
        "pcap": RAW_DIR / "Thursday-WorkingHours.pcap",
        "labels": [
            LABEL_DIR
            / "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",

            LABEL_DIR
            / "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        ],
    },

    "Friday": {
        "pcap": RAW_DIR / "Friday-WorkingHours.pcap",
        "labels": [
            LABEL_DIR
            / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",

            LABEL_DIR
            / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",

            LABEL_DIR
            / "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        ],
    },
}

def get_first_pcap_timestamp(pcap_path):
    """
    Read only the first packet timestamp.

    Supports Scapy PacketMetadataNg used by the
    CIC-IDS2017 pcapng files.

    For the observed CIC-IDS2017 files:
        raw_timestamp = (tshigh << 32) | tslow

    Scapy reports tsresol as the number of timestamp
    ticks per second, e.g. 1,000,000 for microseconds.
    Therefore:

        epoch_seconds = raw_timestamp / tsresol
    """

    with RawPcapReader(str(pcap_path)) as reader:
        pkt_data, pkt_metadata = next(reader)

    if type(pkt_metadata).__name__ == "PacketMetadataNg":

        raw_timestamp = (
            (int(pkt_metadata.tshigh) << 32)
            | int(pkt_metadata.tslow)
        )

        tsresol = int(pkt_metadata.tsresol)

        if tsresol <= 0:
            raise ValueError(
                f"Invalid pcapng timestamp resolution: {tsresol}"
            )

        return float(raw_timestamp / tsresol)

    # Fallback for classic pcap metadata
    if hasattr(pkt_metadata, "sec") and hasattr(
        pkt_metadata, "usec"
    ):
        return (
            float(pkt_metadata.sec)
            + float(pkt_metadata.usec) / 1_000_000
        )

    raise AttributeError(
        f"Unsupported PCAP metadata type: "
        f"{type(pkt_metadata).__name__}. "
        f"Available attributes: {dir(pkt_metadata)}"
    )


def print_epoch_candidates(epoch):
    utc = datetime.datetime.utcfromtimestamp(epoch)

    print(f"  UTC     : {utc}")

    for offset in [-4, -3, -2, 0]:
        local = utc + datetime.timedelta(hours=offset)

        print(
            f"  UTC{offset:+d}    : {local}"
        )


def inspect_label_file(path):
    df = pd.read_csv(
        path,
        encoding="latin1"
    )

    df.columns = df.columns.str.strip()

    if "Timestamp" not in df.columns:
        raise ValueError(
            f"Timestamp column not found: {path}"
        )

    timestamp = pd.to_datetime(
        df["Timestamp"],
        dayfirst=True,
        errors="coerce"
    )

    label_column = "Label"

    if label_column in df.columns:
        labels = (
            df[label_column]
            .astype(str)
            .str.strip()
            .value_counts()
        )
    else:
        labels = pd.Series(dtype="int64")

    print(
        f"\n    {path.name}"
    )

    print(
        f"      Rows          : {len(df):,}"
    )

    print(
        f"      First parsed  : {timestamp.min()}"
    )

    print(
        f"      Last parsed   : {timestamp.max()}"
    )

    print(
        "      Hour distribution:"
    )

    print(
        timestamp.dt.hour
        .value_counts()
        .sort_index()
        .to_string()
    )

    if not labels.empty:
        print(
            "      Label distribution:"
        )

        print(
            labels.to_string()
        )

    return {
        "rows": len(df),
        "first": timestamp.min(),
        "last": timestamp.max(),
        "hours": timestamp.dt.hour.value_counts(),
    }


def main():

    print("=" * 80)
    print("CIC-IDS2017 FIVE-DAY INPUT VERIFICATION")
    print("=" * 80)

    all_ok = True

    for day, info in DATASETS.items():

        print("\n" + "=" * 80)
        print(f"{day.upper()}")
        print("=" * 80)

        pcap = info["pcap"]
        labels = info["labels"]

        # --------------------------------------------------
        # PCAP
        # --------------------------------------------------

        print("\nPCAP:")

        if not pcap.exists():
            print(f"  [MISSING] {pcap}")
            all_ok = False
            continue

        size_gb = pcap.stat().st_size / (1024 ** 3)

        print(
            f"  File  : {pcap.name}"
        )

        print(
            f"  Size  : {size_gb:.2f} GB"
        )

        print(
            "  Reading first packet timestamp..."
        )

        first_epoch = get_first_pcap_timestamp(pcap)

        print(
            f"\n  First PCAP epoch: {first_epoch}"
        )

        print_epoch_candidates(
            first_epoch
        )

        # --------------------------------------------------
        # LABELS
        # --------------------------------------------------

        print("\nOfficial label CSVs:")

        for label_path in labels:

            if not label_path.exists():

                print(
                    f"  [MISSING] {label_path}"
                )

                all_ok = False
                continue

            try:

                inspect_label_file(
                    label_path
                )

            except Exception as exc:

                print(
                    f"  [ERROR] {exc}"
                )

                all_ok = False

    print("\n" + "=" * 80)

    if all_ok:
        print(
            "PASS: All configured PCAP and label files exist."
        )
    else:
        print(
            "FAIL: One or more inputs are missing or invalid."
        )


if __name__ == "__main__":
    main()