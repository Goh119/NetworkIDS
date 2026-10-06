"""
merge_label_csvs.py
-------------------
Merge the multiple CIC-IDS2017 label CSVs per day into a single
CSV per day, so the existing single-day label_matcher.py can be
reused without modification.

Thursday : Morning-WebAttacks + Afternoon-Infilteration
Friday   : Morning + Afternoon-DDos + Afternoon-PortScan
Monday   : already single file
Tuesday  : already single file
Wednesday: already single file

Encoding: CIC-IDS2017 CSVs are Latin-1, not UTF-8.

Usage:
    python src/merge_label_csvs.py
"""

from pathlib import Path
import sys

import pandas as pd # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
LABEL_DIR = RAW_DIR / "GeneratedLabelledFlows" / "TrafficLabelling"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "multiday_labels"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


DAY_SOURCES = {
    "Monday": [
        "Monday-WorkingHours.pcap_ISCX.csv",
    ],
    "Tuesday": [
        "Tuesday-WorkingHours.pcap_ISCX.csv",
    ],
    "Wednesday": [
        "Wednesday-workingHours.pcap_ISCX.csv",
    ],
    "Thursday": [
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    ],
    "Friday": [
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    ],
}


def merge_day(day, filenames):
    print(f"\n[{day}]")
    frames = []

    for fn in filenames:
        path = LABEL_DIR / fn
        if not path.exists():
            print(f"  [ERROR] Missing: {path}")
            continue

        try:
            df = pd.read_csv(path, encoding="latin1", low_memory=False)
        except Exception as exc:
            print(f"  [ERROR] {fn}: {exc}")
            continue

        df.columns = [c.strip() for c in df.columns]
        df["_source_file"] = fn
        frames.append(df)
        print(f"  Loaded: {fn}  ({len(df):,} rows)")

    if not frames:
        print(f"  [SKIP] no data for {day}")
        return

    merged = pd.concat(frames, ignore_index=True)
    out = OUTPUT_DIR / f"{day}_combined.csv"
    merged.to_csv(out, index=False)
    print(f"  Written: {out}")
    print(f"  Total rows: {len(merged):,}")

    if "Label" in merged.columns:
        print(f"  Label distribution:")
        print(merged["Label"].value_counts().to_string())


def main():
    print("=" * 70)
    print("Merge CIC-IDS2017 label CSVs per day")
    print("=" * 70)

    for day, files in DAY_SOURCES.items():
        merge_day(day, files)

    print()
    print("=" * 70)
    print("Done. Merged files are in:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()