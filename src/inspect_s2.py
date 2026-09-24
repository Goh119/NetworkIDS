#后期可以删掉，这是查一下s2
"""
inspect_s2.py
-------------
Inspect S2_mac_ip_inconsistency packets in packet_integrity.csv.

The integrity CSV is ~1.5 GB and its text fields contain commas and
semicolons, so pandas' C engine may misparse certain rows. This script
uses the Python engine and reads in chunks.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import src.config as config


INTEGRITY_CSV = config.PROCESSED_DIR / "integrity" / "packet_integrity.csv"

CHUNK_SIZE = 500_000

USECOLS = [
    "packet_id",
    "src_ip",
    "src_mac",
    "dst_ip",
    "dst_mac",
    "S2_mac_ip_inconsistency",
    "integrity_state",
    "integrity_reasons",
]


def main():
    print("=" * 70)
    print("NetworkIDS - S2 MAC-IP Inconsistency Inspection")
    print("=" * 70)
    print(f"Input: {INTEGRITY_CSV}")
    print()

    if not INTEGRITY_CSV.exists():
        print(f"[ERROR] Not found: {INTEGRITY_CSV}")
        return

    chunks = []
    total = 0

    for i, chunk in enumerate(
        pd.read_csv(
            INTEGRITY_CSV,
            usecols=USECOLS,
            chunksize=CHUNK_SIZE,
            engine="python",
        ),
        start=1,
    ):
        total += len(chunk)
        flagged = chunk[chunk["S2_mac_ip_inconsistency"] == 1]
        if len(flagged) > 0:
            chunks.append(flagged)
        print(f"  chunk {i:>3} | processed {total:>12,} | "
              f"flagged so far: {sum(len(c) for c in chunks):,}")

    print()
    print("=" * 70)
    print("Result")
    print("=" * 70)

    if not chunks:
        print("No S2-flagged packets found.")
        return

    s2 = pd.concat(chunks, ignore_index=True)

    print(f"Total S2-flagged packets : {len(s2):,}")
    print()

    print("By source IP:")
    print(s2["src_ip"].value_counts().to_string())
    print()

    print("By source MAC:")
    print(s2["src_mac"].value_counts().to_string())
    print()

    print("All flagged packets:")
    show_cols = [
        "packet_id", "src_ip", "src_mac",
        "dst_ip", "dst_mac",
        "integrity_state", "integrity_reasons",
    ]
    show_cols = [c for c in show_cols if c in s2.columns]
    print(s2[show_cols].head(20).to_string(index=False))

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()