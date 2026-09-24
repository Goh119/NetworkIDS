"""
NetworkIDS - Attach ground-truth labels from the official CIC-IDS2017
GeneratedLabelledFlows CSV onto our own flow_features.csv.
 
Why this isn't a simple merge on the 5-tuple:
    - About 30% of 5-tuples repeat within a single day's CSV (port reuse),
      so 5-tuple alone is not unique. Timestamp (minute-resolution) is
      needed to disambiguate.
    - Timezone mismatch: our packet timestamps are raw Unix epoch (UTC).
      The official CSV's "Timestamp" column is in Atlantic Daylight Time
      (UTC-3) -- the local time in New Brunswick, Canada, where the
      dataset was captured. This has been verified empirically: epoch
      1499254962.08 (UTC) == "5/7/2017 8:42" in the CSV after a -3h shift.
    - Date format: the CSV's "5/7/2017" is DAY/MONTH/YEAR (5 July 2017),
      not the US month/day convention. pandas needs dayfirst=True or it
      will silently misparse this as May 7th.
    - Direction mismatch: which side the CSV calls "Source" for a given
      flow record does not necessarily match which side our own code
      picked as "forward" (the two flow-splitting algorithms differ).
      Matching is therefore done on an UNDIRECTED (canonical) 5-tuple.
 
Matching strategy:
    1. Build a canonical (undirected) key from both datasets:
       (sorted endpoint pair, protocol).
    2. Convert our epoch timestamps to Atlantic Daylight Time and floor
       to the minute. Try the flow's own minute plus the two neighbouring
       minutes (+-1), since CICFlowMeter's flow-start clock and ours can
       land on opposite sides of a minute boundary by a second or two.
    3. Where a (key, minute) bucket contains more than one official flow
       record, disambiguate by picking the record whose Flow Duration is
       closest to ours.
    4. Flows with no match at all get label = NaN. They are kept (not
       silently dropped) so the mismatch rate is visible; drop them
       explicitly before training.
 
Usage:
    python src/label_matcher.py
    python src/label_matcher.py --report-only
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import src.config as config

# Config
# ============================================================
# PCAP epoch (UTC) -> CSV local time (Atlantic Daylight Time).
UTC_TO_CSV_LOCAL_HOURS = -3

# +/- minutes around our flow's own minute to search.
MINUTE_TOLERANCE = 1

# Temporary columns for the canonical key; removed before saving.
CKEY_COLUMNS = [
    "ckey_ip1", "ckey_port1", "ckey_ip2", "ckey_port2", "ckey_proto",
]

# Canonical key construction
def build_canonical_columns(df, ip1_col, port1_col, ip2_col, port2_col, proto_col):
    """
    Add canonical endpoint columns so that a flow and its mirror image
    map to the same key. Direction is deliberately ignored.
    """

    ip1 = df[ip1_col].astype(str)
    port1 = pd.to_numeric(df[port1_col], errors="coerce").astype("Int64")
    ip2 = df[ip2_col].astype(str)
    port2 = pd.to_numeric(df[port2_col], errors="coerce").astype("Int64")

    endpoint_a = list(zip(ip1, port1))
    endpoint_b = list(zip(ip2, port2))

    ckey_ip1, ckey_port1, ckey_ip2, ckey_port2 = [], [], [], []

    for a, b in zip(endpoint_a, endpoint_b):
        if a <= b:
            first, second = a, b
        else:
            first, second = b, a
        ckey_ip1.append(first[0])
        ckey_port1.append(first[1])
        ckey_ip2.append(second[0])
        ckey_port2.append(second[1])

    df = df.copy()
    df["ckey_ip1"] = ckey_ip1
    df["ckey_port1"] = ckey_port1
    df["ckey_ip2"] = ckey_ip2
    df["ckey_port2"] = ckey_port2
    df["ckey_proto"] = pd.to_numeric(df[proto_col], errors="coerce").astype("Int64")

    return df

# Loaders
def load_official_labels(csv_path):
    """Load the CIC-IDS2017 GeneratedLabelledFlows CSV for one day."""

    print(f"Loading official label CSV: {csv_path}")
    df = pd.read_csv(csv_path, encoding="latin1")
    df.columns = df.columns.str.strip()

    # CIC-IDS2017 uses DD/MM/YYYY -- dayfirst=True is mandatory.
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")

    # CIC-IDS2017's Timestamp column uses 12-hour clock notation WITHOUT an
    # AM/PM marker (e.g. "1:23" for 1:23 PM). pandas parses these as 01:23,
    # which silently shifts every afternoon record 12 hours early.
    # Verified empirically: hours 1-5 appear in huge volume even though the
    # capture window (08:42-17:10) makes genuine 1am-5am traffic impossible.
    # Fix: treat any parsed hour in 1-5 as actually being that hour + 12pm.
    AFTERNOON_HOUR_FIX = {1, 2, 3, 4, 5}
    hour = df["Timestamp"].dt.hour
    needs_fix = hour.isin(AFTERNOON_HOUR_FIX)
    df.loc[needs_fix, "Timestamp"] = df.loc[needs_fix, "Timestamp"] + pd.Timedelta(hours=12)
    
    df["minute"] = df["Timestamp"].dt.floor("min")

    # Flow Duration in the official CSV is microseconds.
    df["duration_sec"] = pd.to_numeric(df["Flow Duration"], errors="coerce") / 1_000_000.0

    df = build_canonical_columns(
        df,
        "Source IP", "Source Port",
        "Destination IP", "Destination Port",
        "Protocol",
    )

    keep_cols = ["Label", "minute", "duration_sec"] + CKEY_COLUMNS

    print(f"  Rows loaded: {len(df):,}")
    print("  Raw label distribution:")
    print(df["Label"].astype(str).str.strip().value_counts().to_string())
    print()

    return df[keep_cols]

def load_our_flows(csv_path):
    """Load our own flow_features.csv and add matching key/minute columns."""

    print(f"Loading our flows: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Rows loaded: {len(df):,}")

    # Our timestamps are Unix epoch (UTC). Convert to Atlantic Daylight Time
    # to align with the official CSV's local-time clock.
    local_time = (
        pd.to_datetime(df["timestamp"], unit="s", utc=True)
        .dt.tz_convert(None)
        + pd.Timedelta(hours=UTC_TO_CSV_LOCAL_HOURS)
    )
    df["minute"] = local_time.dt.floor("min")

    df = build_canonical_columns(
        df,
        "src_ip", "src_port",
        "dst_ip", "dst_port",
        "ip_protocol",
    )

    return df

# Matching
def match_labels(our_df, official_df):
    """
    Attach labels to our flows.

    Matching priority:
        1. Canonical bidirectional 5-tuple
        2. Closest official timestamp (within +/- MINUTE_TOLERANCE)
        3. Closest flow duration as tie-break

    Returns:
        (our_df_with_label, report_dict)
    """
    # --------------------------------------------------------
    # Build lookup: (canonical key, minute) -> list of candidates.
    # Candidate = (label, official_minute, official_duration_sec)
    # --------------------------------------------------------
    lookup = {}
    for row in official_df.itertuples(index=False):
        key = (
            row.ckey_ip1, row.ckey_port1,
            row.ckey_ip2, row.ckey_port2,
            row.ckey_proto, row.minute,
        )
        lookup.setdefault(key, []).append(
            (row.Label, row.minute, row.duration_sec)
        )

    # Result containers
    labels = []
    original_labels = []
    match_status = []
    time_diffs = []
    duration_diffs = []
    candidate_counts = []

    n_exact = 0
    n_nearby = 0
    n_unmatched = 0

    # Match each flow
    for row in our_df.itertuples(index=False):

        base_key = (
            row.ckey_ip1, row.ckey_port1,
            row.ckey_ip2, row.ckey_port2,
            row.ckey_proto,
        )

        our_duration = (
            float(row.flow_duration)
            if pd.notna(row.flow_duration) else np.nan
        )

        # ---- collect candidates within +/- MINUTE_TOLERANCE ----
        candidates = []
        for delta in range(-MINUTE_TOLERANCE, MINUTE_TOLERANCE + 1):
            minute = row.minute + pd.Timedelta(minutes=delta)
            found = lookup.get(base_key + (minute,))
            if found:
                candidates.extend(found)

        # ---- no candidate -> unmatched ----
        if not candidates:
            labels.append(np.nan)
            original_labels.append(np.nan)
            match_status.append("unmatched")
            time_diffs.append(np.nan)
            duration_diffs.append(np.nan)
            candidate_counts.append(0)
            n_unmatched += 1
            continue

        # ---- score candidates: (timestamp_diff, duration_diff) ----
        def candidate_score(candidate):
            _, cand_minute, cand_duration = candidate

            ts_diff = abs((row.minute - cand_minute).total_seconds())

            if pd.notna(our_duration) and pd.notna(cand_duration):
                dur_diff = abs(our_duration - float(cand_duration))
            else:
                dur_diff = np.inf

            return (ts_diff, dur_diff)

        chosen = min(candidates, key=candidate_score)
        raw_label, chosen_minute, chosen_duration = chosen

        # ---- diagnostics for the CHOSEN candidate ----
        ts_diff_sec = abs((row.minute - chosen_minute).total_seconds())

        if pd.notna(our_duration) and pd.notna(chosen_duration):
            dur_diff_sec = abs(our_duration - float(chosen_duration))
        else:
            dur_diff_sec = np.nan

        # ---- clean official label ----
        raw_label_clean = str(raw_label).strip()
        numeric_label = (
            config.LABEL_BENIGN
            if raw_label_clean.upper() == "BENIGN"
            else config.LABEL_ATTACK
        )

        labels.append(numeric_label)
        original_labels.append(raw_label_clean)

        # ---- status based on the CHOSEN candidate ----
        if ts_diff_sec == 0:
            match_status.append("exact_minute")
            n_exact += 1
        else:
            match_status.append("nearby_minute")
            n_nearby += 1

        time_diffs.append(float(ts_diff_sec))
        duration_diffs.append(
            float(dur_diff_sec) if pd.notna(dur_diff_sec) else np.nan
        )
        candidate_counts.append(len(candidates))

    # Attach results
    our_df = our_df.copy()
    our_df["original_label"] = original_labels
    our_df["label"] = labels
    our_df["label_match_status"] = match_status
    our_df["label_match_time_diff"] = time_diffs
    our_df["label_match_duration_diff"] = duration_diffs
    our_df["label_match_candidates"] = candidate_counts

    report = {
        "total": len(our_df),
        "matched": n_exact + n_nearby,
        "exact_minute": n_exact,
        "nearby_minute": n_nearby,
        "unmatched": n_unmatched,
    }

    return our_df, report

# Main
def main():

    ap = argparse.ArgumentParser(
        description="Attach ground-truth CIC-IDS2017 labels to flow_features.csv."
    )
    ap.add_argument("--flows", default=str(config.FLOW_FEATURES_FILE),
                    help="Input flow-level CSV.")
    ap.add_argument("--labels", default=str(config.LABELLED_WEDNESDAY_FILE),
                    help="Official CIC-IDS2017 labelled flow CSV.")
    ap.add_argument("--out", default=str(config.FINAL_ML_DATASET_FILE),
                    help="Output labelled dataset CSV.")
    ap.add_argument("--report-only", action="store_true",
                    help="Print the report only; do not write a CSV.")
    args = ap.parse_args()

    print("=" * 60)
    print("NetworkIDS - Label Matcher")
    print("=" * 60)
    print(f"Our flows     : {args.flows}")
    print(f"Official CSV  : {args.labels}")
    print(f"Output        : {args.out}")
    print()

    our_df = load_our_flows(args.flows)
    official_df = load_official_labels(args.labels)

    print()
    print("Matching...")
    matched_df, report = match_labels(our_df, official_df)

    # ---- match report ----
    print()
    print("Match report:")
    total = report["total"]
    for k, v in report.items():
        if k == "total":
            print(f"  {k:<16}: {v:,}")
        else:
            pct = f" ({v / total:.1%})" if total else ""
            print(f"  {k:<16}: {v:,}{pct}")

    # ---- label distribution ----
    print()
    print("Binary label distribution (matched rows only):")
    print(matched_df["label"].value_counts(dropna=True).sort_index().to_string())

    print()
    print("Original CIC label distribution (all rows):")
    print(matched_df["original_label"].value_counts(dropna=False).head(15).to_string())

    # ---- unmatched sample ----
    unmatched = matched_df[matched_df["label_match_status"] == "unmatched"]
    if len(unmatched) > 0:
        print()
        print(f"Sample unmatched flows (first 5 of {len(unmatched):,}):")
        show_cols = [
            "flow_id", "src_ip", "dst_ip", "src_port", "dst_port",
            "ip_protocol", "flow_packet_count", "flow_duration",
        ]
        show_cols = [c for c in show_cols if c in unmatched.columns]
        print(unmatched[show_cols].head().to_string(index=False))

    # ---- warnings ----
    print()
    if report["matched"] == 0:
        print("  [ERROR] No flows were matched to any official label.")
        print("          Check timezone / date parsing / canonical key logic.")
    elif report["matched"] / total < 0.5:
        print(f"  [WARN] Match rate is below 50% ({report['matched']}/{total}).")
        print("         Some difference in flow definition is likely.")
    else:
        print(f"  [OK] Match rate: {report['matched'] / total:.1%}")

    if config.LABEL_ATTACK not in matched_df["label"].dropna().values:
        print("  [WARN] No attack samples found.")
        print("         Expected for a benign-only capture like Wednesday morning.")
        print("         RF training will need attack data from another day later.")

    # ---- write ----
    if args.report_only:
        print()
        print("Report-only mode: no CSV written.")
        return

    out_df = matched_df.drop(columns=CKEY_COLUMNS)

    metadata_cols = [c for c in config.METADATA_COLUMNS if c in out_df.columns]
    ml_cols = [c for c in config.ML_FEATURES if c in out_df.columns]
    label_cols = [
        "original_label",
        "label",
        "label_match_status",
        "label_match_time_diff",
        "label_match_duration_diff",
        "label_match_candidates",
    ]
    label_cols = [c for c in label_cols if c in out_df.columns]

    ordered = metadata_cols + ml_cols + label_cols
    remaining = [c for c in out_df.columns if c not in ordered]
    out_df = out_df[ordered + remaining]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print()
    print(f"Written: {out_path}")
    print(f"Shape  : {out_df.shape[0]:,} rows x {out_df.shape[1]} columns")
    print()
    print("NOTE: unmatched rows are kept with label = NaN.")
    print("      Before training, drop them explicitly:")
    print("      df = df.dropna(subset=['label'])")

if __name__ == "__main__":
    main()