from pathlib import Path
import argparse

import pandas as pd

import config


INPUT_FILE = (
    config.PROCESSED_DIR
    / "integrity"
    / "packet_integrity.csv"
)

OUTPUT_DIR = (
    config.PROCESSED_DIR
    / "integrity"
)

SUMMARY_FILE = OUTPUT_DIR / "integrity_result_summary.txt"
STATE_COUNTS_FILE = OUTPUT_DIR / "integrity_state_counts.csv"
RULE_COUNTS_FILE = OUTPUT_DIR / "integrity_rule_counts.csv"
PROTOCOL_FILE = OUTPUT_DIR / "integrity_protocol_breakdown.csv"
EXAMPLES_FILE = OUTPUT_DIR / "integrity_suspicious_examples.csv"
INDICATOR_FILE = OUTPUT_DIR / "integrity_indicator_summary.csv"


TAMPERED_RULES = [
    "T1_invalid_ip_checksum",
    "T2_fragmentation_contradiction",
    "T3_invalid_ip_header_length",
]

SUSPICIOUS_RULES = [
    "S1_ttl_anomaly",
    "S2_mac_ip_inconsistency",
    "S3_syn_fin",
    "S4_syn_rst",
    "S5_fin_rst",
    "S6_null_tcp",
]

INDICATOR_RULES = [
    "S7_invalid_l4_checksum",
]

ALL_RULES = (
    TAMPERED_RULES
    + SUSPICIOUS_RULES
    + INDICATOR_RULES
)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze NetworkIDS integrity checker results."
    )

    parser.add_argument(
        "--examples",
        type=int,
        default=100,
        help="Maximum number of suspicious examples to save.",
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("NetworkIDS - Integrity Result Analysis")
    print("=" * 70)
    print(f"Input: {INPUT_FILE}")
    print()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Integrity output not found: {INPUT_FILE}"
        )

    # --------------------------------------------------------------
    # Load required columns
    # --------------------------------------------------------------

    required_columns = [
        "packet_id",
        "timestamp",
        "src_ip",
        "dst_ip",
        "src_mac",
        "dst_mac",
        "ip_version",
        "ip_protocol",
        "l4_protocol",
        "ttl",
        "integrity_state",
        "tampered_rule_count",
        "suspicious_rule_count",
        "integrity_reasons",
        "integrity_indicators",
    ] + ALL_RULES

    print("[1/6] Loading integrity results...")

    df = pd.read_csv(
        INPUT_FILE,
        usecols=required_columns,
        low_memory=False,
    )

    total_packets = len(df)

    print(
        f"  Packets loaded: {total_packets:,}"
    )

    # --------------------------------------------------------------
    # Validation 1 - required columns
    # --------------------------------------------------------------

    print()
    print("[2/6] Validating integrity output...")

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    validation_results = []

    validation_results.append(
        (
            "Required columns",
            len(missing_columns) == 0,
        )
    )

    # --------------------------------------------------------------
    # Validation 2 - valid states
    # --------------------------------------------------------------

    valid_states = {
        "Intact",
        "Suspicious",
        "Tampered",
    }

    invalid_states = (
        ~df["integrity_state"].isin(valid_states)
    ).sum()

    validation_results.append(
        (
            "Valid integrity states",
            invalid_states == 0,
        )
    )

    # --------------------------------------------------------------
    # Validation 3 - state logic
    #
    # Tampered:
    #   at least one T1-T3
    #
    # Suspicious:
    #   no T1-T3
    #   at least one S1-S6
    #
    # Intact:
    #   no T1-T3
    #   no S1-S6
    #
    # S7 does NOT affect state.
    # --------------------------------------------------------------

    tampered_rule_sum = df[
        TAMPERED_RULES
    ].sum(axis=1)

    suspicious_rule_sum = df[
        SUSPICIOUS_RULES
    ].sum(axis=1)

    expected_state = pd.Series(
        "Intact",
        index=df.index,
    )

    expected_state.loc[
        suspicious_rule_sum.gt(0)
    ] = "Suspicious"

    expected_state.loc[
        tampered_rule_sum.gt(0)
    ] = "Tampered"

    state_mismatch = (
        df["integrity_state"] != expected_state
    )

    mismatch_count = int(
        state_mismatch.sum()
    )

    validation_results.append(
        (
            "Integrity state logic",
            mismatch_count == 0,
        )
    )

    # --------------------------------------------------------------
    # Validation 4 - rule counts
    # --------------------------------------------------------------

    expected_tampered_count = (
        df[TAMPERED_RULES].sum(axis=1)
    )

    expected_suspicious_count = (
        df[SUSPICIOUS_RULES].sum(axis=1)
    )

    tampered_count_mismatch = (
        df["tampered_rule_count"]
        != expected_tampered_count
    ).sum()

    suspicious_count_mismatch = (
        df["suspicious_rule_count"]
        != expected_suspicious_count
    ).sum()

    validation_results.append(
        (
            "Tampered rule count",
            tampered_count_mismatch == 0,
        )
    )

    validation_results.append(
        (
            "Suspicious rule count",
            suspicious_count_mismatch == 0,
        )
    )

    # --------------------------------------------------------------
    # State counts
    # --------------------------------------------------------------

    state_counts = (
        df["integrity_state"]
        .value_counts()
        .reindex(
            ["Intact", "Suspicious", "Tampered"],
            fill_value=0,
        )
    )

    state_rows = []

    for state, count in state_counts.items():
        state_rows.append(
            {
                "integrity_state": state,
                "count": int(count),
                "percentage": (
                    count / total_packets * 100
                    if total_packets
                    else 0.0
                ),
            }
        )

    state_df = pd.DataFrame(state_rows)

    state_df.to_csv(
        STATE_COUNTS_FILE,
        index=False,
    )

    # --------------------------------------------------------------
    # Rule counts
    # --------------------------------------------------------------

    rule_rows = []

    for rule in ALL_RULES:
        count = int(df[rule].sum())

        if rule in TAMPERED_RULES:
            category = "Tampered"
        elif rule in SUSPICIOUS_RULES:
            category = "Suspicious"
        else:
            category = "Indicator"

        rule_rows.append(
            {
                "rule": rule,
                "category": category,
                "count": count,
                "percentage": (
                    count / total_packets * 100
                    if total_packets
                    else 0.0
                ),
            }
        )

    rule_df = pd.DataFrame(rule_rows)

    rule_df.to_csv(
        RULE_COUNTS_FILE,
        index=False,
    )

    # --------------------------------------------------------------
    # Protocol breakdown
    # --------------------------------------------------------------

    print("[3/6] Building protocol/state breakdown...")

    protocol_df = (
        df.assign(
            l4_protocol_clean=(
                df["l4_protocol"]
                .fillna("Unknown")
                .astype(str)
            )
        )
        .groupby(
            [
                "l4_protocol_clean",
                "integrity_state",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    protocol_df["percentage_of_protocol"] = (
        protocol_df.groupby(
            "l4_protocol_clean"
        )["count"]
        .transform(
            lambda x: x / x.sum() * 100
        )
    )

    protocol_df.to_csv(
        PROTOCOL_FILE,
        index=False,
    )

    # --------------------------------------------------------------
    # Suspicious examples
    # --------------------------------------------------------------

    print("[4/6] Extracting suspicious examples...")

    suspicious_mask = (
        df["integrity_state"]
        == "Suspicious"
    )

    suspicious_df = df.loc[
        suspicious_mask
    ].copy()

    example_columns = [
        "packet_id",
        "timestamp",
        "src_ip",
        "dst_ip",
        "src_mac",
        "dst_mac",
        "ip_version",
        "l4_protocol",
        "ttl",
        "integrity_state",
        "tampered_rule_count",
        "suspicious_rule_count",
        "integrity_reasons",
        "integrity_indicators",
    ] + ALL_RULES

    suspicious_examples = suspicious_df[
        example_columns
    ].head(args.examples)

    suspicious_examples.to_csv(
        EXAMPLES_FILE,
        index=False,
    )

    # --------------------------------------------------------------
    # Indicator summary
    # --------------------------------------------------------------

    print("[5/6] Analyzing weak indicators...")

    indicator_rows = []

    for rule in INDICATOR_RULES:
        count = int(
            df[rule].sum()
        )

        # Indicator + integrity state
        if count > 0:
            indicator_and_suspicious = int(
                (
                    (df[rule] == 1)
                    & (
                        df["integrity_state"]
                        == "Suspicious"
                    )
                ).sum()
            )

            indicator_and_tampered = int(
                (
                    (df[rule] == 1)
                    & (
                        df["integrity_state"]
                        == "Tampered"
                    )
                ).sum()
            )

            indicator_and_intact = int(
                (
                    (df[rule] == 1)
                    & (
                        df["integrity_state"]
                        == "Intact"
                    )
                ).sum()
            )
        else:
            indicator_and_suspicious = 0
            indicator_and_tampered = 0
            indicator_and_intact = 0

        indicator_rows.append(
            {
                "indicator": rule,
                "count": count,
                "percentage_of_all_packets": (
                    count / total_packets * 100
                    if total_packets
                    else 0.0
                ),
                "with_intact_state": indicator_and_intact,
                "with_suspicious_state": indicator_and_suspicious,
                "with_tampered_state": indicator_and_tampered,
            }
        )

    indicator_df = pd.DataFrame(
        indicator_rows
    )

    indicator_df.to_csv(
        INDICATOR_FILE,
        index=False,
    )

    # --------------------------------------------------------------
    # Validation summary
    # --------------------------------------------------------------

    print("[6/6] Writing validation summary...")

    all_validation_pass = all(
        passed
        for _, passed in validation_results
    )

    # --------------------------------------------------------------
    # Text summary
    # --------------------------------------------------------------

    summary_lines = [
        "NetworkIDS - Integrity Result Analysis",
        "=" * 70,
        "",
        f"Input file       : {INPUT_FILE}",
        f"Packets analyzed : {total_packets:,}",
        "",
        "INTEGRITY STATE DISTRIBUTION",
        "-" * 70,
    ]

    for row in state_rows:
        summary_lines.append(
            f"{row['integrity_state']:<12} "
            f"{row['count']:>12,} "
            f"({row['percentage']:>8.4f}%)"
        )

    summary_lines.extend(
        [
            "",
            "RULE FREQUENCY",
            "-" * 70,
        ]
    )

    for row in rule_rows:
        summary_lines.append(
            f"{row['rule']:<38} "
            f"{row['count']:>12,} "
            f"({row['percentage']:>8.4f}%) "
            f"[{row['category']}]"
        )

    summary_lines.extend(
        [
            "",
            "SUSPICIOUS PACKETS",
            "-" * 70,
            (
                f"Suspicious packets: "
                f"{int(state_counts['Suspicious']):,}"
            ),
            (
                f"Suspicious examples saved: "
                f"{len(suspicious_examples):,}"
            ),
            "",
            "S7 CHECKSUM INDICATOR",
            "-" * 70,
        ]
    )

    for row in indicator_rows:
        summary_lines.extend(
            [
                (
                    f"{row['indicator']}: "
                    f"{row['count']:,} "
                    f"({row['percentage_of_all_packets']:.4f}%)"
                ),
                (
                    f"  With Intact state    : "
                    f"{row['with_intact_state']:,}"
                ),
                (
                    f"  With Suspicious state: "
                    f"{row['with_suspicious_state']:,}"
                ),
                (
                    f"  With Tampered state  : "
                    f"{row['with_tampered_state']:,}"
                ),
            ]
        )

    summary_lines.extend(
        [
            "",
            "VALIDATION",
            "-" * 70,
        ]
    )

    for name, passed in validation_results:
        summary_lines.append(
            f"{name:<35}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    summary_lines.extend(
        [
            "",
            f"Overall validation: "
            f"{'PASS' if all_validation_pass else 'FAIL'}",
            "",
            "INTERPRETATION",
            "-" * 70,
            "T1-T3 represent strong protocol/header",
            "integrity violations and determine the",
            "Tampered state.",
            "",
            "S1-S6 represent protocol anomalies and",
            "determine the Suspicious state when no",
            "Tampered rule is triggered.",
            "",
            "S7 invalid L4 checksum is retained as a",
            "weak integrity indicator and does not",
            "independently change the integrity state.",
            "",
        ]
    )

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "\n".join(summary_lines)
        )

    # --------------------------------------------------------------
    # Console output
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("INTEGRITY RESULT ANALYSIS - COMPLETE")
    print("=" * 70)

    print(
        f"Packets analyzed: {total_packets:,}"
    )

    print()
    print("Integrity state:")

    for row in state_rows:
        print(
            f"  {row['integrity_state']:<12} "
            f"{row['count']:>12,} "
            f"({row['percentage']:>7.3f}%)"
        )

    print()
    print("Validation:")

    for name, passed in validation_results:
        print(
            f"  {'PASS' if passed else 'FAIL':<5} "
            f"{name}"
        )

    print()
    print(
        "Overall validation: "
        + ("PASS" if all_validation_pass else "FAIL")
    )

    print()
    print("Output files:")
    print(
        f"  Summary   : {SUMMARY_FILE}"
    )
    print(
        f"  States    : {STATE_COUNTS_FILE}"
    )
    print(
        f"  Rules     : {RULE_COUNTS_FILE}"
    )
    print(
        f"  Protocol  : {PROTOCOL_FILE}"
    )
    print(
        f"  Examples  : {EXAMPLES_FILE}"
    )
    print(
        f"  Indicators: {INDICATOR_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()