from pathlib import Path
import json

import numpy as np
import pandas as pd

import config


# ============================================================
# Configuration
# ============================================================

TEST_FILE = (
    config.PROCESSED_DIR
    / "split"
    / "test_dataset.csv"
)

COMPARISON_DIR = (
    config.MODELS_DIR
    / "error_analysis"
    / "comparison"
)

ERROR_ANALYSIS_DIR = (
    config.MODELS_DIR
    / "error_analysis"
)

OUTPUT_DIR = (
    ERROR_ANALYSIS_DIR
    / "error_profiles"
)


# Transition-analysis files
PERSISTENT_FN_FILE = (
    COMPARISON_DIR
    / "persistent_fn.csv"
)

V2_ONLY_FN_FILE = (
    COMPARISON_DIR
    / "v2_only_fn.csv"
)

PERSISTENT_FP_FILE = (
    COMPARISON_DIR
    / "persistent_fp.csv"
)

V2_ONLY_FP_FILE = (
    COMPARISON_DIR
    / "v2_only_fp.csv"
)


# Output files
ERROR_GROUP_COMPARISON_FILE = (
    OUTPUT_DIR
    / "error_group_comparison.csv"
)

ORIGINAL_LABEL_SUMMARY_FILE = (
    OUTPUT_DIR
    / "original_label_summary.csv"
)

PORT_SUMMARY_FILE = (
    OUTPUT_DIR
    / "port_summary.csv"
)

PROTOCOL_SUMMARY_FILE = (
    OUTPUT_DIR
    / "protocol_summary.csv"
)

ERROR_PROFILE_SUMMARY_FILE = (
    OUTPUT_DIR
    / "error_profile_summary.json"
)


# ============================================================
# Feature groups
# ============================================================

# Features that are useful for profiling network behaviour.
PROFILE_FEATURES = [
    "ip_protocol",

    "ttl_mean",
    "tcp_window_mean",
    "tcp_header_length_mean",

    "syn_count",
    "ack_count",
    "fin_count",
    "rst_count",
    "psh_count",

    "syn_seen",
    "syn_ack_seen",
    "ack_after_syn_ack",

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

    "frame_length_mean",
    "ip_total_length_mean",

    "src_port",
    "dst_port",

    "ip_dscp",
    "ip_ecn",
    "ip_header_length",
    "ip_checksum_valid",
    "l4_checksum_valid",

    "flow_direction",
]


# Probability columns available in the transition file.
PROBABILITY_COLUMNS = [
    "baseline_attack_probability",
    "v1_attack_probability",
    "v2_attack_probability",
]


# ============================================================
# Helpers
# ============================================================

def require_columns(df, columns, name):
    missing = [
        col for col in columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name}: missing required columns:\n"
            f"  {missing}"
        )


def load_csv(path, name):
    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found:\n"
            f"  {path}"
        )

    df = pd.read_csv(path)

    print(
        f"  {name}: {len(df):,} rows"
    )

    return df


def safe_numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def numeric_summary(df, feature):
    """
    Return useful descriptive statistics for one feature.
    """
    values = safe_numeric(df[feature]).dropna()

    if len(values) == 0:
        return {
            "count": 0,
            "missing": int(df[feature].isna().sum()),
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }

    return {
        "count": int(len(values)),
        "missing": int(df[feature].isna().sum()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def distribution_difference(error_df, reference_df, feature):
    """
    Compare an error group against its corresponding reference group.

    Uses:
      - mean difference
      - median difference
      - missing-rate difference
    """

    error_values = safe_numeric(
        error_df[feature]
    )

    reference_values = safe_numeric(
        reference_df[feature]
    )

    error_non_null = error_values.dropna()
    reference_non_null = reference_values.dropna()

    error_missing_rate = (
        error_values.isna().mean()
    )

    reference_missing_rate = (
        reference_values.isna().mean()
    )

    if len(error_non_null) == 0:
        error_mean = np.nan
        error_median = np.nan
    else:
        error_mean = error_non_null.mean()
        error_median = error_non_null.median()

    if len(reference_non_null) == 0:
        reference_mean = np.nan
        reference_median = np.nan
    else:
        reference_mean = reference_non_null.mean()
        reference_median = reference_non_null.median()

    if (
        pd.notna(error_mean)
        and
        pd.notna(reference_mean)
    ):
        mean_difference = (
            error_mean
            - reference_mean
        )
    else:
        mean_difference = np.nan

    if (
        pd.notna(error_median)
        and
        pd.notna(reference_median)
    ):
        median_difference = (
            error_median
            - reference_median
        )
    else:
        median_difference = np.nan

    return {
        "error_count": len(error_df),
        "reference_count": len(reference_df),

        "error_mean": error_mean,
        "reference_mean": reference_mean,
        "mean_difference": mean_difference,

        "error_median": error_median,
        "reference_median": reference_median,
        "median_difference": median_difference,

        "error_missing_rate":
            error_missing_rate,

        "reference_missing_rate":
            reference_missing_rate,

        "missing_rate_difference":
            (
                error_missing_rate
                - reference_missing_rate
            ),
    }


def create_profile_comparison(
    group_name,
    error_df,
    reference_df,
):
    """
    Compare one error group against its reference group.
    """

    rows = []

    for feature in PROFILE_FEATURES:

        if feature not in error_df.columns:
            continue

        if feature not in reference_df.columns:
            continue

        result = distribution_difference(
            error_df,
            reference_df,
            feature,
        )

        result["group"] = group_name
        result["feature"] = feature

        rows.append(result)

    return pd.DataFrame(rows)


def label_distribution(
    df,
    group_name,
):
    if "original_label" in df.columns:
        column = "original_label"
    elif "baseline_original_label" in df.columns:
        column = "baseline_original_label"
    else:
        return pd.DataFrame()

    counts = (
        df[column]
        .fillna("<NA>")
        .value_counts(dropna=False)
        .rename_axis("original_label")
        .reset_index(name="count")
    )

    counts["group"] = group_name

    counts["percentage"] = (
        counts["count"]
        / len(df)
        * 100
    )

    return counts[
        [
            "group",
            "original_label",
            "count",
            "percentage",
        ]
    ]


def port_distribution(
    df,
    group_name,
):
    rows = []

    for port_type in [
        "src_port",
        "dst_port",
    ]:

        if port_type not in df.columns:
            continue

        values = (
            pd.to_numeric(
                df[port_type],
                errors="coerce",
            )
            .dropna()
            .astype(int)
        )

        counts = (
            values
            .value_counts()
            .head(20)
        )

        for port, count in counts.items():

            rows.append(
                {
                    "group": group_name,
                    "port_type": port_type,
                    "port": int(port),
                    "count": int(count),
                    "percentage": (
                        count
                        / len(df)
                        * 100
                    ),
                }
            )

    return pd.DataFrame(rows)


def protocol_distribution(
    df,
    group_name,
):
    if "ip_protocol" not in df.columns:
        return pd.DataFrame()

    values = pd.to_numeric(
        df["ip_protocol"],
        errors="coerce",
    )

    counts = (
        values
        .value_counts(dropna=False)
        .rename_axis("ip_protocol")
        .reset_index(name="count")
    )

    counts["group"] = group_name

    counts["percentage"] = (
        counts["count"]
        / len(df)
        * 100
    )

    return counts[
        [
            "group",
            "ip_protocol",
            "count",
            "percentage",
        ]
    ]


def probability_profile(
    df,
    group_name,
):
    rows = []

    for column in PROBABILITY_COLUMNS:

        if column not in df.columns:
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        if len(values) == 0:
            continue

        near_threshold = (
            (values >= 0.45)
            &
            (values <= 0.55)
        ).sum()

        rows.append(
            {
                "group": group_name,
                "probability": column,
                "count": int(len(values)),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
                "near_threshold_count":
                    int(near_threshold),
                "near_threshold_percentage":
                    float(
                        near_threshold
                        / len(values)
                        * 100
                    ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("NetworkIDS - Error Profile Analysis")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Load test dataset
    # --------------------------------------------------------

    print("\n[1/8] Loading test dataset...")

    test_df = load_csv(
        TEST_FILE,
        "Test dataset",
    )

    require_columns(
        test_df,
        ["flow_id", "label"],
        "Test dataset",
    )

    print(
        f"  Attack: "
        f"{(test_df['label'] == 1).sum():,}"
    )

    print(
        f"  Benign: "
        f"{(test_df['label'] == 0).sum():,}"
    )

    # --------------------------------------------------------
    # 2. Load error groups
    # --------------------------------------------------------

    print("\n[2/8] Loading error groups...")

    persistent_fn = load_csv(
        PERSISTENT_FN_FILE,
        "Persistent FN",
    )

    v2_only_fn = load_csv(
        V2_ONLY_FN_FILE,
        "V2-only FN",
    )

    persistent_fp = load_csv(
        PERSISTENT_FP_FILE,
        "Persistent FP",
    )

    v2_only_fp = load_csv(
        V2_ONLY_FP_FILE,
        "V2-only FP",
    )

    error_groups = {
        "Persistent FN": persistent_fn,
        "V2-only FN": v2_only_fn,
        "Persistent FP": persistent_fp,
        "V2-only FP": v2_only_fp,
    }

    # --------------------------------------------------------
    # 3. Prepare reference populations
    # --------------------------------------------------------

    print(
        "\n[3/8] Preparing Attack / Benign reference groups..."
    )

    all_attack = test_df[
        test_df["label"] == 1
    ].copy()

    all_benign = test_df[
        test_df["label"] == 0
    ].copy()

    print(
        f"  All Attack : "
        f"{len(all_attack):,}"
    )

    print(
        f"  All Benign : "
        f"{len(all_benign):,}"
    )

    # --------------------------------------------------------
    # 4. Merge test dataset features into error groups
    # --------------------------------------------------------

    print(
        "\n[4/8] Attaching full feature data to error groups..."
    )

    feature_columns = [
        "flow_id",
        "label",
        "original_label",
    ]

    feature_columns.extend(
        [
            feature
            for feature in PROFILE_FEATURES
            if feature in test_df.columns
        ]
    )

    # Add metadata that may be useful.
    for column in [
        "timestamp",
        "src_ip",
        "dst_ip",
        "src_mac",
        "dst_mac",
    ]:
        if column in test_df.columns:
            feature_columns.append(column)

    feature_source = (
        test_df[
            list(dict.fromkeys(feature_columns))
        ]
        .copy()
    )

    enriched_groups = {}

    for group_name, group_df in error_groups.items():

        require_columns(
            group_df,
            ["flow_id"],
            group_name,
        )

        enriched = group_df.merge(
            feature_source,
            on="flow_id",
            how="left",
            suffixes=("", "_test"),
            validate="one_to_one",
        )

        # Prefer test-dataset feature values.
        for feature in PROFILE_FEATURES:

            test_column = (
                f"{feature}_test"
            )

            if test_column in enriched.columns:

                if feature in enriched.columns:

                    enriched[feature] = (
                        enriched[feature]
                        .combine_first(
                            enriched[test_column]
                        )
                    )

                    enriched.drop(
                        columns=[
                            test_column
                        ],
                        inplace=True,
                    )

                else:

                    enriched.rename(
                        columns={
                            test_column:
                                feature
                        },
                        inplace=True,
                    )

        enriched_groups[
            group_name
        ] = enriched

        print(
            f"  {group_name}: "
            f"{len(enriched):,} rows"
        )

    # --------------------------------------------------------
    # 5. Feature profile comparison
    # --------------------------------------------------------

    print(
        "\n[5/8] Comparing feature distributions..."
    )

    comparison_frames = []

    comparison_frames.append(
        create_profile_comparison(
            "Persistent FN vs All Attack",
            enriched_groups["Persistent FN"],
            all_attack,
        )
    )

    comparison_frames.append(
        create_profile_comparison(
            "V2-only FN vs All Attack",
            enriched_groups["V2-only FN"],
            all_attack,
        )
    )

    comparison_frames.append(
        create_profile_comparison(
            "Persistent FP vs All Benign",
            enriched_groups["Persistent FP"],
            all_benign,
        )
    )

    comparison_frames.append(
        create_profile_comparison(
            "V2-only FP vs All Benign",
            enriched_groups["V2-only FP"],
            all_benign,
        )
    )

    comparison_df = pd.concat(
        comparison_frames,
        ignore_index=True,
    )

    # Add absolute standardized-ish difference.
    # This is descriptive only; it is NOT a statistical test.
    def add_relative_difference(df):

        reference_abs = (
            df["reference_mean"]
            .abs()
        )

        df["relative_mean_difference"] = np.where(
            reference_abs > 1e-12,
            (
                df["mean_difference"].abs()
                / reference_abs
                * 100
            ),
            np.nan,
        )

        return df

    comparison_df["relative_mean_difference"] = np.where(
        comparison_df["reference_mean"].abs() > 1e-12,
        (
            comparison_df["mean_difference"].abs()
            / comparison_df["reference_mean"].abs()
            * 100
        ),
        np.nan,
    )

    comparison_df.to_csv(
        ERROR_GROUP_COMPARISON_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # 6. Label / port / protocol profiles
    # --------------------------------------------------------

    print(
        "\n[6/8] Generating categorical profiles..."
    )

    label_frames = []

    for group_name, group_df in enriched_groups.items():

        label_frames.append(
            label_distribution(
                group_df,
                group_name,
            )
        )

    # Add full references.
    label_frames.append(
        label_distribution(
            all_attack,
            "All Attack",
        )
    )

    label_frames.append(
        label_distribution(
            all_benign,
            "All Benign",
        )
    )

    label_summary = pd.concat(
        label_frames,
        ignore_index=True,
    )

    label_summary.to_csv(
        ORIGINAL_LABEL_SUMMARY_FILE,
        index=False,
    )

    # Ports.
    port_frames = []

    for group_name, group_df in enriched_groups.items():

        port_frames.append(
            port_distribution(
                group_df,
                group_name,
            )
        )

    port_frames.append(
        port_distribution(
            all_attack,
            "All Attack",
        )
    )

    port_frames.append(
        port_distribution(
            all_benign,
            "All Benign",
        )
    )

    port_summary = pd.concat(
        port_frames,
        ignore_index=True,
    )

    port_summary.to_csv(
        PORT_SUMMARY_FILE,
        index=False,
    )

    # Protocols.
    protocol_frames = []

    for group_name, group_df in enriched_groups.items():

        protocol_frames.append(
            protocol_distribution(
                group_df,
                group_name,
            )
        )

    protocol_frames.append(
        protocol_distribution(
            all_attack,
            "All Attack",
        )
    )

    protocol_frames.append(
        protocol_distribution(
            all_benign,
            "All Benign",
        )
    )

    protocol_summary = pd.concat(
        protocol_frames,
        ignore_index=True,
    )

    protocol_summary.to_csv(
        PROTOCOL_SUMMARY_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # 7. Probability analysis
    # --------------------------------------------------------

    print(
        "\n[7/8] Analysing model probability profiles..."
    )

    probability_frames = []

    for group_name, group_df in enriched_groups.items():

        probability_frames.append(
            probability_profile(
                group_df,
                group_name,
            )
        )

    probability_df = pd.concat(
        probability_frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # 8. Human-readable terminal summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ERROR PROFILE SUMMARY")
    print("=" * 70)

    for group_name, group_df in enriched_groups.items():

        print("\n" + "-" * 70)
        print(group_name.upper())
        print("-" * 70)

        print(
            f"  Rows: "
            f"{len(group_df):,}"
        )

        # Original labels.
        if "original_label" in group_df.columns:

            print(
                "\n  Original labels:"
            )

            counts = (
                group_df["original_label"]
                .fillna("<NA>")
                .value_counts()
            )

            for label, count in counts.head(10).items():

                print(
                    f"    {label}: "
                    f"{count:,} "
                    f"({count / len(group_df) * 100:.2f}%)"
                )

        # Ports.
        for port_type in [
            "src_port",
            "dst_port",
        ]:

            if port_type not in group_df.columns:
                continue

            values = pd.to_numeric(
                group_df[port_type],
                errors="coerce",
            ).dropna()

            if len(values) == 0:
                continue

            counts = (
                values
                .astype(int)
                .value_counts()
                .head(10)
            )

            print(
                f"\n  Top {port_type}:"
            )

            for port, count in counts.items():

                print(
                    f"    {int(port)}: "
                    f"{count:,} "
                    f"({count / len(group_df) * 100:.2f}%)"
                )

        # Important numeric features.
        print(
            "\n  Numeric profile:"
        )

        for feature in [
            "ttl_mean",
            "flow_packet_count",
            "flow_byte_count",
            "flow_duration",
            "tcp_window_mean",
            "tcp_header_length_mean",
            "rst_count",
            "syn_count",
            "ack_count",
            "byte_rate",
        ]:

            if feature not in group_df.columns:
                continue

            summary = numeric_summary(
                group_df,
                feature,
            )

            if summary["count"] == 0:
                print(
                    f"    {feature}: all NaN"
                )
                continue

            print(
                f"    {feature}: "
                f"mean={summary['mean']:.4f}, "
                f"median={summary['median']:.4f}, "
                f"min={summary['min']:.4f}, "
                f"max={summary['max']:.4f}"
            )

        # Probabilities.
        print(
            "\n  Model probabilities:"
        )

        for column in PROBABILITY_COLUMNS:

            if column not in group_df.columns:
                continue

            values = pd.to_numeric(
                group_df[column],
                errors="coerce",
            ).dropna()

            if len(values) == 0:
                continue

            near_threshold = (
                (values >= 0.45)
                &
                (values <= 0.55)
            ).sum()

            print(
                f"    {column}: "
                f"mean={values.mean():.4f}, "
                f"median={values.median():.4f}, "
                f"0.45-0.55="
                f"{near_threshold:,}"
            )

    # --------------------------------------------------------
    # Save JSON summary
    # --------------------------------------------------------

    summary = {
        "test_rows": int(len(test_df)),
        "all_attack_rows": int(len(all_attack)),
        "all_benign_rows": int(len(all_benign)),
        "groups": {},
    }

    for group_name, group_df in enriched_groups.items():

        group_summary = {
            "count": int(len(group_df)),
        }

        # Original label distribution.
        if "original_label" in group_df.columns:

            label_counts = (
                group_df["original_label"]
                .fillna("<NA>")
                .value_counts()
                .to_dict()
            )

            group_summary[
                "original_label_counts"
            ] = {
                str(key): int(value)
                for key, value
                in label_counts.items()
            }

        # Probability summary.
        probability_summary = {}

        for column in PROBABILITY_COLUMNS:

            if column not in group_df.columns:
                continue

            values = pd.to_numeric(
                group_df[column],
                errors="coerce",
            ).dropna()

            if len(values) == 0:
                continue

            probability_summary[column] = {
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
                "near_threshold_count": int(
                    (
                        (values >= 0.45)
                        &
                        (values <= 0.55)
                    ).sum()
                ),
            }

        group_summary[
            "probabilities"
        ] = probability_summary

        # Numeric summaries.
        numeric_summary_dict = {}

        for feature in PROFILE_FEATURES:

            if feature not in group_df.columns:
                continue

            numeric_summary_dict[
                feature
            ] = numeric_summary(
                group_df,
                feature,
            )

        group_summary[
            "numeric_features"
        ] = numeric_summary_dict

        summary[
            "groups"
        ][group_name] = group_summary

    with open(
        ERROR_PROFILE_SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            allow_nan=True,
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"\n  Feature comparison:"
        f"\n    {ERROR_GROUP_COMPARISON_FILE}"
    )

    print(
        f"\n  Original labels:"
        f"\n    {ORIGINAL_LABEL_SUMMARY_FILE}"
    )

    print(
        f"\n  Ports:"
        f"\n    {PORT_SUMMARY_FILE}"
    )

    print(
        f"\n  Protocols:"
        f"\n    {PROTOCOL_SUMMARY_FILE}"
    )

    print(
        f"\n  JSON summary:"
        f"\n    {ERROR_PROFILE_SUMMARY_FILE}"
    )

    print("\n" + "=" * 70)
    print("ERROR PROFILE ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()