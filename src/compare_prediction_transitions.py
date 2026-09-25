from pathlib import Path
import argparse
import json

import pandas as pd

import config


# ============================================================
# Argument Parser
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Compare prediction transitions between a baseline "
            "model and a current model."
        )
    )

    parser.add_argument(
        "--baseline",
        required=True,
        help=(
            "Path to baseline prediction_results.csv"
        ),
    )

    parser.add_argument(
        "--current",
        required=True,
        help=(
            "Path to current model prediction_results.csv"
        ),
    )

    parser.add_argument(
        "--name",
        required=True,
        help=(
            "Output analysis name, e.g. baseline_vs_v2_final"
        ),
    )

    return parser.parse_args()


# ============================================================
# Helper Functions
# ============================================================

def prediction_status(y_true, y_pred):
    """
    Convert true/predicted binary labels into:
      TP / TN / FP / FN
    """
    if y_true == 1 and y_pred == 1:
        return "TP"

    if y_true == 0 and y_pred == 0:
        return "TN"

    if y_true == 0 and y_pred == 1:
        return "FP"

    if y_true == 1 and y_pred == 0:
        return "FN"

    raise ValueError(
        f"Invalid combination: y_true={y_true}, y_pred={y_pred}"
    )


def load_prediction_file(name, file_path):
    """
    Load one model's prediction_results.csv.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Prediction result file not found for '{name}':\n"
            f"  {file_path}"
        )

    df = pd.read_csv(file_path)

    required_columns = {
        "flow_id",
        "label",
        "original_label",
        "predicted_label",
        "attack_probability",
        "prediction_type",
        "correct_prediction",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{name}: missing required columns: "
            f"{sorted(missing)}"
        )

    df = df[
        [
            "flow_id",
            "label",
            "original_label",
            "predicted_label",
            "attack_probability",
            "prediction_type",
            "correct_prediction",
        ]
    ].copy()

    # Make sure flow_id is unique.
    if df["flow_id"].duplicated().any():
        duplicate_count = int(
            df["flow_id"].duplicated().sum()
        )

        raise ValueError(
            f"{name}: duplicate flow_id detected: "
            f"{duplicate_count}"
        )

    # Rename model-specific columns.
    df = df.rename(
        columns={
            "label": f"{name}_label",
            "original_label": f"{name}_original_label",
            "predicted_label": f"{name}_predicted_label",
            "attack_probability": f"{name}_attack_probability",
            "prediction_type": f"{name}_status",
            "correct_prediction": f"{name}_correct",
        }
    )

    return df


def check_same_test_set(baseline_df, current_df):
    """
    Confirm baseline and current model evaluated
    exactly the same flow IDs.
    """

    baseline_ids = set(
        baseline_df["flow_id"]
    )

    current_ids = set(
        current_df["flow_id"]
    )

    if baseline_ids != current_ids:

        missing = len(
            baseline_ids - current_ids
        )

        extra = len(
            current_ids - baseline_ids
        )

        raise ValueError(
            "Test-set mismatch between baseline and current model.\n"
            f"  Missing flow IDs in current: {missing}\n"
            f"  Extra flow IDs in current   : {extra}"
        )


def calculate_model_summary(
    merged,
    model_name,
):
    """
    Calculate TP/TN/FP/FN, accuracy, FPR and FNR.
    """

    status_counts = (
        merged[f"{model_name}_status"]
        .value_counts()
        .to_dict()
    )

    tp = int(
        status_counts.get("TP", 0)
    )

    tn = int(
        status_counts.get("TN", 0)
    )

    fp = int(
        status_counts.get("FP", 0)
    )

    fn = int(
        status_counts.get("FN", 0)
    )

    total = (
        tp + tn + fp + fn
    )

    return {
        "total": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (
            (tp + tn) / total
            if total > 0
            else 0
        ),
        "fpr_percent": (
            fp / (fp + tn) * 100
            if (fp + tn) > 0
            else 0
        ),
        "fnr_percent": (
            fn / (fn + tp) * 100
            if (fn + tp) > 0
            else 0
        ),
    }


def print_original_label_distribution(
    df,
    title,
    original_label_column,
):
    """
    Print original CIC label distribution
    for an error group.
    """

    print(f"\n{title}")

    if len(df) == 0:
        print("  None")
        return

    counts = (
        df[original_label_column]
        .fillna("<NA>")
        .value_counts()
    )

    for label, count in counts.items():

        percentage = (
            count / len(df) * 100
        )

        print(
            f"  {label}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    baseline_file = Path(
        args.baseline
    )

    current_file = Path(
        args.current
    )

    analysis_name = args.name

    output_dir = (
        config.MODELS_DIR
        / "error_analysis"
        / analysis_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    transition_file = (
        output_dir
        / "prediction_transitions.csv"
    )

    transition_summary_file = (
        output_dir
        / "transition_summary.csv"
    )

    attack_summary_file = (
        output_dir
        / "attack_transition_summary.csv"
    )

    benign_summary_file = (
        output_dir
        / "benign_transition_summary.csv"
    )

    persistent_fn_file = (
        output_dir
        / "persistent_fn.csv"
    )

    persistent_fp_file = (
        output_dir
        / "persistent_fp.csv"
    )

    baseline_fn_repaired_file = (
        output_dir
        / "baseline_fn_repaired.csv"
    )

    baseline_fp_repaired_file = (
        output_dir
        / "baseline_fp_repaired.csv"
    )

    current_only_fn_file = (
        output_dir
        / "current_only_fn.csv"
    )

    current_only_fp_file = (
        output_dir
        / "current_only_fp.csv"
    )

    summary_json_file = (
        output_dir
        / "comparison_summary.json"
    )

    # ========================================================
    # Header
    # ========================================================

    print("=" * 70)
    print(
        "NetworkIDS - Baseline / Current Prediction "
        "Transition Analysis"
    )
    print("=" * 70)

    print("\nAnalysis:")
    print(
        f"  Name     : {analysis_name}"
    )

    print("\nPrediction files:")
    print(
        f"  Baseline : {baseline_file}"
    )

    print(
        f"  Current  : {current_file}"
    )

    # ========================================================
    # 1. Load prediction results
    # ========================================================

    print(
        "\n[1/7] Loading prediction results..."
    )

    print("  Loading baseline...")

    baseline = load_prediction_file(
        "baseline",
        baseline_file,
    )

    print(
        f"    Rows: {len(baseline):,}"
    )

    print("  Loading current model...")

    current = load_prediction_file(
        "current",
        current_file,
    )

    print(
        f"    Rows: {len(current):,}"
    )

    # ========================================================
    # 2. Check same test set
    # ========================================================

    print(
        "\n[2/7] Checking test-set consistency..."
    )

    check_same_test_set(
        baseline,
        current,
    )

    print(
        "  Flow ID overlap check: PASS"
    )

    print(
        f"  Common test flows: "
        f"{len(baseline):,}"
    )

    # ========================================================
    # 3. Merge predictions
    # ========================================================

    print(
        "\n[3/7] Merging Baseline / Current predictions..."
    )

    merged = baseline.merge(
        current,
        on="flow_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(baseline):
        raise ValueError(
            "Merged dataset row count does not match "
            "the baseline test set."
        )

    print(
        f"  Merged rows: {len(merged):,}"
    )

    # ========================================================
    # 4. Verify labels
    # ========================================================

    print(
        "\n[4/7] Checking labels and prediction status..."
    )

    # Binary labels must match.
    if not merged[
        "baseline_label"
    ].equals(
        merged["current_label"]
    ):

        raise ValueError(
            "Binary label mismatch detected "
            "between baseline and current model."
        )

    print(
        "  Binary label consistency: PASS"
    )

    # Original labels should also match.
    original_match = (
        merged[
            "baseline_original_label"
        ].fillna("<NA>")
        ==
        merged[
            "current_original_label"
        ].fillna("<NA>")
    ).all()

    if not original_match:

        print(
            "  WARNING: Original label mismatch detected."
        )

    else:

        print(
            "  Original label consistency: PASS"
        )

    # ========================================================
    # 5. Create transition columns
    # ========================================================

    print(
        "\n[5/7] Creating prediction transitions..."
    )

    merged["transition"] = (
        merged["baseline_status"]
        + " -> "
        + merged["current_status"]
    )

    merged["baseline_result"] = (
        merged["baseline_status"]
    )

    merged["current_result"] = (
        merged["current_status"]
    )

    merged["traffic_class"] = (
        merged["baseline_label"].map(
            {
                0: "Benign",
                1: "Attack",
            }
        )
    )

    # --------------------------------------------------------
    # Attack transitions
    # --------------------------------------------------------

    # FN in both models.
    merged["persistent_fn"] = (
        (merged["baseline_status"] == "FN")
        &
        (merged["current_status"] == "FN")
    )

    # Baseline FN -> Current TP.
    merged["baseline_fn_repaired"] = (
        (merged["baseline_status"] == "FN")
        &
        (merged["current_status"] == "TP")
    )

    # Baseline TP -> Current FN.
    merged["current_only_fn"] = (
        (merged["baseline_status"] == "TP")
        &
        (merged["current_status"] == "FN")
    )

    # --------------------------------------------------------
    # Benign transitions
    # --------------------------------------------------------

    # FP in both models.
    merged["persistent_fp"] = (
        (merged["baseline_status"] == "FP")
        &
        (merged["current_status"] == "FP")
    )

    # Baseline FP -> Current TN.
    merged["baseline_fp_repaired"] = (
        (merged["baseline_status"] == "FP")
        &
        (merged["current_status"] == "TN")
    )

    # Baseline TN -> Current FP.
    merged["current_only_fp"] = (
        (merged["baseline_status"] == "TN")
        &
        (merged["current_status"] == "FP")
    )

    # ========================================================
    # Save full transition dataset
    # ========================================================

    merged.to_csv(
        transition_file,
        index=False,
    )

    print(
        "  Full transition file saved:"
    )

    print(
        f"    {transition_file}"
    )

    # ========================================================
    # 6. Generate summaries
    # ========================================================

    print(
        "\n[6/7] Generating transition summaries..."
    )

    transition_summary = (
        merged
        .groupby(
            [
                "traffic_class",
                "transition",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    transition_summary[
        "percentage_within_class"
    ] = (
        transition_summary
        .groupby(
            "traffic_class"
        )["count"]
        .transform(
            lambda x:
            x / x.sum() * 100
        )
    )

    transition_summary = (
        transition_summary
        .sort_values(
            [
                "traffic_class",
                "count",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    transition_summary.to_csv(
        transition_summary_file,
        index=False,
    )

    # --------------------------------------------------------
    # Attack transitions
    # --------------------------------------------------------

    attack_df = merged[
        merged["traffic_class"] == "Attack"
    ].copy()

    attack_summary = (
        attack_df
        .groupby("transition")
        .size()
        .reset_index(
            name="count"
        )
        .sort_values(
            "count",
            ascending=False,
        )
    )

    attack_summary[
        "percentage_of_attack_flows"
    ] = (
        attack_summary["count"]
        / len(attack_df)
        * 100
    )

    attack_summary.to_csv(
        attack_summary_file,
        index=False,
    )

    # --------------------------------------------------------
    # Benign transitions
    # --------------------------------------------------------

    benign_df = merged[
        merged["traffic_class"] == "Benign"
    ].copy()

    benign_summary = (
        benign_df
        .groupby("transition")
        .size()
        .reset_index(
            name="count"
        )
        .sort_values(
            "count",
            ascending=False,
        )
    )

    benign_summary[
        "percentage_of_benign_flows"
    ] = (
        benign_summary["count"]
        / len(benign_df)
        * 100
    )

    benign_summary.to_csv(
        benign_summary_file,
        index=False,
    )

    # --------------------------------------------------------
    # Important groups
    # --------------------------------------------------------

    persistent_fn = merged[
        merged["persistent_fn"]
    ].copy()

    persistent_fp = merged[
        merged["persistent_fp"]
    ].copy()

    baseline_fn_repaired = merged[
        merged["baseline_fn_repaired"]
    ].copy()

    baseline_fp_repaired = merged[
        merged["baseline_fp_repaired"]
    ].copy()

    current_only_fn = merged[
        merged["current_only_fn"]
    ].copy()

    current_only_fp = merged[
        merged["current_only_fp"]
    ].copy()

    persistent_fn.to_csv(
        persistent_fn_file,
        index=False,
    )

    persistent_fp.to_csv(
        persistent_fp_file,
        index=False,
    )

    baseline_fn_repaired.to_csv(
        baseline_fn_repaired_file,
        index=False,
    )

    baseline_fp_repaired.to_csv(
        baseline_fp_repaired_file,
        index=False,
    )

    current_only_fn.to_csv(
        current_only_fn_file,
        index=False,
    )

    current_only_fp.to_csv(
        current_only_fp_file,
        index=False,
    )

    # ========================================================
    # 7. Detailed summary
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "PREDICTION TRANSITION ANALYSIS"
    )

    print(
        "=" * 70
    )

    print("\nTest set:")

    print(
        f"  Total flows: {len(merged):,}"
    )

    print(
        f"  Attack: {len(attack_df):,}"
    )

    print(
        f"  Benign: {len(benign_df):,}"
    )

    # ========================================================
    # Attack analysis
    # ========================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "ATTACK FLOW ANALYSIS"
    )

    print(
        "-" * 70
    )

    print(
        "\n  Persistent FN "
        "(FN -> FN): "
        f"{len(persistent_fn):,}"
    )

    print(
        "  Baseline FN repaired "
        "(FN -> TP): "
        f"{len(baseline_fn_repaired):,}"
    )

    print(
        "  Current-only FN "
        "(TP -> FN): "
        f"{len(current_only_fn):,}"
    )

    # ========================================================
    # Benign analysis
    # ========================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "BENIGN FLOW ANALYSIS"
    )

    print(
        "-" * 70
    )

    print(
        "\n  Persistent FP "
        "(FP -> FP): "
        f"{len(persistent_fp):,}"
    )

    print(
        "  Baseline FP repaired "
        "(FP -> TN): "
        f"{len(baseline_fp_repaired):,}"
    )

    print(
        "  Current-only FP "
        "(TN -> FP): "
        f"{len(current_only_fp):,}"
    )

    # ========================================================
    # Model summaries
    # ========================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "MODEL ERROR SUMMARY"
    )

    print(
        "-" * 70
    )

    baseline_summary = calculate_model_summary(
        merged,
        "baseline",
    )

    current_summary = calculate_model_summary(
        merged,
        "current",
    )

    model_summary = {
        "baseline": baseline_summary,
        "current": current_summary,
    }

    for name, summary in model_summary.items():

        print(
            f"\n{name.upper()}:"
        )

        print(
            f"  TP: {summary['tp']:,}"
        )

        print(
            f"  TN: {summary['tn']:,}"
        )

        print(
            f"  FP: {summary['fp']:,}"
        )

        print(
            f"  FN: {summary['fn']:,}"
        )

        print(
            f"  Accuracy: "
            f"{summary['accuracy'] * 100:.4f}%"
        )

        print(
            f"  FPR: "
            f"{summary['fpr_percent']:.4f}%"
        )

        print(
            f"  FNR: "
            f"{summary['fnr_percent']:.4f}%"
        )

    # ========================================================
    # Original label analysis
    # ========================================================

    print_original_label_distribution(
        persistent_fn,
        "Persistent FN - Original Label Distribution",
        "baseline_original_label",
    )

    print_original_label_distribution(
        baseline_fn_repaired,
        "Baseline FN Repaired - Original Label Distribution",
        "baseline_original_label",
    )

    print_original_label_distribution(
        current_only_fn,
        "Current-only FN - Original Label Distribution",
        "baseline_original_label",
    )

    print_original_label_distribution(
        persistent_fp,
        "Persistent FP - Original Label Distribution",
        "baseline_original_label",
    )

    print_original_label_distribution(
        baseline_fp_repaired,
        "Baseline FP Repaired - Original Label Distribution",
        "baseline_original_label",
    )

    print_original_label_distribution(
        current_only_fp,
        "Current-only FP - Original Label Distribution",
        "baseline_original_label",
    )

    # ========================================================
    # Probability analysis
    # ========================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "CURRENT MODEL PROBABILITY ANALYSIS"
    )

    print(
        "-" * 70
    )

    probability_groups = {
        "Persistent FN": persistent_fn,
        "Baseline FN Repaired": baseline_fn_repaired,
        "Current-only FN": current_only_fn,
        "Persistent FP": persistent_fp,
        "Baseline FP Repaired": baseline_fp_repaired,
        "Current-only FP": current_only_fp,
    }

    for group_name, df in probability_groups.items():

        print(
            f"\n  {group_name}:"
        )

        if len(df) == 0:

            print(
                "    None"
            )

            continue

        probability = df[
            "current_attack_probability"
        ]

        print(
            f"    Count : {len(df):,}"
        )

        print(
            f"    Mean  : {probability.mean():.4f}"
        )

        print(
            f"    Median: {probability.median():.4f}"
        )

        print(
            f"    Min   : {probability.min():.4f}"
        )

        print(
            f"    Max   : {probability.max():.4f}"
        )

        near_threshold = (
            (probability >= 0.45)
            &
            (probability <= 0.55)
        ).sum()

        print(
            f"    0.45-0.55: "
            f"{near_threshold:,} "
            f"({near_threshold / len(df) * 100:.2f}%)"
        )

    # ========================================================
    # JSON summary
    # ========================================================

    summary = {

        "analysis_name": analysis_name,

        "baseline_prediction_file": str(
            baseline_file
        ),

        "current_prediction_file": str(
            current_file
        ),

        "test_rows": int(
            len(merged)
        ),

        "attack_rows": int(
            len(attack_df)
        ),

        "benign_rows": int(
            len(benign_df)
        ),

        "models": model_summary,

        "important_transition_counts": {

            "persistent_fn":
                int(len(persistent_fn)),

            "baseline_fn_repaired":
                int(len(baseline_fn_repaired)),

            "current_only_fn":
                int(len(current_only_fn)),

            "persistent_fp":
                int(len(persistent_fp)),

            "baseline_fp_repaired":
                int(len(baseline_fp_repaired)),

            "current_only_fp":
                int(len(current_only_fp)),
        },
    }

    with open(
        summary_json_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # ========================================================
    # Files
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FILES SAVED"
    )

    print(
        "=" * 70
    )

    print(
        f"\n  Output directory:\n"
        f"    {output_dir}"
    )

    print(
        f"\n  Full transitions:\n"
        f"    {transition_file}"
    )

    print(
        f"\n  Transition summary:\n"
        f"    {transition_summary_file}"
    )

    print(
        f"\n  Attack transitions:\n"
        f"    {attack_summary_file}"
    )

    print(
        f"\n  Benign transitions:\n"
        f"    {benign_summary_file}"
    )

    print(
        f"\n  Persistent FN:\n"
        f"    {persistent_fn_file}"
    )

    print(
        f"\n  Persistent FP:\n"
        f"    {persistent_fp_file}"
    )

    print(
        f"\n  Baseline FN repaired:\n"
        f"    {baseline_fn_repaired_file}"
    )

    print(
        f"\n  Baseline FP repaired:\n"
        f"    {baseline_fp_repaired_file}"
    )

    print(
        f"\n  Current-only FN:\n"
        f"    {current_only_fn_file}"
    )

    print(
        f"\n  Current-only FP:\n"
        f"    {current_only_fp_file}"
    )

    print(
        f"\n  JSON summary:\n"
        f"    {summary_json_file}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()