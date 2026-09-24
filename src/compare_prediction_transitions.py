from pathlib import Path
import json

import pandas as pd

import config


# ============================================================
# Configuration
# ============================================================

TEST_FILE = config.PROCESSED_DIR / "split" / "test_dataset.csv"

ERROR_ANALYSIS_DIR = config.MODELS_DIR / "error_analysis"

MODEL_DIRS = {
    "baseline": ERROR_ANALYSIS_DIR,
    "v1": ERROR_ANALYSIS_DIR / "v1",
    "v2": ERROR_ANALYSIS_DIR / "v2",
}

OUTPUT_DIR = ERROR_ANALYSIS_DIR / "comparison"

TRANSITION_FILE = OUTPUT_DIR / "prediction_transitions.csv"
TRANSITION_SUMMARY_FILE = OUTPUT_DIR / "transition_summary.csv"
ATTACK_SUMMARY_FILE = OUTPUT_DIR / "attack_transition_summary.csv"
BENIGN_SUMMARY_FILE = OUTPUT_DIR / "benign_transition_summary.csv"
PERSISTENT_FN_FILE = OUTPUT_DIR / "persistent_fn.csv"
PERSISTENT_FP_FILE = OUTPUT_DIR / "persistent_fp.csv"
V2_ONLY_FN_FILE = OUTPUT_DIR / "v2_only_fn.csv"
V2_ONLY_FP_FILE = OUTPUT_DIR / "v2_only_fp.csv"
V1_REPAIRED_V2_REGRESSED_FN_FILE = (
    OUTPUT_DIR / "v1_repaired_v2_regressed_fn.csv"
)
V1_REPAIRED_V2_REGRESSED_FP_FILE = (
    OUTPUT_DIR / "v1_repaired_v2_regressed_fp.csv"
)
SUMMARY_JSON_FILE = OUTPUT_DIR / "comparison_summary.json"


# ============================================================
# Helper functions
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


def load_prediction_file(name, directory):
    """
    Load one model's prediction_results.csv.
    """
    file_path = directory / "prediction_results.csv"

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
            f"{name}: missing required columns: {sorted(missing)}"
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
        duplicate_count = int(df["flow_id"].duplicated().sum())

        raise ValueError(
            f"{name}: duplicate flow_id detected: {duplicate_count}"
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


def check_same_test_set(dfs):
    """
    Confirm all three models evaluated exactly the same flow IDs.
    """
    names = list(dfs.keys())

    base_ids = set(dfs[names[0]]["flow_id"])

    for name in names[1:]:
        current_ids = set(dfs[name]["flow_id"])

        if current_ids != base_ids:
            missing = len(base_ids - current_ids)
            extra = len(current_ids - base_ids)

            raise ValueError(
                f"Test-set mismatch between {names[0]} and {name}.\n"
                f"  Missing flow IDs: {missing}\n"
                f"  Extra flow IDs   : {extra}"
            )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("NetworkIDS - Baseline / V1 / V2 Prediction Transition Analysis")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Load prediction results
    # --------------------------------------------------------

    print("\n[1/7] Loading prediction results...")

    predictions = {}

    for name, directory in MODEL_DIRS.items():
        print(f"  Loading {name}...")

        predictions[name] = load_prediction_file(
            name,
            directory
        )

        print(
            f"    Rows: {len(predictions[name]):,}"
        )

    # --------------------------------------------------------
    # 2. Verify same test set
    # --------------------------------------------------------

    print("\n[2/7] Checking test-set consistency...")

    check_same_test_set(predictions)

    print("  Flow ID overlap check: PASS")
    print(
        f"  Common test flows: "
        f"{len(predictions['baseline']):,}"
    )

    # --------------------------------------------------------
    # 3. Merge predictions
    # --------------------------------------------------------

    print("\n[3/7] Merging Baseline / V1 / V2 predictions...")

    merged = predictions["baseline"].copy()

    merged = merged.merge(
        predictions["v1"],
        on="flow_id",
        how="inner",
        validate="one_to_one",
    )

    merged = merged.merge(
        predictions["v2"],
        on="flow_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(predictions["baseline"]):
        raise ValueError(
            "Merged dataset row count does not match test set."
        )

    print(
        f"  Merged rows: {len(merged):,}"
    )

    # --------------------------------------------------------
    # 4. Verify labels are identical
    # --------------------------------------------------------

    print("\n[4/7] Checking labels and prediction status...")

    # Label must be the same across all models.
    if not (
        merged["baseline_label"].equals(
            merged["v1_label"]
        )
        and
        merged["baseline_label"].equals(
            merged["v2_label"]
        )
    ):
        raise ValueError(
            "Label mismatch detected between model outputs."
        )

    # Original labels should also normally match.
    original_match_v1 = (
        merged["baseline_original_label"].fillna("<NA>")
        ==
        merged["v1_original_label"].fillna("<NA>")
    ).all()

    original_match_v2 = (
        merged["baseline_original_label"].fillna("<NA>")
        ==
        merged["v2_original_label"].fillna("<NA>")
    ).all()

    if not original_match_v1 or not original_match_v2:
        print(
            "  WARNING: Original label mismatch detected."
        )
    else:
        print(
            "  Original label consistency: PASS"
        )

    print("  Binary label consistency: PASS")

    # --------------------------------------------------------
    # 5. Create transition columns
    # --------------------------------------------------------

    print("\n[5/7] Creating prediction transitions...")

    merged["transition"] = (
        merged["baseline_status"]
        + " -> "
        + merged["v1_status"]
        + " -> "
        + merged["v2_status"]
    )

    # More readable transition fields.
    merged["baseline_result"] = merged["baseline_status"]
    merged["v1_result"] = merged["v1_status"]
    merged["v2_result"] = merged["v2_status"]

    # Attack / Benign category.
    merged["traffic_class"] = merged["baseline_label"].map(
        {
            0: "Benign",
            1: "Attack",
        }
    )

    # --------------------------------------------------------
    # Important transition flags
    # --------------------------------------------------------

    # Attack:
    # FN in all three models.
    merged["persistent_fn"] = (
        (merged["baseline_status"] == "FN")
        &
        (merged["v1_status"] == "FN")
        &
        (merged["v2_status"] == "FN")
    )

    # Baseline FN -> V1 TP -> V2 FN
    merged["v1_repaired_v2_regressed_fn"] = (
        (merged["baseline_status"] == "FN")
        &
        (merged["v1_status"] == "TP")
        &
        (merged["v2_status"] == "FN")
    )

    # Baseline TP -> V1 TP -> V2 FN
    merged["v2_only_fn"] = (
        (merged["baseline_status"] == "TP")
        &
        (merged["v1_status"] == "TP")
        &
        (merged["v2_status"] == "FN")
    )

    # Baseline FN -> V1 TP -> V2 TP
    merged["fn_repaired_and_kept"] = (
        (merged["baseline_status"] == "FN")
        &
        (merged["v1_status"] == "TP")
        &
        (merged["v2_status"] == "TP")
    )

    # Benign:
    # FP in all three models.
    merged["persistent_fp"] = (
        (merged["baseline_status"] == "FP")
        &
        (merged["v1_status"] == "FP")
        &
        (merged["v2_status"] == "FP")
    )

    # Baseline FP -> V1 TN -> V2 FP
    merged["v1_repaired_v2_regressed_fp"] = (
        (merged["baseline_status"] == "FP")
        &
        (merged["v1_status"] == "TN")
        &
        (merged["v2_status"] == "FP")
    )

    # Baseline TN -> V1 TN -> V2 FP
    merged["v2_only_fp"] = (
        (merged["baseline_status"] == "TN")
        &
        (merged["v1_status"] == "TN")
        &
        (merged["v2_status"] == "FP")
    )

    # Baseline FP -> V1 TN -> V2 TN
    merged["fp_repaired_and_kept"] = (
        (merged["baseline_status"] == "FP")
        &
        (merged["v1_status"] == "TN")
        &
        (merged["v2_status"] == "TN")
    )

    # --------------------------------------------------------
    # Save full transition dataset
    # --------------------------------------------------------

    merged.to_csv(
        TRANSITION_FILE,
        index=False,
    )

    print(
        f"  Full transition file saved:\n"
        f"    {TRANSITION_FILE}"
    )

    # --------------------------------------------------------
    # 6. Transition summaries
    # --------------------------------------------------------

    print("\n[6/7] Generating transition summaries...")

    transition_summary = (
        merged
        .groupby(
            ["traffic_class", "transition"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )

    transition_summary["percentage_within_class"] = (
        transition_summary
        .groupby("traffic_class")["count"]
        .transform(
            lambda x: x / x.sum() * 100
        )
    )

    transition_summary = transition_summary.sort_values(
        ["traffic_class", "count"],
        ascending=[True, False],
    )

    transition_summary.to_csv(
        TRANSITION_SUMMARY_FILE,
        index=False,
    )

    # Attack-only transitions.
    attack_df = merged[
        merged["traffic_class"] == "Attack"
    ].copy()

    attack_summary = (
        attack_df
        .groupby("transition")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    attack_summary["percentage_of_attack_flows"] = (
        attack_summary["count"]
        / len(attack_df)
        * 100
    )

    attack_summary.to_csv(
        ATTACK_SUMMARY_FILE,
        index=False,
    )

    # Benign-only transitions.
    benign_df = merged[
        merged["traffic_class"] == "Benign"
    ].copy()

    benign_summary = (
        benign_df
        .groupby("transition")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    benign_summary["percentage_of_benign_flows"] = (
        benign_summary["count"]
        / len(benign_df)
        * 100
    )

    benign_summary.to_csv(
        BENIGN_SUMMARY_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Special groups
    # --------------------------------------------------------

    persistent_fn = merged[
        merged["persistent_fn"]
    ].copy()

    persistent_fp = merged[
        merged["persistent_fp"]
    ].copy()

    v2_only_fn = merged[
        merged["v2_only_fn"]
    ].copy()

    v2_only_fp = merged[
        merged["v2_only_fp"]
    ].copy()

    v1_repaired_v2_regressed_fn = merged[
        merged["v1_repaired_v2_regressed_fn"]
    ].copy()

    v1_repaired_v2_regressed_fp = merged[
        merged["v1_repaired_v2_regressed_fp"]
    ].copy()

    persistent_fn.to_csv(
        PERSISTENT_FN_FILE,
        index=False,
    )

    persistent_fp.to_csv(
        PERSISTENT_FP_FILE,
        index=False,
    )

    v2_only_fn.to_csv(
        V2_ONLY_FN_FILE,
        index=False,
    )

    v2_only_fp.to_csv(
        V2_ONLY_FP_FILE,
        index=False,
    )

    v1_repaired_v2_regressed_fn.to_csv(
        V1_REPAIRED_V2_REGRESSED_FN_FILE,
        index=False,
    )

    v1_repaired_v2_regressed_fp.to_csv(
        V1_REPAIRED_V2_REGRESSED_FP_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # 7. Detailed summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREDICTION TRANSITION ANALYSIS")
    print("=" * 70)

    print("\nTest set:")
    print(
        f"  Total flows: {len(merged):,}"
    )

    print(
        f"  Attack: "
        f"{len(attack_df):,}"
    )

    print(
        f"  Benign: "
        f"{len(benign_df):,}"
    )

    # --------------------------------------------------------
    # Attack transitions
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("ATTACK FLOW ANALYSIS")
    print("-" * 70)

    print(
        f"\n  Persistent FN "
        f"(FN -> FN -> FN): "
        f"{len(persistent_fn):,}"
    )

    print(
        f"  V1 repaired, V2 regressed "
        f"(FN -> TP -> FN): "
        f"{len(v1_repaired_v2_regressed_fn):,}"
    )

    print(
        f"  V2-only FN "
        f"(TP -> TP -> FN): "
        f"{len(v2_only_fn):,}"
    )

    print(
        f"  FN repaired and kept "
        f"(FN -> TP -> TP): "
        f"{len(merged[merged['fn_repaired_and_kept']]):,}"
    )

    # --------------------------------------------------------
    # Benign transitions
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("BENIGN FLOW ANALYSIS")
    print("-" * 70)

    print(
        f"\n  Persistent FP "
        f"(FP -> FP -> FP): "
        f"{len(persistent_fp):,}"
    )

    print(
        f"  V1 repaired, V2 regressed "
        f"(FP -> TN -> FP): "
        f"{len(v1_repaired_v2_regressed_fp):,}"
    )

    print(
        f"  V2-only FP "
        f"(TN -> TN -> FP): "
        f"{len(v2_only_fp):,}"
    )

    print(
        f"  FP repaired and kept "
        f"(FP -> TN -> TN): "
        f"{len(merged[merged['fp_repaired_and_kept']]):,}"
    )

    # --------------------------------------------------------
    # Standard model metrics
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("MODEL ERROR SUMMARY")
    print("-" * 70)

    model_summary = {}

    for name in ["baseline", "v1", "v2"]:

        status_counts = (
            merged[f"{name}_status"]
            .value_counts()
            .to_dict()
        )

        tp = int(status_counts.get("TP", 0))
        tn = int(status_counts.get("TN", 0))
        fp = int(status_counts.get("FP", 0))
        fn = int(status_counts.get("FN", 0))

        total = tp + tn + fp + fn

        model_summary[name] = {
            "total": total,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "accuracy": (tp + tn) / total,
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

        print(
            f"\n{name.upper()}:"
        )

        print(
            f"  TP: {tp:,}"
        )
        print(
            f"  TN: {tn:,}"
        )
        print(
            f"  FP: {fp:,}"
        )
        print(
            f"  FN: {fn:,}"
        )

        print(
            f"  Accuracy: "
            f"{model_summary[name]['accuracy'] * 100:.4f}%"
        )

        print(
            f"  FPR: "
            f"{model_summary[name]['fpr_percent']:.4f}%"
        )

        print(
            f"  FNR: "
            f"{model_summary[name]['fnr_percent']:.4f}%"
        )

    # --------------------------------------------------------
    # Original label analysis for important error groups
    # --------------------------------------------------------

    def print_original_label_distribution(
        df,
        title,
    ):
        print(f"\n{title}")

        if len(df) == 0:
            print("  None")
            return

        counts = (
            df["baseline_original_label"]
            .fillna("<NA>")
            .value_counts()
        )

        for label, count in counts.items():
            percentage = count / len(df) * 100

            print(
                f"  {label}: "
                f"{count:,} "
                f"({percentage:.2f}%)"
            )

    print_original_label_distribution(
        persistent_fn,
        "Persistent FN - Original Label Distribution",
    )

    print_original_label_distribution(
        v1_repaired_v2_regressed_fn,
        "V1 Repaired / V2 Regressed FN - Original Label Distribution",
    )

    print_original_label_distribution(
        v2_only_fn,
        "V2-only FN - Original Label Distribution",
    )

    print_original_label_distribution(
        persistent_fp,
        "Persistent FP - Original Label Distribution",
    )

    print_original_label_distribution(
        v1_repaired_v2_regressed_fp,
        "V1 Repaired / V2 Regressed FP - Original Label Distribution",
    )

    print_original_label_distribution(
        v2_only_fp,
        "V2-only FP - Original Label Distribution",
    )

    # --------------------------------------------------------
    # Probability summaries for important error groups
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("ATTACK PROBABILITY ANALYSIS")
    print("-" * 70)

    probability_groups = {
        "Persistent FN": persistent_fn,
        "V1 repaired / V2 regressed FN":
            v1_repaired_v2_regressed_fn,
        "V2-only FN": v2_only_fn,
        "Persistent FP": persistent_fp,
        "V1 repaired / V2 regressed FP":
            v1_repaired_v2_regressed_fp,
        "V2-only FP": v2_only_fp,
    }

    for group_name, df in probability_groups.items():

        print(f"\n  {group_name}:")

        if len(df) == 0:
            print("    None")
            continue

        probability = df[
            "v2_attack_probability"
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

    # --------------------------------------------------------
    # Save JSON summary
    # --------------------------------------------------------

    summary = {
        "test_rows": int(len(merged)),
        "attack_rows": int(len(attack_df)),
        "benign_rows": int(len(benign_df)),
        "models": model_summary,
        "important_transition_counts": {
            "persistent_fn": int(len(persistent_fn)),
            "v1_repaired_v2_regressed_fn":
                int(len(v1_repaired_v2_regressed_fn)),
            "v2_only_fn":
                int(len(v2_only_fn)),
            "fn_repaired_and_kept":
                int(
                    merged["fn_repaired_and_kept"].sum()
                ),
            "persistent_fp":
                int(len(persistent_fp)),
            "v1_repaired_v2_regressed_fp":
                int(len(v1_repaired_v2_regressed_fp)),
            "v2_only_fp":
                int(len(v2_only_fp)),
            "fp_repaired_and_kept":
                int(
                    merged["fp_repaired_and_kept"].sum()
                ),
        },
    }

    with open(
        SUMMARY_JSON_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"\n  Full transitions:\n"
        f"    {TRANSITION_FILE}"
    )

    print(
        f"\n  Transition summary:\n"
        f"    {TRANSITION_SUMMARY_FILE}"
    )

    print(
        f"\n  Attack transitions:\n"
        f"    {ATTACK_SUMMARY_FILE}"
    )

    print(
        f"\n  Benign transitions:\n"
        f"    {BENIGN_SUMMARY_FILE}"
    )

    print(
        f"\n  Persistent FN:\n"
        f"    {PERSISTENT_FN_FILE}"
    )

    print(
        f"\n  Persistent FP:\n"
        f"    {PERSISTENT_FP_FILE}"
    )

    print(
        f"\n  V2-only FN:\n"
        f"    {V2_ONLY_FN_FILE}"
    )

    print(
        f"\n  V2-only FP:\n"
        f"    {V2_ONLY_FP_FILE}"
    )

    print(
        f"\n  V1 repaired / V2 regressed FN:\n"
        f"    {V1_REPAIRED_V2_REGRESSED_FN_FILE}"
    )

    print(
        f"\n  V1 repaired / V2 regressed FP:\n"
        f"    {V1_REPAIRED_V2_REGRESSED_FP_FILE}"
    )

    print(
        f"\n  JSON summary:\n"
        f"    {SUMMARY_JSON_FILE}"
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()