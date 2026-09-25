# src/analyze_rst_overlap.py

from pathlib import Path
import argparse
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTION_FILE = (
    PROJECT_ROOT
    / "models"
    / "error_analysis"
    / "v2_test1"
    / "prediction_results.csv"
)
DEFAULT_ROBUSTNESS_FILE = (
    PROJECT_ROOT
    / "models"
    / "robustness"
    / "robustness_results_robust_v2.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "error_analysis"
    / "rst_overlap"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze overlap between Current V2 clean-test errors "
                    "and RST-zeroing robustness evasions."
    )

    parser.add_argument(
        "--prediction",
        type=str,
        default=str(DEFAULT_PREDICTION_FILE),
        help="Current V2 clean-test prediction_results.csv",
    )

    parser.add_argument(
        "--robustness",
        type=str,
        default=str(DEFAULT_ROBUSTNESS_FILE),
        help="Current V2 robustness_results_robust_v2.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for overlap analysis outputs.",
    )

    return parser.parse_args()


def detect_column(df, candidates, description):
    """
    Return the first matching column from candidates.
    """
    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not find {description} column.\n"
        f"Tried: {candidates}\n"
        f"Available columns:\n{list(df.columns)}"
    )


def main():
    args = parse_args()

    prediction_file = Path(args.prediction)
    robustness_file = Path(args.robustness)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RST Robustness ↔ Clean-Test Error Overlap Analysis")
    print("=" * 80)

    print(f"Prediction file : {prediction_file}")
    print(f"Robustness file : {robustness_file}")
    print(f"Output directory: {output_dir}")
    print()

    if not prediction_file.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {prediction_file}"
        )

    if not robustness_file.exists():
        raise FileNotFoundError(
            f"Robustness file not found: {robustness_file}"
        )

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------

    pred_df = pd.read_csv(prediction_file)
    robust_df = pd.read_csv(robustness_file)

    print(f"Prediction rows : {len(pred_df):,}")
    print(f"Prediction cols : {len(pred_df.columns)}")
    print(f"Robustness rows : {len(robust_df):,}")
    print(f"Robustness cols : {len(robust_df.columns)}")
    print()

    print("Prediction columns:")
    print(list(pred_df.columns))
    print()

    print("Robustness columns:")
    print(list(robust_df.columns))
    print()

    # ------------------------------------------------------------------
    # 2. Detect important columns
    # ------------------------------------------------------------------

    pred_flow_col = detect_column(
        pred_df,
        ["flow_id", "Flow ID"],
        "prediction flow ID",
    )

    robust_flow_col = detect_column(
        robust_df,
        ["flow_id", "Flow ID"],
        "robustness flow ID",
    )

    print(f"Prediction flow ID column: {pred_flow_col}")
    print(f"Robustness flow ID column: {robust_flow_col}")
    print()

    # ------------------------------------------------------------------
    # 3. Identify clean-test errors
    # ------------------------------------------------------------------

    required_prediction_cols = [
        pred_flow_col,
        "actual",
        "predicted",
    ]

    missing_prediction_cols = [
        c for c in required_prediction_cols
        if c not in pred_df.columns
    ]

    if missing_prediction_cols:
        raise ValueError(
            "Missing required prediction columns: "
            + ", ".join(missing_prediction_cols)
        )

    pred_df["actual"] = pd.to_numeric(pred_df["actual"], errors="coerce")
    pred_df["predicted"] = pd.to_numeric(
        pred_df["predicted"],
        errors="coerce",
    )

    clean_fn = pred_df[
        (pred_df["actual"] == 1)
        & (pred_df["predicted"] == 0)
    ].copy()

    clean_fp = pred_df[
        (pred_df["actual"] == 0)
        & (pred_df["predicted"] == 1)
    ].copy()

    clean_attack_correct = pred_df[
        (pred_df["actual"] == 1)
        & (pred_df["predicted"] == 1)
    ].copy()

    clean_benign_correct = pred_df[
        (pred_df["actual"] == 0)
        & (pred_df["predicted"] == 0)
    ].copy()

    print("-" * 80)
    print("Clean-test prediction groups")
    print("-" * 80)

    print(f"Attack correctly detected : {len(clean_attack_correct):,}")
    print(f"Attack false negatives    : {len(clean_fn):,}")
    print(f"Benign correctly cleared  : {len(clean_benign_correct):,}")
    print(f"Benign false positives    : {len(clean_fp):,}")
    print()

    # ------------------------------------------------------------------
    # 4. Identify RST-zeroing scenario
    # ------------------------------------------------------------------

    if "scenario" in robust_df.columns:
        rst_df = robust_df[
            robust_df["scenario"].astype(str)
            == "rst_count_zeroed"
        ].copy()

    elif "test_name" in robust_df.columns:
        rst_df = robust_df[
            robust_df["test_name"].astype(str)
            == "rst_count_zeroed"
        ].copy()

    else:
        raise ValueError(
            "Could not find a scenario/test_name column in robustness results."
        )

    print("-" * 80)
    print("RST-zeroing robustness data")
    print("-" * 80)

    print(f"RST-zeroing rows: {len(rst_df):,}")
    print()

    # ------------------------------------------------------------------
    # 5. Detect robustness prediction columns
    # ------------------------------------------------------------------

    print("RST-zeroing columns:")
    print(list(rst_df.columns))
    print()

    # Different versions of robustness_evaluation.py may use different
    # names. Detect them rather than hardcoding one schema.

    baseline_pred_col = None
    perturbed_pred_col = None

    for candidate in [
        "baseline_prediction",
        "baseline_predicted",
        "baseline_pred",
        "original_prediction",
    ]:
        if candidate in rst_df.columns:
            baseline_pred_col = candidate
            break

    for candidate in [
        "perturbed_prediction",
        "perturbed_predicted",
        "perturbed_pred",
        "new_prediction",
    ]:
        if candidate in rst_df.columns:
            perturbed_pred_col = candidate
            break

    if baseline_pred_col is None or perturbed_pred_col is None:
        raise ValueError(
            "Could not identify baseline/perturbed prediction columns.\n"
            f"Available columns: {list(rst_df.columns)}"
        )

    print(f"Baseline prediction column : {baseline_pred_col}")
    print(f"Perturbed prediction column: {perturbed_pred_col}")
    print()

    rst_df[baseline_pred_col] = pd.to_numeric(
        rst_df[baseline_pred_col],
        errors="coerce",
    )

    rst_df[perturbed_pred_col] = pd.to_numeric(
        rst_df[perturbed_pred_col],
        errors="coerce",
    )

    # ------------------------------------------------------------------
    # 6. Identify Attack → Benign RST evasions
    # ------------------------------------------------------------------

    if "actual" in rst_df.columns:
        rst_df["actual"] = pd.to_numeric(
            rst_df["actual"],
            errors="coerce",
        )

        rst_attack = rst_df[rst_df["actual"] == 1].copy()

    else:
        # The robustness evaluator should normally contain attack samples.
        # If actual labels are absent, use baseline prediction as a fallback.
        rst_attack = rst_df[
            rst_df[baseline_pred_col] == 1
        ].copy()

    rst_evasion = rst_attack[
        (rst_attack[baseline_pred_col] == 1)
        & (rst_attack[perturbed_pred_col] == 0)
    ].copy()

    print("-" * 80)
    print("RST-zeroing attack evasion")
    print("-" * 80)

    print(f"Attack samples evaluated : {len(rst_attack):,}")
    print(f"Attack → Benign evasions : {len(rst_evasion):,}")

    if len(rst_attack) > 0:
        evasion_rate = (
            len(rst_evasion) / len(rst_attack) * 100
        )
    else:
        evasion_rate = 0.0

    print(f"Attack Evasion Rate       : {evasion_rate:.4f}%")
    print()

    # ------------------------------------------------------------------
    # 7. Calculate overlap
    # ------------------------------------------------------------------

    clean_fn_ids = set(
        clean_fn[pred_flow_col]
        .dropna()
        .astype(str)
    )

    rst_evasion_ids = set(
        rst_evasion[robust_flow_col]
        .dropna()
        .astype(str)
    )

    overlap_ids = clean_fn_ids & rst_evasion_ids

    print("-" * 80)
    print("Overlap")
    print("-" * 80)

    print(f"Clean-test FN                  : {len(clean_fn_ids):,}")
    print(f"RST-zeroing evasion            : {len(rst_evasion_ids):,}")
    print(f"Overlap                        : {len(overlap_ids):,}")

    if len(rst_evasion_ids) > 0:
        overlap_of_rst = (
            len(overlap_ids) / len(rst_evasion_ids) * 100
        )
    else:
        overlap_of_rst = 0.0

    if len(clean_fn_ids) > 0:
        overlap_of_fn = (
            len(overlap_ids) / len(clean_fn_ids) * 100
        )
    else:
        overlap_of_fn = 0.0

    print(
        f"Overlap among RST evasions    : "
        f"{overlap_of_rst:.4f}%"
    )

    print(
        f"Overlap among clean-test FN    : "
        f"{overlap_of_fn:.4f}%"
    )
    print()

    # ------------------------------------------------------------------
    # 8. Save overlap rows
    # ------------------------------------------------------------------

    overlap_pred = clean_fn[
        clean_fn[pred_flow_col].astype(str).isin(overlap_ids)
    ].copy()

    overlap_robust = rst_evasion[
        rst_evasion[robust_flow_col].astype(str).isin(overlap_ids)
    ].copy()

    clean_fn.to_csv(
        output_dir / "clean_false_negatives.csv",
        index=False,
    )

    rst_evasion.to_csv(
        output_dir / "rst_evasion_cases.csv",
        index=False,
    )

    overlap_pred.to_csv(
        output_dir / "rst_clean_fn_overlap_prediction.csv",
        index=False,
    )

    overlap_robust.to_csv(
        output_dir / "rst_clean_fn_overlap_robustness.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 9. Feature profile comparison if columns are available
    # ------------------------------------------------------------------

    profile_columns = [
        "rst_count",
        "syn_count",
        "ack_count",
        "fin_count",
        "psh_count",
        "packet_count",
        "flow_duration",
        "dst_port",
        "src_port",
        "ttl_mean",
        "tcp_window_mean",
        "tcp_header_length_mean",
    ]

    available_profile_columns = [
        c for c in profile_columns
        if c in clean_fn.columns
    ]

    profile_rows = []

    if available_profile_columns:

        groups = {
            "clean_fn": clean_fn,
            "rst_evasion": rst_evasion,
            "overlap": overlap_pred,
        }

        for group_name, group_df in groups.items():
            row = {
                "group": group_name,
                "count": len(group_df),
            }

            for col in available_profile_columns:
                values = pd.to_numeric(
                    group_df[col],
                    errors="coerce",
                )

                row[f"{col}_mean"] = values.mean()
                row[f"{col}_median"] = values.median()

            profile_rows.append(row)

        profile_df = pd.DataFrame(profile_rows)

        profile_df.to_csv(
            output_dir / "rst_overlap_feature_profile.csv",
            index=False,
        )

    # ------------------------------------------------------------------
    # 10. Summary JSON
    # ------------------------------------------------------------------

    summary = {
        "prediction_file": str(prediction_file),
        "robustness_file": str(robustness_file),
        "clean_test_rows": int(len(pred_df)),
        "clean_fn": int(len(clean_fn_ids)),
        "clean_fp": int(len(clean_fp)),
        "rst_attack_samples": int(len(rst_attack)),
        "rst_evasion": int(len(rst_evasion_ids)),
        "rst_attack_evasion_rate_percent": float(evasion_rate),
        "overlap_count": int(len(overlap_ids)),
        "overlap_of_rst_evasions_percent": float(overlap_of_rst),
        "overlap_of_clean_fn_percent": float(overlap_of_fn),
    }

    with open(
        output_dir / "rst_overlap_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------
    # 11. Final summary
    # ------------------------------------------------------------------

    print("=" * 80)
    print("RESULT SUMMARY")
    print("=" * 80)

    print(f"Clean-test FN                 : {len(clean_fn_ids):,}")
    print(f"RST-zeroing evasions          : {len(rst_evasion_ids):,}")
    print(f"Overlap                       : {len(overlap_ids):,}")
    print(
        f"Overlap / RST evasions        : "
        f"{overlap_of_rst:.4f}%"
    )
    print(
        f"Overlap / clean-test FN       : "
        f"{overlap_of_fn:.4f}%"
    )

    print()
    print(f"Outputs saved to:")
    print(output_dir)
    print()
    print("PASS: RST overlap analysis completed.")


if __name__ == "__main__":
    main()