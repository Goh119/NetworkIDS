from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd

import config


# ============================================================
# Configuration
# ============================================================

TEST_FILE = config.PROCESSED_DIR / "split" / "test_dataset.csv"

DEFAULT_MODEL_FILE = (
    config.MODELS_DIR / "random_forest_robust_v2.pkl"
)

DEFAULT_PREDICTION_FILE = (
    config.MODELS_DIR
    / "error_analysis"
    / "v2_test1"
    / "prediction_results.csv"
)

DEFAULT_OUTPUT_DIR = (
    config.MODELS_DIR
    / "error_analysis"
    / "rst_overlap"
)

RANDOM_STATE = 42
SAMPLE_SIZE_PER_CLASS = 2000


# ============================================================
# Utility
# ============================================================

def load_model(model_path):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    print(f"Loading model: {model_path}")

    return joblib.load(model_path)


def predict(model, X):
    """
    Return predictions and Attack probabilities.
    """

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)

    attack_class_index = list(model.classes_).index(
        config.LABEL_ATTACK
    )

    attack_probabilities = probabilities[
        :, attack_class_index
    ]

    return predictions, attack_probabilities


# ============================================================
# RST perturbation
# ============================================================

def rst_count_zeroed(X):
    """
    Set rst_count to zero.
    """

    X_new = X.copy()

    if "rst_count" in X_new.columns:
        X_new["rst_count"] = 0

    return X_new


# ============================================================
# Reproduce original robustness sampling
# ============================================================

def select_original_robustness_samples(
    df,
    model,
    sample_size,
    random_state,
):
    """
    Reproduce the exact sampling logic used by the original
    robustness_evaluation.py.

    IMPORTANT:
    - Same test dataset
    - Same model
    - Same random state
    - Same Attack -> Benign sampling order
    """

    X = df[config.ML_FEATURES]
    y = df[config.LABEL_COLUMN].astype(int)

    baseline_predictions, baseline_probabilities = predict(
        model,
        X,
    )

    correct_attack_mask = (
        (y == config.LABEL_ATTACK)
        & (baseline_predictions == config.LABEL_ATTACK)
    )

    correct_benign_mask = (
        (y == config.LABEL_BENIGN)
        & (baseline_predictions == config.LABEL_BENIGN)
    )

    attack_indices = df.index[
        correct_attack_mask
    ].to_numpy()

    benign_indices = df.index[
        correct_benign_mask
    ].to_numpy()

    if len(attack_indices) < sample_size:
        raise ValueError(
            "Not enough correctly classified Attack samples."
        )

    if len(benign_indices) < sample_size:
        raise ValueError(
            "Not enough correctly classified Benign samples."
        )

    # EXACT SAME RNG LOGIC AS ORIGINAL SCRIPT
    rng = np.random.default_rng(random_state)

    selected_attack_indices = rng.choice(
        attack_indices,
        size=sample_size,
        replace=False,
    )

    selected_benign_indices = rng.choice(
        benign_indices,
        size=sample_size,
        replace=False,
    )

    return (
        selected_attack_indices,
        selected_benign_indices,
        baseline_predictions,
        baseline_probabilities,
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Analyze overlap between Current V2 clean-test "
            "false negatives and RST-zeroing robustness "
            "evasion cases."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_FILE),
        help="Current V2 robust Random Forest model.",
    )

    parser.add_argument(
        "--prediction",
        type=str,
        default=str(DEFAULT_PREDICTION_FILE),
        help="Current V2 clean-test prediction results.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory.",
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    prediction_path = Path(args.prediction)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Header
    # ========================================================

    print("=" * 80)
    print("RST ROBUSTNESS ↔ CLEAN-TEST ERROR OVERLAP")
    print("=" * 80)

    print(f"Test dataset : {TEST_FILE}")
    print(f"Model        : {model_path}")
    print(f"Prediction   : {prediction_path}")
    print(f"Output dir   : {output_dir}")
    print()

    # ========================================================
    # Load test dataset
    # ========================================================

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test dataset not found:\n{TEST_FILE}"
        )

    df = pd.read_csv(TEST_FILE)

    print(
        f"Test dataset: "
        f"{len(df):,} rows x {len(df.columns)} columns"
    )

    # ========================================================
    # Validate
    # ========================================================

    if len(config.ML_FEATURES) != 42:
        raise ValueError(
            f"Expected 42 ML features, "
            f"found {len(config.ML_FEATURES)}."
        )

    required_columns = (
        config.ML_FEATURES
        + config.METADATA_COLUMNS
        + [config.LABEL_COLUMN]
    )

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )

    # ========================================================
    # Load model
    # ========================================================

    model = load_model(model_path)

    if hasattr(model, "n_features_in_"):
        if model.n_features_in_ != 42:
            raise ValueError(
                f"Model expects "
                f"{model.n_features_in_} features, "
                f"not 42."
            )

    print(
        f"Model classes: {list(model.classes_)}"
    )

    # ========================================================
    # Reproduce original robustness sampling
    # ========================================================

    (
        attack_indices,
        benign_indices,
        baseline_predictions,
        baseline_probabilities,
    ) = select_original_robustness_samples(
        df=df,
        model=model,
        sample_size=SAMPLE_SIZE_PER_CLASS,
        random_state=RANDOM_STATE,
    )

    print()
    print("-" * 80)
    print("REPRODUCED ROBUSTNESS SAMPLE")
    print("-" * 80)

    print(
        f"Attack samples selected: "
        f"{len(attack_indices):,}"
    )

    print(
        f"Benign samples selected: "
        f"{len(benign_indices):,}"
    )

    # ========================================================
    # Attack sample information
    # ========================================================

    attack_df = df.loc[
        attack_indices
    ].copy()

    attack_X = attack_df[
        config.ML_FEATURES
    ].copy()

    attack_baseline_predictions = (
        baseline_predictions[
            df.index.get_indexer(attack_indices)
        ]
    )

    attack_baseline_probabilities = (
        baseline_probabilities[
            df.index.get_indexer(attack_indices)
        ]
    )

    # ========================================================
    # Apply RST-zeroing
    # ========================================================

    attack_X_rst = rst_count_zeroed(
        attack_X
    )

    (
        attack_rst_predictions,
        attack_rst_probabilities,
    ) = predict(
        model,
        attack_X_rst,
    )

    # ========================================================
    # Build sample-level result
    # ========================================================

    sample_results = attack_df[
        config.METADATA_COLUMNS
        + [config.LABEL_COLUMN]
    ].copy()

    sample_results = sample_results.rename(
        columns={
            config.LABEL_COLUMN: "actual_label"
        }
    )

    sample_results["baseline_prediction"] = (
        attack_baseline_predictions
    )

    sample_results["baseline_attack_probability"] = (
        attack_baseline_probabilities
    )

    sample_results["rst_prediction"] = (
        attack_rst_predictions
    )

    sample_results["rst_attack_probability"] = (
        attack_rst_probabilities
    )

    sample_results["rst_prediction_changed"] = (
        sample_results["baseline_prediction"]
        != sample_results["rst_prediction"]
    )

    sample_results["rst_evasion"] = (
        (sample_results["actual_label"] == config.LABEL_ATTACK)
        &
        (sample_results["baseline_prediction"] == config.LABEL_ATTACK)
        &
        (sample_results["rst_prediction"] == config.LABEL_BENIGN)
    )

    # Original RST count
    sample_results["original_rst_count"] = (
        attack_df["rst_count"].to_numpy()
    )

    # ========================================================
    # Save all RST sample-level results
    # ========================================================

    sample_results_path = (
        output_dir
        / "rst_zeroing_sample_results.csv"
    )

    sample_results.to_csv(
        sample_results_path,
        index=False,
    )

    # ========================================================
    # Extract evasion cases
    # ========================================================

    rst_evasion = sample_results[
        sample_results["rst_evasion"]
    ].copy()

    rst_evasion_path = (
        output_dir
        / "rst_evasion_cases.csv"
    )

    rst_evasion.to_csv(
        rst_evasion_path,
        index=False,
    )

    # ========================================================
    # RST summary
    # ========================================================

    n_attack = len(sample_results)
    n_evasion = len(rst_evasion)

    evasion_rate = (
        n_evasion / n_attack * 100
        if n_attack > 0
        else 0.0
    )

    print()
    print("-" * 80)
    print("RST-ZEROING RESULT")
    print("-" * 80)

    print(
        f"Attack samples tested : {n_attack:,}"
    )

    print(
        f"RST evasion cases     : {n_evasion:,}"
    )

    print(
        f"Attack Evasion Rate   : "
        f"{evasion_rate:.4f}%"
    )

    # ========================================================
    # Load clean prediction results
    # ========================================================

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Prediction results not found:\n"
            f"{prediction_path}"
        )

    prediction_df = pd.read_csv(
        prediction_path
    )

    required_prediction_columns = [
        "flow_id",
        "label",
        "predicted_label",
        "attack_probability",
    ]

    missing_prediction = [
        col
        for col in required_prediction_columns
        if col not in prediction_df.columns
    ]

    if missing_prediction:
        raise ValueError(
            "Missing prediction columns:\n"
            + "\n".join(missing_prediction)
        )

    # ========================================================
    # Clean-test FN
    # ========================================================

    clean_fn = prediction_df[
        (prediction_df["label"] == config.LABEL_ATTACK)
        &
        (
            prediction_df["predicted_label"]
            == config.LABEL_BENIGN
        )
    ].copy()

    clean_fn_ids = set(
        clean_fn["flow_id"]
        .astype(str)
    )

    rst_evasion_ids = set(
        rst_evasion["flow_id"]
        .astype(str)
    )

    # ========================================================
    # Overlap
    # ========================================================

    overlap_ids = (
        clean_fn_ids
        & rst_evasion_ids
    )

    overlap_prediction = clean_fn[
        clean_fn["flow_id"]
        .astype(str)
        .isin(overlap_ids)
    ].copy()

    overlap_rst = rst_evasion[
        rst_evasion["flow_id"]
        .astype(str)
        .isin(overlap_ids)
    ].copy()

    # ========================================================
    # Save overlap
    # ========================================================

    overlap_prediction_path = (
        output_dir
        / "rst_clean_fn_overlap_prediction.csv"
    )

    overlap_rst_path = (
        output_dir
        / "rst_clean_fn_overlap_robustness.csv"
    )

    overlap_prediction.to_csv(
        overlap_prediction_path,
        index=False,
    )

    overlap_rst.to_csv(
        overlap_rst_path,
        index=False,
    )

    # ========================================================
    # Calculate percentages
    # ========================================================

    overlap_count = len(overlap_ids)

    overlap_of_rst = (
        overlap_count / len(rst_evasion_ids) * 100
        if len(rst_evasion_ids) > 0
        else 0.0
    )

    overlap_of_clean_fn = (
        overlap_count / len(clean_fn_ids) * 100
        if len(clean_fn_ids) > 0
        else 0.0
    )

    # ========================================================
    # Feature comparison
    # ========================================================

    profile_columns = [
        "rst_count",
        "syn_count",
        "ack_count",
        "fin_count",
        "psh_count",
        "flow_packet_count",
        "flow_duration",
        "dst_port",
        "src_port",
        "ttl_mean",
        "tcp_window_mean",
        "tcp_header_length_mean",
    ]

    available_profile_columns = [
        col
        for col in profile_columns
        if col in attack_df.columns
    ]

    profile_rows = []

    groups = {
        "all_rst_tested_attacks": attack_df,
        "rst_evasion": attack_df.loc[
            attack_df.index.isin(
                rst_evasion.index
            )
        ],
        "clean_test_fn": df[
            df["flow_id"]
            .astype(str)
            .isin(clean_fn_ids)
        ],
        "overlap": df[
            df["flow_id"]
            .astype(str)
            .isin(overlap_ids)
        ],
    }

    for group_name, group_df in groups.items():

        row = {
            "group": group_name,
            "count": len(group_df),
        }

        for column in available_profile_columns:

            values = pd.to_numeric(
                group_df[column],
                errors="coerce",
            )

            row[f"{column}_mean"] = (
                values.mean()
            )

            row[f"{column}_median"] = (
                values.median()
            )

        profile_rows.append(row)

    profile_df = pd.DataFrame(
        profile_rows
    )

    profile_path = (
        output_dir
        / "rst_overlap_feature_profile.csv"
    )

    profile_df.to_csv(
        profile_path,
        index=False,
    )

    # ========================================================
    # Summary JSON
    # ========================================================

    summary = {
        "test_rows": int(len(df)),
        "sample_size_per_class": SAMPLE_SIZE_PER_CLASS,
        "random_state": RANDOM_STATE,
        "rst_attack_samples": int(n_attack),
        "rst_evasion_cases": int(n_evasion),
        "rst_attack_evasion_rate_percent": float(
            evasion_rate
        ),
        "clean_test_false_negatives": int(
            len(clean_fn_ids)
        ),
        "overlap_count": int(
            overlap_count
        ),
        "overlap_of_rst_evasions_percent": float(
            overlap_of_rst
        ),
        "overlap_of_clean_test_fn_percent": float(
            overlap_of_clean_fn
        ),
        "model_path": str(model_path),
        "prediction_file": str(prediction_path),
    }

    summary_path = (
        output_dir
        / "rst_overlap_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # ========================================================
    # Final output
    # ========================================================

    print()
    print("=" * 80)
    print("OVERLAP ANALYSIS RESULT")
    print("=" * 80)

    print(
        f"Clean-test FN              : "
        f"{len(clean_fn_ids):,}"
    )

    print(
        f"RST-zeroing evasion       : "
        f"{len(rst_evasion_ids):,}"
    )

    print(
        f"Overlap                   : "
        f"{overlap_count:,}"
    )

    print(
        f"Overlap / RST evasions    : "
        f"{overlap_of_rst:.4f}%"
    )

    print(
        f"Overlap / clean-test FN   : "
        f"{overlap_of_clean_fn:.4f}%"
    )

    print()
    print("Outputs:")
    print(f"  {sample_results_path}")
    print(f"  {rst_evasion_path}")
    print(f"  {overlap_prediction_path}")
    print(f"  {overlap_rst_path}")
    print(f"  {profile_path}")
    print(f"  {summary_path}")

    print()
    print("PASS: RST overlap analysis completed.")


if __name__ == "__main__":
    main()