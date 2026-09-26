from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
)

import config


# ============================================================
# Configuration
# ============================================================

TEST_FILE = (
    config.PROCESSED_DIR
    / "split"
    / "test_dataset.csv"
)

DEFAULT_MODEL_FILE = (
    config.MODELS_DIR
    / "random_forest_robust_v2.pkl"
)

DEFAULT_OUTPUT_DIR = (
    config.MODELS_DIR
    / "feature_dependency"
)

RANDOM_STATE = 42

# Number of test rows used for permutation importance.
# 50,000 gives a good balance between stability and runtime.
PERMUTATION_SAMPLE_SIZE = 50_000

# Number of random repetitions for each permutation.
N_REPEATS = 3


# ============================================================
# Model utilities
# ============================================================

def load_model(model_path):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    print(f"Loading model: {model_path}")

    model = joblib.load(model_path)

    return model


def predict(model, X):
    """
    Return prediction and Attack probability.
    """

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)

    attack_index = list(model.classes_).index(
        config.LABEL_ATTACK
    )

    attack_probability = probabilities[:, attack_index]

    return predictions, attack_probability


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(y_true, predictions, probabilities):
    """
    Calculate the main binary classification metrics.
    """

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),

        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "auc_roc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
    }


# ============================================================
# Feature groups
# ============================================================

def build_feature_groups():
    """
    Define the four project feature groups.

    These groups follow the current 42-feature design.
    """

    groups = {
        "L2": [
            "frame_length_mean",
            "eth_type",
            "vlan_id",
            "vlan_priority",
        ],

        "L3": [
            "ip_version",
            "ip_dscp",
            "ip_ecn",
            "ip_total_length_mean",
            "ip_id_mean",
            "ip_df",
            "ip_mf",
            "fragment_offset",
            "ttl_mean",
            "ip_protocol",
            "ip_header_length",
            "ip_checksum_valid",
        ],

        "L4": [
            "src_port",
            "dst_port",
            "tcp_window_mean",
            "tcp_header_length_mean",
            "l4_checksum_valid",
        ],

        "Flow": [
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
            "flow_direction",
            "syn_count",
            "ack_count",
            "fin_count",
            "rst_count",
            "psh_count",
            "syn_seen",
            "syn_ack_seen",
            "ack_after_syn_ack",
        ],
    }

    return groups


# ============================================================
# Validation
# ============================================================

def validate_features(df):
    """
    Validate that the test dataset contains exactly the
    configured 42 ML features.
    """

    if len(config.ML_FEATURES) != 42:
        raise ValueError(
            f"Expected exactly 42 ML features, "
            f"found {len(config.ML_FEATURES)}."
        )

    missing = [
        feature
        for feature in config.ML_FEATURES
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing ML features:\n"
            + "\n".join(missing)
        )


def validate_feature_groups(groups):
    """
    Ensure every configured ML feature belongs to exactly
    one feature group.
    """

    grouped_features = []

    for group_name, features in groups.items():

        print(
            f"{group_name}: "
            f"{len(features)} features"
        )

        grouped_features.extend(features)

    configured = list(config.ML_FEATURES)

    if set(grouped_features) != set(configured):

        missing = sorted(
            set(configured)
            - set(grouped_features)
        )

        extra = sorted(
            set(grouped_features)
            - set(configured)
        )

        raise ValueError(
            "Feature-group definition does not match "
            "config.ML_FEATURES.\n"
            f"Missing from groups: {missing}\n"
            f"Extra in groups: {extra}"
        )

    if len(grouped_features) != len(set(grouped_features)):

        raise ValueError(
            "Some ML features appear in more than one group."
        )

    print("Feature groups: PASS")


# ============================================================
# Baseline evaluation
# ============================================================

def evaluate_baseline(
    model,
    X,
    y,
):
    """
    Evaluate the unchanged Current V2 model.
    """

    predictions, probabilities = predict(
        model,
        X,
    )

    metrics = calculate_metrics(
        y,
        predictions,
        probabilities,
    )

    return (
        predictions,
        probabilities,
        metrics,
    )


# ============================================================
# Single-feature ablation
# ============================================================

def run_single_feature_ablation(
    model,
    X,
    y,
    baseline_metrics,
):
    """
    Remove one feature at a time by setting that feature to NaN.

    IMPORTANT:
    This does NOT retrain the model.

    It tests how much the existing model's prediction changes
    when information from one feature is removed.

    NaN is used because the current pipeline already permits
    missing values and does not use -1 or 0 as missing markers.
    """

    results = []

    print()
    print("=" * 80)
    print("SINGLE-FEATURE ABLATION")
    print("=" * 80)

    for feature in config.ML_FEATURES:

        X_modified = X.copy()

        X_modified[feature] = np.nan

        predictions, probabilities = predict(
            model,
            X_modified,
        )

        metrics = calculate_metrics(
            y,
            predictions,
            probabilities,
        )

        row = {
            "feature": feature,

            "baseline_accuracy":
                baseline_metrics["accuracy"],

            "ablation_accuracy":
                metrics["accuracy"],

            "accuracy_drop":
                baseline_metrics["accuracy"]
                - metrics["accuracy"],

            "baseline_balanced_accuracy":
                baseline_metrics["balanced_accuracy"],

            "ablation_balanced_accuracy":
                metrics["balanced_accuracy"],

            "balanced_accuracy_drop":
                baseline_metrics["balanced_accuracy"]
                - metrics["balanced_accuracy"],

            "baseline_precision":
                baseline_metrics["precision"],

            "ablation_precision":
                metrics["precision"],

            "precision_drop":
                baseline_metrics["precision"]
                - metrics["precision"],

            "baseline_recall":
                baseline_metrics["recall"],

            "ablation_recall":
                metrics["recall"],

            "recall_drop":
                baseline_metrics["recall"]
                - metrics["recall"],

            "baseline_f1":
                baseline_metrics["f1"],

            "ablation_f1":
                metrics["f1"],

            "f1_drop":
                baseline_metrics["f1"]
                - metrics["f1"],

            "baseline_auc":
                baseline_metrics["auc_roc"],

            "ablation_auc":
                metrics["auc_roc"],

            "auc_drop":
                baseline_metrics["auc_roc"]
                - metrics["auc_roc"],
        }

        results.append(row)

        print(
            f"{feature:<30} "
            f"F1 drop: "
            f"{row['f1_drop']:+.6f} | "
            f"Recall drop: "
            f"{row['recall_drop']:+.6f}"
        )

    return pd.DataFrame(results)


# ============================================================
# Feature-group ablation
# ============================================================

def run_group_ablation(
    model,
    X,
    y,
    baseline_metrics,
    groups,
):
    """
    Remove an entire feature group by setting all features
    in that group to NaN.

    Again, this does NOT retrain the model.
    """

    results = []

    print()
    print("=" * 80)
    print("FEATURE-GROUP ABLATION")
    print("=" * 80)

    for group_name, features in groups.items():

        X_modified = X.copy()

        for feature in features:
            X_modified[feature] = np.nan

        predictions, probabilities = predict(
            model,
            X_modified,
        )

        metrics = calculate_metrics(
            y,
            predictions,
            probabilities,
        )

        row = {
            "group": group_name,
            "n_features_removed": len(features),

            "baseline_accuracy":
                baseline_metrics["accuracy"],

            "ablation_accuracy":
                metrics["accuracy"],

            "accuracy_drop":
                baseline_metrics["accuracy"]
                - metrics["accuracy"],

            "baseline_balanced_accuracy":
                baseline_metrics["balanced_accuracy"],

            "ablation_balanced_accuracy":
                metrics["balanced_accuracy"],

            "balanced_accuracy_drop":
                baseline_metrics["balanced_accuracy"]
                - metrics["balanced_accuracy"],

            "baseline_precision":
                baseline_metrics["precision"],

            "ablation_precision":
                metrics["precision"],

            "precision_drop":
                baseline_metrics["precision"]
                - metrics["precision"],

            "baseline_recall":
                baseline_metrics["recall"],

            "ablation_recall":
                metrics["recall"],

            "recall_drop":
                baseline_metrics["recall"]
                - metrics["recall"],

            "baseline_f1":
                baseline_metrics["f1"],

            "ablation_f1":
                metrics["f1"],

            "f1_drop":
                baseline_metrics["f1"]
                - metrics["f1"],

            "baseline_auc":
                baseline_metrics["auc_roc"],

            "ablation_auc":
                metrics["auc_roc"],

            "auc_drop":
                baseline_metrics["auc_roc"]
                - metrics["auc_roc"],
        }

        results.append(row)

        print(
            f"{group_name:<10} "
            f"Features removed: {len(features):>2} | "
            f"F1 drop: {row['f1_drop']:+.6f} | "
            f"Recall drop: {row['recall_drop']:+.6f}"
        )

    return pd.DataFrame(results)


# ============================================================
# Permutation importance
# ============================================================

def run_permutation_importance(
    model,
    X,
    y,
    baseline_metrics,
    sample_size,
    n_repeats,
    random_state,
):
    """
    Calculate permutation importance manually.

    For each feature:
      1. Shuffle that feature.
      2. Predict using the unchanged model.
      3. Measure performance degradation.

    This avoids requiring sklearn.inspection.permutation_importance
    and keeps the implementation explicit.
    """

    print()
    print("=" * 80)
    print("PERMUTATION IMPORTANCE")
    print("=" * 80)

    rng = np.random.default_rng(
        random_state
    )

    if len(X) > sample_size:

        sampled_indices = rng.choice(
            len(X),
            size=sample_size,
            replace=False,
        )

        X_sample = X.iloc[
            sampled_indices
        ].copy()

        y_sample = y.iloc[
            sampled_indices
        ].copy()

        print(
            f"Permutation sample: "
            f"{sample_size:,}"
        )

    else:

        X_sample = X.copy()
        y_sample = y.copy()

        print(
            f"Permutation sample: "
            f"{len(X_sample):,}"
        )

    baseline_predictions, baseline_probabilities = predict(
        model,
        X_sample,
    )

    sample_baseline_metrics = calculate_metrics(
        y_sample,
        baseline_predictions,
        baseline_probabilities,
    )

    results = []

    for feature in config.ML_FEATURES:

        accuracy_drops = []
        balanced_accuracy_drops = []
        f1_drops = []
        recall_drops = []
        auc_drops = []

        for repeat in range(n_repeats):

            X_permuted = X_sample.copy()

            permutation_indices = rng.permutation(
                len(X_permuted)
            )

            X_permuted[feature] = (
                X_permuted[feature]
                .iloc[permutation_indices]
                .to_numpy()
            )

            predictions, probabilities = predict(
                model,
                X_permuted,
            )

            metrics = calculate_metrics(
                y_sample,
                predictions,
                probabilities,
            )

            accuracy_drops.append(
                sample_baseline_metrics["accuracy"]
                - metrics["accuracy"]
            )

            balanced_accuracy_drops.append(
                sample_baseline_metrics["balanced_accuracy"]
                - metrics["balanced_accuracy"]
            )

            f1_drops.append(
                sample_baseline_metrics["f1"]
                - metrics["f1"]
            )

            recall_drops.append(
                sample_baseline_metrics["recall"]
                - metrics["recall"]
            )

            auc_drops.append(
                sample_baseline_metrics["auc_roc"]
                - metrics["auc_roc"]
            )

        results.append({
            "feature": feature,

            "accuracy_drop_mean":
                float(np.mean(accuracy_drops)),

            "accuracy_drop_std":
                float(np.std(accuracy_drops)),

            "balanced_accuracy_drop_mean":
                float(
                    np.mean(
                        balanced_accuracy_drops
                    )
                ),

            "balanced_accuracy_drop_std":
                float(
                    np.std(
                        balanced_accuracy_drops
                    )
                ),

            "f1_drop_mean":
                float(np.mean(f1_drops)),

            "f1_drop_std":
                float(np.std(f1_drops)),

            "recall_drop_mean":
                float(np.mean(recall_drops)),

            "recall_drop_std":
                float(np.std(recall_drops)),

            "auc_drop_mean":
                float(np.mean(auc_drops)),

            "auc_drop_std":
                float(np.std(auc_drops)),
        })

        print(
            f"{feature:<30} "
            f"AUC drop: "
            f"{np.mean(auc_drops):+.6f}"
        )

    result_df = pd.DataFrame(
        results
    )

    result_df = result_df.sort_values(
        "auc_drop_mean",
        ascending=False,
    ).reset_index(
        drop=True
    )

    return result_df


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Analyze feature dependency of the Current V2 "
            "Random Forest model."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_FILE),
        help="Current V2 Random Forest model.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory.",
    )

    parser.add_argument(
        "--permutation-sample",
        type=int,
        default=PERMUTATION_SAMPLE_SIZE,
        help="Number of test rows for permutation importance.",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=N_REPEATS,
        help="Number of permutation repetitions.",
    )

    args = parser.parse_args()

    model_path = Path(
        args.model
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Header
    # ========================================================

    print("=" * 80)
    print("FEATURE DEPENDENCY ANALYSIS")
    print("=" * 80)

    print(
        f"Test dataset : {TEST_FILE}"
    )

    print(
        f"Model        : {model_path}"
    )

    print(
        f"Output dir   : {output_dir}"
    )

    print()

    # ========================================================
    # Load data
    # ========================================================

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test dataset not found:\n{TEST_FILE}"
        )

    df = pd.read_csv(
        TEST_FILE
    )

    print(
        f"Loaded test dataset: "
        f"{len(df):,} rows x "
        f"{len(df.columns)} columns"
    )

    validate_features(
        df
    )

    groups = build_feature_groups()

    validate_feature_groups(
        groups
    )

    # ========================================================
    # Prepare X / y
    # ========================================================

    X = df[
        config.ML_FEATURES
    ].copy()

    y = df[
        config.LABEL_COLUMN
    ].astype(int)

    print()
    print(
        f"X shape: {X.shape}"
    )

    print(
        f"NaN cells: "
        f"{int(X.isna().sum().sum()):,}"
    )

    # ========================================================
    # Load model
    # ========================================================

    model = load_model(
        model_path
    )

    if hasattr(
        model,
        "n_features_in_",
    ):

        if model.n_features_in_ != 42:

            raise ValueError(
                f"Model expects "
                f"{model.n_features_in_} features, "
                f"but config contains 42."
            )

    print(
        f"Model classes: "
        f"{list(model.classes_)}"
    )

    # ========================================================
    # Baseline
    # ========================================================

    print()
    print("=" * 80)
    print("BASELINE CURRENT V2")
    print("=" * 80)

    (
        baseline_predictions,
        baseline_probabilities,
        baseline_metrics,
    ) = evaluate_baseline(
        model,
        X,
        y,
    )

    for name, value in baseline_metrics.items():

        print(
            f"{name:<20}: "
            f"{value:.6f}"
        )

    # ========================================================
    # Single feature ablation
    # ========================================================

    single_feature_df = (
        run_single_feature_ablation(
            model,
            X,
            y,
            baseline_metrics,
        )
    )

    single_feature_df = (
        single_feature_df.sort_values(
            "f1_drop",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    single_feature_path = (
        output_dir
        / "single_feature_ablation.csv"
    )

    single_feature_df.to_csv(
        single_feature_path,
        index=False,
    )

    # ========================================================
    # Group ablation
    # ========================================================

    group_ablation_df = (
        run_group_ablation(
            model,
            X,
            y,
            baseline_metrics,
            groups,
        )
    )

    group_ablation_path = (
        output_dir
        / "feature_group_ablation.csv"
    )

    group_ablation_df.to_csv(
        group_ablation_path,
        index=False,
    )

    # ========================================================
    # Permutation importance
    # ========================================================

    permutation_df = (
        run_permutation_importance(
            model=model,
            X=X,
            y=y,
            baseline_metrics=baseline_metrics,
            sample_size=args.permutation_sample,
            n_repeats=args.repeats,
            random_state=RANDOM_STATE,
        )
    )

    permutation_path = (
        output_dir
        / "permutation_importance.csv"
    )

    permutation_df.to_csv(
        permutation_path,
        index=False,
    )

    # ========================================================
    # Save baseline summary
    # ========================================================

    summary = {
        "model_path": str(
            model_path
        ),

        "test_dataset": str(
            TEST_FILE
        ),

        "test_rows": int(
            len(df)
        ),

        "n_ml_features": int(
            len(config.ML_FEATURES)
        ),

        "nan_cells": int(
            X.isna().sum().sum()
        ),

        "permutation_sample_size": int(
            min(
                args.permutation_sample,
                len(X),
            )
        ),

        "permutation_repeats": int(
            args.repeats
        ),

        "random_state": RANDOM_STATE,

        "baseline_metrics":
            baseline_metrics,

        "feature_groups": groups,
    }

    summary_path = (
        output_dir
        / "feature_dependency_summary.json"
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
    # Print important results
    # ========================================================

    print()
    print("=" * 80)
    print("TOP SINGLE-FEATURE ABLATION")
    print("=" * 80)

    print(
        single_feature_df[
            [
                "feature",
                "f1_drop",
                "recall_drop",
                "auc_drop",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("FEATURE-GROUP ABLATION")
    print("=" * 80)

    print(
        group_ablation_df[
            [
                "group",
                "n_features_removed",
                "f1_drop",
                "recall_drop",
                "auc_drop",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("TOP PERMUTATION IMPORTANCE")
    print("=" * 80)

    print(
        permutation_df[
            [
                "feature",
                "auc_drop_mean",
                "auc_drop_std",
                "f1_drop_mean",
                "recall_drop_mean",
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # Output paths
    # ========================================================

    print()
    print("=" * 80)
    print("OUTPUTS")
    print("=" * 80)

    print(
        single_feature_path
    )

    print(
        group_ablation_path
    )

    print(
        permutation_path
    )

    print(
        summary_path
    )

    print()
    print(
        "PASS: Feature dependency analysis completed."
    )


if __name__ == "__main__":
    main()