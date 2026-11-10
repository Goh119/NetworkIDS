
"""
robustness_model_comparison.py
------------------------------
NetworkIDS - Fair Multi-Model Feature-Level Robustness Evaluation.

Models:
    Random Forest
    Decision Tree
    Support Vector Machine

Input:
    data/test/test_multiday.csv

Reuses the eight perturbation scenarios defined in:
    src/robustness_evaluation.py

Evaluation design:
    1. Select the same held-out test samples for all models.
    2. Generate each perturbed dataset only once.
    3. Evaluate all three models on the exact same data.
    4. Compare clean and perturbed classification metrics.
    5. Calculate prediction flips.
    6. Calculate attack evasion and benign induction only for
       samples that the respective model classified correctly
       before perturbation.

Important:
    - Models are NOT retrained.
    - Ground-truth labels remain unchanged.
    - Perturbations modify flow feature values, not PCAP packets.
    - NaN values are preserved by the original perturbation functions.
    - Each saved model is responsible for its own preprocessing.

Outputs:
    robustness/model_comparison_v2/
        model_comparison_detailed.csv
        model_comparison_summary.csv
        model_comparison_metadata.json
"""

from pathlib import Path
import argparse
import json
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ------------------------------------------------------------
# Project imports
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config as config

# Reuse the ORIGINAL perturbations, unchanged.
from src.robustness_evaluation import build_scenarios


# ------------------------------------------------------------
# Paths and defaults
# ------------------------------------------------------------

DEFAULT_TEST_FILE = (
    PROJECT_ROOT / "data" / "test" / "test_multiday.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "robustness" / "model_comparison_v2"
)

MODEL_PATHS = {
    "Random Forest": (
        PROJECT_ROOT / "random_forest" / "rf_model.joblib"
    ),
    "Decision Tree": (
        PROJECT_ROOT / "decision_tree" / "dt_model.joblib"
    ),
    "SVM": (
        PROJECT_ROOT
        / "support_vector_machine"
        / "svm_model.joblib"
    ),
}

DEFAULT_SAMPLE_SIZE = 2000
DEFAULT_RANDOM_STATE = 42


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def get_attack_class_index(model):
    if not hasattr(model, "classes_"):
        raise RuntimeError(
            "Model does not expose classes_."
        )

    classes = list(model.classes_)

    if config.LABEL_ATTACK not in classes:
        raise RuntimeError(
            f"Attack class {config.LABEL_ATTACK} "
            f"is missing from model classes {classes}."
        )

    return classes.index(config.LABEL_ATTACK)


def validate_model(model, model_name):
    required_features = list(config.ML_FEATURES)

    if not hasattr(model, "predict"):
        raise RuntimeError(
            f"{model_name} does not support predict()."
        )

    if not hasattr(model, "predict_proba"):
        raise RuntimeError(
            f"{model_name} does not support predict_proba()."
        )

    if hasattr(model, "n_features_in_"):
        actual_count = int(model.n_features_in_)

        if actual_count != len(required_features):
            raise RuntimeError(
                f"{model_name}: feature count mismatch. "
                f"Expected {len(required_features)}, "
                f"model requires {actual_count}."
            )

    if hasattr(model, "feature_names_in_"):
        actual_features = list(model.feature_names_in_)

        if actual_features != required_features:
            raise RuntimeError(
                f"{model_name}: feature order mismatch "
                "with config.ML_FEATURES."
            )

    get_attack_class_index(model)


def load_models():
    models = {}

    for name, path in MODEL_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{name} model not found: {path}"
            )

        print(f"Loading {name}: {path}")

        model = joblib.load(path)

        validate_model(model, name)

        models[name] = model

    return models


def load_test_dataset(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {path}"
        )

    config.validate_ml_features()

    features = list(config.ML_FEATURES)

    if len(features) != len(set(features)):
        raise ValueError(
            "Duplicate feature names in config.ML_FEATURES."
        )

    required = list(
        dict.fromkeys(
            features + [config.LABEL_COLUMN]
        )
    )

    header = pd.read_csv(path, nrows=0)

    missing = [
        col for col in required
        if col not in header.columns
    ]

    if missing:
        raise ValueError(
            "Missing test dataset columns:\n"
            + "\n".join(missing)
        )

    dtype_map = {
        col: np.float32
        for col in features
    }

    dtype_map[config.LABEL_COLUMN] = np.int8

    df = pd.read_csv(
        path,
        usecols=required,
        dtype=dtype_map,
        low_memory=False,
    )

    if df.empty:
        raise ValueError("Test dataset is empty.")

    if df[config.LABEL_COLUMN].isna().any():
        raise ValueError("Ground-truth labels contain NaN.")

    actual_labels = set(
        df[config.LABEL_COLUMN].unique().tolist()
    )

    expected_labels = {
        config.LABEL_ATTACK,
        config.LABEL_BENIGN,
    }

    if actual_labels != expected_labels:
        raise ValueError(
            f"Expected both labels {expected_labels}; "
            f"found {actual_labels}."
        )

    X = df[features]

    for feature in features:
        if np.isinf(
            X[feature].to_numpy(copy=False)
        ).any():
            raise ValueError(
                f"Infinite values in feature: {feature}"
            )

    print(f"Test rows: {len(df):,}")
    print(f"ML features: {len(features)}")
    print(f"NaN cells: {int(X.isna().sum().sum()):,}")

    return df


# ------------------------------------------------------------
# Shared sample selection
# ------------------------------------------------------------

def select_shared_samples(df, n_per_class, seed):
    """
    Select 2,000 Attack + 2,000 Benign by ground truth.

    Importantly, sample selection does NOT depend on any
    model's prediction. All models use the same rows.
    """

    labels = df[config.LABEL_COLUMN].to_numpy()

    attack_positions = np.flatnonzero(
        labels == config.LABEL_ATTACK
    )

    benign_positions = np.flatnonzero(
        labels == config.LABEL_BENIGN
    )

    if len(attack_positions) < n_per_class:
        raise ValueError(
            f"Not enough Attack samples. "
            f"Available: {len(attack_positions):,}"
        )

    if len(benign_positions) < n_per_class:
        raise ValueError(
            f"Not enough Benign samples. "
            f"Available: {len(benign_positions):,}"
        )

    rng = np.random.default_rng(seed)

    selected_attack = rng.choice(
        attack_positions,
        size=n_per_class,
        replace=False,
    )

    selected_benign = rng.choice(
        benign_positions,
        size=n_per_class,
        replace=False,
    )

    selected_positions = np.concatenate(
        [selected_attack, selected_benign]
    )

    sample_df = df.iloc[
        selected_positions
    ].reset_index(drop=True)

    X = sample_df[
        config.ML_FEATURES
    ].copy().astype(np.float32)

    y = sample_df[
        config.LABEL_COLUMN
    ].to_numpy(dtype=np.int8)

    print(f"Shared Attack samples: {n_per_class:,}")
    print(f"Shared Benign samples: {n_per_class:,}")
    print(f"Total shared samples: {len(X):,}")

    return X, y, selected_positions


# ------------------------------------------------------------
# Model prediction
# ------------------------------------------------------------

def predict_model(model, X):
    index = get_attack_class_index(model)

    predicted = np.asarray(
        model.predict(X)
    )

    probabilities = np.asarray(
        model.predict_proba(X)
    )

    if probabilities.ndim != 2:
        raise RuntimeError(
            "predict_proba() returned invalid dimensions."
        )

    if probabilities.shape[0] != len(X):
        raise RuntimeError(
            "Prediction probability row count mismatch."
        )

    if probabilities.shape[1] <= index:
        raise RuntimeError(
            "Attack probability column is missing."
        )

    attack_probability = probabilities[:, index]

    if not np.all(np.isfinite(attack_probability)):
        raise RuntimeError(
            "Model returned non-finite probabilities."
        )

    return predicted, attack_probability


# ------------------------------------------------------------
# Classification metrics
# ------------------------------------------------------------

def classification_metrics(y, predicted, probability):
    """
    Attack=positive class.
    """

    return {
        "accuracy": float(
            accuracy_score(y, predicted)
        ),
        "precision": float(
            precision_score(
                y, predicted,
                pos_label=config.LABEL_ATTACK,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y, predicted,
                pos_label=config.LABEL_ATTACK,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y, predicted,
                pos_label=config.LABEL_ATTACK,
                zero_division=0,
            )
        ),
        "auc_roc": float(
            roc_auc_score(
                (y == config.LABEL_ATTACK).astype(int),
                probability,
            )
        ),
    }


# ------------------------------------------------------------
# Feature-change measurement
# ------------------------------------------------------------

def changed_feature_statistics(original, perturbed):
    """
    NaN -> NaN is unchanged.
    """

    before = original.to_numpy(dtype=np.float64)
    after = perturbed.to_numpy(dtype=np.float64)

    same = (
        (before == after)
        |
        (np.isnan(before) & np.isnan(after))
    )

    changed_cells = int((~same).sum())
    total_cells = int(before.size)

    changed_rows = int((~same).any(axis=1).sum())
    total_rows = int(before.shape[0])

    return {
        "changed_feature_cells": changed_cells,
        "changed_feature_percentage": (
            100.0 * changed_cells / total_cells
            if total_cells else 0.0
        ),
        "changed_flow_count": changed_rows,
        "changed_flow_percentage": (
            100.0 * changed_rows / total_rows
            if total_rows else 0.0
        ),
    }


# ------------------------------------------------------------
# Robustness metrics
# ------------------------------------------------------------

def robustness_metrics(
    y,
    clean_pred,
    perturbed_pred,
    clean_prob,
    perturbed_prob,
):
    """
    Flip rate:
        All shared samples.

    Attack evasion:
        Only originally correctly detected Attacks.

    Benign induction:
        Only originally correctly classified Benign samples.
    """

    clean_correct = (clean_pred == y)

    clean_attack_correct = (
        (y == config.LABEL_ATTACK)
        & clean_correct
    )

    clean_benign_correct = (
        (y == config.LABEL_BENIGN)
        & clean_correct
    )

    attack_denominator = int(
        clean_attack_correct.sum()
    )

    benign_denominator = int(
        clean_benign_correct.sum()
    )

    attack_to_benign = int(
        np.sum(
            clean_attack_correct
            & (perturbed_pred == config.LABEL_BENIGN)
        )
    )

    benign_to_attack = int(
        np.sum(
            clean_benign_correct
            & (perturbed_pred == config.LABEL_ATTACK)
        )
    )

    flip_count = int(
        np.sum(perturbed_pred != clean_pred)
    )

    n = len(y)

    attack_probability_drop = (
        clean_prob[clean_attack_correct]
        - perturbed_prob[clean_attack_correct]
    )

    benign_probability_rise = (
        perturbed_prob[clean_benign_correct]
        - clean_prob[clean_benign_correct]
    )

    return {
        "prediction_flip_count": flip_count,
        "prediction_flip_rate": (
            flip_count / n if n else np.nan
        ),
        "baseline_correct_attack_count": attack_denominator,
        "baseline_correct_benign_count": benign_denominator,
        "attack_to_benign_count": attack_to_benign,
        "benign_to_attack_count": benign_to_attack,
        "attack_evasion_rate": (
            attack_to_benign / attack_denominator
            if attack_denominator else np.nan
        ),
        "benign_induction_rate": (
            benign_to_attack / benign_denominator
            if benign_denominator else np.nan
        ),
        "mean_attack_probability_drop": (
            float(np.mean(attack_probability_drop))
            if attack_denominator else np.nan
        ),
        "mean_benign_probability_rise": (
            float(np.mean(benign_probability_rise))
            if benign_denominator else np.nan
        ),
        "mean_absolute_probability_shift": float(
            np.mean(
                np.abs(perturbed_prob - clean_prob)
            )
        ),
    }


# ------------------------------------------------------------
# Shared perturbation generation
# ------------------------------------------------------------

def generate_perturbed_dataset(
    X,
    perturbation_function,
    scenario_seed,
    n_attack,
):
    """
    Apply the ORIGINAL scenario exactly once.

    Use separate Attack and Benign partitions to match the
    original script's perturbation function calling pattern.

    The resulting X_perturbed is then shared by all models.
    """

    rng = np.random.default_rng(scenario_seed)

    attack_X = X.iloc[:n_attack].copy()
    benign_X = X.iloc[n_attack:].copy()

    modified_attack = perturbation_function(
        attack_X, rng
    )

    modified_benign = perturbation_function(
        benign_X, rng
    )

    perturbed = pd.concat(
        [modified_attack, modified_benign],
        ignore_index=True,
    )

    perturbed = perturbed[
        config.ML_FEATURES
    ].astype(np.float32)

    return perturbed


# ------------------------------------------------------------
# Main evaluation
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate RF, DT and SVM robustness on "
            "identical held-out flow samples."
        )
    )

    parser.add_argument(
        "--test-file",
        type=Path,
        default=DEFAULT_TEST_FILE,
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of test samples per class.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    if args.sample_size < 1:
        parser.error("--sample-size must be positive.")

    output_dir = args.out_dir.resolve()
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    section("NETWORKIDS - FAIR MODEL ROBUSTNESS COMPARISON")

    print(f"Test dataset: {args.test_file.resolve()}")
    print(f"Samples/class: {args.sample_size:,}")
    print(f"Random state: {args.random_state}")
    print(f"Output: {output_dir}")

    section("[1/5] Load held-out test data")

    df = load_test_dataset(args.test_file)

    section("[2/5] Select common test samples")

    X, y, selected_positions = select_shared_samples(
        df,
        args.sample_size,
        args.random_state,
    )

    del df

    section("[3/5] Load all trained models")

    models = load_models()

    clean_results = {}

    for name, model in models.items():
        predicted, probability = predict_model(
            model, X
        )

        metrics = classification_metrics(
            y, predicted, probability
        )

        clean_results[name] = {
            "prediction": predicted,
            "probability": probability,
            "metrics": metrics,
        }

        print(
            f"{name:15} "
            f"Accuracy={metrics['accuracy']:.4%}  "
            f"F1={metrics['f1_score']:.4%}  "
            f"AUC={metrics['auc_roc']:.4%}"
        )

    section("[4/5] Evaluate shared perturbations")

    rows = []

    scenarios = build_scenarios()

    for scenario_number, (
        scenario_name,
        description,
        perturbation,
    ) in enumerate(scenarios, start=1):

        print()
        print(
            f"[{scenario_number}/{len(scenarios)}] "
            f"{scenario_name}"
        )

        scenario_seed = (
            args.random_state + scenario_number
        )

        X_perturbed = generate_perturbed_dataset(
            X,
            perturbation,
            scenario_seed,
            args.sample_size,
        )

        change_stats = changed_feature_statistics(
            X, X_perturbed
        )

        print(
            "  Changed flows: "
            f"{change_stats['changed_flow_count']:,} / "
            f"{len(X):,}"
        )

        if change_stats["changed_flow_count"] == 0:
            print(
                "  WARNING: This perturbation did not "
                "change any selected flow feature."
            )

        for name, model in models.items():
            clean = clean_results[name]

            perturbed_pred, perturbed_prob = (
                predict_model(model, X_perturbed)
            )

            after = classification_metrics(
                y,
                perturbed_pred,
                perturbed_prob,
            )

            before = clean["metrics"]

            robustness = robustness_metrics(
                y,
                clean["prediction"],
                perturbed_pred,
                clean["probability"],
                perturbed_prob,
            )

            row = {
                "model": name,
                "scenario": scenario_name,
                "description": description,
                "n_samples": int(len(X)),
                "n_attack": int(args.sample_size),
                "n_benign": int(args.sample_size),

                "baseline_accuracy": before["accuracy"],
                "perturbed_accuracy": after["accuracy"],
                "accuracy_change": (
                    after["accuracy"] - before["accuracy"]
                ),

                "baseline_precision": before["precision"],
                "perturbed_precision": after["precision"],
                "precision_change": (
                    after["precision"] - before["precision"]
                ),

                "baseline_recall": before["recall"],
                "perturbed_recall": after["recall"],
                "recall_change": (
                    after["recall"] - before["recall"]
                ),

                "baseline_f1": before["f1_score"],
                "perturbed_f1": after["f1_score"],
                "f1_change": (
                    after["f1_score"] - before["f1_score"]
                ),

                "baseline_auc": before["auc_roc"],
                "perturbed_auc": after["auc_roc"],
                "auc_change": (
                    after["auc_roc"] - before["auc_roc"]
                ),

                **robustness,
                **change_stats,
            }

            rows.append(row)

            print(
                f"  {name:15} "
                f"Flip={robustness['prediction_flip_rate']:.2%}  "
                f"Evasion={robustness['attack_evasion_rate']:.2%}  "
                f"Induction={robustness['benign_induction_rate']:.2%}  "
                f"F1={after['f1_score']:.4f}"
            )

        del X_perturbed

    section("[5/5] Save results")

    detailed = pd.DataFrame(rows)

    detailed_path = (
        output_dir / "model_comparison_detailed.csv"
    )

    detailed.to_csv(
        detailed_path,
        index=False,
    )

    # One row per model with mean and worst-case results.
    summary_rows = []

    for name, group in detailed.groupby(
        "model", sort=False
    ):
        clean = clean_results[name]["metrics"]

        summary_rows.append({
            "model": name,
            "baseline_accuracy": clean["accuracy"],
            "baseline_precision": clean["precision"],
            "baseline_recall": clean["recall"],
            "baseline_f1": clean["f1_score"],
            "baseline_auc": clean["auc_roc"],

            "mean_perturbed_accuracy": float(
                group["perturbed_accuracy"].mean()
            ),
            "mean_perturbed_f1": float(
                group["perturbed_f1"].mean()
            ),
            "worst_perturbed_f1": float(
                group["perturbed_f1"].min()
            ),

            "mean_prediction_flip_rate": float(
                group["prediction_flip_rate"].mean()
            ),
            "max_prediction_flip_rate": float(
                group["prediction_flip_rate"].max()
            ),

            "mean_attack_evasion_rate": float(
                group["attack_evasion_rate"].mean()
            ),
            "max_attack_evasion_rate": float(
                group["attack_evasion_rate"].max()
            ),

            "mean_benign_induction_rate": float(
                group["benign_induction_rate"].mean()
            ),
            "max_benign_induction_rate": float(
                group["benign_induction_rate"].max()
            ),
        })

    summary = pd.DataFrame(summary_rows)

    summary_path = (
        output_dir / "model_comparison_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    metadata = {
        "evaluation_type": (
            "Controlled feature-level robustness comparison"
        ),
        "test_dataset": str(args.test_file.resolve()),
        "models": {
            name: str(path.resolve())
            for name, path in MODEL_PATHS.items()
        },
        "feature_count": len(config.ML_FEATURES),
        "features": list(config.ML_FEATURES),
        "sample_size_per_class": args.sample_size,
        "total_selected_samples": len(X),
        "random_state": args.random_state,
        "scenarios": [
            name for name, _, _ in scenarios
        ],
        "shared_sample_selection": (
            "Fixed stratified sample from ground truth; "
            "independent of model predictions"
        ),
        "shared_perturbations": True,
        "selected_test_row_positions": (
            selected_positions.astype(int).tolist()
        ),
        "evasion_denominator": (
            "Originally correctly detected Attack samples "
            "within the shared test sample, separately per model"
        ),
        "induction_denominator": (
            "Originally correctly classified Benign samples "
            "within the shared test sample, separately per model"
        ),
        "ground_truth_modified": False,
        "packet_level_modification": False,
        "model_retraining": False,
        "metric_scale": (
            "Rates and metrics stored as fractions from 0 to 1; "
            "multiply by 100 to display percentages"
        ),
        "limitations": [
            "Feature perturbations are not actual PCAP changes.",
            "Modified features may not form physically consistent traffic.",
            "Conditional flip-rate denominators differ across models.",
            "Mean results across scenarios are descriptive, not weighted.",
            "The sampled baseline is not the full test-set baseline.",
        ],
    }

    metadata_path = (
        output_dir / "model_comparison_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print()
    print("Saved:")
    print(f"  {detailed_path}")
    print(f"  {summary_path}")
    print(f"  {metadata_path}")

    print()
    print("SUMMARY")
    print(
        summary[
            [
                "model",
                "baseline_accuracy",
                "baseline_f1",
                "mean_prediction_flip_rate",
                "mean_attack_evasion_rate",
                "max_attack_evasion_rate",
                "worst_perturbed_f1",
            ]
        ].to_string(
            index=False,
            float_format=lambda v: f"{v:.4f}",
        )
    )

    print()
    print("=" * 72)
    print("MODEL ROBUSTNESS COMPARISON COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
