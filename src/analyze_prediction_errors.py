from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
import src.config as config


# ============================================================
# NetworkIDS - Prediction Error Analysis
# ============================================================

RANDOM_FOREST_MODEL_FILE = config.RANDOM_FOREST_MODEL_FILE
TEST_FILE = config.PROCESSED_DIR / "split" / "test_dataset.csv"

OUTPUT_DIR = config.MODELS_DIR / "error_analysis"

PREDICTION_RESULTS_FILE = OUTPUT_DIR / "prediction_results.csv"
FALSE_POSITIVE_FILE = OUTPUT_DIR / "false_positive_flows.csv"
FALSE_NEGATIVE_FILE = OUTPUT_DIR / "false_negative_flows.csv"
SUMMARY_FILE = OUTPUT_DIR / "prediction_error_summary.json"


THRESHOLD = 0.5


def ensure_output_directory():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def validate_dataset(df):
    missing_features = [
        feature for feature in config.ML_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing ML features: {missing_features}"
        )

    if config.LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"Missing label column: {config.LABEL_COLUMN}"
        )

    if df[config.LABEL_COLUMN].isna().any():
        raise ValueError(
            "Test dataset contains NaN labels."
        )

    labels = set(df[config.LABEL_COLUMN].unique())

    if not labels.issubset({0, 1}):
        raise ValueError(
            f"Unexpected labels found: {labels}"
        )

    numeric_data = df[config.ML_FEATURES].select_dtypes(
        include=[np.number]
    )

    if np.isinf(numeric_data.to_numpy()).any():
        raise ValueError(
            "Infinite values detected in ML features."
        )


def probability_statistics(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)

    if len(probabilities) == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "within_0_45_0_55": 0,
            "within_0_40_0_60": 0,
        }

    return {
        "count": int(len(probabilities)),
        "mean": float(np.mean(probabilities)),
        "median": float(np.median(probabilities)),
        "min": float(np.min(probabilities)),
        "max": float(np.max(probabilities)),
        "within_0_45_0_55": int(
            np.sum(
                (probabilities >= 0.45)
                & (probabilities <= 0.55)
            )
        ),
        "within_0_40_0_60": int(
            np.sum(
                (probabilities >= 0.40)
                & (probabilities <= 0.60)
            )
        ),
    }


def main():

    print("=" * 70)
    print("NetworkIDS - Prediction Error Analysis")
    print("=" * 70)

    ensure_output_directory()

    # --------------------------------------------------------
    # 1. Load model and test data
    # --------------------------------------------------------
    print("\n[1/5] Loading model and test dataset...")

    print(f"Model:")
    print(f"  {RANDOM_FOREST_MODEL_FILE}")

    print(f"Test dataset:")
    print(f"  {TEST_FILE}")

    model = joblib.load(RANDOM_FOREST_MODEL_FILE)
    df = pd.read_csv(TEST_FILE)

    print(f"  Test rows: {len(df):,}")
    print(f"  Features : {len(config.ML_FEATURES)}")

    validate_dataset(df)

    # --------------------------------------------------------
    # 2. Generate predictions
    # --------------------------------------------------------
    print("\n[2/5] Generating predictions...")

    X = df[config.ML_FEATURES]
    y_true = df[config.LABEL_COLUMN].astype(int).to_numpy()

    probabilities = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X).astype(int)

    print(f"  Threshold: {THRESHOLD:.2f}")
    print("  Predictions generated.")

    # --------------------------------------------------------
    # 3. Categorize TP / TN / FP / FN
    # --------------------------------------------------------
    print("\n[3/5] Categorizing TP / TN / FP / FN...")

    true_positive = (y_true == 1) & (y_pred == 1)
    true_negative = (y_true == 0) & (y_pred == 0)
    false_positive = (y_true == 0) & (y_pred == 1)
    false_negative = (y_true == 1) & (y_pred == 0)

    tp_count = int(true_positive.sum())
    tn_count = int(true_negative.sum())
    fp_count = int(false_positive.sum())
    fn_count = int(false_negative.sum())

    total = len(df)

    print(f"\n  True Positive : {tp_count:,}")
    print(f"  True Negative : {tn_count:,}")
    print(f"  False Positive: {fp_count:,}")
    print(f"  False Negative: {fn_count:,}")

    print("\n  Percentages:")
    print(f"    TP: {tp_count / total * 100:.4f}%")
    print(f"    TN: {tn_count / total * 100:.4f}%")
    print(f"    FP: {fp_count / total * 100:.4f}%")
    print(f"    FN: {fn_count / total * 100:.4f}%")

    # --------------------------------------------------------
    # 4. Build prediction result table
    # --------------------------------------------------------
    print("\n[4/5] Saving prediction results and error cases...")

    results = df[
        config.METADATA_COLUMNS
        + [config.LABEL_COLUMN, "original_label"]
    ].copy()

    results["predicted_label"] = y_pred
    results["attack_probability"] = probabilities

    results["prediction_type"] = np.select(
        [
            true_positive,
            true_negative,
            false_positive,
            false_negative,
        ],
        [
            "TP",
            "TN",
            "FP",
            "FN",
        ],
        default="UNKNOWN",
    )

    results["correct_prediction"] = (
        results[config.LABEL_COLUMN]
        == results["predicted_label"]
    )

    results.to_csv(
        PREDICTION_RESULTS_FILE,
        index=False
    )

    fp_results = results[
        results["prediction_type"] == "FP"
    ].copy()

    fn_results = results[
        results["prediction_type"] == "FN"
    ].copy()

    fp_results.to_csv(
        FALSE_POSITIVE_FILE,
        index=False
    )

    fn_results.to_csv(
        FALSE_NEGATIVE_FILE,
        index=False
    )

    print(f"  Prediction results:")
    print(f"    {PREDICTION_RESULTS_FILE}")

    print(f"  False positives:")
    print(f"    {FALSE_POSITIVE_FILE}")

    print(f"  False negatives:")
    print(f"    {FALSE_NEGATIVE_FILE}")

    # --------------------------------------------------------
    # 5. Probability analysis
    # --------------------------------------------------------
    print("\n[5/5] Analysing probability distributions...")

    tp_prob = probabilities[true_positive]
    tn_prob = probabilities[true_negative]
    fp_prob = probabilities[false_positive]
    fn_prob = probabilities[false_negative]

    summary = {
        "model": "RandomForestClassifier",
        "model_path": str(RANDOM_FOREST_MODEL_FILE),
        "test_dataset": str(TEST_FILE),

        "test_rows": int(total),
        "threshold": THRESHOLD,
        "feature_count": len(config.ML_FEATURES),

        "confusion_matrix": {
            "true_positive": tp_count,
            "true_negative": tn_count,
            "false_positive": fp_count,
            "false_negative": fn_count,
        },

        "prediction_percentages": {
            "true_positive": tp_count / total * 100,
            "true_negative": tn_count / total * 100,
            "false_positive": fp_count / total * 100,
            "false_negative": fn_count / total * 100,
        },

        "probability_statistics": {
            "true_positive": probability_statistics(tp_prob),
            "true_negative": probability_statistics(tn_prob),
            "false_positive": probability_statistics(fp_prob),
            "false_negative": probability_statistics(fn_prob),
        },

        "error_cases": {
            "false_positive_count": fp_count,
            "false_negative_count": fn_count,
            "total_errors": fp_count + fn_count,
            "overall_error_rate": (
                (fp_count + fn_count) / total * 100
            ),
        },

        "output_files": [
            "prediction_results.csv",
            "false_positive_flows.csv",
            "false_negative_flows.csv",
        ],
    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    print("\n" + "=" * 70)
    print("PREDICTION ERROR ANALYSIS - COMPLETE")
    print("=" * 70)

    print(f"  Test rows       : {total:,}")
    print(f"  TP              : {tp_count:,}")
    print(f"  TN              : {tn_count:,}")
    print(f"  FP              : {fp_count:,}")
    print(f"  FN              : {fn_count:,}")
    print(
        f"  Overall errors  : "
        f"{fp_count + fn_count:,} "
        f"({(fp_count + fn_count) / total * 100:.4f}%)"
    )

    print("\nProbability summary:")

    print(
        f"  TP mean attack probability: "
        f"{np.mean(tp_prob):.4f}"
    )

    print(
        f"  TN mean attack probability: "
        f"{np.mean(tn_prob):.4f}"
    )

    if len(fp_prob) > 0:
        print(
            f"  FP mean attack probability: "
            f"{np.mean(fp_prob):.4f}"
        )
        print(
            f"  FP within 0.45-0.55: "
            f"{np.sum((fp_prob >= 0.45) & (fp_prob <= 0.55)):,}"
        )

    if len(fn_prob) > 0:
        print(
            f"  FN mean attack probability: "
            f"{np.mean(fn_prob):.4f}"
        )
        print(
            f"  FN within 0.45-0.55: "
            f"{np.sum((fn_prob >= 0.45) & (fn_prob <= 0.55)):,}"
        )

    print(f"\nSummary saved:")
    print(f"  {SUMMARY_FILE}")

    print("=" * 70)


if __name__ == "__main__":
    main()