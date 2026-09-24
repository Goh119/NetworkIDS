from pathlib import Path
import argparse
import json
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import src.config as config


# ============================================================
# NetworkIDS - Prediction Error Analysis
# ============================================================

TEST_FILE = (
    config.PROCESSED_DIR
    / "split"
    / "test_dataset.csv"
)

BASE_OUTPUT_DIR = (
    config.MODELS_DIR
    / "error_analysis"
)

THRESHOLD = 0.5


# ============================================================
# Helpers
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "NetworkIDS prediction error analysis "
            "for a specified Random Forest model."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default=str(
            config.RANDOM_FOREST_MODEL_FILE
        ),
        help=(
            "Path to the Random Forest model. "
            "Default: baseline model."
        ),
    )

    parser.add_argument(
        "--name",
        type=str,
        default="baseline",
        help=(
            "Short model name used for the output "
            "directory. Examples: baseline, v1, v2."
        ),
    )

    return parser.parse_args()


def validate_dataset(df):

    missing_features = [
        feature
        for feature in config.ML_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing ML features: {missing_features}"
        )

    if config.LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"Missing label column: "
            f"{config.LABEL_COLUMN}"
        )

    if df[config.LABEL_COLUMN].isna().any():
        raise ValueError(
            "Test dataset contains NaN labels."
        )

    labels = set(
        df[config.LABEL_COLUMN].unique()
    )

    if not labels.issubset({0, 1}):
        raise ValueError(
            f"Unexpected labels found: {labels}"
        )

    numeric_data = df[
        config.ML_FEATURES
    ].select_dtypes(
        include=[np.number]
    )

    if np.isinf(
        numeric_data.to_numpy()
    ).any():
        raise ValueError(
            "Infinite values detected in ML features."
        )


def probability_statistics(probabilities):

    probabilities = np.asarray(
        probabilities,
        dtype=float
    )

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
        "mean": float(
            np.mean(probabilities)
        ),
        "median": float(
            np.median(probabilities)
        ),
        "min": float(
            np.min(probabilities)
        ),
        "max": float(
            np.max(probabilities)
        ),
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


# ============================================================
# Main
# ============================================================

def main():

    args = parse_arguments()

    model_path = Path(args.model)

    if not model_path.is_absolute():
        model_path = (
            Path.cwd()
            / model_path
        )

    output_dir = (
        BASE_OUTPUT_DIR
        / args.name
    )

    prediction_results_file = (
        output_dir
        / "prediction_results.csv"
    )

    false_positive_file = (
        output_dir
        / "false_positive_flows.csv"
    )

    false_negative_file = (
        output_dir
        / "false_negative_flows.csv"
    )

    summary_file = (
        output_dir
        / "prediction_error_summary.json"
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print("=" * 70)
    print(
        "NetworkIDS - Prediction Error Analysis"
    )
    print("=" * 70)

    print("\nModel name:")
    print(f"  {args.name}")

    print("\nModel:")
    print(f"  {model_path}")

    print("\nTest dataset:")
    print(f"  {TEST_FILE}")

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 1. Load model and test data
    # --------------------------------------------------------

    print(
        "\n[1/5] Loading model and test dataset..."
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found:\n"
            f"{model_path}"
        )

    model = joblib.load(
        model_path
    )

    df = pd.read_csv(
        TEST_FILE,
        low_memory=False
    )

    print(
        f"  Test rows: {len(df):,}"
    )

    print(
        f"  Features : "
        f"{len(config.ML_FEATURES)}"
    )

    validate_dataset(df)

    # --------------------------------------------------------
    # 2. Generate predictions
    # --------------------------------------------------------

    print(
        "\n[2/5] Generating predictions..."
    )

    X = df[
        config.ML_FEATURES
    ]

    y_true = (
        df[config.LABEL_COLUMN]
        .astype(int)
        .to_numpy()
    )

    probabilities = (
        model.predict_proba(X)[:, 1]
    )

    # IMPORTANT:
    # Use model.predict() so that the prediction
    # semantics exactly match the model evaluation.
    y_pred = (
        model.predict(X)
        .astype(int)
    )

    print(
        f"  Threshold reference: "
        f"{THRESHOLD:.2f}"
    )

    print(
        "  Predictions generated."
    )

    # --------------------------------------------------------
    # 3. Categorize TP / TN / FP / FN
    # --------------------------------------------------------

    print(
        "\n[3/5] Categorizing TP / TN / FP / FN..."
    )

    true_positive = (
        (y_true == 1)
        & (y_pred == 1)
    )

    true_negative = (
        (y_true == 0)
        & (y_pred == 0)
    )

    false_positive = (
        (y_true == 0)
        & (y_pred == 1)
    )

    false_negative = (
        (y_true == 1)
        & (y_pred == 0)
    )

    tp_count = int(
        true_positive.sum()
    )

    tn_count = int(
        true_negative.sum()
    )

    fp_count = int(
        false_positive.sum()
    )

    fn_count = int(
        false_negative.sum()
    )

    total = len(df)

    print(
        f"\n  True Positive : {tp_count:,}"
    )

    print(
        f"  True Negative : {tn_count:,}"
    )

    print(
        f"  False Positive: {fp_count:,}"
    )

    print(
        f"  False Negative: {fn_count:,}"
    )

    print("\n  Percentages:")

    print(
        f"    TP: "
        f"{tp_count / total * 100:.4f}%"
    )

    print(
        f"    TN: "
        f"{tn_count / total * 100:.4f}%"
    )

    print(
        f"    FP: "
        f"{fp_count / total * 100:.4f}%"
    )

    print(
        f"    FN: "
        f"{fn_count / total * 100:.4f}%"
    )

    # --------------------------------------------------------
    # 4. Build prediction result table
    # --------------------------------------------------------

    print(
        "\n[4/5] Saving prediction results "
        "and error cases..."
    )

    results = df[
        config.METADATA_COLUMNS
        + [
            config.LABEL_COLUMN,
            "original_label",
        ]
    ].copy()

    results["predicted_label"] = (
        y_pred
    )

    results["attack_probability"] = (
        probabilities
    )

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
        prediction_results_file,
        index=False
    )

    fp_results = results[
        results["prediction_type"] == "FP"
    ].copy()

    fn_results = results[
        results["prediction_type"] == "FN"
    ].copy()

    fp_results.to_csv(
        false_positive_file,
        index=False
    )

    fn_results.to_csv(
        false_negative_file,
        index=False
    )

    print(
        "\n  Prediction results:"
    )

    print(
        f"    {prediction_results_file}"
    )

    print(
        "  False positives:"
    )

    print(
        f"    {false_positive_file}"
    )

    print(
        "  False negatives:"
    )

    print(
        f"    {false_negative_file}"
    )

    # --------------------------------------------------------
    # 5. Probability analysis
    # --------------------------------------------------------

    print(
        "\n[5/5] Analysing probability distributions..."
    )

    tp_prob = probabilities[
        true_positive
    ]

    tn_prob = probabilities[
        true_negative
    ]

    fp_prob = probabilities[
        false_positive
    ]

    fn_prob = probabilities[
        false_negative
    ]

    summary = {

        "model_name": args.name,

        "model_path": str(
            model_path
        ),

        "test_dataset": str(
            TEST_FILE
        ),

        "test_rows": int(
            total
        ),

        "threshold_reference": THRESHOLD,

        "feature_count": len(
            config.ML_FEATURES
        ),

        "confusion_matrix": {

            "true_positive": tp_count,

            "true_negative": tn_count,

            "false_positive": fp_count,

            "false_negative": fn_count,
        },

        "prediction_percentages": {

            "true_positive":
                tp_count / total * 100,

            "true_negative":
                tn_count / total * 100,

            "false_positive":
                fp_count / total * 100,

            "false_negative":
                fn_count / total * 100,
        },

        "probability_statistics": {

            "true_positive":
                probability_statistics(
                    tp_prob
                ),

            "true_negative":
                probability_statistics(
                    tn_prob
                ),

            "false_positive":
                probability_statistics(
                    fp_prob
                ),

            "false_negative":
                probability_statistics(
                    fn_prob
                ),
        },

        "error_cases": {

            "false_positive_count":
                fp_count,

            "false_negative_count":
                fn_count,

            "total_errors":
                fp_count + fn_count,

            "overall_error_rate":
                (
                    (fp_count + fn_count)
                    / total
                    * 100
                ),
        },

        "output_files": {

            "prediction_results":
                str(
                    prediction_results_file
                ),

            "false_positive_flows":
                str(
                    false_positive_file
                ),

            "false_negative_flows":
                str(
                    false_negative_file
                ),
        },
    }

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "PREDICTION ERROR ANALYSIS - COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"  Model           : "
        f"{args.name}"
    )

    print(
        f"  Test rows       : "
        f"{total:,}"
    )

    print(
        f"  TP              : "
        f"{tp_count:,}"
    )

    print(
        f"  TN              : "
        f"{tn_count:,}"
    )

    print(
        f"  FP              : "
        f"{fp_count:,}"
    )

    print(
        f"  FN              : "
        f"{fn_count:,}"
    )

    print(
        f"  Overall errors  : "
        f"{fp_count + fn_count:,} "
        f"("
        f"{(fp_count + fn_count) / total * 100:.4f}%"
        f")"
    )

    print(
        "\nProbability summary:"
    )

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

    print(
        "\nSummary saved:"
    )

    print(
        f"  {summary_file}"
    )

    print(
        "\n" + "=" * 70
    )


if __name__ == "__main__":
    main()