"""
error_analysis.py
-----------------
NetworkIDS - Primary Random Forest Error Analysis.

This script analyses prediction errors on the held-out test set
using the currently trained primary Random Forest model.

Primary inputs:
    config.TEST_DATASET_FILE
    config.RANDOM_FOREST_MODEL_FILE

Primary outputs:
    models/error_analysis/
        error_summary.json
        error_predictions.csv
        false_positive.csv
        false_negative.csv
        fp_feature_summary.csv
        fn_feature_summary.csv

The Random Forest feature set is defined entirely by:

    config.ML_FEATURES

Metadata such as:
    flow_id
    timestamp
    src_ip
    dst_ip

is retained for traceability but is NOT passed into the model.

Missing-value policy:
    NaN values are retained.
    No imputation is performed.

Classification:
    config.LABEL_BENIGN
    config.LABEL_ATTACK

The Attack probability column is determined from:

    model.classes_

This script does not assume that Attack is always stored
at probability column index 1.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# Project import
# ============================================================

sys.path.append(
    str(
        Path(__file__).resolve().parents[1]
    )
)

import src.config as config


# ============================================================
# Configuration
# ============================================================

TEST_FILE = (
    config.TEST_DATASET_FILE
)

MODEL_FILE = (
    config.RANDOM_FOREST_MODEL_FILE
)

OUTPUT_DIR = (
    config.MODELS_DIR
    / "error_analysis"
)

FEATURE_DTYPE = np.float32
LABEL_DTYPE = np.int8

TRACEABILITY_COLUMNS = [
    "flow_id",
    "timestamp",
    "src_ip",
    "dst_ip",
]


# ============================================================
# Helpers
# ============================================================

def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def label_to_text(label):
    """
    Convert configured label values into readable names.

    Numeric label mapping is taken from config.py.
    """

    label = int(label)

    if label == config.LABEL_BENIGN:
        return "Benign"

    if label == config.LABEL_ATTACK:
        return "Attack"

    return str(label)


def load_model():
    """
    Load the saved bare Random Forest model.
    """

    model_path = Path(
        MODEL_FILE
    )

    if not model_path.exists():
        raise FileNotFoundError(
            "Random Forest model not found:\n"
            f"{model_path}"
        )

    print(
        f"  Model: {model_path}"
    )

    model = joblib.load(
        model_path
    )

    return model


def validate_model_compatibility(model):
    """
    Verify that the saved Random Forest matches
    the current config.ML_FEATURES definition.
    """

    configured_features = list(
        config.ML_FEATURES
    )

    configured_count = len(
        configured_features
    )

    # --------------------------------------------------------
    # Feature count
    # --------------------------------------------------------

    if hasattr(
        model,
        "n_features_in_",
    ):

        model_count = int(
            model.n_features_in_
        )

        if model_count != configured_count:

            raise RuntimeError(
                "Model/config feature-count mismatch.\n"
                f"Model expects : {model_count}\n"
                f"Config defines: {configured_count}\n"
                "\nRetrain the model or restore the "
                "feature configuration used during training."
            )

    # --------------------------------------------------------
    # Feature names
    # --------------------------------------------------------

    if hasattr(
        model,
        "feature_names_in_",
    ):

        model_features = list(
            model.feature_names_in_
        )

        if model_features != configured_features:

            raise RuntimeError(
                "Model/config feature-order mismatch.\n"
                "\nThe current config.ML_FEATURES does not "
                "match the exact feature order used to train "
                "the saved Random Forest."
            )

    print(
        f"  Model/config features: "
        f"OK ({configured_count})"
    )


def get_attack_class_index(model):
    """
    Locate the Attack class inside model.classes_.
    """

    if not hasattr(
        model,
        "classes_",
    ):
        raise RuntimeError(
            "Loaded model does not contain classes_."
        )

    classes = list(
        model.classes_
    )

    if config.LABEL_ATTACK not in classes:

        raise RuntimeError(
            f"Configured Attack label "
            f"{config.LABEL_ATTACK} was not found "
            f"in model.classes_.\n"
            f"Model classes: {classes}"
        )

    return classes.index(
        config.LABEL_ATTACK
    )


def load_test_dataset(path):
    """
    Load only the columns required for:

        - model prediction
        - error analysis
        - flow traceability

    ML feature columns are loaded using float32.
    Label is loaded using int8.
    Metadata keeps its normal representation.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            "Test dataset not found:\n"
            f"{path}"
        )

    required_columns = [
        *TRACEABILITY_COLUMNS,
        *config.ML_FEATURES,
        config.LABEL_COLUMN,
    ]

    # Remove accidental duplicate column names.
    required_columns = list(
        dict.fromkeys(
            required_columns
        )
    )

    # --------------------------------------------------------
    # Validate header
    # --------------------------------------------------------

    header = pd.read_csv(
        path,
        nrows=0,
    )

    missing = [
        column
        for column in required_columns
        if column not in header.columns
    ]

    if missing:
        raise ValueError(
            "Test dataset is missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    # --------------------------------------------------------
    # Dtypes
    # --------------------------------------------------------

    dtype_map = {
        feature: FEATURE_DTYPE
        for feature in config.ML_FEATURES
    }

    dtype_map[
        config.LABEL_COLUMN
    ] = LABEL_DTYPE

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = pd.read_csv(
        path,
        usecols=required_columns,
        dtype=dtype_map,
        low_memory=False,
    )

    print(
        f"  Test rows: {len(df):,}"
    )

    return df


def validate_dataset(df):
    """
    Validate labels and feature values.
    """

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    label_series = df[
        config.LABEL_COLUMN
    ]

    if label_series.isna().any():

        count = int(
            label_series.isna().sum()
        )

        raise ValueError(
            f"Test dataset contains "
            f"{count:,} NaN labels."
        )

    allowed_labels = {
        config.LABEL_BENIGN,
        config.LABEL_ATTACK,
    }

    actual_labels = set(
        label_series.unique()
        .tolist()
    )

    if not actual_labels.issubset(
        allowed_labels
    ):

        unexpected = (
            actual_labels
            - allowed_labels
        )

        raise ValueError(
            "Unexpected labels found: "
            f"{sorted(unexpected)}"
        )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X = df[
        config.ML_FEATURES
    ]

    missing_feature_count = 0
    infinite_count = 0

    for feature in config.ML_FEATURES:

        values = (
            X[feature]
            .to_numpy(
                copy=False
            )
        )

        missing_feature_count += int(
            np.isnan(values).sum()
        )

        infinite_count += int(
            np.isinf(values).sum()
        )

    if infinite_count:

        raise ValueError(
            f"Found {infinite_count:,} "
            "infinite ML feature values."
        )

    total_feature_cells = (
        len(X)
        * len(config.ML_FEATURES)
    )

    nan_percentage = (
        (
            missing_feature_count
            / total_feature_cells
        )
        * 100.0
        if total_feature_cells
        else 0.0
    )

    print(
        f"  ML features        : "
        f"{len(config.ML_FEATURES)}"
    )

    print(
        f"  NaN feature cells  : "
        f"{missing_feature_count:,} "
        f"({nan_percentage:.4f}%)"
    )

    print(
        "  NaN values retained."
    )


def get_predictions(
    model,
    X,
    attack_class_index,
):
    """
    Generate predictions and Attack probabilities.
    """

    y_pred = model.predict(
        X
    )

    probabilities = model.predict_proba(
        X
    )

    attack_probability = probabilities[
        :,
        attack_class_index
    ]

    return (
        y_pred,
        attack_probability,
    )


def calculate_confusion_counts(
    y_true,
    y_pred,
):
    """
    Calculate TN, FP, FN, TP using the
    configured label ordering.
    """

    benign_label = (
        config.LABEL_BENIGN
    )

    attack_label = (
        config.LABEL_ATTACK
    )

    tn = int(
        (
            (y_true == benign_label)
            &
            (y_pred == benign_label)
        ).sum()
    )

    fp = int(
        (
            (y_true == benign_label)
            &
            (y_pred == attack_label)
        ).sum()
    )

    fn = int(
        (
            (y_true == attack_label)
            &
            (y_pred == benign_label)
        ).sum()
    )

    tp = int(
        (
            (y_true == attack_label)
            &
            (y_pred == attack_label)
        ).sum()
    )

    return (
        tn,
        fp,
        fn,
        tp,
    )


def numeric_profile(
    df,
    feature,
):
    """
    Return descriptive statistics for one feature.
    NaN values are counted separately.
    """

    values = pd.to_numeric(
        df[feature],
        errors="coerce",
    )

    valid = (
        values.dropna()
    )

    total = len(df)

    if len(valid) == 0:

        return {
            "count": 0,
            "missing": int(
                values.isna().sum()
            ),
            "missing_rate": (
                float(
                    values.isna().mean()
                )
                if total
                else 0.0
            ),
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }

    return {
        "count": int(
            len(valid)
        ),

        "missing": int(
            values.isna().sum()
        ),

        "missing_rate": float(
            values.isna().mean()
        ),

        "mean": float(
            valid.mean()
        ),

        "median": float(
            valid.median()
        ),

        "std": float(
            valid.std(
                ddof=0
            )
        ),

        "min": float(
            valid.min()
        ),

        "max": float(
            valid.max()
        ),
    }


def build_feature_comparison(
    error_df,
    reference_df,
):
    """
    Compare an error group against its
    corresponding correct population.

    FP:
        error_df     = False Positives
        reference_df = True Negatives

    FN:
        error_df     = False Negatives
        reference_df = True Positives

    The comparison is descriptive only.
    """

    rows = []

    for feature in config.ML_FEATURES:

        error_values = pd.to_numeric(
            error_df[feature],
            errors="coerce",
        )

        reference_values = pd.to_numeric(
            reference_df[feature],
            errors="coerce",
        )

        error_valid = (
            error_values.dropna()
        )

        reference_valid = (
            reference_values.dropna()
        )

        error_missing_rate = float(
            error_values.isna().mean()
        )

        reference_missing_rate = float(
            reference_values.isna().mean()
        )

        if len(error_valid):

            error_mean = float(
                error_valid.mean()
            )

            error_median = float(
                error_valid.median()
            )

        else:

            error_mean = None
            error_median = None

        if len(reference_valid):

            reference_mean = float(
                reference_valid.mean()
            )

            reference_median = float(
                reference_valid.median()
            )

        else:

            reference_mean = None
            reference_median = None

        if (
            error_mean is not None
            and reference_mean is not None
        ):

            mean_difference = (
                error_mean
                - reference_mean
            )

        else:

            mean_difference = None

        if (
            error_median is not None
            and reference_median is not None
        ):

            median_difference = (
                error_median
                - reference_median
            )

        else:

            median_difference = None

        rows.append(
            {
                "feature": feature,

                "error_count": int(
                    len(error_df)
                ),

                "reference_count": int(
                    len(reference_df)
                ),

                "error_mean":
                    error_mean,

                "reference_mean":
                    reference_mean,

                "mean_difference":
                    mean_difference,

                "error_median":
                    error_median,

                "reference_median":
                    reference_median,

                "median_difference":
                    median_difference,

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
        )

    comparison = pd.DataFrame(
        rows
    )

    # Useful descriptive ranking.
    reference_abs = (
        comparison[
            "reference_mean"
        ]
        .abs()
    )

    comparison[
        "relative_mean_difference_pct"
    ] = np.where(
        reference_abs > 1e-12,

        (
            comparison[
                "mean_difference"
            ].abs()
            /
            reference_abs
            * 100.0
        ),

        np.nan,
    )

    return comparison


def save_csv(
    df,
    path,
):
    """
    Save DataFrame to CSV.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        path,
        index=False,
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "NetworkIDS - PRIMARY RANDOM FOREST ERROR ANALYSIS"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. Load model
    # ========================================================

    print_section(
        "[1/7] Loading model"
    )

    config.validate_ml_features()

    model = load_model()

    validate_model_compatibility(
        model
    )

    attack_class_index = (
        get_attack_class_index(
            model
        )
    )

    print(
        f"  Model classes      : "
        f"{list(model.classes_)}"
    )

    print(
        f"  Benign label       : "
        f"{config.LABEL_BENIGN}"
    )

    print(
        f"  Attack label       : "
        f"{config.LABEL_ATTACK}"
    )

    print(
        f"  Attack class index : "
        f"{attack_class_index}"
    )

    # ========================================================
    # 2. Load test dataset
    # ========================================================

    print_section(
        "[2/7] Loading held-out test dataset"
    )

    test_df = load_test_dataset(
        TEST_FILE
    )

    validate_dataset(
        test_df
    )

    # ========================================================
    # 3. Prepare X and y
    # ========================================================

    print_section(
        "[3/7] Preparing test data"
    )

    X_test = (
        test_df[
            config.ML_FEATURES
        ]
        .copy()
        .astype(
            FEATURE_DTYPE
        )
    )

    y_test = (
        test_df[
            config.LABEL_COLUMN
        ]
        .copy()
        .astype(
            LABEL_DTYPE
        )
    )

    print(
        f"  X_test shape: "
        f"{X_test.shape}"
    )

    print(
        f"  X_test dtype: "
        f"{X_test.dtypes.iloc[0]}"
    )

    # ========================================================
    # 4. Predict
    # ========================================================

    print_section(
        "[4/7] Generating predictions"
    )

    print(
        "  Predicting held-out test set..."
    )

    (
        y_pred,
        attack_probability,
    ) = get_predictions(
        model,
        X_test,
        attack_class_index,
    )

    y_pred = np.asarray(
        y_pred
    )

    attack_probability = np.asarray(
        attack_probability
    )

    print(
        "  Prediction complete."
    )

    # --------------------------------------------------------
    # Prediction dataframe
    # --------------------------------------------------------

    prediction_df = pd.DataFrame(
        {
            "actual_label":
                y_test.to_numpy(),

            "predicted_label":
                y_pred,

            "actual_label_text":
                [
                    label_to_text(
                        value
                    )
                    for value
                    in y_test.to_numpy()
                ],

            "predicted_label_text":
                [
                    label_to_text(
                        value
                    )
                    for value
                    in y_pred
                ],

            "attack_probability":
                attack_probability,
        }
    )

    # --------------------------------------------------------
    # Traceability columns
    # --------------------------------------------------------

    for column in TRACEABILITY_COLUMNS:

        if column in test_df.columns:

            prediction_df.insert(
                0,
                column,
                test_df[column].to_numpy(),
            )

    # --------------------------------------------------------
    # ML feature columns
    # --------------------------------------------------------

    for feature in config.ML_FEATURES:

        prediction_df[
            feature
        ] = X_test[
            feature
        ].to_numpy()

    # ========================================================
    # 5. Identify TP / TN / FP / FN
    # ========================================================

    print_section(
        "[5/7] Identifying prediction errors"
    )

    benign_label = (
        config.LABEL_BENIGN
    )

    attack_label = (
        config.LABEL_ATTACK
    )

    true_negative_mask = (
        (y_test == benign_label)
        &
        (y_pred == benign_label)
    )

    false_positive_mask = (
        (y_test == benign_label)
        &
        (y_pred == attack_label)
    )

    false_negative_mask = (
        (y_test == attack_label)
        &
        (y_pred == benign_label)
    )

    true_positive_mask = (
        (y_test == attack_label)
        &
        (y_pred == attack_label)
    )

    tn = int(
        true_negative_mask.sum()
    )

    fp = int(
        false_positive_mask.sum()
    )

    fn = int(
        false_negative_mask.sum()
    )

    tp = int(
        true_positive_mask.sum()
    )

    print(
        f"  True Negative  : {tn:,}"
    )

    print(
        f"  False Positive : {fp:,}"
    )

    print(
        f"  False Negative : {fn:,}"
    )

    print(
        f"  True Positive  : {tp:,}"
    )

    # --------------------------------------------------------
    # Add error type
    # --------------------------------------------------------

    error_type = np.full(
        len(y_test),
        "Correct",
        dtype=object,
    )

    error_type[
        false_positive_mask.to_numpy()
    ] = "False Positive"

    error_type[
        false_negative_mask.to_numpy()
    ] = "False Negative"

    error_type[
        true_positive_mask.to_numpy()
    ] = "True Positive"

    error_type[
        true_negative_mask.to_numpy()
    ] = "True Negative"

    prediction_df[
        "error_type"
    ] = error_type

    # Put error_type near labels.
    front_columns = []

    for column in [
        "flow_id",
        "timestamp",
        "src_ip",
        "dst_ip",
        "actual_label",
        "actual_label_text",
        "predicted_label",
        "predicted_label_text",
        "error_type",
        "attack_probability",
    ]:

        if column in prediction_df.columns:

            front_columns.append(
                column
            )

    remaining_columns = [
        column
        for column in prediction_df.columns
        if column not in front_columns
    ]

    prediction_df = prediction_df[
        front_columns
        + remaining_columns
    ]

    # --------------------------------------------------------
    # Extract groups
    # --------------------------------------------------------

    false_positive_df = (
        prediction_df[
            false_positive_mask.to_numpy()
        ]
        .copy()
    )

    false_negative_df = (
        prediction_df[
            false_negative_mask.to_numpy()
        ]
        .copy()
    )

    true_positive_df = (
        prediction_df[
            true_positive_mask.to_numpy()
        ]
        .copy()
    )

    true_negative_df = (
        prediction_df[
            true_negative_mask.to_numpy()
        ]
        .copy()
    )

    print(
        f"\n  False Positive rows: "
        f"{len(false_positive_df):,}"
    )

    print(
        f"  False Negative rows: "
        f"{len(false_negative_df):,}"
    )

    # ========================================================
    # 6. Error feature profiles
    # ========================================================

    print_section(
        "[6/7] Analysing FP/FN feature profiles"
    )

    # --------------------------------------------------------
    # FP:
    # Compare False Positives against correctly
    # classified Benign traffic (True Negatives).
    # --------------------------------------------------------

    fp_feature_summary = (
        build_feature_comparison(
            false_positive_df,
            true_negative_df,
        )
    )

    # Rank by largest absolute relative mean difference.
    fp_feature_summary = (
        fp_feature_summary
        .sort_values(
            by=[
                "relative_mean_difference_pct",
                "mean_difference",
            ],
            ascending=False,
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # FN:
    # Compare False Negatives against correctly
    # classified Attack traffic (True Positives).
    # --------------------------------------------------------

    fn_feature_summary = (
        build_feature_comparison(
            false_negative_df,
            true_positive_df,
        )
    )

    fn_feature_summary = (
        fn_feature_summary
        .sort_values(
            by=[
                "relative_mean_difference_pct",
                "mean_difference",
            ],
            ascending=False,
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # 7. Save outputs
    # ========================================================

    print_section(
        "[7/7] Saving error analysis outputs"
    )

    # --------------------------------------------------------
    # All predictions
    # --------------------------------------------------------

    all_predictions_file = (
        OUTPUT_DIR
        / "error_predictions.csv"
    )

    false_positive_file = (
        OUTPUT_DIR
        / "false_positive.csv"
    )

    false_negative_file = (
        OUTPUT_DIR
        / "false_negative.csv"
    )

    fp_feature_file = (
        OUTPUT_DIR
        / "fp_feature_summary.csv"
    )

    fn_feature_file = (
        OUTPUT_DIR
        / "fn_feature_summary.csv"
    )

    summary_file = (
        OUTPUT_DIR
        / "error_summary.json"
    )

    save_csv(
        prediction_df,
        all_predictions_file,
    )

    save_csv(
        false_positive_df,
        false_positive_file,
    )

    save_csv(
        false_negative_df,
        false_negative_file,
    )

    save_csv(
        fp_feature_summary,
        fp_feature_file,
    )

    save_csv(
        fn_feature_summary,
        fn_feature_file,
    )

    # --------------------------------------------------------
    # Rates
    # --------------------------------------------------------

    benign_total = (
        tn + fp
    )

    attack_total = (
        fn + tp
    )

    total = (
        benign_total
        + attack_total
    )

    fpr = (
        fp / benign_total
        if benign_total
        else 0.0
    )

    fnr = (
        fn / attack_total
        if attack_total
        else 0.0
    )

    accuracy = (
        (tn + tp) / total
        if total
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else 0.0
    )

    f1 = (
        2.0
        * precision
        * recall
        /
        (precision + recall)
        if (precision + recall)
        else 0.0
    )

    # --------------------------------------------------------
    # Summary JSON
    # --------------------------------------------------------

    summary = {
        "model": "RandomForestClassifier",

        "model_path":
            str(
                Path(
                    MODEL_FILE
                ).resolve()
            ),

        "test_dataset":
            str(
                Path(
                    TEST_FILE
                ).resolve()
            ),

        "feature_count":
            len(
                config.ML_FEATURES
            ),

        "features":
            list(
                config.ML_FEATURES
            ),

        "labels": {
            "benign":
                int(
                    config.LABEL_BENIGN
                ),

            "attack":
                int(
                    config.LABEL_ATTACK
                ),
        },

        "test_rows":
            int(total),

        "confusion_matrix": {
            "true_negative":
                tn,

            "false_positive":
                fp,

            "false_negative":
                fn,

            "true_positive":
                tp,
        },

        "metrics": {
            "accuracy":
                float(accuracy),

            "precision":
                float(precision),

            "recall":
                float(recall),

            "f1":
                float(f1),

            "fpr":
                float(fpr),

            "fnr":
                float(fnr),
        },

        "class_counts": {
            "benign_total":
                int(benign_total),

            "attack_total":
                int(attack_total),
        },

        "error_counts": {
            "false_positive":
                int(fp),

            "false_negative":
                int(fn),
        },

        "attack_probability": {
            "min":
                float(
                    np.min(
                        attack_probability
                    )
                ),

            "max":
                float(
                    np.max(
                        attack_probability
                    )
                ),

            "mean":
                float(
                    np.mean(
                        attack_probability
                    )
                ),

            "median":
                float(
                    np.median(
                        attack_probability
                    )
                ),
        },

        "files": {
            "all_predictions":
                str(
                    all_predictions_file
                ),

            "false_positive":
                str(
                    false_positive_file
                ),

            "false_negative":
                str(
                    false_negative_file
                ),

            "fp_feature_summary":
                str(
                    fp_feature_file
                ),

            "fn_feature_summary":
                str(
                    fn_feature_file
                ),
        },
    }

    with open(
        summary_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    # ========================================================
    # Terminal summary
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "ERROR ANALYSIS SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"\n  Test rows       : "
        f"{total:,}"
    )

    print(
        f"  Benign total    : "
        f"{benign_total:,}"
    )

    print(
        f"  Attack total    : "
        f"{attack_total:,}"
    )

    print(
        "\n  Confusion Matrix:"
    )

    print(
        "                 Predicted"
    )

    print(
        f"                 "
        f"{label_to_text(benign_label):<10} "
        f"{label_to_text(attack_label):<10}"
    )

    print(
        f"Actual "
        f"{label_to_text(benign_label):<8} "
        f"{tn:>10,} "
        f"{fp:>10,}"
    )

    print(
        f"Actual "
        f"{label_to_text(attack_label):<8} "
        f"{fn:>10,} "
        f"{tp:>10,}"
    )

    print(
        "\n  Metrics:"
    )

    print(
        f"    Accuracy  : "
        f"{accuracy:.6f}"
    )

    print(
        f"    Precision : "
        f"{precision:.6f}"
    )

    print(
        f"    Recall    : "
        f"{recall:.6f}"
    )

    print(
        f"    F1-score  : "
        f"{f1:.6f}"
    )

    print(
        f"    FPR       : "
        f"{fpr:.6f} "
        f"({fpr * 100:.4f}%)"
    )

    print(
        f"    FNR       : "
        f"{fnr:.6f} "
        f"({fnr * 100:.4f}%)"
    )

    # --------------------------------------------------------
    # Top FP profile differences
    # --------------------------------------------------------

    print(
        "\n  Top FP feature differences:"
    )

    for _, row in (
        fp_feature_summary
        .head(10)
        .iterrows()
    ):

        print(
            f"    {row['feature']:<35} "
            f"mean_diff={row['mean_difference']!s:<14} "
            f"missing_diff="
            f"{row['missing_rate_difference']:.6f}"
        )

    # --------------------------------------------------------
    # Top FN profile differences
    # --------------------------------------------------------

    print(
        "\n  Top FN feature differences:"
    )

    for _, row in (
        fn_feature_summary
        .head(10)
        .iterrows()
    ):

        print(
            f"    {row['feature']:<35} "
            f"mean_diff={row['mean_difference']!s:<14} "
            f"missing_diff="
            f"{row['missing_rate_difference']:.6f}"
        )

    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "FILES SAVED"
    )

    print(
        "-" * 70
    )

    print(
        f"  All predictions : "
        f"{all_predictions_file}"
    )

    print(
        f"  False positives: "
        f"{false_positive_file}"
    )

    print(
        f"  False negatives: "
        f"{false_negative_file}"
    )

    print(
        f"  FP profile     : "
        f"{fp_feature_file}"
    )

    print(
        f"  FN profile     : "
        f"{fn_feature_file}"
    )

    print(
        f"  Summary JSON   : "
        f"{summary_file}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "ERROR ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()