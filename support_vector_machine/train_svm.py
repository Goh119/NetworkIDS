"""
NetworkIDS - Calibrated Linear Support Vector Machine
training and evaluation.

Location:
    support_vector_machine/train_svm.py

Default inputs:
    data/train/training_dataset.csv
    data/test/test_multiday.csv

Default outputs:
    support_vector_machine/
        svm_model.joblib
        svm_model_meta.json
        svm_model_report.txt

The calibrated Linear SVM follows the same experimental framework
as the Random Forest and Decision Tree comparison:

    - same final training dataset
    - same predefined held-out test dataset
    - same config.ML_FEATURES
    - same binary label mapping
    - same primary evaluation metrics

Important preprocessing difference:

    Random Forest and Decision Tree can retain NaN values.

    LinearSVC cannot directly train on NaN values. Therefore the
    SVM uses model-specific preprocessing:

        1. Median imputation
        2. Preserve fully-empty features
        3. Standard scaling
        4. LinearSVC
        5. Sigmoid probability calibration

The preprocessing pipeline is placed INSIDE the estimator passed
to CalibratedClassifierCV. Therefore each calibration fold fits
its own imputation and scaling parameters using only that fold's
training portion.

Fully-empty training features are preserved rather than dropped.
SimpleImputer with keep_empty_features=True assigns 0 to such
fully-empty numerical features.

No additional features or missing-value indicators are added.

LinearSVC does not natively provide predict_proba(). Therefore
CalibratedClassifierCV is used to provide calibrated class
probabilities suitable for:

    - P(Benign)
    - P(Attack)
    - confidence display
    - ROC-AUC evaluation
    - possible future deployment

Calibration:
    method   = sigmoid
    cv       = 3
    ensemble = False

With ensemble=False, cross-validation is used to obtain unbiased
scores for calibration, while the final base estimator is fitted
on the complete training dataset.

Classification:
    config.LABEL_BENIGN = Benign
    config.LABEL_ATTACK = Attack

Usage:
    python support_vector_machine/train_svm.py

Optional:
    python support_vector_machine/train_svm.py \
        --train path/to/train.csv \
        --test path/to/test.csv \
        --model path/to/model.joblib \
        --meta path/to/model_meta.json \
        --report path/to/model_report.txt
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


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
# Primary Dataset Paths
# ============================================================

TRAIN_FILE = (
    config.TRAIN_DATASET_FILE
)

TEST_FILE = (
    config.TEST_DATASET_FILE
)


# ============================================================
# Linear SVM Parameters
#
# LinearSVC is selected instead of kernel SVC because the
# training dataset contains more than 3 million flows.
#
# LinearSVC provides a practical non-tree comparison model for
# the Decision Tree and Random Forest.
# ============================================================

RANDOM_STATE = 42

C = 1.0

PENALTY = "l2"

LOSS = "squared_hinge"

CLASS_WEIGHT = "balanced"

MAX_ITER = 20_000

TOL = 1e-4

DUAL = "auto"


# ============================================================
# SVM Preprocessing
# ============================================================

IMPUTER_STRATEGY = "median"

KEEP_EMPTY_FEATURES = True

SCALER_WITH_MEAN = True

SCALER_WITH_STD = True


# ============================================================
# Probability Calibration
#
# sigmoid:
#     Platt-style calibration suitable for SVM decision scores.
#
# cv=3:
#     Three-fold calibration.
#
# ensemble=False:
#     Use CV-generated unbiased predictions for calibration,
#     then fit one final base estimator on the complete training
#     dataset.
#
# This is more deployment-friendly than retaining three
# independently calibrated classifiers.
# ============================================================

CALIBRATION_METHOD = "sigmoid"

CALIBRATION_CV = 3

CALIBRATION_ENSEMBLE = False

CALIBRATION_N_JOBS = -1


# ============================================================
# Data types
# ============================================================

FEATURE_DTYPE = np.float32

LABEL_DTYPE = np.int8


# ============================================================
# Helpers
# ============================================================

def print_section(
    title,
):
    """
    Print a formatted console section.
    """

    print()

    print(
        "=" * 70
    )

    print(
        title
    )

    print(
        "=" * 70
    )


# ============================================================
# Label helper
# ============================================================

def label_to_text(
    label,
):
    """
    Convert configured numeric labels into readable names.
    """

    label = int(
        label
    )

    if (
        label
        == config.LABEL_BENIGN
    ):

        return "Benign"

    if (
        label
        == config.LABEL_ATTACK
    ):

        return "Attack"

    return str(
        label
    )


# ============================================================
# Dataset loading
# ============================================================

def load_training_split(
    path,
    name,
):
    """
    Load only columns required by the calibrated Linear SVM.

    Metadata such as:

        flow_id
        timestamp
        src_ip
        dst_ip
        src_mac
        dst_mac
        day

    is intentionally not loaded.

    Only:

        config.ML_FEATURES
        config.LABEL_COLUMN

    are loaded.
    """

    path = Path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"{name} dataset not found:\n"
            f"{path}"
        )

    required_columns = [
        *config.ML_FEATURES,
        config.LABEL_COLUMN,
    ]

    # --------------------------------------------------------
    # Read header only
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
            f"{name}: missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    # --------------------------------------------------------
    # Explicit dtypes
    # --------------------------------------------------------

    dtype_map = {

        feature:
            FEATURE_DTYPE

        for feature
        in config.ML_FEATURES
    }

    dtype_map[
        config.LABEL_COLUMN
    ] = LABEL_DTYPE

    # --------------------------------------------------------
    # Load only required columns
    # --------------------------------------------------------

    df = pd.read_csv(
        path,
        usecols=required_columns,
        dtype=dtype_map,
        low_memory=False,
    )

    print(
        f"  {name:<6}: "
        f"{len(df):>10,} rows "
        f"from {path.name}"
    )

    return df


# ============================================================
# Label validation
# ============================================================

def validate_labels(
    df,
    name,
):
    """
    Validate binary labels.
    """

    y = df[
        config.LABEL_COLUMN
    ]

    if y.isna().any():

        count = int(
            y.isna().sum()
        )

        raise ValueError(
            f"{name}: "
            f"{count:,} NaN labels found."
        )

    expected = {
        config.LABEL_BENIGN,
        config.LABEL_ATTACK,
    }

    actual = set(
        y.unique()
        .tolist()
    )

    if not actual.issubset(
        expected
    ):

        unexpected = (
            actual
            - expected
        )

        raise ValueError(
            f"{name}: unexpected labels: "
            f"{sorted(unexpected)}"
        )

    if len(
        actual
    ) != 2:

        raise ValueError(
            f"{name}: both "
            f"{label_to_text(config.LABEL_BENIGN)} and "
            f"{label_to_text(config.LABEL_ATTACK)} "
            "classes must be present."
        )


# ============================================================
# Feature validation
# ============================================================

def validate_features(
    df,
    name,
):
    """
    Validate feature values.

    NaN:
        allowed in the raw dataset.

        LinearSVC cannot directly consume NaN values.
        Missing values are handled later inside the
        model-specific preprocessing pipeline.

    Inf:
        rejected.
    """

    missing = [

        feature

        for feature in config.ML_FEATURES

        if feature not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{name}: missing ML features:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing
            )
        )

    X = df[
        config.ML_FEATURES
    ]

    # --------------------------------------------------------
    # NaN count
    # --------------------------------------------------------

    nan_count = int(
        X.isna()
        .sum()
        .sum()
    )

    if nan_count:

        print(
            f"  {name}: "
            f"{nan_count:,} NaN feature cells detected."
        )

        print(
            "  SVM-specific median imputation will be applied "
            "inside each calibrated training pipeline."
        )

    else:

        print(
            f"  {name}: "
            "no NaN feature cells."
        )

    # --------------------------------------------------------
    # Fully-empty features
    # --------------------------------------------------------

    empty_features = [

        feature

        for feature in config.ML_FEATURES

        if X[
            feature
        ].isna().all()
    ]

    if empty_features:

        print(
            f"  {name}: "
            f"{len(empty_features)} fully-empty feature(s):"
        )

        for feature in empty_features:

            print(
                f"    - {feature}"
            )

        print(
            "  They will be preserved by "
            "keep_empty_features=True."
        )

    # --------------------------------------------------------
    # Infinite values
    # --------------------------------------------------------

    inf_count = 0

    for feature in config.ML_FEATURES:

        values = (
            X[
                feature
            ]
            .to_numpy(
                copy=False
            )
        )

        inf_count += int(
            np.isinf(
                values
            ).sum()
        )

    if inf_count:

        raise ValueError(
            f"{name}: "
            f"{inf_count:,} infinite feature values found."
        )

    return {

        "nan_count":
            nan_count,

        "fully_empty_features":
            empty_features,
    }


# ============================================================
# Prepare X / y
# ============================================================

def prepare_xy(
    df,
    name,
):
    """
    Prepare raw X and y.

    X:
        original config.ML_FEATURES
        float32
        NaN still retained at this stage

    y:
        binary labels
        int8

    Important:

    Imputation and scaling are NOT performed here.

    They are part of the base sklearn Pipeline passed into
    CalibratedClassifierCV so that each calibration fold fits
    its preprocessing using only its own training partition.
    """

    validate_labels(
        df,
        name,
    )

    validation_info = validate_features(
        df,
        name,
    )

    X = (
        df[
            config.ML_FEATURES
        ]
        .copy()
    )

    y = (
        df[
            config.LABEL_COLUMN
        ]
        .copy()
    )

    X = X.astype(
        FEATURE_DTYPE
    )

    y = y.astype(
        LABEL_DTYPE
    )

    print(
        f"  {name} X shape: "
        f"{X.shape}"
    )

    print(
        f"  {name} X dtype: "
        f"{X.dtypes.iloc[0]}"
    )

    print(
        f"  {name} y shape: "
        f"{y.shape}"
    )

    print(
        f"  {name} y dtype: "
        f"{y.dtype}"
    )

    return (
        X,
        y,
        validation_info,
    )


# ============================================================
# Label distribution
# ============================================================

def print_label_distribution(
    y,
    name,
):
    """
    Print binary class distribution.
    """

    counts = (
        y.value_counts()
        .sort_index()
    )

    print(
        f"  {name}:"
    )

    for label, count in counts.items():

        label_text = label_to_text(
            label
        )

        percentage = (
            100.0
            * count
            / len(y)
        )

        print(
            f"    {label_text:<8}: "
            f"{count:>10,} "
            f"({percentage:6.2f}%)"
        )


# ============================================================
# Attack probability
# ============================================================

def get_attack_probability(
    model,
    X,
):
    """
    Return calibrated P(Attack).

    CalibratedClassifierCV provides predict_proba().
    The Attack probability column is determined from
    model.classes_ rather than assuming column index 1.
    """

    probabilities = model.predict_proba(
        X
    )

    classes = list(
        model.classes_
    )

    if (
        config.LABEL_ATTACK
        not in classes
    ):

        raise RuntimeError(
            f"{label_to_text(config.LABEL_ATTACK)} "
            "class not found in model.classes_."
        )

    attack_index = classes.index(
        config.LABEL_ATTACK
    )

    return probabilities[
        :,
        attack_index
    ]


# ============================================================
# Extract final fitted LinearSVC information
# ============================================================

def get_final_linear_svc(
    calibrated_model,
):
    """
    Try to retrieve the final fitted LinearSVC stored inside
    CalibratedClassifierCV.

    With ensemble=False, calibrated_classifiers_ should contain
    one fitted classifier/calibrator pair.

    Returns:
        fitted LinearSVC or None
    """

    calibrated_models = getattr(
        calibrated_model,
        "calibrated_classifiers_",
        None,
    )

    if not calibrated_models:

        return None

    calibrated_pair = calibrated_models[
        0
    ]

    fitted_estimator = getattr(
        calibrated_pair,
        "estimator",
        None,
    )

    if fitted_estimator is None:

        return None

    if not hasattr(
        fitted_estimator,
        "named_steps",
    ):

        return None

    return fitted_estimator.named_steps.get(
        "svm"
    )


# ============================================================
# FPR / FNR
# ============================================================

def calculate_rates(
    cm,
):
    """
    Calculate FPR and FNR.

    Confusion matrix:

        [[TN, FP],
         [FN, TP]]
    """

    tn = int(
        cm[
            0,
            0
        ]
    )

    fp = int(
        cm[
            0,
            1
        ]
    )

    fn = int(
        cm[
            1,
            0
        ]
    )

    tp = int(
        cm[
            1,
            1
        ]
    )

    fpr = (
        fp
        / (
            fp
            + tn
        )
        if (
            fp
            + tn
        )
        else 0.0
    )

    fnr = (
        fn
        / (
            fn
            + tp
        )
        if (
            fn
            + tp
        )
        else 0.0
    )

    return (
        fpr,
        fnr,
    )


# ============================================================
# Save human-readable report
# ============================================================

def save_report(
    path,
    train_file,
    test_file,
    train_size,
    test_size,
    y_train,
    y_test,
    train_validation,
    test_validation,
    accuracy,
    precision,
    recall,
    f1,
    auc,
    fpr,
    fnr,
    cm,
    classification_report_text,
    svm_iterations,
):
    """
    Save the complete calibrated Linear SVM report.
    """

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    benign_text = label_to_text(
        config.LABEL_BENIGN
    )

    attack_text = label_to_text(
        config.LABEL_ATTACK
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as report:

        report.write(
            "NetworkIDS - Calibrated Linear SVM Training Report\n"
        )

        report.write(
            "=" * 70
            + "\n\n"
        )

        # ====================================================
        # Model configuration
        # ====================================================

        report.write(
            "Model Configuration\n"
        )

        report.write(
            "-" * 70
            + "\n"
        )

        report.write(
            "Model              : CalibratedClassifierCV\n"
        )

        report.write(
            "Base estimator     : LinearSVC\n"
        )

        report.write(
            f"C                  : {C}\n"
        )

        report.write(
            f"penalty            : {PENALTY}\n"
        )

        report.write(
            f"loss               : {LOSS}\n"
        )

        report.write(
            f"class_weight       : {CLASS_WEIGHT}\n"
        )

        report.write(
            f"dual               : {DUAL}\n"
        )

        report.write(
            f"tol                : {TOL}\n"
        )

        report.write(
            f"max_iter           : {MAX_ITER}\n"
        )

        report.write(
            f"random_state       : {RANDOM_STATE}\n"
        )

        report.write(
            f"actual_iterations  : "
            f"{svm_iterations if svm_iterations is not None else 'Unavailable'}\n"
        )

        report.write(
            f"scikit-learn       : {sklearn.__version__}\n"
        )

        report.write(
            f"ML feature count   : "
            f"{len(config.ML_FEATURES)}\n"
        )

        # ====================================================
        # Calibration configuration
        # ====================================================

        report.write(
            "\nProbability Calibration\n"
        )

        report.write(
            "-" * 70
            + "\n"
        )

        report.write(
            f"Method             : "
            f"{CALIBRATION_METHOD}\n"
        )

        report.write(
            f"CV folds           : "
            f"{CALIBRATION_CV}\n"
        )

        report.write(
            f"Ensemble           : "
            f"{CALIBRATION_ENSEMBLE}\n"
        )

        report.write(
            f"Calibration jobs   : "
            f"{CALIBRATION_N_JOBS}\n"
        )

        report.write(
            "Probability output : "
            "Calibrated P(Benign) and P(Attack)\n"
        )

        report.write(
            "Calibration design : "
            "Cross-validation is used to obtain unbiased "
            "classifier scores for sigmoid calibration. "
            "With ensemble=False, one final base estimator is "
            "trained on the complete training dataset for "
            "deployment-oriented inference.\n"
        )

        # ====================================================
        # Comparison purpose
        # ====================================================

        report.write(
            "\nComparison Purpose\n"
        )

        report.write(
            "-" * 70
            + "\n"
        )

        report.write(
            "The calibrated Linear SVM is evaluated as a "
            "deployment-capable non-tree alternative to the "
            "Decision Tree and primary Random Forest. "
            "All three models use the same final training "
            "dataset, held-out test dataset, 42-feature schema "
            "and binary labels. Model-specific preprocessing "
            "is applied only where required by the estimator.\n"
        )

        # ====================================================
        # Preprocessing
        # ====================================================

        report.write(
            "\nSVM-Specific Preprocessing\n"
        )

        report.write(
            "-" * 70
            + "\n"
        )

        report.write(
            f"Imputer            : SimpleImputer\n"
        )

        report.write(
            f"Imputer strategy   : "
            f"{IMPUTER_STRATEGY}\n"
        )

        report.write(
            f"Keep empty features: "
            f"{KEEP_EMPTY_FEATURES}\n"
        )

        report.write(
            "Fully-empty fill   : "
            "0 for fully-empty numerical training features\n"
        )

        report.write(
            "Scaler             : StandardScaler\n"
        )

        report.write(
            f"Scaler with_mean   : "
            f"{SCALER_WITH_MEAN}\n"
        )

        report.write(
            f"Scaler with_std    : "
            f"{SCALER_WITH_STD}\n"
        )

        report.write(
            "Feature expansion  : None; no missing indicators "
            "or additional model features are added.\n"
        )

        report.write(
            "Leakage control    : Imputer and scaler are inside "
            "the estimator supplied to CalibratedClassifierCV, "
            "so calibration folds fit preprocessing only from "
            "their corresponding training partitions.\n"
        )

        # ====================================================
        # Labels
        # ====================================================

        report.write(
            "\nLabel Configuration\n"
        )

        report.write(
            "-" * 70
            + "\n"
        )

        report.write(
            f"{benign_text:<18}: "
            f"{config.LABEL_BENIGN}\n"
        )

        report.write(
            f"{attack_text:<18}: "
            f"{config.LABEL_ATTACK}\n"
        )

        # ====================================================
        # Dataset
        # ====================================================

        report.write(
            "\nDataset\n"
        )

        report.write(
            "-" * 70
            + "\n"
        )

        report.write(
            f"Training file      : "
            f"{Path(train_file).resolve()}\n"
        )

        report.write(
            f"Test file          : "
            f"{Path(test_file).resolve()}\n"
        )

        report.write(
            f"Training rows      : "
            f"{train_size:,}\n"
        )

        report.write(
            f"Test rows          : "
            f"{test_size:,}\n"
        )

        report.write(
            "Evaluation         : "
            "Predefined held-out test dataset; "
            "test data was not used for model fitting "
            "or probability calibration.\n"
        )

        # ====================================================
        # Missing values
        # ====================================================

        report.write(
            "\nMissing-Value Summary\n"
        )

        report.write(
            "-" * 70
            + "\n"
        )

        report.write(
            f"Training NaN cells : "
            f"{train_validation['nan_count']:,}\n"
        )

        report.write(
            f"Test NaN cells     : "
            f"{test_validation['nan_count']:,}\n"
        )

        report.write(
            "Fully-empty training features:\n"
        )

        if train_validation[
            "fully_empty_features"
        ]:

            for feature in train_validation[
                "fully_empty_features"
            ]:

                report.write(
                    f"  - {feature}\n"
                )

        else:

            report.write(
                "  None\n"
            )

        # ====================================================
        # Label distributions
        # ====================================================

        report.write(
            "\nTrain Label Distribution\n"
        )

        report.write(
            "-" * 70
            + "\n"
        )

        for label, count in (
            y_train
            .value_counts()
            .sort_index()
            .items()
        ):

            label_value = int(
                label
            )

            report.write(
                f"{label_value} "
                f"{label_to_text(label_value):<10}"
                f"{int(count):>12,}\n"
            )

        report.write(
            "\nTest Label Distribution\n"
        )

        report.write(
            "-" * 70
            + "\n"
        )

        for label, count in (
            y_test
            .value_counts()
            .sort_index()
            .items()
        ):

            label_value = int(
                label
            )

            report.write(
                f"{label_value} "
                f"{label_to_text(label_value):<10}"
                f"{int(count):>12,}\n"
            )

        # ====================================================
        # Metrics
        # ====================================================

        report.write(
            "\nClassification Metrics\n"
        )

        report.write(
            "-" * 70
            + "\n"
        )

        report.write(
            f"Accuracy            : "
            f"{accuracy:.6f}\n"
        )

        report.write(
            f"Precision           : "
            f"{precision:.6f}\n"
        )

        report.write(
            f"Recall              : "
            f"{recall:.6f}\n"
        )

        report.write(
            f"F1-score            : "
            f"{f1:.6f}\n"
        )

        report.write(
            f"AUC-ROC             : "
            f"{auc:.6f}\n"
        )

        report.write(
            f"False Positive Rate : "
            f"{fpr:.6f}\n"
        )

        report.write(
            f"False Negative Rate : "
            f"{fnr:.6f}\n"
        )

        report.write(
            "AUC score source    : "
            "Calibrated P(Attack)\n"
        )

        # ====================================================
        # Confusion matrix
        # ====================================================

        report.write(
            "\nConfusion Matrix\n"
        )

        report.write(
            "-" * 70
            + "\n"
        )

        report.write(
            "                 Predicted\n"
        )

        report.write(
            f"                 "
            f"{benign_text:<10} "
            f"{attack_text:<10}\n"
        )

        report.write(
            f"Actual {benign_text:<8} "
            f"{cm[0, 0]:>10,} "
            f"{cm[0, 1]:>10,}\n"
        )

        report.write(
            f"Actual {attack_text:<8} "
            f"{cm[1, 0]:>10,} "
            f"{cm[1, 1]:>10,}\n"
        )

        # ====================================================
        # Classification report
        # ====================================================

        report.write(
            "\nClassification Report\n"
        )

        report.write(
            "-" * 70
            + "\n"
        )

        report.write(
            classification_report_text
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the calibrated "
            "NetworkIDS Linear SVM."
        )
    )

    # ========================================================
    # Dataset arguments
    # ========================================================

    parser.add_argument(
        "--train",
        default=str(
            TRAIN_FILE
        ),
        help="Training CSV path.",
    )

    parser.add_argument(
        "--test",
        default=str(
            TEST_FILE
        ),
        help="Held-out test CSV path.",
    )

    # ========================================================
    # Output arguments
    # ========================================================

    parser.add_argument(
        "--model",
        default=str(
            config.SUPPORT_VECTOR_MACHINE_MODEL_FILE
        ),
        help="Output .joblib model path.",
    )

    parser.add_argument(
        "--meta",
        default=str(
            config.SUPPORT_VECTOR_MACHINE_META_FILE
        ),
        help="Output metadata JSON path.",
    )

    parser.add_argument(
        "--report",
        default=str(
            config.SUPPORT_VECTOR_MACHINE_REPORT_FILE
        ),
        help="Output human-readable report path.",
    )

    args = parser.parse_args()

    # ========================================================
    # Output overwrite protection
    # ========================================================

    output_paths = [

        Path(
            args.model
        ),

        Path(
            args.meta
        ),

        Path(
            args.report
        ),
    ]

    existing_outputs = [

        path

        for path in output_paths

        if path.exists()
    ]

    if existing_outputs:

        raise FileExistsError(
            "Refusing to overwrite existing calibrated "
            "Linear SVM outputs:\n"
            + "\n".join(
                f"  - {path}"
                for path in existing_outputs
            )
        )

    # ========================================================
    # Header
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "NetworkIDS - CALIBRATED LINEAR SVM TRAINING"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # 1. Configuration
    # ========================================================

    print_section(
        "[1/6] Configuration"
    )

    config.validate_ml_features()

    print(
        f"  ML feature count : "
        f"{len(config.ML_FEATURES)}"
    )

    print(
        f"  {label_to_text(config.LABEL_BENIGN):<16}: "
        f"{config.LABEL_BENIGN}"
    )

    print(
        f"  {label_to_text(config.LABEL_ATTACK):<16}: "
        f"{config.LABEL_ATTACK}"
    )

    print(
        f"  Train dataset    : "
        f"{args.train}"
    )

    print(
        f"  Test dataset     : "
        f"{args.test}"
    )

    print(
        f"  SVM directory    : "
        f"{config.SUPPORT_VECTOR_MACHINE_DIR}"
    )

    print(
        f"  Model output     : "
        f"{args.model}"
    )

    print(
        f"  Metadata output  : "
        f"{args.meta}"
    )

    print(
        f"  Report output    : "
        f"{args.report}"
    )

    # ========================================================
    # 2. Load datasets
    # ========================================================

    print_section(
        "[2/6] Loading datasets"
    )

    train_df = load_training_split(
        Path(
            args.train
        ),
        "Train",
    )

    test_df = load_training_split(
        Path(
            args.test
        ),
        "Test",
    )

    # ========================================================
    # 3. Prepare X/y
    # ========================================================

    print_section(
        "[3/6] Preparing X and y"
    )

    (
        X_train,
        y_train,
        train_validation,
    ) = prepare_xy(
        train_df,
        "Train",
    )

    (
        X_test,
        y_test,
        test_validation,
    ) = prepare_xy(
        test_df,
        "Test",
    )

    print()

    print_label_distribution(
        y_train,
        "Train",
    )

    print_label_distribution(
        y_test,
        "Test",
    )

    print()

    print(
        f"  X_train shape: "
        f"{X_train.shape}"
    )

    print(
        f"  X_test shape : "
        f"{X_test.shape}"
    )

    del train_df
    del test_df

    # ========================================================
    # 4. Train
    # ========================================================

    print_section(
        "[4/6] TRAINING CALIBRATED LINEAR SVM"
    )

    print(
        f"  C                   = "
        f"{C}"
    )

    print(
        f"  penalty             = "
        f"{PENALTY}"
    )

    print(
        f"  loss                = "
        f"{LOSS}"
    )

    print(
        f"  class_weight        = "
        f"{CLASS_WEIGHT}"
    )

    print(
        f"  dual                = "
        f"{DUAL}"
    )

    print(
        f"  tol                 = "
        f"{TOL}"
    )

    print(
        f"  max_iter            = "
        f"{MAX_ITER}"
    )

    print(
        f"  random_state        = "
        f"{RANDOM_STATE}"
    )

    print(
        f"  imputer             = "
        f"{IMPUTER_STRATEGY}"
    )

    print(
        f"  keep_empty_features = "
        f"{KEEP_EMPTY_FEATURES}"
    )

    print(
        "  scaler              = "
        "StandardScaler"
    )

    print(
        f"  calibration method  = "
        f"{CALIBRATION_METHOD}"
    )

    print(
        f"  calibration CV      = "
        f"{CALIBRATION_CV}"
    )

    print(
        f"  calibration ensemble= "
        f"{CALIBRATION_ENSEMBLE}"
    )

    print(
        f"  calibration n_jobs  = "
        f"{CALIBRATION_N_JOBS}"
    )

    print(
        f"  n_features          = "
        f"{len(config.ML_FEATURES)}"
    )

    print(
        f"  train rows          = "
        f"{len(X_train):,}"
    )

    # ========================================================
    # Base estimator
    #
    # IMPORTANT:
    #
    # Preprocessing is INSIDE this pipeline.
    #
    # CalibratedClassifierCV clones this complete pipeline for
    # calibration folds, preventing preprocessing statistics
    # from being fitted on calibration validation data.
    # ========================================================

    base_estimator = Pipeline(
        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy=IMPUTER_STRATEGY,
                    keep_empty_features=(
                        KEEP_EMPTY_FEATURES
                    ),
                ),
            ),

            (
                "scaler",
                StandardScaler(
                    with_mean=SCALER_WITH_MEAN,
                    with_std=SCALER_WITH_STD,
                ),
            ),

            (
                "svm",
                LinearSVC(
                    C=C,
                    penalty=PENALTY,
                    loss=LOSS,
                    dual=DUAL,
                    class_weight=CLASS_WEIGHT,
                    random_state=RANDOM_STATE,
                    max_iter=MAX_ITER,
                    tol=TOL,
                ),
            ),
        ]
    )

    # ========================================================
    # Calibrated classifier
    # ========================================================

    clf = CalibratedClassifierCV(
        estimator=base_estimator,
        method=CALIBRATION_METHOD,
        cv=CALIBRATION_CV,
        n_jobs=CALIBRATION_N_JOBS,
        ensemble=CALIBRATION_ENSEMBLE,
    )

    print(
        "\n  Fitting calibrated Linear SVM..."
    )

    print(
        "  This includes cross-validated probability "
        "calibration."
    )

    clf.fit(
        X_train,
        y_train,
    )

    print(
        "\n  TRAINING COMPLETE."
    )

    # --------------------------------------------------------
    # Try to retrieve final LinearSVC iteration information.
    # --------------------------------------------------------

    final_svm = get_final_linear_svc(
        clf
    )

    svm_iterations = None

    if (
        final_svm is not None
        and hasattr(
            final_svm,
            "n_iter_",
        )
    ):

        svm_iterations = int(
            np.max(
                np.atleast_1d(
                    final_svm.n_iter_
                )
            )
        )

        print(
            f"  Final SVM iterations : "
            f"{svm_iterations:,}"
        )

        if (
            svm_iterations
            >= MAX_ITER
        ):

            print(
                "  [WARN] Final LinearSVC reached max_iter. "
                "Convergence should be reviewed."
            )

        else:

            print(
                "  Convergence          : "
                "completed before max_iter"
            )

    else:

        print(
            "  Final SVM iterations : "
            "Unavailable from calibrated wrapper"
        )

    # ========================================================
    # 5. Held-out evaluation
    # ========================================================

    print_section(
        "[5/6] HELD-OUT TEST EVALUATION"
    )

    y_pred = clf.predict(
        X_test
    )

    y_attack_proba = (
        get_attack_probability(
            clf,
            X_test,
        )
    )

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        pos_label=config.LABEL_ATTACK,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label=config.LABEL_ATTACK,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label=config.LABEL_ATTACK,
        zero_division=0,
    )

    auc = roc_auc_score(
        y_test,
        y_attack_proba,
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[
            config.LABEL_BENIGN,
            config.LABEL_ATTACK,
        ],
    )

    fpr, fnr = calculate_rates(
        cm
    )

    benign_text = label_to_text(
        config.LABEL_BENIGN
    )

    attack_text = label_to_text(
        config.LABEL_ATTACK
    )

    classification_report_text = (
        classification_report(
            y_test,
            y_pred,
            labels=[
                config.LABEL_BENIGN,
                config.LABEL_ATTACK,
            ],
            target_names=[
                benign_text,
                attack_text,
            ],
            digits=6,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Probability sanity check
    # --------------------------------------------------------

    if (
        np.any(
            y_attack_proba < 0.0
        )
        or np.any(
            y_attack_proba > 1.0
        )
    ):

        raise RuntimeError(
            "Calibrated probabilities outside [0, 1] detected."
        )

    # --------------------------------------------------------
    # Console metrics
    # --------------------------------------------------------

    print()

    print(
        f"  Accuracy  : "
        f"{accuracy:.6f}"
    )

    print(
        f"  Precision : "
        f"{precision:.6f}"
    )

    print(
        f"  Recall    : "
        f"{recall:.6f}"
    )

    print(
        f"  F1-score  : "
        f"{f1:.6f}"
    )

    print(
        f"  AUC-ROC   : "
        f"{auc:.6f}"
    )

    print(
        f"  FPR       : "
        f"{fpr:.6f}"
        f" ({fpr * 100:.4f}%)"
    )

    print(
        f"  FNR       : "
        f"{fnr:.6f}"
        f" ({fnr * 100:.4f}%)"
    )

    print()

    print(
        f"  P(Attack) min : "
        f"{float(np.min(y_attack_proba)):.6f}"
    )

    print(
        f"  P(Attack) max : "
        f"{float(np.max(y_attack_proba)):.6f}"
    )

    print(
        "\nConfusion matrix:"
    )

    print(
        "                 Predicted"
    )

    print(
        f"                 "
        f"{benign_text:<10} "
        f"{attack_text:<10}"
    )

    print(
        f"Actual {benign_text:<8} "
        f"{cm[0, 0]:>10,} "
        f"{cm[0, 1]:>10,}"
    )

    print(
        f"Actual {attack_text:<8} "
        f"{cm[1, 0]:>10,} "
        f"{cm[1, 1]:>10,}"
    )

    # ========================================================
    # 6. Save model / metadata / report
    # ========================================================

    print_section(
        "[6/6] Saving model and reports"
    )

    # --------------------------------------------------------
    # Model
    #
    # Save the entire calibrated estimator.
    #
    # Inference can therefore use:
    #
    #     model.predict(X)
    #     model.predict_proba(X)
    #
    # directly from raw 42-feature rows containing NaNs.
    # --------------------------------------------------------

    model_path = Path(
        args.model
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        clf,
        model_path,
        compress=3,
    )

    print(
        f"  Model saved : "
        f"{model_path}"
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata_path = Path(
        args.meta
    )

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {

        "model_type":
            "CalibratedClassifierCV",

        "base_estimator":
            "LinearSVC",

        "comparison_role":
            (
                "Deployment-capable calibrated linear "
                "non-tree alternative for comparison with "
                "Decision Tree and Random Forest"
            ),

        "feature_count":
            len(
                config.ML_FEATURES
            ),

        "features":
            list(
                config.ML_FEATURES
            ),

        "label_column":
            config.LABEL_COLUMN,

        "label_mapping": {

            str(
                config.LABEL_BENIGN
            ):
                benign_text,

            str(
                config.LABEL_ATTACK
            ):
                attack_text,
        },

        "model_classes": [
            int(
                value
            )
            for value
            in clf.classes_
        ],

        "parameters": {

            "C":
                C,

            "penalty":
                PENALTY,

            "loss":
                LOSS,

            "dual":
                DUAL,

            "class_weight":
                CLASS_WEIGHT,

            "tol":
                TOL,

            "max_iter":
                MAX_ITER,

            "random_state":
                RANDOM_STATE,

            "actual_final_svm_iterations":
                svm_iterations,
        },

        "calibration": {

            "method":
                CALIBRATION_METHOD,

            "cv":
                CALIBRATION_CV,

            "ensemble":
                CALIBRATION_ENSEMBLE,

            "n_jobs":
                CALIBRATION_N_JOBS,

            "probability_output":
                True,

            "probability_fields": [
                "P(Benign)",
                "P(Attack)",
            ],

            "design":
                (
                    "Cross-validation is used to obtain "
                    "unbiased classifier outputs for sigmoid "
                    "probability calibration. With "
                    "ensemble=False, the final base estimator "
                    "is trained on the complete training set."
                ),
        },

        "comparison_design": {

            "purpose":
                (
                    "Evaluate a deployment-capable calibrated "
                    "linear non-tree classifier against the "
                    "Decision Tree and primary Random Forest."
                ),

            "controlled_conditions":
                (
                    "The same final training dataset, held-out "
                    "test dataset, 42-feature schema and binary "
                    "labels are used across all three models."
                ),

            "model_specific_difference":
                (
                    "LinearSVC requires finite, scaled inputs "
                    "and does not natively produce probability "
                    "estimates. Median imputation, standard "
                    "scaling and sigmoid probability "
                    "calibration are therefore applied inside "
                    "the SVM model pipeline."
                ),
        },

        "dataset": {

            "train_file":
                str(
                    Path(
                        args.train
                    ).resolve()
                ),

            "test_file":
                str(
                    Path(
                        args.test
                    ).resolve()
                ),

            "train_rows":
                int(
                    len(
                        X_train
                    )
                ),

            "test_rows":
                int(
                    len(
                        X_test
                    )
                ),

            "train_attack_rate":
                float(
                    (
                        y_train
                        == config.LABEL_ATTACK
                    ).mean()
                ),

            "test_attack_rate":
                float(
                    (
                        y_test
                        == config.LABEL_ATTACK
                    ).mean()
                ),
        },

        "preprocessing": {

            "pipeline_location":
                (
                    "Inside estimator supplied to "
                    "CalibratedClassifierCV"
                ),

            "imputer":
                "SimpleImputer",

            "imputer_strategy":
                IMPUTER_STRATEGY,

            "keep_empty_features":
                KEEP_EMPTY_FEATURES,

            "fully_empty_feature_fill":
                (
                    "0 for fully-empty numerical "
                    "training features"
                ),

            "scaler":
                "StandardScaler",

            "scaler_with_mean":
                SCALER_WITH_MEAN,

            "scaler_with_std":
                SCALER_WITH_STD,

            "feature_dtype":
                "float32",

            "label_dtype":
                "int8",

            "additional_features_added":
                False,

            "infinite_values":
                "rejected",
        },

        "missing_values": {

            "train_nan_cells":
                int(
                    train_validation[
                        "nan_count"
                    ]
                ),

            "test_nan_cells":
                int(
                    test_validation[
                        "nan_count"
                    ]
                ),

            "fully_empty_training_features":
                list(
                    train_validation[
                        "fully_empty_features"
                    ]
                ),
        },

        "evaluation": {

            "accuracy":
                float(
                    accuracy
                ),

            "precision":
                float(
                    precision
                ),

            "recall":
                float(
                    recall
                ),

            "f1":
                float(
                    f1
                ),

            "auc_roc":
                float(
                    auc
                ),

            "auc_score_source":
                "Calibrated P(Attack)",

            "fpr":
                float(
                    fpr
                ),

            "fnr":
                float(
                    fnr
                ),

            "confusion_matrix":
                cm.tolist(),

            "attack_probability_min":
                float(
                    np.min(
                        y_attack_proba
                    )
                ),

            "attack_probability_max":
                float(
                    np.max(
                        y_attack_proba
                    )
                ),
        },

        "software": {

            "python":
                sys.version,

            "scikit_learn":
                sklearn.__version__,

            "numpy":
                np.__version__,

            "pandas":
                pd.__version__,
        },
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        f"  Metadata saved: "
        f"{metadata_path}"
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report_path = Path(
        args.report
    )

    save_report(
        path=report_path,
        train_file=args.train,
        test_file=args.test,
        train_size=len(
            X_train
        ),
        test_size=len(
            X_test
        ),
        y_train=y_train,
        y_test=y_test,
        train_validation=train_validation,
        test_validation=test_validation,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auc=auc,
        fpr=fpr,
        fnr=fnr,
        cm=cm,
        classification_report_text=(
            classification_report_text
        ),
        svm_iterations=svm_iterations,
    )

    print(
        f"  Report saved  : "
        f"{report_path}"
    )

    # ========================================================
    # Final summary
    # ========================================================

    print_section(
        "CALIBRATED LINEAR SVM TRAINING COMPLETE"
    )

    print(
        f"Train rows : "
        f"{len(X_train):,}"
    )

    print(
        f"Test rows  : "
        f"{len(X_test):,}"
    )

    print(
        f"Features   : "
        f"{len(config.ML_FEATURES)}"
    )

    print(
        f"Calibration: "
        f"{CALIBRATION_METHOD}, "
        f"cv={CALIBRATION_CV}, "
        f"ensemble={CALIBRATION_ENSEMBLE}"
    )

    print(
        f"Iterations : "
        f"{svm_iterations if svm_iterations is not None else 'Unavailable'}"
    )

    print(
        f"Accuracy   : "
        f"{accuracy:.6f}"
    )

    print(
        f"Precision  : "
        f"{precision:.6f}"
    )

    print(
        f"Recall     : "
        f"{recall:.6f}"
    )

    print(
        f"F1-score   : "
        f"{f1:.6f}"
    )

    print(
        f"AUC-ROC    : "
        f"{auc:.6f}"
    )

    print()

    print(
        f"Model  : "
        f"{model_path}"
    )

    print(
        f"Meta   : "
        f"{metadata_path}"
    )

    print(
        f"Report : "
        f"{report_path}"
    )

# ============================================================
if __name__ == "__main__":
    main()