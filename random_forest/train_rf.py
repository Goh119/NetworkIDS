"""
NetworkIDS - Primary Random Forest training and evaluation.

Location:
    random_forest/train_rf.py

Primary inputs:
    data/processed/multiday_split/train_multiday.csv
    data/processed/multiday_split/test_multiday.csv

Primary outputs:
    random_forest/
        rf_model.joblib
        rf_model_meta.json
        rf_model_report.txt

The Random Forest uses exactly the features currently defined in:

    config.ML_FEATURES

The number of ML features is dynamic and is NOT hard-coded.

Classification:
    config.LABEL_BENIGN = Benign
    config.LABEL_ATTACK = Attack

Missing-value policy:
    NaN values are retained.
    No placeholder values such as -1 or 0 are introduced.

This script performs:
    1. Configuration validation
    2. Train/test loading
    3. ML feature validation
    4. Label validation
    5. X/y preparation
    6. Random Forest training
    7. Held-out test evaluation
    8. Gini feature importance
    9. Model saving
   10. Metadata saving
   11. Human-readable report generation

Usage:
    python random_forest/train_rf.py

Optional:
    python random_forest/train_rf.py \
        --train path/to/train.csv \
        --test path/to/test.csv \
        --model path/to/model.joblib
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


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
# Random Forest Parameters
#
# These are the PRIMARY model parameters.
#
# max_depth = 20 is intentionally selected because this IDS
# prioritizes reducing False Negatives / missed attacks.
#
# The resulting higher False Positive rate can be handled
# through SOC analyst triage.
# ============================================================

RANDOM_STATE = 42

N_ESTIMATORS = 200

MAX_DEPTH = 20

CLASS_WEIGHT = "balanced"

N_JOBS = -1


# ============================================================
# Data types
# ============================================================

FEATURE_DTYPE = np.float32

LABEL_DTYPE = np.int8


# ============================================================
# Helpers
# ============================================================

def print_section(title):

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
# Central label-name helper
# ============================================================

def label_to_text(label):
    """
    Convert the configured numeric label into a readable name.
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
    Load only columns needed by the Random Forest.

    Metadata such as:
        flow_id
        timestamp
        src_ip
        dst_ip
        src_mac
        dst_mac
        day

    is intentionally not loaded.

    Only ML_FEATURES + LABEL_COLUMN are loaded.
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

    if len(actual) != 2:

        raise ValueError(
            f"{name}: both "
            f"{label_to_text(config.LABEL_BENIGN)} and "
            f"{label_to_text(config.LABEL_ATTACK)} "
            f"classes must be present."
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
        allowed

    Inf:
        rejected
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
    # NaN
    # --------------------------------------------------------

    nan_count = int(
        X.isna()
        .sum()
        .sum()
    )

    if nan_count:

        print(
            f"  {name}: "
            f"{nan_count:,} NaN feature cells retained."
        )

        print(
            "  No imputation is applied."
        )

    else:

        print(
            f"  {name}: "
            "no NaN feature cells."
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


# ============================================================
# Prepare X / y
# ============================================================

def prepare_xy(
    df,
    name,
):
    """
    Prepare X and y.

    X:
        dynamic ML feature list
        float32

    y:
        binary labels
        int8
    """

    validate_labels(
        df,
        name,
    )

    validate_features(
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

    # ========================================================
    # No deprecated copy= argument.
    # ========================================================

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

    return X, y


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
    Extract Attack probability using model.classes_
    rather than assuming probability column 1.
    """

    probabilities = (
        model.predict_proba(
            X
        )
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
# FPR / FNR
# ============================================================

def calculate_rates(cm):
    """
    Calculate FPR and FNR.

    Confusion matrix:

        [[TN, FP],
         [FN, TP]]
    """

    tn = int(
        cm[0, 0]
    )

    fp = int(
        cm[0, 1]
    )

    fn = int(
        cm[1, 0]
    )

    tp = int(
        cm[1, 1]
    )

    fpr = (
        fp
        / (fp + tn)
        if (fp + tn)
        else 0.0
    )

    fnr = (
        fn
        / (fn + tp)
        if (fn + tp)
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
    train_size,
    test_size,
    y_train,
    y_test,
    accuracy,
    precision,
    recall,
    f1,
    auc,
    fpr,
    fnr,
    cm,
    classification_report_text,
    importances,
):

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
            "NetworkIDS - Random Forest Training Report\n"
        )

        report.write(
            "=" * 70
            + "\n\n"
        )

        report.write(
            "Model Configuration\n"
        )

        report.write(
            "-" * 70
            + "\n"
        )

        report.write(
            "Model              : RandomForestClassifier\n"
        )

        report.write(
            f"n_estimators       : {N_ESTIMATORS}\n"
        )

        report.write(
            f"max_depth          : {MAX_DEPTH}\n"
        )

        report.write(
            f"class_weight       : {CLASS_WEIGHT}\n"
        )

        report.write(
            f"random_state       : {RANDOM_STATE}\n"
        )

        report.write(
            f"n_jobs             : {N_JOBS}\n"
        )

        report.write(
            f"scikit-learn       : {sklearn.__version__}\n"
        )

        report.write(
            f"ML feature count   : "
            f"{len(config.ML_FEATURES)}\n"
        )

        report.write(
            "\nModel Selection Rationale\n"
        )

        report.write(
            "-" * 70
            + "\n"
        )

        report.write(
            "max_depth=20 prioritizes high attack recall and "
            "low false-negative rate. In the SOC use case, "
            "false-positive alerts can be reviewed by analysts, "
            "whereas false-negative attacks may remain "
            "undetected.\n"
        )

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

        report.write(
            "\nDataset\n"
        )

        report.write(
            "-" * 70
            + "\n"
        )

        report.write(
            f"Training rows      : {train_size:,}\n"
        )

        report.write(
            f"Test rows          : {test_size:,}\n"
        )

        report.write(
            "Split              : "
            "5-day merged dataset; identical ML-feature-vector "
            "grouped StratifiedGroupKFold; first fold holdout\n"
        )

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

            label_text = label_to_text(
                label_value
            )

            report.write(
                f"{label_value} "
                f"{label_text:<10}"
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

            label_text = label_to_text(
                label_value
            )

            report.write(
                f"{label_value} "
                f"{label_text:<10}"
                f"{int(count):>12,}\n"
            )

        report.write(
            "\nClassification Metrics\n"
        )

        report.write(
            "-" * 70
            + "\n"
        )

        report.write(
            f"Accuracy            : {accuracy:.6f}\n"
        )

        report.write(
            f"Precision           : {precision:.6f}\n"
        )

        report.write(
            f"Recall              : {recall:.6f}\n"
        )

        report.write(
            f"F1-score            : {f1:.6f}\n"
        )

        report.write(
            f"AUC-ROC             : {auc:.6f}\n"
        )

        report.write(
            f"False Positive Rate : {fpr:.6f}\n"
        )

        report.write(
            f"False Negative Rate : {fnr:.6f}\n"
        )

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

        report.write(
            "\nTop Gini Feature Importances\n"
        )

        report.write(
            "-" * 70
            + "\n"
        )

        for rank, (
            feature,
            importance,
        ) in enumerate(
            importances.items(),
            start=1,
        ):

            report.write(
                f"{rank:>2}. "
                f"{feature:<35} "
                f"{importance:.6f}\n"
            )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the primary "
            "NetworkIDS Random Forest."
        )
    )

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

    parser.add_argument(
        "--model",
        default=str(
            config.RANDOM_FOREST_MODEL_FILE
        ),
        help="Output .joblib model path.",
    )

    args = parser.parse_args()

    print(
        "=" * 70
    )

    print(
        "NetworkIDS - PRIMARY RANDOM FOREST TRAINING"
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

    # ==================== CHANGED ============================

    print(
        f"  RF directory     : "
        f"{config.RANDOM_FOREST_DIR}"
    )

    print(
        f"  Model output     : "
        f"{args.model}"
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

    X_train, y_train = prepare_xy(
        train_df,
        "Train",
    )

    X_test, y_test = prepare_xy(
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
    # 4. TRAIN
    # ========================================================

    print_section(
        "[4/6] TRAINING RANDOM FOREST"
    )

    print(
        f"  n_estimators = {N_ESTIMATORS}"
    )

    print(
        f"  max_depth    = {MAX_DEPTH}"
    )

    print(
        f"  class_weight = {CLASS_WEIGHT}"
    )

    print(
        f"  random_state = {RANDOM_STATE}"
    )

    print(
        f"  n_jobs       = {N_JOBS}"
    )

    print(
        f"  n_features   = "
        f"{len(config.ML_FEATURES)}"
    )

    print(
        f"  train rows   = "
        f"{len(X_train):,}"
    )

    print(
        "\n  Fitting model..."
    )

    clf = RandomForestClassifier(

        n_estimators=N_ESTIMATORS,

        max_depth=MAX_DEPTH,

        class_weight=CLASS_WEIGHT,

        random_state=RANDOM_STATE,

        n_jobs=N_JOBS,
    )

    clf.fit(
        X_train,
        y_train,
    )

    print(
        "\n  TRAINING COMPLETE."
    )

    # ========================================================
    # 5. Evaluation
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

    importances = (

        pd.Series(

            clf.feature_importances_,

            index=config.ML_FEATURES,
        )

        .sort_values(
            ascending=False
        )
    )

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

    print(
        "\nTop Gini feature importances:"
    )

    for rank, (
        feature,
        importance,
    ) in enumerate(
        importances.items(),
        start=1,
    ):

        print(
            f"  {rank:>2}. "
            f"{feature:<35} "
            f"{importance:.6f}"
        )

    # ========================================================
    # 6. Save model / metadata / report
    # ========================================================

    print_section(
        "[6/6] Saving model and reports"
    )

    model_path = Path(
        args.model
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

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
    # Save metadata
    # --------------------------------------------------------

    metadata_path = Path(
        config.RANDOM_FOREST_META_FILE
    )

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {

        "model_type":
            "RandomForestClassifier",

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
                label_to_text(
                    config.LABEL_BENIGN
                ),

            str(
                config.LABEL_ATTACK
            ):
                label_to_text(
                    config.LABEL_ATTACK
                ),
        },

        "model_classes": [
            int(
                value
            )
            for value
            in clf.classes_
        ],

        "parameters": {

            "n_estimators":
                N_ESTIMATORS,

            "max_depth":
                MAX_DEPTH,

            "class_weight":
                CLASS_WEIGHT,

            "random_state":
                RANDOM_STATE,

            "n_jobs":
                N_JOBS,
        },

        "model_selection": {

            "priority":
                "Minimize false negatives",

            "rationale":
                (
                    "For the SOC-oriented IDS use case, "
                    "missed attacks are considered more "
                    "critical than false-positive alerts. "
                    "False positives can be reviewed through "
                    "analyst triage."
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

        "split_strategy":
            (
                "Five-day merged dataset; "
                "identical ML-feature-vector grouped "
                "StratifiedGroupKFold; first fold used "
                "as fixed holdout test set."
            ),

        "preprocessing": {

            "feature_dtype":
                "float32",

            "label_dtype":
                "int8",

            "missing_value_policy":
                "NaN retained; no imputation",

            "infinite_values":
                "rejected",
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
    # Save report
    # --------------------------------------------------------

    report_path = Path(
        config.RANDOM_FOREST_REPORT_FILE
    )

    save_report(

        path=report_path,

        train_size=len(
            X_train
        ),

        test_size=len(
            X_test
        ),

        y_train=y_train,

        y_test=y_test,

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

        importances=importances,
    )

    print(
        f"  Report saved  : "
        f"{report_path}"
    )

    # ========================================================
    # Final summary
    # ========================================================

    print_section(
        "PRIMARY RANDOM FOREST TRAINING COMPLETE"
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

if __name__ == "__main__":
    main()