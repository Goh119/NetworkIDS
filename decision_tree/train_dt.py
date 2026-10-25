"""
NetworkIDS - Decision Tree training and evaluation.

Location:
    decision_tree/train_dt.py

Default inputs:
    data/train/training_dataset.csv
    data/test/test_multiday.csv

Default outputs:
    decision_tree/
        dt_model.joblib
        dt_model_meta.json
        dt_model_report.txt

Alternative train/test datasets and output files can be supplied
through command-line arguments.

The Decision Tree uses exactly the features currently defined in:

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
    6. Decision Tree training
    7. Held-out test evaluation
    8. Gini feature importance
    9. Model saving
   10. Metadata saving
   11. Human-readable report generation

Usage:
    python decision_tree/train_dt.py

Optional:
    python decision_tree/train_dt.py \
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

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.tree import DecisionTreeClassifier


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
# Decision Tree Parameters
#
# The Decision Tree is used as a comparison baseline against
# the primary Random Forest.
#
# max_depth = 20 is intentionally aligned with the Random
# Forest configuration so that the comparison evaluates:
#
#     single decision tree
#
# versus:
#
#     ensemble of decision trees
#
# under comparable depth constraints.
# ============================================================

RANDOM_STATE = 42

CRITERION = "gini"

MAX_DEPTH = 20

MIN_SAMPLES_SPLIT = 2

MIN_SAMPLES_LEAF = 1

CLASS_WEIGHT = "balanced"


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
# Central label-name helper
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
    Load only columns required by the Decision Tree.

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
    Extract Attack probability using model.classes_ rather
    than assuming that Attack is probability column 1.
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
    """
    Save the complete Decision Tree training/evaluation report.
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
            "NetworkIDS - Decision Tree Training Report\n"
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
            "Model              : DecisionTreeClassifier\n"
        )

        report.write(
            f"criterion          : {CRITERION}\n"
        )

        report.write(
            f"max_depth          : {MAX_DEPTH}\n"
        )

        report.write(
            f"min_samples_split  : {MIN_SAMPLES_SPLIT}\n"
        )

        report.write(
            f"min_samples_leaf   : {MIN_SAMPLES_LEAF}\n"
        )

        report.write(
            f"class_weight       : {CLASS_WEIGHT}\n"
        )

        report.write(
            f"random_state       : {RANDOM_STATE}\n"
        )

        report.write(
            f"scikit-learn       : {sklearn.__version__}\n"
        )

        report.write(
            f"ML feature count   : "
            f"{len(config.ML_FEATURES)}\n"
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
            "The Decision Tree is evaluated as a single-tree "
            "baseline against the primary Random Forest. "
            "The same training dataset, held-out test dataset, "
            "ML feature set, binary labels, class weighting, "
            "and maximum tree depth are used where applicable "
            "to support a controlled model comparison.\n"
        )

        # ====================================================
        # Label configuration
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
            "test data was not used for model fitting.\n"
        )

        # ====================================================
        # Train labels
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

            label_text = label_to_text(
                label_value
            )

            report.write(
                f"{label_value} "
                f"{label_text:<10}"
                f"{int(count):>12,}\n"
            )

        # ====================================================
        # Test labels
        # ====================================================

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

        # ====================================================
        # Feature importance
        # ====================================================

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
            "Train and evaluate the comparison "
            "NetworkIDS Decision Tree."
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
            config.DECISION_TREE_MODEL_FILE
        ),
        help="Output .joblib model path.",
    )

    parser.add_argument(
        "--meta",
        default=str(
            config.DECISION_TREE_META_FILE
        ),
        help="Output metadata JSON path.",
    )

    parser.add_argument(
        "--report",
        default=str(
            config.DECISION_TREE_REPORT_FILE
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
            "Refusing to overwrite existing Decision Tree outputs:\n"
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
        "NetworkIDS - DECISION TREE TRAINING"
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
        f"  DT directory     : "
        f"{config.DECISION_TREE_DIR}"
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
    # 4. Train
    # ========================================================

    print_section(
        "[4/6] TRAINING DECISION TREE"
    )

    print(
        f"  criterion         = "
        f"{CRITERION}"
    )

    print(
        f"  max_depth         = "
        f"{MAX_DEPTH}"
    )

    print(
        f"  min_samples_split = "
        f"{MIN_SAMPLES_SPLIT}"
    )

    print(
        f"  min_samples_leaf  = "
        f"{MIN_SAMPLES_LEAF}"
    )

    print(
        f"  class_weight      = "
        f"{CLASS_WEIGHT}"
    )

    print(
        f"  random_state      = "
        f"{RANDOM_STATE}"
    )

    print(
        f"  n_features        = "
        f"{len(config.ML_FEATURES)}"
    )

    print(
        f"  train rows        = "
        f"{len(X_train):,}"
    )

    print(
        "\n  Fitting model..."
    )

    clf = DecisionTreeClassifier(
        criterion=CRITERION,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight=CLASS_WEIGHT,
        random_state=RANDOM_STATE,
    )

    clf.fit(
        X_train,
        y_train,
    )

    print(
        "\n  TRAINING COMPLETE."
    )

    print(
        f"  Tree depth : "
        f"{clf.get_depth()}"
    )

    print(
        f"  Leaf nodes : "
        f"{clf.get_n_leaves():,}"
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

    # ========================================================
    # Feature importance
    # ========================================================

    importances = (
        pd.Series(
            clf.feature_importances_,
            index=config.ML_FEATURES,
        )
        .sort_values(
            ascending=False
        )
    )

    # ========================================================
    # Console metrics
    # ========================================================

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

    # --------------------------------------------------------
    # Model
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
            "DecisionTreeClassifier",

        "comparison_role":
            "Single-tree baseline for Random Forest comparison",

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

            "criterion":
                CRITERION,

            "max_depth":
                MAX_DEPTH,

            "min_samples_split":
                MIN_SAMPLES_SPLIT,

            "min_samples_leaf":
                MIN_SAMPLES_LEAF,

            "class_weight":
                CLASS_WEIGHT,

            "random_state":
                RANDOM_STATE,
        },

        "trained_tree": {

            "actual_depth":
                int(
                    clf.get_depth()
                ),

            "leaf_nodes":
                int(
                    clf.get_n_leaves()
                ),
        },

        "comparison_design": {

            "purpose":
                (
                    "Evaluate a single Decision Tree against "
                    "the primary Random Forest."
                ),

            "controlled_conditions":
                (
                    "The Decision Tree uses the same final "
                    "training dataset, held-out test dataset, "
                    "42 ML features, binary labels and class "
                    "weighting as the Random Forest. "
                    "max_depth=20 is also aligned with the "
                    "Random Forest configuration."
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
                "Training and evaluation use predefined CSV "
                "datasets. The held-out test dataset is not "
                "used for model fitting. The same final train "
                "and test datasets are used for the Random "
                "Forest and Decision Tree comparison."
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
        "DECISION TREE TRAINING COMPLETE"
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
        f"Tree depth : "
        f"{clf.get_depth()}"
    )

    print(
        f"Leaf nodes : "
        f"{clf.get_n_leaves():,}"
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