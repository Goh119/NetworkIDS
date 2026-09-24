"""
train_model.py
--------------
NetworkIDS - Train and evaluate a Random Forest classifier.

Inputs:
    data/processed/split/train_dataset.csv
    data/processed/split/test_dataset.csv

Missing-value policy:
    NaN values in ML features are retained as-is.

    No placeholder values such as -1 or 0 are used to represent
    missing values. Protocol-specific missing values remain NaN.

    The Random Forest implementation used in this project supports
    native missing-value handling.

This script:
    1. Loads the pre-split train/test datasets.
    2. Checks for flow_id leakage between train and test.
    3. Checks for identical 42-feature vectors across train/test.
    4. Selects exactly the 42 ML features.
    5. Trains a Random Forest classifier.
    6. Evaluates the model on the held-out test set.
    7. Reports classification metrics and confusion matrix.
    8. Reports Gini feature importance.
    9. Saves the trained model.
   10. Saves evaluation metrics as JSON.

Usage:
    python src/train_model.py
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

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
    str(Path(__file__).resolve().parents[1])
)

import src.config as config


# ============================================================
# Configuration
# ============================================================

SPLIT_DIR = config.PROCESSED_DIR / "split"

TRAIN_FILE = SPLIT_DIR / "train_dataset.csv"
TEST_FILE = SPLIT_DIR / "test_dataset.csv"

RANDOM_STATE = 42
N_ESTIMATORS = 200
N_JOBS = -1


# ============================================================
# Helper functions
# ============================================================

def load_split(path, name):
    """
    Load a pre-split dataset.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"{name} dataset not found:\n{path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False
    )

    print(
        f"  {name:<6}: "
        f"{len(df):>9,} rows "
        f"({path.name})"
    )

    return df


def prepare_xy(df, name):
    """
    Prepare feature matrix X and target y.

    Exactly the 42 ML features defined in config.py are used.

    NaN values are retained as-is.
    Infinite values are rejected because they indicate
    an upstream processing problem.
    """

    # --------------------------------------------------------
    # Select features and label
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in config.ML_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"{name}: missing ML features: "
            f"{missing_features}"
        )

    if config.LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"{name}: label column "
            f"'{config.LABEL_COLUMN}' not found."
        )

    X = df[config.ML_FEATURES].copy()
    y = df[config.LABEL_COLUMN].copy()

    # --------------------------------------------------------
    # Validate labels
    # --------------------------------------------------------

    if y.isna().any():
        raise ValueError(
            f"{name}: label column contains NaN values. "
            f"Run prepare_training_data.py before training."
        )

    y = y.astype(int)

    valid_labels = {
        config.LABEL_BENIGN,
        config.LABEL_ATTACK
    }

    actual_labels = set(y.unique())

    if not actual_labels.issubset(valid_labels):
        unexpected = actual_labels - valid_labels

        raise ValueError(
            f"{name}: unexpected label values: "
            f"{unexpected}"
        )

    # --------------------------------------------------------
    # Validate feature count
    # --------------------------------------------------------

    if len(config.ML_FEATURES) != 42:
        raise ValueError(
            "Expected exactly 42 ML features, "
            f"but config contains "
            f"{len(config.ML_FEATURES)}."
        )

    # --------------------------------------------------------
    # Check infinite values
    # --------------------------------------------------------

    numeric_X = X.select_dtypes(
        include=[np.number]
    )

    inf_count = int(
        np.isinf(
            numeric_X.to_numpy(
                dtype=float
            )
        ).sum()
    )

    if inf_count > 0:
        raise ValueError(
            f"{name}: found "
            f"{inf_count:,} infinite feature values. "
            f"Check the upstream feature extraction pipeline."
        )

    # --------------------------------------------------------
    # Report NaN values
    # --------------------------------------------------------

    nan_count = int(
        X.isna().sum().sum()
    )

    if nan_count > 0:
        print(
            f"  {name}: "
            f"{nan_count:,} NaN feature cells retained."
        )
        print(
            "  Missing values are not imputed."
        )

    return X, y


def check_flow_id_leakage(train_df, test_df):
    """
    Ensure that no flow_id occurs in both datasets.
    """

    train_ids = set(
        train_df["flow_id"]
    )

    test_ids = set(
        test_df["flow_id"]
    )

    overlap = train_ids.intersection(
        test_ids
    )

    if overlap:
        raise RuntimeError(
            "Flow ID leakage detected: "
            f"{len(overlap):,} overlapping flow IDs."
        )

    print(
        "[OK] Flow ID overlap: 0"
    )


def check_feature_group_leakage(train_df, test_df):
    """
    Ensure that identical 42-feature vectors do not
    occur in both train and test datasets.

    This is consistent with split_dataset.py, which keeps
    identical feature vectors within the same split.
    """

    train_features = train_df[
        config.ML_FEATURES
    ].copy()

    test_features = test_df[
        config.ML_FEATURES
    ].copy()

    # Use merge so that NaN values are handled consistently
    # when checking identical feature combinations.
    train_features["_feature_group"] = 1

    test_features["_feature_group"] = 1

    overlap = train_features.merge(
        test_features,
        on=config.ML_FEATURES,
        how="inner"
    )

    if len(overlap) > 0:
        raise RuntimeError(
            "Feature-vector leakage detected: "
            f"{len(overlap):,} matching feature-vector "
            "combinations exist in both train and test."
        )

    print(
        "[OK] 42-feature vector overlap: 0"
    )


def print_label_distribution(name, y):
    """
    Print label counts and percentages.
    """

    counts = (
        y.value_counts()
        .sort_index()
    )

    print(f"  {name}:")

    total = len(y)

    for label, count in counts.items():

        if label == config.LABEL_BENIGN:
            label_name = "Benign"
        elif label == config.LABEL_ATTACK:
            label_name = "Attack"
        else:
            label_name = str(label)

        percentage = (
            count / total * 100
        )

        print(
            f"    {label_name:<8}: "
            f"{count:>9,} "
            f"({percentage:.2f}%)"
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate Random Forest "
            "on pre-split NetworkIDS datasets."
        )
    )

    parser.add_argument(
        "--train",
        default=str(TRAIN_FILE)
    )

    parser.add_argument(
        "--test",
        default=str(TEST_FILE)
    )

    parser.add_argument(
        "--out-model",
        default=str(
            config.RANDOM_FOREST_MODEL_FILE
        )
    )

    args = parser.parse_args()

    # ========================================================
    # Header
    # ========================================================

    print("=" * 70)
    print("NetworkIDS - Random Forest Training")
    print("=" * 70)

    # ========================================================
    # 1. Load datasets
    # ========================================================

    print("\n[1/7] Loading pre-split datasets...")

    train_df = load_split(
        Path(args.train),
        "Train"
    )

    test_df = load_split(
        Path(args.test),
        "Test"
    )

    # ========================================================
    # 2. Leakage checks
    # ========================================================

    print("\n[2/7] Checking train/test leakage...")

    check_flow_id_leakage(
        train_df,
        test_df
    )

    check_feature_group_leakage(
        train_df,
        test_df
    )

    print(
        "[OK] Leakage checks passed."
    )

    # ========================================================
    # 3. Prepare X and y
    # ========================================================

    print("\n[3/7] Preparing feature matrices...")

    X_train, y_train = prepare_xy(
        train_df,
        "Train"
    )

    X_test, y_test = prepare_xy(
        test_df,
        "Test"
    )

    print(
        f"  X_train : {X_train.shape}"
    )

    print(
        f"  X_test  : {X_test.shape}"
    )

    print(
        f"  Features: {len(config.ML_FEATURES)}"
    )

    print("\nLabel distributions:")

    print_label_distribution(
        "Train",
        y_train
    )

    print_label_distribution(
        "Test",
        y_test
    )

    # ========================================================
    # 4. Train Random Forest
    # ========================================================

    print("\n" + "=" * 70)
    print("Training Random Forest")
    print("=" * 70)

    print(
        f"  n_estimators = {N_ESTIMATORS}"
    )

    print(
        "  class_weight = balanced"
    )

    print(
        f"  random_state = {RANDOM_STATE}"
    )

    print(
        f"  n_jobs       = {N_JOBS}"
    )

    print()

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    clf.fit(
        X_train,
        y_train
    )

    print(
        "  Random Forest training completed."
    )

    # ========================================================
    # 5. Evaluate on held-out test set
    # ========================================================

    print("\n" + "=" * 70)
    print("Evaluation on Held-Out Test Set")
    print("=" * 70)

    y_pred = clf.predict(
        X_test
    )

    y_proba = clf.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        y_proba
    )

    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print()

    print(
        f"  Accuracy  : {accuracy:.4f}"
    )

    print(
        f"  Precision : {precision:.4f}"
    )

    print(
        f"  Recall    : {recall:.4f}"
    )

    print(
        f"  F1-score  : {f1:.4f}"
    )

    print(
        f"  AUC-ROC   : {auc:.4f}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Benign",
                "Attack"
            ],
            digits=4,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("Confusion matrix:")
    print()
    print("                 Predicted")
    print("                 Benign     Attack")
    print(
        f"  Actual Benign  "
        f"{cm[0, 0]:>8,}  "
        f"{cm[0, 1]:>8,}"
    )
    print(
        f"  Actual Attack  "
        f"{cm[1, 0]:>8,}  "
        f"{cm[1, 1]:>8,}"
    )

    # ========================================================
    # 6. Feature importance
    # ========================================================

    print("\n" + "=" * 70)
    print("Top 15 Feature Importances (Gini)")
    print("=" * 70)

    importances = (
        pd.Series(
            clf.feature_importances_,
            index=config.ML_FEATURES
        )
        .sort_values(
            ascending=False
        )
    )

    for feature, value in (
        importances.head(15).items()
    ):
        print(
            f"  {feature:<30} "
            f"{value:.4f}"
        )

    # ========================================================
    # 7. Save model and metrics
    # ========================================================

    print("\n" + "=" * 70)
    print("Saving Model and Metrics")
    print("=" * 70)

    out_model = Path(
        args.out_model
    )

    out_model.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(
        clf,
        out_model
    )

    print(
        f"Model saved : {out_model}"
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics = {
        "model": "RandomForestClassifier",
        "random_state": RANDOM_STATE,
        "n_estimators": N_ESTIMATORS,
        "class_weight": "balanced",

        "n_train": int(
            len(X_train)
        ),

        "n_test": int(
            len(X_test)
        ),

        "n_features": int(
            len(config.ML_FEATURES)
        ),

        "label_train": {
            str(int(k)): int(v)
            for k, v in (
                y_train
                .value_counts()
                .sort_index()
                .items()
            )
        },

        "label_test": {
            str(int(k)): int(v)
            for k, v in (
                y_test
                .value_counts()
                .sort_index()
                .items()
            )
        },

        "accuracy": float(
            accuracy
        ),

        "precision": float(
            precision
        ),

        "recall": float(
            recall
        ),

        "f1": float(
            f1
        ),

        "auc_roc": float(
            auc
        ),

        "confusion_matrix": cm.tolist(),

        "top_15_importances": {
            str(k): float(v)
            for k, v in (
                importances
                .head(15)
                .items()
            )
        }
    }

    metrics_path = (
        out_model.parent
        / "training_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2
        )

    print(
        f"Metrics saved: {metrics_path}"
    )

    # ========================================================
    # Final summary
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAINING AND EVALUATION COMPLETE - PASS")
    print("=" * 70)

    print(
        f"Train rows : {len(X_train):,}"
    )

    print(
        f"Test rows  : {len(X_test):,}"
    )

    print(
        f"Features   : {len(config.ML_FEATURES)}"
    )

    print(
        f"F1-score   : {f1:.4f}"
    )

    print(
        f"AUC-ROC    : {auc:.4f}"
    )

    print(
        f"Model      : {out_model}"
    )

if __name__ == "__main__":
    main()