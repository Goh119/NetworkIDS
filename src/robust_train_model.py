"""
NetworkIDS - Robust Random Forest Training

Purpose:
    Train a robustness-enhanced Random Forest using controlled
    TTL feature augmentation on the TRAINING DATA ONLY.

Important:
    - The original training samples are preserved.
    - Only Attack samples are augmented.
    - The held-out test dataset is NEVER modified or used for training.
    - The 42 ML features remain unchanged.
    - NaN values are retained.
    - Original labels are not changed.
    - The baseline Random Forest is not overwritten.

Augmentation:
    1. TTL +/- 5  -> 30% of Attack training samples
    2. TTL +/- 30 -> 15% of Attack training samples
    3. TTL = 128  -> 5% of Attack training samples

All augmented samples retain:
    label = Attack (1)

Usage:
    python src/robust_train_model.py
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

RANDOM_STATE = 42

N_ESTIMATORS = 200
N_JOBS = -1

# ------------------------------------------------------------
# Robust augmentation ratios
# ------------------------------------------------------------

TTL_SMALL_RATIO = 0.30
TTL_LARGE_RATIO = 0.15
TTL_FORCED_RATIO = 0.05

# Output files
ROBUST_MODEL_FILE = (
    config.MODELS_DIR / "random_forest_robust.pkl"
)

ROBUST_METRICS_FILE = (
    config.MODELS_DIR / "robust_training_metrics.json"
)

ROBUST_TRAINING_FILE = (
    config.PROCESSED_DIR
    / "robust_training_dataset.csv"
)


# ============================================================
# Dataset loading
# ============================================================

def load_dataset(path, name):
    """
    Load a CSV dataset.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"{name} dataset not found:\n{path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    print(
        f"  {name:<10}: "
        f"{len(df):>9,} rows "
        f"({path.name})"
    )

    return df


# ============================================================
# Feature / label validation
# ============================================================

def prepare_xy(df, name):
    """
    Prepare X and y using exactly the 42 configured
    ML features.

    NaN values are retained.
    Infinite values are rejected.
    """

    missing_features = [
        feature
        for feature in config.ML_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"{name}: missing ML features:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing_features
            )
        )

    if config.LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"{name}: missing label column "
            f"'{config.LABEL_COLUMN}'."
        )

    X = df[
        config.ML_FEATURES
    ].copy()

    y = df[
        config.LABEL_COLUMN
    ].copy()

    if y.isna().any():
        raise ValueError(
            f"{name}: label contains NaN values."
        )

    y = y.astype(int)

    valid_labels = {
        config.LABEL_BENIGN,
        config.LABEL_ATTACK,
    }

    actual_labels = set(
        y.unique()
    )

    if not actual_labels.issubset(
        valid_labels
    ):
        raise ValueError(
            f"{name}: unexpected labels: "
            f"{actual_labels}"
        )

    if len(config.ML_FEATURES) != 42:
        raise ValueError(
            "Expected exactly 42 ML features, "
            f"but found "
            f"{len(config.ML_FEATURES)}."
        )

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
            f"{inf_count:,} infinite values."
        )

    nan_count = int(
        X.isna().sum().sum()
    )

    print(
        f"  {name}: "
        f"{nan_count:,} NaN feature cells retained."
    )

    return X, y


# ============================================================
# Leakage checks
# ============================================================

def check_flow_id_leakage(
    train_df,
    test_df,
):
    """
    Ensure no flow_id occurs in both datasets.
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
            f"{len(overlap):,} overlapping IDs."
        )

    print(
        "[OK] Flow ID overlap: 0"
    )


def check_feature_group_leakage(
    train_df,
    test_df,
):
    """
    Ensure identical 42-feature vectors do not
    occur across train and test.
    """

    train_features = train_df[
        config.ML_FEATURES
    ].copy()

    test_features = test_df[
        config.ML_FEATURES
    ].copy()

    train_features[
        "_feature_group"
    ] = 1

    test_features[
        "_feature_group"
    ] = 1

    overlap = train_features.merge(
        test_features,
        on=config.ML_FEATURES,
        how="inner",
    )

    if len(overlap) > 0:
        raise RuntimeError(
            "Feature-vector leakage detected: "
            f"{len(overlap):,} "
            "matching feature combinations."
        )

    print(
        "[OK] 42-feature vector overlap: 0"
    )


# ============================================================
# Label distribution
# ============================================================

def print_label_distribution(
    name,
    y,
):
    """
    Print benign / attack counts.
    """

    counts = (
        y.value_counts()
        .sort_index()
    )

    total = len(y)

    print(f"\n{name} label distribution:")

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
            f"  {label_name:<8}: "
            f"{count:>10,} "
            f"({percentage:.2f}%)"
        )


# ============================================================
# TTL augmentation
# ============================================================

def create_ttl_augmented_attack_samples(
    attack_df,
):
    """
    Create controlled TTL-perturbed copies
    from Attack training samples.

    The original rows are NOT modified.

    Returns:
        augmented_df
        augmentation_summary
    """

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    attack_count = len(attack_df)

    if attack_count == 0:
        raise ValueError(
            "No Attack samples available "
            "for augmentation."
        )

    # --------------------------------------------------------
    # Determine sample counts
    # --------------------------------------------------------

    n_small = int(
        attack_count
        * TTL_SMALL_RATIO
    )

    n_large = int(
        attack_count
        * TTL_LARGE_RATIO
    )

    n_forced = int(
        attack_count
        * TTL_FORCED_RATIO
    )

    print("\nTTL augmentation plan:")
    print(
        f"  Attack training rows : "
        f"{attack_count:,}"
    )
    print(
        f"  TTL +/- 5             : "
        f"{n_small:,}"
    )
    print(
        f"  TTL +/- 30            : "
        f"{n_large:,}"
    )
    print(
        f"  TTL = 128             : "
        f"{n_forced:,}"
    )

    # --------------------------------------------------------
    # Randomly sample Attack rows
    #
    # Each scenario uses a fresh sample.
    # --------------------------------------------------------

    small_indices = rng.choice(
        attack_df.index,
        size=n_small,
        replace=False,
    )

    large_indices = rng.choice(
        attack_df.index,
        size=n_large,
        replace=False,
    )

    forced_indices = rng.choice(
        attack_df.index,
        size=n_forced,
        replace=False,
    )

    # --------------------------------------------------------
    # TTL +/- 5
    # --------------------------------------------------------

    small_df = attack_df.loc[
        small_indices
    ].copy()

    small_valid = (
        small_df["ttl_mean"].notna()
    )

    small_noise = rng.randint(
        -5,
        6,
        size=len(small_df),
    )

    small_df.loc[
        small_valid,
        "ttl_mean",
    ] = (
        small_df.loc[
            small_valid,
            "ttl_mean",
        ]
        + small_noise[small_valid.to_numpy()]
    ).clip(
        1,
        255,
    )

    # Keep label explicitly Attack
    small_df[
        config.LABEL_COLUMN
    ] = config.LABEL_ATTACK

    small_df[
        "augmentation_type"
    ] = "ttl_jitter_small"

    # --------------------------------------------------------
    # TTL +/- 30
    # --------------------------------------------------------

    large_df = attack_df.loc[
        large_indices
    ].copy()

    large_valid = (
        large_df["ttl_mean"].notna()
    )

    large_noise = rng.randint(
        -30,
        31,
        size=len(large_df),
    )

    large_df.loc[
        large_valid,
        "ttl_mean",
    ] = (
        large_df.loc[
            large_valid,
            "ttl_mean",
        ]
        + large_noise[large_valid.to_numpy()]
    ).clip(
        1,
        255,
    )

    large_df[
        config.LABEL_COLUMN
    ] = config.LABEL_ATTACK

    large_df[
        "augmentation_type"
    ] = "ttl_jitter_large"

    # --------------------------------------------------------
    # TTL = 128
    # --------------------------------------------------------

    forced_df = attack_df.loc[
        forced_indices
    ].copy()

    forced_valid = (
        forced_df["ttl_mean"].notna()
    )

    forced_df.loc[
        forced_valid,
        "ttl_mean",
    ] = 128

    forced_df[
        config.LABEL_COLUMN
    ] = config.LABEL_ATTACK

    forced_df[
        "augmentation_type"
    ] = "ttl_forced_128"

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    augmented_df = pd.concat(
        [
            small_df,
            large_df,
            forced_df,
        ],
        axis=0,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Important validation
    # --------------------------------------------------------

    if len(augmented_df) == 0:
        raise RuntimeError(
            "No augmented samples were created."
        )

    if not (
        augmented_df[
            config.LABEL_COLUMN
        ] == config.LABEL_ATTACK
    ).all():
        raise RuntimeError(
            "Augmented samples contain "
            "non-Attack labels."
        )

    # Make sure original attack data was not modified.
    # We created copies throughout, so this should always pass.
    print(
        f"\nAugmented samples created: "
        f"{len(augmented_df):,}"
    )

    print(
        "\nAugmentation distribution:"
    )

    print(
        augmented_df[
            "augmentation_type"
        ]
        .value_counts()
        .to_string()
    )

    return augmented_df


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train a robustness-enhanced "
            "Random Forest using controlled "
            "TTL augmentation."
        )
    )

    parser.add_argument(
        "--train",
        default=str(
            config.PROCESSED_DIR
            / "split"
            / "train_dataset.csv"
        ),
    )

    parser.add_argument(
        "--test",
        default=str(
            config.PROCESSED_DIR
            / "split"
            / "test_dataset.csv"
        ),
    )

    parser.add_argument(
        "--out-model",
        default=str(
            ROBUST_MODEL_FILE
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # Header
    # ========================================================

    print("=" * 70)
    print(
        "NetworkIDS - Robust Random Forest Training"
    )
    print("=" * 70)

    print(
        "\nThis experiment:"
    )
    print(
        "  - keeps the original training samples"
    )
    print(
        "  - augments Attack samples only"
    )
    print(
        "  - perturbs ttl_mean only"
    )
    print(
        "  - never modifies the held-out test set"
    )
    print(
        "  - keeps all 42 ML features"
    )

    # ========================================================
    # 1. Load datasets
    # ========================================================

    print(
        "\n[1/8] Loading pre-split datasets..."
    )

    train_df = load_dataset(
        Path(args.train),
        "Train",
    )

    test_df = load_dataset(
        Path(args.test),
        "Test",
    )

    # ========================================================
    # 2. Leakage checks
    # ========================================================

    print(
        "\n[2/8] Checking train/test leakage..."
    )

    check_flow_id_leakage(
        train_df,
        test_df,
    )

    check_feature_group_leakage(
        train_df,
        test_df,
    )

    print(
        "[OK] Original train/test leakage checks passed."
    )

    # ========================================================
    # 3. Prepare original X/y
    # ========================================================

    print(
        "\n[3/8] Preparing original feature matrices..."
    )

    X_train_original, y_train_original = (
        prepare_xy(
            train_df,
            "Train",
        )
    )

    X_test, y_test = prepare_xy(
        test_df,
        "Test",
    )

    print(
        f"  Original X_train : "
        f"{X_train_original.shape}"
    )

    print(
        f"  X_test           : "
        f"{X_test.shape}"
    )

    print_label_distribution(
        "Original training",
        y_train_original,
    )

    print_label_distribution(
        "Test",
        y_test,
    )

    # ========================================================
    # 4. Create augmentation
    # ========================================================

    print(
        "\n[4/8] Creating TTL-augmented Attack samples..."
    )

    attack_train_df = train_df[
        train_df[
            config.LABEL_COLUMN
        ].astype(int)
        == config.LABEL_ATTACK
    ].copy()

    augmented_df = (
        create_ttl_augmented_attack_samples(
            attack_train_df
        )
    )

    # ========================================================
    # 5. Build augmented training dataset
    # ========================================================

    print(
        "\n[5/8] Building robust training dataset..."
    )

    # Original training rows
    original_training_df = (
        train_df.copy()
    )

    original_training_df[
        "augmentation_type"
    ] = "original"

    # Combine original + augmented
    robust_training_df = pd.concat(
        [
            original_training_df,
            augmented_df,
        ],
        axis=0,
        ignore_index=True,
    )

    print(
        f"  Original rows  : "
        f"{len(original_training_df):,}"
    )

    print(
        f"  Augmented rows : "
        f"{len(augmented_df):,}"
    )

    print(
        f"  Total rows     : "
        f"{len(robust_training_df):,}"
    )

    # --------------------------------------------------------
    # Validate augmented dataset
    # --------------------------------------------------------

    if (
        robust_training_df[
            config.LABEL_COLUMN
        ].isna().any()
    ):
        raise RuntimeError(
            "Robust training dataset contains "
            "NaN labels."
        )

    robust_labels = (
        robust_training_df[
            config.LABEL_COLUMN
        ].astype(int)
    )

    if not set(
        robust_labels.unique()
    ).issubset({0, 1}):
        raise RuntimeError(
            "Unexpected labels found "
            "in robust training dataset."
        )

    # --------------------------------------------------------
    # Check augmentation counts
    # --------------------------------------------------------

    print(
        "\nTraining dataset composition:"
    )

    print(
        robust_training_df[
            "augmentation_type"
        ]
        .value_counts()
        .to_string()
    )

    print_label_distribution(
        "Robust training",
        robust_labels,
    )

    # --------------------------------------------------------
    # Save robust training dataset
    # --------------------------------------------------------

    ROBUST_TRAINING_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    robust_training_df.to_csv(
        ROBUST_TRAINING_FILE,
        index=False,
    )

    print(
        f"\nRobust training dataset saved:"
        f"\n{ROBUST_TRAINING_FILE}"
    )

    # ========================================================
    # 6. Prepare robust X/y
    # ========================================================

    print(
        "\n[6/8] Preparing robust training features..."
    )

    X_train_robust, y_train_robust = (
        prepare_xy(
            robust_training_df,
            "Robust Train",
        )
    )

    print(
        f"  X_train_robust : "
        f"{X_train_robust.shape}"
    )

    print(
        f"  X_test         : "
        f"{X_test.shape}"
    )

    # ========================================================
    # 7. Train Robust Random Forest
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "Training Robust Random Forest"
    )

    print(
        "=" * 70
    )

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
        X_train_robust,
        y_train_robust,
    )

    print(
        "  Robust Random Forest training completed."
    )

    # ========================================================
    # 8. Evaluate on untouched test set
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "Evaluation on Untouched Held-Out Test Set"
    )

    print(
        "=" * 70
    )

    y_pred = clf.predict(
        X_test
    )

    y_proba = clf.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    auc = roc_auc_score(
        y_test,
        y_proba,
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
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

    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Benign",
                "Attack",
            ],
            digits=4,
            zero_division=0,
        )
    )

    print(
        "Confusion matrix:"
    )

    print()

    print(
        "                 Predicted"
    )

    print(
        "                 Benign     Attack"
    )

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

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "Top 15 Feature Importances (Gini)"
    )

    print(
        "=" * 70
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

    for feature, value in (
        importances.head(15).items()
    ):
        print(
            f"  {feature:<30} "
            f"{value:.4f}"
        )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    out_model = Path(
        args.out_model
    )

    out_model.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        clf,
        out_model,
    )

    print(
        f"\nRobust model saved:"
        f"\n{out_model}"
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics = {
        "model": "RandomForestClassifier",
        "model_type": "robust_ttl_augmented",
        "random_state": RANDOM_STATE,
        "n_estimators": N_ESTIMATORS,
        "class_weight": "balanced",

        "n_original_train": int(
            len(train_df)
        ),

        "n_augmented": int(
            len(augmented_df)
        ),

        "n_robust_train": int(
            len(robust_training_df)
        ),

        "n_test": int(
            len(X_test)
        ),

        "n_features": int(
            len(config.ML_FEATURES)
        ),

        "augmentation": {
            "ttl_small_ratio": TTL_SMALL_RATIO,
            "ttl_large_ratio": TTL_LARGE_RATIO,
            "ttl_forced_ratio": TTL_FORCED_RATIO,
            "ttl_small_samples": int(
                (augmented_df[
                    "augmentation_type"
                ] == "ttl_jitter_small").sum()
            ),
            "ttl_large_samples": int(
                (augmented_df[
                    "augmentation_type"
                ] == "ttl_jitter_large").sum()
            ),
            "ttl_forced_samples": int(
                (augmented_df[
                    "augmentation_type"
                ] == "ttl_forced_128").sum()
            ),
        },

        "label_train": {
            str(int(k)): int(v)
            for k, v in (
                y_train_robust
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

        "confusion_matrix": (
            cm.tolist()
        ),

        "top_15_importances": {
            str(k): float(v)
            for k, v in (
                importances
                .head(15)
                .items()
            )
        },
    }

    with open(
        ROBUST_METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )

    print(
        f"Metrics saved:"
        f"\n{ROBUST_METRICS_FILE}"
    )

    # ========================================================
    # Final summary
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "ROBUST TRAINING COMPLETE - PASS"
    )

    print(
        "=" * 70
    )

    print(
        f"\nOriginal train rows : "
        f"{len(train_df):,}"
    )

    print(
        f"Augmented rows      : "
        f"{len(augmented_df):,}"
    )

    print(
        f"Robust train rows   : "
        f"{len(robust_training_df):,}"
    )

    print(
        f"Test rows           : "
        f"{len(X_test):,}"
    )

    print(
        f"Features            : "
        f"{len(config.ML_FEATURES)}"
    )

    print(
        f"Accuracy            : "
        f"{accuracy:.4f}"
    )

    print(
        f"F1-score            : "
        f"{f1:.4f}"
    )

    print(
        f"AUC-ROC             : "
        f"{auc:.4f}"
    )

    print(
        f"Model               : "
        f"{out_model}"
    )


if __name__ == "__main__":
    main()