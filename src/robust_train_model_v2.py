"""
robust_train_model_v2.py
---------------------------
NetworkIDS - Robust Random Forest v2
(v1 base + attack-only OOF hard-example mining)

Lineage of this design:
    v2 original (full-class TTL forcing) : eliminated TTL dependency
        but shifted it onto rst_count (34.35% RST evasion).

    v2 hybrid+OOF (benign+attack hard oversampling) :
        reduced FP slightly (1016 -> 975) but increased FN
        substantially (150 -> 234).

        4,375 of the 5,381 oversampled hard examples were BENIGN.
        The model was therefore exposed more heavily to difficult
        benign samples, while some previously correctly detected
        attacks became false negatives.

This version starts from v1's proportional attack-only TTL
augmentation and adds ONLY attack-side hard-example oversampling.

Structure:
    Part 1 (from v1, unchanged):
        ATTACK ROWS ONLY
            30% of attack rows -> TTL +/- 5
            15% of attack rows -> TTL +/- 30
             5% of attack rows -> TTL forced to 128

    Part 2 (new):
        OOF (5-fold) hard-example mining.

        A hard example is:
            - OOF prediction != true label
              OR
            - OOF P(Attack) is between 0.40 and 0.60

        Only ATTACK-LABEL hard examples are oversampled.

        Benign hard examples are identified and reported for
        transparency, but deliberately NOT oversampled.

    Attack hard-example oversampling:
        +4 extra copies per hard attack row.

Leakage guardrails:
    - Flow IDs must not overlap between train and test.
    - Exact 42-feature vectors must not overlap between train
      and test.
    - OOF predictions are generated entirely inside TRAIN.
    - Test data is never used during OOF mining.
    - Test data is never duplicated or augmented.

Usage:
    python src\\robust_train_model_v2.py
"""

import json
import sys
import time
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
from sklearn.model_selection import StratifiedKFold


# ---------------------------------------------------------------------
# Project import
# ---------------------------------------------------------------------

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

import src.config as config


# =====================================================================
# Configuration
# =====================================================================

RANDOM_STATE = 42

# Final Random Forest
N_ESTIMATORS = 200
N_JOBS = -1


# ---------------------------------------------------------------------
# Part 1: V1 proportional attack-only TTL augmentation
# ---------------------------------------------------------------------

TTL_SMALL_RATIO = 0.30
TTL_LARGE_RATIO = 0.15
TTL_FORCED_RATIO = 0.05

TTL_SMALL_RANGE = (-5, 5)
TTL_LARGE_RANGE = (-30, 30)

TTL_FORCED_VALUE = 128


# ---------------------------------------------------------------------
# Part 2: OOF hard-example mining
# ---------------------------------------------------------------------

OOF_FOLDS = 5
OOF_ESTIMATORS = 100

UNCERTAIN_LOW = 0.40
UNCERTAIN_HIGH = 0.60

# Number of EXTRA copies for each hard ATTACK row.
#
# Original attack row is always retained.
# Therefore:
#     1 original + 4 extra copies = 5 total occurrences
#
ATTACK_HARD_EXTRA_COPIES = 4


# ---------------------------------------------------------------------
# Input / output paths
# ---------------------------------------------------------------------

TRAIN_FILE = (
    config.PROCESSED_DIR
    / "split"
    / "train_dataset.csv"
)

TEST_FILE = (
    config.PROCESSED_DIR
    / "split"
    / "test_dataset.csv"
)

MODEL_OUTPUT = (
    config.MODELS_DIR
    / "random_forest_robust_v2.pkl"
)

METRICS_OUTPUT = (
    config.MODELS_DIR
    / "robust_v2_training_metrics.json"
)

DATASET_OUTPUT = (
    config.PROCESSED_DIR
    / "robust_v2_training_dataset.csv"
)


# =====================================================================
# Leakage checks
# =====================================================================

def check_flow_id_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """
    Check that no flow_id occurs in both train and test.
    """

    train_ids = set(
        train_df["flow_id"].astype(str)
    )

    test_ids = set(
        test_df["flow_id"].astype(str)
    )

    overlap = train_ids & test_ids

    print(
        f"  Flow ID overlap: {len(overlap):,}"
    )

    if overlap:
        raise ValueError(
            "Data leakage detected: "
            "flow IDs overlap between train and test."
        )


def check_feature_vector_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """
    Check that no identical 42-feature vector occurs in both
    train and test.

    This is stricter than the flow_id check because different
    flow IDs can still have exactly the same ML feature vector.

    NaN values are treated consistently by constructing a
    MultiIndex from the complete feature matrix.
    """

    print(
        "  Checking 42-feature vector overlap..."
    )

    train_feature_keys = pd.MultiIndex.from_frame(
        train_df[config.ML_FEATURES]
    )

    test_feature_keys = pd.MultiIndex.from_frame(
        test_df[config.ML_FEATURES]
    )

    overlap = train_feature_keys.intersection(
        test_feature_keys
    )

    print(
        f"  42-feature vector overlap: "
        f"{len(overlap):,}"
    )

    if len(overlap) > 0:
        raise ValueError(
            "Data leakage detected: "
            "identical 42-feature vectors overlap "
            "between train and test."
        )


def check_required_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """
    Validate that both datasets contain the required
    metadata, ML features and labels.
    """

    required_columns = (
        config.METADATA_COLUMNS
        + config.ML_FEATURES
        + [
            config.LABEL_COLUMN,
            "original_label",
        ]
    )

    missing_train = [
        column
        for column in required_columns
        if column not in train_df.columns
    ]

    missing_test = [
        column
        for column in required_columns
        if column not in test_df.columns
    ]

    if missing_train:
        raise ValueError(
            "Training dataset missing columns: "
            f"{missing_train}"
        )

    if missing_test:
        raise ValueError(
            "Test dataset missing columns: "
            f"{missing_test}"
        )


def check_labels(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """
    Validate binary labels.
    """

    train_labels = set(
        train_df[
            config.LABEL_COLUMN
        ]
        .dropna()
        .unique()
    )

    test_labels = set(
        test_df[
            config.LABEL_COLUMN
        ]
        .dropna()
        .unique()
    )

    if not train_labels.issubset({0, 1}):
        raise ValueError(
            f"Unexpected training labels: "
            f"{train_labels}"
        )

    if not test_labels.issubset({0, 1}):
        raise ValueError(
            f"Unexpected test labels: "
            f"{test_labels}"
        )

    print(
        f"  Training labels: "
        f"{sorted(train_labels)}"
    )

    print(
        f"  Test labels: "
        f"{sorted(test_labels)}"
    )


# =====================================================================
# Part 1: V1 TTL augmentation
# =====================================================================

def ttl_jitter_subset(
    df: pd.DataFrame,
    indices,
    low: int,
    high: int,
    seed: int,
):
    """
    Create a TTL-jittered copy of selected rows.

    Only non-NaN ttl_mean values are modified.

    NaN values remain NaN.
    """

    out = df.loc[indices].copy()

    rng = np.random.RandomState(
        seed
    )

    valid = out["ttl_mean"].notna()

    noise = rng.randint(
        low,
        high + 1,
        size=len(out),
    )

    valid_noise = noise[
        valid.to_numpy()
    ]

    out.loc[
        valid,
        "ttl_mean"
    ] = (
        out.loc[
            valid,
            "ttl_mean"
        ].to_numpy()
        + valid_noise
    ).clip(
        1,
        255,
    )

    return out


def build_v1_ttl_augmentation(
    attack_df: pd.DataFrame,
):
    """
    Reproduce the V1 TTL augmentation exactly.

    Attack rows only:
        30% -> TTL +/- 5
        15% -> TTL +/- 30
         5% -> TTL = 128
    """

    print(
        "\nPart 1: proportional TTL augmentation "
        "(attack rows only, from v1)"
    )

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    n = len(attack_df)

    n_small = int(
        n * TTL_SMALL_RATIO
    )

    n_large = int(
        n * TTL_LARGE_RATIO
    )

    n_forced = int(
        n * TTL_FORCED_RATIO
    )

    # -------------------------------------------------------------
    # TTL +/- 5
    # -------------------------------------------------------------

    small_idx = rng.choice(
        attack_df.index,
        size=n_small,
        replace=False,
    )

    small_df = ttl_jitter_subset(
        attack_df,
        small_idx,
        *TTL_SMALL_RANGE,
        seed=101,
    )

    # -------------------------------------------------------------
    # TTL +/- 30
    # -------------------------------------------------------------

    large_idx = rng.choice(
        attack_df.index,
        size=n_large,
        replace=False,
    )

    large_df = ttl_jitter_subset(
        attack_df,
        large_idx,
        *TTL_LARGE_RANGE,
        seed=102,
    )

    # -------------------------------------------------------------
    # TTL = 128
    # -------------------------------------------------------------

    forced_idx = rng.choice(
        attack_df.index,
        size=n_forced,
        replace=False,
    )

    forced_df = attack_df.loc[
        forced_idx
    ].copy()

    valid = forced_df[
        "ttl_mean"
    ].notna()

    forced_df.loc[
        valid,
        "ttl_mean"
    ] = TTL_FORCED_VALUE

    # -------------------------------------------------------------
    # Report
    # -------------------------------------------------------------

    print(
        f"  TTL +/- 5   : "
        f"{len(small_df):,} rows"
    )

    print(
        f"  TTL +/- 30  : "
        f"{len(large_df):,} rows"
    )

    print(
        f"  TTL = 128   : "
        f"{len(forced_df):,} rows"
    )

    ttl_augmented = pd.concat(
        [
            small_df,
            large_df,
            forced_df,
        ],
        ignore_index=True,
    )

    print(
        f"  Total TTL augmented: "
        f"{len(ttl_augmented):,} rows"
    )

    return ttl_augmented


# =====================================================================
# Part 2: OOF hard-example mining
# =====================================================================

def compute_oof_predictions(
    train_df: pd.DataFrame,
    n_folds: int,
    n_estimators: int,
):
    """
    Generate out-of-fold predictions for the TRAIN set only.

    Every training row receives a prediction from a model that
    was NOT trained on that row.

    The test set is not used here.
    """

    X = train_df[
        config.ML_FEATURES
    ]

    y = train_df[
        config.LABEL_COLUMN
    ].astype(int).to_numpy()

    oof_pred = np.zeros(
        len(train_df),
        dtype=int,
    )

    oof_proba = np.zeros(
        len(train_df),
        dtype=float,
    )

    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    print(
        f"\nPart 2: {n_folds}-fold OOF "
        f"hard-example mining "
        f"({n_estimators} trees/fold)"
    )

    for fold, (
        tr_idx,
        va_idx,
    ) in enumerate(
        skf.split(X, y),
        start=1,
    ):

        t0 = time.time()

        print(
            f"    Fold "
            f"{fold}/{n_folds}..."
        )

        fold_clf = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=(
                RANDOM_STATE + fold
            ),
            n_jobs=N_JOBS,
        )

        fold_clf.fit(
            X.iloc[tr_idx],
            y[tr_idx],
        )

        oof_pred[
            va_idx
        ] = fold_clf.predict(
            X.iloc[va_idx]
        )

        oof_proba[
            va_idx
        ] = fold_clf.predict_proba(
            X.iloc[va_idx]
        )[:, 1]

        elapsed_minutes = (
            time.time() - t0
        ) / 60

        print(
            f"    Fold "
            f"{fold}/{n_folds} done in "
            f"{elapsed_minutes:.1f} min"
        )

        del fold_clf

    return (
        oof_pred,
        oof_proba,
    )


def identify_hard_examples(
    train_df: pd.DataFrame,
    oof_pred: np.ndarray,
    oof_proba: np.ndarray,
):
    """
    Identify OOF hard examples.

    Hard example:
        prediction != true label
        OR
        P(Attack) in [0.40, 0.60]

    Only hard ATTACK rows are returned for oversampling.

    Hard BENIGN rows are reported but not oversampled.
    """

    y_train = train_df[
        config.LABEL_COLUMN
    ].astype(int).to_numpy()

    # -------------------------------------------------------------
    # Misclassified OOF samples
    # -------------------------------------------------------------

    misclassified = (
        oof_pred != y_train
    )

    # -------------------------------------------------------------
    # Borderline / uncertain samples
    # -------------------------------------------------------------

    uncertain = (
        (oof_proba >= UNCERTAIN_LOW)
        &
        (oof_proba <= UNCERTAIN_HIGH)
    )

    # -------------------------------------------------------------
    # Combined hard-example set
    # -------------------------------------------------------------

    hard_mask = (
        misclassified
        |
        uncertain
    )

    is_attack = (
        y_train
        == config.LABEL_ATTACK
    )

    hard_attack_mask = (
        hard_mask
        &
        is_attack
    )

    hard_benign_mask = (
        hard_mask
        &
        (~is_attack)
    )

    print(
        "\nOOF hard-example analysis:"
    )

    print(
        f"  OOF misclassified: "
        f"{int(misclassified.sum()):,}"
    )

    print(
        f"  OOF uncertain "
        f"[{UNCERTAIN_LOW:.2f}, "
        f"{UNCERTAIN_HIGH:.2f}]: "
        f"{int(uncertain.sum()):,}"
    )

    print(
        f"  Hard examples union: "
        f"{int(hard_mask.sum()):,}"
    )

    print(
        f"    Hard Attack: "
        f"{int(hard_attack_mask.sum()):,}"
    )

    print(
        f"    Hard Benign: "
        f"{int(hard_benign_mask.sum()):,}"
    )

    print(
        "\n  Oversampling policy:"
    )

    print(
        f"    Attack hard examples: "
        f"+{ATTACK_HARD_EXTRA_COPIES} "
        f"extra copies"
    )

    print(
        "    Benign hard examples: "
        "0 extra copies"
    )

    return (
        hard_attack_mask,
        hard_benign_mask,
        hard_mask,
    )


# =====================================================================
# Training dataset construction
# =====================================================================

def build_attack_hard_augmentation(
    train_df: pd.DataFrame,
    hard_attack_mask: np.ndarray,
):
    """
    Create extra copies of hard ATTACK rows only.

    Original rows remain in the training dataset.

    ATTACK_HARD_EXTRA_COPIES = 4 means:

        original row
        + 4 extra copies
        = 5 total occurrences
    """

    hard_attack_rows = train_df[
        hard_attack_mask
    ].copy()

    attack_oversampled = pd.concat(
        [
            hard_attack_rows
        ]
        * ATTACK_HARD_EXTRA_COPIES,
        ignore_index=True,
    )

    print(
        f"\nAttack hard-example extra copies "
        f"({ATTACK_HARD_EXTRA_COPIES}x): "
        f"{len(attack_oversampled):,}"
    )

    return attack_oversampled


def build_augmented_training_dataset(
    train_df: pd.DataFrame,
    ttl_augmented: pd.DataFrame,
    attack_oversampled: pd.DataFrame,
):
    """
    Combine:

        Original train
        +
        V1 TTL augmentation
        +
        Attack-only hard-example augmentation
    """

    augmented_train = pd.concat(
        [
            train_df,
            ttl_augmented,
            attack_oversampled,
        ],
        ignore_index=True,
    )

    # Shuffle after augmentation.
    augmented_train = (
        augmented_train
        .sample(
            frac=1.0,
            random_state=RANDOM_STATE,
        )
        .reset_index(drop=True)
    )

    return augmented_train


# =====================================================================
# Final model training and evaluation
# =====================================================================

def train_final_model(
    augmented_train: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """
    Train final Random Forest and evaluate ONLY on untouched test set.
    """

    X_train = augmented_train[
        config.ML_FEATURES
    ]

    y_train = augmented_train[
        config.LABEL_COLUMN
    ].astype(int)

    X_test = test_df[
        config.ML_FEATURES
    ]

    y_test = test_df[
        config.LABEL_COLUMN
    ].astype(int)

    print(
        "\n" + "=" * 70
    )

    print(
        "Training final Robust RF v2"
    )

    print(
        "=" * 70
    )

    print(
        f"  X_train: {X_train.shape}"
    )

    print(
        f"  X_test : {X_test.shape}"
    )

    print(
        f"  Train NaN cells: "
        f"{int(X_train.isna().sum().sum()):,}"
    )

    print(
        f"  Test NaN cells : "
        f"{int(X_test.isna().sum().sum()):,}"
    )

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    print(
        f"\nRandom Forest:"
    )

    print(
        f"  Trees: "
        f"{N_ESTIMATORS}"
    )

    print(
        f"  class_weight: "
        f"balanced"
    )

    print(
        f"  random_state: "
        f"{RANDOM_STATE}"
    )

    clf.fit(
        X_train,
        y_train,
    )

    print(
        "  Training complete."
    )

    # -------------------------------------------------------------
    # Clean evaluation
    # -------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "Clean Evaluation "
        "(untouched test set)"
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
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    fpr = (
        fp
        / (fp + tn)
    )

    fnr = (
        fn
        / (fn + tp)
    )

    # -------------------------------------------------------------
    # Print metrics
    # -------------------------------------------------------------

    print(
        f"  Accuracy  : "
        f"{accuracy:.4f}"
    )

    print(
        f"  Precision : "
        f"{precision:.4f}"
    )

    print(
        f"  Recall    : "
        f"{recall:.4f}"
    )

    print(
        f"  F1-score  : "
        f"{f1:.4f}"
    )

    print(
        f"  AUC-ROC   : "
        f"{auc:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        f"  TN: {tn:,}"
    )

    print(
        f"  FP: {fp:,}"
    )

    print(
        f"  FN: {fn:,}"
    )

    print(
        f"  TP: {tp:,}"
    )

    print(
        f"\nFPR: "
        f"{fpr * 100:.4f}%"
    )

    print(
        f"FNR: "
        f"{fnr * 100:.4f}%"
    )

    # -------------------------------------------------------------
    # Comparison reference
    # -------------------------------------------------------------

    print(
        "\nReference comparison:"
    )

    print(
        "  Baseline:"
        " FP=1,016 | FN=150"
    )

    print(
        "  V1:"
        " FP=1,016 | FN=180"
    )

    print(
        "  Hybrid+OOF:"
        " FP=975 | FN=234"
    )

    print(
        f"  Current V2:"
        f" FP={fp:,} | FN={fn:,}"
    )

    # -------------------------------------------------------------
    # Classification report
    # -------------------------------------------------------------

    print(
        "\nClassification Report:"
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

    # -------------------------------------------------------------
    # Feature importance
    # -------------------------------------------------------------

    importances = (
        pd.Series(
            clf.feature_importances_,
            index=config.ML_FEATURES,
        )
        .sort_values(
            ascending=False
        )
    )

    print(
        "Top 15 feature importances:"
    )

    for (
        name,
        value,
    ) in importances.head(15).items():

        print(
            f"  {name:<30} "
            f"{value:.4f}"
        )

    metrics = {
        "model_type": (
            "RandomForestClassifier "
            "(v1 base + attack-only "
            "OOF hard-example oversampling)"
        ),

        "random_state": RANDOM_STATE,

        "n_estimators": N_ESTIMATORS,

        "class_weight": "balanced",

        "part1_ttl_augmentation": {
            "small_ratio": (
                TTL_SMALL_RATIO
            ),
            "large_ratio": (
                TTL_LARGE_RATIO
            ),
            "forced_ratio": (
                TTL_FORCED_RATIO
            ),
            "small_range": list(
                TTL_SMALL_RANGE
            ),
            "large_range": list(
                TTL_LARGE_RANGE
            ),
            "forced_value": (
                TTL_FORCED_VALUE
            ),
        },

        "part2_hard_mining": {
            "oof_folds": OOF_FOLDS,
            "oof_estimators": OOF_ESTIMATORS,

            "uncertain_range": [
                UNCERTAIN_LOW,
                UNCERTAIN_HIGH,
            ],

            "n_hard_attack": int(
                (
                    # This value will be overwritten
                    # in main() with the actual count.
                    0
                )
            ),

            "n_hard_benign_not_oversampled": int(
                0
            ),

            "extra_copies_attack": (
                ATTACK_HARD_EXTRA_COPIES
            ),
        },

        "n_train_original": int(
            0
        ),

        "n_ttl_augmented": int(
            0
        ),

        "n_attack_hard_augmented": int(
            0
        ),

        "n_train_augmented": int(
            len(augmented_train)
        ),

        "test_metrics": {
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
            "true_negative": int(
                tn
            ),
            "false_positive": int(
                fp
            ),
            "false_negative": int(
                fn
            ),
            "true_positive": int(
                tp
            ),
            "fpr": float(
                fpr
            ),
            "fnr": float(
                fnr
            ),
            "confusion_matrix": (
                cm.tolist()
            ),
        },

        "top_15_importances": {
            key: float(value)
            for key, value
            in importances.head(15).items()
        },
    }

    return (
        clf,
        metrics,
    )


# =====================================================================
# Main
# =====================================================================

def main():

    print(
        "=" * 70
    )

    print(
        "NetworkIDS - Robust RF v2"
    )

    print(
        "v1 base + attack-only OOF hard mining"
    )

    print(
        "=" * 70
    )

    # -------------------------------------------------------------
    # Load datasets
    # -------------------------------------------------------------

    print(
        "\nLoading datasets..."
    )

    train_df = pd.read_csv(
        TRAIN_FILE,
        low_memory=False,
    )

    test_df = pd.read_csv(
        TEST_FILE,
        low_memory=False,
    )

    print(
        f"  Train: "
        f"{len(train_df):,} rows"
    )

    print(
        f"  Test : "
        f"{len(test_df):,} rows "
        f"(untouched)"
    )

    # -------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------

    print(
        "\nDataset validation:"
    )

    check_required_columns(
        train_df,
        test_df,
    )

    check_labels(
        train_df,
        test_df,
    )

    # Make labels integer after validation.
    train_df[
        config.LABEL_COLUMN
    ] = train_df[
        config.LABEL_COLUMN
    ].astype(int)

    test_df[
        config.LABEL_COLUMN
    ] = test_df[
        config.LABEL_COLUMN
    ].astype(int)

    # -------------------------------------------------------------
    # Leakage guardrails
    # -------------------------------------------------------------

    print(
        "\nLeakage guardrail:"
    )

    check_flow_id_leakage(
        train_df,
        test_df,
    )

    check_feature_vector_leakage(
        train_df,
        test_df,
    )

    print(
        "  Leakage checks PASS."
    )

    # -------------------------------------------------------------
    # Original class distribution
    # -------------------------------------------------------------

    attack_df = train_df[
        train_df[
            config.LABEL_COLUMN
        ]
        == config.LABEL_ATTACK
    ]

    benign_df = train_df[
        train_df[
            config.LABEL_COLUMN
        ]
        == config.LABEL_BENIGN
    ]

    print(
        "\nOriginal training distribution:"
    )

    print(
        f"  Benign: "
        f"{len(benign_df):,} "
        f"({len(benign_df) / len(train_df) * 100:.2f}%)"
    )

    print(
        f"  Attack: "
        f"{len(attack_df):,} "
        f"({len(attack_df) / len(train_df) * 100:.2f}%)"
    )

    # -------------------------------------------------------------
    # Part 1: V1 TTL augmentation
    # -------------------------------------------------------------

    ttl_augmented = (
        build_v1_ttl_augmentation(
            attack_df
        )
    )

    # -------------------------------------------------------------
    # Part 2: OOF hard-example mining
    # -------------------------------------------------------------

    (
        oof_pred,
        oof_proba,
    ) = compute_oof_predictions(
        train_df,
        OOF_FOLDS,
        OOF_ESTIMATORS,
    )

    (
        hard_attack_mask,
        hard_benign_mask,
        hard_mask,
    ) = identify_hard_examples(
        train_df,
        oof_pred,
        oof_proba,
    )

    # -------------------------------------------------------------
    # Attack-only hard-example augmentation
    # -------------------------------------------------------------

    attack_oversampled = (
        build_attack_hard_augmentation(
            train_df,
            hard_attack_mask,
        )
    )

    # -------------------------------------------------------------
    # Combine
    # -------------------------------------------------------------

    augmented_train = (
        build_augmented_training_dataset(
            train_df,
            ttl_augmented,
            attack_oversampled,
        )
    )

    # -------------------------------------------------------------
    # Final class distribution
    # -------------------------------------------------------------

    final_counts = (
        augmented_train[
            config.LABEL_COLUMN
        ]
        .value_counts()
        .sort_index()
    )

    final_benign = int(
        final_counts.get(
            config.LABEL_BENIGN,
            0,
        )
    )

    final_attack = int(
        final_counts.get(
            config.LABEL_ATTACK,
            0,
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "Final augmented training dataset"
    )

    print(
        "=" * 70
    )

    print(
        f"  Original rows: "
        f"{len(train_df):,}"
    )

    print(
        f"  TTL augmented: "
        f"{len(ttl_augmented):,}"
    )

    print(
        f"  Attack hard-example copies: "
        f"{len(attack_oversampled):,}"
    )

    print(
        f"  Final rows: "
        f"{len(augmented_train):,}"
    )

    print(
        f"  Expansion: "
        f"{len(augmented_train) / len(train_df):.2f}x"
    )

    print(
        f"  Benign: "
        f"{final_benign:,} "
        f"({final_benign / len(augmented_train) * 100:.2f}%)"
    )

    print(
        f"  Attack: "
        f"{final_attack:,} "
        f"({final_attack / len(augmented_train) * 100:.2f}%)"
    )

    # -------------------------------------------------------------
    # Save augmented training dataset
    # -------------------------------------------------------------

    DATASET_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    augmented_train.to_csv(
        DATASET_OUTPUT,
        index=False,
    )

    print(
        f"\nTraining dataset saved:"
        f"\n  {DATASET_OUTPUT}"
    )

    # -------------------------------------------------------------
    # Train final model
    # -------------------------------------------------------------

    (
        clf,
        metrics,
    ) = train_final_model(
        augmented_train,
        test_df,
    )

    # -------------------------------------------------------------
    # Complete metadata
    # -------------------------------------------------------------

    metrics[
        "part2_hard_mining"
    ][
        "n_hard_attack"
    ] = int(
        hard_attack_mask.sum()
    )

    metrics[
        "part2_hard_mining"
    ][
        "n_hard_benign_not_oversampled"
    ] = int(
        hard_benign_mask.sum()
    )

    metrics[
        "part2_hard_mining"
    ][
        "n_hard_total"
    ] = int(
        hard_mask.sum()
    )

    metrics[
        "part2_hard_mining"
    ][
        "n_oof_misclassified"
    ] = int(
        (
            oof_pred
            != train_df[
                config.LABEL_COLUMN
            ].astype(int).to_numpy()
        ).sum()
    )

    metrics[
        "part2_hard_mining"
    ][
        "n_oof_uncertain"
    ] = int(
        (
            (
                oof_proba
                >= UNCERTAIN_LOW
            )
            &
            (
                oof_proba
                <= UNCERTAIN_HIGH
            )
        ).sum()
    )

    metrics[
        "n_train_original"
    ] = int(
        len(train_df)
    )

    metrics[
        "n_ttl_augmented"
    ] = int(
        len(ttl_augmented)
    )

    metrics[
        "n_attack_hard_augmented"
    ] = int(
        len(attack_oversampled)
    )

    # -------------------------------------------------------------
    # Save model
    # -------------------------------------------------------------

    config.MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        clf,
        MODEL_OUTPUT,
    )

    print(
        f"\nModel saved:"
        f"\n  {MODEL_OUTPUT}"
    )

    # -------------------------------------------------------------
    # Save metrics
    # -------------------------------------------------------------

    with open(
        METRICS_OUTPUT,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )

    print(
        f"\nMetrics saved:"
        f"\n  {METRICS_OUTPUT}"
    )

    # -------------------------------------------------------------
    # Next step
    # -------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nNext step:"
    )

    print(
        "  python src\\robustness_evaluation.py "
        "--model models\\random_forest_robust_v2.pkl "
        "--out-name robustness_results_robust_v2.csv"
    )


if __name__ == "__main__":
    main()