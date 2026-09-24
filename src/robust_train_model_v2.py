"""
NetworkIDS - Robust Random Forest v2 (HYBRID with OOF hard-example mining)

Goal:
    Reduce FP/FN while retaining TTL robustness.

Strategy:
    Part 1: 20% of each class receives TTL ±30 jitter.
    Part 2: Hard-example oversampling using out-of-fold (OOF)
            predictions on the TRAIN set. Each training row is scored
            by a model that did NOT see it during training, so the
            hardness measure reflects genuine generalisation
            difficulty rather than training-set memorisation.

Hard-example definition (union):
    - OOF-predicted label != true label, OR
    - OOF attack probability in [0.40, 0.60] (near decision boundary)

Critical leakage guardrail:
    OOF predictions are computed entirely within the TRAIN set
    (5-fold cross-validation). The test set is never touched, never
    scored, never duplicated.

Usage:
    python src/robust_train_model_v2.py
    python src/robust_train_model_v2.py --oof-fold 5 --oof-estimators 50
    python src/robust_train_model_v2.py --skip-oof   # fallback: baseline train predict
"""

import argparse
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

sys.path.append(str(Path(__file__).resolve().parents[1]))
import src.config as config


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42
N_ESTIMATORS = 200
N_JOBS = -1

# Part 1
AUGMENT_FRACTION = 0.20
TTL_JITTER_MIN = -30
TTL_JITTER_MAX = 30

# Part 2
UNCERTAIN_LOW = 0.40
UNCERTAIN_HIGH = 0.60
HARD_EXTRA_COPIES = 2

# OOF settings
OOF_FOLDS = 5
OOF_ESTIMATORS = 100   # lightweight for speed; not the final model

TRAIN_FILE = config.PROCESSED_DIR / "split" / "train_dataset.csv"
TEST_FILE  = config.PROCESSED_DIR / "split" / "test_dataset.csv"

MODEL_OUTPUT   = config.MODELS_DIR / "random_forest_robust_v2.pkl"
METRICS_OUTPUT = config.MODELS_DIR / "robust_v2_training_metrics.json"
DATASET_OUTPUT = config.PROCESSED_DIR / "robust_v2_training_dataset.csv"


# ============================================================
# Helpers
# ============================================================

def check_flow_id_leakage(train_df, test_df):
    train_ids = set(train_df["flow_id"].astype(str))
    test_ids  = set(test_df["flow_id"].astype(str))
    overlap = train_ids & test_ids
    print(f"  Flow ID overlap: {len(overlap):,}")
    if overlap:
        raise ValueError("Data leakage: flow IDs overlap train/test.")


def check_feature_vector_leakage(train_df, test_df):
    sentinel = "__NAN__"
    train_keys = set(
        train_df[config.ML_FEATURES]
        .fillna(sentinel).astype(str)
        .agg("|".join, axis=1)
    )
    test_keys = set(
        test_df[config.ML_FEATURES]
        .fillna(sentinel).astype(str)
        .agg("|".join, axis=1)
    )
    overlap = train_keys & test_keys
    print(f"  42-feature vector overlap: {len(overlap):,}")
    if overlap:
        raise ValueError(
            "Data leakage: identical 42-feature vectors across train/test."
        )


def validate_features(df, name):
    missing = [c for c in config.ML_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing ML features: {missing}")
    if len(config.ML_FEATURES) != 42:
        raise ValueError(
            f"Expected exactly 42 ML features, found {len(config.ML_FEATURES)}."
        )


def ttl_augment(df, fraction, random_state):
    rng = np.random.default_rng(random_state)
    n_samples = int(len(df) * fraction)
    if n_samples <= 0:
        return df.iloc[0:0].copy()

    indices = rng.choice(df.index.to_numpy(), size=n_samples, replace=False)
    augmented = df.loc[indices].copy()

    ttl_mask = augmented["ttl_mean"].notna()
    if ttl_mask.any():
        noise = rng.integers(
            TTL_JITTER_MIN, TTL_JITTER_MAX + 1,
            size=int(ttl_mask.sum()),
        )
        augmented.loc[ttl_mask, "ttl_mean"] = (
            augmented.loc[ttl_mask, "ttl_mean"] + noise
        ).clip(1, 255)

    return augmented


# ============================================================
# OOF hard-example mining
# ============================================================

def compute_oof_predictions(train_df, n_folds, n_estimators):
    """
    Compute out-of-fold predictions for every TRAIN row.

    Each row is predicted by a model trained on the other folds,
    so the resulting probability reflects generalisation difficulty,
    not training-set memorisation.

    Returns:
        oof_pred  : np.ndarray of int labels
        oof_proba : np.ndarray of attack probabilities
    """
    X = train_df[config.ML_FEATURES]
    y = train_df[config.LABEL_COLUMN].astype(int).to_numpy()

    n = len(train_df)
    oof_pred  = np.zeros(n, dtype=int)
    oof_proba = np.zeros(n, dtype=float)

    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_num = 0
    total_start = time.time()

    for train_idx, val_idx in skf.split(X, y):
        fold_num += 1
        t0 = time.time()

        X_tr = X.iloc[train_idx]
        y_tr = y[train_idx]
        X_va = X.iloc[val_idx]

        clf_fold = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=RANDOM_STATE + fold_num,
            n_jobs=N_JOBS,
        )
        clf_fold.fit(X_tr, y_tr)

        oof_pred[val_idx]  = clf_fold.predict(X_va)
        oof_proba[val_idx] = clf_fold.predict_proba(X_va)[:, 1]

        del clf_fold

        elapsed = time.time() - t0
        print(f"    fold {fold_num}/{n_folds} done in {elapsed/60:.1f} min")

    total = time.time() - total_start
    print(f"  OOF complete in {total/60:.1f} min")

    return oof_pred, oof_proba


def find_hard_examples(train_df, oof_pred, oof_proba):
    """
    Identify hard training rows using OOF predictions.

    A row is hard if EITHER:
      - OOF predicted label != true label (misclassified), OR
      - OOF attack probability is in [UNCERTAIN_LOW, UNCERTAIN_HIGH].
    """
    y_train = train_df[config.LABEL_COLUMN].astype(int).to_numpy()

    misclassified = oof_pred != y_train
    uncertain = (
        (oof_proba >= UNCERTAIN_LOW)
        & (oof_proba <= UNCERTAIN_HIGH)
    )
    hard_mask = misclassified | uncertain

    print(f"  Misclassified (OOF) : {int(misclassified.sum()):,} "
          f"({misclassified.mean()*100:.3f}%)")
    print(f"  Uncertain   (OOF)   : {int(uncertain.sum()):,} "
          f"({uncertain.mean()*100:.3f}%)")
    print(f"  Hard examples (union): {int(hard_mask.sum()):,} "
          f"({hard_mask.mean()*100:.3f}%)")

    return hard_mask, misclassified, uncertain


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Robust RF v2 (hybrid + OOF hard-example mining).")
    ap.add_argument("--oof-folds", type=int, default=OOF_FOLDS)
    ap.add_argument("--oof-estimators", type=int, default=OOF_ESTIMATORS)
    ap.add_argument("--skip-oof", action="store_true",
                    help="Skip OOF; use baseline train-set predictions instead (fast fallback).")
    args = ap.parse_args()

    print("=" * 70)
    print("NetworkIDS - Robust RF v2 (HYBRID)")
    print("=" * 70)
    print("\nStrategy:")
    print(f"  Part 1: TTL jitter on {AUGMENT_FRACTION*100:.0f}% of each class "
          f"({TTL_JITTER_MIN} to {TTL_JITTER_MAX})")
    if args.skip_oof:
        print("  Part 2: Hard-example oversampling via BASELINE train predictions")
    else:
        print(f"  Part 2: Hard-example oversampling via {args.oof_folds}-fold OOF "
              f"({args.oof_estimators} trees per fold)")

    # ---- load ----
    print("\nLoading datasets...")
    train_df = pd.read_csv(TRAIN_FILE, low_memory=False)
    test_df  = pd.read_csv(TEST_FILE,  low_memory=False)
    print(f"  Train: {len(train_df):,} rows")
    print(f"  Test : {len(test_df):,} rows (untouched)")

    validate_features(train_df, "Train")
    validate_features(test_df,  "Test")

    print("\nLeakage guardrail:")
    check_flow_id_leakage(train_df, test_df)
    check_feature_vector_leakage(train_df, test_df)
    print("  Leakage check: PASS")

    train_df[config.LABEL_COLUMN] = train_df[config.LABEL_COLUMN].astype(int)
    test_df[config.LABEL_COLUMN]  = test_df[config.LABEL_COLUMN].astype(int)

    benign_df = train_df[train_df[config.LABEL_COLUMN] == 0].copy()
    attack_df = train_df[train_df[config.LABEL_COLUMN] == 1].copy()

    print("\nOriginal train distribution:")
    print(f"  Benign: {len(benign_df):,} ({len(benign_df)/len(train_df)*100:.2f}%)")
    print(f"  Attack: {len(attack_df):,} ({len(attack_df)/len(train_df)*100:.2f}%)")

    # ---- Part 2: hard examples ----
    print("\n" + "=" * 70)
    if args.skip_oof:
        print("PART 2: Hard-example mining (BASELINE train predictions - fast mode)")
        print("=" * 70)
        baseline_model = joblib.load(config.RANDOM_FOREST_MODEL_FILE)
        X_train_full = train_df[config.ML_FEATURES]
        oof_pred  = baseline_model.predict(X_train_full)
        oof_proba = baseline_model.predict_proba(X_train_full)[:, 1]
        method = "baseline_train_predict"
    else:
        print(f"PART 2: Hard-example mining ({args.oof_folds}-fold OOF)")
        print("=" * 70)
        oof_pred, oof_proba = compute_oof_predictions(
            train_df,
            n_folds=args.oof_folds,
            n_estimators=args.oof_estimators,
        )
        method = f"{args.oof_folds}_fold_oof"

    hard_mask, misclassified_mask, uncertain_mask = find_hard_examples(
        train_df, oof_pred, oof_proba
    )

    hard_rows = train_df[hard_mask].copy()
    hard_label_counts = hard_rows[config.LABEL_COLUMN].value_counts().sort_index()
    print("\n  Hard examples by class:")
    for label, count in hard_label_counts.items():
        name = "Benign" if label == 0 else "Attack"
        print(f"    {name}: {count:,}")

    oversampled = pd.concat([hard_rows] * HARD_EXTRA_COPIES, ignore_index=True)
    print(f"\n  Extra copies added: {len(oversampled):,} "
          f"({HARD_EXTRA_COPIES}x per hard row)")

    # ---- Part 1: TTL augmentation ----
    print("\n" + "=" * 70)
    print("PART 1: TTL jitter augmentation")
    print("=" * 70)

    benign_ttl_aug = ttl_augment(benign_df, AUGMENT_FRACTION, RANDOM_STATE)
    attack_ttl_aug = ttl_augment(attack_df, AUGMENT_FRACTION, RANDOM_STATE + 1)
    print(f"  Benign TTL jittered: {len(benign_ttl_aug):,}")
    print(f"  Attack TTL jittered: {len(attack_ttl_aug):,}")

    # ---- Combine ----
    print("\n" + "=" * 70)
    print("BUILDING COMBINED TRAINING DATASET")
    print("=" * 70)

    augmented_train = pd.concat(
        [train_df, benign_ttl_aug, attack_ttl_aug, oversampled],
        ignore_index=True,
    ).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    final_counts = augmented_train[config.LABEL_COLUMN].value_counts().sort_index()
    final_benign = int(final_counts.get(0, 0))
    final_attack = int(final_counts.get(1, 0))

    print(f"  Original rows          : {len(train_df):,}")
    print(f"  + TTL jitter (Part 1)  : {len(benign_ttl_aug) + len(attack_ttl_aug):,}")
    print(f"  + Hard oversamp (Part 2): {len(oversampled):,}")
    print(f"  = Total rows           : {len(augmented_train):,} "
          f"({len(augmented_train)/len(train_df):.2f}x)")
    print()
    print(f"  Final Benign: {final_benign:,} ({final_benign/len(augmented_train)*100:.2f}%)")
    print(f"  Final Attack: {final_attack:,} ({final_attack/len(augmented_train)*100:.2f}%)")

    DATASET_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    augmented_train.to_csv(DATASET_OUTPUT, index=False)
    print(f"\n  Saved: {DATASET_OUTPUT}")

    # ---- Prepare X/y ----
    X_train = augmented_train[config.ML_FEATURES]
    y_train = augmented_train[config.LABEL_COLUMN].astype(int)
    X_test  = test_df[config.ML_FEATURES]
    y_test  = test_df[config.LABEL_COLUMN].astype(int)

    print(f"\n  NaN cells X_train: {int(X_train.isna().sum().sum()):,} (retained)")
    print(f"  NaN cells X_test : {int(X_test.isna().sum().sum()):,} (retained)")
    print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")

    # ---- Train ----
    print("\n" + "=" * 70)
    print("TRAINING FINAL ROBUST RF V2")
    print("=" * 70)
    print(f"  n_estimators={N_ESTIMATORS}  class_weight=balanced  seed={RANDOM_STATE}")

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    clf.fit(X_train, y_train)
    print("  Done.")

    # ---- Evaluate ----
    print("\n" + "=" * 70)
    print("CLEAN EVALUATION (untouched test set)")
    print("=" * 70)

    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    auc       = roc_auc_score(y_test, y_proba)
    cm        = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print()
    print(f"  TN: {tn:,}   FP: {fp:,}   (baseline FP: 1,016)")
    print(f"  FN: {fn:,}   TP: {tp:,}   (baseline FN:   150)")
    print(f"  FPR: {fpr:.4%}   FNR: {fnr:.4%}")
    print()
    print(classification_report(
        y_test, y_pred,
        target_names=["Benign", "Attack"],
        digits=4,
        zero_division=0,
    ))

    importances = (
        pd.Series(clf.feature_importances_, index=config.ML_FEATURES)
        .sort_values(ascending=False)
    )

    print("Top 15 feature importances (Gini):")
    for name, value in importances.head(15).items():
        print(f"  {name:<30} {value:.4f}")

    # ---- Save model ----
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_OUTPUT)
    print(f"\nModel saved: {MODEL_OUTPUT}")

    # ---- Save metrics ----
    metrics = {
        "model_type": "RandomForestClassifier (v2 hybrid: TTL jitter + OOF hard-example oversampling)",
        "random_state": RANDOM_STATE,
        "n_estimators": N_ESTIMATORS,
        "n_features": len(config.ML_FEATURES),

        "strategy": {
            "part1_ttl_jitter": {
                "fraction_per_class": AUGMENT_FRACTION,
                "ttl_jitter_range": [TTL_JITTER_MIN, TTL_JITTER_MAX],
                "benign_augmented": int(len(benign_ttl_aug)),
                "attack_augmented": int(len(attack_ttl_aug)),
            },
            "part2_hard_example_mining": {
                "method": method,
                "oof_folds": args.oof_folds if not args.skip_oof else None,
                "oof_estimators": args.oof_estimators if not args.skip_oof else None,
                "uncertain_probability_range": [UNCERTAIN_LOW, UNCERTAIN_HIGH],
                "extra_copies_per_hard_row": HARD_EXTRA_COPIES,
                "n_misclassified_oof": int(misclassified_mask.sum()),
                "n_uncertain_oof": int(uncertain_mask.sum()),
                "n_hard_examples_union": int(hard_mask.sum()),
                "hard_examples_by_class": {
                    "benign": int(hard_label_counts.get(0, 0)),
                    "attack": int(hard_label_counts.get(1, 0)),
                },
                "oversampled_rows_added": int(len(oversampled)),
            },
        },

        "dataset": {
            "n_train_original": int(len(train_df)),
            "n_train_augmented": int(len(augmented_train)),
            "augmentation_multiplier": float(len(augmented_train) / len(train_df)),
            "n_test": int(len(test_df)),
            "final_benign": final_benign,
            "final_attack": final_attack,
        },

        "test_metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc_roc": float(auc),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
            "false_positive_rate": float(fpr),
            "false_negative_rate": float(fnr),
            "confusion_matrix": cm.tolist(),
        },

        "top_15_importances": {
            k: float(v) for k, v in importances.head(15).items()
        },
    }

    with open(METRICS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {METRICS_OUTPUT}")

    print("\n" + "=" * 70)
    print("V2 (HYBRID with OOF) TRAINING COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()