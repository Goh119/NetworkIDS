from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd

import config


# ============================================================
# Configuration
# ============================================================

TEST_FILE = config.PROCESSED_DIR / "split" / "test_dataset.csv"
OUTPUT_DIR = config.MODELS_DIR / "robustness"

RANDOM_STATE = 42
SAMPLE_SIZE_PER_CLASS = 2000


# ============================================================
# Utility functions
# ============================================================

def load_dataset():
    """Load the fixed test dataset used for robustness evaluation."""

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test dataset not found:\n{TEST_FILE}"
        )

    df = pd.read_csv(TEST_FILE)

    required_columns = (
        config.ML_FEATURES
        + config.METADATA_COLUMNS
        + [config.LABEL_COLUMN]
    )

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing_columns)
        )

    return df


def validate_feature_count(df):
    """Ensure exactly the configured ML feature set is used."""

    if len(config.ML_FEATURES) != 42:
        raise ValueError(
            f"Expected exactly 42 ML features, "
            f"but config contains {len(config.ML_FEATURES)}."
        )

    missing = [
        feature for feature in config.ML_FEATURES
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing ML features:\n"
            + "\n".join(missing)
        )


def load_model(model_path):
    """Load the requested Random Forest model."""

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{model_path}"
        )

    print(f"Loading model: {model_path}")

    model = joblib.load(model_path)

    return model


def predict(model, X):
    """
    Return model predictions and attack probabilities.

    Column order is preserved according to config.ML_FEATURES.
    """

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)

    # Find probability column corresponding to Attack = 1.
    attack_class_index = list(model.classes_).index(
        config.LABEL_ATTACK
    )

    attack_probabilities = probabilities[:, attack_class_index]

    return predictions, attack_probabilities


# ============================================================
# Perturbation functions
# ============================================================

def ttl_jitter_small(X, rng):
    """
    Add random TTL jitter in the range [-5, +5].

    NaN values remain NaN.
    """

    X_new = X.copy()

    if "ttl_mean" not in X_new.columns:
        return X_new

    mask = X_new["ttl_mean"].notna()

    noise = rng.integers(
        low=-5,
        high=6,
        size=mask.sum()
    )

    X_new.loc[mask, "ttl_mean"] = np.clip(
        X_new.loc[mask, "ttl_mean"].to_numpy()
        + noise,
        1,
        255
    )

    return X_new


def ttl_jitter_large(X, rng):
    """
    Add random TTL jitter in the range [-30, +30].

    NaN values remain NaN.
    """

    X_new = X.copy()

    if "ttl_mean" not in X_new.columns:
        return X_new

    mask = X_new["ttl_mean"].notna()

    noise = rng.integers(
        low=-30,
        high=31,
        size=mask.sum()
    )

    X_new.loc[mask, "ttl_mean"] = np.clip(
        X_new.loc[mask, "ttl_mean"].to_numpy()
        + noise,
        1,
        255
    )

    return X_new


def ttl_forced_normal(X):
    """
    Force observed TTL values to 128.

    NaN values remain NaN.
    """

    X_new = X.copy()

    if "ttl_mean" not in X_new.columns:
        return X_new

    mask = X_new["ttl_mean"].notna()

    X_new.loc[mask, "ttl_mean"] = 128

    return X_new


def checksum_forced_valid(X):
    """
    Force observed IP and L4 checksum validity indicators to valid.

    NaN values remain NaN.
    """

    X_new = X.copy()

    for column in [
        "ip_checksum_valid",
        "l4_checksum_valid",
    ]:
        if column not in X_new.columns:
            continue

        mask = X_new[column].notna()
        X_new.loc[mask, column] = 1

    return X_new


def checksum_forced_invalid(X):
    """
    Force observed IP and L4 checksum validity indicators to invalid.

    NaN values remain NaN.
    """

    X_new = X.copy()

    for column in [
        "ip_checksum_valid",
        "l4_checksum_valid",
    ]:
        if column not in X_new.columns:
            continue

        mask = X_new[column].notna()
        X_new.loc[mask, column] = 0

    return X_new


def rst_count_zeroed(X):
    """
    Set RST packet count to zero.
    """

    X_new = X.copy()

    if "rst_count" in X_new.columns:
        X_new["rst_count"] = 0

    return X_new


def syn_count_normalized(X):
    """
    Normalize SYN-related features:

    syn_count = 1
    syn_seen = 1

    Other features remain unchanged.
    """

    X_new = X.copy()

    if "syn_count" in X_new.columns:
        X_new["syn_count"] = 1

    if "syn_seen" in X_new.columns:
        X_new["syn_seen"] = 1

    return X_new


def combined_evasion(X):
    """
    Combined controlled perturbation:

    - TTL forced to 128
    - checksum validity forced to valid
    - RST count zeroed
    """

    X_new = X.copy()

    # TTL
    if "ttl_mean" in X_new.columns:
        mask = X_new["ttl_mean"].notna()
        X_new.loc[mask, "ttl_mean"] = 128

    # Checksums
    for column in [
        "ip_checksum_valid",
        "l4_checksum_valid",
    ]:
        if column in X_new.columns:
            mask = X_new[column].notna()
            X_new.loc[mask, column] = 1

    # RST
    if "rst_count" in X_new.columns:
        X_new["rst_count"] = 0

    return X_new


# ============================================================
# Scenario definitions
# ============================================================

def build_scenarios():
    """
    Return the fixed set of robustness scenarios.

    The same scenarios are used for baseline and robust models.
    """

    return [
        (
            "ttl_jitter_small",
            "TTL +/- 5 random jitter",
            lambda X, rng: ttl_jitter_small(X, rng),
        ),
        (
            "ttl_jitter_large",
            "TTL +/- 30 random jitter",
            lambda X, rng: ttl_jitter_large(X, rng),
        ),
        (
            "ttl_forced_normal",
            "TTL forced to 128 (common default TTL)",
            lambda X, rng: ttl_forced_normal(X),
        ),
        (
            "checksum_forced_valid",
            "IP/L4 checksum validity flags forced valid",
            lambda X, rng: checksum_forced_valid(X),
        ),
        (
            "checksum_forced_invalid",
            "IP/L4 checksum validity flags forced invalid",
            lambda X, rng: checksum_forced_invalid(X),
        ),
        (
            "rst_count_zeroed",
            "RST packet count zeroed",
            lambda X, rng: rst_count_zeroed(X),
        ),
        (
            "syn_count_normalized",
            "SYN count forced to 1 and SYN presence enabled",
            lambda X, rng: syn_count_normalized(X),
        ),
        (
            "combined_evasion",
            "TTL + checksum + RST evasion combined",
            lambda X, rng: combined_evasion(X),
        ),
    ]


# ============================================================
# Sampling
# ============================================================

def select_correctly_classified_samples(
    df,
    model,
    sample_size,
    random_state,
):
    """
    Select correctly classified attack and benign samples.

    Attack:
        Ground truth = Attack
        Baseline prediction = Attack

    Benign:
        Ground truth = Benign
        Baseline prediction = Benign

    The same selected samples are used for every scenario.
    """

    X = df[config.ML_FEATURES]
    y = df[config.LABEL_COLUMN].astype(int)

    baseline_predictions, baseline_attack_probability = predict(
        model,
        X,
    )

    correct_attack_mask = (
        (y == config.LABEL_ATTACK)
        & (baseline_predictions == config.LABEL_ATTACK)
    )

    correct_benign_mask = (
        (y == config.LABEL_BENIGN)
        & (baseline_predictions == config.LABEL_BENIGN)
    )

    attack_indices = df.index[correct_attack_mask].to_numpy()
    benign_indices = df.index[correct_benign_mask].to_numpy()

    if len(attack_indices) < sample_size:
        raise ValueError(
            f"Not enough correctly classified Attack samples. "
            f"Required={sample_size}, available={len(attack_indices)}"
        )

    if len(benign_indices) < sample_size:
        raise ValueError(
            f"Not enough correctly classified Benign samples. "
            f"Required={sample_size}, available={len(benign_indices)}"
        )

    rng = np.random.default_rng(random_state)

    selected_attack_indices = rng.choice(
        attack_indices,
        size=sample_size,
        replace=False,
    )

    selected_benign_indices = rng.choice(
        benign_indices,
        size=sample_size,
        replace=False,
    )

    attack_X = df.loc[
        selected_attack_indices,
        config.ML_FEATURES
    ].copy()

    benign_X = df.loc[
        selected_benign_indices,
        config.ML_FEATURES
    ].copy()

    attack_baseline_prob = baseline_attack_probability[
        df.index.get_indexer(selected_attack_indices)
    ]

    benign_baseline_prob = baseline_attack_probability[
        df.index.get_indexer(selected_benign_indices)
    ]

    return (
        attack_X,
        benign_X,
        attack_baseline_prob,
        benign_baseline_prob,
    )


# ============================================================
# Scenario evaluation
# ============================================================

def evaluate_scenario(
    model,
    attack_X,
    benign_X,
    attack_baseline_prob,
    benign_baseline_prob,
    perturbation,
    scenario_name,
    description,
    rng,
):
    """
    Apply one perturbation scenario and calculate:

    - Attack Evasion Rate
    - Benign Induction Rate
    - Mean Probability Shift
    - Mean Probability Drop for attacks
    - Mean Probability Rise for benign samples
    """

    attack_X_perturbed = perturbation(
        attack_X.copy(),
        rng,
    )

    benign_X_perturbed = perturbation(
        benign_X.copy(),
        rng,
    )

    attack_predictions, attack_perturbed_prob = predict(
        model,
        attack_X_perturbed,
    )

    benign_predictions, benign_perturbed_prob = predict(
        model,
        benign_X_perturbed,
    )

    # --------------------------------------------------------
    # Attack Evasion Rate
    # --------------------------------------------------------
    #
    # Ground truth remains Attack.
    #
    # If the perturbed model prediction becomes Benign,
    # this is an evasion / false negative.
    #

    attack_evasion_rate = np.mean(
        attack_predictions == config.LABEL_BENIGN
    )

    # --------------------------------------------------------
    # Benign Induction Rate
    # --------------------------------------------------------
    #
    # Ground truth remains Benign.
    #
    # If the perturbed model prediction becomes Attack,
    # this is benign induction / false positive.
    #

    benign_induction_rate = np.mean(
        benign_predictions == config.LABEL_ATTACK
    )

    # --------------------------------------------------------
    # Absolute probability shifts
    # --------------------------------------------------------

    attack_probability_shift = np.abs(
        attack_perturbed_prob
        - attack_baseline_prob
    )

    benign_probability_shift = np.abs(
        benign_perturbed_prob
        - benign_baseline_prob
    )

    mean_prob_shift_attack = np.mean(
        attack_probability_shift
    )

    mean_prob_shift_benign = np.mean(
        benign_probability_shift
    )

    # --------------------------------------------------------
    # Directional probability changes
    # --------------------------------------------------------
    #
    # Attack:
    # baseline - perturbed
    #
    # Positive value means attack probability decreased.
    #

    attack_probability_drop = (
        attack_baseline_prob
        - attack_perturbed_prob
    )

    mean_prob_drop_attack = np.mean(
        attack_probability_drop
    )

    # --------------------------------------------------------
    # Benign:
    # perturbed - baseline
    #
    # Positive value means attack probability increased.
    #

    benign_probability_rise = (
        benign_perturbed_prob
        - benign_baseline_prob
    )

    mean_prob_rise_benign = np.mean(
        benign_probability_rise
    )

    return {
        "scenario": scenario_name,
        "description": description,
        "attack_evasion_rate": float(
            attack_evasion_rate
        ),
        "benign_induction_rate": float(
            benign_induction_rate
        ),
        "mean_prob_shift_attack": float(
            mean_prob_shift_attack
        ),
        "mean_prob_shift_benign": float(
            mean_prob_shift_benign
        ),
        "mean_prob_drop_attack": float(
            mean_prob_drop_attack
        ),
        "mean_prob_rise_benign": float(
            mean_prob_rise_benign
        ),
        "n_attack_tested": int(len(attack_X)),
        "n_benign_tested": int(len(benign_X)),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Random Forest robustness using "
            "controlled flow-level feature perturbations."
        )
    )

    # --------------------------------------------------------
    # Model argument
    # --------------------------------------------------------

    parser.add_argument(
        "--model",
        type=str,
        default=str(config.RANDOM_FOREST_MODEL_FILE),
        help=(
            "Path to the model to evaluate "
            "(default: baseline model)."
        ),
    )

    # --------------------------------------------------------
    # Output filename argument
    # --------------------------------------------------------

    parser.add_argument(
        "--out-name",
        type=str,
        default="robustness_results.csv",
        help="Output CSV filename.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Prepare output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("=" * 70)
    print("ROBUSTNESS EVALUATION")
    print("=" * 70)

    print(f"Test dataset: {TEST_FILE}")

    df = load_dataset()

    print(
        f"Loaded test dataset: "
        f"{len(df):,} rows x {len(df.columns)} columns"
    )

    validate_feature_count(df)

    # --------------------------------------------------------
    # Validate labels
    # --------------------------------------------------------

    if df[config.LABEL_COLUMN].isna().any():
        raise ValueError(
            "Test dataset contains NaN labels."
        )

    unique_labels = set(
        df[config.LABEL_COLUMN].astype(int).unique()
    )

    expected_labels = {
        config.LABEL_BENIGN,
        config.LABEL_ATTACK,
    }

    if unique_labels != expected_labels:
        raise ValueError(
            f"Unexpected labels: {unique_labels}. "
            f"Expected: {expected_labels}"
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model(args.model)

    print(
        f"Model classes: {list(model.classes_)}"
    )

    # --------------------------------------------------------
    # Model feature check
    # --------------------------------------------------------

    if hasattr(model, "n_features_in_"):
        if model.n_features_in_ != len(config.ML_FEATURES):
            raise ValueError(
                f"Model expects {model.n_features_in_} features, "
                f"but config contains {len(config.ML_FEATURES)}."
            )

    # --------------------------------------------------------
    # Prepare X / y
    # --------------------------------------------------------

    X = df[config.ML_FEATURES]
    y = df[config.LABEL_COLUMN].astype(int)

    print(
        f"ML feature matrix: {X.shape}"
    )

    print(
        f"NaN cells in test features: "
        f"{int(X.isna().sum().sum()):,}"
    )

    # --------------------------------------------------------
    # Baseline model predictions
    # --------------------------------------------------------

    baseline_predictions, baseline_probabilities = predict(
        model,
        X,
    )

    baseline_accuracy = np.mean(
        baseline_predictions == y.to_numpy()
    )

    print(
        f"Model accuracy on clean test set: "
        f"{baseline_accuracy:.4f}"
    )

    # --------------------------------------------------------
    # Select correctly classified samples
    # --------------------------------------------------------

    (
        attack_X,
        benign_X,
        attack_baseline_prob,
        benign_baseline_prob,
    ) = select_correctly_classified_samples(
        df=df,
        model=model,
        sample_size=SAMPLE_SIZE_PER_CLASS,
        random_state=RANDOM_STATE,
    )

    print()
    print("Correctly classified sample pool:")
    print(
        f"  Attack samples: "
        f"{len(attack_X):,}"
    )
    print(
        f"  Benign samples: "
        f"{len(benign_X):,}"
    )

    print()
    print(
        "Robustness sample configuration:"
    )
    print(
        f"  Samples per class: "
        f"{SAMPLE_SIZE_PER_CLASS:,}"
    )
    print(
        f"  Random state: "
        f"{RANDOM_STATE}"
    )

    # --------------------------------------------------------
    # Run scenarios
    # --------------------------------------------------------

    scenarios = build_scenarios()

    results = []

    for (
        scenario_name,
        description,
        perturbation,
    ) in scenarios:

        print()
        print("-" * 70)
        print(f"Scenario: {scenario_name}")
        print(f"Description: {description}")

        # Use the same deterministic random seed pattern
        # for each scenario.
        scenario_rng = np.random.default_rng(
            RANDOM_STATE
        )

        result = evaluate_scenario(
            model=model,
            attack_X=attack_X,
            benign_X=benign_X,
            attack_baseline_prob=attack_baseline_prob,
            benign_baseline_prob=benign_baseline_prob,
            perturbation=perturbation,
            scenario_name=scenario_name,
            description=description,
            rng=scenario_rng,
        )

        results.append(result)

        print(
            f"Attack Evasion Rate: "
            f"{result['attack_evasion_rate']:.4%}"
        )

        print(
            f"Benign Induction Rate: "
            f"{result['benign_induction_rate']:.4%}"
        )

        print(
            f"Mean Attack Probability Shift: "
            f"{result['mean_prob_shift_attack']:.4f}"
        )

        print(
            f"Mean Benign Probability Shift: "
            f"{result['mean_prob_shift_benign']:.4f}"
        )

        print(
            f"Mean Attack Probability Drop: "
            f"{result['mean_prob_drop_attack']:.4f}"
        )

        print(
            f"Mean Benign Probability Rise: "
            f"{result['mean_prob_rise_benign']:.4f}"
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    out_path = OUTPUT_DIR / args.out_name

    results_df.to_csv(
        out_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save experiment metadata
    # --------------------------------------------------------

    metadata = {
        "model_path": str(Path(args.model)),
        "test_dataset": str(TEST_FILE),
        "output_file": str(out_path),
        "random_state": RANDOM_STATE,
        "sample_size_per_class": SAMPLE_SIZE_PER_CLASS,
        "n_ml_features": len(config.ML_FEATURES),
        "n_scenarios": len(scenarios),
        "attack_ground_truth_preserved": True,
        "benign_ground_truth_preserved": True,
        "baseline_model_reused_without_retraining": True,
    }

    metadata_path = out_path.with_name(
        out_path.stem + "_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ROBUSTNESS EVALUATION SUMMARY")
    print("=" * 70)

    for _, row in results_df.iterrows():

        print(
            f"{row['scenario']:<25} "
            f"Attack Evasion: "
            f"{row['attack_evasion_rate']:.2%} | "
            f"Benign Induction: "
            f"{row['benign_induction_rate']:.2%}"
        )

    print()
    print(f"Results saved to:")
    print(out_path)

    print()
    print(f"Metadata saved to:")
    print(metadata_path)

    print()
    print("PASS: Robustness evaluation completed.")

if __name__ == "__main__":
    main()