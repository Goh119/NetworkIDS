"""
robustness_evaluation.py
------------------------
NetworkIDS - Primary Random Forest Robustness Evaluation.

This script evaluates the robustness of the trained primary
Random Forest model under controlled feature perturbations.

Primary inputs:
    config.TEST_DATASET_FILE
    config.RANDOM_FOREST_MODEL_FILE

Primary output directory:
    <model directory>/robustness/

Default outputs:
    robustness_results.csv
    robustness_results_metadata.json

Important:
    The Random Forest uses exactly config.ML_FEATURES.

    The number of ML features is dynamic and is NOT hard-coded.

    NaN values are retained.

    No imputation is performed.

This evaluation uses correctly classified test samples as the
baseline population:

    Attack baseline population:
        Ground truth = Attack
        Clean prediction = Attack

    Benign baseline population:
        Ground truth = Benign
        Clean prediction = Benign

The same selected samples are reused across all scenarios.

Metrics:

    Attack Evasion Rate
        Percentage of correctly detected Attack flows that become
        Benign after perturbation.

    Benign Induction Rate
        Percentage of correctly classified Benign flows that become
        Attack after perturbation.

    Mean Attack Probability Drop
        Average decrease in P(Attack) for Attack samples.

    Mean Benign Probability Rise
        Average increase in P(Attack) for Benign samples.

Important interpretation:

    This script performs controlled feature perturbation.

    It does NOT directly modify the original PCAP packets.

    Therefore, the results should be described as feature-level
    robustness / perturbation evaluation unless a separate
    packet-level perturbation pipeline is implemented.
"""

from pathlib import Path
import argparse
import json
import sys

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
    config.TEST_DIR / "test_multiday.csv"
)

MODEL_FILE = (
    config.RANDOM_FOREST_MODEL_FILE
)

# Keep robustness results beside the Random Forest model.
#
# Current expected structure:
#
# random_forest/
#     rf_model.joblib
#     rf_model_meta.json
#     rf_model_report.txt
#     robustness/
#
ROBUSTNESS_DIR = (
    Path(MODEL_FILE).parent.parent
    / "robustness"
)

RANDOM_STATE = 42

DEFAULT_SAMPLE_SIZE_PER_CLASS = 2000

FEATURE_DTYPE = np.float32

LABEL_DTYPE = np.int8


# ============================================================
# Utility functions
# ============================================================

def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def label_to_text(label):
    """
    Convert configured label value into readable text.

    Numeric label mapping is taken from config.py.
    """

    label = int(label)

    if label == config.LABEL_BENIGN:
        return "Benign"

    if label == config.LABEL_ATTACK:
        return "Attack"

    return str(label)


def validate_config():
    """
    Validate the current ML feature configuration.

    No fixed feature count is assumed.
    """

    config.validate_ml_features()

    if not config.ML_FEATURES:
        raise ValueError(
            "config.ML_FEATURES is empty."
        )

    if len(
        set(config.ML_FEATURES)
    ) != len(config.ML_FEATURES):

        raise ValueError(
            "Duplicate feature names detected in "
            "config.ML_FEATURES."
        )

    print(
        f"  ML feature count: "
        f"{len(config.ML_FEATURES)}"
    )


def load_dataset():
    """
    Load the fixed held-out test dataset.

    Only:
        - ML features
        - label

    are required.

    Traceability metadata is not required for robustness
    calculations.
    """

    path = Path(TEST_FILE)

    if not path.exists():
        raise FileNotFoundError(
            "Test dataset not found:\n"
            f"{path}"
        )

    required_columns = [
        *config.ML_FEATURES,
        config.LABEL_COLUMN,
    ]

    required_columns = list(
        dict.fromkeys(
            required_columns
        )
    )

    # --------------------------------------------------------
    # Validate CSV header
    # --------------------------------------------------------

    header = pd.read_csv(
        path,
        nrows=0,
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in header.columns
    ]

    if missing_columns:
        raise ValueError(
            "Test dataset is missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_columns
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
        f"  Test rows: "
        f"{len(df):,}"
    )

    return df


def validate_dataset(df):
    """
    Validate labels and ML feature values.
    """

    # --------------------------------------------------------
    # Label validation
    # --------------------------------------------------------

    y = df[
        config.LABEL_COLUMN
    ]

    if y.isna().any():

        raise ValueError(
            "Test dataset contains NaN labels."
        )

    expected_labels = {
        config.LABEL_BENIGN,
        config.LABEL_ATTACK,
    }

    actual_labels = set(
        y.unique()
        .tolist()
    )

    if not actual_labels.issubset(
        expected_labels
    ):

        unexpected = (
            actual_labels
            - expected_labels
        )

        raise ValueError(
            "Unexpected labels found:\n"
            f"{sorted(unexpected)}"
        )

    if len(actual_labels) != 2:

        raise ValueError(
            "Both configured classes must be present."
        )

    # --------------------------------------------------------
    # Feature validation
    # --------------------------------------------------------

    X = df[
        config.ML_FEATURES
    ]

    nan_count = int(
        X.isna()
        .sum()
        .sum()
    )

    infinite_count = 0

    for feature in config.ML_FEATURES:

        values = (
            X[feature]
            .to_numpy(
                copy=False
            )
        )

        infinite_count += int(
            np.isinf(values)
            .sum()
        )

    if infinite_count:

        raise ValueError(
            f"Found {infinite_count:,} "
            "infinite ML feature values."
        )

    total_cells = (
        X.shape[0]
        * X.shape[1]
    )

    nan_percentage = (
        nan_count
        / total_cells
        * 100.0
        if total_cells
        else 0.0
    )

    print(
        f"  ML features: "
        f"{len(config.ML_FEATURES)}"
    )

    print(
        f"  NaN cells: "
        f"{nan_count:,} "
        f"({nan_percentage:.4f}%)"
    )

    print(
        "  NaN values are retained."
    )

    print(
        "  No feature imputation is applied."
    )


def load_model(model_path):
    """
    Load the saved bare Random Forest model.
    """

    model_path = Path(
        model_path
    )

    if not model_path.exists():

        raise FileNotFoundError(
            "Model file not found:\n"
            f"{model_path}"
        )

    print(
        f"  Model: "
        f"{model_path}"
    )

    model = joblib.load(
        model_path
    )

    return model


def validate_model_compatibility(model):
    """
    Ensure the model matches the current feature configuration.
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

        model_feature_count = int(
            model.n_features_in_
        )

        if (
            model_feature_count
            != configured_count
        ):

            raise RuntimeError(
                "Model/config feature-count mismatch.\n"
                f"Model expects : "
                f"{model_feature_count}\n"
                f"Config defines: "
                f"{configured_count}"
            )

    # --------------------------------------------------------
    # Feature order
    # --------------------------------------------------------

    if hasattr(
        model,
        "feature_names_in_",
    ):

        model_features = list(
            model.feature_names_in_
        )

        if (
            model_features
            != configured_features
        ):

            raise RuntimeError(
                "Model/config feature-order mismatch.\n"
                "\nThe current config.ML_FEATURES does "
                "not match the exact feature order used "
                "during training."
            )

    print(
        f"  Model/config features: "
        f"OK ({configured_count})"
    )


def get_attack_class_index(model):
    """
    Locate the configured Attack class inside model.classes_.
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

    if (
        config.LABEL_ATTACK
        not in classes
    ):

        raise RuntimeError(
            f"Configured Attack label "
            f"{config.LABEL_ATTACK} was not found "
            f"in model.classes_.\n"
            f"Model classes: {classes}"
        )

    return classes.index(
        config.LABEL_ATTACK
    )


def predict(
    model,
    X,
    attack_class_index,
):
    """
    Generate predictions and P(Attack).

    The Attack probability column is selected using
    model.classes_ rather than assuming column index 1.
    """

    predictions = model.predict(
        X
    )

    probabilities = model.predict_proba(
        X
    )

    attack_probabilities = (
        probabilities[
            :,
            attack_class_index
        ]
    )

    return (
        np.asarray(
            predictions
        ),
        np.asarray(
            attack_probabilities
        ),
    )


# ============================================================
# Perturbation helpers
# ============================================================

def apply_numeric_change(
    X,
    column,
    transform,
):
    """
    Apply a transformation only to non-NaN values.

    NaN positions remain NaN.
    """

    if column not in X.columns:
        return X

    mask = X[column].notna()

    if not mask.any():
        return X

    original = (
        X.loc[
            mask,
            column
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    transformed = transform(
        original
    )

    X.loc[
        mask,
        column
    ] = transformed

    return X


# ============================================================
# TTL perturbations
# ============================================================

def ttl_jitter_small(
    X,
    rng,
):
    """
    Add random TTL jitter in [-5, +5].

    NaN values remain NaN.

    The perturbed values are explicitly converted back
    to float32 so they remain compatible with the model
    feature dtype.
    """

    X_new = X.copy()

    if "ttl_mean" not in X_new.columns:
        return X_new

    mask = (
        X_new["ttl_mean"]
        .notna()
    )

    count = int(
        mask.sum()
    )

    if count == 0:
        return X_new

    noise = rng.integers(
        low=-5,
        high=6,
        size=count,
    )

    original = (
        X_new.loc[
            mask,
            "ttl_mean"
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    perturbed = np.clip(
        original + noise,
        1,
        255,
    ).astype(
        np.float32
    )

    X_new.loc[
        mask,
        "ttl_mean"
    ] = perturbed

    return X_new


def ttl_jitter_large(
    X,
    rng,
):
    """
    Add random TTL jitter in [-30, +30].

    NaN values remain NaN.

    The perturbed values are explicitly converted back
    to float32.
    """

    X_new = X.copy()

    if "ttl_mean" not in X_new.columns:
        return X_new

    mask = (
        X_new["ttl_mean"]
        .notna()
    )

    count = int(
        mask.sum()
    )

    if count == 0:
        return X_new

    noise = rng.integers(
        low=-30,
        high=31,
        size=count,
    )

    original = (
        X_new.loc[
            mask,
            "ttl_mean"
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    perturbed = np.clip(
        original + noise,
        1,
        255,
    ).astype(
        np.float32
    )

    X_new.loc[
        mask,
        "ttl_mean"
    ] = perturbed

    return X_new


def ttl_forced_normal(X):
    """
    Force non-NaN TTL values to 128.

    NaN values remain NaN.
    """

    X_new = X.copy()

    if "ttl_mean" not in X_new.columns:
        return X_new

    mask = (
        X_new["ttl_mean"]
        .notna()
    )

    X_new.loc[
        mask,
        "ttl_mean"
    ] = np.float32(128.0)

    return X_new


# ============================================================
# Checksum perturbations
# ============================================================

def checksum_forced_valid(
    X,
):
    """
    Force observed checksum-validity indicators to valid.

    NaN values remain NaN.
    """

    X_new = X.copy()

    for column in [
        "ip_checksum_valid",
        "l4_checksum_valid",
    ]:

        if column not in X_new.columns:
            continue

        mask = (
            X_new[column]
            .notna()
        )

        X_new.loc[
            mask,
            column
        ] = 1

    return X_new


def checksum_forced_invalid(
    X,
):
    """
    Force observed checksum-validity indicators to invalid.

    NaN values remain NaN.
    """

    X_new = X.copy()

    for column in [
        "ip_checksum_valid",
        "l4_checksum_valid",
    ]:

        if column not in X_new.columns:
            continue

        mask = (
            X_new[column]
            .notna()
        )

        X_new.loc[
            mask,
            column
        ] = 0

    return X_new


# ============================================================
# TCP / flow perturbations
# ============================================================

def rst_count_zeroed(
    X,
):
    """
    Set RST packet count to zero.

    This is a flow-feature perturbation proxy for
    removing observed reset behaviour.
    """

    X_new = X.copy()

    if "rst_count" in X_new.columns:
        mask = X_new["rst_count"].notna()
        X_new.loc[mask, "rst_count"] = np.float32(0.0)

    return X_new


def syn_count_normalized(
    X,
):
    """
    Normalize SYN-related flow features:

        syn_count = 1
        syn_seen = 1

    Other features remain unchanged.
    """

    X_new = X.copy()

    if "syn_count" in X_new.columns:
        mask = X_new["syn_count"].notna()
        X_new.loc[mask, "syn_count"] = np.float32(1.0)

    if "syn_seen" in X_new.columns:
        mask = X_new["syn_seen"].notna()
        X_new.loc[mask, "syn_seen"] = np.float32(1.0)

    return X_new


# ============================================================
# Combined perturbation
# ============================================================

def combined_evasion(
    X,
):
    """
    Combined controlled feature perturbation:

        TTL -> 128
        checksum validity -> valid
        RST count -> 0

    This is intentionally a feature-level stress test.
    """

    X_new = X.copy()

    # --------------------------------------------------------
    # TTL
    # --------------------------------------------------------

    if "ttl_mean" in X_new.columns:

        mask = (
            X_new["ttl_mean"]
            .notna()
        )

        X_new.loc[
            mask,
            "ttl_mean"
        ] = np.float32(128.0)

    # --------------------------------------------------------
    # Checksums
    # --------------------------------------------------------

    for column in [
        "ip_checksum_valid",
        "l4_checksum_valid",
    ]:

        if column not in X_new.columns:
            continue

        mask = (
            X_new[column]
            .notna()
        )

        X_new.loc[
            mask,
            column
        ] = 1

    # --------------------------------------------------------
    # RST
    # --------------------------------------------------------

    if "rst_count" in X_new.columns:

        mask = X_new["rst_count"].notna()
        X_new.loc[mask, "rst_count"] = np.float32(0.0)

    return X_new


# ============================================================
# Scenario definitions
# ============================================================

def build_scenarios():
    """
    Return the fixed robustness scenario set.

    These scenarios are applied identically to the selected
    Attack and Benign baseline populations.
    """

    return [

        (
            "ttl_jitter_small",
            "TTL +/- 5 random jitter",
            lambda X, rng:
                ttl_jitter_small(
                    X,
                    rng,
                ),
        ),

        (
            "ttl_jitter_large",
            "TTL +/- 30 random jitter",
            lambda X, rng:
                ttl_jitter_large(
                    X,
                    rng,
                ),
        ),

        (
            "ttl_forced_normal",
            "TTL forced to 128",
            lambda X, rng:
                ttl_forced_normal(
                    X,
                ),
        ),

        (
            "checksum_forced_valid",
            "IP/L4 checksum validity forced valid",
            lambda X, rng:
                checksum_forced_valid(
                    X,
                ),
        ),

        (
            "checksum_forced_invalid",
            "IP/L4 checksum validity forced invalid",
            lambda X, rng:
                checksum_forced_invalid(
                    X,
                ),
        ),

        (
            "rst_count_zeroed",
            "RST packet count zeroed",
            lambda X, rng:
                rst_count_zeroed(
                    X,
                ),
        ),

        (
            "syn_count_normalized",
            "SYN count and SYN presence normalized",
            lambda X, rng:
                syn_count_normalized(
                    X,
                ),
        ),

        (
            "combined_evasion",
            "TTL + checksum + RST combined perturbation",
            lambda X, rng:
                combined_evasion(
                    X,
                ),
        ),
    ]


# ============================================================
# Baseline sample selection
# ============================================================

def select_correctly_classified_samples(
    df,
    model,
    attack_class_index,
    sample_size,
    random_state,
):
    """
    Select correctly classified Attack and Benign samples.

    Attack population:
        actual = Attack
        baseline prediction = Attack

    Benign population:
        actual = Benign
        baseline prediction = Benign

    The same rows are reused for every scenario.
    """

    X = (
        df[
            config.ML_FEATURES
        ]
        .copy()
        .astype(
            FEATURE_DTYPE
        )
    )

    y = (
        df[
            config.LABEL_COLUMN
        ]
        .copy()
        .astype(
            LABEL_DTYPE
        )
    )

    (
        baseline_predictions,
        baseline_attack_probability,
    ) = predict(
        model,
        X,
        attack_class_index,
    )

    y_array = (
        y.to_numpy()
    )

    correct_attack_mask = (
        (y_array == config.LABEL_ATTACK)
        &
        (
            baseline_predictions
            == config.LABEL_ATTACK
        )
    )

    correct_benign_mask = (
        (y_array == config.LABEL_BENIGN)
        &
        (
            baseline_predictions
            == config.LABEL_BENIGN
        )
    )

    attack_positions = np.flatnonzero(
        correct_attack_mask
    )

    benign_positions = np.flatnonzero(
        correct_benign_mask
    )

    if len(attack_positions) < sample_size:

        raise ValueError(
            "Not enough correctly classified "
            "Attack samples.\n"
            f"Required : {sample_size:,}\n"
            f"Available: {len(attack_positions):,}"
        )

    if len(benign_positions) < sample_size:

        raise ValueError(
            "Not enough correctly classified "
            "Benign samples.\n"
            f"Required : {sample_size:,}\n"
            f"Available: {len(benign_positions):,}"
        )

    rng = np.random.default_rng(
        random_state
    )

    selected_attack_positions = (
        rng.choice(
            attack_positions,
            size=sample_size,
            replace=False,
        )
    )

    selected_benign_positions = (
        rng.choice(
            benign_positions,
            size=sample_size,
            replace=False,
        )
    )

    attack_X = (
        X.iloc[
            selected_attack_positions
        ]
        .reset_index(
            drop=True
        )
    )

    benign_X = (
        X.iloc[
            selected_benign_positions
        ]
        .reset_index(
            drop=True
        )
    )

    attack_baseline_prob = (
        baseline_attack_probability[
            selected_attack_positions
        ]
    )

    benign_baseline_prob = (
        baseline_attack_probability[
            selected_benign_positions
        ]
    )

    return (
        attack_X,
        benign_X,
        attack_baseline_prob,
        benign_baseline_prob,
    )


# ============================================================
# Feature-change measurement
# ============================================================

def calculate_changed_cells(
    original_X,
    perturbed_X,
):
    """
    Count changed non-NaN feature cells.

    NaN -> NaN is not treated as a change.

    Returns:
        changed_cells
        total_cells
        changed_percentage
    """

    original = (
        original_X
        .to_numpy(
            dtype=np.float64
        )
    )

    perturbed = (
        perturbed_X
        .to_numpy(
            dtype=np.float64
        )
    )

    both_nan = (
        np.isnan(original)
        &
        np.isnan(perturbed)
    )

    equal_values = (
        original == perturbed
    )

    changed = (
        ~both_nan
        &
        ~equal_values
    )

    changed_cells = int(
        changed.sum()
    )

    total_cells = int(
        changed.size
    )

    percentage = (
        changed_cells
        / total_cells
        * 100.0
        if total_cells
        else 0.0
    )

    return (
        changed_cells,
        total_cells,
        percentage,
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
    random_state,
    attack_class_index,
):
    """
    Apply one perturbation scenario.

    Ground-truth labels are never modified.
    """

    # --------------------------------------------------------
    # Independent deterministic RNG
    # --------------------------------------------------------

    rng = np.random.default_rng(
        random_state
    )

    # --------------------------------------------------------
    # Perturb Attack samples
    # --------------------------------------------------------

    attack_X_perturbed = (
        perturbation(
            attack_X.copy(),
            rng,
        )
    )

    # --------------------------------------------------------
    # Perturb Benign samples
    # --------------------------------------------------------

    benign_X_perturbed = (
        perturbation(
            benign_X.copy(),
            rng,
        )
    )

    # --------------------------------------------------------
    # Measure feature changes
    # --------------------------------------------------------

    (
        attack_changed_cells,
        attack_total_cells,
        attack_changed_percentage,
    ) = calculate_changed_cells(
        attack_X,
        attack_X_perturbed,
    )

    (
        benign_changed_cells,
        benign_total_cells,
        benign_changed_percentage,
    ) = calculate_changed_cells(
        benign_X,
        benign_X_perturbed,
    )

    # --------------------------------------------------------
    # Predict perturbed Attack samples
    # --------------------------------------------------------

    (
        attack_predictions,
        attack_perturbed_prob,
    ) = predict(
        model,
        attack_X_perturbed,
        attack_class_index,
    )

    # --------------------------------------------------------
    # Predict perturbed Benign samples
    # --------------------------------------------------------

    (
        benign_predictions,
        benign_perturbed_prob,
    ) = predict(
        model,
        benign_X_perturbed,
        attack_class_index,
    )

    # --------------------------------------------------------
    # Attack Evasion Rate
    # --------------------------------------------------------
    #
    # Correctly detected Attack -> predicted Benign
    #

    attack_evasion_rate = float(
        np.mean(
            attack_predictions
            == config.LABEL_BENIGN
        )
    )

    # --------------------------------------------------------
    # Benign Induction Rate
    # --------------------------------------------------------
    #
    # Correctly classified Benign -> predicted Attack
    #

    benign_induction_rate = float(
        np.mean(
            benign_predictions
            == config.LABEL_ATTACK
        )
    )

    # --------------------------------------------------------
    # Prediction flip counts
    # --------------------------------------------------------

    attack_to_benign_count = int(
        np.sum(
            attack_predictions
            == config.LABEL_BENIGN
        )
    )

    benign_to_attack_count = int(
        np.sum(
            benign_predictions
            == config.LABEL_ATTACK
        )
    )

    # Absolute probability shifts
    # --------------------------------------------------------

    attack_probability_shift = (
        np.abs(
            attack_perturbed_prob
            - attack_baseline_prob
        )
    )

    benign_probability_shift = (
        np.abs(
            benign_perturbed_prob
            - benign_baseline_prob
        )
    )

    mean_prob_shift_attack = float(
        np.mean(
            attack_probability_shift
        )
    )

    mean_prob_shift_benign = float(
        np.mean(
            benign_probability_shift
        )
    )

    # --------------------------------------------------------
    # Directional probability changes
    # --------------------------------------------------------
    #
    # Attack:
    #   baseline - perturbed
    #
    # Positive = attack probability decreased.
    #

    attack_probability_drop = (
        attack_baseline_prob
        - attack_perturbed_prob
    )

    mean_prob_drop_attack = float(
        np.mean(
            attack_probability_drop
        )
    )

    # --------------------------------------------------------
    # Benign:
    #   perturbed - baseline
    #
    # Positive = attack probability increased.
    #

    benign_probability_rise = (
        benign_perturbed_prob
        - benign_baseline_prob
    )

    mean_prob_rise_benign = float(
        np.mean(
            benign_probability_rise
        )
    )

    return {
        "scenario":
            scenario_name,

        "description":
            description,

        "attack_evasion_rate":
            attack_evasion_rate,

        "benign_induction_rate":
            benign_induction_rate,

        "attack_to_benign_count":
            attack_to_benign_count,

        "benign_to_attack_count":
            benign_to_attack_count,

        "mean_prob_shift_attack":
            mean_prob_shift_attack,

        "mean_prob_shift_benign":
            mean_prob_shift_benign,

        "mean_prob_drop_attack":
            mean_prob_drop_attack,

        "mean_prob_rise_benign":
            mean_prob_rise_benign,

        "attack_changed_cells":
            attack_changed_cells,

        "attack_total_cells":
            attack_total_cells,

        "attack_changed_percentage":
            attack_changed_percentage,

        "benign_changed_cells":
            benign_changed_cells,

        "benign_total_cells":
            benign_total_cells,

        "benign_changed_percentage":
            benign_changed_percentage,

        "n_attack_tested":
            int(len(attack_X)),

        "n_benign_tested":
            int(len(benign_X)),
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Random Forest robustness using "
            "controlled feature-level perturbations."
        )
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    parser.add_argument(
        "--model",
        type=str,
        default=str(
            MODEL_FILE
        ),
        help=(
            "Random Forest .joblib path. "
            "Defaults to config.RANDOM_FOREST_MODEL_FILE."
        ),
    )

    # --------------------------------------------------------
    # Sample size
    # --------------------------------------------------------

    parser.add_argument(
        "--sample-size",
        type=int,
        default=(
            DEFAULT_SAMPLE_SIZE_PER_CLASS
        ),
        help=(
            "Correctly classified Attack and Benign "
            "samples used per scenario."
        ),
    )

    # --------------------------------------------------------
    # Random state
    # --------------------------------------------------------

    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed for reproducible sampling.",
    )

    # --------------------------------------------------------
    # Output name
    # --------------------------------------------------------

    parser.add_argument(
        "--out-name",
        type=str,
        default="robustness_results.csv",
        help="Output CSV filename.",
    )

    args = parser.parse_args()

    if args.sample_size <= 0:

        raise ValueError(
            "--sample-size must be greater than 0."
        )

    # ========================================================
    # Header
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "NetworkIDS - ROBUSTNESS EVALUATION"
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

    validate_config()

    print(
        f"  Test dataset : "
        f"{TEST_FILE}"
    )

    print(
        f"  Model        : "
        f"{args.model}"
    )

    print(
        f"  Sample/class : "
        f"{args.sample_size:,}"
    )

    print(
        f"  Random state : "
        f"{args.random_state}"
    )

    # ========================================================
    # 2. Load dataset
    # ========================================================

    print_section(
        "[2/6] Loading held-out test dataset"
    )

    df = load_dataset()

    validate_dataset(
        df
    )

    X = (
        df[
            config.ML_FEATURES
        ]
        .copy()
        .astype(
            FEATURE_DTYPE
        )
    )

    y = (
        df[
            config.LABEL_COLUMN
        ]
        .copy()
        .astype(
            LABEL_DTYPE
        )
    )

    print(
        f"  X shape: "
        f"{X.shape}"
    )

    # ========================================================
    # 3. Load model
    # ========================================================

    print_section(
        "[3/6] Loading Random Forest"
    )

    model = load_model(
        args.model
    )

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
        f"  Attack label       : "
        f"{config.LABEL_ATTACK}"
    )

    print(
        f"  Attack class index : "
        f"{attack_class_index}"
    )

    # --------------------------------------------------------
    # Clean baseline
    # --------------------------------------------------------

    baseline_predictions, baseline_probability = (
        predict(
            model,
            X,
            attack_class_index,
        )
    )

    baseline_accuracy = float(
        np.mean(
            baseline_predictions
            == y.to_numpy()
        )
    )

    print(
        f"  Clean test accuracy: "
        f"{baseline_accuracy:.6f}"
    )

    # ========================================================
    # 4. Select baseline-correct samples
    # ========================================================

    print_section(
        "[4/6] Selecting baseline-correct samples"
    )

    (
        attack_X,
        benign_X,
        attack_baseline_prob,
        benign_baseline_prob,
    ) = select_correctly_classified_samples(
        df=df,
        model=model,
        attack_class_index=attack_class_index,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )

    print(
        f"  Correct Attack samples: "
        f"{len(attack_X):,}"
    )

    print(
        f"  Correct Benign samples: "
        f"{len(benign_X):,}"
    )

    print(
        "  The same baseline-correct samples "
        "are reused across all scenarios."
    )

    # ========================================================
    # 5. Run robustness scenarios
    # ========================================================

    print_section(
        "[5/6] Running perturbation scenarios"
    )

    scenarios = (
        build_scenarios()
    )

    results = []

    for (
        scenario_number,
        (
            scenario_name,
            description,
            perturbation,
        ),
    ) in enumerate(
        scenarios,
        start=1,
    ):

        print(
            f"\n[{scenario_number}/"
            f"{len(scenarios)}] "
            f"{scenario_name}"
        )

        print(
            f"  {description}"
        )

        # Use a deterministic but different
        # seed for every scenario.
        scenario_seed = (
            args.random_state
            + scenario_number
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
            random_state=scenario_seed,
            attack_class_index=attack_class_index,
        )

        results.append(
            result
        )

        print(
            f"  Attack evasion rate : "
            f"{result['attack_evasion_rate']:.4%}"
        )

        print(
            f"  Benign induction    : "
            f"{result['benign_induction_rate']:.4%}"
        )

        print(
            f"  Attack -> Benign    : "
            f"{result['attack_to_benign_count']:,} / {result['n_attack_tested']:,}"
        )

        print(
            f"  Benign -> Attack    : "
            f"{result['benign_to_attack_count']:,} / {result['n_benign_tested']:,}"
        )

        print(
            f"  Attack P drop       : "
            f"{result['mean_prob_drop_attack']:.4f}"
        )

        print(
            f"  Benign P rise       : "
            f"{result['mean_prob_rise_benign']:.4f}"
        )

        print(
            f"  Attack changed cells: "
            f"{result['attack_changed_percentage']:.4f}%"
        )

        print(
            f"  Benign changed cells: "
            f"{result['benign_changed_percentage']:.4f}%"
        )

    # ========================================================
    # 6. Save results
    # ========================================================

    print_section(
        "[6/6] Saving robustness results"
    )

    ROBUSTNESS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df = pd.DataFrame(
        results
    )

    output_path = (
        ROBUSTNESS_DIR
        / args.out_name
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    report_path = (
        ROBUSTNESS_DIR
        / "robustness_report.txt"
    )

    report_lines = [
        "NetworkIDS - Random Forest Robustness Evaluation Report",
        "=" * 70,
        "",
        f"Model: {Path(args.model).resolve()}",
        f"Test dataset: {Path(TEST_FILE).resolve()}",
        f"Feature count: {len(config.ML_FEATURES)}",
        f"Sample size per class: {args.sample_size:,}",
        f"Random state: {args.random_state}",
        f"Clean test accuracy: {baseline_accuracy:.6f}",
        "",
        "Evaluation type: Controlled feature-level perturbation",
        "Packet-level PCAP modification: False",
        "NaN policy: NaN retained; no imputation",
        "",
        "Scenario Results",
        "-" * 70,
    ]

    for row in results:
        report_lines.extend([
            f"Scenario: {row['scenario']}",
            f"Description: {row['description']}",
            f"Attack evasion rate: {row['attack_evasion_rate']:.4%} ({row['attack_to_benign_count']:,} / {row['n_attack_tested']:,})",
            f"Benign induction rate: {row['benign_induction_rate']:.4%} ({row['benign_to_attack_count']:,} / {row['n_benign_tested']:,})",
            f"Mean P(Attack) drop - Attack: {row['mean_prob_drop_attack']:.6f}",
            f"Mean P(Attack) rise - Benign: {row['mean_prob_rise_benign']:.6f}",
            f"Mean absolute probability shift - Attack: {row['mean_prob_shift_attack']:.6f}",
            f"Mean absolute probability shift - Benign: {row['mean_prob_shift_benign']:.6f}",
            "",
        ])

    report_lines += ["=" * 70, "End of robustness evaluation report."]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    metadata = {

        "model":
            "RandomForestClassifier",

        "model_path":
            str(
                Path(
                    args.model
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

        "sample_size_per_class":
            int(
                args.sample_size
            ),

        "random_state":
            int(
                args.random_state
            ),

        "scenarios":
            len(
                scenarios
            ),

        "baseline_clean_accuracy":
            baseline_accuracy,

        "sampling_policy":
            (
                "Correctly classified "
                "Attack and Benign test samples; "
                "same rows reused across scenarios."
            ),

        "ground_truth_policy":
            (
                "Ground-truth labels are retained "
                "unchanged during perturbation."
            ),

        "nan_policy":
            "NaN retained; no imputation",

        "evaluation_type":
            (
                "Controlled feature-level "
                "perturbation robustness evaluation"
            ),

        "packet_level_modification":
            False,

        "results_file":
            str(
                output_path.resolve()
            ),

        "report_file":
            str(
                report_path.resolve()
            ),
    }

    metadata_path = (
        ROBUSTNESS_DIR
        / "robustness_results_metadata.json"
    )

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

    # ========================================================
    # Final summary
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "ROBUSTNESS EVALUATION SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"\nClean test accuracy: "
        f"{baseline_accuracy:.6f}"
    )

    print(
        "\nScenario results:"
    )

    for _, row in (
        results_df.iterrows()
    ):

        print(
            f"  {row['scenario']:<25} "
            f"Attack Evasion="
            f"{row['attack_evasion_rate']:.2%} | "
            f"Benign Induction="
            f"{row['benign_induction_rate']:.2%} | "
            f"A->B={row['attack_to_benign_count']:,} | "
            f"B->A={row['benign_to_attack_count']:,}"
        )

    print(
        "\nResults:"
    )

    print(
        f"  {output_path}"
    )

    print(
        "\nMetadata:"
    )

    print(
        f"  {metadata_path}"
    )

    print(
        "\nReport:"
    )

    print(
        f"  {report_path}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "ROBUSTNESS EVALUATION COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()