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

TEST_FILE = (
    config.PROCESSED_DIR
    / "split"
    / "test_dataset.csv"
)

DEFAULT_MODEL_FILE = (
    config.MODELS_DIR
    / "random_forest_robust_v2.pkl"
)

DEFAULT_OUTPUT_DIR = (
    config.MODELS_DIR
    / "robustness"
    / "port_analysis"
)

RANDOM_STATE = 42
SAMPLE_SIZE_PER_CLASS = 2000

MIN_PORT = 1
MAX_PORT = 65535


# ============================================================
# Utility
# ============================================================

def load_model(model_path):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    print(f"Loading model: {model_path}")

    return joblib.load(model_path)


def predict(model, X):
    """
    Return predictions and Attack probabilities.
    """

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)

    attack_index = list(model.classes_).index(
        config.LABEL_ATTACK
    )

    attack_probability = probabilities[
        :, attack_index
    ]

    return predictions, attack_probability


# ============================================================
# Perturbation functions
# ============================================================

def randomize_port_column(
    X,
    column,
    rng,
):
    """
    Replace observed port values with random valid TCP/UDP
    port numbers from 1 to 65535.

    NaN values remain NaN.

    This is a controlled perturbation rather than a simulation
    of a specific real-world attack.
    """

    X_new = X.copy()

    if column not in X_new.columns:
        return X_new

    mask = X_new[column].notna()

    random_ports = rng.integers(
        low=MIN_PORT,
        high=MAX_PORT + 1,
        size=mask.sum(),
    )

    X_new.loc[
        mask,
        column
    ] = random_ports

    return X_new


def dst_port_randomized(X, rng):
    return randomize_port_column(
        X,
        "dst_port",
        rng,
    )


def src_port_randomized(X, rng):
    return randomize_port_column(
        X,
        "src_port",
        rng,
    )


def both_ports_randomized(X, rng):
    X_new = randomize_port_column(
        X,
        "src_port",
        rng,
    )

    X_new = randomize_port_column(
        X_new,
        "dst_port",
        rng,
    )

    return X_new


def ports_masked(X):
    """
    Remove both source and destination port information.

    NaN is used consistently with the existing pipeline.
    """

    X_new = X.copy()

    if "src_port" in X_new.columns:
        X_new["src_port"] = np.nan

    if "dst_port" in X_new.columns:
        X_new["dst_port"] = np.nan

    return X_new


# ============================================================
# Scenario definitions
# ============================================================

def build_scenarios():

    return [
        (
            "dst_port_randomized",
            "Destination port replaced with random valid port",
            lambda X, rng: dst_port_randomized(
                X,
                rng,
            ),
        ),
        (
            "src_port_randomized",
            "Source port replaced with random valid port",
            lambda X, rng: src_port_randomized(
                X,
                rng,
            ),
        ),
        (
            "both_ports_randomized",
            "Source and destination ports randomized",
            lambda X, rng: both_ports_randomized(
                X,
                rng,
            ),
        ),
        (
            "ports_masked",
            "Source and destination ports removed",
            lambda X, rng: ports_masked(
                X,
            ),
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
    Reproduce the same sampling procedure used by the
    existing robustness evaluation.

    Attack:
        Ground truth = Attack
        Baseline prediction = Attack

    Benign:
        Ground truth = Benign
        Baseline prediction = Benign
    """

    X = df[
        config.ML_FEATURES
    ]

    y = df[
        config.LABEL_COLUMN
    ].astype(int)

    baseline_predictions, baseline_probabilities = predict(
        model,
        X,
    )

    correct_attack_mask = (
        (y == config.LABEL_ATTACK)
        &
        (
            baseline_predictions
            == config.LABEL_ATTACK
        )
    )

    correct_benign_mask = (
        (y == config.LABEL_BENIGN)
        &
        (
            baseline_predictions
            == config.LABEL_BENIGN
        )
    )

    attack_indices = df.index[
        correct_attack_mask
    ].to_numpy()

    benign_indices = df.index[
        correct_benign_mask
    ].to_numpy()

    if len(attack_indices) < sample_size:
        raise ValueError(
            "Not enough correctly classified "
            "Attack samples."
        )

    if len(benign_indices) < sample_size:
        raise ValueError(
            "Not enough correctly classified "
            "Benign samples."
        )

    # EXACT SAME RNG ORDER AS ORIGINAL ROBUSTNESS TEST
    rng = np.random.default_rng(
        random_state
    )

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

    return (
        selected_attack_indices,
        selected_benign_indices,
        baseline_predictions,
        baseline_probabilities,
    )


# ============================================================
# Scenario evaluation
# ============================================================

def evaluate_attack_and_benign(
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
    Evaluate one port perturbation scenario.
    """

    attack_X_perturbed = perturbation(
        attack_X.copy(),
        rng,
    )

    benign_X_perturbed = perturbation(
        benign_X.copy(),
        rng,
    )

    attack_predictions, attack_probabilities = predict(
        model,
        attack_X_perturbed,
    )

    benign_predictions, benign_probabilities = predict(
        model,
        benign_X_perturbed,
    )

    # --------------------------------------------------------
    # Attack evasion
    # --------------------------------------------------------

    attack_evasion = (
        attack_predictions
        == config.LABEL_BENIGN
    )

    attack_evasion_rate = np.mean(
        attack_evasion
    )

    # --------------------------------------------------------
    # Benign induction
    # --------------------------------------------------------

    benign_induction = (
        benign_predictions
        == config.LABEL_ATTACK
    )

    benign_induction_rate = np.mean(
        benign_induction
    )

    # --------------------------------------------------------
    # Probability shifts
    # --------------------------------------------------------

    attack_probability_shift = np.abs(
        attack_probabilities
        - attack_baseline_prob
    )

    benign_probability_shift = np.abs(
        benign_probabilities
        - benign_baseline_prob
    )

    attack_probability_drop = (
        attack_baseline_prob
        - attack_probabilities
    )

    benign_probability_rise = (
        benign_probabilities
        - benign_baseline_prob
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
            np.mean(
                attack_probability_shift
            )
        ),

        "mean_prob_shift_benign": float(
            np.mean(
                benign_probability_shift
            )
        ),

        "mean_prob_drop_attack": float(
            np.mean(
                attack_probability_drop
            )
        ),

        "mean_prob_rise_benign": float(
            np.mean(
                benign_probability_rise
            )
        ),

        "n_attack_tested": int(
            len(attack_X)
        ),

        "n_benign_tested": int(
            len(benign_X)
        ),
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Current V2 robustness against "
            "controlled source/destination port perturbations."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default=str(
            DEFAULT_MODEL_FILE
        ),
        help="Current V2 Random Forest model.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
        help="Output directory.",
    )

    args = parser.parse_args()

    model_path = Path(
        args.model
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Header
    # ========================================================

    print("=" * 80)
    print("PORT ROBUSTNESS ANALYSIS")
    print("=" * 80)

    print(
        f"Test dataset : {TEST_FILE}"
    )

    print(
        f"Model        : {model_path}"
    )

    print(
        f"Output dir   : {output_dir}"
    )

    print()

    # ========================================================
    # Load dataset
    # ========================================================

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test dataset not found:\n{TEST_FILE}"
        )

    df = pd.read_csv(
        TEST_FILE
    )

    print(
        f"Loaded test dataset: "
        f"{len(df):,} rows x "
        f"{len(df.columns)} columns"
    )

    if len(config.ML_FEATURES) != 42:
        raise ValueError(
            f"Expected 42 ML features, "
            f"found {len(config.ML_FEATURES)}."
        )

    missing_features = [
        feature
        for feature in config.ML_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing ML features:\n"
            + "\n".join(
                missing_features
            )
        )

    # ========================================================
    # Load model
    # ========================================================

    model = load_model(
        model_path
    )

    if hasattr(
        model,
        "n_features_in_",
    ):

        if model.n_features_in_ != 42:
            raise ValueError(
                f"Model expects "
                f"{model.n_features_in_} features, "
                f"not 42."
            )

    print(
        f"Model classes: "
        f"{list(model.classes_)}"
    )

    # ========================================================
    # Prepare data
    # ========================================================

    X = df[
        config.ML_FEATURES
    ].copy()

    y = df[
        config.LABEL_COLUMN
    ].astype(int)

    print(
        f"ML feature matrix: "
        f"{X.shape}"
    )

    print(
        f"NaN cells: "
        f"{int(X.isna().sum().sum()):,}"
    )

    # ========================================================
    # Baseline predictions
    # ========================================================

    baseline_predictions, baseline_probabilities = predict(
        model,
        X,
    )

    baseline_accuracy = np.mean(
        baseline_predictions
        == y.to_numpy()
    )

    print(
        f"Clean test accuracy: "
        f"{baseline_accuracy:.6f}"
    )

    # ========================================================
    # Reproduce robustness samples
    # ========================================================

    (
        attack_indices,
        benign_indices,
        baseline_predictions,
        baseline_probabilities,
    ) = select_correctly_classified_samples(
        df=df,
        model=model,
        sample_size=SAMPLE_SIZE_PER_CLASS,
        random_state=RANDOM_STATE,
    )

    print()
    print(
        "-" * 80
    )
    print("ROBUSTNESS SAMPLE")
    print(
        "-" * 80
    )

    print(
        f"Attack samples: "
        f"{len(attack_indices):,}"
    )

    print(
        f"Benign samples: "
        f"{len(benign_indices):,}"
    )

    # ========================================================
    # Prepare sample data
    # ========================================================

    attack_df = df.loc[
        attack_indices
    ].copy()

    benign_df = df.loc[
        benign_indices
    ].copy()

    attack_X = attack_df[
        config.ML_FEATURES
    ].copy()

    benign_X = benign_df[
        config.ML_FEATURES
    ].copy()

    attack_baseline_prob = (
        baseline_probabilities[
            df.index.get_indexer(
                attack_indices
            )
        ]
    )

    benign_baseline_prob = (
        baseline_probabilities[
            df.index.get_indexer(
                benign_indices
            )
        ]
    )

    # ========================================================
    # Build scenarios
    # ========================================================

    scenarios = build_scenarios()

    results = []

    sample_level_outputs = {}

    # ========================================================
    # Run scenarios
    # ========================================================

    for (
        scenario_name,
        description,
        perturbation,
    ) in scenarios:

        print()
        print(
            "-" * 80
        )

        print(
            f"Scenario: "
            f"{scenario_name}"
        )

        print(
            f"Description: "
            f"{description}"
        )

        # Each scenario uses a deterministic RNG.
        scenario_rng = np.random.default_rng(
            RANDOM_STATE
        )

        # ----------------------------------------------------
        # Perturb attack
        # ----------------------------------------------------

        attack_X_perturbed = perturbation(
            attack_X.copy(),
            scenario_rng,
        )

        # ----------------------------------------------------
        # Perturb benign
        # ----------------------------------------------------

        benign_X_perturbed = perturbation(
            benign_X.copy(),
            scenario_rng,
        )

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        attack_predictions, attack_probabilities = predict(
            model,
            attack_X_perturbed,
        )

        benign_predictions, benign_probabilities = predict(
            model,
            benign_X_perturbed,
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        attack_evasion = (
            attack_predictions
            == config.LABEL_BENIGN
        )

        benign_induction = (
            benign_predictions
            == config.LABEL_ATTACK
        )

        attack_evasion_rate = np.mean(
            attack_evasion
        )

        benign_induction_rate = np.mean(
            benign_induction
        )

        attack_prob_shift = np.abs(
            attack_probabilities
            - attack_baseline_prob
        )

        benign_prob_shift = np.abs(
            benign_probabilities
            - benign_baseline_prob
        )

        attack_prob_drop = (
            attack_baseline_prob
            - attack_probabilities
        )

        benign_prob_rise = (
            benign_probabilities
            - benign_baseline_prob
        )

        result = {
            "scenario": scenario_name,
            "description": description,

            "attack_evasion_rate": float(
                attack_evasion_rate
            ),

            "benign_induction_rate": float(
                benign_induction_rate
            ),

            "mean_prob_shift_attack": float(
                np.mean(
                    attack_prob_shift
                )
            ),

            "mean_prob_shift_benign": float(
                np.mean(
                    benign_prob_shift
                )
            ),

            "mean_prob_drop_attack": float(
                np.mean(
                    attack_prob_drop
                )
            ),

            "mean_prob_rise_benign": float(
                np.mean(
                    benign_prob_rise
                )
            ),

            "n_attack_tested": int(
                len(attack_X)
            ),

            "n_benign_tested": int(
                len(benign_X)
            ),
        }

        results.append(
            result
        )

        # ----------------------------------------------------
        # Sample-level attack results
        # ----------------------------------------------------

        attack_output = attack_df[
            config.METADATA_COLUMNS
            + [config.LABEL_COLUMN]
        ].copy()

        attack_output = attack_output.rename(
            columns={
                config.LABEL_COLUMN:
                    "actual_label"
            }
        )

        attack_output[
            "baseline_prediction"
        ] = attack_predictions * 0 + config.LABEL_ATTACK

        # Use the actual clean baseline predictions
        attack_output[
            "baseline_prediction"
        ] = baseline_predictions[
            df.index.get_indexer(
                attack_indices
            )
        ]

        attack_output[
            "baseline_attack_probability"
        ] = attack_baseline_prob

        attack_output[
            "perturbed_prediction"
        ] = attack_predictions

        attack_output[
            "perturbed_attack_probability"
        ] = attack_probabilities

        attack_output[
            "prediction_changed"
        ] = (
            attack_output[
                "baseline_prediction"
            ]
            != attack_output[
                "perturbed_prediction"
            ]
        )

        attack_output[
            "attack_evasion"
        ] = (
            attack_output[
                "baseline_prediction"
            ]
            == config.LABEL_ATTACK
        ) & (
            attack_output[
                "perturbed_prediction"
            ]
            == config.LABEL_BENIGN
        )

        if "dst_port" in attack_df.columns:
            attack_output[
                "original_dst_port"
            ] = attack_df[
                "dst_port"
            ].to_numpy()

        if "src_port" in attack_df.columns:
            attack_output[
                "original_src_port"
            ] = attack_df[
                "src_port"
            ].to_numpy()

        attack_output[
            "scenario"
        ] = scenario_name

        # ----------------------------------------------------
        # Sample-level benign results
        # ----------------------------------------------------

        benign_output = benign_df[
            config.METADATA_COLUMNS
            + [config.LABEL_COLUMN]
        ].copy()

        benign_output = benign_output.rename(
            columns={
                config.LABEL_COLUMN:
                    "actual_label"
            }
        )

        benign_output[
            "baseline_prediction"
        ] = baseline_predictions[
            df.index.get_indexer(
                benign_indices
            )
        ]

        benign_output[
            "baseline_attack_probability"
        ] = benign_baseline_prob

        benign_output[
            "perturbed_prediction"
        ] = benign_predictions

        benign_output[
            "perturbed_attack_probability"
        ] = benign_probabilities

        benign_output[
            "prediction_changed"
        ] = (
            benign_output[
                "baseline_prediction"
            ]
            != benign_output[
                "perturbed_prediction"
            ]
        )

        benign_output[
            "benign_induction"
        ] = (
            benign_output[
                "baseline_prediction"
            ]
            == config.LABEL_BENIGN
        ) & (
            benign_output[
                "perturbed_prediction"
            ]
            == config.LABEL_ATTACK
        )

        if "dst_port" in benign_df.columns:
            benign_output[
                "original_dst_port"
            ] = benign_df[
                "dst_port"
            ].to_numpy()

        if "src_port" in benign_df.columns:
            benign_output[
                "original_src_port"
            ] = benign_df[
                "src_port"
            ].to_numpy()

        benign_output[
            "scenario"
        ] = scenario_name

        sample_level_outputs[
            scenario_name
        ] = (
            attack_output,
            benign_output,
        )

        print(
            f"Attack Evasion Rate : "
            f"{attack_evasion_rate:.4%}"
        )

        print(
            f"Benign Induction Rate: "
            f"{benign_induction_rate:.4%}"
        )

        print(
            f"Mean Attack Prob Shift: "
            f"{np.mean(attack_prob_shift):.4f}"
        )

        print(
            f"Mean Benign Prob Shift: "
            f"{np.mean(benign_prob_shift):.4f}"
        )

    # ========================================================
    # Save summary
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    summary_path = (
        output_dir
        / "port_robustness_results.csv"
    )

    results_df.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # Save sample-level results
    # ========================================================

    for scenario_name, (
        attack_output,
        benign_output,
    ) in sample_level_outputs.items():

        attack_path = (
            output_dir
            / f"{scenario_name}_attack.csv"
        )

        benign_path = (
            output_dir
            / f"{scenario_name}_benign.csv"
        )

        attack_output.to_csv(
            attack_path,
            index=False,
        )

        benign_output.to_csv(
            benign_path,
            index=False,
        )

    # ========================================================
    # Save evasion / induction cases
    # ========================================================

    for scenario_name, (
        attack_output,
        benign_output,
    ) in sample_level_outputs.items():

        evasion = attack_output[
            attack_output[
                "attack_evasion"
            ]
        ].copy()

        induction = benign_output[
            benign_output[
                "benign_induction"
            ]
        ].copy()

        evasion.to_csv(
            output_dir
            / f"{scenario_name}_attack_evasion.csv",
            index=False,
        )

        induction.to_csv(
            output_dir
            / f"{scenario_name}_benign_induction.csv",
            index=False,
        )

    # ========================================================
    # Metadata
    # ========================================================

    metadata = {
        "model_path": str(
            model_path
        ),

        "test_dataset": str(
            TEST_FILE
        ),

        "random_state": RANDOM_STATE,

        "sample_size_per_class":
            SAMPLE_SIZE_PER_CLASS,

        "n_ml_features":
            len(config.ML_FEATURES),

        "port_range":
            [MIN_PORT, MAX_PORT],

        "controlled_perturbation":
            True,

        "ground_truth_preserved":
            True,

        "baseline_model_reused_without_retraining":
            True,

        "scenarios":
            [
                name
                for name, _, _ in scenarios
            ],
    }

    metadata_path = (
        output_dir
        / "port_robustness_metadata.json"
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

    # ========================================================
    # Final summary
    # ========================================================

    print()
    print("=" * 80)
    print("PORT ROBUSTNESS SUMMARY")
    print("=" * 80)

    print(
        results_df[
            [
                "scenario",
                "attack_evasion_rate",
                "benign_induction_rate",
                "mean_prob_shift_attack",
                "mean_prob_shift_benign",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Summary saved to:"
    )

    print(
        summary_path
    )

    print()
    print(
        "Sample-level results saved to:"
    )

    print(
        output_dir
    )

    print()
    print(
        "PASS: Port robustness analysis completed."
    )


if __name__ == "__main__":
    main()