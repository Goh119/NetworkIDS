"""
NetworkIDS - External / Modern Dataset Model Comparison.

Location:
    src/evaluate_external_tests.py

Purpose:
    Compare the baseline Random Forest (Model A) and the enhanced
    Random Forest (Model B) on external / modern test scenarios.

Models:
    Model A:
        random_forest/rf_model.joblib

    Model B:
        random_forest/rf_model_v1.joblib

External test groups:
    1. MQTT known-attack generalization
    2. Mirai unseen evaluation
    3. MQTT Malformed unseen evaluation
    4. MITM Impersonation unseen evaluation
    5. IP Spoofing supplementary unseen evaluation

Important evaluation rule:
    If a test CSV contains a valid label column, true attack Recall/FNR
    are calculated using rows where label == Attack.

    If no label column exists, the script does NOT claim strict
    per-flow Recall. Instead, it reports Scenario Detection Rate:

        predicted Attack flows / total scenario flows

    This is necessary because attack-scenario captures may also
    contain background traffic.

Missing-value policy:
    NaN values are retained.
    No imputation is applied.

Usage:
    python src/evaluate_external_tests.py
"""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


# ============================================================
# Project import
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)

import src.config as config


# ============================================================
# Paths
# ============================================================

TEST_DIR = (
    ROOT
    / "data"
    / "test"
)

MODEL_A_FILE = (
    ROOT
    / "random_forest"
    / "rf_model.joblib"
)

MODEL_B_FILE = (
    ROOT
    / "random_forest"
    / "rf_model_v1.joblib"
)

OUTPUT_DIR = (
    ROOT
    / "random_forest"
    / "external_evaluation"
)

FILE_RESULTS_OUT = (
    OUTPUT_DIR
    / "external_test_file_results.csv"
)

FAMILY_RESULTS_OUT = (
    OUTPUT_DIR
    / "external_test_family_results.csv"
)

REPORT_OUT = (
    OUTPUT_DIR
    / "external_test_comparison_report.txt"
)


# ============================================================
# Settings
# ============================================================

CHUNK_SIZE = 100_000

FEATURES = list(
    config.ML_FEATURES
)

LABEL_COLUMN = (
    config.LABEL_COLUMN
)

BENIGN_LABEL = (
    config.LABEL_BENIGN
)

ATTACK_LABEL = (
    config.LABEL_ATTACK
)


# ============================================================
# Test scenario definitions
#
# Only FLOW CSV files should be evaluated.
#
# The patterns are intentionally specific so unrelated files
# inside data/test are not accidentally included.
# ============================================================

TEST_GROUPS = {

    "MQTT Known Attack Generalization": {

        "evaluation_type":
            "KNOWN_ATTACK_GENERALIZATION",

        "patterns": [
            "MQTT-DDoS-Connect_Flood_test*flow*.csv",
            "MQTT-DDoS-Publish_Flood_test*flow*.csv",
            "MQTT-DoS-Connect_Flood_test*flow*.csv",
            "MQTT-DoS-Publish_Flood_test*flow*.csv",
        ],
    },

    "Mirai": {

        "evaluation_type":
            "PRIMARY_UNSEEN",

        "patterns": [
            "attack_malware_mirai-syn-flood_*flow*.csv",
            "attack_malware_mirai-udp-flood_*flow*.csv",
        ],
    },

    "MQTT Malformed Data": {

        "evaluation_type":
            "PRIMARY_UNSEEN",

        "patterns": [
            "MQTT-Malformed_Data_train*flow*.csv",
            "MQTT-Malformed_Data_test*flow*.csv",
        ],
    },

    "MITM Impersonation": {

        "evaluation_type":
            "PRIMARY_UNSEEN",

        "patterns": [
            "attack_mitm_impersonation_*flow*.csv",
        ],
    },

    "IP Spoofing": {

        "evaluation_type":
            "SUPPLEMENTARY_UNSEEN",

        "patterns": [
            "attack_mitm_ip-spoofing_*flow*.csv",
        ],
    },
}


# ============================================================
# Helpers
# ============================================================

def print_section(title):

    print()

    print(
        "=" * 75
    )

    print(
        title
    )

    print(
        "=" * 75
    )


# ============================================================
# File discovery
# ============================================================

def discover_test_files():
    """
    Discover only the external FLOW CSV files defined above.

    Returns:
        list of dictionaries:
            family
            evaluation_type
            path
    """

    discovered = []

    used_paths = set()

    for family, info in TEST_GROUPS.items():

        family_files = []

        for pattern in info["patterns"]:

            matches = sorted(
                TEST_DIR.glob(
                    pattern
                )
            )

            for path in matches:

                if not path.is_file():
                    continue

                resolved = (
                    path
                    .resolve()
                )

                if resolved in used_paths:
                    continue

                # Extra safety:
                # never accidentally evaluate packet CSV.
                if "packet" in path.name.lower():
                    continue

                family_files.append(
                    path
                )

                used_paths.add(
                    resolved
                )

        if not family_files:

            raise FileNotFoundError(
                f"No flow CSV files found for test family:\n"
                f"  {family}\n\n"
                f"Expected under:\n"
                f"  {TEST_DIR}\n\n"
                f"Patterns:\n"
                + "\n".join(
                    f"  - {pattern}"
                    for pattern
                    in info["patterns"]
                )
            )

        for path in family_files:

            discovered.append({

                "family":
                    family,

                "evaluation_type":
                    info[
                        "evaluation_type"
                    ],

                "path":
                    path,
            })

    return discovered


# ============================================================
# Model validation
# ============================================================

def load_and_validate_model(
    path,
    name,
):
    """
    Load a trained Random Forest and validate its feature schema.
    """

    if not path.is_file():

        raise FileNotFoundError(
            f"{name} not found:\n"
            f"{path}"
        )

    print(
        f"Loading {name}: "
        f"{path}"
    )

    model = joblib.load(
        path
    )

    if not hasattr(
        model,
        "predict"
    ):

        raise TypeError(
            f"{name} does not provide predict()."
        )

    if not hasattr(
        model,
        "predict_proba"
    ):

        raise TypeError(
            f"{name} does not provide predict_proba()."
        )

    # --------------------------------------------------------
    # Validate feature count
    # --------------------------------------------------------

    if hasattr(
        model,
        "n_features_in_"
    ):

        if (
            int(
                model.n_features_in_
            )
            != len(FEATURES)
        ):

            raise ValueError(
                f"{name}: expected "
                f"{len(FEATURES)} features, "
                f"model expects "
                f"{model.n_features_in_}."
            )

    # --------------------------------------------------------
    # Validate feature order if sklearn stored names.
    # --------------------------------------------------------

    if hasattr(
        model,
        "feature_names_in_"
    ):

        model_features = list(
            model.feature_names_in_
        )

        if (
            model_features
            != FEATURES
        ):

            raise ValueError(
                f"{name}: feature order does not match "
                "config.ML_FEATURES."
            )

    # --------------------------------------------------------
    # Validate Attack class
    # --------------------------------------------------------

    classes = list(
        model.classes_
    )

    if ATTACK_LABEL not in classes:

        raise ValueError(
            f"{name}: Attack label "
            f"{ATTACK_LABEL} not found "
            f"in model.classes_: {classes}"
        )

    return model


# ============================================================
# Attack probability
# ============================================================

def get_attack_probability(
    model,
    X,
):
    """
    Return P(Attack) using model.classes_.
    """

    probabilities = (
        model.predict_proba(
            X
        )
    )

    classes = list(
        model.classes_
    )

    attack_index = (
        classes.index(
            ATTACK_LABEL
        )
    )

    return (
        probabilities[
            :,
            attack_index
        ]
    )


# ============================================================
# Header validation
# ============================================================

def inspect_test_header(
    path,
):
    """
    Validate required features and determine whether a real
    label column exists.
    """

    header = list(
        pd.read_csv(
            path,
            nrows=0,
        ).columns
    )

    missing = [

        feature

        for feature
        in FEATURES

        if feature not in header
    ]

    if missing:

        raise ValueError(
            f"{path.name} is missing ML features:\n"
            + "\n".join(
                f"  - {feature}"
                for feature
                in missing
            )
        )

    has_label = (
        LABEL_COLUMN
        in header
    )

    return has_label


# ============================================================
# Evaluate one file
# ============================================================

def evaluate_file(
    path,
    family,
    evaluation_type,
    model_a,
    model_b,
):
    """
    Evaluate both models on one test flow CSV.

    When a true binary label column exists:
        true Attack Recall/FNR is calculated.

    When no label exists:
        Scenario Detection Rate is calculated instead.
    """

    has_label = (
        inspect_test_header(
            path
        )
    )

    usecols = list(
        FEATURES
    )

    if has_label:

        usecols.append(
            LABEL_COLUMN
        )

    dtype_map = {

        feature:
            np.float32

        for feature
        in FEATURES
    }

    if has_label:

        dtype_map[
            LABEL_COLUMN
        ] = np.float32

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    total_rows = 0

    model_a_attack = 0
    model_a_benign = 0

    model_b_attack = 0
    model_b_benign = 0

    model_a_probability_sum = 0.0
    model_b_probability_sum = 0.0

    # True-label counters, only used when labels exist.
    labeled_attack_rows = 0
    labeled_benign_rows = 0

    model_a_tp = 0
    model_a_fn = 0

    model_b_tp = 0
    model_b_fn = 0

    # --------------------------------------------------------
    # Read test file in chunks
    # --------------------------------------------------------

    reader = pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtype_map,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk in reader:

        X = (
            chunk[
                FEATURES
            ]
            .astype(
                np.float32
            )
        )

        # ----------------------------------------------------
        # NaN allowed, Inf rejected.
        # ----------------------------------------------------

        values = X.to_numpy(
            copy=False
        )

        if np.isinf(
            values
        ).any():

            raise ValueError(
                f"Infinite feature value detected "
                f"in {path.name}."
            )

        # ----------------------------------------------------
        # Model A
        # ----------------------------------------------------

        pred_a = (
            model_a.predict(
                X
            )
        )

        prob_a = (
            get_attack_probability(
                model_a,
                X,
            )
        )

        # ----------------------------------------------------
        # Model B
        # ----------------------------------------------------

        pred_b = (
            model_b.predict(
                X
            )
        )

        prob_b = (
            get_attack_probability(
                model_b,
                X,
            )
        )

        # ----------------------------------------------------
        # Prediction counts
        # ----------------------------------------------------

        rows = len(
            chunk
        )

        total_rows += rows

        a_attack_count = int(
            np.sum(
                pred_a
                == ATTACK_LABEL
            )
        )

        b_attack_count = int(
            np.sum(
                pred_b
                == ATTACK_LABEL
            )
        )

        model_a_attack += (
            a_attack_count
        )

        model_a_benign += (
            rows
            - a_attack_count
        )

        model_b_attack += (
            b_attack_count
        )

        model_b_benign += (
            rows
            - b_attack_count
        )

        model_a_probability_sum += float(
            np.sum(
                prob_a,
                dtype=np.float64,
            )
        )

        model_b_probability_sum += float(
            np.sum(
                prob_b,
                dtype=np.float64,
            )
        )

        # ----------------------------------------------------
        # True-label metrics, if labels are available.
        # ----------------------------------------------------

        if has_label:

            labels = pd.to_numeric(
                chunk[
                    LABEL_COLUMN
                ],
                errors="raise",
            )

            if labels.isna().any():

                raise ValueError(
                    f"Missing labels detected "
                    f"in {path.name}."
                )

            if not labels.isin(
                [
                    BENIGN_LABEL,
                    ATTACK_LABEL,
                ]
            ).all():

                bad = sorted(
                    labels[
                        ~labels.isin(
                            [
                                BENIGN_LABEL,
                                ATTACK_LABEL,
                            ]
                        )
                    ]
                    .unique()
                    .tolist()
                )

                raise ValueError(
                    f"Unexpected labels in "
                    f"{path.name}: {bad}"
                )

            y = labels.to_numpy(
                dtype=np.int8
            )

            attack_mask = (
                y
                == ATTACK_LABEL
            )

            benign_mask = (
                y
                == BENIGN_LABEL
            )

            attack_count = int(
                attack_mask.sum()
            )

            benign_count = int(
                benign_mask.sum()
            )

            labeled_attack_rows += (
                attack_count
            )

            labeled_benign_rows += (
                benign_count
            )

            model_a_tp += int(
                np.sum(
                    (
                        pred_a
                        == ATTACK_LABEL
                    )
                    & attack_mask
                )
            )

            model_a_fn += int(
                np.sum(
                    (
                        pred_a
                        == BENIGN_LABEL
                    )
                    & attack_mask
                )
            )

            model_b_tp += int(
                np.sum(
                    (
                        pred_b
                        == ATTACK_LABEL
                    )
                    & attack_mask
                )
            )

            model_b_fn += int(
                np.sum(
                    (
                        pred_b
                        == BENIGN_LABEL
                    )
                    & attack_mask
                )
            )

        if (
            total_rows
            % 500_000
            < rows
        ):

            print(
                f"      Scanned "
                f"{total_rows:,} rows...",
                flush=True,
            )

    if total_rows == 0:

        raise ValueError(
            f"No rows found in "
            f"{path.name}."
        )

    # --------------------------------------------------------
    # Scenario-level detection rates
    # --------------------------------------------------------

    detection_a = (
        model_a_attack
        / total_rows
    )

    detection_b = (
        model_b_attack
        / total_rows
    )

    mean_probability_a = (
        model_a_probability_sum
        / total_rows
    )

    mean_probability_b = (
        model_b_probability_sum
        / total_rows
    )

    # --------------------------------------------------------
    # True Recall/FNR only if attack labels actually exist.
    # --------------------------------------------------------

    if (
        has_label
        and labeled_attack_rows > 0
    ):

        recall_a = (
            model_a_tp
            / labeled_attack_rows
        )

        fnr_a = (
            model_a_fn
            / labeled_attack_rows
        )

        recall_b = (
            model_b_tp
            / labeled_attack_rows
        )

        fnr_b = (
            model_b_fn
            / labeled_attack_rows
        )

    else:

        recall_a = np.nan
        fnr_a = np.nan

        recall_b = np.nan
        fnr_b = np.nan

    return {

        "family":
            family,

        "evaluation_type":
            evaluation_type,

        "source_file":
            path.name,

        "has_ground_truth_label":
            has_label,

        "total_rows":
            total_rows,

        "ground_truth_attack_rows":
            (
                labeled_attack_rows
                if has_label
                else np.nan
            ),

        "ground_truth_benign_rows":
            (
                labeled_benign_rows
                if has_label
                else np.nan
            ),

        # ----------------------------------------------------
        # Model A
        # ----------------------------------------------------

        "model_a_attack_predictions":
            model_a_attack,

        "model_a_benign_predictions":
            model_a_benign,

        "model_a_scenario_detection_rate":
            detection_a,

        "model_a_mean_attack_probability":
            mean_probability_a,

        "model_a_true_attack_recall":
            recall_a,

        "model_a_true_attack_fnr":
            fnr_a,

        # ----------------------------------------------------
        # Model B
        # ----------------------------------------------------

        "model_b_attack_predictions":
            model_b_attack,

        "model_b_benign_predictions":
            model_b_benign,

        "model_b_scenario_detection_rate":
            detection_b,

        "model_b_mean_attack_probability":
            mean_probability_b,

        "model_b_true_attack_recall":
            recall_b,

        "model_b_true_attack_fnr":
            fnr_b,

        # ----------------------------------------------------
        # Difference
        # ----------------------------------------------------

        "detection_rate_change_b_minus_a":
            (
                detection_b
                - detection_a
            ),

        "mean_attack_probability_change_b_minus_a":
            (
                mean_probability_b
                - mean_probability_a
            ),
    }


# ============================================================
# Aggregate family results
# ============================================================

def aggregate_family_results(
    file_results,
):
    """
    Aggregate file-level results by test family.

    Uses row-weighted counts, not simple averaging of file rates.
    """

    family_rows = []

    grouped = (
        file_results
        .groupby(
            [
                "family",
                "evaluation_type",
            ],
            sort=False,
        )
    )

    for (
        family,
        evaluation_type,
    ), group in grouped:

        total_rows = int(
            group[
                "total_rows"
            ].sum()
        )

        a_attack = int(
            group[
                "model_a_attack_predictions"
            ].sum()
        )

        b_attack = int(
            group[
                "model_b_attack_predictions"
            ].sum()
        )

        a_benign = int(
            group[
                "model_a_benign_predictions"
            ].sum()
        )

        b_benign = int(
            group[
                "model_b_benign_predictions"
            ].sum()
        )

        detection_a = (
            a_attack
            / total_rows
        )

        detection_b = (
            b_attack
            / total_rows
        )

        # ----------------------------------------------------
        # Weighted mean attack probability
        # ----------------------------------------------------

        probability_a = float(
            (
                group[
                    "model_a_mean_attack_probability"
                ]
                * group[
                    "total_rows"
                ]
            ).sum()
            / total_rows
        )

        probability_b = float(
            (
                group[
                    "model_b_mean_attack_probability"
                ]
                * group[
                    "total_rows"
                ]
            ).sum()
            / total_rows
        )

        # ----------------------------------------------------
        # Ground-truth attack recall if labels exist.
        # ----------------------------------------------------

        labeled_group = group[
            group[
                "has_ground_truth_label"
            ]
            == True
        ]

        true_attack_rows = int(
            labeled_group[
                "ground_truth_attack_rows"
            ]
            .fillna(0)
            .sum()
        )

        if true_attack_rows > 0:

            # Recover TP from recall * support.
            a_tp = float(
                (
                    labeled_group[
                        "model_a_true_attack_recall"
                    ]
                    * labeled_group[
                        "ground_truth_attack_rows"
                    ]
                )
                .fillna(0)
                .sum()
            )

            b_tp = float(
                (
                    labeled_group[
                        "model_b_true_attack_recall"
                    ]
                    * labeled_group[
                        "ground_truth_attack_rows"
                    ]
                )
                .fillna(0)
                .sum()
            )

            recall_a = (
                a_tp
                / true_attack_rows
            )

            recall_b = (
                b_tp
                / true_attack_rows
            )

            fnr_a = (
                1.0
                - recall_a
            )

            fnr_b = (
                1.0
                - recall_b
            )

        else:

            recall_a = np.nan
            recall_b = np.nan

            fnr_a = np.nan
            fnr_b = np.nan

        family_rows.append({

            "family":
                family,

            "evaluation_type":
                evaluation_type,

            "files":
                len(
                    group
                ),

            "total_rows":
                total_rows,

            "model_a_attack_predictions":
                a_attack,

            "model_a_benign_predictions":
                a_benign,

            "model_a_scenario_detection_rate":
                detection_a,

            "model_a_mean_attack_probability":
                probability_a,

            "model_b_attack_predictions":
                b_attack,

            "model_b_benign_predictions":
                b_benign,

            "model_b_scenario_detection_rate":
                detection_b,

            "model_b_mean_attack_probability":
                probability_b,

            "detection_rate_change_b_minus_a":
                (
                    detection_b
                    - detection_a
                ),

            "true_attack_rows_if_available":
                (
                    true_attack_rows
                    if true_attack_rows > 0
                    else np.nan
                ),

            "model_a_true_attack_recall":
                recall_a,

            "model_a_true_attack_fnr":
                fnr_a,

            "model_b_true_attack_recall":
                recall_b,

            "model_b_true_attack_fnr":
                fnr_b,
        })

    return pd.DataFrame(
        family_rows
    )


# ============================================================
# Report writer
# ============================================================

def save_report(
    file_results,
    family_results,
):
    """
    Save human-readable comparison report.
    """

    with open(
        REPORT_OUT,
        "w",
        encoding="utf-8",
    ) as report:

        report.write(
            "NetworkIDS - External Test Model Comparison\n"
        )

        report.write(
            "=" * 75
            + "\n\n"
        )

        report.write(
            "Models\n"
        )

        report.write(
            "-" * 75
            + "\n"
        )

        report.write(
            f"Model A: {MODEL_A_FILE}\n"
        )

        report.write(
            f"Model B: {MODEL_B_FILE}\n"
        )

        report.write(
            f"Features: {len(FEATURES)}\n"
        )

        report.write(
            "NaN policy: retained; no imputation\n\n"
        )

        report.write(
            "Metric Interpretation\n"
        )

        report.write(
            "-" * 75
            + "\n"
        )

        report.write(
            "Scenario Detection Rate = predicted Attack flows "
            "/ total flows in the scenario capture.\n"
        )

        report.write(
            "Scenario Detection Rate is NOT claimed as strict "
            "per-flow Recall when no verified label column is "
            "available, because scenario captures may include "
            "background traffic.\n"
        )

        report.write(
            "True Attack Recall/FNR are only reported when a "
            "valid ground-truth label column exists.\n\n"
        )

        report.write(
            "Family-Level Results\n"
        )

        report.write(
            "-" * 75
            + "\n"
        )

        for _, row in (
            family_results.iterrows()
        ):

            report.write(
                f"\n{row['family']}\n"
            )

            report.write(
                f"  Evaluation type : "
                f"{row['evaluation_type']}\n"
            )

            report.write(
                f"  Files           : "
                f"{int(row['files'])}\n"
            )

            report.write(
                f"  Total flows     : "
                f"{int(row['total_rows']):,}\n"
            )

            report.write(
                f"  Model A detection rate : "
                f"{row['model_a_scenario_detection_rate']:.6f} "
                f"({row['model_a_scenario_detection_rate'] * 100:.2f}%)\n"
            )

            report.write(
                f"  Model B detection rate : "
                f"{row['model_b_scenario_detection_rate']:.6f} "
                f"({row['model_b_scenario_detection_rate'] * 100:.2f}%)\n"
            )

            report.write(
                f"  Change B - A            : "
                f"{row['detection_rate_change_b_minus_a']:+.6f} "
                f"({row['detection_rate_change_b_minus_a'] * 100:+.2f} pp)\n"
            )

            report.write(
                f"  Model A mean P(Attack)  : "
                f"{row['model_a_mean_attack_probability']:.6f}\n"
            )

            report.write(
                f"  Model B mean P(Attack)  : "
                f"{row['model_b_mean_attack_probability']:.6f}\n"
            )

            if not pd.isna(
                row[
                    "model_a_true_attack_recall"
                ]
            ):

                report.write(
                    f"  Model A true Recall     : "
                    f"{row['model_a_true_attack_recall']:.6f}\n"
                )

                report.write(
                    f"  Model B true Recall     : "
                    f"{row['model_b_true_attack_recall']:.6f}\n"
                )

        report.write(
            "\n\nFile-Level Results\n"
        )

        report.write(
            "-" * 75
            + "\n"
        )

        for _, row in (
            file_results.iterrows()
        ):

            report.write(
                f"\n{row['source_file']}\n"
            )

            report.write(
                f"  Family          : "
                f"{row['family']}\n"
            )

            report.write(
                f"  Total flows     : "
                f"{int(row['total_rows']):,}\n"
            )

            report.write(
                f"  Model A detected: "
                f"{int(row['model_a_attack_predictions']):,} "
                f"({row['model_a_scenario_detection_rate'] * 100:.2f}%)\n"
            )

            report.write(
                f"  Model B detected: "
                f"{int(row['model_b_attack_predictions']):,} "
                f"({row['model_b_scenario_detection_rate'] * 100:.2f}%)\n"
            )

            report.write(
                f"  Change B - A    : "
                f"{row['detection_rate_change_b_minus_a'] * 100:+.2f} pp\n"
            )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 75
    )

    print(
        "NetworkIDS - EXTERNAL TEST MODEL COMPARISON"
    )

    print(
        "=" * 75
    )

    # ========================================================
    # Safety checks
    # ========================================================

    if not TEST_DIR.is_dir():

        raise FileNotFoundError(
            f"Test directory not found:\n"
            f"{TEST_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_files = [
        FILE_RESULTS_OUT,
        FAMILY_RESULTS_OUT,
        REPORT_OUT,
    ]

    existing = [

        path

        for path in output_files

        if path.exists()
    ]

    if existing:

        raise FileExistsError(
            "Refusing to overwrite existing evaluation outputs:\n"
            + "\n".join(
                f"  - {path}"
                for path
                in existing
            )
        )

    config.validate_ml_features()

    if len(FEATURES) != 42:

        raise ValueError(
            f"Expected current model schema "
            f"to contain 42 features, "
            f"found {len(FEATURES)}."
        )

    # ========================================================
    # Load models
    # ========================================================

    print_section(
        "[1/4] Loading models"
    )

    model_a = load_and_validate_model(
        MODEL_A_FILE,
        "Model A",
    )

    model_b = load_and_validate_model(
        MODEL_B_FILE,
        "Model B",
    )

    print(
        f"\nFeatures validated: "
        f"{len(FEATURES)}"
    )

    # ========================================================
    # Discover tests
    # ========================================================

    print_section(
        "[2/4] Discovering external test datasets"
    )

    tests = (
        discover_test_files()
    )

    print(
        f"Found {len(tests)} test flow files."
    )

    family_counts = {}

    for item in tests:

        family = item[
            "family"
        ]

        family_counts[
            family
        ] = (
            family_counts.get(
                family,
                0,
            )
            + 1
        )

    for family, count in (
        family_counts.items()
    ):

        print(
            f"  {family:<35}: "
            f"{count} file(s)"
        )

    # ========================================================
    # Evaluate
    # ========================================================

    print_section(
        "[3/4] Evaluating Model A vs Model B"
    )

    results = []

    for number, item in enumerate(
        tests,
        start=1,
    ):

        path = item[
            "path"
        ]

        family = item[
            "family"
        ]

        evaluation_type = item[
            "evaluation_type"
        ]

        print()

        print(
            f"[{number}/{len(tests)}] "
            f"{path.name}"
        )

        print(
            f"  Family: "
            f"{family}"
        )

        result = evaluate_file(

            path=path,

            family=family,

            evaluation_type=(
                evaluation_type
            ),

            model_a=model_a,

            model_b=model_b,
        )

        results.append(
            result
        )

        print(
            f"  Total flows : "
            f"{result['total_rows']:,}"
        )

        print(
            f"  Model A     : "
            f"{result['model_a_scenario_detection_rate'] * 100:.2f}% "
            f"detected as Attack"
        )

        print(
            f"  Model B     : "
            f"{result['model_b_scenario_detection_rate'] * 100:.2f}% "
            f"detected as Attack"
        )

        print(
            f"  Change B-A  : "
            f"{result['detection_rate_change_b_minus_a'] * 100:+.2f} "
            "percentage points"
        )

    # ========================================================
    # Save results
    # ========================================================

    print_section(
        "[4/4] Saving results"
    )

    file_results = (
        pd.DataFrame(
            results
        )
    )

    family_results = (
        aggregate_family_results(
            file_results
        )
    )

    file_results.to_csv(
        FILE_RESULTS_OUT,
        index=False,
    )

    family_results.to_csv(
        FAMILY_RESULTS_OUT,
        index=False,
    )

    save_report(
        file_results,
        family_results,
    )

    print(
        f"File-level CSV : "
        f"{FILE_RESULTS_OUT}"
    )

    print(
        f"Family CSV     : "
        f"{FAMILY_RESULTS_OUT}"
    )

    print(
        f"Report         : "
        f"{REPORT_OUT}"
    )

    # ========================================================
    # Final summary
    # ========================================================

    print_section(
        "EXTERNAL TEST COMPARISON COMPLETE"
    )

    for _, row in (
        family_results.iterrows()
    ):

        print(
            f"{row['family']}"
        )

        print(
            f"  Flows   : "
            f"{int(row['total_rows']):,}"
        )

        print(
            f"  Model A : "
            f"{row['model_a_scenario_detection_rate'] * 100:.2f}%"
        )

        print(
            f"  Model B : "
            f"{row['model_b_scenario_detection_rate'] * 100:.2f}%"
        )

        print(
            f"  B - A   : "
            f"{row['detection_rate_change_b_minus_a'] * 100:+.2f} pp"
        )

        print()

    print(
        "Important: for files without verified ground-truth labels, "
        "the percentages above are Scenario Detection Rates, "
        "not strict per-flow Recall."
    )

    print(
        "No model, dataset, or config.py file was modified."
    )


if __name__ == "__main__":
    main()