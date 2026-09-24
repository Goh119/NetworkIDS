"""
NetworkIDS - SHAP explainability for the trained Random Forest.

Purpose:
    Explain the predictions of the trained Random Forest classifier
    using SHAP (SHapley Additive exPlanations).

Input:
    data/processed/split/train_dataset.csv
    data/processed/split/test_dataset.csv
    models/random_forest.pkl

Outputs:
    models/shap/
        global_importance_bar.png
        global_importance.csv
        global_beeswarm.png
        waterfall_true_positive.png
        waterfall_true_negative.png
        waterfall_false_positive.png
        waterfall_false_negative.png
        shap_values_attack.npy
        shap_sample_features.csv
        shap_sample_metadata.csv
        shap_explanation_summary.json

============================================================
Missing-value policy
============================================================

The Random Forest model was trained using scikit-learn's native
missing-value support.

Therefore:

    NaN values are retained as-is.

No placeholder values such as -1 or 0 are introduced to represent
missing values.

The SHAP explanation is first performed directly on the original
NaN-containing feature data.

An additivity consistency check is then performed:

    base_value + sum(SHAP values)
        ≈
    model predicted probability for the Attack class

A small numerical tolerance is allowed.

If the raw-NaN SHAP explanation fails this consistency check,
the script STOPS instead of silently changing the feature
representation.

This is intentional.

The project uses the same feature representation for:

    Model training
        ↓
    Model prediction
        ↓
    Model explanation

No median imputation is performed in this script.

============================================================
SHAP interpretation
============================================================

The project uses binary classification:

    Benign = 0
    Attack = 1

SHAP values are therefore interpreted for:

    Attack class (class = 1)

Positive SHAP value:
    pushes the model prediction toward Attack.

Negative SHAP value:
    pushes the model prediction toward Benign.

Mean absolute SHAP value:
    measures the average magnitude of a feature's contribution
    across the SHAP sample.

============================================================
Sampling
============================================================

The complete test set is NOT used for SHAP computation.

A reproducible sample is selected from the held-out test set:

    Default sample size = 5,000
    Random state = 42

The sample is used only for model explanation.

It is NOT used to train or modify the Random Forest model.


Usage:
    python src/shap_explain.py
"""

import argparse
import json
import sys
from pathlib import Path

import joblib

import matplotlib

# Use a non-interactive backend because plots are saved directly
# to PNG files.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


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

OUTPUT_DIR = (
    config.MODELS_DIR
    / "shap"
)

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

DEFAULT_SAMPLE_SIZE = 5000

RANDOM_STATE = 42

# Project label definition:
# Benign = 0
# Attack = 1
ATTACK_CLASS_INDEX = 1


# ============================================================
# Loaders
# ============================================================

def load_model():
    """
    Load the trained Random Forest model.
    """

    if not config.RANDOM_FOREST_MODEL_FILE.exists():
        raise FileNotFoundError(
            "Random Forest model not found:\n"
            f"{config.RANDOM_FOREST_MODEL_FILE}"
        )

    print(
        f"Loading model:\n"
        f"  {config.RANDOM_FOREST_MODEL_FILE}"
    )

    model = joblib.load(
        config.RANDOM_FOREST_MODEL_FILE
    )

    return model


def load_data():
    """
    Load the pre-split training and testing datasets.
    """

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training dataset not found:\n{TRAIN_FILE}"
        )

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Testing dataset not found:\n{TEST_FILE}"
        )

    train_df = pd.read_csv(
        TRAIN_FILE,
        low_memory=False
    )

    test_df = pd.read_csv(
        TEST_FILE,
        low_memory=False
    )

    print(
        f"  Train: {len(train_df):,} rows"
    )

    print(
        f"  Test : {len(test_df):,} rows"
    )

    return train_df, test_df


# ============================================================
# Validation
# ============================================================

def validate_features(df, name):
    """
    Validate that all 42 ML features are present.
    """

    missing_features = [
        feature
        for feature in config.ML_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"{name}: missing ML features:\n"
            f"{missing_features}"
        )

    if len(config.ML_FEATURES) != 42:
        raise ValueError(
            "Expected exactly 42 ML features, "
            f"but config contains "
            f"{len(config.ML_FEATURES)}."
        )


def validate_no_infinite_values(X, name):
    """
    Ensure that no infinite values exist.

    NaN values are allowed because they are legitimate
    missing values in the project's feature representation.
    """

    numeric_X = X.select_dtypes(
        include=[np.number]
    )

    if numeric_X.empty:
        return

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
            f"{inf_count:,} infinite feature values."
        )


def check_flow_id_leakage(train_df, test_df):
    """
    Check that no flow_id occurs in both train and test.
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
        "  [OK] Flow ID overlap: 0"
    )


# ============================================================
# SHAP output handling
# ============================================================

def extract_attack_values(shap_output):
    """
    Extract SHAP values for the Attack class.

    SHAP versions may represent binary classification output as:

        (samples, features, classes)

    or:

        (samples, features)

    This function normalises the output to:

        (samples, features)
    """

    values = np.asarray(
        shap_output.values
    )

    if values.ndim == 3:

        if values.shape[2] <= ATTACK_CLASS_INDEX:
            raise ValueError(
                "Attack class index is outside "
                f"SHAP output shape: {values.shape}"
            )

        values_attack = (
            values[:, :, ATTACK_CLASS_INDEX]
        )

    elif values.ndim == 2:

        values_attack = values

    else:

        raise ValueError(
            "Unexpected SHAP value shape: "
            f"{values.shape}"
        )

    return values_attack


def extract_attack_base_values(shap_output, n_samples):
    """
    Extract the SHAP base value for the Attack class.

    Returns one base value per sample.
    """

    base = np.asarray(
        shap_output.base_values
    )

    # --------------------------------------------------------
    # Case 1:
    # (samples, classes)
    # --------------------------------------------------------

    if base.ndim == 2:

        if base.shape[1] <= ATTACK_CLASS_INDEX:
            raise ValueError(
                "Attack class index is outside "
                f"base value shape: {base.shape}"
            )

        return base[
            :,
            ATTACK_CLASS_INDEX
        ]

    # --------------------------------------------------------
    # Case 2:
    # (samples,)
    # --------------------------------------------------------

    if base.ndim == 1:

        if base.shape[0] == n_samples:
            return base.astype(float)

        # Single base value
        if base.shape[0] == 1:
            return np.full(
                n_samples,
                float(base[0])
            )

    # --------------------------------------------------------
    # Case 3:
    # scalar
    # --------------------------------------------------------

    if base.ndim == 0:

        return np.full(
            n_samples,
            float(base)
        )

    raise ValueError(
        "Unexpected SHAP base value shape: "
        f"{base.shape}"
    )


# ============================================================
# Additivity consistency check
# ============================================================

def check_additivity(
    shap_values_attack,
    base_values_attack,
    model,
    X_data
):
    """
    Check whether:

        base value + sum(SHAP values)

    approximately equals:

        model.predict_proba(X)[:, Attack]

    Returns:

        passed
        max_discrepancy
        mean_discrepancy
    """

    # --------------------------------------------------------
    # Actual model probability
    # --------------------------------------------------------

    model_probability = (
        model.predict_proba(X_data)
        [:, ATTACK_CLASS_INDEX]
    )

    # --------------------------------------------------------
    # SHAP reconstruction
    # --------------------------------------------------------

    reconstructed_probability = (
        shap_values_attack.sum(axis=1)
        + base_values_attack
    )

    # --------------------------------------------------------
    # Difference
    # --------------------------------------------------------

    discrepancy = np.abs(
        reconstructed_probability
        - model_probability
    )

    max_discrepancy = float(
        discrepancy.max()
    )

    mean_discrepancy = float(
        discrepancy.mean()
    )

    # Numerical tolerance
    tolerance = 1e-3

    passed = (
        max_discrepancy < tolerance
    )

    return (
        passed,
        max_discrepancy,
        mean_discrepancy
    )


# ============================================================
# Build single-row explanation
# ============================================================

def build_row_explanation(
    shap_values_attack,
    base_values_attack,
    X_data,
    position
):
    """
    Create a shap.Explanation object for one flow.
    """

    return shap.Explanation(
        values=shap_values_attack[position],
        base_values=base_values_attack[position],
        data=X_data.iloc[position].values,
        feature_names=config.ML_FEATURES,
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate SHAP explanations for "
            "the trained Random Forest."
        )
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=(
            "Number of test rows to explain "
            "(default: 5000)."
        ),
    )

    args = parser.parse_args()

    if args.sample_size <= 0:
        raise ValueError(
            "--sample-size must be greater than 0."
        )

    # ========================================================
    # Header
    # ========================================================

    print("=" * 70)
    print("NetworkIDS - SHAP Explainability")
    print("=" * 70)

    print(
        f"SHAP version: {shap.__version__}"
    )

    # ========================================================
    # 1. Load model and data
    # ========================================================

    print(
        "\n[1/6] Loading model and datasets..."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model = load_model()

    train_df, test_df = load_data()

    # --------------------------------------------------------
    # Validate features
    # --------------------------------------------------------

    validate_features(
        train_df,
        "Train"
    )

    validate_features(
        test_df,
        "Test"
    )

    # --------------------------------------------------------
    # Leakage guard
    # --------------------------------------------------------

    check_flow_id_leakage(
        train_df,
        test_df
    )

    # --------------------------------------------------------
    # Select features
    # --------------------------------------------------------

    X_train = train_df[
        config.ML_FEATURES
    ].copy()

    X_test = test_df[
        config.ML_FEATURES
    ].copy()

    y_test = test_df[
        config.LABEL_COLUMN
    ].astype(int)

    validate_no_infinite_values(
        X_train,
        "Train"
    )

    validate_no_infinite_values(
        X_test,
        "Test"
    )

    print(
        f"  Features: {len(config.ML_FEATURES)}"
    )

    # ========================================================
    # 2. Sample test rows
    # ========================================================

    print(
        "\n[2/6] Sampling test rows..."
    )

    sample_size = min(
        args.sample_size,
        len(X_test)
    )

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    sample_idx = rng.choice(
        X_test.index.to_numpy(),
        size=sample_size,
        replace=False
    )

    # Preserve original test data values.
    X_sample = (
        X_test
        .loc[sample_idx]
        .reset_index(drop=True)
        .copy()
    )

    y_sample = (
        y_test
        .loc[sample_idx]
        .reset_index(drop=True)
        .copy()
    )

    nan_cells_in_sample = int(
        X_sample.isna()
        .sum()
        .sum()
    )

    print(
        f"  Sample size: {sample_size:,}"
    )

    print(
        f"  NaN cells in sample: "
        f"{nan_cells_in_sample:,}"
    )

    print(
        f"  Random state: {RANDOM_STATE}"
    )

    # ========================================================
    # 3. SHAP explanation on original NaN data
    # ========================================================

    print(
        "\n[3/6] Building SHAP TreeExplainer..."
    )

    print(
        "  Explaining the original "
        "NaN-containing feature data."
    )

    explainer = shap.TreeExplainer(
        model
    )

    print(
        "  TreeExplainer created."
    )

    print(
        "\n  Computing SHAP values..."
    )

    try:

        shap_output = explainer(
            X_sample
        )

    except Exception as exc:

        print("\n" + "=" * 70)
        print("SHAP EXPLANATION FAILED")
        print("=" * 70)

        print(
            "TreeExplainer could not process "
            "the original NaN-containing data."
        )

        print(
            "\nError:"
        )

        print(
            str(exc)
        )

        print(
            "\nNo imputation was performed."
        )

        print(
            "No SHAP results were generated."
        )

        print(
            "\nThe model itself was NOT modified."
        )

        print("=" * 70)

        raise RuntimeError(
            "Raw-NaN SHAP explanation failed. "
            "Investigate SHAP/model compatibility "
            "instead of changing the feature representation."
        ) from exc

    # --------------------------------------------------------
    # Extract Attack-class values
    # --------------------------------------------------------

    shap_values_attack = (
        extract_attack_values(
            shap_output
        )
    )

    base_values_attack = (
        extract_attack_base_values(
            shap_output,
            sample_size
        )
    )

    print(
        f"\n  SHAP matrix shape: "
        f"{shap_values_attack.shape}"
    )

    print(
        f"  Expected feature count: "
        f"{len(config.ML_FEATURES)}"
    )

    # --------------------------------------------------------
    # Validate shape
    # --------------------------------------------------------

    expected_shape = (
        sample_size,
        len(config.ML_FEATURES)
    )

    if shap_values_attack.shape != expected_shape:
        raise ValueError(
            "Unexpected SHAP matrix shape.\n"
            f"Expected: {expected_shape}\n"
            f"Actual:   {shap_values_attack.shape}"
        )

    # ========================================================
    # Additivity consistency check
    # ========================================================

    print(
        "\n  Checking SHAP additivity consistency..."
    )

    (
        additivity_passed,
        max_discrepancy,
        mean_discrepancy
    ) = check_additivity(
        shap_values_attack,
        base_values_attack,
        model,
        X_sample
    )

    print(
        f"  Max discrepancy : "
        f"{max_discrepancy:.8f}"
    )

    print(
        f"  Mean discrepancy: "
        f"{mean_discrepancy:.8f}"
    )

    if not additivity_passed:

        print("\n" + "=" * 70)
        print("SHAP ADDITIVITY CHECK FAILED")
        print("=" * 70)

        print(
            "The SHAP explanation could not reproduce "
            "the model's Attack probability within "
            "the configured tolerance."
        )

        print(
            f"\nMaximum discrepancy: "
            f"{max_discrepancy:.8f}"
        )

        print(
            "\nNo imputation was performed."
        )

        print(
            "No explanation plots were generated."
        )

        print(
            "\nInvestigate SHAP / scikit-learn "
            "compatibility before continuing."
        )

        print("=" * 70)

        raise RuntimeError(
            "SHAP additivity consistency check failed."
        )

    print(
        "  [OK] Additivity consistency check passed."
    )

    # ========================================================
    # 4. Global SHAP plots
    # ========================================================

    print(
        "\n[4/6] Generating global SHAP explanations..."
    )

    # --------------------------------------------------------
    # Mean absolute SHAP
    # --------------------------------------------------------

    mean_abs_shap = (
        np.abs(
            shap_values_attack
        )
        .mean(axis=0)
    )

    importance = (
        pd.Series(
            mean_abs_shap,
            index=config.ML_FEATURES
        )
        .sort_values(
            ascending=False
        )
    )

    # --------------------------------------------------------
    # Print top 15
    # --------------------------------------------------------

    print(
        "\nTop 15 features by mean |SHAP|:"
    )

    for rank, (
        feature,
        value
    ) in enumerate(
        importance.head(15).items(),
        start=1
    ):

        print(
            f"  {rank:>2}. "
            f"{feature:<30} "
            f"{value:.6f}"
        )

    # --------------------------------------------------------
    # Save importance CSV
    # --------------------------------------------------------

    importance_df = pd.DataFrame({
        "rank": range(
            1,
            len(importance) + 1
        ),
        "feature": importance.index,
        "mean_abs_shap": importance.values,
    })

    importance_file = (
        OUTPUT_DIR
        / "global_importance.csv"
    )

    importance_df.to_csv(
        importance_file,
        index=False
    )

    print(
        f"\n  Importance table saved:\n"
        f"  {importance_file}"
    )

    # --------------------------------------------------------
    # Global bar chart
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 8)
    )

    (
        importance
        .head(20)
        .sort_values()
        .plot(
            kind="barh"
        )
    )

    plt.xlabel(
        "Mean |SHAP value| "
        "(impact on Attack prediction)"
    )

    plt.title(
        "Global Feature Importance (SHAP)"
    )

    plt.tight_layout()

    bar_file = (
        OUTPUT_DIR
        / "global_importance_bar.png"
    )

    plt.savefig(
        bar_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"  Global bar chart saved:\n"
        f"  {bar_file}"
    )

    # --------------------------------------------------------
    # Global beeswarm
    # --------------------------------------------------------

    plt.figure()

    shap.summary_plot(
        shap_values_attack,
        X_sample,
        show=False,
        max_display=20,
    )

    plt.tight_layout()

    beeswarm_file = (
        OUTPUT_DIR
        / "global_beeswarm.png"
    )

    plt.savefig(
        beeswarm_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"  Global beeswarm saved:\n"
        f"  {beeswarm_file}"
    )

    # ========================================================
    # 5. Per-prediction waterfall plots
    # ========================================================

    print(
        "\n[5/6] Generating TP / TN / FP / FN "
        "waterfall plots..."
    )

    # IMPORTANT:
    #
    # Predictions are generated from the ORIGINAL
    # NaN-containing sample.
    #
    # This preserves consistency with the actual model
    # training and test evaluation pipeline.

    y_pred_sample = model.predict(
        X_sample
    )

    # --------------------------------------------------------
    # Verify prediction probabilities
    # --------------------------------------------------------

    y_proba_sample = (
        model.predict_proba(
            X_sample
        )[:, ATTACK_CLASS_INDEX]
    )

    # --------------------------------------------------------
    # Define four prediction cases
    # --------------------------------------------------------

    cases = {

        "true_positive": (
            (y_sample.to_numpy() == 1)
            &
            (y_pred_sample == 1)
        ),

        "true_negative": (
            (y_sample.to_numpy() == 0)
            &
            (y_pred_sample == 0)
        ),

        "false_positive": (
            (y_sample.to_numpy() == 0)
            &
            (y_pred_sample == 1)
        ),

        "false_negative": (
            (y_sample.to_numpy() == 1)
            &
            (y_pred_sample == 0)
        ),
    }

    case_indices = {}

    # --------------------------------------------------------
    # Generate waterfall for each case
    # --------------------------------------------------------

    for case_name, mask in cases.items():

        positions = np.where(
            mask
        )[0]

        if len(positions) == 0:

            print(
                f"  [skip] No "
                f"{case_name.replace('_', ' ')} "
                f"example in sample."
            )

            case_indices[
                case_name
            ] = None

            continue

        # ----------------------------------------------------
        # Select first matching example
        # ----------------------------------------------------

        position = int(
            positions[0]
        )

        case_indices[
            case_name
        ] = position

        # ----------------------------------------------------
        # Build SHAP explanation
        # ----------------------------------------------------

        explanation = (
            build_row_explanation(
                shap_values_attack,
                base_values_attack,
                X_sample,
                position
            )
        )

        # ----------------------------------------------------
        # Generate waterfall
        # ----------------------------------------------------

        plt.figure()

        shap.plots.waterfall(
            explanation,
            max_display=15,
            show=False
        )

        plt.title(
            f"{case_name.replace('_', ' ').title()} "
            f"(sample row {position})"
        )

        plt.tight_layout()

        waterfall_file = (
            OUTPUT_DIR
            / f"waterfall_{case_name}.png"
        )

        plt.savefig(
            waterfall_file,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        actual_label = int(
            y_sample.iloc[position]
        )

        predicted_label = int(
            y_pred_sample[position]
        )

        attack_probability = float(
            y_proba_sample[position]
        )

        print(
            f"  [OK] "
            f"{case_name}: "
            f"sample row {position}, "
            f"actual={actual_label}, "
            f"predicted={predicted_label}, "
            f"attack_probability={attack_probability:.4f}"
        )

    # ========================================================
    # 6. Save raw outputs and summary
    # ========================================================

    print(
        "\n[6/6] Saving SHAP outputs and summary..."
    )

    # --------------------------------------------------------
    # Save SHAP values
    # --------------------------------------------------------

    shap_values_file = (
        OUTPUT_DIR
        / "shap_values_attack.npy"
    )

    np.save(
        shap_values_file,
        shap_values_attack
    )

    print(
        f"  SHAP values saved:\n"
        f"  {shap_values_file}"
    )

    # --------------------------------------------------------
    # Save feature sample
    # --------------------------------------------------------

    sample_features_file = (
        OUTPUT_DIR
        / "shap_sample_features.csv"
    )

    X_sample.to_csv(
        sample_features_file,
        index=True
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata_columns = [
        "flow_id",
        "timestamp",
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "flow_packet_count",
        "flow_duration",
        "label",
        "original_label",
    ]

    metadata_columns = [
        column
        for column in metadata_columns
        if column in test_df.columns
    ]

    sample_metadata = (
        test_df
        .loc[
            sample_idx,
            metadata_columns
        ]
        .reset_index(drop=True)
        .copy()
    )

    # Add actual model prediction information.
    sample_metadata[
        "predicted_label"
    ] = y_pred_sample

    sample_metadata[
        "attack_probability"
    ] = y_proba_sample

    sample_metadata_file = (
        OUTPUT_DIR
        / "shap_sample_metadata.csv"
    )

    sample_metadata.to_csv(
        sample_metadata_file,
        index=True
    )

    # --------------------------------------------------------
    # Build case details
    # --------------------------------------------------------

    case_details = {}

    for case_name, position in case_indices.items():

        if position is None:

            case_details[
                case_name
            ] = None

            continue

        case_details[
            case_name
        ] = {
            "sample_position": int(
                position
            ),
            "flow_id": str(
                sample_metadata.iloc[
                    position
                ]["flow_id"]
            )
            if "flow_id"
            in sample_metadata.columns
            else None,
            "actual_label": int(
                y_sample.iloc[position]
            ),
            "predicted_label": int(
                y_pred_sample[position]
            ),
            "attack_probability": float(
                y_proba_sample[position]
            ),
        }

    # --------------------------------------------------------
    # Summary JSON
    # --------------------------------------------------------

    summary = {

        "model": "RandomForestClassifier",

        "model_path": str(
            config.RANDOM_FOREST_MODEL_FILE
        ),

        "shap_version": shap.__version__,

        "sample_size": int(
            sample_size
        ),

        "random_state": int(
            RANDOM_STATE
        ),

        "n_features": int(
            len(config.ML_FEATURES)
        ),

        "class_explained": "Attack",

        "attack_class_index": int(
            ATTACK_CLASS_INDEX
        ),

        "nan_cells_in_sample": int(
            nan_cells_in_sample
        ),

        "nan_policy": (
            "NaN retained as-is; "
            "no imputation"
        ),

        "imputation_used": False,

        "shap_matrix_shape": [
            int(value)
            for value in shap_values_attack.shape
        ],

        "additivity_check": {

            "passed": bool(
                additivity_passed
            ),

            "tolerance": 1e-3,

            "max_discrepancy": float(
                max_discrepancy
            ),

            "mean_discrepancy": float(
                mean_discrepancy
            ),
        },

        "top_10_features": {
            str(feature): float(value)
            for feature, value
            in importance.head(10).items()
        },

        "case_examples": case_details,

        "output_files": sorted(
            path.name
            for path in OUTPUT_DIR.glob("*")
        ),
    }

    summary_file = (
        OUTPUT_DIR
        / "shap_explanation_summary.json"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=2
        )

    # ========================================================
    # Final
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "SHAP EXPLAINABILITY COMPLETE - PASS"
    )

    print(
        "=" * 70
    )

    print(
        f"  SHAP version       : "
        f"{shap.__version__}"
    )

    print(
        f"  Test rows          : "
        f"{len(test_df):,}"
    )

    print(
        f"  SHAP sample        : "
        f"{sample_size:,}"
    )

    print(
        f"  Features explained : "
        f"{len(config.ML_FEATURES)}"
    )

    print(
        f"  NaN cells          : "
        f"{nan_cells_in_sample:,}"
    )

    print(
        f"  Imputation used    : "
        f"False"
    )

    print(
        f"  Additivity check   : "
        f"PASSED"
    )

    print(
        f"  Max discrepancy    : "
        f"{max_discrepancy:.8f}"
    )

    print(
        f"  Output directory   : "
        f"{OUTPUT_DIR}"
    )

    print(
        f"  Summary file       : "
        f"{summary_file}"
    )

    print(
        "=" * 70
    )

if __name__ == "__main__":
    main()