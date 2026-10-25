"""
shap_explain.py
---------------
NetworkIDS - SHAP explainability for the primary trained
Random Forest classifier.

============================================================
TEST SCOPE
============================================================

The SHAP test scope is NOT limited to config.TEST_DATASET_FILE.

Instead, this script uses the project's unseen test scope:

    data/test/

All flow-level CSV files matching:

    *_flow.csv

are included.

In addition:

    data/test/test_multiday.csv

is included through:

    config.TEST_ADDITIONAL_FLOW_FILES

IMPORTANT:

A filename containing "train" is NOT excluded.

The project defines the entire data/test directory as
unseen test data.

Packet-level files such as:

    *_packets.csv

are automatically excluded because SHAP operates on the
flow-level Random Forest feature vector.

============================================================
MODEL
============================================================

Primary model:

    config.RANDOM_FOREST_MODEL_FILE

The SHAP feature set is defined entirely by:

    config.ML_FEATURES

The number of ML features is dynamic and is NOT hard-coded.

============================================================
CLASSIFICATION
============================================================

The Random Forest performs binary classification using:

    config.LABEL_BENIGN
    config.LABEL_ATTACK

The Attack probability column is determined from:

    model.classes_

The script does NOT assume that Attack is always at
probability column index 1.

============================================================
GROUND-TRUTH LABEL POLICY
============================================================

The data/test directory contains unseen inference datasets.

Some flow CSV files contain:

    label

while others do not have ground-truth labels.

Therefore:

    label is OPTIONAL.

SHAP does NOT require ground truth in order to explain a
Random Forest prediction.

If a dataset contains label:

    Actual label is preserved and can be shown in a local
    explanation.

If a dataset does not contain label:

    label = missing
    Actual = Unknown

Prediction, P(Attack), SHAP feature contributions, global
importance, beeswarm plots, and local waterfall explanations
still work normally.

============================================================
MISSING-VALUE POLICY
============================================================

NaN values are retained.

No:

    NaN -> 0

and no:

    median imputation

is performed.

SHAP receives the same feature representation used by the
Random Forest model.

============================================================
GLOBAL SHAP
============================================================

A reproducible sample is selected from the complete unseen
test scope.

Default:

    sample size = 500
    random state = 42

Global outputs:

    models/shap/global_importance.csv
    models/shap/global_importance_bar.png
    models/shap/global_beeswarm.png
    models/shap/shap_summary.json

============================================================
LOCAL SHAP
============================================================

Optional:

    python src/shap_explain.py --local-row 100

The row number refers to the zero-based row position in the
combined unseen test scope.

The local explanation preserves where available:

    dataset_file
    day
    flow_id
    timestamp
    src_ip
    dst_ip
    src_port
    dst_port
    label_available

These metadata fields are used only for investigation /
dashboard traceability.

They do NOT contribute additional features to the Random
Forest.

============================================================
USAGE
============================================================

Global SHAP:

    python src/shap_explain.py

Custom sample size:

    python src/shap_explain.py --sample-size 1000

Global + local:

    python src/shap_explain.py --local-row 100
"""

import argparse
import json
import sys
from pathlib import Path

import joblib

import matplotlib

# Non-interactive backend because plots are written directly
# to files.
matplotlib.use(
    "Agg"
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


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

OUTPUT_DIR = (
    config.MODELS_DIR
    / "shap"
)

DEFAULT_SAMPLE_SIZE = 500

RANDOM_STATE = 42

FEATURE_DTYPE = np.float32

ADDITIVITY_TOLERANCE = 1e-3


# ============================================================
# Local metadata
#
# These columns are required for normal flow traceability.
#
# They are NOT ML features.
# ============================================================

LOCAL_METADATA_COLUMNS = [
    "flow_id",
    "timestamp",
    "src_ip",
    "dst_ip",
]


# ============================================================
# Optional metadata
#
# test_multiday.csv contains "day", but most external unseen
# flow files do not.
#
# Therefore "day" must NOT be mandatory.
# ============================================================

OPTIONAL_METADATA_COLUMNS = [
    "day",
]


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

def label_to_text(
    label,
):
    """
    Convert configured numeric labels into readable names.

    None / missing labels are returned as Unknown.
    """

    if label is None:

        return "Unknown"

    if pd.isna(
        label
    ):

        return "Unknown"

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
# Model loading
# ============================================================

def load_model():
    """
    Load the trained Random Forest model.
    """

    model_path = Path(
        config.RANDOM_FOREST_MODEL_FILE
    )

    if not model_path.exists():

        raise FileNotFoundError(
            "Random Forest model not found:\n"
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


# ============================================================
# Discover unseen test flow files
# ============================================================

def discover_test_files():
    """
    Discover all flow-level CSV files belonging to the
    project's unseen test scope.

    Included:

        data/test/*_flow.csv

    plus:

        config.TEST_ADDITIONAL_FLOW_FILES

    IMPORTANT:

    Files containing "train" are NOT excluded.

    Packet files are excluded automatically because they
    normally end with "_packets.csv" and therefore do not
    match "*_flow.csv".
    """

    test_dir = Path(
        config.TEST_DIR
    )

    if not test_dir.exists():

        raise FileNotFoundError(
            "Test directory not found:\n"
            f"{test_dir}"
        )

    test_flow_glob = getattr(
        config,
        "TEST_FLOW_GLOB",
        "*_flow.csv",
    )

    discovered = []

    # --------------------------------------------------------
    # Main flow-file discovery
    # --------------------------------------------------------

    for path in sorted(
        test_dir.glob(
            test_flow_glob
        )
    ):

        if path.is_file():

            discovered.append(
                path.resolve()
            )

    # --------------------------------------------------------
    # Additional explicitly configured flow datasets
    # --------------------------------------------------------

    additional_files = getattr(
        config,
        "TEST_ADDITIONAL_FLOW_FILES",
        [],
    )

    for path in additional_files:

        path = Path(
            path
        )

        if not path.is_absolute():

            path = (
                Path(
                    config.PROJECT_ROOT
                )
                / path
            )

        if not path.exists():

            raise FileNotFoundError(
                "Configured additional test flow dataset "
                "does not exist:\n"
                f"{path}"
            )

        if not path.is_file():

            raise ValueError(
                "Configured additional test dataset "
                "is not a file:\n"
                f"{path}"
            )

        discovered.append(
            path.resolve()
        )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    unique_files = []

    seen = set()

    for path in discovered:

        key = str(
            path
        ).lower()

        if key not in seen:

            seen.add(
                key
            )

            unique_files.append(
                path
            )

    unique_files = sorted(
        unique_files,
        key=lambda p: str(
            p
        ).lower(),
    )

    if not unique_files:

        raise FileNotFoundError(
            "No unseen test flow CSV files were found.\n"
            f"Test directory: {test_dir}\n"
            f"Pattern      : {test_flow_glob}"
        )

    return unique_files


# ============================================================
# Dataset loading
# ============================================================

def load_single_test_dataset(
    path,
):
    """
    Load one flow-level unseen test CSV.

    Required:

        LOCAL_METADATA_COLUMNS
        config.ML_FEATURES

    Optional:

        config.LABEL_COLUMN
        OPTIONAL_METADATA_COLUMNS

    Ground-truth labels are NOT required for SHAP.
    """

    path = Path(
        path
    )

    print()

    print(
        f"  Loading: "
        f"{path.name}"
    )

    # ========================================================
    # Required model + traceability columns
    # ========================================================

    required_columns = [
        *LOCAL_METADATA_COLUMNS,
        *config.ML_FEATURES,
    ]

    required_columns = list(
        dict.fromkeys(
            required_columns
        )
    )

    # ========================================================
    # Read header first
    # ========================================================

    header = pd.read_csv(
        path,
        nrows=0,
    )

    header_columns = list(
        header.columns
    )

    # ========================================================
    # Validate required columns
    # ========================================================

    missing = [

        column

        for column in required_columns

        if column not in header_columns
    ]

    if missing:

        raise ValueError(
            "Dataset is missing required columns:\n"
            f"File: {path}\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    # ========================================================
    # Ground-truth label availability
    # ========================================================

    label_available = (
        config.LABEL_COLUMN
        in header_columns
    )

    # ========================================================
    # Optional metadata present in this file
    # ========================================================

    optional_metadata_present = [

        column

        for column in OPTIONAL_METADATA_COLUMNS

        if column in header_columns
    ]

    # ========================================================
    # Columns to load
    # ========================================================

    columns_to_load = [
        *required_columns,
        *optional_metadata_present,
    ]

    if label_available:

        columns_to_load.append(
            config.LABEL_COLUMN
        )

    columns_to_load = list(
        dict.fromkeys(
            columns_to_load
        )
    )

    # ========================================================
    # Explicit dtype only for ML features.
    #
    # NaN values remain NaN.
    # ========================================================

    dtype_map = {

        feature:
            FEATURE_DTYPE

        for feature
        in config.ML_FEATURES
    }

    # ========================================================
    # Load dataset
    # ========================================================

    df = pd.read_csv(
        path,
        usecols=columns_to_load,
        dtype=dtype_map,
        low_memory=False,
    )

    # ========================================================
    # Ground-truth label handling
    #
    # Use pandas nullable Int8 because missing ground truth
    # cannot be represented by numpy int8.
    # ========================================================

    if label_available:

        label_values = pd.to_numeric(
            df[
                config.LABEL_COLUMN
            ],
            errors="coerce",
        )

        df[
            config.LABEL_COLUMN
        ] = label_values.astype(
            "Int8"
        )

        invalid_label_mask = (
            df[
                config.LABEL_COLUMN
            ]
            .notna()
            &
            ~df[
                config.LABEL_COLUMN
            ]
            .isin(
                [
                    config.LABEL_BENIGN,
                    config.LABEL_ATTACK,
                ]
            )
        )

        if invalid_label_mask.any():

            invalid_values = (
                df.loc[
                    invalid_label_mask,
                    config.LABEL_COLUMN,
                ]
                .drop_duplicates()
                .tolist()
            )

            raise ValueError(
                "Unexpected ground-truth labels detected.\n"
                f"File: {path}\n"
                f"Values: {invalid_values}"
            )

    else:

        df[
            config.LABEL_COLUMN
        ] = pd.Series(
            pd.NA,
            index=df.index,
            dtype="Int8",
        )

    # ========================================================
    # Optional metadata normalization
    #
    # Ensure every combined dataframe contains the same
    # optional metadata columns.
    # ========================================================

    for column in OPTIONAL_METADATA_COLUMNS:

        if column not in df.columns:

            df[
                column
            ] = pd.NA

    # ========================================================
    # Preserve source dataset identity
    # ========================================================

    df.insert(
        0,
        "dataset_file",
        path.name,
    )

    df.insert(
        1,
        "label_available",
        bool(
            label_available
        ),
    )

    print(
        f"    Rows              : "
        f"{len(df):,}"
    )

    print(
        f"    Ground-truth label: "
        f"{'YES' if label_available else 'NO'}"
    )

    if label_available:

        labelled_rows = int(
            df[
                config.LABEL_COLUMN
            ]
            .notna()
            .sum()
        )

        print(
            f"    Labelled rows     : "
            f"{labelled_rows:,}"
        )

    return df


# ============================================================

def load_test_scope(
    test_files,
):
    """
    Load and combine all unseen flow-level test datasets.
    """

    print(
        f"  Test datasets discovered: "
        f"{len(test_files)}"
    )

    for index, path in enumerate(
        test_files,
        start=1,
    ):

        print(
            f"    [{index:>2}] "
            f"{path.name}"
        )

    frames = []

    for path in test_files:

        df = load_single_test_dataset(
            path
        )

        frames.append(
            df
        )

    if not frames:

        raise RuntimeError(
            "No test datasets were loaded."
        )

    test_df = pd.concat(
        frames,
        axis=0,
        ignore_index=True,
    )

    print()

    print(
        f"  Combined unseen test rows: "
        f"{len(test_df):,}"
    )

    labelled_rows = int(
        test_df[
            config.LABEL_COLUMN
        ]
        .notna()
        .sum()
    )

    unlabelled_rows = (
        len(
            test_df
        )
        - labelled_rows
    )

    labelled_datasets = int(
        test_df[
            [
                "dataset_file",
                "label_available",
            ]
        ]
        .drop_duplicates(
            subset=[
                "dataset_file"
            ]
        )[
            "label_available"
        ]
        .sum()
    )

    unlabelled_datasets = (
        len(
            test_files
        )
        - labelled_datasets
    )

    print(
        f"  Labelled datasets   : "
        f"{labelled_datasets:,}"
    )

    print(
        f"  Unlabelled datasets : "
        f"{unlabelled_datasets:,}"
    )

    print(
        f"  Labelled rows       : "
        f"{labelled_rows:,}"
    )

    print(
        f"  Unlabelled rows     : "
        f"{unlabelled_rows:,}"
    )

    return test_df


# ============================================================
# Feature validation
# ============================================================

def validate_features(
    df,
):
    """
    Validate configured model features.

    NaN:
        allowed

    Infinite values:
        rejected
    """

    config.validate_ml_features()

    missing = [

        feature

        for feature in config.ML_FEATURES

        if feature not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing ML features:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in missing
            )
        )

    X = df[
        config.ML_FEATURES
    ]

    # --------------------------------------------------------
    # NaN statistics
    # --------------------------------------------------------

    nan_count = int(
        X.isna()
        .sum()
        .sum()
    )

    total_cells = (
        X.shape[0]
        * X.shape[1]
    )

    nan_percentage = (
        (
            nan_count
            / total_cells
        )
        * 100.0
        if total_cells
        else 0.0
    )

    print(
        f"  NaN cells: "
        f"{nan_count:,} "
        f"({nan_percentage:.4f}%)"
    )

    print(
        "  NaN values are retained."
    )

    # --------------------------------------------------------
    # Infinite-value validation
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
            f"Found "
            f"{inf_count:,} infinite ML feature values."
        )


# ============================================================
# Metadata validation
# ============================================================

def validate_local_metadata(
    df,
):
    """
    Validate metadata required for local flow traceability.

    These columns are display/context fields only.
    They are never passed to the Random Forest.
    """

    missing = [

        column

        for column in LOCAL_METADATA_COLUMNS

        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing local SHAP metadata columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    print(
        "  Local flow traceability metadata: OK"
    )


# ============================================================
# Model / config compatibility
# ============================================================

def validate_model_compatibility(
    model,
):
    """
    Ensure the trained model is compatible with the current
    config.ML_FEATURES definition.
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
                f"{configured_count}\n"
                "\nRetrain the model or restore the exact "
                "feature configuration used during training."
            )

    # --------------------------------------------------------
    # Exact feature names / order
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
                "Model/config feature order mismatch.\n"
                "\nThe current config.ML_FEATURES does not "
                "match the exact feature order used to train "
                "the saved Random Forest."
            )

    print(
        f"  Model/config features: "
        f"OK ({configured_count})"
    )


# ============================================================
# Label / probability helpers
# ============================================================

def get_attack_class_index(
    model,
):
    """
    Return the probability / SHAP class index corresponding
    to config.LABEL_ATTACK.
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
            f"{config.LABEL_ATTACK} "
            "was not found in model.classes_.\n"
            f"Model classes: "
            f"{classes}"
        )

    return classes.index(
        config.LABEL_ATTACK
    )


# ============================================================

def get_attack_probability(
    model,
    X,
    attack_class_index,
):
    """
    Return P(Attack).
    """

    probabilities = model.predict_proba(
        X
    )

    return probabilities[
        :,
        attack_class_index
    ]


# ============================================================
# SHAP output handling
# ============================================================

def extract_attack_values(
    shap_output,
    attack_class_index,
):
    """
    Extract SHAP values for the configured Attack class.

    Supported formats:

        (samples, features, classes)

    or:

        (samples, features)

    Returns:

        (samples, features)
    """

    values = np.asarray(
        shap_output.values
    )

    # --------------------------------------------------------
    # Multi-output classifier
    # --------------------------------------------------------

    if values.ndim == 3:

        if (
            attack_class_index
            >= values.shape[2]
        ):

            raise ValueError(
                "Attack class index is outside "
                "the SHAP output dimensions.\n"
                f"Attack index: "
                f"{attack_class_index}\n"
                f"SHAP shape  : "
                f"{values.shape}"
            )

        return values[
            :,
            :,
            attack_class_index
        ]

    # --------------------------------------------------------
    # Single-output SHAP representation
    # --------------------------------------------------------

    if values.ndim == 2:

        return values

    raise ValueError(
        "Unexpected SHAP value shape: "
        f"{values.shape}"
    )


# ============================================================

def extract_attack_base_values(
    shap_output,
    n_samples,
    attack_class_index,
):
    """
    Extract one Attack-class base value per sample.
    """

    base = np.asarray(
        shap_output.base_values
    )

    # --------------------------------------------------------
    # (samples, classes)
    # --------------------------------------------------------

    if base.ndim == 2:

        if (
            attack_class_index
            >= base.shape[1]
        ):

            raise ValueError(
                "Attack class index is outside "
                "SHAP base-value dimensions.\n"
                f"Attack index: "
                f"{attack_class_index}\n"
                f"Base shape  : "
                f"{base.shape}"
            )

        return base[
            :,
            attack_class_index
        ]

    # --------------------------------------------------------
    # (samples,)
    # --------------------------------------------------------

    if base.ndim == 1:

        if (
            base.shape[0]
            == n_samples
        ):

            return base.astype(
                float
            )

        if (
            base.shape[0]
            == 1
        ):

            return np.full(
                n_samples,
                float(
                    base[0]
                ),
            )

    # --------------------------------------------------------
    # scalar
    # --------------------------------------------------------

    if base.ndim == 0:

        return np.full(
            n_samples,
            float(
                base
            ),
        )

    raise ValueError(
        "Unexpected SHAP base value shape: "
        f"{base.shape}"
    )


# ============================================================
# Additivity validation
# ============================================================

def check_additivity(
    shap_values_attack,
    base_values_attack,
    model,
    X_data,
    attack_class_index,
):
    """
    Verify:

        base value + sum(SHAP values)

    approximately equals:

        P(Attack)

    for the same Random Forest prediction.
    """

    model_probability = (
        get_attack_probability(
            model,
            X_data,
            attack_class_index,
        )
    )

    reconstructed_probability = (
        base_values_attack
        +
        shap_values_attack.sum(
            axis=1
        )
    )

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

    passed = (
        max_discrepancy
        <= ADDITIVITY_TOLERANCE
    )

    return (
        passed,
        max_discrepancy,
        mean_discrepancy,
    )


# ============================================================
# Build a local SHAP explanation
# ============================================================

def build_row_explanation(
    shap_values_attack,
    base_values_attack,
    X_data,
    position,
):
    """
    Build a shap.Explanation object for one row.
    """

    return shap.Explanation(

        values=(
            shap_values_attack[
                position
            ]
        ),

        base_values=(
            base_values_attack[
                position
            ]
        ),

        data=(
            X_data
            .iloc[
                position
            ]
            .to_numpy()
        ),

        feature_names=(
            list(
                config.ML_FEATURES
            )
        ),
    )


# ============================================================
# Local flow context
# ============================================================

def build_local_flow_context(
    test_df,
    local_position,
):
    """
    Build traceability information for one explained flow.

    Metadata is returned separately from ML / SHAP features.
    """

    row = test_df.iloc[
        local_position
    ]

    context = {

        "dataset_file":
            (
                str(
                    row[
                        "dataset_file"
                    ]
                )
                if pd.notna(
                    row[
                        "dataset_file"
                    ]
                )
                else None
            ),

        "label_available":
            bool(
                row[
                    "label_available"
                ]
            ),

        "day":
            (
                str(
                    row[
                        "day"
                    ]
                )
                if (
                    "day" in test_df.columns
                    and pd.notna(
                        row[
                            "day"
                        ]
                    )
                )
                else None
            ),

        "flow_id":
            (
                str(
                    row[
                        "flow_id"
                    ]
                )
                if pd.notna(
                    row[
                        "flow_id"
                    ]
                )
                else None
            ),

        "timestamp":
            (
                str(
                    row[
                        "timestamp"
                    ]
                )
                if pd.notna(
                    row[
                        "timestamp"
                    ]
                )
                else None
            ),

        "src_ip":
            (
                str(
                    row[
                        "src_ip"
                    ]
                )
                if pd.notna(
                    row[
                        "src_ip"
                    ]
                )
                else None
            ),

        "dst_ip":
            (
                str(
                    row[
                        "dst_ip"
                    ]
                )
                if pd.notna(
                    row[
                        "dst_ip"
                    ]
                )
                else None
            ),
    }

    # --------------------------------------------------------
    # src_port / dst_port are already ML features.
    # --------------------------------------------------------

    if "src_port" in test_df.columns:

        src_port = row[
            "src_port"
        ]

        context[
            "src_port"
        ] = (
            int(
                src_port
            )
            if pd.notna(
                src_port
            )
            else None
        )

    if "dst_port" in test_df.columns:

        dst_port = row[
            "dst_port"
        ]

        context[
            "dst_port"
        ] = (
            int(
                dst_port
            )
            if pd.notna(
                dst_port
            )
            else None
        )

    return context


# ============================================================
# Dataset summary helper
# ============================================================

def build_dataset_summary(
    test_df,
):
    """
    Build dataset-level provenance and label-availability
    information for shap_summary.json.
    """

    results = []

    dataset_names = (
        test_df[
            "dataset_file"
        ]
        .drop_duplicates()
        .tolist()
    )

    for dataset_name in dataset_names:

        subset = test_df[
            test_df[
                "dataset_file"
            ]
            .eq(
                dataset_name
            )
        ]

        label_available = bool(
            subset[
                "label_available"
            ]
            .iloc[
                0
            ]
        )

        labelled_rows = int(
            subset[
                config.LABEL_COLUMN
            ]
            .notna()
            .sum()
        )

        results.append(
            {
                "dataset_file":
                    str(
                        dataset_name
                    ),

                "rows":
                    int(
                        len(
                            subset
                        )
                    ),

                "label_available":
                    label_available,

                "labelled_rows":
                    labelled_rows,
            }
        )

    return results


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate global and optional local SHAP "
            "explanations for the trained NetworkIDS "
            "Random Forest."
        )
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=(
            "Number of unseen test rows used for "
            "global SHAP explanation."
        ),
    )

    parser.add_argument(
        "--local-row",
        type=int,
        default=None,
        help=(
            "Optional zero-based row position from the "
            "combined unseen test scope for local SHAP."
        ),
    )

    args = parser.parse_args()

    if (
        args.sample_size
        <= 0
    ):

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
        "NetworkIDS - SHAP EXPLAINABILITY"
    )

    print(
        "=" * 70
    )

    print(
        f"SHAP version : "
        f"{shap.__version__}"
    )

    # ========================================================
    # 1. Configuration / model
    # ========================================================

    print_section(
        "[1/5] Loading model"
    )

    config.validate_ml_features()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = load_model()

    validate_model_compatibility(
        model
    )

    attack_class_index = (
        get_attack_class_index(
            model
        )
    )

    print(
        f"  Model classes       : "
        f"{list(model.classes_)}"
    )

    print(
        f"  Attack label        : "
        f"{config.LABEL_ATTACK}"
    )

    print(
        f"  Attack class index  : "
        f"{attack_class_index}"
    )

    # ========================================================
    # 2. Discover + load unseen test scope
    # ========================================================

    print_section(
        "[2/5] Loading unseen test scope"
    )

    test_files = (
        discover_test_files()
    )

    test_df = (
        load_test_scope(
            test_files
        )
    )

    validate_features(
        test_df
    )

    validate_local_metadata(
        test_df
    )

    X_test = (
        test_df[
            config.ML_FEATURES
        ]
        .copy()
    )

    y_test = (
        test_df[
            config.LABEL_COLUMN
        ]
        .copy()
    )

    # ========================================================
    # Optional local row validation
    # ========================================================

    if (
        args.local_row
        is not None
    ):

        if (
            args.local_row < 0
            or
            args.local_row
            >= len(
                X_test
            )
        ):

            raise ValueError(
                "--local-row is outside the combined "
                "unseen test scope.\n"
                f"Valid range: "
                f"0 to {len(X_test) - 1}"
            )

    # ========================================================
    # 3. Sample + compute SHAP
    # ========================================================

    print_section(
        "[3/5] Computing SHAP values"
    )

    sample_size = min(
        args.sample_size,
        len(
            X_test
        ),
    )

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    sample_positions = rng.choice(

        len(
            X_test
        ),

        size=sample_size,

        replace=False,
    )

    X_sample = (
        X_test
        .iloc[
            sample_positions
        ]
        .reset_index(
            drop=True
        )
        .copy()
    )

    nan_cells_in_sample = int(
        X_sample
        .isna()
        .sum()
        .sum()
    )

    print(
        f"  SHAP sample size : "
        f"{sample_size:,}"
    )

    print(
        f"  ML features      : "
        f"{len(config.ML_FEATURES)}"
    )

    print(
        f"  NaN cells        : "
        f"{nan_cells_in_sample:,}"
    )

    print(
        "  Building TreeExplainer..."
    )

    explainer = shap.TreeExplainer(
        model,
        feature_perturbation=(
            "tree_path_dependent"
        ),
    )

    print(
        "  Computing SHAP values..."
    )

    shap_output = explainer(
        X_sample
    )

    shap_values_attack = (
        extract_attack_values(

            shap_output,

            attack_class_index,
        )
    )

    base_values_attack = (
        extract_attack_base_values(

            shap_output,

            sample_size,

            attack_class_index,
        )
    )

    expected_shape = (

        sample_size,

        len(
            config.ML_FEATURES
        ),
    )

    if (
        shap_values_attack.shape
        != expected_shape
    ):

        raise RuntimeError(
            "Unexpected SHAP matrix shape.\n"
            f"Expected: "
            f"{expected_shape}\n"
            f"Actual  : "
            f"{shap_values_attack.shape}"
        )

    print(
        f"  SHAP matrix      : "
        f"{shap_values_attack.shape}"
    )

    # --------------------------------------------------------
    # Additivity check
    # --------------------------------------------------------

    print(
        "  Checking additivity..."
    )

    (
        additivity_passed,
        max_discrepancy,
        mean_discrepancy,
    ) = check_additivity(

        shap_values_attack,

        base_values_attack,

        model,

        X_sample,

        attack_class_index,
    )

    print(
        f"  Max discrepancy  : "
        f"{max_discrepancy:.8f}"
    )

    print(
        f"  Mean discrepancy : "
        f"{mean_discrepancy:.8f}"
    )

    if not additivity_passed:

        raise RuntimeError(
            "SHAP additivity validation failed.\n"
            f"Maximum discrepancy: "
            f"{max_discrepancy:.8f}\n"
            f"Allowed tolerance  : "
            f"{ADDITIVITY_TOLERANCE}"
        )

    print(
        "  [OK] SHAP additivity passed."
    )

    # ========================================================
    # 4. Global SHAP outputs
    # ========================================================

    print_section(
        "[4/5] Saving global SHAP explanations"
    )

    mean_abs_shap = (

        np.abs(
            shap_values_attack
        )

        .mean(
            axis=0
        )
    )

    importance = (

        pd.Series(

            mean_abs_shap,

            index=(
                config.ML_FEATURES
            ),
        )

        .sort_values(
            ascending=False
        )
    )

    # --------------------------------------------------------
    # Console top features
    # --------------------------------------------------------

    top_count = min(
        15,
        len(
            importance
        ),
    )

    print(
        f"Top {top_count} features "
        "by mean |SHAP|:"
    )

    for rank, (
        feature,
        value,
    ) in enumerate(

        importance
        .head(
            top_count
        )
        .items(),

        start=1,
    ):

        print(
            f"  {rank:>2}. "
            f"{feature:<35} "
            f"{value:.6f}"
        )

    # --------------------------------------------------------
    # Importance CSV
    # --------------------------------------------------------

    importance_df = pd.DataFrame(
        {
            "rank":
                range(
                    1,
                    len(
                        importance
                    )
                    + 1
                ),

            "feature":
                importance.index,

            "mean_abs_shap":
                importance.values,
        }
    )

    importance_file = (
        OUTPUT_DIR
        / "global_importance.csv"
    )

    importance_df.to_csv(
        importance_file,
        index=False,
    )

    # --------------------------------------------------------
    # Global bar plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(
            9,
            8
        )
    )

    (
        importance
        .head(
            min(
                20,
                len(
                    importance
                )
            )
        )
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
        "Global SHAP Feature Importance"
    )

    plt.tight_layout()

    global_bar_file = (
        OUTPUT_DIR
        / "global_importance_bar.png"
    )

    plt.savefig(
        global_bar_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------------
    # Global beeswarm
    # --------------------------------------------------------

    plt.figure()

    shap.summary_plot(

        shap_values_attack,

        X_sample,

        feature_names=(
            config.ML_FEATURES
        ),

        max_display=(
            min(
                20,
                len(
                    config.ML_FEATURES
                )
            )
        ),

        show=False,
    )

    plt.tight_layout()

    beeswarm_file = (
        OUTPUT_DIR
        / "global_beeswarm.png"
    )

    plt.savefig(
        beeswarm_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"  Importance CSV : "
        f"{importance_file}"
    )

    print(
        f"  Global bar     : "
        f"{global_bar_file}"
    )

    print(
        f"  Beeswarm       : "
        f"{beeswarm_file}"
    )

    # ========================================================
    # 5. Optional local explanation + summary
    # ========================================================

    print_section(
        "[5/5] Local explanation and summary"
    )

    local_summary = None

    if (
        args.local_row
        is not None
    ):

        local_position = int(
            args.local_row
        )

        X_local = (
            X_test
            .iloc[
                [
                    local_position
                ]
            ]
            .copy()
        )

        local_output = explainer(
            X_local
        )

        local_values_attack = (
            extract_attack_values(

                local_output,

                attack_class_index,
            )
        )

        local_base_attack = (
            extract_attack_base_values(

                local_output,

                1,

                attack_class_index,
            )
        )

        prediction = int(
            model.predict(
                X_local
            )[0]
        )

        attack_probability = float(
            get_attack_probability(

                model,

                X_local,

                attack_class_index,
            )[0]
        )

        # ====================================================
        # Optional ground truth
        # ====================================================

        actual_value = (
            y_test.iloc[
                local_position
            ]
        )

        if pd.isna(
            actual_value
        ):

            actual_label = None

        else:

            actual_label = int(
                actual_value
            )

        actual_display = (
            label_to_text(
                actual_label
            )
        )

        # ====================================================
        # Flow traceability
        # ====================================================

        flow_context = (
            build_local_flow_context(
                test_df,
                local_position,
            )
        )

        local_explanation = (
            build_row_explanation(

                local_values_attack,

                local_base_attack,

                X_local.reset_index(
                    drop=True
                ),

                0,
            )
        )

        plt.figure()

        shap.plots.waterfall(

            local_explanation,

            max_display=(
                min(
                    15,
                    len(
                        config.ML_FEATURES
                    )
                )
            ),

            show=False,
        )

        flow_id_display = (
            flow_context.get(
                "flow_id"
            )
            or
            "Unknown Flow"
        )

        dataset_display = (
            flow_context.get(
                "dataset_file"
            )
            or
            "Unknown Dataset"
        )

        plt.title(
            "Local SHAP Explanation\n"
            f"Dataset: {dataset_display} | "
            f"Flow: {flow_id_display}\n"
            f"Actual: {actual_display} | "
            f"Predicted: "
            f"{label_to_text(prediction)} | "
            f"P(Attack): "
            f"{attack_probability:.4f}"
        )

        plt.tight_layout()

        local_file = (
            OUTPUT_DIR
            / "local_waterfall.png"
        )

        plt.savefig(
            local_file,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()

        # ====================================================
        # Local summary
        # ====================================================

        local_summary = {

            "test_row":
                local_position,

            "dataset_file":
                flow_context.get(
                    "dataset_file"
                ),

            "label_available":
                flow_context.get(
                    "label_available"
                ),

            "day":
                flow_context.get(
                    "day"
                ),

            "flow_id":
                flow_context.get(
                    "flow_id"
                ),

            "timestamp":
                flow_context.get(
                    "timestamp"
                ),

            "src_ip":
                flow_context.get(
                    "src_ip"
                ),

            "src_port":
                flow_context.get(
                    "src_port"
                ),

            "dst_ip":
                flow_context.get(
                    "dst_ip"
                ),

            "dst_port":
                flow_context.get(
                    "dst_port"
                ),

            "actual_label":
                actual_label,

            "actual_label_text":
                actual_display,

            "predicted_label":
                prediction,

            "predicted_label_text":
                label_to_text(
                    prediction
                ),

            "attack_probability":
                attack_probability,

            "output_file":
                local_file.name,
        }

        # ====================================================
        # Console local-flow information
        # ====================================================

        print(
            f"  Local row       : "
            f"{local_position}"
        )

        print(
            f"  Dataset file    : "
            f"{flow_context.get('dataset_file')}"
        )

        if (
            flow_context.get(
                "day"
            )
            is not None
        ):

            print(
                f"  Day             : "
                f"{flow_context.get('day')}"
            )

        print(
            f"  Flow ID         : "
            f"{flow_context.get('flow_id')}"
        )

        print(
            f"  Timestamp       : "
            f"{flow_context.get('timestamp')}"
        )

        print(
            f"  Source          : "
            f"{flow_context.get('src_ip')}:"
            f"{flow_context.get('src_port')}"
        )

        print(
            f"  Destination     : "
            f"{flow_context.get('dst_ip')}:"
            f"{flow_context.get('dst_port')}"
        )

        print(
            f"  Ground truth    : "
            f"{'Available' if flow_context.get('label_available') else 'Unavailable'}"
        )

        print(
            f"  Actual          : "
            f"{actual_display}"
        )

        print(
            f"  Predicted       : "
            f"{label_to_text(prediction)}"
        )

        print(
            f"  P(Attack)       : "
            f"{attack_probability:.6f}"
        )

        print(
            f"  Waterfall       : "
            f"{local_file}"
        )

    else:

        print(
            "  No --local-row provided."
        )

        print(
            "  Global SHAP outputs only."
        )

    # ========================================================
    # Dataset-level label information
    # ========================================================

    dataset_summary = (
        build_dataset_summary(
            test_df
        )
    )

    labelled_dataset_count = sum(
        1
        for item in dataset_summary
        if item[
            "label_available"
        ]
    )

    unlabelled_dataset_count = (
        len(
            dataset_summary
        )
        - labelled_dataset_count
    )

    labelled_row_count = int(
        test_df[
            config.LABEL_COLUMN
        ]
        .notna()
        .sum()
    )

    unlabelled_row_count = (
        len(
            test_df
        )
        - labelled_row_count
    )

    # ========================================================
    # Summary JSON
    # ========================================================

    summary = {

        "model":
            "RandomForestClassifier",

        "model_path":
            str(
                config.RANDOM_FOREST_MODEL_FILE
            ),

        "test_scope":
            str(
                config.TEST_DIR
            ),

        "test_dataset_pattern":
            getattr(
                config,
                "TEST_FLOW_GLOB",
                "*_flow.csv",
            ),

        "additional_test_datasets":
            [
                str(
                    path
                )
                for path
                in getattr(
                    config,
                    "TEST_ADDITIONAL_FLOW_FILES",
                    [],
                )
            ],

        "test_dataset_files":
            [
                path.name
                for path
                in test_files
            ],

        "test_dataset_count":
            len(
                test_files
            ),

        "test_rows":
            int(
                len(
                    test_df
                )
            ),

        "ground_truth": {

            "label_optional":
                True,

            "label_column":
                config.LABEL_COLUMN,

            "labelled_dataset_count":
                labelled_dataset_count,

            "unlabelled_dataset_count":
                unlabelled_dataset_count,

            "labelled_row_count":
                labelled_row_count,

            "unlabelled_row_count":
                unlabelled_row_count,

            "datasets":
                dataset_summary,
        },

        "shap_version":
            shap.__version__,

        "sample_size":
            int(
                sample_size
            ),

        "random_state":
            RANDOM_STATE,

        "feature_count":
            len(
                config.ML_FEATURES
            ),

        "attack_label":
            int(
                config.LABEL_ATTACK
            ),

        "attack_class_index":
            int(
                attack_class_index
            ),

        "nan_policy":
            "NaN retained; no imputation",

        "nan_cells_in_sample":
            nan_cells_in_sample,

        "sampling_policy":
            (
                "Uniform random row sampling across the combined "
                "unseen test flow population"
            ),

        "additivity_check": {

            "passed":
                bool(
                    additivity_passed
                ),

            "tolerance":
                ADDITIVITY_TOLERANCE,

            "max_discrepancy":
                max_discrepancy,

            "mean_discrepancy":
                mean_discrepancy,
        },

        "top_features": {

            str(
                feature
            ):
                float(
                    value
                )

            for feature, value
            in importance
            .head(
                min(
                    10,
                    len(
                        importance
                    )
                )
            )
            .items()
        },

        "local_explanation":
            local_summary,
    }

    summary_file = (
        OUTPUT_DIR
        / "shap_summary.json"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    print(
        f"  Summary         : "
        f"{summary_file}"
    )

    # ========================================================
    # Final
    # ========================================================

    print_section(
        "SHAP EXPLAINABILITY COMPLETE"
    )

    print(
        f"Model           : "
        f"{config.RANDOM_FOREST_MODEL_FILE}"
    )

    print(
        f"Test datasets   : "
        f"{len(test_files)}"
    )

    print(
        f"Test rows       : "
        f"{len(X_test):,}"
    )

    print(
        f"Labelled rows   : "
        f"{labelled_row_count:,}"
    )

    print(
        f"Unlabelled rows : "
        f"{unlabelled_row_count:,}"
    )

    print(
        f"SHAP sample     : "
        f"{sample_size:,}"
    )

    print(
        f"Features        : "
        f"{len(config.ML_FEATURES)}"
    )

    print(
        f"Attack class    : "
        f"{config.LABEL_ATTACK}"
    )

    print(
        f"Class index     : "
        f"{attack_class_index}"
    )

    print(
        "NaN imputation : None"
    )

    print(
        "Ground truth    : Optional"
    )

    print(
        "Additivity      : PASS"
    )

    print(
        f"Output          : "
        f"{OUTPUT_DIR}"
    )


# ============================================================

if __name__ == "__main__":

    main()