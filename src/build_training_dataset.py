from pathlib import Path
import os
import sys

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
# Input / Output Paths
# ============================================================

PROCESSED = (
    config.PROCESSED_DIR
)

LABELED = (
    PROCESSED
    / "labeled"
)


# Original CIC multiday training split.
#
# This is one source used to construct the final enhanced
# training dataset.
TRAIN = (
    config.CIC_TRAIN_SPLIT_FILE
)


# Final enhanced training dataset.
OUT = (
    config.TRAIN_DATASET_FILE
)


# Final training dataset provenance manifest.
MANIFEST = (
    config.TRAIN_MANIFEST_FILE
)


# Temporary files used during construction.
#
# The final dataset is published only after all validation
# checks pass.
TEMP = (
    config.TRAIN_DIR
    / "training_dataset.building.tmp"
)


TEMP_MANIFEST = (
    config.TRAIN_DIR
    / "training_dataset_manifest.building.tmp"
)


# ============================================================
# Feature Definition
#
# Use config.ML_FEATURES as the single source of truth.
# ============================================================

FEATURES = list(
    config.ML_FEATURES
)


COLS = (
    FEATURES
    + [
        config.LABEL_COLUMN
    ]
)


# ============================================================
# Build Settings
# ============================================================

CHUNK_SIZE = 100_000

SEED = 20261018

MQTT_N = 50_000


# Final dataset size validated during the original build.
EXPECTED_FINAL_ROWS = 3_484_498


# ============================================================
# Helpers
# ============================================================

def require_file(
    path,
):
    """
    Ensure that a required source file exists.
    """

    path = Path(
        path
    )

    if not path.is_file():

        raise FileNotFoundError(
            f"File not found: {path}"
        )

    return path


# ============================================================

def check_header(
    path,
):
    """
    Ensure that a source CSV contains the complete ML feature
    set plus the binary label column.
    """

    header = pd.read_csv(
        path,
        nrows=0,
    ).columns

    missing = [

        column

        for column in COLS

        if column not in header
    ]

    if missing:

        raise ValueError(
            f"{path.name} is missing columns: "
            f"{missing}"
        )


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # Configuration validation
    # ========================================================

    config.validate_ml_features()

    # ========================================================
    # SAFETY CHECK
    #
    # Never overwrite the published final training dataset or
    # its manifest automatically.
    # ========================================================

    if (
        OUT.exists()
        or MANIFEST.exists()
    ):

        raise FileExistsError(
            "training_dataset.csv or its manifest already exists. "
            "Nothing was overwritten. "
            "Please inspect or move the existing outputs first."
        )

    config.TRAIN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Remove only incomplete temporary files from a previous
    # failed build.
    # --------------------------------------------------------

    for path in [
        TEMP,
        TEMP_MANIFEST,
    ]:

        if path.exists():

            path.unlink()

    # ========================================================
    # DISCOVER LABELED FILES
    # ========================================================

    brute = sorted(
        LABELED.glob(
            "attack_bruteforce_*_flow_source_labeled.csv"
        )
    )

    recon = sorted(
        LABELED.glob(
            "attack_recon_*_flow_source_labeled.csv"
        )
    )

    if len(
        brute
    ) != 8:

        raise ValueError(
            "Expected 8 labeled Bruteforce files, "
            f"found {len(brute)}"
        )

    if len(
        recon
    ) != 8:

        raise ValueError(
            "Expected 8 labeled Recon files, "
            f"found {len(recon)}"
        )

    # ========================================================
    # BUILD JOB LIST
    #
    # Tuple:
    #
    # (
    #     path,
    #     dataset_group,
    #     sample_n,
    #     expected_source_rows,
    #     expected_label
    # )
    #
    # expected_label:
    #
    #     None = mixed labels allowed
    #     0    = all rows must be Benign
    #     1    = all rows must be Attack
    # ========================================================

    jobs = [

        (
            require_file(
                TRAIN
            ),
            "MultiDay Train",
            None,
            None,
            None,
        ),

        (
            require_file(
                LABELED
                / "benign_whole-network3_flow_source_labeled.csv"
            ),
            "Additional Benign",
            None,
            22_618,
            config.LABEL_BENIGN,
        ),
    ]

    # --------------------------------------------------------
    # Bruteforce
    # --------------------------------------------------------

    jobs += [

        (
            path,
            "Bruteforce",
            None,
            None,
            config.LABEL_ATTACK,
        )

        for path in brute
    ]

    # --------------------------------------------------------
    # Recon
    # --------------------------------------------------------

    jobs += [

        (
            path,
            "Recon",
            None,
            None,
            config.LABEL_ATTACK,
        )

        for path in recon
    ]

    # ========================================================
    # MQTT
    #
    # Take exactly 50,000 flows from each large attack
    # scenario using the fixed project seed.
    # ========================================================

    jobs += [

        (
            require_file(
                LABELED
                / "MQTT-DDoS-Connect_Flood_train_flow_source_labeled.csv"
            ),
            "MQTT DDoS Connect",
            MQTT_N,
            6_680_854,
            config.LABEL_ATTACK,
        ),

        (
            require_file(
                LABELED
                / "MQTT-DDoS-Publish_Flood_train_flow_source_labeled.csv"
            ),
            "MQTT DDoS Publish",
            MQTT_N,
            775_095,
            config.LABEL_ATTACK,
        ),

        (
            require_file(
                LABELED
                / "MQTT-DoS-Connect_Flood_train_flow_source_labeled.csv"
            ),
            "MQTT DoS Connect",
            MQTT_N,
            288_101,
            config.LABEL_ATTACK,
        ),

        (
            require_file(
                LABELED
                / "MQTT-DoS-Publish_Flood_mqtt_attack_subset_labeled.csv"
            ),
            "MQTT DoS Publish subset",
            None,
            132,
            config.LABEL_ATTACK,
        ),
    ]

    # ========================================================
    # HEADER VALIDATION
    # ========================================================

    for (
        path,
        _,
        _,
        _,
        _,
    ) in jobs:

        check_header(
            path
        )

    # ========================================================
    # RANDOM GENERATOR
    # ========================================================

    rng = np.random.default_rng(
        SEED
    )

    manifest = []

    label_total = {

        config.LABEL_BENIGN:
            0,

        config.LABEL_ATTACK:
            0,
    }

    total_written = 0

    header_written = False

    # ========================================================
    # PROCESS SOURCES
    # ========================================================

    for number, (
        path,
        group,
        sample_n,
        expected_rows,
        expected_label,
    ) in enumerate(
        jobs,
        start=1,
    ):

        print(
            f"\n[{number}/{len(jobs)}] "
            f"{path.name}",
            flush=True,
        )

        # ====================================================
        # Generate exact sample positions
        # ====================================================

        sample_indices = None

        if sample_n is not None:

            if expected_rows is None:

                raise ValueError(
                    f"{path.name}: sampled dataset "
                    "must have expected_rows."
                )

            if (
                sample_n
                > expected_rows
            ):

                raise ValueError(
                    f"{path.name}: requested sample "
                    f"{sample_n:,} is larger than "
                    f"source {expected_rows:,}."
                )

            sample_indices = np.sort(
                rng.choice(
                    expected_rows,
                    size=sample_n,
                    replace=False,
                )
            )

            print(
                f"Sampling {sample_n:,} rows "
                f"from {expected_rows:,}; "
                f"seed={SEED}",
                flush=True,
            )

        else:

            print(
                "Including all rows",
                flush=True,
            )

        rows_seen = 0

        rows_written = 0

        labels_for_file = {

            config.LABEL_BENIGN:
                0,

            config.LABEL_ATTACK:
                0,
        }

        # ====================================================
        # Chunked reading
        # ====================================================

        reader = pd.read_csv(
            path,
            usecols=COLS,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        )

        for chunk in reader:

            chunk = chunk[
                COLS
            ]

            start = rows_seen

            rows_seen += len(
                chunk
            )

            # ------------------------------------------------
            # Validate labels across EVERY source row,
            # including MQTT rows that will not be sampled.
            # ------------------------------------------------

            source_labels = pd.to_numeric(
                chunk[
                    config.LABEL_COLUMN
                ],
                errors="raise",
            )

            if source_labels.isna().any():

                raise ValueError(
                    f"Missing label in source file: "
                    f"{path.name}"
                )

            found_source_labels = set(
                source_labels.unique()
            )

            valid_labels = {
                config.LABEL_BENIGN,
                config.LABEL_ATTACK,
            }

            if not found_source_labels.issubset(
                valid_labels
            ):

                raise ValueError(
                    f"Unexpected labels "
                    f"{found_source_labels} "
                    f"in {path.name}"
                )

            if (
                expected_label
                is not None
            ):

                if not source_labels.eq(
                    expected_label
                ).all():

                    bad_labels = sorted(
                        source_labels[
                            source_labels
                            != expected_label
                        ]
                        .unique()
                        .tolist()
                    )

                    raise ValueError(
                        f"{path.name}: expected all source "
                        f"labels to be {expected_label}, "
                        f"but found unexpected labels "
                        f"{bad_labels}."
                    )

            chunk[
                config.LABEL_COLUMN
            ] = source_labels

            # =================================================
            # Apply MQTT sample positions
            # =================================================

            if sample_indices is not None:

                left = np.searchsorted(
                    sample_indices,
                    start,
                    side="left",
                )

                right = np.searchsorted(
                    sample_indices,
                    rows_seen,
                    side="left",
                )

                if left == right:

                    continue

                positions = (
                    sample_indices[
                        left:right
                    ]
                    - start
                )

                chunk = (
                    chunk
                    .iloc[
                        positions
                    ]
                    .copy()
                )

            else:

                chunk = chunk.copy()

            if chunk.empty:

                continue

            # =================================================
            # Convert all ML features to numeric
            # =================================================

            for feature in FEATURES:

                chunk[
                    feature
                ] = pd.to_numeric(
                    chunk[
                        feature
                    ],
                    errors="raise",
                )

            chunk[
                config.LABEL_COLUMN
            ] = pd.to_numeric(
                chunk[
                    config.LABEL_COLUMN
                ],
                errors="raise",
            )

            # =================================================
            # Label validation
            # =================================================

            if chunk[
                config.LABEL_COLUMN
            ].isna().any():

                raise ValueError(
                    f"Missing label in "
                    f"{path.name}"
                )

            found_labels = set(
                chunk[
                    config.LABEL_COLUMN
                ]
                .unique()
            )

            if not found_labels.issubset(
                valid_labels
            ):

                raise ValueError(
                    f"Unexpected labels "
                    f"{found_labels} "
                    f"in {path.name}"
                )

            # =================================================
            # Extra safety
            #
            # Newly labeled datasets must contain only their
            # expected class.
            #
            # MultiDay Train may contain both classes.
            # =================================================

            if (
                expected_label
                is not None
            ):

                unexpected = (
                    chunk[
                        config.LABEL_COLUMN
                    ]
                    != expected_label
                )

                if unexpected.any():

                    bad_labels = sorted(
                        chunk.loc[
                            unexpected,
                            config.LABEL_COLUMN,
                        ]
                        .unique()
                        .tolist()
                    )

                    raise ValueError(
                        f"{path.name}: expected all labels "
                        f"to be {expected_label}, "
                        f"but found {bad_labels}."
                    )

            chunk[
                config.LABEL_COLUMN
            ] = (
                chunk[
                    config.LABEL_COLUMN
                ]
                .astype(
                    "int8"
                )
            )

            # =================================================
            # Count labels
            # =================================================

            counts = (
                chunk[
                    config.LABEL_COLUMN
                ]
                .value_counts()
            )

            for label in (
                config.LABEL_BENIGN,
                config.LABEL_ATTACK,
            ):

                count = int(
                    counts.get(
                        label,
                        0,
                    )
                )

                labels_for_file[
                    label
                ] += count

                label_total[
                    label
                ] += count

            # =================================================
            # Append to temporary final training dataset
            # =================================================

            chunk.to_csv(
                TEMP,
                index=False,
                mode=(
                    "a"
                    if header_written
                    else "w"
                ),
                header=(
                    not header_written
                ),
            )

            header_written = True

            rows_written += len(
                chunk
            )

            total_written += len(
                chunk
            )

        # ====================================================
        # SOURCE VALIDATION
        # ====================================================

        if (
            expected_rows
            is not None
            and rows_seen
            != expected_rows
        ):

            raise ValueError(
                f"{path.name}: expected "
                f"{expected_rows:,} rows, "
                f"but read {rows_seen:,}. "
                "Temporary output was not published."
            )

        if (
            sample_n
            is not None
            and rows_written
            != sample_n
        ):

            raise ValueError(
                f"{path.name}: expected sample "
                f"of {sample_n:,}, "
                f"but wrote {rows_written:,}."
            )

        # ====================================================
        # MANIFEST ENTRY
        # ====================================================

        manifest.append(
            {
                "source_file":
                    path.name,

                "dataset_group":
                    group,

                "source_rows_scanned":
                    rows_seen,

                "rows_included":
                    rows_written,

                "label_0_benign":
                    labels_for_file[
                        config.LABEL_BENIGN
                    ],

                "label_1_attack":
                    labels_for_file[
                        config.LABEL_ATTACK
                    ],

                "sampling":
                    (
                        "all rows"
                        if sample_n is None
                        else
                        f"random sample of "
                        f"{sample_n}"
                    ),

                "seed":
                    (
                        ""
                        if sample_n is None
                        else
                        SEED
                    ),
            }
        )

        print(
            f"Included: "
            f"{rows_written:,} | "
            f"Benign: "
            f"{labels_for_file[config.LABEL_BENIGN]:,} | "
            f"Attack: "
            f"{labels_for_file[config.LABEL_ATTACK]:,}",
            flush=True,
        )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    if not header_written:

        raise RuntimeError(
            "No rows were written."
        )

    if (
        total_written
        != EXPECTED_FINAL_ROWS
    ):

        raise ValueError(
            "Expected final training dataset to contain "
            f"{EXPECTED_FINAL_ROWS:,} rows, "
            f"but wrote {total_written:,}. "
            "Temporary output was not published."
        )

    if (
        label_total[
            config.LABEL_BENIGN
        ]
        +
        label_total[
            config.LABEL_ATTACK
        ]
        != total_written
    ):

        raise ValueError(
            "Final label totals do not match "
            "the number of written rows."
        )

    # ========================================================
    # SAVE MANIFEST TEMPORARILY
    # ========================================================

    pd.DataFrame(
        manifest
    ).to_csv(
        TEMP_MANIFEST,
        index=False,
    )

    # ========================================================
    # PUBLISH ONLY AFTER ALL CHECKS PASS
    # ========================================================

    os.replace(
        TEMP,
        OUT,
    )

    os.replace(
        TEMP_MANIFEST,
        MANIFEST,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()

    print(
        "========== BUILD COMPLETE =========="
    )

    print(
        f"Output: "
        f"{OUT}"
    )

    print(
        f"Manifest: "
        f"{MANIFEST}"
    )

    print(
        f"Total rows: "
        f"{total_written:,}"
    )

    print(
        f"Benign ({config.LABEL_BENIGN}): "
        f"{label_total[config.LABEL_BENIGN]:,}"
    )

    print(
        f"Attack ({config.LABEL_ATTACK}): "
        f"{label_total[config.LABEL_ATTACK]:,}"
    )

    print(
        f"Features: "
        f"{len(FEATURES)} + "
        f"{config.LABEL_COLUMN}"
    )

    print(
        "Original source files were not modified."
    )


# ============================================================

if __name__ == "__main__":

    main()