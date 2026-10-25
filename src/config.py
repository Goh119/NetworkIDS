from pathlib import Path


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
)

RAW_DIR = (
    DATA_DIR
    / "raw"
)

PROCESSED_DIR = (
    DATA_DIR
    / "processed"
)

TRAIN_DIR = (
    DATA_DIR
    / "train"
)

TEST_DIR = (
    DATA_DIR
    / "test"
)

MODELS_DIR = (
    PROJECT_ROOT
    / "models"
)


# ============================================================
# Model Directories
#
# Primary final model:
#
#     random_forest/
#
# Comparison models:
#
#     decision_tree/
#     support_vector_machine/
#
# SHAP outputs remain under:
#
#     models/shap/
#
# No separate model-comparison directory is required.
# ============================================================

RANDOM_FOREST_DIR = (
    PROJECT_ROOT
    / "random_forest"
)

DECISION_TREE_DIR = (
    PROJECT_ROOT
    / "decision_tree"
)

SUPPORT_VECTOR_MACHINE_DIR = (
    PROJECT_ROOT
    / "support_vector_machine"
)


# ============================================================
# CIC-IDS2017 Days
#
# Used only by the historical multiday preparation pipeline.
#
# Raw CIC-IDS2017 PCAP files are no longer stored inside the
# project repository.
# ============================================================

CIC_DAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
]


# ============================================================
# Processed Multiday Labels
#
# Canonical CIC-IDS2017 label files used by the historical
# multiday preparation pipeline.
#
# These remain useful for reproducibility and rebuilding the
# original CIC multiday split.
# ============================================================

MULTIDAY_LABELS_DIR = (
    PROCESSED_DIR
    / "multiday_labels"
)


MULTIDAY_LABEL_FILES = {

    "Monday":
        MULTIDAY_LABELS_DIR
        / "Monday_combined.csv",

    "Tuesday":
        MULTIDAY_LABELS_DIR
        / "Tuesday_combined.csv",

    "Wednesday":
        MULTIDAY_LABELS_DIR
        / "Wednesday_combined.csv",

    "Thursday":
        MULTIDAY_LABELS_DIR
        / "Thursday_combined.csv",

    "Friday":
        MULTIDAY_LABELS_DIR
        / "Friday_combined.csv",
}


# ============================================================
# Processed Directories
# ============================================================

MULTIDAY_PACKET_DIR = (
    PROCESSED_DIR
    / "multiday_packets"
)


MULTIDAY_SPLIT_DIR = (
    PROCESSED_DIR
    / "multiday_split"
)


# ============================================================
# Final Machine-Learning Datasets
#
# training_dataset.csv
# --------------------
# Final enhanced training dataset.
#
# Used by:
#
#     Random Forest
#     Decision Tree
#     Support Vector Machine
#
#
# test_multiday.csv
# -----------------
# Final labelled held-out flow-level test dataset.
#
#
# test_multiday_packets.csv
# -------------------------
# Packet companion for test_multiday.csv.
#
# Join key:
#
#     (day, flow_id)
# ============================================================

TRAIN_DATASET_FILE = (
    TRAIN_DIR
    / "training_dataset.csv"
)


TRAIN_MANIFEST_FILE = (
    TRAIN_DIR
    / "training_dataset_manifest.csv"
)


TEST_DATASET_FILE = (
    TEST_DIR
    / "test_multiday.csv"
)


TEST_PACKET_DATASET_FILE = (
    TEST_DIR
    / "test_multiday_packets.csv"
)


# ============================================================
# Original CIC Multiday Training Split
#
# This is NOT the final enhanced training dataset.
#
# It is retained because it is one source used to build:
#
#     data/train/training_dataset.csv
# ============================================================

CIC_TRAIN_SPLIT_FILE = (
    MULTIDAY_SPLIT_DIR
    / "train_multiday.csv"
)


# ============================================================
# External / Unseen Test Discovery
#
# All flow-level CSV files under data/test matching:
#
#     *_flow.csv
#
# are treated as external unseen inference datasets.
#
# test_multiday.csv is added explicitly because it does not
# end with "_flow.csv".
# ============================================================

TEST_FLOW_GLOB = (
    "*_flow.csv"
)


TEST_ADDITIONAL_FLOW_FILES = [
    TEST_DATASET_FILE,
]


# ============================================================
# Standalone Preprocessing Defaults
#
# These are default working/output paths for standalone
# preprocessing scripts.
#
# They are NOT the final RF / DT / SVM train or test datasets.
#
# packet_parser.py:
#
#     PCAP
#       -> packet_features.csv
#
# flow_builder.py:
#
#     packet_features.csv
#       -> flow_features.csv
#
# label_matcher.py:
#
#     flow_features.csv
#       -> final_ml_dataset.csv
#
# These files do not need to exist permanently.
# ============================================================

PACKET_FEATURES_FILE = (
    PROCESSED_DIR
    / "packet_features.csv"
)


FLOW_FEATURES_FILE = (
    PROCESSED_DIR
    / "flow_features.csv"
)


FINAL_ML_DATASET_FILE = (
    PROCESSED_DIR
    / "final_ml_dataset.csv"
)


# ============================================================
# Random Forest Artifacts
#
# Primary classifier used by the final NetworkIDS system.
# ============================================================

RANDOM_FOREST_MODEL_FILE = (
    RANDOM_FOREST_DIR
    / "rf_model.joblib"
)


RANDOM_FOREST_META_FILE = (
    RANDOM_FOREST_DIR
    / "rf_model_meta.json"
)


RANDOM_FOREST_REPORT_FILE = (
    RANDOM_FOREST_DIR
    / "rf_model_report.txt"
)


# ============================================================
# Decision Tree Artifacts
#
# Comparison model only.
#
# It is not used by the final dashboard.
# ============================================================

DECISION_TREE_MODEL_FILE = (
    DECISION_TREE_DIR
    / "dt_model.joblib"
)


DECISION_TREE_META_FILE = (
    DECISION_TREE_DIR
    / "dt_model_meta.json"
)


DECISION_TREE_REPORT_FILE = (
    DECISION_TREE_DIR
    / "dt_model_report.txt"
)


# ============================================================
# Support Vector Machine Artifacts
#
# Comparison model only.
#
# It is not used by the final dashboard.
# ============================================================

SUPPORT_VECTOR_MACHINE_MODEL_FILE = (
    SUPPORT_VECTOR_MACHINE_DIR
    / "svm_model.joblib"
)


SUPPORT_VECTOR_MACHINE_META_FILE = (
    SUPPORT_VECTOR_MACHINE_DIR
    / "svm_model_meta.json"
)


SUPPORT_VECTOR_MACHINE_REPORT_FILE = (
    SUPPORT_VECTOR_MACHINE_DIR
    / "svm_model_report.txt"
)


# ============================================================
# Classification Labels
# ============================================================

LABEL_BENIGN = 0

LABEL_ATTACK = 1

LABEL_COLUMN = "label"


# ============================================================
# ML Features
#
# IMPORTANT:
#
# This list is the SINGLE SOURCE OF TRUTH for model input.
#
# Random Forest, Decision Tree and SVM must all use the same
# feature list for fair comparison.
#
# Do NOT hard-code the feature count elsewhere.
# ============================================================

ML_FEATURES = [

    # --------------------------------------------------------
    # L2
    # --------------------------------------------------------

    "frame_length_mean",
    "eth_type",
    "vlan_id",
    "vlan_priority",


    # --------------------------------------------------------
    # L3
    # --------------------------------------------------------

    "ip_version",
    "ip_dscp",
    "ip_ecn",
    "ip_total_length_mean",
    "ip_id_mean",
    "ip_df",
    "ip_mf",
    "fragment_offset",
    "ttl_mean",
    "ip_protocol",
    "ip_header_length",
    "ip_checksum_valid",


    # --------------------------------------------------------
    # L4
    # --------------------------------------------------------

    "src_port",
    "dst_port",
    "tcp_window_mean",
    "tcp_header_length_mean",
    "l4_checksum_valid",


    # --------------------------------------------------------
    # Flow
    # --------------------------------------------------------

    "flow_duration",
    "flow_packet_count",
    "flow_byte_count",
    "forward_packet_count",
    "reverse_packet_count",
    "forward_byte_count",
    "reverse_byte_count",
    "iat_min",
    "iat_mean",
    "iat_max",
    "iat_std",
    "byte_rate",
    "flow_direction",
    "syn_count",
    "ack_count",
    "fin_count",
    "rst_count",
    "psh_count",
    "syn_seen",
    "syn_ack_seen",
    "ack_after_syn_ack",
]


# ============================================================
# Metadata Columns
#
# Used for traceability / investigation only.
#
# They are NOT ML features.
# ============================================================

METADATA_COLUMNS = [

    "flow_id",
    "timestamp",
    "src_ip",
    "dst_ip",
    "src_mac",
    "dst_mac",
]


# ============================================================
# MAC-IP Packet Integrity Columns
# ============================================================

MAC_IP_PACKET_COLUMNS = [

    "src_ip_mac_consistent",
    "src_ip_mac_seen_mac_count",
    "src_mac_ip_seen_ip_count",
    "src_mac_ip_spoof_suspect",
]


# ============================================================
# MAC-IP Flow Integrity Columns
# ============================================================

MAC_IP_FLOW_COLUMNS = [

    "flow_mac_ip_inconsistent_count",
    "flow_mac_ip_spoof_suspect",
    "src_ip_mac_seen_mac_count_max",
    "src_mac_ip_seen_ip_count_max",
]


# ============================================================
# Integrity Columns
# ============================================================

INTEGRITY_COLUMNS = [

    "ip_checksum",
    "l4_checksum",

    "ip_checksum_valid",
    "l4_checksum_valid",

    "src_mac",
    "dst_mac",

    "src_ip",
    "dst_ip",

    "ttl",

    "tcp_syn",
    "tcp_ack",
    "tcp_fin",
    "tcp_rst",
    "tcp_psh",
    "tcp_urg",

    "ip_df",
    "ip_mf",

    "fragment_offset",

    "ip_dscp",

    *MAC_IP_PACKET_COLUMNS,

    *MAC_IP_FLOW_COLUMNS,
]


# ============================================================
# Flow Settings
# ============================================================

FLOW_IDLE_TIMEOUT = 120.0


# ============================================================
# Directory Utilities
# ============================================================

def ensure_directories():
    """
    Create project-owned working/output directories.

    Source datasets themselves are not generated here.
    """

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TRAIN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RANDOM_FOREST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DECISION_TREE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUPPORT_VECTOR_MACHINE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MULTIDAY_LABELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MULTIDAY_PACKET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MULTIDAY_SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# ML Feature Validation
# ============================================================

def validate_ml_features():
    """
    Validate the shared model feature definition.
    """

    if len(
        ML_FEATURES
    ) == 0:

        raise ValueError(
            "ML_FEATURES must contain at least one feature."
        )

    if len(
        set(
            ML_FEATURES
        )
    ) != len(
        ML_FEATURES
    ):

        raise ValueError(
            "Duplicate feature names detected in ML_FEATURES."
        )

    forbidden_features = {

        "flow_id",
        "timestamp",

        "src_ip",
        "dst_ip",

        "src_mac",
        "dst_mac",

        "day",
    }

    overlap = (
        set(
            ML_FEATURES
        )
        &
        forbidden_features
    )

    if overlap:

        raise ValueError(
            "Metadata fields must not appear in ML_FEATURES:\n"
            + "\n".join(
                f"  - {feature}"
                for feature
                in sorted(
                    overlap
                )
            )
        )

    return True


# ============================================================
# MAC-IP Column Validation
# ============================================================

def validate_mac_ip_columns():
    """
    Validate packet-level and flow-level MAC-IP integrity
    column definitions.
    """

    if len(
        MAC_IP_PACKET_COLUMNS
    ) != len(
        set(
            MAC_IP_PACKET_COLUMNS
        )
    ):

        raise ValueError(
            "Duplicate packet-level MAC-IP columns detected."
        )

    if len(
        MAC_IP_FLOW_COLUMNS
    ) != len(
        set(
            MAC_IP_FLOW_COLUMNS
        )
    ):

        raise ValueError(
            "Duplicate flow-level MAC-IP columns detected."
        )

    overlap = (
        set(
            ML_FEATURES
        )
        &
        (
            set(
                MAC_IP_PACKET_COLUMNS
            )
            |
            set(
                MAC_IP_FLOW_COLUMNS
            )
        )
    )

    if overlap:

        raise ValueError(
            "MAC-IP integrity columns unexpectedly appear "
            "inside ML_FEATURES:\n"
            + "\n".join(
                f"  - {feature}"
                for feature
                in sorted(
                    overlap
                )
            )
        )

    return True


# ============================================================
# Primary Dataset Validation
# ============================================================

def validate_training_files():
    """
    Validate final train/test datasets used by the ML models.
    """

    required_files = {

        "Final training dataset":
            TRAIN_DATASET_FILE,

        "Held-out test dataset":
            TEST_DATASET_FILE,
    }

    missing = []

    for name, path in (
        required_files.items()
    ):

        if not path.exists():

            missing.append(
                f"{name}: {path}"
            )

    if missing:

        print(
            "Missing primary ML dataset files:"
        )

        for item in missing:

            print(
                f"  - {item}"
            )

        return False

    return True


# ============================================================
# Training Manifest Validation
# ============================================================

def validate_training_manifest():
    """
    Check whether the final training dataset provenance
    manifest is available.
    """

    if not TRAIN_MANIFEST_FILE.exists():

        print(
            "Missing training dataset manifest:"
        )

        print(
            f"  - {TRAIN_MANIFEST_FILE}"
        )

        return False

    return True


# ============================================================
# Test Packet Companion Validation
# ============================================================

def validate_test_packet_file():
    """
    Check whether the packet companion for test_multiday.csv
    is available.
    """

    if not TEST_PACKET_DATASET_FILE.exists():

        print(
            "Missing test packet companion:"
        )

        print(
            f"  - {TEST_PACKET_DATASET_FILE}"
        )

        return False

    return True


# ============================================================
# Configuration Check
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 70
    )

    print(
        "NetworkIDS Configuration Check"
    )

    print(
        "=" * 70
    )

    ensure_directories()

    # ========================================================
    # Project paths
    # ========================================================

    print()

    print(
        "Project:"
    )

    print(
        f"  Root      : "
        f"{PROJECT_ROOT}"
    )

    print(
        f"  Data      : "
        f"{DATA_DIR}"
    )

    print(
        f"  Raw       : "
        f"{RAW_DIR}"
    )

    print(
        f"  Processed : "
        f"{PROCESSED_DIR}"
    )

    print(
        f"  Train     : "
        f"{TRAIN_DIR}"
    )

    print(
        f"  Test      : "
        f"{TEST_DIR}"
    )

    # ========================================================
    # CIC preparation
    # ========================================================

    print()

    print(
        "CIC multiday preparation:"
    )

    print(
        f"  CIC train split: "
        f"{CIC_TRAIN_SPLIT_FILE}"
    )

    print()

    print(
        "Multiday label files:"
    )

    for day in CIC_DAYS:

        print(
            f"  {day:<10}: "
            f"{MULTIDAY_LABEL_FILES[day]}"
        )

    # ========================================================
    # Final ML datasets
    # ========================================================

    print()

    print(
        "Final ML datasets:"
    )

    print(
        f"  Train    : "
        f"{TRAIN_DATASET_FILE}"
    )

    print(
        f"  Manifest : "
        f"{TRAIN_MANIFEST_FILE}"
    )

    print(
        f"  Test     : "
        f"{TEST_DATASET_FILE}"
    )

    print(
        f"  Packets  : "
        f"{TEST_PACKET_DATASET_FILE}"
    )

    # ========================================================
    # Feature validation
    # ========================================================

    print()

    validate_ml_features()

    print(
        f"ML features: OK "
        f"({len(ML_FEATURES)} features)"
    )

    validate_mac_ip_columns()

    print(
        "MAC-IP integrity columns: OK"
    )

    # ========================================================
    # Dataset validation
    # ========================================================

    print()

    datasets_ok = (
        validate_training_files()
    )

    manifest_ok = (
        validate_training_manifest()
    )

    packet_test_ok = (
        validate_test_packet_file()
    )

    print()

    print(
        "Dataset availability:"
    )

    print(
        f"  Train/test ready : "
        f"{'YES' if datasets_ok else 'NO'}"
    )

    print(
        f"  Train manifest   : "
        f"{'YES' if manifest_ok else 'NO'}"
    )

    print(
        f"  Packet companion : "
        f"{'YES' if packet_test_ok else 'NO'}"
    )

    # ========================================================
    # Random Forest
    # ========================================================

    print()

    print(
        "Random Forest:"
    )

    print(
        f"  Directory : "
        f"{RANDOM_FOREST_DIR}"
    )

    print(
        f"  Model     : "
        f"{RANDOM_FOREST_MODEL_FILE}"
    )

    print(
        f"  Meta      : "
        f"{RANDOM_FOREST_META_FILE}"
    )

    print(
        f"  Report    : "
        f"{RANDOM_FOREST_REPORT_FILE}"
    )

    # ========================================================
    # Decision Tree
    # ========================================================

    print()

    print(
        "Decision Tree:"
    )

    print(
        f"  Directory : "
        f"{DECISION_TREE_DIR}"
    )

    print(
        f"  Model     : "
        f"{DECISION_TREE_MODEL_FILE}"
    )

    print(
        f"  Meta      : "
        f"{DECISION_TREE_META_FILE}"
    )

    print(
        f"  Report    : "
        f"{DECISION_TREE_REPORT_FILE}"
    )

    # ========================================================
    # Support Vector Machine
    # ========================================================

    print()

    print(
        "Support Vector Machine:"
    )

    print(
        f"  Directory : "
        f"{SUPPORT_VECTOR_MACHINE_DIR}"
    )

    print(
        f"  Model     : "
        f"{SUPPORT_VECTOR_MACHINE_MODEL_FILE}"
    )

    print(
        f"  Meta      : "
        f"{SUPPORT_VECTOR_MACHINE_META_FILE}"
    )

    print(
        f"  Report    : "
        f"{SUPPORT_VECTOR_MACHINE_REPORT_FILE}"
    )

    # ========================================================
    # Completed
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "Configuration check completed."
    )

    print(
        "=" * 70
    )