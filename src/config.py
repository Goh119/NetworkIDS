from pathlib import Path


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TEST_DIR = DATA_DIR / "test"

MODELS_DIR = PROJECT_ROOT / "models"


# ============================================================
# ==================== ADDED ================================
# Random Forest Directory
#
# Keep the Random Forest training script and all primary
# Random Forest artifacts together in one dedicated folder:
#
#     random_forest/
#         train_rf.py
#         rf_model.joblib
#         rf_model_meta.json
#         rf_model_report.txt
#
# SHAP outputs remain under:
#
#     models/shap/
# ============================================================

RANDOM_FOREST_DIR = (
    PROJECT_ROOT
    / "random_forest"
)


# ============================================================
# CIC-IDS2017 Days
# ============================================================

CIC_DAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
]


# ============================================================
# Raw CIC-IDS2017 PCAP Files
#
# These are source/reference files used during dataset
# preparation.
#
# User-uploaded PCAP files must be supplied by the runtime
# application and should NOT modify config.py.
# ============================================================

CIC_PCAP_FILES = {

    "Monday":
        RAW_DIR / "Monday-WorkingHours.pcap",

    "Tuesday":
        RAW_DIR / "Tuesday-WorkingHours.pcap",

    "Wednesday":
        RAW_DIR / "Wednesday-workingHours.pcap",

    "Thursday":
        RAW_DIR / "Thursday-WorkingHours.pcap",

    "Friday":
        RAW_DIR / "Friday-WorkingHours.pcap",
}


# Backward-compatible default/reference PCAP.
#
# This is only a default development input.
# It is NOT the primary multiday training input.

DEFAULT_CIC_DAY = "Wednesday"

PCAP_FILE = CIC_PCAP_FILES[
    DEFAULT_CIC_DAY
]


# ============================================================
# Raw GeneratedLabelledFlows
#
# Original CIC-IDS2017 official label files.
#
# These are kept as source/provenance files.
# ============================================================

RAW_LABELLED_FLOWS_DIR = (
    RAW_DIR
    / "GeneratedLabelledFlows"
    / "TrafficLabelling"
)


# ============================================================
# Processed Multiday Labels
#
# These are the canonical label files used by the multiday
# preparation pipeline.
#
# Thursday and Friday have already been combined.
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


# Backward-compatible Wednesday label reference.

LABELLED_WEDNESDAY_FILE = (
    MULTIDAY_LABEL_FILES[
        "Wednesday"
    ]
)


# ============================================================
# Official MachineLearningCSV Reference
#
# This is a reference/sanity-check source only.
# It is NOT the primary training dataset.
# ============================================================

ML_REFERENCE_DIR = (
    RAW_DIR
    / "MachineLearningCSV"
    / "MachineLearningCVE"
)


ML_REFERENCE_WEDNESDAY_FILE = (
    ML_REFERENCE_DIR
    / "Wednesday-workingHours.pcap_ISCX.csv"
)


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
# Primary Train/Test Dataset
# ============================================================

TRAIN_DATASET_FILE = (
    MULTIDAY_SPLIT_DIR
    / "train_multiday.csv"
)


TEST_DATASET_FILE = (
    MULTIDAY_SPLIT_DIR
    / "test_multiday.csv"
)


SPLIT_SUMMARY_FILE = (
    MULTIDAY_SPLIT_DIR
    / "split_summary.txt"
)


# ============================================================
# Legacy / Intermediate Processed Files
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
# Random Forest Outputs
#
# ==================== CHANGED ===============================
#
# Random Forest artifacts are now stored together under:
#
#     random_forest/
#
# Files:
#
#   rf_model.joblib
#       Bare trained sklearn model.
#
#   rf_model_meta.json
#       Reproducibility and model metadata.
#
#   rf_model_report.txt
#       Human-readable evaluation report.
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
# This list is the SINGLE SOURCE OF TRUTH for the model input.
#
# Do NOT hard-code the feature count anywhere else.
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

    TEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # ==================== ADDED ==============================
    # Dedicated Random Forest directory.
    # ========================================================

    RANDOM_FOREST_DIR.mkdir(
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

    if len(ML_FEATURES) == 0:

        raise ValueError(
            "ML_FEATURES must contain at least one feature."
        )


    if len(
        set(ML_FEATURES)
    ) != len(ML_FEATURES):

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
        set(ML_FEATURES)
        &
        forbidden_features
    )


    if overlap:

        raise ValueError(
            "Metadata fields must not appear in ML_FEATURES:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in sorted(overlap)
            )
        )


    return True


# ============================================================
# MAC-IP Column Validation
# ============================================================

def validate_mac_ip_columns():

    if len(
        MAC_IP_PACKET_COLUMNS
    ) != len(
        set(MAC_IP_PACKET_COLUMNS)
    ):

        raise ValueError(
            "Duplicate packet-level MAC-IP columns detected."
        )


    if len(
        MAC_IP_FLOW_COLUMNS
    ) != len(
        set(MAC_IP_FLOW_COLUMNS)
    ):

        raise ValueError(
            "Duplicate flow-level MAC-IP columns detected."
        )


    overlap = (
        set(ML_FEATURES)
        &
        (
            set(MAC_IP_PACKET_COLUMNS)
            |
            set(MAC_IP_FLOW_COLUMNS)
        )
    )


    if overlap:

        raise ValueError(
            "MAC-IP integrity columns unexpectedly appear "
            "inside ML_FEATURES:\n"
            + "\n".join(
                f"  - {feature}"
                for feature in sorted(overlap)
            )
        )


    return True


# ============================================================
# Primary Training File Validation
# ============================================================

def validate_training_files():

    required_files = {

        "Train dataset":
            TRAIN_DATASET_FILE,

        "Test dataset":
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
            "Missing primary training files:"
        )

        for item in missing:

            print(
                f"  - {item}"
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

    print(
        f"Project root: "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Default CIC day: "
        f"{DEFAULT_CIC_DAY}"
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

    print()

    print(
        "Primary datasets:"
    )

    print(
        f"  Train: {TRAIN_DATASET_FILE}"
    )

    print(
        f"  Test : {TEST_DATASET_FILE}"
    )

    print()


    # ========================================================
    # ==================== CHANGED ============================
    # Show the dedicated RF directory and its artifacts.
    # ========================================================

    print(
        "Random Forest:"
    )

    print(
        f"  Directory: {RANDOM_FOREST_DIR}"
    )

    print(
        f"  Model    : {RANDOM_FOREST_MODEL_FILE}"
    )

    print(
        f"  Meta     : {RANDOM_FOREST_META_FILE}"
    )

    print(
        f"  Report   : {RANDOM_FOREST_REPORT_FILE}"
    )

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