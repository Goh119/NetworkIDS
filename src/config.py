from pathlib import Path

# ============================================================
# Project Paths
# Project root:
# C:\FYP\NetworkIDS
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TEST_DIR = DATA_DIR / "test"

MODELS_DIR = PROJECT_ROOT / "models"

# ============================================================
# Raw Input Files
# Original CIC-IDS2017 PCAP used for packet-level extraction
PCAP_FILE = RAW_DIR / "Wednesday-workingHours.pcap"

# ------------------------------------------------------------
# GeneratedLabelledFlows
# Official labelled flow CSV.
# This is the primary ground-truth source for our labels.
LABELLED_FLOWS_DIR = (
    RAW_DIR
    / "GeneratedLabelledFlows"
    / "TrafficLabelling"
)

LABELLED_WEDNESDAY_FILE = (
    LABELLED_FLOWS_DIR
    / "Wednesday-workingHours.pcap_ISCX.csv"
)

# ------------------------------------------------------------
# MachineLearningCSV
# Official CIC-IDS2017 machine-learning CSV.
# This is NOT our primary training dataset.
# It is kept as a reference/sanity-check dataset.
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
# Processed Output Files
# Packet-level extracted features
PACKET_FEATURES_FILE = (
    PROCESSED_DIR / "packet_features.csv"
)

# Flow-level features before label matching
FLOW_FEATURES_FILE = (
    PROCESSED_DIR / "flow_features.csv"
)

# Final labelled dataset used by the ML pipeline
FINAL_ML_DATASET_FILE = (
    PROCESSED_DIR / "final_ml_dataset.csv"
)

# ============================================================
# Model Output
RANDOM_FOREST_MODEL_FILE = (
    MODELS_DIR / "random_forest.pkl"
)

# ============================================================
# Classification Labels
# Binary classification:
# 0 = Benign
# 1 = Attack

LABEL_BENIGN = 0
LABEL_ATTACK = 1

LABEL_COLUMN = "label"

# ============================================================
# Final ML Feature Columns

# IMPORTANT:
# These are the features that are allowed to enter
# the Random Forest model.
# Metadata, raw IP/MAC addresses, raw checksums,
# integrity results and rule-based risk scores are
# intentionally excluded from this list.

ML_FEATURES = [
    # --------------------------------------------------------
    # L2 - Data Link Layer
    "frame_length_mean",
    "eth_type",
    "vlan_id",
    "vlan_priority",

    # --------------------------------------------------------
    # L3 - Network Layer
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
    # L4 - Transport Layer
    "src_port",
    "dst_port",
    "tcp_window_mean",
    "tcp_header_length_mean",
    "l4_checksum_valid",

    #"tcp_syn",
    #"tcp_ack",
    #"tcp_fin",
    #"tcp_rst",
    #"tcp_psh",
    #"tcp_urg",

    # --------------------------------------------------------
    # Flow Features
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
# These columns are useful for tracing, debugging,
# investigation and displaying results.
# They are NOT used as Random Forest input features.

METADATA_COLUMNS = [
    "flow_id",
    "timestamp",
    "src_ip",
    "dst_ip",
    "src_mac",
    "dst_mac",
]

# ============================================================
# Integrity-Only Columns
# These fields are extracted because they are needed
# by the independent packet header integrity checker.
# They are NOT Random Forest features, except for the
# *_valid Boolean indicators already included in ML_FEATURES.

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
]

# ============================================================
# PCAP / Flow Settings

# Maximum allowed idle time between packets before
# considering the traffic to belong to a new flow.
# This value will be used by flow_builder.py.
# NOTE:
# We will validate/tune this against the Wednesday PCAP
# and the official CIC flow reference before finalising
# the flow-building behaviour.
FLOW_IDLE_TIMEOUT = 120.0

# ============================================================
# Utility Functions
def ensure_directories():
    """
    Create project directories required by the pipeline.
    """
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    TEST_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def validate_input_files():
    """
    Check whether the required input files exist.
    """
    required_files = {
        "PCAP": PCAP_FILE,
        "Labelled Wednesday CSV": LABELLED_WEDNESDAY_FILE,
        "ML Reference Wednesday CSV": ML_REFERENCE_WEDNESDAY_FILE,
    }

    missing_files = []

    for name, path in required_files.items():

        if not path.exists():
            missing_files.append(
                f"{name}: {path}"
            )

    if missing_files:
        print("Missing input files:")

        for file_info in missing_files:
            print(f"  - {file_info}")

        return False

    return True


def validate_ml_features():
    """
    Check the final ML feature list.
    """
    expected_feature_count = 42

    actual_feature_count = len(ML_FEATURES)

    if actual_feature_count != expected_feature_count:

        raise ValueError(
            f"Expected {expected_feature_count} ML features, "
            f"but found {actual_feature_count}."
        )

    if len(set(ML_FEATURES)) != actual_feature_count:

        raise ValueError(
            "Duplicate feature names detected in ML_FEATURES."
        )

    return True


# ============================================================
# Main Test
# ============================================================
if __name__ == "__main__":

    print("=" * 60)
    print("NetworkIDS Configuration Check")
    print("=" * 60)

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("PCAP:")
    print(PCAP_FILE)

    print()
    print("Labelled flow CSV:")
    print(LABELLED_WEDNESDAY_FILE)

    print()
    print("ML reference CSV:")
    print(ML_REFERENCE_WEDNESDAY_FILE)

    print()
    print("Processed output directory:")
    print(PROCESSED_DIR)

    print()
    print("Checking directories...")
    ensure_directories()
    print("Directories: OK")

    print()
    print("Checking input files...")

    if validate_input_files():
        print("Input files: OK")
    else:
        print("Input files: FAILED")

    print()
    print("Checking ML feature list...")
    validate_ml_features()
    print(
        f"ML features: OK ({len(ML_FEATURES)} features)"
    )

    print()
    print("Binary labels:")
    print(f"  Benign = {LABEL_BENIGN}")
    print(f"  Attack = {LABEL_ATTACK}")

    print()
    print("=" * 60)
    print("Configuration check completed.")
    print("=" * 60)