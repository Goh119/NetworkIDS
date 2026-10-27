"""
app.py
------
NetworkIDS - Flask Dashboard Backend.

Single-page FYP prototype for offline PCAP analysis.

Supported runtime mode:
    Upload a PCAP / PCAPNG file.

Runtime pipeline:
    PCAP
        -> packet_parser.py
        -> packet_features.csv
        -> flow_builder.py
        -> flow_features.csv
        -> Packet Integrity Checker
        -> Random Forest classification

Packet-level integrity checking is performed independently
using src.integrity_checker.apply_rules().

SHAP explanations are calculated lazily only when the user
opens the explanation section for a selected flow.

Only the latest analysis is retained inside runtime/current/.

No database is used.
"""

from pathlib import Path
import json
import math
import shutil
import subprocess
import sys
import threading
import time

import joblib
import numpy as np
import pandas as pd

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
)

from werkzeug.utils import secure_filename


# ============================================================
# Project imports
# ============================================================

import src.config as config

from src.integrity_checker import (
    apply_rules,
    NEEDED_COLUMNS as INTEGRITY_NEEDED_COLUMNS,
    ALL_RULES,
)


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent


RUNTIME_DIR = (
    PROJECT_ROOT
    / "runtime"
)


CURRENT_RUNTIME_DIR = (
    RUNTIME_DIR
    / "current"
)


# ============================================================
# Runtime files
# ============================================================

RUNTIME_PACKET_FEATURES = (
    CURRENT_RUNTIME_DIR
    / "packet_features.csv"
)

RUNTIME_FLOW_FEATURES = (
    CURRENT_RUNTIME_DIR
    / "flow_features.csv"
)

RUNTIME_PREDICTIONS = (
    CURRENT_RUNTIME_DIR
    / "flow_predictions.csv"
)

RUNTIME_PACKET_INTEGRITY = (
    CURRENT_RUNTIME_DIR
    / "packet_integrity.csv"
)

RUNTIME_SUMMARY = (
    CURRENT_RUNTIME_DIR
    / "analysis_summary.json"
)


# ============================================================
# General settings
# ============================================================

ALLOWED_EXTENSIONS = {
    ".pcap",
    ".pcapng",
}

MAX_UPLOAD_BYTES = (
    512
    * 1024
    * 1024
)

INTEGRITY_CHUNK_SIZE = 250_000

DEFAULT_PAGE_SIZE = 25

MAX_PAGE_SIZE = 100

RELATED_PACKET_LIMIT = 100


# ============================================================
# Friendly SHAP feature names
# ============================================================

FEATURE_LABELS = {

    "frame_length_mean":
        "average frame length",

    "ip_total_length_mean":
        "average IP packet length",

    "ttl_mean":
        "average TTL",

    "src_port":
        "source port",

    "dst_port":
        "destination port",

    "tcp_header_length_mean":
        "average TCP header length",

    "tcp_window_mean":
        "average TCP window size",

    "flow_packet_count":
        "flow packet count",

    "flow_byte_count":
        "flow byte count",

    "forward_packet_count":
        "forward packet count",

    "reverse_packet_count":
        "reverse packet count",

    "forward_byte_count":
        "forward byte count",

    "reverse_byte_count":
        "reverse byte count",

    "flow_duration":
        "flow duration",

    "iat_mean":
        "average packet inter-arrival time",

    "iat_min":
        "minimum packet inter-arrival time",

    "iat_max":
        "maximum packet inter-arrival time",

    "byte_rate":
        "byte rate",

    "packet_rate":
        "packet rate",

    "syn_count":
        "SYN flag count",

    "ack_count":
        "ACK flag count",

    "fin_count":
        "FIN flag count",

    "rst_count":
        "RST flag count",

    "psh_count":
        "PSH flag count",

    "urg_count":
        "URG flag count",

    "ip_id_mean":
        "average IPv4 identification value",

    "ip_header_length_mean":
        "average IP header length",

    "fragment_offset_mean":
        "average fragmentation offset",

    "ip_dscp":
        "DSCP value",

    "ip_ecn":
        "ECN value",

    "ip_version":
        "IP version",

    "ip_protocol":
        "IP protocol",
}


# ============================================================
# Flask
# ============================================================

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)


app.config[
    "MAX_CONTENT_LENGTH"
] = MAX_UPLOAD_BYTES


# ============================================================
# Ensure directories
# ============================================================

RUNTIME_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Load Random Forest once
# ============================================================

MODEL_FILE = Path(
    config.RANDOM_FOREST_MODEL_FILE
)


if not MODEL_FILE.exists():

    raise FileNotFoundError(
        "Random Forest model not found:\n"
        f"{MODEL_FILE}"
    )


MODEL = joblib.load(
    MODEL_FILE
)


# ============================================================
# Validate model
# ============================================================

if hasattr(
    MODEL,
    "n_features_in_",
):

    if (
        int(
            MODEL.n_features_in_
        )
        !=
        len(
            config.ML_FEATURES
        )
    ):

        raise RuntimeError(
            "Random Forest/config feature-count mismatch."
        )


if hasattr(
    MODEL,
    "feature_names_in_",
):

    model_features = list(
        MODEL.feature_names_in_
    )

    if (
        model_features
        !=
        list(
            config.ML_FEATURES
        )
    ):

        raise RuntimeError(
            "Random Forest/config feature-order mismatch."
        )


if not hasattr(
    MODEL,
    "classes_",
):

    raise RuntimeError(
        "Random Forest model does not contain classes_."
    )


MODEL_CLASSES = list(
    MODEL.classes_
)


if (
    config.LABEL_ATTACK
    not in MODEL_CLASSES
):

    raise RuntimeError(
        "Configured Attack label was not found "
        "inside model.classes_."
    )


ATTACK_CLASS_INDEX = (
    MODEL_CLASSES.index(
        config.LABEL_ATTACK
    )
)


# ============================================================
# Application state
# ============================================================

STATE_LOCK = threading.Lock()

ANALYSIS_CONTROL_LOCK = (
    threading.Lock()
)


APP_STATE = {

    "ready":
        False,

    "analyzing":
        False,

    "source_name":
        None,

    "flow_df":
        None,

    "packet_integrity_path":
        None,

    "summary":
        None,

    "shap_cache":
        {},

    "packet_cache":
        {},

    # Analysis progress
    "stage":
        None,

    "stage_message":
        None,

    "started_at":
        None,

    "elapsed_seconds":
        0.0,

    "error":
        None,
}


# ============================================================
# SHAP state
# ============================================================

SHAP_EXPLAINER = None

SHAP_LOCK = threading.Lock()


# ============================================================
# Utility helpers
# ============================================================

def json_safe(
    value,
):

    if value is None:

        return None


    if isinstance(
        value,
        (
            np.integer,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
        ),
    ):

        return int(
            value
        )


    if isinstance(
        value,
        (
            np.floating,
            np.float16,
            np.float32,
            np.float64,
        ),
    ):

        if np.isnan(
            value
        ):

            return None


        if np.isinf(
            value
        ):

            return None


        return float(
            value
        )


    try:

        if pd.isna(
            value
        ):

            return None

    except (
        TypeError,
        ValueError,
    ):

        pass


    return value


def safe_float(
    value,
):

    value = json_safe(
        value
    )


    if value is None:

        return None


    try:

        return float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return None


def safe_int(
    value,
):

    value = json_safe(
        value
    )


    if value is None:

        return None


    try:

        return int(
            float(
                value
            )
        )

    except (
        TypeError,
        ValueError,
    ):

        return None


def format_protocol(
    ip_protocol,
):

    protocol = safe_int(
        ip_protocol
    )


    mapping = {

        1:
            "ICMP",

        6:
            "TCP",

        17:
            "UDP",

        58:
            "ICMPv6",
    }


    if protocol in mapping:

        return mapping[
            protocol
        ]


    if protocol is None:

        return "Unknown"


    return str(
        protocol
    )


def label_to_text(
    label,
):

    label = int(
        label
    )


    if (
        label
        ==
        config.LABEL_ATTACK
    ):

        return "Attack"


    if (
        label
        ==
        config.LABEL_BENIGN
    ):

        return "Benign"


    return str(
        label
    )


def allowed_file(
    filename,
):

    suffix = (
        Path(
            filename
        )
        .suffix
        .lower()
    )


    return (
        suffix
        in
        ALLOWED_EXTENSIONS
    )


def get_feature_label(
    feature,
):

    if feature in FEATURE_LABELS:

        return FEATURE_LABELS[
            feature
        ]


    return (
        feature
        .replace(
            "_",
            " ",
        )
        .strip()
        .lower()
    )


def format_feature_value_for_text(
    value,
):

    value = safe_float(
        value
    )


    if value is None:

        return "an unavailable value"


    if float(
        value
    ).is_integer():

        return f"{int(value):,}"


    absolute = abs(
        value
    )


    if (
        absolute > 0
        and
        absolute < 0.0001
    ):

        return (
            f"{value:.8f}"
            .rstrip(
                "0"
            )
            .rstrip(
                "."
            )
        )


    return (
        f"{value:.2f}"
        .rstrip(
            "0"
        )
        .rstrip(
            "."
        )
    )


# ============================================================
# Runtime state helpers
# ============================================================

def reset_runtime_directory():

    if CURRENT_RUNTIME_DIR.exists():

        shutil.rmtree(
            CURRENT_RUNTIME_DIR
        )


    CURRENT_RUNTIME_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def reset_application_state():

    with STATE_LOCK:

        APP_STATE.update({

            "ready":
                False,

            "analyzing":
                False,

            "source_name":
                None,

            "flow_df":
                None,

            "packet_integrity_path":
                None,

            "summary":
                None,

            "shap_cache":
                {},

            "packet_cache":
                {},

            "stage":
                None,

            "stage_message":
                None,

            "started_at":
                None,

            "elapsed_seconds":
                0.0,

            "error":
                None,
        })


def get_elapsed_seconds_locked():

    started_at = APP_STATE[
        "started_at"
    ]


    if (
        APP_STATE[
            "analyzing"
        ]
        and
        started_at is not None
    ):

        return max(
            0.0,
            time.monotonic()
            -
            started_at,
        )


    return float(
        APP_STATE[
            "elapsed_seconds"
        ]
        or
        0.0
    )


def update_analysis_stage(
    stage,
    message,
):

    with STATE_LOCK:

        APP_STATE[
            "stage"
        ] = stage

        APP_STATE[
            "stage_message"
        ] = message


def mark_analysis_started(
    source_name,
):

    with STATE_LOCK:

        APP_STATE[
            "ready"
        ] = False

        APP_STATE[
            "analyzing"
        ] = True

        APP_STATE[
            "source_name"
        ] = source_name

        APP_STATE[
            "stage"
        ] = "extract"

        APP_STATE[
            "stage_message"
        ] = (
            "Extracting packet and header features..."
        )

        APP_STATE[
            "started_at"
        ] = time.monotonic()

        APP_STATE[
            "elapsed_seconds"
        ] = 0.0

        APP_STATE[
            "error"
        ] = None


def mark_analysis_failed(
    error_message,
):

    with STATE_LOCK:

        elapsed = (
            get_elapsed_seconds_locked()
        )

        APP_STATE[
            "elapsed_seconds"
        ] = elapsed

        APP_STATE[
            "analyzing"
        ] = False

        APP_STATE[
            "ready"
        ] = False

        APP_STATE[
            "error"
        ] = str(
            error_message
        )

        APP_STATE[
            "stage_message"
        ] = (
            "Analysis stopped because an error occurred."
        )


# ============================================================
# External script execution
# ============================================================

def run_command(
    command,
    stage_name,
):

    completed = subprocess.run(
        command,
        cwd=str(
            PROJECT_ROOT
        ),
        capture_output=True,
        text=True,
    )


    if (
        completed.returncode
        !=
        0
    ):

        stdout_tail = (
            completed.stdout[
                -4000:
            ]
            if completed.stdout
            else ""
        )


        stderr_tail = (
            completed.stderr[
                -4000:
            ]
            if completed.stderr
            else ""
        )


        raise RuntimeError(
            f"{stage_name} failed.\n\n"
            f"STDOUT:\n{stdout_tail}\n\n"
            f"STDERR:\n{stderr_tail}"
        )


    return completed


# ============================================================
# Packet Parser
# ============================================================

def run_packet_parser(
    pcap_path,
    output_csv,
):

    command = [

        sys.executable,

        str(
            PROJECT_ROOT
            / "src"
            / "packet_parser.py"
        ),

        "--pcap",
        str(
            pcap_path
        ),

        "--limit",
        "0",

        "--preview",
        "0",

        "--out",
        str(
            output_csv
        ),
    ]


    run_command(
        command,
        "Packet parser",
    )


    if not output_csv.exists():

        raise RuntimeError(
            "Packet parser completed but did not "
            "create packet_features.csv."
        )


# ============================================================
# Flow Builder
# ============================================================

def run_flow_builder(
    packet_csv,
    flow_csv,
):

    command = [

        sys.executable,

        str(
            PROJECT_ROOT
            / "src"
            / "flow_builder.py"
        ),

        "--in",
        str(
            packet_csv
        ),

        "--out",
        str(
            flow_csv
        ),
    ]


    run_command(
        command,
        "Flow builder",
    )


    if not flow_csv.exists():

        raise RuntimeError(
            "Flow builder completed but did not "
            "create flow_features.csv."
        )


# ============================================================
# Runtime packet integrity
# ============================================================

def run_runtime_integrity(
    packet_csv,
    output_csv,
):

    header = pd.read_csv(
        packet_csv,
        nrows=0,
    )


    missing = [

        column

        for column
        in INTEGRITY_NEEDED_COLUMNS

        if column
        not in
        header.columns
    ]


    if missing:

        raise ValueError(
            "Runtime packet features are missing "
            "integrity columns:\n"
            +
            "\n".join(
                f"  - {column}"
                for column
                in missing
            )
        )


    if output_csv.exists():

        output_csv.unlink()


    state_counts = {

        "Intact":
            0,

        "Suspicious":
            0,

        "Tampered":
            0,
    }


    rule_counts = {

        rule:
            0

        for rule
        in ALL_RULES
    }


    total_packets = 0

    first_write = True


    reader = pd.read_csv(
        packet_csv,
        usecols=INTEGRITY_NEEDED_COLUMNS,
        chunksize=INTEGRITY_CHUNK_SIZE,
        low_memory=False,
    )


    for chunk in reader:

        chunk = apply_rules(
            chunk
        )


        total_packets += len(
            chunk
        )


        for rule in ALL_RULES:

            rule_counts[
                rule
            ] += int(
                chunk[
                    rule
                ]
                .sum()
            )


        for state in state_counts:

            state_counts[
                state
            ] += int(
                (
                    chunk[
                        "integrity_state"
                    ]
                    ==
                    state
                )
                .sum()
            )


        chunk.to_csv(
            output_csv,
            mode=(
                "w"
                if first_write
                else "a"
            ),
            header=first_write,
            index=False,
        )


        first_write = False


    return {

        "total_packets":
            int(
                total_packets
            ),

        "intact_packets":
            int(
                state_counts[
                    "Intact"
                ]
            ),

        "suspicious_packets":
            int(
                state_counts[
                    "Suspicious"
                ]
            ),

        "tampered_packets":
            int(
                state_counts[
                    "Tampered"
                ]
            ),

        "l4_checksum_indicators":
            int(
                rule_counts[
                    "S7_invalid_l4_checksum"
                ]
            ),

        "rule_counts": {

            rule:
                int(
                    rule_counts[
                        rule
                    ]
                )

            for rule
            in ALL_RULES
        },
    }


# ============================================================
# Random Forest inference
# ============================================================

def run_random_forest(
    flow_csv,
    output_csv,
):

    flow_df = pd.read_csv(
        flow_csv,
        low_memory=False,
    )


    if flow_df.empty:

        raise RuntimeError(
            "No TCP/UDP flows were generated "
            "from the selected capture."
        )


    missing_features = [

        feature

        for feature
        in config.ML_FEATURES

        if feature
        not in
        flow_df.columns
    ]


    if missing_features:

        raise ValueError(
            "Flow dataset is missing Random Forest features:\n"
            +
            "\n".join(
                f"  - {feature}"
                for feature
                in missing_features
            )
        )


    X = (
        flow_df[
            config.ML_FEATURES
        ]
        .copy()
        .astype(
            np.float32
        )
    )


    infinite_count = 0


    for feature in config.ML_FEATURES:

        values = (
            X[
                feature
            ]
            .to_numpy(
                copy=False
            )
        )


        infinite_count += int(
            np.isinf(
                values
            )
            .sum()
        )


    if infinite_count:

        raise ValueError(
            f"Generated flow features contain "
            f"{infinite_count:,} infinite values."
        )


    predictions = MODEL.predict(
        X
    )


    probabilities = MODEL.predict_proba(
        X
    )


    attack_probabilities = (
        probabilities[
            :,
            ATTACK_CLASS_INDEX
        ]
    )


    flow_df[
        "prediction_label"
    ] = predictions


    flow_df[
        "prediction"
    ] = [

        label_to_text(
            prediction
        )

        for prediction
        in predictions
    ]


    flow_df[
        "attack_probability"
    ] = attack_probabilities


    flow_df.to_csv(
        output_csv,
        index=False,
    )


    return flow_df


# ============================================================
# Analysis summary
# ============================================================

def build_summary(
    flow_df,
    integrity_summary,
    source_name,
):

    total_flows = int(
        len(
            flow_df
        )
    )


    attack_flows = int(
        (
            flow_df[
                "prediction_label"
            ]
            ==
            config.LABEL_ATTACK
        )
        .sum()
    )


    benign_flows = int(
        (
            flow_df[
                "prediction_label"
            ]
            ==
            config.LABEL_BENIGN
        )
        .sum()
    )


    attack_rate = (
        attack_flows
        /
        total_flows
        *
        100.0

        if total_flows

        else 0.0
    )


    return {

        "source_name":
            source_name,

        "total_packets":
            integrity_summary[
                "total_packets"
            ],

        "total_flows":
            total_flows,

        "attack_flows":
            attack_flows,

        "benign_flows":
            benign_flows,

        "attack_rate":
            attack_rate,

        "suspicious_packets":
            integrity_summary[
                "suspicious_packets"
            ],

        "tampered_packets":
            integrity_summary[
                "tampered_packets"
            ],

        "intact_packets":
            integrity_summary[
                "intact_packets"
            ],

        "l4_checksum_indicators":
            integrity_summary[
                "l4_checksum_indicators"
            ],

        "integrity_rule_counts":
            integrity_summary[
                "rule_counts"
            ],
    }


# ============================================================
# Main runtime analysis
# ============================================================

def analyze_runtime_capture(
    pcap_path,
    source_name,
):

    # --------------------------------------------------------
    # 1. Extract
    # --------------------------------------------------------

    update_analysis_stage(
        "extract",
        "Extracting packet and header features...",
    )


    run_packet_parser(
        pcap_path,
        RUNTIME_PACKET_FEATURES,
    )


    # --------------------------------------------------------
    # 2. Build flows
    # --------------------------------------------------------

    update_analysis_stage(
        "build",
        "Building bidirectional network flows...",
    )


    run_flow_builder(
        RUNTIME_PACKET_FEATURES,
        RUNTIME_FLOW_FEATURES,
    )


    # --------------------------------------------------------
    # 3. Check headers
    # --------------------------------------------------------

    update_analysis_stage(
        "headers",
        "Checking packet header integrity...",
    )


    integrity_summary = (
        run_runtime_integrity(
            RUNTIME_PACKET_FEATURES,
            RUNTIME_PACKET_INTEGRITY,
        )
    )


    # --------------------------------------------------------
    # 4. Classify
    # --------------------------------------------------------

    update_analysis_stage(
        "classify",
        "Classifying network flows with Random Forest...",
    )


    flow_df = (
        run_random_forest(
            RUNTIME_FLOW_FEATURES,
            RUNTIME_PREDICTIONS,
        )
    )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = build_summary(
        flow_df=flow_df,
        integrity_summary=(
            integrity_summary
        ),
        source_name=source_name,
    )


    with open(
        RUNTIME_SUMMARY,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )


    # --------------------------------------------------------
    # Store completed state
    # --------------------------------------------------------

    with STATE_LOCK:

        elapsed = (
            get_elapsed_seconds_locked()
        )


        APP_STATE[
            "ready"
        ] = True

        APP_STATE[
            "analyzing"
        ] = False

        APP_STATE[
            "source_name"
        ] = source_name

        APP_STATE[
            "flow_df"
        ] = flow_df

        APP_STATE[
            "packet_integrity_path"
        ] = (
            RUNTIME_PACKET_INTEGRITY
        )

        APP_STATE[
            "summary"
        ] = summary

        APP_STATE[
            "shap_cache"
        ] = {}

        APP_STATE[
            "packet_cache"
        ] = {}

        APP_STATE[
            "stage"
        ] = "classify"

        APP_STATE[
            "stage_message"
        ] = (
            "Analysis completed successfully."
        )

        APP_STATE[
            "elapsed_seconds"
        ] = elapsed

        APP_STATE[
            "error"
        ] = None


    return summary


def analysis_worker(
    pcap_path,
    source_name,
):

    try:

        analyze_runtime_capture(
            pcap_path,
            source_name,
        )

    except Exception as exc:

        mark_analysis_failed(
            exc
        )


# ============================================================
# Flow serialization
# ============================================================

def serialize_flow(
    row,
):

    src_ip = json_safe(
        row.get(
            "src_ip"
        )
    )


    dst_ip = json_safe(
        row.get(
            "dst_ip"
        )
    )


    src_port = safe_int(
        row.get(
            "src_port"
        )
    )


    dst_port = safe_int(
        row.get(
            "dst_port"
        )
    )


    source = (
        f"{src_ip}:{src_port}"
        if src_port is not None
        else str(
            src_ip
        )
    )


    destination = (
        f"{dst_ip}:{dst_port}"
        if dst_port is not None
        else str(
            dst_ip
        )
    )


    return {

        "flow_id":
            json_safe(
                row.get(
                    "flow_id"
                )
            ),

        "timestamp":
            safe_float(
                row.get(
                    "timestamp"
                )
            ),

        "source":
            source,

        "destination":
            destination,

        "src_ip":
            src_ip,

        "dst_ip":
            dst_ip,

        "src_port":
            src_port,

        "dst_port":
            dst_port,

        "protocol":
            format_protocol(
                row.get(
                    "ip_protocol"
                )
            ),

        "avg_ttl":
            safe_float(
                row.get(
                    "ttl_mean"
                )
            ),

        "packets":
            safe_int(
                row.get(
                    "flow_packet_count"
                )
            ),

        "bytes":
            safe_int(
                row.get(
                    "flow_byte_count"
                )
            ),

        "duration":
            safe_float(
                row.get(
                    "flow_duration"
                )
            ),

        "prediction":
            json_safe(
                row.get(
                    "prediction"
                )
            ),

        "attack_probability":
            safe_float(
                row.get(
                    "attack_probability"
                )
            ),
    }


def serialize_flow_details(
    row,
):

    base = serialize_flow(
        row
    )


    base.update({

        "frame_length_mean":
            safe_float(
                row.get(
                    "frame_length_mean"
                )
            ),

        "ip_total_length_mean":
            safe_float(
                row.get(
                    "ip_total_length_mean"
                )
            ),

        "ip_version":
            safe_int(
                row.get(
                    "ip_version"
                )
            ),

        "ip_dscp":
            safe_float(
                row.get(
                    "ip_dscp"
                )
            ),

        "ip_ecn":
            safe_float(
                row.get(
                    "ip_ecn"
                )
            ),

        "tcp_window_mean":
            safe_float(
                row.get(
                    "tcp_window_mean"
                )
            ),

        "forward_packets":
            safe_int(
                row.get(
                    "forward_packet_count"
                )
            ),

        "reverse_packets":
            safe_int(
                row.get(
                    "reverse_packet_count"
                )
            ),

        "forward_bytes":
            safe_int(
                row.get(
                    "forward_byte_count"
                )
            ),

        "reverse_bytes":
            safe_int(
                row.get(
                    "reverse_byte_count"
                )
            ),

        "syn_count":
            safe_int(
                row.get(
                    "syn_count"
                )
            ),

        "ack_count":
            safe_int(
                row.get(
                    "ack_count"
                )
            ),

        "fin_count":
            safe_int(
                row.get(
                    "fin_count"
                )
            ),

        "rst_count":
            safe_int(
                row.get(
                    "rst_count"
                )
            ),

        "psh_count":
            safe_int(
                row.get(
                    "psh_count"
                )
            ),

        "byte_rate":
            safe_float(
                row.get(
                    "byte_rate"
                )
            ),
    })


    return base


# ============================================================
# Flow lookup
# ============================================================

def get_flow_row(
    flow_id,
):

    flow_df = APP_STATE[
        "flow_df"
    ]


    if flow_df is None:

        return None


    matches = flow_df[
        flow_df[
            "flow_id"
        ]
        .astype(
            str
        )
        ==
        str(
            flow_id
        )
    ]


    if matches.empty:

        return None


    return matches.iloc[
        0
    ]


# ============================================================
# Related packet matching
# ============================================================

def get_related_packets(
    flow_id,
):

    cached = (
        APP_STATE[
            "packet_cache"
        ]
        .get(
            flow_id
        )
    )


    if cached is not None:

        return cached


    flow = get_flow_row(
        flow_id
    )


    if flow is None:

        raise KeyError(
            "Flow not found."
        )


    packet_path = APP_STATE[
        "packet_integrity_path"
    ]


    if (
        packet_path is None
        or
        not Path(
            packet_path
        ).exists()
    ):

        raise FileNotFoundError(
            "Runtime packet integrity file not found."
        )


    src_ip = str(
        flow[
            "src_ip"
        ]
    )


    dst_ip = str(
        flow[
            "dst_ip"
        ]
    )


    src_port = safe_int(
        flow[
            "src_port"
        ]
    )


    dst_port = safe_int(
        flow[
            "dst_port"
        ]
    )


    protocol = safe_int(
        flow[
            "ip_protocol"
        ]
    )


    start_time = safe_float(
        flow[
            "timestamp"
        ]
    )


    duration = (
        safe_float(
            flow[
                "flow_duration"
            ]
        )
        or
        0.0
    )


    if start_time is None:

        raise ValueError(
            "The selected flow does not contain a valid timestamp."
        )


    end_time = (
        start_time
        +
        max(
            duration,
            0.0,
        )
        +
        0.000001
    )


    required_columns = [

        "packet_id",
        "timestamp",

        "src_ip",
        "dst_ip",

        "src_port",
        "dst_port",

        "ip_protocol",
        "l4_protocol",

        "ttl",

        "tcp_syn",
        "tcp_ack",
        "tcp_fin",
        "tcp_rst",
        "tcp_psh",
        "tcp_urg",

        "ip_checksum_valid",
        "l4_checksum_valid",

        "integrity_state",
        "integrity_reasons",
        "integrity_indicators",
    ]


    total_packets = 0

    intact = 0

    suspicious = 0

    tampered = 0

    indicator_count = 0

    packet_rows = []


    reader = pd.read_csv(
        packet_path,
        usecols=required_columns,
        chunksize=INTEGRITY_CHUNK_SIZE,
        low_memory=False,
    )


    for chunk in reader:

        timestamps = pd.to_numeric(
            chunk[
                "timestamp"
            ],
            errors="coerce",
        )


        time_mask = (
            timestamps.ge(
                start_time
            )
            &
            timestamps.le(
                end_time
            )
        )


        if not time_mask.any():

            continue


        protocol_values = pd.to_numeric(
            chunk[
                "ip_protocol"
            ],
            errors="coerce",
        )


        protocol_mask = (
            protocol_values.eq(
                protocol
            )
        )


        src_port_values = pd.to_numeric(
            chunk[
                "src_port"
            ],
            errors="coerce",
        )


        dst_port_values = pd.to_numeric(
            chunk[
                "dst_port"
            ],
            errors="coerce",
        )


        forward_mask = (

            chunk[
                "src_ip"
            ]
            .astype(
                str
            )
            .eq(
                src_ip
            )

            &

            chunk[
                "dst_ip"
            ]
            .astype(
                str
            )
            .eq(
                dst_ip
            )

            &

            src_port_values.eq(
                src_port
            )

            &

            dst_port_values.eq(
                dst_port
            )
        )


        reverse_mask = (

            chunk[
                "src_ip"
            ]
            .astype(
                str
            )
            .eq(
                dst_ip
            )

            &

            chunk[
                "dst_ip"
            ]
            .astype(
                str
            )
            .eq(
                src_ip
            )

            &

            src_port_values.eq(
                dst_port
            )

            &

            dst_port_values.eq(
                src_port
            )
        )


        mask = (
            time_mask
            &
            protocol_mask
            &
            (
                forward_mask
                |
                reverse_mask
            )
        )


        selected = chunk.loc[
            mask
        ]


        if selected.empty:

            continue


        total_packets += len(
            selected
        )


        intact += int(
            (
                selected[
                    "integrity_state"
                ]
                ==
                "Intact"
            )
            .sum()
        )


        suspicious += int(
            (
                selected[
                    "integrity_state"
                ]
                ==
                "Suspicious"
            )
            .sum()
        )


        tampered += int(
            (
                selected[
                    "integrity_state"
                ]
                ==
                "Tampered"
            )
            .sum()
        )


        indicator_count += int(
            (
                selected[
                    "integrity_indicators"
                ]
                .fillna(
                    "None"
                )
                !=
                "None"
            )
            .sum()
        )


        remaining = (
            RELATED_PACKET_LIMIT
            -
            len(
                packet_rows
            )
        )


        if remaining <= 0:

            continue


        for _, packet in (
            selected
            .head(
                remaining
            )
            .iterrows()
        ):

            flags = []


            flag_map = {

                "SYN":
                    "tcp_syn",

                "ACK":
                    "tcp_ack",

                "FIN":
                    "tcp_fin",

                "RST":
                    "tcp_rst",

                "PSH":
                    "tcp_psh",

                "URG":
                    "tcp_urg",
            }


            for (
                flag_name,
                column,
            ) in flag_map.items():

                if (
                    safe_int(
                        packet.get(
                            column
                        )
                    )
                    ==
                    1
                ):

                    flags.append(
                        flag_name
                    )


            packet_rows.append({

                "packet_id":
                    safe_int(
                        packet.get(
                            "packet_id"
                        )
                    ),

                "timestamp":
                    safe_float(
                        packet.get(
                            "timestamp"
                        )
                    ),

                "source":
                    (
                        f"{packet.get('src_ip')}:"
                        f"{safe_int(packet.get('src_port'))}"
                    ),

                "destination":
                    (
                        f"{packet.get('dst_ip')}:"
                        f"{safe_int(packet.get('dst_port'))}"
                    ),

                "ttl":
                    safe_float(
                        packet.get(
                            "ttl"
                        )
                    ),

                "flags":
                    (
                        ", ".join(
                            flags
                        )
                        if flags
                        else "None"
                    ),

                "ip_checksum":
                    safe_int(
                        packet.get(
                            "ip_checksum_valid"
                        )
                    ),

                "l4_checksum":
                    safe_int(
                        packet.get(
                            "l4_checksum_valid"
                        )
                    ),

                "integrity_state":
                    json_safe(
                        packet.get(
                            "integrity_state"
                        )
                    ),

                "reasons":
                    json_safe(
                        packet.get(
                            "integrity_reasons"
                        )
                    ),

                "indicators":
                    json_safe(
                        packet.get(
                            "integrity_indicators"
                        )
                    ),
            })


    if tampered > 0:

        overall_state = (
            "Tampered"
        )

    elif suspicious > 0:

        overall_state = (
            "Suspicious"
        )

    else:

        overall_state = (
            "Intact"
        )


    result = {

        "integrity_state":
            overall_state,

        "total_packets":
            total_packets,

        "intact_packets":
            intact,

        "suspicious_packets":
            suspicious,

        "tampered_packets":
            tampered,

        "indicator_packets":
            indicator_count,

        "packets":
            packet_rows,

        "displayed_packets":
            len(
                packet_rows
            ),

        "packet_display_limit":
            RELATED_PACKET_LIMIT,
    }


    APP_STATE[
        "packet_cache"
    ][
        flow_id
    ] = result


    return result


# ============================================================
# SHAP
# ============================================================

def get_shap_explainer():

    global SHAP_EXPLAINER


    if SHAP_EXPLAINER is not None:

        return SHAP_EXPLAINER


    with SHAP_LOCK:

        if SHAP_EXPLAINER is None:

            import shap


            SHAP_EXPLAINER = (
                shap.TreeExplainer(
                    MODEL,
                    feature_perturbation=(
                        "tree_path_dependent"
                    ),
                )
            )


    return SHAP_EXPLAINER


def build_plain_shap_explanation(
    prediction,
    attack_probability,
    contributions,
):

    if prediction == "Attack":

        target_word = (
            "Attack"
        )

        support_direction = (
            "pushed the model toward Attack"
        )

        oppose_direction = (
            "pushed the model toward Benign"
        )

        supporting = [
            item
            for item
            in contributions
            if item[
                "shap_value"
            ] > 0
        ][
            :3
        ]

        opposing = [
            item
            for item
            in contributions
            if item[
                "shap_value"
            ] < 0
        ][
            :2
        ]

    else:

        target_word = (
            "Benign"
        )

        support_direction = (
            "pushed the model toward Benign"
        )

        oppose_direction = (
            "pushed the model toward Attack"
        )

        supporting = [
            item
            for item
            in contributions
            if item[
                "shap_value"
            ] < 0
        ][
            :3
        ]

        opposing = [
            item
            for item
            in contributions
            if item[
                "shap_value"
            ] > 0
        ][
            :2
        ]


    sentences = [

        f"This flow was classified as {target_word}."
    ]


    if supporting:

        strongest = supporting[
            0
        ]


        strongest_label = (
            strongest[
                "feature_label"
            ]
        )


        strongest_value = (
            format_feature_value_for_text(
                strongest[
                    "feature_value"
                ]
            )
        )


        sentence = (
            f"The strongest factor was {strongest_label} "
            f"({strongest_value}), which {support_direction}."
        )


        other_names = [

            item[
                "feature_label"
            ]

            for item
            in supporting[
                1:
            ]
        ]


        if len(
            other_names
        ) == 1:

            sentence += (
                f" {other_names[0].capitalize()} "
                "also supported this result."
            )

        elif len(
            other_names
        ) >= 2:

            sentence += (
                f" {other_names[0].capitalize()} and "
                f"{other_names[1]} also supported this result."
            )


        sentences.append(
            sentence
        )


    if opposing:

        opposing_names = [

            item[
                "feature_label"
            ]

            for item
            in opposing
        ]


        if len(
            opposing_names
        ) == 1:

            sentences.append(
                (
                    f"{opposing_names[0].capitalize()} "
                    f"{oppose_direction}, but it was not strong "
                    "enough to change the final classification."
                )
            )

        else:

            sentences.append(
                (
                    f"{opposing_names[0].capitalize()} and "
                    f"{opposing_names[1]} {oppose_direction}, "
                    "but their influence was not strong enough "
                    "to change the final classification."
                )
            )


    if len(
        sentences
    ) == 1:

        sentences.append(
            (
                "No single displayed feature had a strong enough "
                "individual contribution to provide a more detailed "
                "plain-language explanation."
            )
        )


    return " ".join(
        sentences
    )


def calculate_shap(
    flow_id,
):

    cached = (
        APP_STATE[
            "shap_cache"
        ]
        .get(
            flow_id
        )
    )


    if cached is not None:

        return cached


    flow = get_flow_row(
        flow_id
    )


    if flow is None:

        raise KeyError(
            "Flow not found."
        )


    X = pd.DataFrame(
        [
            {
                feature:
                    flow[
                        feature
                    ]

                for feature
                in config.ML_FEATURES
            }
        ],
        columns=config.ML_FEATURES,
    ).astype(
        np.float32
    )


    explainer = (
        get_shap_explainer()
    )


    raw_shap = (
        explainer.shap_values(
            X,
            check_additivity=False,
        )
    )


    if isinstance(
        raw_shap,
        list,
    ):

        shap_vector = np.asarray(
            raw_shap[
                ATTACK_CLASS_INDEX
            ]
        )[
            0
        ]

    else:

        shap_array = np.asarray(
            raw_shap
        )


        if (
            shap_array.ndim
            ==
            3
        ):

            shap_vector = (
                shap_array[
                    0,
                    :,
                    ATTACK_CLASS_INDEX,
                ]
            )

        elif (
            shap_array.ndim
            ==
            2
        ):

            shap_vector = (
                shap_array[
                    0
                ]
            )

        else:

            raise RuntimeError(
                "Unexpected SHAP output shape: "
                f"{shap_array.shape}"
            )


    contributions = []


    for (
        index,
        feature,
    ) in enumerate(
        config.ML_FEATURES
    ):

        shap_value = float(
            shap_vector[
                index
            ]
        )


        feature_value = json_safe(
            X.iloc[
                0
            ][
                feature
            ]
        )


        if shap_value > 0:

            direction = (
                "Attack"
            )

        elif shap_value < 0:

            direction = (
                "Benign"
            )

        else:

            direction = (
                "Neutral"
            )


        contributions.append({

            "feature":
                feature,

            "feature_label":
                get_feature_label(
                    feature
                ),

            "feature_value":
                feature_value,

            "shap_value":
                shap_value,

            "absolute_shap":
                abs(
                    shap_value
                ),

            "direction":
                direction,
        })


    contributions.sort(
        key=lambda item:
            item[
                "absolute_shap"
            ],
        reverse=True,
    )


    prediction = json_safe(
        flow.get(
            "prediction"
        )
    )


    attack_probability = safe_float(
        flow.get(
            "attack_probability"
        )
    )


    plain_explanation = (
        build_plain_shap_explanation(
            prediction=prediction,
            attack_probability=(
                attack_probability
            ),
            contributions=contributions,
        )
    )


    result = {

        "flow_id":
            flow_id,

        "prediction":
            prediction,

        "attack_probability":
            attack_probability,

        "plain_explanation":
            plain_explanation,

        "top_contributions":
            contributions[
                :10
            ],
    }


    APP_STATE[
        "shap_cache"
    ][
        flow_id
    ] = result


    return result


# ============================================================
# Routes
# ============================================================

@app.route(
    "/"
)
def index():

    return render_template(
        "dashboard.html"
    )


# ============================================================
# Analyze uploaded PCAP
# ============================================================

@app.route(
    "/api/analyze/upload",
    methods=[
        "POST"
    ],
)
def analyze_upload():

    with ANALYSIS_CONTROL_LOCK:

        try:

            # ------------------------------------------------
            # Do not start another analysis while one is active.
            # ------------------------------------------------

            with STATE_LOCK:

                if APP_STATE[
                    "analyzing"
                ]:

                    return jsonify({

                        "success":
                            False,

                        "error":
                            (
                                "An analysis is already running. "
                                "Please wait for it to finish."
                            ),

                    }), 409


            # ------------------------------------------------
            # Validate upload
            # ------------------------------------------------

            if (
                "file"
                not in
                request.files
            ):

                return jsonify({

                    "success":
                        False,

                    "error":
                        "No PCAP file was provided.",

                }), 400


            file = request.files[
                "file"
            ]


            if (
                file.filename is None
                or
                file.filename.strip()
                ==
                ""
            ):

                return jsonify({

                    "success":
                        False,

                    "error":
                        "No file was selected.",

                }), 400


            original_filename = (
                secure_filename(
                    file.filename
                )
            )


            if not original_filename:

                return jsonify({

                    "success":
                        False,

                    "error":
                        "The selected filename is not valid.",

                }), 400


            if not allowed_file(
                original_filename
            ):

                return jsonify({

                    "success":
                        False,

                    "error":
                        (
                            "Unsupported file type. "
                            "Please upload a .pcap or .pcapng file."
                        ),

                }), 400


            # ------------------------------------------------
            # Only retain newest analysis
            # ------------------------------------------------

            reset_application_state()

            reset_runtime_directory()


            suffix = (
                Path(
                    original_filename
                )
                .suffix
                .lower()
            )


            runtime_input = (
                CURRENT_RUNTIME_DIR
                /
                f"input{suffix}"
            )


            # Save directly to runtime/current/.
            # No uploads/ directory is retained.

            file.save(
                runtime_input
            )


            # ------------------------------------------------
            # Start analysis state
            # ------------------------------------------------

            mark_analysis_started(
                original_filename
            )


            # ------------------------------------------------
            # Run pipeline in a background thread.
            # ------------------------------------------------

            worker = threading.Thread(
                target=analysis_worker,
                args=(
                    runtime_input,
                    original_filename,
                ),
                daemon=True,
                name="networkids-analysis",
            )


            worker.start()


            return jsonify({

                "success":
                    True,

                "message":
                    "Analysis started.",

            }), 202


        except Exception as exc:

            mark_analysis_failed(
                exc
            )


            return jsonify({

                "success":
                    False,

                "error":
                    str(
                        exc
                    ),

            }), 500


# ============================================================
# Analysis status
# ============================================================

@app.route(
    "/api/status",
    methods=[
        "GET"
    ],
)
def analysis_status():

    with STATE_LOCK:

        elapsed = (
            get_elapsed_seconds_locked()
        )


        return jsonify({

            "success":
                True,

            "analyzing":
                bool(
                    APP_STATE[
                        "analyzing"
                    ]
                ),

            "ready":
                bool(
                    APP_STATE[
                        "ready"
                    ]
                ),

            "source_name":
                APP_STATE[
                    "source_name"
                ],

            "stage":
                APP_STATE[
                    "stage"
                ],

            "stage_message":
                APP_STATE[
                    "stage_message"
                ],

            "elapsed_seconds":
                float(
                    elapsed
                ),

            "error":
                APP_STATE[
                    "error"
                ],
        })


# ============================================================
# Current summary
# ============================================================

@app.route(
    "/api/summary",
    methods=[
        "GET"
    ],
)
def get_summary():

    if not APP_STATE[
        "ready"
    ]:

        return jsonify({

            "success":
                False,

            "error":
                "No analysis is currently available.",

        }), 404


    return jsonify({

        "success":
            True,

        "summary":
            APP_STATE[
                "summary"
            ],
    })


# ============================================================
# Flow table
# ============================================================

@app.route(
    "/api/flows",
    methods=[
        "GET"
    ],
)
def get_flows():

    if not APP_STATE[
        "ready"
    ]:

        return jsonify({

            "success":
                False,

            "error":
                "No analysis is currently available.",

        }), 404


    flow_df = APP_STATE[
        "flow_df"
    ]


    page = max(
        1,
        request.args.get(
            "page",
            1,
            type=int,
        ),
    )


    page_size = request.args.get(
        "page_size",
        DEFAULT_PAGE_SIZE,
        type=int,
    )


    page_size = min(
        max(
            1,
            page_size,
        ),
        MAX_PAGE_SIZE,
    )


    prediction_filter = (
        request.args.get(
            "prediction",
            "All",
        )
        .strip()
    )


    protocol_filter = (
        request.args.get(
            "protocol",
            "All",
        )
        .strip()
    )


    search_query = (
        request.args.get(
            "q",
            "",
        )
        .strip()
        .lower()
    )


    filtered = flow_df


    # Prediction filter

    if prediction_filter in {
        "Attack",
        "Benign",
    }:

        filtered = filtered[
            filtered[
                "prediction"
            ]
            ==
            prediction_filter
        ]


    # Protocol filter

    if (
        protocol_filter
        ==
        "TCP"
    ):

        filtered = filtered[
            pd.to_numeric(
                filtered[
                    "ip_protocol"
                ],
                errors="coerce",
            )
            .eq(
                6
            )
        ]

    elif (
        protocol_filter
        ==
        "UDP"
    ):

        filtered = filtered[
            pd.to_numeric(
                filtered[
                    "ip_protocol"
                ],
                errors="coerce",
            )
            .eq(
                17
            )
        ]


    # Search IP, port and protocol.

    if search_query:

        src_ip_text = (
            filtered[
                "src_ip"
            ]
            .astype(
                str
            )
            .str
            .lower()
        )


        dst_ip_text = (
            filtered[
                "dst_ip"
            ]
            .astype(
                str
            )
            .str
            .lower()
        )


        src_port_text = (
            filtered[
                "src_port"
            ]
            .astype(
                str
            )
            .str
            .lower()
        )


        dst_port_text = (
            filtered[
                "dst_port"
            ]
            .astype(
                str
            )
            .str
            .lower()
        )


        protocol_text = (
            filtered[
                "ip_protocol"
            ]
            .apply(
                format_protocol
            )
            .astype(
                str
            )
            .str
            .lower()
        )


        search_mask = (

            src_ip_text.str.contains(
                search_query,
                na=False,
                regex=False,
            )

            |

            dst_ip_text.str.contains(
                search_query,
                na=False,
                regex=False,
            )

            |

            src_port_text.str.contains(
                search_query,
                na=False,
                regex=False,
            )

            |

            dst_port_text.str.contains(
                search_query,
                na=False,
                regex=False,
            )

            |

            protocol_text.str.contains(
                search_query,
                na=False,
                regex=False,
            )
        )


        filtered = filtered[
            search_mask
        ]


    # Newest flow first.

    filtered = (
        filtered
        .assign(
            _sort_timestamp=(
                pd.to_numeric(
                    filtered[
                        "timestamp"
                    ],
                    errors="coerce",
                )
            )
        )
        .sort_values(
            by="_sort_timestamp",
            ascending=False,
            na_position="last",
        )
        .drop(
            columns=[
                "_sort_timestamp"
            ]
        )
    )


    total_rows = int(
        len(
            filtered
        )
    )


    total_pages = max(
        1,
        math.ceil(
            total_rows
            /
            page_size
        ),
    )


    if page > total_pages:

        page = total_pages


    start = (
        page
        -
        1
    ) * page_size


    end = (
        start
        +
        page_size
    )


    page_df = filtered.iloc[
        start:end
    ]


    rows = [

        serialize_flow(
            row
        )

        for _,
        row
        in page_df.iterrows()
    ]


    return jsonify({

        "success":
            True,

        "rows":
            rows,

        "pagination": {

            "page":
                page,

            "page_size":
                page_size,

            "total_rows":
                total_rows,

            "total_pages":
                total_pages,
        },
    })


# ============================================================
# Flow details
# ============================================================

@app.route(
    "/api/flow/<flow_id>",
    methods=[
        "GET"
    ],
)
def flow_details(
    flow_id,
):

    if not APP_STATE[
        "ready"
    ]:

        return jsonify({

            "success":
                False,

            "error":
                "No analysis is currently available.",

        }), 404


    row = get_flow_row(
        flow_id
    )


    if row is None:

        return jsonify({

            "success":
                False,

            "error":
                "Flow not found.",

        }), 404


    return jsonify({

        "success":
            True,

        "flow":
            serialize_flow_details(
                row
            ),
    })


# ============================================================
# Related packets / integrity
# ============================================================

@app.route(
    "/api/flow/<flow_id>/packets",
    methods=[
        "GET"
    ],
)
def flow_packets(
    flow_id,
):

    try:

        result = get_related_packets(
            flow_id
        )


        return jsonify({

            "success":
                True,

            "result":
                result,
        })


    except Exception as exc:

        return jsonify({

            "success":
                False,

            "error":
                str(
                    exc
                ),

        }), 500


# ============================================================
# SHAP explanation
# ============================================================

@app.route(
    "/api/flow/<flow_id>/explain",
    methods=[
        "GET"
    ],
)
def flow_explain(
    flow_id,
):

    try:

        explanation = (
            calculate_shap(
                flow_id
            )
        )


        return jsonify({

            "success":
                True,

            "explanation":
                explanation,
        })


    except Exception as exc:

        return jsonify({

            "success":
                False,

            "error":
                str(
                    exc
                ),

        }), 500


# ============================================================
# Flask error: oversized upload
# ============================================================

@app.errorhandler(
    413
)
def too_large(
    _error,
):

    return jsonify({

        "success":
            False,

        "error":
            (
                "The uploaded PCAP exceeds the "
                "configured 512 MB limit."
            ),

    }), 413


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 70
    )

    print(
        "NetworkIDS Dashboard"
    )

    print(
        "=" * 70
    )

    print(
        f"Random Forest : "
        f"{MODEL_FILE}"
    )

    print(
        "Upload limit  : "
        "512 MB"
    )

    print(
        "Runtime       : "
        f"{CURRENT_RUNTIME_DIR}"
    )

    print(
        "Dashboard     : "
        "http://127.0.0.1:5000"
    )

    print(
        "=" * 70
    )


    app.run(
        host="127.0.0.1",
        port=5000,

        debug=True,

        # Prevent Flask debug mode from loading the model twice.
        use_reloader=False,

        threaded=True,
    )