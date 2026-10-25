"""
app.py
------
NetworkIDS - Flask Dashboard Backend.

Single-page FYP prototype.

Supported runtime modes:
    1. Upload a PCAP file.
    2. Use the bundled demo PCAP.

Runtime pipeline:
    PCAP
        -> packet_parser.py
        -> packet_features.csv
        -> flow_builder.py
        -> flow_features.csv
        -> Random Forest classification

Packet-level integrity checking is performed independently
using src.integrity_checker.apply_rules().

SHAP explanations are calculated lazily only when the user
opens the explanation section for a selected flow.

No database is used.
"""

from pathlib import Path
from datetime import datetime
import json
import math
import shutil
import subprocess
import sys
import threading

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
)


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

UPLOAD_DIR = (
    PROJECT_ROOT
    / "uploads"
)

RUNTIME_DIR = (
    PROJECT_ROOT
    / "runtime"
)

CURRENT_RUNTIME_DIR = (
    RUNTIME_DIR
    / "current"
)

DEMO_DIR = (
    PROJECT_ROOT
    / "demo"
)

DEMO_PCAP = (
    DEMO_DIR
    / "demo_traffic.pcap"
)


# ============================================================
# Runtime files
# ============================================================

RUNTIME_INPUT_PCAP = (
    CURRENT_RUNTIME_DIR
    / "input.pcap"
)

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

INTEGRITY_CHUNK_SIZE = 250_000

DEFAULT_PAGE_SIZE = 25

MAX_PAGE_SIZE = 100

RELATED_PACKET_LIMIT = 100


# ============================================================
# Flask
# ============================================================

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)


# Maximum upload:
# 512 MB for prototype use.
#
# You may increase this later if needed.

app.config[
    "MAX_CONTENT_LENGTH"
] = 512 * 1024 * 1024


# ============================================================
# Ensure directories
# ============================================================

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RUNTIME_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DEMO_DIR.mkdir(
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
        int(MODEL.n_features_in_)
        != len(config.ML_FEATURES)
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

    if model_features != list(
        config.ML_FEATURES
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

if config.LABEL_ATTACK not in MODEL_CLASSES:

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
#
# This is intentionally simple because this is a single-user
# local FYP prototype rather than a multi-user production app.
# ============================================================

STATE_LOCK = threading.Lock()

APP_STATE = {

    "ready":
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
}


# ============================================================
# SHAP state
#
# Created lazily the first time a user requests an explanation.
# ============================================================

SHAP_EXPLAINER = None

SHAP_LOCK = threading.Lock()


# ============================================================
# Utility helpers
# ============================================================

def json_safe(value):
    """
    Convert numpy/pandas values into JSON-safe Python values.
    """

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
        return int(value)

    if isinstance(
        value,
        (
            np.floating,
            np.float16,
            np.float32,
            np.float64,
        ),
    ):

        if np.isnan(value):
            return None

        if np.isinf(value):
            return None

        return float(value)

    try:

        if pd.isna(value):
            return None

    except (TypeError, ValueError):
        pass

    return value


def safe_float(value):
    """
    Convert a value into float when possible.
    """

    value = json_safe(value)

    if value is None:
        return None

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return None


def safe_int(value):
    """
    Convert a value into int when possible.
    """

    value = json_safe(value)

    if value is None:
        return None

    try:
        return int(float(value))

    except (
        TypeError,
        ValueError,
    ):
        return None


def format_protocol(
    ip_protocol,
):
    """
    Convert IP protocol number into readable name.
    """

    protocol = safe_int(
        ip_protocol
    )

    mapping = {
        1: "ICMP",
        6: "TCP",
        17: "UDP",
        58: "ICMPv6",
    }

    if protocol in mapping:
        return mapping[protocol]

    if protocol is None:
        return "Unknown"

    return str(protocol)


def label_to_text(label):
    """
    Convert configured prediction label into text.
    """

    label = int(label)

    if label == config.LABEL_ATTACK:
        return "Attack"

    if label == config.LABEL_BENIGN:
        return "Benign"

    return str(label)


def allowed_file(
    filename,
):
    """
    Validate uploaded PCAP extension.
    """

    suffix = (
        Path(filename)
        .suffix
        .lower()
    )

    return suffix in ALLOWED_EXTENSIONS


def reset_runtime_directory():
    """
    Remove previous runtime analysis.

    Only the latest dashboard analysis is retained.
    """

    if CURRENT_RUNTIME_DIR.exists():

        shutil.rmtree(
            CURRENT_RUNTIME_DIR
        )

    CURRENT_RUNTIME_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def reset_application_state():
    """
    Clear previous in-memory dashboard state.
    """

    with STATE_LOCK:

        APP_STATE[
            "ready"
        ] = False

        APP_STATE[
            "source_name"
        ] = None

        APP_STATE[
            "flow_df"
        ] = None

        APP_STATE[
            "packet_integrity_path"
        ] = None

        APP_STATE[
            "summary"
        ] = None

        APP_STATE[
            "shap_cache"
        ] = {}

        APP_STATE[
            "packet_cache"
        ] = {}


# ============================================================
# External script execution
# ============================================================

def run_command(
    command,
    stage_name,
):
    """
    Run one existing project script.

    The scripts remain independent from Flask.
    """

    completed = subprocess.run(
        command,
        cwd=str(
            PROJECT_ROOT
        ),
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:

        stdout_tail = (
            completed.stdout[-4000:]
            if completed.stdout
            else ""
        )

        stderr_tail = (
            completed.stderr[-4000:]
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
    """
    Run the existing packet_parser.py.

    Current packet_parser CLI:
        --pcap
        --limit
        --preview
        --out

    limit=0 means process the whole capture.
    """

    command = [

        sys.executable,

        str(
            PROJECT_ROOT
            / "src"
            / "packet_parser.py"
        ),

        "--pcap",
        str(pcap_path),

        "--limit",
        "0",

        "--preview",
        "0",

        "--out",
        str(output_csv),
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
    """
    Run the existing streaming flow_builder.py.
    """

    command = [

        sys.executable,

        str(
            PROJECT_ROOT
            / "src"
            / "flow_builder.py"
        ),

        "--in",
        str(packet_csv),

        "--out",
        str(flow_csv),
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
    """
    Apply the current packet integrity rules to the uploaded/demo
    capture without using the permanent five-day research output.

    Returns aggregate packet integrity statistics.
    """

    header = pd.read_csv(
        packet_csv,
        nrows=0,
    )

    missing = [

        column

        for column
        in INTEGRITY_NEEDED_COLUMNS

        if column not in header.columns
    ]

    if missing:

        raise ValueError(
            "Runtime packet features are missing "
            "integrity columns:\n"
            +
            "\n".join(
                f"  - {column}"
                for column in missing
            )
        )


    if output_csv.exists():

        output_csv.unlink()


    state_counts = {

        "Intact": 0,

        "Suspicious": 0,

        "Tampered": 0,
    }

    total_packets = 0

    indicator_count = 0

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


        for state in state_counts:

            state_counts[state] += int(
                (
                    chunk[
                        "integrity_state"
                    ]
                    ==
                    state
                )
                .sum()
            )


        indicator_count += int(
            chunk[
                "S7_invalid_l4_checksum"
            ]
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
                indicator_count
            ),
    }


# ============================================================
# Random Forest inference
# ============================================================

def run_random_forest(
    flow_csv,
    output_csv,
):
    """
    Classify all generated flows using the final Random Forest.
    """

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

        if feature not in flow_df.columns
    ]


    if missing_features:

        raise ValueError(
            "Flow dataset is missing Random Forest features:\n"
            +
            "\n".join(
                f"  - {feature}"
                for feature in missing_features
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
            X[feature]
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
    ] = (
        attack_probabilities
    )


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
    """
    Build runtime dashboard KPI values.
    """

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
    }


# ============================================================
# Main PCAP analysis
# ============================================================

def analyze_pcap(
    pcap_path,
    source_name,
):
    """
    Run the complete runtime dashboard pipeline.
    """

    reset_application_state()

    reset_runtime_directory()


    # --------------------------------------------------------
    # Copy selected capture into runtime/current/
    # --------------------------------------------------------

    shutil.copy2(
        pcap_path,
        RUNTIME_INPUT_PCAP,
    )


    # --------------------------------------------------------
    # 1. Packet parser
    # --------------------------------------------------------

    run_packet_parser(
        RUNTIME_INPUT_PCAP,
        RUNTIME_PACKET_FEATURES,
    )


    # --------------------------------------------------------
    # 2. Flow builder
    # --------------------------------------------------------

    run_flow_builder(
        RUNTIME_PACKET_FEATURES,
        RUNTIME_FLOW_FEATURES,
    )


    # --------------------------------------------------------
    # 3. Integrity checker
    # --------------------------------------------------------

    integrity_summary = (
        run_runtime_integrity(
            RUNTIME_PACKET_FEATURES,
            RUNTIME_PACKET_INTEGRITY,
        )
    )


    # --------------------------------------------------------
    # 4. Random Forest
    # --------------------------------------------------------

    flow_df = (
        run_random_forest(
            RUNTIME_FLOW_FEATURES,
            RUNTIME_PREDICTIONS,
        )
    )


    # --------------------------------------------------------
    # 5. Summary
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
    # Store current state
    # --------------------------------------------------------

    with STATE_LOCK:

        APP_STATE[
            "ready"
        ] = True

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


    return summary


# ============================================================
# Flow serialization
# ============================================================

def serialize_flow(
    row,
):
    """
    Convert one flow row into dashboard-friendly JSON.
    """

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
        else str(src_ip)
    )


    destination = (
        f"{dst_ip}:{dst_port}"
        if dst_port is not None
        else str(dst_ip)
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
    """
    More detailed information for the modal popup.
    """

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
    """
    Find one currently analysed flow.
    """

    flow_df = APP_STATE[
        "flow_df"
    ]

    if flow_df is None:

        return None


    matches = flow_df[
        flow_df[
            "flow_id"
        ].astype(str)
        ==
        str(flow_id)
    ]


    if matches.empty:

        return None


    return matches.iloc[0]


# ============================================================
# Related packet matching
# ============================================================

def get_related_packets(
    flow_id,
):
    """
    Find packet-level rows associated with one bidirectional flow.

    Matching uses:
        - source/destination IP
        - source/destination ports
        - protocol
        - flow start/end timestamps

    Both forward and reverse directions are accepted.
    """

    cached = APP_STATE[
        "packet_cache"
    ].get(
        flow_id
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
        or not Path(
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

    start_time = float(
        flow[
            "timestamp"
        ]
    )

    duration = float(
        flow[
            "flow_duration"
        ]
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
        chunksize=250_000,
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


        forward_mask = (

            chunk[
                "src_ip"
            ].astype(str).eq(
                src_ip
            )

            &

            chunk[
                "dst_ip"
            ].astype(str).eq(
                dst_ip
            )

            &

            pd.to_numeric(
                chunk[
                    "src_port"
                ],
                errors="coerce",
            ).eq(
                src_port
            )

            &

            pd.to_numeric(
                chunk[
                    "dst_port"
                ],
                errors="coerce",
            ).eq(
                dst_port
            )
        )


        reverse_mask = (

            chunk[
                "src_ip"
            ].astype(str).eq(
                dst_ip
            )

            &

            chunk[
                "dst_ip"
            ].astype(str).eq(
                src_ip
            )

            &

            pd.to_numeric(
                chunk[
                    "src_port"
                ],
                errors="coerce",
            ).eq(
                dst_port
            )

            &

            pd.to_numeric(
                chunk[
                    "dst_port"
                ],
                errors="coerce",
            ).eq(
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
                .fillna("None")
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


        if remaining > 0:

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

                    if safe_int(
                        packet.get(
                            column
                        )
                    ) == 1:

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
    ][flow_id] = result


    return result


# ============================================================
# SHAP
# ============================================================

def get_shap_explainer():
    """
    Create TreeExplainer only when first requested.
    """

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


def calculate_shap(
    flow_id,
):
    """
    Calculate local SHAP explanation for one selected flow.
    """

    cached = APP_STATE[
        "shap_cache"
    ].get(
        flow_id
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


    # --------------------------------------------------------
    # Handle SHAP output formats across versions
    # --------------------------------------------------------

    if isinstance(
        raw_shap,
        list,
    ):

        shap_vector = np.asarray(
            raw_shap[
                ATTACK_CLASS_INDEX
            ]
        )[0]

    else:

        shap_array = np.asarray(
            raw_shap
        )


        if shap_array.ndim == 3:

            shap_vector = (
                shap_array[
                    0,
                    :,
                    ATTACK_CLASS_INDEX,
                ]
            )

        elif shap_array.ndim == 2:

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


        feature_value = (
            json_safe(
                X.iloc[
                    0
                ][
                    feature
                ]
            )
        )


        contributions.append({

            "feature":
                feature,

            "feature_value":
                feature_value,

            "shap_value":
                shap_value,

            "absolute_shap":
                abs(
                    shap_value
                ),

            "direction":
                (
                    "Attack"
                    if shap_value > 0
                    else "Benign"
                ),
        })


    contributions.sort(
        key=lambda item:
            item[
                "absolute_shap"
            ],
        reverse=True,
    )


    result = {

        "flow_id":
            flow_id,

        "prediction":
            json_safe(
                flow.get(
                    "prediction"
                )
            ),

        "attack_probability":
            safe_float(
                flow.get(
                    "attack_probability"
                )
            ),

        "top_contributions":
            contributions[:10],
    }


    APP_STATE[
        "shap_cache"
    ][flow_id] = result


    return result


# ============================================================
# Routes
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# Analyze uploaded PCAP
# ============================================================

@app.route(
    "/api/analyze/upload",
    methods=["POST"],
)
def analyze_upload():

    try:

        if "file" not in request.files:

            return jsonify({
                "success": False,
                "error": "No PCAP file was provided.",
            }), 400


        file = request.files[
            "file"
        ]


        if (
            file.filename is None
            or file.filename.strip() == ""
        ):

            return jsonify({
                "success": False,
                "error": "No file was selected.",
            }), 400


        filename = secure_filename(
            file.filename
        )


        if not allowed_file(
            filename
        ):

            return jsonify({
                "success": False,
                "error": (
                    "Unsupported file type. "
                    "Please upload a .pcap or .pcapng file."
                ),
            }), 400


        upload_path = (
            UPLOAD_DIR
            /
            filename
        )


        file.save(
            upload_path
        )


        summary = analyze_pcap(
            upload_path,
            source_name=filename,
        )


        return jsonify({
            "success": True,
            "summary": summary,
        })


    except Exception as exc:

        return jsonify({
            "success": False,
            "error": str(exc),
        }), 500


# ============================================================
# Analyze demo PCAP
# ============================================================

@app.route(
    "/api/analyze/demo",
    methods=["POST"],
)
def analyze_demo():

    try:

        if not DEMO_PCAP.exists():

            return jsonify({
                "success": False,
                "error": (
                    "Demo traffic file was not found. "
                    "Place demo_traffic.pcap inside "
                    f"{DEMO_DIR}"
                ),
            }), 404


        summary = analyze_pcap(
            DEMO_PCAP,
            source_name=(
                DEMO_PCAP.name
            ),
        )


        return jsonify({
            "success": True,
            "summary": summary,
        })


    except Exception as exc:

        return jsonify({
            "success": False,
            "error": str(exc),
        }), 500


# ============================================================
# Current summary
# ============================================================

@app.route(
    "/api/summary",
    methods=["GET"],
)
def get_summary():

    if not APP_STATE[
        "ready"
    ]:

        return jsonify({
            "success": False,
            "error": "No analysis is currently available.",
        }), 404


    return jsonify({
        "success": True,
        "summary": APP_STATE[
            "summary"
        ],
    })


# ============================================================
# Flow table
# ============================================================

@app.route(
    "/api/flows",
    methods=["GET"],
)
def get_flows():

    if not APP_STATE[
        "ready"
    ]:

        return jsonify({
            "success": False,
            "error": "No analysis is currently available.",
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


    # --------------------------------------------------------
    # Prediction filter
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Protocol filter
    # --------------------------------------------------------

    if protocol_filter == "TCP":

        filtered = filtered[
            pd.to_numeric(
                filtered[
                    "ip_protocol"
                ],
                errors="coerce",
            )
            .eq(6)
        ]

    elif protocol_filter == "UDP":

        filtered = filtered[
            pd.to_numeric(
                filtered[
                    "ip_protocol"
                ],
                errors="coerce",
            )
            .eq(17)
        ]


    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    if search_query:

        search_mask = (

            filtered[
                "flow_id"
            ]
            .astype(str)
            .str.lower()
            .str.contains(
                search_query,
                na=False,
            )

            |

            filtered[
                "src_ip"
            ]
            .astype(str)
            .str.lower()
            .str.contains(
                search_query,
                na=False,
            )

            |

            filtered[
                "dst_ip"
            ]
            .astype(str)
            .str.lower()
            .str.contains(
                search_query,
                na=False,
            )
        )


        filtered = filtered[
            search_mask
        ]


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
        page - 1
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
    methods=["GET"],
)
def flow_details(
    flow_id,
):

    if not APP_STATE[
        "ready"
    ]:

        return jsonify({
            "success": False,
            "error": "No analysis is currently available.",
        }), 404


    row = get_flow_row(
        flow_id
    )


    if row is None:

        return jsonify({
            "success": False,
            "error": "Flow not found.",
        }), 404


    return jsonify({
        "success": True,
        "flow": serialize_flow_details(
            row
        ),
    })


# ============================================================
# Related packets / integrity
# ============================================================

@app.route(
    "/api/flow/<flow_id>/packets",
    methods=["GET"],
)
def flow_packets(
    flow_id,
):

    try:

        result = get_related_packets(
            flow_id
        )


        return jsonify({
            "success": True,
            "result": result,
        })


    except Exception as exc:

        return jsonify({
            "success": False,
            "error": str(exc),
        }), 500


# ============================================================
# SHAP explanation
# ============================================================

@app.route(
    "/api/flow/<flow_id>/explain",
    methods=["GET"],
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
            "success": True,
            "explanation": explanation,
        })


    except Exception as exc:

        return jsonify({
            "success": False,
            "error": str(exc),
        }), 500


# ============================================================
# Flask error: oversized upload
# ============================================================

@app.errorhandler(413)
def too_large(
    _error,
):

    return jsonify({
        "success": False,
        "error": (
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
        f"Random Forest: {MODEL_FILE}"
    )

    print(
        f"Demo PCAP    : {DEMO_PCAP}"
    )

    print(
        "Dashboard    : http://127.0.0.1:5000"
    )

    print(
        "=" * 70
    )


    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )