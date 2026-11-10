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
    redirect,
    url_for,
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


# Maximum upload:
# 512 MB for prototype use.

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


# Current analysis stage for the upload page.
ANALYSIS_STATUS = {
    "analyzing": False,
    "ready": False,
    "error": None,
    "stage": None,
    "stage_message": "No analysis in progress.",
    "started_at": None,
    "elapsed_final": 0.0,
}
OVERVIEW_CACHE = None
TIMELINE_CACHE = None
RUNTIME_OVERVIEW = CURRENT_RUNTIME_DIR / "overview_stats.json"
RUNTIME_TIMELINE = CURRENT_RUNTIME_DIR / "events_timeline.json"
RUNTIME_FLOW_INTEGRITY = CURRENT_RUNTIME_DIR / "flow_integrity.csv"

def mark_stage(stage, message):
    with STATE_LOCK:
        ANALYSIS_STATUS["stage"] = stage
        ANALYSIS_STATUS["stage_message"] = message

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
    """
    Convert a value into float when possible.
    """

    value = json_safe(
        value
    )


    if value is None:

        return None


    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def safe_int(
    value,
):
    """
    Convert a value into int when possible.
    """

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
    """
    Convert IP protocol number into readable name.
    """

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
    """
    Convert configured prediction label into text.
    """

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
    """
    Validate uploaded PCAP extension.
    """

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
    """
    Convert an ML feature name into a simpler readable label.
    """

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
    """
    Format a feature value for plain-language SHAP explanation.
    """

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
        and absolute < 0.0001
    ):

        return (
            f"{value:.8f}"
            .rstrip("0")
            .rstrip(".")
        )


    return (
        f"{value:.2f}"
        .rstrip("0")
        .rstrip(".")
    )


def reset_runtime_directory():
    """
    Remove previous runtime analysis.

    Only the latest analysis is retained.
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

    global OVERVIEW_CACHE, TIMELINE_CACHE
    with STATE_LOCK:
        OVERVIEW_CACHE = None
        TIMELINE_CACHE = None
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
    """

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

            f"STDOUT:\n"
            f"{stdout_tail}\n\n"

            f"STDERR:\n"
            f"{stderr_tail}"
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
    Run packet_parser.py.

    limit=0 means process the complete capture.
    """

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
    """
    Apply packet integrity rules to the current capture.

    Returns:
        - packet integrity state counts
        - T1-T3 rule counts
        - S1-S6 rule counts
        - S7 indicator count
    """

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
    """
    Classify generated flows using the final Random Forest.
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

        # Retained for future integrity-rule UI.
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
    """
    Run the complete pipeline on a PCAP already stored inside
    runtime/current/.
    """

    # --------------------------------------------------------
    # 1. Packet parser
    # --------------------------------------------------------

    mark_stage("extract", "Extracting L2-L4 packet features...")

    run_packet_parser(
        pcap_path,
        RUNTIME_PACKET_FEATURES,
    )


    # --------------------------------------------------------
    # 2. Flow builder
    # --------------------------------------------------------

    mark_stage("build", "Aggregating packets into network flows...")

    run_flow_builder(
        RUNTIME_PACKET_FEATURES,
        RUNTIME_FLOW_FEATURES,
    )


    # --------------------------------------------------------
    # 3. Integrity checker
    # --------------------------------------------------------

    mark_stage("headers", "Applying packet header integrity rules...")

    integrity_summary = (
        run_runtime_integrity(
            RUNTIME_PACKET_FEATURES,
            RUNTIME_PACKET_INTEGRITY,
        )
    )


    # --------------------------------------------------------
    # 4. Random Forest
    # --------------------------------------------------------

    mark_stage("classify", "Classifying network flows with Random Forest...")

    flow_df = (
        run_random_forest(
            RUNTIME_FLOW_FEATURES,
            RUNTIME_PREDICTIONS,
        )
    )


    # --------------------------------------------------------
    # 5. Prepare packet-to-flow integrity and dashboard data
    # --------------------------------------------------------

    mark_stage("prepare", "Preparing flow integrity, dashboard charts, timeline and SHAP importance...")
    flow_df = add_flow_integrity_states(flow_df, RUNTIME_PACKET_INTEGRITY)
    flow_df.to_csv(RUNTIME_PREDICTIONS, index=False)
    overview = build_overview_payload(flow_df)
    timeline = build_timeline_payload(flow_df)
    with open(RUNTIME_OVERVIEW, "w", encoding="utf-8") as file:
        json.dump(overview, file, indent=2)
    with open(RUNTIME_TIMELINE, "w", encoding="utf-8") as file:
        json.dump(timeline, file, indent=2)

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


    # Prepared caches are published before ready=True.
    global OVERVIEW_CACHE, TIMELINE_CACHE
    with STATE_LOCK:
        OVERVIEW_CACHE = overview
        TIMELINE_CACHE = timeline

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
# Precompute all dashboard results BEFORE making results ready
# ============================================================

def normalized_endpoint_key(ip, port):
    """Normalize endpoints for bidirectional 5-tuple matching."""
    if pd.isna(ip) or pd.isna(port):
        return None
    try:
        return f"{str(ip)}:{int(float(port))}"
    except (TypeError, ValueError, OverflowError):
        return None


def add_flow_integrity_states(flow_df, packet_csv):
    """Assign existing packet integrity results to flow time windows.

    Matches protocol + bidirectional endpoints + capture timestamp.
    In case of re-used 5-tuples, a packet is assigned to the latest
    eligible flow start and only if inside that flow's time window.
    No new integrity rules or ML predictions are introduced here.
    """
    source = flow_df.reset_index(drop=True).copy()
    source["integrity_state"] = None
    if source.empty:
        return source

    starts = pd.to_numeric(source["timestamp"], errors="coerce")
    durations = pd.to_numeric(source["flow_duration"], errors="coerce").fillna(0).clip(lower=0)
    protocols = pd.to_numeric(source["ip_protocol"], errors="coerce")

    flow_keys = []
    for row in source.itertuples(index=False):
        a = normalized_endpoint_key(row.src_ip, row.src_port)
        b = normalized_endpoint_key(row.dst_ip, row.dst_port)
        protocol = safe_int(row.ip_protocol)
        flow_keys.append("|".join([str(protocol), *sorted([a, b])]) if a and b and protocol is not None else None)

    reference = pd.DataFrame({
        "_flow_index": np.arange(len(source)),
        "_key": flow_keys,
        "_start": starts,
        "_end": starts + durations + 0.001,
        "_protocol": protocols,
    }).dropna(subset=["_key", "_start"])

    if reference.empty:
        source.to_csv(RUNTIME_FLOW_INTEGRITY, index=False, columns=["flow_id", "integrity_state"])
        return source

    # Store valid windows by normalized five-tuple.
    groups = {}
    for key, group in reference.groupby("_key", sort=False):
        ordered = group.sort_values("_start")
        groups[key] = (
            ordered["_start"].to_numpy(dtype=float),
            ordered["_end"].to_numpy(dtype=float),
            ordered["_flow_index"].to_numpy(dtype=int),
        )

    # 0 = no matching packet, 1 = Intact, 2 = Suspicious, 3 = Tampered.
    severity = np.zeros(len(source), dtype=np.int8)
    packet_counts = np.zeros(len(source), dtype=np.int64)
    rule_counts_by_flow = {
        rule: np.zeros(len(source), dtype=np.int64)
        for rule in ALL_RULES
    }
    usecols = ["timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "ip_protocol", "integrity_state", *ALL_RULES]
    for chunk in pd.read_csv(packet_csv, usecols=usecols, chunksize=INTEGRITY_CHUNK_SIZE, low_memory=False):
        chunk["_time"] = pd.to_numeric(chunk["timestamp"], errors="coerce")
        chunk = chunk.loc[np.isfinite(chunk["_time"])].copy()
        if chunk.empty:
            continue
        chunk["_key"] = [
            "|".join([str(safe_int(proto)), *sorted([a, b])]) if a and b and safe_int(proto) is not None else None
            for a, b, proto in zip(
                (normalized_endpoint_key(ip, port) for ip, port in zip(chunk["src_ip"], chunk["src_port"])),
                (normalized_endpoint_key(ip, port) for ip, port in zip(chunk["dst_ip"], chunk["dst_port"])),
                chunk["ip_protocol"]
            )
        ]
        chunk = chunk.dropna(subset=["_key"])
        for key, packet_group in chunk.groupby("_key", sort=False):
            if key not in groups:
                continue
            flow_starts, flow_ends, flow_indices = groups[key]
            packet_times = packet_group["_time"].to_numpy(dtype=float)
            indices = np.searchsorted(flow_starts, packet_times, side="right") - 1
            valid = indices >= 0
            valid_positions = np.flatnonzero(valid)
            if not len(valid_positions):
                continue
            candidates = indices[valid_positions]
            valid_positions = valid_positions[packet_times[valid_positions] <= flow_ends[candidates]]
            if not len(valid_positions):
                continue
            mapped = flow_indices[indices[valid_positions]]
            states = packet_group["integrity_state"].to_numpy()[valid_positions]
            levels = np.array([{"Intact": 1, "Suspicious": 2, "Tampered": 3}.get(str(v), 0) for v in states], dtype=np.int8)
            np.maximum.at(severity, mapped, levels)
            np.add.at(packet_counts, mapped, 1)
            for rule in ALL_RULES:
                flags = pd.to_numeric(packet_group[rule], errors="coerce").fillna(0).to_numpy()[valid_positions]
                np.add.at(rule_counts_by_flow[rule], mapped, (flags == 1).astype(np.int64))

    for rule, counts in rule_counts_by_flow.items():
        source[f"integrity_rule_count_{rule}"] = counts
    source["integrity_packet_count"] = packet_counts

    states = np.array([None, "Intact", "Suspicious", "Tampered"], dtype=object)
    source["integrity_state"] = states[severity]
    source[["flow_id", "integrity_state"]].to_csv(RUNTIME_FLOW_INTEGRITY, index=False)
    return source


def build_timeline_payload(df):
    times = pd.to_numeric(df["timestamp"], errors="coerce")
    valid = np.isfinite(times)
    if not bool(valid.any()):
        return {"success": True, "buckets": [], "interval_seconds": 0}
    first, last = float(times[valid].min()), float(times[valid].max())
    span = last - first
    if span <= 60:
        interval = 5
    elif span <= 600:
        interval = 30
    elif span <= 3600:
        interval = 60
    elif span <= 21600:
        interval = 300
    else:
        interval = max(300, int(math.ceil(span / 70 / 300)) * 300)
    start = int(math.floor(first / interval) * interval)
    series = pd.DataFrame({
        "bucket": (np.floor(times[valid].to_numpy(dtype=float) / interval).astype(np.int64) * interval),
        "prediction": df.loc[valid, "prediction"].astype(str).to_numpy(),
    })
    counts = series.groupby(["bucket", "prediction"]).size().unstack(fill_value=0)
    # Preserve empty time periods, up to 300 bars (normal range <= ~72).
    final = int(math.floor(last / interval) * interval)
    buckets = []
    for t in range(start, final + interval, interval):
        values = counts.loc[t] if t in counts.index else None
        buckets.append({
            "start": t, "end": t + interval,
            "attack": int(values.get("Attack", 0)) if values is not None else 0,
            "benign": int(values.get("Benign", 0)) if values is not None else 0,
        })
    return {"success": True, "buckets": buckets, "interval_seconds": interval}


def build_overview_payload(df):
    times = pd.to_numeric(df["timestamp"], errors="coerce")
    times = times[np.isfinite(times)]
    attacks = df.loc[df["prediction_label"] == config.LABEL_ATTACK]
    protocols = df["ip_protocol"].map(format_protocol).value_counts()
    overview = {
        "capture_start": float(times.min()) if not times.empty else None,
        "capture_end": float(times.max()) if not times.empty else None,
        "protocol_distribution": [{"name": str(k), "value": int(v)} for k, v in protocols.items()],
        "top_attack_sources": top_counts(attacks, "src_ip", 5),
        "top_attack_destinations": top_counts(attacks, "dst_ip", 5),
        "top_attack_ports": top_counts(attacks, "dst_port", 5),
        "shap_importance": [], "shap_error": None,
    }
    try:
        overview["shap_importance"] = global_shap_importance(df)
    except Exception as exc:
        overview["shap_error"] = str(exc)
    return overview

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

        "integrity_state":
            json_safe(row.get("integrity_state")),
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


    # These integrity counts were prepared during PCAP analysis.
    # The popup therefore does not need to re-scan packet_integrity.csv
    # to show the status and triggered-rule labels.
    precomputed_rules = []
    precomputed_indicators = []
    for rule in ALL_RULES:
        count = safe_int(row.get(f"integrity_rule_count_{rule}")) or 0
        if count <= 0:
            continue
        info = rule_information(rule, count)
        if info["category"] == "Indicator":
            precomputed_indicators.append(info)
        else:
            precomputed_rules.append(info)

    base["integrity_explanation"] = {
        "integrity_state": json_safe(row.get("integrity_state")),
        "total_packets": safe_int(row.get("integrity_packet_count")) or 0,
        "triggered_rules": precomputed_rules,
        "additional_indicators": precomputed_indicators,
    }
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
# Integrity rule descriptions for the Events popup
# ============================================================

INTEGRITY_RULE_DESCRIPTIONS = {
    "T1": "Invalid IPv4 header checksum",
    "T2": "Contradictory IPv4 fragmentation fields",
    "T3": "Invalid IPv4 header length",
    "S1": "Unusually low TTL",
    "S2": "Capture-local MAC-IP inconsistency",
    "S3": "TCP SYN and FIN flags set together",
    "S4": "TCP SYN and RST flags set together",
    "S5": "TCP FIN and RST flags set together",
    "S6": "TCP NULL flags",
    "S7": "Invalid Layer 4 checksum (indicator only)",
}


def integrity_rule_id(column):
    return str(column).split("_", 1)[0]


def rule_information(column, packet_count):
    rule_id = integrity_rule_id(column)
    return {
        "rule_id": rule_id,
        "rule_name": str(column),
        "description": INTEGRITY_RULE_DESCRIPTIONS.get(
            rule_id, str(column).replace("_", " ")
        ),
        "category": (
            "Indicator" if rule_id == "S7"
            else "Tampered" if rule_id.startswith("T")
            else "Suspicious"
        ),
        "packet_count": int(packet_count),
    }


# ============================================================
# Related packet matching
# ============================================================

def get_related_packets(
    flow_id,
):
    """
    Find packet-level rows associated with one bidirectional flow.
    """

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
        *ALL_RULES,
    ]


    # Count rule triggers across ALL matching packets, not only
    # the first 100 packet details shown in the popup.
    triggered_rule_counts = {rule: 0 for rule in ALL_RULES}

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

            pd.to_numeric(
                chunk[
                    "src_port"
                ],
                errors="coerce",
            )
            .eq(
                src_port
            )

            &

            pd.to_numeric(
                chunk[
                    "dst_port"
                ],
                errors="coerce",
            )
            .eq(
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

            pd.to_numeric(
                chunk[
                    "src_port"
                ],
                errors="coerce",
            )
            .eq(
                dst_port
            )

            &

            pd.to_numeric(
                chunk[
                    "dst_port"
                ],
                errors="coerce",
            )
            .eq(
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


        # Rule S7 is an additional indicator. Count packets
        # with that indicator independently of their status.
        indicator_columns = [
            rule for rule in ALL_RULES
            if integrity_rule_id(rule) == "S7"
        ]
        if indicator_columns:
            indicator_mask = pd.Series(False, index=selected.index)
            for rule in indicator_columns:
                indicator_mask |= pd.to_numeric(
                    selected[rule], errors="coerce"
                ).fillna(0).ne(0)
            indicator_count += int(indicator_mask.sum())

        for rule in ALL_RULES:
            triggered_rule_counts[rule] += int(
                pd.to_numeric(selected[rule], errors="coerce")
                .fillna(0).ne(0).sum()
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

                    # Rule flags are authoritative: no guessing
                    # based on the packet's integrity_state.
                    "triggered_rules": [
                        integrity_rule_id(rule)
                        for rule in ALL_RULES
                        if safe_int(packet.get(rule)) == 1
                        and integrity_rule_id(rule) != "S7"
                    ],
                    "indicator_rules": [
                        integrity_rule_id(rule)
                        for rule in ALL_RULES
                        if safe_int(packet.get(rule)) == 1
                        and integrity_rule_id(rule) == "S7"
                    ],
                })


    if total_packets == 0:
        overall_state = None
    elif tampered > 0:
        overall_state = "Tampered"
    elif suspicious > 0:
        overall_state = "Suspicious"
    elif intact > 0:
        overall_state = "Intact"
    else:
        overall_state = None


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

        "triggered_rules": [
            rule_information(rule, count)
            for rule, count in triggered_rule_counts.items()
            if count > 0 and integrity_rule_id(rule) != "S7"
        ],

        "additional_indicators": [
            rule_information(rule, count)
            for rule, count in triggered_rule_counts.items()
            if count > 0 and integrity_rule_id(rule) == "S7"
        ],

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


def build_plain_shap_explanation(
    prediction,
    attack_probability,
    contributions,
):
    """
    Build a simple flow-specific explanation using the actual
    local SHAP values.

    This is deterministic and does not use an LLM.
    """

    positive = [
        item
        for item in contributions
        if item[
            "shap_value"
        ] > 0
    ]


    negative = [
        item
        for item in contributions
        if item[
            "shap_value"
        ] < 0
    ]


    if prediction == "Attack":

        supporting = positive[
            :3
        ]

        opposing = negative[
            :2
        ]

        confidence = (
            attack_probability
            if attack_probability is not None
            else None
        )

        target_word = "Attack"

        support_direction = (
            "increased the model's attack score"
        )

        oppose_direction = (
            "pushed the model toward Benign"
        )

    else:

        supporting = negative[
            :3
        ]

        opposing = positive[
            :2
        ]

        confidence = (

            1.0
            -
            attack_probability

            if attack_probability is not None

            else None
        )

        target_word = "Benign"

        support_direction = (
            "reduced the model's attack score"
        )

        oppose_direction = (
            "pushed the model toward Attack"
        )


    sentences = []


    if confidence is not None:

        sentences.append(
            (
                f"This flow was classified as {target_word} "
                f"with {confidence * 100:.2f}% model confidence."
            )
        )

    else:

        sentences.append(
            (
                f"This flow was classified as {target_word}."
            )
        )


    if supporting:

        strongest = supporting[
            0
        ]

        strongest_label = strongest[
            "feature_label"
        ]

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


        if len(
            supporting
        ) > 1:

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

            else:

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
                    f"{oppose_direction}, but it was not "
                    "strong enough to change the final classification."
                )
            )

        else:

            sentences.append(
                (
                    f"{opposing_names[0].capitalize()} and "
                    f"{opposing_names[1]} {oppose_direction}, "
                    "but their combined influence was not strong "
                    "enough to change the final classification."
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
    """
    Calculate local SHAP explanation for one selected flow.

    Direction:
        positive -> Attack
        negative -> Benign
        zero     -> Neutral
    """

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


        feature_value = (
            json_safe(
                X.iloc[
                    0
                ][
                    feature
                ]
            )
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

    # The landing/upload page stays separate from the results.
    return render_template("upload.html")


@app.route("/dashboard", methods=["GET"])
def dashboard_page():

    # Dashboard data still comes from the existing summary and overview APIs.
    # Without an analysis, return the user to the upload page.
    if not APP_STATE["ready"]:
        return redirect(url_for("index"))

    return render_template("dashboard.html")


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

    try:

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


        # Prevent simultaneous uploads from deleting an active analysis.
        with STATE_LOCK:
            if ANALYSIS_STATUS["analyzing"]:
                return jsonify({"success": False, "error": "An analysis is already running."}), 409
            ANALYSIS_STATUS.update({
                "analyzing": True, "ready": False, "error": None,
                "stage": "extract", "stage_message": "Preparing uploaded capture...",
                "started_at": time.monotonic(), "elapsed_final": 0.0,
            })

        try:
            reset_application_state()
            reset_runtime_directory()
            suffix = Path(original_filename).suffix.lower()
            runtime_input = CURRENT_RUNTIME_DIR / f"input{suffix}"
            file.save(runtime_input)
        except Exception:
            with STATE_LOCK:
                ANALYSIS_STATUS["analyzing"] = False
                ANALYSIS_STATUS["error"] = "Unable to save uploaded capture."
            raise

        def worker():
            global OVERVIEW_CACHE
            try:
                analyze_runtime_capture(runtime_input, source_name=original_filename)
                with STATE_LOCK:
                    ANALYSIS_STATUS["analyzing"] = False
                    ANALYSIS_STATUS["ready"] = True
                    ANALYSIS_STATUS["elapsed_final"] = time.monotonic() - ANALYSIS_STATUS["started_at"]
                    ANALYSIS_STATUS["stage_message"] = "Analysis completed."
            except Exception as exc:
                with STATE_LOCK:
                    ANALYSIS_STATUS["analyzing"] = False
                    ANALYSIS_STATUS["ready"] = False
                    ANALYSIS_STATUS["error"] = str(exc)
                    ANALYSIS_STATUS["elapsed_final"] = time.monotonic() - ANALYSIS_STATUS["started_at"]

        threading.Thread(target=worker, daemon=True).start()
        return jsonify({"success": True, "analyzing": True}), 202


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
# Upload progress & Overview statistics
# ============================================================
@app.route("/api/status", methods=["GET"])
def get_analysis_status():
    with STATE_LOCK:
        result = dict(ANALYSIS_STATUS)
        started_at = result.pop("started_at")
        result["elapsed_seconds"] = (
            time.monotonic() - started_at if result["analyzing"] and started_at
            else result["elapsed_final"]
        )
        result["ready"] = bool(APP_STATE["ready"]) and not result["analyzing"]
    result["success"] = True
    return jsonify(result)


def top_counts(frame, field, limit=5):
    if field not in frame.columns: return []
    values = frame[field].dropna()
    values = values[~values.astype(str).str.strip().str.lower().isin(["", "nan", "none", "null", "inf", "-inf"])]
    counts = values.astype(str).value_counts().head(limit)
    return [{"name": str(k), "value": int(v)} for k, v in counts.items()]


def global_shap_importance(frame, sample_size=40):
    if frame.empty: return []
    sample = frame.sample(n=min(len(frame), sample_size), random_state=42)
    X = sample[config.ML_FEATURES].astype(np.float32)
    explainer = get_shap_explainer()
    with SHAP_LOCK:
        raw = explainer.shap_values(X, check_additivity=False)
    if isinstance(raw, list):
        values = np.asarray(raw[ATTACK_CLASS_INDEX])
    else:
        values = np.asarray(raw)
        if values.ndim == 3:
            values = values[:, :, ATTACK_CLASS_INDEX]
    if values.shape != X.shape:
        raise ValueError(f"SHAP output shape {values.shape} does not match ML inputs {X.shape}.")
    importance = np.mean(np.abs(values), axis=0)
    order = np.argsort(importance)[::-1][:6]
    return [{"name": get_feature_label(str(config.ML_FEATURES[i])), "value": float(importance[i])} for i in order]


@app.route("/api/overview", methods=["GET"])
def get_overview():
    if not APP_STATE["ready"]:
        return jsonify({"success": False, "error": "No analysis is currently available."}), 404
    with STATE_LOCK:
        overview = OVERVIEW_CACHE
    if overview is None:
        return jsonify({"success": False, "error": "Overview data is not prepared."}), 503
    return jsonify({"success": True, "overview": overview})


# ============================================================
# Export PDF report (read-only: reuse the current analysis)
# ============================================================

@app.route("/api/report/pdf", methods=["GET"])
def export_pdf_report():
    from io import BytesIO
    from flask import send_file
    from werkzeug.utils import secure_filename
    from src.report_generator import build_security_report

    # Take a stable reference to the current prepared analysis.
    with STATE_LOCK:
        if not APP_STATE["ready"] or ANALYSIS_STATUS["analyzing"]:
            return jsonify({"success": False, "error": "No completed analysis is available for PDF export."}), 409
        summary = dict(APP_STATE["summary"] or {})
        overview = dict(OVERVIEW_CACHE or {})
        flow_df = APP_STATE["flow_df"]

    if not summary or not overview or flow_df is None:
        return jsonify({"success": False, "error": "The current analysis report data is incomplete."}), 503

    # Only a small subset is needed, so no PCAP reprocessing or SHAP recomputation.
    try:
        attack_rows = flow_df.loc[
            flow_df["prediction_label"] == config.LABEL_ATTACK,
            ["flow_id", "src_ip", "dst_ip", "src_port", "dst_port",
             "ip_protocol", "attack_probability", "integrity_state"]
        ]
        top_attacks = (
            attack_rows.sort_values("attack_probability", ascending=False)
            .head(10).to_dict(orient="records")
        )
        report_bytes = BytesIO()
        build_security_report(report_bytes, summary, overview, top_attacks)
        report_bytes.seek(0)
        filename = secure_filename(str(summary.get("source_name") or "capture"))
        stem = Path(filename).stem[:55] or "capture"
        return send_file(
            report_bytes,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"NetworkIDS_{stem}_Security_Report.pdf",
        )
    except Exception:
        app.logger.exception("PDF export failed")
        return jsonify({"success": False, "error": "Unable to generate the PDF report. Check the Flask console for details."}), 500


# ============================================================
# Events page and timeline
# ============================================================

@app.route("/events", methods=["GET"])
def events_page():
    return render_template("events.html")


@app.route("/api/events/timeline", methods=["GET"])
def events_timeline():
    if not APP_STATE["ready"]:
        return jsonify({"success": False, "error": "No analysis is currently available."}), 404
    with STATE_LOCK:
        timeline = TIMELINE_CACHE
    if timeline is None:
        return jsonify({"success": False, "error": "Timeline data is not prepared."}), 503
    return jsonify(timeline)


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

    from_ts = request.args.get("from_ts", type=float)
    to_ts = request.args.get("to_ts", type=float)
    if from_ts is not None and math.isfinite(from_ts):
        filtered = filtered[pd.to_numeric(filtered["timestamp"], errors="coerce") >= from_ts]
    if to_ts is not None and math.isfinite(to_ts):
        filtered = filtered[pd.to_numeric(filtered["timestamp"], errors="coerce") < to_ts]



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


    elif protocol_filter == "ICMP":
        filtered = filtered[
            pd.to_numeric(filtered["ip_protocol"], errors="coerce").isin([1, 58])
        ]


    # --------------------------------------------------------
    # Search
    #
    # Search:
    #   - source IP
    #   - destination IP
    #   - source port
    #   - destination port
    #   - protocol name
    # --------------------------------------------------------

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


        timestamp_text = (
            pd.to_datetime(
                pd.to_numeric(filtered["timestamp"], errors="coerce"),
                unit="s", errors="coerce", utc=True
            )
            .dt.tz_convert("Asia/Kuala_Lumpur")
            .dt.strftime("%Y-%m-%d %H:%M:%S")
            .fillna("")
            .str.lower()
        )

        search_mask = (

            timestamp_text.str.contains(
                search_query, na=False, regex=False
            )
            |

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


    # --------------------------------------------------------
    # Sort by newest timestamp first.
    #
    # Flow ID remains an internal identifier, but timestamp is
    # the primary visible ordering for the event-style table.
    # --------------------------------------------------------

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
        use_reloader=False,
    )