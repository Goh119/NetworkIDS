from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TEST_DIR = PROJECT_ROOT / "data" / "test"

RULES_PATH = (
    PROJECT_ROOT
    / "data"
    / "scenario_filter_rules.csv"
)

TEXT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "ground_truth_candidate_inspection.txt"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "ground_truth_candidate_summary.csv"
)

MATCHED_SAMPLE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "ground_truth_candidate_matched_samples.csv"
)

UNMATCHED_SAMPLE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "ground_truth_candidate_unmatched_samples.csv"
)

CHUNK_SIZE = 100_000

TOP_N = 15

SAMPLE_N = 20


# ============================================================
# SAMPLE COLUMNS
# ============================================================

SAMPLE_COLUMNS = [
    "flow_id",
    "timestamp",

    "src_ip",
    "dst_ip",

    "src_port",
    "dst_port",

    "ip_protocol",

    "ttl_mean",

    "flow_duration",
    "flow_packet_count",
    "flow_byte_count",

    "forward_packet_count",
    "reverse_packet_count",

    "syn_count",
    "ack_count",
    "fin_count",
    "rst_count",
    "psh_count",

    "flow_mac_ip_inconsistent_count",
    "flow_mac_ip_spoof_suspect",
]


# ============================================================
# HELPERS
# ============================================================

def normalize_rule_value(value) -> str:
    """
    Empty value or '*' means unrestricted.
    """

    if pd.isna(value):
        return "*"

    text = str(value).strip()

    if not text:
        return "*"

    return text


def safe_percent(
    count: int,
    total: int,
) -> float:

    if total <= 0:
        return 0.0

    return (
        count / total
    ) * 100.0


def locate_flow_file(
    filename: str,
) -> tuple[str, Path] | None:

    processed_path = (
        PROCESSED_DIR
        / filename
    )

    if processed_path.is_file():
        return (
            "processed",
            processed_path,
        )

    test_path = (
        TEST_DIR
        / filename
    )

    if test_path.is_file():
        return (
            "test",
            test_path,
        )

    return None


def write_line(
    handle,
    text: str = "",
) -> None:

    handle.write(
        text + "\n"
    )


# ============================================================
# BUILD RULE MASK
# ============================================================

def build_mask(
    chunk: pd.DataFrame,
    rule: pd.Series,
) -> pd.Series:
    """
    Apply only explicit endpoint/protocol constraints.

    IMPORTANT:
    This function intentionally does NOT use behavioural
    ML features such as:
        - ttl_mean
        - syn_count
        - rst_count
        - flow_packet_count
        - flow_byte_count
        - flow_duration

    Those features may be inspected later, but are not used
    to construct the candidate ground-truth rule.
    """

    mask = pd.Series(
        True,
        index=chunk.index,
    )

    # --------------------------------------------------------
    # Source IP
    # --------------------------------------------------------

    src_ip = normalize_rule_value(
        rule.get(
            "src_ip",
            "*",
        )
    )

    if src_ip != "*":

        mask &= (
            chunk["src_ip"]
            .astype(str)
            .eq(src_ip)
        )

    # --------------------------------------------------------
    # Destination IP
    # --------------------------------------------------------

    dst_ip = normalize_rule_value(
        rule.get(
            "dst_ip",
            "*",
        )
    )

    if dst_ip != "*":

        mask &= (
            chunk["dst_ip"]
            .astype(str)
            .eq(dst_ip)
        )

    # --------------------------------------------------------
    # Destination Port
    # --------------------------------------------------------

    dst_port = normalize_rule_value(
        rule.get(
            "dst_port",
            "*",
        )
    )

    if dst_port != "*":

        numeric_port = pd.to_numeric(
            chunk["dst_port"],
            errors="coerce",
        )

        try:
            expected_port = float(
                dst_port
            )

        except ValueError as exc:

            raise ValueError(
                f"Invalid dst_port "
                f"in rule: {dst_port}"
            ) from exc

        mask &= (
            numeric_port
            == expected_port
        )

    # --------------------------------------------------------
    # IP Protocol
    #
    # TCP = 6
    # UDP = 17
    # --------------------------------------------------------

    protocol = normalize_rule_value(
        rule.get(
            "ip_protocol",
            "*",
        )
    )

    if protocol != "*":

        numeric_protocol = pd.to_numeric(
            chunk["ip_protocol"],
            errors="coerce",
        )

        try:
            expected_protocol = float(
                protocol
            )

        except ValueError as exc:

            raise ValueError(
                f"Invalid ip_protocol "
                f"in rule: {protocol}"
            ) from exc

        mask &= (
            numeric_protocol
            == expected_protocol
        )

    return mask


# ============================================================
# TRAFFIC PROFILE ACCUMULATOR
# ============================================================

class TrafficProfile:
    """
    Collect simple descriptive statistics.

    These statistics are diagnostic only.
    They do NOT create attack labels.
    """

    def __init__(self) -> None:

        self.rows = 0

        self.src_ips = Counter()
        self.dst_ips = Counter()
        self.dst_ports = Counter()

        self.endpoint_pairs = Counter()

        self.protocols = Counter()

    def update(
        self,
        frame: pd.DataFrame,
    ) -> None:

        if frame.empty:
            return

        self.rows += len(frame)

        # ----------------------------------------------------
        # Source IP
        # ----------------------------------------------------

        if "src_ip" in frame.columns:

            values = (
                frame["src_ip"]
                .dropna()
                .astype(str)
            )

            self.src_ips.update(
                values
            )

        # ----------------------------------------------------
        # Destination IP
        # ----------------------------------------------------

        if "dst_ip" in frame.columns:

            values = (
                frame["dst_ip"]
                .dropna()
                .astype(str)
            )

            self.dst_ips.update(
                values
            )

        # ----------------------------------------------------
        # Destination Port
        # ----------------------------------------------------

        if "dst_port" in frame.columns:

            values = (
                frame["dst_port"]
                .dropna()
                .astype(str)
            )

            self.dst_ports.update(
                values
            )

        # ----------------------------------------------------
        # Protocol
        # ----------------------------------------------------

        if "ip_protocol" in frame.columns:

            values = (
                frame["ip_protocol"]
                .dropna()
                .astype(str)
            )

            self.protocols.update(
                values
            )

        # ----------------------------------------------------
        # Endpoint Combination
        #
        # src_ip -> dst_ip:dst_port / protocol
        #
        # Source port is intentionally omitted here because
        # brute-force / scan traffic often uses many ephemeral
        # source ports.
        # ----------------------------------------------------

        required = {
            "src_ip",
            "dst_ip",
            "dst_port",
            "ip_protocol",
        }

        if required.issubset(
            frame.columns
        ):

            subset = frame[
                [
                    "src_ip",
                    "dst_ip",
                    "dst_port",
                    "ip_protocol",
                ]
            ]

            for row in subset.itertuples(
                index=False
            ):

                key = (
                    str(row.src_ip),
                    str(row.dst_ip),
                    str(row.dst_port),
                    str(row.ip_protocol),
                )

                self.endpoint_pairs[
                    key
                ] += 1


# ============================================================
# SAMPLE COLLECTION
# ============================================================

def collect_sample_rows(
    target: list[dict],
    frame: pd.DataFrame,
    rule: pd.Series,
    sample_type: str,
) -> None:
    """
    Collect the first SAMPLE_N deterministic rows.

    This intentionally does NOT use random sampling so the
    inspection is reproducible across runs.
    """

    remaining = (
        SAMPLE_N
        - len(target)
    )

    if remaining <= 0:
        return

    if frame.empty:
        return

    available_columns = [
        column
        for column in SAMPLE_COLUMNS
        if column in frame.columns
    ]

    sample = (
        frame[
            available_columns
        ]
        .head(remaining)
        .copy()
    )

    for _, row in sample.iterrows():

        output = {
            "rule_id": rule["rule_id"],
            "flow_file": rule["flow_file"],
            "family": rule["family"],
            "sample_type": sample_type,
        }

        for column in available_columns:

            output[column] = (
                row[column]
            )

        target.append(
            output
        )


# ============================================================
# TEXT PROFILE WRITER
# ============================================================

def write_counter(
    handle,
    title: str,
    counter: Counter,
    total: int,
) -> None:

    write_line(
        handle,
        f"  {title}:",
    )

    if not counter:

        write_line(
            handle,
            "    <none>",
        )

        return

    for rank, (
        value,
        count,
    ) in enumerate(
        counter.most_common(
            TOP_N
        ),
        start=1,
    ):

        percent = safe_percent(
            count,
            total,
        )

        write_line(
            handle,
            f"    {rank:>2}. "
            f"{value:<45} "
            f"{count:>12,} "
            f"({percent:>8.3f}%)",
        )


def write_endpoint_pairs(
    handle,
    profile: TrafficProfile,
) -> None:

    write_line(
        handle,
        "  Top Endpoint Combinations:",
    )

    if not profile.endpoint_pairs:

        write_line(
            handle,
            "    <none>",
        )

        return

    for rank, (
        key,
        count,
    ) in enumerate(
        profile.endpoint_pairs.most_common(
            TOP_N
        ),
        start=1,
    ):

        (
            src_ip,
            dst_ip,
            dst_port,
            protocol,
        ) = key

        percent = safe_percent(
            count,
            profile.rows,
        )

        write_line(
            handle,
            f"    {rank:>2}. "
            f"{src_ip} "
            f"-> {dst_ip}:{dst_port} "
            f"(protocol={protocol}) "
            f"| {count:,} "
            f"| {percent:.3f}%",
        )


def write_sample_section(
    handle,
    title: str,
    rows: list[dict],
) -> None:

    write_line(handle)
    write_line(
        handle,
        f"  {title}:",
    )

    if not rows:

        write_line(
            handle,
            "    <none>",
        )

        return

    for index, row in enumerate(
        rows,
        start=1,
    ):

        write_line(
            handle,
            f"    Sample {index}",
        )

        for column in SAMPLE_COLUMNS:

            if column not in row:
                continue

            write_line(
                handle,
                f"      {column:<32} "
                f"= {row[column]}",
            )

        write_line(handle)


# ============================================================
# INSPECT ONE RULE
# ============================================================

def inspect_rule(
    rule: pd.Series,
    text_handle,
) -> tuple[
    dict,
    list[dict],
    list[dict],
]:

    filename = str(
        rule["flow_file"]
    ).strip()

    located = locate_flow_file(
        filename
    )

    summary = {
        "rule_id": rule["rule_id"],
        "family": rule["family"],
        "flow_file": filename,
        "rule_status": rule.get(
            "rule_status",
            "CANDIDATE",
        ),

        "src_ip_rule": rule.get(
            "src_ip",
            "*",
        ),

        "dst_ip_rule": rule.get(
            "dst_ip",
            "*",
        ),

        "dst_port_rule": rule.get(
            "dst_port",
            "*",
        ),

        "ip_protocol_rule": rule.get(
            "ip_protocol",
            "*",
        ),

        "total_flows": 0,
        "matched_flows": 0,
        "matched_percent": 0.0,

        "unmatched_flows": 0,
        "unmatched_percent": 0.0,

        "status": "",

        "notes": rule.get(
            "notes",
            "",
        ),
    }

    if located is None:

        summary["status"] = (
            "FILE NOT FOUND"
        )

        return (
            summary,
            [],
            [],
        )

    location, flow_path = located

    matched_profile = (
        TrafficProfile()
    )

    unmatched_profile = (
        TrafficProfile()
    )

    matched_samples: list[dict] = []
    unmatched_samples: list[dict] = []

    total = 0
    matched_total = 0
    unmatched_total = 0

    write_line(text_handle)
    write_line(
        text_handle,
        "=" * 100,
    )

    write_line(
        text_handle,
        f"Rule ID     : "
        f"{rule['rule_id']}",
    )

    write_line(
        text_handle,
        f"Family      : "
        f"{rule['family']}",
    )

    write_line(
        text_handle,
        f"File        : "
        f"{filename}",
    )

    write_line(
        text_handle,
        f"Location    : "
        f"{location}",
    )

    write_line(
        text_handle,
        f"Rule status : "
        f"{rule.get('rule_status', 'CANDIDATE')}",
    )

    write_line(
        text_handle,
        "-" * 100,
    )

    write_line(
        text_handle,
        "Candidate conditions:",
    )

    write_line(
        text_handle,
        f"  src_ip      = "
        f"{rule.get('src_ip', '*')}",
    )

    write_line(
        text_handle,
        f"  dst_ip      = "
        f"{rule.get('dst_ip', '*')}",
    )

    write_line(
        text_handle,
        f"  dst_port    = "
        f"{rule.get('dst_port', '*')}",
    )

    write_line(
        text_handle,
        f"  ip_protocol = "
        f"{rule.get('ip_protocol', '*')}",
    )

    try:

        reader = pd.read_csv(
            flow_path,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        )

        for chunk_number, chunk in enumerate(
            reader,
            start=1,
        ):

            required_columns = {
                "src_ip",
                "dst_ip",
                "dst_port",
                "ip_protocol",
            }

            missing_columns = (
                required_columns
                - set(chunk.columns)
            )

            if missing_columns:

                summary["status"] = (
                    "MISSING COLUMNS: "
                    + ", ".join(
                        sorted(
                            missing_columns
                        )
                    )
                )

                return (
                    summary,
                    matched_samples,
                    unmatched_samples,
                )

            total += len(chunk)

            mask = build_mask(
                chunk,
                rule,
            )

            matched = (
                chunk.loc[mask]
            )

            unmatched = (
                chunk.loc[~mask]
            )

            matched_total += len(
                matched
            )

            unmatched_total += len(
                unmatched
            )

            matched_profile.update(
                matched
            )

            unmatched_profile.update(
                unmatched
            )

            collect_sample_rows(
                matched_samples,
                matched,
                rule,
                "MATCHED",
            )

            collect_sample_rows(
                unmatched_samples,
                unmatched,
                rule,
                "UNMATCHED",
            )

            print(
                f"    chunk {chunk_number}: "
                f"{total:,} flows inspected",
                end="\r",
            )

    except Exception as exc:

        print()

        summary["status"] = (
            f"READ ERROR: {exc}"
        )

        return (
            summary,
            matched_samples,
            unmatched_samples,
        )

    print(
        " " * 100,
        end="\r",
    )

    matched_percent = safe_percent(
        matched_total,
        total,
    )

    unmatched_percent = safe_percent(
        unmatched_total,
        total,
    )

    summary.update({
        "total_flows": total,

        "matched_flows": matched_total,
        "matched_percent": round(
            matched_percent,
            6,
        ),

        "unmatched_flows": unmatched_total,
        "unmatched_percent": round(
            unmatched_percent,
            6,
        ),

        "status": "INSPECTED",
    })

    # ========================================================
    # WRITE SUMMARY
    # ========================================================

    write_line(text_handle)
    write_line(
        text_handle,
        "Coverage:",
    )

    write_line(
        text_handle,
        f"  Total flows   : "
        f"{total:,}",
    )

    write_line(
        text_handle,
        f"  Matched       : "
        f"{matched_total:,} "
        f"({matched_percent:.3f}%)",
    )

    write_line(
        text_handle,
        f"  Unmatched     : "
        f"{unmatched_total:,} "
        f"({unmatched_percent:.3f}%)",
    )

    # ========================================================
    # MATCHED PROFILE
    # ========================================================

    write_line(text_handle)
    write_line(
        text_handle,
        "-" * 100,
    )

    write_line(
        text_handle,
        "MATCHED TRAFFIC PROFILE",
    )

    write_line(
        text_handle,
        "-" * 100,
    )

    write_counter(
        text_handle,
        "Top Source IPs",
        matched_profile.src_ips,
        matched_profile.rows,
    )

    write_line(text_handle)

    write_counter(
        text_handle,
        "Top Destination IPs",
        matched_profile.dst_ips,
        matched_profile.rows,
    )

    write_line(text_handle)

    write_counter(
        text_handle,
        "Top Destination Ports",
        matched_profile.dst_ports,
        matched_profile.rows,
    )

    write_line(text_handle)

    write_counter(
        text_handle,
        "Protocols",
        matched_profile.protocols,
        matched_profile.rows,
    )

    write_line(text_handle)

    write_endpoint_pairs(
        text_handle,
        matched_profile,
    )

    write_sample_section(
        text_handle,
        f"First {SAMPLE_N} Matched Flow Samples",
        matched_samples,
    )

    # ========================================================
    # UNMATCHED PROFILE
    # ========================================================

    write_line(text_handle)
    write_line(
        text_handle,
        "-" * 100,
    )

    write_line(
        text_handle,
        "UNMATCHED TRAFFIC PROFILE",
    )

    write_line(
        text_handle,
        "-" * 100,
    )

    write_counter(
        text_handle,
        "Top Source IPs",
        unmatched_profile.src_ips,
        unmatched_profile.rows,
    )

    write_line(text_handle)

    write_counter(
        text_handle,
        "Top Destination IPs",
        unmatched_profile.dst_ips,
        unmatched_profile.rows,
    )

    write_line(text_handle)

    write_counter(
        text_handle,
        "Top Destination Ports",
        unmatched_profile.dst_ports,
        unmatched_profile.rows,
    )

    write_line(text_handle)

    write_counter(
        text_handle,
        "Protocols",
        unmatched_profile.protocols,
        unmatched_profile.rows,
    )

    write_line(text_handle)

    write_endpoint_pairs(
        text_handle,
        unmatched_profile,
    )

    write_sample_section(
        text_handle,
        f"First {SAMPLE_N} Unmatched Flow Samples",
        unmatched_samples,
    )

    write_line(
        text_handle,
        "=" * 100,
    )

    return (
        summary,
        matched_samples,
        unmatched_samples,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    if not (
        PROJECT_ROOT
        / "src"
        / "flow_builder.py"
    ).exists():

        print(
            "ERROR: Run this script "
            "inside the NetworkIDS project.",
            file=sys.stderr,
        )

        return 2

    if not RULES_PATH.is_file():

        print(
            "ERROR: Candidate rule file "
            "was not found:",
            file=sys.stderr,
        )

        print(
            f"  {RULES_PATH}",
            file=sys.stderr,
        )

        return 2

    rules = pd.read_csv(
        RULES_PATH,
        dtype=str,
        keep_default_na=False,
    )

    required_rule_columns = {
        "flow_file",
        "family",
        "rule_id",
        "src_ip",
        "dst_ip",
        "dst_port",
        "ip_protocol",
    }

    missing_rule_columns = (
        required_rule_columns
        - set(rules.columns)
    )

    if missing_rule_columns:

        print(
            "ERROR: scenario_filter_rules.csv "
            "is missing columns:",
            file=sys.stderr,
        )

        print(
            "  "
            + ", ".join(
                sorted(
                    missing_rule_columns
                )
            ),
            file=sys.stderr,
        )

        return 2

    print(
        "=" * 100
    )

    print(
        "NetworkIDS - Ground Truth Candidate Inspector"
    )

    print(
        "=" * 100
    )

    print(
        f"Candidate rules : "
        f"{len(rules)}"
    )

    print(
        f"Sample rows/rule: "
        f"{SAMPLE_N}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "  This script does NOT assign labels."
    )

    print(
        "  This script does NOT modify source CSVs."
    )

    print(
        "  Rule coverage is NOT ground-truth accuracy."
    )

    print()

    summaries = []

    all_matched_samples = []
    all_unmatched_samples = []

    TEXT_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TEXT_OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as text_handle:

        write_line(
            text_handle,
            "=" * 100,
        )

        write_line(
            text_handle,
            "NetworkIDS - Ground Truth Candidate Inspection",
        )

        write_line(
            text_handle,
            "=" * 100,
        )

        write_line(text_handle)

        write_line(
            text_handle,
            "IMPORTANT:",
        )

        write_line(
            text_handle,
            "Candidate rule coverage is descriptive only.",
        )

        write_line(
            text_handle,
            "A matched flow is NOT automatically confirmed as attack.",
        )

        write_line(
            text_handle,
            "An unmatched flow is NOT automatically confirmed as benign.",
        )

        write_line(
            text_handle,
            "No labels are assigned by this script.",
        )

        write_line(text_handle)

        for index, rule in rules.iterrows():

            print(
                f"[{index + 1}/{len(rules)}] "
                f"{rule['rule_id']} "
                f"-> {rule['flow_file']}"
            )

            (
                summary,
                matched_samples,
                unmatched_samples,
            ) = inspect_rule(
                rule,
                text_handle,
            )

            summaries.append(
                summary
            )

            all_matched_samples.extend(
                matched_samples
            )

            all_unmatched_samples.extend(
                unmatched_samples
            )

    # ========================================================
    # SAVE CSV REPORTS
    # ========================================================

    summary_df = pd.DataFrame(
        summaries
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )

    pd.DataFrame(
        all_matched_samples
    ).to_csv(
        MATCHED_SAMPLE_OUTPUT_PATH,
        index=False,
    )

    pd.DataFrame(
        all_unmatched_samples
    ).to_csv(
        UNMATCHED_SAMPLE_OUTPUT_PATH,
        index=False,
    )

    # ========================================================
    # TERMINAL SUMMARY
    # ========================================================

    print()
    print(
        "=" * 100
    )

    print(
        "GROUND TRUTH CANDIDATE INSPECTION COMPLETE"
    )

    print(
        "=" * 100
    )

    for summary in summaries:

        print(
            f"{summary['rule_id']:<24} "
            f"{summary['matched_flows']:>10,} / "
            f"{summary['total_flows']:>10,} "
            f"matched "
            f"({summary['matched_percent']:>7.2f}%)"
        )

    print()
    print(
        "Reports saved:"
    )

    print(
        f"  {TEXT_OUTPUT_PATH}"
    )

    print(
        f"  {SUMMARY_OUTPUT_PATH}"
    )

    print(
        f"  {MATCHED_SAMPLE_OUTPUT_PATH}"
    )

    print(
        f"  {UNMATCHED_SAMPLE_OUTPUT_PATH}"
    )

    print()
    print(
        "No labels were assigned."
    )

    print(
        "No source datasets were modified."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )