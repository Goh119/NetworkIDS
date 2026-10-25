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

SUMMARY_PATH = PROJECT_ROOT / "data" / "dataset_profile_summary.csv"
TOP_SRC_PATH = PROJECT_ROOT / "data" / "dataset_profile_top_src_ips.csv"
TOP_DST_PATH = PROJECT_ROOT / "data" / "dataset_profile_top_dst_ips.csv"
TOP_PORT_PATH = PROJECT_ROOT / "data" / "dataset_profile_top_dst_ports.csv"
PAIR_PATH = PROJECT_ROOT / "data" / "dataset_profile_endpoint_pairs.csv"

TEXT_OUTPUT_PATH = PROJECT_ROOT / "data" / "dataset_profile_output.txt"

CHUNK_SIZE = 100_000
TOP_N = 20


# ============================================================
# OUTPUT HELPER
# ============================================================

def write_line(
    handle,
    text: str = "",
) -> None:
    """
    Write one line to the text report.
    """

    handle.write(text + "\n")


# ============================================================
# HELPERS
# ============================================================

def find_flow_files() -> list[tuple[str, Path]]:
    """
    Find only top-level *_flow.csv files.

    This intentionally does NOT recurse into CIC-IDS2017
    subdirectories.
    """

    files: list[tuple[str, Path]] = []

    if PROCESSED_DIR.exists():
        for path in sorted(PROCESSED_DIR.glob("*_flow.csv")):
            files.append(("processed", path))

    if TEST_DIR.exists():
        for path in sorted(TEST_DIR.glob("*_flow.csv")):
            files.append(("test", path))

    return files


def classify_basic_family(path: Path) -> str:
    """
    Lightweight filename-based family classification.

    This is descriptive only.
    It does NOT assign row-level ground truth labels.
    """

    lower = path.name.lower()

    if lower == "benign_whole-network3_flow.csv":
        return "Benign"

    if lower.startswith("attack_bruteforce_dictionary-"):
        return "Bruteforce"

    if lower.startswith("attack_recon_"):
        return "Recon"

    if lower.startswith("attack_malware_mirai-"):
        return "Mirai"

    if lower.startswith("attack_mitm_impersonation_"):
        return "MITM Impersonation"

    if lower.startswith("attack_mitm_ip-spoofing_"):
        return "IP Spoofing"

    if lower.startswith("mqtt-malformed_data_"):
        return "MQTT Malformed"

    if lower.startswith("mqtt-"):
        return "MQTT DoS/DDoS"

    return "Unknown"


def normalize_value(value) -> str:
    if pd.isna(value):
        return "<NA>"

    return str(value)


def safe_percent(
    count: int,
    total: int,
) -> float:
    if total == 0:
        return 0.0

    return round(
        (count / total) * 100.0,
        6,
    )


# ============================================================
# PROFILE ONE FLOW CSV
# ============================================================

def profile_flow_file(
    location: str,
    flow_path: Path,
    text_handle,
) -> tuple[
    dict,
    list[dict],
    list[dict],
    list[dict],
    list[dict],
]:

    write_line(text_handle)
    write_line(text_handle, "=" * 88)
    write_line(
        text_handle,
        f"Profiling : {flow_path}",
    )
    write_line(
        text_handle,
        f"Location  : {location}",
    )
    write_line(text_handle, "=" * 88)

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    total_flows = 0

    tcp_flows = 0
    udp_flows = 0

    unique_src_ips: set[str] = set()
    unique_dst_ips: set[str] = set()
    unique_src_ports: set[str] = set()
    unique_dst_ports: set[str] = set()

    src_ip_counter = Counter()
    dst_ip_counter = Counter()
    dst_port_counter = Counter()
    endpoint_pair_counter = Counter()

    # Optional behaviour counters
    syn_only_like = 0
    syn_flagged = 0
    rst_flagged = 0
    fin_flagged = 0

    # --------------------------------------------------------
    # Chunked read
    # --------------------------------------------------------

    reader = pd.read_csv(
        flow_path,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(
        reader,
        start=1,
    ):

        rows = len(chunk)
        total_flows += rows

        # ----------------------------------------------------
        # Protocol
        # ----------------------------------------------------

        if "ip_protocol" in chunk.columns:

            proto = pd.to_numeric(
                chunk["ip_protocol"],
                errors="coerce",
            )

            tcp_flows += int(
                (proto == 6).sum()
            )

            udp_flows += int(
                (proto == 17).sum()
            )

        # ----------------------------------------------------
        # Source IP
        # ----------------------------------------------------

        if "src_ip" in chunk.columns:

            values = (
                chunk["src_ip"]
                .dropna()
                .astype(str)
            )

            unique_src_ips.update(
                values.unique()
            )

            src_ip_counter.update(
                values
            )

        # ----------------------------------------------------
        # Destination IP
        # ----------------------------------------------------

        if "dst_ip" in chunk.columns:

            values = (
                chunk["dst_ip"]
                .dropna()
                .astype(str)
            )

            unique_dst_ips.update(
                values.unique()
            )

            dst_ip_counter.update(
                values
            )

        # ----------------------------------------------------
        # Source port
        # ----------------------------------------------------

        if "src_port" in chunk.columns:

            values = (
                chunk["src_port"]
                .dropna()
                .astype(str)
            )

            unique_src_ports.update(
                values.unique()
            )

        # ----------------------------------------------------
        # Destination port
        # ----------------------------------------------------

        if "dst_port" in chunk.columns:

            values = (
                chunk["dst_port"]
                .dropna()
                .astype(str)
            )

            unique_dst_ports.update(
                values.unique()
            )

            dst_port_counter.update(
                values
            )

        # ----------------------------------------------------
        # Endpoint pair
        #
        # Direction is preserved:
        # src_ip:src_port -> dst_ip:dst_port
        # ----------------------------------------------------

        required = {
            "src_ip",
            "dst_ip",
            "src_port",
            "dst_port",
            "ip_protocol",
        }

        if required.issubset(chunk.columns):

            for row in chunk[
                [
                    "src_ip",
                    "dst_ip",
                    "src_port",
                    "dst_port",
                    "ip_protocol",
                ]
            ].itertuples(index=False):

                src_ip = normalize_value(
                    row.src_ip
                )

                dst_ip = normalize_value(
                    row.dst_ip
                )

                src_port = normalize_value(
                    row.src_port
                )

                dst_port = normalize_value(
                    row.dst_port
                )

                protocol = normalize_value(
                    row.ip_protocol
                )

                key = (
                    src_ip,
                    src_port,
                    dst_ip,
                    dst_port,
                    protocol,
                )

                endpoint_pair_counter[key] += 1

        # ----------------------------------------------------
        # TCP behavioural indicators
        #
        # These are descriptive only.
        # They are NOT used to assign labels.
        # ----------------------------------------------------

        if "syn_count" in chunk.columns:

            syn = pd.to_numeric(
                chunk["syn_count"],
                errors="coerce",
            ).fillna(0)

            syn_flagged += int(
                (syn > 0).sum()
            )

        else:
            syn = None

        if "rst_count" in chunk.columns:

            rst = pd.to_numeric(
                chunk["rst_count"],
                errors="coerce",
            ).fillna(0)

            rst_flagged += int(
                (rst > 0).sum()
            )

        else:
            rst = None

        if "fin_count" in chunk.columns:

            fin = pd.to_numeric(
                chunk["fin_count"],
                errors="coerce",
            ).fillna(0)

            fin_flagged += int(
                (fin > 0).sum()
            )

        else:
            fin = None

        if (
            syn is not None
            and rst is not None
            and fin is not None
        ):

            syn_only_like += int(
                (
                    (syn > 0)
                    & (rst == 0)
                    & (fin == 0)
                ).sum()
            )

        # Short terminal progress only
        print(
            f"[{location}] {flow_path.name} "
            f"- {total_flows:,} flows",
            end="\r",
        )

    print(
        " " * 140,
        end="\r",
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    top_src_ip = ""
    top_src_ip_count = 0

    if src_ip_counter:
        top_src_ip, top_src_ip_count = (
            src_ip_counter.most_common(1)[0]
        )

    top_dst_ip = ""
    top_dst_ip_count = 0

    if dst_ip_counter:
        top_dst_ip, top_dst_ip_count = (
            dst_ip_counter.most_common(1)[0]
        )

    top_dst_port = ""
    top_dst_port_count = 0

    if dst_port_counter:
        top_dst_port, top_dst_port_count = (
            dst_port_counter.most_common(1)[0]
        )

    summary = {
        "location": location,
        "flow_file": flow_path.name,
        "attack_family": classify_basic_family(
            flow_path
        ),

        "total_flows": total_flows,

        "tcp_flows": tcp_flows,
        "udp_flows": udp_flows,

        "tcp_percent": safe_percent(
            tcp_flows,
            total_flows,
        ),

        "udp_percent": safe_percent(
            udp_flows,
            total_flows,
        ),

        "unique_src_ips": len(
            unique_src_ips
        ),

        "unique_dst_ips": len(
            unique_dst_ips
        ),

        "unique_src_ports": len(
            unique_src_ports
        ),

        "unique_dst_ports": len(
            unique_dst_ports
        ),

        "top_src_ip": top_src_ip,
        "top_src_ip_flows": top_src_ip_count,
        "top_src_ip_percent": safe_percent(
            top_src_ip_count,
            total_flows,
        ),

        "top_dst_ip": top_dst_ip,
        "top_dst_ip_flows": top_dst_ip_count,
        "top_dst_ip_percent": safe_percent(
            top_dst_ip_count,
            total_flows,
        ),

        "top_dst_port": top_dst_port,
        "top_dst_port_flows": top_dst_port_count,
        "top_dst_port_percent": safe_percent(
            top_dst_port_count,
            total_flows,
        ),

        "flows_with_syn": syn_flagged,
        "flows_with_rst": rst_flagged,
        "flows_with_fin": fin_flagged,
        "syn_without_rst_fin": syn_only_like,
    }

    # --------------------------------------------------------
    # Detail reports
    # --------------------------------------------------------

    top_src_rows = []

    for rank, (ip, count) in enumerate(
        src_ip_counter.most_common(TOP_N),
        start=1,
    ):

        top_src_rows.append({
            "location": location,
            "flow_file": flow_path.name,
            "attack_family": classify_basic_family(
                flow_path
            ),
            "rank": rank,
            "src_ip": ip,
            "flow_count": count,
            "percent_of_file": safe_percent(
                count,
                total_flows,
            ),
        })

    top_dst_rows = []

    for rank, (ip, count) in enumerate(
        dst_ip_counter.most_common(TOP_N),
        start=1,
    ):

        top_dst_rows.append({
            "location": location,
            "flow_file": flow_path.name,
            "attack_family": classify_basic_family(
                flow_path
            ),
            "rank": rank,
            "dst_ip": ip,
            "flow_count": count,
            "percent_of_file": safe_percent(
                count,
                total_flows,
            ),
        })

    top_port_rows = []

    for rank, (port, count) in enumerate(
        dst_port_counter.most_common(TOP_N),
        start=1,
    ):

        top_port_rows.append({
            "location": location,
            "flow_file": flow_path.name,
            "attack_family": classify_basic_family(
                flow_path
            ),
            "rank": rank,
            "dst_port": port,
            "flow_count": count,
            "percent_of_file": safe_percent(
                count,
                total_flows,
            ),
        })

    pair_rows = []

    for rank, (key, count) in enumerate(
        endpoint_pair_counter.most_common(TOP_N),
        start=1,
    ):

        (
            src_ip,
            src_port,
            dst_ip,
            dst_port,
            protocol,
        ) = key

        pair_rows.append({
            "location": location,
            "flow_file": flow_path.name,
            "attack_family": classify_basic_family(
                flow_path
            ),
            "rank": rank,

            "src_ip": src_ip,
            "src_port": src_port,

            "dst_ip": dst_ip,
            "dst_port": dst_port,

            "ip_protocol": protocol,

            "flow_count": count,

            "percent_of_file": safe_percent(
                count,
                total_flows,
            ),
        })

    # --------------------------------------------------------
    # Full text report
    # --------------------------------------------------------

    write_line(
        text_handle,
        f"  Total flows       : {total_flows:,}",
    )

    write_line(
        text_handle,
        f"  TCP / UDP         : "
        f"{tcp_flows:,} / {udp_flows:,}",
    )

    write_line(
        text_handle,
        f"  Unique src IPs    : "
        f"{len(unique_src_ips):,}",
    )

    write_line(
        text_handle,
        f"  Unique dst IPs    : "
        f"{len(unique_dst_ips):,}",
    )

    write_line(
        text_handle,
        f"  Unique src ports  : "
        f"{len(unique_src_ports):,}",
    )

    write_line(
        text_handle,
        f"  Unique dst ports  : "
        f"{len(unique_dst_ports):,}",
    )

    write_line(
        text_handle,
        f"  Top src IP        : "
        f"{top_src_ip} "
        f"({top_src_ip_count:,}, "
        f"{safe_percent(top_src_ip_count, total_flows):.2f}%)",
    )

    write_line(
        text_handle,
        f"  Top dst IP        : "
        f"{top_dst_ip} "
        f"({top_dst_ip_count:,}, "
        f"{safe_percent(top_dst_ip_count, total_flows):.2f}%)",
    )

    write_line(
        text_handle,
        f"  Top dst port      : "
        f"{top_dst_port} "
        f"({top_dst_port_count:,}, "
        f"{safe_percent(top_dst_port_count, total_flows):.2f}%)",
    )

    write_line(
        text_handle,
        f"  Flows with SYN    : "
        f"{syn_flagged:,}",
    )

    write_line(
        text_handle,
        f"  Flows with RST    : "
        f"{rst_flagged:,}",
    )

    write_line(
        text_handle,
        f"  Flows with FIN    : "
        f"{fin_flagged:,}",
    )

    write_line(
        text_handle,
        f"  SYN no FIN/RST    : "
        f"{syn_only_like:,}",
    )

    # --------------------------------------------------------
    # Top source IP detail
    # --------------------------------------------------------

    write_line(text_handle)
    write_line(
        text_handle,
        f"  Top {TOP_N} Source IPs:",
    )

    for rank, (ip, count) in enumerate(
        src_ip_counter.most_common(TOP_N),
        start=1,
    ):

        write_line(
            text_handle,
            f"    {rank:>2}. "
            f"{ip:<40} "
            f"{count:>12,} "
            f"({safe_percent(count, total_flows):>8.3f}%)",
        )

    # --------------------------------------------------------
    # Top destination IP detail
    # --------------------------------------------------------

    write_line(text_handle)
    write_line(
        text_handle,
        f"  Top {TOP_N} Destination IPs:",
    )

    for rank, (ip, count) in enumerate(
        dst_ip_counter.most_common(TOP_N),
        start=1,
    ):

        write_line(
            text_handle,
            f"    {rank:>2}. "
            f"{ip:<40} "
            f"{count:>12,} "
            f"({safe_percent(count, total_flows):>8.3f}%)",
        )

    # --------------------------------------------------------
    # Top destination ports detail
    # --------------------------------------------------------

    write_line(text_handle)
    write_line(
        text_handle,
        f"  Top {TOP_N} Destination Ports:",
    )

    for rank, (port, count) in enumerate(
        dst_port_counter.most_common(TOP_N),
        start=1,
    ):

        write_line(
            text_handle,
            f"    {rank:>2}. "
            f"{port:<12} "
            f"{count:>12,} "
            f"({safe_percent(count, total_flows):>8.3f}%)",
        )

    # --------------------------------------------------------
    # Top endpoint pairs detail
    # --------------------------------------------------------

    write_line(text_handle)
    write_line(
        text_handle,
        f"  Top {TOP_N} Endpoint Pairs:",
    )

    for rank, (key, count) in enumerate(
        endpoint_pair_counter.most_common(TOP_N),
        start=1,
    ):

        (
            src_ip,
            src_port,
            dst_ip,
            dst_port,
            protocol,
        ) = key

        write_line(
            text_handle,
            f"    {rank:>2}. "
            f"{src_ip}:{src_port} "
            f"-> "
            f"{dst_ip}:{dst_port} "
            f"(protocol={protocol}) "
            f"| flows={count:,} "
            f"| {safe_percent(count, total_flows):.3f}%",
        )

    # Short terminal completion line
    print(
        f"[DONE] {location:<9} | "
        f"{flow_path.name} | "
        f"{total_flows:,} flows"
    )

    return (
        summary,
        top_src_rows,
        top_dst_rows,
        top_port_rows,
        pair_rows,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    if not (
        PROJECT_ROOT
        / "src"
        / "packet_parser.py"
    ).exists():

        print(
            "ERROR: Run this script from the "
            "NetworkIDS project.",
            file=sys.stderr,
        )

        return 2

    flow_files = find_flow_files()

    if not flow_files:

        print(
            "ERROR: No top-level *_flow.csv files found.",
            file=sys.stderr,
        )

        return 1

    print("=" * 88)
    print("NetworkIDS - New Dataset Traffic Profiling")
    print("=" * 88)
    print(f"Flow files found : {len(flow_files)}")
    print(f"Top-N details    : {TOP_N}")
    print()
    print(
        "Terminal output is shortened."
    )
    print(
        "Full profiling details will be written to:"
    )
    print(
        f"  {TEXT_OUTPUT_PATH}"
    )
    print()

    summaries = []
    src_rows = []
    dst_rows = []
    port_rows = []
    pair_rows = []

    # ========================================================
    # TEXT REPORT
    # ========================================================

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
            "=" * 88,
        )

        write_line(
            text_handle,
            "NetworkIDS - New Dataset Traffic Profiling",
        )

        write_line(
            text_handle,
            "=" * 88,
        )

        write_line(
            text_handle,
            f"Project root     : {PROJECT_ROOT}",
        )

        write_line(
            text_handle,
            f"Flow files found : {len(flow_files)}",
        )

        write_line(
            text_handle,
            f"Top-N details    : {TOP_N}",
        )

        write_line(text_handle)

        # ----------------------------------------------------
        # Process each file
        # ----------------------------------------------------

        for index, (
            location,
            flow_path,
        ) in enumerate(
            flow_files,
            start=1,
        ):

            print(
                f"[{index}/{len(flow_files)}] "
                f"Profiling {flow_path.name}"
            )

            (
                summary,
                file_src_rows,
                file_dst_rows,
                file_port_rows,
                file_pair_rows,
            ) = profile_flow_file(
                location,
                flow_path,
                text_handle,
            )

            summaries.append(
                summary
            )

            src_rows.extend(
                file_src_rows
            )

            dst_rows.extend(
                file_dst_rows
            )

            port_rows.extend(
                file_port_rows
            )

            pair_rows.extend(
                file_pair_rows
            )

        # ====================================================
        # SAVE CSV REPORTS
        # ====================================================

        summary_df = pd.DataFrame(
            summaries
        )

        summary_df.to_csv(
            SUMMARY_PATH,
            index=False,
        )

        pd.DataFrame(
            src_rows
        ).to_csv(
            TOP_SRC_PATH,
            index=False,
        )

        pd.DataFrame(
            dst_rows
        ).to_csv(
            TOP_DST_PATH,
            index=False,
        )

        pd.DataFrame(
            port_rows
        ).to_csv(
            TOP_PORT_PATH,
            index=False,
        )

        pd.DataFrame(
            pair_rows
        ).to_csv(
            PAIR_PATH,
            index=False,
        )

        # ====================================================
        # FULL SUMMARY INTO TXT
        # ====================================================

        write_line(text_handle)
        write_line(
            text_handle,
            "=" * 88,
        )

        write_line(
            text_handle,
            "DATASET PROFILING SUMMARY",
        )

        write_line(
            text_handle,
            "=" * 88,
        )

        display_columns = [
            "location",
            "attack_family",
            "flow_file",
            "total_flows",
            "unique_src_ips",
            "unique_dst_ips",
            "unique_dst_ports",
            "top_src_ip",
            "top_dst_ip",
            "top_dst_port",
        ]

        with pd.option_context(
            "display.max_rows",
            None,
            "display.max_colwidth",
            55,
            "display.width",
            240,
        ):

            summary_text = (
                summary_df[
                    display_columns
                ]
                .to_string(
                    index=False
                )
            )

        write_line(
            text_handle,
            summary_text,
        )

        write_line(text_handle)
        write_line(
            text_handle,
            "Reports saved:",
        )

        write_line(
            text_handle,
            f"  {SUMMARY_PATH}",
        )

        write_line(
            text_handle,
            f"  {TOP_SRC_PATH}",
        )

        write_line(
            text_handle,
            f"  {TOP_DST_PATH}",
        )

        write_line(
            text_handle,
            f"  {TOP_PORT_PATH}",
        )

        write_line(
            text_handle,
            f"  {PAIR_PATH}",
        )

        write_line(
            text_handle,
            f"  {TEXT_OUTPUT_PATH}",
        )

        write_line(text_handle)

        write_line(
            text_handle,
            "IMPORTANT: These reports are descriptive only.",
        )

        write_line(
            text_handle,
            "No row-level attack or benign labels were assigned.",
        )

        write_line(
            text_handle,
            "Use these reports to design scenario-specific "
            "ground-truth filtering rules.",
        )

    # ========================================================
    # SHORT TERMINAL SUMMARY
    # ========================================================

    total_flows = int(
        sum(
            row["total_flows"]
            for row in summaries
        )
    )

    processed_count = sum(
        1
        for row in summaries
        if row["location"] == "processed"
    )

    test_count = sum(
        1
        for row in summaries
        if row["location"] == "test"
    )

    print()
    print("=" * 88)
    print("DATASET PROFILING COMPLETE")
    print("=" * 88)

    print(
        f"Files profiled    : "
        f"{len(summaries)}"
    )

    print(
        f"Processed files   : "
        f"{processed_count}"
    )

    print(
        f"Test files        : "
        f"{test_count}"
    )

    print(
        f"Total flows       : "
        f"{total_flows:,}"
    )

    print()
    print(
        "Full text output:"
    )

    print(
        f"  {TEXT_OUTPUT_PATH}"
    )

    print()
    print(
        "CSV reports:"
    )

    print(
        f"  {SUMMARY_PATH}"
    )

    print(
        f"  {TOP_SRC_PATH}"
    )

    print(
        f"  {TOP_DST_PATH}"
    )

    print(
        f"  {TOP_PORT_PATH}"
    )

    print(
        f"  {PAIR_PATH}"
    )

    print()
    print(
        "No labels were assigned."
    )

    print(
        "No dataset files were modified."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )