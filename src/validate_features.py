from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # intended location: NetworkIDS/src/validate_features.py
DATA_DIRS = {
    "processed": ROOT / "data" / "processed",
    "test": ROOT / "data" / "test",
}

PACKET_SUFFIX = "_packets.csv"
FLOW_SUFFIX = "_flow.csv"
LABEL_NAMES = {"label", "binary_label", "class", "target", "attack_label"}


def inspect_csv(path: Path) -> dict:
    result = {
        "path": path,
        "size": path.stat().st_size if path.exists() else 0,
        "header": None,
        "rows": 0,
        "bad_width_rows": 0,
        "first_bad_width_lines": [],
        "error": None,
        "duplicate_columns": [],
        "label_columns": [],
    }

    if result["size"] == 0:
        result["error"] = "ZERO-BYTE FILE"
        return result

    try:
        with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                result["error"] = "MISSING HEADER"
                return result

            header = [str(col).strip() for col in header]
            result["header"] = header

            seen = set()
            duplicates = []
            for col in header:
                key = col.casefold()
                if key in seen and col not in duplicates:
                    duplicates.append(col)
                seen.add(key)
            result["duplicate_columns"] = duplicates
            result["label_columns"] = [
                col for col in header if col.casefold() in LABEL_NAMES
            ]

            expected_width = len(header)
            for line_no, row in enumerate(reader, start=2):
                if not row or all(not cell.strip() for cell in row):
                    continue
                result["rows"] += 1
                if len(row) != expected_width:
                    result["bad_width_rows"] += 1
                    if len(result["first_bad_width_lines"]) < 3:
                        result["first_bad_width_lines"].append(
                            f"line {line_no}: {len(row)} fields; expected {expected_width}"
                        )
    except (OSError, csv.Error) as exc:
        result["error"] = f"READ ERROR: {exc}"

    return result


def describe_folder(name: str, folder: Path) -> tuple[list[Path], list[Path], list[Path]]:
    print(f"\n=== data\\{name} ===")
    if not folder.exists():
        print(f"ERROR: folder does not exist: {folder}")
        return [], [], []

    csv_files = sorted(p for p in folder.rglob("*.csv") if p.is_file())
    packets = [p for p in csv_files if p.name.lower().endswith(PACKET_SUFFIX)]
    flows = [p for p in csv_files if p.name.lower().endswith(FLOW_SUFFIX)]
    other = [p for p in csv_files if p not in packets and p not in flows]

    print(f"CSV files total : {len(csv_files)}")
    print(f"Packet CSVs     : {len(packets)}")
    print(f"Flow CSVs       : {len(flows)}")
    print(f"Other CSVs      : {len(other)}")
    if other:
        print("Other CSV filenames:")
        for p in other:
            print(f"  - {p.relative_to(ROOT)}")

    return packets, flows, other


def check_pairs(packets: list[Path], flows: list[Path], folder_name: str) -> None:
    packet_names = {p.name for p in packets}
    flow_names = {p.name for p in flows}
    missing_flow = []
    missing_packet = []

    for p in packets:
        expected = p.name[:-len(PACKET_SUFFIX)] + FLOW_SUFFIX
        if expected not in flow_names:
            missing_flow.append((p, expected))

    for f in flows:
        expected = f.name[:-len(FLOW_SUFFIX)] + PACKET_SUFFIX
        if expected not in packet_names:
            missing_packet.append((f, expected))

    print(f"\nPair check ({folder_name}):")
    if not missing_flow and not missing_packet:
        print("  PASS: every packet CSV has a matching flow CSV and vice versa.")
    else:
        for p, expected in missing_flow:
            print(f"  MISSING FLOW for {p.name}: expected {expected}")
        for f, expected in missing_packet:
            print(f"  MISSING PACKET for {f.name}: expected {expected}")


def main() -> None:
    print("=" * 72)
    print("NetworkIDS - Read-only Feature CSV Validation")
    print(f"Project root: {ROOT}")
    print("This script does not edit CSVs, assign labels, or train a model.")
    print("=" * 72)

    all_packet_infos = []
    all_flow_infos = []
    folder_files = {}

    for name, folder in DATA_DIRS.items():
        packets, flows, _ = describe_folder(name, folder)
        folder_files[name] = (packets, flows)
        check_pairs(packets, flows, name)

        for path in packets:
            info = inspect_csv(path)
            all_packet_infos.append((name, info))
        for path in flows:
            info = inspect_csv(path)
            all_flow_infos.append((name, info))

    print("\n" + "=" * 72)
    print("FILE CONTENT CHECK")
    print("=" * 72)

    bad_files = []
    zero_row_files = []
    malformed_rows = []
    duplicate_columns = []
    label_columns = []

    for folder_name, info in all_packet_infos + all_flow_infos:
        short = str(info["path"].relative_to(ROOT))
        if info["error"]:
            bad_files.append((short, info["error"]))
        elif info["rows"] == 0:
            zero_row_files.append(short)
        if info["bad_width_rows"]:
            malformed_rows.append((short, info["bad_width_rows"], info["first_bad_width_lines"]))
        if info["duplicate_columns"]:
            duplicate_columns.append((short, info["duplicate_columns"]))
        if info["label_columns"]:
            label_columns.append((short, info["label_columns"]))

    print(f"Packet CSVs inspected: {len(all_packet_infos)}")
    print(f"Flow CSVs inspected  : {len(all_flow_infos)}")
    print(f"Files with read/empty-file errors: {len(bad_files)}")
    print(f"Files with zero data rows        : {len(zero_row_files)}")
    print(f"Files with inconsistent row width: {len(malformed_rows)}")
    print(f"Files with duplicate column names: {len(duplicate_columns)}")

    if bad_files:
        print("\nRead/empty-file errors:")
        for path, error in bad_files:
            print(f"  - {path}: {error}")
    if zero_row_files:
        print("\nFiles with headers but no data rows:")
        for path in zero_row_files:
            print(f"  - {path}")
    if malformed_rows:
        print("\nRows whose field count does not match the header:")
        for path, count, examples in malformed_rows:
            print(f"  - {path}: {count} malformed row(s)")
            for example in examples:
                print(f"      {example}")
    if duplicate_columns:
        print("\nDuplicate column names:")
        for path, cols in duplicate_columns:
            print(f"  - {path}: {cols}")

    print("\nFLOW SCHEMA CONSISTENCY")
    readable_flows = [
        (folder_name, info) for folder_name, info in all_flow_infos
        if info["header"] is not None and not info["error"]
    ]
    if not readable_flows:
        print("  ERROR: no readable flow CSV headers found.")
    else:
        ref_folder, ref_info = readable_flows[0]
        ref_header = ref_info["header"]
        print(f"  Reference: {ref_info['path'].relative_to(ROOT)}")
        print(f"  Reference column count: {len(ref_header)}")
        mismatches = []
        for folder_name, info in readable_flows[1:]:
            if info["header"] != ref_header:
                current = info["header"]
                missing = [c for c in ref_header if c not in current]
                extra = [c for c in current if c not in ref_header]
                same_set_different_order = not missing and not extra
                mismatches.append((info["path"], missing, extra, same_set_different_order))

        if not mismatches:
            print("  PASS: all readable flow CSVs have identical columns in identical order.")
        else:
            print(f"  WARNING: {len(mismatches)} flow CSV(s) differ from the reference schema:")
            for path, missing, extra, order_diff in mismatches:
                print(f"  - {path.relative_to(ROOT)}")
                if missing:
                    print(f"      Missing columns: {missing}")
                if extra:
                    print(f"      Extra columns: {extra}")
                if order_diff:
                    print("      Column names match, but column order differs.")

    if label_columns:
        print("\nNOTE: label-like columns were found. Review before training:")
        for path, cols in label_columns:
            print(f"  - {path}: {cols}")
    else:
        print("\nLabel columns: none detected (expected at feature-extraction stage).")

    print("\nFLOW ROW COUNTS")
    for folder_name in ("processed", "test"):
        infos = [info for name, info in all_flow_infos if name == folder_name]
        print(f"\n[{folder_name}] ({len(infos)} flow CSVs)")
        for info in sorted(infos, key=lambda x: x["path"].name.lower()):
            status = "ERROR" if info["error"] else ("ZERO ROWS" if info["rows"] == 0 else "OK")
            print(f"  {status:9} {info['rows']:>10,} rows  {info['path'].name}")

    print("\n" + "=" * 72)
    print("Validation finished. No files were modified.")
    print("=" * 72)


if __name__ == "__main__":
    main()
