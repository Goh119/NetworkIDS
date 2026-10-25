
from pathlib import Path
import os
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.config as config

SOURCE_DIR = config.TEST_DIR
OUTPUT_DIR = ROOT / "data" / "test_labeled"
MANIFEST = OUTPUT_DIR / "test_label_manifest.csv"
TEMP_MANIFEST = OUTPUT_DIR / "test_label_manifest.building.tmp"

CHUNK_SIZE = 100_000
EXPECTED_FILES = 39

FEATURES = list(config.ML_FEATURES)
LABEL_COLUMN = config.LABEL_COLUMN
ATTACK_LABEL = config.LABEL_ATTACK


def main():
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(
            f"Source directory not found: {SOURCE_DIR}"
        )

    files = sorted(
        path
        for path in SOURCE_DIR.rglob("*.csv")
        if path.name.lower().endswith("flow.csv")
    )

    if len(files) != EXPECTED_FILES:
        raise ValueError(
            f"Expected {EXPECTED_FILES} flow CSV files, "
            f"found {len(files)}. Stop and review the directory."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_pairs = []
    for source in files:
        output = OUTPUT_DIR / f"{source.stem}_source_labeled.csv"
        temp = OUTPUT_DIR / f".{output.name}.building.tmp"
        output_pairs.append((source, output, temp))

    # Do not overwrite any previous results.
    existing = [
        str(path)
        for _, output, temp in output_pairs
        for path in (output, temp)
        if path.exists()
    ]

    if MANIFEST.exists() or TEMP_MANIFEST.exists():
        existing.extend(
            str(path)
            for path in (MANIFEST, TEMP_MANIFEST)
            if path.exists()
        )

    if existing:
        raise FileExistsError(
            "Existing output files found; nothing was overwritten:\n"
            + "\n".join(existing)
        )

    # Validate every source before creating any labeled copies.
    for source in files:
        header = list(pd.read_csv(source, nrows=0).columns)

        missing_features = [
            feature for feature in FEATURES
            if feature not in header
        ]

        if missing_features:
            raise ValueError(
                f"{source.name} is missing features: "
                f"{missing_features}"
            )

        if LABEL_COLUMN in header:
            raise ValueError(
                f"{source.name} already contains '{LABEL_COLUMN}'. "
                "Inspect its existing labels instead of relabeling."
            )

    manifest_rows = []
    temp_paths = [temp for _, _, temp in output_pairs]

    try:
        for index, (source, output, temp) in enumerate(
            output_pairs, start=1
        ):
            print(
                f"\n[{index}/{len(files)}] {source.name}",
                flush=True,
            )

            rows_written = 0

            reader = pd.read_csv(
                source,
                chunksize=CHUNK_SIZE,
                low_memory=False,
            )

            first_chunk = True

            for chunk in reader:
                # Provisional source-level label:
                # this capture must represent an attack scenario.
                chunk[LABEL_COLUMN] = ATTACK_LABEL

                chunk.to_csv(
                    temp,
                    index=False,
                    mode="w" if first_chunk else "a",
                    header=first_chunk,
                )

                first_chunk = False
                rows_written += len(chunk)

            if rows_written == 0:
                raise ValueError(
                    f"{source.name} contains no data rows."
                )

            manifest_rows.append({
                "source_file": source.name,
                "output_file": output.name,
                "rows_labeled": rows_written,
                "label": int(ATTACK_LABEL),
                "label_name": "Attack",
                "label_method": "Provisional source-level",
                "label_basis": (
                    "Attack-scenario capture; confirm against "
                    "the source PCAP scenario documentation."
                ),
            })

            print(
                f"  Rows: {rows_written:,} | "
                f"Provisional label: {ATTACK_LABEL}",
                flush=True,
            )

        pd.DataFrame(manifest_rows).to_csv(
            TEMP_MANIFEST,
            index=False,
        )

        # Publish the labeled copies only after all inputs are processed.
        for _, output, temp in output_pairs:
            os.replace(temp, output)

        os.replace(TEMP_MANIFEST, MANIFEST)

    except Exception:
        # Preserve original sources; remove incomplete temporary outputs.
        for temp in temp_paths + [TEMP_MANIFEST]:
            if temp.exists():
                temp.unlink()
        raise

    print("\n========== LABELING COMPLETE ==========")
    print(f"Source flow files: {len(files)}")
    print(f"Labeled copies: {OUTPUT_DIR}")
    print(f"Manifest: {MANIFEST}")
    print("Label assigned: Attack (1)")
    print("Original files under data/test were not modified.")
    print(
        "Labels are provisional source-level labels, "
        "not independently verified per-flow ground truth."
    )


if __name__ == "__main__":
    main()
