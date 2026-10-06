from pathlib import Path

import pandas as pd

import config


TRAIN_FILE = config.PROCESSED_DIR / "split" / "train_dataset.csv"
TEST_FILE = config.PROCESSED_DIR / "split" / "test_dataset.csv"

OUTPUT_DIR = config.MODELS_DIR / "robustness" / "port_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_file(path, name):

    print("\n" + "=" * 80)
    print(f"{name.upper()} PORT / LABEL DISTRIBUTION")
    print("=" * 80)

    df = pd.read_csv(path)

    print(f"Dataset: {path}")
    print(f"Rows   : {len(df):,}")

    df["port_group"] = df["dst_port"].apply(
        lambda x: (
            "Missing"
            if pd.isna(x)
            else "Port 80"
            if int(x) == 80
            else "Other port"
        )
    )

    table = pd.crosstab(
        df["port_group"],
        df["label"],
        margins=True
    )

    table = table.rename(
        columns={
            0: "Benign",
            1: "Attack"
        }
    )

    print("\nCounts:")
    print(table)

    # Remove unmatched if any
    labeled = df[df["label"].notna()].copy()

    distribution = (
        labeled
        .groupby("port_group")["label"]
        .agg(
            sample_count="size",
            attack_count=lambda x: int((x == 1).sum()),
            benign_count=lambda x: int((x == 0).sum()),
            attack_rate=lambda x: float((x == 1).mean()),
        )
        .reset_index()
    )

    distribution["attack_rate_percent"] = (
        distribution["attack_rate"] * 100
    )

    print("\nAttack rate:")
    print(
        distribution.to_string(index=False)
    )

    output_file = (
        OUTPUT_DIR /
        f"{name.lower()}_port_label_distribution.csv"
    )

    distribution.to_csv(
        output_file,
        index=False
    )

    print(f"\nSaved: {output_file}")


analyze_file(TRAIN_FILE, "TRAIN")
analyze_file(TEST_FILE, "TEST")

print("\nPASS: Port label distribution analysis completed.")