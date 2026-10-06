from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

import config


# ============================================================
# CONFIG
# ============================================================

TEST_FILE = config.PROCESSED_DIR / "split" / "test_dataset.csv"
MODEL_FILE = config.MODELS_DIR / "random_forest_robust_v2.pkl"

OUTPUT_DIR = config.MODELS_DIR / "robustness" / "port_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
SAMPLE_SIZE_PER_CLASS = 2000

ATTACK_LABEL = 1
BENIGN_LABEL = 0

FEATURES = config.ML_FEATURES


# ============================================================
# HELPERS
# ============================================================

def attack_evasion_rate(group):
    if len(group) == 0:
        return np.nan
    return float((group["perturbed_prediction"] == BENIGN_LABEL).mean())


def safe_mean(series):
    if len(series) == 0:
        return np.nan
    return float(series.mean())


def print_table(title, df):
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)

    if df.empty:
        print("No data.")
        return

    print(df.to_string(index=False))


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("DESTINATION PORT BREAKDOWN ANALYSIS")
print("=" * 80)

print(f"Test dataset : {TEST_FILE}")
print(f"Model        : {MODEL_FILE}")
print(f"Output dir   : {OUTPUT_DIR}")

df = pd.read_csv(TEST_FILE)

print(f"\nLoaded test dataset: {len(df):,} rows x {len(df.columns)} columns")

model = joblib.load(MODEL_FILE)

X = df[FEATURES].copy()
y = df["label"].astype(int).to_numpy()

print(f"ML feature matrix: {X.shape}")

# ------------------------------------------------------------
# Clean predictions
# ------------------------------------------------------------

clean_pred = model.predict(X)
clean_prob = model.predict_proba(X)[:, 1]

df["clean_prediction"] = clean_pred
df["clean_attack_probability"] = clean_prob

clean_accuracy = float((clean_pred == y).mean())

print(f"Clean test accuracy: {clean_accuracy:.6f}")


# ============================================================
# REPRODUCE EXACT ROBUSTNESS SAMPLE
# ============================================================

attack_indices = np.where(
    (y == ATTACK_LABEL) &
    (clean_pred == ATTACK_LABEL)
)[0]

benign_indices = np.where(
    (y == BENIGN_LABEL) &
    (clean_pred == BENIGN_LABEL)
)[0]

rng = np.random.default_rng(RANDOM_STATE)

selected_attack_indices = rng.choice(
    attack_indices,
    size=SAMPLE_SIZE_PER_CLASS,
    replace=False
)

selected_benign_indices = rng.choice(
    benign_indices,
    size=SAMPLE_SIZE_PER_CLASS,
    replace=False
)

print("\n" + "-" * 80)
print("ROBUSTNESS SAMPLE")
print("-" * 80)

print(f"Correctly classified Attack pool : {len(attack_indices):,}")
print(f"Correctly classified Benign pool : {len(benign_indices):,}")
print(f"Selected Attack samples          : {len(selected_attack_indices):,}")
print(f"Selected Benign samples          : {len(selected_benign_indices):,}")


# ============================================================
# RANDOMIZE DESTINATION PORT
# ============================================================

X_attack = X.iloc[selected_attack_indices].copy()
X_benign = X.iloc[selected_benign_indices].copy()

attack_original = df.iloc[selected_attack_indices].copy()
benign_original = df.iloc[selected_benign_indices].copy()


def randomize_dst_port(X_input, rng):
    X_out = X_input.copy()

    observed = X_out["dst_port"].notna()

    X_out.loc[observed, "dst_port"] = rng.integers(
        low=1,
        high=65536,
        size=int(observed.sum())
    )

    return X_out


attack_rng = np.random.default_rng(RANDOM_STATE)
benign_rng = np.random.default_rng(RANDOM_STATE)

X_attack_randomized = randomize_dst_port(
    X_attack,
    attack_rng
)

X_benign_randomized = randomize_dst_port(
    X_benign,
    benign_rng
)


# ============================================================
# PREDICT
# ============================================================

attack_perturbed_pred = model.predict(X_attack_randomized)
attack_perturbed_prob = model.predict_proba(
    X_attack_randomized
)[:, 1]

benign_perturbed_pred = model.predict(X_benign_randomized)
benign_perturbed_prob = model.predict_proba(
    X_benign_randomized
)[:, 1]


# ============================================================
# BUILD SAMPLE TABLES
# ============================================================

attack_results = attack_original.copy()

attack_results["sample_type"] = "Attack"

attack_results["baseline_prediction"] = attack_original[
    "clean_prediction"
].to_numpy()

attack_results["perturbed_prediction"] = attack_perturbed_pred

attack_results["baseline_attack_probability"] = attack_original[
    "clean_attack_probability"
].to_numpy()

attack_results["perturbed_attack_probability"] = attack_perturbed_prob

attack_results["probability_shift"] = (
    attack_results["perturbed_attack_probability"]
    - attack_results["baseline_attack_probability"]
).abs()

attack_results["evasion"] = (
    attack_results["perturbed_prediction"] == BENIGN_LABEL
)


benign_results = benign_original.copy()

benign_results["sample_type"] = "Benign"

benign_results["baseline_prediction"] = benign_original[
    "clean_prediction"
].to_numpy()

benign_results["perturbed_prediction"] = benign_perturbed_pred

benign_results["baseline_attack_probability"] = benign_original[
    "clean_attack_probability"
].to_numpy()

benign_results["perturbed_attack_probability"] = benign_perturbed_prob

benign_results["probability_shift"] = (
    benign_results["perturbed_attack_probability"]
    - benign_results["baseline_attack_probability"]
).abs()

benign_results["induction"] = (
    benign_results["perturbed_prediction"] == ATTACK_LABEL
)


# ============================================================
# ATTACK TYPE BREAKDOWN
# ============================================================

attack_label_column = "original_label"

attack_group = (
    attack_results
    .groupby(attack_label_column, dropna=False)
    .agg(
        sample_count=("flow_id", "size"),
        evaded_count=("evasion", "sum"),
        attack_evasion_rate=("evasion", "mean"),
        mean_baseline_probability=(
            "baseline_attack_probability",
            "mean"
        ),
        mean_perturbed_probability=(
            "perturbed_attack_probability",
            "mean"
        ),
        mean_probability_shift=(
            "probability_shift",
            "mean"
        )
    )
    .reset_index()
)

attack_group["attack_evasion_rate"] *= 100


# ============================================================
# DESTINATION PORT BREAKDOWN
# ============================================================

port_group = (
    attack_results
    .groupby("dst_port", dropna=False)
    .agg(
        sample_count=("flow_id", "size"),
        evaded_count=("evasion", "sum"),
        attack_evasion_rate=("evasion", "mean"),
        mean_baseline_probability=(
            "baseline_attack_probability",
            "mean"
        ),
        mean_perturbed_probability=(
            "perturbed_attack_probability",
            "mean"
        ),
        mean_probability_shift=(
            "probability_shift",
            "mean"
        )
    )
    .reset_index()
)

port_group["attack_evasion_rate"] *= 100

port_group = port_group.sort_values(
    ["sample_count", "attack_evasion_rate"],
    ascending=[False, False]
)


# ============================================================
# PORT CATEGORY BREAKDOWN
# ============================================================

def classify_port(port):

    if pd.isna(port):
        return "Missing"

    port = int(port)

    if port == 80:
        return "80"
    elif port == 443:
        return "443"
    elif port < 1024:
        return "Other well-known (<1024)"
    elif port < 49152:
        return "Registered (1024-49151)"
    else:
        return "Dynamic (49152-65535)"


attack_results["dst_port_category"] = (
    attack_results["dst_port"]
    .apply(classify_port)
)

port_category_group = (
    attack_results
    .groupby("dst_port_category", dropna=False)
    .agg(
        sample_count=("flow_id", "size"),
        evaded_count=("evasion", "sum"),
        attack_evasion_rate=("evasion", "mean"),
        mean_baseline_probability=(
            "baseline_attack_probability",
            "mean"
        ),
        mean_perturbed_probability=(
            "perturbed_attack_probability",
            "mean"
        ),
        mean_probability_shift=(
            "probability_shift",
            "mean"
        )
    )
    .reset_index()
)

port_category_group["attack_evasion_rate"] *= 100


# ============================================================
# TOP PORTS
# ============================================================

top_ports = (
    port_group
    .sort_values("sample_count", ascending=False)
    .head(20)
)


# ============================================================
# SAVE
# ============================================================

attack_group.to_csv(
    OUTPUT_DIR / "dst_port_attack_type_breakdown.csv",
    index=False
)

port_group.to_csv(
    OUTPUT_DIR / "dst_port_breakdown.csv",
    index=False
)

port_category_group.to_csv(
    OUTPUT_DIR / "dst_port_category_breakdown.csv",
    index=False
)

top_ports.to_csv(
    OUTPUT_DIR / "dst_port_top20.csv",
    index=False
)

attack_results.to_csv(
    OUTPUT_DIR / "dst_port_attack_sample_results.csv",
    index=False
)

benign_results.to_csv(
    OUTPUT_DIR / "dst_port_benign_sample_results.csv",
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print_table(
    "ATTACK TYPE BREAKDOWN",
    attack_group
)

print_table(
    "DESTINATION PORT CATEGORY BREAKDOWN",
    port_category_group
)

print_table(
    "TOP 20 DESTINATION PORTS BY SAMPLE COUNT",
    top_ports
)


# ============================================================
# OVERALL SUMMARY
# ============================================================

overall_evasion = float(
    attack_results["evasion"].mean() * 100
)

overall_induction = float(
    benign_results["induction"].mean() * 100
)

summary = {
    "test_rows": int(len(df)),
    "clean_accuracy": clean_accuracy,
    "attack_sample_size": int(len(attack_results)),
    "benign_sample_size": int(len(benign_results)),
    "overall_attack_evasion_rate_percent": overall_evasion,
    "overall_benign_induction_rate_percent": overall_induction,
    "random_state": RANDOM_STATE,
    "sample_size_per_class": SAMPLE_SIZE_PER_CLASS,
}

with open(
    OUTPUT_DIR / "dst_port_breakdown_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)


print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(
    f"Overall Attack Evasion Rate : "
    f"{overall_evasion:.4f}%"
)

print(
    f"Overall Benign Induction Rate: "
    f"{overall_induction:.4f}%"
)

print("\nOutputs saved to:")
print(OUTPUT_DIR)

print("\nPASS: Destination-port breakdown completed.")