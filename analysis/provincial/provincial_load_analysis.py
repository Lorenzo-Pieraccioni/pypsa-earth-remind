"""
provincial_load_analysis.py
===========================
Analisi della distribuzione provinciale del carico elettrico.
Confronta il carico modellato (PyPSA-Earth) con i dati reali EFC a livello provinciale.

Uso:
  python analysis/provincial/provincial_load_analysis.py --run CN2020
  python analysis/provincial/provincial_load_analysis.py --run CN2024
"""

import os
import sys
import argparse
import pypsa
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── PARAMETERS ────────────────────────────────────────────────────────────────

RUNS = {
    "CN2020": {
        "network": "results/CN2020_03_hydro_solar1N/networks/elec_s_250_ec_lcopt_3h.nc",
        "efc_year": 2020,
        "output_dir": "analysis/provincial/output/CN2020_1N",
    },
    "CN2024": {
        "network": "results/CN2024_01_hydro_solar1N/networks/elec_s_250_ec_lcopt_3h.nc",
        "efc_year": 2021,  # proxy — EFC 2024 non disponibile
        "output_dir": "analysis/provincial/output/CN2024_1N",
    },
}

EFC_FILE  = "resources/data/validation/EFC_power_his_data.xlsx"
GADM_FILE = "resources/shapes/gadm_shapes.geojson"

GADM_NAMES = {
    "CN.1_1":  "Anhui",          "CN.2_1":  "Beijing",
    "CN.3_1":  "Chongqing",      "CN.4_1":  "Fujian",
    "CN.5_1":  "Gansu",          "CN.6_1":  "Guangdong",
    "CN.7_1":  "Guangxi",        "CN.8_1":  "Guizhou",
    "CN.9_1":  "Hainan",         "CN.10_1": "Hebei",
    "CN.11_1": "Heilongjiang",   "CN.12_1": "Henan",
    "CN.13_1": "Hubei",          "CN.14_1": "Hunan",
    "CN.15_1": "Jiangsu",        "CN.16_1": "Jiangxi",
    "CN.17_1": "Jilin",          "CN.18_1": "Liaoning",
    "CN.19_1": "Inner Mongolia", "CN.20_1": "Ningxia",
    "CN.21_1": "Qinghai",        "CN.22_1": "Shaanxi",
    "CN.23_1": "Shandong",       "CN.24_1": "Shanghai",
    "CN.25_1": "Shanxi",         "CN.26_1": "Sichuan",
    "CN.27_1": "Tianjin",        "CN.28_1": "Xinjiang",
    "CN.29_1": "Tibet",          "CN.30_1": "Yunnan",
    "CN.31_1": "Zhejiang",
}

EFC_TO_GADM = {"InnerMongolia": "Inner Mongolia"}

# ── FUNZIONI ──────────────────────────────────────────────────────────────────

def gadm_id_to_admin1(series):
    return series.str.split('.').str[:2].str.join('.') + "_1"

def load_model_by_province(network_file, gadm_file):
    n = pypsa.Network(network_file)
    w = n.snapshot_weightings.generators
    load_per_bus = (n.loads_t.p_set.multiply(w, axis=0)).sum() / 1e6
    load_per_bus.index = load_per_bus.index.str.replace(r'_(AC|DC)$', '', regex=True)
    load_df = load_per_bus.to_frame("load_twh")
    load_df.index.name = "GADM_ID"
    load_df["admin1_id"] = gadm_id_to_admin1(pd.Series(load_df.index)).values
    load_df["province"]  = load_df["admin1_id"].map(GADM_NAMES)
    model_by_prov = load_df.groupby("province")["load_twh"].sum()
    gadm = gpd.read_file(gadm_file)
    china = gadm[gadm["country"] == "CN"].copy()
    china["admin1_id"] = gadm_id_to_admin1(china["GADM_ID"]).values
    china["province"]  = china["admin1_id"].map(GADM_NAMES)
    china_prov = china.dissolve(by="province").reset_index()
    return model_by_prov, china_prov

def load_efc_by_province(efc_file, year):
    efc = pd.read_excel(efc_file, sheet_name="A-1-1", header=5)
    efc_prov = efc[efc["region"] != "China"][["region", "lang", year]].copy()
    efc_load = efc_prov[efc_prov["lang"] == "Load"][["region", year]].copy()
    efc_load.columns = ["region", "load_raw"]
    efc_load["load_twh"] = efc_load["load_raw"] / 10
    efc_load["province"] = efc_load["region"].replace(EFC_TO_GADM)
    return efc_load.set_index("province")["load_twh"]

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=["CN2020", "CN2024"], required=True)
    args = parser.parse_args()

    cfg = RUNS[args.run]
    OUTPUT_DIR = cfg["output_dir"]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading network: {cfg['network']}")
    model_load, china_prov = load_model_by_province(cfg["network"], GADM_FILE)

    print(f"Loading EFC data: year={cfg['efc_year']}")
    efc_load = load_efc_by_province(EFC_FILE, cfg["efc_year"])

    df = pd.DataFrame({"model": model_load, "efc": efc_load}).dropna()
    df["error_pct"] = (df["model"] - df["efc"]) / df["efc"] * 100
    df = df.sort_values("efc", ascending=False)

    mae  = (df["model"] - df["efc"]).abs().mean()
    rmse = np.sqrt(((df["model"] - df["efc"])**2).mean())
    mape = df["error_pct"].abs().mean()
    bias = (df["model"] - df["efc"]).mean()

    print(f"\n{'Province':<20} {'Model (TWh)':>12} {'EFC (TWh)':>10} {'Error %':>9}")
    print("-" * 54)
    for prov, row in df.iterrows():
        print(f"{str(prov):<20} {row['model']:>12.1f} {row['efc']:>10.1f} {row['error_pct']:>8.1f}%")
    print("-" * 54)
    print(f"{'TOTAL':<20} {df['model'].sum():>12.1f} {df['efc'].sum():>10.1f}")
    print(f"\nMAE={mae:.1f} TWh | RMSE={rmse:.1f} TWh | MAPE={mape:.1f}% | Bias={bias:.1f} TWh")

    # ── Plot 1+2: bar chart + errore % ────────────────────────────────────────
    x = np.arange(len(df))
    width = 0.4
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    ax = axes[0]
    ax.bar(x - width/2, df["model"], width, label=f"Model {args.run}", color="#1f77b4", alpha=0.85)
    ax.bar(x + width/2, df["efc"],   width, label=f"EFC {cfg['efc_year']}", color="#ff7f0e", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(df.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("TWh")
    ax.set_title(f"Provincial electricity load: {args.run} vs EFC {cfg['efc_year']}\nMAE={mae:.1f} TWh | RMSE={rmse:.1f} TWh | MAPE={mape:.1f}%")
    ax.legend(); ax.grid(True, alpha=0.2, axis="y")

    ax = axes[1]
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in df["error_pct"]]
    ax.bar(x, df["error_pct"], color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(df.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Error %")
    ax.set_title(f"Provincial load error {args.run} (red=overestimate, blue=underestimate)")
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"{args.run}_provincial_load_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # ── Plot 3: mappa coropletica ─────────────────────────────────────────────
    china_err = china_prov.merge(df["error_pct"].reset_index(), on="province", how="left")
    fig, ax = plt.subplots(figsize=(14, 10))
    china_err.plot(
        column="error_pct", ax=ax, legend=True, cmap="RdBu_r",
        vmin=-100, vmax=100,
        missing_kwds={"color": "lightgrey"},
        legend_kwds={"label": "Error % (model vs EFC)", "shrink": 0.6}
    )
    ax.set_title(f"{args.run} — Provincial load error %\n(red=overestimate, blue=underestimate)\nMAE={mae:.1f} | RMSE={rmse:.1f} | MAPE={mape:.1f}%", fontsize=12)
    ax.set_axis_off()
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"{args.run}_provincial_load_error_map.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    print("\nDone.")
