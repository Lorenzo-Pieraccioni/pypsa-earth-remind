"""
provincial_error_metrics_1N.py
==============================
Confronto distribuzione solare provinciale: 1/N vs load×CF.
Reference: EFC 2020 per carico provinciale.
Output: analysis/provincial/output/CN2020_1N/
"""

import os
import pypsa
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NETWORK_FILE = "results/CN2020_03_hydro_solar1N/networks/elec_s_250_ec_lcopt_3h.nc"
EFC_FILE     = "resources/data/validation/EFC_power_his_data.xlsx"
OUTPUT_DIR   = "analysis/provincial/output/CN2020_1N"
EFC_YEAR     = 2020
IRENA_CN2020 = 253.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def gadm_id_to_admin1(series):
    return series.str.split('.').str[:2].str.join('.') + "_1"

print(f"Loading: {NETWORK_FILE}")
n = pypsa.Network(NETWORK_FILE)
w = n.snapshot_weightings.generators

# ── CARICO PROVINCIALE ────────────────────────────────────────────────────────
load_per_bus = (n.loads_t.p_set.multiply(w, axis=0)).sum() / 1e6
load_per_bus.index = load_per_bus.index.str.replace(r'_(AC|DC)$', '', regex=True)
load_df = load_per_bus.to_frame("load_twh")
load_df["admin1_id"] = gadm_id_to_admin1(pd.Series(load_df.index)).values
load_df["province"]  = load_df["admin1_id"].map(GADM_NAMES)
model_load_by_prov = load_df.groupby("province")["load_twh"].sum()

# ── SOLARE 1/N ────────────────────────────────────────────────────────────────
solar = n.generators[n.generators.carrier == 'solar'].copy()
solar["admin1_id"] = gadm_id_to_admin1(solar.bus.str.replace(r'_(AC|DC)$', '', regex=True))
solar["province"]  = solar["admin1_id"].map(GADM_NAMES)
solar_1n_by_prov   = solar.groupby("province")["p_nom"].sum() / 1e3

# ── SOLARE load×CF ────────────────────────────────────────────────────────────
cf = n.generators_t.p_max_pu[solar.index].mean()
bus_load_annual = (n.loads_t.p_set.multiply(w, axis=0)).sum()
bus_load_annual.index = n.loads.bus[bus_load_annual.index].values
gen_load = solar.bus.map(bus_load_annual).fillna(0.0)
cf_aligned = cf.reindex(solar.index)
hybrid_weight = pd.Series(gen_load.values * cf_aligned.values, index=solar.index)
p_nom_hybrid = IRENA_CN2020 * hybrid_weight / hybrid_weight.sum()
solar["p_nom_hybrid"] = p_nom_hybrid.values
solar_hybrid_by_prov = solar.groupby("province")["p_nom_hybrid"].sum()

# ── EFC 2020 ──────────────────────────────────────────────────────────────────
efc_raw  = pd.read_excel(EFC_FILE, sheet_name="A-1-1", header=5)
efc_prov = efc_raw[efc_raw["region"] != "China"][["region", "lang", EFC_YEAR]].copy()
efc_load = efc_prov[efc_prov["lang"] == "Load"][["region", EFC_YEAR]].copy()
efc_load.columns = ["region", "load_raw"]
efc_load["load_twh"] = efc_load["load_raw"] / 10
efc_load["province"] = efc_load["region"].replace({"InnerMongolia": "Inner Mongolia"})
efc_by_prov = efc_load.set_index("province")["load_twh"]

# ── TABELLA SOLARE ────────────────────────────────────────────────────────────
solar_df = pd.DataFrame({
    "solar_1n_gw":     solar_1n_by_prov,
    "solar_hybrid_gw": solar_hybrid_by_prov,
}).dropna().sort_values("solar_1n_gw", ascending=False)

print(f"\n{'='*60}")
print(f"SOLAR CAPACITY — 1/N vs load×CF (IRENA={IRENA_CN2020} GW)")
print(f"{'='*60}")
print(f"{'Province':<20} {'1/N (GW)':>10} {'loadxCF (GW)':>13}")
print("-"*45)
for prov, row in solar_df.iterrows():
    print(f"{prov:<20} {row.solar_1n_gw:>10.2f} {row.solar_hybrid_gw:>13.2f}")
print(f"\nTotal 1/N:    {solar_df.solar_1n_gw.sum():.1f} GW")
print(f"Total hybrid: {solar_df.solar_hybrid_gw.sum():.1f} GW")
top5_1n     = solar_df.solar_1n_gw.nlargest(5).sum()     / solar_df.solar_1n_gw.sum()     * 100
top5_hybrid = solar_df.solar_hybrid_gw.nlargest(5).sum() / solar_df.solar_hybrid_gw.sum() * 100
print(f"Top 5 share 1/N:    {top5_1n:.1f}%")
print(f"Top 5 share hybrid: {top5_hybrid:.1f}%")

# ── METRICHE CARICO ───────────────────────────────────────────────────────────
load_cmp = pd.DataFrame({"model": model_load_by_prov, "efc": efc_by_prov}).dropna()
load_cmp["error_pct"] = (load_cmp["model"] - load_cmp["efc"]) / load_cmp["efc"] * 100
mae_l  = (load_cmp["model"] - load_cmp["efc"]).abs().mean()
rmse_l = np.sqrt(((load_cmp["model"] - load_cmp["efc"])**2).mean())
mape_l = load_cmp["error_pct"].abs().mean()
bias_l = (load_cmp["model"] - load_cmp["efc"]).mean()
print(f"\nPROVINCIAL LOAD METRICS vs EFC 2020")
print(f"MAE={mae_l:.1f} TWh | RMSE={rmse_l:.1f} TWh | MAPE={mape_l:.1f}% | Bias={bias_l:.1f} TWh")

# ── PLOT 1: Solar 1/N vs load×CF ─────────────────────────────────────────────
x = np.arange(len(solar_df))
width = 0.4
fig, ax = plt.subplots(figsize=(16, 6))
ax.bar(x - width/2, solar_df["solar_1n_gw"],     width, label="1/N",     color="#1f77b4", alpha=0.85)
ax.bar(x + width/2, solar_df["solar_hybrid_gw"], width, label="load×CF", color="#2ca02c", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(solar_df.index, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("GW")
ax.set_title(f"Solar capacity by province: 1/N vs load×CF (CN2020, IRENA={IRENA_CN2020} GW)\n"
             f"Top5 share — 1/N: {top5_1n:.1f}%  |  load×CF: {top5_hybrid:.1f}%")
ax.legend()
ax.grid(True, alpha=0.2, axis="y")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "CN2020_solar_1N_vs_hybrid.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# ── PLOT 2: Carico provinciale ────────────────────────────────────────────────
df_err = load_cmp.sort_values("efc", ascending=False)
x2 = np.arange(len(df_err))
colors = ["#d62728" if v > 0 else "#1f77b4" for v in df_err["error_pct"]]
fig, axes = plt.subplots(2, 1, figsize=(16, 11))
ax = axes[0]
ax.bar(x2 - width/2, df_err["model"], width, label="Model CN2020", color="#1f77b4", alpha=0.85)
ax.bar(x2 + width/2, df_err["efc"],   width, label="EFC 2020",     color="#ff7f0e", alpha=0.85)
ax.set_xticks(x2)
ax.set_xticklabels(df_err.index, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("TWh")
ax.set_title(f"Provincial load: CN2020 vs EFC 2020\n"
             f"MAE={mae_l:.1f} | RMSE={rmse_l:.1f} | MAPE={mape_l:.1f}% | Bias={bias_l:.1f} TWh")
ax.legend()
ax.grid(True, alpha=0.2, axis="y")
ax = axes[1]
ax.bar(x2, df_err["error_pct"], color=colors, alpha=0.85)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x2)
ax.set_xticklabels(df_err.index, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Error %")
ax.set_title("Provincial load error % (red=overestimate, blue=underestimate)")
ax.grid(True, alpha=0.2, axis="y")
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "CN2020_provincial_load_metrics.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# ── CSV ───────────────────────────────────────────────────────────────────────
solar_df.to_csv(os.path.join(OUTPUT_DIR, "CN2020_solar_1N_vs_hybrid.csv"))
load_cmp.to_csv(os.path.join(OUTPUT_DIR, "CN2020_provincial_load_vs_efc.csv"))
print(f"Saved CSVs in {OUTPUT_DIR}")
print("\nDone.")
