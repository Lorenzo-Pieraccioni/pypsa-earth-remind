"""
solar_cf_counterfactual.py
==========================
Counterfactual analysis: compare ERA5 CF under 1/N vs load×CF distribution.

For each bus computes:
  - CF_ERA5: mean(p_max_pu) — the raw ERA5 solar resource
  - weight_1N: uniform (1/N per province)
  - weight_loadCF: load × CF_ERA5 (Ivan's method)

Then computes national weighted-average CF under each weighting scheme
and compares to observed CF from Ember/IRENA.

Output:
  solar_cf_counterfactual_{year}.csv  — per-bus table
  solar_cf_counterfactual_{year}.png  — scatter + bar summary

Usage:
  python analysis/network/solar_cf_counterfactual.py \
    results/CN2020_03_hydro_solar1N/networks/elec_s_250_ec_lcopt_3h.nc \
    --year 2020 --ember_twh 261.1 --irena_gw 253.0

  python analysis/network/solar_cf_counterfactual.py \
    results/CN2024_01_hydro_solar1N/networks/elec_s_250_ec_lcopt_3h.nc \
    --year 2024 --ember_twh 839.0 --irena_gw 885.7
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pypsa

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("network_file")
parser.add_argument("--year", default="")
parser.add_argument("--ember_twh", type=float, required=True,
                    help="Observed solar generation from Ember (TWh)")
parser.add_argument("--irena_gw", type=float, required=True,
                    help="Observed solar capacity from IRENA (GW)")
args = parser.parse_args()

year = args.year
cf_observed = args.ember_twh / (args.irena_gw * 8.76)

run_name = os.path.basename(os.path.dirname(os.path.dirname(args.network_file)))
output_dir = os.path.join(os.environ.get("PYPSA_OUTPUT_DIR", os.path.join("analysis","network","output", run_name)), "solar_cf_counterfactual")
os.makedirs(output_dir, exist_ok=True)

# ── LOAD NETWORK ─────────────────────────────────────────────────────────────
print(f"\nLoading: {args.network_file}")
n = pypsa.Network(args.network_file)
weights_t = n.snapshot_weightings.generators

# ── SOLAR GENERATORS ─────────────────────────────────────────────────────────
solar = n.generators[n.generators.carrier == "solar"].copy()
print(f"Solar generators: {len(solar)}")

# ERA5 CF per bus = mean(p_max_pu)
p_max = n.generators_t.p_max_pu.reindex(columns=solar.index).fillna(0)
cf_era5 = (p_max.multiply(weights_t, axis=0).sum() /
           weights_t.sum() /
           solar.p_nom_opt.replace(0, np.nan)).fillna(0)
# simpler: weighted mean of p_max_pu
cf_era5 = (p_max.multiply(weights_t, axis=0)).sum() / weights_t.sum()

# ── LOAD PER BUS ──────────────────────────────────────────────────────────────
load_per_bus = n.loads_t.p_set.mean()
bus_province = n.loads.bus.map(lambda b: ".".join(b.split(".")[:2]))
province_load = load_per_bus.groupby(bus_province).sum()

gen_province = solar.bus.map(lambda b: ".".join(b.split(".")[:2]))

# ── WEIGHTS ──────────────────────────────────────────────────────────────────
# Weight 1/N: uniform within province
province_n = gen_province.value_counts()
w_1N = 1.0 / gen_province.map(province_n)
w_1N = w_1N / w_1N.groupby(gen_province).transform("sum")  # normalise to sum=1 per province

# Weight load×CF: load_province × cf_era5_bus
province_load_mapped = gen_province.map(province_load).fillna(0)
w_loadCF_raw = cf_era5 * province_load_mapped
# normalise per province
w_loadCF = w_loadCF_raw / w_loadCF_raw.groupby(gen_province).transform("sum")
w_loadCF = w_loadCF.fillna(0)

# ── IRENA CAPACITY PER PROVINCE ───────────────────────────────────────────────
# Use model capacity distribution as proxy (total is fixed to IRENA national)
cap_per_bus = solar.p_nom_opt  # GW (already in MW — divide by 1000)

# ── NATIONAL WEIGHTED CF ─────────────────────────────────────────────────────
# Under 1/N: capacity is uniform per bus within province
# national CF = sum(cf_era5_bus * cap_bus_1N) / sum(cap_bus_1N)
# Since 1/N gives uniform cap, this simplifies to mean cf_era5 weighted by province size

total_cap_gw = args.irena_gw  # fixed to IRENA

# 1/N: each bus gets same capacity = total / N_buses
n_buses = len(solar)
cap_1N = pd.Series(total_cap_gw * 1e3 / n_buses, index=solar.index)  # MW
cf_national_1N = (cf_era5 * cap_1N).sum() / cap_1N.sum()

# load×CF: capacity proportional to load×CF weight within province
# total provincial capacity allocated proportionally
cap_loadCF = w_loadCF * cap_1N.groupby(gen_province).transform("sum")
cf_national_loadCF = (cf_era5 * cap_loadCF).sum() / cap_loadCF.sum()

# ── PER-BUS TABLE ─────────────────────────────────────────────────────────────
df = pd.DataFrame({
    "bus": solar.index,
    "province": gen_province.values,
    "cf_era5": cf_era5.values,
    "cap_1N_MW": cap_1N.values,
    "cap_loadCF_MW": cap_loadCF.values,
    "w_1N": w_1N.values,
    "w_loadCF": w_loadCF.values,
})
csv_path = os.path.join(output_dir, f"solar_cf_counterfactual_{year}.csv")
df.to_csv(csv_path, index=False)

# ── PROVINCIAL SUMMARY ────────────────────────────────────────────────────────
prov = df.groupby("province").agg(
    cf_era5_mean=("cf_era5", "mean"),
    cap_1N_GW=("cap_1N_MW", lambda x: x.sum() / 1e3),
    cap_loadCF_GW=("cap_loadCF_MW", lambda x: x.sum() / 1e3),
).reset_index()
prov["gen_1N_TWh"] = prov.cf_era5_mean * prov.cap_1N_GW * 8.76
prov["gen_loadCF_TWh"] = prov.cf_era5_mean * prov.cap_loadCF_GW * 8.76

# ── RESULTS ───────────────────────────────────────────────────────────────────
gen_1N_TWh = cf_national_1N * total_cap_gw * 8.76
gen_loadCF_TWh = cf_national_loadCF * total_cap_gw * 8.76

print(f"\n=== SOLAR CF COUNTERFACTUAL ANALYSIS — CN{year} ===")
print(f"\nObserved (Ember/IRENA):")
print(f"  Capacity:   {args.irena_gw:.1f} GW")
print(f"  Generation: {args.ember_twh:.1f} TWh")
print(f"  CF:         {cf_observed:.4f}")
print(f"\nModel with 1/N distribution:")
print(f"  National CF (ERA5-weighted): {cf_national_1N:.4f}")
print(f"  Expected generation:         {gen_1N_TWh:.1f} TWh")
print(f"  Error vs Ember:              {(gen_1N_TWh - args.ember_twh)/args.ember_twh*100:+.1f}%")
print(f"\nCounterfactual with load×CF distribution:")
print(f"  National CF (ERA5-weighted): {cf_national_loadCF:.4f}")
print(f"  Expected generation:         {gen_loadCF_TWh:.1f} TWh")
print(f"  Error vs Ember:              {(gen_loadCF_TWh - args.ember_twh)/args.ember_twh*100:+.1f}%")
print(f"\nKey question: does ERA5 itself overestimate CF?")
print(f"  CF ERA5 under 1/N:     {cf_national_1N:.4f}")
print(f"  CF ERA5 under load×CF: {cf_national_loadCF:.4f}")
print(f"  CF observed:           {cf_observed:.4f}")
print(f"  Residual gap (load×CF vs observed): {cf_national_loadCF - cf_observed:+.4f}")
if cf_national_loadCF > cf_observed:
    print(f"  → ERA5 still overestimates CF even with load×CF distribution")
    print(f"  → ERA5 bias confirmed as independent cause")
else:
    print(f"  → Distribution alone explains the gap; ERA5 bias not demonstrated")

# ── FIGURE ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: CF comparison bar chart
ax1 = axes[0]
labels = ["Observed\n(Ember/IRENA)", "Model\n1/N distrib.", "Counterfactual\nload×CF distrib."]
values = [cf_observed, cf_national_1N, cf_national_loadCF]
colors = ["#2196F3", "#FF5722", "#4CAF50"]
bars = ax1.bar(labels, values, color=colors, width=0.5, edgecolor="black", linewidth=0.8)
for bar, val in zip(bars, values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax1.set_ylabel("National solar capacity factor", fontsize=11)
ax1.set_title(f"Solar CF comparison — CN{year}", fontsize=12, fontweight="bold")
ax1.set_ylim(0, max(values) * 1.2)
ax1.axhline(cf_observed, color="#2196F3", linestyle="--", alpha=0.5)
ax1.grid(True, axis="y", alpha=0.3)

# Panel 2: Provincial CF scatter — 1/N vs load×CF capacity allocation
ax2 = axes[1]
prov_sorted = prov.sort_values("cf_era5_mean", ascending=False).head(31)
x = range(len(prov_sorted))
ax2.bar([i - 0.2 for i in x], prov_sorted.cap_1N_GW,
        width=0.4, label="1/N capacity (GW)", color="#FF5722", alpha=0.7)
ax2.bar([i + 0.2 for i in x], prov_sorted.cap_loadCF_GW,
        width=0.4, label="load×CF capacity (GW)", color="#4CAF50", alpha=0.7)
ax2_twin = ax2.twinx()
ax2_twin.plot(x, prov_sorted.cf_era5_mean, "ko-", markersize=4, label="ERA5 CF")
ax2_twin.set_ylabel("ERA5 CF", fontsize=10)
ax2.set_xticks(list(x))
ax2.set_xticklabels(prov_sorted.province, rotation=90, fontsize=7)
ax2.set_ylabel("Allocated capacity (GW)", fontsize=10)
ax2.set_title(f"Provincial capacity allocation: 1/N vs load×CF — CN{year}",
              fontsize=11, fontweight="bold")
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")
ax2.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(output_dir, f"solar_cf_counterfactual_{year}.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFigure saved: {fig_path}")
print(f"CSV saved:    {csv_path}")
print(f"\nDone. Outputs in: {output_dir}")
