"""
nuclear_analysis.py
===================
Nuclear capacity and generation analysis for thesis.

Compares model nuclear capacity against IAEA reactor database,
identifies excess capacity by province, and produces figures.

Produces:
  1. Bar chart: model vs IAEA operative vs IAEA all by province
  2. Bar chart: generation model vs Ember by province  
  3. Scatter: capacity vs generation by province
  4. Timeline: cumulative IAEA operative capacity 2010-2026

Usage:
  python analysis/network/nuclear_analysis.py \
    results/CN2020_03_hydro_solar1N/networks/elec_s_250_ec_lcopt_3h.nc \
    --year 2020 --ember_twh 366.2 --irena_gw 49.9 --ref_date 2020-12-31

  python analysis/network/nuclear_analysis.py \
    results/CN2024_01_hydro_solar1N/networks/elec_s_250_ec_lcopt_3h.nc \
    --year 2024 --ember_twh 450.9 --irena_gw 60.8 --ref_date 2024-12-31
"""

import argparse, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pypsa

parser = argparse.ArgumentParser()
parser.add_argument("network_file")
parser.add_argument("--year",      default="")
parser.add_argument("--ember_twh", type=float, required=True)
parser.add_argument("--irena_gw",  type=float, required=True)
parser.add_argument("--ref_date",  default="2020-12-31",
                    help="Reference date for IAEA operative filter (YYYY-MM-DD)")
parser.add_argument("--iaea_csv",  
                    default="data/iaea_china_reactors_2026.csv")
args = parser.parse_args()

PROVINCE_NAMES = {
    "CN.4":"Fujian","CN.6":"Guangdong","CN.7":"Guangxi",
    "CN.9":"Hainan","CN.15":"Jiangsu","CN.18":"Liaoning",
    "CN.23":"Shandong","CN.31":"Zhejiang",
    "CN.2":"Beijing*","CN.5":"Gansu*",
}
PROVINCE_TO_CODE = {v.rstrip("*"):k for k,v in PROVINCE_NAMES.items()}

run_name = os.path.basename(os.path.dirname(os.path.dirname(args.network_file)))
out_dir  = os.path.join(os.environ.get("PYPSA_OUTPUT_DIR", os.path.join("analysis","network","output", run_name)), "nuclear")
os.makedirs(out_dir, exist_ok=True)

def save(fig, name):
    p = os.path.join(out_dir, name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {p}")

# ── LOAD NETWORK ─────────────────────────────────────────────────────────────
print(f"Loading {args.network_file}")
n = pypsa.Network(args.network_file)
w = n.snapshot_weightings.generators
nuclear_g = n.generators[n.generators.carrier=="nuclear"].copy()
nuclear_g["province"] = nuclear_g.bus.map(lambda b: ".".join(b.split(".")[:2]))
nuclear_g["gen_TWh"]  = (n.generators_t.p.reindex(columns=nuclear_g.index)
                          .fillna(0).multiply(w,axis=0)).sum() / 1e6

model_prov = nuclear_g.groupby("province").agg(
    cap_GW=("p_nom", lambda x: x.sum()/1e3),
    gen_TWh=("gen_TWh","sum"),
    n_units=("p_nom","count")
).reset_index()
model_prov["province_name"] = model_prov.province.map(PROVINCE_NAMES).fillna(model_prov.province)
model_prov["cf"] = model_prov.gen_TWh / (model_prov.cap_GW * 8.76)

model_total_cap = model_prov.cap_GW.sum()
model_total_gen = model_prov.gen_TWh.sum()
cf_model = model_total_gen / (model_total_cap * 8.76)

# ── LOAD IAEA ─────────────────────────────────────────────────────────────────
iaea = pd.read_csv(args.iaea_csv)
iaea["grid_connection"] = pd.to_datetime(iaea["grid_connection"], errors="coerce")
ref_date = pd.Timestamp(args.ref_date)

# operative at ref_date: status=Operational AND grid_connection <= ref_date
iaea_op = iaea[
    (iaea.status == "Operational") &
    (iaea.grid_connection <= ref_date)
].copy()

# NOTE: units with missing grid_connection date and status=Under Construction
# cannot be classified as operative/under-construction at ref_date with certainty
iaea_uc = iaea[iaea.status == "Under Construction"].copy()
iaea_op_after = iaea[
    (iaea.status == "Operational") &
    (iaea.grid_connection > ref_date)
].copy()

print(f"\nIAEA operative at {args.ref_date}: {len(iaea_op)} units, "
      f"{iaea_op.net_mw.sum()/1e3:.1f} GW (net)")
print(f"IAEA grid connection after {args.ref_date}: {len(iaea_op_after)} units")
print(f"IAEA under construction (no date): {len(iaea_uc)} units")

# aggregate IAEA operative by province
iaea_prov = iaea_op.groupby("province").net_mw.sum().reset_index()
iaea_prov.columns = ["province_name","iaea_op_GW"]
iaea_prov["iaea_op_GW"] /= 1e3

# units that connected AFTER ref_date (potential excess in model)
iaea_after_prov = iaea_op_after.groupby("province").net_mw.sum().reset_index()
iaea_after_prov.columns = ["province_name","iaea_after_GW"]
iaea_after_prov["iaea_after_GW"] /= 1e3

# ── MERGE ─────────────────────────────────────────────────────────────────────
model_prov2 = model_prov.copy()
model_prov2["province_name_clean"] = model_prov2.province_name.str.rstrip("*")

merged = model_prov2.merge(iaea_prov, 
                            left_on="province_name_clean", 
                            right_on="province_name", how="outer")
merged = merged.merge(iaea_after_prov,
                      left_on="province_name_clean",
                      right_on="province_name", how="left")
merged["cap_GW"]      = merged["cap_GW"].fillna(0)
merged["gen_TWh"]     = merged["gen_TWh"].fillna(0)
merged["iaea_op_GW"]  = merged["iaea_op_GW"].fillna(0)
merged["iaea_after_GW"] = merged["iaea_after_GW"].fillna(0)
merged["excess_GW"]   = merged["cap_GW"] - merged["iaea_op_GW"]
merged["pname"]       = merged["province_name_clean"].fillna(merged["province_name_x"])
merged = merged.sort_values("cap_GW", ascending=False)

# print summary
print("\nProvince comparison (GW):")
print(f"{'Province':<15} {'IAEA_op':>8} {'After_ref':>10} {'Model':>8} {'Excess':>8}")
for _,row in merged.iterrows():
    if row.cap_GW > 0 or row.iaea_op_GW > 0:
        print(f"{str(row.pname):<15} {row.iaea_op_GW:>8.2f} "
              f"{row.iaea_after_GW:>10.2f} {row.cap_GW:>8.2f} {row.excess_GW:>+8.2f}")

iaea_total_op = iaea_op.net_mw.sum()/1e3
cf_observed   = args.ember_twh / (iaea_total_op * 8.76)
cf_irena      = args.ember_twh / (args.irena_gw * 8.76)

print(f"\nNational summary:")
print(f"  Model capacity:          {model_total_cap:.1f} GW")
print(f"  IAEA operative:          {iaea_total_op:.1f} GW")
print(f"  IRENA:                   {args.irena_gw:.1f} GW")
print(f"  Model generation:        {model_total_gen:.1f} TWh")
print(f"  Ember reference:         {args.ember_twh:.1f} TWh")
print(f"  CF model:                {cf_model:.3f}")
print(f"  CF observed (IAEA cap):  {cf_observed:.3f}")
print(f"  CF observed (IRENA cap): {cf_irena:.3f}")

# ── FIGURE 1: CAPACITY BY PROVINCE ───────────────────────────────────────────
plot_df = merged[(merged.cap_GW > 0) | (merged.iaea_op_GW > 0)].copy()
plot_df = plot_df.sort_values("iaea_op_GW", ascending=False)
x = np.arange(len(plot_df))
w_bar = 0.35

fig, ax = plt.subplots(figsize=(13, 6))
bars1 = ax.bar(x - w_bar/2, plot_df.iaea_op_GW, w_bar,
               label=f"IAEA operative at {args.ref_date}",
               color="#1565c0", edgecolor="black", linewidth=0.7)
bars2 = ax.bar(x + w_bar/2, plot_df.cap_GW, w_bar,
               label="Model (powerplantmatching)",
               color="#ef5350", edgecolor="black", linewidth=0.7)

# annotate excess
for i, (_, row) in enumerate(plot_df.iterrows()):
    if row.excess_GW > 0.1:
        ax.annotate(f"+{row.excess_GW:.1f} GW",
                    xy=(i + w_bar/2, row.cap_GW),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8, color="#c62828", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(plot_df.pname, rotation=30, ha="right", fontsize=10)
ax.set_ylabel("Installed capacity (GW)", fontsize=11)
ax.set_title(f"Nuclear capacity: model vs IAEA operative — CN{args.year}\n"
             f"Model total: {model_total_cap:.1f} GW  |  "
             f"IAEA operative: {iaea_total_op:.1f} GW  |  "
             f"IRENA: {args.irena_gw:.1f} GW",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, axis="y", alpha=0.3)
ax.set_ylim(0, plot_df[["iaea_op_GW","cap_GW"]].max().max() * 1.25)
plt.tight_layout()
save(fig, f"nuclear_capacity_province_{args.year}.png")

# ── FIGURE 2: DECOMPOSITION BAR CHART ────────────────────────────────────────
# Counterfactual generation under different assumptions
gen_iaea_obs_cf  = iaea_total_op * cf_observed * 8.76   # IAEA cap + obs CF
gen_model_obs_cf = model_total_cap * cf_observed * 8.76  # model cap + obs CF
gen_iaea_model_cf = iaea_total_op * cf_model * 8.76      # IAEA cap + model CF
gen_model_model_cf = model_total_cap * cf_model * 8.76   # model cap + model CF

scenarios = [
    ("Ember\nreference",         args.ember_twh,      "#43a047"),
    ("IAEA cap\nobs. CF",        gen_iaea_obs_cf,     "#1565c0"),
    ("Model cap\nobs. CF",       gen_model_obs_cf,    "#f57c00"),
    ("IAEA cap\nCF=1.0",         gen_iaea_model_cf,   "#7b1fa2"),
    ("Model cap\nCF=1.0\n(actual model)", gen_model_model_cf, "#ef5350"),
]

fig, ax = plt.subplots(figsize=(11, 6))
colors = [s[2] for s in scenarios]
vals   = [s[1] for s in scenarios]
lbls   = [s[0] for s in scenarios]
bars   = ax.bar(lbls, vals, color=colors, edgecolor="black", linewidth=0.8, width=0.6)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
            f"{val:.0f} TWh", ha="center", fontsize=10, fontweight="bold")
ax.axhline(args.ember_twh, color="#43a047", ls="--", alpha=0.5)
ax.set_ylabel("Nuclear generation (TWh)", fontsize=11)
ax.set_title(f"Nuclear generation decomposition — CN{args.year}\n"
             f"Observed CF (IAEA cap): {cf_observed:.3f}  |  Model CF: {cf_model:.3f}",
             fontsize=11, fontweight="bold")
ax.set_ylim(0, max(vals)*1.2)
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
save(fig, f"nuclear_generation_decomposition_{args.year}.png")

# ── FIGURE 3: IAEA TIMELINE ───────────────────────────────────────────────────
iaea_timeline = iaea[iaea.status=="Operational"].copy()
iaea_timeline = iaea_timeline.dropna(subset=["grid_connection"])
iaea_timeline = iaea_timeline.sort_values("grid_connection")
iaea_timeline["cum_gw"] = iaea_timeline.net_mw.cumsum() / 1e3
iaea_timeline["year"]   = iaea_timeline.grid_connection.dt.year

fig, ax = plt.subplots(figsize=(12, 5))
ax.step(iaea_timeline.grid_connection, iaea_timeline.cum_gw,
        where="post", color="#1565c0", linewidth=2, label="IAEA operative (cumulative)")
ax.axvline(pd.Timestamp(args.ref_date), color="red", ls="--", linewidth=1.5,
           label=f"Reference date: {args.ref_date}")
ax.axhline(args.irena_gw, color="#f57c00", ls=":", linewidth=1.5,
           label=f"IRENA {args.year}: {args.irena_gw:.1f} GW")
ax.axhline(model_total_cap, color="#ef5350", ls="-.", linewidth=1.5,
           label=f"Model: {model_total_cap:.1f} GW")

# annotate ref date value
ref_val = iaea_timeline[iaea_timeline.grid_connection <= ref_date].cum_gw.max()
ax.annotate(f"IAEA operative\nat {args.ref_date}:\n{ref_val:.1f} GW",
            xy=(pd.Timestamp(args.ref_date), ref_val),
            xytext=(30, -30), textcoords="offset points",
            fontsize=9, color="red",
            arrowprops=dict(arrowstyle="->", color="red"))

ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("Cumulative capacity (GW)", fontsize=11)
ax.set_title(f"China nuclear capacity timeline (IAEA) — CN{args.year}",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(pd.Timestamp("2010-01-01"), pd.Timestamp("2027-01-01"))
plt.tight_layout()
save(fig, f"nuclear_timeline_{args.year}.png")

print(f"\nDone. All outputs in: {out_dir}")
