"""
hydro_spillage_analysis.py
Hydropower spillage analysis with figures for thesis.
Usage:
  python analysis/network/hydro_spillage_analysis.py \
    results/CN2020_03_hydro_solar1N/networks/elec_s_250_ec_lcopt_3h.nc \
    --year 2020 --ember_twh 1321.7 --irena_gw 338.7
"""
import argparse, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
from province_utils import overlay_admin_boundaries
import pypsa

parser = argparse.ArgumentParser()
parser.add_argument("network_file")
parser.add_argument("--year", default="")
parser.add_argument("--ember_twh", type=float, required=True)
parser.add_argument("--irena_gw",  type=float, required=True)
parser.add_argument("--shapes", default="resources/shapes/gadm_shapes.geojson")
args = parser.parse_args()

COLORS = {"A_hydrobasins":"#d32f2f", "B_moderate":"#f57c00", "C_economic":"#388e3c"}
LABELS = {
    "A_hydrobasins": "A: HydroBASINS mismatch (ratio > 10)",
    "B_moderate":    "B: Moderate ratio (3–10)",
    "C_economic":    "C: Economic mechanism (ratio < 3)",
}
PROVINCE_NAMES = {
    "CN.1":"Anhui","CN.2":"Beijing","CN.3":"Chongqing","CN.4":"Fujian",
    "CN.5":"Gansu","CN.6":"Guangdong","CN.7":"Guangxi","CN.8":"Guizhou",
    "CN.9":"Hainan","CN.10":"Hebei","CN.11":"Heilongjiang","CN.12":"Henan",
    "CN.13":"Hubei","CN.14":"Hunan","CN.15":"Jiangsu","CN.16":"Jiangxi",
    "CN.17":"Jilin","CN.18":"Liaoning","CN.19":"Inner Mongolia","CN.20":"Ningxia",
    "CN.21":"Qinghai","CN.22":"Shaanxi","CN.23":"Shandong","CN.24":"Shanghai",
    "CN.25":"Shanxi","CN.26":"Sichuan","CN.27":"Tianjin","CN.28":"Xinjiang",
    "CN.29":"Tibet","CN.30":"Yunnan","CN.31":"Zhejiang",
}

run_name  = os.path.basename(os.path.dirname(os.path.dirname(args.network_file)))
out_dir   = os.path.join(os.environ.get("PYPSA_OUTPUT_DIR", os.path.join("analysis","network","output", run_name)), "hydro_spillage_analysis")
os.makedirs(out_dir, exist_ok=True)

def save(fig, name):
    p = os.path.join(out_dir, name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {p}")

# ── LOAD ─────────────────────────────────────────────────────────────────────
print(f"Loading {args.network_file}")
n       = pypsa.Network(args.network_file)
w       = n.snapshot_weightings.generators
hydro   = n.storage_units[n.storage_units.carrier == "hydro"]
inf_t   = n.storage_units_t.inflow.reindex(columns=hydro.index).fillna(0)
dis_t   = n.storage_units_t.p.reindex(columns=hydro.index).fillna(0)

# all in TWh, indexed by StorageUnit name
inflow    = (inf_t.multiply(w, axis=0)).sum() / 1e6
dispatch  = (dis_t.multiply(w, axis=0)).sum() / 1e6
spillage  = inflow - dispatch
cap_gw    = hydro.p_nom / 1e3                        # GW
max_gen   = cap_gw * 8.76                             # TWh = GW * 8760 h
ratio     = inflow / max_gen.replace(0, np.nan)

province  = hydro.bus.map(lambda b: ".".join(b.split(".")[:2]))

def cat(r):
    if pd.isna(r) or r > 10: return "A_hydrobasins"
    if r > 3:                 return "B_moderate"
    return "C_economic"

category = ratio.apply(cat)

# ── BUILD DATAFRAME ───────────────────────────────────────────────────────────
df = pd.DataFrame({
    "bus":          hydro.bus,
    "province":     province,
    "cap_gw":       cap_gw,
    "max_gen_twh":  max_gen,
    "inflow_twh":   inflow,
    "dispatch_twh": dispatch,
    "spillage_twh": spillage,
    "ratio":        ratio,
    "category":     category,
})

# sanity check
print("\nSanity check top 5:")
print(df.nlargest(5,"spillage_twh")[["bus","cap_gw","inflow_twh","spillage_twh","ratio","category"]].to_string())

# category summary
cat_sum = (df.groupby("category", sort=False)
             .agg(n_buses=("spillage_twh","count"),
                  spillage_twh=("spillage_twh","sum"),
                  inflow_twh=("inflow_twh","sum"))
             .reindex(["A_hydrobasins","B_moderate","C_economic"]))
cat_sum["spill_rate_pct"] = cat_sum.spillage_twh / cat_sum.inflow_twh * 100
cat_sum["pct_of_total"]   = cat_sum.spillage_twh / spillage.sum() * 100

print("\nCategory summary:")
print(cat_sum.to_string())

# ── CATEGORY C: COMPONENT TYPE + STATE OF CHARGE AT SPILL ────────────────────
print("\nCategory C component + SoC-at-spill check:")
print("Component type: all hydro spillage units are PyPSA StorageUnit (no Store+Link pairs).")

cat_c = df[df.category == "C_economic"].index
has_spill = hasattr(n.storage_units_t, "spill") and not n.storage_units_t.spill.empty
has_soc   = hasattr(n.storage_units_t, "state_of_charge") and not n.storage_units_t.state_of_charge.empty

if has_spill and has_soc and len(cat_c) > 0:
    spill_t    = n.storage_units_t.spill.reindex(columns=cat_c).fillna(0)
    soc_t      = n.storage_units_t.state_of_charge.reindex(columns=cat_c).fillna(0)
    energy_cap = (hydro.loc[cat_c, "p_nom"] * hydro.loc[cat_c, "max_hours"]).replace(0, np.nan)
    soc_pct    = soc_t.divide(energy_cap, axis=1) * 100

    spill_mask  = spill_t > 1e-3
    n_spill_hrs = int(spill_mask.values.sum())
    if n_spill_hrs > 0:
        soc_at_spill = soc_pct[spill_mask].stack()
        mean_soc     = soc_at_spill.mean()
        pct_full     = (soc_at_spill >= 99).mean() * 100
        print(f"Category C units: {len(cat_c)} | spill hours (unit-hours): {n_spill_hrs}")
        print(f"Mean SoC during spill: {mean_soc:.1f}% of energy capacity")
        print(f"Share of spill unit-hours at >=99% SoC (physical overflow): {pct_full:.1f}%")
    else:
        print("No spill hours found via n.storage_units_t.spill for Category C units (check threshold/units).")
else:
    print("WARNING: n.storage_units_t.spill and/or .state_of_charge not available on this network export.")
    print("Cannot test the SoC-at-spill hypothesis directly. Component-type answer above still holds.")

total_spill   = spillage.sum()
total_inflow  = inflow.sum()
total_dis     = dispatch.sum()

# ── FIGURE 1: WATERFALL ───────────────────────────────────────────────────────
spA = cat_sum.loc["A_hydrobasins","spillage_twh"]
spB = cat_sum.loc["B_moderate",   "spillage_twh"]
spC = cat_sum.loc["C_economic",   "spillage_twh"]

fig, ax = plt.subplots(figsize=(11,6))
bars_data = [
    ("ERA5\nInflow",      total_inflow, 0,                         "#1565c0"),
    ("Spill A\nHydroB.",  spA,          total_inflow - spA,        COLORS["A_hydrobasins"]),
    ("Spill B\nModerate", spB,          total_inflow - spA - spB,  COLORS["B_moderate"]),
    ("Spill C\nEcon.",    spC,          total_inflow-spA-spB-spC,  COLORS["C_economic"]),
    ("Dispatched\n(model)",total_dis,   0,                         "#1565c0"),
    ("Ember\nreference",  args.ember_twh, 0,                       "#43a047"),
]
for i,(lbl,val,bot,col) in enumerate(bars_data):
    if i in (0,4,5):
        ax.bar(i, val, color=col, edgecolor="black", linewidth=0.8)
        ax.text(i, val+8, f"{val:.0f}", ha="center", fontsize=10, fontweight="bold")
    else:
        ax.bar(i, val, bottom=bot, color=col, edgecolor="black", linewidth=0.8, alpha=0.9)
        ax.text(i, bot+val/2, f"{val:.0f}", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
ax.axhline(args.ember_twh, color="#43a047", ls="--", alpha=0.4)
ax.set_xticks(range(len(bars_data)))
ax.set_xticklabels([b[0] for b in bars_data], fontsize=10)
ax.set_ylabel("TWh", fontsize=11)
ax.set_title(f"Hydropower: inflow → spillage decomposition → dispatch  —  CN{args.year}",
             fontsize=12, fontweight="bold")
ax.set_ylim(0, total_inflow*1.15)
ax.grid(True, axis="y", alpha=0.3)
save(fig, f"hydro_waterfall_{args.year}.png")

# ── FIGURE 2: CATEGORY BAR CHARTS ────────────────────────────────────────────
cats   = ["A_hydrobasins","B_moderate","C_economic"]
xlbls  = ["A\nHydroBASINS\n(ratio>10)","B\nModerate\n(3–10)","C\nEconomic\n(<3)"]
colors = [COLORS[c] for c in cats]

fig, axes = plt.subplots(1,2, figsize=(12,5))

ax = axes[0]
vals = [cat_sum.loc[c,"spillage_twh"] for c in cats]
bars = ax.bar(xlbls, vals, color=colors, edgecolor="black", linewidth=0.8)
for bar,val in zip(bars,vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
            f"{val:.1f} TWh", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("Spillage (TWh)", fontsize=11)
ax.set_title("Spillage by mechanism (TWh)", fontsize=11, fontweight="bold")
ax.grid(True, axis="y", alpha=0.3)

ax = axes[1]
vals = [cat_sum.loc[c,"spill_rate_pct"] for c in cats]
bars = ax.bar(xlbls, vals, color=colors, edgecolor="black", linewidth=0.8)
for bar,val in zip(bars,vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("Spill rate (% of inflow)", fontsize=11)
ax.set_title("Spill rate by mechanism", fontsize=11, fontweight="bold")
ax.set_ylim(0, 110)
ax.grid(True, axis="y", alpha=0.3)

plt.suptitle(f"Hydropower spillage by mechanism  —  CN{args.year}",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, f"hydro_spillage_categories_{args.year}.png")

# ── FIGURE 3: TOP 15 UNITS ────────────────────────────────────────────────────
top15 = df.nlargest(15,"spillage_twh")
fig, ax = plt.subplots(figsize=(12,7))
y = np.arange(len(top15))
bars = ax.barh(y, top15.spillage_twh,
               color=[COLORS[c] for c in top15.category],
               edgecolor="black", linewidth=0.6)
pnames = top15.province.map(PROVINCE_NAMES).fillna(top15.province)
ax.set_yticks(y)
ax.set_yticklabels([f"{pn} ({idx.split()[0]})"
                    for pn,idx in zip(pnames, top15.index)], fontsize=9)
for bar,val,rat,cat_ in zip(bars, top15.spillage_twh, top15.ratio, top15.category):
    ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
            f"{val:.1f} TWh | ratio {rat:.1f} | {cat_[0]}",
            va="center", fontsize=8)
ax.set_xlabel("Spillage (TWh)", fontsize=11)
ax.set_title(f"Top 15 hydro units by spillage  —  CN{args.year}",
             fontsize=12, fontweight="bold")
patches = [mpatches.Patch(color=COLORS[k], label=LABELS[k]) for k in cats]
ax.legend(handles=patches, fontsize=8, loc="lower right")
ax.set_xlim(0, top15.spillage_twh.max()*1.45)
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
save(fig, f"hydro_top15_units_{args.year}.png")

# ── FIGURE 4: SCATTER inflow vs spillage ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10,7))
for cat_,col in COLORS.items():
    sub = df[df.category==cat_]
    ax.scatter(sub.inflow_twh, sub.spillage_twh, c=col,
               label=LABELS[cat_], alpha=0.7, s=40, edgecolors="none")
for _,row in df.nlargest(8,"spillage_twh").iterrows():
    pn = PROVINCE_NAMES.get(row.province, row.province)
    ax.annotate(f"{pn}\n({row.spillage_twh:.0f} TWh)",
                xy=(row.inflow_twh, row.spillage_twh),
                xytext=(6,0), textcoords="offset points", fontsize=7)
lim = df.inflow_twh.max()*1.05
ax.plot([0,lim],[0,lim],"k--",alpha=0.3, label="100% spill")
ax.set_xlabel("Annual inflow (TWh)", fontsize=11)
ax.set_ylabel("Annual spillage (TWh)", fontsize=11)
ax.set_title(f"Hydro inflow vs spillage per unit  —  CN{args.year}",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
save(fig, f"hydro_scatter_{args.year}.png")

# ── FIGURE 5: MAP ─────────────────────────────────────────────────────────────
if os.path.exists(args.shapes):
    prov_df = (df.groupby("province")
                 .agg(spillage_twh=("spillage_twh","sum"),
                      inflow_twh=("inflow_twh","sum"))
                 .reset_index())
    prov_df["spill_rate"] = prov_df.spillage_twh / prov_df.inflow_twh.replace(0,np.nan)*100

    gdf   = gpd.read_file(args.shapes)
    china = gdf[gdf.country=="CN"].copy()
    china["admin1"] = china.GADM_ID.apply(lambda b: ".".join(str(b).split(".")[:2]))
    adm1  = china.dissolve(by="admin1").reset_index()[["admin1","geometry"]]
    merged = adm1.merge(prov_df, left_on="admin1", right_on="province", how="left")

    fig, axes = plt.subplots(1,2, figsize=(18,7))
    for ax,col,lbl,cmap in [
        (axes[0],"spillage_twh","Spillage (TWh)","Reds"),
        (axes[1],"spill_rate",  "Spill rate (%)","Oranges"),
    ]:
        merged.plot(column=col, ax=ax, cmap=cmap, legend=True,
                    edgecolor="none", linewidth=0.0,
                    missing_kwds={"color":"lightgrey"},
                    legend_kwds={"label":lbl,"shrink":0.6})
        overlay_admin_boundaries(ax, linewidth=0.6, edgecolor="black")
        ax.set_title(f"Hydropower {lbl} by province — CN{args.year}",
                     fontsize=11, fontweight="bold")
        ax.set_xlim(73,136); ax.set_ylim(17,54); ax.axis("off")
    plt.suptitle(f"Hydropower spillage spatial distribution — CN{args.year}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, f"hydro_map_{args.year}.png")

print(f"\nDone. All outputs in: {out_dir}")

# ── EXPORT CSV ────────────────────────────────────────────────────────────────
csv_path = os.path.join(out_dir, f"hydro_spillage_{args.year}.csv")
df.to_csv(csv_path)
print(f"Saved: {csv_path}")
