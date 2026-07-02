"""
thermal_maps.py
===============
Provincial maps of installed capacity (GW) and generation (TWh)
for coal+lignite and gas (CCGT) carriers.

Usage:
  python analysis/network/thermal_maps.py \
    results/CN2020_03_hydro_solar1N/networks/elec_s_250_ec_lcopt_3h.nc \
    --year 2020

  python analysis/network/thermal_maps.py \
    results/CN2024_01_hydro_solar1N/networks/elec_s_250_ec_lcopt_3h.nc \
    --year 2024
"""
import argparse, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
from province_utils import overlay_admin_boundaries
import pypsa

parser = argparse.ArgumentParser()
parser.add_argument("network_file")
parser.add_argument("--year", default="")
parser.add_argument("--shapes", default="resources/shapes/gadm_shapes.geojson")
args = parser.parse_args()

run_name = os.path.basename(os.path.dirname(os.path.dirname(args.network_file)))
out_dir  = os.path.join(os.environ.get("PYPSA_OUTPUT_DIR", os.path.join("analysis","network","output", run_name)), "thermal_maps")
os.makedirs(out_dir, exist_ok=True)

def save(fig, name):
    p = os.path.join(out_dir, name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {p}")

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

# ── LOAD NETWORK ─────────────────────────────────────────────────────────────
print(f"Loading {args.network_file}")
n = pypsa.Network(args.network_file)
w = n.snapshot_weightings.generators

def get_provincial_data(carriers):
    gens = n.generators[n.generators.carrier.isin(carriers)].copy()
    gens["province"] = gens.bus.map(lambda b: ".".join(b.split(".")[:2]))
    gens["gen_TWh"]  = (n.generators_t.p.reindex(columns=gens.index)
                         .fillna(0).multiply(w, axis=0)).sum() / 1e6
    prov = gens.groupby("province").agg(
        cap_GW=("p_nom", lambda x: x.sum()/1e3),
        gen_TWh=("gen_TWh","sum")
    ).reset_index()
    return prov

coal_prov = get_provincial_data(["coal","lignite"])
gas_prov  = get_provincial_data(["CCGT"])

print(f"\nCoal+lignite: {coal_prov.cap_GW.sum():.1f} GW, {coal_prov.gen_TWh.sum():.1f} TWh")
print(f"Gas (CCGT):   {gas_prov.cap_GW.sum():.1f} GW, {gas_prov.gen_TWh.sum():.3f} TWh")

# ── LOAD SHAPES ───────────────────────────────────────────────────────────────
if not os.path.exists(args.shapes):
    print(f"ERROR: shapes not found at {args.shapes}")
    exit(1)

gdf   = gpd.read_file(args.shapes)
china = gdf[gdf.country=="CN"].copy()
china["admin1"] = china.GADM_ID.apply(lambda b: ".".join(str(b).split(".")[:2]))
adm1  = china.dissolve(by="admin1").reset_index()[["admin1","geometry"]]

def make_map(prov_df, carrier_label, cmap_cap, cmap_gen, filename,
             cap_label="Installed capacity (GW)",
             gen_label="Generation (TWh)"):

    merged = adm1.merge(prov_df, left_on="admin1", right_on="province", how="left")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Panel left: capacity
    ax = axes[0]
    merged.plot(column="cap_GW", ax=ax, cmap=cmap_cap,
                legend=True, edgecolor="none", linewidth=0.0,
                missing_kwds={"color":"lightgrey"},
                legend_kwds={"label": cap_label, "shrink":0.6})
    overlay_admin_boundaries(ax, linewidth=0.6, edgecolor="black")
    ax.set_title(f"{carrier_label} — {cap_label}\nCN{args.year}",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(73,136); ax.set_ylim(17,54); ax.axis("off")

    # Panel right: generation
    ax = axes[1]
    merged.plot(column="gen_TWh", ax=ax, cmap=cmap_gen,
                legend=True, edgecolor="none", linewidth=0.0,
                missing_kwds={"color":"lightgrey"},
                legend_kwds={"label": gen_label, "shrink":0.6})
    overlay_admin_boundaries(ax, linewidth=0.6, edgecolor="black")
    ax.set_title(f"{carrier_label} — {gen_label}\nCN{args.year}",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(73,136); ax.set_ylim(17,54); ax.axis("off")

    total_cap = prov_df.cap_GW.sum()
    total_gen = prov_df.gen_TWh.sum()
    plt.suptitle(
        f"{carrier_label} — CN{args.year}  |  "
        f"Total capacity: {total_cap:.1f} GW  |  "
        f"Total generation: {total_gen:.1f} TWh",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    save(fig, filename)

# ── FIGURE 1: COAL+LIGNITE ────────────────────────────────────────────────────
make_map(coal_prov, "Coal + Lignite", "Reds", "YlOrRd",
         f"thermal_map_coal_{args.year}.png")

# ── FIGURE 2: GAS (CCGT) ─────────────────────────────────────────────────────
make_map(gas_prov, "Gas (CCGT)", "Blues", "PuBu",
         f"thermal_map_gas_{args.year}.png",
         gen_label="Generation (TWh) — near zero everywhere")

print(f"\nDone. All outputs in: {out_dir}")
