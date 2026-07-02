"""
nuclear_spatial_analysis.py
============================
Analisi spaziale capacità e generazione nucleare aggregata a livello Admin1.

Carrier: nuclear (Generator).
Per dispatch-only runs (CN2020, CN2024): p_nom_opt = p_nom (capacità fissa).
Per capacity expansion (CN2060): p_nom_opt è la capacità ottimizzata dal solver.

Per ogni provincia estrae:
  - Capacità installata (GW, somma p_nom_opt)
  - Generazione (TWh)
  - Capacity factor (CF)
  - Carico locale (TWh)
  - Rapporto generazione/carico

Output:
  nuclear_per_admin1.csv
  admin1_nuclear_capacity_gw.png
  admin1_nuclear_generation_twh.png
  admin1_nuclear_cf.png
  admin1_nuclear_load_ratio.png

Uso:
  PYPSA_OUTPUT_DIR="/path/to/CN2020_NUCAP" python nuclear_spatial_analysis.py results/.../elec_s_425_ec_lcopt_3h.nc
"""

import argparse
import os
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
from province_utils import overlay_admin_boundaries
import pandas as pd
import pypsa
import seaborn as sns

warnings.filterwarnings("ignore")

GADM_FILE  = "resources/shapes/gadm_shapes.geojson"
OUTPUT_DIR = os.path.join(os.environ.get("PYPSA_OUTPUT_DIR", "analysis/network/output"), "nuclear_spatial_analysis")

sns.set_style("white")
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "font.size": 11,
})


def save(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def bus_to_admin1(name):
    parts = str(name).split(".")
    return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else name


def extract_year_label(network_path):
    match = re.search(r'CN(\d{4})', network_path)
    return f"CN{match.group(1)}" if match else os.path.basename(os.path.dirname(os.path.dirname(network_path)))


def choropleth(gdf, col, title, cbar_label, cmap, filename):
    fig, ax = plt.subplots(figsize=(14, 10))
    gdf.plot(
        column=col,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=gdf[col].max(),
        edgecolor="none",
        linewidth=0.0,
        legend=True,
        legend_kwds={
            "label": cbar_label,
            "orientation": "vertical",
            "shrink": 0.6,
            "pad": 0.02,
        },
        missing_kwds={"color": "lightgray", "label": "No data"},
    )
    overlay_admin_boundaries(ax, linewidth=0.6, edgecolor="black")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(73, 136)
    ax.set_ylim(17, 54)
    ax.grid(True, alpha=0.3, linewidth=0.4)
    save(fig, filename)


def main():
    parser = argparse.ArgumentParser(description="Nuclear spatial analysis — Admin1, PyPSA-Earth")
    parser.add_argument("network", help="Path to .nc network file")
    args = parser.parse_args()

    if not os.path.exists(args.network):
        print(f"[ERROR] File not found: {args.network}")
        raise SystemExit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    year_label = extract_year_label(args.network)
    print(f"Loading: {args.network} (label: {year_label})")

    n = pypsa.Network(args.network)
    n.buses["province"] = n.buses.index.map(bus_to_admin1)

    dt = (n.snapshots[1] - n.snapshots[0]).total_seconds() / 3600.0 if len(n.snapshots) > 1 else 1.0
    total_hours = len(n.snapshots) * dt

    # Nuclear generators
    nuc = n.generators[n.generators.carrier == "nuclear"].copy()
    print(f"  Nuclear generators: {len(nuc)}")

    nuc_p = n.generators_t.p.reindex(columns=nuc.index, fill_value=0.0)
    nuc_energy = (nuc_p * dt).sum() / 1e6      # TWh
    nuc_gw = nuc["p_nom_opt"] / 1e3            # GW

    nuc_province = nuc["bus"].map(n.buses["province"])
    nuc_province.name = "admin1"

    nuc_admin1 = pd.DataFrame({
        "nuclear_gw":  nuc_gw,
        "nuclear_twh": nuc_energy,
    }).groupby(nuc_province)[["nuclear_gw", "nuclear_twh"]].sum()

    # Load per province
    load_ts = n.loads_t.p if not n.loads_t.p.empty else n.loads_t.p_set
    load_energy = (load_ts * dt).sum() / 1e6
    load_province = n.loads["bus"].map(n.buses["province"])
    load_province.name = "admin1"
    load_admin1 = (load_energy.groupby(load_province).sum()).rename("load_twh")

    admin1_df = nuc_admin1.join(load_admin1, how="outer").reset_index()
    admin1_df["nuclear_cf"] = (
        admin1_df["nuclear_twh"] / (admin1_df["nuclear_gw"] * total_hours / 1e3)
    )
    admin1_df["nuclear_load_ratio"] = admin1_df["nuclear_twh"] / admin1_df["load_twh"]

    total_gw  = admin1_df["nuclear_gw"].sum()
    total_twh = admin1_df["nuclear_twh"].sum()
    total_load = admin1_df["load_twh"].sum()

    print(f"\nNational totals:")
    print(f"  Nuclear installed:  {total_gw:.1f} GW")
    print(f"  Nuclear generation: {total_twh:.1f} TWh")
    print(f"  Nuclear share:      {total_twh / total_load * 100:.1f}%")

    csv_path = os.path.join(OUTPUT_DIR, "nuclear_per_admin1.csv")
    admin1_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Saved: {csv_path}")

    # Admin1 geometries
    gadm = gpd.read_file(GADM_FILE)
    china = gadm[gadm["country"] == "CN"].copy()
    name_col = next((c for c in ["GADM_ID", "name", "id"] if c in china.columns), None)
    if name_col is None:
        china["_rid"] = china.index.astype(str)
        name_col = "_rid"
    china["admin1"] = china[name_col].apply(bus_to_admin1)
    admin1_geo = china.dissolve(by="admin1").reset_index()[["admin1", "geometry"]]
    gdf = admin1_geo.merge(admin1_df, on="admin1", how="left")

    choropleth(gdf,
        col="nuclear_gw",
        title=f"Nuclear installed capacity per province — {year_label}  |  Total: {total_gw:.1f} GW",
        cbar_label="Nuclear capacity (GW)",
        cmap="YlOrRd",
        filename="admin1_nuclear_capacity_gw.png",
    )
    choropleth(gdf,
        col="nuclear_twh",
        title=f"Nuclear generation per province — {year_label}  |  Total: {total_twh:.1f} TWh",
        cbar_label="Nuclear generation (TWh)",
        cmap="YlOrRd",
        filename="admin1_nuclear_generation_twh.png",
    )
    choropleth(gdf,
        col="nuclear_cf",
        title=f"Nuclear capacity factor per province — {year_label}",
        cbar_label="CF (dimensionless)",
        cmap="plasma",
        filename="admin1_nuclear_cf.png",
    )
    choropleth(gdf,
        col="nuclear_load_ratio",
        title=f"Nuclear generation / local load per province — {year_label}",
        cbar_label="Nuclear / Load ratio",
        cmap="RdYlGn_r",
        filename="admin1_nuclear_load_ratio.png",
    )

    print(f"\nDone. All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()