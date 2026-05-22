"""
solar_spatial_analysis.py
=========================
Analisi spaziale della generazione solare aggregata a livello Admin1 (provincia).

Per ogni provincia estrae:
  - Capacità solare installata (GW, somma p_nom_opt)
  - Generazione solare (TWh)
  - Availability factor (CF = generazione / capacità / ore totali)
  - Carico locale (TWh)
  - Rapporto solar/load

Mappe: choropleth con confini provinciali Admin1
       (dissolve del GADM Admin2 → Admin1, niente prefetture visibili).

Output:
  solar_per_admin1.csv
  admin1_solar_capacity_gw.png
  admin1_solar_generation_twh.png
  admin1_load_twh.png
  admin1_solar_load_ratio.png
  admin1_solar_cf.png

Uso:
  PYPSA_OUTPUT_DIR="analysis/network/solar_spatial" python analysis/network/solar_spatial_analysis.py results/networks/geospatial_v2_approccio1/CN2020_Admin2/elec_s_250_ec_lcopt_3h.nc
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
import seaborn as sns

warnings.filterwarnings("ignore")

# ── PARAMETERS ────────────────────────────────────────────────────────────────

GADM_FILE  = "resources/shapes/gadm_shapes.geojson"
OUTPUT_DIR = os.environ.get("PYPSA_OUTPUT_DIR", "analysis/network/solar_spatial")

# ── STYLE ─────────────────────────────────────────────────────────────────────

sns.set_style("white")
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "font.size": 11,
})

# ── HELPERS ───────────────────────────────────────────────────────────────────

def save(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def bus_to_admin1(name):
    """'CN.28.3_1_AC' → 'CN.28'"""
    parts = str(name).split(".")
    return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else name


def choropleth(gdf, col, title, cbar_label, cmap, filename):
    """
    Choropleth map con confini Admin1.
    Province senza dati in grigio chiaro.
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    gdf.plot(
        column=col,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=gdf[col].max(),
        edgecolor="black",
        linewidth=0.6,
        legend=True,
        legend_kwds={
            "label": cbar_label,
            "orientation": "vertical",
            "shrink": 0.6,
            "pad": 0.02,
        },
        missing_kwds={"color": "lightgray", "label": "No data"},
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(73, 136)
    ax.set_ylim(17, 54)
    ax.grid(True, alpha=0.3, linewidth=0.4)

    save(fig, filename)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Spatial analysis solar — Admin1 level, PyPSA-Earth"
    )
    parser.add_argument("network", help="Path to .nc network file")
    args = parser.parse_args()

    if not os.path.exists(args.network):
        print(f"[ERROR] File not found: {args.network}")
        raise SystemExit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading: {args.network}")
    n = pypsa.Network(args.network)
    print(f"  Buses:     {len(n.buses)}")
    print(f"  Snapshots: {len(n.snapshots)}")

    # ── Enrich network: province attribute on buses ─────────────────────────
    # Definito una volta sola; tutti i groupby usano n.buses["province"].
    # Path: generator → bus (topologia) → province (attributo bus).
    n.buses["province"] = n.buses.index.map(bus_to_admin1)

    # ── Time resolution ────────────────────────────────────────────────
    dt = (n.snapshots[1] - n.snapshots[0]).total_seconds() / 3600.0 if len(n.snapshots) > 1 else 1.0
    total_hours = len(n.snapshots) * dt
    print(f"  Resolution: {dt}h | Total hours: {total_hours:.0f}h")

    # ── Solar generators ──────────────────────────────────────────
    solar = n.generators[n.generators.carrier == "solar"].copy()
    print(f"  Solar generators: {len(solar)}")

    solar_p = n.generators_t.p.reindex(columns=solar.index, fill_value=0.0)
    solar_energy_per_gen = (solar_p * dt).sum() / 1e6   # TWh per generator
    p_nom_opt_gw = solar["p_nom_opt"] / 1e3              # GW per generator

    # ── Per-generator → per-province ─────────────────────────────────────────
    # generator → bus → province via n.buses["province"]
    solar_province = solar["bus"].map(n.buses["province"])
    solar_province.name = "admin1"  # preserva il nome colonna nel groupby index

    solar_admin1 = pd.DataFrame({
        "solar_gw":  p_nom_opt_gw,
        "solar_twh": solar_energy_per_gen,
    }).groupby(solar_province)[["solar_gw", "solar_twh"]].sum()

    # ── Load per province ───────────────────────────────────────────────
    load_ts = n.loads_t.p if not n.loads_t.p.empty else n.loads_t.p_set
    load_energy = (load_ts * dt).sum() / 1e6

    load_province = n.loads["bus"].map(n.buses["province"])
    load_province.name = "admin1"
    load_admin1 = pd.DataFrame({
        "load_twh": load_energy,
    }).groupby(load_province)["load_twh"].sum()

    admin1_df = solar_admin1.join(load_admin1, how="outer").reset_index()

    # CF e ratio calcolati al livello aggregato (non media di CF per bus)
    admin1_df["solar_cf"] = (
        admin1_df["solar_twh"] / (admin1_df["solar_gw"] * total_hours / 1e3)
    )
    admin1_df["solar_load_ratio"] = admin1_df["solar_twh"] / admin1_df["load_twh"]

    # ── National totals ───────────────────────────────────────────────────────
    total_solar_gw  = admin1_df["solar_gw"].sum()
    total_solar_twh = admin1_df["solar_twh"].sum()
    total_load_twh  = admin1_df["load_twh"].sum()

    print(f"\nNational totals:")
    print(f"  Solar installed:  {total_solar_gw:.1f} GW")
    print(f"  Solar generation: {total_solar_twh:.1f} TWh")
    print(f"  Total load:       {total_load_twh:.1f} TWh")
    print(f"  Solar share:      {total_solar_twh / total_load_twh * 100:.1f}%")

    print(f"\nPer-province table ({len(admin1_df)} provinces), sorted by solar generation:")
    print(
        admin1_df.sort_values("solar_twh", ascending=False)
        [["admin1", "solar_gw", "solar_twh", "solar_cf", "load_twh", "solar_load_ratio"]]
        .to_string(index=False)
    )

    csv_path = os.path.join(OUTPUT_DIR, "solar_per_admin1.csv")
    admin1_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nSaved: {csv_path}")

    # ── Build Admin1 geometries (dissolve GADM Admin2 → Admin1) ──────────────
    if not os.path.exists(GADM_FILE):
        raise FileNotFoundError(f"GADM file not found: {GADM_FILE}")

    print(f"\nBuilding Admin1 geometries from: {GADM_FILE}")
    gadm = gpd.read_file(GADM_FILE)
    china = gadm[gadm["country"] == "CN"].copy()
    print(f"  GADM shapes (Admin2): {len(china)}")
    print(f"  GADM columns: {china.columns.tolist()}")
    print(f"  GADM index sample: {str(china.index[0])}")

    # Cerca la colonna con il codice regione
    name_col = None
    for candidate in ["GADM_ID", "name", "id", "GID_2", "GID_1", "region"]:
        if candidate in china.columns:
            name_col = candidate
            print(f"  Using column '{name_col}' for region ID")
            break

    if name_col is None:
        # Prova l'indice
        sample = str(china.index[0])
        if "CN." in sample:
            china["_rid"] = china.index.astype(str)
            name_col = "_rid"
            print(f"  Using index as region ID")
        else:
            raise ValueError(
                f"Cannot identify region ID field.\n"
                f"Columns: {china.columns.tolist()}\n"
                f"Index sample: {sample}\n"
                f"Provide the correct field name manually."
            )

    china["admin1"] = china[name_col].apply(bus_to_admin1)
    admin1_geo = china.dissolve(by="admin1").reset_index()[["admin1", "geometry"]]
    print(f"  Admin1 provinces after dissolve: {len(admin1_geo)}")

    # Merge geometrie con dati
    gdf = admin1_geo.merge(admin1_df, on="admin1", how="left")

    # ── Choropleth maps ───────────────────────────────────────────────────────
    choropleth(gdf,
        col="solar_gw",
        title=f"Solar installed capacity per province (Admin1) — CN2020  |  Total: {total_solar_gw:.1f} GW",
        cbar_label="Solar capacity (GW)",
        cmap="YlOrRd",
        filename="admin1_solar_capacity_gw.png",
    )

    choropleth(gdf,
        col="solar_twh",
        title=f"Solar generation per province (Admin1) — CN2020  |  Total: {total_solar_twh:.1f} TWh",
        cbar_label="Solar generation (TWh)",
        cmap="YlOrRd",
        filename="admin1_solar_generation_twh.png",
    )

    choropleth(gdf,
        col="load_twh",
        title=f"Local load per province (Admin1) — CN2020  |  Total: {total_load_twh:.1f} TWh",
        cbar_label="Load (TWh)",
        cmap="Blues",
        filename="admin1_load_twh.png",
    )

    choropleth(gdf,
        col="solar_load_ratio",
        title="Solar generation / local load ratio per province (Admin1) — CN2020",
        cbar_label="Solar / Load ratio  (>1 = generation exceeds local demand)",
        cmap="RdYlGn_r",
        filename="admin1_solar_load_ratio.png",
    )

    choropleth(gdf,
        col="solar_cf",
        title="Solar availability factor per province (Admin1) — CN2020",
        cbar_label="CF = generation / (capacity × total_hours)  [dimensionless]",
        cmap="plasma",
        filename="admin1_solar_cf.png",
    )

    print(f"\nDone. All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()