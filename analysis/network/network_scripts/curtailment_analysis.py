"""
curtailment_analysis.py
=======================
Analisi del curtailment VRE per provincia — rete PyPSA-Earth CN2020.

Per ogni provincia e carrier (solar, wind) calcola:
  - Energia disponibile  = sum(p_max_pu * p_nom_opt * dt)  [TWh]
  - Energia dispatchata  = sum(p * dt)                      [TWh]
  - Curtailment          = disponibile - dispatchata        [TWh]
  - Curtailment rate     = curtailment / disponibile        [0-1]

Aggregazione: generator → bus → province (via n.buses["province"]).

Output:
  curtailment_per_province.csv
  map_curtailment_solar.png
  map_curtailment_wind.png
  bar_curtailment_twh.png

Uso:
  PYPSA_OUTPUT_DIR="analysis/network/curtailment" python analysis/network/curtailment_analysis.py results/networks/geospatial_v2_approccio1/CN2020_Admin2/elec_s_250_ec_lcopt_3h.nc
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
from province_utils import overlay_admin_boundaries
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import pypsa
import seaborn as sns

warnings.filterwarnings("ignore")

# ── PARAMETERS ────────────────────────────────────────────────────────────────

GADM_FILE  = "resources/shapes/gadm_shapes.geojson"
OUTPUT_DIR = os.path.join(os.environ.get("PYPSA_OUTPUT_DIR", "analysis/network/output"), "curtailment_analysis")

PROVINCE_NAMES = {
    1: "Anhui",        2: "Beijing",      3: "Chongqing",    4: "Fujian",
    5: "Gansu",        6: "Guangdong",    7: "Guangxi",      8: "Guizhou",
    9: "Hainan",      10: "Hebei",       11: "Heilongjiang", 12: "Henan",
   13: "Hubei",       14: "Hunan",       15: "Jiangsu",      16: "Jiangxi",
   17: "Jilin",       18: "Liaoning",    19: "Inner Mongolia",20: "Ningxia",
   21: "Qinghai",     22: "Shaanxi",     23: "Shandong",     24: "Shanghai",
   25: "Shanxi",      26: "Sichuan",     27: "Tianjin",      28: "Xinjiang",
   29: "Tibet",       30: "Yunnan",      31: "Zhejiang",
}

WIND_CARRIERS = ["onwind", "offwind-ac", "offwind-dc"]

# ── HELPERS ───────────────────────────────────────────────────────────────────

def save(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def bus_to_admin1(name):
    parts = str(name).split(".")
    return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else name


def admin1_to_name(code):
    try:
        return PROVINCE_NAMES.get(int(code.split(".")[1]), code)
    except (IndexError, ValueError):
        return code


def curtailment_by_province(n, carriers, dt):
    """
    Calcola curtailment per i generatori dei carrier indicati,
    aggregato per provincia via n.buses["province"].

    Returns DataFrame con colonne:
      available_twh, dispatched_twh, curtailment_twh, curtailment_rate
    """
    gens = n.generators[n.generators.carrier.isin(carriers)].copy()
    if gens.empty:
        return pd.DataFrame()

    # p_max_pu disponibile (time series)
    p_max_pu = n.generators_t.p_max_pu.reindex(columns=gens.index, fill_value=0.0)

    # Energia disponibile per generatore [TWh]
    available = (p_max_pu * gens["p_nom_opt"] * dt).sum() / 1e6

    # Energia dispatchata per generatore [TWh]
    p_dispatch = n.generators_t.p.reindex(columns=gens.index, fill_value=0.0)
    dispatched = (p_dispatch * dt).sum() / 1e6

    # Curtailment per generatore
    curtailment = available - dispatched

    # Aggrega per provincia via bus → province
    province_key = gens["bus"].map(n.buses["province"])
    province_key.name = "admin1"

    df = pd.DataFrame({
        "available_twh":  available,
        "dispatched_twh": dispatched,
        "curtailment_twh": curtailment,
    }).groupby(province_key).sum()

    df["curtailment_rate"] = df["curtailment_twh"] / df["available_twh"].replace(0, np.nan)
    return df


def build_admin1_geo():
    gadm  = gpd.read_file(GADM_FILE)
    china = gadm[gadm["country"] == "CN"].copy()
    china["admin1"] = china["GADM_ID"].apply(bus_to_admin1)
    return china.dissolve(by="admin1").reset_index()[["admin1", "geometry"]]


def choropleth(gdf, col, title, cbar_label, cmap, filename, vmin=0, vmax=None):
    fig, ax = plt.subplots(figsize=(14, 10))
    _vmax = vmax if vmax is not None else gdf[col].max()
    gdf.plot(
        column=col, ax=ax, cmap=cmap,
        vmin=vmin, vmax=_vmax,
        edgecolor="none", linewidth=0.0,
        legend=True,
        legend_kwds={"label": cbar_label, "orientation": "vertical",
                     "shrink": 0.6, "pad": 0.02},
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


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Curtailment VRE analysis — Admin1 level, PyPSA-Earth"
    )
    parser.add_argument("network", help="Path to .nc network file")
    args = parser.parse_args()

    if not os.path.exists(args.network):
        print(f"[ERROR] File not found: {args.network}")
        raise SystemExit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sns.set_style("white")
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "font.size": 11,
    })

    # Extract year label from network path (e.g. CN2020, CN2024)
    import re
    match = re.search(r'CN(\d{4})', args.network)
    year_label = f"CN{match.group(1)}" if match else os.path.basename(os.path.dirname(os.path.dirname(args.network)))
    print(f"Loading: {args.network} (label: {year_label})")
    n = pypsa.Network(args.network)
    print(f"  Buses:     {len(n.buses)}")
    print(f"  Snapshots: {len(n.snapshots)}")

    # Enrich network
    n.buses["province"] = n.buses.index.map(bus_to_admin1)

    dt = (n.snapshots[1] - n.snapshots[0]).total_seconds() / 3600.0 if len(n.snapshots) > 1 else 1.0
    print(f"  Resolution: {dt}h | Total hours: {len(n.snapshots) * dt:.0f}h")

    # ── Curtailment per carrier group ─────────────────────────────────────────
    print("\nComputing curtailment...")
    solar_df = curtailment_by_province(n, ["solar"],       dt)
    wind_df  = curtailment_by_province(n, WIND_CARRIERS,   dt)

    # ── National summary ──────────────────────────────────────────────────────
    def nat(df, label):
        avail = df["available_twh"].sum()
        disp  = df["dispatched_twh"].sum()
        curt  = df["curtailment_twh"].sum()
        rate  = curt / avail if avail > 0 else 0
        print(f"  {label}: available {avail:.1f} TWh | dispatched {disp:.1f} TWh | "
              f"curtailment {curt:.1f} TWh ({rate:.1%})")

    print("\nNational curtailment:")
    nat(solar_df, "Solar")
    nat(wind_df,  "Wind")

    # ── Per-province table ────────────────────────────────────────────────────
    print("\nSolar curtailment by province (sorted by curtailment_twh):")
    solar_display = solar_df.copy()
    solar_display.index = solar_display.index.map(
        lambda x: f"{admin1_to_name(x)} ({x})"
    )
    print(
        solar_display.sort_values("curtailment_twh", ascending=False)
        [["available_twh", "dispatched_twh", "curtailment_twh", "curtailment_rate"]]
        .to_string()
    )

    print("\nWind curtailment by province (sorted by curtailment_twh):")
    wind_display = wind_df.copy()
    wind_display.index = wind_display.index.map(
        lambda x: f"{admin1_to_name(x)} ({x})"
    )
    print(
        wind_display.sort_values("curtailment_twh", ascending=False)
        [["available_twh", "dispatched_twh", "curtailment_twh", "curtailment_rate"]]
        .to_string()
    )

    # ── Save CSV ──────────────────────────────────────────────────────────────
    combined = solar_df.add_suffix("_solar").join(
        wind_df.add_suffix("_wind"), how="outer"
    ).reset_index()
    combined["province_name"] = combined["admin1"].apply(admin1_to_name)
    csv_path = os.path.join(OUTPUT_DIR, "curtailment_per_province.csv")
    combined.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nSaved: {csv_path}")

    # ── Build Admin1 geometries ───────────────────────────────────────────────
    print("\nBuilding Admin1 geometries...")
    admin1_geo = build_admin1_geo()

    # ── Map: solar curtailment rate ───────────────────────────────────────────
    gdf_solar = admin1_geo.merge(
        solar_df[["curtailment_rate", "curtailment_twh"]].reset_index(),
        on="admin1", how="left"
    )
    nat_solar_rate = solar_df["curtailment_twh"].sum() / solar_df["available_twh"].sum()
    vmax_solar = max(round(solar_df["curtailment_rate"].max() * 1.1, 3), 0.01)
    choropleth(gdf_solar,
        col="curtailment_rate",
        title=f"Solar curtailment rate per province — {year_label}  |  National: {nat_solar_rate:.1%}",
        cbar_label="Curtailment rate  (curtailed / available)",
        cmap="YlOrRd",
        filename="map_curtailment_solar.png",
        vmax=vmax_solar,
    )

    # ── Map: wind curtailment rate ────────────────────────────────────────────
    gdf_wind = admin1_geo.merge(
        wind_df[["curtailment_rate", "curtailment_twh"]].reset_index(),
        on="admin1", how="left"
    )
    nat_wind_rate = wind_df["curtailment_twh"].sum() / wind_df["available_twh"].sum()
    vmax_wind = max(round(wind_df["curtailment_rate"].max() * 1.1, 3), 0.01)
    choropleth(gdf_wind,
        col="curtailment_rate",
        title=f"Wind curtailment rate per province — {year_label}  |  National: {nat_wind_rate:.1%}",
        cbar_label="Curtailment rate  (curtailed / available)",
        cmap="YlOrRd",
        filename="map_curtailment_wind.png",
        vmax=vmax_wind,
    )

    # ── Bar chart: absolute curtailment TWh per province ─────────────────────
    # Mostra solar e wind impilati, ordinati per curtailment totale.
    all_prov = sorted(
        set(solar_df.index) | set(wind_df.index),
        key=lambda p: (
            solar_df.loc[p, "curtailment_twh"] if p in solar_df.index else 0
          + wind_df.loc[p,  "curtailment_twh"] if p in wind_df.index  else 0
        ),
        reverse=True,
    )
    # Filtra province con curtailment > 0
    all_prov = [p for p in all_prov if (
        (solar_df.loc[p, "curtailment_twh"] if p in solar_df.index else 0) +
        (wind_df.loc[p,  "curtailment_twh"] if p in wind_df.index  else 0)
    ) > 0.01]

    labels = [f"{admin1_to_name(p)}" for p in all_prov]
    solar_curt = [solar_df.loc[p, "curtailment_twh"] if p in solar_df.index else 0 for p in all_prov]
    wind_curt  = [wind_df.loc[p,  "curtailment_twh"] if p in wind_df.index  else 0 for p in all_prov]

    fig, ax = plt.subplots(figsize=(12, max(5, len(all_prov) * 0.4)))
    y = np.arange(len(all_prov))
    ax.barh(y, solar_curt, color="#f4a620", label="Solar", edgecolor="none")
    ax.barh(y, wind_curt,  left=solar_curt, color="#1f77b4", label="Wind", edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Curtailment  (TWh)", fontsize=11)
    ax.set_title(
        "VRE curtailment by province — " + year_label + "\n"
        f"Solar: {solar_df['curtailment_twh'].sum():.1f} TWh  |  "
        f"Wind: {wind_df['curtailment_twh'].sum():.1f} TWh",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    save(fig, "bar_curtailment_twh.png")

    print(f"\nDone. All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
