"""
solar_spatial_analysis.py
=========================
Analisi spaziale della generazione solare per bus — rete PyPSA-Earth CN2020.

Per ogni bus estrae:
  - Capacità solare installata (GW, p_nom_opt)
  - Generazione solare (TWh, integrale temporale del dispatch)
  - Availability factor solare (CF = generazione / capacità / ore totali)
  - Carico locale (TWh)
  - Rapporto solar/load

Output:
  solar_per_bus.csv
  top10_solar_buses.csv
  map_solar_capacity_gw.png
  map_solar_generation_twh.png
  map_load_twh.png
  map_solar_load_ratio.png
  map_solar_cf.png
  scatter_cf_vs_ratio.png

Uso:
  PYPSA_OUTPUT_DIR="analysis/network/solar_spatial" \\
  python analysis/network/solar_spatial_analysis.py \\
  results/networks/geospatial_v2_approccio1/CN2020_Admin2/elec_s_250_ec_lcopt_3h.nc
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
import seaborn as sns

warnings.filterwarnings("ignore")

# ── PARAMETERS ────────────────────────────────────────────────────────────────

GADM_FILE   = "resources/shapes/gadm_shapes.geojson"
OUTPUT_DIR  = os.environ.get("PYPSA_OUTPUT_DIR", "analysis/network/solar_spatial")

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


def add_china_boundaries(ax, proj):
    if os.path.exists(GADM_FILE):
        gadm = gpd.read_file(GADM_FILE)
        china = gadm[gadm["country"] == "CN"]
        china.boundary.plot(ax=ax, color="black", linewidth=0.5, transform=proj)
    else:
        print(f"[WARNING] GADM file not found: {GADM_FILE} — skipping boundaries")


def bubble_map(df, col, title, cbar_label, cmap, filename,
               size_col=None, vmax_pct=95):
    """
    Bubble map su proiezione cartopy PlateCarree.

    Parameters
    ----------
    df       : DataFrame con colonne x, y, e la colonna `col`
    col      : colonna da codificare con colore
    size_col : colonna da usare per la dimensione dei bubble (default: col)
    vmax_pct : percentile per il limite superiore della scala colori
               (evita che outlier schiaccino la colormap)
    """
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(14, 10), subplot_kw={"projection": proj})
    add_china_boundaries(ax, proj)

    values = df[col].fillna(0).clip(lower=0)
    pos = values[values > 0]
    vmax = float(np.percentile(pos, vmax_pct)) if len(pos) > 0 else 1.0
    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    sz_vals = df[size_col].fillna(0).clip(lower=0) if size_col else values
    sz_max  = sz_vals.max()
    sizes   = (20 + 280 * (sz_vals / sz_max)).values if sz_max > 0 else np.full(len(df), 40)

    sc = ax.scatter(
        df["x"], df["y"],
        c=values, cmap=cmap, norm=norm,
        s=sizes, alpha=0.85,
        edgecolors="gray", linewidths=0.25,
        transform=proj, zorder=4,
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.025, pad=0.02, shrink=0.7)
    cbar.set_label(cbar_label, fontsize=10)
    if vmax_pct < 100:
        cbar.ax.set_title(f"(cap {vmax_pct}th pct)", fontsize=8, pad=4)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_extent([73, 136, 17, 54], crs=proj)
    ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)

    save(fig, filename)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Spatial analysis of solar generation — PyPSA-Earth"
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
    print(f"  Generators:{len(n.generators)}")
    print(f"  Snapshots: {len(n.snapshots)}")

    # ── Time resolution ───────────────────────────────────────────────────────
    if len(n.snapshots) > 1:
        dt = (n.snapshots[1] - n.snapshots[0]).total_seconds() / 3600.0
    else:
        dt = 1.0
    total_hours = len(n.snapshots) * dt
    print(f"  Resolution: {dt}h | Total hours: {total_hours:.0f}h")

    # ── Solar generators ──────────────────────────────────────────────────────
    solar = n.generators[n.generators.carrier == "solar"].copy()
    print(f"\n  Solar generators: {len(solar)}")

    # Dispatch time series [MW]; reindex per includere generatori con dispatch zero
    solar_p = n.generators_t.p.reindex(columns=solar.index, fill_value=0.0)

    # Energia per generatore [TWh]
    solar_energy = (solar_p * dt).sum() / 1e6

    # Capacità per generatore [GW]
    p_nom_opt_gw = solar["p_nom_opt"] / 1e3

    # ── Aggregate by bus ──────────────────────────────────────────────────────
    tmp = pd.DataFrame({
        "bus":       solar["bus"],
        "solar_gw":  p_nom_opt_gw,
        "solar_twh": solar_energy,
    })
    per_bus = tmp.groupby("bus").sum()

    # Availability factor per bus: CF = solar_twh / (solar_gw * total_hours / 1000)
    # Unità: [TWh] / ([GW] × [h] / 1000) = [TWh] / [TWh] = adimensionale
    denom = per_bus["solar_gw"] * total_hours / 1e3
    per_bus["solar_cf"] = np.where(denom > 0, per_bus["solar_twh"] / denom, np.nan)

    # ── Load per bus ──────────────────────────────────────────────────────────
    if not n.loads_t.p.empty:
        load_ts = n.loads_t.p
    elif not n.loads_t.p_set.empty:
        load_ts = n.loads_t.p_set
    else:
        raise ValueError("No load time series found in n.loads_t.p or n.loads_t.p_set")

    load_energy = (load_ts * dt).sum() / 1e6  # Series indexed by load name
    load_df = pd.DataFrame({
        "bus":      n.loads.loc[load_energy.index, "bus"],
        "load_twh": load_energy,
    })
    load_bus = load_df.groupby("bus")["load_twh"].sum()

    per_bus = per_bus.join(load_bus, how="left")

    # Rapporto solar/load
    per_bus["solar_load_ratio"] = per_bus["solar_twh"] / per_bus["load_twh"]

    # ── Bus coordinates ───────────────────────────────────────────────────────
    per_bus = per_bus.join(n.buses[["x", "y"]], how="left")
    per_bus = per_bus.reset_index()  # porta "bus" da index a colonna

    # ── National summary ──────────────────────────────────────────────────────
    total_solar_gw  = per_bus["solar_gw"].sum()
    total_solar_twh = per_bus["solar_twh"].sum()
    total_load_twh  = per_bus["load_twh"].sum()
    national_cf     = total_solar_twh / (total_solar_gw * total_hours / 1e3)

    print(f"\nNational totals:")
    print(f"  Solar installed:  {total_solar_gw:.1f} GW")
    print(f"  Solar generation: {total_solar_twh:.1f} TWh")
    print(f"  Total load:       {total_load_twh:.1f} TWh")
    print(f"  Solar share:      {total_solar_twh / total_load_twh * 100:.1f}%")
    print(f"  National CF:      {national_cf:.3f}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, "solar_per_bus.csv")
    per_bus.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nSaved: {csv_path}")

    top10 = per_bus.nlargest(10, "solar_twh")[
        ["bus", "solar_gw", "solar_twh", "solar_cf", "load_twh", "solar_load_ratio", "x", "y"]
    ]
    top10_path = os.path.join(OUTPUT_DIR, "top10_solar_buses.csv")
    top10.to_csv(top10_path, index=False, float_format="%.3f")
    print(f"Saved: {top10_path}")
    print("\nTop 10 buses by solar generation:")
    print(top10.to_string(index=False))

    # ── Maps ──────────────────────────────────────────────────────────────────
    df = per_bus.copy()

    bubble_map(df,
        col="solar_gw", size_col="solar_gw",
        title=f"Solar installed capacity per bus — CN2020  |  Total: {total_solar_gw:.1f} GW",
        cbar_label="Solar capacity (GW)",
        cmap="YlOrRd",
        filename="map_solar_capacity_gw.png",
    )

    bubble_map(df,
        col="solar_twh", size_col="solar_twh",
        title=f"Solar generation per bus — CN2020  |  Total: {total_solar_twh:.1f} TWh",
        cbar_label="Solar generation (TWh)",
        cmap="YlOrRd",
        filename="map_solar_generation_twh.png",
    )

    bubble_map(df,
        col="load_twh", size_col="load_twh",
        title=f"Local load per bus — CN2020  |  Total: {total_load_twh:.1f} TWh",
        cbar_label="Load (TWh)",
        cmap="Blues",
        filename="map_load_twh.png",
    )

    bubble_map(df,
        col="solar_load_ratio", size_col="solar_twh",
        title="Solar generation / local load ratio per bus — CN2020\n(bubble size = solar TWh)",
        cbar_label="Solar / Load ratio  (>1 = generation exceeds local demand)",
        cmap="RdYlGn_r",
        filename="map_solar_load_ratio.png",
        vmax_pct=100,
    )

    bubble_map(df,
        col="solar_cf", size_col="solar_gw",
        title="Solar availability factor per bus — CN2020\n(bubble size = installed capacity GW)",
        cbar_label="CF = generation / (capacity × total_hours)  [dimensionless]",
        cmap="plasma",
        filename="map_solar_cf.png",
        vmax_pct=100,
    )

    # ── Scatter: CF vs solar/load ratio ──────────────────────────────────────
    # Questo è il plot chiave per motivare (o falsificare) la teoria:
    # il modello piazza solare dove l'availability factor è massimo,
    # producendo un rapporto solar/load elevato in quei bus.
    fig, ax = plt.subplots(figsize=(9, 7))
    valid = df.dropna(subset=["solar_cf", "solar_load_ratio", "solar_gw"])
    valid = valid[valid["solar_gw"] > 0]

    sc = ax.scatter(
        valid["solar_cf"], valid["solar_load_ratio"],
        c=valid["solar_gw"], cmap="YlOrRd",
        s=60, alpha=0.75,
        edgecolors="gray", linewidths=0.3,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Solar installed capacity (GW)", fontsize=10)

    ax.set_xlabel("Availability factor  [CF = generation / (capacity × hours)]", fontsize=11)
    ax.set_ylabel("Solar generation / local load ratio", fontsize=11)
    ax.set_title(
        "Availability factor vs solar/load ratio — CN2020\n"
        "If theory holds: positive correlation, high-CF buses drive overgeneration",
        fontsize=11, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    # Etichetta top 5 per capacità installata
    for _, row in valid.nlargest(5, "solar_gw").iterrows():
        ax.annotate(
            str(row["bus"]),
            (row["solar_cf"], row["solar_load_ratio"]),
            fontsize=7, xytext=(4, 4), textcoords="offset points",
        )

    save(fig, "scatter_cf_vs_ratio.png")

    print(f"\nDone. All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
