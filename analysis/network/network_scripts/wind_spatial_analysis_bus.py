"""
wind_spatial_analysis_bus.py
============================
Spatial analysis of wind generation per bus — PyPSA-Earth.

EPISTEMIC NOTE (read before interpreting plots):
─────────────────────────────────────────────────────────────────────────────
  SOLVER OUTPUT  (optimisation decisions, run-specific):
    p_nom_opt      → installed capacity
    p (dispatch)   → actual generation
    curtailment    → (ERA5_potential − dispatch) / ERA5_potential
    CF_realised    = generation [TWh] / (capacity [GW] × total_hours [h])

  ERA5 INPUT  (fixed BEFORE the solver, atlite on ERA5, weather year varies by run):
    p_max_pu_t     → normalised hourly availability [0–1]
    CF_ERA5        = mean(p_max_pu_t)
                   p_nom cancels out — independent of installed capacity
                   NOT a model decision — meteorological input only

  Relationship (verifiable from numbers):
    CF_realised = CF_ERA5 × (1 − curtailment_rate)
─────────────────────────────────────────────────────────────────────────────

Output directory: {PYPSA_OUTPUT_DIR}/wind_spatial_analysis_bus/
  wind_per_bus_onwind.csv
  wind_per_bus_offwind.csv
  map_onwind_capacity_gw.png       [SOLVER OUTPUT]
  map_onwind_curtailment_rate.png  [SOLVER OUTPUT]
  map_onwind_cf_realised.png       [SOLVER OUTPUT]
  map_onwind_cf_era5.png           [ERA5 INPUT — solver-independent]
  map_offwind_capacity_gw.png      [SOLVER OUTPUT]
  map_offwind_curtailment_rate.png [SOLVER OUTPUT]
  map_offwind_cf_era5.png          [ERA5 INPUT]

Usage:
  cd /p/tmp/lorenzop/pypsa-earth-ivan && \\
  PYPSA_OUTPUT_DIR=analysis/network/output/CN2060_NUCAP_BIO_95 \\
  python3 analysis/network/network_scripts/wind_spatial_analysis_bus.py \\
      results/CN2060_loadCF/networks/elec_s_250_ec_lcopt_NUCAP-BIOCAP-Co2L0.05-3h.nc \\
      --run-tag CN2060_95pct
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

warnings.filterwarnings("ignore")

# ── PARAMETERS ────────────────────────────────────────────────────────────────

ADMIN2_FILE = "/p/tmp/ivanra/PyPSA-China-PIK/resources/data/regions/admin2_shapes.geojson"

# Output goes to {PYPSA_OUTPUT_DIR}/wind_spatial_analysis_bus/
# PYPSA_OUTPUT_DIR is set per-run via environment variable — never hardcoded here.
_BASE_DIR  = os.environ.get("PYPSA_OUTPUT_DIR", "analysis/network/output")
OUTPUT_DIR = os.path.join(_BASE_DIR, "wind_spatial_analysis_bus")

# ── STYLE ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "font.size": 11,
    "pdf.fonttype": 42,   # vector text in PDF (editable in Illustrator/Inkscape)
})


# ── HELPERS ───────────────────────────────────────────────────────────────────

def save(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def add_china_boundaries(ax, proj):
    """Overlay Admin2 (thin) and Admin1 (thick) boundaries from admin2_shapes.geojson."""
    if os.path.exists(ADMIN2_FILE):
        admin = gpd.read_file(ADMIN2_FILE)
        admin.boundary.plot(
            ax=ax, color="black", linewidth=0.3, alpha=0.5, transform=proj
        )
        admin.dissolve("NAME_1").boundary.plot(
            ax=ax, color="black", linewidth=0.8, transform=proj
        )
    else:
        print(f"  [WARNING] Admin2 boundary file not found: {ADMIN2_FILE}")


def bubble_map(df, col, title, cbar_label, cmap, filename,
               size_col=None, size_unit="GW", vmax_pct=95, vmin=0):
    """
    Bubble map on PlateCarree projection.
    Each bubble = one bus. Position from columns x, y.

    Bubble colour encodes `col`.
    Bubble size  encodes `size_col` (default: same as col).

    A size legend is drawn in the lower-left corner so the reader
    knows what bubble diameter means without guessing.

    Parameters
    ----------
    col       : column to encode with colour
    size_col  : column to encode with bubble size (default: col)
    size_unit : unit label shown in the size legend (e.g. "GW", "TWh")
    vmax_pct  : percentile cap for the colour scale (avoids outliers
                collapsing the colour range)
    """
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(14, 10), subplot_kw={"projection": proj})
    add_china_boundaries(ax, proj)

    values = df[col].fillna(0).clip(lower=vmin)
    pos    = values[values > 0]
    vmax   = float(np.percentile(pos, vmax_pct)) if len(pos) > 0 else 1.0
    norm   = mcolors.Normalize(vmin=vmin, vmax=vmax)

    sz_col  = size_col if size_col else col
    sz_vals = df[sz_col].fillna(0).clip(lower=0)
    sz_max  = sz_vals.max()
    sizes   = (20 + 300 * (sz_vals / sz_max)).values if sz_max > 0 else np.full(len(df), 40)

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

    # ── Size legend ───────────────────────────────────────────────────────────
    # Shows what bubble diameter means. Uses 3 representative values.
    if sz_max > 0:
        legend_vals   = [sz_max * 0.1, sz_max * 0.5, sz_max]
        legend_sizes  = [20 + 300 * (v / sz_max) for v in legend_vals]
        legend_labels = [f"{v:.0f} {size_unit}" for v in legend_vals]
        legend_handles = [
            plt.scatter([], [], s=s, c="gray", alpha=0.6,
                        edgecolors="gray", linewidths=0.5)
            for s in legend_sizes
        ]
        ax.legend(
            legend_handles, legend_labels,
            title=f"Bubble size\n({sz_col})",
            loc="lower left",
            framealpha=0.8,
            fontsize=9,
            title_fontsize=9,
        )

    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    ax.set_extent([73, 136, 17, 54], crs=proj)
    ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)

    save(fig, filename)


# ── CORE ANALYSIS ─────────────────────────────────────────────────────────────

def analyse_carrier(n, carrier, dt, total_hours):
    """
    Extract per-bus statistics for a given wind carrier.

    Per generator:
      CF_ERA5        = mean(p_max_pu_t)                             [ERA5 INPUT]
      actual_twh     = sum(p_dispatch × dt) / 1e6                  [SOLVER OUTPUT]
      potential_twh  = sum(p_max_pu_t × p_nom_opt × dt) / 1e6     [ERA5 × solver cap]
      curtailment    = potential_twh − actual_twh                   [SOLVER OUTPUT]
      CF_realised    = actual_twh / (p_nom_opt_gw × total_hours)   [SOLVER OUTPUT]

    Aggregated to bus level (CF_ERA5 capacity-weighted by p_nom_opt).

    Returns
    -------
    per_bus : DataFrame with columns:
              bus, x, y, gw, actual_twh, potential_twh,
              curtailment_twh, curtailment_rate, cf_realised, cf_era5
    None if no generators found for this carrier.
    """
    gens = n.generators[n.generators.carrier == carrier].copy()
    if gens.empty:
        print(f"  [SKIP] No generators found for carrier: {carrier}")
        return None

    print(f"\n  {carrier}: {len(gens)} generators, "
          f"p_nom_opt total = {gens['p_nom_opt'].sum()/1e3:.1f} GW")

    # ── Actual dispatch [MW] — SOLVER OUTPUT ─────────────────────────────────
    p_actual = n.generators_t.p.reindex(columns=gens.index, fill_value=0.0)

    # ── ERA5 availability p_max_pu_t — ERA5 INPUT ─────────────────────────────
    # CF_ERA5 = mean(p_max_pu_t)  — p_nom cancels, independent of capacity
    has_ts = (
        hasattr(n.generators_t, "p_max_pu")
        and not n.generators_t.p_max_pu.empty
        and any(g in n.generators_t.p_max_pu.columns for g in gens.index)
    )
    if has_ts:
        p_max_pu = n.generators_t.p_max_pu.reindex(columns=gens.index)
        missing  = [g for g in gens.index if g not in n.generators_t.p_max_pu.columns]
        if missing:
            print(f"  [WARNING] {len(missing)} generators missing p_max_pu_t "
                  f"— using static p_max_pu fallback")
            for g in missing:
                static = float(gens.loc[g, "p_max_pu"]) if "p_max_pu" in gens.columns else 1.0
                p_max_pu[g] = static
    else:
        print(f"  [WARNING] No time-varying p_max_pu_t found for {carrier} "
              f"— using static values. CF_ERA5 will be a constant per generator.")
        static_vals = gens["p_max_pu"].fillna(1.0).values
        p_max_pu = pd.DataFrame(
            np.tile(static_vals, (len(n.snapshots), 1)),
            index=n.snapshots,
            columns=gens.index,
        )

    # ── Per-generator quantities ──────────────────────────────────────────────
    p_nom_opt_mw = gens["p_nom_opt"]        # [MW]
    p_nom_opt_gw = p_nom_opt_mw / 1e3       # [GW]

    # CF_ERA5 = mean(p_max_pu_t)  [ERA5 INPUT — independent of p_nom]
    cf_era5_per_gen = p_max_pu.mean()

    # Actual generation [TWh]  [SOLVER OUTPUT]
    actual_twh = (p_actual * dt).sum() / 1e6

    # ERA5 potential [TWh] = sum(p_max_pu_t × p_nom_opt × dt) / 1e6
    # What ERA5 made available regardless of dispatch decisions
    potential_twh = (p_max_pu.multiply(p_nom_opt_mw) * dt).sum() / 1e6

    # Curtailment [TWh]  [SOLVER OUTPUT]
    curtailment_twh = (potential_twh - actual_twh).clip(lower=0)

    # ── Aggregate by bus ──────────────────────────────────────────────────────
    # CF_ERA5 aggregated as capacity-weighted mean (weight = p_nom_opt)
    # This gives the ERA5 quality of what was actually built, not of empty sites
    tmp = pd.DataFrame({
        "bus":             gens["bus"].values,
        "gw":              p_nom_opt_gw.values,
        "actual_twh":      actual_twh.values,
        "potential_twh":   potential_twh.values,
        "curtailment_twh": curtailment_twh.values,
        "cf_era5_x_gw":    (cf_era5_per_gen * p_nom_opt_gw).values,
    }, index=gens.index)

    per_bus = tmp.groupby("bus")[
        ["gw", "actual_twh", "potential_twh", "curtailment_twh", "cf_era5_x_gw"]
    ].sum()

    # Curtailment rate per bus  [SOLVER OUTPUT]
    per_bus["curtailment_rate"] = np.where(
        per_bus["potential_twh"] > 0,
        per_bus["curtailment_twh"] / per_bus["potential_twh"],
        np.nan,
    )

    # CF_realised per bus = generation [TWh] / (capacity [GW] × total_hours [h])
    # [SOLVER OUTPUT]
    denom_bus = per_bus["gw"] * total_hours / 1e3   # [GW × h / 1000 = TWh]
    per_bus["cf_realised"] = np.where(
        denom_bus > 0,
        per_bus["actual_twh"] / denom_bus,
        np.nan,
    )

    # CF_ERA5 per bus — capacity-weighted mean  [ERA5 INPUT]
    per_bus["cf_era5"] = np.where(
        per_bus["gw"] > 0,
        per_bus["cf_era5_x_gw"] / per_bus["gw"],
        np.nan,
    )

    # Drop helper column
    per_bus.drop(columns=["cf_era5_x_gw"], inplace=True)

    # Bus coordinates
    per_bus = per_bus.join(n.buses[["x", "y"]], how="left")
    per_bus = per_bus.reset_index()

    # ── National summary ──────────────────────────────────────────────────────
    total_gw   = per_bus["gw"].sum()
    total_twh  = per_bus["actual_twh"].sum()
    total_curt = per_bus["curtailment_twh"].sum()
    total_pot  = per_bus["potential_twh"].sum()
    nat_curt_rate = total_curt / total_pot if total_pot > 0 else 0.0

    cf_era5_nat  = (per_bus["cf_era5"].fillna(0) * per_bus["gw"]).sum() / total_gw if total_gw > 0 else 0.0
    cf_real_nat  = total_twh / (total_gw * total_hours / 1e3) if total_gw > 0 else 0.0
    cf_check     = cf_era5_nat * (1.0 - nat_curt_rate)
    discrepancy  = abs(cf_check - cf_real_nat)

    print(f"  Installed:         {total_gw:.1f} GW          [SOLVER OUTPUT]")
    print(f"  Generation:        {total_twh:.1f} TWh        [SOLVER OUTPUT]")
    print(f"  Curtailment:       {total_curt:.1f} TWh ({nat_curt_rate*100:.1f}%)  [SOLVER OUTPUT]")
    print(f"  CF_realised:       {cf_real_nat:.3f}            [SOLVER OUTPUT]")
    print(f"  CF_ERA5 (cap-wtd): {cf_era5_nat:.3f}            [ERA5 INPUT]")
    print(f"  Check CF_ERA5×(1−curt) = {cf_check:.3f}  "
          f"{'OK' if discrepancy < 0.005 else f'DISCREPANCY {discrepancy:.4f} — check p_max_pu coverage'}")

    return per_bus


# ── PLOTS ─────────────────────────────────────────────────────────────────────

def make_plots(per_bus, carrier_label, prefix, run_tag, weather_year):
    """
    Produce 4 maps per carrier.
    Every title explicitly declares [SOLVER OUTPUT] or [ERA5 INPUT].
    Bubble size always encodes installed capacity (GW) so the reader
    can judge how much weight each bus carries in the national total.
    """
    df = per_bus[per_bus["gw"] > 0].copy()
    total_gw   = df["gw"].sum()
    total_curt = df["curtailment_twh"].sum()
    total_pot  = df["potential_twh"].sum()
    nat_curt_pct = total_curt / total_pot * 100 if total_pot > 0 else 0.0

    # 1. Installed capacity — SOLVER OUTPUT
    bubble_map(
        df,
        col="gw", size_col="gw", size_unit="GW",
        title=(
            f"{carrier_label} installed capacity per bus — {run_tag}\n"
            f"[SOLVER OUTPUT: p_nom_opt]   Total: {total_gw:.0f} GW"
        ),
        cbar_label="Installed capacity (GW)  [solver output]",
        cmap="YlOrRd",
        filename=f"map_{prefix}_capacity_gw.png",
    )

    # 2. Curtailment rate — SOLVER OUTPUT
    # Bubble size = GW so large-capacity buses are visually prominent
    bubble_map(
        df,
        col="curtailment_rate", size_col="gw", size_unit="GW",
        title=(
            f"{carrier_label} curtailment rate per bus — {run_tag}\n"
            f"[SOLVER OUTPUT: (ERA5_potential − dispatch) / ERA5_potential]"
            f"   National: {nat_curt_pct:.1f}%\n"
            f"Bubble size = installed capacity (GW)"
        ),
        cbar_label="Curtailment rate  [0 = none,  1 = full curtailment]",
        cmap="RdYlGn_r",
        filename=f"map_{prefix}_curtailment_rate.png",
        vmax_pct=100,
    )

    # 3. CF_realised — SOLVER OUTPUT
    # Formula: generation [TWh] / (capacity [GW] × total_hours [h])
    bubble_map(
        df,
        col="cf_realised", size_col="gw", size_unit="GW",
        title=(
            f"{carrier_label} realised CF per bus — {run_tag}\n"
            f"[SOLVER OUTPUT]   CF = generation [TWh] / (capacity [GW] × total hours [h])\n"
            f"Bubble size = installed capacity (GW)"
        ),
        cbar_label="CF realised = TWh / (GW × h)  [dimensionless]",
        cmap="plasma",
        filename=f"map_{prefix}_cf_realised.png",
        vmax_pct=100,
    )

    # 4. CF_ERA5 — ERA5 INPUT (fixed before solver)
    bubble_map(
        df,
        col="cf_era5", size_col="gw", size_unit="GW",
        title=(
            f"{carrier_label} CF_ERA5 per bus — {run_tag}\n"
            f"[ERA5 INPUT: mean(p_max_pu_t), atlite on ERA5 {weather_year}]"
            f"   Independent of solver decisions\n"
            f"Bubble size = installed capacity (GW)"
        ),
        cbar_label="CF_ERA5 = mean(p_max_pu_t)  [ERA5 input, independent of solver]",
        cmap="rainbow",
        filename=f"map_{prefix}_cf_era5.png",
        vmax_pct=100,
    )

    print(f"  Maps for {carrier_label} saved.")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Wind spatial analysis per bus — PyPSA-Earth.\n"
            "Output goes to {PYPSA_OUTPUT_DIR}/wind_spatial_analysis_bus/"
        )
    )
    parser.add_argument("network", help="Path to the solved .nc network file")
    parser.add_argument(
        "--run-tag", default="",
        help="Run label for plot titles, e.g. CN2060_95pct (default: extracted from filename)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.network):
        print(f"[ERROR] File not found: {args.network}")
        raise SystemExit(1)

    # If no run tag provided, use the network filename stem as fallback
    run_tag = args.run_tag if args.run_tag else os.path.splitext(os.path.basename(args.network))[0]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Run tag:          {run_tag}")

    print(f"\nLoading: {args.network}")
    n = pypsa.Network(args.network)
    print(f"  Buses:      {len(n.buses)}")
    print(f"  Generators: {len(n.generators)}")
    print(f"  Snapshots:  {len(n.snapshots)}")
    weather_year = n.snapshots[0].year
    print(f"  Weather year: {weather_year}")

    # ── Time resolution ───────────────────────────────────────────────────────
    if len(n.snapshots) > 1:
        dt = (n.snapshots[1] - n.snapshots[0]).total_seconds() / 3600.0
    else:
        dt = 1.0
    total_hours = len(n.snapshots) * dt
    print(f"  Resolution: {dt}h  |  Total hours: {total_hours:.0f}h")

    # ── Analysis per carrier ──────────────────────────────────────────────────
    carriers = [
        ("onwind",     "Onshore Wind",  "onwind"),
        ("offwind-dc", "Offshore Wind", "offwind"),
    ]

    for carrier, label, prefix in carriers:
        per_bus = analyse_carrier(n, carrier, dt, total_hours)
        if per_bus is None:
            continue

        csv_path = os.path.join(OUTPUT_DIR, f"wind_per_bus_{prefix}.csv")
        per_bus.to_csv(csv_path, index=False, float_format="%.4f")
        print(f"  CSV saved: {csv_path}")

        make_plots(per_bus, label, prefix, run_tag, weather_year)

    print(f"\nDone. All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()