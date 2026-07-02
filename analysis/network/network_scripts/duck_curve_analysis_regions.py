"""
duck_curve_analysis_regions.py
===============================
Duck curve e dispatch stack per macro-regione — rete PyPSA-Earth CN2020.

Stesse analisi di duck_curve_analysis.py ma filtrate per 8 macro-regioni
geografiche invece che aggregate a livello nazionale.

Zone sincrone cinesi ufficiali (fonte: grids.csv PyPSA-China-PIK):
  NC  North China Sync Zone     — Beijing (CN.2), Hebei (CN.10), Inner Mongolia (CN.19),
                                   Shandong (CN.23), Shanxi (CN.25), Tianjin (CN.27)
  NE  Northeast China Sync Zone — Heilongjiang (CN.11), Jilin (CN.17), Liaoning (CN.18)
  EC  East China Sync Zone      — Anhui (CN.1), Fujian (CN.4), Jiangsu (CN.15),
                                   Shanghai (CN.24), Zhejiang (CN.31)
  CC  Central China Sync Zone   — Chongqing (CN.3), Henan (CN.12), Hubei (CN.13),
                                   Hunan (CN.14), Jiangxi (CN.16), Sichuan (CN.26)
  NW  Northwest China Sync Zone — Gansu (CN.5), Ningxia (CN.20), Qinghai (CN.21),
                                   Shaanxi (CN.22), Xinjiang (CN.28)
  SC  South China Sync Zone     — Guangdong (CN.6), Guangxi (CN.7), Guizhou (CN.8),
                                   Hainan (CN.9), Yunnan (CN.30)
  TB  Tibet (DC connected)      — Tibet (CN.29)

Per ogni regione vengono prodotti 6 file:
  dispatch_stack_annual_{REGION}.png
  dispatch_stack_seasonal_{REGION}.png
  duck_curve_annual_{REGION}.png
  solar_heatmap_{REGION}.png
  wind_heatmap_{REGION}.png   — onwind + offwind-ac + offwind-dc
  hydro_heatmap_{REGION}.png  — hydro (reservoir) + ror

Con risoluzione 3h e UTC+8, i dati esistono a 8 ore per giorno:
[2, 5, 8, 11, 14, 17, 20, 23] CST.
Nessun reindex a 24 ore — si usano solo le ore reali per evitare il
pattern a dente di sega.

Output dir default: /p/tmp/lorenzop/pypsa-earth-ivan/analysis/network/duck_curve_regions
  (override con env var PYPSA_OUTPUT_DIR)

Uso:
  PYPSA_OUTPUT_DIR="/p/tmp/lorenzop/pypsa-earth-ivan/analysis/network/duck_curve_regions" \\
  python analysis/network/duck_curve_analysis_regions.py \\
  results/networks/geospatial_v2_approccio1/CN2020_Admin2/elec_s_250_ec_lcopt_3h.nc
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pypsa
import seaborn as sns

warnings.filterwarnings("ignore")

# ── PARAMETERS ────────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(os.environ.get("PYPSA_OUTPUT_DIR", "analysis/network/output"), "duck_curve_analysis_regions")
UTC_OFFSET = 8  # China Standard Time

SEASONS = {
    "Winter (Dec–Feb)": [12, 1,  2],
    "Spring (Mar–May)": [3,  4,  5],
    "Summer (Jun–Aug)": [6,  7,  8],
    "Autumn (Sep–Nov)": [9,  10, 11],
}

# Macro-regions: code → (label, province_admin1_ids)
# Province number N maps to admin1 ID "CN.N" (e.g. 28 → "CN.28")
# Fonte: grids.csv PyPSA-China-PIK (Ivan Ramirez)
# Inner Mongolia (CN.19) assegnata a NC — connessione prevalente.
# Beijing (CN.2) e Tianjin (CN.27) inclusi per correttezza ma senza bus nel network s_250.
REGIONAL_GROUPS = {
    "NC": ("North China Sync Zone",     [2,  10, 19, 23, 25, 27]),
    "NE": ("Northeast China Sync Zone", [11, 17, 18]),
    "EC": ("East China Sync Zone",      [1,  4,  15, 24, 31]),
    "CC": ("Central China Sync Zone",   [3,  12, 13, 14, 16, 26]),
    "NW": ("Northwest China Sync Zone", [5,  20, 21, 22, 28]),
    "SC": ("South China Sync Zone",     [6,  7,  8,  9,  30]),
    "TB": ("Tibet  (DC connected)",     [29]),
}

CARRIER_COLORS = {
    "nuclear":    "#e75480",
    "coal":       "#4d4d4d",
    "lignite":    "#9e5a00",
    "oil":        "#2c3e50",
    "gas":        "#f1948a",
    "OCGT":       "#e59866",
    "CCGT":       "#d35400",
    "hydro":      "#2980b9",
    "ror":        "#5dade2",
    "onwind":     "#1f77b4",
    "offwind-ac": "#aec6cf",
    "offwind-dc": "#7fb3d3",
    "solar":      "#f4a620",
    "biomass":    "#27ae60",
    "PHS":        "#1abc9c",
    "battery":    "#2ecc71",
    "other":      "#bdc3c7",
}

STACK_ORDER = [
    "nuclear", "coal", "lignite", "oil", "gas", "OCGT", "CCGT",
    "biomass", "hydro", "ror", "onwind", "offwind-ac", "offwind-dc",
    "solar", "PHS", "battery",
]

CARRIER_LABELS = {
    "offwind-ac": "offshore wind AC",
    "offwind-dc": "offshore wind DC",
    "onwind":     "onshore wind",
    "ror":        "run-of-river",
    "PHS":        "pumped hydro",
}

WIND_CARRIERS  = ["onwind", "offwind-ac", "offwind-dc"]
HYDRO_CARRIERS = ["hydro", "ror"]

HEATMAP_SPECS = {
    "solar": {
        "carriers":    ["solar"],
        "storage":     False,
        "cmap":        "YlOrRd",
        "cbar_label":  "Average solar generation  (GW)",
        "title_tpl":   "Solar generation — {region_label}  [{code}]\n"
                       "monthly × hourly average, CST  |  3h resolution",
    },
    "wind": {
        "carriers":    WIND_CARRIERS,
        "storage":     False,
        "cmap":        "Blues",
        "cbar_label":  "Average wind generation  (GW)",
        "title_tpl":   "Wind generation (onshore + offshore) — {region_label}  [{code}]\n"
                       "monthly × hourly average, CST  |  3h resolution",
    },
    "hydro": {
        "carriers":    HYDRO_CARRIERS,
        "storage":     False,
        "cmap":        "GnBu",
        "cbar_label":  "Average hydro generation  (GW)",
        "title_tpl":   "Hydro generation (reservoir + run-of-river) — {region_label}  [{code}]\n"
                       "monthly × hourly average, CST  |  3h resolution",
    },
}

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ── HELPERS ───────────────────────────────────────────────────────────────────

def save(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def bus_to_admin1(name):
    parts = str(name).split(".")
    return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else name


def cst_hour(idx):
    return (idx.hour + UTC_OFFSET) % 24


def cst_month(idx):
    return (idx + pd.Timedelta(hours=UTC_OFFSET)).month


def hourly_mean_gw(ts, idx, mask=None):
    """
    Media per ora CST di una Series in MW → GW.
    Restituisce Series indexata sulle ore realmente presenti nel dataset.
    """
    s = ts[mask].values if mask is not None else ts.values
    h = cst_hour(idx[mask] if mask is not None else idx)
    return pd.Series(s, index=h).groupby(level=0).mean() / 1e3


def add_cycle_close(hours, values):
    return (
        np.append(hours,  hours[0] + 24),
        np.append(values, values[0]),
    )


def region_buses(n, admin1_ids):
    """Indici dei bus appartenenti alla regione."""
    return n.buses[n.buses["province"].isin(admin1_ids)].index


def build_dispatch_stack(n, idx, bus_set, mask=None):
    """
    Dispatch stack filtrato per bus_set e (opzionalmente) stagione.

    Differenza rispetto a duck_curve_analysis.py: accetta bus_set per
    filtrare generators, storage_units e loads alla regione specifica.
    """
    gen_stack = {}
    for carrier in n.generators.carrier.unique():
        gens = n.generators[
            (n.generators.carrier == carrier) &
            (n.generators.bus.isin(bus_set))
        ].index
        if gens.empty:
            continue
        p_ts = n.generators_t.p.reindex(columns=gens, fill_value=0.0).sum(axis=1)
        avg  = hourly_mean_gw(p_ts, idx, mask)
        if avg.sum() > 0.01:
            gen_stack[carrier] = avg

    sto_disch, sto_charg = {}, {}
    for carrier in n.storage_units.carrier.unique():
        sus = n.storage_units[
            (n.storage_units.carrier == carrier) &
            (n.storage_units.bus.isin(bus_set))
        ].index
        if sus.empty:
            continue
        p_ts = n.storage_units_t.p.reindex(columns=sus, fill_value=0.0).sum(axis=1)
        avg  = hourly_mean_gw(p_ts, idx, mask)
        d = avg.clip(lower=0)
        c = avg.clip(upper=0)
        if d.sum() > 0.01:
            sto_disch[carrier] = d
        if c.abs().sum() > 0.01:
            sto_charg[carrier] = c

    region_loads = n.loads[n.loads.bus.isin(bus_set)].index
    if not region_loads.empty:
        load_ts = (
            n.loads_t.p.reindex(columns=region_loads, fill_value=0.0).sum(axis=1)
            if not n.loads_t.p.empty
            else n.loads_t.p_set.reindex(columns=region_loads, fill_value=0.0).sum(axis=1)
        )
    else:
        load_ts = pd.Series(0.0, index=idx)

    load_h = hourly_mean_gw(load_ts, idx, mask)
    return gen_stack, sto_disch, sto_charg, load_h


def plot_stack(ax, gen_stack, sto_disch, sto_charg, load_h, title):
    """Stacked area chart identico a duck_curve_analysis.py."""
    def sort_key(c):
        return STACK_ORDER.index(c) if c in STACK_ORDER else len(STACK_ORDER)

    hours = sorted(load_h.index.tolist())
    carriers_sorted = sorted(
        list(gen_stack.keys()) + list(sto_disch.keys()),
        key=sort_key,
    )

    bottom = np.zeros(len(hours))
    legend_handles = []

    for carrier in carriers_sorted:
        src  = gen_stack if carrier in gen_stack else sto_disch
        vals = src[carrier].reindex(hours, fill_value=0.0).values
        color = CARRIER_COLORS.get(carrier, CARRIER_COLORS["other"])

        h_c, b_c = add_cycle_close(np.array(hours), bottom)
        _, v_c   = add_cycle_close(np.array(hours), vals)
        ax.fill_between(h_c, b_c, b_c + v_c, color=color, alpha=0.88, linewidth=0)
        bottom += vals
        legend_handles.append(
            mpatches.Patch(facecolor=color, label=CARRIER_LABELS.get(carrier, carrier))
        )

    bottom_neg = np.zeros(len(hours))
    for carrier, charg in sto_charg.items():
        vals  = charg.reindex(hours, fill_value=0.0).values
        color = CARRIER_COLORS.get(carrier, "#bdc3c7")
        h_c, bn_c = add_cycle_close(np.array(hours), bottom_neg)
        _, v_c    = add_cycle_close(np.array(hours), vals)
        ax.fill_between(h_c, bn_c + v_c, bn_c,
                        color=color, alpha=0.5, linewidth=0, hatch="//")
        bottom_neg += vals

    load_vals = load_h.reindex(hours, fill_value=0.0).values
    h_c, lv_c = add_cycle_close(np.array(hours), load_vals)
    ax.plot(h_c, lv_c, color="black", linewidth=2.2, zorder=5,
            label=f"Load  ({load_vals.mean():.0f} GW avg)")
    legend_handles.append(mpatches.Patch(facecolor="black", label="Load"))

    all_x = list(hours) + [hours[0] + 24]
    ax.set_xticks(all_x)
    ax.set_xticklabels([f"{h % 24:02d}:00" for h in all_x], fontsize=8, rotation=45)
    ax.set_xlim(all_x[0], all_x[-1])
    ax.set_ylabel("GW", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")
    ax.axhline(0, color="black", linewidth=0.5)
    return legend_handles


def plot_heatmap(n, idx, bus_set, spec, region_label, code):
    """
    Heatmap mese × ora CST per una famiglia di carrier.
    Funziona per generator carriers; hydro reservoir è in storage_units.
    """
    # Generators
    gen_carriers = [c for c in spec["carriers"]
                    if c in n.generators.carrier.values]
    gen_cols = n.generators[
        (n.generators.carrier.isin(gen_carriers)) &
        (n.generators.bus.isin(bus_set))
    ].index
    gen_ts = (
        n.generators_t.p.reindex(columns=gen_cols, fill_value=0.0).sum(axis=1)
        if not gen_cols.empty else pd.Series(0.0, index=idx)
    )

    # Storage units (hydro reservoir, PHS …)
    sto_carriers = [c for c in spec["carriers"]
                    if c in n.storage_units.carrier.values]
    sto_cols = n.storage_units[
        (n.storage_units.carrier.isin(sto_carriers)) &
        (n.storage_units.bus.isin(bus_set))
    ].index
    sto_ts = (
        n.storage_units_t.p.reindex(columns=sto_cols, fill_value=0.0)
        .clip(lower=0).sum(axis=1)           # solo discharge
        if not sto_cols.empty else pd.Series(0.0, index=idx)
    )

    combined_ts = gen_ts + sto_ts

    local_h = cst_hour(idx)
    local_m = cst_month(idx)

    pivot = (
        pd.DataFrame({
            "gw":    combined_ts.values / 1e3,
            "month": local_m,
            "hour":  local_h,
        })
        .groupby(["month", "hour"])["gw"].mean()
        .unstack("hour")
        .reindex(index=range(1, 13))          # tutti i 12 mesi, anche vuoti
    )

    hour_cols = sorted(pivot.columns.tolist())
    pivot     = pivot.reindex(hour_cols, axis=1)
    pivot     = pivot.reindex(hour_cols, axis=1)
    xlabels   = [f"{h:02d}:00" for h in hour_cols]
    vmax      = np.nanmax(pivot.values) if np.nanmax(pivot.values) > 0 else 1.0

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(
        pivot.values,
        aspect="auto", cmap=spec["cmap"],
        vmin=0, vmax=vmax, origin="upper",
    )
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=spec["cbar_label"])
    ax.set_xticks(range(len(hour_cols)))
    ax.set_xticklabels(xlabels, rotation=45, fontsize=9)
    ax.set_yticks(range(12))
    ax.set_yticklabels(MONTH_LABELS, fontsize=10)
    ax.set_xlabel("Hour of day  (CST)", fontsize=11)
    ax.set_ylabel("Month", fontsize=11)
    ax.set_title(
        spec["title_tpl"].format(region_label=region_label, code=code),
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    return fig


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
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

    print(f"Loading: {args.network}")
    n   = pypsa.Network(args.network)
    idx = n.snapshots
    dt  = (idx[1] - idx[0]).total_seconds() / 3600.0 if len(idx) > 1 else 1.0
    print(f"  Buses: {len(n.buses)} | Snapshots: {len(idx)} | Resolution: {dt}h")
    print(f"  CST hours in data: {sorted(set(cst_hour(idx).tolist()))}")

    # Province attribute — fatto una volta, riusato ovunque
    n.buses["province"] = n.buses.index.map(bus_to_admin1)

    local_months = cst_month(idx)

    # ── Loop over regions ─────────────────────────────────────────────────────
    for code, (region_label, prov_nums) in REGIONAL_GROUPS.items():
        admin1_ids = [f"CN.{n_}" for n_ in prov_nums]
        bus_set    = region_buses(n, admin1_ids)

        n_buses = len(bus_set)
        print(f"\n{'─'*60}")
        print(f"Region: {code}  ({region_label})  |  {n_buses} buses")
        if n_buses == 0:
            print(f"  [SKIP] No buses found for admin1 IDs: {admin1_ids}")
            continue

        # ── Plot 1: Annual dispatch stack ─────────────────────────────────
        gs, sd, sc, lh = build_dispatch_stack(n, idx, bus_set)
        fig, ax = plt.subplots(figsize=(13, 7))
        handles = plot_stack(
            ax, gs, sd, sc, lh,
            title=f"Dispatch stack — CN2020  |  {region_label} [{code}]",
        )
        ax.set_xlabel("Hour of day  (CST)", fontsize=10)
        ax.legend(handles=handles, loc="upper left", fontsize=8,
                  framealpha=0.92, ncol=2)
        plt.tight_layout()
        save(fig, f"dispatch_stack_annual_{code}.png")

        # ── Plot 2: Seasonal dispatch stack ───────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(16, 11))
        all_handles = None
        for i, (season_name, months) in enumerate(SEASONS.items()):
            mask = pd.Series(local_months, index=idx).isin(months)
            gs_s, sd_s, sc_s, lh_s = build_dispatch_stack(n, idx, bus_set, mask=mask)
            ax = axes.flatten()[i]
            handles = plot_stack(ax, gs_s, sd_s, sc_s, lh_s, title=season_name)
            ax.set_xlabel("Hour of day  (CST)", fontsize=9)
            n_days = int(mask.sum() * dt / 24)
            ax.text(0.98, 0.02, f"{n_days} days",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=8, color="gray")
            all_handles = handles

        fig.legend(handles=all_handles, loc="lower center", fontsize=8,
                   ncol=5, bbox_to_anchor=(0.5, -0.04), framealpha=0.92)
        fig.suptitle(
            f"Dispatch stack by season — CN2020  |  {region_label} [{code}]",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        save(fig, f"dispatch_stack_seasonal_{code}.png")

        # ── Plot 3: Duck curve (Load vs Net load) ─────────────────────────
        solar_cols = n.generators[
            (n.generators.carrier == "solar") &
            (n.generators.bus.isin(bus_set))
        ].index
        wind_cols = n.generators[
            (n.generators.carrier.isin(WIND_CARRIERS)) &
            (n.generators.bus.isin(bus_set))
        ].index
        region_load_cols = n.loads[n.loads.bus.isin(bus_set)].index

        solar_ts = n.generators_t.p.reindex(columns=solar_cols, fill_value=0.0).sum(axis=1)
        wind_ts  = n.generators_t.p.reindex(columns=wind_cols,  fill_value=0.0).sum(axis=1)
        load_ts  = (
            n.loads_t.p.reindex(columns=region_load_cols, fill_value=0.0).sum(axis=1)
            if not n.loads_t.p.empty
            else n.loads_t.p_set.reindex(columns=region_load_cols, fill_value=0.0).sum(axis=1)
        )

        load_h_s  = hourly_mean_gw(load_ts,  idx)
        solar_h   = hourly_mean_gw(solar_ts, idx)
        wind_h    = hourly_mean_gw(wind_ts,  idx)
        hours     = sorted(load_h_s.index.tolist())
        netload_h = (load_h_s - solar_h - wind_h).reindex(hours)

        fig, ax = plt.subplots(figsize=(11, 6))
        for series, color, label, lw, ls in [
            (load_h_s,  "#333333", "Load",  2.0, "-"),
            (solar_h,   "#f4a620", "Solar", 1.8, "-"),
            (wind_h,    "#1f77b4", "Wind",  1.8, "-"),
            (netload_h, "#d62728", "Net load  (Load − Solar − Wind)", 2.5, "--"),
        ]:
            vals = series.reindex(hours, fill_value=0.0).values
            h_c, v_c = add_cycle_close(np.array(hours), vals)
            ax.plot(h_c, v_c, color=color, linewidth=lw, linestyle=ls, label=label)

        min_h = netload_h.idxmin()
        max_h = netload_h.idxmax()
        for h, lbl, off in [(min_h, "Min net load", (1, -30)),
                             (max_h, "Max net load", (-4,  20))]:
            val = netload_h[h]
            ax.annotate(
                f"{lbl}\n{val:.0f} GW  ({h:02d}:00)",
                xy=(h, val), xytext=(h + off[0], val + off[1]),
                fontsize=8, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8),
            )

        all_x = hours + [hours[0] + 24]
        ax.set_xticks(all_x)
        ax.set_xticklabels([f"{h % 24:02d}:00" for h in all_x], fontsize=9, rotation=45)
        ax.set_xlim(all_x[0], all_x[-1])
        ax.set_xlabel("Hour of day  (CST)", fontsize=10)
        ax.set_ylabel("GW", fontsize=10)
        ax.set_title(
            f"Duck curve — CN2020  |  {region_label} [{code}]  (annual average, CST)",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save(fig, f"duck_curve_annual_{code}.png")

        # ── Plots 4–6: Heatmaps (solar, wind, hydro) ──────────────────────
        for hmap_key, spec in HEATMAP_SPECS.items():
            fig = plot_heatmap(n, idx, bus_set, spec, region_label, code)
            save(fig, f"{hmap_key}_heatmap_{code}.png")

    print(f"\n{'='*60}")
    print(f"Done. All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()