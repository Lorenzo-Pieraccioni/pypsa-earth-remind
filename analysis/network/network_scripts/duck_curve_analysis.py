"""
duck_curve_analysis.py
======================
Duck curve e dispatch stack — rete PyPSA-Earth CN2020.

Con risoluzione 3h e UTC+8, i dati esistono a 8 ore per giorno:
[2, 5, 8, 11, 14, 17, 20, 23] CST.
Il plot usa solo queste 8 ore — nessun riempimento con zero
per evitare il pattern a dente di sega.

Tutti i tempi in China Standard Time (CST = UTC+8).

Output:
  dispatch_stack_annual.png    — load + fonti impilate, media annuale
  dispatch_stack_seasonal.png  — stacked dispatch per stagione (4 subplot)
  duck_curve_annual.png        — Load / Net load per ora
  solar_heatmap.png            — heatmap mese x ora generazione solare

Uso:
  PYPSA_OUTPUT_DIR="analysis/network/duck_curve" python analysis/network/duck_curve_analysis.py results/networks/geospatial_v2_approccio1/CN2020_Admin2/elec_s_250_ec_lcopt_3h.nc
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

OUTPUT_DIR   = os.path.join(os.environ.get("PYPSA_OUTPUT_DIR", "analysis/network/output"), "duck_curve_analysis")
UTC_OFFSET   = 8   # China Standard Time (CST)

SEASONS = {
    "Winter (Dec–Feb)": [12, 1,  2],
    "Spring (Mar–May)": [3,  4,  5],
    "Summer (Jun–Aug)": [6,  7,  8],
    "Autumn (Sep–Nov)": [9,  10, 11],
}

CARRIER_COLORS = {
    "nuclear":     "#e75480",
    "coal":        "#4d4d4d",
    "lignite":     "#9e5a00",
    "oil":         "#2c3e50",
    "gas":         "#f1948a",
    "OCGT":        "#e59866",
    "CCGT":        "#d35400",
    "hydro":       "#2980b9",
    "ror":         "#5dade2",
    "onwind":      "#1f77b4",
    "offwind-ac":  "#aec6cf",
    "offwind-dc":  "#7fb3d3",
    "solar":       "#f4a620",
    "biomass":     "#27ae60",
    "PHS":         "#1abc9c",
    "battery":     "#2ecc71",
    "other":       "#bdc3c7",
}

STACK_ORDER = [
    "nuclear", "coal", "lignite", "oil", "gas", "OCGT", "CCGT",
    "biomass", "hydro", "ror", "onwind", "offwind-ac", "offwind-dc",
    "solar", "PHS", "battery",
]

CARRIER_LABELS = {
    "offwind-ac":  "offshore wind AC",
    "offwind-dc":  "offshore wind DC",
    "onwind":      "onshore wind",
    "ror":         "run-of-river",
    "PHS":         "pumped hydro",
}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def save(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def cst_hour(idx):
    """Ora locale CST (UTC+8) per ogni snapshot."""
    return (idx.hour + UTC_OFFSET) % 24


def cst_month(idx):
    """Mese locale CST."""
    return (idx + pd.Timedelta(hours=UTC_OFFSET)).month


def hourly_mean_gw(ts, idx, mask=None):
    """
    Media per ora CST di una Series in MW → GW.
    Restituisce Series indexata sulle ore realmente presenti nel dataset
    (tipicamente 8 ore con risoluzione 3h).
    """
    s = ts[mask].values if mask is not None else ts.values
    h = cst_hour(idx[mask] if mask is not None else idx)
    return pd.Series(s, index=h).groupby(level=0).mean() / 1e3


def add_cycle_close(hours, values):
    """
    Aggiunge un punto finale = primo punto + 24 per chiudere visivamente
    il ciclo giornaliero senza toccare lo zero.
    """
    return (
        np.append(hours,  hours[0]  + 24),
        np.append(values, values[0]),
    )


def build_dispatch_stack(n, idx, mask=None):
    """
    Costruisce il dispatch per ora CST, aggregato per carrier.
    Restituisce:
      gen_stack : dict carrier → Series GW (8 ore reali)
      sto_disch : dict carrier → Series GW discharge
      sto_charg : dict carrier → Series GW charge (negativo)
      load_h    : Series GW
    """
    gen_stack = {}
    for carrier in n.generators.carrier.unique():
        gens = n.generators[n.generators.carrier == carrier].index
        p_ts = n.generators_t.p.reindex(columns=gens, fill_value=0.0).sum(axis=1)
        avg  = hourly_mean_gw(p_ts, idx, mask)
        if avg.sum() > 0.01:
            gen_stack[carrier] = avg

    sto_disch, sto_charg = {}, {}
    for carrier in n.storage_units.carrier.unique():
        sus  = n.storage_units[n.storage_units.carrier == carrier].index
        p_ts = n.storage_units_t.p.reindex(columns=sus, fill_value=0.0).sum(axis=1)
        avg  = hourly_mean_gw(p_ts, idx, mask)
        d = avg.clip(lower=0)
        c = avg.clip(upper=0)
        if d.sum() > 0.01:
            sto_disch[carrier] = d
        if c.abs().sum() > 0.01:
            sto_charg[carrier] = c

    load_ts = (n.loads_t.p.sum(axis=1) if not n.loads_t.p.empty
               else n.loads_t.p_set.sum(axis=1))
    load_h = hourly_mean_gw(load_ts, idx, mask)

    return gen_stack, sto_disch, sto_charg, load_h


def plot_stack(ax, gen_stack, sto_disch, sto_charg, load_h, title):
    """
    Stacked area: fonti impilate dal basso, linea nera = load.
    X-axis = ore CST reali (8 punti con 3h resolution).
    """
    # Ordine stack
    def sort_key(c):
        return STACK_ORDER.index(c) if c in STACK_ORDER else len(STACK_ORDER)

    hours = sorted(load_h.index.tolist())

    carriers_sorted = sorted(
        list(gen_stack.keys()) + list(sto_disch.keys()),
        key=sort_key
    )

    bottom = np.zeros(len(hours))
    legend_handles = []

    for carrier in carriers_sorted:
        src  = gen_stack if carrier in gen_stack else sto_disch
        vals = src[carrier].reindex(hours, fill_value=0.0).values
        color = CARRIER_COLORS.get(carrier, CARRIER_COLORS.get("other", "#bdc3c7"))

        h_closed, b_closed = add_cycle_close(np.array(hours), bottom)
        _, v_closed         = add_cycle_close(np.array(hours), vals)

        ax.fill_between(h_closed, b_closed, b_closed + v_closed,
                        color=color, alpha=0.88, linewidth=0)
        bottom += vals

        label = CARRIER_LABELS.get(carrier, carrier)
        legend_handles.append(
                mpatches.Patch(facecolor=color,
                               label=label)
        )

    # Storage charging (area negativa)
    bottom_neg = np.zeros(len(hours))
    for carrier, charg in sto_charg.items():
        vals  = charg.reindex(hours, fill_value=0.0).values
        color = CARRIER_COLORS.get(carrier, "#bdc3c7")
        h_c, bn_c = add_cycle_close(np.array(hours), bottom_neg)
        _, v_c     = add_cycle_close(np.array(hours), vals)
        ax.fill_between(h_c, bn_c + v_c, bn_c,
                        color=color, alpha=0.5, linewidth=0, hatch="//")
        bottom_neg += vals

    # Load line
    load_vals = load_h.reindex(hours, fill_value=0.0).values
    h_c, lv_c = add_cycle_close(np.array(hours), load_vals)
    ax.plot(h_c, lv_c, color="black", linewidth=2.2, zorder=5,
            label=f"Load  ({load_vals.mean():.0f} GW avg)")
    legend_handles.append(
        mpatches.Patch(facecolor="black",
                       label="Load")
    )

    # Asse x: le ore reali + la chiusura del ciclo
    all_x = list(hours) + [hours[0] + 24]
    ax.set_xticks(all_x)
    ax.set_xticklabels([f"{h % 24:02d}:00" for h in all_x], fontsize=8, rotation=45)
    ax.set_xlim(all_x[0], all_x[-1])
    ax.set_ylabel("GW", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")
    ax.axhline(0, color="black", linewidth=0.5)

    return legend_handles


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
    print(f"  Generator carriers: {sorted(n.generators.carrier.unique().tolist())}")
    print(f"  Storage carriers:   {sorted(n.storage_units.carrier.unique().tolist())}")

    local_months = cst_month(idx)

    # ── Plot 1: Annual dispatch stack ─────────────────────────────────────────
    gs, sd, sc, lh = build_dispatch_stack(n, idx)

    fig, ax = plt.subplots(figsize=(13, 7))
    handles = plot_stack(ax, gs, sd, sc, lh,
        title="Dispatch stack — CN2020")
    ax.set_xlabel("Hour of day", fontsize=10)
    ax.legend(handles=handles, loc="upper left", fontsize=8,
              framealpha=0.92, ncol=2)
    plt.tight_layout()
    save(fig, "dispatch_stack_annual.png")

    # ── Plot 2: Seasonal dispatch stack ───────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    all_handles = None
    for i, (season_name, months) in enumerate(SEASONS.items()):
        mask = pd.Series(local_months, index=idx).isin(months)
        gs_s, sd_s, sc_s, lh_s = build_dispatch_stack(n, idx, mask=mask)
        ax = axes.flatten()[i]
        handles = plot_stack(ax, gs_s, sd_s, sc_s, lh_s, title=season_name)
        ax.set_xlabel("Hour of day  (CST)", fontsize=9)
        n_days = int(mask.sum() * dt / 24)
        ax.text(0.98, 0.02, f"{n_days} days",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="gray")
        all_handles = handles   # usa ultima stagione per legenda

    fig.legend(handles=all_handles, loc="lower center", fontsize=8,
               ncol=5, bbox_to_anchor=(0.5, -0.04), framealpha=0.92)
    fig.suptitle("Dispatch stack by season — CN2020",
             fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "dispatch_stack_seasonal.png")

    # ── Plot 3: Duck curve (Load vs Net load) ─────────────────────────────────
    solar_ts = n.generators_t.p.reindex(
        columns=n.generators[n.generators.carrier == "solar"].index,
        fill_value=0.0).sum(axis=1)
    wind_ts = n.generators_t.p.reindex(
        columns=n.generators[n.generators.carrier.isin(
            ["onwind","offwind-ac","offwind-dc"])].index,
        fill_value=0.0).sum(axis=1)
    load_ts = (n.loads_t.p.sum(axis=1) if not n.loads_t.p.empty
               else n.loads_t.p_set.sum(axis=1))

    load_h_s  = hourly_mean_gw(load_ts,  idx)
    solar_h   = hourly_mean_gw(solar_ts, idx)
    wind_h    = hourly_mean_gw(wind_ts,  idx)

    hours = sorted(load_h_s.index.tolist())
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
    for h, lbl, off in [(min_h, "Min net load", (1, -60)),
                         (max_h, "Max net load", (-4, 30))]:
        val = netload_h[h]
        ax.annotate(f"{lbl}\n{val:.0f} GW  ({h:02d}:00)",
                    xy=(h, val), xytext=(h+off[0], val+off[1]),
                    fontsize=8, color="#d62728",
                    arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8))

    all_x = hours + [hours[0] + 24]
    ax.set_xticks(all_x)
    ax.set_xticklabels([f"{h%24:02d}:00" for h in all_x], fontsize=9, rotation=45)
    ax.set_xlim(all_x[0], all_x[-1])
    ax.set_xlabel("Hour of day", fontsize=10)
    ax.set_ylabel("GW", fontsize=10)
    ax.set_title("Duck curve — CN2020  (annual average, CST)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig, "duck_curve_annual.png")

    # ── Plot 4: Solar heatmap (month × CST hour) ──────────────────────────────
    # Usa solo le ore realmente presenti nel dataset — nessun reindex a 24 ore.
    local_h = cst_hour(idx)
    local_m = cst_month(idx)

    pivot = (
        pd.DataFrame({
            "gw":    solar_ts.values / 1e3,
            "month": local_m,
            "hour":  local_h,
        })
        .groupby(["month", "hour"])["gw"].mean()
        .unstack("hour")
    )
    # Ordina le colonne per ora crescente
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    hour_cols   = pivot.columns.tolist()
    xlabels     = [f"{h:02d}:00" for h in hour_cols]
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(
        pivot.values,
        aspect="auto", cmap="YlOrRd",
        vmin=0, vmax=np.nanmax(pivot.values),
        origin="upper",
    )
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02,
                 label="Average solar generation  (GW)")
    ax.set_xticks(range(len(hour_cols)))
    ax.set_xticklabels(xlabels, rotation=45, fontsize=9)
    ax.set_yticks(range(12))
    ax.set_yticklabels(month_labels, fontsize=10)
    ax.set_xlabel("Hour of day", fontsize=11)
    ax.set_ylabel("Month", fontsize=11)
    ax.set_title(
        "Average solar generation by month and hour — CN2020  (national)\n"
        "8 data points per day (3h resolution)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    save(fig, "solar_heatmap.png")

    # ── Plot 5: Wind heatmap (month × CST hour) ───────────────────────────────
    pivot_wind = (
        pd.DataFrame({
            "gw":    wind_ts.values / 1e3,
            "month": local_m,
            "hour":  local_h,
        })
        .groupby(["month", "hour"])["gw"].mean()
        .unstack("hour")
    )
    pivot_wind = pivot_wind.reindex(sorted(pivot_wind.columns), axis=1)

    hour_cols_w = pivot_wind.columns.tolist()
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(
        pivot_wind.values,
        aspect="auto", cmap="Blues",
        vmin=0, vmax=np.nanmax(pivot_wind.values),
        origin="upper",
    )
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02,
                 label="Average wind generation  (GW)")
    ax.set_xticks(range(len(hour_cols_w)))
    ax.set_xticklabels([f"{h:02d}:00" for h in hour_cols_w], rotation=45, fontsize=9)
    ax.set_yticks(range(12))
    ax.set_yticklabels(month_labels, fontsize=10)
    ax.set_xlabel("Hour of day", fontsize=11)
    ax.set_ylabel("Month", fontsize=11)
    ax.set_title(
        "Average wind generation (onshore + offshore) by month and hour — CN2020  (national)\n"
        "8 data points per day (3h resolution)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    save(fig, "wind_heatmap.png")

    # ── Plot 6: Hydro heatmap (month × CST hour) ──────────────────────────────
    # ror è in n.generators; hydro reservoir è in n.storage_units (solo discharge)
    ror_ts = n.generators_t.p.reindex(
        columns=n.generators[n.generators.carrier == "ror"].index,
        fill_value=0.0).sum(axis=1)
    reservoir_ts = n.storage_units_t.p.reindex(
        columns=n.storage_units[n.storage_units.carrier == "hydro"].index,
        fill_value=0.0).clip(lower=0).sum(axis=1)
    hydro_ts = ror_ts + reservoir_ts

    pivot_hydro = (
        pd.DataFrame({
            "gw":    hydro_ts.values / 1e3,
            "month": local_m,
            "hour":  local_h,
        })
        .groupby(["month", "hour"])["gw"].mean()
        .unstack("hour")
    )
    pivot_hydro = pivot_hydro.reindex(sorted(pivot_hydro.columns), axis=1)

    hour_cols_h = pivot_hydro.columns.tolist()
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(
        pivot_hydro.values,
        aspect="auto", cmap="GnBu",
        vmin=0, vmax=np.nanmax(pivot_hydro.values),
        origin="upper",
    )
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02,
                 label="Average hydro generation  (GW)")
    ax.set_xticks(range(len(hour_cols_h)))
    ax.set_xticklabels([f"{h:02d}:00" for h in hour_cols_h], rotation=45, fontsize=9)
    ax.set_yticks(range(12))
    ax.set_yticklabels(month_labels, fontsize=10)
    ax.set_xlabel("Hour of day", fontsize=11)
    ax.set_ylabel("Month", fontsize=11)
    ax.set_title(
        "Average hydro generation (reservoir + run-of-river) by month and hour — CN2020  (national)\n"
        "8 data points per day (3h resolution)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    save(fig, "hydro_heatmap.png")

    print(f"\nDone. All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()