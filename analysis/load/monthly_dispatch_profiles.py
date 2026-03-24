"""
monthly_dispatch_profiles.py
=============================
Profili mensili di generazione per carrier per i tre scenari Admin2 noCO2:
  - CN2020_Admin2_noCO2
  - CN2025_Admin2_noCO2
  - CN2060_Admin2_noCO2

Produce:
  1. monthly_dispatch_stacked.png
     Grafico a barre impilate mensili per carrier, un pannello per scenario.
     Mostra come cambia il mix di generazione mese per mese.

  2. monthly_dispatch_lines.png
     Profili mensili per carrier (linee), un pannello per scenario.
     Mostra la stagionalita' di ogni fonte energetica.

  3. monthly_dispatch_normalized.png
     Come monthly_dispatch_stacked.png ma normalizzato al 100% per mese.
     Mostra la quota percentuale di ogni carrier mese per mese.

Uso:
  python analysis/load/monthly_dispatch_profiles.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pypsa
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── PARAMETERS ────────────────────────────────────────────────────────────────

OUTPUT_DIR = "analysis/load/output"

SCENARIOS = {
    "CN2020 noCO2": {
        "file":  "results/networks/CN2020_Admin2_noCO2/elec_s_250_ec_lcopt_6h.nc",
        "color": "#1f77b4",
    },
    "CN2025 noCO2": {
        "file":  "results/networks/CN2025_Admin2_noCO2/elec_s_250_ec_lcopt_6h.nc",
        "color": "#ff7f0e",
    },
    "CN2060 noCO2": {
        "file":  "results/networks/CN2060_Admin2_noCO2/elec_s_250_ec_lcopt_6h.nc",
        "color": "#2ca02c",
    },
}

CARRIER_MAP = {
    "coal":       "Coal",
    "lignite":    "Coal",
    "CCGT":       "Gas",
    "oil":        "Oil",
    "ror":        "Hydro",
    "hydro":      "Hydro",
    "PHS":        "Hydro",
    "onwind":     "Wind",
    "offwind-ac": "Wind",
    "offwind-dc": "Wind",
    "solar":      "Solar",
    "nuclear":    "Nuclear",
}

CARRIER_COLORS = {
    "Coal":    "#4a4a4a",
    "Gas":     "#f4a620",
    "Solar":   "#f9d62e",
    "Wind":    "#4393c3",
    "Nuclear": "#7b2d8b",
    "Hydro":   "#1a9850",
    "Oil":     "#d73027",
}

CARRIER_ORDER = ["Nuclear", "Hydro", "Wind", "Solar", "Gas"]

MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

# ─────────────────────────────────────────────────────────────────────────────


def extract_monthly_dispatch(network_file):
    """
    Extract monthly generation by carrier (TWh/month).
    Includes generators and storage_units (hydro dispatch only, clipped to positive).
    Returns a DataFrame: index=month (1-12), columns=carrier categories.
    """
    n = pypsa.Network(network_file)
    w = n.snapshot_weightings.generators

    # Generators dispatch
    gen = n.generators_t.p.multiply(w, axis=0)
    gen_by_carrier = {}
    for col in gen.columns:
        carrier = n.generators.loc[col, "carrier"]
        if carrier == "load shedding":
            continue
        cat = CARRIER_MAP.get(carrier, carrier)
        if cat not in gen_by_carrier:
            gen_by_carrier[cat] = gen[col].copy()
        else:
            gen_by_carrier[cat] += gen[col]

    # Storage units dispatch (positive = discharge = generation)
    if not n.storage_units_t.p.empty:
        stor = n.storage_units_t.p.clip(lower=0).multiply(w, axis=0)
        for col in stor.columns:
            carrier = n.storage_units.loc[col, "carrier"]
            cat = CARRIER_MAP.get(carrier, carrier)
            if cat not in gen_by_carrier:
                gen_by_carrier[cat] = stor[col].copy()
            else:
                gen_by_carrier[cat] += stor[col]

    # Resample to monthly TWh
    monthly = {}
    for cat, series in gen_by_carrier.items():
        monthly[cat] = series.resample("M").sum() / 1e6  # TWh/month

    df = pd.DataFrame(monthly)
    df.index = df.index.month
    df = df.groupby(df.index).sum()  # aggregate by month number

    # Keep only carriers with non-zero generation
    df = df[[c for c in df.columns if df[c].sum() > 0]]

    # Reorder columns
    ordered = [c for c in CARRIER_ORDER if c in df.columns]
    df = df[ordered]

    return df


# ── PLOT 1: Stacked bar per scenario ─────────────────────────────────────────

def plot_stacked(data, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    for ax, (name, df) in zip(axes, data.items()):
        x = np.arange(1, 13)
        bottom = np.zeros(12)

        for carrier in [c for c in CARRIER_ORDER if c in df.columns]:
            vals = df[carrier].reindex(range(1, 13), fill_value=0).values
            ax.bar(x, vals, bottom=bottom,
                   color=CARRIER_COLORS[carrier], label=carrier,
                   alpha=0.88, edgecolor="white", linewidth=0.3)
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(MONTH_LABELS, fontsize=8)
        ax.set_ylabel("Generation (TWh/month)", fontsize=10)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
        ax.legend(fontsize=7, loc="upper left")

    plt.suptitle("Monthly generation by carrier — Admin2 noCO2 scenarios",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(output_dir, "monthly_dispatch_stacked.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


# ── PLOT 2: Lines per carrier ─────────────────────────────────────────────────

def plot_lines(data, output_dir):
    # One row per scenario, one line per carrier
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, (name, df) in zip(axes, data.items()):
        x = range(1, 13)
        for carrier in [c for c in CARRIER_ORDER if c in df.columns]:
            vals = df[carrier].reindex(range(1, 13), fill_value=0).values
            ax.plot(x, vals, color=CARRIER_COLORS[carrier],
                    label=carrier, linewidth=2, marker="o", markersize=4)

        ax.set_xticks(list(x))
        ax.set_xticklabels(MONTH_LABELS, fontsize=8)
        ax.set_ylabel("Generation (TWh/month)", fontsize=10)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    plt.suptitle("Monthly generation by carrier (lines) — Admin2 noCO2 scenarios",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(output_dir, "monthly_dispatch_lines.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


# ── PLOT 3: Normalized stacked ────────────────────────────────────────────────

def plot_normalized(data, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, (name, df) in zip(axes, data.items()):
        x = np.arange(1, 13)

        # Normalize to 100% per month
        df_norm = df.reindex(range(1, 13), fill_value=0)
        totals = df_norm.sum(axis=1)
        df_pct = df_norm.div(totals, axis=0) * 100

        bottom = np.zeros(12)
        for carrier in [c for c in CARRIER_ORDER if c in df_pct.columns]:
            vals = df_pct[carrier].values
            ax.bar(x, vals, bottom=bottom,
                   color=CARRIER_COLORS[carrier], label=carrier,
                   alpha=0.88, edgecolor="white", linewidth=0.3)
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(MONTH_LABELS, fontsize=8)
        ax.set_ylabel("Share of generation (%)", fontsize=10)
        ax.set_ylim(0, 100)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
        ax.legend(fontsize=7, loc="lower left")

    plt.suptitle("Monthly generation mix (%) — Admin2 noCO2 scenarios",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(output_dir, "monthly_dispatch_normalized.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = {}
    for name, meta in SCENARIOS.items():
        print(f"Loading {name}...")
        if not os.path.exists(meta["file"]):
            print(f"  [ERROR] File not found: {meta['file']}")
            continue
        data[name] = extract_monthly_dispatch(meta["file"])
        print(f"  Carriers: {list(data[name].columns)}")
        print(f"  Total generation (TWh): {data[name].sum().sum():.1f}")

    if not data:
        print("[ERROR] No networks loaded.")
        raise SystemExit(1)

    print("\nProducing plots...")
    plot_stacked(data, OUTPUT_DIR)
    plot_lines(data, OUTPUT_DIR)
    plot_normalized(data, OUTPUT_DIR)

    print("\nDone.")
