"""
capacity_scenarios_2025_2060.py
================================
Produce due grafici di capacità installata per carrier:

  1. capacity_2025_comparison.png
     Modello CN2025 Admin2 (Co2L) vs Ember 2024 (proxy 2025)

  2. capacity_2060_comparison.png
     Modello CN2060 Admin2 (Co2L) vs CETO 2024 BCNS (lettura visiva, ±15%)

Nota: i run 2025 e 2060 hanno il vincolo Co2L attivo.
La capacità installata non dipende dal vincolo CO2 (è un input, non un output
dell'ottimizzazione), quindi il confronto di capacità è valido anche con Co2L.
Il confronto di generazione non è invece rappresentativo e richiede run noCO2.

Uso:
  python analysis/validation/capacity_scenarios_2025_2060.py
"""

import os
import pypsa
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── PARAMETERS ────────────────────────────────────────────────────────────────

NETWORK_2025 = "results/networks/CN2025_Admin2/elec_s_250_ec_lcopt_Co2L-6h.nc"
NETWORK_2060 = "results/networks/CN2060_Admin2/elec_s_250_ec_lcopt_Co2L-6h.nc"
OUTPUT_DIR   = "analysis/comparison/output"

# ── REFERENCE DATA ────────────────────────────────────────────────────────────

# Ember 2024 (proxy for 2025) — exact values from CSV
EMBER_CAP_2024 = {
    "coal":    1174.5,
    "gas":      153.0,
    "solar":    887.9,
    "wind":     521.8,
    "nuclear":   60.8,
    "hydro":    385.0,
    "oil":       15.4,
}

# CETO 2024 BCNS scenario at 2060 — read graphically from Figure 5-3
# Uncertainty approximately ±15%
CETO_CAP_2060 = {
    "coal":    200.0,
    "gas":     100.0,
    "solar":  5500.0,
    "wind":   3500.0,
    "nuclear": 400.0,
    "hydro":   600.0,
}

# PyPSA carrier -> reference category mapping
CARRIER_MAP = {
    "coal":       "coal",
    "lignite":    "coal",
    "CCGT":       "gas",
    "oil":        "oil",
    "ror":        "hydro",
    "hydro":      "hydro",
    "PHS":        "hydro",
    "onwind":     "wind",
    "offwind-ac": "wind",
    "offwind-dc": "wind",
    "solar":      "solar",
    "nuclear":    "nuclear",
}

CARRIER_COLORS = {
    "coal":    "#4a4a4a",
    "gas":     "#f4a620",
    "solar":   "#f9d62e",
    "wind":    "#4393c3",
    "nuclear": "#7b2d8b",
    "hydro":   "#1a9850",
    "oil":     "#d73027",
}

# ─────────────────────────────────────────────────────────────────────────────


def load_capacity(network_file):
    """Extract and aggregate installed capacity from a PyPSA network."""
    n = pypsa.Network(network_file)

    cap = n.generators.groupby("carrier")["p_nom"].sum() / 1e3
    cap = cap[cap.index != "load shedding"]
    stor_cap = n.storage_units.groupby("carrier")["p_nom"].sum() / 1e3

    agg = {}
    for carrier, cat in CARRIER_MAP.items():
        val = cap.get(carrier, 0) + stor_cap.get(carrier, 0)
        agg[cat] = agg.get(cat, 0) + val

    return pd.Series(agg)


def plot_capacity(model_cap, ref_cap, ref_label, title, output_file,
                  uncertainty_pct=None):
    """Bar chart: model vs reference capacity by carrier."""
    carriers = sorted(set(list(model_cap.index) + list(ref_cap.keys())))
    carriers = [c for c in carriers if c in CARRIER_COLORS]

    x = np.arange(len(carriers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))

    model_vals = [model_cap.get(c, 0) for c in carriers]
    ref_vals   = [ref_cap.get(c, 0)   for c in carriers]
    colors     = [CARRIER_COLORS[c]   for c in carriers]

    ax.bar(x - width/2, model_vals, width,
           label="Model", color=colors, alpha=0.85,
           edgecolor="white", linewidth=0.5)

    bars_ref = ax.bar(x + width/2, ref_vals, width,
                      label=ref_label, color=colors, alpha=0.40,
                      edgecolor="#333333", linewidth=0.8, hatch="//")

    # Uncertainty bars for CETO (visual reading error)
    if uncertainty_pct is not None:
        for i, rv in enumerate(ref_vals):
            if rv > 0:
                err = rv * uncertainty_pct / 100
                ax.errorbar(x[i] + width/2, rv, yerr=err,
                            fmt="none", color="#333333",
                            capsize=4, linewidth=1.2)

    # Error % annotation
    for i, (mv, rv) in enumerate(zip(model_vals, ref_vals)):
        if rv > 0:
            err = (mv - rv) / rv * 100
            ypos = max(mv, rv) + max(ref_vals) * 0.03
            ax.text(x[i], ypos, f"{err:+.0f}%",
                    ha="center", fontsize=8, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in carriers], fontsize=11)
    ax.set_ylabel("Installed capacity (GW)", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    plt.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_file}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # CN2025 vs Ember 2024
    print(f"Loading: {NETWORK_2025}")
    cap_2025 = load_capacity(NETWORK_2025)
    print("CN2025 model capacity (GW):")
    print(cap_2025.sort_values(ascending=False).to_string())
    print()

    plot_capacity(
        model_cap=cap_2025,
        ref_cap=EMBER_CAP_2024,
        ref_label="Ember 2024 (proxy for 2025)",
        title=("Installed capacity by carrier — Model vs Ember 2024\n"
               "CN2025 Admin2 (Co2L-6h) | Note: generation comparison"
               " requires noCO2 run"),
        output_file=os.path.join(OUTPUT_DIR, "capacity_2025_comparison.png"),
    )

    # CN2060 vs CETO 2024 BCNS
    print(f"Loading: {NETWORK_2060}")
    cap_2060 = load_capacity(NETWORK_2060)
    print("CN2060 model capacity (GW):")
    print(cap_2060.sort_values(ascending=False).to_string())
    print()

    plot_capacity(
        model_cap=cap_2060,
        ref_cap=CETO_CAP_2060,
        ref_label="CETO 2024 BCNS (visual reading, ±15%)",
        title=("Installed capacity by carrier — Model vs CETO 2024 BCNS\n"
               "CN2060 Admin2 (Co2L-6h) | Note: generation comparison"
               " requires noCO2 run"),
        output_file=os.path.join(OUTPUT_DIR, "capacity_2060_comparison.png"),
        uncertainty_pct=15,
    )

    print("\nDone.")
