"""
capacity_generation_validation.py
==================================
Produce tre grafici per il documento di validazione:

  1. capacity_comparison.png
     Capacità installata per carrier: modello CN2020_Admin2_noCO2 vs Ember 2020 (GW)

  2. generation_comparison.png
     Mix di generazione: modello CN2020_Admin2_noCO2 vs Ember 2020 (TWh)

  3. load_scenarios_comparison.png
     Carico totale modello vs riferimento per i tre scenari Admin2 (2020, 2025, 2060)

Dati di riferimento Ember (estratti da Ember Yearly Electricity Data CSV, China, 2020):
  Capacity (GW): Coal 1048.3, Gas 101.9, Solar 253.9, Wind 282.0,
                 Nuclear 49.9, Hydro 338.7, Other Fossil 9.5
  Generation (TWh): Coal 4921.9, Gas 252.5, Solar 261.1, Wind 466.5,
                    Nuclear 366.2, Hydro 1321.7
  Total demand 2020: 7762.0 TWh

Uso:
  python analysis/validation/capacity_generation_validation.py
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

NETWORK_CN2020_NOCO2 = "results/networks/CN2020_Admin2_noCO2/elec_s_250_ec_lcopt_6h.nc"
OUTPUT_DIR           = "analysis/capacity/output"

# ── EMBER 2020 REFERENCE DATA ─────────────────────────────────────────────────

# Capacity (GW) — Ember Yearly Electricity Data, China 2020
EMBER_CAP_2020 = {
    "coal":    1048.3,
    "gas":      101.9,
    "solar":    253.9,
    "wind":     282.0,
    "nuclear":   49.9,
    "hydro":    338.7,
    "oil":        9.5,
}

# Generation (TWh) — Ember Yearly Electricity Data, China 2020
EMBER_GEN_2020 = {
    "coal":    4921.9,
    "gas":      252.5,
    "solar":    261.1,
    "wind":     466.5,
    "nuclear":  366.2,
    "hydro":   1321.7,
}

# Load reference (TWh) — Ember demand + Provincial_Load CSV (Ivan)
LOAD_REF = {
    "CN2020\nAdmin2": {"model": 6462.4, "ref": 7588.0,  "ref_label": "Provincial Load CSV"},
    "CN2025\nAdmin2": {"model": 8388.4, "ref": 9849.0,  "ref_label": "Provincial Load CSV"},
    "CN2060\nAdmin2": {"model": 13884.2,"ref": 16301.0, "ref_label": "Provincial Load CSV"},
}

# PyPSA carrier -> Ember category mapping
CARRIER_MAP = {
    "coal":       "coal",
    "lignite":    "coal",
    "CCGT":       "gas",
    "oil":        "oil",
    "ror":        "hydro",
    "onwind":     "wind",
    "offwind-ac": "wind",
    "offwind-dc": "wind",
    "solar":      "solar",
    "nuclear":    "nuclear",
}

# Colors per carrier (consistent across plots)
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


def load_model_metrics(network_file):
    """Extract capacity and generation from a PyPSA network."""
    n = pypsa.Network(network_file)
    w = n.snapshot_weightings.generators

    # Capacity from generators
    cap = n.generators.groupby("carrier")["p_nom"].sum() / 1e3
    cap = cap[cap.index != "load shedding"]

    # Capacity from storage units (hydro, PHS)
    stor_cap = n.storage_units.groupby("carrier")["p_nom"].sum() / 1e3

    # Generation from generators
    gen = (
        n.generators_t.p.multiply(w, axis=0).sum()
        .groupby(n.generators.carrier).sum() / 1e6
    )
    gen = gen[gen.index != "load shedding"]

    # Generation from storage units
    if not n.storage_units_t.p.empty:
        stor_gen = (
            n.storage_units_t.p.multiply(w, axis=0)
            .clip(lower=0).sum()
            .groupby(n.storage_units.carrier).sum() / 1e6
        )
    else:
        stor_gen = pd.Series(dtype=float)

    return cap, stor_cap, gen, stor_gen


def aggregate_to_ember(cap, stor_cap, gen, stor_gen):
    """Aggregate PyPSA carriers to Ember categories."""
    agg_cap = {}
    agg_gen = {}

    for carrier, cat in CARRIER_MAP.items():
        agg_cap[cat] = agg_cap.get(cat, 0) + cap.get(carrier, 0)
        agg_gen[cat] = agg_gen.get(cat, 0) + gen.get(carrier, 0)

    # Add hydro from storage units
    agg_cap["hydro"] = agg_cap.get("hydro", 0) + stor_cap.get("hydro", 0)
    agg_gen["hydro"] = agg_gen.get("hydro", 0) + stor_gen.get("hydro", 0)

    return pd.Series(agg_cap), pd.Series(agg_gen)


def plot_capacity_comparison(model_cap, output_dir):
    """Bar chart: model vs Ember 2020 installed capacity by carrier (GW)."""
    carriers = sorted(set(list(model_cap.index) + list(EMBER_CAP_2020.keys())))
    carriers = [c for c in carriers if c in CARRIER_COLORS]

    x = np.arange(len(carriers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    model_vals = [model_cap.get(c, 0) for c in carriers]
    ember_vals = [EMBER_CAP_2020.get(c, 0) for c in carriers]
    colors     = [CARRIER_COLORS.get(c, "#888888") for c in carriers]

    bars1 = ax.bar(x - width/2, model_vals, width,
                   label="Model (CN2020 Admin2, no CO2 limit)",
                   color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, ember_vals, width,
                   label="Ember 2020 (reference)",
                   color=colors, alpha=0.40, edgecolor="#333333",
                   linewidth=0.8, hatch="//")

    # Error % annotation
    for i, (mv, ev) in enumerate(zip(model_vals, ember_vals)):
        if ev > 0:
            err = (mv - ev) / ev * 100
            ax.text(x[i], max(mv, ev) + 8, f"{err:+.0f}%",
                    ha="center", fontsize=8, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in carriers], fontsize=11)
    ax.set_ylabel("Installed capacity (GW)", fontsize=12)
    ax.set_title("Installed capacity by carrier — Model vs Ember 2020\n"
                 "CN2020 Admin2, no CO2 limit", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))

    plt.tight_layout()
    out = os.path.join(output_dir, "capacity_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_generation_comparison(model_gen, output_dir):
    """Bar chart: model vs Ember 2020 generation by carrier (TWh)."""
    carriers = sorted(set(list(model_gen.index) + list(EMBER_GEN_2020.keys())))
    carriers = [c for c in carriers if c in CARRIER_COLORS]

    x = np.arange(len(carriers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    model_vals = [model_gen.get(c, 0) for c in carriers]
    ember_vals = [EMBER_GEN_2020.get(c, 0) for c in carriers]
    colors     = [CARRIER_COLORS.get(c, "#888888") for c in carriers]

    ax.bar(x - width/2, model_vals, width,
           label="Model (CN2020 Admin2, no CO2 limit)",
           color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.bar(x + width/2, ember_vals, width,
           label="Ember 2020 (reference)",
           color=colors, alpha=0.40, edgecolor="#333333",
           linewidth=0.8, hatch="//")

    # Error % annotation
    for i, (mv, ev) in enumerate(zip(model_vals, ember_vals)):
        if ev > 0:
            err = (mv - ev) / ev * 100
            ax.text(x[i], max(mv, ev) + 30, f"{err:+.0f}%",
                    ha="center", fontsize=8, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in carriers], fontsize=11)
    ax.set_ylabel("Electricity generation (TWh)", fontsize=12)
    ax.set_title("Generation mix by carrier — Model vs Ember 2020\n"
                 "CN2020 Admin2, no CO2 limit", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    out = os.path.join(output_dir, "generation_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_load_scenarios(output_dir):
    """Grouped bar chart: model load vs reference for three Admin2 scenarios."""
    scenarios = list(LOAD_REF.keys())
    model_vals = [LOAD_REF[s]["model"] for s in scenarios]
    ref_vals   = [LOAD_REF[s]["ref"]   for s in scenarios]
    errors     = [(m - r) / r * 100 for m, r in zip(model_vals, ref_vals)]

    x = np.arange(len(scenarios))
    width = 0.35
    colors = ["#4393c3", "#f4a620", "#d73027"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (mv, rv, col) in enumerate(zip(model_vals, ref_vals, colors)):
        ax.bar(x[i] - width/2, mv, width, color=col, alpha=0.85,
               edgecolor="white", label=f"Model {scenarios[i].replace(chr(10), ' ')}")
        ax.bar(x[i] + width/2, rv, width, color=col, alpha=0.35,
               edgecolor="#333333", linewidth=0.8, hatch="//")
        err = (mv - rv) / rv * 100
        ax.text(x[i], max(mv, rv) + 150, f"{err:+.1f}%",
                ha="center", fontsize=10, color="#333333", fontweight="bold")

    # Add reference lines
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("\n", " ") for s in scenarios], fontsize=11)
    ax.set_ylabel("Total electricity demand (TWh)", fontsize=12)
    ax.set_title("Total electricity load — Model vs reference\n"
                 "Admin2 scenarios (2020, 2025, 2060)", fontsize=12)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#888888", alpha=0.85, label="Model"),
        Patch(facecolor="#888888", alpha=0.35, hatch="//",
              edgecolor="#333333", label="Reference (Provincial Load CSV)"),
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    plt.tight_layout()
    out = os.path.join(output_dir, "load_scenarios_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading network: {NETWORK_CN2020_NOCO2}")
    cap, stor_cap, gen, stor_gen = load_model_metrics(NETWORK_CN2020_NOCO2)

    print("Aggregating to Ember categories...")
    model_cap, model_gen = aggregate_to_ember(cap, stor_cap, gen, stor_gen)

    print("\nModel capacity (GW):")
    print(model_cap.sort_values(ascending=False).to_string())
    print("\nEmber 2020 capacity (GW):")
    print(pd.Series(EMBER_CAP_2020).sort_values(ascending=False).to_string())

    print("\nModel generation (TWh):")
    print(model_gen.sort_values(ascending=False).to_string())
    print("\nEmber 2020 generation (TWh):")
    print(pd.Series(EMBER_GEN_2020).sort_values(ascending=False).to_string())

    print("\nProducing plots...")
    plot_capacity_comparison(model_cap, OUTPUT_DIR)
    plot_generation_comparison(model_gen, OUTPUT_DIR)
    plot_load_scenarios(OUTPUT_DIR)

    print("\nDone.")
