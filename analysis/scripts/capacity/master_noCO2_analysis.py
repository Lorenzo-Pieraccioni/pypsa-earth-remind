"""
master_noCO2_analysis.py
========================
Master script per l'analisi comparativa dei tre scenari Admin2 senza vincolo CO2:
  - CN2020_Admin2
  - CN2025_Admin2
  - CN2060_Admin2

Produce:
  1. report_noCO2.txt              : tabelle numeriche complete
  2. load_noCO2_comparison.png     : carico totale modello vs riferimento
  3. capacity_noCO2_comparison.png : capacità installata per carrier vs riferimenti
  4. generation_noCO2_comparison.png: mix di generazione per carrier vs riferimenti
  5. generation_noCO2_shares.png   : quota % di generazione per carrier
  6. capacity_noCO2_trajectory.png : traiettoria capacità 2020->2025->2060

Dati di riferimento:
  2020: Ember 2020 CSV (dati esatti)
  2025: Ember 2024 CSV (proxy, dati esatti)
  2060: CETO 2024 BCNS (lettura visiva, ±15%)

Uso:
  python analysis/scripts/capacity/master_noCO2_analysis.py
  python analysis/scripts/capacity/master_noCO2_analysis.py --run fix-hydro
  python analysis/scripts/capacity/master_noCO2_analysis.py --run noCO2

I file di input vengono letti da:
  results/networks/{run}/CN2020_Admin2/elec_s_250_ec_lcopt_6h.nc
  results/networks/{run}/CN2025_Admin2/elec_s_250_ec_lcopt_6h.nc
  results/networks/{run}/CN2060_Admin2/elec_s_250_ec_lcopt_6h.nc

I risultati vengono salvati in:
  analysis/output/{run}/capacity/
"""

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import pypsa
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── ARGUMENT PARSING ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Master analysis script for noCO2 scenarios"
)
parser.add_argument(
    "--run",
    default="noCO2",
    help="Name of the run folder under results/networks/ (default: noCO2)",
)
args = parser.parse_args()
RUN = args.run

# ── PARAMETERS ────────────────────────────────────────────────────────────────

NETWORK_DIR = f"results/networks/{RUN}"
OUTPUT_DIR  = f"analysis/output/{RUN}/capacity"
NETFILE     = "elec_s_250_ec_lcopt_6h.nc"

SCENARIOS = {
    "CN2020\nnoCO2": {
        "file":         os.path.join(NETWORK_DIR, "CN2020_Admin2", NETFILE),
        "year":         2020,
        "load_ref_twh": 7588.0,
        "color":        "#1f77b4",
    },
    "CN2025\nnoCO2": {
        "file":         os.path.join(NETWORK_DIR, "CN2025_Admin2", NETFILE),
        "year":         2025,
        "load_ref_twh": 9849.0,
        "color":        "#ff7f0e",
    },
    "CN2060\nnoCO2": {
        "file":         os.path.join(NETWORK_DIR, "CN2060_Admin2", NETFILE),
        "year":         2060,
        "load_ref_twh": 16301.0,
        "color":        "#2ca02c",
    },
}

# ── REFERENCE DATA ────────────────────────────────────────────────────────────

# Ember 2020 — exact values from CSV
REF_CAP = {
    2020: {
        "coal":    1048.3,
        "gas":      101.9,
        "solar":    253.9,
        "wind":     282.0,
        "nuclear":   49.9,
        "hydro":    338.7,
        "oil":        9.5,
    },
    # Ember 2024 — proxy for 2025
    2025: {
        "coal":    1174.5,
        "gas":      153.0,
        "solar":    887.9,
        "wind":     521.8,
        "nuclear":   60.8,
        "hydro":    385.0,
        "oil":       15.4,
    },
    # CETO 2024 BCNS — visual reading ±15%
    2060: {
        "coal":    200.0,
        "gas":     100.0,
        "solar":  5500.0,
        "wind":   3500.0,
        "nuclear": 400.0,
        "hydro":   600.0,
    },
}

REF_GEN = {
    2020: {
        "coal":    4921.9,
        "gas":      252.5,
        "solar":    261.1,
        "wind":     466.5,
        "nuclear":  366.2,
        "hydro":   1321.7,
    },
    # Ember 2024 — proxy for 2025
    2025: {
        "coal":    5827.6,
        "gas":      320.7,
        "solar":    839.0,
        "wind":     997.0,
        "nuclear":  450.9,
        "hydro":   1354.3,
    },
    # CETO 2024 BCNS — visual reading ±15%
    2060: {
        "coal":     500.0,
        "gas":      300.0,
        "solar":   8000.0,
        "wind":    7000.0,
        "nuclear": 2000.0,
        "hydro":   1800.0,
    },
}

REF_LABELS = {
    2020: "Ember 2020",
    2025: "Ember 2024 (proxy)",
    2060: "CETO 2024 BCNS (±15%)",
}

# ── CARRIER MAPPING AND COLORS ────────────────────────────────────────────────

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

MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

# ─────────────────────────────────────────────────────────────────────────────


def extract_metrics(network_file):
    """Extract key quantities from a PyPSA network."""
    n = pypsa.Network(network_file)
    w = n.snapshot_weightings.generators

    load_twh = (n.loads_t.p_set.multiply(w, axis=0)).sum().sum() / 1e6

    # Capacity
    cap = n.generators.groupby("carrier")["p_nom"].sum() / 1e3
    cap = cap[cap.index != "load shedding"]
    stor_cap = n.storage_units.groupby("carrier")["p_nom"].sum() / 1e3

    # Generation from generators
    gen = (
        n.generators_t.p.multiply(w, axis=0).sum()
        .groupby(n.generators.carrier).sum() / 1e6
    )
    gen = gen[gen.index != "load shedding"]

    # Generation from storage units (hydro, PHS)
    if not n.storage_units_t.p.empty:
        stor_gen = (
            n.storage_units_t.p.multiply(w, axis=0)
            .clip(lower=0).sum()
            .groupby(n.storage_units.carrier).sum() / 1e6
        )
    else:
        stor_gen = pd.Series(dtype=float)

    # Aggregate to reference categories
    agg_cap, agg_gen = {}, {}
    for carrier, cat in CARRIER_MAP.items():
        agg_cap[cat] = agg_cap.get(cat, 0) + cap.get(carrier, 0) + stor_cap.get(carrier, 0)
        agg_gen[cat] = agg_gen.get(cat, 0) + gen.get(carrier, 0) + stor_gen.get(carrier, 0)

    # Monthly load
    load_ts = n.loads_t.p_set.sum(axis=1)
    load_monthly = (load_ts * w).resample("M").sum() / 1e6

    gen_total = sum(agg_gen.values())
    balance_err = (gen_total - load_twh) / load_twh * 100 if load_twh > 0 else float("nan")

    return {
        "load_twh":      load_twh,
        "cap":           pd.Series(agg_cap),
        "gen":           pd.Series(agg_gen),
        "gen_total":     gen_total,
        "balance_err":   balance_err,
        "load_monthly":  load_monthly,
        "n_buses":       len(n.buses),
        "n_snapshots":   len(n.snapshots),
    }


# ── PLOTS ─────────────────────────────────────────────────────────────────────

def plot_load_comparison(metrics, output_dir):
    names = list(metrics.keys())
    colors = [SCENARIOS[n]["color"] for n in names]
    model_vals = [metrics[n]["load_twh"] for n in names]
    ref_vals   = [SCENARIOS[n]["load_ref_twh"] for n in names]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (mv, rv, col) in enumerate(zip(model_vals, ref_vals, colors)):
        ax.bar(x[i] - width/2, mv, width, color=col, alpha=0.85, edgecolor="white")
        ax.bar(x[i] + width/2, rv, width, color=col, alpha=0.35,
               edgecolor="#333333", linewidth=0.8, hatch="//")
        err = (mv - rv) / rv * 100
        ax.text(x[i], max(mv, rv) + 200, f"{err:+.1f}%",
                ha="center", fontsize=10, fontweight="bold", color="#333333")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#888888", alpha=0.85, label=f"Model ({RUN})"),
        Patch(facecolor="#888888", alpha=0.35, hatch="//",
              edgecolor="#333333", label="Reference (Provincial Load CSV)"),
    ], fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("\n", " ") for n in names], fontsize=11)
    ax.set_ylabel("Total electricity demand (TWh)", fontsize=12)
    ax.set_title(f"Total electricity demand — Model ({RUN}) vs reference\n"
                 "Admin2 scenarios: 2020, 2025, 2060", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    plt.tight_layout()
    out = os.path.join(output_dir, "load_noCO2_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


def plot_capacity_comparison(metrics, output_dir):
    names = list(metrics.keys())
    carriers = sorted(CARRIER_COLORS.keys())
    years = [SCENARIOS[n]["year"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    for ax, name, year in zip(axes, names, years):
        model_vals = [metrics[name]["cap"].get(c, 0) for c in carriers]
        ref_vals   = [REF_CAP[year].get(c, 0) for c in carriers]
        colors     = [CARRIER_COLORS[c] for c in carriers]
        x = np.arange(len(carriers))
        width = 0.35

        ax.bar(x - width/2, model_vals, width, color=colors, alpha=0.85,
               edgecolor="white", linewidth=0.5, label=f"Model ({RUN})")
        ax.bar(x + width/2, ref_vals, width, color=colors, alpha=0.40,
               edgecolor="#333333", linewidth=0.8, hatch="//",
               label=REF_LABELS[year])

        if year == 2060:
            for i, rv in enumerate(ref_vals):
                if rv > 0:
                    ax.errorbar(x[i] + width/2, rv, yerr=rv * 0.15,
                                fmt="none", color="#333333", capsize=3, linewidth=1)

        for i, (mv, rv) in enumerate(zip(model_vals, ref_vals)):
            if rv > 0:
                err = (mv - rv) / rv * 100
                ypos = max(mv, rv) + max(max(model_vals), max(ref_vals)) * 0.04
                ax.text(x[i], ypos, f"{err:+.0f}%",
                        ha="center", fontsize=7, color="#333333")

        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() for c in carriers],
                           rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Installed capacity (GW)", fontsize=10)
        ax.set_title(f"{name.replace(chr(10), ' ')}\nvs {REF_LABELS[year]}", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    plt.suptitle(f"Installed capacity by carrier — Model ({RUN}) vs reference",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(output_dir, "capacity_noCO2_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


def plot_generation_comparison(metrics, output_dir):
    names = list(metrics.keys())
    carriers = sorted(CARRIER_COLORS.keys())
    years = [SCENARIOS[n]["year"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    for ax, name, year in zip(axes, names, years):
        model_vals = [metrics[name]["gen"].get(c, 0) for c in carriers]
        ref_vals   = [REF_GEN[year].get(c, 0) for c in carriers]
        colors     = [CARRIER_COLORS[c] for c in carriers]
        x = np.arange(len(carriers))
        width = 0.35

        ax.bar(x - width/2, model_vals, width, color=colors, alpha=0.85,
               edgecolor="white", linewidth=0.5, label=f"Model ({RUN})")
        ax.bar(x + width/2, ref_vals, width, color=colors, alpha=0.40,
               edgecolor="#333333", linewidth=0.8, hatch="//",
               label=REF_LABELS[year])

        if year == 2060:
            for i, rv in enumerate(ref_vals):
                if rv > 0:
                    ax.errorbar(x[i] + width/2, rv, yerr=rv * 0.15,
                                fmt="none", color="#333333", capsize=3, linewidth=1)

        for i, (mv, rv) in enumerate(zip(model_vals, ref_vals)):
            if rv > 0:
                err = (mv - rv) / rv * 100
                ypos = max(mv, rv) + max(max(model_vals), max(ref_vals)) * 0.04
                ax.text(x[i], ypos, f"{err:+.0f}%",
                        ha="center", fontsize=7, color="#333333")

        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() for c in carriers],
                           rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Electricity generation (TWh)", fontsize=10)
        ax.set_title(f"{name.replace(chr(10), ' ')}\nvs {REF_LABELS[year]}", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle(f"Electricity generation by carrier — Model ({RUN}) vs reference",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(output_dir, "generation_noCO2_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


def plot_generation_shares(metrics, output_dir):
    names = list(metrics.keys())
    carriers = [c for c in sorted(CARRIER_COLORS.keys()) if c != "oil"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, name in zip(axes, names):
        gen = metrics[name]["gen"]
        total = sum(gen.get(c, 0) for c in carriers)
        shares = [gen.get(c, 0) / total * 100 if total > 0 else 0 for c in carriers]
        colors = [CARRIER_COLORS[c] for c in carriers]

        wedges, texts, autotexts = ax.pie(
            shares, labels=None, colors=colors,
            autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
            startangle=90, pctdistance=0.75,
        )
        for at in autotexts:
            at.set_fontsize(8)

        ax.set_title(f"{name.replace(chr(10), ' ')}\nTotal: {total:.0f} TWh", fontsize=10)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=CARRIER_COLORS[c], label=c.capitalize())
               for c in carriers]
    fig.legend(handles=handles, loc="lower center", ncol=len(carriers),
               fontsize=9, bbox_to_anchor=(0.5, -0.05))

    plt.suptitle(f"Generation mix — Model ({RUN}) scenarios", fontsize=13)
    plt.tight_layout()
    out = os.path.join(output_dir, "generation_noCO2_shares.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


def plot_capacity_trajectory(metrics, output_dir):
    names = list(metrics.keys())
    years = [SCENARIOS[n]["year"] for n in names]
    carriers = [c for c in sorted(CARRIER_COLORS.keys()) if c != "oil"]

    fig, ax = plt.subplots(figsize=(12, 6))

    for c in carriers:
        model_vals = [metrics[n]["cap"].get(c, 0) for n in names]
        if max(model_vals) > 0:
            ax.plot(years, model_vals, marker="o", color=CARRIER_COLORS[c],
                    linewidth=2, markersize=7, label=c.capitalize())

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Installed capacity (GW)", fontsize=12)
    ax.set_title(f"Capacity trajectory — Model ({RUN}): 2020 → 2025 → 2060", fontsize=12)
    ax.set_xticks(years)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    plt.tight_layout()
    out = os.path.join(output_dir, "capacity_noCO2_trajectory.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Run: {RUN}")
    print(f"Input:  {NETWORK_DIR}/")
    print(f"Output: {OUTPUT_DIR}/")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_lines = []

    # Load all networks
    metrics = {}
    for name, meta in SCENARIOS.items():
        print(f"Loading {name.replace(chr(10), ' ')}...")
        if not os.path.exists(meta["file"]):
            print(f"  [WARNING] File not found: {meta['file']} — skipping")
            continue
        metrics[name] = extract_metrics(meta["file"])

    if not metrics:
        print("[ERROR] No networks loaded.")
        raise SystemExit(1)

    # ── REPORT ────────────────────────────────────────────────────────────────
    def log(s=""):
        print(s); report_lines.append(s)

    log("=" * 70)
    log(f"MASTER noCO2 ANALYSIS — run: {RUN}")
    log("CN2020 / CN2025 / CN2060 Admin2")
    log("=" * 70)

    log("\n1. NETWORK STRUCTURE")
    log("-" * 50)
    for name in metrics:
        m = metrics[name]
        log(f"  {name.replace(chr(10), ' '):<25} "
            f"buses={m['n_buses']}  snapshots={m['n_snapshots']}")

    log("\n2. TOTAL ELECTRICITY LOAD")
    log("-" * 60)
    log(f"  {'Scenario':<25} {'Model (TWh)':>12} {'Ref (TWh)':>12} {'Error %':>10}")
    for name, meta in SCENARIOS.items():
        if name not in metrics:
            continue
        m = metrics[name]
        ref = meta["load_ref_twh"]
        err = (m["load_twh"] - ref) / ref * 100
        log(f"  {name.replace(chr(10), ' '):<25} {m['load_twh']:>12.1f} {ref:>12.1f} {err:>9.1f}%")

    log("\n3. ENERGY BALANCE (generation vs load)")
    log("-" * 70)
    log(f"  {'Scenario':<25} {'Load (TWh)':>12} {'Gen (TWh)':>12} {'Error %':>10}")
    for name in metrics:
        m = metrics[name]
        log(f"  {name.replace(chr(10), ' '):<25} "
            f"{m['load_twh']:>12.1f} {m['gen_total']:>12.1f} {m['balance_err']:>9.2f}%")

    log("\n4. INSTALLED CAPACITY BY CARRIER (GW)")
    log("-" * 70)
    carriers = sorted(CARRIER_COLORS.keys())
    header = f"  {'Carrier':<12}" + "".join(
        f" {n.replace(chr(10), ' '):>18}" for n in metrics
    )
    log(header)
    for c in carriers:
        row = f"  {c:<12}" + "".join(
            f" {metrics[n]['cap'].get(c, 0):>18.1f}" for n in metrics
        )
        log(row)

    log("\n5. GENERATION BY CARRIER (TWh)")
    log("-" * 70)
    log(header)
    for c in carriers:
        row = f"  {c:<12}" + "".join(
            f" {metrics[n]['gen'].get(c, 0):>18.1f}" for n in metrics
        )
        log(row)

    report_path = os.path.join(OUTPUT_DIR, "report_noCO2.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nReport saved: {report_path}")

    print("\nProducing plots...")
    plot_load_comparison(metrics, OUTPUT_DIR)
    plot_capacity_comparison(metrics, OUTPUT_DIR)
    plot_generation_comparison(metrics, OUTPUT_DIR)
    plot_generation_shares(metrics, OUTPUT_DIR)
    plot_capacity_trajectory(metrics, OUTPUT_DIR)

    print("\nDone.")