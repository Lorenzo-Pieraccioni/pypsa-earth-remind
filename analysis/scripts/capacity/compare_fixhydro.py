"""
compare_fixhydro.py
===================
Confronto diretto tra CN2020_Admin2_noCO2 (baseline) e CN2020_Admin2_noCO2_fixhydro
per valutare l'impatto del fix idro (PR #1707 + hydro_capacities.csv China row).

Produce:
  1. report_fixhydro.txt              : tabelle numeriche complete
  2. capacity_fixhydro_comparison.png : capacità installata per carrier
  3. generation_fixhydro_comparison.png: mix di generazione per carrier vs Ember 2020
  4. hydro_monthly_fixhydro.png       : profilo mensile generazione idro vs Ember 2020

Dati di riferimento: Ember 2020 (valori esatti)

Uso:
  python analysis/capacity/compare_fixhydro.py
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
import matplotlib.ticker as mticker

# ── PARAMETERS ────────────────────────────────────────────────────────────────

OUTPUT_DIR = "analysis/output/fix-hydro/capacity"

RUNS = {
    "Baseline\n(noCO2)": {
        "file":  "results/networks/noCO2/CN2020_Admin2/elec_s_250_ec_lcopt_6h.nc",
        "color": "#1f77b4",
    },
    "Fix-hydro\n(noCO2)": {
        "file":  "results/networks/fix-hydro/CN2020_Admin2/elec_s_250_ec_lcopt_6h.nc",
        "color": "#2ca02c",
    },
}

# Ember 2020 — valori esatti
REF_CAP_2020 = {
    "coal":    1048.3,
    "gas":      101.9,
    "solar":    253.9,
    "wind":     282.0,
    "nuclear":   49.9,
    "hydro":    338.7,
    "oil":        9.5,
}

REF_GEN_2020 = {
    "coal":    4921.9,
    "gas":      252.5,
    "solar":    261.1,
    "wind":     466.5,
    "nuclear":  366.2,
    "hydro":   1321.7,
}

# Generazione idro mensile Ember 2020 (TWh) — lettura da dati pubblici
# Valori approssimativi basati su stagionalità tipica cinese
REF_HYDRO_MONTHLY_2020 = [
    75.0, 68.0, 72.0, 85.0, 110.0, 130.0,
    160.0, 175.0, 155.0, 130.0, 95.0, 80.0,
]

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

MONTH_LABELS = ["Gen","Feb","Mar","Apr","Mag","Giu",
                "Lug","Ago","Set","Ott","Nov","Dic"]

# ─────────────────────────────────────────────────────────────────────────────


def extract_metrics(network_file):
    n = pypsa.Network(network_file)
    w = n.snapshot_weightings.generators

    load_twh = (n.loads_t.p_set.multiply(w, axis=0)).sum().sum() / 1e6

    cap = n.generators.groupby("carrier")["p_nom"].sum() / 1e3
    cap = cap[cap.index != "load shedding"]
    stor_cap = n.storage_units.groupby("carrier")["p_nom"].sum() / 1e3

    gen = (
        n.generators_t.p.multiply(w, axis=0).sum()
        .groupby(n.generators.carrier).sum() / 1e6
    )
    gen = gen[gen.index != "load shedding"]

    if not n.storage_units_t.p.empty:
        stor_gen = (
            n.storage_units_t.p.multiply(w, axis=0)
            .clip(lower=0).sum()
            .groupby(n.storage_units.carrier).sum() / 1e6
        )
    else:
        stor_gen = pd.Series(dtype=float)

    agg_cap, agg_gen = {}, {}
    for carrier, cat in CARRIER_MAP.items():
        agg_cap[cat] = agg_cap.get(cat, 0) + cap.get(carrier, 0) + stor_cap.get(carrier, 0)
        agg_gen[cat] = agg_gen.get(cat, 0) + gen.get(carrier, 0) + stor_gen.get(carrier, 0)

    # Profilo mensile idro (hydro + ror, escluso PHS)
    hydro_carriers = [c for c in n.storage_units.index
                      if n.storage_units.loc[c, "carrier"] == "hydro"]
    ror_carriers   = [c for c in n.generators.index
                      if n.generators.loc[c, "carrier"] == "ror"]

    hydro_ts = pd.Series(0.0, index=n.snapshots)
    if hydro_carriers and not n.storage_units_t.p.empty:
        hydro_ts += n.storage_units_t.p[hydro_carriers].clip(lower=0).sum(axis=1)
    if ror_carriers and not n.generators_t.p.empty:
        hydro_ts += n.generators_t.p[ror_carriers].sum(axis=1)

    hydro_monthly = (hydro_ts * w).resample("ME").sum() / 1e6

    gen_total = sum(agg_gen.values())
    balance_err = (gen_total - load_twh) / load_twh * 100 if load_twh > 0 else float("nan")

    return {
        "load_twh":      load_twh,
        "cap":           pd.Series(agg_cap),
        "gen":           pd.Series(agg_gen),
        "gen_total":     gen_total,
        "balance_err":   balance_err,
        "hydro_monthly": hydro_monthly,
    }


# ── PLOTS ─────────────────────────────────────────────────────────────────────

def plot_capacity_comparison(metrics, output_dir):
    carriers = sorted(CARRIER_COLORS.keys())
    names    = list(RUNS.keys())
    x        = np.arange(len(carriers))
    width    = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))

    # Baseline
    vals_base = [metrics[names[0]]["cap"].get(c, 0) for c in carriers]
    ax.bar(x - width, vals_base, width,
           color=[CARRIER_COLORS[c] for c in carriers],
           alpha=0.85, edgecolor="white", linewidth=0.5, label="Baseline (noCO2)")

    # Fix-hydro
    vals_fix = [metrics[names[1]]["cap"].get(c, 0) for c in carriers]
    ax.bar(x, vals_fix, width,
           color=[CARRIER_COLORS[c] for c in carriers],
           alpha=0.55, edgecolor="#333333", linewidth=0.8,
           hatch="//", label="Fix-hydro (noCO2)")

    # Ember 2020
    vals_ref = [REF_CAP_2020.get(c, 0) for c in carriers]
    ax.bar(x + width, vals_ref, width,
           color=[CARRIER_COLORS[c] for c in carriers],
           alpha=0.30, edgecolor="#cc0000", linewidth=1.2,
           hatch="xx", label="Ember 2020")

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in carriers],
                       rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Installed capacity (GW)", fontsize=12)
    ax.set_title("Installed capacity by carrier\nBaseline vs Fix-hydro vs Ember 2020", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    plt.tight_layout()
    out = os.path.join(output_dir, "capacity_fixhydro_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


def plot_generation_comparison(metrics, output_dir):
    carriers = sorted(CARRIER_COLORS.keys())
    names    = list(RUNS.keys())
    x        = np.arange(len(carriers))
    width    = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))

    vals_base = [metrics[names[0]]["gen"].get(c, 0) for c in carriers]
    ax.bar(x - width, vals_base, width,
           color=[CARRIER_COLORS[c] for c in carriers],
           alpha=0.85, edgecolor="white", linewidth=0.5, label="Baseline (noCO2)")

    vals_fix = [metrics[names[1]]["gen"].get(c, 0) for c in carriers]
    ax.bar(x, vals_fix, width,
           color=[CARRIER_COLORS[c] for c in carriers],
           alpha=0.55, edgecolor="#333333", linewidth=0.8,
           hatch="//", label="Fix-hydro (noCO2)")

    vals_ref = [REF_GEN_2020.get(c, 0) for c in carriers]
    ax.bar(x + width, vals_ref, width,
           color=[CARRIER_COLORS[c] for c in carriers],
           alpha=0.30, edgecolor="#cc0000", linewidth=1.2,
           hatch="xx", label="Ember 2020")

    # Errore percentuale fix-hydro vs Ember
    for i, (mv, rv) in enumerate(zip(vals_fix, vals_ref)):
        if rv > 0:
            err = (mv - rv) / rv * 100
            ypos = max(vals_base[i], mv, rv) + max(vals_ref) * 0.03
            ax.text(x[i], ypos, f"{err:+.0f}%",
                    ha="center", fontsize=7, color="#2ca02c", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in carriers],
                       rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Electricity generation (TWh)", fontsize=12)
    ax.set_title("Generation by carrier\nBaseline vs Fix-hydro vs Ember 2020\n"
                 "(% labels: Fix-hydro error vs Ember)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir, "generation_fixhydro_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


def plot_hydro_monthly(metrics, output_dir):
    names  = list(RUNS.keys())
    colors = [RUNS[n]["color"] for n in names]

    fig, ax = plt.subplots(figsize=(12, 5))

    for name, color in zip(names, colors):
        hm = metrics[name]["hydro_monthly"]
        if len(hm) == 12:
            ax.plot(range(1, 13), hm.values, marker="o", color=color,
                    linewidth=2, markersize=6, label=name.replace("\n", " "))

    ax.plot(range(1, 13), REF_HYDRO_MONTHLY_2020, marker="s",
            color="#cc0000", linewidth=2, markersize=6,
            linestyle="--", label="Ember 2020 (reference)")

    ax.fill_between(range(1, 13), REF_HYDRO_MONTHLY_2020,
                    alpha=0.08, color="#cc0000")

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS, fontsize=10)
    ax.set_ylabel("Hydropower generation (TWh)", fontsize=12)
    ax.set_title("Monthly hydropower generation profile\n"
                 "Baseline vs Fix-hydro vs Ember 2020", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Totali annui
    for name, color in zip(names, colors):
        hm = metrics[name]["hydro_monthly"]
        if len(hm) == 12:
            ax.text(12.2, hm.values[-1],
                    f"{hm.sum():.0f} TWh",
                    color=color, fontsize=9, va="center")
    ax.text(12.2, REF_HYDRO_MONTHLY_2020[-1],
            f"{sum(REF_HYDRO_MONTHLY_2020):.0f} TWh",
            color="#cc0000", fontsize=9, va="center")

    plt.tight_layout()
    out = os.path.join(output_dir, "hydro_monthly_fixhydro.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_lines = []

    def log(s=""):
        print(s); report_lines.append(s)

    metrics = {}
    for name, meta in RUNS.items():
        print(f"Loading {name.replace(chr(10), ' ')}...")
        if not os.path.exists(meta["file"]):
            print(f"  [ERROR] File not found: {meta['file']}")
            continue
        metrics[name] = extract_metrics(meta["file"])

    if len(metrics) < 2:
        print("[ERROR] One or both network files not found.")
        raise SystemExit(1)

    names = list(RUNS.keys())

    log("=" * 70)
    log("FIX-HYDRO COMPARISON — CN2020_Admin2_noCO2 vs CN2020_Admin2_noCO2_fixhydro")
    log("=" * 70)

    log("\n1. TOTAL LOAD AND ENERGY BALANCE")
    log("-" * 70)
    log(f"  {'Run':<30} {'Load (TWh)':>12} {'Gen (TWh)':>12} {'Error %':>10}")
    for name in names:
        m = metrics[name]
        log(f"  {name.replace(chr(10), ' '):<30} "
            f"{m['load_twh']:>12.1f} {m['gen_total']:>12.1f} {m['balance_err']:>9.2f}%")

    log("\n2. INSTALLED CAPACITY BY CARRIER (GW)")
    log("-" * 70)
    carriers = sorted(CARRIER_COLORS.keys())
    log(f"  {'Carrier':<12} {'Baseline':>12} {'Fix-hydro':>12} {'Ember 2020':>12} {'Delta %':>10}")
    for c in carriers:
        base = metrics[names[0]]["cap"].get(c, 0)
        fix  = metrics[names[1]]["cap"].get(c, 0)
        ref  = REF_CAP_2020.get(c, 0)
        delta = (fix - base) / base * 100 if base > 0 else 0
        log(f"  {c:<12} {base:>12.1f} {fix:>12.1f} {ref:>12.1f} {delta:>9.1f}%")

    log("\n3. GENERATION BY CARRIER (TWh)")
    log("-" * 70)
    log(f"  {'Carrier':<12} {'Baseline':>12} {'Fix-hydro':>12} {'Ember 2020':>12} "
        f"{'Fix vs Ember %':>15}")
    for c in carriers:
        base = metrics[names[0]]["gen"].get(c, 0)
        fix  = metrics[names[1]]["gen"].get(c, 0)
        ref  = REF_GEN_2020.get(c, 0)
        err  = (fix - ref) / ref * 100 if ref > 0 else float("nan")
        log(f"  {c:<12} {base:>12.1f} {fix:>12.1f} {ref:>12.1f} {err:>14.1f}%")

    log("\n4. HYDRO MONTHLY PROFILE (TWh)")
    log("-" * 70)
    log(f"  {'Month':<8} {'Baseline':>12} {'Fix-hydro':>12} {'Ember 2020':>12}")
    for i, month in enumerate(MONTH_LABELS):
        base_hm = metrics[names[0]]["hydro_monthly"]
        fix_hm  = metrics[names[1]]["hydro_monthly"]
        bv = base_hm.iloc[i] if len(base_hm) == 12 else float("nan")
        fv = fix_hm.iloc[i]  if len(fix_hm)  == 12 else float("nan")
        rv = REF_HYDRO_MONTHLY_2020[i]
        log(f"  {month:<8} {bv:>12.1f} {fv:>12.1f} {rv:>12.1f}")

    report_path = os.path.join(OUTPUT_DIR, "report_fixhydro.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nReport saved: {report_path}")

    print("\nProducing plots...")
    plot_capacity_comparison(metrics, output_dir=OUTPUT_DIR)
    plot_generation_comparison(metrics, output_dir=OUTPUT_DIR)
    plot_hydro_monthly(metrics, output_dir=OUTPUT_DIR)

    print("\nDone.")