"""
compare_scenarios.py
====================
Analisi comparativa delle simulazioni CN2020 / CN2025 / CN2060.

Confronta le tre simulazioni su:
  - struttura della rete (bus, snapshot)
  - carico totale annuo vs riferimento Provincial_Load
  - energy balance check (generazione vs carico)
  - capacità installata per carrier (GW)
  - generazione per carrier (TWh)
  - confronto CN2020 vs dati reali EFC 2020 (capacità e generazione)

Output (salvato in analysis/comparison/output/):
  - comparison_report.txt      : tabelle numeriche complete
  - load_comparison.png        : carico totale modello vs riferimento
  - capacity_comparison.png    : capacità installata per carrier (stacked bar)
  - generation_comparison.png  : mix di generazione per carrier (stacked bar)
  - load_profile_comparison.png: profili mensili sovrapposti
  - capacity_trajectory.png    : traiettoria capacità 2020->2025->2060

Uso:
  python analysis/comparison/compare_scenarios.py --phase 1
  python analysis/comparison/compare_scenarios.py --phase 2
  python analysis/comparison/compare_scenarios.py --phase 2 --no-efc
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pypsa
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── PARAMETERS ────────────────────────────────────────────────────────────────

# Paths are relative to the project root (pypsa-earth-ivan/)
SCENARIOS = {
    "CN2020": {
        "file":         "results/networks/CN2020/elec_s_250_ec_lcopt_Co2L-3h.nc",
        "load_ref_twh": 7587.6,
        "efc_year":     2020,
        "color":        "#1f77b4",
    },
    "CN2025": {
        "file":         "results/networks/CN2025/elec_s_250_ec_lcopt_Co2L-6h.nc",
        "load_ref_twh": 9849.0,
        "efc_year":     None,
        "color":        "#ff7f0e",
    },
    "CN2060": {
        "file":         "results/networks/CN2060/elec_s_250_ec_lcopt_Co2L-6h.nc",
        "load_ref_twh": 16301.0,
        "efc_year":     None,
        "color":        "#2ca02c",
    },
}

EFC_FILE   = "resources/data/validation/EFC_power_his_data.xlsx"
OUTPUT_DIR = "analysis/capacity/output"

# Maps PyPSA-Earth carriers to EFC technology categories.
# Units note: EFC capacity values are in units of 10 MW -> divide by 100 to get GW.
#             EFC generation values are in units of 100 GWh -> divide by 10 to get TWh.
EFC_CARRIER_MAP = {
    "coal":       "coal",
    "lignite":    "coal",
    "CCGT":       "gas",
    "oil":        "other fossile",
    "ror":        "hydroelectricity",
    "hydro":      "hydroelectricity",
    "PHS":        "hydroelectricity",
    "onwind":     "onwind",
    "offwind-ac": "onwind",
    "offwind-dc": "onwind",
    "solar":      "solar",
    "nuclear":    "nuclear",
}

# ─────────────────────────────────────────────────────────────────────────────


def load_network(name, meta):
    path = meta["file"]
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        return None
    print(f"  Loading {name}: {path}")
    return pypsa.Network(path)


def extract_metrics(n):
    """Extract key quantities from a PyPSA network."""
    w = n.snapshot_weightings.generators
    load_twh    = (n.loads_t.p_set.multiply(w, axis=0)).sum().sum() / 1e6
    cap_gw      = n.generators.groupby("carrier")["p_nom"].sum() / 1e3
    cap_gw      = cap_gw[cap_gw.index != "load shedding"]
    stor_gw     = n.storage_units.groupby("carrier")["p_nom"].sum() / 1e3
    gen_twh     = (
        n.generators_t.p.multiply(w, axis=0).sum()
        .groupby(n.generators.carrier).sum() / 1e6
    )
    gen_twh     = gen_twh[gen_twh.index != "load shedding"]
    load_ts     = n.loads_t.p_set.sum(axis=1)
    load_monthly = (load_ts * w).resample("ME").sum() / 1e6
    gen_total   = gen_twh.sum()
    return {
        "load_twh":          load_twh,
        "cap_gw":            cap_gw,
        "stor_gw":           stor_gw,
        "gen_twh":           gen_twh,
        "load_monthly":      load_monthly,
        "gen_total":         gen_total,
        "balance_error_pct": (gen_total - load_twh) / load_twh * 100,
        "n_buses":           len(n.buses),
        "n_snapshots":       len(n.snapshots),
        "period":            f"{n.snapshots[0]} -> {n.snapshots[-1]}",
    }


# ── PHASE 1: Explore EFC labels ───────────────────────────────────────────────

def phase1_explore_efc():
    print("\n" + "=" * 60)
    print("PHASE 1 -- RAW EFC LABELS")
    print("=" * 60)
    if not os.path.exists(EFC_FILE):
        print(f"[ERROR] EFC file not found: {EFC_FILE}")
        return
    for sheet, label in [("A-1-1", "GENERATION/LOAD"), ("B-2-1", "CAPACITY")]:
        print(f"\nSheet {sheet} -- {label}")
        print("-" * 40)
        try:
            df = pd.read_excel(EFC_FILE, sheet_name=sheet, header=5)
            print(f"Available columns: {list(df.columns)}")
            df_cn = df[df["region"] == "China"] if "region" in df.columns else df
            print(f"Rows for China: {len(df_cn)}")
            label_col = next((c for c in ["lang", "Lang", "carrier"] if c in df_cn.columns), None)
            if label_col:
                print(f"\nLabels in column '{label_col}':")
                for val in df_cn[label_col].dropna().unique():
                    print(f"  - {val}")
        except Exception as e:
            print(f"[ERROR] {e}")
    print("\nPhase 1 complete.")


# ── PHASE 2: Full comparative analysis ───────────────────────────────────────

def phase2_compare(use_efc=True):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_lines = []

    def log(s=""):
        print(s)
        report_lines.append(s)

    log("=" * 60)
    log("COMPARATIVE ANALYSIS: CN2020 / CN2025 / CN2060")
    log("=" * 60)

    metrics = {}
    for name, meta in SCENARIOS.items():
        print(f"\nLoading {name}...")
        n = load_network(name, meta)
        if n is not None:
            metrics[name] = extract_metrics(n)

    if not metrics:
        print("[ERROR] No networks loaded.")
        sys.exit(1)

    # 1. Network structure
    log("\n" + "=" * 60)
    log("1. NETWORK STRUCTURE")
    log("=" * 60)
    log(f"{'':20} {'CN2020':>12} {'CN2025':>12} {'CN2060':>12}")
    log("-" * 58)
    for key in ["n_buses", "n_snapshots"]:
        row = f"{key:<20}"
        for name in SCENARIOS:
            row += f" {metrics[name][key]:>12}" if name in metrics else f" {'N/A':>12}"
        log(row)
    for name in SCENARIOS:
        if name in metrics:
            log(f"  {name} period: {metrics[name]['period']}")

    # 2. Total load
    log("\n" + "=" * 60)
    log("2. TOTAL ELECTRICITY LOAD")
    log("=" * 60)
    log(f"{'Scenario':12} {'Model (TWh)':>15} {'Reference (TWh)':>16} {'Error %':>10}")
    log("-" * 56)
    for name, meta in SCENARIOS.items():
        if name not in metrics:
            continue
        m = metrics[name]
        ref = meta["load_ref_twh"]
        err = (m["load_twh"] - ref) / ref * 100
        log(f"{name:<12} {m['load_twh']:>15.1f} {ref:>16.1f} {err:>9.1f}%")

    # 3. Energy balance
    log("\n" + "=" * 60)
    log("3. ENERGY BALANCE CHECK (generation vs load)")
    log("=" * 60)
    log(f"{'Scenario':12} {'Load (TWh)':>12} {'Generation (TWh)':>17} {'Error %':>10}")
    log("-" * 54)
    for name in SCENARIOS:
        if name not in metrics:
            continue
        m = metrics[name]
        log(f"{name:<12} {m['load_twh']:>12.1f} {m['gen_total']:>17.1f} {m['balance_error_pct']:>9.2f}%")

    # 4. Installed capacity
    log("\n" + "=" * 60)
    log("4. INSTALLED CAPACITY BY CARRIER (GW)")
    log("=" * 60)
    all_carriers = sorted(set().union(*[set(metrics[n]["cap_gw"].index) for n in metrics]))
    header = f"{'Carrier':18}" + "".join(f" {n:>10}" for n in SCENARIOS if n in metrics)
    log(header)
    log("-" * (18 + 11 * len(metrics)))
    for c in all_carriers:
        row = f"{c:<18}" + "".join(
            f" {metrics[n]['cap_gw'].get(c, 0.0):>10.1f}" for n in SCENARIOS if n in metrics
        )
        log(row)
    log(f"{'TOTAL':18}" + "".join(
        f" {metrics[n]['cap_gw'].sum():>10.1f}" for n in SCENARIOS if n in metrics
    ))

    # 5. Generation by carrier
    log("\n" + "=" * 60)
    log("5. GENERATION BY CARRIER (TWh)")
    log("=" * 60)
    all_gen = sorted(set().union(*[set(metrics[n]["gen_twh"].index) for n in metrics]))
    log(header)
    log("-" * (18 + 11 * len(metrics)))
    for c in all_gen:
        row = f"{c:<18}" + "".join(
            f" {metrics[n]['gen_twh'].get(c, 0.0):>10.1f}" for n in SCENARIOS if n in metrics
        )
        log(row)
    log(f"{'TOTAL':18}" + "".join(
        f" {metrics[n]['gen_twh'].sum():>10.1f}" for n in SCENARIOS if n in metrics
    ))

    # 6. CN2020 vs EFC
    if use_efc and "CN2020" in metrics and os.path.exists(EFC_FILE):
        log("\n" + "=" * 60)
        log("6. CN2020 vs EFC 2020 COMPARISON")
        log("=" * 60)
        try:
            # Capacity (units: 10 MW -> /100 -> GW)
            efc_cap = pd.read_excel(EFC_FILE, sheet_name="B-2-1", header=5)
            efc_cap = efc_cap[efc_cap["region"] == "China"][["lang", 2020]].dropna()
            efc_cap.columns = ["carrier_efc", "raw"]
            efc_cap["cap_gw"] = efc_cap["raw"] / 100
            efc_cap = efc_cap.set_index("carrier_efc")["cap_gw"]

            # Generation (units: 100 GWh -> /10 -> TWh)
            efc_gen = pd.read_excel(EFC_FILE, sheet_name="A-1-1", header=5)
            efc_gen = efc_gen[efc_gen["region"] == "China"][["lang", 2020]].dropna()
            efc_gen.columns = ["carrier_efc", "raw"]
            efc_gen = efc_gen[efc_gen["carrier_efc"] != "Load"]
            efc_gen["gen_twh"] = efc_gen["raw"] / 10
            efc_gen = efc_gen.set_index("carrier_efc")["gen_twh"]

            pypsa_cap = pd.concat([metrics["CN2020"]["cap_gw"], metrics["CN2020"]["stor_gw"]])
            pypsa_gen = metrics["CN2020"]["gen_twh"]

            agg_cap, agg_gen = {}, {}
            for carrier, cat in EFC_CARRIER_MAP.items():
                agg_cap[cat] = agg_cap.get(cat, 0) + pypsa_cap.get(carrier, 0)
                agg_gen[cat] = agg_gen.get(cat, 0) + pypsa_gen.get(carrier, 0)

            log("\nInstalled capacity -- model vs EFC (GW):")
            log(f"{'EFC category':20} {'Model':>10} {'EFC 2020':>10} {'Error %':>10}")
            log("-" * 54)
            for cat in sorted(set(list(agg_cap.keys()) + list(efc_cap.index))):
                mod = agg_cap.get(cat, 0)
                ref = efc_cap.get(cat, 0)
                err = (mod - ref) / ref * 100 if ref != 0 else float("nan")
                log(f"{cat:<20} {mod:>10.1f} {ref:>10.1f} {err:>9.1f}%")

            log("\nGeneration -- model vs EFC (TWh):")
            log(f"{'EFC category':20} {'Model':>10} {'EFC 2020':>10} {'Error %':>10}")
            log("-" * 54)
            for cat in sorted(set(list(agg_gen.keys()) + list(efc_gen.index))):
                mod = agg_gen.get(cat, 0)
                ref = efc_gen.get(cat, 0)
                err = (mod - ref) / ref * 100 if ref != 0 else float("nan")
                log(f"{cat:<20} {mod:>10.1f} {ref:>10.1f} {err:>9.1f}%")

        except Exception as e:
            log(f"[ERROR] Reading EFC file: {e}")

    # Save report
    report_path = os.path.join(OUTPUT_DIR, "comparison_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nReport saved: {report_path}")

    # Plots
    names  = [n for n in SCENARIOS if n in metrics]
    colors = [SCENARIOS[n]["color"] for n in names]
    import numpy as np

    # Plot A: total load
    fig, ax = plt.subplots(figsize=(8, 5))
    loads = [metrics[n]["load_twh"] for n in names]
    refs  = [SCENARIOS[n]["load_ref_twh"] for n in names]
    x = range(len(names))
    ax.bar([i - 0.2 for i in x], loads, width=0.35, label="Model",     color=colors, alpha=0.85)
    ax.bar([i + 0.2 for i in x], refs,  width=0.35, label="Reference", color=colors, alpha=0.35, hatch="//")
    ax.set_xticks(list(x)); ax.set_xticklabels(names)
    ax.set_ylabel("TWh"); ax.set_title("Total electricity load: model vs reference")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "load_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(); print(f"Saved: {OUTPUT_DIR}/load_comparison.png")

    # Plot B: capacity stacked bar
    fig, ax = plt.subplots(figsize=(12, 6))
    pd.DataFrame({n: metrics[n]["cap_gw"] for n in names}).fillna(0).T.plot(
        kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_ylabel("GW"); ax.set_title("Installed capacity by carrier -- scenario comparison")
    ax.set_xticklabels(names, rotation=0)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "capacity_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(); print(f"Saved: {OUTPUT_DIR}/capacity_comparison.png")

    # Plot C: generation stacked bar
    fig, ax = plt.subplots(figsize=(12, 6))
    pd.DataFrame({n: metrics[n]["gen_twh"] for n in names}).fillna(0).T.plot(
        kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_ylabel("TWh"); ax.set_title("Generation mix by carrier -- scenario comparison")
    ax.set_xticklabels(names, rotation=0)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "generation_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(); print(f"Saved: {OUTPUT_DIR}/generation_comparison.png")

    # Plot D: monthly load profiles
    MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig, ax = plt.subplots(figsize=(12, 5))
    for name in names:
        ml = metrics[name]["load_monthly"]
        ax.plot(range(1, len(ml) + 1), ml.values, label=name,
                color=SCENARIOS[name]["color"], linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Month"); ax.set_ylabel("TWh / month")
    ax.set_title("Monthly load profile -- scenario comparison")
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(MONTH_LABELS)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "load_profile_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(); print(f"Saved: {OUTPUT_DIR}/load_profile_comparison.png")

    # Plot E: capacity trajectory
    years_map = {"CN2020": 2020, "CN2025": 2025, "CN2060": 2060}
    years = [years_map[n] for n in names]
    all_c = sorted(set().union(*[set(metrics[n]["cap_gw"].index) for n in names]))
    fig, ax = plt.subplots(figsize=(12, 6))
    for c in all_c:
        vals = [metrics[n]["cap_gw"].get(c, 0) for n in names]
        if max(vals) > 0:
            ax.plot(years, vals, marker="o", label=c, linewidth=1.5)
    ax.set_xlabel("Year"); ax.set_ylabel("GW")
    ax.set_title("Installed capacity trajectory by carrier (2020 -> 2025 -> 2060)")
    ax.set_xticks(years)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "capacity_trajectory.png"), dpi=150, bbox_inches="tight")
    plt.close(); print(f"Saved: {OUTPUT_DIR}/capacity_trajectory.png")

    print(f"\nPhase 2 complete. Output: {OUTPUT_DIR}/")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparative analysis of PyPSA-Earth China scenarios")
    parser.add_argument("--phase", type=int, choices=[1, 2], required=True)
    parser.add_argument("--no-efc", action="store_true")
    args = parser.parse_args()
    if args.phase == 1:
        phase1_explore_efc()
    elif args.phase == 2:
        phase2_compare(use_efc=not args.no_efc)
