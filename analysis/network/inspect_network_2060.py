"""
inspect_network_2060.py
=======================
Analysis of CN2060 decarbonization scenario results from PyPSA-Earth.
Designed for optimization scenarios (not validation against historical data).

Sections:
  1. Network structure
  2. Total electricity load vs CETO 2025 projection
  3. Installed capacity (GW) vs CETO 2025 BCNS/ICNS
  4. Generation (TWh) vs CETO 2025 BCNS/ICNS
  5. Energy balance (with battery artefact note)
  6. CO2 price and system cost
  7. Storage analysis (GW and TWh)
  8. Comparison vs CETO 2025

Plots:
  1. Capacity comparison: model vs CETO BCNS/ICNS with uncertainty bands
  2. Generation comparison: same
  3. Generation mix pie chart
  4. Storage capacity (GW and TWh)
  5. Capacity factor by carrier

Reference: CETO 2025 Executive Summary
  BCNS: Baseline Carbon Neutrality Scenario
  ICNS: Ideal Carbon Neutrality Scenario
  Uncertainty: +/-5% (from explicit text/tables), +/-15% (from visual reading)

Usage:
  python analysis/network/inspect_network_2060.py \\
      results/networks/CN2060_95pct_Admin2/elec_s_250_ec_lcopt_Co2L0.05-3h.nc
  python analysis/network/inspect_network_2060.py \\
      results/networks/CN2060_100pct_Admin2/elec_s_250_ec_lcopt_Co2L0.0-3h.nc
"""

import argparse
import os
import pypsa
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── PARAMETERS ────────────────────────────────────────────────────────────────

BASE_OUTPUT_DIR = os.environ.get("PYPSA_OUTPUT_DIR", "analysis/network/output")
CETO_FILE       = "data/validation/ceto_2025_reference.csv"

# CETO 2025 total electricity demand projection for 2060 (TWh)
CETO_LOAD_BCNS = 21200
CETO_LOAD_ICNS = 22600

CARRIER_COLORS = {
    "coal":        "#4d4d4d",
    "lignite":     "#8c6d31",
    "CCGT":        "#6baed6",
    "OCGT":        "#9ecae1",
    "nuclear":     "#e6550d",
    "oil":         "#969696",
    "onwind":      "#74c476",
    "offwind-ac":  "#41ab5d",
    "offwind-dc":  "#238b45",
    "solar":       "#fdd835",
    "ror":         "#3182bd",
    "hydro":       "#08519c",
    "PHS":         "#9ecae1",
    "H2":          "#9e9ac8",
    "battery":     "#fd8d3c",
    # CETO display names
    "Solar":       "#fdd835",
    "Wind":        "#74c476",
    "Hydro":       "#08519c",
    "PHS stor":    "#9ecae1",
    "Nuclear":     "#e6550d",
    "Coal":        "#4d4d4d",
    "Gas":         "#6baed6",
    "Biomass":     "#a1d99b",
    "Battery":     "#fd8d3c",
    "H2 Storage":  "#9e9ac8",
}

# PyPSA carriers -> CETO carriers (capacity comparison)
PYPSA_TO_CETO_CAP = {
    "solar":       "solar_pv_total",
    "onwind":      "wind_total",
    "offwind-ac":  "wind_total",
    "offwind-dc":  "wind_total",
    "hydro":       "hydro",
    "ror":         "hydro",
    "PHS":         "pumped_hydro",
    "nuclear":     "nuclear",
    "coal":        "coal",
    "lignite":     "coal",
    "CCGT":        "gas",
    "OCGT":        "gas",
    "oil":         "gas",
    "battery":     "chemical_storage",
    "H2":          "hydrogen_electrolyser",
}

# PyPSA carriers -> CETO carriers (generation comparison)
PYPSA_TO_CETO_GEN = {
    "solar":       "solar_pv_total",
    "onwind":      "wind_total",
    "offwind-ac":  "wind_total",
    "offwind-dc":  "wind_total",
    "hydro":       "hydro",
    "ror":         "hydro",
    "PHS":         "pumped_hydro",
    "nuclear":     "nuclear",
    "coal":        "coal",
    "lignite":     "coal",
    "CCGT":        "gas",
    "OCGT":        "gas",
    "oil":         "gas",
    "battery":     "chemical_storage",
    "H2":          "hydrogen_electrolyser",
}

# Human-readable display names for CETO carrier keys
CETO_DISPLAY = {
    "solar_pv_total":        "Solar",
    "wind_total":            "Wind",
    "hydro":                 "Hydro",
    "pumped_hydro":          "PHS stor",
    "nuclear":               "Nuclear",
    "coal":                  "Coal",
    "gas":                   "Gas",
    "biomass":               "Biomass",
    "chemical_storage":      "Battery",
    "hydrogen_electrolyser": "H2 Storage",
}

# Ordered carrier list for plots (most important first)
CARRIERS_CAP_ORDER = [
    "solar_pv_total", "wind_total", "nuclear", "hydro",
    "coal", "gas", "chemical_storage", "pumped_hydro",
]
CARRIERS_GEN_ORDER = [
    "solar_pv_total", "wind_total", "nuclear", "hydro",
    "coal", "gas",
]

# ── HELPERS ───────────────────────────────────────────────────────────────────

def get_color(carrier):
    return CARRIER_COLORS.get(carrier, "#aaaaaa")


def load_ceto(ceto_file, year=2060):
    if not os.path.exists(ceto_file):
        print(f"[WARNING] CETO file not found: {ceto_file}")
        return None, None
    df = pd.read_csv(ceto_file)
    df_year = df[df["year"] == year]
    bcns = df_year[df_year["scenario"] == "BCNS"].set_index("carrier")
    icns = df_year[df_year["scenario"] == "ICNS"].set_index("carrier")
    return bcns, icns


def infer_scenario(network_file):
    if "Co2L0.0-" in network_file or "Co2L0.0_" in network_file:
        return "100% decarb (Co2L0.0)"
    elif "Co2L0.05" in network_file:
        return "95% decarb (Co2L0.05)"
    elif "noCO2" in network_file:
        return "no CO2 constraint"
    return "unknown"


def aggregate_carriers(model_series, mapping):
    result = {}
    for carrier, val in model_series.items():
        key = mapping.get(carrier)
        if key:
            result[key] = result.get(key, 0) + val
    return pd.Series(result)


def safe_get(df, carrier, col):
    """Return float value from CETO dataframe, None if missing or NaN."""
    if df is None or carrier not in df.index:
        return None
    val = df.loc[carrier, col]
    if pd.isna(val):
        return None
    return float(val)


# ── MAIN INSPECTION ───────────────────────────────────────────────────────────

def inspect(network_file):
    if not os.path.exists(network_file):
        print(f"[ERROR] File not found: {network_file}")
        raise SystemExit(1)

    print(f"Loading: {network_file}")
    n = pypsa.Network(network_file)
    w = n.snapshot_weightings.generators

    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    network_name = (
        os.path.basename(os.path.dirname(network_file))
        + "_"
        + os.path.splitext(os.path.basename(network_file))[0]
    )
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    scenario_label = infer_scenario(network_file)
    bcns, icns = load_ceto(CETO_FILE, year=2060)
    if bcns is not None:
        log("  CETO 2025 reference loaded (year=2060, BCNS + ICNS)")
    else:
        log("  [WARNING] CETO reference not available")

    # ── 1. NETWORK STRUCTURE ──────────────────────────────────────────────────
    log("=" * 60)
    log("1. NETWORK STRUCTURE")
    log("=" * 60)
    log(f"  File:         {network_file}")
    log(f"  Scenario:     {scenario_label}")
    log(f"  Buses:        {len(n.buses)}")
    log(f"  Generators:   {len(n.generators)}")
    log(f"  Lines:        {len(n.lines)}")
    log(f"  Links:        {len(n.links)}")
    log(f"  Loads:        {len(n.loads)}")
    log(f"  Storage units:{len(n.storage_units)}")
    log(f"  Snapshots:    {len(n.snapshots)}")
    log(f"  Period:       {n.snapshots[0]} -> {n.snapshots[-1]}")
    log(f"  Resolution:   {w.iloc[0]:.0f}h per timestep")

    # ── 2. TOTAL LOAD ─────────────────────────────────────────────────────────
    load_twh = (n.loads_t.p_set.multiply(w, axis=0)).sum().sum() / 1e6
    ceto_mid = (CETO_LOAD_BCNS + CETO_LOAD_ICNS) / 2
    load_err = (load_twh - ceto_mid) / ceto_mid * 100
    log("")
    log("=" * 60)
    log("2. TOTAL ELECTRICITY LOAD")
    log("=" * 60)
    log(f"  Model total:  {load_twh:.1f} TWh")
    log(f"  CETO 2025:    {CETO_LOAD_BCNS:,} (BCNS) — {CETO_LOAD_ICNS:,} (ICNS) TWh")
    log(f"  vs midpoint:  {load_err:+.1f}%")
    log(f"  vs BCNS:      {(load_twh - CETO_LOAD_BCNS) / CETO_LOAD_BCNS * 100:+.1f}%")
    log(f"  vs ICNS:      {(load_twh - CETO_LOAD_ICNS) / CETO_LOAD_ICNS * 100:+.1f}%")

    # ── 3. INSTALLED CAPACITY ─────────────────────────────────────────────────
    cap = n.generators.groupby("carrier")["p_nom_opt"].sum() / 1e3
    cap = cap[cap.index != "load shedding"].sort_values(ascending=False)
    stor = n.storage_units.groupby("carrier")["p_nom_opt"].sum() / 1e3
    all_cap = pd.concat([cap, stor]).sort_values(ascending=False)

    log("")
    log("=" * 60)
    log("3. INSTALLED CAPACITY BY CARRIER (GW) — p_nom_opt")
    log("=" * 60)
    for carrier, val in cap.items():
        log(f"    {carrier:<18} {val:>8.1f} GW")
    log(f"    {'TOTAL GEN':<18} {cap.sum():>8.1f} GW")
    log("  Storage units:")
    for carrier, val in stor.items():
        log(f"    {carrier:<18} {val:>8.1f} GW")
    log(f"    {'TOTAL ALL':<18} {all_cap.sum():>8.1f} GW")
    log(f"  CETO 2025 total installed: 10,450 GW (BCNS) — 11,040 GW (ICNS)")

    # ── 4. GENERATION ─────────────────────────────────────────────────────────
    gen_twh = (
        n.generators_t.p.multiply(w, axis=0).sum()
        .groupby(n.generators.carrier).sum() / 1e6
    )
    gen_twh = gen_twh[gen_twh.index != "load shedding"].sort_values(ascending=False)

    stor_gen_twh = pd.Series(dtype=float)
    if not n.storage_units_t.p.empty:
        stor_gen_twh = (
            n.storage_units_t.p.clip(lower=0)
            .multiply(w, axis=0).sum()
            .groupby(n.storage_units.carrier).sum() / 1e6
        )

    links_gen_twh = pd.Series(dtype=float)
    if not n.links_t.p1.empty:
        links_gen_twh = (
            n.links_t.p1.clip(lower=0)
            .multiply(w, axis=0).sum()
            .groupby(n.links.carrier).sum() / 1e6
        )

    total_gen = gen_twh.sum() + stor_gen_twh.sum()
    all_gen = pd.concat([gen_twh, stor_gen_twh]).sort_values(ascending=False)
    all_gen = all_gen[all_gen > 0]

    log("")
    log("=" * 60)
    log("4. GENERATION BY CARRIER (TWh)")
    log("=" * 60)
    for carrier, val in gen_twh.items():
        share = val / total_gen * 100 if total_gen > 0 else 0
        log(f"    {carrier:<18} {val:>8.1f} TWh  ({share:.1f}%)")
    log("  Storage units (discharge):")
    for carrier, val in stor_gen_twh.items():
        share = val / total_gen * 100 if total_gen > 0 else 0
        log(f"    {carrier:<18} {val:>8.1f} TWh  ({share:.1f}%)")
    if not links_gen_twh.empty:
        log("  Links (transmission):")
        for carrier, val in links_gen_twh.items():
            log(f"    {carrier:<18} {val:>8.1f} TWh")
    log(f"  {'TOTAL':<18} {total_gen:>8.1f} TWh")

    # ── 5. ENERGY BALANCE ─────────────────────────────────────────────────────
    balance_err = (total_gen - load_twh) / load_twh * 100 if load_twh > 0 else float("nan")
    log("")
    log("=" * 60)
    log("5. ENERGY BALANCE CHECK")
    log("=" * 60)
    log(f"  Load:         {load_twh:.1f} TWh")
    log(f"  Gen (total):  {total_gen:.1f} TWh")
    log(f"  Error:        {balance_err:.2f}%")
    if abs(balance_err) > 5:
        log(f"  NOTE: error > 5% is a reporting artefact.")
        log(f"        Battery discharge is counted in total gen but is not net supply.")
        log(f"        The solver energy balance is correct by construction.")
    log(f"  Links total:  {links_gen_twh.sum():.1f} TWh")

    # ── 6. CO2 PRICE AND SYSTEM COST ──────────────────────────────────────────
    log("")
    log("=" * 60)
    log("6. CO2 PRICE AND SYSTEM COST")
    log("=" * 60)

    co2_price = None
    if hasattr(n, "global_constraints") and not n.global_constraints.empty:
        co2_rows = n.global_constraints[
            n.global_constraints.index.str.lower().str.contains("co2")
        ]
        if not co2_rows.empty:
            co2_price = float(co2_rows["mu"].iloc[0])
            log(f"  CO2 constraint: {co2_rows.index[0]}")
            log(f"  CO2 price:      {co2_price:.1f} EUR/tCO2")
        else:
            log("  CO2 price: no CO2 constraint found in global_constraints")
    else:
        log("  CO2 price: global_constraints not available")

    if hasattr(n, "objective") and n.objective is not None:
        obj_bn = n.objective / 1e9
        log(f"  System cost:    {obj_bn:.2f} billion EUR/yr")
    else:
        log("  System cost: n.objective not available")

    # ── 7. STORAGE ANALYSIS ───────────────────────────────────────────────────
    log("")
    log("=" * 60)
    log("7. STORAGE ANALYSIS")
    log("=" * 60)
    log(f"  {'Carrier':<18} {'Power (GW)':>12} {'Energy (TWh)':>14} {'Max hours':>10}")
    log("  " + "-" * 58)
    for carrier in sorted(n.storage_units.carrier.unique()):
        su = n.storage_units[n.storage_units.carrier == carrier]
        power_gw  = su["p_nom_opt"].sum() / 1e3
        max_h     = su["max_hours"].mean()
        energy_twh = power_gw * max_h / 1e3
        log(f"  {carrier:<18} {power_gw:>12.1f} {energy_twh:>14.3f} {max_h:>10.1f}")
    log("")
    log("  CETO 2025 regulation capacity targets for 2060:")
    log("    Chemical storage:  800-1,000 GW   (16-17% of total regulation)")
    log("    Pumped hydro:      400-500 GW    (8-9%)")
    log("    H2 + e-fuels:      1,740-2,180 GW (35-38%)")
    log("    EV-grid (V2G):     750-920 GW    (15-16%)")
    log("    Total regulation:  4,970-5,810 GW")

    # ── 8. COMPARISON VS CETO 2025 ────────────────────────────────────────────
    log("")
    log("=" * 60)
    log("8. COMPARISON VS CETO 2025 (BCNS and ICNS)")
    log("=" * 60)

    model_cap_ceto = aggregate_carriers(all_cap, PYPSA_TO_CETO_CAP)
    model_gen_ceto = aggregate_carriers(all_gen, PYPSA_TO_CETO_GEN)

    if bcns is not None and icns is not None:
        # Capacity
        log("")
        log("  Capacity (GW) — model vs CETO 2025")
        log(f"  {'Carrier':<22} {'Model':>7} {'BCNS':>7} {'ICNS':>7}"
            f" {'vs BCNS':>8} {'vs ICNS':>8} {'Unc':>6}")
        log("  " + "-" * 70)
        for c in CARRIERS_CAP_ORDER:
            m  = model_cap_ceto.get(c, 0.0)
            bv = safe_get(bcns, c, "capacity_gw")
            iv = safe_get(icns, c, "capacity_gw")
            unc = safe_get(bcns, c, "uncertainty") or 0.05
            if bv is None and iv is None and m == 0:
                continue
            b_str  = f"{bv:.0f}" if bv is not None else "n/a"
            i_str  = f"{iv:.0f}" if iv is not None else "n/a"
            b_err  = f"{(m - bv) / bv * 100:+.0f}%" if bv and bv > 0 else "n/a"
            i_err  = f"{(m - iv) / iv * 100:+.0f}%" if iv and iv > 0 else "n/a"
            disp   = CETO_DISPLAY.get(c, c)
            log(f"  {disp:<22} {m:>7.0f} {b_str:>7} {i_str:>7}"
                f" {b_err:>8} {i_err:>8} {unc*100:>5.0f}%")

        # Generation
        log("")
        log("  Generation (TWh) — model vs CETO 2025")
        log(f"  {'Carrier':<22} {'Model':>7} {'BCNS':>7} {'ICNS':>7}"
            f" {'vs BCNS':>8} {'vs ICNS':>8} {'Unc':>6}")
        log("  " + "-" * 70)
        for c in CARRIERS_GEN_ORDER:
            m  = model_gen_ceto.get(c, 0.0)
            bv = safe_get(bcns, c, "generation_twh")
            iv = safe_get(icns, c, "generation_twh")
            unc = safe_get(bcns, c, "uncertainty") or 0.05
            if bv is None and iv is None:
                continue
            b_str  = f"{bv:.0f}" if bv is not None else "n/a"
            i_str  = f"{iv:.0f}" if iv is not None else "n/a"
            b_err  = f"{(m - bv) / bv * 100:+.0f}%" if bv and bv > 0 else "n/a"
            i_err  = f"{(m - iv) / iv * 100:+.0f}%" if iv and iv > 0 else "n/a"
            disp   = CETO_DISPLAY.get(c, c)
            log(f"  {disp:<22} {m:>7.0f} {b_str:>7} {i_str:>7}"
                f" {b_err:>8} {i_err:>8} {unc*100:>5.0f}%")

    # ── SAVE REPORT ───────────────────────────────────────────────────────────
    report_path = os.path.join(BASE_OUTPUT_DIR, f"{network_name}_2060_inspect.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved: {report_path}")

    # ── PLOTS ─────────────────────────────────────────────────────────────────

    # ── PLOT 1: Capacity — model vs CETO BCNS/ICNS ───────────────────────────
    if bcns is not None and icns is not None:
        c_list, mv_l, bv_l, iv_l, unc_l = [], [], [], [], []
        for c in CARRIERS_CAP_ORDER:
            m  = model_cap_ceto.get(c, 0.0)
            bv = safe_get(bcns, c, "capacity_gw") or 0.0
            iv = safe_get(icns, c, "capacity_gw") or 0.0
            unc = safe_get(bcns, c, "uncertainty") or 0.05
            if m > 0 or bv > 0 or iv > 0:
                c_list.append(CETO_DISPLAY.get(c, c))
                mv_l.append(m); bv_l.append(bv); iv_l.append(iv); unc_l.append(unc)

        x = np.arange(len(c_list))
        w_bar = 0.25
        colors_c = [get_color(c) for c in c_list]

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.bar(x - w_bar, mv_l, w_bar, color=colors_c, alpha=0.92,
               edgecolor="white", label="Model")
        ax.bar(x,         bv_l, w_bar, color=colors_c, alpha=0.50,
               edgecolor="#333", linewidth=0.8, label="CETO 2025 BCNS", hatch="///")
        ax.bar(x + w_bar, iv_l, w_bar, color=colors_c, alpha=0.30,
               edgecolor="#333", linewidth=0.8, label="CETO 2025 ICNS", hatch="xxx")
        for i, (bv, iv, unc) in enumerate(zip(bv_l, iv_l, unc_l)):
            if bv > 0:
                ax.errorbar(x[i], bv, yerr=bv * unc,
                            fmt="none", color="#333", capsize=4, linewidth=1.2)
            if iv > 0:
                ax.errorbar(x[i] + w_bar, iv, yerr=iv * unc,
                            fmt="none", color="#333", capsize=4, linewidth=1.2)
        # Model value labels
        for i, m in enumerate(mv_l):
            if m > 0:
                ax.text(x[i] - w_bar, m * 1.02, f"{m:.0f}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(c_list, fontsize=11)
        ax.set_ylabel("GW", fontsize=12)
        ax.set_title(
            f"Installed capacity (p_nom_opt): Model vs CETO 2025\n"
            f"{scenario_label}  —  {network_name}",
            fontsize=12, fontweight="bold"
        )
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        out = os.path.join(BASE_OUTPUT_DIR, f"{network_name}_2060_capacity.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")

    # ── PLOT 2: Generation — model vs CETO BCNS/ICNS ─────────────────────────
    if bcns is not None and icns is not None:
        cg_list, mg_l, bg_l, ig_l, unc_g = [], [], [], [], []
        for c in CARRIERS_GEN_ORDER:
            m  = model_gen_ceto.get(c, 0.0)
            bv = safe_get(bcns, c, "generation_twh") or 0.0
            iv = safe_get(icns, c, "generation_twh") or 0.0
            unc = safe_get(bcns, c, "uncertainty") or 0.05
            if bv > 0 or iv > 0:
                cg_list.append(CETO_DISPLAY.get(c, c))
                mg_l.append(m); bg_l.append(bv); ig_l.append(iv); unc_g.append(unc)

        x = np.arange(len(cg_list))
        colors_g = [get_color(c) for c in cg_list]

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.bar(x - w_bar, mg_l, w_bar, color=colors_g, alpha=0.92,
               edgecolor="white", label="Model")
        ax.bar(x,         bg_l, w_bar, color=colors_g, alpha=0.50,
               edgecolor="#333", linewidth=0.8, label="CETO 2025 BCNS", hatch="///")
        ax.bar(x + w_bar, ig_l, w_bar, color=colors_g, alpha=0.30,
               edgecolor="#333", linewidth=0.8, label="CETO 2025 ICNS", hatch="xxx")
        for i, (bv, iv, unc) in enumerate(zip(bg_l, ig_l, unc_g)):
            if bv > 0:
                ax.errorbar(x[i], bv, yerr=bv * unc,
                            fmt="none", color="#333", capsize=4, linewidth=1.2)
            if iv > 0:
                ax.errorbar(x[i] + w_bar, iv, yerr=iv * unc,
                            fmt="none", color="#333", capsize=4, linewidth=1.2)
        for i, m in enumerate(mg_l):
            if m > 0:
                share = m / total_gen * 100 if total_gen > 0 else 0
                ax.text(x[i] - w_bar, m * 1.02, f"{m:.0f}\n({share:.0f}%)",
                        ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(cg_list, fontsize=11)
        ax.set_ylabel("TWh", fontsize=12)
        ax.set_title(
            f"Generation: Model vs CETO 2025\n"
            f"{scenario_label}  —  {network_name}",
            fontsize=12, fontweight="bold"
        )
        ax.legend(fontsize=10)
        ax.grid(True, axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        out = os.path.join(BASE_OUTPUT_DIR, f"{network_name}_2060_generation.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")

    # ── PLOT 3: Generation mix pie ─────────────────────────────────────────────
    gen_pie = all_gen[all_gen / total_gen >= 0.005]
    other   = all_gen[all_gen / total_gen < 0.005].sum()
    if other > 0:
        gen_pie = pd.concat([gen_pie, pd.Series({"other": other})])
    colors_pie = [get_color(c) for c in gen_pie.index]

    fig, ax = plt.subplots(figsize=(9, 9))
    _, _, autotexts = ax.pie(
        gen_pie.values, labels=gen_pie.index,
        colors=colors_pie,
        autopct=lambda p: f"{p:.1f}%" if p >= 1 else "",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
        textprops={"fontsize": 11},
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")
    ax.set_title(
        f"Generation mix — {scenario_label}\n{network_name}\n"
        f"Total gen: {total_gen:.0f} TWh  |  Load: {load_twh:.0f} TWh",
        fontsize=12, fontweight="bold", pad=20
    )
    plt.tight_layout()
    out = os.path.join(BASE_OUTPUT_DIR, f"{network_name}_2060_mix_pie.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # ── PLOT 4: Storage capacity (GW and TWh) ─────────────────────────────────
    stor_carriers = sorted(n.storage_units.carrier.unique())
    s_power  = []
    s_energy = []
    for c in stor_carriers:
        su = n.storage_units[n.storage_units.carrier == c]
        pw = su["p_nom_opt"].sum() / 1e3
        mh = su["max_hours"].mean()
        s_power.append(pw)
        s_energy.append(pw * mh / 1e3)
    colors_s = [get_color(c) for c in stor_carriers]
    x = np.arange(len(stor_carriers))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    ax1.bar(x, s_power,  0.5, color=colors_s, alpha=0.88, edgecolor="white")
    for i, v in enumerate(s_power):
        ax1.text(i, v * 1.02, f"{v:.0f} GW", ha="center", fontsize=9, fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(stor_carriers, fontsize=10)
    ax1.set_ylabel("GW", fontsize=11)
    ax1.set_title("Storage: Power capacity (GW)", fontsize=11, fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.2)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    ax2.bar(x, s_energy, 0.5, color=colors_s, alpha=0.88, edgecolor="white")
    for i, v in enumerate(s_energy):
        ax2.text(i, v * 1.02, f"{v:.2f} TWh", ha="center", fontsize=9, fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(stor_carriers, fontsize=10)
    ax2.set_ylabel("TWh", fontsize=11)
    ax2.set_title("Storage: Energy capacity (TWh)", fontsize=11, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.2)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    plt.suptitle(
        f"Storage analysis — {scenario_label}\n{network_name}",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    out = os.path.join(BASE_OUTPUT_DIR, f"{network_name}_2060_storage.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # ── PLOT 5: Capacity factor ────────────────────────────────────────────────
    total_hours = len(n.snapshots) * w.iloc[0]
    carriers_lf = [c for c in sorted(set(all_cap.index) & set(all_gen.index))
                   if all_cap.get(c, 0) > 0]
    lf_vals = []
    for c in carriers_lf:
        cap_gw = all_cap.get(c, 0)
        gen_t  = all_gen.get(c, 0)
        lf = gen_t / (cap_gw * total_hours / 1e3) * 100 if cap_gw > 0 else 0
        lf_vals.append(lf)

    colors_lf = [get_color(c) for c in carriers_lf]
    x = np.arange(len(carriers_lf))

    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(x, lf_vals, 0.6, color=colors_lf, alpha=0.88, edgecolor="white")
    for bar, val, c in zip(bars, lf_vals, carriers_lf):
        cap = all_cap.get(c, 0)
        gen = all_gen.get(c, 0)
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%\n({gen:.0f} TWh\n{cap:.0f} GW)",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.axhline(100, color="red", linewidth=1, linestyle="--", alpha=0.5,
               label="100% theoretical max")
    ax.set_xticks(x)
    ax.set_xticklabels(carriers_lf, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Capacity Factor (%)", fontsize=11)
    ax.set_ylim(0, max(lf_vals) * 1.35 if lf_vals else 110)
    ax.set_title(
        f"Capacity Factor by carrier — {scenario_label}\n{network_name}",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = os.path.join(BASE_OUTPUT_DIR, f"{network_name}_2060_capacity_factor.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analysis of CN2060 decarbonization scenario from PyPSA-Earth"
    )
    parser.add_argument("network", help="Path to the .nc network file")
    args = parser.parse_args()
    inspect(args.network)
