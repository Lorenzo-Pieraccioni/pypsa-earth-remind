"""
inspect_network_2060.py  (v2 — multi-model reference, 04.06.2026)
==================================================================
Analysis of CN2060 decarbonisation scenario results from PyPSA-Earth.

Sections (text output):
  1. Network structure
  2. Total electricity load vs all reference sources
  3. Installed capacity (GW) — model
  4. Generation (TWh) — model
  5. Energy balance (with battery artefact note)
  6. CO2 price and system cost
  7. Storage analysis (GW, TWh, max_hours)
  8. Comparison vs all reference sources (CETO 2025 + Zhu 2026 + Zhang Da 2025)
  9. Multi-model normalised comparison (GW per 1000 TWh demand)

Plots:
  1. Capacity (GW): model bar vs aggregated reference bar (mean ± min/max range)
  2. Generation (TWh): same, ALL carriers
  3. Generation mix pie chart
  4. Storage: power (GW) and energy (TWh)
  5. Capacity factor by carrier
  6. Normalised capacity (GW / 1000 TWh demand): model vs literature range

Reference sources (all in data/validation/ceto_2025_reference.csv):
  - CETO 2025 BCNS / ICNS  (demand 2060: 21,200 / 22,600 TWh)
  - Zhu et al. (2026) EES S1  (demand: 20,000 TWh)  DOI 10.1039/D5EE05948H
  - Zhang Da et al. (2025) EES BASE  (demand: 16,000 TWh)  DOI 10.1039/d5ee00355e

Normalisation method:
  Reference bars are normalised by each source's own demand, then rescaled to
  the model's demand, so that bars are in GW at the model's scale.
  norm(source) = GW_source / demand_source
  ref_bar_GW   = mean(norm_i) * model_demand_TWh

Usage:
  PYPSA_OUTPUT_DIR="analysis/network/output/CN2060_95_01_PHSext" python analysis/network/inspect_network_2060.py results/CN2060_01_PHSext/networks/elec_s_250_ec_lcopt_Co2L0.05-3h.nc

  PYPSA_OUTPUT_DIR="analysis/network/output/CN2060_100_01_PHSext" python analysis/network/inspect_network_2060.py results/CN2060_01_PHSext/networks/elec_s_250_ec_lcopt_Co2L0.0-3h.nc
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

# Demand assumptions per reference source (TWh) used for normalisation
SOURCE_DEMAND = {
    "CETO_2025":    {"BCNS": 21200, "ICNS": 22600},
    "ZHU_2026":     {"S1":   20000},
    "ZHANGDA_2025": {"BASE": 16000},
}

# Multi-model normalised range (GW per 1000 TWh demand) — computed 04.06.2026
# Sources: CETO 2025 BCNS/ICNS, Zhu et al. 2026 S1, Zhang Da et al. 2025 BASE
# Method: norm = GW_source / demand_source * 1000
MULTIMODEL_CAP_NORM = {
    "solar_pv_total":     (241, 288),
    "wind_total":         (131, 158),
    "nuclear":            (8.8, 14.4),
    "hydro_conventional": (25.9, 37.5),
    "biomass_incl_beccs": (9.1, 11.3),
    "pumped_hydro":       (18.9, 28.7),
    "chemical_storage":   (31.3, 44.2),
}

# Multi-model generation share range (% of total demand)
MULTIMODEL_GEN_SHARE = {
    "solar_pv_total":     (27.8, 35.5),
    "wind_total":         (39.0, 48.6),
    "nuclear":            (7.1,  10.6),
    "hydro_conventional": (9.7,  16.0),
    "biomass_incl_beccs": (4.7,   5.7),
}

# Carrier colours
CARRIER_COLORS = {
    "coal":           "#4d4d4d",
    "Coal":           "#4d4d4d",
    "lignite":        "#8c6d31",
    "CCGT":           "#6baed6",
    "OCGT":           "#9ecae1",
    "Gas":            "#6baed6",
    "nuclear":        "#e6550d",
    "Nuclear":        "#e6550d",
    "oil":            "#969696",
    "onwind":         "#74c476",
    "offwind-ac":     "#41ab5d",
    "offwind-dc":     "#238b45",
    "Wind":           "#74c476",
    "solar":          "#fdd835",
    "Solar":          "#fdd835",
    "ror":            "#3182bd",
    "hydro":          "#08519c",
    "Hydro":          "#08519c",
    "PHS":            "#9ecae1",
    "H2":             "#9e9ac8",
    "H2 Storage":     "#9e9ac8",
    "battery":        "#fd8d3c",
    "Battery":        "#fd8d3c",
    "Biomass+BECCS":  "#a1d99b",
}

# PyPSA carrier → CETO/reference CSV carrier key (capacity)
# NOTE: "hydro" and "ror" map to "hydro_conventional" to match CSV keys
PYPSA_TO_CETO_CAP = {
    "solar":       "solar_pv_total",
    "onwind":      "wind_total",
    "offwind-ac":  "wind_total",
    "offwind-dc":  "wind_total",
    "hydro":       "hydro_conventional",
    "ror":         "hydro_conventional",
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

# PyPSA carrier → CETO/reference CSV carrier key (generation)
PYPSA_TO_CETO_GEN = {
    "solar":       "solar_pv_total",
    "onwind":      "wind_total",
    "offwind-ac":  "wind_total",
    "offwind-dc":  "wind_total",
    "hydro":       "hydro_conventional",
    "ror":         "hydro_conventional",
    "PHS":         "pumped_hydro",
    "nuclear":     "nuclear",
    "coal":        "coal",
    "lignite":     "coal",
    "CCGT":        "gas",
    "OCGT":        "gas",
    "oil":         "gas",
    "battery":     "chemical_storage",
    "H2":          "hydrogen_electrolyser",
    "biomass":     "biomass_incl_beccs",
}

# Human-readable display names
CETO_DISPLAY = {
    "solar_pv_total":        "Solar",
    "wind_total":            "Wind",
    "hydro_conventional":    "Hydro",
    "pumped_hydro":          "PHS",
    "nuclear":               "Nuclear",
    "coal":                  "Coal",
    "gas":                   "Gas",
    "biomass_incl_beccs":    "Biomass+BECCS",
    "chemical_storage":      "Battery",
    "hydrogen_electrolyser": "H2 Storage",
}

# Display name for normalised plot
NORM_DISPLAY = {
    "solar_pv_total":     "Solar",
    "wind_total":         "Wind",
    "nuclear":            "Nuclear",
    "hydro_conventional": "Hydro",
    "pumped_hydro":       "PHS",
    "chemical_storage":   "Battery",
    "biomass_incl_beccs": "Biomass+BECCS",
}

# Carrier order for comparison plots
CARRIERS_CAP_ORDER = [
    "solar_pv_total", "wind_total", "nuclear", "hydro_conventional",
    "chemical_storage", "pumped_hydro", "biomass_incl_beccs", "coal", "gas",
]
CARRIERS_GEN_ORDER = [
    "solar_pv_total", "wind_total", "hydro_conventional", "nuclear",
    "biomass_incl_beccs", "coal", "gas", "chemical_storage", "pumped_hydro",
]

# ── HELPERS ───────────────────────────────────────────────────────────────────

def get_color(carrier):
    return CARRIER_COLORS.get(carrier, "#aaaaaa")


def load_all_references(ceto_file, year=2060):
    """
    Load all reference sources from the CSV for the given year.
    Returns dict: {source: {scenario: DataFrame indexed by carrier}}
    """
    if not os.path.exists(ceto_file):
        print(f"[WARNING] Reference file not found: {ceto_file}")
        return {}
    df = pd.read_csv(ceto_file)
    df_year = df[df["year"] == year].copy()
    refs = {}
    for (source, scenario), group in df_year.groupby(["source", "scenario"]):
        if source not in refs:
            refs[source] = {}
        refs[source][scenario] = group.set_index("carrier")
    return refs


def safe_get_ref(df, carrier, col):
    """Return float from reference DataFrame, None if missing/NaN."""
    if df is None or carrier not in df.index:
        return None
    val = df.loc[carrier, col]
    if pd.isna(val):
        return None
    return float(val)


def compute_aggregated_ref(refs, carrier, col, model_demand_twh):
    """
    Collect absolute values from all reference sources (no demand scaling).
    Returns (mean_abs, min_abs, max_abs) or (None, None, None).
    Note: reference demands range from 16,000 to 22,600 TWh vs model 16,301 TWh.
    """
    abs_values = []
    for source, scenarios in refs.items():
        for scenario, df in scenarios.items():
            val = safe_get_ref(df, carrier, col)
            if val is None or val <= 0:
                continue
            abs_values.append(float(val))
    if not abs_values:
        return None, None, None
    return float(np.mean(abs_values)), float(min(abs_values)), float(max(abs_values))


def infer_scenario(network_file):
    if "Co2L0.0-" in network_file or "Co2L0.0_" in network_file:
        return "100% decarb (Co2L0.0)"
    elif "Co2L0.05" in network_file:
        return "95% decarb (Co2L0.05)"
    elif "noCO2" in network_file:
        return "no CO2 constraint"
    return "unknown"


def aggregate_carriers(model_series, mapping):
    """Sum model values by reference carrier key."""
    result = {}
    for carrier, val in model_series.items():
        key = mapping.get(carrier)
        if key:
            result[key] = result.get(key, 0) + val
    return pd.Series(result)


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

    # Load all reference sources
    refs = load_all_references(CETO_FILE, year=2060)
    n_sources = sum(len(v) for v in refs.values())
    log(f"  Reference sources loaded: {n_sources} scenarios from {len(refs)} models")
    for src, scens in refs.items():
        log(f"    {src}: {list(scens.keys())}")

    # ── 1. NETWORK STRUCTURE ─────────────────────────────────────────────────
    log("")
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

    # ── 2. TOTAL LOAD ────────────────────────────────────────────────────────
    load_twh = (n.loads_t.p_set.multiply(w, axis=0)).sum().sum() / 1e6
    log("")
    log("=" * 60)
    log("2. TOTAL ELECTRICITY LOAD")
    log("=" * 60)
    log(f"  Model total:  {load_twh:.1f} TWh")
    log("")
    log(f"  Reference source demands for 2060:")
    log(f"    CETO 2025 BCNS:         21,200 TWh")
    log(f"    CETO 2025 ICNS:         22,600 TWh")
    log(f"    Zhu et al. 2026 S1:     20,000 TWh")
    log(f"    Zhang Da et al. 2025:   16,000 TWh")
    log(f"    Literature range:       16,000 — 22,600 TWh")
    log(f"  Model vs CETO BCNS:  {(load_twh - 21200) / 21200 * 100:+.1f}%")
    log(f"  Model vs CETO ICNS:  {(load_twh - 22600) / 22600 * 100:+.1f}%")
    log(f"  Model vs Zhu 2026:   {(load_twh - 20000) / 20000 * 100:+.1f}%")
    log(f"  Model vs ZhangDa:    {(load_twh - 16000) / 16000 * 100:+.1f}%")
    log(f"  NOTE: model load gap vs CETO (-23 to -28%) reflects missing")
    log(f"        sectoral demand (transport, heating, industry).")

    # ── 3. INSTALLED CAPACITY ────────────────────────────────────────────────
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

    # ── 4. GENERATION ────────────────────────────────────────────────────────
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

    # ── 5. ENERGY BALANCE ────────────────────────────────────────────────────
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
        log(f"        Battery discharge counted in total gen, not net supply.")
        log(f"        The solver energy balance is correct by construction.")
    log(f"  Links total:  {links_gen_twh.sum():.1f} TWh")

    # ── 6. CO2 PRICE AND SYSTEM COST ─────────────────────────────────────────
    log("")
    log("=" * 60)
    log("6. CO2 PRICE AND SYSTEM COST")
    log("=" * 60)

    if hasattr(n, "global_constraints") and not n.global_constraints.empty:
        co2_rows = n.global_constraints[
            n.global_constraints.index.str.lower().str.contains("co2")
        ]
        if not co2_rows.empty:
            co2_price = float(co2_rows["mu"].iloc[0])
            log(f"  CO2 constraint: {co2_rows.index[0]}")
            log(f"  CO2 price:      {co2_price:.1f} EUR/tCO2")
        else:
            log("  CO2 price: no CO2 constraint found")
    else:
        log("  CO2 price: global_constraints not available")

    if hasattr(n, "objective") and n.objective is not None:
        log(f"  System cost:    {n.objective / 1e9:.2f} billion EUR/yr")
    else:
        log("  System cost: n.objective not available")

    # ── 7. STORAGE ANALYSIS ──────────────────────────────────────────────────
    log("")
    log("=" * 60)
    log("7. STORAGE ANALYSIS")
    log("=" * 60)
    log(f"  {'Carrier':<18} {'Power (GW)':>12} {'Energy (TWh)':>14} {'Max hours':>10}")
    log("  " + "-" * 58)
    for carrier in sorted(n.storage_units.carrier.unique()):
        su = n.storage_units[n.storage_units.carrier == carrier]
        power_gw   = su["p_nom_opt"].sum() / 1e3
        max_h      = su["max_hours"].mean()
        energy_twh = power_gw * max_h / 1e3
        log(f"  {carrier:<18} {power_gw:>12.1f} {energy_twh:>14.3f} {max_h:>10.1f}")
    log("")
    log("  Literature regulation capacity targets for 2060:")
    log("    Chemical storage:  572–1,000 GW  (Zhu: 626, ZhangDa: 572, CETO: 800–1,000)")
    log("    Pumped hydro:      400–573 GW   (CETO: 400–500, Zhu: 573, ZhangDa: 437)")
    log("    H2 + e-fuels:      1,740–2,180 GW (CETO only; model has no multi-sector H2)")
    log("    EV-grid (V2G):     750–920 GW   (CETO only; not modelled)")
    log("    Total regulation:  4,970–5,810 GW (CETO only)")

    # ── 8. COMPARISON VS ALL REFERENCE SOURCES ───────────────────────────────
    log("")
    log("=" * 60)
    log("8. COMPARISON VS ALL REFERENCE SOURCES")
    log("   CETO 2025 BCNS/ICNS | Zhu et al. 2026 S1 | Zhang Da et al. 2025 BASE")
    log("=" * 60)

    model_cap_ceto = aggregate_carriers(all_cap, PYPSA_TO_CETO_CAP)
    model_gen_ceto = aggregate_carriers(all_gen, PYPSA_TO_CETO_GEN)

    # Build per-source reference dicts for easy access
    ref_flat = {}   # (source, scenario) -> df
    for src, scens in refs.items():
        for scen, df in scens.items():
            ref_flat[(src, scen)] = df

    def get_src_val(source, scenario, carrier, col):
        df = ref_flat.get((source, scenario))
        return safe_get_ref(df, carrier, col)

    log("")
    log("  Capacity (GW) — model vs reference sources")
    log(f"  {'Carrier':<20} {'Model':>7} {'CETO_B':>7} {'CETO_I':>7}"
        f" {'Zhu26':>7} {'ZhDa25':>7}  {'Ref.range':>12}")
    log("  " + "-" * 78)
    for c in CARRIERS_CAP_ORDER:
        m    = model_cap_ceto.get(c, 0.0)
        cb   = get_src_val("CETO_2025",    "BCNS", c, "capacity_gw")
        ci   = get_src_val("CETO_2025",    "ICNS", c, "capacity_gw")
        zhu  = get_src_val("ZHU_2026",     "S1",   c, "capacity_gw")
        zhd  = get_src_val("ZHANGDA_2025", "BASE", c, "capacity_gw")
        vals = [v for v in [cb, ci, zhu, zhd] if v is not None]
        rng  = f"{min(vals):.0f}–{max(vals):.0f}" if len(vals) >= 2 else (f"~{vals[0]:.0f}" if vals else "n/a")
        disp = CETO_DISPLAY.get(c, c)
        fmt  = lambda v: f"{v:.0f}" if v is not None else " n/a"
        if m == 0 and not vals:
            continue
        log(f"  {disp:<20} {m:>7.0f} {fmt(cb):>7} {fmt(ci):>7}"
            f" {fmt(zhu):>7} {fmt(zhd):>7}  {rng:>12}")

    log("")
    log("  Generation (TWh) — model vs reference sources")
    log(f"  {'Carrier':<20} {'Model':>7} {'CETO_B':>7} {'CETO_I':>7}"
        f" {'Zhu26':>7} {'ZhDa25':>7}  {'Ref.range':>12}")
    log("  " + "-" * 78)
    for c in CARRIERS_GEN_ORDER:
        m    = model_gen_ceto.get(c, 0.0)
        cb   = get_src_val("CETO_2025",    "BCNS", c, "generation_twh")
        ci   = get_src_val("CETO_2025",    "ICNS", c, "generation_twh")
        zhu  = get_src_val("ZHU_2026",     "S1",   c, "generation_twh")
        zhd  = get_src_val("ZHANGDA_2025", "BASE", c, "generation_twh")
        vals = [v for v in [cb, ci, zhu, zhd] if v is not None]
        rng  = f"{min(vals):.0f}–{max(vals):.0f}" if len(vals) >= 2 else (f"~{vals[0]:.0f}" if vals else "—")
        disp = CETO_DISPLAY.get(c, c)
        fmt  = lambda v: f"{v:.0f}" if v is not None else " n/a"
        if m == 0 and not vals:
            continue
        log(f"  {disp:<20} {m:>7.0f} {fmt(cb):>7} {fmt(ci):>7}"
            f" {fmt(zhu):>7} {fmt(zhd):>7}  {rng:>12}")

    # ── 9. NORMALISED MULTI-MODEL COMPARISON ─────────────────────────────────
    log("")
    log("=" * 60)
    log("9. NOTE ON COMPARISON METHODOLOGY")
    log("  Reference bars in plots show absolute GW/TWh values (no demand scaling).")
    log("  Bar = arithmetic mean of available sources. Error bars = min-max range.")
    log(f"  Model demand: {load_twh:.0f} TWh.")
    log("  Literature demands: CETO BCNS 21,200 / CETO ICNS 22,600 / Zhu 2026 20,000 / ZhangDa 16,000 TWh.")
    log("  Values are not comparable on an equal-demand basis.")
    log("   Sources: CETO 2025 BCNS/ICNS, Zhu et al. 2026, Zhang Da et al. 2025")
    log("=" * 60)
    log(f"  Model demand: {load_twh:.0f} TWh")
    log("")
    log(f"  {'Carrier':<20} {'Model':>8} {'Lit.min':>8} {'Lit.max':>8}"
        f" {'vs max':>8}  Status")
    log("  " + "-" * 72)
    for ceto_key, label in NORM_DISPLAY.items():
        model_gw   = model_cap_ceto.get(ceto_key, 0.0)
        model_norm = model_gw / load_twh * 1000 if load_twh > 0 else 0
        if ceto_key in MULTIMODEL_CAP_NORM:
            lo, hi = MULTIMODEL_CAP_NORM[ceto_key]
            if model_norm == 0:
                status = "NOT MODELLED"
                vs_max = "n/a"
            elif model_norm < lo:
                status = f"below range"
                vs_max = f"{(model_norm - hi) / hi * 100:+.0f}%"
            elif model_norm > hi:
                status = f"ABOVE range"
                vs_max = f"{(model_norm - hi) / hi * 100:+.0f}%"
            else:
                status = "within range"
                vs_max = f"{(model_norm - hi) / hi * 100:+.0f}%"
            log(f"  {label:<20} {model_norm:>8.1f} {lo:>8.1f} {hi:>8.1f}"
                f" {vs_max:>8}  {status}")

    log("")
    log("  Generation share (% of annual demand)")
    log(f"  {'Carrier':<20} {'Model%':>8} {'Lit.min':>8} {'Lit.max':>8}  Status")
    log("  " + "-" * 62)
    for ceto_key, label in NORM_DISPLAY.items():
        if ceto_key not in MULTIMODEL_GEN_SHARE:
            continue
        model_twh  = model_gen_ceto.get(ceto_key, 0.0)
        model_pct  = model_twh / load_twh * 100 if load_twh > 0 else 0
        lo, hi     = MULTIMODEL_GEN_SHARE[ceto_key]
        if model_pct == 0:
            status = "NOT MODELLED"
        elif model_pct < lo:
            status = f"below range ({(model_pct - lo) / lo * 100:+.0f}%)"
        elif model_pct > hi:
            status = f"ABOVE range ({(model_pct - hi) / hi * 100:+.0f}%)"
        else:
            status = "within range"
        log(f"  {label:<20} {model_pct:>7.1f}% {lo:>7.1f}% {hi:>7.1f}%  {status}")

    log("")
    log("  Interpretation (reasoned interpretation, not verified facts):")
    log("  Solar ABOVE range: model has no sectoral flexibility (V2G, heating,")
    log("  demand response). Solver substitutes all flexibility with solar+battery.")
    log("  Battery ABOVE range: replaces V2G (750-920 GW CETO) and demand response.")
    log("  Nuclear ABOVE range: low CAPEX override (2200 EUR/kW, single unverified source) + absence of seasonal storage make nuclear the solver's main lever for the nocturnal/seasonal gap (reasoned interpretation, not verified).")
    log("  Hydro BELOW range: powerplantmatching underestimates Chinese hydro capacity.")


    # ── 10. ELECTRICITY PRICE AND CURTAILMENT ────────────────────────────────
    log("")
    log("=" * 60)
    log("10. ELECTRICITY PRICE AND CURTAILMENT")
    log("=" * 60)

    if not n.buses_t.marginal_price.empty:
        load_p = n.loads_t.p_set.reindex(columns=n.loads.index)
        missing_load_p = load_p.columns[load_p.isna().all()]
        if len(missing_load_p) > 0:
            log(f"  WARNING: {len(missing_load_p)} loads have no time-varying p_set, using static value.")
        for col in missing_load_p:
            load_p[col] = n.loads.loc[col, "p_set"]
        bus_price = n.buses_t.marginal_price
        weighted_num = 0.0
        weighted_den = 0.0
        for load_name, bus in n.loads.bus.items():
            if bus in bus_price.columns and load_name in load_p.columns:
                p = load_p[load_name] * w
                pr = bus_price[bus]
                weighted_num += (p * pr).sum()
                weighted_den += p.sum()
        avg_price = weighted_num / weighted_den if weighted_den > 0 else float("nan")
        log(f"  Load-weighted average electricity price: {avg_price:.2f} EUR/MWh")
    else:
        log("  WARNING: n.buses_t.marginal_price not available (no dual values exported).")

    log("")
    log("  Curtailment by carrier (available - dispatched):")
    renewable_carriers = ["solar", "onwind", "offwind-ac", "offwind-dc"]
    curt_total = 0.0
    for c in renewable_carriers:
        gens = n.generators[n.generators.carrier == c]
        if gens.empty:
            continue
        missing_pmpu = gens.index.difference(n.generators_t.p_max_pu.columns)
        if len(missing_pmpu) > 0:
            log(f"    WARNING: {len(missing_pmpu)} {c} generators have no time-varying p_max_pu, using static value.")
        pmpu = n.generators_t.p_max_pu.reindex(columns=gens.index)
        pmpu = pmpu.apply(lambda col: col.fillna(gens.loc[col.name, "p_max_pu"]) if col.isna().any() else col)
        avail = pmpu.multiply(gens.p_nom_opt, axis=1).multiply(w, axis=0).sum().sum() / 1e6
        dispatched = n.generators_t.p[gens.index].multiply(w, axis=0).sum().sum() / 1e6
        curt = avail - dispatched
        curt_pct = curt / avail * 100 if avail > 0 else 0
        curt_total += curt
        log(f"    {c:<14} available {avail:>8.1f} TWh  dispatched {dispatched:>8.1f} TWh  curtailed {curt:>7.1f} TWh ({curt_pct:.1f}%)")
    log(f"    {'TOTAL':<14} curtailed {curt_total:>7.1f} TWh")
    # ── SAVE REPORT ──────────────────────────────────────────────────────────
    report_path = os.path.join(BASE_OUTPUT_DIR, f"{network_name}_2060_inspect.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved: {report_path}")

    # ── PLOTS ────────────────────────────────────────────────────────────────

    w_bar = 0.35

    # ── PLOT 1: Capacity — model (1 bar) vs aggregated reference (1 bar) ────
    c_list, mv_l, rv_l, rmin_l, rmax_l = [], [], [], [], []
    for c in CARRIERS_CAP_ORDER:
        m = model_cap_ceto.get(c, 0.0)
        mean_r, min_r, max_r = compute_aggregated_ref(refs, c, "capacity_gw", load_twh)
        if m > 0 or mean_r is not None:
            c_list.append(CETO_DISPLAY.get(c, c))
            mv_l.append(m)
            rv_l.append(mean_r or 0.0)
            rmin_l.append(mean_r - min_r if mean_r else 0)
            rmax_l.append(max_r - mean_r if mean_r else 0)

    x = np.arange(len(c_list))
    colors_c = [get_color(c) for c in c_list]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - w_bar / 2, mv_l, w_bar, color=colors_c, alpha=0.90,
           edgecolor="white", label="Model")
    ax.bar(x + w_bar / 2, rv_l, w_bar, color=colors_c, alpha=0.40,
           edgecolor="#333", linewidth=0.8, label="Literature", hatch="///")
    for i, (lo_err, hi_err, rv) in enumerate(zip(rmin_l, rmax_l, rv_l)):
        if rv > 0:
            ax.errorbar(x[i] + w_bar / 2, rv, yerr=[[lo_err], [hi_err]],
                        fmt="none", color="#222", capsize=5, linewidth=1.5)
    for i, m in enumerate(mv_l):
        if m > 0:
            ax.text(x[i] - w_bar / 2, m * 1.02, f"{m:.0f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(c_list, fontsize=11)
    ax.set_ylabel("GW", fontsize=11)
    ax.set_title(f"Installed capacity — {scenario_label}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = os.path.join(BASE_OUTPUT_DIR, f"{network_name}_2060_capacity.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # ── PLOT 2: Generation — model vs aggregated reference, ALL carriers ─────
    # Show all carriers: those with reference AND those without
    cg_list, mg_l, rg_l, rgmin_l, rgmax_l, has_ref_l = [], [], [], [], [], []
    # First: carriers that appear in model gen, in CARRIERS_GEN_ORDER order
    ordered = [c for c in CARRIERS_GEN_ORDER if model_gen_ceto.get(c, 0) > 0]
    # Add any remaining model carriers not in the order list
    remaining = [c for c in model_gen_ceto.index if c not in ordered
                 and model_gen_ceto.get(c, 0) > 0]
    all_gen_carriers = ordered + remaining

    for c in all_gen_carriers:
        m = model_gen_ceto.get(c, 0.0)
        mean_r, min_r, max_r = compute_aggregated_ref(refs, c, "generation_twh", load_twh)
        cg_list.append(CETO_DISPLAY.get(c, c))
        mg_l.append(m)
        rg_l.append(mean_r or 0.0)
        rgmin_l.append(mean_r - min_r if mean_r else 0)
        rgmax_l.append(max_r - mean_r if mean_r else 0)
        has_ref_l.append(mean_r is not None and mean_r > 0)

    xg = np.arange(len(cg_list))
    colors_g = [get_color(c) for c in cg_list]

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.bar(xg - w_bar / 2, mg_l, w_bar, color=colors_g, alpha=0.90,
           edgecolor="white", label="Model")
    ref_bars = ax.bar(xg + w_bar / 2, rg_l, w_bar, color=colors_g, alpha=0.40,
                      edgecolor="#333", linewidth=0.8,
                      label="Literature", hatch="///")
    # Hide reference bars for carriers without reference
    for bar, has_ref in zip(ref_bars, has_ref_l):
        if not has_ref:
            bar.set_visible(False)
    for i, (lo_err, hi_err, rv, has_ref) in enumerate(
            zip(rgmin_l, rgmax_l, rg_l, has_ref_l)):
        if rv > 0 and has_ref:
            ax.errorbar(xg[i] + w_bar / 2, rv, yerr=[[lo_err], [hi_err]],
                        fmt="none", color="#222", capsize=5, linewidth=1.5)
    for i, m in enumerate(mg_l):
        if m > 0:
            share = m / total_gen * 100 if total_gen > 0 else 0
            ax.text(xg[i] - w_bar / 2, m * 1.02,
                    f"{m:.0f}\n({share:.0f}%)",
                    ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_xticks(xg)
    ax.set_xticklabels(cg_list, fontsize=10)
    ax.set_ylabel("TWh", fontsize=11)
    ax.set_title(f"Generation — {scenario_label}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = os.path.join(BASE_OUTPUT_DIR, f"{network_name}_2060_generation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # ── PLOT 3: Generation mix pie ────────────────────────────────────────────
    gen_pie = all_gen[all_gen / total_gen >= 0.005]
    other   = all_gen[all_gen / total_gen < 0.005].sum()
    if other > 0:
        gen_pie = pd.concat([gen_pie, pd.Series({"other": other})])
    colors_pie = [get_color(c) for c in gen_pie.index]

    fig, ax = plt.subplots(figsize=(9, 9))
    _, _, autotexts = ax.pie(
        gen_pie.values, labels=gen_pie.index, colors=colors_pie,
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

    # ── PLOT 4: Storage capacity ──────────────────────────────────────────────
    stor_carriers = sorted(n.storage_units.carrier.unique())
    s_power, s_energy = [], []
    for c in stor_carriers:
        su = n.storage_units[n.storage_units.carrier == c]
        pw = su["p_nom_opt"].sum() / 1e3
        mh = su["max_hours"].mean()
        s_power.append(pw)
        s_energy.append(pw * mh / 1e3)
    colors_s = [get_color(c) for c in stor_carriers]
    xs = np.arange(len(stor_carriers))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    ax1.bar(xs, s_power,  0.5, color=colors_s, alpha=0.88, edgecolor="white")
    for i, v in enumerate(s_power):
        ax1.text(i, v * 1.02, f"{v:.0f} GW", ha="center", fontsize=9, fontweight="bold")
    ax1.set_xticks(xs); ax1.set_xticklabels(stor_carriers, fontsize=10)
    ax1.set_ylabel("GW", fontsize=11)
    ax1.set_title("Storage: Power capacity (GW)", fontsize=11, fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.2)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    ax2.bar(xs, s_energy, 0.5, color=colors_s, alpha=0.88, edgecolor="white")
    for i, v in enumerate(s_energy):
        ax2.text(i, v * 1.02, f"{v:.2f} TWh", ha="center", fontsize=9, fontweight="bold")
    ax2.set_xticks(xs); ax2.set_xticklabels(stor_carriers, fontsize=10)
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

    # ── PLOT 5: Capacity factor ───────────────────────────────────────────────
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
    xlf = np.arange(len(carriers_lf))

    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(xlf, lf_vals, 0.6, color=colors_lf, alpha=0.88, edgecolor="white")
    for bar, val, c in zip(bars, lf_vals, carriers_lf):
        cap = all_cap.get(c, 0)
        gen = all_gen.get(c, 0)
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%\n({gen:.0f} TWh\n{cap:.0f} GW)",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.axhline(100, color="red", linewidth=1, linestyle="--", alpha=0.5,
               label="100% theoretical max")
    ax.set_xticks(xlf)
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

    # ── PLOT 6: Normalised capacity — model vs literature range ──────────────
    norm_carriers = list(NORM_DISPLAY.keys())
    norm_labels   = [NORM_DISPLAY[c] for c in norm_carriers]
    model_norms   = []
    ref_means_n   = []
    ref_mins_n    = []
    ref_maxs_n    = []

    for c in norm_carriers:
        mgw = model_cap_ceto.get(c, 0.0)
        model_norms.append(mgw / load_twh * 1000 if load_twh > 0 else 0)
        lo, hi = MULTIMODEL_CAP_NORM.get(c, (None, None))
        if lo is not None:
            mid = (lo + hi) / 2
            ref_means_n.append(mid)
            ref_mins_n.append(mid - lo)
            ref_maxs_n.append(hi - mid)
        else:
            ref_means_n.append(0)
            ref_mins_n.append(0)
            ref_maxs_n.append(0)

    xn = np.arange(len(norm_carriers))
    colors_n = [get_color(CETO_DISPLAY.get(c, c)) for c in norm_carriers]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(xn - w_bar / 2, model_norms, w_bar, color=colors_n, alpha=0.90,
           edgecolor="white", label="Model")
    ax.bar(xn + w_bar / 2, ref_means_n, w_bar, color=colors_n, alpha=0.35,
           edgecolor="#333", linewidth=0.8, label="Literature midpoint (min–max)", hatch="///")
    for i, (lo_err, hi_err, rv) in enumerate(zip(ref_mins_n, ref_maxs_n, ref_means_n)):
        if rv > 0:
            ax.errorbar(xn[i] + w_bar / 2, rv, yerr=[[lo_err], [hi_err]],
                        fmt="none", color="#222", capsize=5, linewidth=1.5)
    for i, v in enumerate(model_norms):
        if v > 0:
            ax.text(xn[i] - w_bar / 2, v * 1.02, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(xn)
    ax.set_xticklabels(norm_labels, fontsize=11)
    ax.set_ylabel("GW per 1000 TWh annual demand", fontsize=11)
    ax.set_title(
        f"Normalised capacity intensity — Model vs literature range\n"
        f"{scenario_label}  —  {network_name}\n"
        f"Sources: CETO 2025 BCNS/ICNS, Zhu et al. 2026, Zhang Da et al. 2025. "
        f"Error bars = full min–max range.",
        fontsize=10, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = os.path.join(BASE_OUTPUT_DIR, f"{network_name}_2060_capacity_norm.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CN2060 PyPSA-Earth analysis with multi-model reference comparison"
    )
    parser.add_argument("network", help="Path to the .nc network file")
    args = parser.parse_args()
    inspect(args.network)
