"""
inspect_network.py
==================
Ispezione rapida della struttura e dei risultati principali di una rete PyPSA-Earth.

Stampa a schermo (e salva in un file .txt) le informazioni principali:
  - struttura della rete (bus, generatori, linee, links, carichi, storage, snapshots)
  - carico totale annuo (TWh)
  - capacità installata per carrier (GW), inclusi storage units
  - generazione per carrier (TWh) con quota percentuale
  - energy balance check (generazione vs carico)

Produce 4 grafici con confronto Ember:
  - Plot 1: capacità installata modello vs Ember (GW)
  - Plot 2: generazione modello vs Ember (TWh con quota %)
  - Plot 3: pie chart mix energetico del modello
  - Plot 4: capacità vs generazione doppio asse

L'anno di riferimento Ember viene inferito automaticamente dal path del file
(CN2020 -> 2020, CN2025 -> 2024 proxy, CN2060 -> confronto Ember omesso).

Uso:
  python analysis/network/inspect_network.py results/networks/geospatial_v2/CN2020_Admin2/elec_s_250_ec_lcopt_6h.nc
  python analysis/network/inspect_network.py results/networks/fix-hydro/CN2025_Admin2/elec_s_250_ec_lcopt_6h.nc
  python analysis/network/inspect_network.py results/networks/fix-hydro/CN2060_Admin2/elec_s_250_ec_lcopt_6h.nc
"""

import argparse
import os
import re
import pypsa
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── PARAMETERS ────────────────────────────────────────────────────────────────

BASE_OUTPUT_DIR = "analysis/network/output"
EMBER_FILE      = "data/validation/ember_CN_2025.xlsx"
IRENA_FILE      = "data/validation/irena_CN_2025.xlsx"

CARRIER_COLORS = {
    "coal":         "#4d4d4d",
    "lignite":      "#8c6d31",
    "CCGT":         "#6baed6",
    "OCGT":         "#9ecae1",
    "nuclear":      "#e6550d",
    "oil":          "#969696",
    "onwind":       "#74c476",
    "offwind-ac":   "#41ab5d",
    "offwind-dc":   "#238b45",
    "solar":        "#fdd835",
    "ror":          "#3182bd",
    "hydro":        "#08519c",
    "PHS":          "#6baed6",
    "H2":           "#9e9ac8",
    "battery":      "#fd8d3c",
    "Wind":         "#74c476",
    "Solar":        "#fdd835",
    "Hydro":        "#08519c",
    "Coal":         "#4d4d4d",
    "Gas":          "#6baed6",
    "Nuclear":      "#e6550d",
    "Other Fossil": "#969696",
    "Bioenergy":    "#a1d99b",
}

PYPSA_TO_EMBER = {
    "coal":       "Coal",
    "lignite":    "Coal",
    "CCGT":       "Gas",
    "OCGT":       "Gas",
    "nuclear":    "Nuclear",
    "oil":        "Other Fossil",
    "onwind":     "Wind",
    "offwind-ac": "Wind",
    "offwind-dc": "Wind",
    "solar":      "Solar",
    "hydro":      "Hydro",
    "ror":        "Hydro",
    "PHS":        "PHS",
}

PYPSA_TO_DISPLAY_CAP = {
    "solar":      "Solar",
    "onwind":     "Wind",
    "offwind-ac": "Wind",
    "offwind-dc": "Wind",
    "hydro":      "Hydro",
    "ror":        "Hydro",
    "PHS":        "PHS",
    "nuclear":    "Nuclear",
    "coal":       "Coal",
    "lignite":    "Coal",
    "CCGT":       "Gas",
    "OCGT":       "Gas",
    "oil":        "Oil",
}

def get_color(carrier):
    return CARRIER_COLORS.get(carrier, "#aaaaaa")

def infer_year(network_file):
    match = re.search(r"CN(20\d{2})", network_file)
    if match:
        year = int(match.group(1))
        if year == 2025:
            return 2025, 2025
        if year == 2060:
            return 2060, None
        return year, year
    return None, None

def load_ember(ember_file, ember_year):
    if not os.path.exists(ember_file):
        print(f"[WARNING] Ember file not found: {ember_file}")
        return None, None
    df = pd.read_excel(ember_file)
    cn = df[(df["ISO 3 code"] == "CHN") & (df["Year"] == ember_year)]
    cap_ember = (
        cn[(cn["Category"] == "Capacity") & (cn["Subcategory"] == "Fuel")]
        .set_index("Variable")["Value"]
    )
    gen_ember = (
        cn[
            (cn["Category"] == "Electricity generation")
            & (cn["Subcategory"] == "Fuel")
            & (cn["Unit"] == "TWh")
        ]
        .set_index("Variable")["Value"]
    )
    return cap_ember, gen_ember
def load_irena(irena_file, year):
    if not os.path.exists(irena_file):
        print(f"[WARNING] IRENA file not found: {irena_file}")
        return None
    df = pd.read_excel(irena_file, header=0)
    df.columns = ['Country', 'Technology', 'Grid', 'Year', 'Value_MW']
    df = df.ffill()
    df = df[(df['Year'] == year) & (df['Grid'] == 'OnGrid')]
    df['Value_MW'] = pd.to_numeric(df['Value_MW'], errors='coerce')
    IRENA_TO_DISPLAY = {
        'Solar photovoltaic':   'Solar',
        'Onshore wind energy':  'Wind',
        'Offshore wind energy': 'Wind',
        'Renewable hydropower': 'Hydro',
        'Mixed hydropower':     'Hydro',
        'Pumped hydro':          'PHS',
        'Nuclear energy':       'Nuclear',
        'Coal':                 'Coal',
        'Natural gas':          'Gas',
        'Oil':                  'Oil',
    }
    result = {}
    for tech, display_name in IRENA_TO_DISPLAY.items():
        val = df[df['Technology'] == tech]['Value_MW'].sum()
        if val > 0:
            result[display_name] = result.get(display_name, 0) + val / 1e3
    return pd.Series(result)

def aggregate_to_ember(model_series, mapping):
    result = {}
    for carrier, val in model_series.items():
        key = mapping.get(carrier)
        if key:
            result[key] = result.get(key, 0) + val
    return pd.Series(result)

# ─────────────────────────────────────────────────────────────────────────────

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

    model_year, ember_year = infer_year(network_file)
    log(f"  Model year:   {model_year}  |  Ember reference year: {ember_year}")

    irena_cap = None
    gen_ember = None
    if ember_year:
        _, gen_ember = load_ember(EMBER_FILE, ember_year)
        irena_cap = load_irena(IRENA_FILE, ember_year)
        if irena_cap is not None:
            log(f"  Capacity loaded from IRENA for year {ember_year}")
        if gen_ember is not None:
            log(f"  Ember data loaded for year {ember_year}")

    ember_label = f"Ember {ember_year}" if ember_year else "Ember (n/a)"

    # 1. Network structure
    log("=" * 60)
    log("1. NETWORK STRUCTURE")
    log("=" * 60)
    log(f"  File:         {network_file}")
    log(f"  Buses:        {len(n.buses)}")
    log(f"  Generators:   {len(n.generators)}")
    log(f"  Lines:        {len(n.lines)}")
    log(f"  Links:        {len(n.links)}")
    log(f"  Loads:        {len(n.loads)}")
    log(f"  Storage units:{len(n.storage_units)}")
    log(f"  Snapshots:    {len(n.snapshots)}")
    log(f"  Period:       {n.snapshots[0]} -> {n.snapshots[-1]}")
    log(f"  Resolution:   {w.iloc[0]:.0f}h per timestep")

    # 2. Total load
    load_twh = (n.loads_t.p_set.multiply(w, axis=0)).sum().sum() / 1e6
    log("")
    log("=" * 60)
    log("2. TOTAL ELECTRICITY LOAD")
    log("=" * 60)
    log(f"  Model total:  {load_twh:.1f} TWh")

    # 3. Installed capacity
    cap = n.generators.groupby("carrier")["p_nom_opt"].sum() / 1e3
    cap = cap[cap.index != "load shedding"].sort_values(ascending=False)
    stor = n.storage_units.groupby("carrier")["p_nom_opt"].sum() / 1e3
    all_cap = pd.concat([cap, stor]).sort_values(ascending=False)

    log("")
    log("=" * 60)
    log("3. INSTALLED CAPACITY BY CARRIER (GW)")
    log("=" * 60)
    for carrier, val in cap.items():
        log(f"    {carrier:<18} {val:>8.1f} GW")
    log(f"    {'TOTAL':<18} {cap.sum():>8.1f} GW")
    log("  Storage units:")
    for carrier, val in stor.items():
        log(f"    {carrier:<18} {val:>8.1f} GW")

    # 4. Generation
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
        log("  Links (transmission only):")
        for carrier, val in links_gen_twh.items():
            log(f"    {carrier:<18} {val:>8.1f} TWh")
    log(f"  {'TOTAL':<18} {total_gen:>8.1f} TWh")

    # 5. Energy balance
    balance_err = (total_gen - load_twh) / load_twh * 100 if load_twh > 0 else float("nan")
    log("")
    log("=" * 60)
    log("5. ENERGY BALANCE CHECK")
    log("=" * 60)
    log(f"  Load:         {load_twh:.1f} TWh")
    log(f"  Gen (total):  {total_gen:.1f} TWh")
    log(f"  Error:        {balance_err:.2f}%")
    log(f"  Links total:  {links_gen_twh.sum():.1f} TWh")

    # 6. Comparison vs reference data
    log("")
    log("=" * 60)
    log("6. COMPARISON VS REFERENCE DATA")
    log("=" * 60)
    if irena_cap is not None:
        model_cap_display = aggregate_to_ember(all_cap, PYPSA_TO_DISPLAY_CAP)
        log(f"  Capacity (GW) — model vs IRENA {ember_year}")
        log(f"  {'Carrier':<20} {'Model':>10} {'IRENA':>10} {'Error':>8}")
        log("  " + "-" * 52)
        for c in sorted(set(model_cap_display.index) | set(irena_cap.index)):
            m = model_cap_display.get(c, 0.0)
            r = irena_cap.get(c, 0.0)
            if r > 0:
                err = (m - r) / r * 100
                log(f"  {c:<20} {m:>10.1f} {r:>10.1f} {err:>+8.1f}%")
            elif m > 0:
                log(f"  {c:<20} {m:>10.1f} {'n/a':>10} {'n/a':>8}")
    if gen_ember is not None:
        model_gen_agg2 = aggregate_to_ember(all_gen, PYPSA_TO_EMBER)
        log("")
        log(f"  Generation (TWh) — model vs Ember {ember_year}")
        log(f"  {'Carrier':<20} {'Model':>10} {'Ember':>10} {'Error':>8}")
        log("  " + "-" * 52)
        for c in sorted(set(model_gen_agg2.index) | set(gen_ember.index)):
            m = model_gen_agg2.get(c, 0.0)
            r = gen_ember.get(c, 0.0)
            if r > 0:
                err = (m - r) / r * 100
                log(f"  {c:<20} {m:>10.1f} {r:>10.1f} {err:>+8.1f}%")
            elif m > 0:
                log(f"  {c:<20} {m:>10.1f} {'n/a':>10} {'n/a':>8}")

    report_path = os.path.join(BASE_OUTPUT_DIR, f"{network_name}_inspect.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved: {report_path}")

    # ── GRAFICI ───────────────────────────────────────────────────────────────

    model_cap_agg = aggregate_to_ember(all_cap, PYPSA_TO_DISPLAY_CAP)
    model_gen_agg = aggregate_to_ember(all_gen, PYPSA_TO_EMBER)

    # ── PLOT 1: Capacità modello vs IRENA ─────────────────────────────────────
    irena_label = f"IRENA {ember_year}" if ember_year else "IRENA (n/a)"
    carriers_cap = sorted(set(model_cap_agg.index) | (set(irena_cap.index) if irena_cap is not None else set()))
    x = np.arange(len(carriers_cap))
    width = 0.35
    mv = [model_cap_agg.get(c, 0) for c in carriers_cap]
    ev = [irena_cap.get(c, 0) if irena_cap is not None else 0 for c in carriers_cap]
    colors_c = [get_color(c) for c in carriers_cap]

    err_cap = [(m - e) / e * 100 if e > 0 else None for m, e in zip(mv, ev)]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(13, 9),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )
    ax_top.bar(x - width/2, mv, width, color=colors_c, alpha=0.9,
               edgecolor="white", label="Model")
    ax_top.bar(x + width/2, ev, width, color=colors_c, alpha=0.45,
               edgecolor="black", linewidth=0.8, label=irena_label, hatch="///")
    for i, m in enumerate(mv):
        if m > 0:
            ax_top.text(x[i] - width/2, m * 1.15, f"{m:.0f} GW",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax_top.set_ylabel("GW", fontsize=11)
    ax_top.set_title(f"Installed capacity: model vs {irena_label}\n{network_name}",
                     fontsize=12, fontweight="bold")
    ax_top.legend(fontsize=10)
    ax_top.grid(True, axis="y", alpha=0.2, which="both")
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    err_cap_vals = [e if e is not None else 0 for e in err_cap]
    err_cap_colors = ["#d62728" if e > 0 else "#1f77b4" if e < 0 else "#aaaaaa"
                      for e in err_cap_vals]
    ax_bot.bar(x, err_cap_vals, 0.6, color=err_cap_colors, alpha=0.85, edgecolor="white")
    ax_bot.axhline(0, color="black", linewidth=0.8)
    for i, (e, c) in enumerate(zip(err_cap_vals, err_cap_colors)):
        if err_cap[i] is not None:
            va = "bottom" if e >= 0 else "top"
            offset = 1 if e >= 0 else -1
            ax_bot.text(x[i], e + offset, f"{e:+.0f}%",
                        ha="center", va=va, fontsize=8, color=c, fontweight="bold")
    ax_bot.set_ylabel("Error %", fontsize=10)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(carriers_cap, rotation=30, ha="right", fontsize=10)
    ax_bot.grid(True, axis="y", alpha=0.2)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)
    plt.tight_layout()
    out = os.path.join(BASE_OUTPUT_DIR, f"{network_name}_capacity.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # ── PLOT 2: Generazione modello vs Ember (log scale + pannello errore) ────
    carriers_gen = sorted(set(model_gen_agg.index) | (set(gen_ember.index) if gen_ember is not None else set()))
    x = np.arange(len(carriers_gen))
    mg = [model_gen_agg.get(c, 0) for c in carriers_gen]
    eg = [gen_ember.get(c, 0) if gen_ember is not None else 0 for c in carriers_gen]
    colors_g = [get_color(c) for c in carriers_gen]

    # Calcola errori % per pannello inferiore
    err_pct = []
    for m, e in zip(mg, eg):
        if e > 0:
            err_pct.append((m - e) / e * 100)
        else:
            err_pct.append(None)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(13, 9),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )

    # Pannello superiore: barre con scala log
    ax_top.bar(x - width/2, mg, width, color=colors_g, alpha=0.9,
               edgecolor="white", label="Model")
    ax_top.bar(x + width/2, eg, width, color=colors_g, alpha=0.45,
               edgecolor="black", linewidth=0.8, label=ember_label, hatch="///")

    # Etichette TWh + quota % sopra barre modello
    for i, m in enumerate(mg):
        if m > 0:
            share = m / total_gen * 100 if total_gen > 0 else 0
            ax_top.text(x[i] - width/2, m * 1.15,
                        f"{m:.0f} TWh\n({share:.0f}%)",
                        ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax_top.set_ylabel("TWh", fontsize=11)
    ax_top.set_title(f"Generation: model vs {ember_label}\n{network_name}",
                     fontsize=12, fontweight="bold")
    ax_top.legend(fontsize=10)
    ax_top.grid(True, axis="y", alpha=0.2, which="both")
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    # Pannello inferiore: errore %
    err_colors = []
    err_vals = []
    for e in err_pct:
        if e is not None:
            err_colors.append("#d62728" if e > 0 else "#1f77b4")
            err_vals.append(e)
        else:
            err_colors.append("#aaaaaa")
            err_vals.append(0)

    ax_bot.bar(x, err_vals, 0.6, color=err_colors, alpha=0.85, edgecolor="white")
    ax_bot.axhline(0, color="black", linewidth=0.8)
    for i, (e, c) in enumerate(zip(err_vals, err_colors)):
        if err_pct[i] is not None:
            va = "bottom" if e >= 0 else "top"
            offset = 1 if e >= 0 else -1
            ax_bot.text(x[i], e + offset, f"{e:+.0f}%",
                        ha="center", va=va, fontsize=8, color=c, fontweight="bold")

    ax_bot.set_ylabel("Error %", fontsize=10)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(carriers_gen, rotation=30, ha="right", fontsize=10)
    ax_bot.grid(True, axis="y", alpha=0.2)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)

    plt.tight_layout()
    out = os.path.join(BASE_OUTPUT_DIR, f"{network_name}_generation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # ── PLOT 3: Pie chart mix energetico ───────────────────────────────────────
    gen_pie = all_gen[all_gen / total_gen >= 0.005]
    other = all_gen[all_gen / total_gen < 0.005].sum()
    if other > 0:
        gen_pie = pd.concat([gen_pie, pd.Series({"other": other})])
    colors_pie = [get_color(c) for c in gen_pie.index]

    fig, ax = plt.subplots(figsize=(9, 9))
    wedges, texts, autotexts = ax.pie(
        gen_pie.values,
        labels=gen_pie.index,
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
        f"Energy mix — {network_name}\nTotal gen: {total_gen:.0f} TWh  |  Load: {load_twh:.0f} TWh",
        fontsize=12, fontweight="bold", pad=20
    )
    plt.tight_layout()
    out = os.path.join(BASE_OUTPUT_DIR, f"{network_name}_mix_pie.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # ── PLOT 4: Load factor per carrier ───────────────────────────────────────
    carriers_lf = sorted(set(all_cap.index) & set(all_gen.index))
    carriers_lf = [c for c in carriers_lf if all_cap.get(c, 0) > 0]

    lf_vals = []
    for c in carriers_lf:
        cap_gw = all_cap.get(c, 0)
        gen_twh = all_gen.get(c, 0)
        if cap_gw > 0:
            lf = gen_twh / (cap_gw * 8760 / 1e3) * 100
            lf_vals.append(lf)
        else:
            lf_vals.append(0)

    colors_lf = [get_color(c) for c in carriers_lf]
    x = np.arange(len(carriers_lf))

    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(x, lf_vals, 0.6, color=colors_lf, alpha=0.88, edgecolor="white")

    for bar, val, carrier in zip(bars, lf_vals, carriers_lf):
        cap = all_cap.get(carrier, 0)
        gen = all_gen.get(carrier, 0)
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%\n({gen:.0f} TWh\n{cap:.0f} GW)",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.axhline(100, color="red", linewidth=1, linestyle="--", alpha=0.5,
               label="100% (theoretical max)")
    ax.set_xticks(x)
    ax.set_xticklabels(carriers_lf, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Capacity Factor (%)", fontsize=11)
    ax.set_ylim(0, max(lf_vals) * 1.35 if lf_vals else 110)
    ax.set_title(f"Capacity Factor by carrier — {network_name}\n"
                 f"gen (TWh) / (capacity (GW) × 8760 h)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = os.path.join(BASE_OUTPUT_DIR, f"{network_name}_capacity_vs_generation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quick inspection of a PyPSA-Earth network with Ember comparison"
    )
    parser.add_argument("network", help="Path to the .nc network file")
    args = parser.parse_args()
    inspect(args.network)