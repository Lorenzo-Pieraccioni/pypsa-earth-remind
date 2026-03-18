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

Non produce grafici — per i grafici usare compare_scenarios.py o provincial_load_analysis.py.

Output:
  Stampa a schermo + salva report in analysis/tools/output/<nome_rete>_inspect.txt

Uso:
  python analysis/tools/inspect_network.py results/networks/CN2020/elec_s_250_ec_lcopt_Co2L-3h.nc
  python analysis/tools/inspect_network.py results/networks/CN2025/elec_s_250_ec_lcopt_Co2L-6h.nc
  python analysis/tools/inspect_network.py results/networks/CN2060/elec_s_250_ec_lcopt_Co2L-6h.nc
"""

import argparse
import os
import pypsa

# ── PARAMETERS ────────────────────────────────────────────────────────────────

BASE_OUTPUT_DIR = "analysis/tools/output"

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
    cap = n.generators.groupby("carrier")["p_nom"].sum() / 1e3
    cap = cap[cap.index != "load shedding"].sort_values(ascending=False)
    stor = n.storage_units.groupby("carrier")["p_nom"].sum() / 1e3

    log("")
    log("=" * 60)
    log("3. INSTALLED CAPACITY BY CARRIER (GW)")
    log("=" * 60)
    log("  Generators:")
    for carrier, val in cap.items():
        log(f"    {carrier:<18} {val:>8.1f} GW")
    log(f"    {'TOTAL':<18} {cap.sum():>8.1f} GW")
    log("  Storage units:")
    for carrier, val in stor.items():
        log(f"    {carrier:<18} {val:>8.1f} GW")

    # 4. Generation
    gen_twh = (
        n.generators_t.p
        .multiply(w, axis=0)
        .sum()
        .groupby(n.generators.carrier)
        .sum()
        / 1e6
    )
    gen_twh = gen_twh[gen_twh.index != "load shedding"].sort_values(ascending=False)
    total_gen = gen_twh.sum()

    log("")
    log("=" * 60)
    log("4. GENERATION BY CARRIER (TWh)")
    log("=" * 60)
    for carrier, val in gen_twh.items():
        share = val / total_gen * 100 if total_gen > 0 else 0
        log(f"  {carrier:<18} {val:>8.1f} TWh  ({share:.1f}%)")
    log(f"  {'TOTAL':<18} {total_gen:>8.1f} TWh")

    # 5. Energy balance
    balance_err = (total_gen - load_twh) / load_twh * 100 if load_twh > 0 else float("nan")
    log("")
    log("=" * 60)
    log("5. ENERGY BALANCE CHECK")
    log("=" * 60)
    log(f"  Load:         {load_twh:.1f} TWh")
    log(f"  Generation:   {total_gen:.1f} TWh")
    log(f"  Error:        {balance_err:.2f}%")

    # Save report
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    network_name = os.path.basename(os.path.dirname(network_file)) + "_" + os.path.splitext(os.path.basename(network_file))[0]
    report_path = os.path.join(BASE_OUTPUT_DIR, f"{network_name}_inspect.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved: {report_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick inspection of a PyPSA-Earth network")
    parser.add_argument("network", help="Path to the .nc network file")
    args = parser.parse_args()
    inspect(args.network)
    
    #python analysis/tools/inspect_network.py results/networks/CN2020/elec_s_250_ec_lcopt_Co2L-3h.nc