"""
csv_inspection.py
=================
Esporta i componenti principali di una rete PyPSA in file CSV per ispezione manuale.

Esporta:
  Componenti statici:
    - buses.csv          : tutti i bus della rete
    - generators.csv     : generatori (capacità, carrier, bus)
    - lines.csv          : linee AC
    - loads.csv          : carichi
    - storage_units.csv  : unità di storage (hydro, PHS, battery)

  Componenti dinamici (serie temporali):
    - gen_dispatch.csv   : dispatch dei generatori (MW, per snapshot)
    - load_profile.csv   : profilo di carico (MW, per snapshot)
    - marginal_price.csv : prezzo marginale per bus (se disponibile)

Output salvato in una sottocartella nominata come il file .nc (senza estensione),
dentro analysis/tools/output/.

Uso:
  python analysis/tools/csv_inspection.py results/networks/CN2020/elec_s_250_ec_lcopt_Co2L-3h.nc
  python analysis/tools/csv_inspection.py results/networks/CN2025/elec_s_250_ec_lcopt_Co2L-6h.nc
"""

import argparse
import os
import pypsa

# ── PARAMETERS ────────────────────────────────────────────────────────────────

BASE_OUTPUT_DIR = "analysis/tools/output"

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyPSA network components to CSV")
    parser.add_argument("network", help="Path to the .nc network file")
    args = parser.parse_args()

    if not os.path.exists(args.network):
        print(f"[ERROR] File not found: {args.network}")
        raise SystemExit(1)

    # Output folder named after the network file
    network_name = os.path.splitext(os.path.basename(args.network))[0]
    output_dir = os.path.join(BASE_OUTPUT_DIR, network_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading: {args.network}")
    n = pypsa.Network(args.network)

    # Static components
    static = {
        "buses":         n.buses,
        "generators":    n.generators,
        "lines":         n.lines,
        "loads":         n.loads,
        "storage_units": n.storage_units,
    }
    for name, df in static.items():
        path = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(path)
        print(f"  Saved: {path}  ({len(df)} rows)")

    # Dynamic components
    dynamic = {
        "gen_dispatch":  n.generators_t.p,
        "load_profile":  n.loads_t.p_set,
    }
    for name, df in dynamic.items():
        path = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(path)
        print(f"  Saved: {path}  ({len(df)} snapshots x {len(df.columns)} columns)")

    # Marginal price (optional)
    if hasattr(n.buses_t, "marginal_price") and not n.buses_t.marginal_price.empty:
        path = os.path.join(output_dir, "marginal_price.csv")
        n.buses_t.marginal_price.to_csv(path)
        print(f"  Saved: {path}")
    else:
        print("  marginal_price: not available (skipped)")

    print(f"\nDone. Output: {output_dir}/")
    
