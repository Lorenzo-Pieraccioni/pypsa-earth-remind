"""
plot_network.py
===============
Visualizzazione geografica della rete PyPSA-Earth su mappa della Cina con confini provinciali.

Mostra:
  - bus della rete (nodi, dimensione proporzionale alla capacità installata)
  - linee AC della rete
  - confini provinciali cinesi (da GADM)

Output salvato in analysis/tools/output/<nome_rete>_map.png

Uso:
  python analysis/tools/plot_network.py results/networks/CN2020/elec_s_250_ec_lcopt_Co2L-3h.nc
  python analysis/tools/plot_network.py results/networks/CN2025/elec_s_250_ec_lcopt_Co2L-6h.nc
  python analysis/tools/plot_network.py results/networks/CN2060/elec_s_250_ec_lcopt_Co2L-6h.nc
"""

import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pypsa
import geopandas as gpd

# ── PARAMETERS ────────────────────────────────────────────────────────────────

GADM_FILE       = "resources/shapes/gadm_shapes.geojson"
BASE_OUTPUT_DIR = "analysis/tools/output"

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot PyPSA-Earth network on China map")
    parser.add_argument("network", help="Path to the .nc network file")
    args = parser.parse_args()

    if not os.path.exists(args.network):
        print(f"[ERROR] File not found: {args.network}")
        raise SystemExit(1)

    print(f"Loading: {args.network}")
    n = pypsa.Network(args.network)

    # Build title from network metadata
    n_buses = len(n.buses)
    network_name = os.path.basename(os.path.dirname(args.network)) + "_" + os.path.splitext(os.path.basename(args.network))[0]
    title = f"PyPSA-Earth — China network ({n_buses} buses) — {network_name}"

    # Map
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(14, 10), subplot_kw={"projection": proj})

    # Provincial boundaries
    if os.path.exists(GADM_FILE):
        gadm = gpd.read_file(GADM_FILE)
        china = gadm[gadm["country"] == "CN"]
        china.boundary.plot(ax=ax, color="black", linewidth=0.5, transform=proj)
    else:
        print(f"[WARNING] GADM file not found: {GADM_FILE} — skipping provincial boundaries")

    # Network
    n.plot(ax=ax, title=title)

    # Save
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(BASE_OUTPUT_DIR, f"{network_name}_map.png")
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_file}")

    #python analysis/tools/plot_network.py results/networks/CN2020/elec_s_250_ec_lcopt_Co2L-3h.nc