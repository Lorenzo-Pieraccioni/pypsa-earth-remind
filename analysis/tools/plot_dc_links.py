"""
plot_dc_links.py
================
Visualizzazione geografica dei soli link DC e B2B della rete PyPSA-Earth China.
Prodotto su richiesta di Ivan per validazione visiva dei corridoi UHV.

Mostra:
  - link DC  : linee rosse, spessore proporzionale a p_nom
  - link B2B : linee blu, spessore proporzionale a p_nom
  - bus DC   : nodi neri
  - confini provinciali cinesi (da GADM)

La legenda riporta i range di capacità in GW.

Output salvato in analysis/tools/output/<scenario>_dc_links.png

Uso:
  python analysis/tools/plot_dc_links.py results/networks/CN2025_Admin2/elec_s_250_ec_lcopt_Co2L-6h.nc
  python analysis/tools/plot_dc_links.py results/networks/CN2020_Admin1/elec_s_250_ec_lcopt_Co2L-6h.nc
"""

import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import pypsa
import geopandas as gpd
import cartopy.crs as ccrs

# ── PARAMETERS ────────────────────────────────────────────────────────────────

GADM_FILE       = "resources/shapes/gadm_shapes.geojson"
BASE_OUTPUT_DIR = "analysis/tools/output"

# Line width scaling: p_nom in MW -> linewidth in points
# 819 MW -> lw ~1.0, 6400 MW -> lw ~5.0
LW_MIN  = 0.8
LW_MAX  = 6.0
P_REF   = 6400.0  # MW, reference for max linewidth

COLOR_DC  = "#d62728"   # red
COLOR_B2B = "#1f77b4"   # blue
COLOR_BUS = "#2c2c2c"   # dark grey

# ─────────────────────────────────────────────────────────────────────────────


def scale_lw(p_nom_mw):
    """Scale line width proportionally to p_nom."""
    return LW_MIN + (LW_MAX - LW_MIN) * np.clip(p_nom_mw / P_REF, 0, 1)


def plot_dc(network_file):
    if not os.path.exists(network_file):
        print(f"[ERROR] File not found: {network_file}")
        raise SystemExit(1)

    print(f"Loading: {network_file}")
    n = pypsa.Network(network_file)

    # Separate DC and B2B links
    links_dc  = n.links[n.links.carrier == "DC"]
    links_b2b = n.links[n.links.carrier == "B2B"]
    buses_dc  = n.buses[n.buses.carrier == "DC"]

    print(f"  DC links : {len(links_dc)}")
    print(f"  B2B links: {len(links_b2b)}")
    print(f"  DC buses : {len(buses_dc)}")

    # Build title
    scenario_name = (
        os.path.basename(os.path.dirname(network_file))
        + "_"
        + os.path.splitext(os.path.basename(network_file))[0]
    )
    title = (
        f"PyPSA-Earth — China DC/B2B links — {scenario_name}\n"
        f"DC: {len(links_dc)} links | B2B: {len(links_b2b)} links | "
        f"DC buses: {len(buses_dc)}"
    )

    # Map setup
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(16, 11), subplot_kw={"projection": proj})
    ax.set_extent([73, 136, 17, 54], crs=proj)

    # Provincial boundaries
    if os.path.exists(GADM_FILE):
        gadm = gpd.read_file(GADM_FILE)
        china = gadm[gadm["country"] == "CN"]
        china.boundary.plot(ax=ax, color="black", linewidth=0.4,
                            transform=proj, alpha=0.6)
    else:
        print(f"[WARNING] GADM file not found: {GADM_FILE}")

    # Plot DC links
    for _, link in links_dc.iterrows():
        b0 = n.buses.loc[link.bus0] if link.bus0 in n.buses.index else None
        b1 = n.buses.loc[link.bus1] if link.bus1 in n.buses.index else None
        if b0 is None or b1 is None:
            continue
        lw = scale_lw(link.p_nom)
        ax.plot(
            [b0.x, b1.x], [b0.y, b1.y],
            color=COLOR_DC, linewidth=lw, alpha=0.85,
            transform=proj, solid_capstyle="round"
        )

    # Plot B2B links
    for _, link in links_b2b.iterrows():
        b0 = n.buses.loc[link.bus0] if link.bus0 in n.buses.index else None
        b1 = n.buses.loc[link.bus1] if link.bus1 in n.buses.index else None
        if b0 is None or b1 is None:
            continue
        lw = scale_lw(link.p_nom)
        ax.plot(
            [b0.x, b1.x], [b0.y, b1.y],
            color=COLOR_B2B, linewidth=lw, alpha=0.85,
            transform=proj, solid_capstyle="round",
            linestyle="--"
        )

    # Plot DC buses
    ax.scatter(
        buses_dc.x, buses_dc.y,
        color=COLOR_BUS, s=12, zorder=5,
        transform=proj, label="DC bus"
    )

    # Capacity stats for legend
    all_pnom = list(links_dc.p_nom) + list(links_b2b.p_nom)
    p_min = min(all_pnom) / 1e3 if all_pnom else 0
    p_max = max(all_pnom) / 1e3 if all_pnom else 0

    # Legend
    legend_elements = [
        mlines.Line2D([], [], color=COLOR_DC,  linewidth=2.5,
                      label=f"DC links ({len(links_dc)})"),
        mlines.Line2D([], [], color=COLOR_B2B, linewidth=2.5, linestyle="--",
                      label=f"B2B links ({len(links_b2b)})"),
        mlines.Line2D([], [], color=COLOR_BUS, marker="o", markersize=5,
                      linewidth=0, label=f"DC buses ({len(buses_dc)})"),
        mpatches.Patch(color="none",
                       label=f"p_nom range: {p_min:.1f} – {p_max:.1f} GW"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9,
              framealpha=0.9)

    ax.set_title(title, fontsize=11, pad=10)

    # Save
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(BASE_OUTPUT_DIR, f"{scenario_name}_dc_links.png")
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_file}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot DC and B2B links of a PyPSA-Earth China network"
    )
    parser.add_argument("network", help="Path to the .nc network file")
    args = parser.parse_args()
    plot_dc(args.network)

# Example usage:
# python analysis/tools/plot_dc_links.py results/networks/CN2025_Admin2/elec_s_250_ec_lcopt_Co2L-6h.nc
# python analysis/tools/plot_dc_links.py results/networks/CN2020_Admin1/elec_s_250_ec_lcopt_Co2L-6h.nc