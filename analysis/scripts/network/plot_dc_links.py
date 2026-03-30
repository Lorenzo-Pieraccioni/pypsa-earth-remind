"""
plot_dc_links.py v2
===================
Visualizzazione geografica dei link DC/B2B e linee AC della rete PyPSA-Earth China.
Prodotto su richiesta di Ivan Ruiz Ramos per validazione visiva dei corridori UHV.

Mostra:
  - linee AC (380 kV): linee blu chiaro sottili
  - link DC colorati per tensione:
      500 kV  -> arancione
      660 kV  -> rosso
      800 kV  -> viola
      1100 kV -> nero
      nan     -> grigio
  - link B2B: tratteggiati grigi
  - bus DC: nodi con annotazione p_nom in GW del link piu' grande che li tocca
  - confini provinciali (GADM1, derivati da dissolve GADM2): linee nere spesse
  - confini prefetturali (GADM2): linee grigio chiaro sottili (sfondo)

Output salvato in analysis/tools/output/<scenario>_dc_links.png

Uso:
  python analysis/tools/plot_dc_links.py results/networks/CN2020_Admin2_noCO2/elec_s_250_ec_lcopt_6h.nc
  python analysis/tools/plot_dc_links.py results/networks/CN2025_Admin2/elec_s_250_ec_lcopt_Co2L-6h.nc
"""

import argparse
import os
import re
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
BASE_OUTPUT_DIR = "analysis/network/output"

# Line width scaling for DC links
LW_MIN_DC = 0.8
LW_MAX_DC = 6.0
P_REF_DC  = 6400.0  # MW

# Line width for AC lines (fixed, thin)
LW_AC = 0.5

# Colors for DC links by voltage
DC_VOLTAGE_COLORS = {
    500.0:  "#f4a620",   # orange  — HVDC 500 kV
    660.0:  "#d62728",   # red     — HVDC 660 kV
    800.0:  "#7b2d8b",   # purple  — UHV DC 800 kV
    1100.0: "#111111",   # black   — Ultra UHV 1100 kV
    "nan":  "#aaaaaa",   # grey    — unknown voltage
}

COLOR_B2B    = "#888888"   # grey dashed
COLOR_AC     = "#a8d4f5"   # light blue
COLOR_BUS_DC = "#222222"   # dark grey

# Province number -> name mapping (for dissolve)
GADM_PROVINCE = {
    1:  "Anhui",          2:  "Beijing",        3:  "Chongqing",
    4:  "Fujian",         5:  "Gansu",          6:  "Guangdong",
    7:  "Guangxi",        8:  "Guizhou",        9:  "Hainan",
    10: "Hebei",          11: "Heilongjiang",   12: "Henan",
    13: "Hubei",          14: "Hunan",          15: "Jiangsu",
    16: "Jiangxi",        17: "Jilin",          18: "Liaoning",
    19: "Inner Mongolia", 20: "Ningxia",        21: "Qinghai",
    22: "Shaanxi",        23: "Shandong",       24: "Shanghai",
    25: "Shanxi",         26: "Sichuan",        27: "Tianjin",
    28: "Xinjiang",       29: "Tibet",          30: "Yunnan",
    31: "Zhejiang",
}

# ─────────────────────────────────────────────────────────────────────────────


def scale_lw(p_nom_mw):
    return LW_MIN_DC + (LW_MAX_DC - LW_MIN_DC) * np.clip(p_nom_mw / P_REF_DC, 0, 1)


def extract_province_number(gadm_id):
    m = re.match(r"CN\.(\d+)\.", gadm_id)
    return int(m.group(1)) if m else None


def load_gadm_shapes(gadm_file):
    """Load GADM2 prefectures and derive GADM1 provinces by dissolve."""
    gdf = gpd.read_file(gadm_file)
    china_pref = gdf[gdf["country"] == "CN"].copy().to_crs("EPSG:4326")
    china_pref["prov_num"] = china_pref["GADM_ID"].apply(extract_province_number)
    # Dissolve to province level
    china_prov = china_pref.dissolve(by="prov_num").reset_index()
    return china_pref, china_prov


def get_dc_color(v_nom):
    """Return color for a DC link based on v_nom."""
    if v_nom is None or (isinstance(v_nom, float) and np.isnan(v_nom)):
        return DC_VOLTAGE_COLORS["nan"]
    return DC_VOLTAGE_COLORS.get(float(v_nom), DC_VOLTAGE_COLORS["nan"])


def max_pnom_at_bus(bus_id, links):
    """Return max p_nom of all links touching a bus."""
    mask = (links.bus0 == bus_id) | (links.bus1 == bus_id)
    if mask.any():
        return links.loc[mask, "p_nom"].max()
    return None


def plot_dc(network_file):
    if not os.path.exists(network_file):
        print(f"[ERROR] File not found: {network_file}")
        raise SystemExit(1)

    print(f"Loading network: {network_file}")
    n = pypsa.Network(network_file)

    # Separate link types
    links_dc  = n.links[n.links.carrier == "DC"]
    links_b2b = n.links[n.links.carrier == "B2B"]
    buses_dc  = n.buses[n.buses.carrier == "DC"]
    lines_ac  = n.lines

    print(f"  DC links : {len(links_dc)}")
    print(f"  B2B links: {len(links_b2b)}")
    print(f"  DC buses : {len(buses_dc)}")
    print(f"  AC lines : {len(lines_ac)}")

    # Load GADM shapes
    print("Loading GADM shapes...")
    china_pref, china_prov = load_gadm_shapes(GADM_FILE)

    # Build scenario name for title and output file
    scenario_name = (
        os.path.basename(os.path.dirname(network_file))
        + "_"
        + os.path.splitext(os.path.basename(network_file))[0]
    )

    # Voltage stats
    dc_vnoms = links_dc.v_nom.dropna().unique()
    dc_vnoms_str = ", ".join([f"{int(v)} kV" for v in sorted(dc_vnoms)])

    title = (
        f"PyPSA-Earth — China network — {scenario_name}\n"
        f"DC links: {len(links_dc)} ({dc_vnoms_str}) | "
        f"B2B: {len(links_b2b)} | AC lines: {len(lines_ac)}"
    )

    # ── PLOT ──────────────────────────────────────────────────────────────────
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(18, 12), subplot_kw={"projection": proj})
    ax.set_extent([73, 136, 17, 54], crs=proj)
    ax.set_facecolor("#f8f9fa")

    # z=1: prefecture boundaries (thin grey — background only)
    china_pref.boundary.plot(
        ax=ax, color="#cccccc", linewidth=0.25,
        transform=proj, alpha=0.5, zorder=1,
    )

    # z=2: province fills (white, to visually separate from background)
    china_pref.plot(
        ax=ax, color="white", alpha=0.7,
        linewidth=0, zorder=2,
    )

    # z=3: AC lines (thin light blue)
    for _, line in lines_ac.iterrows():
        b0 = n.buses.loc[line.bus0] if line.bus0 in n.buses.index else None
        b1 = n.buses.loc[line.bus1] if line.bus1 in n.buses.index else None
        if b0 is None or b1 is None:
            continue
        ax.plot(
            [b0.x, b1.x], [b0.y, b1.y],
            color=COLOR_AC, linewidth=LW_AC, alpha=0.7,
            transform=proj, solid_capstyle="round", zorder=3,
        )

    # z=4: B2B links (grey dashed)
    for _, link in links_b2b.iterrows():
        b0 = n.buses.loc[link.bus0] if link.bus0 in n.buses.index else None
        b1 = n.buses.loc[link.bus1] if link.bus1 in n.buses.index else None
        if b0 is None or b1 is None:
            continue
        ax.plot(
            [b0.x, b1.x], [b0.y, b1.y],
            color=COLOR_B2B, linewidth=0.8, alpha=0.6,
            transform=proj, linestyle="--", zorder=4,
        )

    # z=5: DC links colored by voltage
    for _, link in links_dc.iterrows():
        b0 = n.buses.loc[link.bus0] if link.bus0 in n.buses.index else None
        b1 = n.buses.loc[link.bus1] if link.bus1 in n.buses.index else None
        if b0 is None or b1 is None:
            continue
        color = get_dc_color(link.v_nom)
        lw = scale_lw(link.p_nom)
        ax.plot(
            [b0.x, b1.x], [b0.y, b1.y],
            color=color, linewidth=lw, alpha=0.88,
            transform=proj, solid_capstyle="round", zorder=5,
        )

    # z=6: provincial boundaries (thick black — on top of everything)
    china_prov.boundary.plot(
        ax=ax, color="#111111", linewidth=1.6,
        transform=proj, zorder=6,
    )

    # z=7: DC buses with p_nom annotation
    for bus_id, bus in buses_dc.iterrows():
        ax.scatter(
            bus.x, bus.y,
            color=COLOR_BUS_DC, s=14, zorder=7,
            transform=proj,
        )
        # Annotate with max p_nom of DC links at this bus
        pmax = max_pnom_at_bus(bus_id, links_dc)
        if pmax is not None and pmax > 0:
            ax.annotate(
                f"{pmax/1e3:.1f}",
                xy=(bus.x, bus.y),
                xytext=(3, 3), textcoords="offset points",
                fontsize=5.5, color="#333333",
                transform=proj, zorder=8,
            )

    # ── LEGEND ────────────────────────────────────────────────────────────────
    legend_elements = []

    # AC line
    legend_elements.append(
        mlines.Line2D([], [], color=COLOR_AC, linewidth=1.5,
                      label=f"AC lines 380 kV ({len(lines_ac)})")
    )

    # DC by voltage
    for v_nom in sorted([k for k in DC_VOLTAGE_COLORS.keys() if k != "nan"]):
        if v_nom == "nan":
            continue
        count = len(links_dc[links_dc.v_nom == v_nom])
        if count == 0:
            continue
        legend_elements.append(
            mlines.Line2D([], [], color=DC_VOLTAGE_COLORS[v_nom],
                          linewidth=2.5,
                          label=f"DC {int(v_nom)} kV ({count})")
        )

    # nan voltage
    nan_count = len(links_dc[links_dc.v_nom.isna()])
    if nan_count > 0:
        legend_elements.append(
            mlines.Line2D([], [], color=DC_VOLTAGE_COLORS["nan"],
                          linewidth=1.5,
                          label=f"DC unknown V ({nan_count})")
        )

    # B2B
    legend_elements.append(
        mlines.Line2D([], [], color=COLOR_B2B, linewidth=1.5,
                      linestyle="--",
                      label=f"B2B ({len(links_b2b)})")
    )

    # p_nom range
    all_pnom = list(links_dc.p_nom)
    if all_pnom:
        legend_elements.append(
            mpatches.Patch(color="none",
                           label=f"p_nom: {min(all_pnom)/1e3:.1f}–{max(all_pnom)/1e3:.1f} GW")
        )

    ax.legend(handles=legend_elements, loc="lower left",
              fontsize=8.5, framealpha=0.92, title="Network elements",
              title_fontsize=9)

    ax.set_title(title, fontsize=10, pad=10)

    # ── SAVE ──────────────────────────────────────────────────────────────────
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(BASE_OUTPUT_DIR, f"{scenario_name}_dc_links.png")
    fig.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_file}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot DC/B2B links and AC lines of a PyPSA-Earth China network"
    )
    parser.add_argument("network", help="Path to the .nc network file")
    args = parser.parse_args()
    plot_dc(args.network)