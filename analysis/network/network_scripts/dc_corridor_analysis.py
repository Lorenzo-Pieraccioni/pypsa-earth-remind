"""
dc_corridor_analysis.py
=======================
Analisi dei corridori DC: capacità installata vs flussi effettivi.

Risponde a una domanda: i corridori Tibet/Xinjiang → est sono saturi?

Output:
  dc_links_summary.csv         — tabella completa per link
  dc_utilization_hist.png      — distribuzione utilization su tutti i 113 link
                                  (Tibet/Xinjiang marcati)
  dc_tibet_xinjiang.png        — capacità vs flusso per i 5 corridori Tibet/Xinjiang
  dc_map_utilization.png       — mappa Admin1, colore = utilization
                                  (rosso = saturo, verde = libero)

Uso:
  PYPSA_OUTPUT_DIR="analysis/network/dc_spatial" python analysis/network/dc_corridor_analysis.py results/networks/geospatial_v2_approccio1/CN2020_Admin2/elec_s_250_ec_lcopt_3h.nc
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import geopandas as gpd
from province_utils import overlay_admin2_only
import numpy as np
import pandas as pd
import pypsa
import seaborn as sns

warnings.filterwarnings("ignore")

# ── PARAMETERS ────────────────────────────────────────────────────────────────

GADM_FILE  = "resources/shapes/gadm_shapes.geojson"
OUTPUT_DIR = os.path.join(os.environ.get("PYPSA_OUTPUT_DIR", "analysis/network/output"), "dc_corridor_analysis")

PROVINCE_NAMES = {
    1: "Anhui",        2: "Beijing",      3: "Chongqing",    4: "Fujian",
    5: "Gansu",        6: "Guangdong",    7: "Guangxi",      8: "Guizhou",
    9: "Hainan",      10: "Hebei",       11: "Heilongjiang", 12: "Henan",
   13: "Hubei",       14: "Hunan",       15: "Jiangsu",      16: "Jiangxi",
   17: "Jilin",       18: "Liaoning",    19: "Inner Mongolia",20: "Ningxia",
   21: "Qinghai",     22: "Shaanxi",     23: "Shandong",     24: "Shanghai",
   25: "Shanxi",      26: "Sichuan",     27: "Tianjin",      28: "Xinjiang",
   29: "Tibet",       30: "Yunnan",      31: "Zhejiang",
}

TIBET_XJ = {"CN.28", "CN.29"}   # Xinjiang, Tibet

# ── HELPERS ───────────────────────────────────────────────────────────────────

def save(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def bus_to_admin1(name):
    parts = str(name).split(".")
    return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else name


def admin1_to_name(code):
    try:
        return PROVINCE_NAMES.get(int(code.split(".")[1]), code)
    except (IndexError, ValueError):
        return code


def is_tibet_xj(row):
    return row["admin1_bus0"] in TIBET_XJ or row["admin1_bus1"] in TIBET_XJ


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("network", help="Path to .nc network file")
    args = parser.parse_args()

    if not os.path.exists(args.network):
        print(f"[ERROR] File not found: {args.network}")
        raise SystemExit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sns.set_style("white")
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "font.size": 11,
    })

    print(f"Loading: {args.network}")
    n = pypsa.Network(args.network)

    dt = (n.snapshots[1] - n.snapshots[0]).total_seconds() / 3600.0 if len(n.snapshots) > 1 else 1.0

    # ── Extract DC links ──────────────────────────────────────────────────────
    dc = n.links[n.links.carrier == "DC"].copy()
    print(f"  DC links: {len(dc)}")

    p0_ts = n.links_t.p0.reindex(columns=dc.index, fill_value=0.0)

    df = dc[["p_nom", "p_nom_opt", "v_nom", "bus0", "bus1"]].copy()
    df["mean_flow_mw"] = p0_ts.abs().mean()
    df["max_flow_mw"]  = p0_ts.abs().max()
    df["net_flow_mw"]  = p0_ts.mean()
    df["energy_twh"]   = (p0_ts.abs() * dt).sum() / 1e6
    df["utilization"]  = df["mean_flow_mw"] / df["p_nom_opt"]
    df["admin1_bus0"]  = df["bus0"].apply(bus_to_admin1)
    df["admin1_bus1"]  = df["bus1"].apply(bus_to_admin1)
    df["prov_bus0"]    = df["admin1_bus0"].apply(admin1_to_name)
    df["prov_bus1"]    = df["admin1_bus1"].apply(admin1_to_name)
    df["is_tibet_xj"]  = df.apply(is_tibet_xj, axis=1)
    df = df.reset_index()

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\nUtilization rate (mean |p0| / p_nom_opt):")
    print(f"  Mean:   {df['utilization'].mean():.3f}")
    print(f"  Median: {df['utilization'].median():.3f}")
    print(f"  < 20%:  {(df['utilization'] < 0.20).sum()} links")
    print(f"  > 80%:  {(df['utilization'] > 0.80).sum()} links")

    txj = df[df["is_tibet_xj"]].sort_values("p_nom", ascending=False)
    print(f"\nTibet / Xinjiang links ({len(txj)}):")
    print(txj[["Link","prov_bus0","prov_bus1","v_nom",
               "p_nom","mean_flow_mw","utilization"]].to_string(index=False))

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df[["Link","prov_bus0","prov_bus1","v_nom","p_nom","p_nom_opt",
        "mean_flow_mw","max_flow_mw","net_flow_mw","energy_twh",
        "utilization","is_tibet_xj"]].to_csv(
        os.path.join(OUTPUT_DIR, "dc_links_summary.csv"),
        index=False, float_format="%.4f"
    )
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'dc_links_summary.csv')}")

    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 1 — Istogramma distribuzione utilization (tutti i link)
    # Messaggio: la rete DC opera vicino alla saturazione.
    # I link Tibet/Xinjiang sono marcati come punti colorati.
    # ─────────────────────────────────────────────────────────────────────────
    util_norm = mcolors.Normalize(vmin=0, vmax=1)
    util_cmap = cm.RdYlGn_r   # rosso = saturo, verde = libero

    bins = np.linspace(0, 1, 21)   # 20 bins da 0 a 1
    util_all  = df[~df["is_tibet_xj"]]["utilization"].values
    util_txj  = df[ df["is_tibet_xj"]]["utilization"].values

    fig, ax = plt.subplots(figsize=(10, 6))

    # Colora ogni bin con il colore corrispondente alla sua utilization media
    n_counts, bin_edges, patches = ax.hist(
        util_all, bins=bins, edgecolor="white", linewidth=0.5, label="All DC links (non-Tibet/XJ)"
    )
    for patch, left_edge in zip(patches, bin_edges[:-1]):
        mid = left_edge + (bins[1] - bins[0]) / 2
        patch.set_facecolor(util_cmap(util_norm(mid)))

    # Tibet/Xinjiang: scatter sopra il grafico
    jitter = np.random.default_rng(42).uniform(-0.3, 0.3, len(util_txj))
    prov_labels = txj["prov_bus0"].values + " → " + txj["prov_bus1"].values
    for i, (u, label) in enumerate(zip(util_txj, prov_labels)):
        ax.scatter(u, -1.5 + jitter[i], s=120,
                   color=util_cmap(util_norm(u)),
                   edgecolors="black", linewidths=0.8, zorder=5)
        ax.annotate(label, (u, -1.5 + jitter[i]),
                    xytext=(0, 10), textcoords="offset points",
                    fontsize=7.5, ha="center", rotation=30)

    ax.axvline(0.20, color="gray", linestyle="--", linewidth=1.0, label="20%")
    ax.axvline(0.80, color="gray", linestyle=":",  linewidth=1.0, label="80%")

    ax.set_xlabel("Utilization rate  (mean |p0| / p_nom_opt)", fontsize=11)
    ax.set_ylabel("Number of DC links", fontsize=11)
    ax.set_title(
        f"Distribution of DC link utilization — CN2020  ({len(df)} links)\n"
        f"Median: {df['utilization'].median():.1%}  |  "
        f"{(df['utilization'] > 0.80).sum()} links > 80%  |  "
        f"{(df['utilization'] < 0.20).sum()} links < 20%",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    save(fig, "dc_utilization_hist.png")

    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 2 — Corridori Tibet / Xinjiang: capacità vs flusso
    # Messaggio: questi sono i corridori rilevanti per la teoria.
    # ─────────────────────────────────────────────────────────────────────────
    txj_sorted = txj.sort_values("p_nom", ascending=True).copy()
    labels = [
        f"{r['prov_bus0']} → {r['prov_bus1']}\n"
        f"({int(r['v_nom']) if not pd.isna(r['v_nom']) else 'n/a'} kV)"
        for _, r in txj_sorted.iterrows()
    ]
    y = np.arange(len(txj_sorted))
    bar_h = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_cap  = ax.barh(y + bar_h/2, txj_sorted["p_nom"] / 1e3,
                        height=bar_h, color="#4878d0", label="Installed capacity (p_nom)")
    bars_flow = ax.barh(y - bar_h/2, txj_sorted["mean_flow_mw"] / 1e3,
                        height=bar_h,
                        color=[util_cmap(util_norm(u)) for u in txj_sorted["utilization"]],
                        label="Mean actual flow  (colored by utilization)")

    # Annotate utilization rate
    for i, (_, row) in enumerate(txj_sorted.iterrows()):
        ax.text(
            row["mean_flow_mw"] / 1e3 + 0.05,
            y[i] - bar_h/2,
            f"{row['utilization']:.0%}",
            va="center", fontsize=9, color="black"
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Power  (GW)", fontsize=11)
    ax.set_title(
        "Tibet / Xinjiang DC corridors — installed capacity vs actual flow — CN2020\n"
        "Flow bar color: red = saturated, green = free capacity",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)

    # Colorbar for utilization
    sm = cm.ScalarMappable(cmap=util_cmap, norm=util_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.02, shrink=0.8)
    cbar.set_label("Utilization", fontsize=9)

    plt.tight_layout()
    save(fig, "dc_tibet_xinjiang.png")

    # ─────────────────────────────────────────────────────────────────────────
    # PLOT 3 — Mappa: Admin1 boundaries, link colorati per utilization
    # Plain matplotlib (no cartopy) per evitare artefatti.
    # Larghezza fissa per tutti i link. Rosso = saturo, verde = libero.
    # ─────────────────────────────────────────────────────────────────────────
    print("\nBuilding Admin1 geometries...")
    gadm   = gpd.read_file(GADM_FILE)
    china  = gadm[gadm["country"] == "CN"].copy()
    china["admin1"] = china["GADM_ID"].apply(bus_to_admin1)
    admin1_geo = china.dissolve(by="admin1").reset_index()[["admin1", "geometry"]]
    print(f"  Admin1 provinces: {len(admin1_geo)}")

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#f5f5f5")

    # Admin1 fill + borders only (no Admin2 artifacts)
    admin1_geo.plot(ax=ax, color="white", edgecolor="#444444",
                    linewidth=0.8, zorder=1)
    overlay_admin2_only(ax, linewidth=0.3, edgecolor="black", alpha=0.5)

    # DC links: colore = utilization, larghezza fissa
    for _, row in df.iterrows():
        b0_name = row["bus0"]
        b1_name = row["bus1"]
        if b0_name not in n.buses.index or b1_name not in n.buses.index:
            continue
        b0 = n.buses.loc[b0_name]
        b1 = n.buses.loc[b1_name]
        u  = row["utilization"]
        lw = 3.5 if row["is_tibet_xj"] else 1.8
        ax.plot(
            [b0.x, b1.x], [b0.y, b1.y],
            color=util_cmap(util_norm(u)),
            linewidth=lw, alpha=0.9,
            solid_capstyle="round", zorder=3,
        )

    # Colorbar
    sm = cm.ScalarMappable(cmap=util_cmap, norm=util_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, shrink=0.6)
    cbar.set_label("Utilization rate  (mean |p0| / p_nom_opt)", fontsize=10)

    ax.set_xlim(73, 136)
    ax.set_ylim(17, 54)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    n_low  = (df["utilization"] < 0.20).sum()
    n_high = (df["utilization"] > 0.80).sum()
    ax.set_title(
        f"DC link utilization — CN2020  |  "
        f"Thicker lines = Tibet/Xinjiang corridors\n"
        f"Red = saturated  |  Green = free capacity  |  "
        f"{n_high} links > 80%  |  {n_low} links < 20%",
        fontsize=12, fontweight="bold", pad=10
    )
    ax.grid(True, alpha=0.25, linewidth=0.4)
    plt.tight_layout()
    save(fig, "dc_map_utilization.png")

    print(f"\nDone. All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
