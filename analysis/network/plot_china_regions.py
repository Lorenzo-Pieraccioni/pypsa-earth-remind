"""
plot_china_regions.py v6
========================
Mappa della Cina con confini prefetturali e provinciali geometricamente separati.

Approccio corretto:
  - i confini interni delle prefetture vengono estratti come differenza geometrica
    tra i boundary delle prefetture e i boundary delle province
  - questo evita la sovrapposizione visiva che causava i trattini neri
  - i confini provinciali vengono disegnati sopra come linee spesse nere

Output salvato in analysis/tools/output/china_regions_map.png

Uso:
  python analysis/tools/plot_china_regions.py
"""

import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import geopandas as gpd
import shapely.geometry as sg
import shapely.ops as so

# ── PARAMETERS ────────────────────────────────────────────────────────────────

GADM_FILE    = "resources/shapes/gadm_shapes.geojson"
OUTPUT_DIR   = "analysis/network/output"
OUTPUT_FILE  = "china_regions_map.png"
EXTENT       = [70, 142, 14, 57]

NEIGHBOR_LABELS = [
    "Russia", "Mongolia", "Kazakhstan", "Kyrgyzstan", "Tajikistan",
    "Afghanistan", "Pakistan", "India", "Nepal", "Bhutan",
    "Myanmar", "Laos", "Vietnam", "North Korea", "South Korea",
    "Japan", "Philippines", "Bangladesh",
]

# ── PROVINCE MAPPING ──────────────────────────────────────────────────────────

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

MUNICIPALITY_NUMS = {2, 27, 24, 3}

REGIONAL_GROUPS = {
    "MC": ("Municipalities",  "#e41a1c", [2, 27, 24, 3]),
    "NC": ("North China",     "#4393c3", [10, 25, 19, 12]),
    "NE": ("Northeast",       "#4daf4a", [18, 17, 11]),
    "NW": ("Northwest",       "#f4a620", [22, 5, 21, 20, 28]),
    "EC": ("East Coast",      "#7b4fa6", [15, 31, 1, 4, 23]),
    "CC": ("Central China",   "#8c6d3f", [13, 14, 16]),
    "SW": ("Southwest",       "#f781bf", [26, 30, 8, 29]),
    "SC": ("South China",     "#888888", [6, 7, 9]),
}

# ─────────────────────────────────────────────────────────────────────────────


def extract_province_number(gadm_id):
    m = re.match(r"CN\.(\d+)\.", gadm_id)
    return int(m.group(1)) if m else None


def build_prov_to_color():
    mapping = {}
    for _, (_, color, pnums) in REGIONAL_GROUPS.items():
        for p in pnums:
            mapping[p] = color
    return mapping


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading GADM...")
    gdf = gpd.read_file(GADM_FILE)
    china = gdf[gdf["country"] == "CN"].copy().to_crs("EPSG:4326")
    china["prov_num"] = china["GADM_ID"].apply(extract_province_number)
    china["prov_name"] = china["prov_num"].map(GADM_PROVINCE)
    prov_to_color = build_prov_to_color()
    china["color"] = china["prov_num"].map(
        lambda x: prov_to_color.get(x, "#cccccc"))

    # Dissolve to province level
    provinces = china.dissolve(by="prov_num").reset_index()
    provinces["prov_name"] = provinces["prov_num"].map(GADM_PROVINCE)
    provinces["is_muni"] = provinces["prov_num"].isin(MUNICIPALITY_NUMS)
    china_outer = china.dissolve()

    # ── Geometric extraction of internal prefecture borders ───────────────────
    print("Computing internal prefecture borders...")

    # All prefecture boundaries unioned
    all_pref_boundaries = so.unary_union(
        [row.geometry.boundary for _, row in china.iterrows()]
    )

    # All provincial boundaries unioned
    all_prov_boundaries = so.unary_union(
        [row.geometry.boundary for _, row in provinces.iterrows()]
    )

    # Internal prefecture borders = prefecture boundaries MINUS provincial boundaries
    # We buffer the provincial boundary slightly to ensure clean subtraction
    prov_boundary_buffered = all_prov_boundaries.buffer(0.001)
    internal_pref_borders = all_pref_boundaries.difference(prov_boundary_buffered)

    print("Internal prefecture borders computed.")

    # ── Natural Earth neighbors ───────────────────────────────────────────────
    print("Loading Natural Earth...")
    shpfile = shpreader.natural_earth(
        resolution="50m", category="cultural", name="admin_0_countries")
    reader = shpreader.Reader(shpfile)
    neighbor_records = []
    for rec in reader.records():
        name = rec.attributes.get("NAME", "")
        if name in NEIGHBOR_LABELS:
            neighbor_records.append((name, rec.geometry))

    # ── PLOT ──────────────────────────────────────────────────────────────────
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(22, 15), subplot_kw={"projection": proj})
    ax.set_extent(EXTENT, crs=proj)

    # z=0: ocean
    ax.add_feature(cfeature.OCEAN.with_scale("50m"),
                   facecolor="#c6e8f5", zorder=0)
    # z=1: land background
    ax.add_feature(cfeature.LAND.with_scale("50m"),
                   facecolor="#eeeeee", zorder=1)

    # z=2: neighbor fills + borders
    for name, geom in neighbor_records:
        ax.add_geometries([geom], crs=proj,
                          facecolor="#e0e0e0", edgecolor="#888888",
                          linewidth=0.9, zorder=2)

    # z=3: neighbor labels
    box = sg.box(EXTENT[0], EXTENT[2], EXTENT[1], EXTENT[3])
    for name, geom in neighbor_records:
        try:
            clipped = geom.intersection(box)
            if clipped.is_empty:
                continue
            cx, cy = clipped.centroid.x, clipped.centroid.y
            if not (EXTENT[0] <= cx <= EXTENT[1] and
                    EXTENT[2] <= cy <= EXTENT[3]):
                continue
            ax.text(cx, cy, name, fontsize=6.5,
                    ha="center", va="center",
                    color="#444444", fontstyle="italic",
                    transform=proj, zorder=3,
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white", edgecolor="none",
                              alpha=0.55))
        except Exception:
            pass

    # z=4: prefecture FILL only (no edges)
    for _, row in china.iterrows():
        ax.add_geometries([row.geometry], crs=proj,
                          facecolor=row["color"], alpha=0.65,
                          edgecolor="none", linewidth=0,
                          zorder=4)

    # z=5: internal prefecture borders only (thin grey)
    ax.add_geometries(
        [internal_pref_borders], crs=proj,
        facecolor="none",
        edgecolor="#777777",
        linewidth=0.5,
        zorder=5,
    )

    # z=6: provincial boundaries (thick black)
    for _, row in provinces[~provinces["is_muni"]].iterrows():
        ax.add_geometries([row.geometry], crs=proj,
                          facecolor="none",
                          edgecolor="#111111", linewidth=1.9,
                          zorder=6)

    # z=7: municipality boundaries (extra thick)
    for _, row in provinces[provinces["is_muni"]].iterrows():
        ax.add_geometries([row.geometry], crs=proj,
                          facecolor="none",
                          edgecolor="#111111", linewidth=3.4,
                          zorder=7)

    # z=8: China outer boundary
    for _, row in china_outer.iterrows():
        ax.add_geometries([row.geometry], crs=proj,
                          facecolor="none",
                          edgecolor="#000000", linewidth=2.3,
                          zorder=8)

    # z=9: province name labels
    for _, row in provinces.iterrows():
        name = row["prov_name"]
        if not name:
            continue
        centroid = row.geometry.centroid
        area = row.geometry.area
        fs = 9 if area > 8 else 8 if area > 2 else 7 if area > 0.5 else 6
        ax.text(centroid.x, centroid.y, name,
                fontsize=fs, ha="center", va="center",
                color="#111111", fontweight="bold",
                transform=proj, zorder=9,
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="white", edgecolor="none",
                          alpha=0.82))

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                      color="gray", alpha=0.4, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    # Legend
    handles = []
    for code, (label, color, pnums) in REGIONAL_GROUPS.items():
        names = [GADM_PROVINCE.get(p, str(p)) for p in pnums]
        lbl = f"[{code}] {label}\n  " + ", ".join(names)
        handles.append(mpatches.Patch(
            facecolor=color, alpha=0.75,
            edgecolor="#333333", linewidth=0.6,
            label=lbl))

    ax.legend(handles=handles, loc="lower left",
              fontsize=7.5, framealpha=0.95,
              title="Regional Groups", title_fontsize=9.5,
              handlelength=1.6, borderpad=1.0, labelspacing=0.75)

    ax.set_title(
        "China — Administrative Map\n"
        "GADM2 prefectures (grey borders) + GADM1 provinces (thick black borders)\n"
        "Colored by regional energy group  |  Extra-thick borders = municipalities",
        fontsize=12, pad=14)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")