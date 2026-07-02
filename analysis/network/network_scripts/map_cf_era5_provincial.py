import pypsa
import sys
import os
import pandas as pd
import numpy as np
import geopandas as gpd
from province_utils import overlay_admin_boundaries
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from shapely.wkt import loads as wkt_loads
import warnings
warnings.filterwarnings("ignore")

GADM_FILE  = "resources/shapes/gadm_shapes.geojson"
OUTPUT_DIR = os.path.join(os.environ.get("PYPSA_OUTPUT_DIR", "analysis/network/output"), "map_cf_era5_provincial")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def bus_to_admin1(name):
    parts = str(name).split(".")
    return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else name

def admin1_to_provnum(code):
    try:
        return int(code.split(".")[1])
    except:
        return None

def save(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

def main():
    network_file = sys.argv[1]
    osm_file     = sys.argv[2]  # all_clean_generators.csv

    import re
    match = re.search(r'CN(\d{4})', network_file)
    year_label = f"CN{match.group(1)}" if match else "CN"

    print(f"Loading network: {network_file}")
    n = pypsa.Network(network_file)
    dt = (n.snapshots[1] - n.snapshots[0]).total_seconds() / 3600.0
    ore_anno = len(n.snapshots) * dt

    # --- CF ERA5 per provincia ---
    wind = n.generators[n.generators.carrier == 'onwind'].copy()
    wind['admin1'] = wind['bus'].apply(bus_to_admin1)

    p_max_pu = n.generators_t.p_max_pu.reindex(columns=wind.index, fill_value=0.0)
    era5_energy = (p_max_pu * wind['p_nom_opt'] * dt).sum()  # MWh per generatore

    wind['era5_MWh'] = era5_energy
    wind['CF_ERA5']  = era5_energy / (wind['p_nom_opt'] * ore_anno)

    prov_cf = wind.groupby('admin1').apply(
        lambda g: np.average(g['CF_ERA5'], weights=g['p_nom_opt']) if g['p_nom_opt'].sum() > 0 else np.nan
    ).rename('CF_ERA5_weighted')

    prov_cap = wind.groupby('admin1')['p_nom_opt'].sum() / 1e3  # GW

    # --- Geometrie provinciali ---
    gadm  = gpd.read_file(GADM_FILE)
    china = gadm[gadm["country"] == "CN"].copy()
    china["admin1"] = china["GADM_ID"].apply(bus_to_admin1)
    china = china.dissolve(by="admin1").reset_index()[["admin1", "geometry"]]

    gdf = china.merge(prov_cf.reset_index(), on="admin1", how="left")
    gdf = gdf.merge(prov_cap.reset_index().rename(columns={'p_nom_opt':'p_nom_GW'}), on="admin1", how="left")

    # --- Coordinate turbine OSM ---
    osm = pd.read_csv(osm_file)
    wind_osm = osm[osm['tags.generator:source'].str.contains('wind', case=False, na=False)].copy()
    wind_osm['lon'] = wind_osm['geometry'].apply(lambda g: wkt_loads(g).x)
    wind_osm['lat'] = wind_osm['geometry'].apply(lambda g: wkt_loads(g).y)

    # --- Plot: due pannelli affiancati ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    plt.rcParams.update({"font.size": 11, "figure.facecolor": "white"})

    xlim = (73, 136)
    ylim = (17, 54)

    # Pannello sinistro: CF ERA5 per provincia
    ax1 = axes[0]
    vmax = gdf['CF_ERA5_weighted'].max()
    gdf.plot(
        column='CF_ERA5_weighted', ax=ax1,
        cmap='YlOrRd',
        vmin=0, vmax=vmax,
        edgecolor='none', linewidth=0.0,
        legend=True,
        legend_kwds={"label": "Capacity factor ERA5 (onwind, weighted by p_nom)",
                     "orientation": "vertical", "shrink": 0.6, "pad": 0.02},
        missing_kwds={"color": "lightgray", "label": "No onwind"},
    )
    overlay_admin_boundaries(ax1, linewidth=0.6, edgecolor="black")
    ax1.set_xlim(xlim); ax1.set_ylim(ylim)
    ax1.set_title(f"ERA5 onwind capacity factor by province — {year_label}", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Longitude"); ax1.set_ylabel("Latitude")
    ax1.grid(True, alpha=0.3, linewidth=0.4)

    # Pannello destro: localizzazione turbine OSM + capacità installata per provincia
    ax2 = axes[1]
    gdf.plot(
        column='p_nom_GW', ax=ax2,
        cmap='Blues',
        vmin=0,
        edgecolor='none', linewidth=0.0,
        legend=True,
        legend_kwds={"label": "Installed onwind capacity (GW)",
                     "orientation": "vertical", "shrink": 0.6, "pad": 0.02},
        missing_kwds={"color": "lightgray", "label": "No onwind"},
    )
    overlay_admin_boundaries(ax2, linewidth=0.6, edgecolor="black")
    ax2.scatter(
        wind_osm['lon'], wind_osm['lat'],
        s=1.5, color='red', alpha=0.4, linewidths=0, label='OSM turbines'
    )
    ax2.set_xlim(xlim); ax2.set_ylim(ylim)
    ax2.set_title(f"Installed onwind capacity and OSM turbine locations — {year_label}", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Longitude"); ax2.set_ylabel("Latitude")
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3, linewidth=0.4)

    plt.tight_layout()
    save(fig, f"map_cf_era5_vs_location_{year_label}.png")
    print("Done.")

if __name__ == "__main__":
    main()
