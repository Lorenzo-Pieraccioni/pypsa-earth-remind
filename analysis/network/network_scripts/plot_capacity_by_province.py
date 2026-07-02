"""
plot_capacity_by_province.py
=============================
Province-level map of installed capacity for a given carrier.
Reusable across runs and carriers: edit NETWORK_PATH, REGIONS_PATH, CARRIER below.
"""

import os
import pypsa
import geopandas as gpd
from province_utils import overlay_admin_boundaries
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- only these need to change per run/carrier ----
NETWORK_PATH = "/p/tmp/lorenzop/pypsa-earth-ivan/results/CN2060_loadCF/networks/elec_s_425_ec_lcopt_Co2L0.0-3h.nc"
REGIONS_PATH = "/p/tmp/lorenzop/pypsa-earth-ivan/resources/CN2060_loadCF/bus_regions/regions_onshore_elec_s_425.geojson"
CARRIER = "nuclear"
OUTPUT_PATH = f"/p/tmp/lorenzop/pypsa-earth-ivan/analysis/network/output/maps/{CARRIER}_by_province.png"
# -----------------------------------------------------

GADM_NAMES = {
    "CN.1_1": "Anhui", "CN.2_1": "Beijing", "CN.3_1": "Chongqing",
    "CN.4_1": "Fujian", "CN.5_1": "Gansu", "CN.6_1": "Guangdong",
    "CN.7_1": "Guangxi", "CN.8_1": "Guizhou", "CN.9_1": "Hainan",
    "CN.10_1": "Hebei", "CN.11_1": "Heilongjiang", "CN.12_1": "Henan",
    "CN.13_1": "Hubei", "CN.14_1": "Hunan", "CN.15_1": "Jiangsu",
    "CN.16_1": "Jiangxi", "CN.17_1": "Jilin", "CN.18_1": "Liaoning",
    "CN.19_1": "Inner Mongolia", "CN.20_1": "Ningxia", "CN.21_1": "Qinghai",
    "CN.22_1": "Shaanxi", "CN.23_1": "Shandong", "CN.24_1": "Shanghai",
    "CN.25_1": "Shanxi", "CN.26_1": "Sichuan", "CN.27_1": "Tianjin",
    "CN.28_1": "Xinjiang", "CN.29_1": "Tibet", "CN.30_1": "Yunnan",
    "CN.31_1": "Zhejiang",
}

def to_admin1(series):
    """'CN.6.6_1_AC' -> 'CN.6_1', matching GADM_NAMES keys."""
    stripped = series.str.replace(r"_(AC|DC)$", "", regex=True)
    return stripped.str.split(".").str[:2].str.join(".") + "_1"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

n = pypsa.Network(NETWORK_PATH)
gen = n.generators[n.generators.carrier == CARRIER]
cap_by_bus = gen.groupby("bus")["p_nom_opt"].sum().div(1000)  # MW -> GW
cap_by_bus.name = "capacity_gw"

regions = gpd.read_file(REGIONS_PATH)
id_col = "name" if "name" in regions.columns else regions.columns[0]

regions = regions.merge(cap_by_bus, left_on=id_col, right_index=True, how="left")
regions["capacity_gw"] = regions["capacity_gw"].fillna(0.0)

n_matched = (regions["capacity_gw"] > 0).sum()
print(f"buses with capacity > 0 after merge: {n_matched} of {len(regions)}")
if n_matched == 0:
    print("WARNING: no match. sample network bus:", cap_by_bus.index[:3].tolist())
    print("sample regions id:", regions[id_col].head(3).tolist())

regions["admin1_id"] = to_admin1(regions[id_col])
regions["province"] = regions["admin1_id"].map(GADM_NAMES)
unmapped = regions["province"].isna().sum()
if unmapped > 0:
    print(f"WARNING: {unmapped} regions did not map to a province name")
    print(regions.loc[regions["province"].isna(), "admin1_id"].unique())

province_gdf = regions.dissolve(by="province", aggfunc="sum")

fig, ax = plt.subplots(figsize=(10, 8))
province_gdf.plot(
    column="capacity_gw",
    cmap="YlOrRd",
    edgecolor="none",
    linewidth=0.0,
    legend=True,
    legend_kwds={"label": f"{CARRIER.capitalize()} capacity (GW)", "shrink": 0.6},
    ax=ax,
)
overlay_admin_boundaries(ax, linewidth=0.6, edgecolor="black")
ax.set_title(f"{CARRIER.capitalize()} installed capacity by province")
ax.axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200)
print("saved to:", OUTPUT_PATH)
