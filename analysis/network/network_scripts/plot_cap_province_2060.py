import sys, os
sys.path.insert(0, "analysis/network/network_scripts")
import pypsa, geopandas as gpd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from province_utils import overlay_admin_boundaries

GADM_NAMES = {"CN.1_1":"Anhui","CN.2_1":"Beijing","CN.3_1":"Chongqing","CN.4_1":"Fujian","CN.5_1":"Gansu","CN.6_1":"Guangdong","CN.7_1":"Guangxi","CN.8_1":"Guizhou","CN.9_1":"Hainan","CN.10_1":"Hebei","CN.11_1":"Heilongjiang","CN.12_1":"Henan","CN.13_1":"Hubei","CN.14_1":"Hunan","CN.15_1":"Jiangsu","CN.16_1":"Jiangxi","CN.17_1":"Jilin","CN.18_1":"Liaoning","CN.19_1":"Inner Mongolia","CN.20_1":"Ningxia","CN.21_1":"Qinghai","CN.22_1":"Shaanxi","CN.23_1":"Shandong","CN.24_1":"Shanghai","CN.25_1":"Shanxi","CN.26_1":"Sichuan","CN.27_1":"Tianjin","CN.28_1":"Xinjiang","CN.29_1":"Tibet","CN.30_1":"Yunnan","CN.31_1":"Zhejiang"}
REGIONS = "resources/CN2060_loadCF/bus_regions/regions_onshore_elec_s_425.geojson"

net_path, carrier, out_path, scenario = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

def to_admin1(s):
    return s.str.replace(r"_(AC|DC)$","",regex=True).str.split(".").str[:2].str.join(".")+"_1"

n = pypsa.Network(net_path)
comp = n.storage_units if carrier in ["battery","H2"] else n.generators
gen = comp[comp.carrier == carrier]
cap_by_bus = gen.groupby("bus")["p_nom_opt"].sum().div(1000)
cap_by_bus.name = "capacity_gw"
regions = gpd.read_file(REGIONS)
id_col = "name" if "name" in regions.columns else regions.columns[0]
regions = regions.merge(cap_by_bus, left_on=id_col, right_index=True, how="left")
regions["capacity_gw"] = regions["capacity_gw"].fillna(0.0)
regions["admin1_id"] = to_admin1(regions[id_col])
regions["province"] = regions["admin1_id"].map(GADM_NAMES)
province_gdf = regions.dissolve(by="province", aggfunc="sum")
fig, ax = plt.subplots(figsize=(10,8))
province_gdf.plot(column="capacity_gw", cmap="YlOrRd", edgecolor="none", linewidth=0.0, legend=True, legend_kwds={"label":f"{carrier} capacity (GW)","shrink":0.6}, ax=ax)
overlay_admin_boundaries(ax, linewidth=0.6, edgecolor="black")
ax.set_title(f"{carrier} installed capacity by province — {scenario}")
ax.axis("off")
plt.tight_layout()
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=200)
print(f"Saved: {out_path}")
