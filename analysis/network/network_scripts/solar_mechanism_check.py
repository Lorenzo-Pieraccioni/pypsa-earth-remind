import os
import sys
import numpy as np
import pandas as pd
import pypsa

network_path = sys.argv[1]
output_csv = sys.argv[2]
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

n = pypsa.Network(network_path)
solar_i = n.generators.query("carrier == 'solar'").index

cf = n.generators_t.p_max_pu[solar_i].mean()
p_nom = n.generators.loc[solar_i, "p_nom"]
p_nom_max = n.generators.loc[solar_i, "p_nom_max"].replace([np.inf, -np.inf], np.nan)
bus = n.generators.loc[solar_i, "bus"]
province = bus.map(lambda b: ".".join(str(b).split(".")[:2]))

bus_load_mw = n.loads_t.p_set.mean()
load_bus_province = n.loads.bus.map(lambda b: ".".join(str(b).split(".")[:2]))
province_load_mw = bus_load_mw.groupby(load_bus_province).sum()
gen_province_load = province.map(province_load_mw).fillna(0)

mask = p_nom_max.notna() & (p_nom_max > 0)
w_pot = (cf * p_nom_max)[mask]
w_load = (cf * gen_province_load)[mask]
corr_pot = np.corrcoef(p_nom[mask], w_pot)[0, 1]
corr_load = np.corrcoef(p_nom[mask], w_load)[0, 1]

print(f"=== {network_path} ===")
print(f"solar generators total: {len(solar_i)} | with finite p_nom_max: {mask.sum()}")
print(f"corr(p_nom, CF x p_nom_max)      = {corr_pot:.4f}")
print(f"corr(p_nom, CF x province_load)  = {corr_load:.4f}")
print("--> method used:", "CF x potential" if corr_pot > corr_load else "load x CF")

df = pd.DataFrame({
    "province": province,
    "cf": cf,
    "p_nom_gw": p_nom / 1e3,
    "p_nom_max_gw": p_nom_max / 1e3,
})
grp = df.groupby("province")
table = pd.DataFrame({
    "p_nom_gw": grp["p_nom_gw"].sum(),
    "p_nom_max_gw": grp["p_nom_max_gw"].sum(),
})
table["cf_mean"] = grp.apply(
    lambda g: (g["cf"] * g["p_nom_gw"]).sum() / g["p_nom_gw"].sum() if g["p_nom_gw"].sum() > 0 else np.nan
)
table["load_twh"] = table.index.map(province_load_mw).fillna(0) * 8760 / 1e6
table["solar_twh"] = table["p_nom_gw"] * table["cf_mean"] * 8.76
table["solar_load_ratio"] = table["solar_twh"] / table["load_twh"]
table = table.sort_values("p_nom_max_gw", ascending=False)
table.to_csv(output_csv)

print(table.round(4).to_string())
nat_cf = (table["p_nom_gw"] * table["cf_mean"]).sum() / table["p_nom_gw"].sum()
print(f"\nNational solar p_nom (GW):     {table['p_nom_gw'].sum():.2f}")
print(f"National solar p_nom_max (GW): {table['p_nom_max_gw'].sum():.2f}")
print(f"National CF (cap-weighted):    {nat_cf:.4f}")
print(f"National solar generation (TWh): {table['solar_twh'].sum():.2f}")
