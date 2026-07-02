import pypsa
import sys
import pandas as pd
import numpy as np

network_file = sys.argv[1] if len(sys.argv) > 1 else \
    "results/CN2020_LCF/networks/elec_s_425_ec_lcopt_3h.nc"

n = pypsa.Network(network_file)
weights = n.snapshot_weightings.generators

print(f"\n=== WIND CURTAILMENT PROVINCIAL ANALYSIS ===")
print(f"File: {network_file}\n")

rows = []

for carrier in ['onwind', 'offwind-dc', 'offwind-ac']:
    gens = n.generators[n.generators.carrier == carrier].copy()
    if gens.empty:
        continue

    gens['province'] = gens['bus'].str.extract(r'CN\.(\d+)\.')

    p_max = n.generators_t.p_max_pu.reindex(columns=gens.index).fillna(1.0)
    p_act = n.generators_t.p.reindex(columns=gens.index).fillna(0.0)

    potential_gen = p_max.multiply(gens.p_nom_opt).multiply(weights, axis=0).sum(axis=0) / 1e6
    actual_gen    = p_act.multiply(weights, axis=0).sum(axis=0) / 1e6

    gens['potential_TWh'] = potential_gen
    gens['actual_TWh']    = actual_gen
    gens['p_nom_GW']      = gens['p_nom_opt'] / 1e3

    prov = gens.groupby('province').agg(
        p_nom_GW    =('p_nom_GW',     'sum'),
        potential_TWh=('potential_TWh','sum'),
        actual_TWh  =('actual_TWh',   'sum'),
    ).reset_index()

    prov['carrier']      = carrier
    prov['curt_TWh']     = prov['potential_TWh'] - prov['actual_TWh']
    prov['curt_pct']     = 100 * prov['curt_TWh'] / prov['potential_TWh']
    prov['CF_ERA5']      = prov['potential_TWh'] / (prov['p_nom_GW'] * 8.76)
    prov['CF_act']       = prov['actual_TWh']    / (prov['p_nom_GW'] * 8.76)

    rows.append(prov)

result = pd.concat(rows, ignore_index=True)

# aggiungi coordinate centroide per identificare la provincia
buses = n.buses[['x', 'y']].copy()
gens_all = n.generators[n.generators.carrier.isin(['onwind','offwind-dc','offwind-ac'])].copy()
gens_all['province'] = gens_all['bus'].str.extract(r'CN\.(\d+)\.')
gens_all = gens_all.join(buses, on='bus')
centroids = gens_all.groupby('province')[['x','y']].mean().rename(columns={'x':'lon','y':'lat'})
result = result.join(centroids, on='province')

result = result.sort_values(['carrier','potential_TWh'], ascending=[True, False])

print(f"{'Carrier':<12} {'Prov':>5} {'lon':>7} {'lat':>6} {'Cap(GW)':>8} {'Pot(TWh)':>9} {'Act(TWh)':>9} {'Curt(TWh)':>10} {'Curt%':>7} {'CF_ERA5':>8} {'CF_act':>7}")
print("-" * 100)

for _, row in result.iterrows():
    print(f"{row['carrier']:<12} {row['province']:>5} {row['lon']:>7.1f} {row['lat']:>6.1f} "
          f"{row['p_nom_GW']:>8.1f} {row['potential_TWh']:>9.1f} {row['actual_TWh']:>9.1f} "
          f"{row['curt_TWh']:>10.2f} {row['curt_pct']:>6.1f}% {row['CF_ERA5']:>8.3f} {row['CF_act']:>7.3f}")

print("-" * 100)
result.to_csv(sys.argv[2] if len(sys.argv) > 2 else 'wind_curtailment_provincial.csv', index=False)
print(f"\nCSV salvato.")
