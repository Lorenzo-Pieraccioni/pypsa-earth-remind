import pypsa
import sys

network_file = sys.argv[1] if len(sys.argv) > 1 else \
    "results/CN2020_03_hydro_solar1N/networks/elec_s_250_ec_lcopt_3h.nc"

n = pypsa.Network(network_file)
weights = n.snapshot_weightings.generators

print(f"\n=== WIND CURTAILMENT ANALYSIS ===")
print(f"File: {network_file}\n")
print(f"{'Carrier':<15} {'Cap(GW)':>8} {'Pot(TWh)':>10} {'Act(TWh)':>10} {'Curt(TWh)':>10} {'Curt%':>7} {'CF_ERA5':>8} {'CF_act':>7}")
print("-" * 85)

total_pot, total_act, total_curt = 0, 0, 0

for carrier in ['onwind', 'offwind-dc', 'offwind-ac']:
    gens = n.generators[n.generators.carrier == carrier]
    if gens.empty:
        continue
    cap = gens.p_nom_opt.sum() / 1e3
    p_max = n.generators_t.p_max_pu.reindex(columns=gens.index).fillna(1.0)
    potential = (p_max.multiply(gens.p_nom_opt).multiply(weights, axis=0)).sum().sum() / 1e6
    actual = (n.generators_t.p.reindex(columns=gens.index).fillna(0).multiply(weights, axis=0)).sum().sum() / 1e6
    curt = potential - actual
    curt_pct = 100 * curt / potential if potential > 0 else 0
    cf_era5 = potential / (cap * 8.76) if cap > 0 else 0
    cf_act = actual / (cap * 8.76) if cap > 0 else 0
    total_pot += potential
    total_act += actual
    total_curt += curt
    print(f"{carrier:<15} {cap:>8.1f} {potential:>10.1f} {actual:>10.1f} {curt:>10.1f} {curt_pct:>6.1f}% {cf_era5:>8.3f} {cf_act:>7.3f}")

print("-" * 85)
total_cap = sum(n.generators[n.generators.carrier == c].p_nom_opt.sum() / 1e3
                for c in ['onwind', 'offwind-dc', 'offwind-ac'])
print(f"{'TOTAL':<15} {total_cap:>8.1f} {total_pot:>10.1f} {total_act:>10.1f} {total_curt:>10.1f} {100*total_curt/total_pot:>6.1f}%")
print(f"\nEmber reference: see handoff")
print(f"ERA5 potential vs actual gap: {total_pot - total_act:.1f} TWh")
