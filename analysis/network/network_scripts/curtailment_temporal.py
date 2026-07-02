import pypsa
import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(os.environ.get("PYPSA_OUTPUT_DIR", "analysis/network/output"), "curtailment_temporal")
os.makedirs(OUTPUT_DIR, exist_ok=True)
WIND_CARRIERS = ['onwind', 'offwind-dc', 'offwind-ac']
SOLAR_CARRIERS = ['solar']

def main():
    network_file = sys.argv[1]
    import re
    match = re.search(r'CN(\d{4})', network_file)
    year_label = f"CN{match.group(1)}" if match else "CN"

    print(f"Loading: {network_file}")
    n = pypsa.Network(network_file)
    dt = (n.snapshots[1] - n.snapshots[0]).total_seconds() / 3600.0
    weights = n.snapshot_weightings.generators

    def curtailment_ts(carriers):
        gens = n.generators[n.generators.carrier.isin(carriers)]
        if gens.empty:
            return pd.Series(0.0, index=n.snapshots)
        p_max = n.generators_t.p_max_pu.reindex(columns=gens.index, fill_value=0.0)
        p_act = n.generators_t.p.reindex(columns=gens.index, fill_value=0.0)
        potential = (p_max * gens['p_nom_opt']).sum(axis=1)
        actual    = p_act.sum(axis=1)
        return (potential - actual).clip(lower=0)

    def generation_ts(carriers):
        gens = n.generators[n.generators.carrier.isin(carriers)]
        if gens.empty:
            return pd.Series(0.0, index=n.snapshots)
        p_act = n.generators_t.p.reindex(columns=gens.index, fill_value=0.0)
        return p_act.sum(axis=1)

    wind_curt  = curtailment_ts(WIND_CARRIERS)   # MW per timestep
    solar_gen  = generation_ts(SOLAR_CARRIERS)
    wind_gen   = generation_ts(WIND_CARRIERS)
    vre_gen    = solar_gen + wind_gen              # MW totale VRE

    # carico totale
    load_ts = n.loads_t.p_set.sum(axis=1)

    # VRE penetration = VRE / load per timestep
    vre_penetration = vre_gen / load_ts.replace(0, np.nan)

    # --- Statistiche curtailment temporale ---
    n_timesteps = len(n.snapshots)
    n_curt      = (wind_curt > 1.0).sum()  # timestep con curtailment > 1 MW
    print(f"\nTotal timesteps: {n_timesteps}")
    print(f"Timesteps with wind curtailment > 1 MW: {n_curt} ({100*n_curt/n_timesteps:.1f}%)")
    print(f"Max curtailment in single timestep: {wind_curt.max():.1f} MW")
    print(f"Mean curtailment when curtailing: {wind_curt[wind_curt>1].mean():.1f} MW")

    # --- Plot 1: distribuzione temporale del curtailment ---
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    ax1 = axes[0]
    ax1.fill_between(n.snapshots, wind_curt / 1e3, alpha=0.7, color='steelblue')
    ax1.set_ylabel("Wind curtailment (GW)")
    ax1.set_title(f"Temporal distribution of wind curtailment — {year_label}", fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.fill_between(n.snapshots, vre_gen / 1e3, alpha=0.6, color='orange', label='VRE generation (solar+wind)')
    ax2.fill_between(n.snapshots, load_ts / 1e3, alpha=0.3, color='gray', label='Load')
    ax2.set_ylabel("Power (GW)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.fill_between(n.snapshots, vre_penetration * 100, alpha=0.6, color='green')
    ax3.set_ylabel("VRE penetration (%)")
    ax3.set_xlabel("Time")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"curtailment_temporal_{year_label}.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")

    # --- Plot 2: curtailment vs VRE penetration (scatter) ---
    fig2, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        vre_penetration * 100,
        wind_curt / 1e3,
        alpha=0.3, s=4, c='steelblue'
    )
    ax.set_xlabel("VRE penetration (% of load)")
    ax.set_ylabel("Wind curtailment (GW)")
    ax.set_title(f"Wind curtailment vs VRE penetration — {year_label}", fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out2 = os.path.join(OUTPUT_DIR, f"curtailment_vs_penetration_{year_label}.png")
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved: {out2}")

    # --- CSV: timestep-level data ---
    df_out = pd.DataFrame({
        'snapshot': n.snapshots,
        'wind_curtailment_MW': wind_curt.values,
        'wind_gen_MW': wind_gen.values,
        'solar_gen_MW': solar_gen.values,
        'vre_gen_MW': vre_gen.values,
        'load_MW': load_ts.values,
        'vre_penetration': vre_penetration.values,
    })
    csv_out = os.path.join(OUTPUT_DIR, f"curtailment_temporal_{year_label}.csv")
    df_out.to_csv(csv_out, index=False, float_format='%.2f')
    print(f"Saved: {csv_out}")

if __name__ == "__main__":
    main()
