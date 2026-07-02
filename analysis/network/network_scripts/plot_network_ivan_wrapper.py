"""
Wrapper standalone per plot_network di Ivan.
Uso:
  python plot_network_ivan_wrapper.py <network.nc> <output_dir> [capacity|energy|cost]
"""
import sys
import os
import yaml
import pypsa

# Aggiungi la cartella scripts di Ivan al path per le dipendenze
IVAN_SCRIPTS = "/p/tmp/ivanra/PyPSA-China-PIK/workflow/scripts"
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, IVAN_SCRIPTS)
sys.path.insert(0, THIS_DIR)

from plot_network import plot_network
from _pypsa_helpers import unscale_biomass_network

PLOT_CONFIG_PATH = os.path.join(THIS_DIR, "plot_config.yaml")
REGIONS_PATH = "/p/tmp/lorenzop/pypsa-earth-ivan/resources/CN2060_loadCF/bus_regions/regions_onshore_elec_s_425.geojson"

def main():
    if len(sys.argv) < 3:
        print("Uso: python plot_network_ivan_wrapper.py <network.nc> <output_dir> [capacity|energy|cost]")
        sys.exit(1)

    network_path = sys.argv[1]
    output_dir = sys.argv[2]
    metric = sys.argv[3] if len(sys.argv) > 3 else "capacity"
    os.makedirs(output_dir, exist_ok=True)

    with open(PLOT_CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    opts = config["plotting"]

    print(f"Loading network: {network_path}")
    n = pypsa.Network(network_path)

    save_path = os.path.join(output_dir, f"network_{metric}.png")
    print(f"Plotting {metric} map...")
    plot_network(
        n, opts,
        metric_type=metric,
        china_only=True,
        regions_path=REGIONS_PATH,
        save_path=save_path,
    )
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()
