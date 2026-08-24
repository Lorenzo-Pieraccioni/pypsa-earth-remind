#!/usr/bin/env python3
"""
plot_profile_cf_by_region.py

Genera mappe del capacity factor medio per bus a partire dai file
resources/{RUN}/renewable_profiles/profile_{carrier}.nc, per tutti i
carrier rinnovabili (onwind, solar, offwind-ac, offwind-dc).

IMPORTANTE (contesto epistemico):
Questa mappa mostra il CF DOPO l'aggregazione per bus (fatta da atlite in
build_renewable_profiles.py), NON il CF grezzo cella-per-cella del cutout.
Per onwind, le regioni del Nordest appaiono a 0.0 a causa del bug NaN->0.0
gia' diagnosticato (celle ERA5 mancanti azzerate silenziosamente da
convert_and_aggregate in atlite). Questo e' quindi il plot che DOCUMENTA il
bug, non il plot del cutout grezzo che Ivan chiede separatamente.

Mapping ID: profile_{carrier}.nc e' indicizzato per ID numerico grezzo
(coord 'bus'), non per nome CN.xx.yy. Il file regions_*.geojson fornisce la
corrispondenza: colonna 'name' = ID grezzo, 'shape_id' = nome finale.
Onshore/solar -> regions_onshore.geojson ; offshore -> regions_offshore.geojson.
(Verified fact, sessione 03.07.2026: mapping via regions_onshore.geojson,
non busmap_elec_s_250.csv.)
"""

import os
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt

# --- Configurazione ---------------------------------------------------------

RUN = os.environ.get("RUN_NAME", "CN2060_loadCF")
PROJECT_ROOT = "/p/tmp/lorenzop/pypsa-earth-ivan"

# Output dir: default esplicito al path richiesto, override via PYPSA_OUTPUT_DIR
DEFAULT_OUTPUT = os.path.join(
    PROJECT_ROOT, "analysis/network/output/CN2060_NUCAP_BIO_95",
    "profile_cf_by_region",
)
OUTPUT_DIR = os.environ.get("PYPSA_OUTPUT_DIR", DEFAULT_OUTPUT)

PROFILE_DIR = os.path.join(PROJECT_ROOT, "resources", RUN, "renewable_profiles")
REGIONS_DIR = os.path.join(PROJECT_ROOT, "resources", RUN, "bus_regions")

# carrier -> file regioni corretto
CARRIERS = {
    "onwind": "regions_onshore.geojson",
    "offwind-dc": "regions_offshore.geojson",
    "solar": "regions_onshore.geojson",
}


def plot_carrier(carrier, regions_file):
    profile_path = os.path.join(PROFILE_DIR, f"profile_{carrier}.nc")
    regions_path = os.path.join(REGIONS_DIR, regions_file)

    if not os.path.exists(profile_path):
        print(f"[SKIP] {carrier}: file profilo assente ({profile_path})")
        return
    if not os.path.exists(regions_path):
        print(f"[SKIP] {carrier}: file regioni assente ({regions_path})")
        return

    regions = gpd.read_file(regions_path)
    regions["name"] = regions["name"].astype(str)

    ds = xr.open_dataset(profile_path)
    if "profile" not in ds.data_vars:
        print(f"[SKIP] {carrier}: variabile 'profile' assente. "
              f"Vars presenti: {list(ds.data_vars)}")
        return

    profile_mean = ds["profile"].mean(dim="time").to_pandas()
    regions["profile_mean"] = regions["name"].map(profile_mean)

    matched = regions["profile_mean"].notna().sum()
    total = len(regions)
    zero = (regions["profile_mean"] == 0.0).sum()
    print(f"[{carrier}] regioni con dato: {matched}/{total}  |  "
          f"CF esattamente 0.0: {zero}")

    fig, ax = plt.subplots(figsize=(10, 8))
    regions.plot(
        column="profile_mean", cmap="viridis", legend=True, ax=ax,
        missing_kwds={"color": "lightgray", "label": "no data"},
    )
    ax.set_title(
        f"profile_{carrier}.nc — mean CF per bus (post-aggregation, {RUN})"
    )
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

    out_path = os.path.join(OUTPUT_DIR, f"profile_{carrier}_cf_map.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{carrier}] salvato: {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"RUN: {RUN}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}\n")
    for carrier, regions_file in CARRIERS.items():
        plot_carrier(carrier, regions_file)


if __name__ == "__main__":
    main()