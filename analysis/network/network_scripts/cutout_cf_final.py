import atlite
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import geopandas as gpd
import os

CUTOUT_PATH = "cutouts/cutout-2013-era5.nc"
ADMIN2_FILE = "/p/tmp/ivanra/PyPSA-China-PIK/resources/data/regions/admin2_shapes.geojson"
REGIONS_ONSHORE_FILE = "resources/CN2060_loadCF/bus_regions/regions_onshore_elec_s_250.geojson"
OUTPUT_DIR = "analysis/network/output/CN2060_NUCAP_BIO_95/wind_spatial_analysis_bus"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading cutout...")
cutout = atlite.Cutout(path=CUTOUT_PATH)

# Dedup coordinate x quasi-duplicate (tolleranza 0.01) — fix verificato 30.06.2026
x_vals = cutout.data.x.values
keep = np.ones(len(x_vals), dtype=bool)
for i in range(1, len(x_vals)):
    if x_vals[i] - x_vals[i-1] < 0.01:
        keep[i] = False
print(f"Removing {(~keep).sum()} near-duplicate x cells")

cutout.data = cutout.data.isel(x=keep)
cutout.data = cutout.data.sel(x=slice(73, 136), y=slice(17, 54))

print("Computing capacity factors (Vestas_V112_3MW)...")
cap_factors = cutout.wind(turbine="Vestas_V112_3MW", capacity_factor=True)

nan_before = int(cap_factors.isnull().sum())
print(f"NaN before fill: {nan_before} / {cap_factors.size}")

# Interpolazione SOLO per display — non modifica i dati del modello
cap_factors_filled = cap_factors.interpolate_na(dim="x", method="linear").interpolate_na(dim="y", method="linear")
nan_after = int(cap_factors_filled.isnull().sum())
print(f"NaN after fill: {nan_after} / {cap_factors_filled.size}")
print(f"CF range (filled): min={float(cap_factors_filled.min()):.3f}  "
      f"mean={float(cap_factors_filled.mean()):.3f}  "
      f"max={float(cap_factors_filled.max()):.3f}")


def make_plot(overlay_mode, filename, title_suffix):
    """
    overlay_mode: 'admin' usa admin2_shapes (confini provinciali reali)
                  'regions' usa regions_onshore_elec_s_250 (celle del modello)
    """
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(14, 10), subplot_kw={"projection": proj})

    cap_factors_filled.plot(
        ax=ax, transform=proj, cmap="gist_earth_r", vmin=0, vmax=0.5,
        cbar_kwargs={
            "label": "CF_ERA5 (Vestas V112 3MW)  [ERA5 input, gaps interpolated for display]",
            "fraction": 0.025, "pad": 0.02, "shrink": 0.7,
        },
    )

    if overlay_mode == "admin" and os.path.exists(ADMIN2_FILE):
        admin = gpd.read_file(ADMIN2_FILE)
        admin.boundary.plot(ax=ax, color="black", linewidth=0.3, alpha=0.5, transform=proj)
        admin.dissolve("NAME_1").boundary.plot(ax=ax, color="black", linewidth=0.8, transform=proj)
    elif overlay_mode == "regions" and os.path.exists(REGIONS_ONSHORE_FILE):
        regions = gpd.read_file(REGIONS_ONSHORE_FILE)
        regions.boundary.plot(ax=ax, color="black", linewidth=0.4, alpha=0.7, transform=proj)
    else:
        print(f"[WARNING] Overlay file non trovato per mode={overlay_mode}")

    ax.set_title(
        f"Onshore wind CF per ERA5 cell — Vestas_V112_3MW\n"
        f"[ERA5 INPUT — gaps interpolated for display only]\n{title_suffix}",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.set_extent([73, 136, 17, 54], crs=proj)
    ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)

    out = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# Versione 1: confini amministrativi (province reali) — per lettura geografica generale
make_plot("admin", "map_onwind_cf_era5_cellres_admin.png",
          "Boundaries: administrative provinces (admin2_shapes)")

# Versione 2: regioni del modello (celle Voronoi clustering) — risposta diretta a Davide
make_plot("regions", "map_onwind_cf_era5_cellres_regions250.png",
          "Boundaries: model bus regions (regions_onshore_elec_s_250)")

print("\nDone. Two versions saved for comparison.")
