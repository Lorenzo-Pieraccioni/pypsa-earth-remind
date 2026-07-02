import atlite
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import geopandas as gpd
import os

CUTOUT_PATH = "cutouts/cutout-2013-era5.nc"
ADMIN2_FILE = "/p/tmp/ivanra/PyPSA-China-PIK/resources/data/regions/admin2_shapes.geojson"
OUTPUT_DIR = "analysis/network/output/CN2060_NUCAP_BIO_95/wind_spatial_analysis_bus"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading cutout...")
cutout = atlite.Cutout(path=CUTOUT_PATH)

# Bounding box Cina — riduce il dominio rispetto al cutout completo (tutta Asia)
print("Restricting to China bounding box...")
cutout.data = cutout.data.sel(x=slice(73, 136), y=slice(17, 54))

print("Computing capacity factors (Vestas_V112_3MW)...")
cap_factors = cutout.wind(turbine="Vestas_V112_3MW", capacity_factor=True)

print(f"CF range: min={float(cap_factors.min()):.3f}  "
      f"mean={float(cap_factors.mean()):.3f}  "
      f"max={float(cap_factors.max()):.3f}")

proj = ccrs.PlateCarree()
fig, ax = plt.subplots(figsize=(14, 10), subplot_kw={"projection": proj})
cap_factors.plot(
    ax=ax, transform=proj, cmap="YlOrRd", vmin=0, vmax=0.6,
    cbar_kwargs={"label": "CF_ERA5 (Vestas V112 3MW)  [ERA5 input]",
                 "fraction": 0.025, "pad": 0.02, "shrink": 0.7},
)
if os.path.exists(ADMIN2_FILE):
    admin = gpd.read_file(ADMIN2_FILE)
    admin.boundary.plot(ax=ax, color="black", linewidth=0.3, alpha=0.5, transform=proj)
    admin.dissolve("NAME_1").boundary.plot(ax=ax, color="black", linewidth=0.8, transform=proj)

ax.set_title(
    "Onshore wind CF per ERA5 cell (0.3°) — Vestas_V112_3MW, China bbox\n"
    "[ERA5 INPUT — independent of solver]",
    fontsize=12, fontweight="bold", pad=12,
)
ax.set_extent([73, 136, 17, 54], crs=proj)
ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)

out = os.path.join(OUTPUT_DIR, "map_onwind_cf_era5_cellres_china.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
