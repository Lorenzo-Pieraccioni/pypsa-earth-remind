import atlite
import xarray as xr
import numpy as np
import os

CUTOUT_PATH = "cutouts/cutout-2013-era5.nc"

print("Loading full cutout (no bbox restriction)...")
cutout = atlite.Cutout(path=CUTOUT_PATH)

print(f"\nFull cutout x range: [{float(cutout.data.x.min()):.2f}, {float(cutout.data.x.max()):.2f}]")
print(f"Full cutout y range: [{float(cutout.data.y.min()):.2f}, {float(cutout.data.y.max()):.2f}]")
print(f"x spacing (dx check): {np.diff(cutout.data.x.values)[:5]}")

# 1. Check raw wnd100m for NaN in the stripe region BEFORE any bbox cut
print("\n=== Check 1: raw wnd100m NaN, full cutout, x in [125,138] ===")
raw_region = cutout.data.wnd100m.sel(x=slice(125, 138), y=slice(40, 53))
nan_count = int(raw_region.isnull().sum())
total_count = raw_region.size
print(f"NaN in raw wnd100m: {nan_count} / {total_count} ({100*nan_count/total_count:.2f}%)")

# 2. Check exact x values in stripe region — is it irregular spacing causing the bbox cut to miss cells?
print("\n=== Check 2: x coordinate values near 127-137 ===")
x_vals = cutout.data.x.sel(x=slice(125, 138)).values
print(f"Number of x cells in [125,138]: {len(x_vals)}")
print(f"x values: {x_vals}")
diffs = np.diff(x_vals)
print(f"Spacing between consecutive x (should be ~0.3): min={diffs.min():.4f} max={diffs.max():.4f}")
irregular = np.where(np.abs(diffs - 0.3) > 0.01)[0]
print(f"Irregular spacing points: {len(irregular)} (indices: {irregular[:10]})")

# 3. Now restrict to bbox as in the previous run and check NaN AFTER cutout.wind()
print("\n=== Check 3: NaN after cutout.wind(), restricted bbox (x:73-136) ===")
cutout2 = atlite.Cutout(path=CUTOUT_PATH)
cutout2.data = cutout2.data.sel(x=slice(73, 136), y=slice(17, 54))
cap_factors = cutout2.wind(turbine="Vestas_V112_3MW", capacity_factor=True)
nan_in_cf = int(cap_factors.isnull().sum())
total_cf = cap_factors.size
print(f"NaN in cap_factors (full bbox): {nan_in_cf} / {total_cf} ({100*nan_in_cf/total_cf:.2f}%)")

stripe_region_cf = cap_factors.sel(x=slice(125, 136), y=slice(40, 53))
nan_stripe = int(stripe_region_cf.isnull().sum())
total_stripe = stripe_region_cf.size
print(f"NaN in cap_factors, stripe region only: {nan_stripe} / {total_stripe} "
      f"({100*nan_stripe/total_stripe:.2f}%)")

# Which exact x columns have NaN
nan_by_x = cap_factors.sel(x=slice(125,136)).isnull().sum(dim="y")
print("\nNaN count per x-column in stripe region:")
for xv, n in zip(nan_by_x.x.values, nan_by_x.values):
    if n > 0:
        print(f"  x={xv:.2f}: {int(n)} NaN cells")
