#!/bin/bash
set -e
echo "=== results/ ==="
[ -d "results/CN2020" ] && mv results/CN2020 results/CN2020_03_hydro_solar1N && echo "✓ CN2020 → CN2020_03_hydro_solar1N"
[ -d "results/networks/CN2060_100pct_Admin2" ] && mv results/networks/CN2060_100pct_Admin2 results/networks/CN2060_100_01_base && echo "✓ CN2060_100pct_Admin2 → CN2060_100_01_base"
[ -d "results/networks/CN2060_95pct_Admin2" ] && mv results/networks/CN2060_95pct_Admin2 results/networks/CN2060_95_01_base && echo "✓ CN2060_95pct_Admin2 → CN2060_95_01_base"
mkdir -p results/networks/archive/old_runs
for d in fix-hydro geospatial geospatial_v2 geospatial_v2_3h geospatial_v2_CN2025_3h geospatial_v2_approccio1 geospatial_v2_fix1 geospatial_v2_fix2_code40 geospatial_v2_fix2_irena2024 noCO2; do
    [ -d "results/networks/$d" ] && mv results/networks/$d results/networks/archive/old_runs/$d && echo "✓ $d → archive/old_runs/"
done
echo ""
echo "=== analysis/network/output/ ==="
[ -d "analysis/network/output/CN2020" ] && mv analysis/network/output/CN2020 analysis/network/output/CN2020_00_baseline && echo "✓ CN2020 → CN2020_00_baseline"
[ -d "analysis/network/output/CN2020_fix_classification" ] && mv analysis/network/output/CN2020_fix_classification analysis/network/output/CN2020_02_hydro && echo "✓ CN2020_fix_classification → CN2020_02_hydro"
[ -d "analysis/network/output/CN2020_solar_fix" ] && mv analysis/network/output/CN2020_solar_fix analysis/network/output/CN2020_03_hydro_solar1N && echo "✓ CN2020_solar_fix → CN2020_03_hydro_solar1N"
[ -d "analysis/network/output/CN2024" ] && mv analysis/network/output/CN2024 analysis/network/output/CN2024_00_old && echo "✓ CN2024 → CN2024_00_old"
[ -d "analysis/network/output/CN2060_95pct" ] && mv analysis/network/output/CN2060_95pct analysis/network/output/CN2060_95_00_base && echo "✓ CN2060_95pct → CN2060_95_00_base"
[ -d "analysis/network/output/CN2060_100pct" ] && mv analysis/network/output/CN2060_100pct analysis/network/output/CN2060_100_00_base && echo "✓ CN2060_100pct → CN2060_100_00_base"
echo ""
echo "=== analysis/old_output/ ==="
[ -d "analysis/old_output" ] && tar -czf analysis/old_output_archive_01062026.tar.gz analysis/old_output/ && rm -rf analysis/old_output/ && echo "✓ old_output → old_output_archive_01062026.tar.gz"
echo ""
echo "=== DONE — nothing deleted ==="
