#!/usr/bin/env bash
# =============================================================================
# run_analysis.sh
# Esegue tutti gli script di analisi su un run CN2020 o CN2024.
#
# Uso:
#   bash analysis/network/run_analysis.sh <RUN_NAME> <YEAR>
#
# Esempi:
#   bash analysis/network/run_analysis.sh CN2020_LCF 2020
#   bash analysis/network/run_analysis.sh CN2024_LCF 2024
#
# Output: analysis/network/output/<RUN_NAME>/<script_name>/
# Log:    analysis/network/output/<RUN_NAME>/run_analysis.log
# =============================================================================

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Uso: bash $0 <RUN_NAME> <YEAR>"
    echo "     YEAR deve essere 2020 o 2024"
    exit 1
fi

RUN_NAME="$1"
YEAR="$2"

if [[ "$YEAR" != "2020" && "$YEAR" != "2024" ]]; then
    echo "ERROR: YEAR deve essere 2020 o 2024 (ricevuto: $YEAR)"
    exit 1
fi

if [ "$YEAR" == "2020" ]; then
    SOLAR_EMBER=261.1;  SOLAR_IRENA=253.4
    HYDRO_EMBER=1321.7; HYDRO_IRENA=338.7
    NUCLEAR_EMBER=366.2; NUCLEAR_IRENA=49.9; NUCLEAR_REF_DATE="2020-12-31"
elif [ "$YEAR" == "2024" ]; then
    SOLAR_EMBER=839.0;  SOLAR_IRENA=885.7
    HYDRO_EMBER=1354.2; HYDRO_IRENA=377.3
    NUCLEAR_EMBER=450.9; NUCLEAR_IRENA=60.8; NUCLEAR_REF_DATE="2024-12-31"
fi

NETWORK="results/${RUN_NAME}/networks/elec_s_425_ec_lcopt_3h.nc"
BASE_OUT="analysis/network/output/${RUN_NAME}"
SCRIPTS="analysis/network/network_scripts"
SHAPES="resources/shapes/gadm_shapes.geojson"
IAEA_CSV="data/iaea_china_reactors_2026.csv"

if [ ! -f "$NETWORK" ]; then
    echo "ERROR: network file not found: $NETWORK"; exit 1
fi

mkdir -p "$BASE_OUT"
LOG="$BASE_OUT/run_analysis.log"
echo "=== run_analysis.sh — $(date) ===" | tee "$LOG"
echo "RUN_NAME : $RUN_NAME" | tee -a "$LOG"
echo "YEAR     : $YEAR"     | tee -a "$LOG"
echo "Network  : $NETWORK"  | tee -a "$LOG"
echo "" | tee -a "$LOG"

run_script() {
    local label="$1"; shift
    echo "--- $label ---" | tee -a "$LOG"
    if "$@" >> "$LOG" 2>&1; then
        echo "OK: $label" | tee -a "$LOG"
    else
        echo "FAILED: $label (exit $?)" | tee -a "$LOG"
    fi
    echo "" | tee -a "$LOG"
}

run_script "inspect_network" \
    env PYPSA_OUTPUT_DIR="$BASE_OUT/inspect" \
    python "$SCRIPTS/inspect_network.py" "$NETWORK" --year "$YEAR"

run_script "curtailment_analysis" \
    env PYPSA_OUTPUT_DIR="$BASE_OUT/curtailment" \
    python "$SCRIPTS/curtailment_analysis.py" "$NETWORK"

run_script "dc_corridor_analysis" \
    env PYPSA_OUTPUT_DIR="$BASE_OUT/dc_corridor" \
    python "$SCRIPTS/dc_corridor_analysis.py" "$NETWORK"

run_script "duck_curve_analysis" \
    env PYPSA_OUTPUT_DIR="$BASE_OUT/duck_curve" \
    python "$SCRIPTS/duck_curve_analysis.py" "$NETWORK"

run_script "duck_curve_analysis_regions" \
    env PYPSA_OUTPUT_DIR="$BASE_OUT/duck_curve_regions" \
    python "$SCRIPTS/duck_curve_analysis_regions.py" "$NETWORK"

run_script "hydro_spillage_analysis" \
    python "$SCRIPTS/hydro_spillage_analysis.py" "$NETWORK" \
        --year "$YEAR" --ember_twh "$HYDRO_EMBER" \
        --irena_gw "$HYDRO_IRENA" --shapes "$SHAPES"

run_script "nuclear_analysis" \
    python "$SCRIPTS/nuclear_analysis.py" "$NETWORK" \
        --year "$YEAR" --ember_twh "$NUCLEAR_EMBER" \
        --irena_gw "$NUCLEAR_IRENA" --ref_date "$NUCLEAR_REF_DATE" \
        --iaea_csv "$IAEA_CSV"

run_script "solar_cf_counterfactual" \
    python "$SCRIPTS/solar_cf_counterfactual.py" "$NETWORK" \
        --year "$YEAR" --ember_twh "$SOLAR_EMBER" --irena_gw "$SOLAR_IRENA"

run_script "solar_spatial_analysis" \
    env PYPSA_OUTPUT_DIR="$BASE_OUT/solar_spatial" \
    python "$SCRIPTS/solar_spatial_analysis.py" "$NETWORK"

run_script "solar_spatial_analysis_amnin2" \
    env PYPSA_OUTPUT_DIR="$BASE_OUT/solar_spatial_amnin2" \
    python "$SCRIPTS/solar_spatial_analysis_amnin2.py" "$NETWORK"

run_script "thermal_maps" \
    python "$SCRIPTS/thermal_maps.py" "$NETWORK" \
        --year "$YEAR" --shapes "$SHAPES"

run_script "plot_network" \
    env PYPSA_OUTPUT_DIR="$BASE_OUT/plot_network" \
    python "$SCRIPTS/plot_network.py" "$NETWORK"

run_script "plot_dc_links" \
    env PYPSA_OUTPUT_DIR="$BASE_OUT/plot_dc_links" \
    python "$SCRIPTS/plot_dc_links.py" "$NETWORK"

run_script "plot_china_regions" \
    python "$SCRIPTS/plot_china_regions.py"

run_script "wind_curtailment" \
    python "$SCRIPTS/wind_curtailment.py" "$NETWORK"

echo "" | tee -a "$LOG"
echo "=== COMPLETATO $(date) ===" | tee -a "$LOG"
echo "Output in: $BASE_OUT" | tee -a "$LOG"
echo "Log completo: $LOG"
OK=$(grep -c "^OK:" "$LOG" || true)
FAIL=$(grep -c "^FAILED:" "$LOG" || true)
echo "Risultati: $OK OK, $FAIL FAILED"