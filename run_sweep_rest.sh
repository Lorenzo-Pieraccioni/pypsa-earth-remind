#!/bin/bash
set -e
CAPEX_LIST=(6000 8000 10000)
TARGET="results/CN2060_admin1_test/networks/elec_s_250_ec_lcopt_BIOCAP-Co2L0.05-3h.nc"
ANN=0.08900913887

for capex in "${CAPEX_LIST[@]}"; do
  echo "===== CAPEX $capex EUR/kW — $(date) ====="
  sed -e "s/name: CN2060_nuc${capex}/name: CN2060_admin1_test/" \
      "config.CN2060_nuc${capex}.yaml" > config.yaml
  echo "override:"; grep -A1 "investment:" config.yaml | grep nuclear

  snakemake --profile _pik_hpc_profile --rerun-triggers mtime -j 20 "$TARGET"

  cp "$TARGET" "results/sweep/nuc_${capex}.nc"

  python3 - "$capex" "$ANN" << 'PYEOF'
import sys, pypsa
capex, ann = int(sys.argv[1]), float(sys.argv[2])
n = pypsa.Network(f"results/sweep/nuc_{capex}.nc")
nuc = n.generators[n.generators.carrier=="nuclear"]
cc = nuc.capital_cost.iloc[0]; gw = nuc.p_nom_opt.sum()/1e3
exp = capex*1000*ann
line = f"{capex} EUR/kW | capital_cost={cc:.0f} (exp {exp:.0f}) | nuclear={gw:.1f} GW | p_min={nuc.p_min_pu.iloc[0]} p_max={nuc.p_max_pu.iloc[0]} | cap_off={'OK' if abs(gw-300)>0.1 else 'FAIL'}"
print(line)
open("results/sweep/sweep_log.txt","a").write(line+"\n")
PYEOF
  echo "done $capex — $(date)"
done
echo "===== SWEEP COMPLETE ====="
cat results/sweep/sweep_log.txt
