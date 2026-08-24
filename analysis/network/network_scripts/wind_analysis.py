#!/usr/bin/env python3
"""
wind_analysis.py

Mappa spaziale dei generatori onwind con capacity factor medio = 0.0,
per una rete risolta di qualunque scenario.

Convenzione di invocazione (coerente con gli altri script del repo):

  cd /p/tmp/lorenzop/pypsa-earth-ivan && \
  PYPSA_OUTPUT_DIR=analysis/network/output/<label_scenario> \
  python3 analysis/network/network_scripts/wind_analysis.py \
      <path/alla/rete.nc> \
      --run-tag <etichetta_run>

PYPSA_OUTPUT_DIR e' obbligatoria. Lo script si ferma se non e' impostata,
non assume un default silenzioso.

Nessuno scenario, anno o path e' hardcoded nel codice.

IPOTESI APERTE (non verificate in questa sessione, da verificare alla
prima esecuzione reale):

  1. regions_onshore.geojson non viene passato esplicitamente nella
     convenzione mostrata da Lorenzo. Lo script lo cerca derivando la
     cartella "resources/<run>/" da "results/<run>/" nel path della
     rete, poi prova due pattern: regions_onshore_elec_s<N>.geojson
     (N = numero cluster estratto dal nome della rete) e
     regions_onshore.geojson generico. Ricostruzione plausibile della
     convenzione Snakemake di PyPSA-Earth, MAI VERIFICATA IN QUESTA
     CONVERSAZIONE contro la struttura reale delle cartelle. Se fallisce
     o trova il file sbagliato, usare --regions-path esplicito.
  2. La colonna di join tra poligoni e bus (shape_id vs name) resta
     un'ipotesi gia' segnalata nella versione precedente: verificata
     solo per profile_onwind.nc nell'indagine sul bug vento Nordest,
     non per reti risolte generiche. Lo script stampa sempre gli
     unmatched, da controllare prima di interpretare la figura.
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pypsa


def find_regions_path(network_path: Path):
    """Deriva resources/<run>/ da results/<run>/networks/file.nc e cerca
    regions_onshore*.geojson. Ricostruzione plausibile, non verificata."""
    parts = list(network_path.resolve().parts)
    if "results" not in parts:
        return None, []
    idx = parts.index("results")
    if idx + 1 >= len(parts):
        return None, []
    run_name = parts[idx + 1]
    base_parts = parts[:idx]
    resources_root = Path(*base_parts, "resources", run_name)

    cluster_match = re.search(r"elec_s_(\d+)", network_path.name)
    patterns_tried = []
    candidates = []

    if cluster_match:
        n_clusters = cluster_match.group(1)
        pattern = str(resources_root / "**" / f"regions_onshore_elec_s{n_clusters}*.geojson")
        patterns_tried.append(pattern)
        candidates += glob.glob(pattern, recursive=True)

    pattern_generic = str(resources_root / "**" / "regions_onshore*.geojson")
    patterns_tried.append(pattern_generic)
    candidates += glob.glob(pattern_generic, recursive=True)

    candidates = sorted(set(candidates))
    return (candidates[0] if candidates else None), patterns_tried


def main():
    parser = argparse.ArgumentParser(
        description="Mappa bus onwind con CF=0.0 per una rete risolta."
    )
    parser.add_argument("network_path", help="Path alla rete .nc risolta.")
    parser.add_argument(
        "--run-tag", required=True,
        help="Etichetta del run, usata nei nomi dei file di output e nel titolo."
    )
    parser.add_argument(
        "--regions-path", default=None,
        help="Path esplicito a regions_onshore.geojson. Se omesso, lo script "
             "tenta una derivazione automatica (vedi docstring, non verificata)."
    )
    parser.add_argument(
        "--cf-threshold", type=float, default=1e-9,
        help="Soglia sotto la quale il CF medio e' considerato zero (default 1e-9)."
    )
    args = parser.parse_args()

    output_dir = os.environ.get("PYPSA_OUTPUT_DIR")
    if not output_dir:
        sys.exit(
            "ERRORE: PYPSA_OUTPUT_DIR non impostata. Lo script non assume "
            "un default: impostarla esplicitamente prima di lanciare, come "
            "negli altri script del repo."
        )
    os.makedirs(output_dir, exist_ok=True)

    network_path = Path(args.network_path)
    if not network_path.exists():
        sys.exit(f"ERRORE: rete non trovata: {network_path}")

    print(f"[{args.run_tag}] rete: {network_path}")

    if args.regions_path:
        regions_path = args.regions_path
    else:
        regions_path, patterns_tried = find_regions_path(network_path)
        if regions_path is None:
            sys.exit(
                "ERRORE: regions_onshore.geojson non trovato automaticamente.\n"
                "Pattern provati:\n  " + "\n  ".join(patterns_tried) + "\n"
                "Passare --regions-path esplicito."
            )
        print(f"[{args.run_tag}] regions_onshore derivato automaticamente "
              f"(ipotesi non verificata): {regions_path}")

    n = pypsa.Network(str(network_path))

    onwind = n.generators[n.generators.carrier == "onwind"]
    if onwind.empty:
        sys.exit(
            f"ERRORE: nessun generatore con carrier 'onwind' in {network_path}."
        )

    n_total = len(onwind)
    print(f"[{args.run_tag}] generatori onwind: {n_total}")

    p_max_pu = n.generators_t.p_max_pu[onwind.index]
    mean_cf = p_max_pu.mean(axis=0)
    zero_mask = mean_cf <= args.cf_threshold
    n_zero = int(zero_mask.sum())

    print(
        f"[{args.run_tag}] generatori con CF medio <= {args.cf_threshold}: "
        f"{n_zero}/{n_total} ({100 * n_zero / n_total:.2f}%)"
    )

    cf_table = pd.DataFrame({
        "bus": onwind.loc[mean_cf.index, "bus"],
        "mean_cf": mean_cf,
        "is_zero": zero_mask,
    })
    cf_table_path = os.path.join(output_dir, f"{args.run_tag}_onwind_cf_table.csv")
    cf_table.to_csv(cf_table_path)
    print(f"[{args.run_tag}] tabella CF: {cf_table_path}")

    regions = gpd.read_file(regions_path)

    id_col = None
    for candidate in ["shape_id", "name", "bus", "id"]:
        if candidate in regions.columns:
            id_col = candidate
            break
    if id_col is None:
        sys.exit(
            f"ERRORE: nessuna colonna ID riconoscibile in {regions_path}. "
            f"Colonne presenti: {list(regions.columns)}."
        )

    print(f"[{args.run_tag}] colonna ID join: '{id_col}' (non verificata per questo run)")

    if id_col == "shape_id":
        regions["_join_key"] = regions[id_col].astype(str) + "_AC"
    else:
        regions["_join_key"] = regions[id_col].astype(str)

    regions = regions.merge(cf_table, left_on="_join_key", right_on="bus", how="left")
    unmatched = int(regions["mean_cf"].isna().sum())
    if unmatched > 0:
        print(
            f"ATTENZIONE: {unmatched}/{len(regions)} poligoni senza corrispondenza. "
            f"'{id_col}' probabilmente sbagliata per questo run: non interpretare "
            f"la figura finche' questo numero non e' zero o spiegato."
        )

    fig, ax = plt.subplots(figsize=(10, 10))
    regions.plot(ax=ax, color="lightgrey", edgecolor="grey", linewidth=0.3)

    nonzero_regions = regions[regions["is_zero"] == False]
    zero_regions = regions[regions["is_zero"] == True]

    from matplotlib.patches import Patch

    nonzero_regions.plot(ax=ax, color="#2c7fb8", edgecolor="grey", linewidth=0.3)
    zero_regions.plot(ax=ax, color="#d7191c", edgecolor="black", linewidth=0.5)

    legend_elements = [
        Patch(facecolor="#2c7fb8", edgecolor="grey",
              label=f"CF > {args.cf_threshold} (n={len(nonzero_regions)})"),
        Patch(facecolor="#d7191c", edgecolor="black",
              label=f"CF <= {args.cf_threshold} (n={len(zero_regions)})"),
    ]

    title = (f"{args.run_tag}: onwind mean CF, zero-CF buses highlighted\n"
             f"{n_zero}/{n_total} buses at CF=0.0 ({100 * n_zero / n_total:.1f}%)")
    if unmatched > 0:
        title += f"\n[{unmatched} unmatched polygons, verify id_col='{id_col}']"
    ax.set_title(title)
    ax.legend(handles=legend_elements, loc="lower left")
    ax.set_axis_off()

    fig_path = os.path.join(output_dir, f"{args.run_tag}_onwind_cf_zero_map.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"[{args.run_tag}] figura: {fig_path}")

    print("\nRiepilogo (verificare prima di usare con Ivan o Davide):")
    print(f"  run_tag: {args.run_tag}")
    print(f"  rete: {network_path}")
    print(f"  regions: {regions_path}")
    print(f"  colonna id: {id_col} (non verificata per questo run)")
    print(f"  poligoni non abbinati: {unmatched}")
    print(f"  onwind totali: {n_total}")
    print(f"  CF=0: {n_zero} ({100 * n_zero / n_total:.2f}%)")


if __name__ == "__main__":
    main()