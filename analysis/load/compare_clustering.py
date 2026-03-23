"""
compare_clustering.py
=====================
Confronto dei profili temporali del carico elettrico tra due metodi di clustering
per lo scenario CN2020:
  - CN2020 k-means (250 cluster)
  - CN2020 Admin1  (GADM level 1, ~31 province)

Produce quattro grafici:
  - load_hourly_profiles_clustering.png            : profilo orario in GW
  - load_daily_profiles_clustering.png             : profilo giornaliero in GWh/giorno
  - load_monthly_profiles_clustering.png           : somma mensile in TWh/mese
  - load_monthly_profiles_normalized_clustering.png: profilo mensile normalizzato

e una tabella testuale con i valori mensili per ogni scenario.

Uso: python analysis/clustering/compare_clustering.py
"""

import pypsa
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ── PARAMETERS ────────────────────────────────────────────────────────────────

SCENARIOS = {
    "CN2020_kmeans": {
        "file":  "results/networks/CN2020/elec_s_250_ec_lcopt_Co2L-3h.nc",
        "color": "#1f77b4",
    },
    "CN2020_Admin1": {
        "file":  "results/networks/CN2020_Admin1/elec_s_250_ec_lcopt_Co2L-6h.nc",
        "color": "#d62728",
    },
}

OUTPUT_DIR   = "analysis/load/output"
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ─────────────────────────────────────────────────────────────────────────────


def load_profile(network_file):
    """
    Load a PyPSA network and return the national load time series,
    daily resample and monthly resample.
    Returns:
      load_ts      : pd.Series, national total load (MW), original temporal resolution
      load_daily   : pd.Series, daily resampled load (GWh/day)
      load_monthly : pd.Series, monthly resampled load (TWh/month)
    """
    n = pypsa.Network(network_file)
    w = n.snapshot_weightings.generators
    load_ts = n.loads_t.p_set.sum(axis=1)                        # MW
    load_daily = (load_ts * w).resample("D").sum() / 1e3         # GWh/day
    load_monthly = (load_ts * w).resample("ME").sum() / 1e6      # TWh/month
    return load_ts, load_daily, load_monthly


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    profiles = {}
    for name, meta in SCENARIOS.items():
        print(f"Loading {name}...")
        load_ts, load_daily, load_monthly = load_profile(meta["file"])
        profiles[name] = {
            "load_ts": load_ts,
            "load_daily": load_daily,
            "load_monthly": load_monthly,
        }

    # Plot 1: time series at original resolution (GW)
    fig, ax = plt.subplots(figsize=(14, 5))
    for name, meta in SCENARIOS.items():
        ts = profiles[name]["load_ts"] / 1e3   # MW -> GW
        ax.plot(ts.index, ts.values, label=name, color=meta["color"],
                linewidth=0.6, alpha=0.8)
    ax.set_ylabel("GW")
    ax.set_title("National electricity load -- k-means vs Admin1 (GW)")
    ax.legend(); ax.grid(True, alpha=0.2)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "load_hourly_profiles_clustering.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(); print(f"Saved: {out}")

    # Plot 2: daily sum (GWh/day)
    fig, ax = plt.subplots(figsize=(14, 5))
    for name, meta in SCENARIOS.items():
        dl = profiles[name]["load_daily"]
        ax.plot(dl.index, dl.values, label=name, color=meta["color"],
                linewidth=1.0, alpha=0.85)
    ax.set_ylabel("GWh / day")
    ax.set_title("National electricity load -- daily profile k-means vs Admin1")
    ax.legend(); ax.grid(True, alpha=0.2)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "load_daily_profiles_clustering.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(); print(f"Saved: {out}")

    # Plot 3: monthly sum (TWh/month)
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, meta in SCENARIOS.items():
        ml = profiles[name]["load_monthly"]
        ax.plot(range(1, len(ml) + 1), ml.values, label=name, color=meta["color"],
                linewidth=2, marker="o", markersize=5)
    ax.set_xlabel("Month"); ax.set_ylabel("TWh / month")
    ax.set_title("National electricity load -- monthly sum k-means vs Admin1")
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(MONTH_LABELS)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "load_monthly_profiles_clustering.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(); print(f"Saved: {out}")

    # Plot 4: normalized monthly profile (fraction of annual total)
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, meta in SCENARIOS.items():
        ml = profiles[name]["load_monthly"]
        ml_norm = ml / ml.sum()
        ax.plot(range(1, len(ml_norm) + 1), ml_norm.values, label=name,
                color=meta["color"], linewidth=2, marker="o", markersize=5)
    ax.set_xlabel("Month"); ax.set_ylabel("Fraction of annual load")
    ax.set_title("National electricity load -- normalized monthly profile k-means vs Admin1")
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(MONTH_LABELS)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "load_monthly_profiles_normalized_clustering.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(); print(f"Saved: {out}")

    # Monthly table
    print()
    print(f"{'Month':<6} {'CN2020_kmeans (TWh)':>20} {'CN2020_Admin1 (TWh)':>20}")
    print("-" * 48)
    for i, month in enumerate(MONTH_LABELS):
        row = f"{month:<6}"
        for name in SCENARIOS:
            row += f" {profiles[name]['load_monthly'].iloc[i]:>20.1f}"
        print(row)
    print("-" * 48)
    print(f"{'TOTAL':<6}" + "".join(
        f" {profiles[n]['load_monthly'].sum():>20.1f}" for n in SCENARIOS
    ))

    print("\nDone.")
