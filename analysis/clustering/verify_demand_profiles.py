"""
verify_demand_profiles.py
=========================
Verifica empirica che i profili di domanda dei bus siano proporzionali tra loro,
ovvero che la correlazione tra tutti i bus sia pari a 1.
Se il modello usa un unico profilo nazionale scalato per fattori statici,
allora tutti i profili dei bus devono essere perfettamente correlati (r = 1).

Output (salvato in analysis/clustering/output/):
  - demand_profiles_correlation.png : heatmap della matrice di correlazione tra bus
  - demand_profiles_normalized.png  : profili normalizzati sovrapposti (3 bus campione)

Uso: python analysis/clustering/verify_demand_profiles.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ── PARAMETERS ────────────────────────────────────────────────────────────────
DEMAND_FILE = "resources/demand_profiles.csv"
OUTPUT_DIR  = "analysis/clustering/output"

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DEMAND_FILE, index_col=0, parse_dates=True)
    print(f"Loaded: {df.shape[0]} timesteps x {df.shape[1]} buses")

    # ── Correlation matrix ────────────────────────────────────────────────────
    corr = df.corr()
    print(f"\nCorrelation matrix stats:")
    print(f"  Min correlation : {corr.values[np.triu_indices_from(corr.values, k=1)].min():.6f}")
    print(f"  Max correlation : {corr.values[np.triu_indices_from(corr.values, k=1)].max():.6f}")
    print(f"  Mean correlation: {corr.values[np.triu_indices_from(corr.values, k=1)].mean():.6f}")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, vmin=0.99, vmax=1.0, cmap="Blues")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    plt.colorbar(im, ax=ax, label="Pearson correlation")
    ax.set_title("Correlation matrix of bus demand profiles")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "demand_profiles_correlation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(); print(f"\nSaved: {out}")

    # ── Normalized profiles (3 sample buses) ─────────────────────────────────
    sample_buses = df.columns[:3].tolist()
    fig, ax = plt.subplots(figsize=(14, 4))
    for bus in sample_buses:
        ts = df[bus]
        ts_norm = ts / ts.max()
        ax.plot(df.index, ts_norm, label=f"bus {bus}", linewidth=0.7, alpha=0.85)
    ax.set_ylabel("Normalized demand (0-1)")
    ax.set_title("Normalized demand profiles -- 3 sample buses (should overlap perfectly)")
    ax.legend(); ax.grid(True, alpha=0.2)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "demand_profiles_normalized.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(); print(f"Saved: {out}")

    print("\nDone.")
