"""
provincial_error_metrics.py
============================
Calcola e confronta le metriche di errore provinciale per due configurazioni
del modello PyPSA-Earth CN2020:
  - fix-hydro (distribuzione uniforme basata su popolazione/GDP)
  - geospatial_v2 OLD (scale: 0.7828, run con Asia.csv provinciale)
  - geospatial_v2 NEW (scale: 1.0, run con Asia.csv provinciale)

Metriche calcolate:
  - MAE  (Mean Absolute Error, TWh)
  - RMSE (Root Mean Square Error, TWh)
  - MAPE (Mean Absolute Percentage Error, %)
  - Bias (errore medio con segno, TWh) — indica se il modello sovra/sottostima

Output: analysis/spatial/output/metrics/
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ── DATI ──────────────────────────────────────────────────────────────────────
# (provincia, EFC_2020, fix_hydro, geo_old_scale0.7828, geo_new_scale1.0)

data = [
    ("Shandong",       578.1, 491.3, 549.3, 701.7),
    ("Inner Mongolia", 570.3, 171.6, 296.5, 378.8),
    ("Jiangsu",        507.4,  48.2, 498.8, 637.2),
    ("Guangdong",      504.8, 207.0, 533.5, 681.5),
    ("Sichuan",        416.7, 640.7, 211.9, 270.7),
    ("Xinjiang",       405.2, 129.8, 245.3, 313.4),
    ("Yunnan",         367.4, 228.0, 146.0, 186.5),
    ("Zhejiang",       352.1, 311.8, 328.6, 419.8),
    ("Shanxi",         339.5, 229.1, 187.8, 239.9),
    ("Hubei",          303.7, 209.3, 180.7, 230.8),
    ("Hebei",          294.5, 312.7, 321.0, 410.1),
    ("Henan",          279.1, 272.7, 288.3, 368.2),
    ("Anhui",          278.5, 244.0, 208.0, 265.7),
    ("Fujian",         263.6, 261.9, 192.2, 245.5),
    ("Shaanxi",        242.6, 185.0, 155.2, 198.3),
    ("Guizhou",        232.7, 165.5, 124.2, 158.7),
    ("Liaoning",       203.9, 314.8, 201.5, 257.5),
    ("Guangxi",        193.9, 252.5, 154.1, 196.8),
    ("Gansu",          178.7, 154.8, 103.3, 132.0),
    ("Ningxia",        176.8,  16.0,  87.7, 112.0),
    ("Hunan",          155.2, 393.0, 148.7, 190.0),
    ("Jiangxi",        147.7, 212.4, 156.0, 199.3),
    ("Heilongjiang",   111.1, 251.2,  80.0, 102.2),
    ("Jilin",           99.0, 200.0,  61.6,  78.7),
    ("Qinghai",         94.8,  37.1,  57.9,  74.0),
    ("Shanghai",        86.4, 240.2, 123.4, 157.7),
    ("Chongqing",       83.7, 235.1,  93.6, 119.6),
    ("Hainan",          34.8,  31.9,  28.7,  36.6),
    ("Tibet",            8.7,  15.1,   6.9,   8.8),
]

provinces  = [d[0] for d in data]
efc        = np.array([d[1] for d in data])
fix_hydro  = np.array([d[2] for d in data])
geo_old    = np.array([d[3] for d in data])
geo_new    = np.array([d[4] for d in data])

OUTPUT_DIR = "analysis/spatial/output/metrics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── FUNZIONI METRICHE ─────────────────────────────────────────────────────────

def mae(pred, ref):
    return np.mean(np.abs(pred - ref))

def rmse(pred, ref):
    return np.sqrt(np.mean((pred - ref) ** 2))

def mape(pred, ref):
    return np.mean(np.abs((pred - ref) / ref)) * 100

def bias(pred, ref):
    return np.mean(pred - ref)

# ── CALCOLO METRICHE ──────────────────────────────────────────────────────────

configs = {
    "fix-hydro":      fix_hydro,
    "geo_old (0.7828)": geo_old,
    "geo_new (1.0)":  geo_new,
}

results = {}
for name, pred in configs.items():
    results[name] = {
        "MAE (TWh)":  mae(pred, efc),
        "RMSE (TWh)": rmse(pred, efc),
        "MAPE (%)":   mape(pred, efc),
        "Bias (TWh)": bias(pred, efc),
        "Total (TWh)": pred.sum(),
        "Total error (%)": (pred.sum() - efc.sum()) / efc.sum() * 100,
    }

# Stampa tabella riepilogativa
print("\n" + "=" * 70)
print("METRICHE DI ERRORE PROVINCIALE — CN2020")
print(f"Riferimento: EFC 2020, {len(provinces)} province, totale = {efc.sum():.1f} TWh")
print("=" * 70)
header = f"{'Metrica':<20} {'fix-hydro':>18} {'geo_old (0.7828)':>18} {'geo_new (1.0)':>18}"
print(header)
print("-" * 70)
for metric in ["MAE (TWh)", "RMSE (TWh)", "MAPE (%)", "Bias (TWh)", "Total (TWh)", "Total error (%)"]:
    row = f"{metric:<20}"
    for name in configs:
        val = results[name][metric]
        row += f" {val:>18.1f}"
    print(row)
print("=" * 70)

# Salva CSV
df_metrics = pd.DataFrame(results).T
df_metrics.to_csv(os.path.join(OUTPUT_DIR, "error_metrics_summary.csv"))
print(f"\nSaved: {os.path.join(OUTPUT_DIR, 'error_metrics_summary.csv')}")

# ── PLOT 1: Barre metriche aggregate ─────────────────────────────────────────
metrics_to_plot = ["MAE (TWh)", "RMSE (TWh)", "MAPE (%)"]
metric_labels   = ["MAE (TWh)", "RMSE (TWh)", "MAPE (%)"]
colors = ["#60ace6", "#ff7f0e", "#1a9641"]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for ax, metric, label in zip(axes, metrics_to_plot, metric_labels):
    vals = [results[n][metric] for n in configs]
    bars = ax.bar(list(configs.keys()), vals, color=colors, alpha=0.88,
                  edgecolor="white", width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.02,
                f"{val:.1f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_xticklabels(list(configs.keys()), rotation=15, ha="right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("Provincial error metrics — CN2020\n(lower is better)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "error_metrics_bar.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# ── PLOT 2: Errore % per provincia — confronto tre configurazioni ──────────────
err_fh  = (fix_hydro - efc) / efc * 100
err_old = (geo_old   - efc) / efc * 100
err_new = (geo_new   - efc) / efc * 100

# Ordina per EFC decrescente
idx = np.argsort(efc)[::-1]
provinces_sorted = [provinces[i] for i in idx]
err_fh_s  = err_fh[idx]
err_old_s = err_old[idx]
err_new_s = err_new[idx]

x = np.arange(len(provinces_sorted))
width = 0.26

fig, ax = plt.subplots(figsize=(22, 7))
ax.bar(x - width, err_fh_s,  width, label="fix-hydro",        color="#60ace6", alpha=0.85, edgecolor="white")
ax.bar(x,         err_old_s, width, label="geo_old (0.7828)", color="#ff7f0e", alpha=0.85, edgecolor="white")
ax.bar(x + width, err_new_s, width, label="geo_new (1.0)",    color="#1a9641", alpha=0.85, edgecolor="white")

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(provinces_sorted, rotation=45, ha="right", fontsize=8.5)
ax.set_ylabel("Error % vs EFC 2020", fontsize=11)
ax.set_title(
    "Provincial load error % — fix-hydro vs geo_old vs geo_new\n"
    f"MAE: fix-hydro={results['fix-hydro']['MAE (TWh)']:.0f} TWh  |  "
    f"geo_old={results['geo_old (0.7828)']['MAE (TWh)']:.0f} TWh  |  "
    f"geo_new={results['geo_new (1.0)']['MAE (TWh)']:.0f} TWh",
    fontsize=12, fontweight="bold"
)
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, axis="y", alpha=0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "error_pct_by_province.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# ── PLOT 3: Scatter errore OLD vs NEW per provincia ───────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))
sc = ax.scatter(err_old, err_new,
                c=efc, cmap="YlOrRd", s=80, edgecolors="black", linewidth=0.5, zorder=3)
plt.colorbar(sc, ax=ax, label="EFC 2020 (TWh)")

ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax.plot([-120, 60], [-120, 60], color="gray", linewidth=0.8,
        linestyle=":", alpha=0.7, label="y = x (no change)")

for i, prov in enumerate(provinces):
    ax.annotate(prov, (err_old[i], err_new[i]),
                fontsize=6.5, ha="left", va="bottom",
                xytext=(3, 2), textcoords="offset points")

ax.set_xlabel("Error % geo_old (scale 0.7828)", fontsize=11)
ax.set_ylabel("Error % geo_new (scale 1.0)", fontsize=11)
ax.set_title(
    "Provincial error: geo_old vs geo_new\n"
    "Points above y=x line: NEW is worse  |  Points below: NEW is better",
    fontsize=11, fontweight="bold"
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "scatter_old_vs_new.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

print("\nDone.")
