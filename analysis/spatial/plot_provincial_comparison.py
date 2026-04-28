"""
plot_provincial_comparison.py
=============================
Confronto provinciale fix-hydro vs geospatial_v2 vs EFC 2020.

Plot 1: totali aggregati (bar chart con 3 barre)
Plot 2: confronto per provincia (grouped bar chart con errore % in cima)

Output: analysis/spatial/output/comparison/
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── DATI ──────────────────────────────────────────────────────────────────────

data = [
    ("Shandong",       578.1, 491.3, 549.3),
    ("Inner Mongolia", 570.3, 171.6, 296.5),
    ("Jiangsu",        507.4,  48.2, 498.8),
    ("Guangdong",      504.8, 207.0, 533.5),
    ("Sichuan",        416.7, 640.7, 211.9),
    ("Xinjiang",       405.2, 129.8, 245.3),
    ("Yunnan",         367.4, 228.0, 146.0),
    ("Zhejiang",       352.1, 311.8, 328.6),
    ("Shanxi",         339.5, 229.1, 187.8),
    ("Hubei",          303.7, 209.3, 180.7),
    ("Hebei",          294.5, 312.7, 321.0),
    ("Henan",          279.1, 272.7, 288.3),
    ("Anhui",          278.5, 244.0, 208.0),
    ("Fujian",         263.6, 261.9, 192.2),
    ("Shaanxi",        242.6, 185.0, 155.2),
    ("Guizhou",        232.7, 165.5, 124.2),
    ("Liaoning",       203.9, 314.8, 201.5),
    ("Guangxi",        193.9, 252.5, 154.1),
    ("Gansu",          178.7, 154.8, 103.3),
    ("Ningxia",        176.8,  16.0,  87.7),
    ("Hunan",          155.2, 393.0, 148.7),
    ("Jiangxi",        147.7, 212.4, 156.0),
    ("Heilongjiang",   111.1, 251.2,  80.0),
    ("Jilin",           99.0, 200.0,  61.6),
    ("Qinghai",         94.8,  37.1,  57.9),
    ("Shanghai",        86.4, 240.2, 123.4),
    ("Chongqing",       83.7, 235.1,  93.6),
    ("Hainan",          34.8,  31.9,  28.7),
    ("Tibet",            8.7,  15.1,   6.9),
]

provinces   = [d[0] for d in data]
efc         = np.array([d[1] for d in data])
fix_hydro   = np.array([d[2] for d in data])
geospatial  = np.array([d[3] for d in data])

err_fh  = (fix_hydro  - efc) / efc * 100
err_geo = (geospatial - efc) / efc * 100

OUTPUT_DIR = "analysis/spatial/output/comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── COLORI ────────────────────────────────────────────────────────────────────
C_EFC  = "#d7191c"
C_FH   = "#60ace6"
C_GEO  = "#1a9641"

# ── PLOT 1: TOTALI ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

runs   = ["EFC 2020", "fix-hydro", "geospatial_v2"]
totals = [efc.sum(), fix_hydro.sum(), geospatial.sum()]
colors = [C_EFC, C_FH, C_GEO]
bars   = ax.bar(runs, totals, color=colors, width=0.5, alpha=0.88, edgecolor="white")

for bar, val, run in zip(bars, totals, runs):
    if run != "EFC 2020":
        err = (val - efc.sum()) / efc.sum() * 100
        label = f"{val:.0f} TWh\n({err:+.1f}%)"
    else:
        label = f"{val:.0f} TWh"
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 40,
            label, ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_ylabel("TWh", fontsize=12)
ax.set_title("Total electricity load — CN2020", fontsize=13, fontweight="bold")
ax.set_ylim(0, max(totals) * 1.18)
ax.grid(True, axis="y", alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "total_comparison.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# ── PLOT 2: CONFRONTO PER PROVINCIA ──────────────────────────────────────────
n      = len(provinces)
x      = np.arange(n)
width  = 0.26

fig, ax = plt.subplots(figsize=(22, 8))

b_efc = ax.bar(x - width,     efc,       width, label="EFC 2020",      color=C_EFC,  alpha=0.88, edgecolor="white")
b_fh  = ax.bar(x,             fix_hydro, width, label="fix-hydro",     color=C_FH,   alpha=0.88, edgecolor="white")
b_geo = ax.bar(x + width,     geospatial,width, label="geospatial_v2", color=C_GEO,  alpha=0.88, edgecolor="white")

# Errore % in cima alle barre del modello
for i, (val_fh, val_geo, e_fh, e_geo) in enumerate(zip(fix_hydro, geospatial, err_fh, err_geo)):
    # fix-hydro
    color_fh = "#8b0000" if e_fh > 0 else "#00008b"
    ax.text(x[i],
            val_fh + 4,
            f"{e_fh:+.0f}%",
            ha="center", va="bottom",
            fontsize=6.2, color=color_fh, fontweight="bold", rotation=90)
    # geospatial_v2
    color_geo = "#8b0000" if e_geo > 0 else "#00008b"
    ax.text(x[i] + width,
            val_geo + 4,
            f"{e_geo:+.0f}%",
            ha="center", va="bottom",
            fontsize=6.2, color=color_geo, fontweight="bold", rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(provinces, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("TWh", fontsize=11)
ax.set_title(
    "Provincial electricity load: fix-hydro vs geospatial_v2 vs EFC 2020\n"
    "Error % on top of model bars  (dark red = overestimate, dark blue = underestimate)",
    fontsize=12, fontweight="bold"
)
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, axis="y", alpha=0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Aggiungi totali in legenda
total_note = (
    f"Totals:  EFC={efc.sum():.0f} TWh  |  "
    f"fix-hydro={fix_hydro.sum():.0f} TWh ({(fix_hydro.sum()-efc.sum())/efc.sum()*100:+.1f}%)  |  "
    f"geospatial_v2={geospatial.sum():.0f} TWh ({(geospatial.sum()-efc.sum())/efc.sum()*100:+.1f}%)"
)
fig.text(0.5, -0.01, total_note, ha="center", fontsize=9, color="gray")

plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "provincial_comparison.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

print("Done.")