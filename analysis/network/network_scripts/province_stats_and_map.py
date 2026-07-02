import pypsa
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely import Point
import os

# Da confermare con Ivan: questo è il run giusto o serve cambiarlo?
NETWORK_PATH = "results/CN2060_loadCF/networks/elec_s_250_ec_lcopt_NUCAP-BIOCAP-Co2L0.05-3h.nc"
REGIONS_PATH = "resources/CN2060_loadCF/bus_regions/regions_onshore_elec_s_250.geojson"
ADMIN_PATH = "/p/tmp/ivanra/PyPSA-China-PIK/resources/data/regions/admin2_shapes.geojson"
OUTPUT_DIR = "analysis/network/output/CN2060_NUCAP_BIO_95/wind_spatial_analysis_bus"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading network: {NETWORK_PATH}")
n = pypsa.Network(NETWORK_PATH)

print(f"Loading regions: {REGIONS_PATH}")
regions = gpd.read_file(REGIONS_PATH)

print(f"Loading admin boundaries: {ADMIN_PATH}")
admin = gpd.read_file(ADMIN_PATH)

# ── Point-in-polygon: assegna ogni bus alla sua provincia ────────────────────
n.buses["point"] = n.buses.apply(lambda row: Point(row.x, row.y), axis=1)

# Fix manuale per bus in mare — verificare se serve anche per la rete a 250 nodi
# Ivan lo applica su CN.6.18_1_AC per la rete 425; potrebbe essere un bus diverso
# o non esistere affatto qui — verificare prima di applicare ciecamente.
if "CN.6.18_1_AC" in n.buses.index:
    n.buses.loc["CN.6.18_1_AC", "point"] = Point(109.870802, 20.718244)
    print("Applied sea-point fix for CN.6.18_1_AC")
else:
    print("[INFO] CN.6.18_1_AC non presente in questa rete — fix non applicato")

admin1 = admin.dissolve("NAME_1").reset_index()

print("Assigning province via point-in-polygon...")
n.buses["province"] = n.buses["point"].apply(
    lambda pt: admin1[admin1.geometry.contains(pt)]["NAME_1"].values[0]
    if len(admin1[admin1.geometry.contains(pt)]) > 0 else "unmatched"
).astype(str)

unmatched = (n.buses["province"] == "unmatched").sum()
print(f"Bus non assegnati a nessuna provincia: {unmatched} / {len(n.buses)}")
if unmatched > 0:
    print(f"Esempi: {n.buses[n.buses['province']=='unmatched'].index.tolist()[:5]}")

# ── Propaga "province" ai componenti via bus, per usarla nel groupby ─────────
for component in ["generators", "storage_units", "links"]:
    df = getattr(n, component)
    if not df.empty and "bus" in df.columns:
        df["province"] = df["bus"].map(n.buses["province"])
    elif not df.empty and "bus0" in df.columns:
        # Links hanno bus0/bus1 — usiamo bus0 come riferimento principale
        df["province"] = df["bus0"].map(n.buses["province"])

# ── Statistics groupby — verifica diretta della funzionalità nativa PyPSA ────
print("\n=== n.statistics.capacity_factor(groupby=['carrier','province']) ===")
try:
    cf_stats = n.statistics.capacity_factor(groupby=["carrier", "province"])
    print(cf_stats)
    cf_stats.to_csv(os.path.join(OUTPUT_DIR, "capacity_factor_by_province.csv"))
    print(f"\nSaved: {OUTPUT_DIR}/capacity_factor_by_province.csv")
except Exception as e:
    print(f"[ERROR] n.statistics.capacity_factor con groupby province ha fallito: {e}")
    print("Verificare la versione PyPSA e la sintassi groupby supportata.")

# ── Plot con confini sovrapposti, come richiesto da Ivan ──────────────────────
fig, ax = plt.subplots(figsize=(12, 12))
regions.plot(facecolor="none", edgecolor="darkgray", ax=ax, linewidth=0.5)
admin.dissolve("NAME_1").boundary.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.0)
ax.set_title("Model regions (gray) + provincial boundaries (black) — CN2060", fontsize=12)
out = os.path.join(OUTPUT_DIR, "map_boundaries_check.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
