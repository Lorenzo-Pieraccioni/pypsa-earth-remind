# """
# provincial_load_analysis.py
# ===========================
# Analisi della distribuzione provinciale del carico elettrico per lo scenario CN2020.
# Confronta il carico modellato (PyPSA-Earth) con i dati reali EFC 2020 a livello provinciale.

# Il carico per bus viene aggregato a livello provinciale tramite spatial join con i confini
# GADM. I bus non mappati su alcuna provincia vengono ignorati.

# Note sulle unità EFC:
#   - Carico (sheet A-1-1, lang == "Load"): unità 100 GWh -> divide by 10 -> TWh
#   - Capacità (sheet B-2-1): unità 10 MW -> divide by 100 -> GW

# Output (salvato in analysis/spatial/output/):
#   - provincial_load_comparison.png : bar chart modello vs EFC per provincia
#   - provincial_load_error.png      : errore % per provincia
#   - provincial_load_error_map.png  : mappa coropletica dell'errore %

# Uso: python analysis/spatial/provincial_load_analysis.py
# """

# import pypsa
# import pandas as pd
# import geopandas as gpd
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import numpy as np

# # ── PARAMETERS ────────────────────────────────────────────────────────────────

# NETWORK_FILE = "results/networks/CN2020_Admin2_noCO2/elec_s_250_ec_lcopt_6h.nc"
# EFC_FILE     = "resources/data/validation/EFC_power_his_data.xlsx"
# GADM_FILE    = "resources/shapes/gadm_shapes.geojson"
# OUTPUT_DIR   = "analysis/spatial/output"
# EFC_YEAR     = 2020

# GADM_NAMES = {
#     "CN.1_1":  "Anhui",          "CN.2_1":  "Beijing",
#     "CN.3_1":  "Chongqing",      "CN.4_1":  "Fujian",
#     "CN.5_1":  "Gansu",          "CN.6_1":  "Guangdong",
#     "CN.7_1":  "Guangxi",        "CN.8_1":  "Guizhou",
#     "CN.9_1":  "Hainan",         "CN.10_1": "Hebei",
#     "CN.11_1": "Heilongjiang",   "CN.12_1": "Henan",
#     "CN.13_1": "Hubei",          "CN.14_1": "Hunan",
#     "CN.15_1": "Jiangsu",        "CN.16_1": "Jiangxi",
#     "CN.17_1": "Jilin",          "CN.18_1": "Liaoning",
#     "CN.19_1": "Inner Mongolia", "CN.20_1": "Ningxia",
#     "CN.21_1": "Qinghai",        "CN.22_1": "Shaanxi",
#     "CN.23_1": "Shandong",       "CN.24_1": "Shanghai",
#     "CN.25_1": "Shanxi",         "CN.26_1": "Sichuan",
#     "CN.27_1": "Tianjin",        "CN.28_1": "Xinjiang",
#     "CN.29_1": "Tibet",          "CN.30_1": "Yunnan",
#     "CN.31_1": "Zhejiang",
# }

# EFC_TO_GADM = {"InnerMongolia": "Inner Mongolia"}

# # ─────────────────────────────────────────────────────────────────────────────


# def load_model_by_province(network_file, gadm_file):
#     n = pypsa.Network(network_file)
#     w = n.snapshot_weightings.generators
#     load_per_bus = (n.loads_t.p_set.multiply(w, axis=0)).sum() / 1e6
#     bus_info = n.buses[["x", "y"]]
#     result = load_per_bus.to_frame("load_twh").join(bus_info)
#     gdf_buses = gpd.GeoDataFrame(
#         result, geometry=gpd.points_from_xy(result.x, result.y), crs="EPSG:4326"
#     )
#     gadm = gpd.read_file(gadm_file)
#     china = gadm[gadm["country"] == "CN"].copy()
#     china["province"] = china["GADM_ID"].map(GADM_NAMES)
#     joined = gpd.sjoin(gdf_buses, china[["GADM_ID", "province", "geometry"]],
#                        how="left", predicate="within")
#     return joined.groupby("province")["load_twh"].sum(), china


# def load_efc_by_province(efc_file, year):
#     efc = pd.read_excel(efc_file, sheet_name="A-1-1", header=5)
#     efc_prov = efc[efc["region"] != "China"][["region", "lang", year]].copy()
#     efc_load = efc_prov[efc_prov["lang"] == "Load"][["region", year]].copy()
#     efc_load.columns = ["region", "load_raw"]
#     efc_load["load_twh"] = efc_load["load_raw"] / 10
#     efc_load["province"] = efc_load["region"].replace(EFC_TO_GADM)
#     return efc_load.set_index("province")["load_twh"]


# if __name__ == "__main__":

#     import os
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     print(f"Loading network: {NETWORK_FILE}")
#     model_load, china = load_model_by_province(NETWORK_FILE, GADM_FILE)

#     print(f"Loading EFC data: {EFC_FILE}")
#     efc_load = load_efc_by_province(EFC_FILE, EFC_YEAR)

#     df = pd.DataFrame({"model": model_load, "efc": efc_load}).dropna()
#     df["error_pct"] = (df["model"] - df["efc"]) / df["efc"] * 100
#     df = df.sort_values("efc", ascending=False)

#     print(f"\n{'Province':<20} {'Model (TWh)':>12} {'EFC (TWh)':>10} {'Error %':>9}")
#     print("-" * 54)
#     for prov, row in df.iterrows():
#         print(f"{str(prov):<20} {row['model']:>12.1f} {row['efc']:>10.1f} {row['error_pct']:>8.1f}%")
#     print("-" * 54)
#     print(f"{'TOTAL':<20} {df['model'].sum():>12.1f} {df['efc'].sum():>10.1f}")

#     # Plot 1: bar chart
#     x = np.arange(len(df)); width = 0.4
#     fig, ax = plt.subplots(figsize=(14, 7))
#     ax.bar(x - width/2, df["model"], width, label="Model",    color="#1f77b4", alpha=0.85)
#     ax.bar(x + width/2, df["efc"],   width, label="EFC 2020", color="#ff7f0e", alpha=0.85)
#     ax.set_xticks(x); ax.set_xticklabels(df.index, rotation=45, ha="right", fontsize=9)
#     ax.set_ylabel("TWh"); ax.set_title("Provincial electricity load: model vs EFC 2020")
#     ax.legend(); ax.grid(True, alpha=0.2, axis="y")
#     plt.tight_layout()
#     out = os.path.join(OUTPUT_DIR, "provincial_load_comparison.png")
#     fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(); print(f"Saved: {out}")

#     # Plot 2: error % bar
#     colors = ["#d62728" if v > 0 else "#1f77b4" for v in df["error_pct"]]
#     fig, ax = plt.subplots(figsize=(14, 5))
#     ax.bar(x, df["error_pct"], color=colors, alpha=0.85)
#     ax.axhline(0, color="black", linewidth=0.8)
#     ax.set_xticks(x); ax.set_xticklabels(df.index, rotation=45, ha="right", fontsize=9)
#     ax.set_ylabel("Error %")
#     ax.set_title("Provincial load error: model vs EFC 2020 (red = overestimate, blue = underestimate)")
#     ax.grid(True, alpha=0.2, axis="y")
#     plt.tight_layout()
#     out = os.path.join(OUTPUT_DIR, "provincial_load_error.png")
#     fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(); print(f"Saved: {out}")

#     # Plot 3: choropleth
#     china_err = china.merge(df["error_pct"].reset_index(), on="province", how="left")
#     fig, ax = plt.subplots(figsize=(14, 10))
#     china_err.plot(column="error_pct", ax=ax, legend=True, cmap="RdBu_r",
#                    vmin=-100, vmax=100, missing_kwds={"color": "lightgrey"},
#                    legend_kwds={"label": "Error % (model vs EFC)", "shrink": 0.6})
#     ax.set_title("CN2020 — Provincial load error % (red = overestimate, blue = underestimate)", fontsize=13)
#     ax.set_axis_off(); plt.tight_layout()
#     out = os.path.join(OUTPUT_DIR, "provincial_load_error_map.png")
#     fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(); print(f"Saved: {out}")

#     print("\nDone.")



"""
provincial_load_analysis.py
===========================
Analisi della distribuzione provinciale del carico elettrico per lo scenario CN2020_Admin2_noCO2.
Confronta il carico modellato (PyPSA-Earth, Admin2) con i dati reali EFC 2020 a livello provinciale.

I bus Admin2 (es. CN.1.11_1_AC) vengono aggregati a livello provinciale (Admin1) estraendo
il codice Admin1 dal GADM_ID tramite split per punto.

Note sulle unità EFC:
  - Carico (sheet A-1-1, lang == "Load"): unità 100 GWh -> divide by 10 -> TWh

Output (salvato in analysis/spatial/output/):
  - provincial_load_comparison.png : bar chart modello vs EFC per provincia
  - provincial_load_error.png      : errore % per provincia
  - provincial_load_error_map.png  : mappa coropletica dell'errore %

Uso: python analysis/spatial/provincial_load_analysis.py
"""

import os
import pypsa
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── PARAMETERS ────────────────────────────────────────────────────────────────

NETWORK_FILE = "results/networks/CN2020_Admin2_noCO2/elec_s_250_ec_lcopt_6h.nc"
EFC_FILE     = "resources/data/validation/EFC_power_his_data.xlsx"
GADM_FILE    = "resources/shapes/gadm_shapes.geojson"
OUTPUT_DIR   = "analysis/spatial/output"
EFC_YEAR     = 2020

GADM_NAMES = {
    "CN.1_1":  "Anhui",          "CN.2_1":  "Beijing",
    "CN.3_1":  "Chongqing",      "CN.4_1":  "Fujian",
    "CN.5_1":  "Gansu",          "CN.6_1":  "Guangdong",
    "CN.7_1":  "Guangxi",        "CN.8_1":  "Guizhou",
    "CN.9_1":  "Hainan",         "CN.10_1": "Hebei",
    "CN.11_1": "Heilongjiang",   "CN.12_1": "Henan",
    "CN.13_1": "Hubei",          "CN.14_1": "Hunan",
    "CN.15_1": "Jiangsu",        "CN.16_1": "Jiangxi",
    "CN.17_1": "Jilin",          "CN.18_1": "Liaoning",
    "CN.19_1": "Inner Mongolia", "CN.20_1": "Ningxia",
    "CN.21_1": "Qinghai",        "CN.22_1": "Shaanxi",
    "CN.23_1": "Shandong",       "CN.24_1": "Shanghai",
    "CN.25_1": "Shanxi",         "CN.26_1": "Sichuan",
    "CN.27_1": "Tianjin",        "CN.28_1": "Xinjiang",
    "CN.29_1": "Tibet",          "CN.30_1": "Yunnan",
    "CN.31_1": "Zhejiang",
}

EFC_TO_GADM = {"InnerMongolia": "Inner Mongolia"}

# ─────────────────────────────────────────────────────────────────────────────


def gadm_id_to_admin1(series):
    """Converte GADM_ID Admin2 in Admin1 ID.
    Es: 'CN.1.11_1' -> 'CN.1_1', 'CN.10.10_1' -> 'CN.10_1'
    """
    return series.str.split('.').str[:2].str.join('.') + "_1"


def load_model_by_province(network_file, gadm_file):
    n = pypsa.Network(network_file)
    w = n.snapshot_weightings.generators

    # Carico totale per bus in TWh
    load_per_bus = (n.loads_t.p_set.multiply(w, axis=0)).sum() / 1e6

    # Rimuovi suffisso _AC/_DC -> ottieni GADM_ID Admin2
    load_per_bus.index = load_per_bus.index.str.replace(r'_(AC|DC)$', '', regex=True)
    load_df = load_per_bus.to_frame("load_twh")
    load_df.index.name = "GADM_ID"

    # Ricava Admin1 ID tramite split (es. "CN.1.11_1" -> "CN.1_1")
    load_df["admin1_id"] = gadm_id_to_admin1(pd.Series(load_df.index)).values
    load_df["province"]  = load_df["admin1_id"].map(GADM_NAMES)

    model_by_prov = load_df.groupby("province")["load_twh"].sum()

    # Carica shapefile e dissolvi Admin2 -> Admin1 per la mappa
    gadm = gpd.read_file(gadm_file)
    china = gadm[gadm["country"] == "CN"].copy()
    china["admin1_id"] = gadm_id_to_admin1(china["GADM_ID"]).values
    china["province"]  = china["admin1_id"].map(GADM_NAMES)
    china_prov = china.dissolve(by="province").reset_index()

    return model_by_prov, china_prov


def load_efc_by_province(efc_file, year):
    efc = pd.read_excel(efc_file, sheet_name="A-1-1", header=5)
    efc_prov = efc[efc["region"] != "China"][["region", "lang", year]].copy()
    efc_load = efc_prov[efc_prov["lang"] == "Load"][["region", year]].copy()
    efc_load.columns = ["region", "load_raw"]
    efc_load["load_twh"] = efc_load["load_raw"] / 10
    efc_load["province"] = efc_load["region"].replace(EFC_TO_GADM)
    return efc_load.set_index("province")["load_twh"]


if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading network: {NETWORK_FILE}")
    model_load, china_prov = load_model_by_province(NETWORK_FILE, GADM_FILE)

    print(f"Loading EFC data: {EFC_FILE}")
    efc_load = load_efc_by_province(EFC_FILE, EFC_YEAR)

    df = pd.DataFrame({"model": model_load, "efc": efc_load}).dropna()
    df["error_pct"] = (df["model"] - df["efc"]) / df["efc"] * 100
    df = df.sort_values("efc", ascending=False)

    print(f"\n{'Province':<20} {'Model (TWh)':>12} {'EFC (TWh)':>10} {'Error %':>9}")
    print("-" * 54)
    for prov, row in df.iterrows():
        print(f"{str(prov):<20} {row['model']:>12.1f} {row['efc']:>10.1f} {row['error_pct']:>8.1f}%")
    print("-" * 54)
    print(f"{'TOTAL':<20} {df['model'].sum():>12.1f} {df['efc'].sum():>10.1f}")

    # ── Plot 1: bar chart confronto ───────────────────────────────────────────
    x = np.arange(len(df))
    width = 0.4
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width/2, df["model"], width, label="Model (CN2020_Admin2_noCO2)", color="#1f77b4", alpha=0.85)
    ax.bar(x + width/2, df["efc"],   width, label="EFC 2020",                    color="#ff7f0e", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("TWh")
    ax.set_title("Provincial electricity load: model vs EFC 2020")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "provincial_load_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # ── Plot 2: errore % per provincia ────────────────────────────────────────
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in df["error_pct"]]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x, df["error_pct"], color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Error %")
    ax.set_title("Provincial load error: model vs EFC 2020  (red = overestimate, blue = underestimate)")
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "provincial_load_error.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    # ── Plot 3: mappa coropletica ─────────────────────────────────────────────
    china_err = china_prov.merge(df["error_pct"].reset_index(), on="province", how="left")
    fig, ax = plt.subplots(figsize=(14, 10))
    china_err.plot(
        column="error_pct", ax=ax, legend=True, cmap="RdBu_r",
        vmin=-100, vmax=100,
        missing_kwds={"color": "lightgrey"},
        legend_kwds={"label": "Error % (model vs EFC)", "shrink": 0.6}
    )
    ax.set_title("CN2020_Admin2_noCO2 — Provincial load error %  (red = overestimate, blue = underestimate)", fontsize=13)
    ax.set_axis_off()
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "provincial_load_error_map.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

    print("\nDone.")