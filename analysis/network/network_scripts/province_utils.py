import geopandas as gpd

ADMIN_PATH = "/p/tmp/ivanra/PyPSA-China-PIK/resources/data/regions/admin2_shapes.geojson"

_admin1_cache = None
_admin2_cache = None

def get_admin1():
    global _admin1_cache
    if _admin1_cache is None:
        admin = gpd.read_file(ADMIN_PATH)
        _admin1_cache = admin.dissolve("NAME_1").reset_index()
    return _admin1_cache

def get_admin2():
    global _admin2_cache
    if _admin2_cache is None:
        _admin2_cache = gpd.read_file(ADMIN_PATH)
    return _admin2_cache

def overlay_admin2_only(ax, linewidth=0.3, edgecolor="black", alpha=0.5):
    """Sovrappone solo confini admin2 (prefetture)."""
    admin2 = get_admin2()
    admin2.plot(ax=ax, facecolor="none", edgecolor=edgecolor,
                linewidth=linewidth, alpha=alpha)

def overlay_admin_boundaries(ax, linewidth=0.5, edgecolor="black", alpha=1.0,
                              linewidth_admin2=0.3, edgecolor_admin2="black", alpha_admin2=0.5):
    """Sovrappone confini admin2 (sottile) e admin1 (marcato) su asse matplotlib.
    Default invariati (nero) per compatibilita' con script esistenti.
    Per sfondi scuri (es. colormap viridis), passare edgecolor="white",
    edgecolor_admin2="white"."""
    admin2 = get_admin2()
    admin1 = get_admin1()
    admin2.plot(ax=ax, facecolor="none", edgecolor=edgecolor_admin2,
                linewidth=linewidth_admin2, alpha=alpha_admin2)
    admin1.plot(ax=ax, facecolor="none", edgecolor=edgecolor,
                linewidth=linewidth, alpha=alpha)


def add_province(n):
    """
    Assegna ogni bus della rete alla sua provincia (NAME_1), via point-in-polygon
    sulle coordinate x/y del bus. Applica il fix noto per il bus che cade in mare
    per imprecisione dello shapefile costiero (CN.6.18_1_AC, rete 425 nodi).
    Propaga "province" a generators, storage_units, links (via bus0).

    Fonte: codice originale di Ivan Ramirez, incapsulato da province_stats_and_map.py.
    """
    from shapely import Point

    admin1 = get_admin1()

    n.buses["point"] = n.buses.apply(lambda row: Point(row.x, row.y), axis=1)

    if "CN.6.18_1_AC" in n.buses.index:
        n.buses.loc["CN.6.18_1_AC", "point"] = Point(109.870802, 20.718244)
        print("Applied sea-point fix for CN.6.18_1_AC")

    n.buses["province"] = n.buses["point"].apply(
        lambda pt: admin1[admin1.geometry.contains(pt)]["NAME_1"].values[0]
        if len(admin1[admin1.geometry.contains(pt)]) > 0 else "unmatched"
    ).astype(str)

    unmatched = (n.buses["province"] == "unmatched").sum()
    print(f"Bus non assegnati a nessuna provincia: {unmatched} / {len(n.buses)}")
    if unmatched > 0:
        print(f"Esempi: {n.buses[n.buses['province']=='unmatched'].index.tolist()[:5]}")

    for component in ["generators", "storage_units", "links"]:
        df = getattr(n, component)
        if not df.empty and "bus" in df.columns:
            df["province"] = df["bus"].map(n.buses["province"])
        elif not df.empty and "bus0" in df.columns:
            df["province"] = df["bus0"].map(n.buses["province"])

    return n
