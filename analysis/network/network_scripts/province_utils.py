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

def overlay_admin_boundaries(ax, linewidth=0.5, edgecolor="black", alpha=1.0):
    """Sovrappone confini admin2 (grigio) e admin1 (nero) su asse matplotlib."""
    admin2 = get_admin2()
    admin1 = get_admin1()
    admin2.plot(ax=ax, facecolor="none", edgecolor="black",
                linewidth=0.3, alpha=0.5)
    admin1.plot(ax=ax, facecolor="none", edgecolor=edgecolor,
                linewidth=linewidth, alpha=alpha)
