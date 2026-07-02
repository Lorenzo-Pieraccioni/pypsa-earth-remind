fr"""Plot network maps and energy system visualizations.

This module creates geographical network maps showing generation capacities,
transmission lines, energy flows, and other network characteristics for
the PyPSA-China energy system model.
"""

import logging
import os

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pypsa
from _helpers import (
    configure_logging,
    mock_snakemake,
    set_plot_test_backend,
    setup_proj_environment,
)
from _pypsa_helpers import filter_carriers, unscale_biomass_network
from _plot_utilities import (
    fix_network_names_colors,
    make_nice_tech_colors,
    set_plot_style,
    expand_heat_carriers,
    sanitize_carriers,
)
from constants import (
    CURRENCY,
    PLOT_CAP_UNITS,
    PLOT_COST_UNITS,
    PLOT_SUPPLY_UNITS,
    CARRIER_HIERARCHY,
)
from readers_geospatial import read_admin2_shapes

# from make_summary import assign_carriers
from pypsa.plot import add_legend_circles, add_legend_lines, add_legend_patches

logger = logging.getLogger(__name__)


# ==============================================================================
# Network plotting helper functions
# ==============================================================================

# Cache for China boundary to avoid repeated file I/O
_china_boundary_cache = None


def calculate_link_widths(
    network: pypsa.Network,
    carriers: str | list[str] | tuple[str, ...] | None = "AC",
    min_p: float = 1,
    additions: bool = False,
) -> pd.Series:
    """Calculate link widths for plotting, filtering by carrier and minimum capacity.

    Consolidates the duplicated calc_link_plot_width logic from plot_network.py.

    Args:
        network: PyPSA network object
        carriers: Transmission carrier(s) to filter. If None, include all link
            carriers with non-zero length.
        min_p: Minimum capacity threshold in MW (default 1)
        additions: If True, use p_nom instead of p_nom_opt (default False)

    Returns:
        Series of link widths indexed by link name
    """

    if carriers is None:
        carrier_filter = None
    elif isinstance(carriers, str):
        carrier_filter = {carriers}
    else:
        carrier_filter = set(carriers)

    def calc_width(row):
        if row.length == 0:
            return 0
        if carrier_filter is not None and row.carrier not in carrier_filter:
            return 0
        elif additions:
            return max(row.p_nom, min_p) if row.p_nom >= min_p else 0
        else:
            return max(row.p_nom_opt, min_p) if row.p_nom_opt >= min_p else 0

    return network.links.apply(calc_width, axis=1)


def prepare_edge_data(
    network: pypsa.Network,
    link_carriers: str | list[str] | tuple[str, ...] | None = ("AC", "DC"),
    linewidth_factor: float = 1.0,
    min_edge_capacity: float = 500,
    additions: bool = False,
) -> tuple[pd.Series, pd.Series]:
    """Prepare edge widths for network plotting.

    Consolidates repeated edge data preparation logic.

    Args:
        network: PyPSA network object
        link_carriers: Transmission link carrier(s) to include
        linewidth_factor: Divisor to scale line widths for visualization
        min_edge_capacity: Minimum capacity to display (MW)
        additions: If True, plot additions only

    Returns:
        Tuple of (line_widths, link_widths) where:
        - line_widths: Series of normalized widths for lines
        - link_widths: Series of normalized widths for links
    """

    def _normalize_widths(capacities: pd.Series) -> pd.Series:
        if capacities.empty:
            return capacities.astype(float)
        widths = capacities.clip(lower=min_edge_capacity)
        widths = widths.replace(min_edge_capacity, 0)
        return widths / linewidth_factor

    link_widths = calculate_link_widths(
        network, link_carriers, min_p=min_edge_capacity, additions=additions
    )

    if additions:
        line_caps = network.lines.s_nom.copy()
    else:
        line_caps = network.lines.s_nom_opt.copy()

    line_widths = _normalize_widths(line_caps)
    link_widths = _normalize_widths(link_widths)

    return line_widths, link_widths


def prepare_network_plot_data(
    network: pypsa.Network,
    metric_type: str,
    carriers: list = None,
    opts: dict = None,
    components=None,
    bus_carrier="AC",
) -> dict:
    """Unified data preparation with stats/other for all network map types.

    Centralizes bus/edge
      logic used across plot_cost_map,
    plot_energy_map, plot_capacity_map, and plot_nodal_prices.

    Args:
        network (pypsa.Network): PyPSA network object
        metric_type (str): One of 'cost', 'energy', 'capacity', 'price'
        carriers (list | None): List of carriers to include (None = all)
        opts (dict): Plottimng options dict
        components (list): components for the stats
        bus_carrier: carrier for bus-based metrics (default "AC")

    Returns:
        Dict with keys: bus_sizes, bus_colors, line_widths, link_widths,
        line_color, link_color, edge_colors, legend_data, metric_data
    """
    if opts is None:
        opts = {}

    config_key = f"{metric_type}_map"
    config = opts.get(config_key, {})

    result = {
        "bus_sizes": pd.Series(dtype=float),
        "bus_colors": pd.Series(dtype=object),
        "line_widths": pd.Series(dtype=float),
        "link_widths": pd.Series(dtype=float),
        "line_color": "indigo",
        "link_color": "indigo",
        "edge_colors": "indigo",
        "legend_data": {},
        "metric_data": None,
    }

    # Prepare edges
    link_carriers = config.get("link_carriers", ("AC", "DC"))
    linewidth_factor = config.get("linewidth_factor", 6e3)
    min_edge_capacity = config.get("min_edge_capacity", 500)

    line_widths, link_widths = prepare_edge_data(
        network, link_carriers, linewidth_factor, min_edge_capacity
    )
    result["line_widths"] = line_widths
    result["link_widths"] = link_widths

    fallback_edge_color = config.get("edge_color", "indigo")
    default_line_color = opts.get("tech_colors", {}).get("AC", fallback_edge_color)
    default_link_color = opts.get("tech_colors", {}).get("DC", fallback_edge_color)

    result["line_color"] = config.get("line_color", default_line_color)
    result["link_color"] = config.get("link_color", default_link_color)
    result["edge_colors"] = fallback_edge_color

    # Prepare buses based on metric type
    if metric_type == "cost":
        # CAPEX by location and carrier
        costs = network.statistics.capex(groupby=["location", "carrier"], components=components)
        costs = costs.groupby(level=["location", "carrier"]).sum()
        if "" in costs.index:
            costs = costs.drop("")

        # Fill missing buses
        bus_idx = pd.MultiIndex.from_product([network.buses.index, ["AC"]])
        costs = costs.reindex(bus_idx.union(costs.index), fill_value=0)

        result["bus_sizes"] = costs.fillna(0)
        result["metric_data"] = costs

    elif metric_type == "energy":
        # Energy supply/withdrawal
        if components is None:
            components = ["Generator", "Link"]
        energy_supply = network.statistics.supply(
            groupby=["bus", "carrier"],
            bus_carrier=bus_carrier,
            carrier=carriers,
            components=components,
        )
        result["bus_sizes"] = energy_supply.groupby(level=[1, 2]).sum()
        result["metric_data"] = result["bus_sizes"]

    elif metric_type == "capacity":
        # Installed capacity
        carrier = config.get("carrier", "AC")
        if components is None:
            components = ["Generator", "Link", "StorageUnit"]
        capacity = network.statistics.optimal_capacity(
            groupby=["bus", "carrier"],
            bus_carrier=bus_carrier,
            components=components,
            carrier=carriers,
        ).clip(lower=0)
        capacity = capacity.groupby(level=["bus", "carrier"]).sum().dropna()
        result["bus_sizes"] = capacity
        result["metric_data"] = capacity

    elif metric_type == "price":
        # Nodal prices (revenue / withdrawal)
        carrier = config.get("carrier", "AC")
        revenue = network.statistics.revenue(
            groupby=["bus", "carrier", "bus_carrier"],
            components="Load",
            bus_carrier=bus_carrier,
        )
        withdrawal = network.statistics.withdrawal(
            components="Load",
            groupby=["bus", "carrier", "bus_carrier"],
            bus_carrier=bus_carrier,
            carrier=carriers,
        )
        nodal_prices = (revenue / withdrawal * -1).droplevel([1, 2])
        result["metric_data"] = nodal_prices
    elif metric_type == "withdrawal":
        # For price maps, bus_sizes show consumption
        energy_consum = network.statistics.withdrawal(
            groupby=["bus", "carrier"],
            components=["Load"],
            bus_carrier=bus_carrier,
            carrier=carriers,
        )
        result["bus_sizes"] = energy_consum.groupby(level=1).sum()

    else:
        raise ValueError(f"Unsupported metric_type: {metric_type}")
    return result


def add_metric_panel(
    fig: plt.Figure,
    df: pd.DataFrame,
    preferred_order: pd.Index,
    tech_colors: dict,
    unit: str,
    ylabel: str,
    ax_loc: list = None,
    plot_additions: bool = False,
    **kwargs,
) -> plt.Axes:
    """Add a metric panel (cost/energy/capacity) to the figure.

    Consolidates add_cost_pannel, add_energy_pannel, add_capacity_pannel.

    Args:
        fig: Figure object to add panel to
        df: DataFrame with metrics to plot (columns are scenarios/years)
        preferred_order: Ordering for carriers in plot
        tech_colors: Technology color mapping
        unit: Unit string for display (e.g., 'bEUR/a', 'TWh/a', 'GW')
        ylabel: Y-axis label
        ax_loc: Panel location [left, bottom, width, height]
        plot_additions: Whether showing additions (for percentage label)
        **kwargs: Additional arguments passed to DataFrame.plot()

    Returns:
        The created axes object
    """
    if ax_loc is None:
        ax_loc = [-0.09, 0.28, 0.09, 0.45]

    ax = fig.add_axes(ax_loc)

    # Reorder carriers by preferred order
    reordered = preferred_order.intersection(df.index).append(df.index.difference(preferred_order))

    # Map colors (case-insensitive)
    colors = {k.lower(): v for k, v in tech_colors.items()}
    color_list = []
    for k in reordered:
        if k.lower() in colors:
            color_list.append(colors[k.lower()])
        else:
            color_list.append("lightgrey")

    # Create stacked bar plot
    df.loc[reordered, df.columns].T.plot(
        kind="bar",
        ax=ax,
        stacked=True,
        color=color_list,
        **kwargs,
    )

    ax.legend().remove()
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(ax.get_xticklabels(), rotation="horizontal")
    ax.grid(axis="y")
    ax.set_ylim([0, df.sum().max() * 1.1])

    # Add percentage label if showing additions
    if plot_additions and "added" in df.columns and "total" in df.columns:
        percent = np.round((df.sum()["added"] / df.sum()["total"]) * 100)
        ax.text(0.85, (df.sum()["added"] + df.sum().max() * 0.02), f"{percent}%", color="black")

    # fig.tight_layout()
    return ax


def get_node_boundaries(
    shapes_path=None, admin2_shapes=None, dissolve: bool = True
) -> gpd.GeoSeries:
    """Get China administrative boundary for plotting.

    Loads and optionally caches shapes for China boundary.

    Args:
        shapes_path: Path to clustered regions GeoJSON (e.g., regions_onshore_s_*.geojson)
                     If provided, loads these instead of raw admin2 shapes
        admin2_shapes: Pre-loaded GeoDataFrame of admin2 shapes (optional)
        dissolve: If True, dissolve into single boundary (default True)

    Returns:
        Geoseries with Node/China boundary/boundaries
    """
    global _china_boundary_cache

    # Use cached boundary if available and appropriate
    if _china_boundary_cache is not None and dissolve and shapes_path is None:
        return _china_boundary_cache

    # Load clustered regions if path provided
    if shapes_path is not None:
        if not os.path.exists(shapes_path):
            raise FileNotFoundError(f"Clustered regions not found at {shapes_path}")
        node_shapes = gpd.read_file(shapes_path).set_index("cluster", drop=True)
        logger.info(f"Loaded {len(node_shapes)} clustered regions from {shapes_path}")

    # Load default admin2 shapes if nothing else provided
    if admin2_shapes is None and shapes_path is None:
        default_path = "resources/data/regions/admin2_shapes.geojson"
        if not os.path.exists(default_path):
            raise FileNotFoundError(f"Admin2 shapes not found at {default_path}")

        node_shapes = read_admin2_shapes(default_path).set_index("cluster", drop=True)
    elif admin2_shapes is not None:
        node_shapes = admin2_shapes

    # Dissolve (total contour) or return as-is
    if dissolve:
        boundary = node_shapes.dissolve().iloc[0]
        # Only cache the default admin2 boundary (not clustered)
        if shapes_path is None:
            _china_boundary_cache = boundary
        return boundary
    else:
        return node_shapes


def apply_china_extent(ax, admin2_shapes=None, margin: float = 2.0):
    """Apply China-focused map extent to axes.

    Args:
        ax: Matplotlib/cartopy axes
        admin2_shapes: Pre-loaded admin2 shapes (optional)
        margin: Margin in degrees around China boundary
    """
    boundary = get_node_boundaries(admin2_shapes=admin2_shapes, dissolve=True)
    bounds = boundary.geometry.bounds  # [minx, miny, maxx, maxy]
    # Set extent in lon/lat order for cartopy: [lon_min, lon_max, lat_min, lat_max]
    ax.set_extent(
        [
            bounds[0] - margin,  # lon_min
            bounds[2] + margin,  # lon_max
            bounds[1] - margin,  # lat_min
            bounds[3] + margin,  # lat_max
        ],
        crs=ccrs.PlateCarree(),
    )


def add_gridlines(ax, gridlines_style: dict = None, frame_style: dict = None):
    """Add gridlines with lat/lon labels to a cartopy axes.

    Args:
        ax: Cartopy GeoAxes
        gridlines_style: Dict with gridlines configuration options:
            - draw_labels (bool): Whether to draw coordinate labels
            - linewidth (float): Width of gridlines
            - color (str): Color of gridlines
            - alpha (float): Transparency of gridlines
            - linestyle (str): Line style
            - x_inline/y_inline (bool): Whether to draw labels inline
            - xlocs/ylocs (list): Specific longitude/latitude values to draw
            - xlabel_style/ylabel_style (dict): Label formatting options
        frame_style: Dict with frame/outline configuration:
            - enabled (bool): Whether to draw a frame around the map
            - linewidth (float): Width of the frame
            - edgecolor (str): Color of the frame
            - facecolor (str): Fill color of the frame
    """
    if gridlines_style is None:
        gridlines_style = {}

    # Create gridlines
    gl = ax.gridlines(
        draw_labels=gridlines_style.get("draw_labels", True),
        linewidth=gridlines_style.get("linewidth", 0.5),
        color=gridlines_style.get("color", "gray"),
        alpha=gridlines_style.get("alpha", 0.5),
        linestyle=gridlines_style.get("linestyle", "--"),
    )

    # Configure label positions
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = gridlines_style.get("draw_labels", True)
    gl.bottom_labels = gridlines_style.get("draw_labels", True)

    # Configure inline labels
    gl.x_inline = gridlines_style.get("x_inline", False)
    gl.y_inline = gridlines_style.get("y_inline", False)

    # Set specific grid locations if provided
    if gridlines_style.get("xlocs") is not None:
        gl.xlocator = mticker.FixedLocator(gridlines_style["xlocs"])
    if gridlines_style.get("ylocs") is not None:
        gl.ylocator = mticker.FixedLocator(gridlines_style["ylocs"])

    # Configure label formatting
    xlabel_style = gridlines_style.get("xlabel_style", {})
    ylabel_style = gridlines_style.get("ylabel_style", {})

    if xlabel_style:
        gl.xlabel_style = xlabel_style
    if ylabel_style:
        gl.ylabel_style = ylabel_style

    # Format labels to show degrees (like "50°E", "30°N")
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    # Add frame/outline around the map if requested
    if frame_style and frame_style.get("enabled", False):
        import matplotlib.patches as mpatches

        # Draw a box in axes coordinates (0,0 to 1,1) to create a frame
        rect = mpatches.Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            fill=False,
            linewidth=frame_style.get("linewidth", 1.0),
            edgecolor=frame_style.get("edgecolor", "black"),
            clip_on=False,
            zorder=100,
        )
        ax.add_patch(rect)

    return gl


def get_projection(projection_name: str = "PlateCarree"):
    """Get a cartopy projection by name.

    Args:
        projection_name: Name of the projection (e.g., 'PlateCarree', 'Mercator', 'LambertConformal')

    Returns:
        Cartopy projection instance
    """
    projection_map = {
        "PlateCarree": ccrs.PlateCarree,
        "Mercator": ccrs.Mercator,
        "LambertConformal": ccrs.LambertConformal,
        "AlbersEqualArea": ccrs.AlbersEqualArea,
        "Robinson": ccrs.Robinson,
        "Orthographic": ccrs.Orthographic,
    }

    projection_class = projection_map.get(projection_name, ccrs.PlateCarree)
    return projection_class()


# ==============================================================================
# Network map plotting functions
# ==============================================================================


def plot_map(
    network: pypsa.Network,
    tech_colors: dict,
    line_widths: pd.Series,
    link_widths: pd.Series,
    bus_colors: dict | pd.Series,
    bus_sizes: pd.Series,
    edge_colors: pd.Series | str = "black",
    line_color: pd.Series | str | None = None,
    link_color: pd.Series | str | None = None,
    add_ref_edge_sizes=True,
    add_ref_bus_sizes=True,
    add_legend=True,
    bus_unit_conv=PLOT_COST_UNITS,
    edge_unit_conv=PLOT_CAP_UNITS,
    ax=None,
    expanded_lines_only=False,
    color_geomap=True,
    projection=None,
    gridlines_config=None,
    do_tight_layout=True,
    **kwargs,
) -> plt.Axes:
    """Plot the network on a map.

    Draws network elements (bus pies, transmission lines, and legends) on the
    given axes.  Region background drawing, choropleth coloring, extent setting,
    and colorbar creation are the responsibility of the wrapper :func:`plot_network`.

    Args:
        network (pypsa.Network): the pypsa network
        tech_colors (dict): technology color mapping
        edge_colors (pd.Series|str): fallback edge colors used when line/link
            colors are not provided
        line_color (pd.Series|str|None): HVAC line color(s)
        link_color (pd.Series|str|None): HVDC link color(s)
        line_widths (pd.Series): line widths indexed by line names
        link_widths (pd.Series): link widths indexed by link names
        bus_colors (pd.Series): the series of bus colors
        bus_sizes (pd.Series): the series of bus sizes
        add_ref_edge_sizes (bool): add reference line sizes in legend. Defaults to True.
        add_ref_bus_sizes (bool): add reference bus sizes in legend. Defaults to True.
        add_legend (bool): add carrier color patches legend. Defaults to True.
        bus_unit_conv (float): unit conversion factor for bus sizes.
        edge_unit_conv (float): unit conversion factor for edge sizes.
        ax (plt.Axes, optional): the plotting ax. Defaults to None (new figure).
        expanded_lines_only (bool): if True, plot expansions only (p_nom_opt - p_nom).
        color_geomap (bool): if True, colour the ocean/land with cartopy. Defaults to True.
        projection: cartopy projection (used only when ax is None).
        gridlines_config (dict): gridlines configuration dict (optional).
        do_tight_layout (bool): call fig.tight_layout() at the end. Defaults to True.
        **kwargs: additional keyword arguments, including:
            - carrier_legend_kw (dict): override legend kwargs for carrier patches.
            - edge_legend_kw (dict): override legend kwargs for edge size reference.
            - bus_legend_kw (dict): override legend kwargs for bus size reference.
            - ref_edge_unit (str): unit label for edge legend sizes.
            - ref_bus_unit (str): unit label for bus legend sizes.
            - linewidth_factor (float): divisor for edge widths.
            - bus_size_factor (float): divisor for bus sizes.
            - ref_edge_sizes (list): reference edge sizes (raw values pre-scaling).
            - ref_bus_sizes (list): reference bus sizes (raw values pre-scaling).
            - edge_ref_title (str): title for edge size legend.
            - bus_ref_title (str): title for bus size legend.
            - boundaries: explicit map boundaries.
    """

    if not ax:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    bus_colors = pd.Series(bus_colors)
    if bus_sizes.index.nlevels > 1:
        missing = bus_sizes.index.get_level_values(1).difference(bus_colors.index)
    else:
        missing = bus_sizes.index.difference(bus_colors.index)
    if not missing.empty:
        raise ValueError(f"Missing colors for bus carriers: {missing.tolist()}")

    if expanded_lines_only:
        line_caps_original = network.links.copy()
        network.links.loc[:, "p_nom_opt"] -= network.links.loc[:, "p_nom"]

    effective_line_color = edge_colors if line_color is None else line_color
    effective_link_color = edge_colors if link_color is None else link_color
    has_visible_lines = not line_widths.empty and line_widths.gt(0).any()
    has_visible_links = not link_widths.empty and link_widths.gt(0).any()

    # Build per-carrier link color Series: AC carrier links (manual backend or OSM HV extras)
    # get line_color (AC color), DC carrier links get link_color (DC color).
    if isinstance(effective_line_color, str) and isinstance(effective_link_color, str):
        plot_link_color = network.links.carrier.map(
            {"AC": effective_line_color, "DC": effective_link_color}
        ).fillna(effective_link_color)
    else:
        plot_link_color = effective_link_color

    # Plot network
    network.plot(
        bus_size=bus_sizes,
        bus_color=bus_colors,
        line_color=effective_line_color,
        link_color=plot_link_color,
        line_width=line_widths,
        link_width=link_widths,
        ax=ax,
        geomap_color=color_geomap,
        boundaries=kwargs.get("boundaries", None),
    )

    if expanded_lines_only:
        # restore original line capacities
        network.links.loc[:, "p_nom_opt"] = line_caps_original.loc[:, "p_nom_opt"]

    if add_legend:
        # Only show carriers that are actually present (non-zero) in the plot
        if bus_sizes.index.nlevels > 1:
            carrier_totals = bus_sizes.groupby(level=1).sum()
            carriers = carrier_totals[carrier_totals > 0].index
        else:
            carriers = bus_sizes[bus_sizes > 0].index
        # Ensure all carriers have colors, use default color for missing ones
        colors = []
        for carrier in carriers:
            if carrier in tech_colors:
                colors.append(tech_colors[carrier])
            else:
                colors.append("lightgrey")  # Default color for missing carriers

        # Determine which transmission carrier types are actually visible
        visible_link_carriers = set()
        if has_visible_links:
            visible_idx = link_widths[link_widths > 0].index
            visible_link_carriers = set(network.links.loc[visible_idx, "carrier"].unique())

        labels = carriers.to_list()
        if isinstance(effective_line_color, str) and isinstance(effective_link_color, str):
            has_visible_ac = has_visible_lines or "AC" in visible_link_carriers
            has_visible_dc = "DC" in visible_link_carriers
            if not has_visible_ac and not has_visible_dc:
                colors.append(effective_line_color)
                labels.append("Transmission grid")
            else:
                if has_visible_ac:
                    colors.append(effective_line_color)
                    # OSM backend: AC in n.lines → "HVAC lines"; manual: AC carrier links → "HVAC links"
                    labels.append("HVAC lines" if has_visible_lines else "HVAC links")
                if has_visible_dc:
                    colors.append(effective_link_color)
                    labels.append("HVDC links")
        elif isinstance(edge_colors, str):
            colors.append(edge_colors)
            labels.append("Transmission grid")
        else:
            colors += edge_colors.values.to_list()
            labels += edge_colors.index.to_list()

        default_carrier_kw = {"bbox_to_anchor": (1.42, 1.04), "frameon": False}
        carrier_legend_kw = kwargs.get("carrier_legend_kw", default_carrier_kw)
        add_legend_patches(ax, colors, labels, legend_kw=carrier_legend_kw)

    _ref_edge_sizes = kwargs.get("ref_edge_sizes", [1e5, 5e5])
    ref_edge_color = None
    if isinstance(effective_line_color, str):
        ref_edge_color = effective_line_color
    elif isinstance(effective_link_color, str):
        ref_edge_color = effective_link_color
    elif isinstance(edge_colors, str):
        ref_edge_color = edge_colors

    if add_ref_edge_sizes and ref_edge_color is not None and _ref_edge_sizes:
        ref_unit = kwargs.get("ref_edge_unit", "GW")
        size_factor = float(kwargs.get("linewidth_factor", 1e5))
        ref_sizes = list(map(lambda x: float(x) / size_factor, _ref_edge_sizes))
        labels = [f"{float(s) / edge_unit_conv} {ref_unit}" for s in _ref_edge_sizes]
        label = f"Grid {'expanded' if expanded_lines_only else ''} capacity"

        default_edge_kw = dict(
            loc="upper left",
            bbox_to_anchor=(0.26, 1.0),
            frameon=False,
            labelspacing=0.8,
            handletextpad=2,
            title=kwargs.get("edge_ref_title", label),
        )
        edge_legend_kw = {**default_edge_kw, **kwargs.get("edge_legend_kw", {})}
        add_legend_lines(
            ax,
            ref_sizes,
            labels,
            patch_kw=dict(color=ref_edge_color),
            legend_kw=edge_legend_kw,
        )

    # add reference bus sizes from the units
    if add_ref_bus_sizes:
        ref_unit = kwargs.get("ref_bus_unit", "bEUR/a")
        size_factor = float(kwargs.get("bus_size_factor", 1e10))
        _ref_bus_sizes = kwargs.get("ref_bus_sizes", [2e10, 1e10, 5e10])
        ref_sizes = list(map(lambda x: float(x) / size_factor, _ref_bus_sizes))
        labels = [f"{float(s) / bus_unit_conv:.0f} {ref_unit}" for s in _ref_bus_sizes]

        default_bus_kw = {
            "loc": "upper left",
            "bbox_to_anchor": (0.0, 1.0),
            "labelspacing": 0.8,
            "frameon": False,
            "handletextpad": 0,
            "title": kwargs.get("bus_ref_title", "UNDEFINED TITLE"),
        }
        bus_legend_kw = {**default_bus_kw, **kwargs.get("bus_legend_kw", {})}

        add_legend_circles(
            ax,
            ref_sizes,
            labels,
            srid=network.srid,
            patch_kw=dict(facecolor="lightgrey"),
            legend_kw=bus_legend_kw,
        )

    # Add gridlines if configured
    if gridlines_config and gridlines_config.get("enabled", False):
        add_gridlines(ax, gridlines_config.get("style", {}), gridlines_config.get("frame", {}))

    if do_tight_layout:
        fig.tight_layout()

    return ax


def plot_network(
    network: pypsa.Network,
    opts: dict,
    metric_type: str = "cost",
    carriers: list = None,
    node_values: pd.Series | None = None,
    china_only: bool | None = None,
    regions_path: str | None = None,
    add_panel: bool = False,
    save_path: os.PathLike | None = None,
    components: list = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Unified network map plotting function with optional China boundary and node heatmap.

    This is the central plotting function that consolidates logic from plot_cost_map,
    plot_energy_map, plot_capacity_map, and plot_nodal_prices.

    Responsibilities of this wrapper (NOT delegated to ``plot_map``):
    - Loading and drawing region shapes (china_only background: static or choropleth).
    - Applying China-focused map extent.
    - Adding the choropleth / heatmap colorbar (sized via ``inset_axes`` to stay
      within the axes frame).
    - Overriding legend positions for choropleth mode.

    Args:
        network: PyPSA network object
        opts: Configuration dict from plot_config.yaml
        metric_type: Type of map - 'cost', 'energy', 'capacity', or 'price'
        carriers: List of technology carriers to include (e.g., ['Solar', 'Wind']).
                  None = all carriers.
        node_values: Optional Series for continuous heatmap coloring of nodes
                     Indexed by bus names. Overrides default bus colors.
        china_only: If True, overlay clustered region boundaries and focus on China extent.
                    If None, use config default. If False, never apply china_only.
        regions_path: Path to clustered regions GeoJSON (e.g., regions_onshore_s_*.geojson)
        add_panel: Whether to add side panel with bar chart
        save_path: Path to save figure (None = no save)
        components: List of component types to include (e.g., ['Generator', 'Link']).
                    None = use defaults for metric_type.
        **kwargs: Additional arguments passed to plot_map

    Returns:
        Tuple of (figure, axes)

    Examples:
        # Basic cost map
        fig, ax = plot_network(network, opts, 'cost')

        # Heatmap of emissions per node
        emissions = pd.Series({bus: value for bus, value in ...}, name='CO2')
        fig, ax = plot_network(network, opts, 'capacity', node_values=emissions)

        # China-only choropleth view
        fig, ax = plot_network(network, opts, 'energy', china_only=True)

        # Filter to show only renewables
        fig, ax = plot_network(network, opts, 'capacity',
                               carriers=['Solar', 'Onshore Wind', 'Offshore Wind'])
    """
    # ── Configuration ────────────────────────────────────────────────────────
    config_key = f"{metric_type}_map"
    config = opts.get(config_key, {})

    # Apply config default only if china_only wasn't explicitly set
    if china_only is None:
        china_only = config.get("china_only", False)
    # Pop bus_carrier early so it is never forwarded to plot_map as an unknown kwarg
    bus_carrier = kwargs.pop("bus_carrier", "AC")
    plot_data = prepare_network_plot_data(
        network, metric_type, carriers, opts, components, bus_carrier=bus_carrier
    )

    # For price maps, automatically derive node_values from nodal prices
    # and bus_sizes from load withdrawal when not explicitly supplied
    if metric_type == "price" and node_values is None:
        node_values = plot_data["metric_data"]
        withdrawal_data = prepare_network_plot_data(
            network, "withdrawal", carriers, opts, components, bus_carrier=bus_carrier
        )
        plot_data["bus_sizes"] = withdrawal_data["bus_sizes"]
        norm = plt.Normalize(vmin=node_values.min(), vmax=node_values.max())
        cmap = plt.get_cmap("plasma")
        plot_data["bus_colors"] = node_values.map(lambda x: cmap(norm(x)) if pd.notnull(x) else "lightgrey")

    projection_name = opts.get("map_projection", "PlateCarree")
    projection = get_projection(projection_name)
    gridlines_config = opts.get("gridlines", {})

    # ── Figure setup ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(subplot_kw={"projection": projection})
    figsize = list(config.get("figsize", [10, 8]))
    if china_only:
        figsize = [figsize[0] + 4, figsize[1] + 2]
    fig.set_size_inches(figsize)

    tech_colors = make_nice_tech_colors(opts["tech_colors"], opts["nice_names"])

    # ── Pop node-coloring kwargs early to avoid passing them to plot_map ─────
    if node_values is not None:
        node_cmap = kwargs.pop("node_cmap", config.get("node_cmap", "plasma"))
        node_vmin = kwargs.pop("node_vmin", node_values.min())
        node_vmax = kwargs.pop("node_vmax", node_values.max())
        norm = plt.Normalize(vmin=node_vmin, vmax=node_vmax)
        cmap_obj = plt.get_cmap(node_cmap)
        nan_color = opts.get("nan_color") or "lightgrey"
    else:
        node_cmap = node_vmin = node_vmax = norm = cmap_obj = nan_color = None

    # ── Region background drawing (china_only) – choropleth or static ────────
    region_shapes = None
    choropleth_sm = None  # ScalarMappable for colorbar, set only when choropleth used
    choropleth_ylabel = ""
    add_choropleth_colorbar = False

    if china_only:
        region_shapes = get_node_boundaries(shapes_path=regions_path, dissolve=False)
        if region_shapes is not None:
            if node_values is not None and not node_values.empty:
                logger.info("Drawing choropleth background for china_only map")
                # Aggregate per-bus values to region level.
                # Bus names may carry a carrier suffix (e.g. "BJ AC"); strip it
                # so the index aligns with the cluster names in region_shapes.
                region_vals = node_values.copy()
                region_vals.index = region_vals.index.str.split(" ").str[0]
                region_vals = region_vals.groupby(level=0).mean()
                region_vals_df = pd.concat(
                    [region_shapes, region_vals.rename("plot_value")], axis=1
                ).dropna(subset=["geometry"])
                region_vals_df.plot(
                    ax=ax,
                    column="plot_value",
                    cmap=node_cmap,
                    vmin=node_vmin,
                    vmax=node_vmax,
                    edgecolor="darkgray",
                    linewidth=1.0,
                    transform=ccrs.PlateCarree(),
                    legend=False,
                    missing_kwds={
                        "color": nan_color,
                        "edgecolor": "darkgray",
                        "linewidth": 1.0,
                    },
                )
                choropleth_sm = plt.cm.ScalarMappable(
                    cmap=node_cmap, norm=plt.Normalize(vmin=node_vmin, vmax=node_vmax)
                )
                choropleth_sm.set_array([])
                choropleth_label = kwargs.get(
                    "colorbar_label",
                    config.get("node_heatmap", {}).get("colorbar_label", "Value"),
                )
                add_choropleth_colorbar = config.get("node_heatmap", {}).get(
                    "add_region_colorbar", True
                )
            else:
                logger.info("Drawing static region background for china_only map")
                region_shapes.plot(
                    ax=ax,
                    color="#f0f0f0",
                    edgecolor="lightgray",
                    linewidth=0.8,
                    transform=ccrs.PlateCarree(),
                )
        else:
            raise RuntimeError(
                "china_only=True but region_shapes could not be loaded. "
                "Provide a valid regions_path or ensure admin2_shapes.geojson exists."
            )

    # ── Bus coloring ──────────────────────────────────────────────────────────
    # ugly logic to be revisited
    add_non_choropleth_colorbar = False
    add_legend_flag = True

    if node_values is not None:
        if china_only:
            # Choropleth background already drawn; bus pies keep carrier colors
            if plot_data["bus_colors"].empty:
                bus_colors = tech_colors
            else:
                bus_colors = plot_data["bus_colors"]
            bus_sizes_to_plot = plot_data["bus_sizes"]
            add_legend_flag = metric_type != "price"  # suppress carrier legend for price maps
        else:
            # Apply heatmap directly to bus pies (scalar coloring)
            bus_sizes_agg = (
                plot_data["bus_sizes"].groupby(level=0).sum()
                if plot_data["bus_sizes"].index.nlevels > 1
                else plot_data["bus_sizes"]
            )
            bus_colors = {
                bus: cmap_obj(norm(node_values[bus])) if bus in node_values.index else nan_color
                for bus in bus_sizes_agg.index
            }
            bus_sizes_to_plot = bus_sizes_agg
            add_non_choropleth_colorbar = config.get("node_heatmap", {}).get("add_colorbar", True)
            add_legend_flag = False  # suppress carrier legend for bus heatmap
    else:
        bus_colors = tech_colors
        bus_sizes_to_plot = plot_data["bus_sizes"]
        add_legend_flag = kwargs.pop("add_legend", True)

    bus_size_factor = kwargs.pop("bus_size_factor", None)
    if bus_size_factor is None:
        bus_size_factor = config.get("bus_size_factor", 1e10)

    # ── Build kwargs for plot_map ─────────────────────────────────────────────
    plot_kwargs = {**config, **kwargs}
    # Keys fully handled here – must not reach plot_map
    for _key in (
        "china_only",
        "regions_path",
        "region_values",
        "region_cmap",
        "region_vmin",
        "region_vmax",
        "add_region_colorbar",
        "region_colorbar_label",
        "line_color",
        "link_color",
    ):
        plot_kwargs.pop(_key, None)
    plot_kwargs.pop("bus_size_factor", None)
    plot_kwargs["gridlines_config"] = gridlines_config

    # Legend position overrides for choropleth mode
    if china_only:
        # Count actual carriers so ncol scales with legend length
        _bs = plot_data["bus_sizes"]
        _n_carriers = (
            len(_bs.index.get_level_values(1).unique())
            if _bs.index.nlevels > 1
            else len(_bs.index.unique())
        )
        n_cols = max(3, (_n_carriers + 2) // 3)
        plot_kwargs.setdefault(
            "carrier_legend_kw",
            {
                # axes coordinates: top of legend sits 2% below axes bottom edge
                "bbox_to_anchor": (0.5, -0.02),
                "loc": "upper center",
                "frameon": False,
                "ncol": n_cols,
            },
        )
        plot_kwargs.setdefault(
            "edge_legend_kw",
            dict(
                loc="upper right",
                bbox_to_anchor=(1.20, 0.7),
                bbox_transform=fig.transFigure,
                frameon=False,
                labelspacing=0.8,
                handletextpad=1,
            ),
        )
        plot_kwargs.setdefault(
            "bus_legend_kw",
            {
                "loc": "upper right",
                "bbox_to_anchor": (1.20, 0.5),
                "bbox_transform": fig.transFigure,
                "labelspacing": 0.8,
                "frameon": False,
                "handletextpad": 0,
            },
        )

    # ── Draw network ──────────────────────────────────────────────────────────
    logger.info("Calling plot_map (metric_type=%s, china_only=%s)", metric_type, china_only)
    ax = plot_map(
        network,
        tech_colors=tech_colors,
        line_widths=plot_data["line_widths"],
        link_widths=plot_data["link_widths"],
        bus_colors=bus_colors,
        bus_sizes=bus_sizes_to_plot / bus_size_factor,
        edge_colors=plot_data["edge_colors"],
        line_color=plot_data["line_color"],
        link_color=plot_data["link_color"],
        ax=ax,
        add_legend=add_legend_flag,
        # For choropleth, always show size legends regardless of add_legend_flag
        add_ref_edge_sizes=plot_kwargs.pop("add_ref_edge_sizes", True)
        or (china_only and choropleth_sm is not None),
        add_ref_bus_sizes=plot_kwargs.pop("add_ref_bus_sizes", True)
        or (china_only and choropleth_sm is not None),
        bus_size_factor=bus_size_factor,
        color_geomap=not china_only,
        do_tight_layout=False,
        **plot_kwargs,
    )
    # ── Apply China extent AFTER network is drawn ─────────────────────────────
    if china_only and region_shapes is not None:
        apply_china_extent(ax, region_shapes, margin=config.get("china_margin", 2.2))

    # ── Layout ────────────────────────────────────────────────────────────────
    # Layout must be finalised BEFORE placing colorbars/panel so that
    # ax.get_position() reflects the true axes bounding box.
    #
    # When add_panel=True we reserve the left side for the panel by constraining
    # tight_layout / subplots_adjust to a rect that starts at panel_reserve.
    # The panel is then drawn flush to the left edge of the axes.
    _panel_w = 0.09  # panel width in figure-fraction
    _panel_gap = 0.015  # gap between panel right-edge and axes left-edge
    _panel_reserve = _panel_w + _panel_gap + 0.04  # +0.04 for ylabel
    # Extra allowance for cartopy lat/lon gridline labels when enabled
    _label_extra = (
        0.06
        if (
            gridlines_config.get("enabled")
            and gridlines_config.get("style", {}).get("draw_labels", True)
        )
        else 0.0
    )
    _panel_reserve_labels = _panel_reserve + _label_extra

    if china_only and not add_panel:
        fig.subplots_adjust(left=0.15, right=0.85, top=0.95, bottom=0.15)
    elif china_only and add_panel:
        fig.subplots_adjust(
            left=_panel_reserve_labels + 0.03,
            right=0.84,
            top=0.95,
            bottom=0.15,
        )
    elif add_panel:
        # subplots_adjust is more reliable than tight_layout(rect=...) with cartopy.
        # We first call tight_layout to let matplotlib compute sensible top/bottom/right
        # margins, then clamp the left so the axes starts to the right of the panel.
        fig.tight_layout()
        pos = ax.get_position()
        if pos.x0 < _panel_reserve_labels:
            # Push axes right; shrink width so the right edge stays put
            new_left = _panel_reserve_labels
            new_width = pos.width - (new_left - pos.x0)
            ax.set_position([new_left, pos.y0, max(new_width, 0.4), pos.height])
    else:
        fig.tight_layout()

    # ── Metric panel (placed after layout so ax position is correct) ──────────
    if add_panel:
        preferred_order = pd.Index(opts.get("preferred_order", []))

        if metric_type == "cost":
            df = pd.DataFrame(columns=["total"])
            df["total"] = network.statistics.capex(nice_names=False).groupby(level=1).sum()
            df = df / PLOT_COST_UNITS
            ylabel = f"annualized system cost {opts.get('cost_panel', {}).get('y_axis', 'bEUR/a')}"
        elif metric_type == "energy":
            df = plot_data["metric_data"].groupby(level=1).sum().to_frame()
            df = df / PLOT_SUPPLY_UNITS
            ylabel = "Energy supply (TWh/a)"
        elif metric_type == "capacity":
            df = plot_data["metric_data"].groupby(level=1).sum().to_frame()
            df = df / PLOT_CAP_UNITS
            ylabel = "Optimal capacity (GW)"

        if metric_type in ["cost", "energy", "capacity"]:
            df = df.fillna(0)
            if df.columns[0] != "total":
                df.columns = ["total"]
            # Derive panel position from the full rendered bounding box of the
            # axes – this includes cartopy gridline lat/lon tick labels that
            # extend beyond ax.get_position().x0.
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            tight_bb = ax.get_tightbbox(renderer)  # display coords (px)
            fig_w_px, fig_h_px = fig.get_size_inches() * fig.dpi
            left_fig = tight_bb.x0 / fig_w_px  # leftmost rendered edge
            pos = ax.get_position()
            panel_left = left_fig - _panel_gap - _panel_w
            panel_bottom = pos.y0 + pos.height * 0.1
            panel_height = pos.height * 0.7
            add_metric_panel(
                fig=fig,
                df=df,
                preferred_order=preferred_order,
                tech_colors=tech_colors,
                unit=ylabel.split()[-1] if " " in ylabel else ylabel,
                ylabel=ylabel,
                ax_loc=[panel_left, panel_bottom, _panel_w, panel_height],
            )

    # ── Colorbars (placed after layout so ax position is final) ───────────────
    def _add_colorbar(sm, label):
        pos = ax.get_position()  # correct Bbox after layout
        cax = fig.add_axes([pos.x1 + 0.01, pos.y0 + 0.01 * pos.height, 0.018, 0.98 * pos.height])
        cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
        cbar.set_label(label)
        return cbar

    # Choropleth colorbar (china_only + node_values)
    if add_choropleth_colorbar and choropleth_sm is not None:
        _add_colorbar(choropleth_sm, choropleth_label)

    # Heatmap colorbar (non-china_only + node_values)
    if add_non_choropleth_colorbar and norm is not None:
        cbar_label = kwargs.get(
            "colorbar_label", config.get("node_heatmap", {}).get("colorbar_label", "Value")
        )
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        _add_colorbar(sm, cbar_label)

    if save_path:
        fig.savefig(save_path, transparent=opts.get("transparent", False), bbox_inches="tight")

    return fig, ax


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "plot_network",
            topology="current+FCG",
            co2_pathway="exp175default",
            planning_horizons="2060",
            cluster_id="IM4GA2QH2TI2XJ4",
            grade="default",
            load="mid",
        )

    config = snakemake.config
    plot_config = snakemake.params.plot_config
    is_heat_coupled = snakemake.params.heat_coupling
    
    set_plot_test_backend(snakemake.params.is_test)
    configure_logging(snakemake, logger=logger)
    set_plot_style(
        style_config_file=plot_config["network_style_config_file"],
        base_styles=["classic", "seaborn-v0_8-white"],
    )

    # Extract regions_path if available
    regions_path = getattr(snakemake.input, "regions_onshore", None)
    if regions_path:
        logger.info(f"Using clustered regions from: {regions_path}")
    else:
        logger.info("No clustered regions provided, will use admin2 shapes if china_only enabled")

    n = pypsa.Network(snakemake.input.network)
    unscale_biomass_network(n, snakemake.params.biomass_scale)

    expand_heat_carriers(n)
    sanitize_carriers(n, plot_config)

    # backward compatibility for old network files
    fix_network_names_colors(n, plot_config)

    # check the timespan
    timespan = n.snapshots.max() - n.snapshots.min()
    if not 365 <= timespan.days <= 366:
        logger.warning(
            "Network timespan is not one year, this may cause issues with the CAPEX calculation,"
            " which is referenced to the time period and not directly annualised"
        )

    # Determine which heat bus carriers are actually present in the network
    network_bus_carriers = set(n.buses.carrier.unique())
    present_heat_carriers = [
        c for c in CARRIER_HIERARCHY.get("heat", ["heat"]) if c in network_bus_carriers
    ]

    # ── Nodal prices maps (one per bus carrier) ───────────────────────────────
    nodal_price_configs = [
        ("AC", snakemake.output.cost_map.replace("cost.png", "nodal_prices.png"))
    ]
    if is_heat_coupled and present_heat_carriers:
        nodal_price_configs.append(
            (
                present_heat_carriers,
                snakemake.output.cost_map.replace("cost.png", "heat_nodal_prices.png"),
            )
        )

    for bus_carrier, out_path in nodal_price_configs:
        plot_network(
            n,
            plot_config,
            metric_type="price",
            china_only=True,
            colorbar_label=f"Nodal price [{CURRENCY}/MWh]",
            save_path=out_path,
            regions_path=regions_path,
            bus_carrier=bus_carrier,
        )

    # ── Cost map ──────────────────────────────────────────────────────────────
    cost_map_configs = [("AC", snakemake.output.cost_map)]

    for bus_carrier, out_path in cost_map_configs:
        primary = bus_carrier[0] if isinstance(bus_carrier, list) else bus_carrier
        carrier_list = filter_carriers(n, [primary])
        plot_network(
            n,
            plot_config,
            metric_type="cost",
            china_only=True,
            add_panel=True,
            components=["Generator", "Link", "StorageUnit"],
            save_path=out_path,
            regions_path=regions_path,
            carriers=carrier_list,
            bus_carrier=bus_carrier,
        )

    # ── Energy supply maps (one per bus carrier) ──────────────────────────────
    energy_supply_configs = [("AC", snakemake.output.el_supply_map)]
    if is_heat_coupled:
        energy_supply_configs.append(
            (
                present_heat_carriers,
                snakemake.output.el_supply_map.replace("el_supply.png", "heat_supply.png"),
            )
        )

    for bus_carrier, out_path in energy_supply_configs:
        primary = bus_carrier[0] if isinstance(bus_carrier, list) else bus_carrier
        carrier_list = filter_carriers(n, [primary])
        if bus_carrier == "AC":
            carrier_list = [c for c in carrier_list if c not in ("AC", "DC")]
        plot_network(
            n,
            plot_config,
            metric_type="energy",
            china_only=True,
            add_panel=True,
            components=["Generator", "Link"],
            save_path=out_path,
            regions_path=regions_path,
            bus_carrier=bus_carrier,
            carriers=carrier_list,
        )

    # ── Capacity maps (one per bus carrier) ───────────────────────────────────
    capacity_map_configs = [("AC", snakemake.output.cost_map.replace("cost.png", "capacity.png"))]
    if is_heat_coupled:
        capacity_map_configs.append(
            (
                present_heat_carriers,
                snakemake.output.cost_map.replace("cost.png", "heat_capacity.png"),
            )
        )

    for bus_carrier, out_path in capacity_map_configs:
        primary = bus_carrier[0] if isinstance(bus_carrier, list) else bus_carrier
        carrier_list = filter_carriers(n, [primary])
        plot_network(
            n,
            plot_config,
            metric_type="capacity",
            china_only=True,
            add_panel=True,
            components=["Generator", "Link", "StorageUnit"],
            save_path=out_path,
            regions_path=regions_path,
            carriers=carrier_list,
            bus_carrier=bus_carrier,
        )

    # ── Brownfield capacity map ────────────────────────────────────────────────
    p = snakemake.output.cost_map.replace("cost.png", "capacity_brownfield.png")
    network_path = snakemake.input["network"]
    brownfield_candidates = [
        network_path.replace("postnetworks", "prenetworks-brownfield"),
        network_path.replace("postnetworks", "prenetworks-sector-brownfield"),
    ]

    brownfield_n = None
    for candidate in brownfield_candidates:
        if os.path.exists(candidate):
            brownfield_n = candidate
            break

    if brownfield_n is None:
        logger.warning(
            f"Brownfield network file not found. Tried: {brownfield_candidates}. "
            "Skipping brownfield capacity map."
        )
    else:
        n_brownfield = pypsa.Network(brownfield_n)
        n_brownfield.generators.loc[:, "p_nom_opt"] = n_brownfield.generators.loc[:, "p_nom"]
        n_brownfield.storage_units.loc[:, "p_nom_opt"] = n_brownfield.storage_units.loc[:, "p_nom"]
        n_brownfield.links.loc[:, "p_nom_opt"] = n_brownfield.links.loc[:, "p_nom"]
        plot_network(
            n_brownfield,
            plot_config,
            metric_type="capacity",
            china_only=True,
            add_panel=True,
            components=["Generator", "StorageUnit"],
            save_path=p,
            regions_path=regions_path,
        )

    logger.info("Network successfully plotted")
