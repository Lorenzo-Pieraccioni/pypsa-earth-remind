"""Helper functions for pypsa network handling"""

import logging
import re
from itertools import product as _product

import geopandas as gpd
import numpy as np
import pypsa
import scipy.sparse as _sp
from pypsa.statistics import groupers as _groupers
from shapely.prepared import prep as _prep
import numpy as np
import pandas as pd
import pytz
from constants import PROV_NAMES, FOM_LINES

# get root logger
logger = logging.getLogger()

# Simplified component mapping - only essential mappings
COMPONENT_MAPPING = {
    "generator": "generators",
    "link": "links",
    "line": "lines",
    "store": "stores",
    "storageunit": "storage_units",
    "bus": "buses",
    "globalconstraint": "global_constraints",
    "load": "loads",
    "transformer": "transformers",
}

YEAR_HRS = 8760


def aggregate_costs(
    n: pypsa.Network,
    flatten=False,
    opts: dict | None = None,
    existing_only=False,
) -> pd.Series | pd.DataFrame:
    """LEGACY FUNCTION used in pypsa heating plots - unclear what it does

    Args:
        n (pypsa.Network): the network object
        flatten (bool, optional):merge capex and marginal ? Defaults to False.
        opts (dict, optional): options for the function. Defaults to None.
        existing_only (bool, optional): use _nom instead of nom_opt. Defaults to False.
    """

    components = dict(
        Link=("p_nom", "p0"),
        Generator=("p_nom", "p"),
        StorageUnit=("p_nom", "p"),
        Store=("e_nom", "p"),
        Line=("s_nom", None),
        Transformer=("s_nom", None),
    )

    costs = {}
    for c, (p_nom, p_attr) in zip(
        n.iterate_components(components.keys(), skip_empty=True), components.values()
    ):
        if not existing_only:
            p_nom += "_opt"
        costs[(c.list_name, "capital")] = (
            (c.df[p_nom] * c.df.capital_cost).groupby(c.df.carrier).sum()
        )
        if p_attr is not None:
            p = c.dynamic[p_attr].sum()
            if c.name == "StorageUnit":
                p = p.loc[p > 0]
            costs[(c.list_name, "marginal")] = (p * c.df.marginal_cost).groupby(c.df.carrier).sum()
    costs = pd.concat(costs)

    if flatten:
        assert opts is not None
        conv_techs = opts["conv_techs"]

        costs = costs.reset_index(level=0, drop=True)
        costs = costs["capital"].add(
            costs["marginal"].rename({t: t + " marginal" for t in conv_techs}),
            fill_value=0.0,
        )

    return costs


def check_heat_supply_attached(n: pypsa.Network) -> None:
    """Check that all heat supply components are attached to heat buses.

    Args:
        n (pypsa.Network): the pypsa network object
    """
    buses = n.loads_t.p_set.sum().index
    heat_mask = buses.map(n.buses.carrier)
    heat_buses = buses[heat_mask == "heat"]

    missing = heat_buses.difference(n.generators.bus)
    missing = missing.difference(n.links.bus0).difference(n.links.bus1)
    missing = missing.difference(n.storage_units.bus)

    if not missing.empty:
        raise ValueError(f"Not heat supply heat loads at buses: {missing.tolist()}")


def compute_line_utilisation(n: pypsa.Network) -> pd.Series:
    """Compute time-mean utilisation (0..1) of each AC Line.

    Utilisation is the snapshot-weighted mean absolute flow divided by the
    usable capacity::

        util = sum_t(|p0| * w_t) / (s_nom_opt * s_max_pu * sum_t(w_t))

    i.e. a line capacity factor on the optimised rating. Lines with zero
    optimised capacity (or no flow) map to 0; the result is clipped to 1.

    Args:
        n: Solved PyPSA network.

    Returns:
        Series indexed by line name, values in [0, 1].
    """
    if n.lines.empty or n.lines_t.p0.empty:
        return pd.Series(dtype=float, index=n.lines.index)
    w = n.snapshot_weightings.objective
    s_max_pu = (
        pd.to_numeric(n.lines.get("s_max_pu", pd.Series(1.0, index=n.lines.index)), errors="coerce")
        .fillna(1.0)
        .replace(0.0, np.nan)
    )
    cap = pd.to_numeric(n.lines["s_nom_opt"], errors="coerce").replace(0.0, np.nan)
    energy = n.lines_t.p0.abs().mul(w, axis=0).sum()          # MWh
    util = energy / (cap * s_max_pu * float(w.sum()))
    return util.clip(upper=1.0).fillna(0.0)


def compute_transmission_utilisation(n: pypsa.Network) -> dict:
    """Compute time-mean utilisation (0..1) of transmission Lines and Links.

    Generalises `compute_line_utilisation` to cover every backend:

    - OSM AC backbone on ``n.lines`` (via `compute_line_utilisation`);
    - transmission ``n.links`` (carrier AC or DC) — manual-backend AC links and
      all HVDC corridors.

    The lossy positive+reversed split builds two independent directional legs,
    so utilisation is computed per leg — ``energy / (p_nom_opt * p_max_pu *
    hours)`` — and the corridor takes the MAX over its legs (the busier
    direction), assigned to the real (non-``reversed``) leg. Reversed legs and
    non-transmission (sector) links map to NaN so callers can hide them.

    Args:
        n: Solved PyPSA network.

    Returns:
        Dict ``{"lines": Series, "links": Series}``. ``lines`` values in [0, 1]
        indexed by line; ``links`` indexed by every link, with utilisation on
        real transmission legs and NaN elsewhere.
    """
    lines_u = compute_line_utilisation(n)
    # maybe not completely correct since one side will refer to p0 and other to p1?
    tr_links = n.statistics.supply(comps="Link", carrier = ["AC","DC"], groupby=["name"])
    cap_lnks = n.statistics.optimal_capacity(comps="Link", carrier = ["AC","DC"], groupby=["name"]).abs()
    hours = float(n.snapshot_weightings.objective.sum())
    max_use = cap_lnks.loc[tr_links.index].replace(0.0, np.nan) * hours * n.links.loc[tr_links.index, "p_max_pu"]
    
    utilisation = (tr_links/max_use).to_frame()
    utilisation["group"] = utilisation.index.str.replace("reversed", "positive")
    utilisation = utilisation.groupby("group").sum()[0].rename_axis("name")

    return {"lines": lines_u, "links": utilisation}


# TODO fix timezones/centralsie, think Shanghai won't work on its own
def generate_periodic_profiles(
    dt_index=None,
    col_tzs=pd.Series(index=PROV_NAMES, data=len(PROV_NAMES) * ["Shanghai"]),
    weekly_profile=range(24 * 7),
):
    """Generate weekly hourly profiles for each province, from pypsa-eur workflow.

    Args:
        dt_index: Time index for the profiles
        col_tzs: Time zones for each province
        weekly_profile: 168-hour weekly profile pattern

    Returns:
        pd.DataFrame: Weekly profiles for each province
    """

    weekly_profile = pd.Series(weekly_profile, range(24 * 7))
    # TODO fix, no longer take into accoutn summer time
    # ALSO ADD A TODO in base_network
    week_df = pd.DataFrame(index=dt_index, columns=col_tzs.index)
    for ct in col_tzs.index:
        week_df[ct] = [24 * dt.weekday() + dt.hour for dt in dt_index.tz_localize(None)]
        week_df[ct] = week_df[ct].map(weekly_profile)
    return week_df


def add_missing_carriers(n: pypsa.Network, carriers: list | set) -> None:
    """Function to add missing carriers to the network without raising errors.

    Args:
        n (pypsa.Network): the pypsa network object
        carriers (list | set): a list of carriers that should be included
    """
    missing_carriers = set(carriers) - set(n.carriers.index)
    if len(missing_carriers) > 0:
        n.add("Carrier", missing_carriers)


def assign_locations(n: pypsa.Network):
    """Assign location based on the node location

    Args:
        n (pypsa.Network): the pypsa network object
    """
    # AC buses
    ac_buses = n.buses.query("carrier=='AC'").index
    n.buses.loc[ac_buses, "location"] = n.buses.loc[ac_buses].index
    for c in n.iterate_components(n.one_port_components):
        c.df["location"] = c.df.bus.map(n.buses.location)

    for c in n.iterate_components(n.branch_components):
        # use bus1 and bus2
        c.df["_loc1"] = c.df.bus0.map(n.buses.location)
        c.df["_loc2"] = c.df.bus1.map(n.buses.location)
        # if only one of buses is in the ntwk node list, make it a loop to the location
        c.df["_loc2"] = c.df.apply(lambda row: row._loc1 if row._loc2 == "" else row._loc2, axis=1)
        c.df["_loc1"] = c.df.apply(lambda row: row._loc2 if row._loc1 == "" else row._loc1, axis=1)
        # add location to loops. Links between nodes have ambiguos location
        c.df["location"] = c.df.apply(
            lambda row: row._loc1 if row._loc1 == row._loc2 else "", axis=1
        )
        c.df.drop(columns=["_loc1", "_loc2"], inplace=True)


def aggregate_p(n: pypsa.Network) -> pd.Series:
    """Make a single series for generators, storage units, loads, and stores power,
    summed over all carriers

    Args:
        n (pypsa.Network): the network object

    Returns:
        pd.Series: the aggregated p data
    """
    return pd.concat(
        [
            n.generators_t.p.sum().groupby(n.generators.carrier).sum(),
            n.storage_units_t.p.sum().groupby(n.storage_units.carrier).sum(),
            n.stores_t.p.sum().groupby(n.stores.carrier).sum(),
            -n.loads_t.p.sum().groupby(n.loads.carrier).sum(),
        ]
    )


def calc_generation_share(df, n, carrier):
    """
    Add generation share column to an existing DataFrame.
    Assumes df has carrier index already.

    Returns: DataFrame with an added 'GenShare' column.
    """
    supply_data = n.statistics.supply(bus_carrier=carrier, comps="Generator")
    total_supply = supply_data.sum()
    gen_shares = (supply_data / total_supply * 100).dropna()
    carrier_map = {c.lower(): row["nice_name"] for c, row in n.carriers.iterrows()}
    gen_shares.index = gen_shares.index.map(lambda idx: carrier_map.get(idx.lower(), idx))
    df = df.copy()
    df["GenShare"] = gen_shares
    return df


def calc_atlite_heating_timeshift(date_range: pd.date_range, use_last_ts=False) -> int:
    """Imperfect function to calculate the heating time shift for atlite
    Atlite is in xarray, which does not have timezone handling. Adapting the UTC ERA5 data
    to the network local time, is therefore limited to a single shift, which is based on the first
    entry of the time range. For a whole year, in the northern Hemisphere -> winter

    Args:
        date_range (pd.date_range): the date range for which the shift is calc
        use_last_ts (bool, optional): use last instead of first. Defaults to False.

    Returns:
        int: a single timezone shift to utc in hours
    """
    # import constants here to not interfere with snakemake
    from constants import TIMEZONE

    idx = 0 if not use_last_ts else -1
    return pytz.timezone(TIMEZONE).utcoffset(date_range[idx]).total_seconds() / 3600


def calc_lcoe(
    n: pypsa.Network,
    grouper=["carrier", "bus_carrier"],
    carriers=["AC", "heat"],  # noqa: F
    **kwargs,
) -> pd.DataFrame:
    """Calculate the LCOE for the network: (capex+opex)/supply.

    Args:
        n (pypsa.Network): the network for which LCOE is to be calaculated
        grouper (function | list, optional): function to group the data in network.statistics.
                Overwritten if groupby is passed in kwargs.
                Defaults to ["carrier", "bus_carrier"].
        carriers (list, optional): list of carriers to include for generators. Defaults to ["AC", "heat"].
        **kwargs: other arguments to be passed to network.statistics
    Returns:
        pd.DataFrame: The LCOE for the network with or without brownfield CAPEX, MV and delta

    """
    if "groupby" in kwargs:
        grouper = kwargs.pop("groupby")

    # store marginal costs we will manipulate to merge fuel costs
    original_marginal_costs = n.links.marginal_cost.copy()
    for carr in ["coal", "gas", "coal ccs"]:
        fueled_links = n.links.query(f"carrier.str.contains('{carr}', case=False)").index
        suffix = " fuel" if carr == "gas" else ""
        fuel_costs = n.generators.loc[
            n.links.loc[fueled_links, "bus0"] + suffix
        ].marginal_cost.values
        # eta is applied by statistics
        n.links.loc[fueled_links, "marginal_cost"] += fuel_costs
    # TODO same with BECCS? & other links?
    rev = n.statistics.revenue(groupby=grouper, **kwargs)
    opex = n.statistics.opex(groupby=grouper, **kwargs)
    capex = n.statistics.expanded_capex(groupby=grouper, **kwargs)
    tot_capex = n.statistics.capex(groupby=grouper, **kwargs)
    supply = n.statistics.supply(groupby=grouper, **kwargs)
    # restore original marginal costs
    n.links.marginal_cost = original_marginal_costs

    # incase no grouper was specified, get different levels
    if grouper is None:
        supply = supply.groupby(level=[0, 1]).sum()

    outputs = pd.concat(
        [opex, capex, tot_capex, rev, supply],
        axis=1,
        keys=["OPEX", "CAPEX", "CAPEX_wBROWN", "Revenue", "supply"],
    ).fillna(0)

    # remove generators that are not of interest (eg. coal fuel)
    if "bus_carrier" in outputs.index.names and "carrier" in outputs.index.names:
        outputs = outputs.query(
            "(component == 'Generator' and bus_carrier in @carriers) or component!='Generator'"
        )
        outputs = outputs.sort_index()
        # in case of links costs can be assigned to a different bus_carrier
        outputs["supp_frac"] = (
            outputs.groupby(["component", "carrier"])
            .apply(lambda x: x.supply / x.supply.sum())
            .values
        )
        outputs.fillna({"supp_frac": 0}, inplace=True)
        outputs["cost"] = (
            outputs.groupby(["component", "carrier"])
            .apply(lambda x: (x.CAPEX + x.OPEX).sum() * x.supp_frac)
            .values
        )
        outputs["cost_w_brownfield"] = (
            outputs.groupby(["component", "carrier"])
            .apply(lambda x: (x.CAPEX_wBROWN + x.OPEX).sum() * x.supp_frac)
            .values
        )
        outputs = outputs[outputs.supp_frac != 0]
    else:
        outputs["cost"] = outputs["CAPEX"] + outputs["OPEX"]
        outputs["cost_w_brownfield"] = outputs["CAPEX_wBROWN"] + outputs["OPEX"]

    outputs["LCOE"] = outputs["cost"] / (outputs["supply"])
    outputs["LCOE_wbrownfield"] = outputs["cost_w_brownfield"] / (outputs["supply"])
    outputs["rev-costs"] = outputs.apply(lambda row: row.Revenue - row.cost, axis=1)
    outputs["MV"] = outputs.apply(lambda row: row.Revenue / row.supply, axis=1)
    outputs["profit_pu"] = outputs["rev-costs"] / outputs.supply
    outputs.sort_values("profit_pu", ascending=False, inplace=True)

    return outputs[outputs.supply > 0]


def unscale_biomass_network(n: pypsa.Network, biomass_scale: float) -> None:
    """Reverse the biomass LP-numerics scaling transformation in post-processing.

    ``prepare_network.add_biomass()`` and ``add_heat_sector.add_biomass_heat()``
    divide biomass Store ``e_nom``/``e_initial`` by ``biomass_scale`` and multiply
    all downstream Link ``efficiency*`` and ``*_cost`` by the same factor to
    reduce the LP coefficient range.  This function restores physical units in the
    in-memory network so that all post-processing statistics, prices, and energy
    time-series are correct.

    Only the in-memory ``pypsa.Network`` object is modified; on-disk ``.nc`` files
    are **not** touched.  The function is idempotent: repeated calls are no-ops.
    When ``biomass_scale == 1.0`` the function is a no-op.

    What is unscaled:
    - ``n.stores``: ``e_nom``, ``e_initial``, ``e_nom_opt``          (× biomass_scale)
    - ``n.links``:  ``efficiency``, ``efficiency2``, ``efficiency3``,
                    ``capital_cost``, ``marginal_cost``               (÷ biomass_scale)
    - ``n.stores_t.e``, ``n.stores_t.p``                             (× biomass_scale)
    - ``n.stores_t.mu_upper``, ``n.stores_t.mu_lower``               (÷ biomass_scale)
    - ``n.links_t.p0`` for biomass links                             (× biomass_scale)
    - ``n.buses_t.marginal_price`` for biomass buses                 (÷ biomass_scale)

    Note: ``links_t.p1 / p2 / p3`` (electricity, heat, CO2 output streams) are
    already in physical units and are **not** modified.

    Args:
        n (pypsa.Network): Solved network that was built with biomass_scale applied.
        biomass_scale (float): The scale factor used during network build.
    """
    if getattr(n, "_biomass_unscaled", False):
        logger.debug("unscale_biomass_network: already unscaled, skipping")
        return

    if biomass_scale == 1.0:
        n._biomass_unscaled = True
        return

    # ── identify components ──────────────────────────────────────────────────
    biomass_stores = n.stores[
        (n.stores.carrier == "biomass") | (n.stores.bus.str.endswith(" biomass"))
    ].index
    biomass_links = n.links[n.links.bus0.str.endswith(" biomass")].index
    biomass_buses = n.buses[n.buses.carrier == "biomass"].index

    logger.info(
        f"unscale_biomass_network: {len(biomass_stores)} stores, "
        f"{len(biomass_links)} links, {len(biomass_buses)} buses"
    )

    # ── static store attributes ──────────────────────────────────────────────
    for col in ["e_nom", "e_initial", "e_nom_opt"]:
        if col in n.stores.columns and not biomass_stores.empty:
            n.stores.loc[biomass_stores, col] = (
                n.stores.loc[biomass_stores, col] * biomass_scale
            )

    # ── static link attributes ───────────────────────────────────────────────
    for col in ["efficiency", "efficiency2", "efficiency3", "capital_cost", "marginal_cost"]:
        if col in n.links.columns and not biomass_links.empty:
            n.links.loc[biomass_links, col] = (
                n.links.loc[biomass_links, col] / biomass_scale
            )

    # ── time-series: store energy and power (× scale) ────────────────────────
    for ts_attr in ["e", "p"]:
        ts = getattr(n.stores_t, ts_attr, None)
        if ts is not None and not ts.empty:
            cols = biomass_stores.intersection(ts.columns)
            if not cols.empty:
                n.stores_t[ts_attr][cols] *= biomass_scale

    # ── time-series: store duals (÷ scale) ────────────────────────────────────
    for ts_attr in ["mu_upper", "mu_lower"]:
        ts = getattr(n.stores_t, ts_attr, None)
        if ts is not None and not ts.empty:
            cols = biomass_stores.intersection(ts.columns)
            if not cols.empty:
                n.stores_t[ts_attr][cols] /= biomass_scale

    # ── time-series: link biomass input flow (× scale) ───────────────────────
    # p0 is the flow drawn from the biomass bus; it is in LP-scale units
    # (1 LP unit = biomass_scale MWh_th) so multiply to get physical MW_th.
    # p1/p2/p3 are already in physical output units – do NOT touch them.
    if not n.links_t.p0.empty and not biomass_links.empty:
        cols = biomass_links.intersection(n.links_t.p0.columns)
        if not cols.empty:
            n.links_t.p0[cols] *= biomass_scale

    # ── bus marginal prices (÷ scale) ─────────────────────────────────────────
    if not n.buses_t.marginal_price.empty and not biomass_buses.empty:
        cols = biomass_buses.intersection(n.buses_t.marginal_price.columns)
        if not cols.empty:
            n.buses_t.marginal_price[cols] /= biomass_scale

    n._biomass_unscaled = True
    logger.info("unscale_biomass_network: done")


def calc_link_length(row, network: pypsa.Network) -> float:
    """Calculate the share of a link in the total link capacity
    for links connecting the same bus pair

    Args:
        row (pd.Series): the link row
        network (pypsa.Network): the network object
    """
    from functions import haversine

    bus0 = row["bus0"]
    bus1 = row["bus1"]
    x0, y0 = network.buses.loc[bus0, ["x", "y"]]
    x1, y1 = network.buses.loc[bus1, ["x", "y"]]
    return haversine([x0, y0], [x1, y1])


def determine_simulation_timespan(config: dict, year: int) -> int:
    """Determine the simulation timespan in years (so the network object is not needed)

    Args:
        config (dict): the snakemake config
        year (int): the year to simulate
    Returns:
        int: the simulation timespan in years
    """

    # make snapshots (drop leap days) -> possibly do all the unpacking in the function
    snapshot_cfg = config["snapshots"]
    snapshots = make_periodic_snapshots(
        year=year,
        freq=snapshot_cfg["freq"],
        start_day_hour=snapshot_cfg["start"],
        end_day_hour=snapshot_cfg["end"],
        bounds=snapshot_cfg["bounds"],
        # naive local timezone
        tz=None,
        end_year=None if not snapshot_cfg["end_year_plus1"] else year + 1,
    )

    # load costs
    n_years = config["snapshots"]["frequency"] * len(snapshots) / YEAR_HRS

    return n_years


def filter_carriers(n: pypsa.Network, bus_carriers=["AC"], comps=["Generator", "Link"]) -> list:
    """Filter carriers for links that attach to a bus of the target carrier

    Args:
        n (pypsa.Network): the pypsa network object
        bus_carrier (str | list, optional): the bus carrier. Defaults to "AC".
        comps (list, optional): the components to check. Defaults to ["Generator", "Link"].

    Returns:
        list: list of carriers that are attached to the bus carrier
    """
    if isinstance(bus_carriers, str):
        bus_carriers = [bus_carriers]

    carriers = []
    for c in comps:
        comp = n.static(c)
        ports = [c for c in comp.columns if c.startswith("bus")]
        comp_df = comp[ports + ["carrier"]]
        is_attached = (
            comp_df[ports].apply(lambda x: x.map(n.buses.carrier).isin(bus_carriers)).T.any()
        )
        carriers += comp_df.loc[is_attached].carrier.unique().tolist()

    if bus_carriers not in carriers:
        carriers += bus_carriers
    return carriers


def get_location_and_carrier(
    n: pypsa.Network, c: str, port: str = "", nice_names: bool = True
) -> list[pd.Series]:
    """Get component location and carrier.

    Args:
        n (pypsa.Network): the network object
        c (str): component name
        port (str, optional): port name. Defaults to "".
        nice_names (bool, optional): use nice names. Defaults to True.

    Returns:
        list[pd.Series]: list of location and carrier series
    """

    location = _groupers.location(n, c, port=port)
    carrier = _groupers.carrier(n, c, nice_names=nice_names)
    return [location, carrier]


# TODO is thsi really good? useful?
# TODO make a standard apply/str op instead ofmap in add_electricity.sanitize_carriers
def rename_techs(label: str, nice_names: dict | pd.Series | None = None) -> str:
    """Rename technology labels for better readability. Removes some prefixes
        and renames if certain conditions  defined in function body are met.

    Args:
        label (str): original technology label
        nice_names (dict, optional): nice names that will overwrite defaults

    Returns:
        str: renamed tech label
    """

    prefix_to_remove = [
        "residential ",
        "services ",
        "urban ",
        "rural ",
        "central ",
        "decentral ",
    ]

    rename_if_contains = [
        "CHP",
        "gas boiler",
        "biogas",
        "solar thermal",
        "air heat pump",
        "ground heat pump",
        "resistive heater",
        "Fischer-Tropsch",
    ]

    rename_if_contains_dict = {
        "water tanks": "hot water storage",
        "retrofitting": "building retrofitting",
        # "H2 Electrolysis": "hydrogen storage",
        # "H2 Fuel Cell": "hydrogen storage",
        # "H2 pipeline": "hydrogen storage",
        "battery": "battery storage",
        "H2 for industry": "H2 for industry",
        "land transport fuel cell": "land transport fuel cell",
        "land transport oil": "land transport oil",
        "oil shipping": "shipping oil",
        # "CC": "CC"
    }

    for ptr in prefix_to_remove:
        if label[: len(ptr)] == ptr:
            label = label[len(ptr) :]

    for rif in rename_if_contains:
        if rif in label:
            label = rif

    for old, new in rename_if_contains_dict.items():
        if old in label:
            label = new
    # import here to not mess with snakemake
    from constants import NICE_NAMES_DEFAULT

    names_new = NICE_NAMES_DEFAULT.copy()
    names_new.update(nice_names)
    for old, new in names_new.items():
        if old == label:
            label = new
    return label


def is_leap_year(year: int) -> bool:
    """Determine whether a year is a leap year.

    Args:
        year (int): the year
    Returns:
        bool: True if leap year, False otherwise
    """
    year = int(year)
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def mock_solve(n: pypsa.Network) -> pypsa.Network:
    """Mock the solving step for tests

    Args:
        n (pypsa.Network): the network object
    """
    for c in n.iterate_components(components=["Generator", "Link", "Store", "LineType"]):
        opt_cols = [col for col in c.df.columns if col.endswith("opt")]
        base_cols = [col.split("_opt")[0] for col in opt_cols]
        c.df[opt_cols] = c.df[base_cols]
    return n


def make_periodic_snapshots(
    year: int,
    freq: int,
    start_day_hour="01-01 00:00:00",
    end_day_hour="12-31 23:00",
    bounds="both",
    end_year: int = None,
    tz: str = None,
) -> pd.date_range:
    """Centralised function to make regular snapshots.
    REMOVES LEAP DAYS

    Args:
        year (int): start time stamp year (end year if end_year None)
        freq (int): snapshot frequency in hours
        start_day_hour (str, optional): Day and hour. Defaults to "01-01 00:00:00".
        end_day_hour (str, optional): _description_. Defaults to "12-31 23:00".
        bounds (str, optional):  bounds behaviour (pd.data_range) . Defaults to "both".
        tz (str, optional): timezone (UTC, None or a timezone). Defaults to None (naive).
        end_year (int, optional): end time stamp year. Defaults to None (use year).

    Returns:
        pd.date_range: the snapshots for the network
    """
    if not end_year:
        end_year = year

    # do not apply freq yet or get inconsistencies with leap years
    snapshots = pd.date_range(
        f"{int(year)}-{start_day_hour}",
        f"{int(end_year)}-{end_day_hour}",
        freq="1h",
        inclusive=bounds,
        tz=tz,
    )
    if is_leap_year(int(year)):
        snapshots = snapshots[~((snapshots.month == 2) & (snapshots.day == 29))]
    freq_hours = int("".join(filter(str.isdigit, str(freq))))

    return snapshots[::freq_hours]  # every freq hour


def shift_profile_to_planning_year(data: pd.DataFrame, planning_yr: int | str) -> pd.DataFrame:
    """Shift the profile to the planning year - this harmonises weather and network timestamps
       which is needed for pandas loc operations
    Args:
        data (pd.DataFrame): profile data, for 1 year
        planning_yr (int): planning year
    Returns:
        pd.DataFrame: shifted profile data
    Raises:
        ValueError: if the profile data crosses years
    """

    years = data.index.year.unique()
    if not len(years) == 1:
        raise ValueError(f"Data should be for one year only but got {years}")

    ref_year = years[0]
    # remove all planning year leap days
    if is_leap_year(ref_year):  # and not is_leap_year(planning_yr):
        data = data.loc[~((data.index.month == 2) & (data.index.day == 29))]

    # TODO CONSIDER CHANGING METHOD TO REINDEX inex = daterange w new year method = FORWARDFILL
    data.index = data.index.map(lambda t: t.replace(year=int(planning_yr)))

    return data


def select_province(n, province_name):
    """
    REMOVE all provinces except the selected one.

    Args:
        n (pypsa.Network): The PyPSA network instance.
        province_name (str): The name of the province to select.
    """

    n.generators.province = n.generators.bus.map(n.buses.province)
    n.stores["province"] = n.stores.bus.map(n.buses.province)
    n.storage_units["province"] = n.storage_units.bus.map(n.buses.province)
    n.links["province_0"] = n.links.bus0.map(n.buses.province)
    n.links["province_1"] = n.links.bus1.map(n.buses.province)
    n.links["province"] = n.links.apply(
        lambda x: x.province_0 if x.province_0 == x.province_1 else "", axis=1
    )
    n.loads["province"] = n.loads.bus.map(n.buses.province)

    n.stores = n.stores[n.stores["province"] == province_name]
    n.links = n.links[n.links["province"] == province_name]
    n.generators = n.generators[n.generators["province"] == province_name]
    n.storage_units = n.storage_units[n.storage_units["province"] == province_name]
    n.loads = n.loads[n.loads["province"] == province_name]
    n.buses = n.buses[n.buses["province"] == province_name]
    for k in n.generators_t.keys():
        cols = [c for c in n.generators_t[k] if c in n.generators.index]
        n.generators_t[k] = n.generators_t[k][cols]
    for k in n.storage_units_t.keys():
        cols = [c for c in n.storage_units_t[k] if c in n.storage_units.index]
        n.storage_units_t[k] = n.storage_units_t[k][cols]
    for k in n.links_t.keys():
        cols = [c for c in n.links_t[k] if c in n.links.index]
        n.links_t[k] = n.links_t[k][cols]
    for k in n.loads_t.keys():
        cols = [c for c in n.loads_t[k] if c in n.loads.index]
        n.loads_t[k] = n.loads_t[k][cols]
    for k in n.stores_t.keys():
        cols = [c for c in n.stores_t[k] if c in n.stores.index]
        n.stores_t[k] = n.stores_t[k][cols]


def _derive_endpoint_l2(n: pypsa.Network, bus_col: pd.Series) -> pd.Series:
    """Map a branch endpoint to its L2 (prefecture) identity.

    Use the bus ``prefecture`` attribute when available (admin-2 name, stable
    across OSM rebuilds and clustering); fall back to the bus name where
    ``prefecture`` is missing (e.g. the manual backend, whose buses are the
    province/node themselves).

    Args:
        n: PyPSA network providing ``n.buses``.
        bus_col: Series of bus names (a branch ``bus0`` or ``bus1`` column).

    Returns:
        Series of L2 identities aligned to ``bus_col``.
    """
    if "prefecture" in n.buses.columns:
        l2 = bus_col.map(n.buses["prefecture"])
    else:
        l2 = pd.Series(np.nan, index=bus_col.index)
    # fall back to the bus name itself where no prefecture is recorded
    return l2.where(l2.notna(), bus_col)


def _derive_hexi_region_mask(
    prov: pd.Series,
    l2: pd.Series,
    region_map: dict,
) -> pd.Series:
    """Flag endpoints that fall in a nested ``province -> [L2 nodes]`` region map.

    ``region_map`` maps a province to either an empty value (None / empty list,
    meaning the *whole* province matches) or a list of L2 (prefecture/node)
    names (meaning only those nodes within the province match). The same format
    drives both the corridor origin side (``west_origins``) and the exempt
    destinations (``exempt_destinations``).

    Args:
        prov: province of each branch endpoint.
        l2: L2 identity of each branch endpoint.
        region_map: province -> sublevel L2 list (or empty for whole province).

    Returns:
        Boolean Series, True where the endpoint matches the region map.
    """
    mask = pd.Series(False, index=prov.index)
    for province, subs in region_map.items():
        in_prov = prov == province
        if not subs:  # None or [] -> match the whole province
            mask |= in_prov
        else:
            mask |= in_prov & l2.isin(set(subs))
    return mask


def _find_hexi_corridor_in_component(
    n: pypsa.Network,
    df: pd.DataFrame,
    origin_map: dict,
    exempt_map: dict,
    min_voltage_kv: float | None,
) -> pd.Index:
    """Select Hexi-corridor crossings within one branch component.

    A branch qualifies when exactly one endpoint is an origin (per
    ``origin_map``) and the *other* (eastern) endpoint is NOT exempt (per
    ``exempt_map``), optionally filtered to UHV by ``min_voltage_kv``. Both maps
    use the nested ``province -> [L2 nodes]`` format, so the origin side can mix
    whole provinces (e.g. Xinjiang) with specific L2 nodes (e.g. the Hexi-strip
    prefectures of Gansu), catching corridor branches that start in Gansu or
    that break into Gansu segments.

    Args:
        n: PyPSA network providing ``n.buses``.
        df: ``n.lines`` or ``n.links`` (must carry ``bus0``/``bus1``).
        origin_map: province -> L2 sublevel list (or empty for whole province)
            forming the corridor's western (origin) side.
        exempt_map: province -> L2 sublevel list (or empty for whole province);
            eastern endpoints matching this are NOT corridor-limited.
        min_voltage_kv: drop branches below this voltage; ``None`` disables.

    Returns:
        Index of qualifying branch names (empty if ``df`` is empty).
    """
    if df.empty:
        return df.index[:0]

    prov0 = df.bus0.map(n.buses.province)
    prov1 = df.bus1.map(n.buses.province)
    l2_0 = _derive_endpoint_l2(n, df.bus0)
    l2_1 = _derive_endpoint_l2(n, df.bus1)

    origin0 = _derive_hexi_region_mask(prov0, l2_0, origin_map)
    origin1 = _derive_hexi_region_mask(prov1, l2_1, origin_map)
    # XOR: exactly one endpoint is a corridor origin
    origin_ok = origin0 ^ origin1
    # the eastern endpoint is whichever one is NOT the origin side
    eastern_prov = prov1.where(origin0, prov0)
    eastern_l2 = l2_1.where(origin0, l2_0)
    # corridor-limited unless the eastern endpoint is exempt
    dest_ok = ~_derive_hexi_region_mask(eastern_prov, eastern_l2, exempt_map)

    mask = origin_ok & dest_ok

    if min_voltage_kv is not None:
        v = df.get("v_nom_original")
        if v is None:
            v = df.get("v_nom")
        if v is not None:
            v = pd.to_numeric(v, errors="coerce").fillna(0)
            mask &= v >= min_voltage_kv

    return df.index[mask]


def _expand_region_map_for_clustering(region_map: dict, node_splits: dict) -> dict:
    """Rewrite a prefecture-keyed region map into post-cluster node identities.

    The Hexi-corridor and terrain-markup region maps are consumed on the
    L2-clustered network, where buses have lost their admin-2 ``prefecture``
    attribute (dropped in ``cluster_network_l2``) and are identified only by
    their cluster-node name: ``Province`` for an unsplit province,
    ``Province_Split`` for a split one (see
    ``cluster_network_l2.build_busmap_from_config``). A region map written with
    prefecture names (e.g. ``{Gansu: [Jiuquan, Zhangye]}``) therefore matches no
    cluster node such as ``Gansu_West``. For each province whose sub-list is a
    prefecture set, add the cluster-node names whose split prefecture list
    intersects the sub-list, keeping the original prefecture names too so
    prefecture-bearing backends still match.

    Args:
        region_map: province -> prefecture sub-list (or empty = whole province).
        node_splits: ``nodes.splits`` config (province -> {split: [prefectures]}).

    Returns:
        Region map with each prefecture sub-list augmented by the matching
        cluster-node names. Returned unchanged when ``node_splits`` is empty.
    """
    if not node_splits:
        return region_map
    expanded: dict = {}
    for province, subs in region_map.items():
        if not subs:
            expanded[province] = subs  # whole-province match (province survives)
            continue
        sub_set = set(subs)
        prov_splits = node_splits.get(province) or {}
        if prov_splits:
            nodes = [
                f"{province}_{split}"
                for split, prefs in prov_splits.items()
                if sub_set & set(prefs)
            ]
            missing = sub_set - {p for prefs in prov_splits.values() for p in prefs}
            if missing:
                logger.warning(
                    "Region map: prefectures %s for province %r are in no "
                    "split cluster and will not match any node.",
                    sorted(missing), province,
                )
            expanded[province] = sorted(sub_set | set(nodes))
        else:
            # Province is not split -> the only node is the whole province, so a
            # prefecture sub-list cannot be honoured at sub-province granularity.
            # Match the whole-province node (fail-safe: over-apply rather than
            # silently miss the region), keeping prefectures for backends that
            # still carry them.
            logger.warning(
                "Region map: province %r is not split but lists prefectures "
                "%s; matching the whole-province node (cannot sub-select after "
                "clustering).", province, sorted(sub_set),
            )
            expanded[province] = sorted(sub_set | {province})
    return expanded


def find_hexi_corridor_branches(
    n: pypsa.Network, hexi_cfg: dict, node_splits: dict | None = None
) -> dict:
    """Find the AC Lines and DC/AC Links that cross the Hexi corridor.

    The Hexi corridor (the narrow Gansu strip) is the sole land bridge between
    Xinjiang and eastern China, so almost every Xinjiang export funnels through
    it. A branch is marked as a corridor crossing iff exactly one endpoint is a
    corridor origin (per ``west_origins``) AND the other (eastern) endpoint is
    NOT exempt (per ``exempt_destinations``). Both are nested mappings
    ``province -> [L2 nodes]`` where an empty value matches the whole province,
    so the origin side can mix whole provinces (Xinjiang) with specific Gansu
    Hexi-strip L2 nodes (catching corridor branches that start in Gansu or that
    break into Gansu segments). Exemptions cover regions reachable from Xinjiang
    WITHOUT the corridor (e.g. Qinghai/Tibet via the southern plateau).
    Optionally filtered to UHV by ``min_voltage_kv``. The ``include`` /
    ``exclude`` name lists provide a final manual override.

    This is fail-safe: an export to an unlisted region is limited by default
    rather than silently escaping the constraint.

    Args:
        n: PyPSA network with ``province`` (and ideally ``prefecture``) on
            ``n.buses``.
        hexi_cfg: ``electricity.lines.hexi_corridor`` config block. Reads
            ``west_origins`` (falls back to legacy ``west_provinces``),
            ``exempt_destinations``, ``min_voltage_kv``, ``include``,
            ``exclude``.
        node_splits: ``nodes.splits`` config (province -> {split: [prefectures]}).
            Used to translate prefecture-keyed origin/exempt maps into the
            split-node names the clustered network actually uses. ``None`` skips
            the translation (correct for unsplit/manual backends).

    Returns:
        Dict with ``"lines"`` and ``"links"`` Index objects of corridor branch
        names.
    """
    origin_map = hexi_cfg.get("west_origins")
    if origin_map is None:
        # legacy flat list of origin provinces -> whole-province map
        origin_map = {prov: None for prov in hexi_cfg.get("west_provinces", ["Xinjiang"])}
    min_voltage_kv = hexi_cfg.get("min_voltage_kv")
    include = set(hexi_cfg.get("include", []) or [])
    exclude = set(hexi_cfg.get("exclude", []) or [])
    exempt_map = hexi_cfg.get("exempt_destinations") or {}

    # Translate prefecture-keyed maps to post-cluster node names. After L2
    # clustering buses lose their prefecture attribute, so a sub-province origin
    # like Gansu's Hexi strip is identified by its split-node name (e.g.
    # Gansu_West) rather than the prefecture. Without this, such corridors
    # silently escape the limit (e.g. the Jiuquan-Hunan UHV-DC link).
    origin_map = _expand_region_map_for_clustering(origin_map, node_splits or {})
    exempt_map = _expand_region_map_for_clustering(exempt_map, node_splits or {})

    result = {}
    for key, df in (("lines", n.lines), ("links", n.links)):
        # Only genuine transmission branches are corridor crossinfs
        if key == "links" and "carrier" in df.columns:
            df = df[df["carrier"].isin(("AC", "DC"))]
        idx = _find_hexi_corridor_in_component(n, df, origin_map, exempt_map, min_voltage_kv)
        # manual overrides, restricted to names that actually exist in df
        idx = idx.union(df.index.intersection(include))
        idx = idx.difference(exclude)
        result[key] = idx
        logger.info("Hexi corridor: %d %s marked", len(idx), key)

    return result


def _derive_elevation_bus_factor(n: pypsa.Network, factors: dict, elevation_cfg: dict) -> pd.Series:
    """Classify each bus into a terrain factor from GEBCO elevation + relief.

    Plateau is detected from absolute ``elevation`` (m); mountain/hill from
    ``local_relief`` (m, the elevation range over a small neighbour stencil
    sampled at base-network time). A bus takes the highest applicable factor.
    Returns the plain factor for every bus when neither column is present.

    Args:
        n: PyPSA network; uses ``n.buses.elevation`` / ``n.buses.local_relief``.
        factors: terrain class -> multiplier (Wei2017).
        elevation_cfg: thresholds ``plateau_min_m``, ``relief_mountain_m``,
            ``relief_hill_m``.

    Returns:
        Per-bus factor Series indexed by ``n.buses.index``.
    """
    plain = factors.get("plain", 1.0)
    bf = pd.Series(plain, index=n.buses.index, dtype=float)

    if "local_relief" in n.buses.columns:
        relief = pd.to_numeric(n.buses["local_relief"], errors="coerce").fillna(0.0)
        hill_t = elevation_cfg.get("relief_hill_m", np.inf)
        mtn_t = elevation_cfg.get("relief_mountain_m", np.inf)
        relief_f = np.where(
            relief >= mtn_t,
            factors.get("mountain", plain),
            np.where(relief >= hill_t, factors.get("hill", plain), plain),
        )
        bf = pd.Series(np.maximum(bf.to_numpy(), relief_f), index=n.buses.index)

    if "elevation" in n.buses.columns:
        elev = pd.to_numeric(n.buses["elevation"], errors="coerce").fillna(0.0)
        plateau_t = elevation_cfg.get("plateau_min_m", np.inf)
        plateau_f = np.where(elev >= plateau_t, factors.get("plateau", plain), plain)
        bf = pd.Series(np.maximum(bf.to_numpy(), plateau_f), index=n.buses.index)

    return bf


def _set_line_expansion_limits(
    df: pd.DataFrame,
    config: dict,
    component: str,
) -> pd.DataFrame:
    """Set per-branch expansion ceiling on AC Lines or DC Links.

    Config schema (under ``electricity.lines.per_line_expansion``) is split
    per-component so that ``s_nom`` (AC Lines) and ``p_nom`` (DC Links) each
    have their own floor/ratio knobs and there is no ambiguity about which
    column drives which limit::

        per_line_expansion:
          enable: True
          lines:                       # AC Lines (PyPSA n.lines, attr s_nom)
            rel_max_limit: 5           # s_nom_max = max(floor, rel × s_nom)
            s_nom_max_floor: 30.e3     # MW
          links:                       # DC Links (PyPSA n.links, attr p_nom)
            rel_max_limit: 5           # p_nom_max = max(floor, rel × p_nom)
            p_nom_max_floor: 30.e3     # MW

    For backends that only use Links (manual), pass ``component='links'``.
    For OSM (Lines + Links), call once per component.

    Args:
        df: DataFrame of branches (subset of ``n.lines`` or ``n.links``).
        config: Snakemake config dict.
        component: ``"lines"`` for AC Lines (reads/writes ``s_nom``/``s_nom_max``)
            or ``"links"`` for DC Links (reads/writes ``p_nom``/``p_nom_max``).

    Returns:
        DataFrame with the ``*_max`` column set; unchanged if disabled or
        the sub-block is missing.
    """
    cfg = config["electricity"]["lines"].get("per_line_expansion", {"enable": False})
    if not cfg.get("enable", False):
        return df

    if component not in ("lines", "links"):
        raise ValueError(
            f"_set_line_expansion_limits: component must be 'lines' or 'links', got {component!r}"
        )
    per_line_limits = cfg.get(component, {})
    cap_col = "s_nom" if component == "lines" else "p_nom"
    max_col = f"{cap_col}_max"
    floor_key = f"{cap_col}_max_floor"

    factor = per_line_limits.get("rel_max_limit", 1)
    floor = per_line_limits.get(floor_key, np.inf)

    # expansion limit: floor + existing or existing * factor
    computed = df[cap_col].apply(lambda x: max(floor + x, factor * x))
    df[max_col] = computed
    return df


def apply_hexi_corridor_limits(network: pypsa.Network, config: dict) -> None:
    """Apply the per-branch Hexi-corridor expansion ceiling (behaviour A).

    Tighten ``s_nom_max`` (AC Lines) and ``p_nom_max`` (DC/AC Links) on the
    branches crossing the Hexi corridor to ``rel × existing`` capacity. This
    overrides the generic per-line ceiling for corridor branches (the tighter
    corridor limit wins). No-op unless ``electricity.lines.hexi_corridor`` is
    enabled with ``mode == "per_line"``.

    Args:
        network: PyPSA network to modify in place.
        config: Snakemake config dict.
    """
    cfg = config["electricity"]["lines"].get("hexi_corridor_limit", {})
    if not cfg.get("enable", False) or cfg.get("mode") != "per_line":
        return

    rel = cfg.get("per_line", {}).get("rel_max_limit", 1)
    node_splits = config.get("nodes", {}).get("splits", {})
    corridor = find_hexi_corridor_branches(network, cfg, node_splits)

    for comp, df, cap_col in (
        ("lines", network.lines, "s_nom"),
        ("links", network.links, "p_nom"),
    ):
        idx = corridor[comp]
        if len(idx) == 0:
            continue
        max_col = f"{cap_col}_max"
        df.loc[idx, max_col] = df.loc[idx, cap_col] * rel
        logger.info(
            "Hexi corridor: set %s = %g × %s on %d %s",
            max_col,
            rel,
            cap_col,
            len(idx),
            comp,
        )


def find_terrain_cost_markup(
    n: pypsa.Network, terrain_cfg: dict, node_splits: dict | None = None
) -> dict:
    """Derive a per-branch capex multiplier from terrain (Wei2017).

    Each bus is assigned a terrain factor from two sources, combined by taking
    the maximum:

    1. **Elevation** (data-driven, OSM backend): ``n.buses.elevation`` /
       ``n.buses.local_relief`` sampled from GEBCO classify the bus as
       plateau / mountain / hill / plain (see ``_derive_elevation_bus_factor``).
    2. **Region overrides** (config / manual-backend fallback): the nested
       ``province -> [L2 nodes]`` maps under ``terrain_cfg["regions"]`` (empty
       value matches the whole province).

    A branch's markup is the mean of its two endpoint factors, so a line
    climbing from plain onto a plateau is penalised proportionally. Note this
    endpoint mean cannot resolve the mountainous *fraction* of a long corridor
    (a range crossed mid-line between two flat nodes is under-billed).

    Factors are calibrated from Wei et al. (2017) line unit-cost ratios to
    flatland, averaged within scheme letters across 110-1000 kV to remove the
    circuit-count confounding present in the raw min/max spread.

    Args:
        n: PyPSA network with ``province`` (and ideally ``prefecture``) on
            ``n.buses``; optionally ``elevation`` / ``local_relief``.
        terrain_cfg: ``electricity.lines.terrain_cost_markup`` block. Reads
            ``factors``, ``elevation`` (thresholds) and ``regions``.
        node_splits: ``nodes.splits`` config (province -> {split: [prefectures]}).
            Used to translate prefecture-keyed ``regions`` sublists into the
            split-node names the clustered network uses (buses lose their
            ``prefecture`` after clustering). ``None`` skips the translation.

    Returns:
        Dict with ``"lines"`` and ``"links"`` Series of per-branch multipliers
        (empty Series where the component is empty).
    """
    factors = terrain_cfg.get("factors", {}) or {}
    regions = terrain_cfg.get("regions", {}) or {}
    elevation_cfg = terrain_cfg.get("elevation", {}) or {}

    prov = n.buses.province
    l2 = _derive_endpoint_l2(n, pd.Series(n.buses.index, index=n.buses.index))

    # 1. elevation-driven classification (no-op if columns absent)
    bus_factor = _derive_elevation_bus_factor(n, factors, elevation_cfg)

    # 2. region overrides (config / manual-backend path), max with elevation.
    # Expand prefecture-keyed sublists to post-cluster node names so a split
    # province's prefecture list (e.g. Xinjiang desert prefectures) still
    # matches the clustered node (e.g. Xinjiang_West) — mirrors the Hexi fix.
    for tclass, region_map in regions.items():
        f = factors.get(tclass, 1.0)
        expanded = _expand_region_map_for_clustering(region_map or {}, node_splits or {})
        mask = _derive_hexi_region_mask(prov, l2, expanded)
        bus_factor.loc[mask] = np.maximum(bus_factor.loc[mask], f)

    result = {}
    for key, df in (("lines", n.lines), ("links", n.links)):
        # Terrain markup is a transmission build-cost premium (lines climbing
        # onto a plateau, crossing mountains). n.links also holds intra-node
        # tech/storage/converter links (gas OCGT, battery, H2 Electrolysis,
        # biomass, ...) whose capital_cost must NOT be scaled by terrain.
        # Restrict the links component to the transmission carriers.
        if key == "links" and "carrier" in df.columns:
            df = df[df["carrier"].isin(("AC", "DC"))]
        if df.empty:
            result[key] = pd.Series(dtype=float)
            continue
        f0 = df.bus0.map(bus_factor).fillna(1.0)
        f1 = df.bus1.map(bus_factor).fillna(1.0)
        result[key] = (f0 + f1) / 2.0
    return result


def update_p_nom_max(n: pypsa.Network) -> None:
    # if extendable carriers (solar/onwind/...) have capacity >= 0,
    # e.g. existing assets from the OPSD project are included to the network,
    # the installed capacity might exceed the expansion limit.
    # Hence, we update the assumptions.

    n.generators.p_nom_max = n.generators[["p_nom_min", "p_nom_max"]].max(1)


def aggregate_parallel_paths_to_link(lines: pd.DataFrame) -> pd.DataFrame:
    """Collapse parallel paths between the same bus pair into a single Link.

    Used at the boundary where PyPSA-China switches to a **transport
    model** for the HVDC/UHV layer (e.g. at admin-L2 aggregation). After
    the per-corridor ratings have been applied upstream, multiple parallel
    branches between the same ``(bus0, bus1, carrier)`` triple are summed
    into one dispatchable Link with aggregated ``p_nom`` so that downstream
    rules see one corridor per pair rather than N fragments.

    Per carrier: sums ``p_nom``, averages ``length``, takes ``min``
    ``capital_cost`` (most favourable), and keeps the first ``type`` /
    other attributes. AC and DC paths are kept distinct (grouped separately
    on ``carrier``) so each preserves its own physics. If no ``carrier``
    column is present, all lines are treated as one bin.

    Note: this is **not** an analogue of pypsa-earth's ``simplify_network``
    chain. pypsa-earth has no standalone ``simplify_lines``; its closest
    helpers are ``simplify_links`` (HVDC multi-node reduction),
    ``remove_stubs``, and ``aggregate_to_substations``. The degree-2 AC
    chain collapse and DC link chain collapse live in
    ``workflow/scripts/osm/rate_osm_lines.py``
    (``simplify_ac_chains`` and ``simplify_dc_chains``).

    Args:
        lines (pd.DataFrame): branches with ``bus0``, ``bus1`` and
            (optionally) ``carrier``, ``p_nom``, ``length``,
            ``capital_cost``, ``type``.

    Returns:
        pd.DataFrame: one row per ``(bus0, bus1, carrier)`` triple, indexed
        by ``"<bus0>-<bus1> UHV-<carrier>"``.
    """

    has_carrier = "carrier" in lines.columns

    # Canonicalise the bus pair (unordered) so A->B and B->A collapse to ONE
    # corridor. The transport-model Link is bidirectional (positive+reversed
    # split downstream), so orientation carries no information here. Without
    # this, opposite-orientation parallels survive clustering and trip the
    # "duplicate links" guard in prepare_network.
    lines = lines.copy()
    swap = lines["bus0"].astype(str) > lines["bus1"].astype(str)
    cb0 = lines["bus0"].where(~swap, lines["bus1"])
    cb1 = lines["bus1"].where(~swap, lines["bus0"])
    lines["bus0"], lines["bus1"] = cb0, cb1

    group_keys = ["bus0", "bus1", "carrier"] if has_carrier else ["bus0", "bus1"]

    sum_columns = ["p_nom"]
    avg_columns = ["length"]
    min_columns = ["capital_cost"]
    first_columns = ["type"] if "type" in lines.columns else []
    other_columns = lines.columns.difference(
        sum_columns + avg_columns + min_columns + first_columns + group_keys
    )

    agg_links = lines.groupby(group_keys)[sum_columns].sum()
    agg_links[avg_columns] = lines.groupby(group_keys)[avg_columns].mean()
    agg_links[other_columns] = lines.groupby(group_keys)[other_columns].first()
    agg_links[min_columns] = lines.groupby(group_keys)[min_columns].min()
    if first_columns:
        agg_links[first_columns] = lines.groupby(group_keys)[first_columns].first()
    agg_links = agg_links.reset_index()
    suffix = (
        " UHV-" + agg_links["carrier"].astype(str)
        if has_carrier
        else pd.Series(" UHV", index=agg_links.index)
    )
    agg_links.index = agg_links.bus0 + "-" + agg_links.bus1 + suffix
    agg_links.index.name = "links_i"
    return agg_links


def summarize_lines_by_voltage(
    n: pypsa.Network,
    link_carriers: tuple[str, ...] | None = ("DC", "B2B"),
    use_voltage_class: bool = True,
    min_length_km: float = 15.0,
) -> pd.DataFrame:
    """Aggregate Lines (AC) and Links (DC/B2B) by carrier and voltage level.

    A validation helper. In the Chinese grid, voltage classes correspond
    to distinct roles (~110 kV distribution, 220-330 kV regional, 500-
    750 kV bulk, 800-1100 kV UHV long-distance), so inventory per class
    is a basic sanity check on the topology pipeline. Chain with
    :func:`filter_lines_crossing_boundary` to restrict to lines that
    cross a given admin level and obtain inter-region transmission
    inventory per voltage class.

    Lines/Links with ``length < min_length_km`` (default 15 km) are
    excluded from the aggregation. Short OSM fragments are commonly
    dead-end stubs, polyline-snapping artefacts or junctions that
    badly skew per-voltage capacity totals (a 1 km 220 kV stub adds
    the same s_nom as a 500 km corridor under the default rating).

    Args:
        n: PyPSA Network. Lines must carry ``v_nom``, ``length``,
            ``s_nom`` and ``num_parallel``. Links must carry the same
            plus ``carrier`` and ``p_nom``.
        link_carriers: Iterable of Link carriers to count. Defaults to
            HVDC carriers (``"DC"`` and ``"B2B"``); pass ``None`` to
            include every Link regardless of carrier.
        use_voltage_class: Group by the ``voltage_class`` categorical
            column when present (else fall back to numeric v_nom).
        min_length_km: Minimum line length (km) for inclusion. Set to
            0 to disable filtering.

    Returns:
        DataFrame indexed by ``(carrier, v_nom)`` with columns
        ``n_lines``, ``total_length_km``, ``circuit_length_km``,
        ``total_capacity_MW``, ``median_length_km``, ``total_TW_km``,
        ``TW_km_via_median_L``. ``total_length_km`` is right-of-way
        length; ``circuit_length_km`` weights by ``num_parallel`` so a
        double-circuit 500 km line counts as 1000 circuit-km.
        ``total_TW_km`` is ``sum(capacity_MW * length_km) / 1e6`` — the
        exact per-line transport-capacity sum. ``TW_km_via_median_L``
        is ``sum(capacity_MW) * median(length_km) / 1e6`` — a
        median-collapsed transport-capacity proxy that ignores the
        capacity-length covariance within a voltage tier; useful as a
        robust comparator when a few long-tail lines dominate
        ``total_TW_km``.
    """
    frames = []

    def _resolve_v_nom(table: pd.DataFrame) -> pd.Series:
        """v_nom — prefer Line ``v_nom_original`` (pre-rebase), else Line
        ``v_nom``, else bus0.v_nom. ``v_nom_original`` is the OSM native
        voltage preserved through ``apply_pypsa_earth_rebase``; without
        it every Line would inherit the unified base voltage and the
        per-voltage validation would collapse to a single class."""
        if "v_nom_original" in table.columns and table["v_nom_original"].notna().any():
            v = table["v_nom_original"].astype(float)
        elif "v_nom" in table.columns and table["v_nom"].notna().any():
            v = table["v_nom"].astype(float)
        else:
            v = pd.Series(np.nan, index=table.index, dtype=float)
        if v.isna().any() and "v_nom" in n.buses.columns:
            fill = n.buses["v_nom"].reindex(table["bus0"]).astype(float)
            fill.index = table.index
            v = v.where(v.notna(), fill)
        return v

    def _voltage_key(table: pd.DataFrame) -> pd.Series:
        """voltage_class (categorical) if present and requested, else numeric v_nom."""
        if use_voltage_class and "voltage_class" in table.columns:
            return table["voltage_class"].astype(str)
        return _resolve_v_nom(table)

    if not n.lines.empty:
        ac = n.lines[["length", "s_nom", "num_parallel", "bus0"]].copy()
        ac["v_nom"] = _voltage_key(n.lines)
        ac["num_parallel"] = ac["num_parallel"].fillna(1.0).astype(float)
        ac["carrier"] = "AC"
        ac.rename(columns={"s_nom": "capacity_MW"}, inplace=True)
        frames.append(ac[["carrier", "v_nom", "length", "num_parallel", "capacity_MW"]])

    if not n.links.empty:
        dc_full = n.links.copy()
        if link_carriers is not None:
            dc_full = dc_full[dc_full["carrier"].isin(link_carriers)]
        if not dc_full.empty:
            dc = dc_full[["carrier", "length", "p_nom", "bus0"]].copy()
            dc["v_nom"] = _voltage_key(dc_full)
            dc["num_parallel"] = (
                dc_full["num_parallel"].fillna(1.0).astype(float)
                if "num_parallel" in dc_full.columns
                else 1.0
            )
            dc.rename(columns={"p_nom": "capacity_MW"}, inplace=True)
            frames.append(dc[["carrier", "v_nom", "length", "num_parallel", "capacity_MW"]])

    cols = [
        "n_lines", "total_length_km", "circuit_length_km",
        "total_capacity_MW", "median_length_km",
        "total_TW_km", "TW_km_via_median_L",
    ]
    if not frames:
        return pd.DataFrame(columns=cols).rename_axis(index=["carrier", "v_nom"])

    combined = pd.concat(frames, ignore_index=True)
    # Drop degenerate short fragments (dead-end stubs / snapped artefacts)
    # before any aggregation. Caller can disable with min_length_km=0.
    if min_length_km and min_length_km > 0:
        combined = combined[combined["length"] >= min_length_km].copy()
        if combined.empty:
            return pd.DataFrame(columns=cols).rename_axis(index=["carrier", "v_nom"])
    combined["circuit_length_km"] = combined["length"] * combined["num_parallel"]
    combined["capacity_length_MW_km"] = combined["capacity_MW"] * combined["length"]
    grouped = combined.groupby(["carrier", "v_nom"], dropna=False).agg(
        n_lines=("length", "size"),
        total_length_km=("length", "sum"),
        circuit_length_km=("circuit_length_km", "sum"),
        total_capacity_MW=("capacity_MW", "sum"),
        median_length_km=("length", "median"),
        total_MW_km=("capacity_length_MW_km", "sum"),
    )
    grouped["total_TW_km"] = grouped.pop("total_MW_km") / 1e6
    # Median-collapsed proxy: sum(MW) × median(km) / 1e6. Differs from
    # total_TW_km when the capacity-length distribution within a voltage
    # tier is skewed (lots of short lines + a few long ones); matches it
    # when lengths cluster tightly. Useful as a sanity check on whether
    # one or two long lines are carrying the per-class TW-km headline.
    grouped["TW_km_via_median_L"] = (
        grouped["total_capacity_MW"] * grouped["median_length_km"] / 1e6
    )
    return grouped[cols].sort_index()


def filter_lines_crossing_boundary(
    n: pypsa.Network,
    regions: gpd.GeoDataFrame,
    region_col: str = "name",
    link_carriers: tuple[str, ...] | None = ("DC", "B2B"),
    inplace: bool = False,
) -> pypsa.Network:
    """Keep only Lines/Links whose endpoints lie in different ``regions``.

    Use case: chain with :func:`summarize_lines_by_voltage` to report
    inter-provincial / inter-L2 transmission inventory per voltage class::

        provinces = gpd.read_file(...)
        inter_n = filter_lines_crossing_boundary(n, provinces, region_col="NAME_1")
        summarize_lines_by_voltage(inter_n)

    Buses are spatially joined to ``regions`` via ``within``. A Line or
    Link is dropped when both endpoints fall in the SAME identified
    region; lines with one or both buses outside every polygon are
    kept (conservative — caller can be stricter by pre-filtering buses).

    Args:
        n: PyPSA Network with bus ``x`` / ``y`` in EPSG:4326.
        regions: GeoDataFrame of polygons. Any CRS — re-projected
            internally if not lat/lon. Must carry ``region_col``.
        region_col: Column on ``regions`` holding the region identifier
            (e.g. ``"province"``, ``"NAME_1"``, ``"NAME_2"``).
        link_carriers: Restrict the Link filter to these carriers
            (default HVDC ``"DC"`` + ``"B2B"``). Other Links pass through
            unchanged. Pass ``None`` to apply the filter to all Links.
        inplace: If True, mutate ``n``; else operate on a copy.

    Returns:
        PyPSA Network with non-crossing Lines/Links removed.
    """
    if regions.crs is None:
        raise ValueError("regions must have a defined CRS")
    if regions.crs.to_epsg() != 4326:
        regions = regions.to_crs("EPSG:4326")

    target = n if inplace else n.copy()
    buses_gdf = gpd.GeoDataFrame(
        index=target.buses.index,
        geometry=gpd.points_from_xy(target.buses.x, target.buses.y),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(
        buses_gdf,
        regions[[region_col, regions.geometry.name]],
        how="left",
        predicate="within",
    )
    # A bus that overlaps multiple regions (boundary cases) takes its first match.
    bus_region = joined[region_col].groupby(joined.index).first()

    def _same_known_region(table: pd.DataFrame) -> pd.Series:
        r0 = bus_region.reindex(table.bus0).to_numpy()
        r1 = bus_region.reindex(table.bus1).to_numpy()
        both_known = pd.notna(r0) & pd.notna(r1)
        return pd.Series(both_known & (r0 == r1), index=table.index)

    if not target.lines.empty:
        drop = target.lines.index[_same_known_region(target.lines)]
        if len(drop):
            target.remove("Line", drop)

    if not target.links.empty:
        link_mask = (
            target.links["carrier"].isin(link_carriers)
            if link_carriers is not None
            else pd.Series(True, index=target.links.index)
        )
        candidate_links = target.links[link_mask]
        if not candidate_links.empty:
            drop = candidate_links.index[_same_known_region(candidate_links)]
            if len(drop):
                target.remove("Link", drop)

    return target


def store_duals_to_network(network: pypsa.Network) -> None:
    """Store dual variables in network components so they get saved to netcdf file."""
    model = getattr(network, "model", None)
    duals = getattr(model, "dual", None)

    for dual_name, dual_value in duals.items():
        # ---- Parse component / constraint ----
        if "-" in dual_name:
            component_type, constraint_type = dual_name.split("-", 1)
        else:
            lt = dual_name.lower()
            component_type = next((k for k in COMPONENT_MAPPING if k in lt), None)
            if not component_type:
                continue
            constraint_type = dual_name

        comp_name = COMPONENT_MAPPING.get(component_type.lower())
        if comp_name is None or not hasattr(network, comp_name):
            continue
        comp_obj = getattr(network, comp_name)

        # ---- Standardize attr name ----
        attr = f"mu_{re.sub(r'[^a-zA-Z0-9_]', '_', constraint_type)}"

        # ---- Normalize to pandas ----
        if hasattr(dual_value, "to_pandas"):
            dual_value = dual_value.to_pandas()

        # ---- Write back (single decision on DataFrame vs not) ----
        if isinstance(dual_value, pd.DataFrame):
            if comp_name == "global_constraints":
                # collapse rows → per-constraint Series, align to component index
                series = dual_value.mean(axis=0).reindex(comp_obj.index)
                comp_obj[attr] = series
            else:
                # align time index & columns safely
                time_index = getattr(network, "snapshots", dual_value.index)
                aligned = dual_value.reindex(index=time_index, columns=comp_obj.index)

                comp_t_name = f"{comp_name}_t"
                if not hasattr(network, comp_t_name):
                    setattr(network, comp_t_name, {})
                getattr(network, comp_t_name)[attr] = aligned
        else:
            # Series / scalar / ndarray → Series aligned to component index
            if isinstance(dual_value, pd.Series):
                series = dual_value.reindex(comp_obj.index)
            else:
                try:
                    val = float(np.asarray(dual_value).ravel()[0])
                except Exception:
                    val = 0.0
                series = pd.Series(val, index=comp_obj.index)
            comp_obj[attr] = series


def to_list(x: str):
    """In case of csv input. convert str to list

    Args:
        x (str): maybe list like string
    """
    if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
        split = x.replace("[", "").replace("]", "").split(", ")
        # in case no space in the text-list sep
        if split[0].find(",") >= 0:
            return x.replace("[", "").replace("]", "").split(",")
        else:
            return split
    return x


# ============================================================
# Spatial downscaling utilities (province → admin2 / node)
# ============================================================


def shapes_to_shapes(orig: gpd.GeoSeries, dest: gpd.GeoSeries) -> _sp.lil_matrix:
    """Build area-overlap transfer matrix from ``orig`` shapes to ``dest`` shapes.

    Each entry ``[i, j]`` is the fraction of ``dest[i]`` covered by ``orig[j]``.
    Adapted from vresutils.transfer.Shapes2Shapes.

    Args:
        orig: Source geometries.
        dest: Destination geometries.

    Returns:
        scipy.sparse.lil_matrix: Transfer matrix of shape (len(dest), len(orig)).
    """
    orig_prepped = list(map(_prep, orig))
    transfer = _sp.lil_matrix((len(dest), len(orig)), dtype=float)
    for i, j in _product(range(len(dest)), range(len(orig))):
        if orig_prepped[j].intersects(dest.iloc[i]):
            area = orig.iloc[j].intersection(dest.iloc[i]).area
            transfer[i, j] = area / dest.iloc[i].area
    return transfer


def _normed(s: pd.Series) -> pd.Series:
    """Normalize a Series to sum to 1.0."""
    return s / s.sum()


def downsample_provincial_annual_to_admin2(
    prov_annual: pd.DataFrame,
    l2_gdp_pop: gpd.GeoDataFrame,
    distribution_key: dict[str, float],
) -> pd.DataFrame:
    """Downscale provincial annual totals to admin-level-2 nodes using GDP and population weights.

    The bus naming convention matches the base network: ``"ProvinceName-0"``,
    ``"ProvinceName-1"``, etc., ordered by the cumulative count within each province.

    Args:
        prov_annual: Annual totals, index=province names, columns=year strings.
        l2_gdp_pop: GeoDataFrame with at least columns ``NAME_1`` (province),
            ``NAME_2`` (admin2 name), ``gdp_l2``, ``population``, and geometry.
        distribution_key: Fractional weights for GDP vs. population, e.g.
            ``{"gdp": 0.6, "pop": 0.4}``. Must sum to 1.

    Returns:
        pd.DataFrame: Admin2-level annual totals, index=bus names, columns=year strings.

    Raises:
        ValueError: If ``distribution_key`` weights do not sum to 1.
    """
    gdp_weight = distribution_key.get("gdp", 0.6)
    pop_weight = distribution_key.get("pop", 0.4)
    if not abs(gdp_weight + pop_weight - 1.0) < 1e-6:
        raise ValueError(
            f"distribution_key weights must sum to 1.0, got gdp={gdp_weight}, pop={pop_weight}"
        )

    l2_gdp_pop = l2_gdp_pop.copy()
    l2_gdp_pop["bus_idx"] = l2_gdp_pop.groupby("NAME_1").cumcount()

    admin2_bus_names = [
        f"{prov}-{row['bus_idx']}"
        for prov, group in l2_gdp_pop.groupby("NAME_1")
        for _, row in group.iterrows()
    ]
    result = pd.DataFrame(index=admin2_bus_names, columns=prov_annual.columns, dtype=float)

    for province in prov_annual.index:
        prov_data = l2_gdp_pop.query("NAME_1 == @province").copy()
        if prov_data.empty:
            logger.warning("No admin2 data found for province '%s'; skipping.", province)
            continue

        if len(prov_data) == 1:
            factors = pd.Series(1.0, index=prov_data.index)
        else:
            transfer = shapes_to_shapes(prov_data.geometry, prov_data.geometry).T.tocsr()
            gdp_n = pd.Series(
                transfer.dot(prov_data["gdp_l2"].fillna(1.0).values), index=prov_data.index
            )
            pop_n = pd.Series(
                transfer.dot(prov_data["population"].fillna(1.0).values), index=prov_data.index
            )
            factors = _normed(gdp_weight * _normed(gdp_n) + pop_weight * _normed(pop_n))

        prov_annual_data = prov_annual.loc[province]
        for idx, factor in factors.items():
            bus_name = f"{province}-{prov_data.loc[idx, 'bus_idx']}"
            result.loc[bus_name] = prov_annual_data * factor

    return result


def compute_line_utilisation(n: pypsa.Network) -> pd.Series:
    """Compute time-mean utilisation (0..1) of each AC Line.

    Utilisation is the snapshot-weighted mean absolute flow divided by the
    usable capacity::

        util = sum_t(|p0| * w_t) / (s_nom_opt * s_max_pu * sum_t(w_t))

    i.e. a line capacity factor on the optimised rating. Lines with zero
    optimised capacity (or no flow) map to 0; the result is clipped to 1.

    Args:
        n: Solved PyPSA network.

    Returns:
        Series indexed by line name, values in [0, 1].
    """
    if n.lines.empty or n.lines_t.p0.empty:
        return pd.Series(dtype=float, index=n.lines.index)
    w = n.snapshot_weightings.objective
    s_max_pu = (
        pd.to_numeric(n.lines.get("s_max_pu", pd.Series(1.0, index=n.lines.index)), errors="coerce")
        .fillna(1.0)
        .replace(0.0, np.nan)
    )
    cap = pd.to_numeric(n.lines["s_nom_opt"], errors="coerce").replace(0.0, np.nan)
    energy = n.lines_t.p0.abs().mul(w, axis=0).sum()          # MWh
    util = energy / (cap * s_max_pu * float(w.sum()))
    return util.clip(upper=1.0).fillna(0.0)


def compute_transmission_utilisation(n: pypsa.Network) -> dict:
    """Compute time-mean utilisation (0..1) of transmission Lines and Links.

    Generalises `compute_line_utilisation` to cover every backend:

    - OSM AC backbone on ``n.lines`` (via `compute_line_utilisation`);
    - transmission ``n.links`` (carrier AC or DC) — manual-backend AC links and
      all HVDC corridors.

    The lossy positive+reversed split builds two independent directional legs,
    so utilisation is computed per leg — ``energy / (p_nom_opt * p_max_pu *
    hours)`` — and the corridor takes the MAX over its legs (the busier
    direction), assigned to the real (non-``reversed``) leg. Reversed legs and
    non-transmission (sector) links map to NaN so callers can hide them.

    Args:
        n: Solved PyPSA network.

    Returns:
        Dict ``{"lines": Series, "links": Series}``. ``lines`` values in [0, 1]
        indexed by line; ``links`` indexed by every link, with utilisation on
        real transmission legs and NaN elsewhere.
    """
    lines_u = compute_line_utilisation(n)

    links_u = pd.Series(np.nan, index=n.links.index, dtype=float)
    if not n.links.empty and not n.links_t.p0.empty:
        w = n.snapshot_weightings.objective
        hrs = float(w.sum())
        tr_lnk = n.links[n.links.carrier.isin(["AC", "DC"])]
        if not  tr_lnk.empty:
            names = tr_lnk.index.to_series()
            base = names.str.replace(" positive", "", regex=False).str.replace(
                " reversed", "", regex=False
            )
            p_max_pu = (
                pd.to_numeric(tr_lnk.get("p_max_pu", pd.Series(1.0, index=tr_lnk.index)), errors="coerce")
                .fillna(1.0)
                .replace(0.0, np.nan)
            )
            cap = pd.to_numeric(tr_lnk["p_nom_opt"], errors="coerce").replace(0.0, np.nan)
            energy = n.links_t.p0[tr_lnk.index].abs().mul(w, axis=0).sum()       # MWh per leg
            u_leg = (energy / (cap * p_max_pu * hrs)).clip(upper=1.0)         # per leg
            u_corr = u_leg.groupby(base).max()                               # busier direction
            real = tr_lnk.index[~names.str.endswith(" reversed").values]
            links_u.loc[real] = base.loc[real].map(u_corr).values
    return {"lines": lines_u, "links": links_u}


def compute_ac_line_losses(n: pypsa.Network) -> dict:
    """Compute AC-line transmission losses from ``n.statistics`` (energy, MWh).

    Losses are ``withdrawal - supply`` restricted to AC buses, so the figure is
    snapshot-weighted, robust to carrier renaming, and captures the solver's
    tangent-linearized I^2R line losses. Every Line sits on AC buses, so the
    ``bus_carrier="AC"`` filter selects exactly the AC backbone.

    DC-link losses are intentionally NOT reported here: the lossy
    positive+reversed Link split makes ``supply - withdrawal`` an unreliable
    (often negative) loss proxy for the DC corridors.

    Args:
        n: Solved PyPSA network.

    Returns:
        Dict with ``inflow_MWh`` (energy entering the lines), ``supply_MWh``
        (delivered at receiving ends), ``losses_MWh`` and ``loss_fraction``
        (losses / inflow; 0 when there is no AC flow).
    """
    wd = float(n.statistics.withdrawal(comps="Line", bus_carrier="AC").sum())
    sp = float(n.statistics.supply(comps="Line", bus_carrier="AC").sum())
    loss = wd - sp
    return {
        "inflow_MWh": wd,
        "supply_MWh": sp,
        "losses_MWh": loss,
        "loss_fraction": (loss / wd) if wd else 0.0,
    }


def compute_link_losses(
    n: pypsa.Network, carriers: tuple[str, ...] = ("AC", "DC", "B2B")
) -> dict:
    """Compute transmission-link losses from the lossy positive+reversed split (energy, MWh).

    Under ``force_dc`` / the transport preset the AC backbone is converted to
    lossy transport-model Links, so the line-based ``compute_ac_line_losses``
    reports zero. For a lossy Link ``p1 = -efficiency * p0``; per snapshot the
    dissipated power is ``p0 + p1 = p0 * (1 - efficiency) >= 0`` whenever
    ``p0 >= 0`` (the positive+reversed split sets ``p_min_pu = 0``). Summed over
    both legs and weighted by ``snapshot_weightings.objective`` this is the exact
    corridor loss energy. Restricted to transmission carriers (``AC``/``DC``/
    ``B2B``) so storage and conversion Links are excluded.

    Args:
        n: Solved PyPSA network.
        carriers: Link carriers treated as transmission corridors.

    Returns:
        Dict with ``inflow_MWh`` (energy entering the corridors), ``supply_MWh``
        (delivered at receiving ends), ``losses_MWh`` and ``loss_fraction``
        (losses / inflow; 0 when there is no link flow).
    """
    links = (
        n.links[n.links.carrier.isin(carriers)]
        if "carrier" in n.links.columns
        else n.links.iloc[:0]
    )
    if links.empty or n.links_t.p0.empty:
        return {"inflow_MWh": 0.0, "supply_MWh": 0.0, "losses_MWh": 0.0, "loss_fraction": 0.0}

    w = n.snapshot_weightings.objective
    p0 = n.links_t.p0.reindex(columns=links.index, fill_value=0.0)
    p1 = n.links_t.p1.reindex(columns=links.index, fill_value=0.0)
    inflow = float(p0.clip(lower=0).mul(w, axis=0).to_numpy().sum())
    loss = float((p0 + p1).clip(lower=0).mul(w, axis=0).to_numpy().sum())
    return {
        "inflow_MWh": inflow,
        "supply_MWh": inflow - loss,
        "losses_MWh": loss,
        "loss_fraction": (loss / inflow) if inflow else 0.0,
    }


def convert_ac_lines_to_links(
    network: pypsa.Network, costs: pd.DataFrame | None = None
) -> pd.Index:
    """Convert all AC ``n.lines`` into transport-model ``n.links`` (carrier='DC').

    Drops PyPSA's native KVL/KCL on the AC backbone and treats every corridor as
    a uniform dispatchable Link (transport model). Expansion bounds and
    extendability carry over (``s_nom*`` -> ``p_nom*``). The caller is
    responsible for the lossy positive+reversed split on the returned links.

    Capital cost is PRESERVED from the per-kV AC capex already stored on the
    Lines (``capital_cost``); rows whose capex is missing fall back to a flat
    HVDC-overhead basis (requires ``costs``). When ``costs`` is None every Line
    must already carry a positive ``capital_cost``.

    Args:
        network: PyPSA network mutated in place.
        costs: Optional cost table for the flat HVDC-overhead fallback.

    Returns:
        Index of the newly created Links (empty if there were no AC Lines).
    """
    if network.lines.empty:
        return pd.Index([])

    lines = network.lines
    s_max_pu = lines.get("s_max_pu", pd.Series(1.0, index=lines.index))
    new_index = pd.Index([f"{i} AC2DC" for i in lines.index])

    ac_capex = pd.to_numeric(
        lines.get("capital_cost", pd.Series(np.nan, index=lines.index)), errors="coerce"
    ).values
    if costs is not None:
        flat_hvdc = lines.length.values * costs.at["HVDC overhead", "capital_cost"] * FOM_LINES
        capital_cost = np.where(np.nan_to_num(ac_capex, nan=0.0) > 0, ac_capex, flat_hvdc)
    else:
        if not (np.nan_to_num(ac_capex, nan=0.0) > 0).all():
            raise ValueError(
                "convert_ac_lines_to_links: some AC Lines lack a positive "
                "capital_cost and no `costs` fallback was provided."
            )
        capital_cost = ac_capex

    links_data = pd.DataFrame(
        {
            "bus0": lines.bus0.values,
            "bus1": lines.bus1.values,
            "length": lines.length.values,
            "p_nom": lines.s_nom.values,
            "p_nom_min": lines.get("s_nom_min", pd.Series(0.0, index=lines.index)).values,
            "p_nom_max": lines.get("s_nom_max", pd.Series(np.inf, index=lines.index)).values,
            "p_nom_extendable": lines.get(
                "s_nom_extendable", pd.Series(False, index=lines.index)
            ).values,
            "p_max_pu": s_max_pu.values,
            "p_min_pu": -s_max_pu.values,
            "carrier": "DC",
            "capital_cost": capital_cost,
        },
        index=new_index,
    )

    network.remove("Line", lines.index)
    network.add("Link", links_data.index, **links_data)
    logger.info(
        "convert_ac_lines_to_links: converted %d AC Lines into transport-model "
        "DC Links (AC per-kV capex preserved)",
        len(links_data),
    )
    return new_index


def split_lossy_links_positive_reversed(
    network: pypsa.Network, link_mask: pd.Index, config: dict
) -> None:
    """Split each transport-model Link into a positive + reversed pair.

    Each direction's losses are accounted for independently (PyPSA-EUR
    convention)::

        eff = eta_static * eta_per_1000km(v_nom) ** (length_km / 1000)

    ``eta_per_1000km`` is voltage-dependent (DC I^2R loss ~1/V^2): each link's
    ``v_nom`` (falling back to its bus0 voltage) is snapped to the nearest
    listed kV in ``transmission_efficiency.DC.efficiency_per_1000km_by_kV``,
    with the scalar ``efficiency_per_1000km`` as fallback. Operates only on the
    Links named by ``link_mask``.
    """
    lossy_links = network.links.loc[link_mask].copy()
    dc_cfg = config["transmission_efficiency"]["DC"]
    eta_stat = dc_cfg["efficiency_static"]
    by_kv = dc_cfg.get("efficiency_per_1000km_by_kV", {}) or {}
    fallback = float(dc_cfg.get("efficiency_per_1000km", 0.95))

    # Per-link voltage: prefer the link's own v_nom, else its bus0 voltage.
    v_nom = pd.to_numeric(
        lossy_links.get("v_nom", pd.Series(np.nan, index=lossy_links.index)), errors="coerce"
    )
    bus_v = lossy_links["bus0"].map(network.buses["v_nom"])
    v_nom = v_nom.where(v_nom > 0, bus_v)

    def _eta_per_1000km(v: float) -> float:
        if not by_kv or pd.isna(v):
            return fallback
        nearest = min(by_kv, key=lambda k: abs(float(k) - float(v)))
        return float(by_kv[nearest])

    eta_km = v_nom.map(_eta_per_1000km)
    lossy_links["efficiency"] = eta_stat * eta_km ** (lossy_links.length / 1000)
    lossy_links["p_min_pu"] = 0
    network.links.loc[link_mask] = lossy_links
    rename_map = dict(zip(link_mask, link_mask + " positive"))
    network.links.rename(index=rename_map, inplace=True)
    # Reversed leg has zero length / cost so it doesn't double-count line volume.
    lossy_links["length"] = 0
    lossy_links["capital_cost"] = 0
    lossy_links_reversed = lossy_links.copy()
    lossy_links_reversed["bus0"] = lossy_links["bus1"]
    lossy_links_reversed["bus1"] = lossy_links["bus0"]
    network.add("Link", lossy_links_reversed.index, suffix=" reversed", **lossy_links_reversed)
