# SPDX-FileCopyrightText: : 2022 The PyPSA-Eur Authors
# SPDX-License-Identifier: MIT

# WARNING: DO NOT DO "import snakemake"

"""
Helper functions for the PyPSA China workflow including
- HPC helpers (gurobi tunnel setup)
- Snakemake helpers (logging, path management and emulators for testing)
"""
import os

if "PROJ_DATA" not in os.environ:
    _conda = os.environ.get("CONDA_PREFIX", "")
    if _conda:
        os.environ["PROJ_DATA"] = os.path.join(_conda, "share", "proj")


import atexit
import functools
import importlib
import importlib.util
import logging
import multiprocessing
import os
import subprocess
import sys
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

# get root logger
logger = logging.getLogger()

DEFAULT_TUNNEL_PORT = 1080
LOGIN_NODE = "01"


# ============== Path & config Management ==================


class ConfigManager:
    """Config manager class for the snakemake configs"""

    def __init__(self, config: dict):
        self._raw_config = deepcopy(config)
        self.config = deepcopy(config)
        self.wildcards = {}

    def find_screened_params(self) -> list:
        """Determine which scenario variations are applied.
        Excludes planning horizons & co2 pathways
        """
        names = []
        for k, v in self.config["scenario"].items():
            if k not in ["planning_horizons", "co2_pathway"] and type(v) is list:
                if len(v) > 1:
                    names.append(k)
        return names

    def get_scenario_wildcards(self) -> list:
        """Return scenario wildcard names used in path templates.

        Extracts all scenario variable names except 'planning_horizons',
        which are used as wildcards for path expansion (e.g., in expand patterns).

        Returns:
            list: Scenario wildcard names (e.g., ['co2_pathway', 'topology']).

        Example:
            >>> config_manager = ConfigManager(config)
            >>> wildcards = config_manager.get_scenario_wildcards()
            >>> print(wildcards)
            ['co2_pathway', 'topology']
        """
        return [key for key in self.config["scenario"] if key != "planning_horizons"]

    def _normalize_legacy_config(self) -> dict:
        """Migrate legacy config keys to the new sectors.enabled.* format.

        Migrates:
          - ``heat_coupling: true`` → ``sectors.enabled.heating: true``
          - ``sectors.electric_vehicles: true`` → ``sectors.enabled.transport: true``

        The original keys are preserved for backward compatibility.

        Returns:
            dict: The modified config dict (same object as self.config).
        """
        if "sectors" not in self.config:
            self.config["sectors"] = {}
        if "enabled" not in self.config["sectors"]:
            self.config["sectors"]["enabled"] = {}

        if self.config.get("heat_coupling", False):
            logger.warning(
                "Config key 'heat_coupling' is deprecated. "
                "Use 'sectors.enabled.heating' instead."
            )
            self.config["sectors"]["enabled"].setdefault("heating", True)

        ev_enabled = (
            isinstance(self.config.get("sectors"), dict)
            and self.config["sectors"].get("electric_vehicles") is True
        )
        if ev_enabled:
            logger.warning(
                "Config key 'sectors.electric_vehicles: true' is deprecated. "
                "Use 'sectors.enabled.transport: true' instead."
            )
            self.config["sectors"]["enabled"].setdefault("transport", True)

        # Reverse sync: ensure legacy keys reflect new-style config so that
        # existing scripts and rule params that read config["heat_coupling"] still work.
        if self.config["sectors"]["enabled"].get("heating", False):
            self.config["heat_coupling"] = True

        return self.config

    def handle_scenarios(self) -> dict:
        """Unpack and filter scenarios from the configuration.

        Processes planning horizons by converting them to integers and handles
        GHG scenarios through the GHGConfigHandler. This method modifies the
        internal config state.

        Returns:
            dict: The processed configuration with validated scenarios.

        Example:
            >>> config_manager = ConfigManager(raw_config)
            >>> processed_config = config_manager.handle_scenarios()
            >>> print(processed_config['scenario']['planning_horizons'])
            [2020, 2030, 2040, 2050]
        """
        self._normalize_legacy_config()
        self.config["scenario"]["planning_horizons"] = [
            int(v) for v in self._raw_config["scenario"]["planning_horizons"]
        ]
        ghg_handler = GHGConfigHandler(self.config.copy())
        self.config = ghg_handler.handle_ghg_scenarios()

        return self.config

    @property
    def sectors(self) -> "SectorConfigHandler":
        """Lazy-loaded SectorConfigHandler for sector enable/config queries.

        Returns:
            SectorConfigHandler: Handler for sector configuration.
        """
        if not hasattr(self, "_sectors_handler"):
            self._sectors_handler = SectorConfigHandler(self.config)
        return self._sectors_handler

    def fetch_co2_restriction(self, pthw_name: str, year: str) -> dict:
        """Fetch CO2 restriction parameters for a specific scenario and year.

        Retrieves the CO2 emission reduction or price limit for a given scenario
        pathway and planning year from the configuration.

        Args:
            pthw_name (str): The name of the CO2 scenario pathway (e.g., 'exp175default').
            year (str): The planning year as a string (e.g., '2030').

        Returns:
            dict: A dictionary containing:
                - 'co2_pr_or_limit': The CO2 reduction fraction or price limit
                - 'control': The control method ('reduction', 'price', 'budget', etc.)

        Raises:
            KeyError: If the pathway name or year is not found in the configuration.

        Example:
            >>> config_manager = ConfigManager(config)
            >>> restriction = config_manager.fetch_co2_restriction('exp175default', '2030')
            >>> print(restriction)
            {'co2_pr_or_limit': 0.41175086, 'control': 'reduction'}
        """
        scenario = self.config["co2_scenarios"][pthw_name]
        return {
            "co2_pr_or_limit": scenario["pathway"][year],
            "control": scenario["control"],
        }

    def make_wildcards(self) -> list:
        """Expand wildcards in config"""
        raise NotImplementedError


# TODO add dataclass for Config ? or otherwise validate
class GHGConfigHandler:
    """A class to handle & validate GHG scenarios in the config"""

    def __init__(self, config: dict):
        self.config = deepcopy(config)
        self._raw_config = deepcopy(config)
        self._validate_scenarios()

    def handle_ghg_scenarios(self) -> dict:
        """Handle ghg scenarios (parse, valdiate & unpack to config[scenario])

        Returns:
            dict: validated and parsed
        """
        # HACK for snakemake access
        scripts_dir = os.path.abspath(os.path.dirname(__file__))
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        heat_enabled = self.config.get("heat_coupling", False) or self.config.get(
            "sectors", {}
        ).get("enabled", {}).get("heating", False)
        if heat_enabled:
            # HACK import here for snakemake access
            from constants import CO2_BASEYEAR_EM as base_year_ems
        else:
            from constants import CO2_EL_2020 as base_year_ems

        for name, co2_scen in self.config["co2_scenarios"].items():
            co2_scen["pathway"] = {int(k): v for k, v in co2_scen.get("pathway", {}).items()}
        self._reduction_to_budget(base_year_ems)
        self._filter_active_scenarios()
        return self.config

    def _filter_active_scenarios(self):
        """Select active ghg scenarios"""
        scenarios = self.config["scenario"].get("co2_pathway", [])
        if not isinstance(scenarios, list):
            scenarios = [scenarios]

        self.config["co2_scenarios"] = {
            k: v for k, v in self.config["co2_scenarios"].items() if k in scenarios
        }

    def _reduction_to_budget(self, base_yr_ems: float):
        """Transform reduction to budget

        Args:
            base_yr_ems (float): Base year emissions value
        """
        for name, co2_scen in self.config["co2_scenarios"].items():
            if co2_scen["control"] == "reduction":
                budget = {yr: base_yr_ems * (1 - redu) for yr, redu in co2_scen["pathway"].items()}
                self.config["co2_scenarios"][name]["pathway"] = budget
                self.config["co2_scenarios"][name]["control"] = "budget_from_reduction"

    # TODO switch to config validate, see
    # https://snakemake.readthedocs.io/en/stable/snakefiles/configuration.html#validation
    def _validate_scenarios(self):
        """Validate CO2 scenarios"""

        for name, scen in self._raw_config["co2_scenarios"].items():
            # do not validate if not selected
            if name not in self.config["scenario"]["co2_pathway"]:
                continue

            # check type
            if not isinstance(scen, dict):
                raise ValueError(f"Expected a dictionary for co2 scenario but got {scen}")

            # control type none = free emissions. DOn't validate
            if "control" in set(scen) and scen["control"] is None:
                continue

            # otherwise check expected keys in scenario
            if {"control", "pathway"} - set(scen):
                raise ValueError(f"Scenario {scen} must contain 'control' and 'pathway'")

            ALLOWED = ["price", "reduction", "budget", "budget_from_reduction", None]

            if scen["control"] not in ALLOWED:
                err = f"Control must be {','.join([str(x) for x in ALLOWED])} but was"
                err += f" {name}:{scen.get('control', 'missing')}"
                raise ValueError(err)

            years_int = set(map(int, self.config["scenario"]["planning_horizons"]))
            missing_yrs = years_int - set(map(int, scen["pathway"]))
            if missing_yrs:
                raise ValueError(f"Years in scenario {scen['pathway']} missing {missing_yrs}")


# TODO return pathlib objects? so can just use / to combine paths?
# TODO unit tests for path manager
class PathManager:
    """Manages file system paths for the Snakemake workflow.

    This class provides centralized path management for the PyPSA-China workflow,
    handling different path configurations for production runs vs. CI/CD test runs.
    It constructs paths based on scenario configurations and wildcards.

    Attributes:
        config: The Snakemake configuration dictionary.
        root_dir: The root directory of the project.

    Example:
        >>> pm = PathManager(snakemake_config)
        >>> results_path = pm.results_dir()
        >>> cutout_path = pm.cutouts_dir
    """

    def __init__(self, snmk_config: dict, scenario_vars: list | None = None):
        """Initialize the PathManager with configuration.

        Args:
            snmk_config: The Snakemake configuration containing run settings.
            scenario_vars: Optional list of scenario variations to implement as a wildcard.
        """
        self.config = snmk_config
        self._is_test_run = self.config["run"].get("is_test", False)
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.scenario_vars = scenario_vars or []

    @functools.lru_cache(maxsize=1)
    def _get_version(self) -> str:
        """Get version from workflow package.

        Returns:
            Version string from the workflow package.
        """
        spec = importlib.util.spec_from_file_location(
            "workflow", os.path.abspath("./workflow/__init__.py")
        )
        workflow = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(workflow)
        return workflow.__version__

    # Scenario dimensions that only influence the solve stage (and downstream
    # post-processing), NOT the prepared network. They are stripped from the
    # prenetwork / derived-data paths so prepare runs once and is shared across
    # variants, while results_dir() keeps them so each variant gets its own
    # results subtree.
    SOLVE_ONLY_SCENARIO_VARS = ("transmission",)

    @functools.lru_cache(maxsize=8)
    def _join_scenario_vars(self, include_solve_only: bool = True) -> str:
        """Join scenario variables into a compact directory name string.

        Constructs a path segment with Snakemake wildcard placeholders like:
        'topology-{topology}_co2pw-{co2_pathway}'

        Extra wildcards from self.scenario_vars are appended as:
        '/base_vars_{extra_var1}_{extra_var2}'

        Args:
            include_solve_only: Keep solve-only dimensions (see
                ``SOLVE_ONLY_SCENARIO_VARS``). Pass False for prenetwork /
                derived-data paths that must be shared across solve variants.

        Returns:
            String with wildcards for use in paths.
        """
        exclude = ["planning_horizons", "co2_reduction"]
        if not include_solve_only:
            exclude = exclude + list(self.SOLVE_ONLY_SCENARIO_VARS)
        short_names = {
            "planning_horizons": "yr",
            "topology": "topo",
            "co2_pathway": "co2pw",
            "transmission": "tx",
        }
        scenario_keys = [k for k in self.config["scenario"] if k not in exclude]

        preferred = ["co2_pathway", "topology", *scenario_keys]
        ordered_keys = []
        for key in preferred:
            if key in scenario_keys and key not in ordered_keys:
                ordered_keys.append(key)

        return "_".join(f"{short_names.get(k, k)}-{{{k}}}" for k in ordered_keys)

    def _resolve_path(self, path: str, default: str) -> str:
        """Resolve a config path, making it absolute if needed.

        Args:
            path: Path from config (may be empty, relative, or absolute).
            default: Default path to use if config path is empty.

        Returns:
            Resolved absolute or relative path.
        """
        if not path:
            return default
        if os.path.exists(path):
            return path
        return os.path.abspath(path)

    def results_dir(self, extra_opts: dict = None, include_solve_only: bool = True) -> str:
        """Generate the results directory path.

        Args:
            extra_opts: Optional extra options to append to path.
            include_solve_only: Keep solve-only scenario dimensions in the path.
                Pass False to obtain the shared prenetwork directory (stripped
                of solve-only variants like ``transmission``).

        Returns:
            Path to results directory.
        """
        run, foresight = self.config["run"]["name"], self.config["foresight"]
        base_dir = f"v-{self._get_version()}_{run}"
        sub_dir = f"{foresight}_{self._join_scenario_vars(include_solve_only)}"

        suffix = ""
        if self.config["heat_coupling"]:
            suffix += "-heat"
        sub_dir += suffix

        if extra_opts:
            sub_dir += "/" + "".join(extra_opts.values())
        return os.path.join(self.config["paths"]["results_dir"], base_dir, sub_dir)

    def derived_data_dir(self, shared: bool = False) -> str:
        """Generate the derived data directory path.

        Args:
            shared: If True, return shared derived data dir (no scenario subdirs).

        Returns:
            Path to derived data directory.
        """
        base = "tests" if self._is_test_run else "resources"
        base_path = os.path.abspath(os.path.join(self.root_dir, base, "derived_data"))

        if shared:
            return base_path
        else:
            base_path += "/scenarios"

        sub_dir = f"{self.config['foresight']}_{self._join_scenario_vars(include_solve_only=False)}"
        return os.path.join(base_path, sub_dir)

    @property
    def logs_dir(self) -> str:
        """Generate logs directory path (shared across solve-only variants)."""
        run, foresight = self.config["run"]["name"], self.config["foresight"]
        base_dir = f"v-{self._get_version()}_{run}"
        sub_dir = f"{foresight}_{self._join_scenario_vars(include_solve_only=False)}"
        return os.path.join("logs", base_dir, sub_dir)

    @property
    def cutouts_dir(self) -> str:
        """Get cutouts directory path."""
        return "tests/testdata" if self._is_test_run else "resources/cutouts"

    @property
    def landuse_raster_data(self) -> str:
        """Get landuse raster data directory path."""
        if self._is_test_run:
            return "tests/testdata/landuse_availability"
        return "resources/data/landuse_availability"

    # ===== Configurable paths with REMIND variants =====

    def _default_costs_dir(self) -> str:
        """Get default costs directory (override in subclass for REMIND)."""
        return "resources/data/costs/default"

    def costs_dir(self) -> str:
        """Get costs directory path from config or default."""
        default = self._default_costs_dir()
        path = self.config["paths"].get("costs_dir", default)
        path = self._resolve_path(path, default)
        return path.rstrip("/")

    def _default_elec_load(self) -> str:
        """Get default electric load path (override in subclass for REMIND)."""
        return "resources/data/load/Provincial_Load_2020_2060_MWh.csv"

    def elec_load(self, variation: str | None = None) -> str:
        """Get electric load data path from config or default.

        Args:
            variation: Optional load variation key, e.g. ``"high"``.
                When provided, reads from ``paths.yearly_regional_load[variation].ac``.
        """
        default = self._default_elec_load()
        if variation is None:
            loads = self.config["paths"].get("yearly_regional_load", {"ac": default})
        else:
            all_loads = self.config["paths"].get("yearly_regional_load")
            if all_loads is None:
                raise ValueError(
                    "A load variation was specified but no paths are configured under "
                    "'yearly_regional_load'"
                )
            if variation not in all_loads:
                raise ValueError(
                    f"Specified load variation '{variation}' not found in config under "
                    "'yearly_regional_load'."
                )
            loads = all_loads[variation]

        path = loads.get("ac", default)
        return self._resolve_path(path, default)

    def _default_heat_load(self) -> str:
        """Get default heat load path (override in subclass for REMIND)."""
        return "resources/data/heating/Zhou_et_al_heat_projection.csv"

    def heat_load(self) -> str:
        """Get heat load data path from config or default."""
        default = self._default_heat_load()
        path = self.config["paths"].get("heat_projections", default)
        return self._resolve_path(path, default)

    def _default_infrastructure(self) -> str:
        """Get default infrastructure path (override in subclass for REMIND)."""
        return f"{self.derived_data_dir()}/existing_infrastructure"

    def infrastructure(self) -> str:
        """Get existing infrastructure data path from config or default."""
        default = self._default_infrastructure()
        path = self.config["paths"].get("existing_infra", default)
        return self._resolve_path(path, default)

    def ev_passenger_loads(self) -> str:
        """Get EV passenger loads input path from config.

        Reads the standalone path from
        ``config["sectors"]["electric_vehicles"]["standalone_passenger_input"]``.

        Returns:
            str: Path to the EV passenger loads file.

        Raises:
            ValueError: If ``data_source`` is explicitly set to ``"remind"`` but
                this is not a REMIND-coupled run (i.e. not a RemindPathManager).
            ValueError: If ``data_source`` is ``"standalone"`` but no path is
                configured under ``standalone_passenger_input``.
        """
        cfg = self.config.get("sectors", {}).get("electric_vehicles", {})
        path = cfg.get("standalone_passenger_input")
        if not path:
            raise ValueError(
                "sectors.electric_vehicles.data_source='standalone' but no "
                "'standalone_passenger_input' is configured under sectors.electric_vehicles."
            )
        return path

    def ev_freight_loads(self) -> str:
        """Get EV freight loads input path from config.

        Reads the standalone path from
        ``config["sectors"]["electric_vehicles"]["standalone_freight_input"]``.

        Returns:
            str: Path to the EV freight loads file.

        Raises:
            ValueError: If ``data_source`` is ``"standalone"`` but no path is
                configured under ``standalone_freight_input``.
        """
        cfg = self.config.get("sectors", {}).get("electric_vehicles", {})
        path = cfg.get("standalone_freight_input")
        if not path:
            raise ValueError(
                "sectors.electric_vehicles.data_source='standalone' but no "
                "'standalone_freight_input' is configured under sectors.electric_vehicles."
            )
        return path

    def h2_loads(self) -> str:
        """Get H2 loads input path from config.

        Reads the standalone path from
        ``config["sectors"]["hydrogen"]["standalone_input"]``.

        Returns:
            str: Path to the H2 loads file.

        Raises:
            ValueError: If ``data_source`` is explicitly set to ``"remind"`` but
                this is not a REMIND-coupled run (i.e. not a RemindPathManager).
            ValueError: If ``data_source`` is ``"standalone"`` but no path is
                configured under ``standalone_input``.
        """
        cfg = self.config.get("sectors", {}).get("hydrogen", {})
        path = cfg.get("standalone_input")
        if not path:
            raise ValueError(
                "sectors.hydrogen.data_source='standalone' but no "
                "'standalone_input' is configured under sectors.hydrogen."
            )
        return path


class RemindPathManager(PathManager):
    """Path manager for REMIND-coupled runs.

    Overrides default paths to use REMIND-specific data locations in derived data.
    """

    def _default_costs_dir(self) -> str:
        """Get REMIND costs directory."""
        return f"{self.derived_data_dir()}/remind/costs"

    def _default_elec_load(self) -> str:
        """Get REMIND electric load path."""
        return f"{self.derived_data_dir()}/remind/ac_load_disagg_{{cluster_id}}.csv"

    def _default_infrastructure(self) -> str:
        """Get REMIND harmonized capacities path."""
        return f"{self.derived_data_dir()}/remind/harmonized_capacities"

    def ev_passenger_loads(self) -> str:
        """Get REMIND EV passenger loads path."""
        return f"{self.derived_data_dir()}/remind/ev_passenger_loads_{{cluster_id}}.csv"

    def ev_freight_loads(self) -> str:
        """Get REMIND EV freight loads path."""
        return f"{self.derived_data_dir()}/remind/ev_freight_loads_{{cluster_id}}.csv"

    def h2_loads(self) -> str:
        """Get REMIND H2 loads path."""
        return f"{self.derived_data_dir()}/remind/h2_loads_{{cluster_id}}.csv"


class WorkflowChainManager:
    """Manages the network preparation chain for sector coupling.

    Determines the suffix-based file paths for each stage in the pipeline:
    ``prepare_networks → [add_heat_sector] → [add_transport_sector]
    → [add_existing_baseyear] → solve_networks``

    Supports both old-style config keys (e.g. ``heat_coupling: true``) and
    new-style keys (e.g. ``sectors.enabled.heating: true``), preferring the
    new format when both are present.

    Attributes:
        SECTOR_ORDER (OrderedDict): Ordered mapping of sector name to config
            metadata (``config_key``, ``suffix``, ``new_config_key``).
    """

    SECTOR_ORDER = OrderedDict(
        [
            (
                "heating",
                {
                    "config_key": "heat_coupling",
                    "suffix": "-heat",
                    "new_config_key": "sectors.enabled.heating",
                },
            ),
            (
                "transport",
                {
                    "config_key": "sectors.electric_vehicles",
                    "suffix": "-transport",
                    "new_config_key": "sectors.enabled.transport",
                },
            ),
            (
                "hydrogen",
                {
                    "config_key": "sectors.hydrogen.enabled",
                    "suffix": "-hydrogen",
                    "new_config_key": "sectors.enabled.hydrogen",
                },
            ),
        ]
    )

    def __init__(self, cfg: dict):
        """Initialize with the workflow configuration dict.

        Args:
            cfg (dict): Snakemake configuration dictionary.
        """
        self.cfg = cfg

    def _is_sector_enabled(self, config_key: str) -> bool:
        """Check whether a sector is enabled by a dotted or simple config key.

        Supports dotted traversal (e.g. ``"sectors.enabled.heating"``).

        Args:
            config_key (str): Dotted key path into the config dict.

        Returns:
            bool: True if the key resolves to a truthy value.
        """
        keys = config_key.split(".")
        node = self.cfg
        for k in keys:
            if not isinstance(node, dict):
                return False
            node = node.get(k, False)
        # Accept only explicit boolean True or non-empty string "true"/"yes".
        # A dict value (e.g. sectors.electric_vehicles being a config sub-block)
        # must NOT be treated as an enable flag.
        if isinstance(node, dict):
            return False
        return bool(node)

    def _sector_enabled(self, sector_name: str) -> bool:
        """Check whether a named sector is enabled, preferring new-style keys.

        Args:
            sector_name (str): Name of the sector (e.g. ``"heating"``).

        Returns:
            bool: True if the sector is enabled in either config format.
        """
        meta = self.SECTOR_ORDER[sector_name]
        new_key = meta["new_config_key"]
        old_key = meta["config_key"]
        # New-style takes precedence; fall back to old-style
        if self._is_sector_enabled(new_key):
            return True
        return self._is_sector_enabled(old_key)

    def get_chain(self) -> dict:
        """Build the full network preparation chain dict.

        Returns a dictionary describing the suffix for each pipeline stage.
        Backward-compatible keys (``heat_enabled``, ``heat_in``,
        ``heat_out``, ``brownfield_enabled``, ``brownfield_in``,
        ``brownfield_out``, ``solve_in``) are always present.

        Returns:
            dict: Chain configuration with suffix strings for each stage.
        """
        heat_enabled = self._sector_enabled("heating")
        transport_enabled = self._sector_enabled("transport")
        hydrogen_enabled = self._sector_enabled("hydrogen")
        brownfield_enabled = self.cfg.get("existing_capacities", {}).get("add", False)

        chain: dict = {
            "prepare_out": "",
            # --- backward-compat heat keys ---
            "heat_enabled": heat_enabled,
            # --- transport keys ---
            "transport_enabled": transport_enabled,
            # --- hydrogen keys ---
            "hydrogen_enabled": hydrogen_enabled,
            # --- brownfield keys ---
            "brownfield_enabled": brownfield_enabled,
        }

        # Build the ordered suffix progression
        current_suffix = ""

        # Heating
        if heat_enabled:
            chain["heat_in"] = current_suffix
            current_suffix += "-heat"
            chain["heat_out"] = current_suffix
        else:
            chain["heat_in"] = None
            chain["heat_out"] = None

        # Transport
        if transport_enabled:
            chain["transport_in"] = current_suffix
            current_suffix += "-transport"
            chain["transport_out"] = current_suffix
        else:
            chain["transport_in"] = None
            chain["transport_out"] = None

        # Brownfield
        if brownfield_enabled:
            chain["brownfield_in"] = current_suffix
            current_suffix += "-brownfield"
            chain["brownfield_out"] = current_suffix
        else:
            chain["brownfield_in"] = None
            chain["brownfield_out"] = None

        chain["solve_in"] = current_suffix

        return chain


class SectorConfigHandler:
    """Handler for sector-level configuration queries.

    Supports both legacy flat keys and new ``sectors.enabled.*`` format.

    Args:
        config (dict): The workflow configuration dictionary.
    """

    def __init__(self, config: dict):
        self._config = config

    @property
    def enabled_sectors(self) -> list:
        """List of enabled sector names from normalized config.

        Returns:
            list: Sector names (e.g. ``['heating', 'transport']``) that are enabled.
        """
        mgr = WorkflowChainManager(self._config)
        return [name for name in WorkflowChainManager.SECTOR_ORDER if mgr._sector_enabled(name)]

    def is_enabled(self, sector_name: str) -> bool:
        """Check whether a named sector is enabled.

        Args:
            sector_name (str): Sector name (e.g. ``"heating"``).

        Returns:
            bool: True if the sector is enabled.
        """
        return WorkflowChainManager(self._config)._sector_enabled(sector_name)

    def get_config(self, sector_name: str) -> dict:
        """Return the sector-specific sub-config.

        Args:
            sector_name (str): Sector name (e.g. ``"heating"``).

        Returns:
            dict: Sector configuration dict from ``config['sectors'][sector_name]``.

        Raises:
            KeyError: If the sector is not present in config.
        """
        try:
            return self._config["sectors"][sector_name]
        except (KeyError, TypeError):
            raise KeyError(f"Sector '{sector_name}' not found in config['sectors']")


def get_network_prep_chain(cfg):
    """Backward-compatible wrapper. Use WorkflowChainManager directly for new code.

    Determine the network preparation chain based on config settings.

    Returns a dictionary with suffix information for each stage:
    - prepare_out: suffix after prepare_networks rule
    - heat_enabled: whether heating sector is enabled
    - heat_in: suffix for add_heat_sector input (if heat coupling enabled)
    - heat_out: suffix for add_heat_sector output (if heat coupling enabled)
    - transport_enabled: whether transport sector is enabled
    - transport_in: suffix for add_transport_sector input (if enabled)
    - transport_out: suffix for add_transport_sector output (if enabled)
    - brownfield_enabled: whether brownfield addition is enabled
    - brownfield_in: suffix for add_existing_baseyear input (if brownfield enabled)
    - brownfield_out: suffix for add_existing_baseyear output (if brownfield enabled)
    - solve_in: suffix for solve_networks input (final stage)
    """
    return WorkflowChainManager(cfg).get_chain()

# ============== MGA helpers ==================

def sanitize_mga_direction_name(name: str) -> str:
    """Sanitize an MGA direction name for use in file/path names.

    Replaces spaces with underscores and forward/backward slashes with dashes
    so the name is safe to embed in file names and Snakemake wildcards.

    Args:
        name: Raw direction name, e.g. ``"utility solar"`` or ``"BECCS/heat"``.

    Returns:
        Sanitized name, e.g. ``"utility_solar"`` or ``"BECCS-heat"``.
    """
    return name.replace(" ", "_").replace("/", "-").replace("\\", "-")

# ============== HPC helpers ==================

def setup_gurobi_tunnel_and_env(
    tunnel_config: dict, logger: logging.Logger, attempts=4
) -> subprocess.Popen | None:
    """A utility function to set up the Gurobi environment variables and establish an
    SSH tunnel on HPCs. Otherwise the license check will fail if the compute nodes do
     not have internet access or a token server isn't set up

    Args:
        config (dict): the snakemake pypsa-china configuration
        logger (logging.Logger, optional): Logger. Defaults to None.
        attempts (int, optional): ssh connection attemps. Defaults to 4.
    """
    if not tunnel_config.get("use_tunnel", False):
        return
    logger.info("setting up tunnel")
    user = os.getenv("USER")  # User is pulled from the environment
    port = tunnel_config.get("tunnel_port", DEFAULT_TUNNEL_PORT)
    login_node = tunnel_config.get("login_node", LOGIN_NODE)
    timeout = tunnel_config.get("timeout_s", 60)

    # Without -f, SSH stays as a direct child of Popen so socks_proc.kill() reliably
    # terminates it. atexit ensures cleanup even if the script crashes before tunnel.kill().
    ssh_args = [
        "ssh",
        # "-vvv",
        "-N",
        "-D",
        str(port),
        "-o",
        "ServerAliveInterval=60",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        f"ConnectTimeout={timeout}",
        f"{user}@login{login_node}",
    ]
    logger.info(f"Attempting ssh tunnel to login node {login_node}")
    socks_proc = subprocess.Popen(ssh_args, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

    # Brief wait: if SSH exits within 3 s it failed (bad host, permission denied, etc.)
    try:
        socks_proc.wait(timeout=3)
        err = socks_proc.stderr.read().decode()
        if "Permission" in err or "Could not resolve hostname" in err:
            logger.error(f"SSH tunnel failed: {err[:300]}")
        else:
            logger.warning(f"SSH tunnel exited unexpectedly: {err[:300]}")
    except subprocess.TimeoutExpired:
        logger.info("Gurobi SSH tunnel established successfully.")

    # Ensure the tunnel is always cleaned up, even if the script exits via exception
    atexit.register(socks_proc.kill)

    os.environ["https_proxy"] = f"socks5://127.0.0.1:{port}"
    os.environ["SSL_CERT_FILE"] = tunnel_config.get(
        "ssl_cert", "/p/projects/rd3mod/ssl/ca-bundle.pem_2022-02-08"
    )
    os.environ["GRB_CAFILE"] = tunnel_config.get(
        "grb_cafile", "/p/projects/rd3mod/ssl/ca-bundle.pem_2022-02-08"
    )

    # Set up Gurobi environment variables
    # TODO soft code
    os.environ["GUROBI_HOME"] = tunnel_config.get(
        "gurobi_home", "/p/projects/rd3mod/gurobi1103/linux64"
    )
    os.environ["PATH"] += f":{os.environ['GUROBI_HOME']}/bin"
    if "LD_LIBRARY_PATH" in os.environ:
        os.environ["LD_LIBRARY_PATH"] += f":{os.environ['GUROBI_HOME']}/lib"
    os.environ["GRB_LICENSE_FILE"] = tunnel_config.get(
        "license_path", "/p/projects/rd3mod/gurobi_rc/gurobi.lic"
    )
    os.environ["GRB_CURLVERBOSE"] = tunnel_config.get("verbose", "1")
    os.environ["GRB_SERVER_TIMEOUT"] = tunnel_config.get("timeout", "10")

    return socks_proc


def _check_gurobi_license_subprocess() -> bool:
    """
    Subprocess function to check Gurobi license availability.
    This function will start the Gurobi environment to verify if a license is available.

    Returns:
        bool: True if the license check succeeded, False otherwise.
    """
    import gurobipy

    try:
        env = gurobipy.Env(empty=True)
        env.start()  # Start the Gurobi environment (this will attempt to acquire the license)
        logger.info("Gurobi license is available.")
        env.dispose()  # Dispose of the environment after use
        return True
    except gurobipy.GurobiError as e:
        logger.error(f"Error checking Gurobi license: {e}")
        return False


def check_gurobi_license(attempts=1, timeout=10) -> bool:
    """
    Checks the availability of the Gurobi license in a subprocess with timeout.

    Args:
        attempts (int): Number of attempts.
        timeout (int): Time to wait before retrying (in seconds).

    Returns:
        bool: True if the license is available, False if the check times out.
    """
    logger.info("Checking Gurobi license availability...")

    for _ in range(attempts):
        # Create a multiprocessing Process to check license
        process = multiprocessing.Process(target=_check_gurobi_license_subprocess)
        process.start()

        process.join(timeout=timeout)  # Wait for the process to finish or timeout

        if process.is_alive():
            # If the process is still alive after the timeout, terminate it
            process.terminate()
            process.join()  # Ensure it is properly joined to clean up
            logger.warning("License check timeout. Retrying...")
        else:
            # If the process completed, check the result
            if process.exitcode == 0:
                # License was available
                return True
            else:
                # License was not available
                logger.warning("License not available during subprocess check. Retrying...")

    return False


# ==== Misc helpers ======
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


# ====== SNAKEMAKE HELPERS =========
def configure_logging(
    snakemake: object, logger: logging.Logger | None = None, skip_handlers=False, level="INFO"
):
    """Configure the logger or the  behaviour for the logging module.

    Note: Must only be called once from the __main__ section of a script.
    The setup includes printing log messages to STDERR and to a log file defined
    by either (in priority order): snakemake.log.python, snakemake.log[0] or "logs/{rulename}.log".
    Additional keywords from logging.basicConfig are accepted via the snakemake configuration
    file under snakemake.config.logging.

    ISSUE: may not work properly with snakemake logging yaml config [to be solved]

    Args:
        snakemake (object):  snakemake script object
        logger (Logger, optional): the script logger. Defaults to None (Root logger).
            Passing a local logger will apply the configuration to the logger instead of root.
        skip_handlers (bool, optional): Do (not) skip the default handlers
            redirecting output to STDERR and file. Defaults to False.
        level (str, optional): the logging level. Defaults to "INFO".
    """

    if not logger:
        logger = logging.getLogger()
        logger.info("Configuring logging")

    kwargs = snakemake.config.get("logging", dict())
    kwargs.setdefault("level", level)

    if skip_handlers is False:
        fallback_path = Path(__file__).parent.joinpath("../..", "logs", f"{snakemake.rule}.log")
        default_logfile = snakemake.log[0] if snakemake.log else fallback_path
        logfile = snakemake.log.get("python", default_logfile)
        logger.setLevel(kwargs["level"])

        formatter = logging.Formatter("%(asctime)s - %(filename)s - %(levelname)s - %(message)s")

        if not os.path.exists(logfile):
            with open(logfile, "a"):
                pass
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # make running log easier to read
        logger.info("=========== NEW RUN ===========")

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    def handle_exception(exc_type, exc_value, exc_traceback):
        # Log the exception
        logger = logging.getLogger()
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        sys.excepthook = handle_exception

    sys.excepthook = handle_exception


configure_logging_china = configure_logging


def get_cutout_params(config: dict) -> dict:
    """Get the cutout parameters from the config file

    Args:
        config (dict): the snakemake config
    Raises:
        ValueError: if no parameters are found for the cutout name
        FileNotFoundError: if the cutout is not built & build_cutout is disabled
    Returns:
        dict: the cutout parameters
    """
    cutout_name = config["atlite"]["cutout_name"]
    cutout_params = config["atlite"]["cutouts"].get(cutout_name, None)

    if cutout_params is None:
        err = f"No cutout parameters found for {cutout_name}"
        raise ValueError(err + " in config['atlite']['cutouts'].")
    elif not config["enable"]["build_cutout"]:
        cutouts_dir = PathManager(config).cutouts_dir
        is_built = os.path.exists(os.path.join(cutouts_dir, f"{cutout_name}.nc"))
        if not is_built:
            err = f"Cutout {cutout_name} not found in {cutouts_dir}, enable build_cutout"
            raise FileNotFoundError(err)
    return cutout_params


def mock_snakemake(
    rulename: str,
    configfiles: list | str = None,
    snakefile_path: os.PathLike = None,
    **wildcards,
):
    """A function to enable scripts to run as standalone, giving them access to
     the snakefile rule input, outputs etc

    WARNING: only to be used if snakemake is not in globals

    Args:
        rulename (str): the name of the rule
        configfiles (list or str, optional): the config file or config file list. Defaults to None.
        wildcards (optional):  keyword arguments fixing the wildcards (if any needed)

    Raises:
        FileNotFoundError: Config file not found
    Example:
        if "snakemake" not in globals():
            snakemake = mock_snakemake(
                rulename="my_rule",
                configfiles="path/to/config.yaml",
                wildcard1="value1")

    Returns:
        snakemake.script.Snakemake: an object storing all the rule inputs/outputs etc
    """

    import snakemake as sm
    from snakemake.api import Workflow
    from snakemake.common import SNAKEFILE_CHOICES
    from snakemake.script import Snakemake
    from snakemake.settings.types import (
        ConfigSettings,
        DAGSettings,
        OutputSettings,
        ResourceSettings,
        StorageSettings,
        WorkflowSettings,
    )
    from snakemake.logging import LoggerManager, logger as sm_logger

    # horrible hack
    curr_path = os.getcwd()

    if snakefile_path:
        os.chdir(os.path.dirname(snakefile_path))
    try:
        snakefile = None
        for p in SNAKEFILE_CHOICES:
            if os.path.exists(p):
                snakefile = p
                break

        if snakefile is None:
            raise FileNotFoundError("Snakefile not found.")

        if configfiles is None:
            configfiles = []
        elif isinstance(configfiles, str):
            configfiles = [configfiles]

        @dataclass
        class FakeStorageProviderSettings:
            shared_fs_usage: list = field(default_factory=list)

        resource_settings = ResourceSettings()
        config_settings = ConfigSettings(configfiles=map(Path, configfiles))
        workflow_settings = WorkflowSettings()
        storage_settings = StorageSettings()
        dag_settings = DAGSettings(rerun_triggers=[])
        output_settings = OutputSettings()
        logger_manager = LoggerManager(logger=sm_logger, settings=output_settings)
        workflow = Workflow(
            config_settings=config_settings,
            resource_settings=resource_settings,
            workflow_settings=workflow_settings,
            logger_manager=logger_manager,
            storage_settings=storage_settings,
            dag_settings=dag_settings,
            output_settings=output_settings,
            storage_provider_settings={"storageprovider": FakeStorageProviderSettings()},
        )
        # configfiles are already provided to ConfigSettings above and loaded during
        # workflow.include(); calling workflow.configfile() again would re-merge the raw
        # YAML over any transformations done in the Snakefile (e.g. GHGConfigHandler
        # converting "reduction" → "budget_from_reduction"), producing mixed int/string
        # keys and restoring the original control value.
        if configfiles:
            for f in configfiles:
                if not os.path.exists(f):
                    raise FileNotFoundError(f"Config file {f} does not exist.")
        workflow.include(snakefile)

        workflow.global_resources = {}
        rule = workflow.get_rule(rulename)
        dag = sm.dag.DAG(workflow, rules=[rule])
        wc = wildcards
        job = sm.jobs.Job(rule, dag, wc)

        def make_accessible(*ios):
            for io in ios:
                for i, _ in enumerate(io):
                    io[i] = os.path.abspath(io[i])

        make_accessible(job.input, job.output, job.log)
        snakemake = Snakemake(
            job.input,
            job.output,
            job.params,
            job.wildcards,
            job.threads,
            job.resources,
            job.log,
            job.dag.workflow.config,
            job.rule.name,
            None,
        )
        # create log and output dir if not existent
        for path in list(snakemake.log) + list(snakemake.output):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise e
    finally:
        os.chdir(curr_path)

    return snakemake


def set_plot_test_backend(is_test: bool):
    """Hack to set the matplotlib backend to Agg for testing
    Not possible via normal conftest.py since snakemake is a subprocess

    Args:
        is_test (bool): whether to set the backend for testing
    """
    if is_test:
        import matplotlib
        matplotlib.use("Agg")


def setup_proj_environment():
    """Set PROJ_DATA and PROJ_LIB so GDAL/pyproj can find the datum database.

    pyproj 3.x uses PROJ_DATA; PROJ_LIB is the legacy name used by GDAL.
    Infers the path from CONDA_PREFIX or the Python executable location so
    this works both in activated conda environments and bare interpreter calls.
    """
    if "PROJ_DATA" not in os.environ:
        conda_prefix = os.environ.get("CONDA_PREFIX") or str(
            Path(sys.executable).resolve().parent.parent
        )
        proj_data = os.path.join(conda_prefix, "share", "proj")
        if os.path.isdir(proj_data):
            os.environ["PROJ_DATA"] = proj_data
            os.environ.setdefault("PROJ_LIB", proj_data)
            logger.info(f"Set PROJ_DATA to: {proj_data}")
        else:
            logger.warning("CONDA_PREFIX not set, PROJ_LIB may not be configured")
    else:
        logger.info(f"PROJ_LIB already set to: {os.environ['PROJ_LIB']}")


# ====== OSM / PyPSA-Earth HELPERS =========

# Absolute path to the workflow/ directory — used by OSM scripts for config lookup
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Standard column names expected in region GeoDataFrames by the OSM pipeline
REGION_COLS = ["geometry", "name", "x", "y", "country"]

# Recognised NA representations for CSV I/O (excludes "NA"/"na" to avoid confusing Namibia ISO code)
NA_VALUES = ["NULL", "", "N/A", "NAN", "NaN", "nan", "Nan", "n/a", "null"]


def _handle_exception_osm(exc_type, exc_value, exc_traceback):
    """Custom exception hook used by create_logger to capture tracebacks."""
    tb = exc_traceback
    while tb.tb_next:
        tb = tb.tb_next
    flname = tb.tb_frame.f_globals.get("__file__")
    funcname = tb.tb_frame.f_code.co_name
    _log = logging.getLogger()
    if issubclass(exc_type, KeyboardInterrupt):
        _log.error("Manual interruption %r, function %r: %s", flname, funcname, exc_value)
    else:
        _log.error(
            "An error happened in module %r, function %r: %s",
            flname,
            funcname,
            exc_value,
            exc_info=(exc_type, exc_value, exc_traceback),
        )


def create_logger(logger_name, level=logging.INFO):
    """Create a named logger that captures exception tracebacks to the log stream.

    Args:
        logger_name (str): Name for the logger (typically ``__name__``).
        level (int): Logging level. Defaults to ``logging.INFO``.

    Returns:
        logging.Logger: Configured logger instance.
    """
    _logger = logging.getLogger(logger_name)
    _logger.setLevel(level)
    handler = logging.StreamHandler(stream=sys.stdout)
    _logger.addHandler(handler)
    sys.excepthook = _handle_exception_osm
    return _logger


def read_osm_config(*args):
    """Read values from ``config/regions_definition_config.yaml``.

    Locates the config file relative to this module, walking up the directory
    tree if necessary (supports running from a submodule context).

    Args:
        *args (str): One or more top-level keys to retrieve. If a single key is
            given the value is returned directly; multiple keys return a tuple.

    Returns:
        Any | tuple: Value(s) for the requested key(s), or the full config dict
            when called with no arguments.
    """
    import yaml as _yaml

    base_folder = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
    config_path = os.path.join(base_folder, "config", "regions_definition_config.yaml")
    if not os.path.exists(config_path):
        # Walk up one level (script lives inside workflow/scripts/, config is at repo root)
        base_folder = os.path.dirname(os.path.dirname(base_folder))
        config_path = os.path.join(base_folder, "config", "regions_definition_config.yaml")
    with open(config_path) as f:
        osm_config = _yaml.safe_load(f)
    if len(args) == 0:
        return osm_config
    if len(args) == 1:
        return osm_config[args[0]]
    return tuple(osm_config[a] for a in args)


def to_csv_nafix(df, path, **kwargs):
    """Write a DataFrame to CSV using standardised NA representation.

    Args:
        df: pandas DataFrame to write.
        path (str | Path): Output file path.
        **kwargs: Additional keyword arguments passed to ``DataFrame.to_csv``.
    """
    kwargs.pop("na_rep", None)
    if not df.empty or not df.columns.empty:
        df.to_csv(path, **kwargs, na_rep=NA_VALUES[0])
    else:
        open(path, "w").close()


def read_csv_nafix(file, **kwargs):
    """Read a CSV file with standardised NA value handling.

    Args:
        file (str | Path): Path to the CSV file.
        **kwargs: Additional keyword arguments passed to ``pandas.read_csv``.

    Returns:
        pandas.DataFrame: Parsed DataFrame, or an empty DataFrame for zero-byte files.
    """
    import pandas as _pd

    kwargs.setdefault("keep_default_na", False)
    kwargs.setdefault("na_values", NA_VALUES)
    if os.stat(file).st_size > 0:
        return _pd.read_csv(file, **kwargs)
    return _pd.DataFrame()


def save_to_geojson(df, fn):
    """Save a GeoDataFrame to a GeoJSON file, removing any existing file first.

    Writes an empty file when ``df`` is empty (keeps Snakemake output targets valid).

    Args:
        df: GeoDataFrame to save.
        fn (str | Path): Output file path.
    """
    if os.path.exists(fn):
        os.unlink(fn)
    if df.empty:
        open(fn, "w").close()
    else:
        df.to_file(fn, driver="GeoJSON")


def read_geojson(fn, cols=None, dtype=None, crs="EPSG:4326"):
    """Read a GeoJSON file, returning an empty GeoDataFrame for zero-byte files.

    Args:
        fn (str | Path): Path to the GeoJSON file.
        cols (list, optional): Columns for the empty GeoDataFrame fallback.
        dtype (dict, optional): Column dtype overrides applied to the empty fallback.
        crs (str): CRS string for the empty fallback. Defaults to ``"EPSG:4326"``.

    Returns:
        geopandas.GeoDataFrame: Parsed GeoDataFrame, or empty GeoDataFrame on zero-byte file.
    """
    import geopandas as _gpd

    if os.path.getsize(fn) > 0:
        return _gpd.read_file(fn)
    df = _gpd.GeoDataFrame(columns=cols or [], geometry=[], crs=crs)
    if isinstance(dtype, dict):
        for k, v in dtype.items():
            df[k] = df[k].astype(v)
    return df

