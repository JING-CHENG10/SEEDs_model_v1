# -*- coding: utf-8 -*-
"""Full-model endpoint reruns for Strategy maximum reduction potential.

This S5.7 script is based on S5.6, but changes the endpoint definition:

- S5.6 uses CONFIG["endpoint_u_by_kind"] where 0 selects Min_bound and
  1 selects Max_bound from Scenario_config_new.xlsx.
- S5.7 uses ``src/S5_7_strategy_process_config.json`` as the primary strategy
  mapping and endpoint source. Emission-factor rows remain
  ``kind="emission_factor"`` for the model engine, but are assigned to
  process-specific SP_M3a measures for scenario combinations and attribution.

The script reruns the full model path through S4_0_main.run_one_pipeline.
It is not a regression surrogate.

The package contains nine plotting measures. There is no aggregate
``emission_factor`` measure. Enteric fermentation and rice are independent
measures, while manure, fertilizer, and crop-residue/soil measures combine
their management lever with the matching process-specific EF rows.

Default scenarios:
- baseline
- global all strategies at configured endpoints
- global one-at-a-time Strategy endpoints

The mitigation package intentionally excludes land carbon price. Land carbon
price is reserved for the separate S5.7 CDR-price workflow, where it is used to
estimate the reforestation incentive needed to offset residual emissions after
the cost-limited mitigation package has been applied.

Singleton strategy runs use the full result path and write strictly owned
strategy rows to ``cost_summary.csv``. Coalition runs explicitly disable cost
ownership; their Shapley allocation is combined with singleton unit costs to
generate MACC-ready strategy contribution curves without double charging.

Outputs:
- <results>/Strategy_Endpoint_Max_Reduction_Potential/scenario_status.csv
- <results>/Strategy_Endpoint_Max_Reduction_Potential/strategy_design_long.csv
- <results>/Strategy_Endpoint_Max_Reduction_Potential/scenario_reduction_summary.csv
- <results>/Strategy_Endpoint_Max_Reduction_Potential/global_strategy_summary.csv
- <results>/Strategy_Endpoint_Max_Reduction_Potential/global_max_reduction_summary.csv
- <results>/Strategy_Endpoint_Max_Reduction_Potential/global_strategy_potential_normalized.csv
- <results>/Strategy_Endpoint_Max_Reduction_Potential/global_strategy_interaction_residual.csv
- <results>/Strategy_Endpoint_Max_Reduction_Potential/shapley_strategy_decomposition.csv, when enabled
- <results>/Strategy_Endpoint_Max_Reduction_Potential/shapley_coalition_values.csv, when enabled
- <results>/Strategy_Endpoint_Max_Reduction_Potential/shapley_decomposition_status.csv, when enabled
- <results>/Strategy_Endpoint_Max_Reduction_Potential/max_package_shapley_decomposition.csv
- <results>/Strategy_Endpoint_Max_Reduction_Potential/max_package_shapley_status.csv
- <results>/Strategy_Endpoint_Max_Reduction_Potential/macc_strategy_cost_profiles.csv
- <results>/Strategy_Endpoint_Max_Reduction_Potential/macc_strategy_curve_long.csv
- <results>/Strategy_Endpoint_Max_Reduction_Potential/macc_strategy_curve_sp_m3a.xlsx
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config_paths import get_results_base
import S5_4_1_monte_carlo_full_variables as fullmc
import S5_6_1_max_emission_reduction_potential as s56

try:
    from cost_database_v2 import load_cost_database_v2
except ImportError:  # Compatibility with checkouts created before cost DB v2.
    load_cost_database_v2 = None


STRATEGY_KIND_ORDER = [
    "ruminant_reduction",
    "losses_ratio",
    "yield_rate",
    "feed_intensity",
    "enteric_fermentation_management",
    "manure_management",
    "crop_residue_soil_management",
    "rice_cultivation",
    "fertilizer_efficiency",
]


# S5 historically calls the rice lever ``rice_cultivation`` while the v2
# database uses ``rice_management`` as its canonical strategy_kind.  Both map
# to the same (and only) solver/database key, ``Rice``.
STRATEGY_KIND_TO_DATABASE_STRATEGY = {
    "ruminant_reduction": "RuminantReduction",
    "losses_ratio": "LossWaste",
    "yield_rate": "YieldRate",
    "feed_intensity": "FeedEfficiency",
    "enteric_fermentation_management": "EntericF",
    "manure_management": "Manure",
    "crop_residue_soil_management": "Residue",
    "rice_cultivation": "Rice",
    "rice_management": "Rice",
    "fertilizer_efficiency": "Fertilizer",
}
SYSTEM_COST_DATABASE_STRATEGIES = frozenset(
    {"RuminantReduction", "LossWaste", "YieldRate", "FeedEfficiency"}
)

CDR_PRICE_KIND = "land_carbon_price"

STRATEGY_DISPLAY_NAMES = {
    "ruminant_reduction": "Reduce ruminate",
    "losses_ratio": "Reduce waste",
    "yield_rate": "Yield rate",
    "feed_intensity": "Feed efficiency",
    "enteric_fermentation_management": "Enteric fermentation management",
    "manure_management": "Manure management",
    "crop_residue_soil_management": "Crop residue+soil management",
    "rice_cultivation": "Rice cultivation",
    "fertilizer_efficiency": "Fertilizer efficiency",
}

DEFAULT_STRATEGY_PROCESS_CONFIG = (
    Path(__file__).resolve().parents[2] / "src" / "S5_7_strategy_process_config.json"
)


CONFIG = {
    **copy.deepcopy(s56.CONFIG),
    "output_dir": "",  # empty -> <NZF_OUTPUT_DIR>/Strategy_Endpoint_Max_Reduction_Potential
    "runs_subdir": "runs",
    "scenario_prefix": "S5_7",
    "baseline_scenario_id": "S5_7_BASE",
    "global_all_scenario_id": "S5_7_GLOBAL_ALL_ENDPOINT_MAX_REDUCTION",
    "nutrition_profile_sheet": "low_land_new",
    "mc_sheet_prefer": "MC_effect_low_land_new",
    "run_baseline": True,
    "run_global_all_levers": True,
    "run_global_individual_levers": True,
    # S5.7 is Strategy-type potential by default. Country runs can be enabled
    # in CONFIG or with --country-runs.
    "run_country_one_at_a_time": False,
    "run_country_individual_levers": False,
    "detailed_country_accounting": False,
    "fast_emis_only": True,
    "resume": False,
    "clear_existing_run_dirs_when_no_resume": False,
    "eligible_kinds": list(STRATEGY_KIND_ORDER),
    # The process mapping table is the primary source for S5.7 strategy
    # membership, row-specific endpoints, and neutral values. It keeps the
    # source model kind as emission_factor while assigning each EF row to an
    # SP_M3a-compatible strategy category.
    "strategy_process_config_path": str(DEFAULT_STRATEGY_PROCESS_CONFIG),
    "strategy_process_config_strict": True,
    # Fallback values are used only if a mapping row omits endpoint_value.
    "endpoint_value_by_kind": {
        "yield_rate": 1.0,
        "feed_intensity": -0.75,
        "losses_ratio": -0.75,
        "ruminant_reduction": 0.0,
        "enteric_fermentation_management": -0.95,
        "rice_cultivation": -0.95,
        "land_carbon_price": 2000.0,
    },
    # Fallback only for kinds missing from endpoint_value_by_kind.
    "endpoint_u_by_kind": {
        "yield_rate": 1.0,
        "feed_intensity": 0.0,
        "losses_ratio": 0.0,
        "ruminant_reduction": 0.0,
        "enteric_fermentation_management": 0.0,
        "manure_management": 0.0,
        "crop_residue_soil_management": 0.0,
        "rice_cultivation": 0.0,
        "fertilizer_efficiency": 0.0,
        "land_carbon_price": 1.0,
        "land_co2_price": 1.0,
    },
    "decomposition": {
        # "single" keeps the baseline, package, and singleton workflow.
        # "shapley" runs every non-empty coalition of eligible strategies.
        "method": "shapley",
        "run_shapley": True,
        "keep_standard_global_scenarios": False,
        "max_strategies": 12,
    },
    "batch": {
        "enabled": False,
        "batch_index": 1,
        "total_batches": 1,
        "assignment": "round_robin",  # round_robin | contiguous
    },
    "cost_curve": {
        "enabled": True,
        "year": 2080,
        "cost_levels_usd_per_tco2eq": [0, 10, 20, 30, 40, 50, 100, 200, 300, 500, 700, 900, 1000],
        "detailed_singleton_runs": True,
        "detailed_full_package_run": False,
        "minimum_positive_abatement_tco2eq": 1e-6,
        "bioenergy_case": "configured_case",
        "write_excel": True,
        "excel_filename": "macc_strategy_curve_sp_m3a.xlsx",
    },
    "cdr_price": {
        "kind": CDR_PRICE_KIND,
        "target_emissions_gt": 0.0,
        "land_price_grid_usd_per_tco2eq": [
            0,
            10,
            20,
            30,
            40,
            50,
            75,
            100,
            150,
            200,
            300,
            500,
            750,
            1000,
            1500,
            2000,
            3000,
            5000,
        ],
        "partial_baseline_value_by_kind": {
            "yield_rate": 0.0,
            "feed_intensity": 0.0,
            "losses_ratio": 0.0,
            "ruminant_reduction": "max_bound",
            "enteric_fermentation_management": 0.0,
            "manure_management": 0.0,
            "crop_residue_soil_management": 0.0,
            "rice_cultivation": 0.0,
            "fertilizer_efficiency": 0.0,
        },
    },
    "override_cfg": {
        **copy.deepcopy(s56.CONFIG.get("override_cfg", {}) or {}),
        "nutrition_profile_sheet": "low_land_new",
        "cost_calculation_method": "unit_cost",
        "debug_level": 0,
        "batch_mode": False,
        "linear_enable_infeasible_iis": False,
        "linear_enable_violation_iis": False,
        "linear_enable_output_diagnostics": False,
        "linear_enable_verbose_logging": False,
    },
    "write_summary_outputs": True,
}


def _default_output_dir() -> Path:
    return Path(get_results_base()) / "Strategy_Endpoint_Max_Reduction_Potential"


def _output_dir(cfg: Mapping[str, object]) -> Path:
    raw = str(cfg.get("output_dir", "") or "").strip()
    return Path(raw) if raw else _default_output_dir()


def _runs_dir(cfg: Mapping[str, object]) -> Path:
    return _output_dir(cfg) / str(cfg.get("runs_subdir", "runs") or "runs")


def _database_strategy_for_kind(kind: object) -> str:
    return STRATEGY_KIND_TO_DATABASE_STRATEGY.get(str(kind or "").strip(), "")


def _cost_component_type_for_database_strategy(database_strategy: object) -> str:
    return (
        "strategy"
        if str(database_strategy or "").strip() in SYSTEM_COST_DATABASE_STRATEGIES
        else "process"
    )


def _active_strategy_cost_keys(plan: s56.ScenarioPlan) -> Tuple[str, ...]:
    """Return one database key only for a strict singleton scenario."""
    if len(tuple(plan.include_kinds)) != 1:
        return ()
    database_strategy = _database_strategy_for_kind(plan.include_kinds[0])
    return (database_strategy,) if database_strategy else ()


def _mitigation_cost_database_path(paths: object) -> Path:
    configured = getattr(paths, "mitigation_cost_database_path", "")
    if configured:
        return Path(str(configured))
    base = Path(str(getattr(paths, "base", "") or ""))
    return base / "Price_Cost" / "Cost" / "food_mitigation_cost_database_v2.0.json"


@lru_cache(maxsize=4)
def _load_cost_database_identity(path_text: str) -> Dict[str, str]:
    path = Path(path_text)
    identity = {
        "cost_database_path": str(path),
        "cost_database_version": "",
        "cost_database_schema_version": "",
        "cost_database_sha256": "",
        "cost_database_source_sha256": "",
        "cost_database_error": "",
    }
    try:
        if load_cost_database_v2 is None:
            raise ImportError("cost_database_v2 is unavailable")
        database = load_cost_database_v2(path)
        identity.update(
            {
                "cost_database_version": str(database.version or ""),
                "cost_database_schema_version": str(database.schema_version or ""),
                "cost_database_sha256": str(database.sha256 or ""),
                "cost_database_source_sha256": str(database.source_sha256 or ""),
            }
        )
        return identity
    except Exception as exc:
        # Keep dry-run/status tooling usable while still recording a mismatch.
        identity["cost_database_error"] = f"{type(exc).__name__}: {exc}"

    if not path.is_file():
        return identity
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        identity["cost_database_sha256"] = digest.hexdigest()
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {}) if isinstance(payload, Mapping) else {}
        if isinstance(metadata, Mapping):
            identity["cost_database_version"] = str(
                metadata.get("database_version", metadata.get("version", "")) or ""
            )
            identity["cost_database_schema_version"] = str(
                metadata.get("schema_version", "") or ""
            )
            identity["cost_database_source_sha256"] = str(
                metadata.get("source_sha256", "") or ""
            )
    except Exception as exc:
        identity["cost_database_error"] = f"{type(exc).__name__}: {exc}"
    return identity


def _cost_database_identity(paths: object) -> Dict[str, str]:
    return dict(_load_cost_database_identity(str(_mitigation_cost_database_path(paths))))


def _cost_resume_fingerprint(
    active_keys: Sequence[str],
    identity: Mapping[str, object],
    reference_scenario_id: str,
    *,
    strategy_cost_regions: Optional[Sequence[str]] = None,
    design_signature: str = "",
) -> str:
    payload = {
        "active_strategy_cost_keys": list(active_keys),
        "strategy_cost_regions": list(strategy_cost_regions or ()),
        "cost_database_version": str(identity.get("cost_database_version", "") or ""),
        "cost_database_sha256": str(identity.get("cost_database_sha256", "") or ""),
        "reference_scenario_id": str(reference_scenario_id or ""),
        "attribution_method": "strict_singleton" if active_keys else "none",
        "design_signature": str(design_signature or ""),
    }
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _cost_design_signature(
    plan: s56.ScenarioPlan,
    param_rows: Sequence[Mapping[str, object]],
) -> str:
    canonical_rows = sorted(
        json.dumps(
            dict(row),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        for row in param_rows
    )
    payload = {
        "scenario_id": str(plan.scenario_id),
        "scope": str(plan.scope),
        "strategy_name": str(plan.strategy_name),
        "include_kinds": list(plan.include_kinds),
        "country": str(plan.country or ""),
        "param_rows": canonical_rows,
    }
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _decorate_cost_status(
    status: Dict[str, object],
    plan: s56.ScenarioPlan,
    identity: Mapping[str, object],
    reference_scenario_id: str,
    *,
    strategy_cost_regions: Optional[Sequence[str]] = None,
    design_signature: str = "",
) -> Dict[str, object]:
    active_keys = _active_strategy_cost_keys(plan)
    status.update({key: value for key, value in identity.items()})
    status["active_strategy_cost_keys"] = "|".join(active_keys)
    status["cost_attribution_method"] = "strict_singleton" if active_keys else "none"
    status["cost_reference_scenario_id"] = str(reference_scenario_id or "")
    status["cost_strategy_regions"] = "|".join(strategy_cost_regions or ())
    status["cost_design_signature"] = str(design_signature or "")
    status["cost_resume_fingerprint"] = _cost_resume_fingerprint(
        active_keys,
        identity,
        reference_scenario_id,
        strategy_cost_regions=strategy_cost_regions,
        design_signature=design_signature,
    )
    return status


def _summary_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _cost_summary_matches_singleton(
    scenario_dir: Path,
    *,
    database_strategy: str,
    identity: Mapping[str, object],
    reference_scenario_id: str,
    strategy_cost_regions: Optional[Sequence[str]] = None,
) -> bool:
    expected_version = str(identity.get("cost_database_version", "") or "").strip()
    expected_sha256 = str(identity.get("cost_database_sha256", "") or "").strip()
    if (
        str(identity.get("cost_database_error", "") or "").strip()
        or not expected_version
        or not expected_sha256
    ):
        # Failure to identify the current database must never weaken resume
        # validation into accepting an arbitrary historical cost file.
        return False
    path = _cost_summary_path(scenario_dir)
    try:
        detail = pd.read_csv(path)
    except Exception:
        return False
    required = {
        "cost_component_type",
        "database_strategy",
        "is_priced",
        "attribution_method",
    }
    if detail.empty or not required.issubset(detail.columns):
        return False
    priced = detail[_summary_bool(detail["is_priced"])].copy()
    priced_owners = set(
        priced["database_strategy"].fillna("").astype(str).str.strip()
    )
    if priced_owners != {database_strategy}:
        # Legacy process-mode solves can contain the requested key alongside
        # four other priced owners.  Such a solution is not a strict
        # singleton and must not be resumed under the v2 attribution contract.
        return False
    if strategy_cost_regions:
        expected_regions = {
            s56._normalize_m49(region) for region in strategy_cost_regions
        }
        actual_regions = {
            s56._normalize_m49(region)
            for region in priced.get("region", pd.Series(dtype=str)).tolist()
        }
        if actual_regions != expected_regions:
            return False
    work = detail[
        detail["cost_component_type"].astype(str).str.strip().str.lower().eq(
            _cost_component_type_for_database_strategy(database_strategy)
        )
        & detail["database_strategy"].astype(str).str.strip().eq(database_strategy)
        & _summary_bool(detail["is_priced"])
        & detail["attribution_method"].fillna("").astype(str).str.strip().eq(
            "singleton_incremental_vs_reference"
        )
    ].copy()
    if work.empty:
        return False
    expected_fields = {
        "cost_database_version": expected_version,
        "cost_database_sha256": expected_sha256,
        "reference_scenario_id": str(reference_scenario_id or ""),
    }
    for column, expected in expected_fields.items():
        if not expected:
            continue
        if column not in work.columns:
            return False
        values = work[column].fillna("").astype(str).str.strip()
        if values.empty or not values.eq(expected).all():
            return False
    return True


def _strategy_process_config_path(cfg: Mapping[str, object]) -> Path:
    raw = str(cfg.get("strategy_process_config_path", "") or "").strip()
    path = Path(raw) if raw else DEFAULT_STRATEGY_PROCESS_CONFIG
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return path.resolve()


def _clean_config_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


@lru_cache(maxsize=8)
def _load_strategy_process_mappings(path_text: str) -> Tuple[Dict[str, object], ...]:
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"Missing S5.7 strategy process config: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_rows = payload.get("mappings") if isinstance(payload, Mapping) else None
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError(f"S5.7 strategy process config has no mapping rows: {path}")

    rows: List[Dict[str, object]] = []
    required = {
        "strategy_kind",
        "strategy_display_name",
        "source_element",
        "source_process",
        "endpoint_value",
        "neutral_value",
        "include_in_package",
        "role",
    }
    for index, raw in enumerate(raw_rows, start=1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"S5.7 strategy mapping row {index} is not an object.")
        missing = sorted(required.difference(raw))
        if missing:
            raise ValueError(f"S5.7 strategy mapping row {index} is missing fields: {missing}")
        row = dict(raw)
        row["strategy_kind"] = _clean_config_text(row.get("strategy_kind"))
        row["strategy_display_name"] = _clean_config_text(row.get("strategy_display_name"))
        row["source_element"] = _clean_config_text(row.get("source_element")).lower()
        row["source_process"] = _clean_config_text(row.get("source_process"))
        row["include_in_package"] = bool(row.get("include_in_package"))
        row["role"] = _clean_config_text(row.get("role"))
        if not row["strategy_kind"] or not row["source_element"]:
            raise ValueError(f"S5.7 strategy mapping row {index} has an empty key.")
        endpoint = row.get("endpoint_value")
        if endpoint is not None:
            endpoint_value = float(endpoint)
            if not np.isfinite(endpoint_value):
                raise ValueError(f"S5.7 strategy mapping row {index} has a non-finite endpoint.")
            row["endpoint_value"] = endpoint_value
        neutral = row.get("neutral_value")
        if isinstance(neutral, str):
            neutral_key = neutral.strip().lower()
            if neutral_key not in {"min_bound", "max_bound"}:
                row["neutral_value"] = float(neutral)
            else:
                row["neutral_value"] = neutral_key
        else:
            row["neutral_value"] = float(neutral)
        rows.append(row)

    package_kinds = {
        str(row["strategy_kind"])
        for row in rows
        if bool(row["include_in_package"])
    }
    if "emission_factor" in package_kinds:
        raise ValueError("S5.7 strategy config cannot use aggregate emission_factor as a package strategy.")
    missing_package_kinds = sorted(set(STRATEGY_KIND_ORDER).difference(package_kinds))
    extra_package_kinds = sorted(package_kinds.difference(STRATEGY_KIND_ORDER))
    if missing_package_kinds or extra_package_kinds:
        raise ValueError(
            "S5.7 strategy config does not match STRATEGY_KIND_ORDER: "
            f"missing={missing_package_kinds}, extra={extra_package_kinds}"
        )
    for kind in STRATEGY_KIND_ORDER:
        displays = {
            str(row["strategy_display_name"])
            for row in rows
            if row["strategy_kind"] == kind
        }
        expected = STRATEGY_DISPLAY_NAMES[kind]
        if displays != {expected}:
            raise ValueError(
                f"S5.7 display name mismatch for {kind}: config={sorted(displays)}, expected={expected}"
            )
    return tuple(rows)


def _prepare_strategy_specs(
    specs: pd.DataFrame,
    cfg: Mapping[str, object],
) -> pd.DataFrame:
    mappings = _load_strategy_process_mappings(str(_strategy_process_config_path(cfg)))
    out = specs.copy().reset_index(drop=True)
    strategy_kinds: List[str] = []
    display_names: List[str] = []
    endpoint_values: List[object] = []
    neutral_values: List[object] = []
    include_flags: List[bool] = []
    roles: List[str] = []
    unmapped: List[str] = []

    for index, row in out.iterrows():
        source_element = _clean_config_text(row.get("__kind", row.get("Element", ""))).lower()
        source_process = _clean_config_text(row.get("Process", ""))
        candidates = [
            mapping
            for mapping in mappings
            if mapping["source_element"] == source_element
            and str(mapping["source_process"]) in {"*", source_process}
        ]
        exact = [
            mapping
            for mapping in candidates
            if str(mapping["source_process"]) == source_process
        ]
        selected = exact if exact else [
            mapping
            for mapping in candidates
            if str(mapping["source_process"]) == "*"
        ]
        if len(selected) != 1:
            unmapped.append(
                f"row={index + 1}, element={source_element}, process={source_process or '<blank>'}, matches={len(selected)}"
            )
            strategy_kinds.append("")
            display_names.append("")
            endpoint_values.append(np.nan)
            neutral_values.append(np.nan)
            include_flags.append(False)
            roles.append("unmapped")
            continue
        mapping = selected[0]
        strategy_kinds.append(str(mapping["strategy_kind"]))
        display_names.append(str(mapping["strategy_display_name"]))
        endpoint_values.append(mapping.get("endpoint_value"))
        neutral_values.append(mapping.get("neutral_value"))
        include_flags.append(bool(mapping.get("include_in_package")))
        roles.append(str(mapping.get("role", "")))

    if unmapped and bool(cfg.get("strategy_process_config_strict", True)):
        details = "; ".join(unmapped[:20])
        raise ValueError(f"Unmapped or ambiguous S5.7 strategy rows: {details}")

    out["__strategy_kind"] = strategy_kinds
    out["__strategy_display_name"] = display_names
    out["__strategy_endpoint_value"] = endpoint_values
    out["__strategy_neutral_value"] = neutral_values
    out["__strategy_include_in_package"] = include_flags
    out["__strategy_role"] = roles
    return out


def _eligible_strategy_kinds(
    cfg: Mapping[str, object],
    specs: pd.DataFrame,
) -> Tuple[str, ...]:
    configured = _ordered_strategy_kinds(
        [str(k) for k in (cfg.get("eligible_kinds") or STRATEGY_KIND_ORDER)]
    )
    available = set(
        specs.loc[
            specs["__strategy_include_in_package"].astype(bool),
            "__strategy_kind",
        ].astype(str)
    )
    return tuple(kind for kind in configured if kind in available)


def _configured_endpoint_value(kind: str, cfg: Mapping[str, object]) -> Optional[float]:
    raw_map = cfg.get("endpoint_value_by_kind") or {}
    if not isinstance(raw_map, Mapping):
        return None
    candidates = [kind]
    if kind == "land_co2_price":
        candidates.append("land_carbon_price")
    if kind == "land_carbon_price":
        candidates.append("land_co2_price")
    for key in candidates:
        if key not in raw_map:
            continue
        raw = raw_map.get(key)
        if raw is None or (isinstance(raw, str) and raw.strip().lower() in {"", "none", "null"}):
            return None
        val = float(raw)
        if not np.isfinite(val):
            raise ValueError(f"CONFIG['endpoint_value_by_kind'][{key!r}] is not finite: {raw!r}")
        return val
    return None


def _configured_endpoint_value_for_row(
    row: Mapping[str, object],
    cfg: Mapping[str, object],
) -> Optional[float]:
    raw = row.get("strategy_endpoint_value")
    if raw is not None and not pd.isna(raw):
        value = float(raw)
        if not np.isfinite(value):
            raise ValueError(f"Non-finite strategy endpoint value for row: {raw!r}")
        return value
    strategy_kind = str(row.get("strategy_kind", "") or "")
    source_kind = str(row.get("kind", "") or "")
    strategy_value = _configured_endpoint_value(strategy_kind, cfg)
    if strategy_value is not None:
        return strategy_value
    return _configured_endpoint_value(source_kind, cfg)


def _endpoint_u(kind: str, cfg: Mapping[str, object]) -> float:
    endpoint_map = cfg.get("endpoint_u_by_kind") or {}
    raw = endpoint_map.get(kind, cfg.get("unknown_kind_u", 0.5))
    try:
        val = float(raw)
    except Exception:
        val = 0.5
    return max(0.0, min(1.0, val))


def _endpoint_label_for_row(
    row: Mapping[str, object],
    cfg: Mapping[str, object],
    u: float,
) -> Tuple[str, str]:
    configured = _configured_endpoint_value_for_row(row, cfg)
    if configured is not None:
        raw = row.get("strategy_endpoint_value")
        if raw is not None and not pd.isna(raw):
            return "CONFIG_endpoint_value", "strategy_process_config"
        return "CONFIG_endpoint_value", "endpoint_value_by_kind"
    return s56._endpoint_label(u), "endpoint_u_by_kind"


def _build_param_rows(
    specs: pd.DataFrame,
    cfg: Mapping[str, object],
    *,
    include_kinds: Sequence[str],
    country: Optional[str],
) -> List[Dict[str, object]]:
    """Build ScenarioEffect parameter rows using configured actual endpoints."""
    include = {str(k) for k in include_kinds}
    work = specs[specs["__strategy_kind"].astype(str).isin(include)].copy().reset_index(drop=True)
    if work.empty:
        return []

    unit_row = np.array(
        [_endpoint_u(str(k), cfg) for k in work["__strategy_kind"].astype(str)],
        dtype=float,
    )
    sampling_cfg = copy.deepcopy(cfg.get("sampling", {}) or {})
    q_bounds = sampling_cfg.get("quantile_bounds", (0.0, 1.0))
    try:
        q_bounds = (float(q_bounds[0]), float(q_bounds[1]))
    except Exception:
        q_bounds = (0.0, 1.0)

    rows = fullmc._draw_mc_param_rows(
        work,
        unit_row=unit_row,
        quantile_bounds=q_bounds,
        sampling_cfg=sampling_cfg,
    )
    rows = fullmc._attach_param_metadata(rows, work)

    for row, (_, spec_row) in zip(rows, work.iterrows()):
        source_kind = str(row.get("kind", "") or "")
        strategy_kind = str(spec_row.get("__strategy_kind", "") or "")
        row["source_kind"] = source_kind
        row["strategy_kind"] = strategy_kind
        row["strategy_display_name"] = str(
            spec_row.get("__strategy_display_name", STRATEGY_DISPLAY_NAMES.get(strategy_kind, strategy_kind))
        )
        row["strategy_endpoint_value"] = spec_row.get("__strategy_endpoint_value")
        row["strategy_neutral_value"] = spec_row.get("__strategy_neutral_value")
        row["strategy_role"] = spec_row.get("__strategy_role")

        u = _endpoint_u(strategy_kind, cfg)
        configured_value = _configured_endpoint_value_for_row(row, cfg)
        endpoint_label, endpoint_source = _endpoint_label_for_row(row, cfg, u)

        if configured_value is not None:
            row["abs_value"] = configured_value
            row["continuous_draw"] = configured_value
            lo = row.get("min_bound")
            hi = row.get("max_bound")
            try:
                lo_f = float(lo)
                hi_f = float(hi)
                row["mc_u"] = (configured_value - lo_f) / (hi_f - lo_f) if hi_f != lo_f else np.nan
            except Exception:
                row["mc_u"] = np.nan

        if country is not None:
            row["region"] = country
            row["region_selector"] = country

        row["strategy_endpoint_u"] = u
        row["strategy_endpoint"] = endpoint_label
        row["strategy_endpoint_source"] = endpoint_source
        row["configured_endpoint_value"] = configured_value
    return rows


def _build_effects(
    param_rows: List[Dict[str, object]],
    universe,
    cfg: Mapping[str, object],
    *,
    scenario_id: str,
) -> List[object]:
    effects = fullmc._build_scenario_effects(
        param_rows,
        universe,
        scenario_id=scenario_id,
        mc_y2020_mode=str(cfg.get("mc_non_ef_mode", "shared") or "shared"),
        mc_mode_non_ef=str(cfg.get("mc_non_ef_mode", "shared") or "shared"),
        mc_mode_ef=str(cfg.get("mc_ef_mode", "shared") or "shared"),
        ef_process_mode=str(cfg.get("ef_process_mode", "all") or "all"),
    )
    return fullmc._attach_effect_metadata(effects, param_rows)


def _strategy_design_rows(
    plan: s56.ScenarioPlan,
    param_rows: List[Dict[str, object]],
    universe,
) -> List[Dict[str, object]]:
    country_info = s56._country_label(plan.country, universe) if plan.country else {}
    rows: List[Dict[str, object]] = []
    for row in param_rows:
        rows.append(
            {
                "scenario_id": plan.scenario_id,
                "scope": plan.scope,
                "strategy_name": plan.strategy_name,
                "country": country_info.get("country", ""),
                "iso3": country_info.get("iso3", ""),
                "country_name": country_info.get("country_name", ""),
                "region_aggMC": country_info.get("region_aggMC", ""),
                "spec_row_id": row.get("spec_row_id"),
                "kind": row.get("strategy_kind"),
                "database_strategy": _database_strategy_for_kind(row.get("strategy_kind")),
                "strategy_display_name": row.get("strategy_display_name"),
                "source_kind": row.get("kind"),
                "element_name": row.get("element_name"),
                "element_unit": row.get("element_unit"),
                "item_selector": row.get("item_selector"),
                "process_selector": row.get("process_selector"),
                "ghg_selector": row.get("ghg_selector"),
                "region_selector": row.get("region_selector"),
                "strategy_endpoint": row.get("strategy_endpoint"),
                "strategy_endpoint_source": row.get("strategy_endpoint_source"),
                "strategy_endpoint_u": row.get("strategy_endpoint_u"),
                "configured_endpoint_value": row.get("configured_endpoint_value"),
                "strategy_neutral_value": row.get("strategy_neutral_value"),
                "strategy_role": row.get("strategy_role"),
                "value_2080": row.get("abs_value"),
                "min_bound": row.get("min_bound"),
                "max_bound": row.get("max_bound"),
                "mc_u": row.get("mc_u"),
                "q_low": row.get("q_low"),
                "q_high": row.get("q_high"),
            }
        )
    return rows


def _decomposition_cfg(cfg: Mapping[str, object]) -> Mapping[str, object]:
    raw = cfg.get("decomposition") or {}
    return raw if isinstance(raw, Mapping) else {}


def _shapley_enabled(cfg: Mapping[str, object]) -> bool:
    decomp = _decomposition_cfg(cfg)
    method = str(decomp.get("method", "") or "").strip().lower()
    return bool(decomp.get("run_shapley", False)) or method == "shapley"


def _shapley_keep_standard(cfg: Mapping[str, object]) -> bool:
    return bool(_decomposition_cfg(cfg).get("keep_standard_global_scenarios", False))


def _cost_curve_cfg(cfg: Mapping[str, object]) -> Mapping[str, object]:
    raw = cfg.get("cost_curve") or {}
    return raw if isinstance(raw, Mapping) else {}


def _cost_curve_enabled(cfg: Mapping[str, object]) -> bool:
    return bool(_cost_curve_cfg(cfg).get("enabled", False))


def _requires_detailed_cost_output(
    cfg: Mapping[str, object],
    include_kinds: Sequence[str],
    eligible: Sequence[str],
) -> bool:
    if not _cost_curve_enabled(cfg):
        return False
    include_count = len(tuple(include_kinds))
    cost_cfg = _cost_curve_cfg(cfg)
    if include_count == 1 and bool(cost_cfg.get("detailed_singleton_runs", True)):
        return True
    if include_count == len(tuple(eligible)) and include_count > 1:
        return bool(cost_cfg.get("detailed_full_package_run", False))
    return False


def _ordered_strategy_kinds(kinds: Sequence[str]) -> Tuple[str, ...]:
    available = {str(k) for k in kinds}
    ordered = [kind for kind in STRATEGY_KIND_ORDER if kind in available]
    extras = sorted(k for k in available if k not in set(ordered))
    return tuple(ordered + extras)


def _shapley_mask_width(n_kinds: int) -> int:
    return max(2, len(f"{(1 << max(n_kinds, 1)) - 1:X}"))


def _shapley_scenario_id(prefix: str, mask: int, n_kinds: int) -> str:
    return f"{prefix}_SHAPLEY_{mask:0{_shapley_mask_width(n_kinds)}X}"


def _build_shapley_plans(
    cfg: Mapping[str, object],
    eligible: Sequence[str],
    *,
    detailed: bool,
) -> List[s56.ScenarioPlan]:
    kinds = _ordered_strategy_kinds(eligible)
    n = len(kinds)
    max_n = int(_decomposition_cfg(cfg).get("max_strategies", 12) or 12)
    if n <= 0:
        return []
    if n > max_n:
        raise ValueError(
            f"Shapley decomposition would require 2^{n} coalition scenarios. "
            f"CONFIG['decomposition']['max_strategies'] is {max_n}."
        )

    prefix = str(cfg.get("scenario_prefix", "S5_7") or "S5_7")
    plans: List[s56.ScenarioPlan] = []
    for mask in range(1, 1 << n):
        include = tuple(kinds[idx] for idx in range(n) if mask & (1 << idx))
        detailed_cost = _requires_detailed_cost_output(cfg, include, kinds)
        plans.append(
            s56.ScenarioPlan(
                scenario_id=_shapley_scenario_id(prefix, mask, n),
                scope="global",
                strategy_name=f"shapley_coalition_{mask:0{_shapley_mask_width(n)}X}",
                include_kinds=include,
                fast_emis_only=not (detailed or detailed_cost),
                require_country_detail=detailed,
            )
        )
    return plans


def _select_batch_items(
    items: Sequence[object],
    *,
    batch_count: int,
    batch_index: int,
    assignment: str,
) -> List[Tuple[int, object]]:
    """Return one-based original indices and items assigned to this batch."""
    if batch_count <= 1:
        return [(idx, item) for idx, item in enumerate(items, start=1)]
    if batch_index < 1 or batch_index > batch_count:
        raise ValueError(f"batch_index must be within 1..{batch_count}, got {batch_index}")
    assignment_l = str(assignment or "round_robin").strip().lower()
    n = len(items)
    if n == 0:
        return []
    if assignment_l in {"contiguous", "block", "blocks"}:
        start = (n * (batch_index - 1)) // batch_count
        end = (n * batch_index) // batch_count
        return [(idx + 1, item) for idx, item in enumerate(items[start:end], start=start)]
    if assignment_l in {"round_robin", "round-robin", "rr"}:
        return [
            (idx, item)
            for idx, item in enumerate(items, start=1)
            if ((idx - 1) % batch_count) == (batch_index - 1)
        ]
    raise ValueError("assignment must be 'round_robin' or 'contiguous'")


def _batch_config(cfg: Mapping[str, object]) -> Mapping[str, object]:
    raw = cfg.get("batch") or {}
    return raw if isinstance(raw, Mapping) else {}


def _batch_tag(batch_index: int, total_batches: int) -> str:
    return f"batch_{int(batch_index):02d}_of_{int(total_batches):02d}"


def _apply_batch_plan_filter(
    plans: Sequence[s56.ScenarioPlan],
    cfg: Mapping[str, object],
) -> Tuple[List[s56.ScenarioPlan], Dict[str, object], Dict[str, int]]:
    batch = _batch_config(cfg)
    total = int(batch.get("total_batches", 1) or 1)
    index = int(batch.get("batch_index", 1) or 1)
    assignment = str(batch.get("assignment", "round_robin") or "round_robin")
    enabled = bool(batch.get("enabled", False)) and total > 1
    if not enabled:
        return list(plans), {
            "enabled": False,
            "batch_index": 1,
            "total_batches": 1,
            "batch_tag": "",
            "assignment": assignment,
            "full_plan_count": len(plans),
            "selected_plan_count": len(plans),
        }, {plan.scenario_id: idx for idx, plan in enumerate(plans, start=1)}

    selected = _select_batch_items(plans, batch_count=total, batch_index=index, assignment=assignment)
    # Every isolated batch writes into its own runs directory.  Prepend the
    # full-output reference plan when round-robin/contiguous assignment placed
    # it in another batch, so each priced singleton has a local BASE ledger.
    baseline_entry = next(
        (
            (plan_index, plan)
            for plan_index, plan in enumerate(plans, start=1)
            if plan.scope == "baseline"
        ),
        None,
    )
    selected_ids = {plan.scenario_id for _, plan in selected}
    if baseline_entry is not None and baseline_entry[1].scenario_id not in selected_ids:
        selected = [baseline_entry, *selected]
    selected_plans = [plan for _, plan in selected]
    plan_indices = {plan.scenario_id: idx for idx, plan in selected}
    tag = _batch_tag(index, total)
    return selected_plans, {
        "enabled": True,
        "batch_index": index,
        "total_batches": total,
        "batch_tag": tag,
        "assignment": assignment,
        "full_plan_count": len(plans),
        "selected_plan_count": len(selected_plans),
    }, plan_indices


def _build_plans(cfg: Mapping[str, object], universe, specs: pd.DataFrame) -> List[s56.ScenarioPlan]:
    eligible = _eligible_strategy_kinds(cfg, specs)
    country_eligible = tuple(k for k in eligible if k not in s56.GLOBAL_ONLY_KINDS)
    prefix = str(cfg.get("scenario_prefix", "S5_7") or "S5_7")
    detailed = bool(cfg.get("detailed_country_accounting", False))
    run_shapley = _shapley_enabled(cfg)
    keep_standard = _shapley_keep_standard(cfg)
    plans: List[s56.ScenarioPlan] = []

    if bool(cfg.get("run_baseline", True)):
        plans.append(
            s56.ScenarioPlan(
                scenario_id=str(cfg.get("baseline_scenario_id") or f"{prefix}_BASE"),
                scope="baseline",
                strategy_name="baseline",
                include_kinds=(),
                # Every priced singleton requires the process-level BASE
                # emissions ledger.  The fast path omits that file, so the
                # dedicated S5.7 reference must always run the full output
                # contract even when country detail is not requested.
                fast_emis_only=False,
                require_country_detail=detailed,
            )
        )

    if bool(cfg.get("run_global_all_levers", True)) and (not run_shapley or keep_standard):
        detailed_cost = _requires_detailed_cost_output(cfg, eligible, eligible)
        plans.append(
            s56.ScenarioPlan(
                scenario_id=str(cfg.get("global_all_scenario_id") or f"{prefix}_GLOBAL_ALL_ENDPOINT_MAX_REDUCTION"),
                scope="global",
                strategy_name="all_levers_configured_endpoint_max_reduction",
                include_kinds=eligible,
                fast_emis_only=not (detailed or detailed_cost),
                require_country_detail=detailed,
            )
        )

    if bool(cfg.get("run_global_individual_levers", True)) and (not run_shapley or keep_standard):
        for kind in eligible:
            detailed_cost = _requires_detailed_cost_output(cfg, (kind,), eligible)
            plans.append(
                s56.ScenarioPlan(
                    scenario_id=f"{prefix}_GLOBAL_{s56._safe_token(kind).upper()}",
                    scope="global",
                    strategy_name=f"single_lever_configured_endpoint_{kind}",
                    include_kinds=(kind,),
                    fast_emis_only=bool(cfg.get("fast_emis_only", True)) and not detailed_cost,
                    require_country_detail=False,
                )
            )

    if run_shapley:
        plans.extend(_build_shapley_plans(cfg, eligible, detailed=detailed))

    countries = s56._active_countries(cfg, universe)
    if bool(cfg.get("run_country_one_at_a_time", False)):
        for country in countries:
            suffix = s56._country_scenario_suffix(country, universe)
            plans.append(
                s56.ScenarioPlan(
                    scenario_id=f"{prefix}_CTRY_{suffix}_ALL",
                    scope="country",
                    strategy_name="country_own_configured_endpoint_strategy",
                    include_kinds=country_eligible,
                    country=country,
                    fast_emis_only=bool(cfg.get("fast_emis_only", True)),
                    require_country_detail=False,
                )
            )

    if bool(cfg.get("run_country_individual_levers", False)):
        for country in countries:
            suffix = s56._country_scenario_suffix(country, universe)
            for kind in country_eligible:
                plans.append(
                    s56.ScenarioPlan(
                        scenario_id=f"{prefix}_CTRY_{suffix}_{s56._safe_token(kind).upper()}",
                        scope="country",
                        strategy_name=f"country_single_lever_configured_endpoint_{kind}",
                        include_kinds=(kind,),
                        country=country,
                        fast_emis_only=bool(cfg.get("fast_emis_only", True)),
                        require_country_detail=False,
                    )
                )
    return plans


def _cost_summary_path(scenario_dir: Path) -> Path:
    return scenario_dir / "cost_summary.csv"


def _can_resume_plan(
    plan: s56.ScenarioPlan,
    scenario_dir: Path,
    *,
    cost_database_identity: Optional[Mapping[str, object]] = None,
    reference_scenario_id: str = "",
    strategy_cost_regions: Optional[Sequence[str]] = None,
    expected_resume_fingerprint: Optional[str] = None,
) -> bool:
    active_keys = _active_strategy_cost_keys(plan)
    identity = dict(cost_database_identity or {})
    identity_valid = bool(
        str(identity.get("cost_database_version", "") or "").strip()
        and str(identity.get("cost_database_sha256", "") or "").strip()
        and not str(identity.get("cost_database_error", "") or "").strip()
    )
    if active_keys and not identity_valid:
        return False
    expected_fingerprint = expected_resume_fingerprint
    if expected_fingerprint is None and identity_valid:
        expected_fingerprint = _cost_resume_fingerprint(
            active_keys,
            identity,
            reference_scenario_id,
            strategy_cost_regions=strategy_cost_regions,
        )
    if not s56._can_resume(
        plan,
        scenario_dir,
        expected_resume_fingerprint=expected_fingerprint,
    ):
        return False
    needs_cost_summary = bool(active_keys) or not plan.fast_emis_only
    if needs_cost_summary and not s56._artifact_can_resume(
        plan,
        scenario_dir,
        _cost_summary_path(scenario_dir),
        expected_resume_fingerprint=expected_fingerprint,
    ):
        return False
    if active_keys and not _cost_summary_matches_singleton(
        scenario_dir,
        database_strategy=active_keys[0],
        identity=cost_database_identity or {},
        reference_scenario_id=reference_scenario_id,
        strategy_cost_regions=strategy_cost_regions,
    ):
        return False
    return True


def _run_plan(
    plan: s56.ScenarioPlan,
    *,
    paths: s56.DataPaths,
    shared_cfg: s56.ScenarioConfig,
    shared_universe,
    shared_run_cache: Dict[str, object],
    specs: pd.DataFrame,
    cfg: Mapping[str, object],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    runs_dir = _runs_dir(cfg)
    scenario_dir = runs_dir / plan.scenario_id
    status = s56._status_base(plan, scenario_dir, shared_universe)
    database_identity = _cost_database_identity(paths)
    baseline_id = str(cfg.get("baseline_scenario_id") or "S5_7_BASE")
    active_cost_keys = _active_strategy_cost_keys(plan)
    strategy_cost_regions = (
        (str(plan.country),) if active_cost_keys and plan.country else None
    )
    param_rows = _build_param_rows(specs, cfg, include_kinds=plan.include_kinds, country=plan.country)
    design_signature = _cost_design_signature(plan, param_rows)
    status = _decorate_cost_status(
        status,
        plan,
        database_identity,
        baseline_id,
        strategy_cost_regions=strategy_cost_regions,
        design_signature=design_signature,
    )
    resume_fingerprint = str(status["cost_resume_fingerprint"])
    design_rows = _strategy_design_rows(plan, param_rows, shared_universe)

    if bool(cfg.get("dry_run", False)):
        status["run_status"] = "dry_run"
        return status, design_rows

    if bool(cfg.get("resume", True)) and _can_resume_plan(
        plan,
        scenario_dir,
        cost_database_identity=database_identity,
        reference_scenario_id=baseline_id,
        strategy_cost_regions=strategy_cost_regions,
        expected_resume_fingerprint=resume_fingerprint,
    ):
        status["run_status"] = "resumed"
        status = s56._finalize_status_from_outputs(status, scenario_dir, cfg)
        if status["run_status"] == "ok":
            status["run_status"] = "resumed"
        status["cost_summary_found"] = bool(_cost_summary_path(scenario_dir).exists())
        return status, design_rows

    if not bool(cfg.get("resume", True)) and bool(cfg.get("clear_existing_run_dirs_when_no_resume", False)):
        s56._clear_scenario_dir(scenario_dir, runs_dir)

    try:
        effects = None
        if param_rows:
            effects = _build_effects(param_rows, shared_universe, cfg, scenario_id=plan.scenario_id)
        outdir = s56.run_one_pipeline(
            paths,
            pre_macc_e0=False,
            scenario_id=plan.scenario_id,
            scenario_effects=effects,
            solve=True,
            use_fao_modules=True,
            save_root=str(runs_dir),
            future_last_only=True,
            use_linear=True,
            # Strategy-priced singleton runs must reach cost_summary.csv even
            # when the emissions workflow would otherwise use its fast exit.
            fast_emis_only=bool(plan.fast_emis_only) and not active_cost_keys,
            fast_emis_year=int(cfg.get("year", 2080) or 2080),
            resume_fingerprint=resume_fingerprint,
            prebuilt_config=shared_cfg,
            prebuilt_universe=shared_universe,
            prebuilt_run_cache=shared_run_cache,
            active_strategy_cost_keys=active_cost_keys,
            strategy_cost_regions=strategy_cost_regions,
        )
        scenario_dir = Path(outdir)
        status["scenario_dir"] = str(scenario_dir)
        status = s56._finalize_status_from_outputs(status, scenario_dir, cfg)
        status["cost_summary_found"] = bool(_cost_summary_path(scenario_dir).exists())
    except s56.MCPrecheckFailed as exc:
        status["run_status"] = "precheck_failed"
        status["error_type"] = type(exc).__name__
        status["error_message"] = str(exc)
    except Exception as exc:
        status["run_status"] = "failed"
        status["error_type"] = type(exc).__name__
        status["error_message"] = str(exc)
        if bool(cfg.get("stop_on_error", False)):
            raise
    return status, design_rows


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S5.7 full-model Strategy endpoint reruns.")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument(
        "--strategy-process-config",
        type=str,
        default=None,
        help="JSON mapping from source scenario rows to SP_M3a strategy categories.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clear-existing-run-dirs", action="store_true")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--no-global-all", action="store_true")
    parser.add_argument("--no-global-individual-levers", action="store_true")
    parser.add_argument("--country-runs", action="store_true")
    parser.add_argument("--country-individual-levers", action="store_true")
    parser.add_argument("--country", action="append", default=[], help="M49, ISO3, or country name. Repeatable.")
    parser.add_argument("--max-countries", type=int, default=None)
    parser.add_argument("--detailed-country-accounting", action="store_true")
    parser.add_argument(
        "--shapley",
        action="store_true",
        default=None,
        help="Run every Strategy coalition and write Shapley decomposition outputs.",
    )
    parser.add_argument(
        "--no-shapley",
        action="store_false",
        dest="shapley",
        help="Run the baseline, full package, and singleton strategy scenarios without Shapley coalitions.",
    )
    parser.add_argument(
        "--shapley-keep-standard",
        action="store_true",
        help="With --shapley, also keep the standard all-levers and single-lever global scenarios.",
    )
    parser.add_argument(
        "--shapley-max-strategies",
        type=int,
        default=None,
        help="Safety cap for number of Strategy kinds in full Shapley decomposition.",
    )
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--threads", type=int, default=None)
    return parser.parse_args(list(argv) if argv is not None else None)


def _effective_config(args: argparse.Namespace) -> Dict[str, object]:
    cfg = copy.deepcopy(CONFIG)
    if args.out_dir:
        cfg["output_dir"] = args.out_dir
    if args.strategy_process_config:
        cfg["strategy_process_config_path"] = args.strategy_process_config
    if args.dry_run:
        cfg["dry_run"] = True
    if args.resume:
        cfg["resume"] = True
    if args.clear_existing_run_dirs:
        cfg["clear_existing_run_dirs_when_no_resume"] = True
    if args.no_baseline:
        cfg["run_baseline"] = False
    if args.no_global_all:
        cfg["run_global_all_levers"] = False
    if args.no_global_individual_levers:
        cfg["run_global_individual_levers"] = False
    if args.country_runs:
        cfg["run_country_one_at_a_time"] = True
    if args.country_individual_levers:
        cfg["run_country_individual_levers"] = True
    if args.country:
        cfg["country_filter"] = list(args.country)
    if args.max_countries is not None:
        cfg["max_countries"] = int(args.max_countries)
    if args.detailed_country_accounting:
        cfg["detailed_country_accounting"] = True
    decomp = copy.deepcopy(cfg.get("decomposition", {}) or {})
    if args.shapley is not None:
        if bool(args.shapley):
            decomp["method"] = "shapley"
            decomp["run_shapley"] = True
            if not args.shapley_keep_standard:
                decomp["keep_standard_global_scenarios"] = False
        else:
            decomp["method"] = "single"
            decomp["run_shapley"] = False
    if args.shapley_keep_standard:
        decomp["keep_standard_global_scenarios"] = True
    if args.shapley_max_strategies is not None:
        decomp["max_strategies"] = int(args.shapley_max_strategies)
    cfg["decomposition"] = decomp
    if args.stop_on_error:
        cfg["stop_on_error"] = True

    override = copy.deepcopy(cfg.get("override_cfg", {}) or {})
    if args.threads is not None:
        override["linear_solver_threads"] = int(args.threads)
    cfg["override_cfg"] = override
    return cfg


def _write_summary_outputs(
    *,
    status_df: pd.DataFrame,
    design_df: pd.DataFrame,
    cfg: Mapping[str, object],
    universe,
) -> None:
    # Reuse S5.6 summary logic. Ensure its helper sees our S5.7 output dir.
    cfg_for_s56 = dict(cfg)
    cfg_for_s56["output_dir"] = str(_output_dir(cfg))
    s56._write_summary_outputs(
        status_df=status_df,
        design_df=design_df,
        cfg=cfg_for_s56,
        universe=universe,
    )
    _write_normalized_global_strategy_potential(status_df=status_df, design_df=design_df, cfg=cfg)
    if _shapley_enabled(cfg):
        _write_shapley_decomposition(status_df=status_df, design_df=design_df, cfg=cfg)
        _write_max_package_shapley(status_df=status_df, cfg=cfg)
    if _cost_curve_enabled(cfg):
        _write_macc_outputs(status_df=status_df, cfg=cfg)


def _single_kind_from_include(raw: object) -> str:
    parts = [p.strip() for p in str(raw or "").split(";") if p.strip()]
    return parts[0] if len(parts) == 1 else ""


def _first_numeric(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return float(vals.iloc[0]) if not vals.empty else np.nan


def _design_endpoint_values(design_df: pd.DataFrame) -> pd.DataFrame:
    if design_df.empty or "scenario_id" not in design_df.columns or "kind" not in design_df.columns:
        return pd.DataFrame(columns=["scenario_id", "kind", "endpoint_value_2080"])
    work = design_df.copy()
    if "value_2080" not in work.columns:
        work["value_2080"] = np.nan
    if "strategy_endpoint_source" not in work.columns:
        work["strategy_endpoint_source"] = ""
    rows: List[Dict[str, object]] = []
    for (scenario_id, kind), group in work.groupby(["scenario_id", "kind"], sort=False):
        values = sorted(
            {
                float(value)
                for value in pd.to_numeric(group["value_2080"], errors="coerce").dropna()
                if np.isfinite(float(value))
            }
        )
        sources = sorted(
            {
                str(value)
                for value in group["strategy_endpoint_source"].dropna()
                if str(value).strip()
            }
        )
        source_kinds = sorted(
            {
                str(value)
                for value in group.get("source_kind", pd.Series(dtype=str)).dropna()
                if str(value).strip()
            }
        )
        rows.append(
            {
                "scenario_id": scenario_id,
                "kind": kind,
                "endpoint_value_2080": values[0] if len(values) == 1 else np.nan,
                "endpoint_values_2080": ";".join(f"{value:.12g}" for value in values),
                "endpoint_source": ";".join(sources),
                "endpoint_source_kinds": ";".join(source_kinds),
                "endpoint_rows": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def _write_normalized_global_strategy_potential(
    *,
    status_df: pd.DataFrame,
    design_df: pd.DataFrame,
    cfg: Mapping[str, object],
) -> None:
    """Normalize one-at-a-time full-model endpoint potentials.

    This is not LMDI. The model response is a discrete endpoint rerun with
    optimization and possible interactions, so the mathematically defensible
    first-order summary is a normalized set of single-lever potentials plus an
    explicit residual between all-lever and single-lever sums.
    """
    out_dir = _output_dir(cfg)
    baseline_id = str(cfg.get("baseline_scenario_id") or "S5_7_BASE")
    global_all_id = str(cfg.get("global_all_scenario_id") or "S5_7_GLOBAL_ALL_ENDPOINT_MAX_REDUCTION")
    reduction_df = s56._add_reduction_columns(status_df, baseline_id)

    global_rows = reduction_df[reduction_df.get("scope", "").astype(str).eq("global")].copy()
    if global_rows.empty:
        s56._write_csv(pd.DataFrame(), out_dir / "global_strategy_potential_normalized.csv")
        s56._write_csv(pd.DataFrame(), out_dir / "global_strategy_interaction_residual.csv")
        return

    global_rows["kind"] = global_rows.get("include_kinds", "").map(_single_kind_from_include)
    single = global_rows[
        (global_rows["kind"] != "")
        & (global_rows["scenario_id"].astype(str) != global_all_id)
        & (global_rows["scenario_id"].astype(str) != baseline_id)
    ].copy()

    endpoints = _design_endpoint_values(design_df)
    if not endpoints.empty:
        single = single.merge(endpoints, on=["scenario_id", "kind"], how="left")

    single["emission_reduction_gt"] = pd.to_numeric(single.get("emission_reduction_gt"), errors="coerce")
    single["positive_emission_reduction_gt"] = single["emission_reduction_gt"].clip(lower=0.0)
    valid_single_reductions = single["emission_reduction_gt"].dropna()
    if valid_single_reductions.empty:
        positive_sum = np.nan
        raw_sum = np.nan
    else:
        positive_sum = float(valid_single_reductions.clip(lower=0.0).sum())
        raw_sum = float(valid_single_reductions.sum())

    all_row = reduction_df[reduction_df["scenario_id"].astype(str).eq(global_all_id)].head(1)
    if all_row.empty and _shapley_enabled(cfg):
        shapley_kinds = _shapley_kinds_from_status(status_df, cfg)
        if shapley_kinds:
            full_mask = (1 << len(shapley_kinds)) - 1
            shapley_all_id = _shapley_scenario_id(
                str(cfg.get("scenario_prefix", "S5_7") or "S5_7"),
                full_mask,
                len(shapley_kinds),
            )
            shapley_all = reduction_df[reduction_df["scenario_id"].astype(str).eq(shapley_all_id)].head(1)
            if not shapley_all.empty:
                all_row = shapley_all
                global_all_id = shapley_all_id
    baseline_total_gt = _first_numeric(reduction_df.loc[reduction_df["scenario_id"].astype(str).eq(baseline_id), "afolu_emissions_gt_co2eq_yr"])
    all_levers_emissions_gt = _first_numeric(all_row.get("afolu_emissions_gt_co2eq_yr", pd.Series(dtype=float)))
    all_levers_reduction_gt = _first_numeric(all_row.get("emission_reduction_gt", pd.Series(dtype=float)))
    all_positive = max(0.0, all_levers_reduction_gt) if np.isfinite(all_levers_reduction_gt) else np.nan

    if np.isfinite(positive_sum) and positive_sum > 0:
        single["normalized_share_of_positive_single_sum"] = single["positive_emission_reduction_gt"] / positive_sum
        single["normalized_percent_of_positive_single_sum"] = single["normalized_share_of_positive_single_sum"] * 100.0
    else:
        single["normalized_share_of_positive_single_sum"] = np.nan
        single["normalized_percent_of_positive_single_sum"] = np.nan

    if np.isfinite(all_positive):
        single["all_levers_scaled_normalized_reduction_gt"] = (
            single["normalized_share_of_positive_single_sum"] * all_positive
        )
    else:
        single["all_levers_scaled_normalized_reduction_gt"] = np.nan

    if np.isfinite(all_levers_reduction_gt) and all_levers_reduction_gt != 0:
        single["single_reduction_share_of_all_levers_raw"] = (
            single["emission_reduction_gt"] / all_levers_reduction_gt
        )
    else:
        single["single_reduction_share_of_all_levers_raw"] = np.nan

    single["negative_or_zero_potential"] = np.where(
        np.isfinite(single["emission_reduction_gt"]),
        single["emission_reduction_gt"] <= 0.0,
        np.nan,
    )
    order_map = {kind: idx for idx, kind in enumerate(STRATEGY_KIND_ORDER)}
    single["_order"] = single["kind"].map(order_map).fillna(999)
    single = single.sort_values(["_order", "scenario_id"]).drop(columns=["_order"], errors="ignore")

    keep_cols = [
        "kind",
        "strategy_name",
        "scenario_id",
        "run_status",
        "afolu_emissions_gt_co2eq_yr",
        "baseline_total_co2eq_gt",
        "emission_reduction_gt",
        "positive_emission_reduction_gt",
        "normalized_share_of_positive_single_sum",
        "normalized_percent_of_positive_single_sum",
        "all_levers_scaled_normalized_reduction_gt",
        "single_reduction_share_of_all_levers_raw",
        "negative_or_zero_potential",
        "endpoint_value_2080",
        "endpoint_values_2080",
        "endpoint_source",
        "endpoint_source_kinds",
        "endpoint_rows",
    ]
    keep_cols = [col for col in keep_cols if col in single.columns]
    normalized_out = single[keep_cols].copy()
    s56._write_csv(normalized_out, out_dir / "global_strategy_potential_normalized.csv")

    residual_gt = (
        all_levers_reduction_gt - raw_sum
        if np.isfinite(all_levers_reduction_gt) and np.isfinite(raw_sum)
        else np.nan
    )
    positive_residual_gt = (
        all_positive - positive_sum
        if np.isfinite(all_positive) and np.isfinite(positive_sum)
        else np.nan
    )
    residual_df = pd.DataFrame(
        [
            {
                "baseline_scenario_id": baseline_id,
                "global_all_scenario_id": global_all_id,
                "baseline_total_co2eq_gt": baseline_total_gt,
                "all_levers_emissions_gt": all_levers_emissions_gt,
                "all_levers_reduction_gt": all_levers_reduction_gt,
                "sum_single_strategy_reduction_gt": raw_sum,
                "sum_positive_single_strategy_reduction_gt": positive_sum,
                "interaction_residual_gt": residual_gt,
                "positive_interaction_residual_gt": positive_residual_gt,
                "interaction_residual_pct_of_all_levers": (
                    residual_gt / all_levers_reduction_gt * 100.0
                    if np.isfinite(residual_gt) and np.isfinite(all_levers_reduction_gt) and all_levers_reduction_gt != 0
                    else np.nan
                ),
                "normalization_method": "positive single-strategy endpoint potential / sum of positive single-strategy endpoint potentials",
                "decomposition_note": (
                    "LMDI is not applied because these are discrete full-model endpoint reruns, "
                    "not a multiplicative identity decomposition. The residual reports non-additivity/interactions."
                ),
            }
        ]
    )
    s56._write_csv(residual_df, out_dir / "global_strategy_interaction_residual.csv")


def _include_kinds_tuple(raw: object) -> Tuple[str, ...]:
    return tuple(p.strip() for p in str(raw or "").split(";") if p.strip())


def _shapley_kinds_from_status(status_df: pd.DataFrame, cfg: Mapping[str, object]) -> Tuple[str, ...]:
    configured = _ordered_strategy_kinds(list(cfg.get("eligible_kinds") or STRATEGY_KIND_ORDER))
    if "strategy_name" in status_df.columns and "include_kinds" in status_df.columns:
        shapley = status_df[
            status_df["strategy_name"].astype(str).str.startswith("shapley_coalition_")
        ].copy()
        seen: List[str] = []
        for raw in shapley.get("include_kinds", []):
            for kind in _include_kinds_tuple(raw):
                if kind not in seen:
                    seen.append(kind)
        if seen:
            seen_set = set(seen)
            configured_set = set(configured)
            if configured and seen_set.issubset(configured_set):
                return configured
            return _ordered_strategy_kinds(seen)
    return configured


def _coalition_mask(kinds: Sequence[str], all_kinds: Sequence[str]) -> int:
    index = {kind: idx for idx, kind in enumerate(all_kinds)}
    mask = 0
    for kind in kinds:
        if kind in index:
            mask |= 1 << index[kind]
    return mask


def _write_shapley_decomposition(
    *,
    status_df: pd.DataFrame,
    design_df: pd.DataFrame,
    cfg: Mapping[str, object],
) -> None:
    """Write exact Shapley allocation from full-model coalition reruns."""
    out_dir = _output_dir(cfg)
    baseline_id = str(cfg.get("baseline_scenario_id") or "S5_7_BASE")
    prefix = str(cfg.get("scenario_prefix", "S5_7") or "S5_7")
    kinds = _shapley_kinds_from_status(status_df, cfg)
    n = len(kinds)
    if n == 0:
        s56._write_csv(pd.DataFrame(), out_dir / "shapley_strategy_decomposition.csv")
        s56._write_csv(pd.DataFrame(), out_dir / "shapley_coalition_values.csv")
        s56._write_csv(pd.DataFrame(), out_dir / "shapley_decomposition_status.csv")
        return

    reduction_df = s56._add_reduction_columns(status_df, baseline_id)
    shapley_rows = reduction_df[
        reduction_df.get("strategy_name", "").astype(str).str.startswith("shapley_coalition_")
    ].copy()
    shapley_rows["coalition_kinds"] = shapley_rows.get("include_kinds", "").map(_include_kinds_tuple)
    shapley_rows["coalition_mask"] = shapley_rows["coalition_kinds"].map(lambda ks: _coalition_mask(ks, kinds))
    shapley_rows["coalition_size"] = shapley_rows["coalition_kinds"].map(len)
    shapley_rows["emission_reduction_gt"] = pd.to_numeric(
        shapley_rows.get("emission_reduction_gt"), errors="coerce"
    )

    values: Dict[int, float] = {0: 0.0}
    available_masks = {0}
    for _, row in shapley_rows.iterrows():
        mask = int(row["coalition_mask"])
        val = row.get("emission_reduction_gt")
        if pd.notna(val) and np.isfinite(float(val)):
            values[mask] = float(val)
            available_masks.add(mask)

    expected_masks = set(range(0, 1 << n))
    missing_masks = sorted(expected_masks - available_masks)
    full_mask = (1 << n) - 1
    all_levers_reduction_gt = values.get(full_mask, np.nan)
    complete = len(missing_masks) == 0 and np.isfinite(all_levers_reduction_gt)

    coalition_out = shapley_rows.copy()
    coalition_out["coalition_label"] = coalition_out["coalition_kinds"].map(lambda ks: ";".join(ks))
    coalition_keep = [
        "scenario_id",
        "run_status",
        "coalition_mask",
        "coalition_size",
        "coalition_label",
        "afolu_emissions_gt_co2eq_yr",
        "baseline_total_co2eq_gt",
        "emission_reduction_gt",
    ]
    coalition_keep = [col for col in coalition_keep if col in coalition_out.columns]
    coalition_out = coalition_out[coalition_keep].sort_values("coalition_mask")
    s56._write_csv(coalition_out, out_dir / "shapley_coalition_values.csv")

    rows: List[Dict[str, object]] = []
    factorial_n = math.factorial(n)
    for idx, kind in enumerate(kinds):
        phi = 0.0
        missing_marginals = 0
        for mask in range(0, 1 << n):
            if mask & (1 << idx):
                continue
            with_i = mask | (1 << idx)
            s = int(mask.bit_count())
            weight = math.factorial(s) * math.factorial(n - s - 1) / factorial_n
            if mask not in values or with_i not in values:
                missing_marginals += 1
                continue
            phi += weight * (values[with_i] - values[mask])
        rows.append(
            {
                "kind": kind,
                "shapley_value_gt": phi if complete else np.nan,
                "shapley_value_partial_gt": phi,
                "missing_marginals": missing_marginals,
                "all_levers_reduction_gt": all_levers_reduction_gt,
                "shapley_share_of_all_levers": (
                    phi / all_levers_reduction_gt
                    if complete and np.isfinite(all_levers_reduction_gt) and all_levers_reduction_gt != 0
                    else np.nan
                ),
                "shapley_percent_of_all_levers": (
                    phi / all_levers_reduction_gt * 100.0
                    if complete and np.isfinite(all_levers_reduction_gt) and all_levers_reduction_gt != 0
                    else np.nan
                ),
                "positive_shapley_value_gt": max(0.0, phi) if complete else np.nan,
            }
        )

    shapley_df = pd.DataFrame(rows)
    positive_sum = pd.to_numeric(shapley_df["positive_shapley_value_gt"], errors="coerce").sum(skipna=True)
    if np.isfinite(positive_sum) and positive_sum > 0:
        shapley_df["positive_shapley_normalized_share"] = shapley_df["positive_shapley_value_gt"] / positive_sum
        shapley_df["positive_shapley_normalized_percent"] = shapley_df["positive_shapley_normalized_share"] * 100.0
    else:
        shapley_df["positive_shapley_normalized_share"] = np.nan
        shapley_df["positive_shapley_normalized_percent"] = np.nan

    endpoints = _design_endpoint_values(design_df)
    if not endpoints.empty:
        endpoint_by_kind = (
            endpoints.sort_values("scenario_id")
            .drop_duplicates("kind", keep="first")
            [[
                "kind",
                "endpoint_value_2080",
                "endpoint_values_2080",
                "endpoint_source",
                "endpoint_source_kinds",
                "endpoint_rows",
            ]]
        )
        shapley_df = shapley_df.merge(endpoint_by_kind, on="kind", how="left")

    order_map = {kind: idx for idx, kind in enumerate(kinds)}
    shapley_df["_order"] = shapley_df["kind"].map(order_map).fillna(999)
    shapley_df = shapley_df.sort_values("_order").drop(columns=["_order"], errors="ignore")
    s56._write_csv(shapley_df, out_dir / "shapley_strategy_decomposition.csv")

    sum_shapley = pd.to_numeric(shapley_df["shapley_value_gt"], errors="coerce").sum(skipna=True)
    status = pd.DataFrame(
        [
            {
                "decomposition_method": "exact_shapley_full_model_coalitions",
                "strategy_count": n,
                "expected_coalitions_including_empty": int(1 << n),
                "expected_nonempty_coalition_runs": int((1 << n) - 1),
                "available_coalitions_including_empty": int(len(available_masks)),
                "missing_coalitions_count": int(len(missing_masks)),
                "complete": bool(complete),
                "baseline_scenario_id": baseline_id,
                "scenario_prefix": prefix,
                "all_levers_reduction_gt": all_levers_reduction_gt,
                "sum_shapley_value_gt": sum_shapley if complete else np.nan,
                "efficiency_gap_gt": (
                    sum_shapley - all_levers_reduction_gt
                    if complete and np.isfinite(all_levers_reduction_gt)
                    else np.nan
                ),
                "missing_masks_first_20": ";".join(
                    f"{mask:0{_shapley_mask_width(n)}X}" for mask in missing_masks[:20]
                ),
                "note": (
                    "Shapley values strictly allocate interaction when all coalition reruns are complete. "
                    "For n strategies this requires 2^n model evaluations including baseline."
                ),
            }
        ]
    )
    s56._write_csv(status, out_dir / "shapley_decomposition_status.csv")


def _write_max_package_shapley(
    *,
    status_df: pd.DataFrame,
    cfg: Mapping[str, object],
) -> None:
    """Select the best tested coalition and allocate its reduction exactly."""
    out_dir = _output_dir(cfg)
    baseline_id = str(cfg.get("baseline_scenario_id") or "S5_7_BASE")
    reduction_df = s56._add_reduction_columns(status_df, baseline_id)
    coalition_rows = reduction_df[
        reduction_df.get("strategy_name", "").astype(str).str.startswith("shapley_coalition_")
        & reduction_df.get("run_status", "").astype(str).isin(["ok", "resumed"])
    ].copy()
    coalition_rows["coalition_kinds"] = coalition_rows.get("include_kinds", "").map(
        _include_kinds_tuple
    )
    coalition_rows["coalition_size"] = coalition_rows["coalition_kinds"].map(len)
    coalition_rows["emission_reduction_gt"] = pd.to_numeric(
        coalition_rows.get("emission_reduction_gt"),
        errors="coerce",
    )
    coalition_rows = coalition_rows.dropna(subset=["emission_reduction_gt"])
    if coalition_rows.empty:
        s56._write_csv(pd.DataFrame(), out_dir / "max_package_shapley_decomposition.csv")
        s56._write_csv(
            pd.DataFrame(
                [
                    {
                        "complete": False,
                        "status": "no_valid_coalitions",
                    }
                ]
            ),
            out_dir / "max_package_shapley_status.csv",
        )
        return

    best = coalition_rows.sort_values(
        ["emission_reduction_gt", "coalition_size", "scenario_id"],
        ascending=[False, True, True],
    ).iloc[0]
    package_kinds = _ordered_strategy_kinds(best["coalition_kinds"])
    package_set = frozenset(package_kinds)
    package_size = len(package_kinds)
    package_reduction = float(best["emission_reduction_gt"])
    values: Dict[frozenset[str], float] = {frozenset(): 0.0}
    for _, row in coalition_rows.iterrows():
        kinds = frozenset(str(kind) for kind in row["coalition_kinds"])
        if kinds.issubset(package_set):
            values[kinds] = float(row["emission_reduction_gt"])

    missing_subsets: List[str] = []
    for mask in range(1 << package_size):
        subset = frozenset(
            package_kinds[idx]
            for idx in range(package_size)
            if mask & (1 << idx)
        )
        if subset not in values:
            missing_subsets.append(";".join(sorted(subset)))
    complete = package_size > 0 and not missing_subsets

    rows: List[Dict[str, object]] = []
    denominator = math.factorial(package_size) if package_size > 0 else 1
    for kind in package_kinds:
        phi = 0.0
        missing_marginals = 0
        others = [candidate for candidate in package_kinds if candidate != kind]
        for mask in range(1 << len(others)):
            subset = frozenset(
                others[idx]
                for idx in range(len(others))
                if mask & (1 << idx)
            )
            with_kind = frozenset(set(subset) | {kind})
            if subset not in values or with_kind not in values:
                missing_marginals += 1
                continue
            subset_size = len(subset)
            weight = (
                math.factorial(subset_size)
                * math.factorial(package_size - subset_size - 1)
                / denominator
            )
            phi += weight * (values[with_kind] - values[subset])
        rows.append(
            {
                "kind": kind,
                "max_package_shapley_value_gt": phi if complete else np.nan,
                "max_package_shapley_value_partial_gt": phi,
                "missing_marginals": missing_marginals,
                "max_package_scenario_id": best.get("scenario_id", ""),
                "max_package_kinds": ";".join(package_kinds),
                "max_package_strategy_count": package_size,
                "max_package_reduction_gt": package_reduction,
                "share_of_max_package": (
                    phi / package_reduction
                    if complete and package_reduction != 0
                    else np.nan
                ),
                "percent_of_max_package": (
                    phi / package_reduction * 100.0
                    if complete and package_reduction != 0
                    else np.nan
                ),
            }
        )
    decomposition = pd.DataFrame(rows)
    if not decomposition.empty:
        order_map = {kind: idx for idx, kind in enumerate(package_kinds)}
        decomposition["_order"] = decomposition["kind"].map(order_map).fillna(999)
        decomposition = decomposition.sort_values("_order").drop(columns=["_order"])
    s56._write_csv(decomposition, out_dir / "max_package_shapley_decomposition.csv")

    sum_phi = pd.to_numeric(
        decomposition.get("max_package_shapley_value_gt", pd.Series(dtype=float)),
        errors="coerce",
    ).sum(skipna=True)
    status = pd.DataFrame(
        [
            {
                "complete": bool(complete),
                "status": "complete" if complete else "missing_package_subsets",
                "max_package_scenario_id": best.get("scenario_id", ""),
                "max_package_kinds": ";".join(package_kinds),
                "max_package_strategy_count": package_size,
                "max_package_reduction_gt": package_reduction,
                "sum_max_package_shapley_gt": sum_phi if complete else np.nan,
                "efficiency_gap_gt": (
                    sum_phi - package_reduction
                    if complete
                    else np.nan
                ),
                "missing_subset_count": len(missing_subsets),
                "missing_subsets_first_20": "|".join(missing_subsets[:20]),
            }
        ]
    )
    s56._write_csv(status, out_dir / "max_package_shapley_status.csv")


def _read_csv_if_present(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError, ValueError):
        return pd.DataFrame()


def _cost_levels(cfg: Mapping[str, object]) -> List[float]:
    raw_levels = _cost_curve_cfg(cfg).get("cost_levels_usd_per_tco2eq") or []
    levels: List[float] = []
    for raw in raw_levels:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value >= 0:
            levels.append(value)
    return sorted(set(levels))


def _singleton_status_rows(status_df: pd.DataFrame) -> pd.DataFrame:
    if status_df.empty:
        return pd.DataFrame()
    work = status_df.copy()
    work["kind"] = work.get("include_kinds", "").map(_single_kind_from_include)
    work = work[
        work.get("scope", "").astype(str).eq("global")
        & work["kind"].astype(str).ne("")
        & work.get("run_status", "").astype(str).isin(["ok", "resumed"])
    ].copy()
    return work


def _strict_singleton_cost_rows(detail: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Select only the strategy-owned row for one singleton run."""
    expected_key = _database_strategy_for_kind(kind)
    required = {
        "cost_component_type",
        "database_strategy",
        "is_priced",
        "attribution_method",
    }
    if detail.empty or not expected_key or not required.issubset(detail.columns):
        return detail.iloc[0:0].copy()
    expected_component = _cost_component_type_for_database_strategy(expected_key)
    return detail[
        detail["cost_component_type"].astype(str).str.strip().str.lower().eq(expected_component)
        & detail["database_strategy"].astype(str).str.strip().eq(expected_key)
        & _summary_bool(detail["is_priced"])
        & detail["attribution_method"].fillna("").astype(str).str.strip().eq(
            "singleton_incremental_vs_reference"
        )
    ].copy()


def _current_database_cost_rows(
    detail: pd.DataFrame,
    cfg: Mapping[str, object],
) -> pd.DataFrame:
    out = detail.copy()
    expected = {
        "cost_database_version": str(cfg.get("cost_database_version", "") or ""),
        "cost_database_sha256": str(cfg.get("cost_database_sha256", "") or ""),
        "reference_scenario_id": str(cfg.get("baseline_scenario_id", "") or ""),
    }
    for column, value in expected.items():
        if not value:
            continue
        if column not in out.columns:
            return out.iloc[0:0].copy()
        out = out[out[column].fillna("").astype(str).str.strip().eq(value)].copy()
    return out


def _collect_singleton_cost_detail(
    status_df: pd.DataFrame,
    cfg: Mapping[str, object],
) -> pd.DataFrame:
    out_dir = _output_dir(cfg)
    merged_cost_path = out_dir / "macc_singleton_cost_detail.csv"
    merged = _read_csv_if_present(merged_cost_path)
    singleton_status = _singleton_status_rows(status_df)
    if not merged.empty and {
        "scenario_id",
        "kind",
        "cost_resume_fingerprint",
    }.issubset(merged.columns):
        strict_frames: List[pd.DataFrame] = []
        for _, status_row in singleton_status.iterrows():
            scenario_id = str(status_row.get("scenario_id", "") or "")
            kind = str(status_row.get("kind", "") or "")
            fingerprint = str(
                status_row.get("cost_resume_fingerprint", "") or ""
            )
            selected = merged[
                merged["scenario_id"].astype(str).eq(scenario_id)
                & merged["kind"].astype(str).eq(kind)
                & merged["cost_resume_fingerprint"].fillna("").astype(str).eq(
                    fingerprint
                )
            ].copy()
            selected = _strict_singleton_cost_rows(selected, kind)
            selected = _current_database_cost_rows(selected, cfg)
            if not selected.empty and fingerprint:
                strict_frames.append(selected)
        if strict_frames and len(strict_frames) == len(singleton_status):
            strict_merged = pd.concat(strict_frames, ignore_index=True)
            return strict_merged

    frames: List[pd.DataFrame] = []
    for _, row in singleton_status.iterrows():
        scenario_dir = Path(str(row.get("scenario_dir", "") or ""))
        cost_path = _cost_summary_path(scenario_dir)
        cost_df = _read_csv_if_present(cost_path)
        if cost_df.empty:
            continue
        kind = str(row.get("kind", "") or "")
        detail = _strict_singleton_cost_rows(cost_df, kind)
        detail = _current_database_cost_rows(detail, cfg)
        if detail.empty:
            continue
        detail.insert(0, "kind", str(row.get("kind", "") or ""))
        detail.insert(0, "scenario_id", str(row.get("scenario_id", "") or ""))
        detail.insert(2, "scenario_dir", str(scenario_dir))
        detail.insert(
            3,
            "cost_resume_fingerprint",
            str(row.get("cost_resume_fingerprint", "") or ""),
        )
        frames.append(detail)

    if not frames:
        return pd.DataFrame()
    detail_all = pd.concat(frames, ignore_index=True)
    s56._write_csv(detail_all, merged_cost_path)
    return detail_all


def _package_contribution_table(
    status_df: pd.DataFrame,
    cfg: Mapping[str, object],
) -> Tuple[pd.DataFrame, str]:
    out_dir = _output_dir(cfg)
    max_package = _read_csv_if_present(out_dir / "max_package_shapley_decomposition.csv")
    max_package_status = _read_csv_if_present(out_dir / "max_package_shapley_status.csv")
    max_package_complete = (
        not max_package_status.empty
        and "complete" in max_package_status.columns
        and str(max_package_status.iloc[0]["complete"]).strip().lower() in {"1", "true", "yes"}
    )
    max_package_col = "max_package_shapley_value_gt"
    if (
        max_package_complete
        and not max_package.empty
        and {"kind", max_package_col}.issubset(max_package.columns)
    ):
        out = max_package[["kind", max_package_col]].copy()
        out = out.rename(columns={max_package_col: "package_contribution_gt"})
        out["contribution_method"] = "exact_max_package_shapley"
        finite = pd.to_numeric(out["package_contribution_gt"], errors="coerce").notna()
        if not out.empty and bool(finite.all()):
            return out, "exact_max_package_shapley"

    shapley = _read_csv_if_present(out_dir / "shapley_strategy_decomposition.csv")
    shapley_status = _read_csv_if_present(out_dir / "shapley_decomposition_status.csv")
    complete = False
    if not shapley_status.empty and "complete" in shapley_status.columns:
        complete = str(shapley_status.iloc[0]["complete"]).strip().lower() in {"1", "true", "yes"}
    elif not shapley.empty and "missing_marginals" in shapley.columns:
        missing = pd.to_numeric(shapley["missing_marginals"], errors="coerce")
        complete = bool(missing.notna().all() and missing.eq(0).all())
    if complete and not shapley.empty and {"kind", "shapley_value_gt"}.issubset(shapley.columns):
        out = shapley[
            shapley["kind"].astype(str).isin(set(STRATEGY_KIND_ORDER))
        ][["kind", "shapley_value_gt"]].copy()
        out = out.rename(columns={"shapley_value_gt": "package_contribution_gt"})
        out["contribution_method"] = "exact_shapley"
        finite = pd.to_numeric(out["package_contribution_gt"], errors="coerce").notna()
        if len(out) == len(STRATEGY_KIND_ORDER) and bool(finite.all()):
            return out, "exact_shapley"

    normalized = _read_csv_if_present(out_dir / "global_strategy_potential_normalized.csv")
    fallback_col = "all_levers_scaled_normalized_reduction_gt"
    if not normalized.empty and {"kind", fallback_col}.issubset(normalized.columns):
        out = normalized[
            normalized["kind"].astype(str).isin(set(STRATEGY_KIND_ORDER))
        ][["kind", fallback_col]].copy()
        out = out.rename(columns={fallback_col: "package_contribution_gt"})
        out["contribution_method"] = "positive_singleton_scaled"
        finite = pd.to_numeric(out["package_contribution_gt"], errors="coerce").notna()
        if len(out) == len(STRATEGY_KIND_ORDER) and bool(finite.all()):
            return out, "positive_singleton_scaled"

    return pd.DataFrame(columns=["kind", "package_contribution_gt", "contribution_method"]), "missing"


def _baseline_total_gt(status_df: pd.DataFrame, cfg: Mapping[str, object]) -> float:
    baseline_id = str(cfg.get("baseline_scenario_id") or "S5_7_BASE")
    if status_df.empty or "scenario_id" not in status_df.columns:
        return np.nan
    rows = status_df[status_df["scenario_id"].astype(str).eq(baseline_id)]
    if rows.empty:
        return np.nan
    return _first_numeric(rows.get("afolu_emissions_gt_co2eq_yr", pd.Series(dtype=float)))


def _strategy_cost_profiles(
    contribution_df: pd.DataFrame,
    cost_detail: pd.DataFrame,
    cfg: Mapping[str, object],
) -> pd.DataFrame:
    levels = _cost_levels(cfg)
    cost_cfg = _cost_curve_cfg(cfg)
    year = int(cost_cfg.get("year", cfg.get("year", 2080)) or 2080)
    min_abatement = float(cost_cfg.get("minimum_positive_abatement_tco2eq", 1e-6) or 1e-6)

    detail = cost_detail.copy()
    for col in ("year", "abatement_tco2eq", "unit_cost_usd_per_tco2eq", "total_cost_usd"):
        if col not in detail.columns:
            detail[col] = np.nan
        detail[col] = pd.to_numeric(detail[col], errors="coerce")
    if not detail.empty:
        priced_mask = (
            _summary_bool(detail["is_priced"])
            if "is_priced" in detail.columns
            else pd.Series(False, index=detail.index)
        )
        detail = detail[
            detail["year"].eq(year)
            & detail["abatement_tco2eq"].gt(min_abatement)
            & detail["unit_cost_usd_per_tco2eq"].ge(0)
            & np.isfinite(detail["unit_cost_usd_per_tco2eq"])
            & priced_mask
        ].copy()

    rows: List[Dict[str, object]] = []
    order_map = {kind: idx for idx, kind in enumerate(STRATEGY_KIND_ORDER)}
    for _, contribution_row in contribution_df.iterrows():
        kind = str(contribution_row.get("kind", "") or "")
        package_contribution = pd.to_numeric(
            pd.Series([contribution_row.get("package_contribution_gt")]),
            errors="coerce",
        ).iloc[0]
        strategy_detail = detail[detail.get("kind", "").astype(str).eq(kind)].copy()
        total_abatement = float(strategy_detail["abatement_tco2eq"].sum()) if not strategy_detail.empty else np.nan
        total_cost = float(strategy_detail["total_cost_usd"].sum()) if not strategy_detail.empty else np.nan
        average_cost = (
            total_cost / total_abatement
            if np.isfinite(total_cost) and np.isfinite(total_abatement) and total_abatement > 0
            else np.nan
        )
        for level in levels:
            eligible_abatement = (
                float(
                    strategy_detail.loc[
                        strategy_detail["unit_cost_usd_per_tco2eq"].le(level),
                        "abatement_tco2eq",
                    ].sum()
                )
                if not strategy_detail.empty
                else np.nan
            )
            fraction = (
                min(1.0, max(0.0, eligible_abatement / total_abatement))
                if np.isfinite(eligible_abatement) and np.isfinite(total_abatement) and total_abatement > 0
                else np.nan
            )
            realized_contribution = (
                float(package_contribution) * fraction
                if np.isfinite(package_contribution) and np.isfinite(fraction)
                else np.nan
            )
            rows.append(
                {
                    "bioenergy_case": str(cost_cfg.get("bioenergy_case", "configured_case") or "configured_case"),
                    "year": year,
                    "cost_level_usd_per_tco2eq": level,
                    "kind": kind,
                    "strategy_display_name": STRATEGY_DISPLAY_NAMES.get(kind, kind),
                    "strategy_order": order_map.get(kind, 999),
                    "contribution_method": contribution_row.get("contribution_method", ""),
                    "package_contribution_gt": package_contribution,
                    "singleton_cost_rows": int(len(strategy_detail)),
                    "singleton_total_abatement_tco2eq": total_abatement,
                    "singleton_total_cost_usd": total_cost,
                    "singleton_average_cost_usd_per_tco2eq": average_cost,
                    "eligible_singleton_abatement_tco2eq": eligible_abatement,
                    "implementation_fraction": fraction,
                    "cost_limited_package_contribution_gt": realized_contribution,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["cost_level_usd_per_tco2eq", "strategy_order", "kind"])
    return out


def _write_macc_excel(
    profiles: pd.DataFrame,
    contribution_df: pd.DataFrame,
    baseline_total_gt: float,
    cfg: Mapping[str, object],
) -> None:
    cost_cfg = _cost_curve_cfg(cfg)
    if not bool(cost_cfg.get("write_excel", True)):
        return
    if profiles.empty or not np.isfinite(baseline_total_gt):
        return

    display_order = [
        STRATEGY_DISPLAY_NAMES.get(kind, kind)
        for kind in STRATEGY_KIND_ORDER
        if kind in set(profiles["kind"].astype(str))
    ]
    wide = profiles.pivot_table(
        index="cost_level_usd_per_tco2eq",
        columns="strategy_display_name",
        values="cost_limited_package_contribution_gt",
        aggfunc="first",
    ).reset_index()
    wide = wide.rename(columns={"cost_level_usd_per_tco2eq": "Price_USD_per_Ton"})
    for col in display_order:
        if col not in wide.columns:
            wide[col] = 0.0
    wide = wide[["Price_USD_per_Ton", *display_order]]

    max_by_kind = (
        contribution_df.assign(
            strategy_display_name=contribution_df["kind"].map(
                lambda kind: STRATEGY_DISPLAY_NAMES.get(str(kind), str(kind))
            )
        )
        .groupby("strategy_display_name", as_index=True)["package_contribution_gt"]
        .sum()
    )
    max_values = [float(max_by_kind.get(col, 0.0) or 0.0) for col in display_order]
    package_total = float(np.nansum(max_values))
    residual_at_max = float(baseline_total_gt - package_total)
    residual_col = "Wood harvest+Fire"
    wide[residual_col] = 0.0

    baseline_row = {
        "Price_USD_per_Ton": "Baseline emission",
        **{col: value for col, value in zip(display_order, max_values)},
        residual_col: residual_at_max,
    }
    max_row = {
        "Price_USD_per_Ton": "Max abatement",
        **{col: value for col, value in zip(display_order, max_values)},
        residual_col: 0.0,
    }
    export = pd.concat([wide, pd.DataFrame([baseline_row, max_row])], ignore_index=True)

    out_dir = _output_dir(cfg)
    filename = str(cost_cfg.get("excel_filename", "macc_strategy_curve_sp_m3a.xlsx") or "macc_strategy_curve_sp_m3a.xlsx")
    sheet_name = str(cost_cfg.get("bioenergy_case", "configured_case") or "configured_case")[:31]
    with pd.ExcelWriter(out_dir / filename, engine="openpyxl") as writer:
        export.to_excel(writer, sheet_name=sheet_name, index=False)


def _write_macc_outputs(
    *,
    status_df: pd.DataFrame,
    cfg: Mapping[str, object],
) -> None:
    out_dir = _output_dir(cfg)
    contribution_df, contribution_method = _package_contribution_table(status_df, cfg)
    cost_detail = _collect_singleton_cost_detail(status_df, cfg)
    baseline_total = _baseline_total_gt(status_df, cfg)

    diagnostic = {
        "cost_curve_enabled": True,
        "contribution_method": contribution_method,
        "baseline_total_co2eq_gt": baseline_total,
        "strategy_count": int(len(contribution_df)),
        "singleton_cost_rows": int(len(cost_detail)),
        "cost_level_count": int(len(_cost_levels(cfg))),
        "macc_ready": False,
        "message": "",
    }
    if contribution_df.empty:
        diagnostic["message"] = "Package contribution data are missing."
        s56._write_csv(pd.DataFrame([diagnostic]), out_dir / "macc_generation_status.csv")
        return
    if cost_detail.empty:
        diagnostic["message"] = "Singleton cost summaries are missing."
        s56._write_csv(pd.DataFrame([diagnostic]), out_dir / "macc_generation_status.csv")
        return
    if not np.isfinite(baseline_total):
        diagnostic["message"] = "Baseline emissions are missing."
        s56._write_csv(pd.DataFrame([diagnostic]), out_dir / "macc_generation_status.csv")
        return

    profiles = _strategy_cost_profiles(contribution_df, cost_detail, cfg)
    profile_kinds = set(profiles.get("kind", pd.Series(dtype=str)).astype(str))
    expected_kinds = set(contribution_df.get("kind", pd.Series(dtype=str)).astype(str))
    finite_fractions = pd.to_numeric(
        profiles.get("implementation_fraction", pd.Series(dtype=float)),
        errors="coerce",
    )
    missing_cost_kinds = sorted(expected_kinds.difference(profile_kinds))
    invalid_fraction_count = int(finite_fractions.isna().sum())
    if missing_cost_kinds or invalid_fraction_count > 0:
        diagnostic["message"] = (
            f"Incomplete strategy cost profiles: missing_kinds={missing_cost_kinds}, "
            f"invalid_fraction_rows={invalid_fraction_count}."
        )
        s56._write_csv(profiles, out_dir / "macc_strategy_cost_profiles.csv")
        s56._write_csv(pd.DataFrame([diagnostic]), out_dir / "macc_generation_status.csv")
        return
    s56._write_csv(profiles, out_dir / "macc_strategy_cost_profiles.csv")

    curve = profiles.copy()
    totals = (
        curve.groupby(["bioenergy_case", "year", "cost_level_usd_per_tco2eq"], as_index=False)
        .agg(
            package_abatement_gt=("cost_limited_package_contribution_gt", "sum"),
            strategies_with_cost_data=("implementation_fraction", "count"),
        )
    )
    totals["baseline_total_co2eq_gt"] = baseline_total
    totals["remaining_emissions_gt"] = baseline_total - totals["package_abatement_gt"]
    curve = curve.merge(
        totals,
        on=["bioenergy_case", "year", "cost_level_usd_per_tco2eq"],
        how="left",
    )
    s56._write_csv(curve, out_dir / "macc_strategy_curve_long.csv")
    s56._write_csv(totals, out_dir / "macc_strategy_curve_totals.csv")
    _write_macc_excel(profiles, contribution_df, baseline_total, cfg)

    diagnostic["macc_ready"] = True
    diagnostic["message"] = "MACC strategy profiles and SP_M3a-compatible workbook were generated."
    s56._write_csv(pd.DataFrame([diagnostic]), out_dir / "macc_generation_status.csv")


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    cfg = _effective_config(args)
    out_dir = _output_dir(cfg)
    runs_dir = _runs_dir(cfg)
    s56._ensure_dir(out_dir)
    s56._ensure_dir(runs_dir)

    baseline_id = str(cfg.get("baseline_scenario_id") or "S5_7_BASE")
    override = copy.deepcopy(cfg.get("override_cfg", {}) or {})
    override["base_case"] = baseline_id
    override["base_reference_dir"] = str(runs_dir / baseline_id)
    override["base_cost_calculation_method"] = "off"
    cfg["override_cfg"] = override

    paths = s56.DataPaths()
    database_identity = _cost_database_identity(paths)
    cfg.update(database_identity)
    shared_cfg = s56.ScenarioConfig()
    shared_universe = s56.build_universe_from_dict_v3(paths.dict_v3_path, shared_cfg)
    specs = _prepare_strategy_specs(s56._load_normalized_specs(cfg), cfg)
    all_plans = _build_plans(cfg, shared_universe, specs)
    plans, batch_meta, plan_indices = _apply_batch_plan_filter(all_plans, cfg)

    print(f"[S5_7] output_dir={out_dir}")
    print(f"[S5_7] strategy_process_config={_strategy_process_config_path(cfg)}")
    print(
        "[S5_7] cost_database="
        f"{database_identity.get('cost_database_version') or 'unknown'} "
        f"sha256={str(database_identity.get('cost_database_sha256') or '')[:16]}..."
    )
    print(f"[S5_7] specs={len(specs)} eligible_kinds={list(_eligible_strategy_kinds(cfg, specs))}")
    if bool(batch_meta.get("enabled", False)):
        print(
            "[S5_7] batch "
            f"{batch_meta.get('batch_index')}/{batch_meta.get('total_batches')} "
            f"tag={batch_meta.get('batch_tag')} "
            f"assignment={batch_meta.get('assignment')} "
            f"selected={batch_meta.get('selected_plan_count')}/{batch_meta.get('full_plan_count')}"
        )
    print(
        f"[S5_7] planned scenarios={len(plans)} "
        f"full_plan_count={len(all_plans)} dry_run={bool(cfg.get('dry_run', False))}"
    )

    backup = s56._apply_cfg_overrides(cfg)
    try:
        if bool(cfg.get("dry_run", False)):
            shared_run_cache = {}
        else:
            shared_run_cache = s56.build_run_baseline_cache(
                paths,
                shared_cfg,
                shared_universe,
                future_last_only=True,
            )

        status_rows: List[Dict[str, object]] = []
        design_rows: List[Dict[str, object]] = []
        for idx, plan in enumerate(plans, start=1):
            print(f"[S5_7] {idx}/{len(plans)} {plan.scenario_id} [{plan.scope}:{plan.strategy_name}]")
            status, rows = _run_plan(
                plan,
                paths=paths,
                shared_cfg=shared_cfg,
                shared_universe=shared_universe,
                shared_run_cache=shared_run_cache,
                specs=specs,
                cfg=cfg,
            )
            status_rows.append(status)
            if bool(batch_meta.get("enabled", False)):
                status["batch_index"] = int(batch_meta.get("batch_index", 1) or 1)
                status["batch_count"] = int(batch_meta.get("total_batches", 1) or 1)
                status["batch_tag"] = str(batch_meta.get("batch_tag", "") or "")
                status["plan_index"] = int(plan_indices.get(plan.scenario_id, idx))
                status["plan_count"] = int(batch_meta.get("full_plan_count", len(all_plans)) or len(all_plans))
            design_rows.extend(rows)
            print(
                f"[S5_7] -> {status.get('run_status')} "
                f"emissions={status.get('afolu_emissions_gt_co2eq_yr')}"
            )

        status_df = pd.DataFrame(status_rows)
        design_df = pd.DataFrame(design_rows)
        s56._write_csv(status_df, out_dir / "scenario_status.csv")
        s56._write_csv(design_df, out_dir / "strategy_design_long.csv")
        if bool(batch_meta.get("enabled", False)):
            manifest_rows = []
            for plan in plans:
                manifest_rows.append(
                    {
                        **batch_meta,
                        "scenario_id": plan.scenario_id,
                        "scope": plan.scope,
                        "strategy_name": plan.strategy_name,
                        "include_kinds": ";".join(plan.include_kinds),
                        "plan_index": int(plan_indices.get(plan.scenario_id, 0)),
                    }
                )
            s56._write_csv(pd.DataFrame(manifest_rows), out_dir / "batch_plan_manifest.csv")
        if bool(cfg.get("write_summary_outputs", True)):
            _write_summary_outputs(
                status_df=status_df,
                design_df=design_df,
                cfg=cfg,
                universe=shared_universe,
            )

        print(f"[S5_7] wrote {out_dir / 'scenario_status.csv'}")
        print(f"[S5_7] wrote {out_dir / 'strategy_design_long.csv'}")
        if bool(cfg.get("write_summary_outputs", True)):
            print(f"[S5_7] wrote {out_dir / 'scenario_reduction_summary.csv'}")
    finally:
        s56._restore_cfg_overrides(backup)


if __name__ == "__main__":
    main()
