# -*- coding: utf-8 -*-
"""Endpoint tests for maximum emission-reduction potential.

This S5.6 script reuses the S5.4/S4 model path and builds deterministic
endpoint strategies from Scenario_config_new.xlsx / MC_effect_low_land_new.

Functional scope:

1. Read the normalized sensitivity rows from the selected MC-effect sheet.
2. Select a low-emission endpoint for each eligible strategy kind using
   ``CONFIG["endpoint_u_by_kind"]``. A selector of 0 uses ``Min_bound`` and a
   selector of 1 uses ``Max_bound``.
3. Run a reference scenario, a global all-strategy endpoint scenario, and
   global singleton strategy scenarios.
4. Optionally run one-country-at-a-time scenarios. In the default country
   workflow, all eligible country-level strategies are applied together in one
   selected country while every other country remains at the reference setting.
5. Report the change in global 2080 AFOLU emissions relative to the S5.6
   reference run.
6. Optionally export country domestic accounting for the global all-strategy
   scenario when detailed emissions are enabled.

Interpretation limits:

- S5.6 is an endpoint stress test. It does not optimize continuously over
  strategy values and does not prove a mathematical global maximum.
- Country one-at-a-time output measures the global-system emissions response to
  one country's action. It is not automatically the selected country's domestic
  emissions reduction.
- Country singleton strategy runs are disabled by default because they require
  approximately countries multiplied by strategies model evaluations.
- The default S5.6 cost calculation is disabled. S5.7 is the workflow used for
  explicit endpoint values, interaction allocation, and MACC-ready cost output.

Outputs are written under:
  <output>/Max_Emission_Reduction_Potential/

Main outputs:
  - strategy_design_long.csv
  - scenario_status.csv
  - scenario_reduction_summary.csv
  - global_strategy_summary.csv
  - global_max_reduction_summary.csv
  - country_own_max_reduction_summary.csv
  - country_own_max_strategy_long.csv
  - country_best_tested_strategy_summary.csv
  - country_best_tested_strategy_long.csv
  - country_reduction_under_global_strategy.csv, when detailed country
    accounting is enabled and the detailed emissions files exist.
  - sensitivity_cost_summary_by_country_measure.csv
  - sensitivity_cost_summary_by_global_measure.csv
  - sensitivity_cost_summary_audit.csv

Interpretation:
  The "maximum" here is an endpoint stress-test maximum under explicit,
  monotonic direction assumptions in CONFIG["endpoint_u_by_kind"]. It is not a
  formal mixed-integer optimization over all possible strategy combinations.
"""
from __future__ import annotations

import argparse
import copy
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config_paths import get_results_base
from S1_0_schema import ScenarioConfig
from S2_0_load_data import DataPaths, build_universe_from_dict_v3
from S4_0_main import CFG, MCPrecheckFailed, build_run_baseline_cache, run_one_pipeline
import S5_4_1_monte_carlo_full_variables as fullmc
from S5_cost_summary_outputs import write_sensitivity_cost_summaries
from model_run_status import (
    ResumeValidation,
    artifact_matches_validated_run,
    validate_run_for_resume,
)


MAX_REDUCTION_KIND_ORDER = [
    "yield_rate",
    "feed_intensity",
    "losses_ratio",
    "ruminant_reduction",
    "emission_factor",
    "fertilizer_rate",
    "manure_management_ratio",
    "crop_soil_management_ratio",
    "land_carbon_price",
]

GLOBAL_ONLY_KINDS = {"land_carbon_price", "land_co2_price"}


CONFIG = {
    "year": 2080,
    "output_dir": "",  # empty -> <NZF_OUTPUT_DIR>/Max_Emission_Reduction_Potential
    "runs_subdir": "runs",
    "baseline_scenario_id": "S5_6_BASE",
    "global_all_scenario_id": "S5_6_GLOBAL_MAX_REDUCTION",
    "nutrition_profile_sheet": "low_land_new",
    # S5.6 is tied to the low-land nutrition profile, so read the matching
    # endpoint range sheet explicitly instead of relying on implicit remapping.
    "mc_sheet_prefer": "MC_effect_low_land_new",
    "aggregate_non_ef": False,
    "run_baseline": True,
    # Apply every eligible endpoint globally in one combined stress-test run.
    "run_global_all_levers": True,
    # Apply one eligible endpoint globally per run.
    "run_global_individual_levers": True,
    # Country one-at-a-time scenarios answer:
    # "How much global 2080 AFOLU emissions fall if only this country applies
    # the endpoint max-reduction strategy?"
    "run_country_one_at_a_time": True,
    # This creates country-by-strategy runs and is normally much larger than the
    # default country all-strategy workflow.
    "run_country_individual_levers": False,
    "country_filter": [],  # [] -> all countries; accepts M49 ('156/156), ISO3, or country name
    "max_countries": None,  # e.g. 5 for smoke tests
    "resume": False,
    "clear_existing_run_dirs_when_no_resume": False,
    "stop_on_error": False,
    "dry_run": False,
    # Fast mode is used for one-at-a-time country scenarios. Baseline and the
    # global-all strategy can be forced through the detailed emissions path below
    # to build country domestic accounting.
    "fast_emis_only": True,
    "detailed_country_accounting": True,
    "validate_fast_nonluc_emissions": True,
    "validate_market_balance": True,
    "market_gap_max_rate": 0.05,
    # IMPORTANT: values below are normalized endpoint selectors, not the actual
    # scenario values. 0.0 selects the sheet Min_bound; 1.0 selects Max_bound.
    # Example: for manure_management_ratio / crop_soil_management_ratio,
    # 0.0 selects the actual MC_effect_low_land_new Min_bound value -0.6.
    # These two levers are direct multipliers in the current emissions modules,
    # so their low-emission endpoint is Min_bound rather than Max_bound.
    "endpoint_u_by_kind": {
        "yield_rate": 1.0,
        "feed_intensity": 0.0,
        "losses_ratio": 0.0,
        "ruminant_reduction": 0.0,
        "emission_factor": 0.0,
        "fertilizer_rate": 0.0,
        "manure_management_ratio": 0.0,
        "crop_soil_management_ratio": 0.0,
        "land_carbon_price": 1.0,
        "land_co2_price": 1.0,
    },
    "unknown_kind_u": 0.5,
    "eligible_kinds": list(MAX_REDUCTION_KIND_ORDER),
    "sampling": {
        **copy.deepcopy(fullmc.CONFIG.get("sampling", {}) or {}),
        "method": "endpoint",
        "scope": "row",
        "shuffle": False,
        "quantile_bounds": (0.0, 1.0),
    },
    "override_cfg": {
        **copy.deepcopy(fullmc.CONFIG.get("override_cfg", {}) or {}),
        "nutrition_profile_sheet": "low_land_new",
        "cost_calculation_method": "off",
        "debug_level": 0,
        "batch_mode": False,
        "linear_enable_infeasible_iis": False,
        "linear_enable_violation_iis": False,
        "linear_enable_output_diagnostics": True,
        "linear_enable_verbose_logging": False,
        "max_slack_rate": 0.1,
        "max_shortage_slack_rate": 0.1,
        "max_excess_slack_rate": 0.1,
        "market_gap_max_rate": 0.05,
    },
}


@dataclass(frozen=True)
class ScenarioPlan:
    scenario_id: str
    scope: str  # baseline | global | country
    strategy_name: str
    include_kinds: Tuple[str, ...]
    country: Optional[str] = None
    fast_emis_only: bool = True
    require_country_detail: bool = False


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _default_output_dir() -> Path:
    return Path(get_results_base()) / "Max_Emission_Reduction_Potential"


def _output_dir(cfg: Mapping[str, object]) -> Path:
    raw = str(cfg.get("output_dir", "") or "").strip()
    return Path(raw) if raw else _default_output_dir()


def _runs_dir(cfg: Mapping[str, object]) -> Path:
    return _output_dir(cfg) / str(cfg.get("runs_subdir", "runs") or "runs")


def _safe_token(text: object, *, max_len: int = 72) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(text or "").strip()).strip("_")
    return (token or "NA")[:max_len]


def _normalize_m49(value: object) -> str:
    s = str(value or "").strip()
    if not s:
        return ""
    if s.startswith("'"):
        return "'" + s[1:].zfill(3) if s[1:].isdigit() else s
    if s.isdigit():
        return "'" + s.zfill(3)
    return s


def _country_label(country: str, universe) -> Dict[str, object]:
    m49 = _normalize_m49(country)
    return {
        "country": m49,
        "m49": m49,
        "iso3": universe.iso3_by_country.get(m49, ""),
        "country_name": universe.country_by_m49.get(m49, m49),
        "region_aggMC": universe.region_aggMC_by_country.get(m49, ""),
    }


def _country_scenario_suffix(country: str, universe) -> str:
    info = _country_label(country, universe)
    m49_num = str(info["m49"]).replace("'", "")
    iso3 = str(info.get("iso3") or "").strip()
    return _safe_token(f"{iso3}_{m49_num}" if iso3 else m49_num)


def _resolve_country_filter(raw_values: Sequence[object], universe) -> List[str]:
    if not raw_values:
        return list(universe.countries)

    by_iso = {
        str(iso).strip().upper(): c
        for c, iso in (universe.iso3_by_country or {}).items()
        if str(iso).strip()
    }
    by_name = {
        str(name).strip().lower(): c
        for c, name in (universe.country_by_m49 or {}).items()
        if str(name).strip()
    }

    out: List[str] = []
    seen = set()
    for raw in raw_values:
        s = str(raw or "").strip()
        if not s:
            continue
        candidates = [
            _normalize_m49(s),
            by_iso.get(s.upper(), ""),
            by_name.get(s.lower(), ""),
        ]
        match = next((c for c in candidates if c in universe.countries), "")
        if not match:
            raise ValueError(f"Unknown country selector: {s!r}")
        if match not in seen:
            out.append(match)
            seen.add(match)
    return out


def _active_countries(cfg: Mapping[str, object], universe) -> List[str]:
    countries = _resolve_country_filter(list(cfg.get("country_filter") or []), universe)
    max_countries = cfg.get("max_countries")
    if max_countries is not None:
        max_n = int(max_countries)
        if max_n <= 0:
            raise ValueError("max_countries must be positive or None.")
        countries = countries[:max_n]
    return countries


def _apply_cfg_overrides(cfg: Mapping[str, object]) -> Dict[str, object]:
    backup: Dict[str, object] = {}
    for key, value in (cfg.get("override_cfg") or {}).items():
        backup[key] = CFG.get(key)
        CFG[key] = value
    return backup


def _restore_cfg_overrides(backup: Mapping[str, object]) -> None:
    for key, value in backup.items():
        CFG[key] = value


def _load_normalized_specs(cfg: Mapping[str, object]) -> pd.DataFrame:
    paths = DataPaths()
    mc_sheet = fullmc.resolve_mc_effect_sheet(
        cfg.get("mc_sheet_prefer", "MC_effect_low_land_new"),
        nutrition_profile_sheet=cfg.get("nutrition_profile_sheet", "low_land_new"),
    )
    specs = fullmc._load_mc_specs_effect(
        paths.scenario_config_xlsx,
        prefer_sheet=mc_sheet,
        nutrition_profile_sheet=cfg.get("nutrition_profile_sheet", "low_land_new"),
    )
    specs = fullmc._normalize_mc_specs(
        specs,
        aggregate_non_ef=bool(cfg.get("aggregate_non_ef", False)),
    ).reset_index(drop=True)
    specs["spec_row_id"] = np.arange(1, len(specs) + 1)
    if specs.empty:
        raise RuntimeError(f"{mc_sheet} sheet is empty.")
    return specs


def _eligible_kinds(cfg: Mapping[str, object], specs: pd.DataFrame) -> Tuple[str, ...]:
    configured = [str(k).strip() for k in (cfg.get("eligible_kinds") or []) if str(k).strip()]
    available = set(specs["__kind"].astype(str))
    if configured:
        return tuple(k for k in configured if k in available)
    return tuple(k for k in MAX_REDUCTION_KIND_ORDER if k in available)


def _endpoint_u(kind: str, cfg: Mapping[str, object]) -> float:
    endpoint_map = cfg.get("endpoint_u_by_kind") or {}
    raw = endpoint_map.get(kind, cfg.get("unknown_kind_u", 0.5))
    try:
        val = float(raw)
    except Exception:
        val = 0.5
    return max(0.0, min(1.0, val))


def _endpoint_label(u: float) -> str:
    if u <= 1e-12:
        return "Min_bound"
    if u >= 1.0 - 1e-12:
        return "Max_bound"
    return "Interpolated"


def _build_param_rows(
    specs: pd.DataFrame,
    cfg: Mapping[str, object],
    *,
    include_kinds: Sequence[str],
    country: Optional[str],
) -> List[Dict[str, object]]:
    """Build ScenarioEffect parameter rows for one endpoint strategy.

    include_kinds controls which levers are active. If country is provided, the
    same endpoint levers are applied only to that country's M49 selector.
    """
    include = {str(k) for k in include_kinds}
    work = specs[specs["__kind"].astype(str).isin(include)].copy().reset_index(drop=True)
    if work.empty:
        return []

    unit_row = np.array([_endpoint_u(str(k), cfg) for k in work["__kind"].astype(str)], dtype=float)
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
    for row in rows:
        kind = str(row.get("kind", "") or "")
        u = _endpoint_u(kind, cfg)
        if country is not None:
            row["region"] = country
            row["region_selector"] = country
        row["strategy_endpoint_u"] = u
        row["strategy_endpoint"] = _endpoint_label(u)
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
    plan: ScenarioPlan,
    param_rows: List[Dict[str, object]],
    universe,
) -> List[Dict[str, object]]:
    country_info = _country_label(plan.country, universe) if plan.country else {}
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
                "kind": row.get("kind"),
                "element_name": row.get("element_name"),
                "element_unit": row.get("element_unit"),
                "item_selector": row.get("item_selector"),
                "process_selector": row.get("process_selector"),
                "ghg_selector": row.get("ghg_selector"),
                "region_selector": row.get("region_selector"),
                "strategy_endpoint": row.get("strategy_endpoint"),
                "strategy_endpoint_u": row.get("strategy_endpoint_u"),
                "value_2080": row.get("abs_value"),
                "min_bound": row.get("min_bound"),
                "max_bound": row.get("max_bound"),
                "mc_u": row.get("mc_u"),
                "q_low": row.get("q_low"),
                "q_high": row.get("q_high"),
            }
        )
    return rows


def _build_plans(cfg: Mapping[str, object], universe, specs: pd.DataFrame) -> List[ScenarioPlan]:
    """Create the scenario run list for global and country-level tests."""
    eligible = _eligible_kinds(cfg, specs)
    country_eligible = tuple(k for k in eligible if k not in GLOBAL_ONLY_KINDS)
    plans: List[ScenarioPlan] = []
    detailed = bool(cfg.get("detailed_country_accounting", False))

    if bool(cfg.get("run_baseline", True)):
        plans.append(
            ScenarioPlan(
                scenario_id=str(cfg.get("baseline_scenario_id") or "S5_6_BASE"),
                scope="baseline",
                strategy_name="baseline",
                include_kinds=(),
                fast_emis_only=not detailed,
                require_country_detail=detailed,
            )
        )

    if bool(cfg.get("run_global_all_levers", True)):
        plans.append(
            ScenarioPlan(
                scenario_id=str(cfg.get("global_all_scenario_id") or "S5_6_GLOBAL_MAX_REDUCTION"),
                scope="global",
                strategy_name="all_levers_endpoint_max_reduction",
                include_kinds=eligible,
                fast_emis_only=not detailed,
                require_country_detail=detailed,
            )
        )

    if bool(cfg.get("run_global_individual_levers", True)):
        for kind in eligible:
            plans.append(
                ScenarioPlan(
                    scenario_id=f"S5_6_GLOBAL_{_safe_token(kind).upper()}",
                    scope="global",
                    strategy_name=f"single_lever_{kind}",
                    include_kinds=(kind,),
                    fast_emis_only=bool(cfg.get("fast_emis_only", True)),
                    require_country_detail=False,
                )
            )

    countries = _active_countries(cfg, universe)
    if bool(cfg.get("run_country_one_at_a_time", True)):
        # Each country gets its own maximum endpoint strategy scenario:
        # only the selected country receives all eligible low-emission endpoint
        # levers, while all other countries stay at baseline. This gives the
        # country's one-at-a-time maximum reduction potential.
        for country in countries:
            suffix = _country_scenario_suffix(country, universe)
            plans.append(
                ScenarioPlan(
                    scenario_id=f"S5_6_CTRY_{suffix}_ALL",
                    scope="country",
                    strategy_name="country_own_max_endpoint_strategy",
                    include_kinds=country_eligible,
                    country=country,
                    fast_emis_only=bool(cfg.get("fast_emis_only", True)),
                    require_country_detail=False,
                )
            )

    if bool(cfg.get("run_country_individual_levers", False)):
        for country in countries:
            suffix = _country_scenario_suffix(country, universe)
            for kind in country_eligible:
                plans.append(
                    ScenarioPlan(
                        scenario_id=f"S5_6_CTRY_{suffix}_{_safe_token(kind).upper()}",
                        scope="country",
                        strategy_name=f"country_single_lever_{kind}",
                        include_kinds=(kind,),
                        country=country,
                        fast_emis_only=bool(cfg.get("fast_emis_only", True)),
                        require_country_detail=False,
                    )
                )
    return plans


def _clear_scenario_dir(scenario_dir: Path, runs_dir: Path) -> None:
    try:
        scenario_resolved = scenario_dir.resolve()
        runs_resolved = runs_dir.resolve()
    except Exception:
        return
    if scenario_resolved == runs_resolved:
        raise RuntimeError(f"Refuse to clear runs root: {scenario_resolved}")
    if not str(scenario_resolved).startswith(str(runs_resolved)):
        raise RuntimeError(f"Refuse to clear path outside runs root: {scenario_resolved}")
    if scenario_dir.exists():
        shutil.rmtree(scenario_dir)


def _country_detail_path(scenario_dir: Path) -> Path:
    return scenario_dir / "Emis" / "emissions_summary_By_Country.csv"


def _resume_validation(
    plan: ScenarioPlan,
    scenario_dir: Path,
    *,
    expected_resume_fingerprint: Optional[str] = None,
) -> ResumeValidation:
    return validate_run_for_resume(
        scenario_dir,
        expected_scenario_id=plan.scenario_id,
        expected_resume_fingerprint=expected_resume_fingerprint,
    )


def _artifact_can_resume(
    plan: ScenarioPlan,
    scenario_dir: Path,
    artifact_path: Path,
    *,
    expected_resume_fingerprint: Optional[str] = None,
) -> bool:
    return artifact_matches_validated_run(
        artifact_path,
        _resume_validation(
            plan,
            scenario_dir,
            expected_resume_fingerprint=expected_resume_fingerprint,
        ),
    )


def _fast_summary_identity_matches(
    fast_df: pd.DataFrame,
    validation: ResumeValidation,
) -> bool:
    if fast_df is None or fast_df.empty or not validation.allowed:
        return False
    expected = {
        "run_id": validation.run_id,
        "scenario_id": validation.scenario_id,
    }
    for column, expected_value in expected.items():
        if column not in fast_df.columns or not expected_value:
            return False
        values = [
            "" if pd.isna(value) else str(value).strip()
            for value in fast_df[column].tolist()
        ]
        if not values or any(value != expected_value for value in values):
            return False
    return True


def _can_resume(
    plan: ScenarioPlan,
    scenario_dir: Path,
    *,
    expected_resume_fingerprint: Optional[str] = None,
) -> bool:
    # Status/manifest provenance is checked before opening any output CSV.  This
    # prevents a failed new run from silently reusing a stale previous summary.
    validation = _resume_validation(
        plan,
        scenario_dir,
        expected_resume_fingerprint=expected_resume_fingerprint,
    )
    if not validation.allowed:
        return False
    fast_path = scenario_dir / "Emis" / "emissions_fast_summary.csv"
    if not artifact_matches_validated_run(fast_path, validation):
        return False
    fast_df = fullmc._read_fast_summary_df(scenario_dir)
    if not _fast_summary_identity_matches(fast_df, validation):
        return False
    if fullmc._fast_summary_total_gt(fast_df) is None:
        return False
    if plan.require_country_detail and not artifact_matches_validated_run(
        _country_detail_path(scenario_dir),
        validation,
    ):
        return False
    return True


def _read_country_co2eq(scenario_dir: Path, *, year: int, universe) -> pd.DataFrame:
    path = _country_detail_path(scenario_dir)
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    ycol = f"Y{int(year)}"
    if df.empty or ycol not in df.columns or "M49_Country_Code" not in df.columns or "GHG" not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    work["m49"] = work["M49_Country_Code"].map(_normalize_m49)
    ghg = work["GHG"].astype(str).str.strip()
    work = work[ghg.str.endswith("_CO2eq")].copy()
    work = work[work["m49"] != "'000"].copy()
    if work.empty:
        return pd.DataFrame()
    work["co2eq_kt"] = pd.to_numeric(work[ycol], errors="coerce").fillna(0.0)
    grouped = work.groupby("m49", as_index=False)["co2eq_kt"].sum()
    labels = [_country_label(m49, universe) for m49 in grouped["m49"]]
    label_df = pd.DataFrame(labels)
    out = pd.concat([label_df.reset_index(drop=True), grouped[["co2eq_kt"]].reset_index(drop=True)], axis=1)
    out["co2eq_gt"] = out["co2eq_kt"] * 1e-6
    return out


def _status_base(plan: ScenarioPlan, scenario_dir: Path, universe) -> Dict[str, object]:
    country_info = _country_label(plan.country, universe) if plan.country else {}
    return {
        "scenario_id": plan.scenario_id,
        "scope": plan.scope,
        "strategy_name": plan.strategy_name,
        "include_kinds": ";".join(plan.include_kinds),
        "country": country_info.get("country", ""),
        "iso3": country_info.get("iso3", ""),
        "country_name": country_info.get("country_name", ""),
        "region_aggMC": country_info.get("region_aggMC", ""),
        "scenario_dir": str(scenario_dir),
        "run_status": "pending",
        "model_status_code": None,
        "model_status_text": "",
        "iis_summary": "",
        "afolu_emissions_gt_co2eq_yr": np.nan,
        "error_type": "",
        "error_message": "",
    }


def _finalize_status_from_outputs(
    row: Dict[str, object],
    scenario_dir: Path,
    cfg: Mapping[str, object],
) -> Dict[str, object]:
    fast_df = fullmc._read_fast_summary_df(scenario_dir)
    log_diag = fullmc._apply_log_diagnostics(row, scenario_dir)
    total_gt = fullmc._fast_summary_total_gt(fast_df)

    if fullmc._apply_model_status_failure(row, log_diag):
        return row

    if bool(cfg.get("validate_fast_nonluc_emissions", True)):
        neg_msg = fullmc._validate_nonluc_fast_emissions(scenario_dir)
        if neg_msg:
            row["run_status"] = "invalid_fast_emissions"
            row["error_type"] = "NegativeNonLUCEmissions"
            row["error_message"] = neg_msg
            return row

    if fullmc._is_invalid_total_gt(total_gt):
        row["run_status"] = "invalid_fast_emissions"
        row["error_type"] = "InvalidFastSummarySentinel"
        row["error_message"] = f"invalid total_co2eq_gt sentinel: {total_gt}"
        return row

    if bool(cfg.get("validate_market_balance", True)):
        gap_msg = fullmc._validate_market_balance_gap(
            scenario_dir,
            max_gap_rate=float(cfg.get("market_gap_max_rate", 0.05) or 0.05),
        )
        if gap_msg:
            row["run_status"] = "invalid_market_balance"
            row["error_type"] = "MarketBalanceGap"
            row["error_message"] = gap_msg
            return row

    if total_gt is None:
        row["run_status"] = "missing_fast_summary"
        row["error_type"] = "MissingFastSummary"
        row["error_message"] = "emissions_fast_summary.csv missing or unreadable"
        return row

    row["run_status"] = "ok"
    row["afolu_emissions_gt_co2eq_yr"] = float(total_gt)
    return row


def _run_plan(
    plan: ScenarioPlan,
    *,
    paths: DataPaths,
    shared_cfg: ScenarioConfig,
    shared_universe,
    shared_run_cache: Dict[str, object],
    specs: pd.DataFrame,
    cfg: Mapping[str, object],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    runs_dir = _runs_dir(cfg)
    scenario_dir = runs_dir / plan.scenario_id
    status = _status_base(plan, scenario_dir, shared_universe)
    param_rows = _build_param_rows(specs, cfg, include_kinds=plan.include_kinds, country=plan.country)
    design_rows = _strategy_design_rows(plan, param_rows, shared_universe)

    if bool(cfg.get("dry_run", False)):
        status["run_status"] = "dry_run"
        return status, design_rows

    if bool(cfg.get("resume", True)) and _can_resume(plan, scenario_dir):
        status["run_status"] = "resumed"
        status = _finalize_status_from_outputs(status, scenario_dir, cfg)
        if status["run_status"] == "ok":
            status["run_status"] = "resumed"
        return status, design_rows

    if not bool(cfg.get("resume", True)) and bool(cfg.get("clear_existing_run_dirs_when_no_resume", False)):
        _clear_scenario_dir(scenario_dir, runs_dir)

    try:
        effects = None
        if param_rows:
            effects = _build_effects(param_rows, shared_universe, cfg, scenario_id=plan.scenario_id)
        outdir = run_one_pipeline(
            paths,
            pre_macc_e0=False,
            scenario_id=plan.scenario_id,
            scenario_effects=effects,
            solve=True,
            use_fao_modules=True,
            save_root=str(runs_dir),
            future_last_only=True,
            use_linear=True,
            fast_emis_only=bool(plan.fast_emis_only),
            fast_emis_year=int(cfg.get("year", 2080) or 2080),
            prebuilt_config=shared_cfg,
            prebuilt_universe=shared_universe,
            prebuilt_run_cache=shared_run_cache,
        )
        scenario_dir = Path(outdir)
        status["scenario_dir"] = str(scenario_dir)
        status = _finalize_status_from_outputs(status, scenario_dir, cfg)
    except MCPrecheckFailed as exc:
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


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    _ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _add_reduction_columns(status_df: pd.DataFrame, baseline_scenario_id: str) -> pd.DataFrame:
    out = status_df.copy()
    ok_total = pd.to_numeric(out["afolu_emissions_gt_co2eq_yr"], errors="coerce")
    baseline_vals = out.loc[out["scenario_id"] == baseline_scenario_id, "afolu_emissions_gt_co2eq_yr"]
    baseline_total = pd.to_numeric(baseline_vals, errors="coerce").dropna()
    base = float(baseline_total.iloc[0]) if not baseline_total.empty else np.nan
    out["baseline_total_co2eq_gt"] = base
    out["emission_reduction_gt"] = base - ok_total if np.isfinite(base) else np.nan
    out["emission_reduction_pct"] = np.where(
        np.isfinite(base) & (base != 0),
        out["emission_reduction_gt"] / base * 100.0,
        np.nan,
    )
    return out


def _write_summary_outputs(
    *,
    status_df: pd.DataFrame,
    design_df: pd.DataFrame,
    cfg: Mapping[str, object],
    universe,
) -> None:
    out_dir = _output_dir(cfg)
    write_sensitivity_cost_summaries(status_df, output_dir=out_dir)
    baseline_id = str(cfg.get("baseline_scenario_id") or "S5_6_BASE")
    global_all_id = str(cfg.get("global_all_scenario_id") or "S5_6_GLOBAL_MAX_REDUCTION")
    reduction_df = _add_reduction_columns(status_df, baseline_id)
    _write_csv(reduction_df, out_dir / "scenario_reduction_summary.csv")

    usable = reduction_df[reduction_df["run_status"].isin(["ok", "resumed"])].copy()
    global_df = usable[(usable["scope"] == "global") & (usable["scenario_id"] != baseline_id)].copy()
    if not global_df.empty:
        global_df = global_df.sort_values("emission_reduction_gt", ascending=False)
    _write_csv(global_df, out_dir / "global_strategy_summary.csv")

    global_max = global_df.head(1).copy() if not global_df.empty else pd.DataFrame()
    _write_csv(global_max, out_dir / "global_max_reduction_summary.csv")
    if not global_max.empty:
        best_id = str(global_max.iloc[0]["scenario_id"])
        _write_csv(
            design_df[design_df["scenario_id"] == best_id].copy(),
            out_dir / "global_max_strategy_long.csv",
        )

    country_all_status = reduction_df[reduction_df["scope"] == "country"].copy()
    country_df = usable[usable["scope"] == "country"].copy()
    country_own = country_all_status[
        country_all_status["strategy_name"] == "country_own_max_endpoint_strategy"
    ].copy()
    if not country_own.empty:
        country_own["_sort_reduction_gt"] = pd.to_numeric(
            country_own.get("emission_reduction_gt"), errors="coerce"
        ).fillna(-np.inf)
        country_own = country_own.sort_values("_sort_reduction_gt", ascending=False).drop(
            columns=["_sort_reduction_gt"],
            errors="ignore",
        )
    _write_csv(country_own, out_dir / "country_own_max_reduction_summary.csv")
    # Backward-compatible filename: each row is one country applying its own
    # all-lever maximum endpoint strategy.
    _write_csv(country_own, out_dir / "country_single_actor_potential.csv")
    own_ids = set(country_own["scenario_id"].astype(str)) if not country_own.empty else set()
    own_strategy = design_df[design_df["scenario_id"].astype(str).isin(own_ids)].copy()
    _write_csv(own_strategy, out_dir / "country_own_max_strategy_long.csv")
    _write_csv(own_strategy, out_dir / "country_max_strategy_long.csv")

    if not country_df.empty:
        # This is a diagnostic "best among scenarios we actually tested". It
        # may differ from country_own_max_endpoint_strategy only when
        # run_country_individual_levers=True adds country-level single-lever
        # scenarios.
        country_best_tested = (
            country_df.sort_values("emission_reduction_gt", ascending=False)
            .groupby("country", as_index=False, sort=False)
            .head(1)
            .sort_values("emission_reduction_gt", ascending=False)
        )
    else:
        country_best_tested = pd.DataFrame()
    _write_csv(country_best_tested, out_dir / "country_best_tested_strategy_summary.csv")
    if not country_best_tested.empty:
        best_ids = set(country_best_tested["scenario_id"].astype(str))
        _write_csv(
            design_df[design_df["scenario_id"].astype(str).isin(best_ids)].copy(),
            out_dir / "country_best_tested_strategy_long.csv",
        )

    # Domestic country accounting under the global-all strategy, if detailed
    # country emission summaries are available for both baseline and global-all.
    base_row = reduction_df[reduction_df["scenario_id"] == baseline_id].head(1)
    global_row = reduction_df[reduction_df["scenario_id"] == global_all_id].head(1)
    if not base_row.empty and not global_row.empty:
        base_dir = Path(str(base_row.iloc[0].get("scenario_dir", "")))
        max_dir = Path(str(global_row.iloc[0].get("scenario_dir", "")))
        base_country = _read_country_co2eq(base_dir, year=int(cfg.get("year", 2080) or 2080), universe=universe)
        max_country = _read_country_co2eq(max_dir, year=int(cfg.get("year", 2080) or 2080), universe=universe)
        if not base_country.empty and not max_country.empty:
            merged = base_country.merge(
                max_country[["m49", "co2eq_kt", "co2eq_gt"]],
                on="m49",
                how="outer",
                suffixes=("_baseline", "_global_max"),
            )
            for col in ("co2eq_kt_baseline", "co2eq_kt_global_max", "co2eq_gt_baseline", "co2eq_gt_global_max"):
                merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
            merged["emission_reduction_kt"] = merged["co2eq_kt_baseline"] - merged["co2eq_kt_global_max"]
            merged["emission_reduction_gt"] = merged["emission_reduction_kt"] * 1e-6
            merged["emission_reduction_pct"] = np.where(
                merged["co2eq_kt_baseline"] != 0,
                merged["emission_reduction_kt"] / merged["co2eq_kt_baseline"] * 100.0,
                np.nan,
            )
            merged = merged.sort_values("emission_reduction_gt", ascending=False)
            _write_csv(merged, out_dir / "country_reduction_under_global_strategy.csv")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S5.6 maximum endpoint emission-reduction tests.")
    parser.add_argument("--output-dir", default="", help="Output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Only write scenario design/status tables.")
    parser.add_argument("--resume", action="store_true", default=None, help="Reuse existing outputs when available.")
    parser.add_argument("--no-resume", action="store_false", dest="resume", help="Rerun scenarios instead of reusing existing outputs.")
    parser.add_argument(
        "--clear-existing-runs",
        action="store_true",
        default=None,
        help="Delete existing scenario run directories before rerunning. Default is to keep them.",
    )
    parser.add_argument(
        "--keep-existing-runs",
        action="store_false",
        dest="clear_existing_runs",
        help="Keep existing scenario run directories before rerunning.",
    )
    parser.add_argument("--max-countries", type=int, default=None, help="Limit country one-at-a-time runs.")
    parser.add_argument(
        "--countries",
        default="",
        help="Comma-separated country selectors: M49, ISO3, or country name. Empty means all countries.",
    )
    parser.add_argument("--no-country-runs", action="store_true", help="Skip country one-at-a-time runs.")
    parser.add_argument("--country-individual-levers", action="store_true", help="Run country-by-country single-lever tests.")
    parser.add_argument("--no-global-individual-levers", action="store_true", help="Skip global single-lever tests.")
    parser.add_argument("--no-detailed-country-accounting", action="store_true", help="Do not run detailed baseline/global emissions.")
    parser.add_argument("--iis", action="store_true", help="Enable infeasible IIS diagnostics.")
    parser.add_argument("--iis-timeout", type=int, default=None, help="IIS timeout seconds.")
    parser.add_argument("--threads", type=int, default=None, help="Override linear_solver_threads.")
    return parser.parse_args(argv)


def _effective_config(args: argparse.Namespace) -> Dict[str, object]:
    cfg = copy.deepcopy(CONFIG)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.dry_run:
        cfg["dry_run"] = True
    if args.resume is not None:
        cfg["resume"] = bool(args.resume)
    if args.clear_existing_runs is not None:
        cfg["clear_existing_run_dirs_when_no_resume"] = bool(args.clear_existing_runs)
    if args.max_countries is not None:
        cfg["max_countries"] = int(args.max_countries)
    if args.countries:
        cfg["country_filter"] = [x.strip() for x in str(args.countries).split(",") if x.strip()]
    if args.no_country_runs:
        cfg["run_country_one_at_a_time"] = False
    if args.country_individual_levers:
        cfg["run_country_individual_levers"] = True
    if args.no_global_individual_levers:
        cfg["run_global_individual_levers"] = False
    if args.no_detailed_country_accounting:
        cfg["detailed_country_accounting"] = False
    override = copy.deepcopy(cfg.get("override_cfg", {}) or {})
    if args.iis:
        override["linear_enable_infeasible_iis"] = True
    if args.iis_timeout is not None:
        override["iis_timeout"] = int(args.iis_timeout)
    if args.threads is not None:
        override["linear_solver_threads"] = int(args.threads)
    cfg["override_cfg"] = override
    return cfg


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    cfg = _effective_config(args)
    out_dir = _output_dir(cfg)
    runs_dir = _runs_dir(cfg)
    _ensure_dir(out_dir)
    _ensure_dir(runs_dir)

    paths = DataPaths()
    shared_cfg = ScenarioConfig()
    shared_universe = build_universe_from_dict_v3(paths.dict_v3_path, shared_cfg)
    specs = _load_normalized_specs(cfg)
    plans = _build_plans(cfg, shared_universe, specs)

    print(f"[S5_6] output_dir={out_dir}")
    print(f"[S5_6] specs={len(specs)} eligible_kinds={list(_eligible_kinds(cfg, specs))}")
    print(f"[S5_6] planned scenarios={len(plans)} dry_run={bool(cfg.get('dry_run', False))}")
    print(
        f"[S5_6] resume={bool(cfg.get('resume', False))} "
        f"clear_existing_runs={bool(cfg.get('clear_existing_run_dirs_when_no_resume', False))}"
    )

    backup = _apply_cfg_overrides(cfg)
    try:
        if bool(cfg.get("dry_run", False)):
            shared_run_cache = {}
        else:
            shared_run_cache = build_run_baseline_cache(
                paths,
                shared_cfg,
                shared_universe,
                future_last_only=True,
            )

        status_rows: List[Dict[str, object]] = []
        design_rows: List[Dict[str, object]] = []
        for idx, plan in enumerate(plans, start=1):
            print(f"[S5_6] {idx}/{len(plans)} {plan.scenario_id} [{plan.scope}:{plan.strategy_name}]")
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
            design_rows.extend(rows)
            print(
                f"[S5_6] -> {status.get('run_status')} "
                f"emissions={status.get('afolu_emissions_gt_co2eq_yr')}"
            )

        status_df = pd.DataFrame(status_rows)
        design_df = pd.DataFrame(design_rows)
        _write_csv(status_df, out_dir / "scenario_status.csv")
        _write_csv(design_df, out_dir / "strategy_design_long.csv")
        _write_summary_outputs(
            status_df=status_df,
            design_df=design_df,
            cfg=cfg,
            universe=shared_universe,
        )

        print(f"[S5_6] wrote {out_dir / 'scenario_status.csv'}")
        print(f"[S5_6] wrote {out_dir / 'scenario_reduction_summary.csv'}")
    finally:
        _restore_cfg_overrides(backup)


if __name__ == "__main__":
    main()
