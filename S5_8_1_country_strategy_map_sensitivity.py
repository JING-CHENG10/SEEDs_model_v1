# -*- coding: utf-8 -*-
"""Country-by-strategy sensitivity runs for SP_M3b map data.

The workflow evaluates one detailed 2080 reference scenario and every
country-by-single-strategy endpoint scenario. Each strategy is applied only to
the selected country. The nine strategy categories and process-specific
emission-factor mappings are inherited from S5.7.

The reference run is also used as the S4 unit-cost reference directory. In
batch mode every batch runs its own reference scenario before its assigned
country-strategy scenarios, so batches can execute independently.

Main outputs:
- scenario_status.csv
- strategy_design_long.csv
- reference_country_emissions.csv
- country_strategy_long.csv
- country_max_reduction_strategy.csv
- country_strategy_argmax.csv
- country_min_unit_cost_strategy.csv
- country_strategy_argmin_cost.csv
- sp_m3b_map_data.csv
- Figure3.csv
- sensitivity_cost_summary_by_country_measure.csv
- sensitivity_cost_summary_by_global_measure.csv
- sensitivity_cost_summary_audit.csv
- figure3d/figure3d_country_strategy_ranked.csv
- figure3d/figure3d_country_dominant_intervention.csv
- figure3d/figure3d_metric_sensitivity_comparison.csv
- figure3d/Figure3d_country_dominant_intervention_source_data.xlsx
- figure3d/figure3d_data_quality.csv
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config_paths import get_results_base
import S5_6_1_max_emission_reduction_potential as s56
import S5_7_1_strategy_endpoint_rerun_max_reduction_potential as s57
from S5_cost_summary_outputs import write_sensitivity_cost_summaries


CODE_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_OUTPUT_ROOT = (CODE_ROOT / "output").resolve()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def ensure_output_child(path: Path) -> Path:
    """Resolve a runtime target and require a strict child of Code/output."""

    resolved = Path(path).expanduser().resolve()
    if resolved == ALLOWED_OUTPUT_ROOT or not _is_within(
        resolved, ALLOWED_OUTPUT_ROOT
    ):
        raise ValueError(
            "S5.8 output must be a strict child of Code/output; "
            f"got {resolved}"
        )
    return resolved


PLOT_PROCESS_NAMES = {
    "ruminant_reduction": "Reduce Ruminate",
    "losses_ratio": "Reduce waste",
    "yield_rate": "Improve yield rate",
    "feed_intensity": "Improve feed efficiency",
    "enteric_fermentation_management": "Enteric fermentation management",
    "manure_management": "Manure management",
    "crop_residue_soil_management": "Crop residue management",
    "rice_cultivation": "Rice cultivation",
    "fertilizer_efficiency": "Improve fertilizer efficiency",
}


CONFIG = {
    "year": 2080,
    "output_dir": "",
    "runs_subdir": "runs",
    "scenario_prefix": "S5_8",
    "baseline_scenario_id": "BASE_S5_8_REFERENCE",
    "nutrition_profile_sheet": "low_land_new",
    "mc_sheet_prefer": "MC_effect_low_land_new",
    "strategy_process_config_path": str(s57.DEFAULT_STRATEGY_PROCESS_CONFIG),
    "strategy_process_config_strict": True,
    "eligible_kinds": list(s57.STRATEGY_KIND_ORDER),
    "country_filter": [],
    "max_countries": None,
    "run_reference": True,
    "resume": False,
    "clear_existing_run_dirs_when_no_resume": False,
    "stop_on_error": False,
    "dry_run": False,
    "validate_fast_nonluc_emissions": True,
    "validate_market_balance": True,
    "market_gap_max_rate": 0.05,
    "minimum_positive_domestic_reduction_gt": 1e-12,
    "minimum_positive_cost_abatement_tco2eq": 1e-6,
    "write_summary_outputs": True,
    "sampling": copy.deepcopy(s57.CONFIG.get("sampling", {}) or {}),
    "endpoint_value_by_kind": copy.deepcopy(s57.CONFIG.get("endpoint_value_by_kind", {}) or {}),
    "endpoint_u_by_kind": copy.deepcopy(s57.CONFIG.get("endpoint_u_by_kind", {}) or {}),
    "unknown_kind_u": s57.CONFIG.get("unknown_kind_u", 0.5),
    "mc_non_ef_mode": s57.CONFIG.get("mc_non_ef_mode", "shared"),
    "mc_ef_mode": s57.CONFIG.get("mc_ef_mode", "shared"),
    "ef_process_mode": s57.CONFIG.get("ef_process_mode", "all"),
    "override_cfg": {
        **copy.deepcopy(s57.CONFIG.get("override_cfg", {}) or {}),
        "nutrition_profile_sheet": "low_land_new",
        # Figure 3d is a country-local intervention comparison against one
        # common reference, not one of the low/medium/high bioenergy panels.
        # Keep this explicit so a caller cannot inherit a panel a-c setting.
        "bioenergy_enabled": False,
        "cost_calculation_method": "unit_cost",
        "base_cost_calculation_method": "off",
        "debug_level": 0,
        "batch_mode": False,
        "linear_enable_infeasible_iis": False,
        "linear_enable_violation_iis": False,
        "linear_enable_output_diagnostics": False,
        "linear_enable_verbose_logging": False,
    },
}


def _default_output_dir() -> Path:
    return Path(get_results_base()) / "Country_Strategy_Map_Sensitivity"


def _output_dir(cfg: Mapping[str, object]) -> Path:
    raw = str(cfg.get("output_dir", "") or "").strip()
    return Path(raw) if raw else _default_output_dir()


def _runs_dir(cfg: Mapping[str, object]) -> Path:
    return _output_dir(cfg) / str(cfg.get("runs_subdir", "runs") or "runs")


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _reference_required_paths(scenario_dir: Path) -> Tuple[Path, ...]:
    return (
        scenario_dir / "Emis" / "emissions_summary_By_Country.csv",
        scenario_dir / "Emis" / "emissions_summary_By_Country_Process_Item.csv",
        scenario_dir / "DS" / "production_summary.csv",
        scenario_dir / "DS" / "market_summary.csv",
    )


def _reference_ready(scenario_dir: Path) -> bool:
    return all(path.exists() for path in _reference_required_paths(scenario_dir))


def _build_plans(cfg: Mapping[str, object], universe, specs: pd.DataFrame) -> List[s56.ScenarioPlan]:
    plans: List[s56.ScenarioPlan] = []
    baseline_id = str(cfg.get("baseline_scenario_id") or "BASE_S5_8_REFERENCE")
    if bool(cfg.get("run_reference", True)):
        plans.append(
            s56.ScenarioPlan(
                scenario_id=baseline_id,
                scope="baseline",
                strategy_name="reference",
                include_kinds=(),
                fast_emis_only=False,
                require_country_detail=True,
            )
        )

    countries = s56._active_countries(cfg, universe)
    eligible = s57._eligible_strategy_kinds(cfg, specs)
    prefix = str(cfg.get("scenario_prefix", "S5_8") or "S5_8")
    for country in countries:
        suffix = s56._country_scenario_suffix(country, universe)
        for kind in eligible:
            plans.append(
                s56.ScenarioPlan(
                    scenario_id=f"{prefix}_{suffix}_{s56._safe_token(kind).upper()}",
                    scope="country_strategy",
                    strategy_name=f"country_single_strategy_{kind}",
                    include_kinds=(kind,),
                    country=country,
                    fast_emis_only=False,
                    require_country_detail=True,
                )
            )
    return plans


def _status_base(plan: s56.ScenarioPlan, scenario_dir: Path, universe) -> Dict[str, object]:
    row = s56._status_base(plan, scenario_dir, universe)
    kind = plan.include_kinds[0] if len(plan.include_kinds) == 1 else ""
    row["strategy_kind"] = kind
    row["database_strategy"] = s57._database_strategy_for_kind(kind)
    row["strategy_display_name"] = s57.STRATEGY_DISPLAY_NAMES.get(kind, kind)
    row["plot_process"] = PLOT_PROCESS_NAMES.get(kind, kind)
    row["country_emissions_found"] = False
    row["cost_summary_found"] = False
    return row


def _can_resume(
    plan: s56.ScenarioPlan,
    scenario_dir: Path,
    *,
    cost_database_identity: Optional[Mapping[str, object]] = None,
    reference_scenario_id: str = "",
    strategy_cost_regions: Optional[Sequence[str]] = None,
    expected_resume_fingerprint: Optional[str] = None,
) -> bool:
    active_keys = s57._active_strategy_cost_keys(plan)
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
        expected_fingerprint = s57._cost_resume_fingerprint(
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
    if plan.scope == "baseline":
        return all(
            s56._artifact_can_resume(
                plan,
                scenario_dir,
                path,
                expected_resume_fingerprint=expected_fingerprint,
            )
            for path in _reference_required_paths(scenario_dir)
        )
    if not s56._artifact_can_resume(
        plan,
        scenario_dir,
        scenario_dir / "cost_summary.csv",
        expected_resume_fingerprint=expected_fingerprint,
    ):
        return False
    if not active_keys:
        return True
    return s57._cost_summary_matches_singleton(
        scenario_dir,
        database_strategy=active_keys[0],
        identity=cost_database_identity or {},
        reference_scenario_id=reference_scenario_id,
        strategy_cost_regions=strategy_cost_regions,
    )


def _reuse_external_reference(
    *,
    baseline_id: str,
    runs_dir: Path,
    universe,
    database_identity: Mapping[str, object],
    cfg: Mapping[str, object],
) -> Tuple[bool, Optional[Dict[str, object]]]:
    """Reuse an externally generated BASE only when its full run provenance matches."""
    reference_plan = s56.ScenarioPlan(
        scenario_id=baseline_id,
        scope="baseline",
        strategy_name="reference",
        include_kinds=(),
        fast_emis_only=False,
        require_country_detail=True,
    )
    scenario_dir = runs_dir / baseline_id
    design_signature = s57._cost_design_signature(reference_plan, ())
    reference_status = _status_base(reference_plan, scenario_dir, universe)
    reference_status = s57._decorate_cost_status(
        reference_status,
        reference_plan,
        database_identity,
        baseline_id,
        strategy_cost_regions=None,
        design_signature=design_signature,
    )
    resume_fingerprint = str(reference_status["cost_resume_fingerprint"])
    if not _can_resume(
        reference_plan,
        scenario_dir,
        cost_database_identity=database_identity,
        reference_scenario_id=baseline_id,
        strategy_cost_regions=None,
        expected_resume_fingerprint=resume_fingerprint,
    ):
        return False, None

    reference_status = s56._finalize_status_from_outputs(
        reference_status,
        scenario_dir,
        cfg,
    )
    if reference_status.get("run_status") != "ok":
        return False, None
    reference_status["run_status"] = "resumed"
    reference_status["country_emissions_found"] = True
    reference_status["cost_summary_found"] = (scenario_dir / "cost_summary.csv").exists()
    return True, reference_status


def _run_plan(
    plan: s56.ScenarioPlan,
    *,
    paths,
    shared_cfg,
    shared_universe,
    shared_run_cache: Dict[str, object],
    specs: pd.DataFrame,
    cfg: Mapping[str, object],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    runs_dir = _runs_dir(cfg)
    scenario_dir = runs_dir / plan.scenario_id
    status = _status_base(plan, scenario_dir, shared_universe)
    database_identity = s57._cost_database_identity(paths)
    baseline_id = str(cfg.get("baseline_scenario_id") or "BASE_S5_8_REFERENCE")
    active_cost_keys = s57._active_strategy_cost_keys(plan)
    strategy_cost_regions = (
        (str(plan.country),) if active_cost_keys and plan.country else None
    )
    param_rows = s57._build_param_rows(
        specs,
        cfg,
        include_kinds=plan.include_kinds,
        country=plan.country,
    )
    design_signature = s57._cost_design_signature(plan, param_rows)
    status = s57._decorate_cost_status(
        status,
        plan,
        database_identity,
        baseline_id,
        strategy_cost_regions=strategy_cost_regions,
        design_signature=design_signature,
    )
    resume_fingerprint = str(status["cost_resume_fingerprint"])
    design_rows = s57._strategy_design_rows(plan, param_rows, shared_universe)

    if bool(cfg.get("dry_run", False)):
        status["run_status"] = "dry_run"
        return status, design_rows

    if bool(cfg.get("resume", True)) and _can_resume(
        plan,
        scenario_dir,
        cost_database_identity=database_identity,
        reference_scenario_id=baseline_id,
        strategy_cost_regions=strategy_cost_regions,
        expected_resume_fingerprint=resume_fingerprint,
    ):
        status = s56._finalize_status_from_outputs(status, scenario_dir, cfg)
        if status.get("run_status") == "ok":
            status["run_status"] = "resumed"
        status["country_emissions_found"] = s56._country_detail_path(scenario_dir).exists()
        status["cost_summary_found"] = (scenario_dir / "cost_summary.csv").exists()
        return status, design_rows

    if (
        not bool(cfg.get("resume", True))
        and bool(cfg.get("clear_existing_run_dirs_when_no_resume", False))
    ):
        s56._clear_scenario_dir(scenario_dir, runs_dir)

    try:
        effects = None
        if param_rows:
            effects = s57._build_effects(
                param_rows,
                shared_universe,
                cfg,
                scenario_id=plan.scenario_id,
            )
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
            fast_emis_only=False,
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
        status["country_emissions_found"] = s56._country_detail_path(scenario_dir).exists()
        status["cost_summary_found"] = (scenario_dir / "cost_summary.csv").exists()
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


def _country_emission_lookup(
    scenario_dir: Path,
    *,
    year: int,
    universe,
) -> Dict[str, float]:
    detail = s56._read_country_co2eq(scenario_dir, year=year, universe=universe)
    if detail.empty:
        return {}
    return {
        s56._normalize_m49(row.m49): float(row.co2eq_gt)
        for row in detail[["m49", "co2eq_gt"]].itertuples(index=False)
        if pd.notna(row.co2eq_gt)
    }


def _read_cost_metrics(
    scenario_dir: Path,
    *,
    country: str,
    year: int,
    minimum_abatement: float,
    database_strategy: str = "",
    reference_scenario_id: str = "",
    cost_database_version: str = "",
    cost_database_sha256: str = "",
) -> Dict[str, object]:
    path = scenario_dir / "cost_summary.csv"
    empty = {
        "database_strategy": str(database_strategy or ""),
        "country_cost_rows": 0,
        "country_cost_abatement_tco2eq": np.nan,
        "country_cost_abatement_gt": np.nan,
        "country_total_cost_usd": np.nan,
        "weighted_unit_cost_usd_per_tco2eq": np.nan,
        "country_cost_is_priced": False,
        "country_cost_unpriced_reason": "missing_strategy_cost_row",
    }
    if not path.exists():
        return empty
    try:
        detail = pd.read_csv(path)
    except Exception:
        return empty
    required = {
        "region",
        "year",
        "cost_component_type",
        "database_strategy",
        "is_priced",
        "abatement_tco2eq",
        "unit_cost_usd_per_tco2eq",
        "total_cost_usd",
    }
    if detail.empty or not required.issubset(detail.columns) or not database_strategy:
        return empty
    work = detail.copy()
    work["m49"] = work["region"].map(s56._normalize_m49)
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    for col in ("abatement_tco2eq", "unit_cost_usd_per_tco2eq", "total_cost_usd"):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    priced_mask = s57._summary_bool(work["is_priced"])
    expected_component = s57._cost_component_type_for_database_strategy(database_strategy)
    selection = (
        work["m49"].eq(s56._normalize_m49(country))
        & work["year"].eq(int(year))
        & work["cost_component_type"].astype(str).str.strip().str.lower().eq(expected_component)
        & work["database_strategy"].astype(str).str.strip().eq(str(database_strategy))
        & work["unit_cost_usd_per_tco2eq"].notna()
        & work["total_cost_usd"].notna()
        & priced_mask
    )
    if reference_scenario_id:
        if "reference_scenario_id" not in work.columns:
            return empty
        selection &= work["reference_scenario_id"].fillna("").astype(str).str.strip().eq(
            str(reference_scenario_id)
        )
    for column, expected in (
        ("cost_database_version", cost_database_version),
        ("cost_database_sha256", cost_database_sha256),
    ):
        if not expected:
            continue
        if column not in work.columns:
            return empty
        selection &= work[column].fillna("").astype(str).str.strip().eq(str(expected))
    priced_work = work[selection].copy()
    if priced_work.empty:
        return empty
    work = priced_work[
        priced_work["abatement_tco2eq"].gt(float(minimum_abatement))
    ].copy()
    if work.empty:
        return {
            "database_strategy": str(database_strategy),
            "country_cost_rows": int(len(priced_work)),
            "country_cost_abatement_tco2eq": 0.0,
            "country_cost_abatement_gt": 0.0,
            "country_total_cost_usd": float(priced_work["total_cost_usd"].sum()),
            "weighted_unit_cost_usd_per_tco2eq": np.nan,
            "country_cost_is_priced": True,
            "country_cost_unpriced_reason": "no_positive_abatement_opportunity",
        }
    abatement = float(work["abatement_tco2eq"].sum())
    total_cost = float(work["total_cost_usd"].sum())
    return {
        "database_strategy": str(database_strategy),
        "country_cost_rows": int(len(work)),
        "country_cost_abatement_tco2eq": abatement,
        "country_cost_abatement_gt": abatement * 1e-9,
        "country_total_cost_usd": total_cost,
        "weighted_unit_cost_usd_per_tco2eq": (
            total_cost / abatement if abatement > 0 else np.nan
        ),
        # Explicit zero unit cost remains a priced observation.
        "country_cost_is_priced": True,
        "country_cost_unpriced_reason": "",
    }


def _build_country_strategy_long(
    status_df: pd.DataFrame,
    cfg: Mapping[str, object],
    universe,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    year = int(cfg.get("year", 2080) or 2080)
    baseline_id = str(cfg.get("baseline_scenario_id") or "BASE_S5_8_REFERENCE")
    baseline_rows = status_df[
        status_df.get("scenario_id", pd.Series(dtype=str)).astype(str).eq(baseline_id)
    ]
    baseline_dir = (
        Path(str(baseline_rows.iloc[0].get("scenario_dir", "") or ""))
        if not baseline_rows.empty
        else Path()
    )
    baseline_global_emissions = (
        pd.to_numeric(
            baseline_rows.get(
                "afolu_emissions_gt_co2eq_yr",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        ).dropna()
    )
    baseline_global_emissions_gt = (
        float(baseline_global_emissions.iloc[0])
        if not baseline_global_emissions.empty
        else np.nan
    )
    baseline_lookup = (
        _country_emission_lookup(baseline_dir, year=year, universe=universe)
        if baseline_dir.exists()
        else {}
    )
    reference_rows: List[Dict[str, object]] = []
    for country, value in baseline_lookup.items():
        info = s56._country_label(country, universe)
        reference_rows.append(
            {
                "M49_Country_Code": country,
                "ISO3": info.get("iso3", ""),
                "country_name": info.get("country_name", ""),
                "Region_aggMC": info.get("region_aggMC", ""),
                "year": year,
                "reference_emissions_gt": value,
                "reference_global_afolu_emissions_gt": baseline_global_emissions_gt,
                "reference_scenario_id": baseline_id,
                "reference_scenario_dir": str(baseline_dir),
            }
        )
    reference_columns = [
        "M49_Country_Code",
        "ISO3",
        "country_name",
        "Region_aggMC",
        "year",
        "reference_emissions_gt",
        "reference_global_afolu_emissions_gt",
        "reference_scenario_id",
        "reference_scenario_dir",
    ]
    reference_df = pd.DataFrame(reference_rows, columns=reference_columns)

    rows: List[Dict[str, object]] = []
    country_status = status_df[
        status_df.get("scope", pd.Series(dtype=str)).astype(str).eq("country_strategy")
    ].copy()
    minimum_reduction = float(
        cfg.get("minimum_positive_domestic_reduction_gt", 1e-12) or 1e-12
    )
    minimum_cost_abatement = float(
        cfg.get("minimum_positive_cost_abatement_tco2eq", 1e-6) or 1e-6
    )
    override_cfg = cfg.get("override_cfg", {}) or {}
    bioenergy_enabled = bool(override_cfg.get("bioenergy_enabled", False))
    bioenergy_scenario = str(override_cfg.get("bioenergy_scenario", "") or "")
    figure3d_reference_context = (
        f"bioenergy_{bioenergy_scenario}" if bioenergy_enabled and bioenergy_scenario
        else "bioenergy_enabled_unspecified" if bioenergy_enabled
        else "bioenergy_disabled_reference"
    )
    for _, status in country_status.iterrows():
        country = s56._normalize_m49(status.get("country", ""))
        kind = str(status.get("strategy_kind", "") or "")
        scenario_dir = Path(str(status.get("scenario_dir", "") or ""))
        scenario_lookup = (
            _country_emission_lookup(scenario_dir, year=year, universe=universe)
            if scenario_dir.exists()
            else {}
        )
        baseline_emissions = baseline_lookup.get(country, np.nan)
        scenario_emissions = scenario_lookup.get(country, np.nan)
        reduction = (
            float(baseline_emissions) - float(scenario_emissions)
            if np.isfinite(baseline_emissions) and np.isfinite(scenario_emissions)
            else np.nan
        )
        cost = _read_cost_metrics(
            scenario_dir,
            country=country,
            year=year,
            minimum_abatement=minimum_cost_abatement,
            database_strategy=s57._database_strategy_for_kind(kind),
            reference_scenario_id=baseline_id,
            cost_database_version=str(cfg.get("cost_database_version", "") or ""),
            cost_database_sha256=str(cfg.get("cost_database_sha256", "") or ""),
        )
        total_cost = float(cost["country_total_cost_usd"]) if pd.notna(cost["country_total_cost_usd"]) else np.nan
        scenario_global_emissions_gt = pd.to_numeric(
            pd.Series([status.get("afolu_emissions_gt_co2eq_yr", np.nan)]),
            errors="coerce",
        ).iloc[0]
        global_system_reduction_gt = (
            float(baseline_global_emissions_gt) - float(scenario_global_emissions_gt)
            if np.isfinite(baseline_global_emissions_gt)
            and np.isfinite(scenario_global_emissions_gt)
            else np.nan
        )
        effective_cost = (
            total_cost / (reduction * 1e9)
            if np.isfinite(total_cost) and np.isfinite(reduction) and reduction > minimum_reduction
            else np.nan
        )
        info = s56._country_label(country, universe)
        weighted_cost = cost["weighted_unit_cost_usd_per_tco2eq"]
        usable_status = str(status.get("run_status", "")) in {"ok", "resumed"}
        max_eligible = bool(
            usable_status and np.isfinite(reduction) and reduction > minimum_reduction
        )
        cost_eligible = bool(
            max_eligible
            and bool(cost.get("country_cost_is_priced", False))
            and pd.notna(weighted_cost)
            and np.isfinite(float(weighted_cost))
            and float(cost.get("country_cost_abatement_tco2eq") or 0.0)
            > minimum_cost_abatement
        )
        rows.append(
            {
                "scenario_id": status.get("scenario_id", ""),
                "scenario_dir": str(scenario_dir),
                "run_status": status.get("run_status", ""),
                "M49_Country_Code": country,
                "ISO3": info.get("iso3", status.get("iso3", "")),
                "country_name": info.get("country_name", status.get("country_name", "")),
                "Region_aggMC": info.get("region_aggMC", status.get("region_aggMC", "")),
                "year": year,
                "figure3d_reference_context": figure3d_reference_context,
                "bioenergy_enabled": bioenergy_enabled,
                "bioenergy_scenario": bioenergy_scenario,
                "strategy_kind": kind,
                "database_strategy": s57._database_strategy_for_kind(kind),
                "strategy_display_name": s57.STRATEGY_DISPLAY_NAMES.get(kind, kind),
                "plot_process": PLOT_PROCESS_NAMES.get(kind, kind),
                "strategy_order": (
                    s57.STRATEGY_KIND_ORDER.index(kind)
                    if kind in s57.STRATEGY_KIND_ORDER
                    else 999
                ),
                "reference_emissions_gt": baseline_emissions,
                "reference_global_afolu_emissions_gt": baseline_global_emissions_gt,
                "strategy_emissions_gt": scenario_emissions,
                "domestic_emission_reduction_gt": reduction,
                "domestic_emission_reduction_pct": (
                    reduction / baseline_emissions * 100.0
                    if np.isfinite(reduction)
                    and np.isfinite(baseline_emissions)
                    and baseline_emissions != 0
                    else np.nan
                ),
                "global_afolu_emissions_gt": scenario_global_emissions_gt,
                "global_system_emission_reduction_gt": global_system_reduction_gt,
                "cost_database_version": status.get("cost_database_version", ""),
                "cost_database_sha256": status.get("cost_database_sha256", ""),
                "cost_reference_scenario_id": status.get(
                    "cost_reference_scenario_id", ""
                ),
                "cost_attribution_method": status.get(
                    "cost_attribution_method", ""
                ),
                "active_strategy_cost_keys": status.get(
                    "active_strategy_cost_keys", ""
                ),
                "cost_strategy_regions": status.get("cost_strategy_regions", ""),
                "cost_design_signature": status.get("cost_design_signature", ""),
                "cost_resume_fingerprint": status.get(
                    "cost_resume_fingerprint", ""
                ),
                **cost,
                "effective_unit_cost_per_net_domestic_reduction_usd_per_tco2eq": effective_cost,
                "max_reduction_eligible": max_eligible,
                "min_cost_eligible": cost_eligible,
                "error_type": status.get("error_type", ""),
                "error_message": status.get("error_message", ""),
            }
        )
    long_df = pd.DataFrame(rows)
    if not long_df.empty:
        long_df = long_df.sort_values(
            ["M49_Country_Code", "strategy_order", "strategy_kind"]
        )
    return long_df, reference_df


def _select_country_strategies(
    long_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if long_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    def _as_bool(series: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(series):
            return series.fillna(False)
        return series.astype(str).str.strip().str.lower().isin(
            {"1", "true", "yes", "y"}
        )

    max_candidates = long_df[_as_bool(long_df["max_reduction_eligible"])].copy()
    if not max_candidates.empty:
        max_candidates["_cost_tie"] = pd.to_numeric(
            max_candidates["weighted_unit_cost_usd_per_tco2eq"],
            errors="coerce",
        ).fillna(np.inf)
        max_selected = (
            max_candidates.sort_values(
                [
                    "M49_Country_Code",
                    "domestic_emission_reduction_gt",
                    "_cost_tie",
                    "strategy_order",
                ],
                ascending=[True, False, True, True],
            )
            .groupby("M49_Country_Code", as_index=False, sort=False)
            .head(1)
            .drop(columns=["_cost_tie"], errors="ignore")
        )
    else:
        max_selected = long_df.iloc[0:0].copy()

    cost_candidates = long_df[_as_bool(long_df["min_cost_eligible"])].copy()
    if not cost_candidates.empty:
        cost_selected = (
            cost_candidates.sort_values(
                [
                    "M49_Country_Code",
                    "weighted_unit_cost_usd_per_tco2eq",
                    "domestic_emission_reduction_gt",
                    "strategy_order",
                ],
                ascending=[True, True, False, True],
            )
            .groupby("M49_Country_Code", as_index=False, sort=False)
            .head(1)
        )
    else:
        cost_selected = long_df.iloc[0:0].copy()
    return max_selected, cost_selected


def _build_map_data(
    long_df: pd.DataFrame,
    max_selected: pd.DataFrame,
    cost_selected: pd.DataFrame,
) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(
            columns=[
                "M49_Country_Code",
                "Region_label_new",
                "Region_aggMC",
                "ISO3",
                "Max_Process",
                "Cost_Process",
            ]
        )
    countries = (
        long_df[
            ["M49_Country_Code", "country_name", "Region_aggMC", "ISO3"]
        ]
        .drop_duplicates("M49_Country_Code")
        .copy()
    )
    if not max_selected.empty:
        max_map = max_selected[
            [
                "M49_Country_Code",
                "plot_process",
                "domestic_emission_reduction_gt",
                "domestic_emission_reduction_pct",
            ]
        ].rename(
            columns={
                "plot_process": "Max_Process",
                "domestic_emission_reduction_gt": "Max_Reduction_Gt",
                "domestic_emission_reduction_pct": "Max_Reduction_Pct",
            }
        )
        countries = countries.merge(max_map, on="M49_Country_Code", how="left")
    else:
        countries["Max_Process"] = ""
        countries["Max_Reduction_Gt"] = np.nan
        countries["Max_Reduction_Pct"] = np.nan
    if not cost_selected.empty:
        cost_map = cost_selected[
            [
                "M49_Country_Code",
                "plot_process",
                "weighted_unit_cost_usd_per_tco2eq",
                "domestic_emission_reduction_gt",
            ]
        ].rename(
            columns={
                "plot_process": "Cost_Process",
                "weighted_unit_cost_usd_per_tco2eq": "Min_Unit_Cost_USD_per_tCO2eq",
                "domestic_emission_reduction_gt": "Cost_Strategy_Reduction_Gt",
            }
        )
        countries = countries.merge(cost_map, on="M49_Country_Code", how="left")
    else:
        countries["Cost_Process"] = ""
        countries["Min_Unit_Cost_USD_per_tCO2eq"] = np.nan
        countries["Cost_Strategy_Reduction_Gt"] = np.nan
    countries = countries.rename(columns={"country_name": "Region_label_new"})
    return countries.sort_values("M49_Country_Code")


def _write_summary_outputs(
    *,
    status_df: pd.DataFrame,
    cfg: Mapping[str, object],
    universe,
) -> None:
    out_dir = _output_dir(cfg)
    write_sensitivity_cost_summaries(status_df, output_dir=out_dir)
    long_df, reference_df = _build_country_strategy_long(status_df, cfg, universe)
    max_selected, cost_selected = _select_country_strategies(long_df)
    map_df = _build_map_data(long_df, max_selected, cost_selected)
    _write_csv(reference_df, out_dir / "reference_country_emissions.csv")
    _write_csv(long_df, out_dir / "country_strategy_long.csv")
    _write_csv(max_selected, out_dir / "country_max_reduction_strategy.csv")
    _write_csv(max_selected, out_dir / "country_strategy_argmax.csv")
    _write_csv(cost_selected, out_dir / "country_min_unit_cost_strategy.csv")
    _write_csv(cost_selected, out_dir / "country_strategy_argmin_cost.csv")
    _write_csv(map_df, out_dir / "sp_m3b_map_data.csv")
    _write_csv(map_df, out_dir / "Figure3.csv")
    import S5_8_3_prepare_country_dominant_mitigation_intervention as figure3d

    figure3d.write_figure3d_outputs(
        long_df,
        status_df,
        output_dir=out_dir / "figure3d",
        settings=figure3d.Figure3dSettings(
            input_dir=out_dir,
            output_dir=out_dir / "figure3d",
            metric="domestic",
            require_v2_provenance=True,
            strict=False,
            plot=False,
        ),
        source_files=(
            out_dir / "country_strategy_long.csv",
            out_dir / "scenario_status.csv",
        ),
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run S5.8 country-by-single-strategy sensitivity scenarios."
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--strategy-process-config", type=str, default=None)
    parser.add_argument("--country", action="append", default=[])
    parser.add_argument("--countries", type=str, default=None)
    parser.add_argument("--max-countries", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", default=None)
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--clear-existing-runs", action="store_true", default=None)
    parser.add_argument(
        "--keep-existing-runs",
        action="store_false",
        dest="clear_existing_runs",
    )
    parser.add_argument("--no-reference", action="store_true", default=None)
    parser.add_argument("--stop-on-error", action="store_true", default=None)
    parser.add_argument("--threads", type=int, default=None)
    return parser


def _effective_config(args: argparse.Namespace) -> Dict[str, object]:
    cfg = copy.deepcopy(CONFIG)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.strategy_process_config:
        cfg["strategy_process_config_path"] = args.strategy_process_config
    selectors: List[str] = []
    if args.countries:
        selectors.extend(
            part.strip() for part in str(args.countries).split(",") if part.strip()
        )
    selectors.extend(str(value).strip() for value in args.country if str(value).strip())
    if selectors:
        cfg["country_filter"] = selectors
    if args.max_countries is not None:
        cfg["max_countries"] = int(args.max_countries)
    if args.dry_run is not None:
        cfg["dry_run"] = bool(args.dry_run)
    if args.resume is not None:
        cfg["resume"] = bool(args.resume)
    if args.clear_existing_runs is not None:
        cfg["clear_existing_run_dirs_when_no_resume"] = bool(args.clear_existing_runs)
    if args.no_reference:
        cfg["run_reference"] = False
    if args.stop_on_error:
        cfg["stop_on_error"] = True
    override = copy.deepcopy(cfg.get("override_cfg", {}) or {})
    if args.threads is not None:
        override["linear_solver_threads"] = int(args.threads)
    cfg["override_cfg"] = override
    return cfg


def _checkpoint(
    status_rows: Sequence[Mapping[str, object]],
    design_rows: Sequence[Mapping[str, object]],
    cfg: Mapping[str, object],
) -> None:
    out_dir = _output_dir(cfg)
    _write_csv(pd.DataFrame(status_rows), out_dir / "scenario_status.csv")
    _write_csv(pd.DataFrame(design_rows), out_dir / "strategy_design_long.csv")


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _build_arg_parser().parse_args(list(argv) if argv is not None else None)
    cfg = _effective_config(args)
    out_dir = ensure_output_child(_output_dir(cfg))
    cfg["output_dir"] = str(out_dir)
    runs_dir = _runs_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    baseline_id = str(cfg.get("baseline_scenario_id") or "BASE_S5_8_REFERENCE")
    override = copy.deepcopy(cfg.get("override_cfg", {}) or {})
    override["base_case"] = baseline_id
    override["base_reference_dir"] = str(runs_dir / baseline_id)
    cfg["override_cfg"] = override

    paths = s56.DataPaths()
    database_identity = s57._cost_database_identity(paths)
    cfg.update(database_identity)
    shared_cfg = s56.ScenarioConfig()
    shared_universe = s56.build_universe_from_dict_v3(paths.dict_v3_path, shared_cfg)
    specs = s57._prepare_strategy_specs(s56._load_normalized_specs(cfg), cfg)
    plans = _build_plans(cfg, shared_universe, specs)
    countries = s56._active_countries(cfg, shared_universe)
    eligible = s57._eligible_strategy_kinds(cfg, specs)

    print(f"[S5_8] output_dir={out_dir}")
    print(f"[S5_8] reference_dir={runs_dir / baseline_id}")
    print(
        "[S5_8] cost_database="
        f"{database_identity.get('cost_database_version') or 'unknown'} "
        f"sha256={str(database_identity.get('cost_database_sha256') or '')[:16]}..."
    )
    print(
        f"[S5_8] countries={len(countries)} strategies={len(eligible)} "
        f"planned_scenarios={len(plans)} dry_run={bool(cfg.get('dry_run', False))}"
    )

    backup = s56._apply_cfg_overrides(cfg)
    try:
        shared_run_cache = (
            {}
            if bool(cfg.get("dry_run", False))
            else s56.build_run_baseline_cache(
                paths,
                shared_cfg,
                shared_universe,
                future_last_only=True,
            )
        )
        status_rows: List[Dict[str, object]] = []
        design_rows: List[Dict[str, object]] = []
        dry_run = bool(cfg.get("dry_run", False))
        run_reference = bool(cfg.get("run_reference", True))
        reference_available = dry_run or (
            run_reference and _reference_ready(runs_dir / baseline_id)
        )
        if not run_reference and not dry_run:
            reference_available, reference_status = _reuse_external_reference(
                baseline_id=baseline_id,
                runs_dir=runs_dir,
                universe=shared_universe,
                database_identity=database_identity,
                cfg=cfg,
            )
            if reference_status is not None:
                status_rows.append(reference_status)
        for index, plan in enumerate(plans, start=1):
            if plan.scope != "baseline" and not reference_available:
                status = _status_base(
                    plan,
                    runs_dir / plan.scenario_id,
                    shared_universe,
                )
                param_rows = s57._build_param_rows(
                    specs,
                    cfg,
                    include_kinds=plan.include_kinds,
                    country=plan.country,
                )
                active_cost_keys = s57._active_strategy_cost_keys(plan)
                strategy_cost_regions = (
                    (str(plan.country),) if active_cost_keys and plan.country else None
                )
                status = s57._decorate_cost_status(
                    status,
                    plan,
                    database_identity,
                    baseline_id,
                    strategy_cost_regions=strategy_cost_regions,
                    design_signature=s57._cost_design_signature(plan, param_rows),
                )
                status["run_status"] = "reference_unavailable"
                status["error_type"] = "MissingReferenceScenario"
                status["error_message"] = (
                    "Required reference run is missing or has incompatible provenance "
                    f"under {runs_dir / baseline_id}"
                )
                rows = s57._strategy_design_rows(
                    plan,
                    param_rows,
                    shared_universe,
                )
            else:
                print(
                    f"[S5_8] {index}/{len(plans)} {plan.scenario_id} "
                    f"[{plan.scope}:{plan.strategy_name}]"
                )
                status, rows = _run_plan(
                    plan,
                    paths=paths,
                    shared_cfg=shared_cfg,
                    shared_universe=shared_universe,
                    shared_run_cache=shared_run_cache,
                    specs=specs,
                    cfg=cfg,
                )
                if plan.scope == "baseline":
                    reference_available = (
                        str(status.get("run_status", "")) in {"ok", "resumed", "dry_run"}
                        and (
                            bool(cfg.get("dry_run", False))
                            or _reference_ready(Path(str(status.get("scenario_dir", ""))))
                        )
                    )
            status_rows.append(status)
            design_rows.extend(rows)
            _checkpoint(status_rows, design_rows, cfg)
            print(
                f"[S5_8] status={status.get('run_status')} "
                f"global_emissions={status.get('afolu_emissions_gt_co2eq_yr')}"
            )

        status_df = pd.DataFrame(status_rows)
        if bool(cfg.get("write_summary_outputs", True)):
            _write_summary_outputs(
                status_df=status_df,
                cfg=cfg,
                universe=shared_universe,
            )
    finally:
        s56._restore_cfg_overrides(backup)

    print(f"[S5_8] wrote {out_dir / 'country_strategy_long.csv'}")
    print(f"[S5_8] wrote {out_dir / 'sp_m3b_map_data.csv'}")


if __name__ == "__main__":
    main()
