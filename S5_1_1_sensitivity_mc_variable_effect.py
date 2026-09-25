# -*- coding: utf-8 -*-
"""
MC sensitivity runner dedicated to:
  A) yield + feed intensity
  B) emission factor (intensity)

This script only generates experimental data for plotting (no plotting).
MC bounds are treated as direct values using Element unit semantics
(rate/multiplier/amount/absolute; default absolute). Outputs include 2080 AFOLU
GHG emissions (CO2eq, Gt) samples for each group.
"""

# Functional overview (S5_1)
# Purpose: run sensitivity experiments in variable_effect and mc modes.
# Inputs: Scenario_config_new.xlsx and baseline/emissions parameter tables.
# Sheet selection:
# variable_effect reads `MC_effect_low_land_new` (overridable via CONFIG["variable_effect"]["mc_sheet_prefer"]).
# mc reads `MC` through load_mc_specs(), with sheet_name='MC' fixed.
# Outputs: targets.csv, <group>/summary/samples.csv, MC/*.xlsx.

from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from config_paths import get_input_base, get_results_base, get_src_base
from market_balance_diagnostics import (
    validate_market_balance_gap as _validate_solver_market_balance_gap,
)
from model_run_status import (
    ResumeValidation,
    artifact_matches_validated_run,
    build_resume_fingerprint,
    validate_run_for_resume,
)
from S2_0_load_data import (
    DataPaths,
    ScenarioConfig,
    build_universe_from_dict_v3,
    load_production_statistics,
)
from S3_0_ds_linear_regional import (
    _final_loss_ratio_from_delta,
    _load_demand_composition_losses_ratio,
    _load_item_demand_extra_and_map,
    _loss_multiplier_from_delta,
    _normalize_comp_item_name,
)
from S3_6_scenarios import (
    ScenarioEffect,
    _normalize_mc_unit,
    _parse_mc_bound_value,
    _select_commodities,
    _select_countries,
    _select_processes,
    load_mc_specs,
)
from S4_0_main import CFG, MCPrecheckFailed, build_run_baseline_cache, run_one_pipeline
from S5_cost_summary_outputs import write_sensitivity_cost_summaries

DEFAULT_NUTRITION_PROFILE_SHEET = "low_land_new"
LOW_LAND_NEW_NUTRITION_PROFILE_SHEET = "low_land_new"
DEFAULT_MC_EFFECT_SHEET = "MC_effect"
LOW_LAND_NEW_MC_EFFECT_SHEET = "MC_effect_low_land_new"
RUMINANT_REDUCTION_DISCRETE_LEVELS = tuple(round(i / 100.0, 2) for i in range(101))
DEFAULT_DISCRETE_LEVELS_BY_KIND = {
    "ruminant_reduction": RUMINANT_REDUCTION_DISCRETE_LEVELS,
}


def _is_low_land_new_profile_sheet(sheet_name: object) -> bool:
    return str(sheet_name or "").strip().lower() == LOW_LAND_NEW_NUTRITION_PROFILE_SHEET


def resolve_mc_effect_sheet(
    prefer_sheet: object = DEFAULT_MC_EFFECT_SHEET,
    *,
    nutrition_profile_sheet: object = None,
) -> str:
    sheet = str(prefer_sheet or DEFAULT_MC_EFFECT_SHEET).strip() or DEFAULT_MC_EFFECT_SHEET
    if _is_low_land_new_profile_sheet(nutrition_profile_sheet) and sheet == DEFAULT_MC_EFFECT_SHEET:
        return LOW_LAND_NEW_MC_EFFECT_SHEET
    return sheet


def _resolve_mc_modes(cfg: Dict[str, object]) -> Tuple[str, str, str]:
    """Return normalized non-EF, EF, and EF-process sampling modes."""
    mc_mode_non_ef = str(cfg.get("mc_non_ef_mode", "shared")).strip().lower() or "shared"
    mc_mode_ef = str(cfg.get("mc_ef_mode", "shared")).strip().lower() or "shared"
    ef_process_mode = str(cfg.get("ef_process_mode", "all")).strip().lower() or "all"
    return mc_mode_non_ef, mc_mode_ef, ef_process_mode


def _normalize_low_land_new_mc_specs(
    specs_df: pd.DataFrame,
    *,
    sheet_name: object,
    nutrition_profile_sheet: object = None,
) -> pd.DataFrame:
    out = specs_df.copy()
    sheet_text = str(sheet_name or "").strip()
    if sheet_text != LOW_LAND_NEW_MC_EFFECT_SHEET and not _is_low_land_new_profile_sheet(nutrition_profile_sheet):
        return out
    if "Element" not in out.columns:
        return out
    mask = out["Element"].astype(str).str.strip().str.lower().eq("ruminant_reduction")
    if not bool(mask.any()):
        return out
    if "Element unit" in out.columns:
        out.loc[mask, "Element unit"] = "rate"
    return out


# Config (edit here)

CONFIG = {
    "seed": 42,
    "year": 2080,
    "unit_scale": 1e-6,  # kt -> Gt
    "mode": "variable_effect",  # "mc" | "variable_effect"
    # Variable-effect sensitivity (fix one variable at set rate levels; MC others)
    "variable_effect": {
        "enabled": True,
        "samples_per_level": 2000,
        "mc_sheet_prefer": LOW_LAND_NEW_MC_EFFECT_SHEET,
        "nutrition_profile_sheet": DEFAULT_NUTRITION_PROFILE_SHEET,
        "target_kinds": {
            # rate relative to Y2020 baseline (e.g., +0.2 means +20%)
            "yield_rate": [-0.2, 0.0, 0.5, 0.9],   # [-0.2, 0.0, 0.4, 0.8, 1.0]
            "emission_factor": [-0.9, -0.5, 0.0, 0.2],
            # low_land_new interprets ruminant_reduction as an absolute
            # ruminant kcal-share cap in [0, 1].
            "ruminant_reduction": [0.0, 0.25, 0.5, 0.75, 1.0],
          # "losses_ratio": [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2],
          # "crop_soil_management_ratio": [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2],
        },
        "output_dir": "",  # empty -> <NZF_OUTPUT_DIR>/MC_Sensitivity_Variable_Effect
        "resume": False,
        "output_fig5": {
            "enabled": True,
            "write_targets": True,
        },
        "targets": [
            ("1.5D", 0.9),
            ("2D", 4.2),
            ("Current", 12.6),
        ],
        "use_linear": True,
        "future_last_only": True,
        "use_regional": False,
        "pre_macc_e0": False,
        "use_fao_modules": True,
        "domestic_supply_simulation_mode": "hard_equation",
        "supply_curtailment_enabled": False,
        "supply_curtailment_penalty": 1e10,
        # Only compute fast emissions summary; skip DS/market/cost outputs
        "fast_emis_only": False,
        "market_gap_max_rate": 0.05,
        "sampling": {
            "method": "lhs_antithetic",
            # MC_effect sampling scope controls how the normalized random U is
            # shared before mapping through each row's Min_bound/Max_bound:
            # "element": one U per Element/kind. This is the default;
            # all ruminant_reduction rows share one sampled rate.
            # "element_process_item": one U per Element + Process + Item.
            # "element_process_item_region": also split by Region_cat.
            # "element_process_item_ghg": also split by GHG.
            # "row": every MC_effect row is independent.
            "scope": "element",
            # Optional discrete snapping after continuous draw. For
            # low_land_new, ruminant_reduction is an absolute share cap with
            # 1%-step candidates. The active Min_bound/Max_bound in
            # MC_effect_low_land_new still controls the sampled range.
            "discrete_levels_enabled": True,
            # False: map raw U directly over the discrete grid inside each
            # row's Min_bound/Max_bound.
            # True: apply quantile_bounds first, then snap to nearest level.
            "discrete_levels_use_quantile_bounds": False,
            "discrete_levels_by_kind": {
                "ruminant_reduction": list(RUMINANT_REDUCTION_DISCRETE_LEVELS),
            },
            "mix_ratio": 0.6,
            "shuffle": True,
            "scramble": True,
            "quantile_bounds": (0.0, 1.0),
        },
        "precheck": {
            "enabled": False,
            "max_resample": 0,
        },
        "nutrition_soft_constraints": {
            "enabled": False,
            "max_slack_rate": 0.1,
            "slack_penalty": 1e16,
            "demand_method": None,
            "nutrition_band_epsilon": 0.1,
        },
        "land_soft_constraints": {
            "enabled": False,
            "max_over_cap_rate": 0.02,
            "slack_penalty_per_ha": 1e10,
        },
        # Non-EF sampling u mode (applies to ALL bounds, not just Y2020_XX):
        # Controls how the random position u is shared across countries/commodities
        # for non-EF variables (e.g., yield_rate, losses_ratio).
        # "shared": one u shared across all countries/commodities
        # "per_country": u differs by country
        # "per_commodity": u differs by commodity (shared across countries)
        # "per_country_commodity": u differs by country+commodity
        "mc_non_ef_mode": "shared",
        # EF sampling u mode (applies to ALL bounds):
        # Controls how u is shared for emission_factor rows (before process grouping).
        # "shared": one u shared across all countries/commodities
        # "per_country": u differs by country
        # "per_commodity": u differs by commodity (shared across countries)
        # "per_country_commodity": u differs by country+commodity
        "mc_ef_mode": "shared",
        # EF u grouping by process:
        # Controls whether Process further splits u for EF.
        # "all": ignore process (one u per country/commodity)
        # "by_process": include actual process (u differs by process)
        "ef_process_mode": "all",
        "batch": {
            "enabled": False,
            "total_batches": 100,
            "batch_index": 1,
            "assignment": "round_robin",  # 'round_robin' | 'contiguous'
            "batches_subdir": "batches",
        },
    },
    "mc": {
        "samples": 20,
        "output_dir": "",  # empty -> <NZF_OUTPUT_DIR>/MC_Sensitivity_Yield_EF
        "group": "both",  # "yield_feed" | "emission_factor" | "both"
        "targets": [
            ("1.5D", 0.9),
            ("2D", 5.0),
            ("Current", 12.6),
        ],
        "resume": False,
        "use_linear": True,
        "future_last_only": True,
        "use_regional": False,
        "pre_macc_e0": False,
        "use_fao_modules": True,
        "domestic_supply_simulation_mode": "hard_equation",
        "supply_curtailment_enabled": False,
        "supply_curtailment_penalty": 1e10,
        # Only compute fast emissions summary; skip DS/market/cost outputs
        "fast_emis_only": False,
        "market_gap_max_rate": 0.05,
        # Sampling config (affects how MC values are drawn within bounds)
        # method:
        # "uniform": independent uniform draws
        # "lhs": Latin Hypercube (good coverage with small samples)
        # "lhs_antithetic": LHS paired with (1-u); default for stable fixed-budget MC
        # "range": deterministic stratified points incl. 0/1 endpoints (small-sample range coverage)
        # "halton": low-discrepancy sequence
        # "mixed": mix of lhs + uniform for multi-dimensional diversity
        # Brief descriptions:
        # uniform: independent uniform sampling, with equal probability across each dimension's interval.
        # lhs: Latin hypercube sampling, providing more even coverage with few samples.
        # lhs_antithetic: LHS with antithetic pairs, stabilizing means/quantiles for a fixed sample count.
        # range: stratified fixed grid including endpoints, suitable for scanning ranges with few samples.
        # halton: a low-discrepancy sequence with relatively uniform coverage.
        # mixed: combines lhs and uniform to balance coverage and randomness.
        "sampling": {
            "method": "lhs_antithetic",
            # MC_effect sampling scope controls how the normalized random U is
            # shared before mapping through each row's Min_bound/Max_bound:
            # "element": one U per Element/kind. This is the default;
            # all ruminant_reduction rows share one sampled rate.
            # "element_process_item": one U per Element + Process + Item.
            # "element_process_item_region": also split by Region_cat.
            # "element_process_item_ghg": also split by GHG.
            # "row": every MC_effect row is independent.
            "scope": "element",
            # Optional discrete snapping after continuous draw. For
            # low_land_new, ruminant_reduction is an absolute share cap with
            # 1%-step candidates. The active Min_bound/Max_bound in
            # MC_effect_low_land_new still controls the sampled range.
            "discrete_levels_enabled": True,
            # False: map raw U directly over the discrete grid inside each
            # row's Min_bound/Max_bound.
            # True: apply quantile_bounds first, then snap to nearest level.
            "discrete_levels_use_quantile_bounds": False,
            "discrete_levels_by_kind": {
                "ruminant_reduction": list(RUMINANT_REDUCTION_DISCRETE_LEVELS),
            },
            "mix_ratio": 0.6,     # used by "mixed": share of LHS samples
            "shuffle": True,      # shuffle mixed samples
            "scramble": True,     # scramble halton sequence
            # Quantile bounds within [Min_bound, Max_bound]; use the full range.
            "quantile_bounds": (0.0, 1.0),
        },
        # Adaptive resampling: shrink quantile ranges for specific groups after MC precheck failure
        # Recommended defaults:
        # enabled: True (works only when precheck.enabled = True)
        # mode: "group_quantile" (shrink only selected groups)
        # target_groups: ["land"] (land-demand related drivers)
        # shrink_step: 0.02 (each failure tightens bounds by 2% on both sides)
        # min_q_low/max_q_high: 0.2/0.8 (avoid over-shrinking to a point mass)
        # Optional enhancements:
        # directional: bias bounds toward feasibility-friendly direction (e.g., higher yield)
        # region_targeting: only adjust rows that overlap failed regions (plus "All" if enabled)
        # severity: scale shrink/bias by over-cap severity (need/cap - 1)
        "adaptive_resample": {
            "enabled": True,
            "mode": "group_quantile",
            "target_groups": ["land"],
            "shrink_step": 0.02,
            "min_q_low": 0.2,
            "max_q_high": 0.8,
            # Group definitions: map group -> kinds (lowercase)
            "groups": {
                "land": ["yield_rate", "feed_intensity", "ruminant_reduction", "fertilizer_rate"],
                "ef": ["emission_factor", "crop_soil_management_ratio"],
                "other": [],
            },
            # Directional bias settings
            "directional": {
                "enabled": True,
                "bias_step": 0.02,   # shift bounds per failure (scaled by severity)
                "max_bias": 0.2,     # cap total shift to avoid extreme clipping
                # kind -> direction ("up" favors higher values, "down" favors lower values)
                "kind_direction": {
                    "yield_rate": "up",
                    "feed_intensity": "up",
                    "ruminant_reduction": "down",
                    "fertilizer_rate": "down",
                },
            },
            # Region-targeted adjustment (only rows intersecting failed regions)
            "region_targeting": {
                "enabled": True,
                "include_all": True,  # also adjust rows with Region_cat=All
            },
            # Severity scaling (based on need/cap ratio)
            "severity": {
                "enabled": True,
                "max_scale": 3.0,     # shrink/bias multiplier cap (1 + min(severity, max_scale))
            },
        },
        # MC feasibility precheck: reject samples with land_need > land_cap + max_expansion
        "precheck": {
            "enabled": False,      # only used by MC sensitivity
            "max_resample": 0,   # max redraw attempts per sample
        },
        # Soft nutrition constraints in energy units (MC only): allow small demand shortfalls/surpluses to avoid hard infeasibility.
        # When enabled, max_slack_rate caps total shortfall/surplus as a fraction of total demand in energy units.
        "nutrition_soft_constraints": {
            "enabled": False,
            "max_slack_rate": 0.1,   # 10% demand slack cap
            "slack_penalty": 1e16,    # keep large to discourage slack
            # Optional override; set to "nutrition_band" to soften nutrition demand
            "demand_method": None,
            "nutrition_band_epsilon": 0.1,
        },
        # Soft land constraints (MC only): allow small land-cap exceedances (ha) and penalize them in the objective.
        # enabled: False
        # max_over_cap_rate: 0.02 allows at most a 2% exceedance.
        # slack_penalty_per_ha: 1e10 is the penalty per excess hectare and must be sufficiently large.
        "land_soft_constraints": {
            "enabled": False,
            "max_over_cap_rate": 0.02,
            "slack_penalty_per_ha": 1e10,
        },
        # Non-EF sampling u mode (applies to ALL bounds, not just Y2020_XX):
        # Control how the sampling fraction u for non-EF variables is shared across countries/commodities.
        # "shared": use the same u for all countries/commodities, improving consistency and interpretability.
        # "per_country": sample independently for each country.
        # "per_commodity": sample independently for each commodity; all countries share that commodity's u.
        # "per_country_commodity": sample independently for each country-commodity pair.
        "mc_non_ef_mode": "shared",
        # EF sampling u mode (applies to ALL bounds):
        # Control how emission_factor u is shared across countries/commodities; ef_process_mode controls Process separately.
        # "shared": use the same u for all countries/commodities.
        # "per_country": sample independently for each country.
        # "per_commodity": sample independently for each commodity; all countries share that commodity's u.
        # "per_country_commodity": independent for each country-commodity pair.
        "mc_ef_mode": "shared",
        # EF u grouping by process:
        # Whether to subdivide u further by Process.
        # "all": ignore Process (default).
        # "by_process": sample independently for each actual Process.
        "ef_process_mode": "all",
    },
    # Element unit notes (Scenario path; MC draws direct values from bounds):
    # rate: 2080 change relative to 2020 (linear ramp to future years); yield/fertilizer/EF use 1+rate,
    # feed_intensity uses 1+rate
    # multiplier: direct multiplier (linear ramp to future years)
    # amount: absolute level (e.g., $/tCO2e), constant for future years
    # absolute: absolute value (e.g., t/ha, t/head, kgN/ha), overwrites the parameter itself
    # best_value: read the specified column from Scenario_variable_historical_range.xlsx and apply like absolute
    # Note: MC values are numeric draws; "best_value" behaves like absolute here, "profile" is not used in MC.
    # Bounds can be "Y2020_XX" (e.g., Y2020_90) to express a multiplier around the 2020 baseline.
}

GROUPS: Dict[str, Dict[str, object]] = {
    "yield_feed": {
        "label": "yield+feed_intensity",
        "kinds": {"yield_rate", "feed_intensity"},
    },
    "emission_factor": {
        "label": "emission_factor",
        "kinds": {"emission_factor", "crop_soil_management_ratio"},
    },
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _append_log(path: Path, message: str) -> None:
    try:
        if path.parent:
            _ensure_dir(path.parent)
        with path.open("a", encoding="utf-8") as f:
            f.write(message.rstrip() + "\n")
    except Exception:
        # Best-effort logging only
        pass


def _reset_output_file(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception as exc:
        raise RuntimeError(f"failed to reset derived output {path}: {exc}") from exc


def _append_rows_csv(rows: List[Dict[str, object]], out_path: Path) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    header = (not out_path.exists()) or out_path.stat().st_size == 0
    df.to_csv(out_path, mode="a", header=header, index=False, encoding="utf-8-sig")
    return int(len(df))


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _dedupe_csv_by_scenario(path: Path) -> pd.DataFrame:
    df = _read_csv_if_exists(path)
    if df.empty or "scenario_id" not in df.columns:
        return df
    df = df.drop_duplicates(subset=["scenario_id"], keep="last").reset_index(drop=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def _read_validated_run_csv(
    path: Path,
    validation: ResumeValidation,
) -> pd.DataFrame:
    """Read a current-run CSV only when every row has the exact run identity."""
    if not artifact_matches_validated_run(path, validation):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    required = {"run_id", "scenario_id"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()
    run_ids_match = (
        df["run_id"].fillna("").astype(str).str.strip().eq(validation.run_id).all()
    )
    scenario_ids_match = (
        df["scenario_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq(validation.scenario_id)
        .all()
    )
    if not bool(run_ids_match and scenario_ids_match):
        return pd.DataFrame()
    return df


def _validated_resume_artifacts(
    scenario_dir: Path,
    *,
    expected_scenario_id: str,
    expected_resume_fingerprint: str,
) -> Tuple[ResumeValidation, pd.DataFrame, pd.DataFrame]:
    """Validate status/manifest first, then open both required fast artifacts."""
    validation = validate_run_for_resume(
        scenario_dir,
        expected_scenario_id=expected_scenario_id,
        expected_resume_fingerprint=expected_resume_fingerprint,
    )
    if not validation.allowed:
        return validation, pd.DataFrame(), pd.DataFrame()
    market_path = (
        scenario_dir
        / "Diagnostics"
        / "commodity_balance_by_commodity.csv"
    )
    if not artifact_matches_validated_run(market_path, validation):
        return validation, pd.DataFrame(), pd.DataFrame()
    summary_df = _read_validated_run_csv(
        scenario_dir / "Emis" / "emissions_fast_summary.csv",
        validation,
    )
    detail_df = _read_validated_run_csv(
        scenario_dir / "Emis" / "emissions_fast_global_detail.csv",
        validation,
    )
    return validation, summary_df, detail_df


def _resume_artifacts_ready(
    scenario_dir: Path,
    validation: Optional[ResumeValidation],
    summary_df: pd.DataFrame,
    detail_df: pd.DataFrame,
) -> bool:
    market_diagnostic = (
        Path(scenario_dir)
        / "Diagnostics"
        / "commodity_balance_by_commodity.csv"
    )
    return bool(
        validation is not None
        and validation.allowed
        and not summary_df.empty
        and not detail_df.empty
        and artifact_matches_validated_run(market_diagnostic, validation)
    )


def _s51_resume_fingerprint(
    *,
    runner_mode: str,
    scenario_id: str,
    param_rows: List[Dict[str, object]],
    run_cfg: Dict[str, object],
    year: int,
) -> str:
    """Bind reuse to this exact S5.1 draw and its effective solve options."""
    return build_resume_fingerprint(
        {
            "schema": 1,
            "runner": "S5_1_1_sensitivity_mc_variable_effect",
            "runner_mode": str(runner_mode),
            "scenario_id": str(scenario_id),
            "param_rows": [dict(row) for row in param_rows],
            "model_options": {
                "year": int(year),
                "fast_emis_only": bool(run_cfg.get("fast_emis_only", False)),
                "future_last_only": bool(CFG.get("future_last_only", True)),
                "use_fao_modules": bool(CFG.get("use_fao_modules", True)),
                "use_linear": bool(CFG.get("use_linear_model", True)),
                "use_regional": bool(CFG.get("use_regional_aggregation", False)),
                "solve": bool(CFG.get("solve", True)),
                "pre_macc_e0": bool(CFG.get("premacc_e0", False)),
                "demand_method": str(CFG.get("demand_method", "") or ""),
                "nutrition_profile_sheet": str(
                    CFG.get("nutrition_profile_sheet", "") or ""
                ),
                "domestic_supply_simulation_mode": str(
                    CFG.get("domestic_supply_simulation_mode", "") or ""
                ),
                "supply_curtailment_enabled": bool(
                    CFG.get("supply_curtailment_enabled", False)
                ),
                "supply_curtailment_penalty": float(
                    CFG.get("supply_curtailment_penalty", 0.0) or 0.0
                ),
                "disable_production_cost_term": bool(
                    CFG.get("disable_production_cost_term", True)
                ),
                "production_cost_weight": float(
                    CFG.get("production_cost_weight", 1.0) or 0.0
                ),
                "bioenergy_enabled": bool(CFG.get("bioenergy_enabled", False)),
                "bioenergy_scenario": str(CFG.get("bioenergy_scenario", "") or ""),
                "mc_non_ef_mode": str(
                    run_cfg.get("mc_non_ef_mode", "shared") or "shared"
                ),
                "mc_ef_mode": str(run_cfg.get("mc_ef_mode", "shared") or "shared"),
                "ef_process_mode": str(
                    run_cfg.get("ef_process_mode", "all") or "all"
                ),
                "precheck_enabled": bool(
                    (run_cfg.get("precheck", {}) or {}).get("enabled", False)
                ),
                "nutrition_soft_constraints": dict(
                    run_cfg.get("nutrition_soft_constraints", {}) or {}
                ),
                "land_soft_constraints": dict(
                    run_cfg.get("land_soft_constraints", {}) or {}
                ),
            },
        }
    )


def _variable_effect_counts_from_outputs(samples_path: Path, status_path: Path) -> Tuple[int, int, int]:
    samples_df = _dedupe_csv_by_scenario(samples_path)
    status_df = _dedupe_csv_by_scenario(status_path)
    sample_ids = set()
    status_ids = set()
    if not samples_df.empty and "scenario_id" in samples_df.columns:
        sample_ids = {
            str(x).strip()
            for x in samples_df["scenario_id"].dropna().tolist()
            if str(x).strip()
        }
    if not status_df.empty and "scenario_id" in status_df.columns:
        status_ids = {
            str(x).strip()
            for x in status_df["scenario_id"].dropna().tolist()
            if str(x).strip()
        }
    attempted_runs = len(status_ids | sample_ids)
    valid_runs = len(sample_ids)
    invalid_runs = max(0, attempted_runs - valid_runs)
    return valid_runs, attempted_runs, invalid_runs


def _parse_targets(raw: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    items: List[Tuple[str, float]] = []
    for label, val in raw or []:
        try:
            items.append((str(label), float(val)))
        except Exception:
            continue
    return items


def _load_mc_specs_effect(
    xlsx_path: str,
    *,
    prefer_sheet: str = DEFAULT_MC_EFFECT_SHEET,
    nutrition_profile_sheet: object = None,
) -> pd.DataFrame:
    """Load MC specs from preferred sheet only; raise if missing."""
    prefer_sheet = resolve_mc_effect_sheet(
        prefer_sheet,
        nutrition_profile_sheet=nutrition_profile_sheet,
    )
    xls = pd.ExcelFile(xlsx_path)
    if prefer_sheet not in xls.sheet_names:
        raise ValueError(f"Missing MC sheet: {prefer_sheet}")
    sheet = prefer_sheet
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    return _normalize_low_land_new_mc_specs(
        df,
        sheet_name=sheet,
        nutrition_profile_sheet=nutrition_profile_sheet,
    )


def _format_level_tag(level: float) -> str:
    sign = "p" if level >= 0 else "m"
    pct = int(round(abs(level) * 100))
    return f"{sign}{pct:02d}"


def _fig5_key_label(kind: str, rate_value: float) -> Tuple[str, str, str]:
    kind_l = str(kind or "").strip().lower()
    if abs(rate_value) < 1e-9:
        if kind_l == "yield_rate":
            return "yield", "yield_current", "当前产率"
        if kind_l == "emission_factor":
            return "emission_factor", "ef_current", "当前EF"
        if kind_l == "ruminant_reduction":
            return "ruminant_reduction", "ruminant_share_0", "反刍热量占比0%"
        if kind_l == "losses_ratio":
            return "losses_ratio", "losses_current", "当前损耗"
        if kind_l == "crop_soil_management_ratio":
            return "crop_soil_management_ratio", "crop_soil_current", "当前作物土壤管理"
    pct = int(round(abs(rate_value) * 100))
    direction = "up" if rate_value > 0 else "down"
    if kind_l == "yield_rate":
        key = f"yield_{direction}_{pct}"
        label = f"产率提升{pct}%" if rate_value > 0 else f"产率降低{pct}%"
        return "yield", key, label
    if kind_l == "emission_factor":
        key = f"ef_{direction}_{pct}"
        label = f"EF升高{pct}%" if rate_value > 0 else f"EF降低{pct}%"
        return "emission_factor", key, label
    if kind_l == "ruminant_reduction":
        key = f"ruminant_share_{pct}"
        label = f"反刍热量占比{pct}%"
        return "ruminant_reduction", key, label
    if kind_l == "losses_ratio":
        key = f"losses_{direction}_{pct}"
        label = f"损耗升高{pct}%" if rate_value > 0 else f"损耗降低{pct}%"
        return "losses_ratio", key, label
    if kind_l == "crop_soil_management_ratio":
        key = f"crop_soil_{direction}_{pct}"
        label = f"作物土壤管理升高{pct}%" if rate_value > 0 else f"作物土壤管理降低{pct}%"
        return "crop_soil_management_ratio", key, label
    return "", "", ""


def _write_fig5_outputs(records: List[Dict[str, object]],
                        *,
                        year: int,
                        targets: List[Tuple[str, float]],
                        write_targets: bool) -> None:
    if not records:
        return
    rows: List[Dict[str, object]] = []
    for r in records:
        panel, key, label = _fig5_key_label(r.get("variable"), float(r.get("rate_value", 0.0)))
        if not panel:
            continue
        rows.append(
            {
                "panel": panel,
                "scenario_key": key,
                "scenario_label": label,
                "sample_id": int(r.get("sample_id", 0)),
                "year": int(year),
                "emissions_2080_gt": float(r.get("emissions_2080_gt")),
            }
        )
    if not rows:
        return
    fig5_dir = Path(get_results_base()) / "Plot" / "Fig5"
    _ensure_dir(fig5_dir)
    samples_path = fig5_dir / "samples.csv"
    pd.DataFrame(rows).to_csv(samples_path, index=False, encoding="utf-8-sig")
    if write_targets and targets:
        targets_df = pd.DataFrame([{"label": lbl, "emission_gt": val} for lbl, val in targets])
        targets_df.to_csv(fig5_dir / "targets.csv", index=False, encoding="utf-8-sig")
    print(f"[DONE] fig5 samples: {samples_path}")


def _override_param_rows(param_rows: List[Dict[str, object]], *, kind: str, rate_value: float) -> int:
    changed = 0
    for row in param_rows:
        if row.get("kind") != kind:
            continue
        row["abs_value"] = float(rate_value)
        row["element_unit"] = "rate"
        row["unit"] = "rate"
        row["min_bound"] = float(rate_value)
        row["max_bound"] = float(rate_value)
        row["min_is_y2020"] = False
        row["max_is_y2020"] = False
        row["y2020_bound"] = False
        changed += 1
    return changed


def _batch_tag(batch_index: int, batch_count: int) -> str:
    return f"batch_{int(batch_index):02d}_of_{int(batch_count):02d}"


def _resolve_batch_settings(batch_cfg: Dict[str, object]) -> Dict[str, object]:
    cfg = batch_cfg or {}
    enabled = bool(cfg.get("enabled", False))
    batch_count = int(cfg.get("total_batches", 1) or 1)
    batch_index = int(cfg.get("batch_index", 1) or 1)
    if batch_count <= 0:
        raise ValueError("variable_effect.batch.total_batches must be a positive integer.")
    if batch_index <= 0 or batch_index > batch_count:
        raise ValueError("variable_effect.batch.batch_index must be within 1..total_batches.")
    assignment = str(cfg.get("assignment", "round_robin") or "round_robin").strip().lower()
    if assignment not in {"round_robin", "contiguous"}:
        raise ValueError("variable_effect.batch.assignment must be 'round_robin' or 'contiguous'.")
    if not enabled:
        batch_count = 1
        batch_index = 1
    return {
        "enabled": enabled,
        "count": batch_count,
        "index": batch_index,
        "tag": _batch_tag(batch_index, batch_count),
        "assignment": assignment,
        "batches_subdir": str(cfg.get("batches_subdir", "batches") or "batches"),
    }


def _select_batch_item_indices(
    *,
    total_items: int,
    batch_count: int,
    batch_index: int,
    assignment: str,
) -> List[int]:
    if total_items <= 0:
        return []
    if batch_count <= 1:
        return list(range(total_items))
    if assignment == "contiguous":
        base = total_items // batch_count
        rem = total_items % batch_count
        start = (batch_index - 1) * base + min(batch_index - 1, rem)
        size = base + (1 if batch_index <= rem else 0)
        return list(range(start, start + size))
    return [idx for idx in range(total_items) if (idx % batch_count) == (batch_index - 1)]


def _resolve_variable_effect_output_paths(root_output_dir: Path, batch_state: Dict[str, object]) -> Dict[str, Path]:
    if bool(batch_state.get("enabled")):
        active_output_dir = (
            root_output_dir / str(batch_state["batches_subdir"]) / str(batch_state["tag"])
        )
        summary_dir = active_output_dir / "summary"
        samples_path = summary_dir / "samples.csv"
        status_path = summary_dir / "run_status.csv"
        meta_path = summary_dir / "run_meta.csv"
    else:
        active_output_dir = root_output_dir
        summary_dir = active_output_dir
        samples_path = active_output_dir / "samples.csv"
        status_path = active_output_dir / "run_status.csv"
        meta_path = active_output_dir / "run_meta.csv"
    return {
        "root_output_dir": root_output_dir,
        "active_output_dir": active_output_dir,
        "summary_dir": summary_dir,
        "runs_dir": active_output_dir / "runs",
        "mc_samples_dir": active_output_dir / "MC",
        "samples_path": samples_path,
        "status_path": status_path,
        "meta_path": meta_path,
    }


def _build_variable_effect_tasks(
    target_kinds: Dict[str, object],
    *,
    samples_per_level: int,
) -> List[Dict[str, object]]:
    tasks: List[Dict[str, object]] = []
    task_id = 0
    for raw_kind, levels in (target_kinds or {}).items():
        if not isinstance(levels, (list, tuple)) or not levels:
            continue
        kind = str(raw_kind).strip()
        if not kind:
            continue
        for level in levels:
            try:
                rate_val = float(level)
            except Exception:
                continue
            level_tag = _format_level_tag(rate_val)
            for sample_idx in range(int(samples_per_level)):
                task_id += 1
                sample_id = sample_idx + 1
                tasks.append(
                    {
                        "task_id": task_id,
                        "kind": kind,
                        "rate_value": rate_val,
                        "level_tag": level_tag,
                        "sample_index": sample_idx,
                        "sample_id": sample_id,
                        "scenario_id": f"VE_{kind}_{level_tag}_{sample_id:05d}",
                    }
                )
    return tasks


def _apply_batch_meta(row: Dict[str, object], batch_state: Dict[str, object]) -> Dict[str, object]:
    row["batch_index"] = int(batch_state["index"])
    row["batch_count"] = int(batch_state["count"])
    row["batch_tag"] = str(batch_state["tag"])
    return row


def _write_variable_effect_run_meta(
    path: Path,
    *,
    requested_tasks: int,
    assigned_tasks: int,
    valid_runs: int,
    attempted_runs: int,
    invalid_runs: int,
    samples_per_level: int,
    seed: int,
    year: int,
    sampling_cfg: Dict[str, object],
    q_bounds: Tuple[float, float],
    run_cfg: Dict[str, object],
    extra_meta: Optional[Dict[str, object]] = None,
) -> Path:
    success_rate = (float(valid_runs) / float(attempted_runs)) if attempted_runs > 0 else float("nan")
    row = {
        "requested_tasks": int(requested_tasks),
        "assigned_tasks": int(assigned_tasks),
        "valid_runs": int(valid_runs),
        "attempted_runs": int(attempted_runs),
        "invalid_runs": int(invalid_runs),
        "success_rate": success_rate,
        "samples_per_level": int(samples_per_level),
        "seed": int(seed),
        "year": int(year),
        "sampling_method": str(sampling_cfg.get("method", "lhs_antithetic")),
        "sampling_scope": _sampling_scope(sampling_cfg),
        "sampling_discrete_levels_enabled": bool(sampling_cfg.get("discrete_levels_enabled", True)),
        "sampling_discrete_use_quantile_bounds": bool(sampling_cfg.get("discrete_levels_use_quantile_bounds", False)),
        "sampling_mix_ratio": sampling_cfg.get("mix_ratio"),
        "sampling_shuffle": bool(sampling_cfg.get("shuffle", True)),
        "sampling_scramble": bool(sampling_cfg.get("scramble", True)),
        "sampling_q_low": float(q_bounds[0]),
        "sampling_q_high": float(q_bounds[1]),
        "use_linear": bool(run_cfg.get("use_linear", True)),
        "future_last_only": bool(run_cfg.get("future_last_only", True)),
        "use_regional": bool(run_cfg.get("use_regional", False)),
        "pre_macc_e0": bool(run_cfg.get("pre_macc_e0", False)),
        "use_fao_modules": bool(run_cfg.get("use_fao_modules", True)),
        "fast_emis_only": bool(run_cfg.get("fast_emis_only", False)),
        "market_gap_max_rate": float(run_cfg.get("market_gap_max_rate", 0.05) or 0.05),
        "mc_non_ef_mode": str(run_cfg.get("mc_non_ef_mode", "shared")),
        "mc_ef_mode": str(run_cfg.get("mc_ef_mode", "shared")),
        "ef_process_mode": str(run_cfg.get("ef_process_mode", "all")),
    }
    if extra_meta:
        row.update(extra_meta)
    pd.DataFrame([row]).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _run_variable_effect(args: Dict[str, object]) -> None:
    cfg = args.get("variable_effect", {}) or {}
    if not bool(cfg.get("enabled", True)):
        raise RuntimeError("variable_effect is disabled in CONFIG.")

    args["paths"] = DataPaths()
    scenario_cfg = ScenarioConfig()
    universe = build_universe_from_dict_v3(args["paths"].dict_v3_path, scenario_cfg)
    base_year = int(scenario_cfg.years_hist_end or 2020)

    nutrition_profile_sheet = str(cfg.get("nutrition_profile_sheet", DEFAULT_NUTRITION_PROFILE_SHEET) or DEFAULT_NUTRITION_PROFILE_SHEET)
    prefer_sheet = resolve_mc_effect_sheet(
        cfg.get("mc_sheet_prefer", DEFAULT_MC_EFFECT_SHEET),
        nutrition_profile_sheet=nutrition_profile_sheet,
    )
    specs_df = _load_mc_specs_effect(
        args["paths"].scenario_config_xlsx,
        prefer_sheet=prefer_sheet,
        nutrition_profile_sheet=nutrition_profile_sheet,
    )
    print(f"[INFO] MC sheet in use (variable_effect): {prefer_sheet}")
    if specs_df is None or specs_df.empty:
        raise RuntimeError("MC specs sheet is empty or missing.")
    specs_df = _normalize_mc_specs(specs_df)

    base_out = (
        Path(cfg.get("output_dir") or "")
        if cfg.get("output_dir")
        else Path(get_results_base()) / "MC_Sensitivity_Variable_Effect"
    )
    _ensure_dir(base_out)
    batch_state = _resolve_batch_settings(cfg.get("batch", {}) or {})
    output_paths = _resolve_variable_effect_output_paths(base_out, batch_state)
    _ensure_dir(output_paths["active_output_dir"])
    _ensure_dir(output_paths["summary_dir"])
    _ensure_dir(output_paths["mc_samples_dir"])
    _ensure_dir(output_paths["runs_dir"])

    out_path = output_paths["samples_path"]
    status_path = output_paths["status_path"]
    meta_path = output_paths["meta_path"]
    mc_samples_dir = output_paths["mc_samples_dir"]
    runs_dir = output_paths["runs_dir"]

    baselines = _load_mc_baselines(args["paths"], universe, base_year=base_year)
    shared_run_cache = build_run_baseline_cache(
        args["paths"],
        scenario_cfg,
        universe,
        future_last_only=bool(cfg.get("future_last_only", True)),
    )
    mc_mode_default = str(CFG.get("mc_y2020_non_ef_mode", "shared")).strip().lower() or "shared"
    mc_mode_non_ef, mc_mode_ef, ef_process_mode = _resolve_mc_modes(cfg)

    target_kinds = cfg.get("target_kinds", {}) or {}
    if not isinstance(target_kinds, dict) or not target_kinds:
        raise RuntimeError("variable_effect.target_kinds is empty.")

    try:
        samples_per_level = int(cfg.get("samples_per_level", 1))
    except Exception:
        samples_per_level = 1
    samples_per_level = max(1, samples_per_level)
    tasks = _build_variable_effect_tasks(target_kinds, samples_per_level=samples_per_level)
    if not tasks:
        raise RuntimeError("variable_effect.target_kinds produced no runnable tasks.")
    task_indices = _select_batch_item_indices(
        total_items=len(tasks),
        batch_count=int(batch_state["count"]) if bool(batch_state.get("enabled")) else 1,
        batch_index=int(batch_state["index"]) if bool(batch_state.get("enabled")) else 1,
        assignment=str(batch_state["assignment"]),
    )
    assigned_tasks = len(task_indices)

    unit_matrix = _sample_unit_matrix_for_specs(
        specs_df,
        samples_per_level,
        seed=int(args["seed"]),
        config=cfg.get("sampling", {}),
    )
    q_bounds = cfg.get("sampling", {}).get("quantile_bounds", (0.0, 1.0))
    try:
        q_bounds = (float(q_bounds[0]), float(q_bounds[1]))
    except Exception:
        q_bounds = (0.0, 1.0)

    precheck_cfg = cfg.get("precheck", {}) or {}
    precheck_enabled = bool(precheck_cfg.get("enabled", False))
    resume_enabled = bool(cfg.get("resume", args.get("resume", False)))
    # Summary tables are derived outputs, never resume authorities.  Rebuild
    # them from the per-scenario provenance decisions below so a newly failed
    # rerun cannot leave an older "valid" row in the final sample table.
    _reset_output_file(out_path)
    _reset_output_file(status_path)
    _reset_output_file(meta_path)

    print(f"[S5_1_1] output_dir={output_paths['active_output_dir']}")
    print(
        f"[S5_1_1] samples_per_level={samples_per_level} total_tasks={len(tasks)} "
        f"fast_emis_only={bool(cfg.get('fast_emis_only', False))}"
    )
    if bool(batch_state.get("enabled")):
        print(
            f"[S5_1_1] batch={batch_state['tag']} assignment={batch_state['assignment']} "
            f"assigned_tasks={assigned_tasks}/{len(tasks)}"
        )
    if resume_enabled:
        print(
            "[S5_1_1] resume enabled: each scenario will be checked against "
            "run_status.json, the emissions manifest, and current-run fast artifacts"
        )

    valid_runs = 0
    invalid_runs = 0
    skipped_runs = 0
    for local_idx, task_index in enumerate(task_indices, start=1):
        task = tasks[task_index]
        kind = str(task["kind"])
        rate_val = float(task["rate_value"])
        level_tag = str(task["level_tag"])
        sample_idx = int(task["sample_index"])
        sample_id = int(task["sample_id"])
        scenario_id = str(task["scenario_id"])

        unit_row = None
        param_rows = None
        effects = None
        rows_by_kind = None
        status_row = _apply_batch_meta(
            {
                "task_id": int(task["task_id"]),
                "variable": kind,
                "rate_value": rate_val,
                "level_tag": level_tag,
                "sample_id": sample_id,
                "scenario_id": scenario_id,
                "scenario_dir": str(runs_dir / scenario_id),
                "run_id": "",
                "resume_fingerprint": "",
                "status": "unknown",
                "emissions_2080_gt": np.nan,
                "message": "",
            },
            batch_state,
        )
        try:
            unit_row = unit_matrix[sample_idx] if sample_idx < len(unit_matrix) else _draw_unit_row_for_specs(
                specs_df,
                seed=int(args["seed"]) + sample_idx * 1000,
                config=cfg.get("sampling", {}),
            )
            param_rows = _draw_mc_param_rows(
                specs_df,
                unit_row=unit_row,
                quantile_bounds=q_bounds,
                group_q_bounds=None,
                group_lookup=None,
                row_q_bounds=None,
                sampling_cfg=cfg.get("sampling", {}),
            )
            changed = _override_param_rows(param_rows, kind=kind, rate_value=rate_val)
            if changed == 0:
                print(f"[WARN] No rate rows overridden for {kind} at {rate_val}.")
            scenario_resume_fingerprint = _s51_resume_fingerprint(
                runner_mode="variable_effect",
                scenario_id=scenario_id,
                param_rows=param_rows,
                run_cfg=cfg,
                year=int(args["year"]),
            )
            status_row["resume_fingerprint"] = scenario_resume_fingerprint

            effects = _build_scenario_effects(
                param_rows,
                universe,
                scenario_id=scenario_id,
                mc_y2020_mode=mc_mode_non_ef,
                mc_mode_non_ef=mc_mode_non_ef,
                mc_mode_ef=mc_mode_ef,
                ef_process_mode=ef_process_mode,
            )
            rows_by_kind = _build_mc_sample_rows(
                effects,
                universe=universe,
                baselines=baselines,
                scenario_id=scenario_id,
                sample_id=sample_id,
                attempt=1,
                mc_mode_default=mc_mode_default,
                mc_mode_non_ef=mc_mode_non_ef,
                mc_mode_ef=mc_mode_ef,
                ef_process_mode=ef_process_mode,
            )
            _write_mc_sample_xlsx(
                mc_samples_dir,
                scenario_id=scenario_id,
                sample_id=sample_id,
                attempt=1,
                rows_by_kind=rows_by_kind,
            )

            scenario_dir = runs_dir / scenario_id
            emis_dir = scenario_dir / "Emis"
            emis_path = emis_dir / "emissions_summary.xlsx"
            resume_validation: Optional[ResumeValidation] = None
            resume_summary_df = pd.DataFrame()
            resume_detail_df = pd.DataFrame()
            reuse_existing = False
            if resume_enabled:
                (
                    resume_validation,
                    resume_summary_df,
                    resume_detail_df,
                ) = _validated_resume_artifacts(
                    scenario_dir,
                    expected_scenario_id=scenario_id,
                    expected_resume_fingerprint=scenario_resume_fingerprint,
                )
                reuse_existing = _resume_artifacts_ready(
                    scenario_dir,
                    resume_validation,
                    resume_summary_df,
                    resume_detail_df,
                )
                if reuse_existing:
                    skipped_runs += 1
                    status_row["message"] = (
                        f"validated resume run_id={resume_validation.run_id}"
                    )
                    if skipped_runs == 1 or skipped_runs % 50 == 0:
                        print(
                            f"[S5_1_1] validated resume {skipped_runs}: {scenario_id} "
                            f"(task {local_idx}/{assigned_tasks})"
                        )
                else:
                    reason = (
                        resume_validation.reason
                        if not resume_validation.allowed
                        else "missing_stale_or_identity_mismatched_required_artifacts"
                    )
                    print(f"[S5_1_1] resume rejected for {scenario_id}: {reason}; rerun")

            if not reuse_existing:
                run_one_pipeline(
                    args["paths"],
                    pre_macc_e0=CFG["premacc_e0"],
                    scenario_id=scenario_id,
                    scenario_params=None,
                    scenario_effects=effects,
                    solve=CFG["solve"],
                    use_fao_modules=CFG["use_fao_modules"],
                    future_last_only=CFG["future_last_only"],
                    use_linear=CFG["use_linear_model"],
                    fast_emis_only=bool(cfg.get("fast_emis_only", False)),
                    fast_emis_year=int(args["year"]),
                    resume_fingerprint=scenario_resume_fingerprint,
                    save_root=str(runs_dir),
                    mc_precheck=precheck_enabled,
                    mc_precheck_year=int(args["year"]),
                    prebuilt_config=scenario_cfg,
                    prebuilt_universe=universe,
                    prebuilt_run_cache=shared_run_cache,
                )

                (
                    resume_validation,
                    resume_summary_df,
                    resume_detail_df,
                ) = _validated_resume_artifacts(
                    scenario_dir,
                    expected_scenario_id=scenario_id,
                    expected_resume_fingerprint=scenario_resume_fingerprint,
                )
                if not _resume_artifacts_ready(
                    scenario_dir,
                    resume_validation,
                    resume_summary_df,
                    resume_detail_df,
                ):
                    reason = (
                        resume_validation.reason
                        if not resume_validation.allowed
                        else "missing_stale_or_identity_mismatched_required_artifacts"
                    )
                    raise RuntimeError(
                        f"run completed without reusable current-generation outputs: {reason}"
                    )
            status_row["run_id"] = resume_validation.run_id

            neg_msg = _validate_nonluc_fast_emissions(
                scenario_dir,
                validation=resume_validation,
            )
            if neg_msg:
                invalid_runs += 1
                status_row["status"] = "invalid_fast_emissions"
                status_row["message"] = neg_msg
                print(f"[S5_1_1] skip {scenario_id}: {neg_msg}")
                continue

            gap_msg = _validate_market_balance_gap(
                scenario_dir,
                max_gap_rate=float(cfg.get("market_gap_max_rate", 0.05) or 0.05),
                validation=resume_validation,
            )
            if gap_msg:
                invalid_runs += 1
                status_row["status"] = "invalid_market_balance"
                status_row["message"] = gap_msg
                print(f"[S5_1_1] skip {scenario_id}: {gap_msg}")
                continue

            emis_gt = _read_global_emissions_2080_gt(
                emis_path,
                year=args["year"],
                unit_scale=args["unit_scale"],
                validation=resume_validation,
            )
            status_row["emissions_2080_gt"] = emis_gt
            if np.isfinite(emis_gt):
                _append_rows_csv(
                    [{
                        "variable": kind,
                        "rate_value": rate_val,
                        "level_tag": level_tag,
                        "sample_id": sample_id,
                        "scenario_id": scenario_id,
                        "run_id": resume_validation.run_id,
                        "resume_fingerprint": scenario_resume_fingerprint,
                        "emissions_2080_gt": emis_gt,
                    }],
                    out_path,
                )
                valid_runs += 1
                status_row["status"] = "valid"
                if valid_runs == 1 or valid_runs % 25 == 0 or local_idx == assigned_tasks:
                    print(
                        f"[S5_1_1] valid {valid_runs}/{assigned_tasks} in active run "
                        f"({scenario_id}, task {local_idx}/{assigned_tasks})"
                    )
            else:
                invalid_runs += 1
                status_row["status"] = "invalid_emissions"
                status_row["message"] = "emissions_2080_gt is NaN or non-finite"
                print(f"[S5_1_1] skip invalid output {scenario_id}: emissions_2080_gt is NaN")
        except Exception as exc:
            invalid_runs += 1
            status_row["status"] = "run_error"
            status_row["message"] = str(exc)
            print(f"[S5_1_1] skip failed task {scenario_id}: {exc}")
        finally:
            _append_rows_csv([status_row], status_path)
            unit_row = None
            param_rows = None
            effects = None
            rows_by_kind = None
            gc.collect()

    final_valid_runs, final_attempted_runs, final_invalid_runs = _variable_effect_counts_from_outputs(
        out_path,
        status_path,
    )
    _write_variable_effect_run_meta(
        meta_path,
        requested_tasks=len(tasks),
        assigned_tasks=assigned_tasks,
        valid_runs=final_valid_runs,
        attempted_runs=final_attempted_runs,
        invalid_runs=final_invalid_runs,
        samples_per_level=samples_per_level,
        seed=int(args["seed"]),
        year=int(args["year"]),
        sampling_cfg=cfg.get("sampling", {}) or {},
        q_bounds=q_bounds,
        run_cfg=cfg,
        extra_meta=_apply_batch_meta(
            {
                "resume_enabled": bool(resume_enabled),
                "resume_skipped_runs": int(skipped_runs),
                "new_valid_runs": int(valid_runs),
                "new_invalid_runs": int(invalid_runs),
            },
            batch_state,
        ),
    )
    status_df = pd.read_csv(status_path) if status_path.exists() else pd.DataFrame()
    write_sensitivity_cost_summaries(
        status_df,
        output_dir=output_paths["summary_dir"],
        run_search_root=output_paths["active_output_dir"],
    )
    print(f"[DONE] samples: {out_path}")
    print(f"[DONE] run_status: {status_path}")
    print(f"[DONE] run_meta: {meta_path}")

    fig5_cfg = cfg.get("output_fig5", {}) or {}
    if not bool(batch_state.get("enabled")) and bool(fig5_cfg.get("enabled", False)):
        targets = _parse_targets(cfg.get("targets", []))
        records = pd.read_csv(out_path).to_dict("records") if out_path.exists() else []
        _write_fig5_outputs(
            records,
            year=int(args.get("year", 2080)),
            targets=targets,
            write_targets=bool(fig5_cfg.get("write_targets", True)),
        )
    elif bool(batch_state.get("enabled")):
        print(
            "[DONE] batch outputs written. Run merge after all batches finish: "
            f"python S5_1_3_merge_variable_effect_batches.py --total-batches {batch_state['count']}"
        )

def _read_global_emissions_2080_gt(
    emis_xlsx: Path,
    *,
    year: int,
    unit_scale: float,
    validation: Optional[ResumeValidation] = None,
) -> float:
    fast_csv = emis_xlsx.parent / "emissions_fast_summary.csv"
    if validation is not None:
        df = _read_validated_run_csv(fast_csv, validation)
        if df.empty:
            return float("nan")
    elif fast_csv.exists():
        df = pd.read_csv(fast_csv)
    else:
        df = pd.DataFrame()
    if not df.empty:
        if "year" in df.columns:
            years = pd.to_numeric(df["year"], errors="coerce")
            df = df[years.eq(int(year))]
        if "total_co2eq_gt" in df.columns:
            return float(pd.to_numeric(df["total_co2eq_gt"], errors="coerce").sum())
        if "total_co2eq_kt" in df.columns:
            return float(pd.to_numeric(df["total_co2eq_kt"], errors="coerce").sum()) * unit_scale
    if validation is not None:
        # A validated run must use its identity-stamped fast artifact.  Falling
        # back to a legacy workbook could silently cross run generations.
        return float("nan")
    if not emis_xlsx.exists():
        return float("nan")
    df = pd.read_excel(emis_xlsx, sheet_name="By_Country")
    if df.empty:
        return float("nan")
    if "GHG" in df.columns:
        df = df[df["GHG"].astype(str).str.upper() == "CO2EQ"]
    if "Region_label_new" in df.columns:
        df = df[df["Region_label_new"].astype(str).str.lower() == "global"]
    elif "M49_Country_Code" in df.columns:
        df = df[df["M49_Country_Code"].astype(str).str.contains("000")]
    ycol = f"Y{year}"
    if ycol not in df.columns:
        return float("nan")
    val = pd.to_numeric(df[ycol], errors="coerce").sum()
    return float(val) * unit_scale


def _validate_nonluc_fast_emissions(
    run_dir: Path,
    *,
    tol_kt: float = 1e-9,
    validation: Optional[ResumeValidation] = None,
) -> Optional[str]:
    """Return an error message if fast GCE/GLE details look physically invalid."""
    detail_path = Path(run_dir) / "Emis" / "emissions_fast_global_detail.csv"
    if validation is not None:
        df = _read_validated_run_csv(detail_path, validation)
        if df.empty:
            return (
                "fast emission detail is missing, stale, empty, or has a "
                "run_id/scenario_id mismatch"
            )
    else:
        if not detail_path.exists():
            return None
        try:
            df = pd.read_csv(detail_path)
        except Exception as exc:
            return f"failed to read fast emission detail: {exc}"
    required = {"source_module", "Process", "Item", "GHG", "emissions_kt", "co2eq_kt"}
    missing = required.difference(df.columns)
    if missing:
        return f"fast emission detail missing columns: {sorted(missing)}"
    work = df.copy()
    work["source_module"] = work["source_module"].astype(str).str.upper().str.strip()
    work["emissions_kt"] = pd.to_numeric(work["emissions_kt"], errors="coerce")
    work["co2eq_kt"] = pd.to_numeric(work["co2eq_kt"], errors="coerce")
    if "row_type" in work.columns:
        work = work[work["row_type"].astype(str).str.strip().ne("co2eq_summary")].copy()
    mask = (
        work["source_module"].isin({"GCE", "GLE"})
        & (
            (work["emissions_kt"] < -float(tol_kt))
            | (work["co2eq_kt"] < -float(tol_kt))
        )
    )
    if mask.any():
        neg = work.loc[mask].copy()
        total_gt = float(neg["co2eq_kt"].sum()) / 1e6
        top = neg.sort_values("co2eq_kt").head(5)
        top_txt = "; ".join(
            f"{r.source_module}/{r.Process}/{r.Item}/{r.GHG}={float(r.co2eq_kt) / 1e6:.3g}Gt"
            for r in top.itertuples(index=False)
        )
        return (
            f"negative non-LUC fast emissions detected: rows={len(neg)}, "
            f"co2eq={total_gt:.6g}Gt; top={top_txt}"
        )
    # Regression guard for the old EF-bound bug: the same erroneous absolute EF
    # could be applied to CH4 and N2O, producing identical large gas masses.
    nonluc = work.loc[work["source_module"].isin({"GCE", "GLE"})].copy()
    nonluc["GHG"] = nonluc["GHG"].astype(str).str.upper().str.strip()
    gas = nonluc.loc[nonluc["GHG"].isin({"CH4", "N2O"})]
    if not gas.empty:
        pivot = gas.pivot_table(
            index=["source_module", "Process", "Item"],
            columns="GHG",
            values="emissions_kt",
            aggfunc="sum",
        )
        if {"CH4", "N2O"}.issubset(set(pivot.columns)):
            pairs = pivot[["CH4", "N2O"]].dropna().copy()
            min_abs_kt = max(1.0, float(tol_kt))
            denom = pairs.abs().max(axis=1).replace(0.0, np.nan)
            rel_diff = (pairs["CH4"] - pairs["N2O"]).abs() / denom
            suspicious = pairs[
                (pairs["CH4"].abs() > min_abs_kt)
                & (pairs["N2O"].abs() > min_abs_kt)
                & (rel_diff <= 1e-9)
            ]
            if not suspicious.empty:
                top = suspicious.head(5).reset_index()
                top_txt = "; ".join(
                    f"{r.source_module}/{r.Process}/{r.Item}:CH4=N2O={float(r.CH4):.6g}kt"
                    for r in top.itertuples(index=False)
                )
                return (
                    f"suspicious identical CH4/N2O non-LUC fast emissions detected: "
                    f"pairs={len(suspicious)}; top={top_txt}"
                )
    return None


def _validate_market_balance_gap(
    scenario_dir: Path,
    *,
    max_gap_rate: float = 0.05,
    validation: Optional[ResumeValidation] = None,
) -> Optional[str]:
    """Fail closed unless solver-postsolve market diagnostics are valid."""
    if validation is not None:
        diagnostic_path = (
            Path(scenario_dir)
            / "Diagnostics"
            / "commodity_balance_by_commodity.csv"
        )
        if not artifact_matches_validated_run(diagnostic_path, validation):
            return "market-balance diagnostic is missing, empty, or stale for this run"
    return _validate_solver_market_balance_gap(
        scenario_dir,
        max_gap_rate=max_gap_rate,
    )


def _mc_element_to_kind(elem: str) -> Optional[str]:
    elem_l = str(elem or "").strip().lower()
    if not elem_l:
        return None
    if "land" in elem_l and ("carbon" in elem_l or "co2" in elem_l) and "price" in elem_l:
        return "land_carbon_price"
    if "ruminant" in elem_l or "ruminate" in elem_l or "intake" in elem_l:
        return "ruminant_reduction"
    if "feed" in elem_l and ("intensity" in elem_l or "eff" in elem_l):
        return "feed_intensity"
    if "fertilizer" in elem_l or "fertlizer" in elem_l:
        return "fertilizer_rate"
    if "manure" in elem_l and ("ratio" in elem_l or "management" in elem_l):
        return "manure_management_ratio"
    if "yield" in elem_l and "feed" not in elem_l:
        return "yield_rate"
    if "loss" in elem_l or "waste" in elem_l:
        return "losses_ratio"
    if "crop_soil_management" in elem_l or ("soil" in elem_l and "management" in elem_l and "ratio" in elem_l):
        return "crop_soil_management_ratio"
    if "ef" in elem_l or "emission" in elem_l:
        return "emission_factor"
    return None

def _normalize_mc_specs(specs_df: pd.DataFrame, *, aggregate_non_ef: bool = False) -> pd.DataFrame:
    """Normalize MC specs. Optionally aggregate non-EF by Element+Unit+Item (default: keep original rows)."""
    df = specs_df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # Ensure required columns exist
    for col in ("Element", "Element unit", "Process", "Item", "GHG", "Min_bound", "Max_bound", "Region_cat"):
        if col not in df.columns:
            df[col] = np.nan
    df["__kind"] = df["Element"].apply(_mc_element_to_kind)
    # Parse Min/Max bounds; support Y2020_XX tokens (relative to 2020 baseline)
    min_vals: List[Optional[float]] = []
    max_vals: List[Optional[float]] = []
    min_is_y2020: List[bool] = []
    max_is_y2020: List[bool] = []
    for r in df.itertuples(index=False):
        lo, lo_y = _parse_mc_bound_value(getattr(r, "Min_bound", None))
        hi, hi_y = _parse_mc_bound_value(getattr(r, "Max_bound", None))
        min_vals.append(lo)
        max_vals.append(hi)
        min_is_y2020.append(bool(lo_y))
        max_is_y2020.append(bool(hi_y))
    df["Min_bound"] = min_vals
    df["Max_bound"] = max_vals
    df["Min_is_y2020"] = min_is_y2020
    df["Max_is_y2020"] = max_is_y2020
    df["y2020_bound"] = df["Min_is_y2020"] | df["Max_is_y2020"]
    df = df[df["__kind"].notna()].copy()
    # Normalize selectors
    df["Item"] = df["Item"].apply(lambda v: _norm_field(v, "All"))
    df["Process"] = df["Process"].apply(lambda v: _norm_field(v, "All"))
    df["GHG"] = df["GHG"].apply(lambda v: _norm_field(v, "All"))
    df["Region_cat"] = df["Region_cat"].apply(lambda v: _norm_field(v, "All"))
    df["Element unit"] = df["Element unit"].apply(lambda v: _norm_field(v, ""))
    # Canonicalize Element unit; must be explicitly provided in MC/MC_effect.
    def _normalize_mc_unit_strict(raw: object) -> str:
        raw_s = str(raw or "").strip()
        if not raw_s:
            raise ValueError("Empty Element unit in MC specs.")
        unit = _normalize_mc_unit(raw_s, "__invalid__")
        if unit == "__invalid__":
            raise ValueError(f"Unsupported Element unit: {raw_s}")
        return unit
    df["Element unit"] = df["Element unit"].apply(_normalize_mc_unit_strict)

    if aggregate_non_ef:
        # Non-EF: collapse Process/GHG/Region into All, aggregate bounds by Element+Unit+Item
        non_ef = df[df["__kind"] != "emission_factor"].copy()
        if not non_ef.empty:
            non_ef["Process"] = "All"
            non_ef["GHG"] = "All"
            non_ef["Region_cat"] = "All"
            non_ef = non_ef.groupby(
                ["Element", "Element unit", "Item", "__kind"],
                as_index=False
            ).agg(
                {
                    "Min_bound": "min",
                    "Max_bound": "max",
                    "y2020_bound": "max",
                    "Min_is_y2020": "max",
                    "Max_is_y2020": "max",
                    "Process": "first",
                    "GHG": "first",
                    "Region_cat": "first",
                }
            )
        ef = df[df["__kind"] == "emission_factor"].copy()
        return pd.concat([non_ef, ef], ignore_index=True)
    # Default: keep original rows (no aggregation)
    return df

def _prime_list(n: int) -> List[int]:
    primes: List[int] = []
    candidate = 2
    while len(primes) < n:
        is_prime = True
        for p in primes:
            if p * p > candidate:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes


def _van_der_corput(n: int, base: int) -> np.ndarray:
    seq = np.zeros(n, dtype=float)
    for i in range(n):
        x = 0.0
        denom = 1.0
        idx = i + 1
        while idx > 0:
            idx, remainder = divmod(idx, base)
            denom *= base
            x += remainder / denom
        seq[i] = x
    return seq


def _halton_sequence(n_samples: int, n_dim: int, *, scramble: bool, seed: int) -> np.ndarray:
    primes = _prime_list(n_dim)
    mat = np.zeros((n_samples, n_dim), dtype=float)
    for j, base in enumerate(primes):
        mat[:, j] = _van_der_corput(n_samples, base)
    if scramble:
        rng = np.random.default_rng(seed)
        # random shift per dimension (Cranley-Patterson rotation)
        shifts = rng.random(n_dim)
        mat = (mat + shifts) % 1.0
    return mat


def _lhs(n_samples: int, n_dim: int, rng: np.random.Generator) -> np.ndarray:
    if n_samples <= 1:
        return rng.random((n_samples, n_dim))
    mat = np.zeros((n_samples, n_dim), dtype=float)
    for j in range(n_dim):
        perm = rng.permutation(n_samples)
        mat[:, j] = (perm + rng.random(n_samples)) / float(n_samples)
    return mat


def _lhs_centered(n_samples: int, n_dim: int, rng: np.random.Generator) -> np.ndarray:
    if n_samples <= 1:
        return np.full((n_samples, n_dim), 0.5, dtype=float)
    centers = (np.arange(n_samples, dtype=float) + 0.5) / float(n_samples)
    mat = np.zeros((n_samples, n_dim), dtype=float)
    for j in range(n_dim):
        mat[:, j] = centers[rng.permutation(n_samples)]
    return mat


def _range_stratified(n_samples: int, n_dim: int, rng: np.random.Generator) -> np.ndarray:
    if n_samples <= 1:
        return np.full((n_samples, n_dim), 0.5, dtype=float)
    grid = np.linspace(0.0, 1.0, n_samples)
    mat = np.zeros((n_samples, n_dim), dtype=float)
    for j in range(n_dim):
        mat[:, j] = grid[rng.permutation(n_samples)]
    return mat


def _antithetic_expand(
    base: np.ndarray,
    *,
    n_samples: int,
    rng: np.random.Generator,
    shuffle: bool,
) -> np.ndarray:
    if n_samples <= 0:
        return np.empty((0, base.shape[1] if base.ndim == 2 else 0), dtype=float)
    if base.size == 0:
        return np.empty((n_samples, 0), dtype=float)
    mat = np.vstack([base, 1.0 - base])
    if len(mat) > n_samples:
        mat = mat[:n_samples]
    if shuffle and len(mat) > 1:
        rng.shuffle(mat, axis=0)
    return mat


def _sample_unit_matrix(n_samples: int, n_dim: int, *, seed: int, config: Dict[str, object]) -> np.ndarray:
    """
    Draw an n_samples x n_dim matrix on [0, 1].

    Supported methods:
      - "uniform": independent pseudo-random sampling.
      - "lhs": Latin Hypercube Sampling; usually the best default for many variables.
      - "lhs_centered": deterministic midpoint LHS; more reproducible, less jitter.
      - "uniform_antithetic": uniform draws paired with (1 - u) for variance reduction.
      - "lhs_antithetic": LHS base draws paired with (1 - u); strong default when
        sample budget is fixed and global moments / quantiles should be more stable.
      - "range": endpoint-to-endpoint stratified coverage; mainly for stress tests.
      - "halton": low-discrepancy sequence; useful in lower/medium dimensions.
      - "mixed": blend of LHS and uniform draws; robust compromise when you want
        both per-dimension coverage and some fully random jitter.
    """
    method = str(config.get("method", "lhs_antithetic")).lower()
    rng = np.random.default_rng(seed)
    if method == "lhs":
        return _lhs(n_samples, n_dim, rng)
    if method == "lhs_centered":
        return _lhs_centered(n_samples, n_dim, rng)
    if method == "uniform_antithetic":
        base_n = max(1, (n_samples + 1) // 2)
        base = rng.random((base_n, n_dim))
        return _antithetic_expand(
            base,
            n_samples=n_samples,
            rng=rng,
            shuffle=bool(config.get("shuffle", True)),
        )
    if method == "lhs_antithetic":
        base_n = max(1, (n_samples + 1) // 2)
        base = _lhs(base_n, n_dim, rng)
        return _antithetic_expand(
            base,
            n_samples=n_samples,
            rng=rng,
            shuffle=bool(config.get("shuffle", True)),
        )
    if method == "range":
        return _range_stratified(n_samples, n_dim, rng)
    if method == "halton":
        return _halton_sequence(n_samples, n_dim, scramble=bool(config.get("scramble", True)), seed=seed)
    if method == "mixed":
        mix_ratio = float(config.get("mix_ratio", 0.6))
        n_lhs = max(1, int(round(n_samples * mix_ratio)))
        n_uni = max(0, n_samples - n_lhs)
        u_lhs = _lhs(n_lhs, n_dim, rng)
        u_uni = rng.random((n_uni, n_dim)) if n_uni > 0 else np.empty((0, n_dim))
        mat = np.vstack([u_lhs, u_uni])
        if bool(config.get("shuffle", True)) and len(mat) > 1:
            rng.shuffle(mat, axis=0)
        return mat
    # default: uniform
    return rng.random((n_samples, n_dim))

def _rescale_unit_to_quantile(u: float, q_low: float, q_high: float) -> float:
    try:
        ql = float(q_low)
        qh = float(q_high)
    except Exception:
        return float(u)
    ql = max(0.0, min(1.0, ql))
    qh = max(0.0, min(1.0, qh))
    if qh < ql:
        ql, qh = qh, ql
    return ql + (qh - ql) * float(u)


def _draw_unit_row(n_dim: int, *, seed: int, config: Dict[str, object]) -> np.ndarray:
    mat = _sample_unit_matrix(1, n_dim, seed=seed, config=config)
    return mat[0] if len(mat) else np.zeros((n_dim,), dtype=float)


def _sampling_scope(config: Dict[str, object]) -> str:
    config = config or {}
    scope = str(config.get("scope", config.get("sampling_scope", "element")) or "element")
    scope = scope.strip().lower().replace("-", "_")
    aliases = {
        "by_element": "element",
        "kind": "element",
        "by_kind": "element",
        "element_item_process": "element_process_item",
        "kind_process_item": "element_process_item",
        "element_process_item_region": "element_process_item_region",
        "element_process_item_ghg": "element_process_item_ghg",
        "independent": "row",
        "per_row": "row",
    }
    scope = aliases.get(scope, scope)
    allowed = {
        "element",
        "element_process_item",
        "element_process_item_region",
        "element_process_item_ghg",
        "row",
    }
    return scope if scope in allowed else "element"


def _sampling_group_keys(specs_df: pd.DataFrame, *, config: Dict[str, object]) -> List[Tuple[object, ...]]:
    """
    Build MC_effect sampling groups.

    The default "element" scope intentionally shares one U draw across every
    row with the same Element/kind. For example, all ruminant_reduction rows
    use the same sampled rate, while each row still maps that shared U through
    its own bounds.
    """
    df = specs_df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    config = config or {}
    scope = _sampling_scope(config)
    keys: List[Tuple[object, ...]] = []
    for pos, (_, row) in enumerate(df.iterrows()):
        kind = _mc_element_to_kind(row.get("Element", ""))
        kind_key = kind or str(row.get("Element", "") or "").strip().lower() or "unknown"
        if scope == "row":
            keys.append(("row", int(pos)))
            continue
        item = _norm_field(row.get("Item", "All"))
        process = _norm_field(row.get("Process", "All"))
        region = _norm_field(row.get("Region_cat", "All"))
        ghg = _norm_field(row.get("GHG", "All"))
        if scope == "element_process_item":
            keys.append((kind_key, process, item))
        elif scope == "element_process_item_region":
            keys.append((kind_key, process, item, region))
        elif scope == "element_process_item_ghg":
            keys.append((kind_key, process, item, ghg))
        else:
            keys.append((kind_key,))
    return keys


def _sample_unit_matrix_for_specs(specs_df: pd.DataFrame,
                                  n_samples: int,
                                  *,
                                  seed: int,
                                  config: Dict[str, object]) -> np.ndarray:
    config = config or {}
    keys = _sampling_group_keys(specs_df, config=config)
    if not keys:
        return np.empty((max(0, int(n_samples)), 0), dtype=float)
    unique_keys = list(dict.fromkeys(keys))
    grouped = _sample_unit_matrix(
        int(n_samples),
        len(unique_keys),
        seed=seed,
        config=config,
    )
    key_index = {key: i for i, key in enumerate(unique_keys)}
    out = np.zeros((grouped.shape[0], len(keys)), dtype=float)
    for col_idx, key in enumerate(keys):
        out[:, col_idx] = grouped[:, key_index[key]]
    return out


def _draw_unit_row_for_specs(specs_df: pd.DataFrame,
                             *,
                             seed: int,
                             config: Dict[str, object]) -> np.ndarray:
    mat = _sample_unit_matrix_for_specs(specs_df, 1, seed=seed, config=config)
    return mat[0] if len(mat) else np.zeros((len(specs_df),), dtype=float)


def _normalize_discrete_levels(raw_levels: object) -> List[float]:
    levels: List[float] = []
    if raw_levels is None:
        return levels
    if isinstance(raw_levels, str):
        raw_iter = [v.strip() for v in raw_levels.split(",") if v.strip()]
    else:
        try:
            raw_iter = list(raw_levels)  # type: ignore[arg-type]
        except Exception:
            raw_iter = []
    for raw in raw_iter:
        try:
            val = float(raw)
        except Exception:
            continue
        if np.isfinite(val):
            val = round(val, 10)
            if abs(val) < 1e-12:
                val = 0.0
            levels.append(float(val))
    return sorted(set(levels))


def _discrete_levels_by_kind(config: Optional[Dict[str, object]]) -> Dict[str, List[float]]:
    cfg = config or {}
    if not bool(cfg.get("discrete_levels_enabled", True)):
        return {}
    raw = cfg.get("discrete_levels_by_kind", DEFAULT_DISCRETE_LEVELS_BY_KIND)
    if raw is None:
        raw = DEFAULT_DISCRETE_LEVELS_BY_KIND
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, List[float]] = {}
    for raw_kind, raw_levels in raw.items():
        kind = str(raw_kind or "").strip().lower()
        if not kind:
            continue
        levels = _normalize_discrete_levels(raw_levels)
        if levels:
            out[kind] = levels
    return out


def _snap_draw_to_discrete_level(kind: str,
                                 value: float,
                                 lo: float,
                                 hi: float,
                                 *,
                                 u_raw: Optional[float] = None,
                                 config: Optional[Dict[str, object]] = None,
                                 levels_by_kind: Optional[Dict[str, List[float]]] = None) -> Tuple[float, bool]:
    levels_lookup = levels_by_kind if levels_by_kind is not None else _discrete_levels_by_kind(config)
    levels = levels_lookup.get(str(kind or "").strip().lower(), [])
    if not levels:
        return float(value), False
    low = min(float(lo), float(hi))
    high = max(float(lo), float(hi))
    tol = 1e-12
    in_range = [v for v in levels if (low - tol) <= v <= (high + tol)]
    if not in_range:
        return float(value), False
    if not bool((config or {}).get("discrete_levels_use_quantile_bounds", False)) and u_raw is not None:
        u_clamped = max(0.0, min(float(u_raw), np.nextafter(1.0, 0.0)))
        pos = int(np.floor(u_clamped * len(in_range)))
        pos = max(0, min(pos, len(in_range) - 1))
        snapped = in_range[pos]
        if abs(snapped) < 1e-12:
            snapped = 0.0
        return float(snapped), True
    snapped = min(in_range, key=lambda v: (abs(v - float(value)), v))
    if abs(snapped) < 1e-12:
        snapped = 0.0
    return float(snapped), True


def _draw_mc_param_rows(specs_df: pd.DataFrame,
                        *,
                        unit_row: np.ndarray,
                        quantile_bounds: Tuple[float, float],
                        group_q_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                        group_lookup: Optional[Dict[str, str]] = None,
                        row_q_bounds: Optional[Dict[int, Tuple[float, float]]] = None,
                        sampling_cfg: Optional[Dict[str, object]] = None) -> List[Dict[str, object]]:
    """Draw one MC sample row: direct values with Element unit semantics."""
    df = specs_df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    element_units = df["Element unit"].tolist() if "Element unit" in df.columns else []
    discrete_levels_cfg = _discrete_levels_by_kind(sampling_cfg)
    rows: List[Dict[str, object]] = []
    for idx, r in enumerate(df.itertuples(index=False)):
        elem_raw = getattr(r, "Element", "")
        kind = _mc_element_to_kind(elem_raw)
        if not kind:
            continue
        q_low, q_high = quantile_bounds
        if row_q_bounds and idx in row_q_bounds:
            q_low, q_high = row_q_bounds[idx]
        elif group_q_bounds and group_lookup:
            grp = group_lookup.get(kind)
            if grp in group_q_bounds:
                q_low, q_high = group_q_bounds[grp]
        lo = getattr(r, "Min_bound", 0.0)
        hi = getattr(r, "Max_bound", 0.0)
        lo_y2020 = bool(getattr(r, "Min_is_y2020", False))
        hi_y2020 = bool(getattr(r, "Max_is_y2020", False))
        try:
            lo = float(lo)
            hi = float(hi)
        except Exception:
            lo = None
            hi = None
        if lo is None or hi is None:
            continue
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue
        mixed_bounds = (lo_y2020 != hi_y2020)
        if not mixed_bounds and hi < lo:
            lo, hi = hi, lo
        u_raw = unit_row[idx] if idx < len(unit_row) else 0.5
        u = u_raw
        u = _rescale_unit_to_quantile(u, q_low, q_high)
        if mixed_bounds:
            # Mixed bound types; defer absolute conversion to baseline-aware stage.
            draw = float(u)
        else:
            draw = lo + (hi - lo) * float(u)
        continuous_draw = float(draw)
        discrete_applied = False
        if not mixed_bounds:
            draw, discrete_applied = _snap_draw_to_discrete_level(
                kind,
                float(draw),
                lo,
                hi,
                u_raw=float(u_raw),
                config=sampling_cfg,
                levels_by_kind=discrete_levels_cfg,
            )
            if discrete_applied and hi != lo:
                u = max(0.0, min(1.0, (float(draw) - lo) / (hi - lo)))

        item = _norm_field(getattr(r, "Item", "All"))
        process = _norm_field(getattr(r, "Process", "All"))
        ghg = _norm_field(getattr(r, "GHG", "All"))
        region = _norm_field(getattr(r, "Region_cat", "All"))
        element_unit = element_units[idx] if idx < len(element_units) else ""
        element_unit = _norm_field(element_unit, "")
        y2020_bound = bool(getattr(r, "y2020_bound", False) or lo_y2020 or hi_y2020)
        unit = element_unit

        rows.append(
            {
                "kind": kind,
                "item": item,
                "process": process,
                "ghg": ghg,
                "region": region,
                "abs_value": float(draw),
                "continuous_draw": float(continuous_draw),
                "discrete_level_applied": bool(discrete_applied),
                "element_unit": element_unit,
                "unit": unit,
                "y2020_bound": y2020_bound,
                "min_is_y2020": lo_y2020,
                "max_is_y2020": hi_y2020,
                "mc_u": float(u),
                "q_low": float(q_low),
                "q_high": float(q_high),
                "min_bound": float(lo),
                "max_bound": float(hi),
            }
        )
    return rows


def _build_scenario_effects(param_rows: List[Dict[str, object]],
                            universe,
                            *,
                            scenario_id: str,
                            mc_y2020_mode: Optional[str] = None,
                            mc_mode_non_ef: Optional[str] = None,
                            mc_mode_ef: Optional[str] = None,
                            ef_process_mode: str = "all") -> List[ScenarioEffect]:
    effects: List[ScenarioEffect] = []
    mode_non_ef = str(mc_mode_non_ef or mc_y2020_mode or "shared").strip().lower() or "shared"
    mode_ef = str(mc_mode_ef or mode_non_ef or "shared").strip().lower() or "shared"
    ef_process_mode_l = str(ef_process_mode or "all").strip().lower() or "all"
    for row in param_rows:
        kind = row["kind"]
        unit = row.get("unit")
        if not unit:
            if kind in ("land_carbon_price", "land_co2_price"):
                unit = "amount"
            elif kind in ("ruminant_reduction", "losses_ratio", "crop_soil_management_ratio"):
                unit = "rate"
            else:
                unit = "absolute"
        value = float(row["abs_value"])
        item = row["item"]
        process = row["process"]
        region = row.get("region") or "All"
        eff = ScenarioEffect(scenario_id, kind, unit, value, region, item, process)
        eff.countries = _select_countries(universe, region)
        eff.commodities = _select_commodities(universe, item)
        eff.processes = _select_processes(universe, process)
        if kind == "emission_factor":
            eff.ghg_sel = row.get("ghg") or "All"
        eff.element_unit = row.get("element_unit")
        lo = row.get("min_bound", None)
        hi = row.get("max_bound", None)
        if lo is not None and hi is not None:
            mode_val = mode_ef if kind == "emission_factor" else mode_non_ef
            bounds = {
                "lo": float(lo),
                "hi": float(hi),
                "lo_is_y2020": bool(row.get("min_is_y2020", False)),
                "hi_is_y2020": bool(row.get("max_is_y2020", False)),
                "unit": str(unit or ""),
                "u": float(row.get("mc_u", 0.5)),
                "u_is_rescaled": True,
                "pre_sampled": True,
                "q_low": row.get("q_low", None),
                "q_high": row.get("q_high", None),
                "mode": mode_val,
            }
            if kind == "emission_factor":
                bounds["ef_process_mode"] = ef_process_mode_l
            eff.mc_bounds_raw = bounds
            # Only Y2020-relative bounds need baseline-aware interpolation in
            # S3/S4. Normal MC_effect rate/multiplier draws are already final
            # in eff.value_2080 and must go through the standard multiplier
            # path; otherwise EF rate bounds such as [-0.9, 0.8] are
            # incorrectly treated as absolute EF values.
            if row.get("y2020_bound"):
                eff.mc_bounds = dict(bounds)
        effects.append(eff)
    return effects


def _build_param_map(param_rows: List[Dict[str, object]],
                     group_key: str) -> Dict[str, float]:
    kinds = GROUPS[group_key]["kinds"]
    params: Dict[str, float] = {}
    for row in param_rows:
        kind = row.get("kind")
        if kind not in kinds:
            continue
        value = float(row.get("abs_value"))
        if kind == "emission_factor":
            key = f"{kind}|{row.get('item')}|{row.get('process')}|{row.get('ghg')}"
        else:
            key = f"{kind}|{row.get('item')}"
        params[key] = value
    return params


def _build_group_lookup(groups_cfg: Dict[str, List[str]]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for grp, kinds in (groups_cfg or {}).items():
        for k in kinds or []:
            lookup[str(k).strip().lower()] = str(grp)
    return lookup


def _init_group_q_bounds(base_bounds: Tuple[float, float],
                         groups_cfg: Dict[str, List[str]],
                         target_groups: List[str]) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    for grp in (groups_cfg or {}).keys():
        out[str(grp)] = base_bounds
    for grp in (target_groups or []):
        out.setdefault(str(grp), base_bounds)
    return out


def _shrink_group_bounds(group_q_bounds: Dict[str, Tuple[float, float]],
                         *,
                         target_groups: List[str],
                         shrink_step: float,
                         min_q_low: float,
                         max_q_high: float,
                         scale: float = 1.0) -> None:
    if not group_q_bounds:
        return
    try:
        step = float(shrink_step)
    except Exception:
        step = 0.0
    if step <= 0:
        return
    try:
        scale_val = float(scale)
    except Exception:
        scale_val = 1.0
    if scale_val <= 0:
        scale_val = 1.0
    step = step * scale_val
    for grp in (target_groups or []):
        if grp not in group_q_bounds:
            continue
        ql, qh = group_q_bounds[grp]
        ql_new = min(max(ql + step, min_q_low), max_q_high)
        qh_new = max(min(qh - step, max_q_high), min_q_low)
        if ql_new > qh_new:
            mid = (ql_new + qh_new) * 0.5
            ql_new = mid
            qh_new = mid
        group_q_bounds[grp] = (ql_new, qh_new)


def _build_region_members(universe) -> Dict[str, List[str]]:
    members: Dict[str, List[str]] = {}
    for c, r in (universe.region_aggMC_by_country or {}).items():
        if c and r:
            members.setdefault(str(r), []).append(str(c))
    return members


_KIND_ALIASES = {
    "yield_multiplier": "yield_rate",
    "yield_improvement": "yield_rate",
    "feed_intensity": "feed_intensity",
    "feed_intensity_rate": "feed_intensity",
    "feed_intensity_improve": "feed_intensity",
    "feed_efficiency": "feed_intensity",
    "feed_efficiency_rate": "feed_intensity",
    "feed_efficiency_improve": "feed_intensity",
    "fertlizer_rate": "fertilizer_rate",
    "fertilizer_efficiency": "fertilizer_rate",
    "manure_ratio": "manure_management_ratio",
    "mm_ratio": "manure_management_ratio",
    "ruminant_intake_ratio": "ruminant_reduction",
    "ruminant_intake_decreasing_ratio": "ruminant_reduction",
    "ruminant_reduction": "ruminant_reduction",
    "crop_soil_ratio": "crop_soil_management_ratio",
    "crop_soil_management_ratio": "crop_soil_management_ratio",
    "waste_rate": "losses_ratio",
    "waste_reduction": "losses_ratio",
}


def _normalize_kind(kind: str) -> str:
    key = str(kind or "").strip().lower()
    return _KIND_ALIASES.get(key, key)


def _norm_m49_key(val: object) -> str:
    s = str(val).strip() if val is not None else ''
    if not s:
        return ''
    if s.startswith("'"):
        return s
    if s.isdigit():
        return f"'{s.zfill(3)}"
    return s


def _normalize_m49(val: object) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip()
    if s.startswith("'"):
        s = s[1:]
    s = s.strip()
    if not s:
        return ""
    if s.count(".") == 1:
        left, right = s.split(".", 1)
        if left.isdigit() and right.strip("0") == "":
            s = left
    if s.isdigit():
        return f"'{s.zfill(3)}"
    return f"'{s}"


def _build_base_map(df: pd.DataFrame, value_col: str, *, base_year: int, universe) -> Dict[Tuple[str, str], float]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {}
    work = df.copy()
    if "year" in work.columns:
        work["year"] = pd.to_numeric(work["year"], errors="coerce")
        work = work[work["year"] == base_year]
    if work.empty or value_col not in work.columns:
        return {}
    if "country" not in work.columns or work["country"].isna().all():
        if "M49_Country_Code" in work.columns:
            work["M49_Country_Code"] = work["M49_Country_Code"].apply(_normalize_m49)
            work["country"] = work["M49_Country_Code"].map(universe.country_by_m49)
    work = work.dropna(subset=["country", "commodity", value_col])
    if work.empty:
        return {}
    grouped = work.groupby(["country", "commodity"], as_index=False)[value_col].mean()
    out: Dict[Tuple[str, str], float] = {}
    for r in grouped.itertuples(index=False):
        try:
            out[(str(r.country), str(r.commodity))] = float(getattr(r, value_col))
        except Exception:
            continue
    return out


def _lookup_with_region(base_map: Dict[Tuple[str, str], float],
                        country: str,
                        commodity: str,
                        region_members: Dict[str, List[str]]) -> Optional[float]:
    val = base_map.get((country, commodity))
    if val is not None and np.isfinite(val):
        return float(val)
    members = region_members.get(country)
    if members:
        vals = [base_map.get((c, commodity)) for c in members]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        if vals:
            return float(np.mean(vals))
    return None


def _lookup_yield_base_value_and_unit(
    country: str,
    commodity: str,
    baselines: Dict[str, object],
    region_members: Dict[str, List[str]],
) -> Tuple[Optional[float], Optional[str]]:
    crop_val = _lookup_with_region(baselines["yield"], country, commodity, region_members)
    livestock_val = _lookup_with_region(baselines["livestock_yield"], country, commodity, region_members)

    crop_ok = crop_val is not None and np.isfinite(crop_val)
    livestock_ok = livestock_val is not None and np.isfinite(livestock_val)
    if crop_ok and livestock_ok:
        crop_f = float(crop_val)
        livestock_f = float(livestock_val)
        if np.isclose(crop_f, livestock_f, rtol=1e-6, atol=1e-12):
            return crop_f, "t/ha"
        if livestock_f != 0.0:
            ratio = abs(crop_f / livestock_f)
            if 500.0 <= ratio <= 1500.0:
                return livestock_f, "t/head"
        return crop_f, "t/ha"
    if crop_ok:
        return float(crop_val), "t/ha"
    if livestock_ok:
        return float(livestock_val), "t/head"
    return None, None


def _rescale_u(u_raw: float, q_low: Optional[float], q_high: Optional[float]) -> float:
    try:
        u = float(u_raw)
    except Exception:
        u = 0.5
    if u < 0.0:
        u = 0.0
    elif u > 1.0:
        u = 1.0
    if q_low is None or q_high is None:
        return u
    try:
        ql = float(q_low)
        qh = float(q_high)
    except Exception:
        return u
    ql = max(0.0, min(1.0, ql))
    qh = max(0.0, min(1.0, qh))
    if qh < ql:
        ql, qh = qh, ql
    return ql + (qh - ql) * u


def _u_for_country(spec: dict, eff, *, country: str, commodity: str, mode: str) -> float:
    base_u = spec.get("u", 0.5)
    if spec.get("pre_sampled", False) or mode == "shared":
        if spec.get("u_is_rescaled", False):
            try:
                u = float(base_u)
            except Exception:
                u = 0.5
            return max(0.0, min(1.0, u))
        return _rescale_u(base_u, spec.get("q_low"), spec.get("q_high"))
    import hashlib
    if mode == "per_country_commodity":
        key = f"{eff.scenario_id}|{eff.kind}|{country}|{commodity}"
    elif mode == "per_commodity":
        key = f"{eff.scenario_id}|{eff.kind}|{commodity}"
    elif mode == "per_country":
        key = f"{eff.scenario_id}|{eff.kind}|{country}"
    else:
        key = f"{eff.scenario_id}|{eff.kind}"
    seed = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    return _rescale_u(float(rng.random()), spec.get("q_low"), spec.get("q_high"))


def _u_for_ef(spec: dict,
              eff,
              *,
              country: str,
              commodity: str,
              process_key: str,
              mode: str) -> float:
    base_u = spec.get("u", 0.5)
    if spec.get("pre_sampled", False) or mode == "shared":
        if spec.get("u_is_rescaled", False):
            try:
                u = float(base_u)
            except Exception:
                u = 0.5
            return max(0.0, min(1.0, u))
        return _rescale_u(base_u, spec.get("q_low"), spec.get("q_high"))
    import hashlib
    if mode == "per_country_commodity":
        key = f"{eff.scenario_id}|{eff.kind}|{process_key}|{country}|{commodity}"
    elif mode == "per_commodity":
        key = f"{eff.scenario_id}|{eff.kind}|{process_key}|{commodity}"
    elif mode == "per_country":
        key = f"{eff.scenario_id}|{eff.kind}|{process_key}|{country}"
    else:
        key = f"{eff.scenario_id}|{eff.kind}|{process_key}"
    seed = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    return _rescale_u(float(rng.random()), spec.get("q_low"), spec.get("q_high"))


def _pick_year_value(row: pd.Series, year_cols: List[str], base_year: int) -> Optional[float]:
    base_col = f"Y{base_year}"
    if base_col in row.index and pd.notna(row.get(base_col)):
        try:
            return float(row.get(base_col))
        except Exception:
            pass
    vals = row[year_cols]
    vals = vals.dropna()
    if vals.empty:
        return None
    try:
        return float(vals.iloc[-1])
    except Exception:
        return None


def _load_ef_params(path: Path, *, sheet_name: Optional[str], base_year: int) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    def _usecols(c: object) -> bool:
        s = str(c).strip()
        if s in {"M49_Country_Code", "Item", "Process", "paramName", "paramMMS", "units"}:
            return True
        if s.startswith("Y") and s[1:].isdigit():
            return True
        return False
    try:
        df = pd.read_excel(path, sheet_name=sheet_name, usecols=_usecols)
    except Exception:
        return pd.DataFrame()
    df.columns = [str(c).strip() for c in df.columns]
    if "paramName" not in df.columns:
        return pd.DataFrame()
    df["paramName"] = df["paramName"].astype(str).str.strip().str.lower()
    df = df[df["paramName"] == "emission factor"].copy()
    if df.empty:
        return pd.DataFrame()
    year_cols = [c for c in df.columns if str(c).startswith("Y") and str(c)[1:].isdigit()]
    year_cols = sorted(year_cols, key=lambda x: int(str(x)[1:]))
    if not year_cols:
        return pd.DataFrame()
    df["__base"] = df.apply(lambda r: _pick_year_value(r, year_cols, base_year), axis=1)
    df = df.dropna(subset=["__base"])
    df["m49"] = df["M49_Country_Code"].apply(_normalize_m49)
    df["ghg"] = df.get("paramMMS", "All").fillna("All").astype(str).str.strip()
    if "units" in df.columns:
        df["__unit"] = df["units"].astype(str).str.strip()
    df = df.dropna(subset=["m49", "Item", "Process"])
    cols = ["m49", "Item", "Process", "ghg", "__base"]
    if "__unit" in df.columns:
        cols.append("__unit")
    return df[cols]


def _load_fish_ef_base(panel_path: Path, *, base_year: int) -> Dict[Tuple[str, str, str, str], float]:
    if not panel_path.exists():
        return {}
    try:
        df = pd.read_excel(panel_path, sheet_name="country_year_panel")
    except Exception:
        return {}
    df.columns = [str(c).strip() for c in df.columns]
    if "Year" not in df.columns or "M49_Country_Code" not in df.columns:
        return {}
    df["M49_Country_Code"] = df["M49_Country_Code"].apply(_normalize_m49)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["M49_Country_Code", "Year"])
    df["Year"] = df["Year"].astype(int)
    df = df.sort_values(["M49_Country_Code", "Year"])
    def _pick(g: pd.DataFrame) -> pd.Series:
        base = g[g["Year"] == base_year]
        if base.empty:
            base = g[g["Year"] <= base_year]
        if base.empty:
            base = g
        return base.iloc[-1]
    base = df.groupby("M49_Country_Code", as_index=False).apply(_pick).reset_index(drop=True)
    out: Dict[Tuple[str, str, str, str], float] = {}
    for r in base.itertuples(index=False):
        m49 = getattr(r, "M49_Country_Code")
        try:
            ch4 = float(getattr(r, "EF_aqua_CH4_kg_per_ha_yr_median"))
        except Exception:
            ch4 = None
        try:
            n2o = float(getattr(r, "EF_aqua_N2O_kg_per_kg_median"))
        except Exception:
            n2o = None
        if m49 and ch4 is not None and np.isfinite(ch4):
            out[(m49, "Fish, Seafood", "Fish farming", "CH4")] = ch4
        if m49 and n2o is not None and np.isfinite(n2o):
            out[(m49, "Fish, Seafood", "Fish farming", "N2O")] = n2o
    return out


def _load_fish_ef_units(panel_path: Path) -> Dict[Tuple[str, str, str, str], str]:
    if not panel_path.exists():
        return {}
    try:
        df = pd.read_excel(panel_path, sheet_name="country_year_panel", nrows=1)
    except Exception:
        return {}
    df.columns = [str(c).strip() for c in df.columns]
    out: Dict[Tuple[str, str, str, str], str] = {}
    if "EF_aqua_CH4_kg_per_ha_yr_median" in df.columns:
        out[("Fish, Seafood", "Fish farming", "CH4")] = "kg/ha/yr"
    if "EF_aqua_N2O_kg_per_kg_median" in df.columns:
        out[("Fish, Seafood", "Fish farming", "N2O")] = "kg/kg"
    return out


def _load_mc_baselines(paths: DataPaths, universe, *, base_year: int) -> Dict[str, object]:
    production_stats = load_production_statistics(
        paths,
        universe,
        feed_requirement_scheme=CFG.get("feed_requirement_scheme"),
    )
    yield_base = _build_base_map(
        production_stats.get("yield", pd.DataFrame()), "yield_t_per_ha", base_year=base_year, universe=universe
    )
    livestock_yield_base = _build_base_map(
        production_stats.get("livestock_yield", pd.DataFrame()), "yield_t_per_head", base_year=base_year, universe=universe
    )
    fert_base = _build_base_map(
        production_stats.get("fertilizer_efficiency", pd.DataFrame()),
        "fertilizer_efficiency_kgN_per_ha",
        base_year=base_year,
        universe=universe,
    )
    manure_base = _build_base_map(
        production_stats.get("manure_management_ratio", pd.DataFrame()),
        "manure_management_ratio",
        base_year=base_year,
        universe=universe,
    )
    feed_base = _build_base_map(
        production_stats.get("feed_requirement", pd.DataFrame()),
        "feed_requirement_kg_per_head",
        base_year=base_year,
        universe=universe,
    )
    comp_path = Path(get_input_base()) / "Production_Trade" / "Demand_composition.xlsx"
    losses_raw = _load_demand_composition_losses_ratio(str(comp_path))
    losses_base: Dict[Tuple[str, str], float] = {}
    demand_item_map: Dict[str, List[str]] = {}
    try:
        _, demand_item_map = _load_item_demand_extra_and_map(paths.dict_v3_path)
    except Exception:
        demand_item_map = {}
    # Precompute item->commodity mapping (normalized) to align with model commodities
    item_to_comms: Dict[str, List[str]] = {}
    for comm, items in (demand_item_map or {}).items():
        if not comm or not items:
            continue
        for item in items:
            item_norm = _normalize_comp_item_name(item)
            if not item_norm:
                continue
            item_to_comms.setdefault(item_norm, []).append(comm)
    # Direct or mapped aggregation from demand-composition items to model commodities
    acc: Dict[Tuple[str, str], float] = {}
    cnt: Dict[Tuple[str, str], int] = {}
    for (m49, item), val in (losses_raw or {}).items():
        m49_norm = _normalize_m49(m49)
        if not m49_norm:
            continue
        item_norm = _normalize_comp_item_name(item)
        try:
            val_f = float(val)
        except Exception:
            continue
        # Direct match (item == commodity)
        losses_base[(m49_norm, item_norm)] = val_f
        # Map to commodities via demand_item_map
        for comm in item_to_comms.get(item_norm, []):
            key = (m49_norm, comm)
            acc[key] = acc.get(key, 0.0) + val_f
            cnt[key] = cnt.get(key, 0) + 1
    for key, total in acc.items():
        n = cnt.get(key, 0)
        if n > 0 and key not in losses_base:
            losses_base[key] = total / float(n)

    ef_base: Dict[Tuple[str, str, str, str], float] = {}
    ef_unit: Dict[Tuple[str, str, str, str], str] = {}
    src_base = Path(get_src_base())
    gle_df = _load_ef_params(src_base / "GLE_parameters.xlsx", sheet_name=0, base_year=base_year)
    gce_df = _load_ef_params(src_base / "GCE_parameters.xlsx", sheet_name="GCE_parameters", base_year=base_year)
    soil_df = _load_ef_params(src_base / "Soil_parameters.xlsx", sheet_name=0, base_year=base_year)
    for df in (gle_df, gce_df, soil_df):
        if df is None or df.empty:
            continue
        grouped = df.groupby(["m49", "Item", "Process", "ghg"], as_index=False)["__base"].mean()
        for r in grouped.to_dict("records"):
            try:
                ef_base[(str(r["m49"]), str(r["Item"]), str(r["Process"]), str(r["ghg"]))] = float(r["__base"])
            except Exception:
                continue
        if "__unit" in df.columns:
            unit_rows = df.dropna(subset=["__unit"])[["m49", "Item", "Process", "ghg", "__unit"]]
            for r in unit_rows.to_dict("records"):
                key = (str(r["m49"]), str(r["Item"]), str(r["Process"]), str(r["ghg"]))
                if key not in ef_unit:
                    try:
                        ef_unit[key] = str(r["__unit"])
                    except Exception:
                        continue
    fish_panel = Path(get_input_base()) / "Aquaculture" / "fish_seafood_country_panel_2000_present.xlsx"
    ef_base.update(_load_fish_ef_base(fish_panel, base_year=base_year))
    fish_units = _load_fish_ef_units(fish_panel)
    for (item, proc, ghg), u in fish_units.items():
        for key in list(ef_base.keys()):
            if key[1] == item and key[2] == proc and key[3] == ghg:
                ef_unit.setdefault(key, u)

    return {
        "yield": yield_base,
        "livestock_yield": livestock_yield_base,
        "fertilizer": fert_base,
        "manure": manure_base,
        "feed": feed_base,
        "losses": losses_base,
        "ef": ef_base,
        "ef_unit": ef_unit,
    }


def _lookup_ef_base_value(
    baselines: Dict[str, object],
    m49: object,
    commodity: object,
    process: object,
    ghg: object,
) -> Optional[float]:
    ef_base = baselines.get("ef", {}) or {}
    m49_norm = _normalize_m49(m49)
    key = (str(m49_norm), str(commodity), str(process), str(ghg))
    val = ef_base.get(key)
    if val is not None and np.isfinite(val):
        return float(val)

    # Fallback for legacy rows where M49 may already be normalized or where
    # GHG was left as All despite process-specific EF tables.
    raw_key = (str(m49), str(commodity), str(process), str(ghg))
    val = ef_base.get(raw_key)
    if val is not None and np.isfinite(val):
        return float(val)

    if str(ghg).strip().lower() in {"", "all", "nan"}:
        matches = [
            float(v)
            for (m49_k, item_k, proc_k, _ghg_k), v in ef_base.items()
            if str(m49_k) == str(m49_norm)
            and str(item_k) == str(commodity)
            and str(proc_k) == str(process)
            and v is not None
            and np.isfinite(v)
        ]
        if matches:
            return float(np.mean(matches))
    return None


def _lookup_ef_unit(
    baselines: Dict[str, object],
    m49: object,
    commodity: object,
    process: object,
    ghg: object,
) -> Optional[str]:
    ef_unit = baselines.get("ef_unit", {}) or {}
    m49_norm = _normalize_m49(m49)
    for key in (
        (str(m49_norm), str(commodity), str(process), str(ghg)),
        (str(m49), str(commodity), str(process), str(ghg)),
    ):
        unit = ef_unit.get(key)
        if unit is not None and str(unit).strip():
            return str(unit)
    if str(ghg).strip().lower() in {"", "all", "nan"}:
        for (m49_k, item_k, proc_k, _ghg_k), unit in ef_unit.items():
            if (
                str(m49_k) == str(m49_norm)
                and str(item_k) == str(commodity)
                and str(proc_k) == str(process)
                and unit is not None
                and str(unit).strip()
            ):
                return str(unit)
    return None


def _lookup_base_value(kind: str,
                       country: str,
                       commodity: str,
                       baselines: Dict[str, object],
                       region_members: Dict[str, List[str]]) -> Optional[float]:
    if kind == "yield_rate":
        val, _unit = _lookup_yield_base_value_and_unit(country, commodity, baselines, region_members)
        return val
    if kind == "fertilizer_rate":
        return _lookup_with_region(baselines["fertilizer"], country, commodity, region_members)
    if kind == "manure_management_ratio":
        return _lookup_with_region(baselines["manure"], country, commodity, region_members)
    if kind == "losses_ratio":
        return _lookup_with_region(baselines["losses"], country, commodity, region_members)
    if kind in ("feed_intensity", "feed_efficiency"):
        return _lookup_with_region(baselines["feed"], country, commodity, region_members)
    if kind == "ruminant_reduction":
        # Baseline rate is 0 (no reduction at Y2020).
        return 0.0
    if kind == "crop_soil_management_ratio":
        # Rate-only scenario for process emissions.
        return 0.0
    return None


def _value_unit_for(kind: str,
                    country: str,
                    commodity: str,
                    baselines: Dict[str, object],
                    region_members: Dict[str, List[str]],
                    *,
                    process: Optional[str] = None,
                    ghg: Optional[str] = None,
                    m49: Optional[str] = None) -> Optional[str]:
    if kind == "yield_rate":
        _val, unit = _lookup_yield_base_value_and_unit(country, commodity, baselines, region_members)
        return unit
    if kind == "fertilizer_rate":
        return "kgN/ha"
    if kind in ("manure_management_ratio", "ruminant_reduction", "losses_ratio", "crop_soil_management_ratio"):
        return "ratio"
    if kind in ("feed_intensity", "feed_efficiency"):
        return "kg/head"
    if kind in ("land_carbon_price", "land_co2_price"):
        return "$/tCO2e"
    if kind == "emission_factor":
        if m49 is None:
            m49 = ""
        return _lookup_ef_unit(baselines, m49, commodity, process, ghg)
    return None


def _calc_ratio(kind: str,
                unit: str,
                value: float,
                base_val: Optional[float],
                *,
                spec: Optional[dict],
                u_val: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    if spec and (spec.get("lo_is_y2020") or spec.get("hi_is_y2020")):
        lo = float(spec.get("lo", 0.0))
        hi = float(spec.get("hi", 0.0))
        u = float(u_val if u_val is not None else spec.get("u", 0.5))
        if base_val is not None and np.isfinite(base_val) and base_val != 0:
            lo_ratio = lo if spec.get("lo_is_y2020") else lo / base_val
            hi_ratio = hi if spec.get("hi_is_y2020") else hi / base_val
            ratio = lo_ratio + (hi_ratio - lo_ratio) * u
            return ratio, base_val * ratio
        if spec.get("lo_is_y2020") and spec.get("hi_is_y2020"):
            ratio = lo + (hi - lo) * u
            return ratio, None
        return None, None

    unit_l = str(unit or "").strip().lower()
    if unit_l == "rate":
        ratio = 1.0 + float(value)
        if base_val is None or not np.isfinite(base_val) or base_val == 0:
            # For rate-only variables with zero/unknown baseline (e.g., ruminant_reduction),
            # report the rate itself as sample value.
            return ratio, float(value)
    elif unit_l == "multiplier":
        ratio = float(value)
    else:
        if base_val is None or not np.isfinite(base_val) or base_val == 0:
            return None, float(value)
        ratio = float(value) / float(base_val)
    if base_val is None or not np.isfinite(base_val):
        return ratio, None
    return ratio, float(base_val) * ratio


def _calc_bound_value(kind: str,
                      unit: str,
                      bound: Optional[float],
                      base_val: Optional[float],
                      *,
                      is_y2020: bool) -> Optional[float]:
    if bound is None:
        return None
    try:
        bound_val = float(bound)
    except Exception:
        return None
    if not np.isfinite(bound_val):
        return None

    unit_l = str(unit or "").strip().lower()
    if is_y2020:
        if base_val is None or not np.isfinite(base_val):
            return None
        return float(base_val) * bound_val
    if unit_l == "rate":
        if base_val is None or not np.isfinite(base_val) or base_val == 0:
            return float(bound_val)
        ratio = 1.0 + bound_val
        return float(base_val) * ratio
    if unit_l == "multiplier":
        if base_val is None or not np.isfinite(base_val):
            return None
        return float(base_val) * bound_val
    return bound_val


def _build_mc_sample_rows(effects: Iterable,
                          *,
                          universe,
                          baselines: Dict[str, object],
                          scenario_id: str,
                          sample_id: int,
                          attempt: int,
                          mc_mode_default: str,
                          mc_mode_non_ef: Optional[str] = None,
                          mc_mode_ef: Optional[str] = None,
                          ef_process_mode: str = "all") -> Dict[str, List[Dict[str, object]]]:
    region_members = _build_region_members(universe)
    out: Dict[str, List[Dict[str, object]]] = {}
    for eff in effects:
        kind = _normalize_kind(getattr(eff, "kind", ""))
        unit = getattr(eff, "unit", "")
        mc_unit = unit
        spec = getattr(eff, "mc_bounds_raw", None)
        mode_non_ef = str(mc_mode_non_ef or mc_mode_default or "shared").strip().lower()
        mode_ef = str(mc_mode_ef or mc_mode_default or "shared").strip().lower()
        def _normalize_mode(m: str) -> str:
            if m in ("a", "country", "per_country", "independent", "per-country"):
                return "per_country"
            if m in ("per_commodity", "commodity", "per-item", "per_item"):
                return "per_commodity"
            if m in ("per_country_commodity", "country_commodity", "per-country-commodity", "per_country_item", "per-item"):
                return "per_country_commodity"
            return "shared"
        mode_non_ef = _normalize_mode(mode_non_ef)
        mode_ef = _normalize_mode(mode_ef)
        ef_process_mode_l = str(ef_process_mode or "all").strip().lower()

        countries = getattr(eff, "countries", None) or []
        commodities = getattr(eff, "commodities", None) or []
        processes = getattr(eff, "processes", None) or []
        ghg_sel = getattr(eff, "ghg_sel", "All") or "All"
        for country in countries:
            for commodity in commodities:
                if kind == "emission_factor":
                    for process in processes:
                        m49 = universe.m49_by_country.get(country)
                        m49_norm = _normalize_m49(m49) if m49 else _normalize_m49(country)
                        base_val = _lookup_ef_base_value(baselines, m49_norm, commodity, process, ghg_sel)
                        value_unit = _value_unit_for(
                            kind,
                            country,
                            commodity,
                            baselines,
                            region_members,
                            process=process,
                            ghg=ghg_sel,
                            m49=m49_norm,
                        )
                        u_val = None
                        is_y2020 = bool(spec and (spec.get("lo_is_y2020") or spec.get("hi_is_y2020")))
                        pre_sampled = bool(spec and spec.get("pre_sampled", False))
                        if pre_sampled:
                            u_val = (spec or {}).get("u")
                        elif spec:
                            process_key = process if ef_process_mode_l == "by_process" else "All"
                            u_val = _u_for_ef(
                                spec,
                                eff,
                                country=country,
                                commodity=commodity,
                                process_key=process_key,
                                mode=mode_ef,
                        )
                        u_used = u_val if u_val is not None else (spec or {}).get("u")
                        value_draw_out = eff.value_2080
                        if u_used is not None and not is_y2020 and spec is not None and not pre_sampled:
                            lo = (spec or {}).get("lo")
                            hi = (spec or {}).get("hi")
                            if lo is not None and hi is not None:
                                value_draw_out = float(lo) + (float(hi) - float(lo)) * float(u_used)
                        if is_y2020 and u_used is not None:
                            value_draw_out = float(u_used)
                        min_bound_val = _calc_bound_value(
                            kind, unit, (spec or {}).get("lo"), base_val, is_y2020=bool((spec or {}).get("lo_is_y2020"))
                        )
                        max_bound_val = _calc_bound_value(
                            kind, unit, (spec or {}).get("hi"), base_val, is_y2020=bool((spec or {}).get("hi_is_y2020"))
                        )
                        ratio = None
                        sample_val = None
                        if is_y2020 and u_used is not None:
                            if min_bound_val is not None and max_bound_val is not None:
                                sample_val = float(min_bound_val) + (float(max_bound_val) - float(min_bound_val)) * float(u_used)
                                if base_val is not None and np.isfinite(base_val) and base_val != 0:
                                    ratio = float(sample_val) / float(base_val)
                            else:
                                lo = (spec or {}).get("lo")
                                hi = (spec or {}).get("hi")
                                if (spec or {}).get("lo_is_y2020") and (spec or {}).get("hi_is_y2020") and lo is not None and hi is not None:
                                    try:
                                        lo_f = float(lo)
                                        hi_f = float(hi)
                                        ratio = lo_f + (hi_f - lo_f) * float(u_used)
                                        if base_val is not None and np.isfinite(base_val):
                                            sample_val = float(base_val) * float(ratio)
                                    except Exception:
                                        ratio = None
                                        sample_val = None
                        else:
                            ratio, sample_val = _calc_ratio(kind, unit, value_draw_out, base_val, spec=None, u_val=None)
                        out.setdefault(kind, []).append({
                            "scenario_id": scenario_id,
                            "sample_id": sample_id,
                            "attempt": attempt,
                            "country": country,
                            "commodity": commodity,
                            "process": process,
                            "ghg": ghg_sel,
                            "mc_unit": mc_unit,
                            "value_unit": value_unit,
                            "value_y2020": base_val,
                            "value_draw": value_draw_out,
                            "value_sample": sample_val,
                            "ratio": ratio,
                            "mc_u": u_used,
                            "min_bound": (spec or {}).get("lo"),
                            "max_bound": (spec or {}).get("hi"),
                            "min_bound_value": min_bound_val,
                            "max_bound_value": max_bound_val,
                            "min_is_y2020": (spec or {}).get("lo_is_y2020"),
                            "max_is_y2020": (spec or {}).get("hi_is_y2020"),
                        })
                else:
                    base_val = _lookup_base_value(kind, country, commodity, baselines, region_members)
                    value_unit = _value_unit_for(
                        kind,
                        country,
                        commodity,
                        baselines,
                        region_members,
                    )
                    u_val = None
                    is_y2020 = bool(spec and (spec.get("lo_is_y2020") or spec.get("hi_is_y2020")))
                    pre_sampled = bool(spec and spec.get("pre_sampled", False))
                    if pre_sampled:
                        u_val = (spec or {}).get("u")
                    elif spec:
                        u_val = _u_for_country(spec, eff, country=country, commodity=commodity, mode=mode_non_ef)
                    u_used = u_val if u_val is not None else (spec or {}).get("u")
                    value_draw_out = eff.value_2080
                    if u_used is not None and not is_y2020 and spec is not None and not pre_sampled:
                        lo = (spec or {}).get("lo")
                        hi = (spec or {}).get("hi")
                        if lo is not None and hi is not None:
                            value_draw_out = float(lo) + (float(hi) - float(lo)) * float(u_used)
                    if is_y2020 and u_used is not None:
                        value_draw_out = float(u_used)
                    min_bound_val = _calc_bound_value(
                        kind, unit, (spec or {}).get("lo"), base_val, is_y2020=bool((spec or {}).get("lo_is_y2020"))
                    )
                    max_bound_val = _calc_bound_value(
                        kind, unit, (spec or {}).get("hi"), base_val, is_y2020=bool((spec or {}).get("hi_is_y2020"))
                    )
                    ratio = None
                    sample_val = None
                    if is_y2020 and u_used is not None:
                        if min_bound_val is not None and max_bound_val is not None:
                            sample_val = float(min_bound_val) + (float(max_bound_val) - float(min_bound_val)) * float(u_used)
                            if base_val is not None and np.isfinite(base_val) and base_val != 0:
                                ratio = float(sample_val) / float(base_val)
                        else:
                            lo = (spec or {}).get("lo")
                            hi = (spec or {}).get("hi")
                            if (spec or {}).get("lo_is_y2020") and (spec or {}).get("hi_is_y2020") and lo is not None and hi is not None:
                                try:
                                    lo_f = float(lo)
                                    hi_f = float(hi)
                                    ratio = lo_f + (hi_f - lo_f) * float(u_used)
                                    if base_val is not None and np.isfinite(base_val):
                                        sample_val = float(base_val) * float(ratio)
                                except Exception:
                                    ratio = None
                                    sample_val = None
                    else:
                        ratio, sample_val = _calc_ratio(kind, unit, value_draw_out, base_val, spec=None, u_val=None)
                    if kind == "losses_ratio":
                        base_loss = base_val if base_val is not None and np.isfinite(base_val) else 0.0
                        sample_val = _final_loss_ratio_from_delta(base_loss, value_draw_out)
                        ratio = _loss_multiplier_from_delta(base_loss, value_draw_out)
                    out.setdefault(kind, []).append({
                        "scenario_id": scenario_id,
                        "sample_id": sample_id,
                        "attempt": attempt,
                        "country": country,
                        "commodity": commodity,
                        "mc_unit": mc_unit,
                        "value_unit": value_unit,
                        "value_y2020": base_val,
                        "value_draw": value_draw_out,
                        "value_sample": sample_val,
                        "ratio": ratio,
                        "mc_u": u_used,
                        "min_bound": (spec or {}).get("lo"),
                        "max_bound": (spec or {}).get("hi"),
                        "min_bound_value": min_bound_val,
                        "max_bound_value": max_bound_val,
                        "min_is_y2020": (spec or {}).get("lo_is_y2020"),
                        "max_is_y2020": (spec or {}).get("hi_is_y2020"),
                    })
    return out


def _write_mc_sample_xlsx(out_dir: Path,
                          *,
                          scenario_id: str,
                          sample_id: int,
                          attempt: int,
                          rows_by_kind: Dict[str, List[Dict[str, object]]]) -> Path:
    _ensure_dir(out_dir)
    out_path = out_dir / f"{scenario_id}_attempt{attempt:02d}.xlsx"
    with pd.ExcelWriter(out_path) as writer:
        meta_rows = [
            {"key": "scenario_id", "value": scenario_id, "description": "MC 场景编号"},
            {"key": "sample_id", "value": sample_id, "description": "样本序号(从1开始)"},
            {"key": "attempt", "value": attempt, "description": "该样本的第几次尝试(重采样计数)"},
            {"key": "column.scenario_id", "value": "", "description": "同 scenario_id"},
            {"key": "column.sample_id", "value": "", "description": "同 sample_id"},
            {"key": "column.attempt", "value": "", "description": "同 attempt"},
            {"key": "column.country", "value": "", "description": "国家/地区名称"},
            {"key": "column.commodity", "value": "", "description": "商品/品类名称"},
            {"key": "column.process", "value": "", "description": "排放过程(仅 emission_factor / crop_soil_management_ratio)"},
            {"key": "column.ghg", "value": "", "description": "GHG 类型(仅 emission_factor)"},
            {"key": "column.mc_unit", "value": "", "description": "MC配置中的单位类型(rate/multiplier/amount/absolute)"},
            {"key": "column.value_unit", "value": "", "description": "模型变量真实单位(如t/ha、t/head、kgN/ha、ratio等)"},
            {"key": "column.value_y2020", "value": "", "description": "基期Y2020的实际值(按国家-商品取值)"},
            {"key": "column.value_draw", "value": "", "description": "抽样值；存在Y2020边界时记为实际使用的u(=mc_u)"},
            {"key": "column.value_sample", "value": "", "description": "用于模型的最终值；若有Y2020边界则先算min/max再用mc_u线性插值"},
            {"key": "column.ratio", "value": "", "description": "相对基期比例(value_sample/value_y2020)。若基期缺失且Y2020边界为双比例，则按lo/hi与u计算"},
            {"key": "column.mc_u", "value": "", "description": "抽样位置u∈[0,1]，已应用q_low/q_high裁剪"},
            {"key": "column.min_bound", "value": "", "description": "原始最小边界(解析后数值，例如Y2020_90转为0.9)"},
            {"key": "column.max_bound", "value": "", "description": "原始最大边界(解析后数值)"},
            {"key": "column.min_bound_value", "value": "", "description": "按国家基期换算后的最小边界值(绝对值)"},
            {"key": "column.max_bound_value", "value": "", "description": "按国家基期换算后的最大边界值(绝对值)"},
            {"key": "column.min_is_y2020", "value": "", "description": "Min_bound 是否为 Y2020 比例边界"},
            {"key": "column.max_is_y2020", "value": "", "description": "Max_bound 是否为 Y2020 比例边界"},
        ]
        meta = pd.DataFrame(meta_rows)
        meta.to_excel(writer, sheet_name="meta", index=False)
        for kind, rows in rows_by_kind.items():
            if not rows:
                continue
            df = pd.DataFrame(rows)
            sheet = str(kind)[:31] if kind else "data"
            df.to_excel(writer, sheet_name=sheet, index=False)
    return out_path


def _extract_failed_regions(exc: Exception, universe) -> Tuple[set, set, float]:
    failed_info = getattr(exc, "failed_regions", None)
    failed_regions: set = set()
    failed_countries: set = set()
    max_severity = 0.0
    if not failed_info:
        return failed_regions, failed_countries, max_severity
    for entry in failed_info:
        try:
            region_key, need, cap, *_ = entry
        except Exception:
            continue
        region_str = str(region_key)
        failed_regions.add(region_str)
        m49_key = _norm_m49_key(region_key)
        country = universe.country_by_m49.get(m49_key, '')
        if country:
            failed_countries.add(country)
        else:
            # fallback: if region_key itself is a country name
            if region_str in (universe.countries or []):
                failed_countries.add(region_str)
        try:
            need_val = float(need)
            cap_val = float(cap)
            if cap_val > 0:
                sev = max(0.0, need_val / cap_val - 1.0)
                if sev > max_severity:
                    max_severity = sev
        except Exception:
            continue
    return failed_regions, failed_countries, max_severity


def _row_is_targeted(region_sel: str,
                     *,
                     failed_regions: set,
                     failed_countries: set,
                     region_members: Dict[str, List[str]],
                     include_all: bool) -> bool:
    region = str(region_sel or '').strip()
    if not region or region.lower() == 'all':
        return bool(include_all)
    if region in failed_regions or region in failed_countries:
        return True
    if region in region_members:
        members = region_members.get(region, [])
        return any(c in failed_countries for c in members)
    return False


def _apply_directional_bias(bounds: Tuple[float, float],
                            *,
                            direction: str,
                            bias_step: float,
                            max_bias: float,
                            min_q_low: float,
                            max_q_high: float,
                            scale: float) -> Tuple[float, float]:
    ql, qh = bounds
    try:
        step = float(bias_step)
    except Exception:
        step = 0.0
    try:
        max_b = float(max_bias)
    except Exception:
        max_b = 0.0
    if step <= 0 or max_b <= 0:
        return ql, qh
    try:
        scale_val = float(scale)
    except Exception:
        scale_val = 1.0
    if scale_val <= 0:
        scale_val = 1.0
    bias = min(step * scale_val, max_b)
    dir_l = str(direction or '').strip().lower()
    if dir_l in ('up', 'high', 'increase', '+', 'higher'):
        ql += bias
        qh += bias
    elif dir_l in ('down', 'low', 'decrease', '-', 'lower'):
        ql -= bias
        qh -= bias
    ql = max(min_q_low, min(max_q_high, ql))
    qh = max(min_q_low, min(max_q_high, qh))
    if ql > qh:
        mid = (ql + qh) * 0.5
        ql = mid
        qh = mid
    return ql, qh


def _build_row_q_bounds(specs_df: pd.DataFrame,
                        *,
                        base_bounds: Tuple[float, float],
                        group_lookup: Dict[str, str],
                        group_q_bounds: Optional[Dict[str, Tuple[float, float]]],
                        direction_cfg: Dict[str, object],
                        region_cfg: Dict[str, object],
                        failed_regions: set,
                        failed_countries: set,
                        region_members: Dict[str, List[str]],
                        min_q_low: float,
                        max_q_high: float,
                        severity_scale: float) -> Tuple[Dict[int, Tuple[float, float]], Dict[str, int]]:
    row_q: Dict[int, Tuple[float, float]] = {}
    stats = {"targeted_rows": 0, "adjusted_rows": 0}
    dir_enabled = bool(direction_cfg.get("enabled", False))
    bias_step = float(direction_cfg.get("bias_step", 0.0) or 0.0)
    max_bias = float(direction_cfg.get("max_bias", 0.0) or 0.0)
    kind_dir = {str(k).strip().lower(): str(v) for k, v in (direction_cfg.get("kind_direction", {}) or {}).items()}
    region_enabled = bool(region_cfg.get("enabled", False))
    include_all = bool(region_cfg.get("include_all", True))

    df = specs_df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for idx, r in enumerate(df.itertuples(index=False)):
        elem_raw = getattr(r, "Element", "")
        kind = _mc_element_to_kind(elem_raw)
        if not kind:
            continue
        base = base_bounds
        if group_q_bounds and group_lookup:
            grp = group_lookup.get(kind)
            if grp in group_q_bounds:
                base = group_q_bounds[grp]
        ql, qh = base
        targeted = True
        if region_enabled:
            region_sel = getattr(r, "Region_cat", "All")
            targeted = _row_is_targeted(
                str(region_sel),
                failed_regions=failed_regions,
                failed_countries=failed_countries,
                region_members=region_members,
                include_all=include_all,
            )
        if targeted:
            stats["targeted_rows"] += 1
        if dir_enabled and targeted:
            direction = kind_dir.get(kind)
            if direction:
                ql2, qh2 = _apply_directional_bias(
                    (ql, qh),
                    direction=direction,
                    bias_step=bias_step,
                    max_bias=max_bias,
                    min_q_low=min_q_low,
                    max_q_high=max_q_high,
                    scale=severity_scale,
                )
                if (ql2, qh2) != (ql, qh):
                    ql, qh = ql2, qh2
                    stats["adjusted_rows"] += 1
        if (ql, qh) != base:
            row_q[idx] = (ql, qh)
    return row_q, stats


def _norm_field(raw: object, default: str = "All") -> str:
    if raw is None:
        return default
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", "-"):
        return default
    return s


def _configure_cfg(mc_cfg: Dict[str, object]) -> None:
    CFG["solve"] = True
    CFG["use_linear_model"] = mc_cfg.get("use_linear", True)
    CFG["future_last_only"] = mc_cfg.get("future_last_only", True)
    CFG["use_regional_aggregation"] = mc_cfg.get("use_regional", False)
    CFG["premacc_e0"] = mc_cfg.get("pre_macc_e0", False)
    CFG["use_fao_modules"] = mc_cfg.get("use_fao_modules", True)
    CFG["nutrition_profile_sheet"] = str(
        mc_cfg.get("nutrition_profile_sheet", DEFAULT_NUTRITION_PROFILE_SHEET)
        or DEFAULT_NUTRITION_PROFILE_SHEET
    )
    CFG["domestic_supply_simulation_mode"] = str(
        mc_cfg.get("domestic_supply_simulation_mode", "hard_equation") or "hard_equation"
    )
    CFG["supply_curtailment_enabled"] = bool(mc_cfg.get("supply_curtailment_enabled", False))
    CFG["supply_curtailment_penalty"] = float(mc_cfg.get("supply_curtailment_penalty", 1e10) or 1e10)
    # Optional nutrition soft-constraint switch for MC sensitivity
    soft_cfg = mc_cfg.get("nutrition_soft_constraints", {}) or {}
    if soft_cfg.get("enabled"):
        if soft_cfg.get("max_slack_rate") is not None:
            CFG["max_slack_rate"] = float(soft_cfg["max_slack_rate"])
        if soft_cfg.get("slack_penalty") is not None:
            CFG["slack_penalty"] = float(soft_cfg["slack_penalty"])
        dm = soft_cfg.get("demand_method")
        if dm:
            CFG["demand_method"] = str(dm)
        if str(CFG.get("demand_method", "")).lower() == "nutrition_band":
            if soft_cfg.get("nutrition_band_epsilon") is not None:
                CFG["nutrition_band_epsilon"] = float(soft_cfg["nutrition_band_epsilon"])
    # Optional land soft-constraint switch (MC only)
    land_soft = mc_cfg.get("land_soft_constraints", {}) or {}
    CFG["land_soft_constraints_enabled"] = bool(land_soft.get("enabled", False))
    try:
        CFG["land_soft_constraints_max_over_cap_rate"] = float(
            land_soft.get("max_over_cap_rate", 0.0) or 0.0
        )
    except Exception:
        CFG["land_soft_constraints_max_over_cap_rate"] = 0.0
    try:
        CFG["land_soft_constraints_penalty_per_ha"] = float(
            land_soft.get("slack_penalty_per_ha", land_soft.get("penalty_per_ha", 0.0)) or 0.0
        )
    except Exception:
        CFG["land_soft_constraints_penalty_per_ha"] = 0.0
    # Handling of non-EF Y2020_XX: shared / per_country.
    CFG["mc_y2020_non_ef_mode"] = str(mc_cfg.get("mc_non_ef_mode", "shared")).strip().lower() or "shared"




def main() -> None:
    args = CONFIG.copy()
    mc_cfg = args.get("mc", {}) or {}
    mode = str(args.get("mode", "mc") or "").strip().lower()
    if mode in ("variable_effect", "effect", "sensitivity"):
        _configure_cfg(args.get("variable_effect", {}) or {})
        _run_variable_effect(args)
        return
    _configure_cfg(mc_cfg)

    args["paths"] = DataPaths()
    cfg = ScenarioConfig()
    universe = build_universe_from_dict_v3(args["paths"].dict_v3_path, cfg)
    region_members = _build_region_members(universe)
    base_year = int(cfg.years_hist_end or 2020)

    specs_df = load_mc_specs(args["paths"].scenario_config_xlsx)
    if specs_df is None or specs_df.empty:
        raise RuntimeError("MC specs sheet is empty or missing.")
    specs_df = _normalize_mc_specs(specs_df)

    base_out = (
        Path(mc_cfg.get("output_dir") or "")
        if mc_cfg.get("output_dir")
        else Path(get_results_base()) / "MC_Sensitivity_Yield_EF"
    )
    _ensure_dir(base_out)
    mc_samples_dir = base_out / "MC"
    _ensure_dir(mc_samples_dir)
    baselines = _load_mc_baselines(args["paths"], universe, base_year=base_year)
    mc_mode_default = str(CFG.get("mc_y2020_non_ef_mode", "shared")).strip().lower() or "shared"
    mc_mode_non_ef, mc_mode_ef, ef_process_mode = _resolve_mc_modes(mc_cfg)

    targets = _parse_targets(mc_cfg.get("targets", []))
    targets_df = pd.DataFrame(
        [{"label": label, "emission_gt": val} for label, val in targets]
    )
    targets_path = base_out / "targets.csv"
    targets_df.to_csv(targets_path, index=False, encoding="utf-8-sig")

    groups_to_run = (
        ["yield_feed", "emission_factor"]
        if str(mc_cfg.get("group", "both")) == "both"
        else [str(mc_cfg.get("group", "both"))]
    )
    runs_dir = base_out / "runs"
    _ensure_dir(runs_dir)
    summary_dirs: Dict[str, Path] = {}
    for group_key in groups_to_run:
        group_out = base_out / group_key
        summary_dir = group_out / "summary"
        _ensure_dir(summary_dir)
        summary_dirs[group_key] = summary_dir
        _reset_output_file(summary_dir / "samples.csv")

    unit_matrix = _sample_unit_matrix_for_specs(
        specs_df,
        int(mc_cfg.get("samples", 0)),
        seed=int(args["seed"]),
        config=mc_cfg.get("sampling", {}),
    )
    q_bounds = mc_cfg.get("sampling", {}).get("quantile_bounds", (0.0, 1.0))
    try:
        q_bounds = (float(q_bounds[0]), float(q_bounds[1]))
    except Exception:
        q_bounds = (0.0, 1.0)
    adapt_cfg = mc_cfg.get("adaptive_resample", {}) or {}
    adapt_enabled = bool(adapt_cfg.get("enabled", False))
    adapt_mode = str(adapt_cfg.get("mode", "group_quantile") or "").strip().lower()
    groups_cfg = adapt_cfg.get("groups", {}) or {}
    group_lookup = _build_group_lookup(groups_cfg) if groups_cfg else {}
    target_groups = [str(g) for g in (adapt_cfg.get("target_groups", []) or [])]
    direction_cfg = adapt_cfg.get("directional", {}) or {}
    region_cfg = adapt_cfg.get("region_targeting", {}) or {}
    severity_cfg = adapt_cfg.get("severity", {}) or {}
    try:
        shrink_step = float(adapt_cfg.get("shrink_step", 0.0))
    except Exception:
        shrink_step = 0.0
    try:
        min_q_low = float(adapt_cfg.get("min_q_low", 0.0))
        max_q_high = float(adapt_cfg.get("max_q_high", 1.0))
    except Exception:
        min_q_low, max_q_high = 0.0, 1.0
    min_q_low = max(0.0, min(1.0, min_q_low))
    max_q_high = max(0.0, min(1.0, max_q_high))
    if min_q_low > max_q_high:
        min_q_low, max_q_high = max_q_high, min_q_low
    precheck_cfg = mc_cfg.get("precheck", {}) or {}
    precheck_enabled = bool(precheck_cfg.get("enabled", False))
    max_resample = int(precheck_cfg.get("max_resample", 0) or 0)

    for idx in range(int(mc_cfg.get("samples", 0))):
        sample_id = idx + 1
        scenario_id = f"MC_{sample_id:05d}"
        attempt = 0
        group_q_bounds = None
        row_q_bounds = None
        if adapt_enabled and adapt_mode == "group_quantile":
            group_q_bounds = _init_group_q_bounds(q_bounds, groups_cfg, target_groups)
        while True:
            if attempt == 0:
                unit_row = unit_matrix[idx]
            else:
                unit_row = _draw_unit_row_for_specs(
                    specs_df,
                    seed=int(args["seed"]) + idx * 1000 + attempt,
                    config=mc_cfg.get("sampling", {}),
                )
            param_rows = _draw_mc_param_rows(
                specs_df,
                unit_row=unit_row,
                quantile_bounds=q_bounds,
                group_q_bounds=group_q_bounds,
                group_lookup=group_lookup,
                row_q_bounds=row_q_bounds,
                sampling_cfg=mc_cfg.get("sampling", {}),
            )
            scenario_resume_fingerprint = _s51_resume_fingerprint(
                runner_mode="mc",
                scenario_id=scenario_id,
                param_rows=param_rows,
                run_cfg=mc_cfg,
                year=int(args["year"]),
            )
            effects = _build_scenario_effects(
                param_rows,
                universe,
                scenario_id=scenario_id,
                mc_y2020_mode=str(mc_cfg.get("mc_non_ef_mode", "shared")),
                mc_mode_non_ef=mc_mode_non_ef,
                mc_mode_ef=mc_mode_ef,
                ef_process_mode=ef_process_mode,
            )
            attempt_id = attempt + 1
            rows_by_kind = _build_mc_sample_rows(
                effects,
                universe=universe,
                baselines=baselines,
                scenario_id=scenario_id,
                sample_id=sample_id,
                attempt=attempt_id,
                mc_mode_default=mc_mode_default,
                mc_mode_non_ef=mc_mode_non_ef,
                mc_mode_ef=mc_mode_ef,
                ef_process_mode=ef_process_mode,
            )
            _write_mc_sample_xlsx(
                mc_samples_dir,
                scenario_id=scenario_id,
                sample_id=sample_id,
                attempt=attempt_id,
                rows_by_kind=rows_by_kind,
            )

            scenario_dir = runs_dir / scenario_id
            emis_dir = scenario_dir / "Emis"
            emis_path = emis_dir / "emissions_summary.xlsx"
            resample_log = base_out / "resample.log"
            model_log = scenario_dir / "Log" / "model.log"
            resume_validation: Optional[ResumeValidation] = None
            resume_summary_df = pd.DataFrame()
            resume_detail_df = pd.DataFrame()
            reuse_existing = False
            if bool(mc_cfg.get("resume")):
                (
                    resume_validation,
                    resume_summary_df,
                    resume_detail_df,
                ) = _validated_resume_artifacts(
                    scenario_dir,
                    expected_scenario_id=scenario_id,
                    expected_resume_fingerprint=scenario_resume_fingerprint,
                )
                reuse_existing = _resume_artifacts_ready(
                    scenario_dir,
                    resume_validation,
                    resume_summary_df,
                    resume_detail_df,
                )
                if reuse_existing:
                    print(
                        f"[S5_1_1] validated resume {scenario_id}: "
                        f"run_id={resume_validation.run_id}"
                    )
                else:
                    reason = (
                        resume_validation.reason
                        if not resume_validation.allowed
                        else "missing_stale_or_identity_mismatched_required_artifacts"
                    )
                    print(f"[S5_1_1] resume rejected for {scenario_id}: {reason}; rerun")

            if not reuse_existing:
                try:
                    run_one_pipeline(
                        args["paths"],
                        pre_macc_e0=CFG["premacc_e0"],
                        scenario_id=scenario_id,
                        scenario_params=None,
                        scenario_effects=effects,
                        solve=CFG["solve"],
                        use_fao_modules=CFG["use_fao_modules"],
                        future_last_only=CFG["future_last_only"],
                        use_linear=CFG["use_linear_model"],
                        fast_emis_only=bool(mc_cfg.get("fast_emis_only", True)),
                        fast_emis_year=int(args["year"]),
                        resume_fingerprint=scenario_resume_fingerprint,
                        save_root=str(runs_dir),
                        mc_precheck=precheck_enabled,
                        mc_precheck_year=int(args["year"]),
                    )
                except MCPrecheckFailed as exc:
                    attempt += 1
                    header = f"[MC-RESAMPLE] {scenario_id} attempt {attempt} failed: {exc}"
                    print(header)
                    _append_log(resample_log, header)
                    _append_log(model_log, header)
                    cfg_msg = (
                        f"[MC-RESAMPLE] config: adapt_enabled={adapt_enabled}, mode={adapt_mode}, "
                        f"target_groups={target_groups}, shrink_step={shrink_step}, "
                        f"min_q_low={min_q_low}, max_q_high={max_q_high}, base_q_bounds={q_bounds}"
                    )
                    print(cfg_msg)
                    _append_log(resample_log, cfg_msg)
                    _append_log(model_log, cfg_msg)
                    failed_regions, failed_countries, severity = _extract_failed_regions(exc, universe)
                    try:
                        max_scale = float(severity_cfg.get("max_scale", 0.0))
                    except Exception:
                        max_scale = 0.0
                    severity_scale = 1.0
                    if bool(severity_cfg.get("enabled", False)) and severity > 0:
                        if max_scale > 0:
                            severity_scale = 1.0 + min(severity, max_scale)
                        else:
                            severity_scale = 1.0 + severity
                    if failed_regions:
                        msg = f"[MC-RESAMPLE] failed regions: {len(failed_regions)}"
                        print(msg)
                        _append_log(resample_log, msg)
                        _append_log(model_log, msg)
                    if severity > 0:
                        msg = f"[MC-RESAMPLE] severity scale: {severity_scale:.2f} (max severity={severity:.3f})"
                        print(msg)
                        _append_log(resample_log, msg)
                        _append_log(model_log, msg)
                    if adapt_enabled and adapt_mode == "group_quantile" and group_q_bounds is not None:
                        before_bounds = dict(group_q_bounds)
                        _shrink_group_bounds(
                            group_q_bounds,
                            target_groups=target_groups,
                            shrink_step=shrink_step,
                            min_q_low=min_q_low,
                            max_q_high=max_q_high,
                            scale=severity_scale,
                        )
                        if before_bounds != group_q_bounds:
                            changed = [
                                f"{g}: {before_bounds[g]} -> {group_q_bounds[g]}"
                                for g in group_q_bounds
                                if before_bounds.get(g) != group_q_bounds.get(g)
                            ]
                            if changed:
                                msg = f"[MC-RESAMPLE] adjusted group quantile bounds: {', '.join(changed)}"
                                print(msg)
                                _append_log(resample_log, msg)
                                _append_log(model_log, msg)
                        else:
                            msg = "[MC-RESAMPLE] no group quantile adjustment applied"
                            print(msg)
                            _append_log(resample_log, msg)
                            _append_log(model_log, msg)
                    else:
                        msg = "[MC-RESAMPLE] no adaptive resample adjustment (disabled or not configured)"
                        print(msg)
                        _append_log(resample_log, msg)
                        _append_log(model_log, msg)
                    if adapt_enabled and adapt_mode == "group_quantile":
                        row_q_bounds, stats = _build_row_q_bounds(
                            specs_df,
                            base_bounds=q_bounds,
                            group_lookup=group_lookup,
                            group_q_bounds=group_q_bounds,
                            direction_cfg=direction_cfg,
                            region_cfg=region_cfg,
                            failed_regions=failed_regions,
                            failed_countries=failed_countries,
                            region_members=region_members,
                            min_q_low=min_q_low,
                            max_q_high=max_q_high,
                            severity_scale=severity_scale,
                        )
                        if stats.get("adjusted_rows", 0) > 0:
                            msg = (
                                f"[MC-RESAMPLE] directional/region adjustments: "
                                f"targeted_rows={stats.get('targeted_rows', 0)}, "
                                f"adjusted_rows={stats.get('adjusted_rows', 0)}"
                            )
                            print(msg)
                            _append_log(resample_log, msg)
                            _append_log(model_log, msg)
                    if max_resample <= 0 or attempt > max_resample:
                        raise RuntimeError(
                            f"MC precheck failed after {attempt} attempts for {scenario_id}: {exc}"
                        ) from exc
                    continue

                (
                    resume_validation,
                    resume_summary_df,
                    resume_detail_df,
                ) = _validated_resume_artifacts(
                    scenario_dir,
                    expected_scenario_id=scenario_id,
                    expected_resume_fingerprint=scenario_resume_fingerprint,
                )
            if not _resume_artifacts_ready(
                scenario_dir,
                resume_validation,
                resume_summary_df,
                resume_detail_df,
            ):
                reason = (
                    resume_validation.reason
                    if resume_validation is not None and not resume_validation.allowed
                    else "missing_stale_or_identity_mismatched_required_artifacts"
                )
                raise RuntimeError(
                    f"run completed without reusable current-generation outputs: {reason}"
                )
            break

        run_dir = runs_dir / scenario_id
        neg_msg = _validate_nonluc_fast_emissions(
            run_dir,
            validation=resume_validation,
        )
        if neg_msg:
            print(f"[S5_1_1] skip {scenario_id}: {neg_msg}")
            continue

        gap_msg = _validate_market_balance_gap(
            run_dir,
            max_gap_rate=float(mc_cfg.get("market_gap_max_rate", 0.05) or 0.05),
            validation=resume_validation,
        )
        if gap_msg:
            print(f"[S5_1_1] skip {scenario_id}: {gap_msg}")
            continue

        emis_gt = _read_global_emissions_2080_gt(
            emis_path,
            year=args["year"],
            unit_scale=args["unit_scale"],
            validation=resume_validation,
        )
        base_row = {
            "sample_id": sample_id,
            "scenario_id": scenario_id,
            "run_id": resume_validation.run_id,
            "resume_fingerprint": scenario_resume_fingerprint,
            "emissions_2080_gt": emis_gt,
        }
        for group_key in groups_to_run:
            row = dict(base_row)
            row.update(_build_param_map(param_rows, group_key))
            _append_rows_csv([row], summary_dirs[group_key] / "samples.csv")

    out_paths: List[Path] = []
    cost_status_frames: List[pd.DataFrame] = []
    for group_key in groups_to_run:
        samples_path = summary_dirs[group_key] / "samples.csv"
        out_paths.append(samples_path)
        if samples_path.exists():
            group_status = pd.read_csv(samples_path)
            if not group_status.empty and "scenario_id" in group_status.columns:
                group_status = group_status.drop_duplicates(subset=["scenario_id"]).copy()
                group_status["status"] = "valid"
                group_status["scenario_dir"] = group_status["scenario_id"].map(
                    lambda value: str(runs_dir / str(value))
                )
                cost_status_frames.append(group_status)

    cost_status = (
        pd.concat(cost_status_frames, ignore_index=True, sort=False)
        if cost_status_frames
        else pd.DataFrame()
    )
    write_sensitivity_cost_summaries(
        cost_status,
        output_dir=base_out,
        run_search_root=base_out,
    )

    print(f"[DONE] targets: {targets_path}")
    for p in out_paths:
        print(f"[DONE] samples: {p}")


if __name__ == "__main__":
    main()
