# -*- coding: utf-8 -*-
"""
S5.3 panel data generator.

Runs the current main model over the yield / emission-factor grid and writes
long-form CSV outputs for the downstream panel plotting scripts.
"""
from __future__ import annotations

import copy
import gc
import os
from pathlib import Path
import re
import shutil
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config_paths import get_results_base
from market_balance_diagnostics import (
    read_market_balance_summary as _read_solver_market_balance_summary,
    validate_market_balance_gap as _validate_solver_market_balance_gap,
)
from model_run_status import (
    ResumeValidation,
    artifact_matches_validated_run,
    build_resume_fingerprint,
    validate_run_for_resume,
)
from S1_0_schema import ScenarioConfig
from S2_0_load_data import DataPaths, build_universe_from_dict_v3, load_population_wpp
from S3_6_scenarios import ScenarioEffect
from S4_0_main import (
    CFG,
    _apply_global_priority_forest_scenario,
    _extend_land_cover_to_future_years,
    build_run_baseline_cache,
    run_one_pipeline,
)
from S5_cost_summary_outputs import write_sensitivity_cost_summaries
from S5_1_1_sensitivity_mc_variable_effect import _validate_nonluc_fast_emissions


RUMINANT_CAP_COL = "ruminant_kcal_share_cap_pct"
RUMINANT_CAP_VALUES_KEY = "ruminant_kcal_share_cap_pct_values"
DEPRECATED_RUMINANT_CAP_COL = "ruminant_intake_change_pct"
DEPRECATED_RUMINANT_CAP_VALUES_KEY = "ruminant_intake_change_pct_values"

PALE_RESULT_COLS = [
    "pale_population",
    "pale_ag_output_kcal",
    "pale_land_cropland_ha",
    "pale_land_pasture_ha",
    "pale_land_cropland_pasture_ha",
    "pale_luc_emissions_gt_co2eq_yr",
    "pale_ag_production_emissions_gt_co2eq_yr",
    "pale_total_emissions_gt_co2eq_yr",
    "pale_a_per_p_kcal_cap_yr",
    "pale_l_per_a_ha_per_kcal",
    "pale_luc_intensity_gt_per_ha",
    "pale_ag_intensity_gt_per_kcal",
    "pale_land_source",
    "pale_ag_output_source",
    "pale_emissions_source",
    "pale_missing_reason",
]

_GLOBAL_POPULATION_BY_YEAR_CACHE: Optional[Dict[int, float]] = None


CONFIG = {
    "year": 2080,
    "output_dir": "",  # empty -> <NZF_OUTPUT_DIR>/Panel_Yield_EF
    "runs_subdir": "runs",
    "results_csv": "figure_panel_dataset_long.csv",
    "global_emissions_detail_csv": "figure_panel_global_emissions_detail_long.csv",
    "resume": False,
    "clear_existing_run_dirs_when_no_resume": True,
    "clear_existing_run_dirs_on_resume_retry": True,
    "save_per_run_dirs": True,
    "write_every_n_runs": 1,
    "max_runs": None,  # Optional debug limit; None runs all grid points.
    "stop_on_error": False,  # Useful for debugging: stop at the first failed scenario.
    # Default scan ranges.
    "emission_factor_change_pct_range": (-60, 60, 2),
    "yield_change_pct_range": (-60, 60, 2),
    "emission_factor_change_pct_values": None,
    "yield_change_pct_values": None,
    # With low_land_new, these values are absolute ruminant kcal-share caps:
    # 0, 1, ..., 100 map to Ruminate_Cap00, Ruminate_Cap01, ..., Ruminate_Cap100.
    "ruminant_kcal_share_cap_pct_values": [1, 13, 25],
    "forest_area_change_pct_values": [-30, 0, 30],
    "forest_area_allocation_mode": "global_priority",  # 'uniform' | 'global_priority'
    "domestic_supply_simulation_mode": "hard_equation",
    "solve": True,
    "use_fao_modules": True,
    "use_linear": True,
    "future_last_only": True,
    "fast_emis_only": True,
    "fast_emis_year": 2080,
    "resume_retry_statuses": [
        "TIME_LIMIT",
        "INTERRUPTED",
        "NUMERIC",
        "MEM_LIMIT",
        "WORK_LIMIT",
        "ITERATION_LIMIT",
        "NODE_LIMIT",
        "SOLUTION_LIMIT",
    ],
    # Batch production runs should not trigger IIS/verbose diagnostic reruns:
    # those reruns are memory-heavy and can OOM even when the normal run fits.
    "failed_diag_sample_n": 0,
    "failed_diag_runs_subdir": "runs_diag",
    # Skip unnecessary BASE cost references for batch scans.
    "override_cfg": {
        "cost_calculation_method": "off",
        "debug_level": 0,
        "domestic_supply_simulation_mode": "hard_equation",
        'supply_curtailment_enabled': False,
        "supply_curtailment_penalty": 1e10,
        "zero_price_shutdown_enabled": True,
        "demand_method": "nutrition",
        "max_slack_rate": 0.1,
        "max_shortage_slack_rate": 0.1,
        "max_excess_slack_rate": 0.1,
        "slack_penalty": 1e10,
        "armington_trade_slack_penalty": 1e10,
        "trade_cap_exempt_all_pairs": True,
        "forest_global_target_slack_enabled": True,
        "forest_global_target_slack_penalty": 1e10,
        "forest_global_target_slack_max_rate": None,
        "luc_accounting_path_mode": "linear_to_target",
        "luc_accounting_path_shape_power": 2.0,
        # Post-run diagnostic is a mass-based all-commodity gap, while the model
        # shortage cap is energy-based. Keep this as a quality screen, not a
        # duplicate feasibility constraint.
        "market_gap_max_rate": 0.08,
        "batch_mode": True,
        "linear_enable_infeasible_iis": False,
        "linear_enable_violation_iis": False,
        "linear_enable_output_diagnostics": False,
        "linear_enable_verbose_logging": False,
        "lightweight_diagnostic_outputs_enabled": True,
        "nutrition_profile_sheet": "low_land_new",
    },
    "batch": {
        "enabled": False,
        "total_batches": 200,
        "batch_index": 1,
        "assignment": "round_robin",  # 'round_robin' | 'contiguous'
        "batches_subdir": "batches",
    },
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_abs_path(raw_path: Path) -> Path:
    path = raw_path.expanduser()
    if not path.is_absolute():
        root = _project_root()
        resolved = (root / path).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Relative S5_3 output path escapes project root: {raw_path}. "
                "Use an absolute path instead."
            ) from exc
        return resolved
    return path.resolve()


def _default_output_base() -> Path:
    return _resolve_abs_path(Path(get_results_base()))


def _resolve_panel_output_root(raw_output_dir: object = "") -> Path:
    text = str(raw_output_dir or "").strip()
    if text:
        return _resolve_abs_path(Path(text))
    return _default_output_base() / "Panel_Yield_EF"


def _sync_panel_output_environment(root_output_dir: Path) -> None:
    root = _resolve_abs_path(root_output_dir)
    os.environ["PANEL_OUTPUT_DIR"] = str(root)
    os.environ["NZF_OUTPUT_DIR"] = str(root.parent)


def _panel_resume_fingerprint(
    *,
    scenario_id: str,
    forest_pct: float,
    ruminant_pct: float,
    yield_pct: float,
    ef_pct: float,
    cfg: Dict[str, object],
) -> str:
    return build_resume_fingerprint(
        {
            "schema": 1,
            "runner": "S5_3_1_sensitivity_panel_yield_ef",
            "scenario_id": str(scenario_id),
            "panel_point": {
                "forest_area_change_pct": float(forest_pct),
                "ruminant_kcal_share_cap_pct": float(ruminant_pct),
                "yield_change_pct": float(yield_pct),
                "emission_factor_change_pct": float(ef_pct),
            },
            "model_options": {
                "future_last_only": bool(cfg.get("future_last_only", True)),
                "use_fao_modules": bool(cfg.get("use_fao_modules", True)),
                "use_linear": bool(cfg.get("use_linear", True)),
                "fast_emis_only": bool(cfg.get("fast_emis_only", True)),
                "fast_emis_year": int(cfg.get("fast_emis_year", 2080) or 2080),
                "forest_area_allocation_mode": str(
                    cfg.get("forest_area_allocation_mode", "") or ""
                ),
                "override_cfg": dict(cfg.get("override_cfg") or {}),
            },
        }
    )


def _reset_output_file(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def _clear_existing_scenario_dir(scenario_dir: Path, runs_dir: Path) -> bool:
    if not scenario_dir.exists():
        return False
    scenario_resolved = scenario_dir.resolve()
    runs_resolved = runs_dir.resolve()
    try:
        scenario_resolved.relative_to(runs_resolved)
    except ValueError as exc:
        raise RuntimeError(
            f"Refuse to clear scenario dir outside runs_dir: {scenario_resolved}"
        ) from exc
    if scenario_resolved == runs_resolved or not scenario_resolved.name.startswith("FIG_PANEL_"):
        raise RuntimeError(f"Refuse to clear unexpected scenario dir: {scenario_resolved}")
    shutil.rmtree(scenario_resolved)
    return True


def _append_rows_csv(rows: List[Dict[str, object]], out_path: Path) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    header = (not out_path.exists()) or out_path.stat().st_size == 0
    df.to_csv(
        out_path,
        mode="a",
        header=header,
        index=False,
        encoding="utf-8-sig",
    )
    return int(len(df))


def _batch_tag(batch_index: int, batch_count: int) -> str:
    return f"batch_{int(batch_index):02d}_of_{int(batch_count):02d}"


def _resolve_batch_settings(cfg: Dict[str, object]) -> Dict[str, object]:
    batch_cfg = cfg.get("batch", {}) or {}
    enabled = bool(batch_cfg.get("enabled", False))
    batch_count = int(batch_cfg.get("total_batches", 1) or 1)
    batch_index = int(batch_cfg.get("batch_index", 1) or 1)
    if batch_count <= 0:
        raise ValueError("batch.total_batches must be a positive integer.")
    if batch_index <= 0 or batch_index > batch_count:
        raise ValueError("batch.batch_index must be within 1..total_batches.")
    assignment = str(batch_cfg.get("assignment", "round_robin") or "round_robin").strip().lower()
    if assignment not in {"round_robin", "contiguous"}:
        raise ValueError("batch.assignment must be 'round_robin' or 'contiguous'.")
    if not enabled:
        batch_count = 1
        batch_index = 1
    return {
        "enabled": enabled,
        "count": batch_count,
        "index": batch_index,
        "tag": _batch_tag(batch_index, batch_count),
        "assignment": assignment,
        "batches_subdir": str(batch_cfg.get("batches_subdir", "batches") or "batches"),
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


def _resolve_output_dir(root_output_dir: Path, batch_state: Dict[str, object]) -> Path:
    if bool(batch_state.get("enabled")):
        return root_output_dir / str(batch_state["batches_subdir"]) / str(batch_state["tag"])
    return root_output_dir


def _apply_batch_meta(row: Dict[str, object], batch_state: Dict[str, object]) -> Dict[str, object]:
    row["batch_index"] = int(batch_state["index"])
    row["batch_count"] = int(batch_state["count"])
    row["batch_tag"] = str(batch_state["tag"])
    return row


def _build_panel_tasks(
    *,
    forest_values: List[float],
    ruminant_values: List[float],
    yield_values: List[float],
    ef_values: List[float],
) -> List[Dict[str, object]]:
    tasks: List[Dict[str, object]] = []
    run_idx = 0
    for forest_idx, forest_pct in enumerate(forest_values, start=1):
        for ruminant_idx, ruminant_pct in enumerate(ruminant_values, start=1):
            for yield_pct in yield_values:
                for ef_pct in ef_values:
                    run_idx += 1
                    scenario_id = _scenario_id(
                        forest_pct=forest_pct,
                        ruminant_pct=ruminant_pct,
                        yield_pct=yield_pct,
                        ef_pct=ef_pct,
                    )
                    tasks.append(
                        {
                            "global_run_idx": run_idx,
                            "scenario_id": scenario_id,
                            "panel_row": forest_idx,
                            "panel_col": ruminant_idx,
                            "forest_pct": float(forest_pct),
                            "ruminant_pct": float(ruminant_pct),
                            "yield_pct": float(yield_pct),
                            "ef_pct": float(ef_pct),
                        }
                    )
    return tasks


def _read_fast_global_emissions_detail(
    scenario_dir: Path,
    *,
    validation: Optional[ResumeValidation] = None,
) -> pd.DataFrame:
    detail_path = scenario_dir / "Emis" / "emissions_fast_global_detail.csv"
    if not detail_path.exists():
        return pd.DataFrame()
    if validation is not None and not artifact_matches_validated_run(detail_path, validation):
        return pd.DataFrame()
    try:
        df = pd.read_csv(detail_path)
    except Exception:
        return pd.DataFrame()
    if validation is not None:
        required = {"run_id", "scenario_id"}
        if df.empty or not required.issubset(df.columns):
            return pd.DataFrame()
        run_ids = set(df["run_id"].dropna().astype(str).str.strip())
        scenario_ids = set(df["scenario_id"].dropna().astype(str).str.strip())
        if run_ids != {validation.run_id} or scenario_ids != {validation.scenario_id}:
            return pd.DataFrame()
    return df


def _read_forest_target_slack_summary(scenario_dir: Path, target_year: int) -> Dict[str, object]:
    default = {
        "forest_target_shortfall_ha": None,
        "forest_target_surplus_ha": None,
        "forest_target_abs_slack_ha": None,
        "forest_target_slack_rate": None,
        "forest_target_actual_ha": None,
        "forest_target_requested_ha": None,
    }
    slack_path = scenario_dir / "Log" / "forest_global_target_slack.csv"
    if not slack_path.exists():
        slack_path = scenario_dir / "DS" / "forest_global_target_slack.csv"
    if not slack_path.exists():
        return default
    try:
        df = pd.read_csv(slack_path)
    except Exception:
        return default
    if df.empty:
        return default
    year_col = "target_year" if "target_year" in df.columns else ("year" if "year" in df.columns else None)
    row_df = df
    if year_col is not None:
        years = pd.to_numeric(df[year_col], errors="coerce")
        matched = df.loc[years == int(target_year)]
        if not matched.empty:
            row_df = matched
    row = row_df.iloc[0]

    def _num(col: str) -> Optional[float]:
        if col not in row.index:
            return None
        try:
            val = float(row[col])
        except Exception:
            return None
        if not np.isfinite(val):
            return None
        return val

    return {
        "forest_target_shortfall_ha": _num("shortfall_ha"),
        "forest_target_surplus_ha": _num("surplus_ha"),
        "forest_target_abs_slack_ha": _num("abs_slack_ha"),
        "forest_target_slack_rate": _num("slack_rate"),
        "forest_target_actual_ha": _num("actual_forest_ha"),
        "forest_target_requested_ha": _num("target_forest_ha"),
    }


def _empty_pale_summary(reason: str = "") -> Dict[str, object]:
    out = {col: None for col in PALE_RESULT_COLS}
    out["pale_missing_reason"] = reason or None
    return out


def _sum_numeric(df: pd.DataFrame, col: str) -> Optional[float]:
    if df is None or df.empty or col not in df.columns:
        return None
    vals = pd.to_numeric(df[col], errors="coerce")
    if vals.notna().sum() <= 0:
        return None
    total = float(vals.fillna(0.0).sum())
    return total if np.isfinite(total) else None


def _global_population_for_year(target_year: int) -> Optional[float]:
    global _GLOBAL_POPULATION_BY_YEAR_CACHE
    if _GLOBAL_POPULATION_BY_YEAR_CACHE is None:
        try:
            paths = DataPaths()
            cfg = ScenarioConfig()
            universe = build_universe_from_dict_v3(paths.dict_v3_path, cfg)
            pop_map = load_population_wpp(paths.population_wpp_csv, universe)
            by_year: Dict[int, float] = {}
            for (_, year), value in (pop_map or {}).items():
                try:
                    year_i = int(year)
                    pop_f = float(value or 0.0)
                except Exception:
                    continue
                if np.isfinite(pop_f) and pop_f > 0:
                    by_year[year_i] = by_year.get(year_i, 0.0) + pop_f
            _GLOBAL_POPULATION_BY_YEAR_CACHE = by_year
        except Exception:
            _GLOBAL_POPULATION_BY_YEAR_CACHE = {}
    val = _GLOBAL_POPULATION_BY_YEAR_CACHE.get(int(target_year), None)
    if val is None or not np.isfinite(float(val)) or float(val) <= 0:
        return None
    return float(val)


def _first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _read_pale_land_summary(scenario_dir: Path, target_year: int) -> Dict[str, object]:
    ds_dir = scenario_dir / "DS"
    land_candidates = [
        (
            ds_dir / "luc_land_area_solver_period.csv",
            ["cropland_actual_ha", "cropland_ha"],
            ["pasture_actual_ha", "grassland_actual_ha", "grassland_ha", "pasture_ha"],
            "DS/luc_land_area_solver_period.csv",
        ),
        (
            ds_dir / "country_year_summary.csv",
            ["cropland_area_ha_luc_actual", "cropland_area_ha"],
            ["pasture_area_ha_luc_actual", "pasture_area_ha"],
            "DS/country_year_summary.csv",
        ),
    ]
    for path, crop_cols, pasture_cols, source in land_candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if df.empty or "year" not in df.columns:
            continue
        years = pd.to_numeric(df["year"], errors="coerce")
        sub = df.loc[years == int(target_year)].copy()
        if sub.empty:
            continue
        crop_col = _first_existing_col(sub, crop_cols)
        pasture_col = _first_existing_col(sub, pasture_cols)
        crop = _sum_numeric(sub, crop_col) if crop_col else None
        pasture = _sum_numeric(sub, pasture_col) if pasture_col else None
        if crop is None and pasture is None:
            continue
        crop_val = float(crop or 0.0)
        pasture_val = float(pasture or 0.0)
        return {
            "pale_land_cropland_ha": crop_val,
            "pale_land_pasture_ha": pasture_val,
            "pale_land_cropland_pasture_ha": crop_val + pasture_val,
            "pale_land_source": source,
        }
    return _read_pale_land_summary_fast_diag(scenario_dir, target_year)


def _read_pale_land_summary_fast_diag(scenario_dir: Path, target_year: int) -> Dict[str, object]:
    path = scenario_dir / "Diagnostics" / "crop_pasture_land_balance.csv"
    if not path.exists():
        return {
            "pale_land_cropland_ha": None,
            "pale_land_pasture_ha": None,
            "pale_land_cropland_pasture_ha": None,
            "pale_land_source": None,
        }
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return {
            "pale_land_cropland_ha": None,
            "pale_land_pasture_ha": None,
            "pale_land_cropland_pasture_ha": None,
            "pale_land_source": None,
        }
    if df.empty or "year" not in df.columns:
        return {
            "pale_land_cropland_ha": None,
            "pale_land_pasture_ha": None,
            "pale_land_cropland_pasture_ha": None,
            "pale_land_source": None,
        }
    years = pd.to_numeric(df["year"], errors="coerce")
    sub = df.loc[years == int(target_year)].copy()
    if sub.empty:
        return {
            "pale_land_cropland_ha": None,
            "pale_land_pasture_ha": None,
            "pale_land_cropland_pasture_ha": None,
            "pale_land_source": None,
        }

    source_series = sub.get("source", pd.Series("", index=sub.index)).astype(str)
    candidates = [
        (
            sub.loc[source_series.eq("luc_land_area_LUH2Based_period")],
            ["crop_area_ha", "cropland_ha", "cropland_area_ha"],
            ["pasture_area_ha", "grassland_ha", "grassland_area_ha", "pasture_ha"],
            "Diagnostics/crop_pasture_land_balance.csv:luc_land_area_LUH2Based_period",
        ),
        (
            sub.loc[source_series.eq("luc_land_area_DS_period")],
            ["crop_area_need_ha", "crop_area_ha", "cropland_ha"],
            ["grass_area_need_ha", "pasture_area_ha", "grassland_ha"],
            "Diagnostics/crop_pasture_land_balance.csv:luc_land_area_DS_period",
        ),
        (
            sub,
            ["crop_area_ha", "cropland_ha", "cropland_area_ha"],
            ["pasture_area_ha", "grassland_ha", "grassland_area_ha", "pasture_ha"],
            "Diagnostics/crop_pasture_land_balance.csv:actual_land",
        ),
        (
            sub,
            ["crop_area_need_ha"],
            ["grass_area_need_ha"],
            "Diagnostics/crop_pasture_land_balance.csv:demand_land",
        ),
    ]
    for candidate_df, crop_cols, pasture_cols, source in candidates:
        if candidate_df.empty:
            continue
        crop_col = _first_existing_col(candidate_df, crop_cols)
        pasture_col = _first_existing_col(candidate_df, pasture_cols)
        crop = _sum_numeric(candidate_df, crop_col) if crop_col else None
        pasture = _sum_numeric(candidate_df, pasture_col) if pasture_col else None
        if crop is None and pasture is None:
            continue
        crop_val = float(crop or 0.0)
        pasture_val = float(pasture or 0.0)
        if crop_val + pasture_val <= 0:
            continue
        return {
            "pale_land_cropland_ha": crop_val,
            "pale_land_pasture_ha": pasture_val,
            "pale_land_cropland_pasture_ha": crop_val + pasture_val,
            "pale_land_source": source,
        }
    return {
        "pale_land_cropland_ha": None,
        "pale_land_pasture_ha": None,
        "pale_land_cropland_pasture_ha": None,
        "pale_land_source": None,
    }


def _read_pale_population_output_summary(scenario_dir: Path, target_year: int) -> Dict[str, object]:
    ds_dir = scenario_dir / "DS"
    candidates = [
        (ds_dir / "nutrition_per_capita.csv", "energy_total_kcal", "DS/nutrition_per_capita.csv"),
        (ds_dir / "country_year_summary.csv", "energy_total_kcal", "DS/country_year_summary.csv"),
    ]
    for path, energy_col, source in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if df.empty or "year" not in df.columns:
            continue
        years = pd.to_numeric(df["year"], errors="coerce")
        sub = df.loc[years == int(target_year)].copy()
        if sub.empty:
            continue
        population = _sum_numeric(sub, "population")
        ag_output_kcal = _sum_numeric(sub, energy_col)
        if population is None and ag_output_kcal is None:
            continue
        return {
            "pale_population": population,
            "pale_ag_output_kcal": ag_output_kcal,
            "pale_ag_output_source": source,
        }
    fast_path = scenario_dir / "Diagnostics" / "realized_ruminant_intake_share.csv"
    population = _global_population_for_year(target_year)
    ag_output_kcal = None
    source = None
    if fast_path.exists():
        try:
            df = pd.read_csv(fast_path, low_memory=False)
        except Exception:
            df = pd.DataFrame()
        if not df.empty and "year" in df.columns:
            years = pd.to_numeric(df["year"], errors="coerce")
            sub = df.loc[years == int(target_year)].copy()
            if "scope" in sub.columns:
                global_sub = sub.loc[sub["scope"].astype(str).str.strip().eq("global_summary")].copy()
                if not global_sub.empty:
                    sub = global_sub
            for col in ["demand_kcal", "demand_intake"]:
                if col not in sub.columns:
                    continue
                vals = pd.to_numeric(sub[col], errors="coerce")
                vals = vals[np.isfinite(vals) & (vals > 0)]
                if not vals.empty:
                    ag_output_kcal = float(vals.iloc[0])
                    source = f"Diagnostics/realized_ruminant_intake_share.csv:{col}"
                    break
    if population is not None or ag_output_kcal is not None:
        return {
            "pale_population": population,
            "pale_ag_output_kcal": ag_output_kcal,
            "pale_ag_output_source": source,
        }
    return {
        "pale_population": None,
        "pale_ag_output_kcal": None,
        "pale_ag_output_source": None,
    }


def _read_pale_emissions_summary(
    scenario_dir: Path,
    target_year: int,
    detail_df: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    df = detail_df
    if df is None or df.empty:
        df = _read_fast_global_emissions_detail(scenario_dir)
    if df is None or df.empty or "co2eq_kt" not in df.columns:
        return {
            "pale_luc_emissions_gt_co2eq_yr": None,
            "pale_ag_production_emissions_gt_co2eq_yr": None,
            "pale_total_emissions_gt_co2eq_yr": None,
            "pale_emissions_source": None,
        }
    work = df.copy()
    if "year" in work.columns:
        years = pd.to_numeric(work["year"], errors="coerce")
        work = work.loc[years == int(target_year)].copy()
    if work.empty:
        return {
            "pale_luc_emissions_gt_co2eq_yr": None,
            "pale_ag_production_emissions_gt_co2eq_yr": None,
            "pale_total_emissions_gt_co2eq_yr": None,
            "pale_emissions_source": None,
        }
    if "row_type" in work.columns:
        row_type = work["row_type"].fillna("").astype(str).str.strip().str.lower()
        filtered = work.loc[row_type.eq("co2eq_summary")].copy()
        if not filtered.empty:
            work = filtered
    elif "GHG" in work.columns:
        ghg = work["GHG"].fillna("").astype(str).str.strip().str.lower()
        filtered = work.loc[ghg.eq("co2eq")].copy()
        if not filtered.empty:
            work = filtered

    module = (
        work.get("source_module", pd.Series("", index=work.index))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    co2eq_gt = pd.to_numeric(work["co2eq_kt"], errors="coerce").fillna(0.0) / 1e6
    luc = float(co2eq_gt.loc[module.eq("LUC")].sum())
    ag = float(co2eq_gt.loc[~module.eq("LUC")].sum())
    return {
        "pale_luc_emissions_gt_co2eq_yr": luc,
        "pale_ag_production_emissions_gt_co2eq_yr": ag,
        "pale_total_emissions_gt_co2eq_yr": luc + ag,
        "pale_emissions_source": "Emis/emissions_fast_global_detail.csv",
    }


def _read_pale_summary(
    scenario_dir: Path,
    target_year: int,
    detail_df: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    out = _empty_pale_summary()
    out.update(_read_pale_land_summary(scenario_dir, target_year))
    out.update(_read_pale_population_output_summary(scenario_dir, target_year))
    out.update(_read_pale_emissions_summary(scenario_dir, target_year, detail_df=detail_df))

    population = out.get("pale_population")
    ag_output = out.get("pale_ag_output_kcal")
    land = out.get("pale_land_cropland_pasture_ha")
    luc_emis = out.get("pale_luc_emissions_gt_co2eq_yr")
    ag_emis = out.get("pale_ag_production_emissions_gt_co2eq_yr")

    missing = []
    if population is None or not np.isfinite(float(population)) or float(population) <= 0:
        missing.append("population")
    if ag_output is None or not np.isfinite(float(ag_output)) or float(ag_output) <= 0:
        missing.append("ag_output_kcal")
    if land is None or not np.isfinite(float(land)) or float(land) <= 0:
        missing.append("cropland_plus_pasture")
    if luc_emis is None:
        missing.append("luc_emissions")
    if ag_emis is None:
        missing.append("ag_production_emissions")

    if not missing:
        population_f = float(population)
        ag_output_f = float(ag_output)
        land_f = float(land)
        out["pale_a_per_p_kcal_cap_yr"] = ag_output_f / population_f
        out["pale_l_per_a_ha_per_kcal"] = land_f / ag_output_f
        out["pale_luc_intensity_gt_per_ha"] = float(luc_emis) / land_f
        out["pale_ag_intensity_gt_per_kcal"] = float(ag_emis) / ag_output_f
        out["pale_missing_reason"] = None
    else:
        out["pale_missing_reason"] = ";".join(missing)
    return out


def _append_global_emissions_detail(
    detail_rows: List[Dict[str, object]],
    detail_df: pd.DataFrame,
    row_meta: Dict[str, object],
) -> None:
    if detail_df is None or detail_df.empty:
        return
    keep_cols = [
        "year",
        "source_module",
        "Process",
        "Item",
        "GHG",
        "emissions_kt",
        "co2eq_kt",
        "row_type",
        "baseline_year",
        "Y2020_emissions_kt",
        "Y2020_co2eq_kt",
        "delta_vs_Y2020_co2eq_kt",
        "reduction_vs_Y2020_pct",
    ]
    use_cols = [c for c in keep_cols if c in detail_df.columns]
    if not use_cols:
        return
    detail_use = detail_df[use_cols].copy()
    meta_cols = {
        "scenario_id": row_meta.get("scenario_id"),
        "panel_row": row_meta.get("panel_row"),
        "panel_col": row_meta.get("panel_col"),
        "forest_area_change_pct": row_meta.get("forest_area_change_pct"),
        RUMINANT_CAP_COL: row_meta.get(RUMINANT_CAP_COL, row_meta.get(DEPRECATED_RUMINANT_CAP_COL)),
        "yield_change_pct": row_meta.get("yield_change_pct"),
        "emission_factor_change_pct": row_meta.get("emission_factor_change_pct"),
        "target_year": row_meta.get("target_year"),
        "run_status": row_meta.get("run_status"),
        "scenario_dir": row_meta.get("scenario_dir"),
    }
    for key, value in meta_cols.items():
        detail_use[key] = value
    ordered = list(meta_cols.keys()) + use_cols
    detail_rows.extend(detail_use[ordered].to_dict("records"))


def _pct_to_rate(pct: float) -> float:
    return float(pct) / 100.0


def _pct_to_multiplier(pct: float) -> float:
    return 1.0 + _pct_to_rate(pct)


def _active_nutrition_profile_sheet(cfg: Optional[Dict[str, object]] = None) -> str:
    cfg_use = cfg or CONFIG
    override = cfg_use.get("override_cfg", {}) or {}
    return str(
        override.get("nutrition_profile_sheet", CFG.get("nutrition_profile_sheet", "low_land_new"))
        or "low_land_new"
    ).strip().lower()


def _uses_absolute_ruminant_cap(cfg: Optional[Dict[str, object]] = None) -> bool:
    return _active_nutrition_profile_sheet(cfg) == "low_land_new"


def _ruminant_cap_values_from_config(cfg: Dict[str, object]) -> List[float]:
    raw_values = cfg.get(RUMINANT_CAP_VALUES_KEY)
    if not raw_values:
        raw_values = cfg.get(DEPRECATED_RUMINANT_CAP_VALUES_KEY)
    return [float(v) for v in raw_values or []]


def _fmt_pct(pct: float) -> str:
    val = int(round(float(pct)))
    sign = "p" if val >= 0 else "m"
    return f"{sign}{abs(val)}"


def _expand_axis_values(cfg: Dict[str, object], value_key: str, range_key: str) -> List[float]:
    raw_values = cfg.get(value_key)
    if isinstance(raw_values, list) and raw_values:
        return [float(v) for v in raw_values]

    raw_range = cfg.get(range_key)
    if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 3:
        raise ValueError(f"{range_key} must be (min, max, step)")

    start, stop, step = raw_range
    start = float(start)
    stop = float(stop)
    step = float(step)
    if step <= 0:
        raise ValueError(f"{range_key} step must be > 0")
    if stop < start:
        raise ValueError(f"{range_key} max must be >= min")

    values: List[float] = []
    cur = start
    tol = abs(step) * 1e-9 + 1e-9
    while cur <= stop + tol:
        rounded = round(cur, 10)
        if abs(rounded - round(rounded)) < 1e-9:
            rounded = float(int(round(rounded)))
        values.append(float(rounded))
        cur += step
    return values


def _scenario_id(
    *,
    forest_pct: float,
    ruminant_pct: float,
    yield_pct: float,
    ef_pct: float,
) -> str:
    return (
        "FIG_PANEL"
        f"_F{_fmt_pct(forest_pct)}"
        f"_R{_fmt_pct(ruminant_pct)}"
        f"_Y{_fmt_pct(yield_pct)}"
        f"_E{_fmt_pct(ef_pct)}"
    )


def _make_effect(
    *,
    universe,
    scenario_id: str,
    kind: str,
    value_2080: Any,
    unit: str = "rate",
) -> ScenarioEffect:
    eff = ScenarioEffect(
        scenario_id=scenario_id,
        kind=kind,
        unit=unit,
        value_2080=value_2080,
        country_sel="All",
        commodity_sel="All",
        process_sel="All",
    )
    eff.countries = list(universe.countries)
    eff.commodities = list(universe.commodities)
    eff.processes = list(universe.processes)
    eff.ghg_sel = "All"
    return eff


def _build_effects(
    *,
    universe,
    scenario_id: str,
    yield_change_pct: float,
    ef_change_pct: float,
    ruminant_change_pct: float,
) -> List[ScenarioEffect]:
    effects = [
        _make_effect(
            universe=universe,
            scenario_id=scenario_id,
            kind="yield_rate",
            value_2080=_pct_to_rate(yield_change_pct),
        ),
        _make_effect(
            universe=universe,
            scenario_id=scenario_id,
            kind="emission_factor",
            value_2080=_pct_to_rate(ef_change_pct),
        ),
    ]
    ruminant_pct = float(ruminant_change_pct)
    profile_sheet = str(
        (CONFIG.get("override_cfg", {}) or {}).get("nutrition_profile_sheet", CFG.get("nutrition_profile_sheet", "low_land_new"))
        or "low_land_new"
    ).strip().lower()
    if profile_sheet == "low_land_new":
        rumi_pct = int(round(max(0.0, min(100.0, ruminant_pct))))
        profile_name = f"Ruminate_Cap{rumi_pct:02d}"
        effects.append(
            _make_effect(
                universe=universe,
                scenario_id=scenario_id,
                kind="nutrition_profile",
                unit="profile",
                value_2080=profile_name,
            )
        )
    elif ruminant_pct != 0:
        rumi_pct = int(round(abs(ruminant_pct)))
        profile_name = (
            f"Ruminate_Cap{rumi_pct:02d}"
            if ruminant_pct < 0
            else f"Ruminate_Cap-{rumi_pct:02d}"
        )
        effects.append(
            _make_effect(
                universe=universe,
                scenario_id=scenario_id,
                kind="nutrition_profile",
                unit="profile",
                value_2080=profile_name,
            )
        )
    return effects


def _build_forest_scenario_params(cfg: Dict[str, Any], forest_mult: float) -> Dict[str, Any]:
    forest_target_enabled = bool(cfg.get("forest_area_change_pct_values") or [])
    return {
        "forest_area_multiplier": forest_mult,
        "forest_area_allocation_mode": str(
            cfg.get("forest_area_allocation_mode", "global_priority") or "global_priority"
        ),
        "forest_global_target_enabled": forest_target_enabled,
    }


def _expected_forest_target_ha(
    *,
    land_cover_base_df: Optional[pd.DataFrame],
    active_years: List[int],
    target_year: int,
    forest_mult: float,
    allocation_mode: str,
    base_year: int = 2020,
) -> Optional[float]:
    if land_cover_base_df is None or land_cover_base_df.empty:
        return None
    required = {"land_use", "area_ha", "year"}
    if not required.issubset(land_cover_base_df.columns):
        return None
    years = sorted({int(y) for y in active_years if y is not None} | {int(target_year)})
    work = land_cover_base_df.copy()
    work = _extend_land_cover_to_future_years(work, years, base_year=int(base_year))

    try:
        mult = float(forest_mult)
    except Exception:
        return None
    if not np.isfinite(mult) or mult <= 0:
        return None

    mode = str(allocation_mode or "global_priority").strip().lower()
    if abs(mult - 1.0) > 1e-12:
        land_use_norm = work["land_use"].fillna("").astype(str).str.strip().str.lower()
        year_vals = pd.to_numeric(work["year"], errors="coerce")
        future_forest = land_use_norm.eq("forest") & year_vals.gt(int(base_year))
        if mode in {"global", "global_priority", "priority_global", "priority"}:
            work = _apply_global_priority_forest_scenario(
                work,
                mult,
                base_year=int(base_year),
            )
        elif int(future_forest.sum()) > 0:
            work = work.copy()
            work.loc[future_forest, "area_ha"] = (
                pd.to_numeric(work.loc[future_forest, "area_ha"], errors="coerce").fillna(0.0)
                * mult
            )

    land_use_norm = work["land_use"].fillna("").astype(str).str.strip().str.lower()
    year_vals = pd.to_numeric(work["year"], errors="coerce")
    target_mask = land_use_norm.eq("forest") & year_vals.eq(int(target_year))
    if int(target_mask.sum()) <= 0:
        return None
    return float(pd.to_numeric(work.loc[target_mask, "area_ha"], errors="coerce").fillna(0.0).sum())


def _build_expected_forest_targets(
    *,
    shared_run_cache: Dict[str, Any],
    forest_values: List[float],
    cfg: Dict[str, Any],
    target_year: int,
    base_year: int = 2020,
) -> Dict[float, float]:
    land_cover_base_df = shared_run_cache.get("land_cover_base_df")
    active_years = [int(y) for y in (shared_run_cache.get("active_years") or [])]
    allocation_mode = str(cfg.get("forest_area_allocation_mode", "global_priority") or "global_priority")
    expected: Dict[float, float] = {}
    for forest_pct in forest_values:
        forest_mult = _pct_to_multiplier(float(forest_pct))
        val = _expected_forest_target_ha(
            land_cover_base_df=land_cover_base_df,
            active_years=active_years,
            target_year=int(target_year),
            forest_mult=forest_mult,
            allocation_mode=allocation_mode,
            base_year=int(base_year),
        )
        if val is not None and np.isfinite(val):
            expected[float(forest_mult)] = float(val)
    return expected


def _validate_forest_target_consistency(
    row: Dict[str, object],
    expected_targets_by_multiplier: Dict[float, float],
    *,
    rel_tol: float = 1e-6,
    abs_tol_ha: float = 10_000.0,
) -> Optional[str]:
    if not expected_targets_by_multiplier:
        return None
    try:
        forest_mult = float(row.get("forest_area_multiplier"))
    except Exception:
        return None
    if not np.isfinite(forest_mult):
        return None
    expected = expected_targets_by_multiplier.get(float(forest_mult))
    if expected is None:
        for key, val in expected_targets_by_multiplier.items():
            if abs(float(key) - forest_mult) <= 1e-12:
                expected = val
                break
    if expected is None:
        return None
    try:
        requested = float(row.get("forest_target_requested_ha"))
    except Exception:
        return None
    if not np.isfinite(requested):
        return None
    diff = requested - float(expected)
    tol = max(float(abs_tol_ha), abs(float(expected)) * float(rel_tol))
    if abs(diff) <= tol:
        return None
    pct = row.get("forest_area_change_pct")
    return (
        f"forest target mismatch for forest={pct}% multiplier={forest_mult:.6g}: "
        f"run target={requested:,.0f} ha, expected={float(expected):,.0f} ha, "
        f"diff={diff:,.0f} ha, tolerance={tol:,.0f} ha"
    )


def _read_fast_emissions_gt(
    scenario_dir: Path,
    *,
    validation: Optional[ResumeValidation] = None,
) -> Optional[float]:
    fast_path = scenario_dir / "Emis" / "emissions_fast_summary.csv"
    if not fast_path.exists():
        return None
    if validation is not None and not artifact_matches_validated_run(fast_path, validation):
        return None
    try:
        df = pd.read_csv(fast_path)
    except Exception:
        return None
    if df.empty:
        return None
    if validation is not None:
        required = {"run_id", "scenario_id"}
        if not required.issubset(df.columns):
            return None
        run_ids = set(df["run_id"].dropna().astype(str).str.strip())
        scenario_ids = set(df["scenario_id"].dropna().astype(str).str.strip())
        if run_ids != {validation.run_id} or scenario_ids != {validation.scenario_id}:
            return None
    if "total_co2eq_gt" in df.columns:
        val = pd.to_numeric(df["total_co2eq_gt"], errors="coerce").dropna()
        if not val.empty:
            return float(val.iloc[0])
    if "total_co2eq_kt" in df.columns:
        val = pd.to_numeric(df["total_co2eq_kt"], errors="coerce").dropna()
        if not val.empty:
            return float(val.iloc[0]) * 1e-6
    return None


def _read_market_gap_summary(scenario_dir: Path) -> Dict[str, object]:
    return _read_solver_market_balance_summary(scenario_dir)


def _validate_market_balance_gap(scenario_dir: Path, *, max_gap_rate: float = 0.01) -> Optional[str]:
    return _validate_solver_market_balance_gap(
        scenario_dir,
        max_gap_rate=max_gap_rate,
    )


def _read_model_log_diagnostics(scenario_dir: Path) -> Dict[str, object]:
    log_path = scenario_dir / "Log" / "model.log"
    diag: Dict[str, object] = {
        "log_exists": False,
        "model_status_code": None,
        "model_status_text": "",
        "iis_summary": "",
        "run_status_hint": "",
        "error_type_hint": "",
        "error_message_hint": "",
    }
    if not log_path.exists():
        return diag

    diag["log_exists"] = True
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            text = log_path.read_text(encoding="gbk", errors="ignore")
        except Exception:
            return diag

    status_matches = re.findall(r"status=(\d+)", text)
    status_code = int(status_matches[-1]) if status_matches else None
    diag["model_status_code"] = status_code

    status_map = {
        1: "LOADED",
        2: "OPTIMAL",
        3: "INFEASIBLE",
        4: "INF_OR_UNBD",
        5: "UNBOUNDED",
        7: "ITERATION_LIMIT",
        8: "NODE_LIMIT",
        9: "TIME_LIMIT",
        10: "SOLUTION_LIMIT",
        11: "INTERRUPTED",
        12: "NUMERIC",
        13: "SUBOPTIMAL",
        16: "WORK_LIMIT",
        17: "MEM_LIMIT",
    }

    infeasible_hit = bool(re.search(r"\bInfeasible model\b", text, flags=re.IGNORECASE))
    iis_match = re.search(
        r"IIS computed:\s*(\d+)\s+constraints?\s+and\s+(\d+)\s+bounds?",
        text,
        flags=re.IGNORECASE,
    )
    iis_summary = ""
    if iis_match:
        iis_summary = (
            f"IIS computed: {iis_match.group(1)} constraints and {iis_match.group(2)} bounds"
        )
    diag["iis_summary"] = iis_summary

    if infeasible_hit or status_code == 3:
        diag["run_status_hint"] = "infeasible"
        diag["model_status_text"] = "Infeasible model"
        diag["error_type_hint"] = "InfeasibleModel"
        parts = []
        if status_code is not None:
            parts.append(f"status={status_code}")
        parts.append("Infeasible model")
        if iis_summary:
            parts.append(iis_summary)
        diag["error_message_hint"] = "; ".join(parts)
        return diag

    if status_code is not None and status_code != 2:
        status_text = status_map.get(status_code, f"STATUS_{status_code}")
        diag["run_status_hint"] = "nonoptimal"
        diag["model_status_text"] = status_text
        diag["error_type_hint"] = "NonOptimalModel"
        parts = [f"status={status_code}", status_text]
        if iis_summary:
            parts.append(iis_summary)
        diag["error_message_hint"] = "; ".join(parts)
        return diag

    if status_code == 2:
        diag["model_status_text"] = "OPTIMAL"
    elif iis_summary:
        diag["model_status_text"] = "IIS_PRESENT"
    return diag


def _apply_log_diagnostics(row: Dict[str, object], scenario_dir: Path) -> Dict[str, object]:
    diag = _read_model_log_diagnostics(scenario_dir)
    row["model_status_code"] = diag.get("model_status_code")
    row["model_status_text"] = diag.get("model_status_text", "")
    row["iis_summary"] = diag.get("iis_summary", "")
    return diag


def _apply_terminal_model_status(row: Dict[str, object], log_diag: Dict[str, object]) -> bool:
    status_hint = str(log_diag.get("run_status_hint") or "")
    if status_hint not in {"infeasible", "nonoptimal"}:
        return False
    row["run_status"] = status_hint
    row["afolu_emissions_gt_co2eq_yr"] = None
    row["error_type"] = str(log_diag.get("error_type_hint") or "ModelStatus")
    row["error_message"] = str(log_diag.get("error_message_hint") or row.get("model_status_text") or "")
    return True


def _is_retryable_resume_status(cfg: Dict[str, object], log_diag: Dict[str, object]) -> bool:
    status_text = str(log_diag.get("model_status_text") or "").strip().upper()
    status_code = log_diag.get("model_status_code")
    retry_list = cfg.get("resume_retry_statuses") or []
    retry_set = {str(item).strip().upper() for item in retry_list if str(item).strip()}
    if status_text and status_text in retry_set:
        return True
    retry_codes = {7, 8, 9, 10, 11, 12, 16, 17}
    return status_code in retry_codes


def _rerun_failed_point_with_diagnostics(
    *,
    cfg: Dict[str, object],
    paths: DataPaths,
    shared_cfg: ScenarioConfig,
    shared_universe,
    shared_run_cache: Dict[str, Any],
    diag_runs_dir: Path,
    scenario_id: str,
    scenario_params: Dict[str, object],
    effects: List[ScenarioEffect],
) -> Optional[str]:
    diag_overrides = {
        "batch_mode": False,
        "linear_enable_infeasible_iis": True,
        "linear_enable_violation_iis": True,
        "linear_enable_output_diagnostics": True,
        "linear_enable_verbose_logging": True,
    }
    backup: Dict[str, object] = {}
    for key, value in diag_overrides.items():
        backup[key] = CFG.get(key)
        CFG[key] = value
    try:
        outdir = run_one_pipeline(
            paths,
            pre_macc_e0=False,
            scenario_id=scenario_id,
            scenario_params=scenario_params,
            scenario_effects=effects,
            solve=bool(cfg.get("solve", True)),
            use_fao_modules=bool(cfg.get("use_fao_modules", True)),
            save_root=str(diag_runs_dir),
            future_last_only=bool(cfg.get("future_last_only", True)),
            use_linear=bool(cfg.get("use_linear", True)),
            fast_emis_only=bool(cfg.get("fast_emis_only", True)),
            fast_emis_year=int(cfg.get("fast_emis_year", 2080) or 2080),
            prebuilt_config=shared_cfg,
            prebuilt_universe=shared_universe,
            prebuilt_run_cache=shared_run_cache,
        )
        return str(outdir)
    finally:
        for key, value in backup.items():
            CFG[key] = value
        gc.collect()


def main() -> None:
    cfg = copy.deepcopy(CONFIG)
    root_output_dir = _resolve_panel_output_root(cfg.get("output_dir", ""))
    _sync_panel_output_environment(root_output_dir)
    batch_state = _resolve_batch_settings(cfg)
    output_dir = _resolve_output_dir(root_output_dir, batch_state)
    runs_dir = output_dir / str(cfg.get("runs_subdir") or "runs")
    out_path = output_dir / str(cfg.get("results_csv") or "figure_panel_dataset_long.csv")
    detail_out_path = output_dir / str(
        cfg.get("global_emissions_detail_csv") or "figure_panel_global_emissions_detail_long.csv"
    )
    meta_path = output_dir / "run_meta.csv"
    _ensure_dir(output_dir)
    if cfg.get("save_per_run_dirs", True):
        _ensure_dir(runs_dir)
    _reset_output_file(out_path)
    _reset_output_file(detail_out_path)
    _reset_output_file(meta_path)

    paths = DataPaths()
    shared_cfg = ScenarioConfig()
    shared_universe = build_universe_from_dict_v3(paths.dict_v3_path, shared_cfg)

    ef_values = _expand_axis_values(
        cfg,
        "emission_factor_change_pct_values",
        "emission_factor_change_pct_range",
    )
    yield_values = _expand_axis_values(
        cfg,
        "yield_change_pct_values",
        "yield_change_pct_range",
    )
    ruminant_values = _ruminant_cap_values_from_config(cfg)
    forest_values = [float(v) for v in cfg.get("forest_area_change_pct_values") or []]
    if not ef_values or not yield_values or not ruminant_values or not forest_values:
        raise ValueError("CONFIG grid value lists must not be empty")
    if forest_values:
        cfg.setdefault("override_cfg", {})["forest_global_target_enabled"] = True
    tasks = _build_panel_tasks(
        forest_values=forest_values,
        ruminant_values=ruminant_values,
        yield_values=yield_values,
        ef_values=ef_values,
    )
    total_runs = len(tasks)
    task_indices = _select_batch_item_indices(
        total_items=total_runs,
        batch_count=int(batch_state["count"]) if bool(batch_state.get("enabled")) else 1,
        batch_index=int(batch_state["index"]) if bool(batch_state.get("enabled")) else 1,
        assignment=str(batch_state["assignment"]),
    )
    assigned_runs = len(task_indices)
    print(f"[FIG-PANEL] root_output_dir: {root_output_dir}")
    print(f"[FIG-PANEL] NZF_OUTPUT_DIR: {os.environ.get('NZF_OUTPUT_DIR', '')}")
    print(f"[FIG-PANEL] PANEL_OUTPUT_DIR: {os.environ.get('PANEL_OUTPUT_DIR', '')}")
    if bool(batch_state.get("enabled")):
        print(
            f"[FIG-PANEL] batch={batch_state['tag']} assignment={batch_state['assignment']} "
            f"assigned_runs={assigned_runs}/{total_runs}"
        )
    print(f"[FIG-PANEL] preparing {total_runs} scenario grid points")
    print(f"[FIG-PANEL] output_dir: {output_dir}")
    print(f"[FIG-PANEL] EF axis: {len(ef_values)} points, {ef_values[0]} -> {ef_values[-1]}")
    print(f"[FIG-PANEL] Yield axis: {len(yield_values)} points, {yield_values[0]} -> {yield_values[-1]}")
    print(
        f"[FIG-PANEL] shared cache: countries={len(shared_universe.countries)}, "
        f"commodities={len(shared_universe.commodities)}, years={len(shared_universe.years)}"
    )

    if bool(cfg.get("resume", False)):
        print("[FIG-PANEL] resume enabled: rebuild CSV outputs and reuse completed run directories")
    else:
        print("[FIG-PANEL] resume disabled: rerun assigned scenarios")
        print("[FIG-PANEL] batch result CSV/detail CSV/run_meta are reset at job start")
        if bool(cfg.get("clear_existing_run_dirs_when_no_resume", True)):
            print("[FIG-PANEL] existing per-scenario run dirs will be cleared before rerun")

    max_runs = cfg.get("max_runs")
    if max_runs is not None:
        raw_max_runs = max_runs
        max_runs = int(max_runs)
        if max_runs <= 0:
            raise ValueError(f"max_runs must be a positive integer or None; got {raw_max_runs!r}")
        print(f"[FIG-PANEL] debug limit: run at most {max_runs} scenario grid points")

    if max_runs is not None:
        task_indices = task_indices[:max_runs]
        assigned_runs = len(task_indices)
        max_runs = None

    write_every = int(cfg.get("write_every_n_runs", 1) or 1)
    if write_every <= 0:
        write_every = 1

    cfg_backup: Dict[str, object] = {}
    for key, value in (cfg.get("override_cfg") or {}).items():
        cfg_backup[key] = CFG.get(key)
        CFG[key] = value

    shared_run_cache = build_run_baseline_cache(
        paths,
        shared_cfg,
        shared_universe,
        future_last_only=bool(cfg.get("future_last_only", True)),
    )
    expected_forest_targets_by_multiplier = _build_expected_forest_targets(
        shared_run_cache=shared_run_cache,
        forest_values=forest_values,
        cfg=cfg,
        target_year=int(cfg.get("fast_emis_year", 2080) or 2080),
        base_year=int(getattr(shared_cfg, "years_hist_end", 2020) or 2020),
    )
    print(
        f"[FIG-PANEL] baseline cache: nodes={len(shared_run_cache.get('node_blueprint') or [])}, "
        f"years={len(shared_run_cache.get('active_years') or [])}"
    )
    if expected_forest_targets_by_multiplier:
        expected_parts = [
            f"{mult:.6g}->{target:,.0f} ha"
            for mult, target in sorted(expected_forest_targets_by_multiplier.items())
        ]
        print(f"[FIG-PANEL] expected forest targets: {', '.join(expected_parts)}")
    failed_diag_budget = max(0, int(cfg.get("failed_diag_sample_n", 0) or 0))
    failed_diag_used = 0
    diag_runs_dir = output_dir / str(cfg.get("failed_diag_runs_subdir") or "runs_diag")
    if failed_diag_budget > 0:
        _ensure_dir(diag_runs_dir)

    rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []
    total_written = 0
    success_count = 0
    resume_count = 0
    infeasible_count = 0
    nonoptimal_count = 0
    failed_count = 0
    detail_written = 0
    writes_ruminant_multiplier = not _uses_absolute_ruminant_cap(cfg)

    def _register_status(row_obj: Dict[str, object]) -> None:
        nonlocal success_count, infeasible_count, nonoptimal_count, failed_count
        status = str(row_obj.get("run_status") or "")
        if status in {"ok", "resumed"}:
            success_count += 1
        elif status == "infeasible":
            infeasible_count += 1
        elif status == "nonoptimal":
            nonoptimal_count += 1
        elif status in {"failed", "invalid_market_balance"}:
            failed_count += 1

    def _flush_buffers() -> None:
        nonlocal total_written, detail_written, rows, detail_rows
        if rows:
            total_written += _append_rows_csv(rows, out_path)
            rows = []
        if detail_rows:
            detail_written += _append_rows_csv(detail_rows, detail_out_path)
            detail_rows = []

    task_index_set = set(task_indices)
    active_run_idx = 0
    run_idx = 0
    try:
        for forest_idx, forest_pct in enumerate(forest_values, start=1):
            forest_mult = _pct_to_multiplier(forest_pct)
            for ruminant_idx, ruminant_pct in enumerate(ruminant_values, start=1):
                ruminant_mult = _pct_to_multiplier(ruminant_pct)
                for yield_pct in yield_values:
                    yield_mult = _pct_to_multiplier(yield_pct)
                    for ef_pct in ef_values:
                        ef_mult = _pct_to_multiplier(ef_pct)
                        scenario_id = _scenario_id(
                            forest_pct=forest_pct,
                            ruminant_pct=ruminant_pct,
                            yield_pct=yield_pct,
                            ef_pct=ef_pct,
                        )
                        scenario_dir = runs_dir / scenario_id
                        run_idx += 1
                        if (run_idx - 1) not in task_index_set:
                            continue
                        active_run_idx += 1

                        print(
                            f"[FIG-PANEL] ({active_run_idx}/{assigned_runs}, global {run_idx}/{total_runs}) {scenario_id} | "
                            f"forest={forest_pct:+.0f}% ruminant_kcal_cap={ruminant_pct:+.0f}% "
                            f"yield={yield_pct:+.0f}% ef={ef_pct:+.0f}%"
                        )

                        row: Dict[str, object] = {
                            "scenario_id": scenario_id,
                            "panel_row": forest_idx,
                            "panel_col": ruminant_idx,
                            "forest_area_change_pct": forest_pct,
                            "forest_area_multiplier": forest_mult,
                            RUMINANT_CAP_COL: ruminant_pct,
                            "yield_change_pct": yield_pct,
                            "yield_multiplier": yield_mult,
                            "emission_factor_change_pct": ef_pct,
                            "emission_factor_multiplier": ef_mult,
                            "target_year": int(cfg.get("fast_emis_year", 2080) or 2080),
                            "run_status": "pending",
                            "afolu_emissions_gt_co2eq_yr": None,
                            "forest_target_shortfall_ha": None,
                            "forest_target_surplus_ha": None,
                            "forest_target_abs_slack_ha": None,
                            "forest_target_slack_rate": None,
                            "forest_target_actual_ha": None,
                            "forest_target_requested_ha": None,
                            "market_shortage_t": None,
                            "market_gap_rate": None,
                            "market_gap_year": None,
                            "scenario_dir": str(scenario_dir),
                            "diagnostic_scenario_dir": "",
                            "model_status_code": None,
                            "model_status_text": "",
                            "iis_summary": "",
                            "error_type": "",
                            "error_message": "",
                        }
                        row.update(_empty_pale_summary())
                        if writes_ruminant_multiplier:
                            row["ruminant_intake_multiplier"] = ruminant_mult
                        _apply_batch_meta(row, batch_state)
                        effects = None
                        scenario_params = None
                        outdir = None
                        emissions_gt = None
                        resume_detail_df = pd.DataFrame()
                        resume_validation: Optional[ResumeValidation] = None
                        log_diag = None
                        neg_msg = None
                        diag_outdir = None
                        scenario_resume_fingerprint = _panel_resume_fingerprint(
                            scenario_id=scenario_id,
                            forest_pct=float(forest_pct),
                            ruminant_pct=float(ruminant_pct),
                            yield_pct=float(yield_pct),
                            ef_pct=float(ef_pct),
                            cfg=cfg,
                        )

                        try:
                            emissions_gt = None
                            resume_run_eligible = False
                            if cfg.get("resume", False):
                                resume_validation = validate_run_for_resume(
                                    scenario_dir,
                                    expected_scenario_id=scenario_id,
                                    expected_resume_fingerprint=scenario_resume_fingerprint,
                                )
                                resume_run_eligible = resume_validation.allowed
                                if not resume_run_eligible:
                                    print(
                                        f"[FIG-PANEL][RESUME-RERUN] {scenario_id}: "
                                        "structured run validation rejected reuse "
                                        f"({resume_validation.reason})"
                                    )
                                    if bool(cfg.get("clear_existing_run_dirs_on_resume_retry", True)):
                                        if _clear_existing_scenario_dir(scenario_dir, runs_dir):
                                            print(
                                                f"[FIG-PANEL][RERUN] cleared old run dir: {scenario_dir}"
                                            )

                            if resume_run_eligible and resume_validation is not None:
                                emissions_gt = _read_fast_emissions_gt(
                                    scenario_dir,
                                    validation=resume_validation,
                                )
                                resume_detail_df = _read_fast_global_emissions_detail(
                                    scenario_dir,
                                    validation=resume_validation,
                                )
                                market_diag_path = (
                                    scenario_dir
                                    / "Diagnostics"
                                    / "commodity_balance_by_commodity.csv"
                                )
                                resume_market_current = artifact_matches_validated_run(
                                    market_diag_path,
                                    resume_validation,
                                )
                                if (
                                    emissions_gt is not None
                                    and not resume_detail_df.empty
                                    and resume_market_current
                                ):
                                    log_path = scenario_dir / "Log" / "model.log"
                                    if artifact_matches_validated_run(log_path, resume_validation):
                                        log_diag = _apply_log_diagnostics(row, scenario_dir)
                                    else:
                                        solver_meta = (resume_validation.payload or {}).get("solver") or {}
                                        row["model_status_code"] = solver_meta.get("status_code")
                                        row["model_status_text"] = str(
                                            solver_meta.get("status_name") or ""
                                        ).upper()
                                        log_diag = {"run_status_hint": ""}
                                    if _apply_terminal_model_status(row, log_diag):
                                        print(
                                            f"[FIG-PANEL][RESUME-SKIP] {scenario_id} -> {row['run_status']}: "
                                            f"{row['error_message']}"
                                        )
                                    else:
                                        neg_msg = _validate_nonluc_fast_emissions(scenario_dir)
                                        row.update(_read_market_gap_summary(scenario_dir))
                                        gap_msg = _validate_market_balance_gap(
                                            scenario_dir,
                                            max_gap_rate=float((cfg.get("override_cfg") or {}).get("market_gap_max_rate", 0.10)),
                                        )
                                        if neg_msg:
                                            row["run_status"] = "invalid_fast_emissions"
                                            row["afolu_emissions_gt_co2eq_yr"] = None
                                            row["error_type"] = "NegativeNonLUCEmissions"
                                            row["error_message"] = neg_msg
                                        elif gap_msg:
                                            print(f"[FIG-PANEL][RESUME-RERUN] {scenario_id}: {gap_msg}")
                                            if _clear_existing_scenario_dir(scenario_dir, runs_dir):
                                                print(f"[FIG-PANEL][RERUN] cleared old run dir: {scenario_dir}")
                                        else:
                                            row["run_status"] = "resumed"
                                            row["afolu_emissions_gt_co2eq_yr"] = emissions_gt
                                            row.update(_read_market_gap_summary(scenario_dir))
                                            resume_count += 1
                                            if resume_count == 1 or resume_count % 50 == 0:
                                                print(
                                                    f"[FIG-PANEL][RESUME] reused {resume_count}: "
                                                    f"{scenario_id} ({active_run_idx}/{assigned_runs})"
                                                )
                                    if row["run_status"] != "pending":
                                        if row["run_status"] in {"ok", "resumed"}:
                                            row.update(
                                                _read_forest_target_slack_summary(
                                                    Path(str(row.get("scenario_dir") or scenario_dir)),
                                                    int(row.get("target_year") or cfg.get("fast_emis_year", 2080) or 2080),
                                                )
                                            )
                                            forest_msg = _validate_forest_target_consistency(
                                                row,
                                                expected_forest_targets_by_multiplier,
                                            )
                                            if forest_msg:
                                                print(f"[FIG-PANEL][RESUME-RERUN] {scenario_id}: {forest_msg}")
                                                if _clear_existing_scenario_dir(scenario_dir, runs_dir):
                                                    print(f"[FIG-PANEL][RERUN] cleared old run dir: {scenario_dir}")
                                                if row["run_status"] == "resumed" and resume_count > 0:
                                                    resume_count -= 1
                                                row["run_status"] = "pending"
                                                row["afolu_emissions_gt_co2eq_yr"] = None
                                                row["error_type"] = ""
                                                row["error_message"] = ""
                                        if row["run_status"] in {"ok", "resumed"}:
                                            row.update(
                                                _read_pale_summary(
                                                    Path(str(row.get("scenario_dir") or scenario_dir)),
                                                    int(row.get("target_year") or cfg.get("fast_emis_year", 2080) or 2080),
                                                    detail_df=resume_detail_df,
                                                )
                                            )
                                            _append_global_emissions_detail(
                                                detail_rows,
                                                resume_detail_df,
                                                row,
                                            )
                                        if row["run_status"] != "pending":
                                            rows.append(row)
                                            _register_status(row)
                                            if len(rows) >= write_every:
                                                _flush_buffers()
                                            continue

                                if (
                                    emissions_gt is None
                                    or resume_detail_df.empty
                                    or not resume_market_current
                                ):
                                    print(
                                        f"[FIG-PANEL][RESUME-RERUN] {scenario_id}: "
                                        "fast emissions files are missing, stale, or belong to another run"
                                    )
                                    if bool(cfg.get("clear_existing_run_dirs_on_resume_retry", True)):
                                        if _clear_existing_scenario_dir(scenario_dir, runs_dir):
                                            print(
                                                f"[FIG-PANEL][RERUN] cleared old run dir: {scenario_dir}"
                                            )
                                    log_diag = {}
                                elif log_diag is None:
                                    log_diag = _apply_log_diagnostics(row, scenario_dir)
                                if log_diag.get("run_status_hint") == "infeasible":
                                    _apply_terminal_model_status(row, log_diag)
                                    if failed_diag_used < failed_diag_budget:
                                        effects = _build_effects(
                                            universe=shared_universe,
                                            scenario_id=scenario_id,
                                            yield_change_pct=yield_pct,
                                            ef_change_pct=ef_pct,
                                            ruminant_change_pct=ruminant_pct,
                                        )
                                        scenario_params = _build_forest_scenario_params(cfg, forest_mult)
                                        try:
                                            diag_outdir = _rerun_failed_point_with_diagnostics(
                                                cfg=cfg,
                                                paths=paths,
                                                shared_cfg=shared_cfg,
                                                shared_universe=shared_universe,
                                                shared_run_cache=shared_run_cache,
                                                diag_runs_dir=diag_runs_dir,
                                                scenario_id=scenario_id,
                                                scenario_params=scenario_params,
                                                effects=effects,
                                            )
                                            if diag_outdir:
                                                row["diagnostic_scenario_dir"] = str(diag_outdir)
                                                failed_diag_used += 1
                                                print(
                                                    f"[FIG-PANEL][DIAG] {scenario_id} -> detailed rerun saved at {diag_outdir}"
                                                )
                                        except Exception as diag_exc:
                                            print(f"[FIG-PANEL][DIAG-FAILED] {scenario_id}: {diag_exc}")
                                    rows.append(row)
                                    _register_status(row)
                                    if len(rows) >= write_every:
                                        _flush_buffers()
                                    print(
                                        f"[FIG-PANEL][RESUME] {scenario_id} -> {row['run_status']} "
                                        f"({row['error_message']})"
                                    )
                                    continue

                                if log_diag.get("run_status_hint") == "nonoptimal":
                                    if _is_retryable_resume_status(cfg, log_diag):
                                        print(
                                            f"[FIG-PANEL][RESUME-RETRY] {scenario_id} -> "
                                            f"{row.get('model_status_text') or 'NONOPTIMAL'}"
                                        )
                                        if bool(cfg.get("clear_existing_run_dirs_on_resume_retry", True)):
                                            try:
                                                if _clear_existing_scenario_dir(scenario_dir, runs_dir):
                                                    print(
                                                        f"[FIG-PANEL][RESUME-RETRY] cleared old run dir: {scenario_dir}"
                                                    )
                                            except Exception as clear_exc:
                                                print(
                                                    f"[FIG-PANEL][RESUME-RETRY] failed to clear {scenario_dir}: {clear_exc}"
                                                )
                                    else:
                                        _apply_terminal_model_status(row, log_diag)
                                        if failed_diag_used < failed_diag_budget:
                                            effects = _build_effects(
                                                universe=shared_universe,
                                                scenario_id=scenario_id,
                                                yield_change_pct=yield_pct,
                                                ef_change_pct=ef_pct,
                                                ruminant_change_pct=ruminant_pct,
                                            )
                                            scenario_params = _build_forest_scenario_params(cfg, forest_mult)
                                            try:
                                                diag_outdir = _rerun_failed_point_with_diagnostics(
                                                    cfg=cfg,
                                                    paths=paths,
                                                    shared_cfg=shared_cfg,
                                                    shared_universe=shared_universe,
                                                    shared_run_cache=shared_run_cache,
                                                    diag_runs_dir=diag_runs_dir,
                                                    scenario_id=scenario_id,
                                                    scenario_params=scenario_params,
                                                    effects=effects,
                                                )
                                                if diag_outdir:
                                                    row["diagnostic_scenario_dir"] = str(diag_outdir)
                                                    failed_diag_used += 1
                                                    print(
                                                        f"[FIG-PANEL][DIAG] {scenario_id} -> detailed rerun saved at {diag_outdir}"
                                                    )
                                            except Exception as diag_exc:
                                                print(f"[FIG-PANEL][DIAG-FAILED] {scenario_id}: {diag_exc}")
                                        rows.append(row)
                                        _register_status(row)
                                        if len(rows) >= write_every:
                                            _flush_buffers()
                                        print(
                                            f"[FIG-PANEL][RESUME] {scenario_id} -> {row['run_status']} "
                                            f"({row['error_message']})"
                                        )
                                        continue
                            effects = _build_effects(
                                universe=shared_universe,
                                scenario_id=scenario_id,
                                yield_change_pct=yield_pct,
                                ef_change_pct=ef_pct,
                                ruminant_change_pct=ruminant_pct,
                            )
                            scenario_params = _build_forest_scenario_params(cfg, forest_mult)
                            if (
                                not bool(cfg.get("resume", False))
                                and bool(cfg.get("clear_existing_run_dirs_when_no_resume", True))
                            ):
                                if _clear_existing_scenario_dir(scenario_dir, runs_dir):
                                    print(f"[FIG-PANEL][RERUN] cleared old run dir: {scenario_dir}")

                            outdir = run_one_pipeline(
                                paths,
                                pre_macc_e0=False,
                                scenario_id=scenario_id,
                                scenario_params=scenario_params,
                                scenario_effects=effects,
                                solve=bool(cfg.get("solve", True)),
                                use_fao_modules=bool(cfg.get("use_fao_modules", True)),
                                save_root=str(runs_dir),
                                future_last_only=bool(cfg.get("future_last_only", True)),
                                use_linear=bool(cfg.get("use_linear", True)),
                                fast_emis_only=bool(cfg.get("fast_emis_only", True)),
                                fast_emis_year=int(cfg.get("fast_emis_year", 2080) or 2080),
                                resume_fingerprint=scenario_resume_fingerprint,
                                prebuilt_config=shared_cfg,
                                prebuilt_universe=shared_universe,
                                prebuilt_run_cache=shared_run_cache,
                            )
                            row["scenario_dir"] = str(outdir)
                            emissions_gt = _read_fast_emissions_gt(Path(outdir))
                            log_diag = _apply_log_diagnostics(row, Path(outdir))
                            neg_msg = _validate_nonluc_fast_emissions(Path(outdir))
                            row.update(_read_market_gap_summary(Path(outdir)))
                            gap_msg = _validate_market_balance_gap(
                                Path(outdir),
                                max_gap_rate=float((cfg.get("override_cfg") or {}).get("market_gap_max_rate", 0.10)),
                            )
                            if _apply_terminal_model_status(row, log_diag):
                                pass
                            elif neg_msg:
                                row["run_status"] = "invalid_fast_emissions"
                                row["afolu_emissions_gt_co2eq_yr"] = None
                                row["error_type"] = "NegativeNonLUCEmissions"
                                row["error_message"] = neg_msg
                            elif gap_msg:
                                row["run_status"] = "invalid_market_balance"
                                row["afolu_emissions_gt_co2eq_yr"] = None
                                row["error_type"] = "MarketBalanceGap"
                                row["error_message"] = gap_msg
                            elif emissions_gt is not None:
                                row["run_status"] = "ok"
                                row["afolu_emissions_gt_co2eq_yr"] = emissions_gt
                            else:
                                row["run_status"] = "missing_fast_summary"
                                row["afolu_emissions_gt_co2eq_yr"] = None
                                row["error_type"] = "MissingFastSummary"
                                row["error_message"] = (
                                    "emissions_fast_summary.csv missing, empty, "
                                    "or missing total_co2eq_gt/kt"
                                )
                            if row["run_status"] not in {"ok", "resumed"}:
                                print(
                                    f"[FIG-PANEL][WARN] {scenario_id} -> {row['run_status']}: "
                                    f"{row['error_message'] or 'missing total_co2eq_gt'}"
                                )
                            if row["run_status"] in {"ok", "resumed"}:
                                row.update(
                                    _read_forest_target_slack_summary(
                                        Path(str(row.get("scenario_dir") or outdir)),
                                        int(row.get("target_year") or cfg.get("fast_emis_year", 2080) or 2080),
                                    )
                                )
                                forest_msg = _validate_forest_target_consistency(
                                    row,
                                    expected_forest_targets_by_multiplier,
                                )
                                if forest_msg:
                                    row["run_status"] = "invalid_forest_target"
                                    row["afolu_emissions_gt_co2eq_yr"] = None
                                    row["error_type"] = "ForestTargetMismatch"
                                    row["error_message"] = forest_msg
                                    print(f"[FIG-PANEL][WARN] {scenario_id} -> invalid_forest_target: {forest_msg}")
                            if row["run_status"] in {"ok", "resumed"}:
                                run_detail_df = _read_fast_global_emissions_detail(Path(outdir))
                                row.update(
                                    _read_pale_summary(
                                        Path(str(row.get("scenario_dir") or outdir)),
                                        int(row.get("target_year") or cfg.get("fast_emis_year", 2080) or 2080),
                                        detail_df=run_detail_df,
                                    )
                                )
                                _append_global_emissions_detail(
                                    detail_rows,
                                    run_detail_df,
                                    row,
                                )
                            if (
                                row["run_status"] not in {"ok", "resumed"}
                                and failed_diag_used < failed_diag_budget
                            ):
                                try:
                                    diag_outdir = _rerun_failed_point_with_diagnostics(
                                        cfg=cfg,
                                        paths=paths,
                                        shared_cfg=shared_cfg,
                                        shared_universe=shared_universe,
                                        shared_run_cache=shared_run_cache,
                                        diag_runs_dir=diag_runs_dir,
                                        scenario_id=scenario_id,
                                        scenario_params=scenario_params,
                                        effects=effects,
                                    )
                                    if diag_outdir:
                                        row["diagnostic_scenario_dir"] = str(diag_outdir)
                                        failed_diag_used += 1
                                        print(
                                            f"[FIG-PANEL][DIAG] {scenario_id} -> detailed rerun saved at {diag_outdir}"
                                        )
                                except Exception as diag_exc:
                                    print(f"[FIG-PANEL][DIAG-FAILED] {scenario_id}: {diag_exc}")
                        except KeyboardInterrupt:
                            row["run_status"] = "interrupted"
                            row["error_type"] = "KeyboardInterrupt"
                            row["error_message"] = "Interrupted by user or runtime"
                            rows.append(row)
                            _register_status(row)
                            _flush_buffers()
                            print(f"[FIG-PANEL][INTERRUPTED] {scenario_id}")
                            raise
                        except Exception as exc:
                            row["run_status"] = "failed"
                            row["error_type"] = type(exc).__name__
                            row["error_message"] = str(exc)
                            print(f"[FIG-PANEL][FAILED] {scenario_id}: {type(exc).__name__}: {exc}")
                            if bool(cfg.get("stop_on_error", False)):
                                rows.append(row)
                                _register_status(row)
                                _flush_buffers()
                                raise
                        finally:
                            effects = None
                            scenario_params = None
                            outdir = None
                            emissions_gt = None
                            log_diag = None
                            neg_msg = None
                            diag_outdir = None
                            gc.collect()

                        rows.append(row)
                        _register_status(row)
                        if len(rows) >= write_every:
                            _flush_buffers()

        _flush_buffers()
        print(f"[FIG-PANEL] results written: {out_path}")
        if detail_written > 0:
            print(f"[FIG-PANEL] global emissions detail written: {detail_out_path}")
        if total_written > 0:
            print(
                f"[FIG-PANEL] completed: total={total_written}, "
                f"success={success_count}, resumed={resume_count}, infeasible={infeasible_count}, "
                f"nonoptimal={nonoptimal_count}, failed={failed_count}"
            )
        meta_row = _apply_batch_meta(
            {
                "root_output_dir": str(root_output_dir),
                "active_output_dir": str(output_dir),
                "requested_runs": int(total_runs),
                "assigned_runs": int(assigned_runs),
                "attempted_runs": int(total_written),
                "success_count": int(success_count),
                "resume_count": int(resume_count),
                "infeasible_count": int(infeasible_count),
                "nonoptimal_count": int(nonoptimal_count),
                "failed_count": int(failed_count),
                "detail_rows": int(detail_written),
                "ef_points": int(len(ef_values)),
                "yield_points": int(len(yield_values)),
                "ruminant_points": int(len(ruminant_values)),
                "forest_points": int(len(forest_values)),
                "runs_subdir": str(cfg.get("runs_subdir") or "runs"),
                "results_csv": str(cfg.get("results_csv") or "figure_panel_dataset_long.csv"),
                "global_emissions_detail_csv": str(
                    cfg.get("global_emissions_detail_csv")
                    or "figure_panel_global_emissions_detail_long.csv"
                ),
            },
            batch_state,
        )
        pd.DataFrame([meta_row]).to_csv(meta_path, index=False, encoding="utf-8-sig")
        cost_status_df = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
        write_sensitivity_cost_summaries(
            cost_status_df,
            output_dir=output_dir,
            run_search_root=output_dir,
        )
        print(f"[FIG-PANEL] run_meta: {meta_path}")
        if bool(batch_state.get("enabled")):
            print(
                "[FIG-PANEL] batch outputs written. Run merge after all batches finish: "
                f"python S5_3_3_merge_panel_yield_ef_batches.py --total-batches {batch_state['count']}"
            )
    finally:
        for key, value in cfg_backup.items():
            CFG[key] = value


if __name__ == "__main__":
    main()
