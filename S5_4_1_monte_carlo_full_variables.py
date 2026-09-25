# -*- coding: utf-8 -*-
"""
S5.4: sample all variables with Monte Carlo using MC_effect_low_land_new, run the main model in batches, and output:
1) A sample-status table.
2) Fast emissions summaries for successful samples.
3) Global CO2eq by process for successful samples.
4) Global calorie-weighted means of each Element for successful samples.

Notes:
- Reuse S5.1 parsing and normalization rules for Scenario_config_new.xlsx / MC_effect_low_land_new.
- Continue calling S4_0_main.run_one_pipeline(...) for complete scenarios to preserve the existing workflow.
- Element means use base-year 2020 production_t * kcal_per_ton as country-commodity weights.
"""
from __future__ import annotations

import copy
import gc
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from config_paths import get_input_base, get_results_base
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
from S2_0_load_data import (
    DataPaths,
    build_universe_from_dict_v3,
    load_nutrient_factors_from_dict_v3,
)
from S4_0_main import CFG, MCPrecheckFailed, build_run_baseline_cache, run_one_pipeline
from S5_cost_summary_outputs import write_sensitivity_cost_summaries
from S5_1_1_sensitivity_mc_variable_effect import (
    _build_mc_sample_rows,
    _build_region_members,
    _build_scenario_effects,
    _calc_bound_value,
    _calc_ratio,
    _draw_mc_param_rows,
    _load_mc_baselines,
    _load_mc_specs_effect,
    _lookup_ef_base_value,
    _lookup_base_value,
    _normalize_kind,
    _normalize_mc_specs,
    resolve_mc_effect_sheet,
    RUMINANT_REDUCTION_DISCRETE_LEVELS,
    _sample_unit_matrix_for_specs,
    _u_for_country,
    _u_for_ef,
    _validate_nonluc_fast_emissions,
    _value_unit_for,
    _write_mc_sample_xlsx,
)


CONFIG = {
    "seed": 42,
    "samples": 20000,
    "year": 2080,
    "output_dir": "",  # empty -> <NZF_OUTPUT_DIR>/MC_Full_Variables
    "runs_subdir": "runs",
    "sample_workbook_subdir": "MC",
    "status_csv": "mc_sample_status.csv",
    "draws_csv": "mc_draws_long.csv",
    "success_summary_csv": "mc_success_fast_summary.csv",
    "success_process_csv": "mc_success_global_process_co2eq.csv",
    "success_weighted_elements_csv": "mc_success_weighted_elements.csv",
    "ef_intensity_baseline_emissions_csv": "",  # empty -> S4 CFG['base_case']/Emis under input/Emission or output
    "require_ef_co2eq_intensity": True,
    "success_realized_ruminant_csv": "mc_success_realized_ruminant_share.csv",
    "success_land_balance_csv": "mc_success_crop_pasture_land_balance.csv",
    "resume": False,
    "clear_existing_run_dirs_when_no_resume": True,
    "clear_existing_run_dirs_on_resume_retry": True,
    "rerun_infeasible_on_resume": True,
    "save_per_run_dirs": True,
    "save_sample_workbook": False,
    "write_every_n_runs": 1,
    "max_runs": None,
    "stop_on_error": False,
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
    "failed_diag_sample_n": 0,
    "failed_diag_runs_subdir": "runs_diag",
    "batch": {
        "enabled": True,
        "total_batches": 200,
        "batch_index": 1,  # 1-based
        "assignment": "round_robin",  # 'round_robin' | 'contiguous'
        "batches_subdir": "batches",
    },
    "solve": True,
    "use_fao_modules": True,
    "use_linear": True,
    "future_last_only": True,
    "fast_emis_only": True,
    "fast_emis_year": 2080,
    "nutrition_profile_sheet": "low_land_new",
    "mc_sheet_prefer": "MC_effect_low_land_new",
    "aggregate_non_ef": False,
    "sampling": {
        # Sampling method for the normalized U(0,1) matrix before quantile re-scaling.
        
        # Supported values:
        # "uniform": independent pseudo-random draws; fastest baseline, but coverage is weakest.
        # "lhs": Latin Hypercube Sampling; good coverage with a fixed sample budget.
        # "lhs_centered": midpoint LHS; very reproducible, with less random jitter.
        # "uniform_antithetic": uniform draws paired with (1-u); helps stabilize means.
        # "lhs_antithetic": LHS paired with (1-u); good when sample budget is fixed and
        # you want more stable global means / quantiles.
        # "range": endpoint-to-endpoint stratified coverage; use for stress tests, not production MC.
        # "halton": low-discrepancy sequence; useful in lower/medium dimensions, but usually
        # not the first choice for this model because variable count is high.
        # "mixed": blend of lhs + uniform; robust fallback when you want some extra random jitter.
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
        # Snap selected kinds to explicit levels after continuous sampling.
        # For low_land_new, ruminant_reduction is an absolute share cap with
        # 1%-step candidates. The active Min_bound/Max_bound in
        # MC_effect_low_land_new still controls the sampled range.
        "discrete_levels_enabled": True,
        # False means discrete variables are sampled directly on the full grid
        # instead of first applying quantile_bounds; ruminant can hit every
        # 1%-step profile level inside each row's Min_bound/Max_bound.
        "discrete_levels_use_quantile_bounds": False,
        "discrete_levels_by_kind": {
            "ruminant_reduction": list(RUMINANT_REDUCTION_DISCRETE_LEVELS),
        },
        # Only used by "mixed": share of rows generated by LHS before appending uniform rows.
        "mix_ratio": 0.5,
        # Used by antithetic / mixed methods: whether to shuffle rows after generation.
        "shuffle": True,
        # Used by Halton sequence only: Cranley-Patterson rotation for random scrambling.
        "scramble": True,
        # Final per-variable sampling range after rescaling U(0,1).
        # Use the full MC_effect Min_bound/Max_bound range.
        "quantile_bounds": (0.0, 1.0),
    },
    "mc_non_ef_mode": "shared",
    "mc_ef_mode": "shared",
    "ef_process_mode": "all",
    "override_cfg": {
        "nutrition_profile_sheet": "low_land_new",
        "cost_calculation_method": "off",
        "debug_level": 0,
        "grassland_coef_verbose_logging": False,
        "domestic_supply_simulation_mode": "hard_equation",
        "supply_curtailment_enabled": False,
        "supply_curtailment_penalty": 1e10,
        "zero_price_shutdown_enabled": True,
        "zero_demand_production_shutdown": False,
        "land_delta_anchor_to_available_stock": False,
        # Keep S5_4 aligned with S4 land headroom semantics:
        # forest_nonneg_ratio expands the stock available to forest_to_* flows,
        # while nonforest expansion is excluded from LUC accounting.
        "forest_nonneg_ratio": 2.0,
        "cropland_nonforest_expand_ratio": 1.05,
        "pasture_nonforest_expand_ratio": 1.02,
        "demand_method": "nutrition",
        "nutrition_band_epsilon": 0.10,
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
        # This is a post-run mass-balance quality screen. The model's hard
        # shortage cap is energy-based, so keep the mass screen less strict.
        "market_gap_max_rate": 0.08,
        "batch_mode": True,
        "linear_enable_infeasible_iis": False,
        "linear_enable_violation_iis": False,
        "linear_enable_output_diagnostics": False,
        "linear_enable_verbose_logging": False,
        "lightweight_diagnostic_outputs_enabled": True,
        "linear_solver_method": 1,
        "linear_solver_threads": 4,
    },
}


EF_INTENSITY_BASELINE_EMISSIONS_NAME = "emissions_summary_By_Country_Process_Item.csv"

INVALID_TOTAL_CO2EQ_GT_VALUES = (1.264874,)

# Recommended production setting for this script:
# CONFIG["sampling"]["method"] = "lhs_antithetic"

# Practical guidance for this model:
# High-dimensional / many-variable full MC (this script): prefer "lhs_antithetic".
# Use "lhs" only when you specifically want non-antithetic LHS rows.
# Small pilot runs to quickly probe the whole range: "range" or "halton".
# Do not use "uniform" as the first choice unless you specifically need the simplest baseline.

# Example configuration for 5 parallel batches:
# Notes:
# All 5 scripts/jobs must use identical configurations except batch_index, especially:
# samples / seed / sampling / mc_sheet_prefer / override_cfg must match.
# Each batch then receives its own subset of sample_id values from the same full sampling matrix.
# Combining all batches is equivalent to one complete Monte Carlo sample set.

# Batch 1:
# CONFIG["samples"] = 5000
# CONFIG["output_dir"] = str(Path(get_results_base()) / "MC_Full_Variables")
# CONFIG["batch"] = {
# "enabled": True,
# "total_batches": 5,
# "batch_index": 1,
# "assignment": "round_robin", # Recommended for a more balanced workload across batches.
# "batches_subdir": "batches",
# }

# For batches 2-5, change only batch_index:
# CONFIG["batch"]["batch_index"] = 2
# CONFIG["batch"]["batch_index"] = 3
# CONFIG["batch"]["batch_index"] = 4
# CONFIG["batch"]["batch_index"] = 5

# To assign a contiguous sample-ID range to each batch, use:
# CONFIG["batch"]["assignment"] = "contiguous"
# For example, 5,000 samples across 5 batches are approximately partitioned as follows:
# 1-1000, 1001-2000, 2001-3000, 3001-4000, 4001-5000


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
                f"Relative S5_4 output path escapes project root: {raw_path}. "
                "Use an absolute path instead."
            ) from exc
        return resolved
    return path.resolve()


def _default_output_base() -> Path:
    return _resolve_abs_path(Path(get_results_base()))


def _resolve_fullmc_output_root(raw_output_dir: object = "") -> Path:
    text = str(raw_output_dir or "").strip()
    if text:
        return _resolve_abs_path(Path(text))
    return _default_output_base() / "MC_Full_Variables"


def _sync_fullmc_output_environment(root_output_dir: Path) -> None:
    root = _resolve_abs_path(root_output_dir)
    os.environ["FULLMC_OUTPUT_DIR_RESOLVED"] = str(root)
    os.environ["NZF_OUTPUT_DIR"] = str(root.parent)


def _clear_existing_scenario_dir(scenario_dir: Path, runs_dir: Path) -> bool:
    if not scenario_dir.exists():
        return False
    scenario_path = scenario_dir.resolve()
    runs_path = runs_dir.resolve()
    try:
        scenario_path.relative_to(runs_path)
    except ValueError as exc:
        raise RuntimeError(f"Refuse to clear run dir outside runs root: {scenario_path}") from exc
    if scenario_path == runs_path:
        raise RuntimeError(f"Refuse to clear runs root itself: {scenario_path}")
    if not scenario_path.name.startswith("MC_FULL_"):
        raise RuntimeError(f"Refuse to clear non-S5_4 scenario dir: {scenario_path}")
    shutil.rmtree(scenario_path)
    return True


def _reset_output_file(path: Path) -> None:
    if not path.exists():
        return
    try:
        path.unlink()
    except Exception as exc:
        raise RuntimeError(f"Cannot reset batch output before rebuild: {path}") from exc


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


def _scenario_id(sample_id: int) -> str:
    return f"MC_FULL_{int(sample_id):05d}"


def _batch_tag(batch_index: int, batch_count: int) -> str:
    return f"batch_{int(batch_index):02d}_of_{int(batch_count):02d}"


def _resolve_batch_settings(cfg: Dict[str, object]) -> Dict[str, object]:
    batch_cfg = cfg.get("batch", {}) or {}
    batch_count = int(batch_cfg.get("total_batches", 1) or 1)
    batch_index = int(batch_cfg.get("batch_index", 1) or 1)
    if batch_count <= 0:
        raise ValueError("batch.total_batches 必须为正整数。")
    if batch_index <= 0 or batch_index > batch_count:
        raise ValueError("batch.batch_index 必须位于 1..total_batches。")
    enabled = bool(batch_cfg.get("enabled", False) or batch_count > 1)
    assignment = str(batch_cfg.get("assignment", "round_robin") or "round_robin").strip().lower()
    if assignment not in {"round_robin", "contiguous"}:
        raise ValueError("batch.assignment 仅支持 'round_robin' 或 'contiguous'。")
    return {
        "enabled": enabled,
        "count": batch_count,
        "index": batch_index,
        "tag": _batch_tag(batch_index, batch_count),
        "assignment": assignment,
        "batches_subdir": str(batch_cfg.get("batches_subdir", "batches") or "batches"),
    }


def _select_batch_sample_indices(
    *,
    total_samples: int,
    batch_count: int,
    batch_index: int,
    assignment: str,
) -> List[int]:
    if total_samples <= 0:
        return []
    if batch_count <= 1:
        return list(range(total_samples))
    if assignment == "contiguous":
        base = total_samples // batch_count
        rem = total_samples % batch_count
        start = (batch_index - 1) * base + min(batch_index - 1, rem)
        size = base + (1 if batch_index <= rem else 0)
        return list(range(start, start + size))
    return [idx for idx in range(total_samples) if (idx % batch_count) == (batch_index - 1)]


def _apply_batch_meta(row: Dict[str, object], batch_state: Dict[str, object]) -> Dict[str, object]:
    row["batch_index"] = int(batch_state["index"])
    row["batch_count"] = int(batch_state["count"])
    row["batch_tag"] = str(batch_state["tag"])
    return row


def _read_fast_summary_df(scenario_dir: Path) -> pd.DataFrame:
    fast_path = scenario_dir / "Emis" / "emissions_fast_summary.csv"
    if not fast_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(fast_path)
    except Exception:
        return pd.DataFrame()


def _read_validated_run_csv(
    path: Path,
    validation: ResumeValidation,
) -> pd.DataFrame:
    """Read a current-run CSV only when its embedded identity is exact."""
    if not artifact_matches_validated_run(path, validation):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    required = {"run_id", "scenario_id"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()
    run_ids = set(df["run_id"].dropna().astype(str).str.strip())
    scenario_ids = set(df["scenario_id"].dropna().astype(str).str.strip())
    if run_ids != {validation.run_id} or scenario_ids != {validation.scenario_id}:
        return pd.DataFrame()
    return df


def _read_validated_fast_summary_df(
    scenario_dir: Path,
    validation: ResumeValidation,
) -> pd.DataFrame:
    return _read_validated_run_csv(
        scenario_dir / "Emis" / "emissions_fast_summary.csv",
        validation,
    )


def _read_validated_fast_global_detail_df(
    scenario_dir: Path,
    validation: ResumeValidation,
) -> pd.DataFrame:
    return _read_validated_run_csv(
        scenario_dir / "Emis" / "emissions_fast_global_detail.csv",
        validation,
    )


def _fast_summary_total_gt(fast_df: pd.DataFrame) -> Optional[float]:
    if fast_df is None or fast_df.empty:
        return None
    if "total_co2eq_gt" in fast_df.columns:
        vals = pd.to_numeric(fast_df["total_co2eq_gt"], errors="coerce").dropna()
        if not vals.empty:
            return float(vals.iloc[0])
    if "total_co2eq_kt" in fast_df.columns:
        vals = pd.to_numeric(fast_df["total_co2eq_kt"], errors="coerce").dropna()
        if not vals.empty:
            return float(vals.iloc[0]) * 1e-6
    return None


def _is_invalid_total_gt(value: Optional[float]) -> bool:
    if value is None:
        return False
    try:
        val = float(value)
    except Exception:
        return False
    if not np.isfinite(val):
        return True
    rounded = round(val, 6)
    return any(rounded == round(float(bad), 6) for bad in INVALID_TOTAL_CO2EQ_GT_VALUES)


def _read_market_gap_summary(scenario_dir: Path) -> Dict[str, object]:
    return _read_solver_market_balance_summary(scenario_dir)


def _validate_market_balance_gap(scenario_dir: Path, *, max_gap_rate: float = 0.01) -> Optional[str]:
    """Reject runs unless solver balance and shortage-rate diagnostics pass."""
    return _validate_solver_market_balance_gap(
        scenario_dir,
        max_gap_rate=max_gap_rate,
    )


def _read_fast_global_detail_df(scenario_dir: Path) -> pd.DataFrame:
    detail_path = scenario_dir / "Emis" / "emissions_fast_global_detail.csv"
    if not detail_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(detail_path)
    except Exception:
        return pd.DataFrame()


def _read_realized_ruminant_share_df(
    scenario_dir: Path,
    *,
    validation: Optional[ResumeValidation] = None,
) -> pd.DataFrame:
    detail_path = scenario_dir / "Diagnostics" / "realized_ruminant_intake_share.csv"
    if not detail_path.exists():
        return pd.DataFrame()
    if validation is not None and not artifact_matches_validated_run(detail_path, validation):
        return pd.DataFrame()
    try:
        df = pd.read_csv(detail_path)
    except Exception:
        return pd.DataFrame()
    if {"scope", "commodity", "year", "demand_t", "demand_kcal"}.issubset(df.columns):
        commodity_rows = df[df["scope"].astype(str).str.strip().eq("commodity")].copy()
        if not commodity_rows.empty:
            ruminant_comms = {
                "Cattle, dairy",
                "Cattle, non-dairy",
                "Buffalo, dairy",
                "Buffalo, non-dairy",
                "Sheep, dairy",
                "Sheep, non-dairy",
                "Goats, dairy",
                "Goats, non-dairy",
                "Camel, dairy",
                "Camel, non-dairy",
                "Llamas",
                "Bovine Meat-cattle",
                "Bovine Meat-buffalo",
                "Mutton & Goat Meat-sheep",
                "Mutton & Goat Meat-goat",
                "Meat, Other-camels",
                "Meat, Other-other domestic camelids",
                "Milk-cattle",
                "Milk-buffalo",
                "Milk-sheep",
                "Milk-goats",
                "Milk-camel",
                "Meat of cattle with the bone, fresh or chilled",
                "Meat of buffalo, fresh or chilled",
                "Meat of sheep, fresh or chilled",
                "Meat of goat, fresh or chilled",
            }
            work = commodity_rows.copy()
            work["year"] = pd.to_numeric(work["year"], errors="coerce")
            work["demand_t"] = pd.to_numeric(work["demand_t"], errors="coerce")
            work["demand_kcal"] = pd.to_numeric(work["demand_kcal"], errors="coerce")
            work["is_ruminant_intake"] = work["commodity"].astype(str).str.strip().isin(ruminant_comms)
            rows = []
            for year, grp in work.dropna(subset=["year"]).groupby("year", sort=False):
                demand_t = float(grp["demand_t"].fillna(0.0).sum())
                demand_kcal = float(grp["demand_kcal"].fillna(0.0).sum())
                rgrp = grp[grp["is_ruminant_intake"]]
                ruminant_t = float(rgrp["demand_t"].fillna(0.0).sum())
                ruminant_kcal = float(rgrp["demand_kcal"].fillna(0.0).sum())
                rows.append(
                    {
                        "scope": "global_summary",
                        "country": "Global",
                        "commodity": "All food commodities",
                        "year": int(year),
                        "demand_t": demand_t,
                        "demand_kcal": demand_kcal,
                        "is_ruminant_intake": False,
                        "ruminant_demand_t": ruminant_t,
                        "ruminant_demand_kcal": ruminant_kcal,
                        "realized_ruminant_share_kcal": (
                            ruminant_kcal / demand_kcal if demand_kcal > 0 else np.nan
                        ),
                        "realized_ruminant_share_t": ruminant_t / demand_t if demand_t > 0 else np.nan,
                    }
                )
            if rows:
                return pd.DataFrame(rows)
    if "scope" in df.columns:
        df = df[df["scope"].astype(str).str.strip().eq("global_summary")].copy()
    return df


def _read_crop_pasture_land_balance_df(
    scenario_dir: Path,
    *,
    validation: Optional[ResumeValidation] = None,
) -> pd.DataFrame:
    detail_path = scenario_dir / "Diagnostics" / "crop_pasture_land_balance.csv"
    if not detail_path.exists():
        return pd.DataFrame()
    if validation is not None and not artifact_matches_validated_run(detail_path, validation):
        return pd.DataFrame()
    try:
        return pd.read_csv(detail_path)
    except Exception:
        return pd.DataFrame()


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
    if iis_match:
        diag["iis_summary"] = (
            f"IIS computed: {iis_match.group(1)} constraints and {iis_match.group(2)} bounds"
        )
    if infeasible_hit or status_code == 3:
        diag["run_status_hint"] = "infeasible"
        diag["model_status_text"] = "Infeasible model"
        diag["error_type_hint"] = "InfeasibleModel"
        parts = []
        if status_code is not None:
            parts.append(f"status={status_code}")
        parts.append("Infeasible model")
        if diag["iis_summary"]:
            parts.append(diag["iis_summary"])
        diag["error_message_hint"] = "; ".join(parts)
        return diag
    if status_code is not None and status_code != 2:
        status_text = status_map.get(status_code, f"STATUS_{status_code}")
        diag["run_status_hint"] = "nonoptimal"
        diag["model_status_text"] = status_text
        diag["error_type_hint"] = "NonOptimalModel"
        parts = [f"status={status_code}", status_text]
        if diag["iis_summary"]:
            parts.append(diag["iis_summary"])
        diag["error_message_hint"] = "; ".join(parts)
        return diag
    if status_code == 2:
        diag["model_status_text"] = "OPTIMAL"
    return diag


def _apply_log_diagnostics(row: Dict[str, object], scenario_dir: Path) -> Dict[str, object]:
    diag = _read_model_log_diagnostics(scenario_dir)
    row["model_status_code"] = diag.get("model_status_code")
    row["model_status_text"] = diag.get("model_status_text", "")
    row["iis_summary"] = diag.get("iis_summary", "")
    return diag


def _apply_model_status_failure(row: Dict[str, object], log_diag: Dict[str, object]) -> bool:
    """Return True when the model log proves this run is not a usable success."""
    status_hint = str(log_diag.get("run_status_hint") or "")
    if status_hint not in {"infeasible", "nonoptimal"}:
        return False
    row["run_status"] = status_hint
    row["afolu_emissions_gt_co2eq_yr"] = None
    row["error_type"] = str(log_diag.get("error_type_hint") or "ModelLog")
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
    effects,
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


def _build_calorie_weight_lookup(
    *,
    paths: DataPaths,
    universe,
    run_cache: Dict[str, Any],
    base_year: int,
) -> Dict[Tuple[str, str], float]:
    """Build country-commodity calorie weights using 2020 production_t * kcal_per_ton."""
    production_df = run_cache.get("production_df")
    if not isinstance(production_df, pd.DataFrame) or production_df.empty:
        return {}
    work = production_df.copy()
    if "year" in work.columns:
        work["year"] = pd.to_numeric(work["year"], errors="coerce")
        work = work[work["year"] == int(base_year)]
    if work.empty or "production_t" not in work.columns or "commodity" not in work.columns:
        return {}
    if "country" not in work.columns or work["country"].isna().all():
        if "M49_Country_Code" in work.columns:
            work["country"] = work["M49_Country_Code"].astype(str).map(universe.country_by_m49)
    work = work.dropna(subset=["country", "commodity", "production_t"]).copy()
    if work.empty:
        return {}
    nutrient_energy_map = load_nutrient_factors_from_dict_v3(paths.dict_v3_path, "energy") or {}
    if not nutrient_energy_map:
        return {}
    work["production_t"] = pd.to_numeric(work["production_t"], errors="coerce")
    work["kcal_per_ton"] = work["commodity"].map(nutrient_energy_map)
    work["weight_kcal"] = work["production_t"] * pd.to_numeric(work["kcal_per_ton"], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["weight_kcal"])
    if work.empty:
        return {}
    grouped = work.groupby(["country", "commodity"], as_index=False)["weight_kcal"].sum()
    out: Dict[Tuple[str, str], float] = {}
    for r in grouped.itertuples(index=False):
        try:
            out[(str(r.country), str(r.commodity))] = float(r.weight_kcal)
        except Exception:
            continue
    return out


def _normalize_m49_for_lookup(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    digits = re.sub(r"\D", "", text)
    if digits:
        return f"'{digits.zfill(3)}"
    return text


def _require_ef_co2eq_intensity(cfg: Dict[str, object]) -> bool:
    return bool(cfg.get("require_ef_co2eq_intensity", False))


def _ef_intensity_unavailable(cfg: Dict[str, object], message: str) -> Dict[Tuple[str, str, str, str], float]:
    if _require_ef_co2eq_intensity(cfg):
        raise RuntimeError(message)
    print(f"[S5_4][WARN] {message}")
    return {}


def _co2eq_ghg_key(value: object, *, allow_plain: bool = False) -> Optional[str]:
    text = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    text = re.sub(r"_+", "_", text)
    aliases = {
        "CH4_CO2EQ": "CH4",
        "CH4_CO2E": "CH4",
        "N2O_CO2EQ": "N2O",
        "N2O_CO2E": "N2O",
        "CO2_CO2EQ": "CO2",
        "CO2_CO2E": "CO2",
        "CO2EQ": "All",
        "CO2E": "All",
    }
    strict = aliases.get(text)
    if strict is not None or not allow_plain:
        return strict
    plain_aliases = {
        "CH4": "CH4",
        "METHANE": "CH4",
        "N2O": "N2O",
        "NITROUS_OXIDE": "N2O",
        "CO2": "CO2",
        "CARBON_DIOXIDE": "CO2",
        "ALL": "All",
    }
    return plain_aliases.get(text)


def _country_weight_for_m49(
    calorie_weights: Dict[Tuple[str, str], float],
    universe,
    m49_norm: str,
    commodity: str,
) -> float:
    unquoted = str(m49_norm or "").lstrip("'")
    candidates = [
        universe.country_by_m49.get(m49_norm),
        m49_norm,
        unquoted,
        f"'{unquoted}",
    ]
    for country in candidates:
        if not country:
            continue
        val = calorie_weights.get((str(country), str(commodity)))
        if val is not None and np.isfinite(val) and float(val) > 0:
            return float(val)
    return 0.0


def _ef_intensity_candidate_paths(raw: str) -> List[Path]:
    base_case = str(CFG.get("base_case", "BASE") or "BASE").strip() or "BASE"
    candidates: List[Path] = []
    if raw:
        raw_path = Path(raw)
        candidates.append(raw_path)
        if raw_path.suffix.lower() != ".csv":
            candidates.extend(
                [
                    raw_path / EF_INTENSITY_BASELINE_EMISSIONS_NAME,
                    raw_path / "Emis" / EF_INTENSITY_BASELINE_EMISSIONS_NAME,
                ]
            )

    base_reference_dirs = []
    cfg_base_reference_dir = str(CFG.get("base_reference_dir", "") or "").strip()
    if cfg_base_reference_dir:
        base_reference_dirs.append(Path(cfg_base_reference_dir))
    base_reference_dirs.extend(
        [
            Path(get_input_base()) / "Emission" / base_case,
            Path(get_results_base(base_case)),
        ]
    )
    for base_dir in base_reference_dirs:
        candidates.append(base_dir / "Emis" / EF_INTENSITY_BASELINE_EMISSIONS_NAME)

    seen = set()
    unique: List[Path] = []
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _resolve_ef_intensity_emissions_path(cfg: Dict[str, object]) -> Optional[Path]:
    raw = str(cfg.get("ef_intensity_baseline_emissions_csv", "") or "").strip()
    candidates = _ef_intensity_candidate_paths(raw)
    for path in candidates:
        if path.exists():
            print(f"[S5_4] EF intensity baseline emissions file: {path}")
            return path
    print("[S5_4][WARN] EF intensity baseline emissions file not found. Checked:")
    for path in candidates[:20]:
        print(f"  - {path}")
    if len(candidates) > 20:
        print(f"  ... {len(candidates) - 20} more")
    return None


def _build_ef_co2eq_intensity_lookup(
    *,
    cfg: Dict[str, object],
    universe,
    calorie_weights: Dict[Tuple[str, str], float],
    base_year: int,
) -> Dict[Tuple[str, str, str, str], float]:
    """Return baseline EF-process intensity as kg CO2eq/kcal.

    The native EF units differ by process. For a common ranking metric, use the
    baseline country-item-process CO2eq emission divided by the same
    country-item 2020 kcal weight, then scale it by the sampled EF ratio.
    """
    path = _resolve_ef_intensity_emissions_path(cfg)
    if path is None:
        return _ef_intensity_unavailable(
            cfg,
            "EF CO2eq/kcal baseline emissions file not found; EF absolute intensity unavailable. "
            "Set CONFIG['ef_intensity_baseline_emissions_csv'] or FULLMC_EF_INTENSITY_BASELINE_EMISSIONS_CSV "
            "to emissions_summary_By_Country_Process_Item.csv, or run the S4 base_case so "
            f"{str(CFG.get('base_case', 'BASE') or 'BASE')}/Emis/{EF_INTENSITY_BASELINE_EMISSIONS_NAME} exists.",
        )

    year_col = f"Y{int(base_year)}"
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception as exc:
        return _ef_intensity_unavailable(
            cfg,
            f"failed to inspect EF intensity baseline emissions file: {path}: {exc}",
        )
    if year_col not in header.columns:
        return _ef_intensity_unavailable(
            cfg,
            f"EF intensity baseline file missing {year_col}: {path}",
        )

    use_cols = ["M49_Country_Code", "Process", "Item", "GHG", year_col]
    try:
        df = pd.read_csv(path, usecols=use_cols, low_memory=False)
    except Exception as exc:
        return _ef_intensity_unavailable(
            cfg,
            f"failed to read EF intensity baseline emissions file: {path}: {exc}",
        )

    df.columns = [str(c).strip() for c in df.columns]
    df["Item"] = df["Item"].astype(str).str.strip()
    df["Process"] = df["Process"].astype(str).str.strip()
    df["m49_norm"] = df["M49_Country_Code"].map(_normalize_m49_for_lookup)
    df["ghg_key_strict"] = df["GHG"].map(_co2eq_ghg_key)
    df["ghg_key_plain"] = df["GHG"].map(lambda x: _co2eq_ghg_key(x, allow_plain=True))
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df = df.dropna(subset=[year_col]).copy()
    df = df[df["m49_norm"].astype(str).str.strip().ne("'000")]
    if df.empty:
        return _ef_intensity_unavailable(
            cfg,
            f"no usable EF CO2eq rows in baseline emissions file: {path}",
        )

    strict = df[df["ghg_key_strict"].notna()].copy()
    strict["ghg_key"] = strict["ghg_key_strict"]
    plain = df[df["ghg_key_strict"].isna() & df["ghg_key_plain"].notna()].copy()
    if strict.empty:
        plain["ghg_key"] = plain["ghg_key_plain"]
        df = plain
        if not df.empty:
            print(
                "[S5_4][WARN] EF baseline emissions file has no *_CO2eq/CO2e GHG labels; "
                "falling back to plain CH4/N2O/CO2 labels."
            )
    elif not plain.empty:
        key_cols = ["m49_norm", "Item", "Process", "ghg_key"]
        strict_keys = set(map(tuple, strict[key_cols].astype(str).to_numpy()))
        plain["ghg_key"] = plain["ghg_key_plain"]
        plain_key_tuples = list(map(tuple, plain[key_cols].astype(str).to_numpy()))
        plain = plain[[key not in strict_keys for key in plain_key_tuples]].copy()
        if not plain.empty:
            print(
                f"[S5_4][WARN] EF baseline emissions file is missing CO2eq labels for "
                f"{len(plain)} rows; using plain GHG labels only where no CO2eq row exists."
            )
        df = pd.concat([strict, plain], ignore_index=True)

    df = df.dropna(subset=["ghg_key"]).copy()
    if df.empty:
        return _ef_intensity_unavailable(
            cfg,
            f"no usable EF CO2eq GHG labels in baseline emissions file: {path}",
        )

    grouped = (
        df.groupby(["m49_norm", "Item", "Process", "ghg_key"], dropna=False, as_index=False)[year_col]
        .sum()
    )
    out: Dict[Tuple[str, str, str, str], float] = {}
    for row in grouped.itertuples(index=False):
        m49_norm = str(getattr(row, "m49_norm"))
        item = str(getattr(row, "Item"))
        process = str(getattr(row, "Process"))
        ghg_key = str(getattr(row, "ghg_key"))
        co2eq_kt = float(getattr(row, year_col))
        if not np.isfinite(co2eq_kt):
            continue
        kcal = _country_weight_for_m49(calorie_weights, universe, m49_norm, item)
        if kcal <= 0:
            continue
        out[(m49_norm, item, process, ghg_key)] = co2eq_kt * 1e6 / kcal

    print(
        f"[S5_4] EF CO2eq/kcal baseline intensities={len(out)} "
        f"from {path}"
    )
    if not out:
        return _ef_intensity_unavailable(
            cfg,
            f"EF CO2eq/kcal baseline lookup is empty after matching emissions rows to calorie weights: {path}",
        )
    return out


def _lookup_ef_co2eq_intensity(
    intensity_lookup: Dict[Tuple[str, str, str, str], float],
    m49: object,
    commodity: object,
    process: object,
    ghg: object,
) -> Optional[float]:
    if not intensity_lookup:
        return None
    m49_norm = _normalize_m49_for_lookup(m49)
    ghg_text = str(ghg or "").strip().upper()
    ghg_keys = [ghg_text] if ghg_text and ghg_text not in {"ALL", "NAN"} else []
    ghg_keys.append("All")
    commodity_key = str(commodity).strip()
    process_key = str(process).strip()
    for ghg_key in ghg_keys:
        val = intensity_lookup.get((m49_norm, commodity_key, process_key, ghg_key))
        if val is not None and np.isfinite(val):
            return float(val)
    return None


def _attach_param_metadata(
    param_rows: List[Dict[str, object]],
    specs_df: pd.DataFrame,
) -> List[Dict[str, object]]:
    spec_records = specs_df.to_dict("records")
    if len(spec_records) != len(param_rows):
        raise ValueError(
            f"MC spec rows ({len(spec_records)}) and param rows ({len(param_rows)}) mismatch."
        )
    merged: List[Dict[str, object]] = []
    for row, spec in zip(param_rows, spec_records):
        merged_row = dict(row)
        merged_row["spec_row_id"] = int(spec["spec_row_id"])
        merged_row["element_name"] = str(spec.get("Element", "") or "")
        merged_row["item_selector"] = str(spec.get("Item", "All") or "All")
        merged_row["process_selector"] = str(spec.get("Process", "All") or "All")
        merged_row["ghg_selector"] = str(spec.get("GHG", "All") or "All")
        merged_row["region_selector"] = str(spec.get("Region_cat", "All") or "All")
        merged_row["element_unit"] = str(spec.get("Element unit", "") or "")
        merged.append(merged_row)
    return merged


def _attach_effect_metadata(
    effects: Iterable,
    param_rows: List[Dict[str, object]],
) -> List[Any]:
    effect_list = list(effects)
    if len(effect_list) != len(param_rows):
        raise ValueError(
            f"ScenarioEffect count ({len(effect_list)}) and param rows ({len(param_rows)}) mismatch."
        )
    for eff, row in zip(effect_list, param_rows):
        eff.spec_row_id = int(row.get("spec_row_id", -1))
        eff.element_name = str(row.get("element_name", "") or "")
        eff.item_selector = str(row.get("item_selector", "All") or "All")
        eff.process_selector = str(row.get("process_selector", "All") or "All")
        eff.ghg_selector = str(row.get("ghg_selector", "All") or "All")
        eff.region_selector = str(row.get("region_selector", "All") or "All")
        # S4 uses this canonical metadata to bind generic physical levers such
        # as emission_factor to exactly one of the nine v2 cost owners.
        eff.strategy_kind = str(row.get("strategy_kind", "") or "")
    return effect_list


def _sample_resume_fingerprint(
    *,
    scenario_id: str,
    param_rows: List[Dict[str, object]],
    cfg: Dict[str, object],
    experiment_fingerprint: str = "",
) -> str:
    """Bind a reusable MC directory to the exact draw and solve options."""
    return build_resume_fingerprint(
        {
            "schema": 1,
            "runner": "S5_4_1_monte_carlo_full_variables",
            "scenario_id": str(scenario_id),
            "experiment_fingerprint": str(experiment_fingerprint),
            "param_rows": [dict(row) for row in param_rows],
            "model_options": {
                "future_last_only": bool(cfg.get("future_last_only", True)),
                "use_fao_modules": bool(cfg.get("use_fao_modules", True)),
                "use_linear": bool(cfg.get("use_linear", True)),
                "fast_emis_only": bool(cfg.get("fast_emis_only", True)),
                "fast_emis_year": int(cfg.get("fast_emis_year", 2080) or 2080),
                "mc_non_ef_mode": str(cfg.get("mc_non_ef_mode", "shared") or "shared"),
                "mc_ef_mode": str(cfg.get("mc_ef_mode", "shared") or "shared"),
                "ef_process_mode": str(cfg.get("ef_process_mode", "all") or "all"),
                "override_cfg": dict(cfg.get("override_cfg") or {}),
            },
        }
    )


def _experiment_resume_fingerprint(
    *,
    specs_df: pd.DataFrame,
    cfg: Dict[str, object],
    batch_state: Dict[str, object],
    q_bounds: Tuple[float, float],
) -> str:
    specs_records = (
        specs_df.astype(object)
        .where(pd.notna(specs_df), None)
        .to_dict("records")
    )
    return build_resume_fingerprint(
        {
            "schema": 1,
            "runner": "S5_4_1_monte_carlo_full_variables",
            "seed": int(cfg.get("seed", 42) or 42),
            "samples": int(cfg.get("samples", 0) or 0),
            "mc_sheet": str(cfg.get("mc_sheet_prefer", "") or ""),
            "nutrition_profile_sheet": str(
                cfg.get("nutrition_profile_sheet", "") or ""
            ),
            "sampling": dict(cfg.get("sampling") or {}),
            "quantile_bounds": [float(q_bounds[0]), float(q_bounds[1])],
            "specs": specs_records,
            "batch": {
                "count": int(batch_state.get("count", 1) or 1),
                "assignment": str(batch_state.get("assignment", "") or ""),
            },
            "model_options": {
                "future_last_only": bool(cfg.get("future_last_only", True)),
                "use_fao_modules": bool(cfg.get("use_fao_modules", True)),
                "use_linear": bool(cfg.get("use_linear", True)),
                "fast_emis_only": bool(cfg.get("fast_emis_only", True)),
                "fast_emis_year": int(cfg.get("fast_emis_year", 2080) or 2080),
                "mc_non_ef_mode": str(cfg.get("mc_non_ef_mode", "shared") or "shared"),
                "mc_ef_mode": str(cfg.get("mc_ef_mode", "shared") or "shared"),
                "ef_process_mode": str(cfg.get("ef_process_mode", "all") or "all"),
                "override_cfg": dict(cfg.get("override_cfg") or {}),
            },
        }
    )


def _resolve_effect_draw_for_logging(
    eff,
    *,
    mc_mode_default: str,
    mc_mode_ef: Optional[str] = None,
    ef_process_mode: str = "all",
) -> Tuple[Optional[float], Optional[float]]:
    spec = getattr(eff, "mc_bounds_raw", None) or getattr(eff, "mc_bounds", None) or {}
    pre_sampled = bool(spec and spec.get("pre_sampled", False))
    value_draw_out = getattr(eff, "value_2080", None)
    u_used = spec.get("u")
    kind = _normalize_kind(getattr(eff, "kind", ""))
    is_y2020 = bool(spec and (spec.get("lo_is_y2020") or spec.get("hi_is_y2020")))

    # EF rows in shared/per-commodity mode have a single row-level draw that should
    # match the active solve path instead of the original raw LHS row draw.
    if kind == "emission_factor" and spec and not pre_sampled:
        mode_ef_eff = str((spec or {}).get("mode") or mc_mode_ef or mc_mode_default or "shared").strip().lower() or "shared"
        process_mode_eff = str((spec or {}).get("ef_process_mode") or ef_process_mode).strip().lower() or "all"
        if mode_ef_eff in {"shared", "per_commodity"}:
            process_sel = str(getattr(eff, "process_selector", getattr(eff, "process_sel", "All")) or "All")
            commodity_sel = str(getattr(eff, "item_selector", getattr(eff, "commodity_sel", "")) or "")
            process_key = process_sel if process_mode_eff == "by_process" else "All"
            u_used = _u_for_ef(
                spec,
                eff,
                country="",
                commodity=commodity_sel,
                process_key=process_key,
                mode=mode_ef_eff,
            )

    if u_used is not None and not is_y2020 and spec and not pre_sampled:
        lo = spec.get("lo")
        hi = spec.get("hi")
        if lo is not None and hi is not None:
            value_draw_out = float(lo) + (float(hi) - float(lo)) * float(u_used)
    if is_y2020 and u_used is not None:
        value_draw_out = float(u_used)
    return value_draw_out, u_used


def _normalize_m49_key(val: object) -> str:
    s = str(val).strip() if val is not None else ""
    if not s:
        return ""
    if s.startswith("'"):
        return s
    if s.isdigit():
        return f"'{s.zfill(3)}"
    return f"'{s}"


def _expand_effect_rows_for_weighting(
    *,
    effects: Iterable,
    universe,
    baselines: Dict[str, object],
    calorie_weights: Dict[Tuple[str, str], float],
    scenario_id: str,
    sample_id: int,
    attempt: int,
    mc_mode_default: str,
    ef_co2eq_intensity_lookup: Optional[Dict[Tuple[str, str, str, str], float]] = None,
    mc_mode_non_ef: Optional[str] = None,
    mc_mode_ef: Optional[str] = None,
    ef_process_mode: str = "all",
) -> pd.DataFrame:
    region_members = _build_region_members(universe)
    rows: List[Dict[str, object]] = []

    def _normalize_mode(mode: str) -> str:
        mode_l = str(mode or "shared").strip().lower()
        if mode_l in ("a", "country", "per_country", "independent", "per-country"):
            return "per_country"
        if mode_l in ("per_commodity", "commodity", "per-item", "per_item"):
            return "per_commodity"
        if mode_l in (
            "per_country_commodity",
            "country_commodity",
            "per-country-commodity",
            "per_country_item",
        ):
            return "per_country_commodity"
        return "shared"

    mode_non_ef = _normalize_mode(mc_mode_non_ef or mc_mode_default or "shared")
    mode_ef = _normalize_mode(mc_mode_ef or mc_mode_default or "shared")
    ef_process_mode_l = str(ef_process_mode or "all").strip().lower()

    for eff in effects:
        kind = _normalize_kind(getattr(eff, "kind", ""))
        unit = getattr(eff, "unit", "")
        mc_unit = unit
        spec = getattr(eff, "mc_bounds_raw", None)
        countries = getattr(eff, "countries", None) or []
        commodities = getattr(eff, "commodities", None) or []
        processes = getattr(eff, "processes", None) or []
        ghg_sel = getattr(eff, "ghg_sel", "All") or "All"

        if kind == "emission_factor":
            for country in countries:
                m49_lookup = universe.m49_by_country.get(country)
                # select_countries may return either country names or normalized M49
                # codes. Fall back to the country value itself so EF baselines are
                # found when countries are already represented as M49 keys.
                m49_norm = _normalize_m49_key(m49_lookup if m49_lookup else country)
                for commodity in commodities:
                    weight_kcal = float(calorie_weights.get((country, commodity), 0.0) or 0.0)
                    for process in processes:
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
                        is_y2020 = bool(spec and (spec.get("lo_is_y2020") or spec.get("hi_is_y2020")))
                        pre_sampled = bool(spec and spec.get("pre_sampled", False))
                        # Keep weighting outputs aligned with the active EF solve path.
                        mode_ef_eff = str((spec or {}).get("mode") or mode_ef or "shared").strip().lower() or "shared"
                        process_mode_eff = (
                            str((spec or {}).get("ef_process_mode") or ef_process_mode_l).strip().lower() or "all"
                        )
                        u_used = None
                        if pre_sampled:
                            u_used = (spec or {}).get("u")
                        elif spec:
                            process_key = process if process_mode_eff == "by_process" else "All"
                            u_used = _u_for_ef(
                                spec,
                                eff,
                                country=country,
                                commodity=commodity,
                                process_key=process_key,
                                mode=mode_ef_eff,
                        )
                        value_draw_out = eff.value_2080
                        if u_used is not None and not is_y2020 and spec is not None and not pre_sampled:
                            lo = (spec or {}).get("lo")
                            hi = (spec or {}).get("hi")
                            if lo is not None and hi is not None:
                                value_draw_out = float(lo) + (float(hi) - float(lo)) * float(u_used)
                        if is_y2020 and u_used is not None:
                            value_draw_out = float(u_used)
                        min_bound_val = _calc_bound_value(
                            kind,
                            unit,
                            (spec or {}).get("lo"),
                            base_val,
                            is_y2020=bool((spec or {}).get("lo_is_y2020")),
                        )
                        max_bound_val = _calc_bound_value(
                            kind,
                            unit,
                            (spec or {}).get("hi"),
                            base_val,
                            is_y2020=bool((spec or {}).get("hi_is_y2020")),
                        )
                        if is_y2020 and u_used is not None and min_bound_val is not None and max_bound_val is not None:
                            value_sample = (
                                float(min_bound_val)
                                + (float(max_bound_val) - float(min_bound_val)) * float(u_used)
                            )
                            ratio = (
                                float(value_sample) / float(base_val)
                                if base_val is not None and np.isfinite(base_val) and float(base_val) != 0.0
                                else None
                            )
                        else:
                            ratio, value_sample = _calc_ratio(
                                kind,
                                unit,
                                value_draw_out,
                                base_val,
                                spec=None,
                                u_val=None,
                            )
                        co2eq_intensity_y2020 = _lookup_ef_co2eq_intensity(
                            ef_co2eq_intensity_lookup or {},
                            m49_norm,
                            commodity,
                            process,
                            ghg_sel,
                        )
                        co2eq_intensity_sample = None
                        if (
                            co2eq_intensity_y2020 is not None
                            and ratio is not None
                            and np.isfinite(ratio)
                        ):
                            co2eq_intensity_sample = float(co2eq_intensity_y2020) * float(ratio)
                        rows.append(
                            {
                                "scenario_id": scenario_id,
                                "sample_id": int(sample_id),
                                "attempt": int(attempt),
                                "spec_row_id": int(getattr(eff, "spec_row_id", -1)),
                                "element_name": str(getattr(eff, "element_name", "") or ""),
                                "kind": kind,
                                "element_unit": str(getattr(eff, "element_unit", "") or ""),
                                "item_selector": str(getattr(eff, "item_selector", "All") or "All"),
                                "process_selector": str(getattr(eff, "process_selector", "All") or "All"),
                                "ghg_selector": str(getattr(eff, "ghg_selector", "All") or "All"),
                                "region_selector": str(getattr(eff, "region_selector", "All") or "All"),
                                "country": str(country),
                                "commodity": str(commodity),
                                "process": str(process),
                                "ghg": str(ghg_sel),
                                "mc_unit": mc_unit,
                                "value_unit": value_unit,
                                "value_y2020": base_val,
                                "value_draw": value_draw_out,
                                "value_sample": value_sample,
                                "ratio": ratio,
                                "mc_u": u_used,
                                "min_bound": (spec or {}).get("lo"),
                                "max_bound": (spec or {}).get("hi"),
                                "min_bound_value": min_bound_val,
                                "max_bound_value": max_bound_val,
                                "weight_kcal": weight_kcal,
                                "co2eq_intensity_y2020_kg_per_kcal": co2eq_intensity_y2020,
                                "co2eq_intensity_sample_kg_per_kcal": co2eq_intensity_sample,
                                "co2eq_intensity_unit": (
                                    "kg CO2eq/kcal" if co2eq_intensity_y2020 is not None else ""
                                ),
                            }
                        )
        else:
            for country in countries:
                for commodity in commodities:
                    weight_kcal = float(calorie_weights.get((country, commodity), 0.0) or 0.0)
                    base_val = _lookup_base_value(kind, country, commodity, baselines, region_members)
                    value_unit = _value_unit_for(kind, country, commodity, baselines, region_members)
                    is_y2020 = bool(spec and (spec.get("lo_is_y2020") or spec.get("hi_is_y2020")))
                    pre_sampled = bool(spec and spec.get("pre_sampled", False))
                    # Keep weighting outputs aligned with the currently active sampled effect.
                    u_used = (spec or {}).get("u")
                    value_draw_out = eff.value_2080
                    if u_used is not None and not is_y2020 and spec is not None and not pre_sampled:
                        lo = (spec or {}).get("lo")
                        hi = (spec or {}).get("hi")
                        if lo is not None and hi is not None:
                            value_draw_out = float(lo) + (float(hi) - float(lo)) * float(u_used)
                    if is_y2020 and u_used is not None:
                        value_draw_out = float(u_used)
                    min_bound_val = _calc_bound_value(
                        kind,
                        unit,
                        (spec or {}).get("lo"),
                        base_val,
                        is_y2020=bool((spec or {}).get("lo_is_y2020")),
                    )
                    max_bound_val = _calc_bound_value(
                        kind,
                        unit,
                        (spec or {}).get("hi"),
                        base_val,
                        is_y2020=bool((spec or {}).get("hi_is_y2020")),
                    )
                    if is_y2020 and u_used is not None and min_bound_val is not None and max_bound_val is not None:
                        value_sample = (
                            float(min_bound_val)
                            + (float(max_bound_val) - float(min_bound_val)) * float(u_used)
                        )
                        ratio = (
                            float(value_sample) / float(base_val)
                            if base_val is not None and np.isfinite(base_val) and float(base_val) != 0.0
                            else None
                        )
                    else:
                        ratio, value_sample = _calc_ratio(
                            kind,
                            unit,
                            value_draw_out,
                            base_val,
                            spec=None,
                            u_val=None,
                        )
                    rows.append(
                        {
                            "scenario_id": scenario_id,
                            "sample_id": int(sample_id),
                            "attempt": int(attempt),
                            "spec_row_id": int(getattr(eff, "spec_row_id", -1)),
                            "element_name": str(getattr(eff, "element_name", "") or ""),
                            "kind": kind,
                            "element_unit": str(getattr(eff, "element_unit", "") or ""),
                            "item_selector": str(getattr(eff, "item_selector", "All") or "All"),
                            "process_selector": str(getattr(eff, "process_selector", "All") or "All"),
                            "ghg_selector": str(getattr(eff, "ghg_selector", "All") or "All"),
                            "region_selector": str(getattr(eff, "region_selector", "All") or "All"),
                            "country": str(country),
                            "commodity": str(commodity),
                            "process": "",
                            "ghg": "",
                            "mc_unit": mc_unit,
                            "value_unit": value_unit,
                            "value_y2020": base_val,
                            "value_draw": value_draw_out,
                            "value_sample": value_sample,
                            "ratio": ratio,
                            "mc_u": u_used,
                            "min_bound": (spec or {}).get("lo"),
                            "max_bound": (spec or {}).get("hi"),
                            "min_bound_value": min_bound_val,
                            "max_bound_value": max_bound_val,
                            "weight_kcal": weight_kcal,
                        }
                    )
    return pd.DataFrame(rows)


def _aggregate_weighted_elements(
    expanded_df: pd.DataFrame,
    *,
    batch_state: Optional[Dict[str, object]] = None,
    require_ef_intensity: bool = False,
) -> List[Dict[str, object]]:
    if not isinstance(expanded_df, pd.DataFrame) or expanded_df.empty:
        return []
    group_cols = [
        "scenario_id",
        "sample_id",
        "attempt",
        "spec_row_id",
        "element_name",
        "kind",
        "element_unit",
        "item_selector",
        "process_selector",
        "ghg_selector",
        "region_selector",
        "mc_unit",
        "value_unit",
    ]
    rows: List[Dict[str, object]] = []
    work = expanded_df.copy()
    numeric_cols = (
        "value_y2020",
        "value_draw",
        "value_sample",
        "ratio",
        "weight_kcal",
        "co2eq_intensity_y2020_kg_per_kcal",
        "co2eq_intensity_sample_kg_per_kcal",
    )
    for col in numeric_cols:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")
    for keys, grp in work.groupby(group_cols, dropna=False):
        grp = grp.copy()
        valid = grp["value_sample"].notna()
        if str(grp["kind"].iloc[0]).strip().lower() == "emission_factor":
            # For EF rate draws, rows without a physical baseline only carry the
            # sampled relative rate. They are valid for ratios but not for the
            # absolute weighted_value_sample/y2020 used by absolute quartiles.
            valid = valid & grp["value_y2020"].notna()
        if not valid.any():
            continue
        weights = grp["weight_kcal"].where(grp["weight_kcal"].notna() & (grp["weight_kcal"] > 0), 0.0)
        weight_sum = float(weights[valid].sum())
        if weight_sum > 0:
            weighted_sample = float(np.average(grp.loc[valid, "value_sample"], weights=weights[valid]))
            weighted_method = "calorie_weighted"
        else:
            weighted_sample = float(grp.loc[valid, "value_sample"].mean())
            weighted_method = "unweighted_mean"
        result = dict(zip(group_cols, keys))
        result["selected_pairs"] = int(valid.sum())
        result["weight_basis"] = "production_t_2020_x_kcal_per_ton"
        result["weight_kcal_total"] = weight_sum
        result["aggregation_method"] = weighted_method
        result["weighted_value_sample"] = weighted_sample
        result["weighted_co2eq_intensity_y2020_kg_per_kcal"] = None
        result["weighted_co2eq_intensity_sample_kg_per_kcal"] = None
        result["co2eq_intensity_unit"] = ""

        for src_col, out_col in (
            ("value_y2020", "weighted_value_y2020"),
            ("value_draw", "weighted_value_draw"),
            ("ratio", "weighted_ratio"),
            ("co2eq_intensity_y2020_kg_per_kcal", "weighted_co2eq_intensity_y2020_kg_per_kcal"),
            ("co2eq_intensity_sample_kg_per_kcal", "weighted_co2eq_intensity_sample_kg_per_kcal"),
        ):
            src_valid = grp[src_col].notna() & valid
            if not src_valid.any():
                result[out_col] = None
                continue
            if weight_sum > 0 and float(weights[src_valid].sum()) > 0:
                result[out_col] = float(np.average(grp.loc[src_valid, src_col], weights=weights[src_valid]))
            else:
                result[out_col] = float(grp.loc[src_valid, src_col].mean())
        if result.get("weighted_co2eq_intensity_sample_kg_per_kcal") is not None:
            result["co2eq_intensity_unit"] = "kg CO2eq/kcal"
        if batch_state:
            _apply_batch_meta(result, batch_state)
        rows.append(result)
    if require_ef_intensity:
        ef_rows = [
            row
            for row in rows
            if str(row.get("kind", "")).strip().lower() == "emission_factor"
        ]
        has_intensity = any(
            pd.notna(row.get("weighted_co2eq_intensity_sample_kg_per_kcal"))
            for row in ef_rows
        )
        if ef_rows and not has_intensity:
            first = ef_rows[0]
            raise RuntimeError(
                "EF absolute-intensity output is required, but this sample produced no "
                "weighted_co2eq_intensity_sample_kg_per_kcal values. "
                f"scenario_id={first.get('scenario_id')} sample_id={first.get('sample_id')}. "
                "Check the EF baseline emissions lookup path and GHG labels."
            )
    return rows


def _stamp_success_provenance(
    rows: List[Dict[str, object]],
    *,
    run_id: str,
    experiment_fingerprint: str,
    resume_fingerprint: str,
) -> List[Dict[str, object]]:
    for row in rows:
        row["run_id"] = str(run_id)
        row["experiment_fingerprint"] = str(experiment_fingerprint)
        row["resume_fingerprint"] = str(resume_fingerprint)
    return rows


def _append_fast_summary(
    summary_rows: List[Dict[str, object]],
    fast_df: pd.DataFrame,
    *,
    scenario_id: str,
    sample_id: int,
    scenario_dir: Path,
    run_id: str = "",
    experiment_fingerprint: str = "",
    resume_fingerprint: str = "",
    batch_state: Optional[Dict[str, object]] = None,
) -> None:
    if fast_df is None or fast_df.empty:
        return
    work = fast_df.copy()
    for col in ("year", "total_co2eq_kt", "total_co2eq_gt"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    for row in work.to_dict("records"):
        row["scenario_id"] = scenario_id
        row["sample_id"] = int(sample_id)
        row["scenario_dir"] = str(scenario_dir)
        row["run_id"] = str(run_id)
        row["experiment_fingerprint"] = str(experiment_fingerprint)
        row["resume_fingerprint"] = str(resume_fingerprint)
        if batch_state:
            _apply_batch_meta(row, batch_state)
        summary_rows.append(row)


def _append_global_process_rows(
    process_rows: List[Dict[str, object]],
    detail_df: pd.DataFrame,
    *,
    scenario_id: str,
    sample_id: int,
    scenario_dir: Path,
    run_id: str = "",
    experiment_fingerprint: str = "",
    resume_fingerprint: str = "",
    batch_state: Optional[Dict[str, object]] = None,
) -> None:
    if detail_df is None or detail_df.empty or "co2eq_kt" not in detail_df.columns:
        return
    work = detail_df.copy()
    for col in ("year", "co2eq_kt"):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    if "row_type" in work.columns:
        work = work[work["row_type"].astype(str).str.strip().ne("co2eq_summary")].copy()
    keep_cols = [c for c in ("year", "source_module", "Process", "co2eq_kt") if c in work.columns]
    if len(keep_cols) < 4:
        return
    grouped = work.groupby(["year", "source_module", "Process"], as_index=False)["co2eq_kt"].sum()
    grouped["co2eq_gt"] = grouped["co2eq_kt"] * 1e-6
    grouped["scenario_id"] = scenario_id
    grouped["sample_id"] = int(sample_id)
    grouped["scenario_dir"] = str(scenario_dir)
    grouped["run_id"] = str(run_id)
    grouped["experiment_fingerprint"] = str(experiment_fingerprint)
    grouped["resume_fingerprint"] = str(resume_fingerprint)
    if batch_state:
        grouped["batch_index"] = int(batch_state["index"])
        grouped["batch_count"] = int(batch_state["count"])
        grouped["batch_tag"] = str(batch_state["tag"])
        ordered = [
            "scenario_id",
            "sample_id",
            "run_id",
            "experiment_fingerprint",
            "resume_fingerprint",
            "batch_index",
            "batch_count",
            "batch_tag",
            "scenario_dir",
            "year",
            "source_module",
            "Process",
            "co2eq_kt",
            "co2eq_gt",
        ]
    else:
        ordered = [
            "scenario_id", "sample_id", "run_id", "experiment_fingerprint",
            "resume_fingerprint", "scenario_dir", "year", "source_module",
            "Process", "co2eq_kt", "co2eq_gt",
        ]
    process_rows.extend(grouped[ordered].to_dict("records"))


def _append_realized_ruminant_rows(
    realized_rows: List[Dict[str, object]],
    realized_df: pd.DataFrame,
    *,
    scenario_id: str,
    sample_id: int,
    scenario_dir: Path,
    run_id: str = "",
    experiment_fingerprint: str = "",
    resume_fingerprint: str = "",
    batch_state: Optional[Dict[str, object]] = None,
) -> None:
    if realized_df is None or realized_df.empty:
        return
    work = realized_df.copy()
    keep_cols = [
        c
        for c in (
            "year",
            "demand_t",
            "demand_kcal",
            "ruminant_demand_t",
            "ruminant_demand_kcal",
            "realized_ruminant_share_kcal",
            "realized_ruminant_share_t",
        )
        if c in work.columns
    ]
    if "year" not in keep_cols or "realized_ruminant_share_kcal" not in keep_cols:
        return
    work = work[keep_cols].copy()
    for col in keep_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["year", "realized_ruminant_share_kcal"]).copy()
    if work.empty:
        return
    work["year"] = work["year"].astype(int)
    work["scenario_id"] = scenario_id
    work["sample_id"] = int(sample_id)
    work["scenario_dir"] = str(scenario_dir)
    work["run_id"] = str(run_id)
    work["experiment_fingerprint"] = str(experiment_fingerprint)
    work["resume_fingerprint"] = str(resume_fingerprint)
    if batch_state:
        work["batch_index"] = int(batch_state["index"])
        work["batch_count"] = int(batch_state["count"])
        work["batch_tag"] = str(batch_state["tag"])
    realized_rows.extend(work.to_dict("records"))


LAND_BALANCE_KEEP_COLS = [
    "year",
    "source",
    "crop_area_need_ha",
    "grass_area_need_ha",
    "d_crop_area_need_ha",
    "d_grass_area_need_ha",
    "d_forest_area_balance_ha",
    "crop_area_ha",
    "pasture_area_ha",
    "forest_area_ha",
    "base_cropland_ha",
    "new_cropland_ha",
    "d_cropland_ha",
    "base_grass_need_ha",
    "yr_grass_need_ha",
    "d_grassland_ha",
    "d_forest_ha",
    "feed_area_delta_ha",
    "grass_to_crop_ha",
    "grass_to_forest_ha",
    "crop_to_grass_ha",
    "crop_to_forest_ha",
    "forest_to_crop_ha",
    "forest_to_grass_ha",
]


def _append_land_balance_rows(
    land_rows: List[Dict[str, object]],
    land_df: pd.DataFrame,
    *,
    scenario_id: str,
    sample_id: int,
    scenario_dir: Path,
    run_id: str = "",
    experiment_fingerprint: str = "",
    resume_fingerprint: str = "",
    batch_state: Optional[Dict[str, object]] = None,
) -> None:
    if land_df is None or land_df.empty:
        return
    keep_cols = [c for c in LAND_BALANCE_KEEP_COLS if c in land_df.columns]
    if "year" not in keep_cols:
        return
    work = land_df[keep_cols].copy()
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work = work.dropna(subset=["year"]).copy()
    if work.empty:
        return
    work["year"] = work["year"].astype(int)
    for col in keep_cols:
        if col not in {"source"}:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    if "source" not in work.columns:
        work["source"] = ""
    work["scenario_id"] = scenario_id
    work["sample_id"] = int(sample_id)
    work["scenario_dir"] = str(scenario_dir)
    work["run_id"] = str(run_id)
    work["experiment_fingerprint"] = str(experiment_fingerprint)
    work["resume_fingerprint"] = str(resume_fingerprint)
    if batch_state:
        work["batch_index"] = int(batch_state["index"])
        work["batch_count"] = int(batch_state["count"])
        work["batch_tag"] = str(batch_state["tag"])
    land_rows.extend(work.to_dict("records"))


def main() -> None:
    cfg = copy.deepcopy(CONFIG)
    CFG["nutrition_profile_sheet"] = str(cfg.get("nutrition_profile_sheet", "low_land_new") or "low_land_new")
    root_output_dir = _resolve_fullmc_output_root(cfg.get("output_dir", ""))
    _sync_fullmc_output_environment(root_output_dir)
    batch_state = _resolve_batch_settings(cfg)

    if bool(batch_state.get("enabled")):
        output_dir = root_output_dir / str(batch_state["batches_subdir"]) / str(batch_state["tag"])
    else:
        output_dir = root_output_dir
    runs_dir = output_dir / str(cfg.get("runs_subdir") or "runs")
    sample_wb_dir = output_dir / str(cfg.get("sample_workbook_subdir") or "MC")
    status_out_path = output_dir / str(cfg.get("status_csv") or "mc_sample_status.csv")
    draws_out_path = output_dir / str(cfg.get("draws_csv") or "mc_draws_long.csv")
    summary_out_path = output_dir / str(cfg.get("success_summary_csv") or "mc_success_fast_summary.csv")
    process_out_path = output_dir / str(cfg.get("success_process_csv") or "mc_success_global_process_co2eq.csv")
    weighted_out_path = output_dir / str(cfg.get("success_weighted_elements_csv") or "mc_success_weighted_elements.csv")
    realized_ruminant_out_path = output_dir / str(
        cfg.get("success_realized_ruminant_csv") or "mc_success_realized_ruminant_share.csv"
    )
    land_balance_out_path = output_dir / str(
        cfg.get("success_land_balance_csv") or "mc_success_crop_pasture_land_balance.csv"
    )
    resume_enabled = bool(cfg.get("resume", False))

    _ensure_dir(output_dir)
    if cfg.get("save_per_run_dirs", True):
        _ensure_dir(runs_dir)
    if cfg.get("save_sample_workbook", False):
        _ensure_dir(sample_wb_dir)
    _reset_output_file(status_out_path)
    _reset_output_file(draws_out_path)
    _reset_output_file(summary_out_path)
    _reset_output_file(process_out_path)
    _reset_output_file(weighted_out_path)
    _reset_output_file(realized_ruminant_out_path)
    _reset_output_file(land_balance_out_path)

    paths = DataPaths()
    shared_cfg = ScenarioConfig()
    shared_universe = build_universe_from_dict_v3(paths.dict_v3_path, shared_cfg)
    base_year = int(shared_cfg.years_hist_end or 2020)

    mc_sheet = resolve_mc_effect_sheet(
        cfg.get("mc_sheet_prefer", "MC_effect_low_land_new"),
        nutrition_profile_sheet=cfg.get("nutrition_profile_sheet", "low_land_new"),
    )
    specs_df = _load_mc_specs_effect(
        paths.scenario_config_xlsx,
        prefer_sheet=mc_sheet,
        nutrition_profile_sheet=cfg.get("nutrition_profile_sheet", "low_land_new"),
    )
    specs_df = _normalize_mc_specs(
        specs_df,
        aggregate_non_ef=bool(cfg.get("aggregate_non_ef", False)),
    ).reset_index(drop=True)
    specs_df["spec_row_id"] = np.arange(1, len(specs_df) + 1)
    if specs_df.empty:
        raise RuntimeError(f"{mc_sheet} sheet is empty; cannot run S5_4.")

    n_samples = int(cfg.get("samples", 0) or 0)
    if n_samples <= 0:
        raise ValueError("samples 必须为正整数。")
    max_runs = cfg.get("max_runs")
    if max_runs is not None:
        max_runs = int(max_runs)
        if max_runs <= 0:
            raise ValueError("max_runs 必须为正整数或 None。")

    sample_indices = _select_batch_sample_indices(
        total_samples=n_samples,
        batch_count=int(batch_state["count"]),
        batch_index=int(batch_state["index"]),
        assignment=str(batch_state["assignment"]),
    )

    print(f"[S5_4] MC sheet in use: {mc_sheet} rows={len(specs_df)}")
    print(f"[S5_4] requested samples={n_samples}")
    print(f"[S5_4] root_output_dir={root_output_dir}")
    print(f"[S5_4] NZF_OUTPUT_DIR={os.environ.get('NZF_OUTPUT_DIR', '')}")
    print(f"[S5_4] output_dir={output_dir}")
    if bool(batch_state.get("enabled")):
        print(
            f"[S5_4] batch={batch_state['tag']} assignment={batch_state['assignment']} "
            f"assigned_samples={len(sample_indices)}/{n_samples}"
        )
    if resume_enabled:
        print("[S5_4] resume enabled: rebuild CSV outputs and reuse completed run directories")
    elif bool(cfg.get("clear_existing_run_dirs_when_no_resume", True)):
        print("[S5_4] resume disabled: clear existing per-scenario run dirs before rerun")

    unit_matrix = _sample_unit_matrix_for_specs(
        specs_df,
        n_samples,
        seed=int(cfg.get("seed", 42) or 42),
        config=cfg.get("sampling", {}) or {},
    )
    q_bounds = cfg.get("sampling", {}).get("quantile_bounds", (0.0, 1.0))
    try:
        q_bounds = (float(q_bounds[0]), float(q_bounds[1]))
    except Exception:
        q_bounds = (0.0, 1.0)
    experiment_resume_fingerprint = _experiment_resume_fingerprint(
        specs_df=specs_df,
        cfg=cfg,
        batch_state=batch_state,
        q_bounds=q_bounds,
    )
    print(
        "[S5_4] experiment_fingerprint="
        f"{experiment_resume_fingerprint[:16]}..."
    )

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
    baselines = _load_mc_baselines(paths, shared_universe, base_year=base_year)
    calorie_weights = _build_calorie_weight_lookup(
        paths=paths,
        universe=shared_universe,
        run_cache=shared_run_cache,
        base_year=base_year,
    )
    ef_co2eq_intensity_lookup: Optional[Dict[Tuple[str, str, str, str], float]] = None
    print(
        f"[S5_4] shared cache: nodes={len(shared_run_cache.get('node_blueprint') or [])}, "
        f"calorie_weights={len(calorie_weights)}, "
        "ef_co2eq_intensities=lazy"
    )
    failed_diag_budget = max(0, int(cfg.get("failed_diag_sample_n", 0) or 0))
    failed_diag_used = 0
    diag_runs_dir = output_dir / str(cfg.get("failed_diag_runs_subdir") or "runs_diag")
    if failed_diag_budget > 0:
        _ensure_dir(diag_runs_dir)

    write_every = int(cfg.get("write_every_n_runs", 1) or 1)
    if write_every <= 0:
        write_every = 1

    status_rows: List[Dict[str, object]] = []
    draw_rows: List[Dict[str, object]] = []
    success_summary_rows: List[Dict[str, object]] = []
    success_process_rows: List[Dict[str, object]] = []
    success_weighted_rows: List[Dict[str, object]] = []
    success_realized_ruminant_rows: List[Dict[str, object]] = []
    success_land_balance_rows: List[Dict[str, object]] = []
    total_written = 0
    success_count = 0
    resume_count = 0
    infeasible_count = 0
    nonoptimal_count = 0
    failed_count = 0
    precheck_failed_count = 0
    interrupted_count = 0
    draws_written = 0
    summary_written = 0
    process_written = 0
    weighted_written = 0
    realized_ruminant_written = 0
    land_balance_written = 0

    def _register_status(row_obj: Dict[str, object]) -> None:
        nonlocal success_count, infeasible_count, nonoptimal_count, failed_count
        nonlocal precheck_failed_count, interrupted_count
        status = str(row_obj.get("run_status") or "")
        if status in {"ok", "resumed"}:
            success_count += 1
        elif status == "infeasible":
            infeasible_count += 1
        elif status == "nonoptimal":
            nonoptimal_count += 1
        elif status in {"failed", "invalid_market_balance", "invalid_fast_emissions", "missing_fast_summary"}:
            failed_count += 1
        elif status == "precheck_failed":
            precheck_failed_count += 1
        elif status == "interrupted":
            interrupted_count += 1

    def _flush_buffers() -> None:
        nonlocal total_written, draws_written, summary_written, process_written, weighted_written
        nonlocal realized_ruminant_written, land_balance_written
        nonlocal status_rows, draw_rows, success_summary_rows, success_process_rows, success_weighted_rows
        nonlocal success_realized_ruminant_rows, success_land_balance_rows
        if status_rows:
            total_written += _append_rows_csv(status_rows, status_out_path)
            status_rows = []
        if draw_rows:
            draws_written += _append_rows_csv(draw_rows, draws_out_path)
            draw_rows = []
        if success_summary_rows:
            summary_written += _append_rows_csv(success_summary_rows, summary_out_path)
            success_summary_rows = []
        if success_process_rows:
            process_written += _append_rows_csv(success_process_rows, process_out_path)
            success_process_rows = []
        if success_weighted_rows:
            weighted_written += _append_rows_csv(success_weighted_rows, weighted_out_path)
            success_weighted_rows = []
        if success_realized_ruminant_rows:
            realized_ruminant_written += _append_rows_csv(
                success_realized_ruminant_rows,
                realized_ruminant_out_path,
            )
            success_realized_ruminant_rows = []
        if success_land_balance_rows:
            land_balance_written += _append_rows_csv(
                success_land_balance_rows,
                land_balance_out_path,
            )
            success_land_balance_rows = []

    def _maybe_run_diagnostics(row_obj: Dict[str, object], effects_obj) -> None:
        nonlocal failed_diag_used
        if failed_diag_used >= failed_diag_budget or effects_obj is None:
            return
        try:
            diag_outdir = _rerun_failed_point_with_diagnostics(
                cfg=cfg,
                paths=paths,
                shared_cfg=shared_cfg,
                shared_universe=shared_universe,
                shared_run_cache=shared_run_cache,
                diag_runs_dir=diag_runs_dir,
                scenario_id=str(row_obj.get("scenario_id") or ""),
                effects=effects_obj,
            )
            if diag_outdir:
                row_obj["diagnostic_scenario_dir"] = str(diag_outdir)
                failed_diag_used += 1
                print(
                    f"[S5_4][DIAG] {row_obj.get('scenario_id')} -> "
                    f"detailed rerun saved at {diag_outdir}"
                )
        except Exception as diag_exc:
            print(f"[S5_4][DIAG-FAILED] {row_obj.get('scenario_id')}: {diag_exc}")

    def _get_ef_co2eq_intensity_lookup() -> Dict[Tuple[str, str, str, str], float]:
        nonlocal ef_co2eq_intensity_lookup
        if ef_co2eq_intensity_lookup is None:
            ef_co2eq_intensity_lookup = _build_ef_co2eq_intensity_lookup(
                cfg=cfg,
                universe=shared_universe,
                calorie_weights=calorie_weights,
                base_year=base_year,
            )
        return ef_co2eq_intensity_lookup

    def _expand_success_weighting_rows(
        effects_obj,
        *,
        scenario_id: str,
        sample_id: int,
        attempt: int,
    ) -> pd.DataFrame:
        return _expand_effect_rows_for_weighting(
            effects=effects_obj,
            universe=shared_universe,
            baselines=baselines,
            calorie_weights=calorie_weights,
            ef_co2eq_intensity_lookup=_get_ef_co2eq_intensity_lookup(),
            scenario_id=scenario_id,
            sample_id=sample_id,
            attempt=attempt,
            mc_mode_default=str(cfg.get("mc_non_ef_mode", "shared") or "shared"),
            mc_mode_non_ef=str(cfg.get("mc_non_ef_mode", "shared") or "shared"),
            mc_mode_ef=str(cfg.get("mc_ef_mode", "shared") or "shared"),
            ef_process_mode=str(cfg.get("ef_process_mode", "all") or "all"),
        )

    try:
        processed_runs = 0
        assigned_total = len(sample_indices)
        for local_idx, sample_idx in enumerate(sample_indices, start=1):
            if max_runs is not None and processed_runs >= max_runs:
                break
            sample_id = sample_idx + 1
            scenario_id = _scenario_id(sample_id)
            scenario_dir = runs_dir / scenario_id
            print(f"[S5_4] ({local_idx}/{assigned_total} | global {sample_id}/{n_samples}) {scenario_id}")

            status_row: Dict[str, object] = _apply_batch_meta({
                "scenario_id": scenario_id,
                "sample_id": sample_id,
                "attempt": 1,
                "target_year": int(cfg.get("fast_emis_year", 2080) or 2080),
                "run_status": "pending",
                "scenario_dir": str(scenario_dir),
                "run_id": "",
                "experiment_fingerprint": experiment_resume_fingerprint,
                "resume_fingerprint": "",
                "diagnostic_scenario_dir": "",
                "afolu_emissions_gt_co2eq_yr": None,
                "market_shortage_t": None,
                "market_gap_rate": None,
                "market_gap_year": None,
                "model_status_code": None,
                "model_status_text": "",
                "iis_summary": "",
                "error_type": "",
                "error_message": "",
            }, batch_state)
            unit_row = None
            param_rows = None
            effects = None
            expanded_df = None
            scenario_path = None
            fast_df = None
            fast_detail_df = None
            log_diag = None
            neg_msg = None
            outdir = None

            try:
                unit_row = unit_matrix[sample_idx] if sample_idx < len(unit_matrix) else np.zeros((len(specs_df),))
                param_rows = _draw_mc_param_rows(
                    specs_df,
                    unit_row=unit_row,
                    quantile_bounds=q_bounds,
                    group_q_bounds=None,
                    group_lookup=None,
                    row_q_bounds=None,
                    sampling_cfg=cfg.get("sampling", {}) or {},
                )
                param_rows = _attach_param_metadata(param_rows, specs_df)

                effects = _build_scenario_effects(
                    param_rows,
                    shared_universe,
                    scenario_id=scenario_id,
                    mc_y2020_mode=str(cfg.get("mc_non_ef_mode", "shared") or "shared"),
                    mc_mode_non_ef=str(cfg.get("mc_non_ef_mode", "shared") or "shared"),
                    mc_mode_ef=str(cfg.get("mc_ef_mode", "shared") or "shared"),
                    ef_process_mode=str(cfg.get("ef_process_mode", "all") or "all"),
                )
                effects = _attach_effect_metadata(effects, param_rows)
                scenario_resume_fingerprint = _sample_resume_fingerprint(
                    scenario_id=scenario_id,
                    param_rows=param_rows,
                    cfg=cfg,
                    experiment_fingerprint=experiment_resume_fingerprint,
                )
                status_row["resume_fingerprint"] = scenario_resume_fingerprint
                for eff in effects:
                    value_draw_out, u_used = _resolve_effect_draw_for_logging(
                        eff,
                        mc_mode_default=str(cfg.get("mc_non_ef_mode", "shared") or "shared"),
                        mc_mode_ef=str(cfg.get("mc_ef_mode", "shared") or "shared"),
                        ef_process_mode=str(cfg.get("ef_process_mode", "all") or "all"),
                    )
                    spec = getattr(eff, "mc_bounds_raw", None) or getattr(eff, "mc_bounds", None) or {}
                    draw_rows.append(
                        _apply_batch_meta(
                            {
                                "scenario_id": scenario_id,
                                "sample_id": sample_id,
                                "attempt": 1,
                                "spec_row_id": int(getattr(eff, "spec_row_id", -1)),
                                "element_name": str(getattr(eff, "element_name", "") or ""),
                                "kind": str(getattr(eff, "kind", "") or ""),
                                "element_unit": str(getattr(eff, "element_unit", "") or ""),
                                "item_selector": str(getattr(eff, "item_selector", "All") or "All"),
                                "process_selector": str(getattr(eff, "process_selector", "All") or "All"),
                                "ghg_selector": str(getattr(eff, "ghg_selector", "All") or "All"),
                                "region_selector": str(getattr(eff, "region_selector", "All") or "All"),
                                "mc_unit": str(getattr(eff, "unit", "") or ""),
                                "value_draw": value_draw_out,
                                "min_bound": spec.get("lo"),
                                "max_bound": spec.get("hi"),
                                "min_is_y2020": spec.get("lo_is_y2020"),
                                "max_is_y2020": spec.get("hi_is_y2020"),
                                "mc_u": u_used,
                                "q_low": spec.get("q_low"),
                                "q_high": spec.get("q_high"),
                                "experiment_fingerprint": experiment_resume_fingerprint,
                                "resume_fingerprint": scenario_resume_fingerprint,
                            },
                            batch_state,
                        )
                    )

                if bool(cfg.get("save_sample_workbook", False)):
                    rows_by_kind = _build_mc_sample_rows(
                        effects,
                        universe=shared_universe,
                        baselines=baselines,
                        scenario_id=scenario_id,
                        sample_id=sample_id,
                        attempt=1,
                        mc_mode_default=str(cfg.get("mc_non_ef_mode", "shared") or "shared"),
                        mc_mode_non_ef=str(cfg.get("mc_non_ef_mode", "shared") or "shared"),
                        mc_mode_ef=str(cfg.get("mc_ef_mode", "shared") or "shared"),
                        ef_process_mode=str(cfg.get("ef_process_mode", "all") or "all"),
                    )
                    _write_mc_sample_xlsx(
                        sample_wb_dir,
                        scenario_id=scenario_id,
                        sample_id=sample_id,
                        attempt=1,
                        rows_by_kind=rows_by_kind,
                    )

                if (
                    not resume_enabled
                    and bool(cfg.get("clear_existing_run_dirs_when_no_resume", True))
                ):
                    _clear_existing_scenario_dir(scenario_dir, runs_dir)

                resume_validation: Optional[ResumeValidation] = None
                if resume_enabled:
                    resume_validation = validate_run_for_resume(
                        scenario_dir,
                        expected_scenario_id=scenario_id,
                        expected_resume_fingerprint=scenario_resume_fingerprint,
                    )
                    if not resume_validation.allowed:
                        print(
                            f"[S5_4][RESUME-RERUN] {scenario_id}: "
                            f"structured run validation rejected reuse "
                            f"({resume_validation.reason})"
                        )
                        if bool(cfg.get("clear_existing_run_dirs_on_resume_retry", True)):
                            if _clear_existing_scenario_dir(scenario_dir, runs_dir):
                                print(f"[S5_4][RESUME-RERUN] cleared old run dir: {scenario_dir}")

                if resume_enabled and resume_validation is not None and resume_validation.allowed:
                    scenario_path = scenario_dir
                    status_row["run_id"] = resume_validation.run_id
                    log_path = scenario_path / "Log" / "model.log"
                    if artifact_matches_validated_run(log_path, resume_validation):
                        log_diag = _apply_log_diagnostics(status_row, scenario_path)
                    else:
                        solver_meta = (resume_validation.payload or {}).get("solver") or {}
                        status_row["model_status_code"] = solver_meta.get("status_code")
                        status_row["model_status_text"] = str(
                            solver_meta.get("status_name") or ""
                        ).upper()
                        log_diag = {
                            "run_status_hint": "",
                            "model_status_code": solver_meta.get("status_code"),
                            "model_status_text": status_row["model_status_text"],
                        }
                    existing_total_gt = None

                    status_hint = str(log_diag.get("run_status_hint") or "")
                    should_rerun_resume = False
                    if status_hint == "infeasible" and bool(cfg.get("rerun_infeasible_on_resume", True)):
                        should_rerun_resume = True
                        print(f"[S5_4][RESUME-RERUN] {scenario_id}: old infeasible run will be rerun")
                    elif status_hint == "nonoptimal" and _is_retryable_resume_status(cfg, log_diag):
                        should_rerun_resume = True
                        print(
                            f"[S5_4][RESUME-RERUN] {scenario_id}: "
                            f"old {status_row.get('model_status_text') or 'NONOPTIMAL'} run will be rerun"
                        )
                    if should_rerun_resume:
                        if bool(cfg.get("clear_existing_run_dirs_on_resume_retry", True)):
                            if _clear_existing_scenario_dir(scenario_path, runs_dir):
                                print(f"[S5_4][RESUME-RERUN] cleared old run dir: {scenario_path}")
                    elif _apply_model_status_failure(status_row, log_diag):
                        _maybe_run_diagnostics(status_row, effects)
                        pass
                    else:
                        fast_df = _read_validated_fast_summary_df(
                            scenario_path,
                            resume_validation,
                        )
                        fast_detail_df = _read_validated_fast_global_detail_df(
                            scenario_path,
                            resume_validation,
                        )
                        market_diag_path = (
                            scenario_path
                            / "Diagnostics"
                            / "commodity_balance_by_commodity.csv"
                        )
                        resume_artifacts_valid = bool(
                            not fast_df.empty
                            and not fast_detail_df.empty
                            and artifact_matches_validated_run(
                                market_diag_path,
                                resume_validation,
                            )
                        )
                        neg_msg = (
                            _validate_nonluc_fast_emissions(scenario_path)
                            if resume_artifacts_valid
                            else None
                        )
                        if resume_artifacts_valid:
                            status_row.update(_read_market_gap_summary(scenario_path))
                            gap_msg = _validate_market_balance_gap(
                                scenario_path,
                                max_gap_rate=float((cfg.get("override_cfg") or {}).get("market_gap_max_rate", 0.10)),
                            )
                        else:
                            gap_msg = None
                        existing_total_gt = _fast_summary_total_gt(fast_df)

                        if not resume_artifacts_valid:
                            print(
                                f"[S5_4][RESUME-RERUN] {scenario_id}: "
                                "fast emissions files are missing, stale, or belong to another run"
                            )
                            if bool(cfg.get("clear_existing_run_dirs_on_resume_retry", True)):
                                if _clear_existing_scenario_dir(scenario_path, runs_dir):
                                    print(
                                        f"[S5_4][RESUME-RERUN] cleared old run dir: {scenario_path}"
                                    )
                        elif neg_msg:
                            status_row["run_status"] = "invalid_fast_emissions"
                            status_row["error_type"] = "NegativeNonLUCEmissions"
                            status_row["error_message"] = neg_msg
                        elif _is_invalid_total_gt(existing_total_gt):
                            status_row["run_status"] = "invalid_fast_emissions"
                            status_row["error_type"] = "InvalidFastSummarySentinel"
                            status_row["error_message"] = (
                                f"invalid total_co2eq_gt sentinel: {existing_total_gt}"
                            )
                        elif gap_msg:
                            print(f"[S5_4][RESUME-RERUN] {scenario_id}: {gap_msg}")
                            _clear_existing_scenario_dir(scenario_path, runs_dir)
                        elif existing_total_gt is not None:
                            status_row["run_status"] = "resumed"
                            status_row["afolu_emissions_gt_co2eq_yr"] = float(existing_total_gt)

                    if status_row["run_status"] != "pending":
                        if status_row["run_status"] == "resumed":
                            _append_fast_summary(
                                success_summary_rows,
                                fast_df,
                                scenario_id=scenario_id,
                                sample_id=sample_id,
                                scenario_dir=scenario_path,
                                run_id=resume_validation.run_id,
                                experiment_fingerprint=experiment_resume_fingerprint,
                                resume_fingerprint=scenario_resume_fingerprint,
                                batch_state=batch_state,
                            )
                            _append_global_process_rows(
                                success_process_rows,
                                fast_detail_df,
                                scenario_id=scenario_id,
                                sample_id=sample_id,
                                scenario_dir=scenario_path,
                                run_id=resume_validation.run_id,
                                experiment_fingerprint=experiment_resume_fingerprint,
                                resume_fingerprint=scenario_resume_fingerprint,
                                batch_state=batch_state,
                            )
                            success_weighted_rows.extend(
                                _stamp_success_provenance(
                                    _aggregate_weighted_elements(
                                        _expand_success_weighting_rows(
                                            effects,
                                            scenario_id=scenario_id,
                                            sample_id=sample_id,
                                            attempt=1,
                                        ),
                                        batch_state=batch_state,
                                        require_ef_intensity=bool(cfg.get("require_ef_co2eq_intensity", False)),
                                    ),
                                    run_id=resume_validation.run_id,
                                    experiment_fingerprint=experiment_resume_fingerprint,
                                    resume_fingerprint=scenario_resume_fingerprint,
                                )
                            )
                            _append_realized_ruminant_rows(
                                success_realized_ruminant_rows,
                                _read_realized_ruminant_share_df(
                                    scenario_path,
                                    validation=resume_validation,
                                ),
                                scenario_id=scenario_id,
                                sample_id=sample_id,
                                scenario_dir=scenario_path,
                                run_id=resume_validation.run_id,
                                experiment_fingerprint=experiment_resume_fingerprint,
                                resume_fingerprint=scenario_resume_fingerprint,
                                batch_state=batch_state,
                            )
                            _append_land_balance_rows(
                                success_land_balance_rows,
                                _read_crop_pasture_land_balance_df(
                                    scenario_path,
                                    validation=resume_validation,
                                ),
                                scenario_id=scenario_id,
                                sample_id=sample_id,
                                scenario_dir=scenario_path,
                                run_id=resume_validation.run_id,
                                experiment_fingerprint=experiment_resume_fingerprint,
                                resume_fingerprint=scenario_resume_fingerprint,
                                batch_state=batch_state,
                            )
                            resume_count += 1
                            if resume_count == 1 or resume_count % 50 == 0:
                                print(
                                    f"[S5_4][RESUME] reused {resume_count}: "
                                    f"{scenario_id} ({local_idx}/{assigned_total})"
                                )
                        else:
                            print(
                                f"[S5_4][RESUME] {scenario_id} -> {status_row['run_status']}: "
                                f"{status_row['error_message'] or status_row['model_status_text']}"
                            )
                        status_rows.append(status_row)
                        _register_status(status_row)
                        processed_runs += 1
                        if len(status_rows) >= write_every:
                            _flush_buffers()
                        continue

                outdir = run_one_pipeline(
                    paths,
                    pre_macc_e0=False,
                    scenario_id=scenario_id,
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
                scenario_path = Path(outdir)
                status_row["scenario_dir"] = str(scenario_path)
                fresh_validation = validate_run_for_resume(
                    scenario_path,
                    expected_scenario_id=scenario_id,
                    expected_resume_fingerprint=scenario_resume_fingerprint,
                )
                status_row["run_id"] = fresh_validation.run_id
                solver_meta = (fresh_validation.payload or {}).get("solver") or {}
                status_row["model_status_code"] = solver_meta.get("status_code")
                status_row["model_status_text"] = str(
                    solver_meta.get("status_name") or ""
                ).upper()
                co2eq_gt = None

                if not fresh_validation.allowed:
                    status_row["run_status"] = (
                        "nonoptimal"
                        if fresh_validation.reason.startswith("solver_not_optimal:")
                        else "failed"
                    )
                    status_row["error_type"] = "RunStatusValidation"
                    status_row["error_message"] = fresh_validation.reason
                else:
                    fast_df = _read_validated_fast_summary_df(
                        scenario_path,
                        fresh_validation,
                    )
                    fast_detail_df = _read_validated_fast_global_detail_df(
                        scenario_path,
                        fresh_validation,
                    )
                    market_diag_path = (
                        scenario_path
                        / "Diagnostics"
                        / "commodity_balance_by_commodity.csv"
                    )
                    fresh_artifacts_valid = bool(
                        not fast_df.empty
                        and not fast_detail_df.empty
                        and artifact_matches_validated_run(
                            market_diag_path,
                            fresh_validation,
                        )
                    )
                    neg_msg = (
                        _validate_nonluc_fast_emissions(scenario_path)
                        if fresh_artifacts_valid
                        else None
                    )
                    if fresh_artifacts_valid:
                        status_row.update(_read_market_gap_summary(scenario_path))
                        gap_msg = _validate_market_balance_gap(
                            scenario_path,
                            max_gap_rate=float((cfg.get("override_cfg") or {}).get("market_gap_max_rate", 0.10)),
                        )
                    else:
                        gap_msg = None
                    co2eq_gt = _fast_summary_total_gt(fast_df)

                    if not fresh_artifacts_valid:
                        status_row["run_status"] = "missing_fast_summary"
                        status_row["error_type"] = "MissingCurrentRunArtifact"
                        status_row["error_message"] = (
                            "required fast-emissions or market artifact is missing, stale, "
                            "or belongs to another run"
                        )
                    elif neg_msg:
                        status_row["run_status"] = "invalid_fast_emissions"
                        status_row["error_type"] = "NegativeNonLUCEmissions"
                        status_row["error_message"] = neg_msg
                    elif _is_invalid_total_gt(co2eq_gt):
                        status_row["run_status"] = "invalid_fast_emissions"
                        status_row["error_type"] = "InvalidFastSummarySentinel"
                        status_row["error_message"] = f"invalid total_co2eq_gt sentinel: {co2eq_gt}"
                    elif gap_msg:
                        status_row["run_status"] = "invalid_market_balance"
                        status_row["error_type"] = "MarketBalanceGap"
                        status_row["error_message"] = gap_msg
                    elif co2eq_gt is not None:
                        status_row["run_status"] = "ok"
                        status_row["afolu_emissions_gt_co2eq_yr"] = float(co2eq_gt)
                    else:
                        status_row["run_status"] = "missing_fast_summary"
                        status_row["error_type"] = "MissingFastSummary"
                        status_row["error_message"] = "emissions_fast_summary.csv missing or unreadable"

                if status_row["run_status"] == "ok":
                    _append_fast_summary(
                        success_summary_rows,
                        fast_df,
                        scenario_id=scenario_id,
                        sample_id=sample_id,
                        scenario_dir=scenario_path,
                        run_id=fresh_validation.run_id,
                        experiment_fingerprint=experiment_resume_fingerprint,
                        resume_fingerprint=scenario_resume_fingerprint,
                        batch_state=batch_state,
                    )
                    _append_global_process_rows(
                        success_process_rows,
                        fast_detail_df,
                        scenario_id=scenario_id,
                        sample_id=sample_id,
                        scenario_dir=scenario_path,
                        run_id=fresh_validation.run_id,
                        experiment_fingerprint=experiment_resume_fingerprint,
                        resume_fingerprint=scenario_resume_fingerprint,
                        batch_state=batch_state,
                    )
                    success_weighted_rows.extend(
                        _stamp_success_provenance(
                            _aggregate_weighted_elements(
                                _expand_success_weighting_rows(
                                    effects,
                                    scenario_id=scenario_id,
                                    sample_id=sample_id,
                                    attempt=1,
                                ),
                                batch_state=batch_state,
                                require_ef_intensity=bool(cfg.get("require_ef_co2eq_intensity", False)),
                            ),
                            run_id=fresh_validation.run_id,
                            experiment_fingerprint=experiment_resume_fingerprint,
                            resume_fingerprint=scenario_resume_fingerprint,
                        )
                    )
                    _append_realized_ruminant_rows(
                        success_realized_ruminant_rows,
                        _read_realized_ruminant_share_df(scenario_path),
                        scenario_id=scenario_id,
                        sample_id=sample_id,
                        scenario_dir=scenario_path,
                        run_id=fresh_validation.run_id,
                        experiment_fingerprint=experiment_resume_fingerprint,
                        resume_fingerprint=scenario_resume_fingerprint,
                        batch_state=batch_state,
                    )
                    _append_land_balance_rows(
                        success_land_balance_rows,
                        _read_crop_pasture_land_balance_df(scenario_path),
                        scenario_id=scenario_id,
                        sample_id=sample_id,
                        scenario_dir=scenario_path,
                        run_id=fresh_validation.run_id,
                        experiment_fingerprint=experiment_resume_fingerprint,
                        resume_fingerprint=scenario_resume_fingerprint,
                        batch_state=batch_state,
                    )
                else:
                    _maybe_run_diagnostics(status_row, effects)
                    print(
                        f"[S5_4][WARN] {scenario_id} -> {status_row['run_status']}: "
                        f"{status_row['error_message'] or status_row['model_status_text']}"
                    )
            except KeyboardInterrupt:
                status_row["run_status"] = "interrupted"
                status_row["error_type"] = "KeyboardInterrupt"
                status_row["error_message"] = "Interrupted by user or runtime"
                status_rows.append(status_row)
                _register_status(status_row)
                _flush_buffers()
                raise
            except MCPrecheckFailed as exc:
                status_row["run_status"] = "precheck_failed"
                status_row["error_type"] = type(exc).__name__
                status_row["error_message"] = str(exc)
                print(f"[S5_4][PRECHECK-FAILED] {scenario_id}: {exc}")
                if bool(cfg.get("stop_on_error", False)):
                    status_rows.append(status_row)
                    _register_status(status_row)
                    _flush_buffers()
                    raise
            except Exception as exc:
                status_row["run_status"] = "failed"
                status_row["error_type"] = type(exc).__name__
                status_row["error_message"] = str(exc)
                print(f"[S5_4][FAILED] {scenario_id}: {type(exc).__name__}: {exc}")
                if bool(cfg.get("stop_on_error", False)):
                    status_rows.append(status_row)
                    _register_status(status_row)
                    _flush_buffers()
                    raise
            finally:
                unit_row = None
                param_rows = None
                effects = None
                expanded_df = None
                scenario_path = None
                fast_df = None
                log_diag = None
                neg_msg = None
                outdir = None
                gc.collect()

            status_rows.append(status_row)
            _register_status(status_row)
            processed_runs += 1
            if len(status_rows) >= write_every:
                _flush_buffers()

        _flush_buffers()
        cost_status_df = (
            pd.read_csv(status_out_path) if status_out_path.exists() else pd.DataFrame()
        )
        write_sensitivity_cost_summaries(
            cost_status_df,
            output_dir=output_dir,
            run_search_root=output_dir,
        )

        print(
            f"[S5_4] done: total={total_written}, "
            f"success={success_count}, resumed={resume_count}, infeasible={infeasible_count}, "
            f"nonoptimal={nonoptimal_count}, failed={failed_count}, "
            f"precheck_failed={precheck_failed_count}, interrupted={interrupted_count}"
        )
        print(f"[S5_4] status -> {status_out_path}")
        print(f"[S5_4] draws -> {draws_out_path}")
        if summary_written > 0:
            print(f"[S5_4] fast summary -> {summary_out_path}")
        if process_written > 0:
            print(f"[S5_4] process co2eq -> {process_out_path}")
        if weighted_written > 0:
            print(f"[S5_4] weighted elements -> {weighted_out_path}")
        if realized_ruminant_written > 0:
            print(f"[S5_4] realized ruminant share -> {realized_ruminant_out_path}")
        if land_balance_written > 0:
            print(f"[S5_4] crop/pasture land balance -> {land_balance_out_path}")
    finally:
        for key, value in cfg_backup.items():
            CFG[key] = value


if __name__ == "__main__":
    main()
