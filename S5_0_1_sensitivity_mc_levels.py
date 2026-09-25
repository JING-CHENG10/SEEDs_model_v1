# -*- coding: utf-8 -*-
"""
MC sensitivity runner for scenario variables with emissions-level importance tables.

This script:
1) Draws MC samples from Scenario_config_new.xlsx (sheet "MC_effect_low_land_new").
2) Runs full pipeline per sample (incl. LUC + GFIRE emissions).
3) Extracts 2080 global net emissions (CO2eq).
4) Computes relative importance for target emission levels.
"""

# Functional overview (S5_0)
# Purpose: assess the relative importance of variables for a target emissions level using MC samples.
# Inputs: MC configuration in Scenario_config_new.xlsx, baseline model data, and pipeline emissions outputs.
# Workflow:
# 1) Draw samples from the MC configuration and generate scenario_effects.
# 2) Call run_one_pipeline for each sample to calculate global emissions in 2080.
# 3) Calculate variable importance near the specified emissions target using correlations/weights.
# 4) Summarize detailed and grouped importance results.
# Main outputs:
# summary/samples.csv
# summary/importance_detail.csv
# summary/importance_by_variable.csv

from __future__ import annotations

import argparse
import gc
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd

from config_paths import get_input_base, get_results_base, get_src_base
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
from S4_0_main import CFG, build_run_baseline_cache, run_one_pipeline
from S5_cost_summary_outputs import write_sensitivity_cost_summaries
from S5_1_1_sensitivity_mc_variable_effect import (
    _build_scenario_effects,
    _draw_mc_param_rows,
    _draw_unit_row_for_specs,
    _load_mc_specs_effect,
    _normalize_mc_specs,
    resolve_mc_effect_sheet,
    _validate_market_balance_gap,
    _validate_nonluc_fast_emissions,
    RUMINANT_REDUCTION_DISCRETE_LEVELS,
    _sampling_scope,
    _sample_unit_matrix_for_specs,
)


@dataclass
class SampleResult:
    sample_id: int
    scenario_id: str
    emissions_2080_gt: float
    status: str
    params: Dict[str, float]


# MC sheet guidance (Scenario_config_new.xlsx -> "MC_effect_low_land_new"):
# Element (case-insensitive): feed_intensity, ruminant_reduction, emission_factor,
# fertilizer_rate, yield_rate, manure_management_ratio, land_carbon_price, losses_ratio,
# crop_soil_management_ratio
# Element unit: rate | multiplier | amount
# rate: value is 2080 relative change vs 2020 (e.g., -0.2 = -20%)
# multiplier: value is 2080 multiplier (e.g., 0.8 -> rate -0.2; 1.2 -> rate +0.2)
# amount: absolute level (used for land_carbon_price, USD/tCO2e)
# Min_bound/Max_bound define uniform draw range for each Element row.
# Region_cat, Item, Process filters follow Scenario rules (All / Region_aggMC / commodity / process).

CONFIG = {
    "seed": 42,
    "year": 2080,
    "unit_scale": 1e-6,  # kt -> Gt
    # Total MC draws. In batch mode these sample IDs are split across jobs.
    # Manuscript Figure 4 uses 20,000 Monte Carlo draws.
    "samples": 20000,
    "mc_sheet": "MC_effect_low_land_new",
    "nutrition_profile_sheet": "low_land_new",
    "output_dir": "",  # empty -> <NZF_OUTPUT_DIR>/MC_Sensitivity
    # Old runs in output/MC_Sensitivity were created with outdated scenario construction.
    # Default to False so a fresh run does not silently reuse them.
    "resume": False,
    "use_linear": True,
    "future_last_only": True,
    "use_regional": False,
    "pre_macc_e0": False,
    "use_fao_modules": True,
    "domestic_supply_simulation_mode": "hard_equation",
    "supply_curtailment_enabled": False,
    "supply_curtailment_penalty": 1e10,
    "fast_emis_only": True,
    "sampling": {
        # Better space coverage than uniform for a fixed sample budget.
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
        "mix_ratio": 0.6,
        "shuffle": True,
        "scramble": True,
        "quantile_bounds": (0.0, 1.0),
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
    "mc_non_ef_mode": "shared",
    "mc_ef_mode": "shared",
    "ef_process_mode": "all",
    # Keep model behaviour unchanged; only disable heavy diagnostics / verbose modes
    # that tend to increase memory and log pressure in large MC batches.
    "override_cfg": {
        "nutrition_profile_sheet": "low_land_new",
        "domestic_supply_simulation_mode": "hard_equation",
        "supply_curtailment_enabled": False,
        "supply_curtailment_penalty": 1e10,
        "market_gap_max_rate": 0.05,
        "batch_mode": True,
        "debug_level": 0,
        "linear_enable_infeasible_iis": False,
        "linear_enable_violation_iis": False,
        "linear_enable_output_diagnostics": False,
        "linear_enable_verbose_logging": False,
    },
    "batch": {
        "enabled": False,
        "total_batches": 100,
        "batch_index": 1,  # 1-based
        "assignment": "round_robin",  # 'round_robin' | 'contiguous'
        "batches_subdir": "batches",
    },
    "importance": {
        "method": "linear",
        # Target grid and sample filter for emissions-level importance.
        # Set either range to None to derive/keep the full available range.
        "target_range_gt": (-3.0, 60.0),
        "emissions_filter_range_gt": (-3.0, 60.0),
        "target_bin_width_gt": 1.0,
        # S5_0 computes target-varying importance by reweighting the same MC
        # samples around each target emission level. A small explicit sigma
        # makes importance local to each target; None falls back to whole-sample
        # std and usually makes target curves nearly flat.
        "window": None,
        "sigma": 2.0,
        "min_samples": 30,
    },
}

# Legacy target grid retained for callers that explicitly pass fixed targets.
TARGETS = np.arange(-3.0, 18.2, 0.5)
TARGET_STEP_GT = 1.0


def _parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip() != ""]


def _parse_optional_range(raw: object, *, name: str) -> Optional[Tuple[float, float]]:
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if text == "" or text.lower() in {"none", "null", "all"}:
            return None
        parts = [p.strip() for p in re.split(r"[,;:]", text) if p.strip()]
    else:
        try:
            parts = list(raw)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError(f"{name} must be None or a two-value range.") from exc
    if len(parts) != 2:
        raise ValueError(f"{name} must be None or a two-value range.")
    lo = float(parts[0])
    hi = float(parts[1])
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError(f"{name} values must be finite.")
    if hi <= lo:
        raise ValueError(f"{name} max must be greater than min.")
    return lo, hi


def _importance_cfg() -> Dict[str, object]:
    return dict(CONFIG.get("importance", {}) or {})


def _configured_target_step_gt() -> float:
    cfg = _importance_cfg()
    raw = cfg.get("target_bin_width_gt", cfg.get("target_step_gt", TARGET_STEP_GT))
    step = float(raw)
    if not np.isfinite(step) or step <= 0:
        raise ValueError(f"CONFIG['importance']['target_bin_width_gt'] must be positive, got {raw!r}.")
    return step


def _target_grid_from_range(target_range: Tuple[float, float], *, step: float) -> List[float]:
    lo, hi = target_range
    targets: List[float] = []
    current = lo
    eps = abs(step) * 1e-9
    while current <= hi + eps:
        targets.append(round(float(current), 10))
        current += step
    if not targets or not np.isclose(targets[-1], hi):
        targets.append(round(float(hi), 10))
    return targets


def _filter_samples_to_importance_range(samples_df: pd.DataFrame) -> pd.DataFrame:
    cfg = _importance_cfg()
    raw_range = cfg.get("emissions_filter_range_gt", cfg.get("target_range_gt"))
    filter_range = _parse_optional_range(raw_range, name="CONFIG['importance']['emissions_filter_range_gt']")
    if filter_range is None:
        return samples_df
    lo, hi = filter_range
    before = len(samples_df)
    values = pd.to_numeric(samples_df["emissions_2080_gt"], errors="coerce")
    out = samples_df.loc[values.between(lo, hi, inclusive="both")].copy().reset_index(drop=True)
    print(
        f"[S5_0] filtered samples for importance to {lo:g}-{hi:g} Gt: "
        f"{len(out)}/{before} retained."
    )
    if out.empty:
        raise RuntimeError(
            f"No samples remain after CONFIG['importance']['emissions_filter_range_gt'] "
            f"filter [{lo:g}, {hi:g}] Gt."
        )
    return out


def _targets_from_emission_range(samples_df: pd.DataFrame, *, step: float = TARGET_STEP_GT) -> List[float]:
    if "emissions_2080_gt" not in samples_df.columns:
        raise KeyError("samples_df missing emissions_2080_gt column.")
    values = pd.to_numeric(samples_df["emissions_2080_gt"], errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        raise RuntimeError("Cannot derive target emissions: no finite emissions_2080_gt values.")

    step = float(step)
    if not np.isfinite(step) or step <= 0:
        raise ValueError(f"Target step must be positive and finite, got {step!r}.")

    lo = float(values.min())
    hi = float(values.max())
    target_lo = np.floor(lo / step) * step
    target_hi = np.ceil(hi / step) * step
    n = int(round((target_hi - target_lo) / step)) + 1
    targets = [round(float(target_lo + i * step), 10) for i in range(max(n, 1))]
    print(
        f"[S5_0] derived {len(targets)} target emissions from sample range "
        f"{lo:.6g}-{hi:.6g} Gt using {step:g} Gt step "
        f"({targets[0]:g}-{targets[-1]:g} Gt)."
    )
    return targets


def _resolve_importance_targets(samples_df: pd.DataFrame, targets: Optional[List[float]]) -> List[float]:
    if targets:
        out = [float(t) for t in targets if np.isfinite(float(t))]
        if not out:
            raise ValueError("Explicit targets were provided but none are finite.")
        return out
    cfg = _importance_cfg()
    step = _configured_target_step_gt()
    target_range = _parse_optional_range(
        cfg.get("target_range_gt"),
        name="CONFIG['importance']['target_range_gt']",
    )
    if target_range is None:
        return _targets_from_emission_range(samples_df, step=step)
    targets_from_cfg = _target_grid_from_range(target_range, step=step)
    print(
        f"[S5_0] using configured target emissions: "
        f"{target_range[0]:g}-{target_range[1]:g} Gt, step {step:g} "
        f"({len(targets_from_cfg)} targets)."
    )
    return targets_from_cfg


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _reset_output_file(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def _append_rows_csv(rows: List[Dict[str, object]], out_path: Path) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    header = (not out_path.exists()) or out_path.stat().st_size == 0
    df.to_csv(out_path, mode="a", header=header, index=False, encoding="utf-8-sig")
    return int(len(df))


def _read_text_safe(path: Path) -> str:
    if not path.exists():
        return ""
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return path.read_text(encoding=encoding, errors="replace")
        except Exception:
            continue
    try:
        return path.read_text(errors="replace")
    except Exception:
        return ""


def _read_run_solver_status(run_dir: Path) -> Tuple[Optional[int], str]:
    """Return Gurobi-style status from run logs. OPTIMAL is status=2."""
    model_log = run_dir / "Log" / "model.log"
    gurobi_log = run_dir / "Log" / "gurobi.log"
    texts = [
        ("model.log", _read_text_safe(model_log)),
        ("gurobi.log", _read_text_safe(gurobi_log)),
    ]

    statuses: List[Tuple[int, str]] = []
    for label, text in texts:
        if not text:
            continue
        for match in re.finditer(r"(?:status|状态)\s*=\s*(\d+)", text, flags=re.IGNORECASE):
            try:
                statuses.append((int(match.group(1)), label))
            except Exception:
                continue
    if statuses:
        status, source = statuses[-1]
        return status, f"solver_status={status} from {source}"

    combined = "\n".join(text for _, text in texts if text)
    if "Infeasible model" in combined:
        return 3, "solver_status=3 from gurobi log: Infeasible model"
    if "Unbounded model" in combined:
        return 5, "solver_status=5 from gurobi log: Unbounded model"
    if "Time limit reached" in combined:
        return 9, "solver_status=9 from gurobi log: Time limit reached"
    if "Optimal objective" in combined or "Optimal solution found" in combined:
        return 2, "solver_status=2 from gurobi log"
    return None, "solver status not found in model.log/gurobi.log"


def _write_run_meta(path: Path,
                    *,
                    requested_samples: int,
                    assigned_samples: int,
                    valid_samples: int,
                    attempted_draws: int,
                    invalid_draws: int,
                    seed: int,
                    year: int,
                    unit_scale: float,
                    sampling_cfg: Dict[str, object],
                    q_bounds: Tuple[float, float],
                    run_cfg: Dict[str, object],
                    extra_meta: Optional[Dict[str, object]] = None) -> Path:
    success_rate = (float(valid_samples) / float(attempted_draws)) if attempted_draws > 0 else float("nan")
    row = {
        "requested_samples": int(requested_samples),
        "assigned_samples": int(assigned_samples),
        "valid_samples": int(valid_samples),
        "attempted_draws": int(attempted_draws),
        "invalid_draws": int(invalid_draws),
        "success_rate": success_rate,
        "seed": int(seed),
        "year": int(year),
        "unit_scale": float(unit_scale),
        "sampling_method": str(sampling_cfg.get("method", "lhs_antithetic")),
        "sampling_scope": _sampling_scope(sampling_cfg),
        "sampling_discrete_levels_enabled": bool(sampling_cfg.get("discrete_levels_enabled", True)),
        "sampling_discrete_use_quantile_bounds": bool(sampling_cfg.get("discrete_levels_use_quantile_bounds", False)),
        "sampling_mix_ratio": sampling_cfg.get("mix_ratio"),
        "sampling_shuffle": bool(sampling_cfg.get("shuffle", True)),
        "sampling_scramble": bool(sampling_cfg.get("scramble", True)),
        "sampling_q_low": float(q_bounds[0]),
        "sampling_q_high": float(q_bounds[1]),
        "mc_sheet": str(run_cfg.get("mc_sheet", CONFIG.get("mc_sheet", "MC_effect_low_land_new"))),
        "use_linear": bool(run_cfg.get("use_linear", True)),
        "future_last_only": bool(run_cfg.get("future_last_only", True)),
        "use_regional": bool(run_cfg.get("use_regional", False)),
        "pre_macc_e0": bool(run_cfg.get("pre_macc_e0", False)),
        "use_fao_modules": bool(run_cfg.get("use_fao_modules", True)),
        "fast_emis_only": bool(run_cfg.get("fast_emis_only", True)),
        "market_gap_max_rate": float(run_cfg.get("market_gap_max_rate", 0.05) or 0.05),
        "mc_non_ef_mode": str(run_cfg.get("mc_non_ef_mode", "shared")),
        "mc_ef_mode": str(run_cfg.get("mc_ef_mode", "shared")),
        "ef_process_mode": str(run_cfg.get("ef_process_mode", "all")),
    }
    if extra_meta:
        row.update(extra_meta)
    meta_df = pd.DataFrame([row])
    meta_df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _scenario_id(sample_id: int) -> str:
    return f"MC_{int(sample_id):05d}"


def _batch_tag(batch_index: int, batch_count: int) -> str:
    return f"batch_{int(batch_index):02d}_of_{int(batch_count):02d}"


def _resolve_batch_settings(cfg: Dict[str, object]) -> Dict[str, object]:
    batch_cfg = cfg.get("batch", {}) or {}
    batch_count = int(batch_cfg.get("total_batches", 1) or 1)
    batch_index = int(batch_cfg.get("batch_index", 1) or 1)
    if batch_count <= 0:
        raise ValueError("batch.total_batches must be a positive integer.")
    if batch_index <= 0 or batch_index > batch_count:
        raise ValueError("batch.batch_index must be within 1..total_batches.")
    assignment = str(batch_cfg.get("assignment", "round_robin") or "round_robin").strip().lower()
    if assignment not in {"round_robin", "contiguous"}:
        raise ValueError("batch.assignment must be 'round_robin' or 'contiguous'.")
    enabled = bool(batch_cfg.get("enabled", False))
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


def _resolve_output_paths(root_output_dir: Path, batch_state: Dict[str, object]) -> Dict[str, Path]:
    if bool(batch_state.get("enabled")):
        active_output_dir = (
            root_output_dir / str(batch_state["batches_subdir"]) / str(batch_state["tag"])
        )
    else:
        active_output_dir = root_output_dir
    summary_dir = active_output_dir / "summary"
    return {
        "root_output_dir": root_output_dir,
        "active_output_dir": active_output_dir,
        "runs_dir": active_output_dir / "runs",
        "mc_samples_dir": active_output_dir / "MC",
        "summary_dir": summary_dir,
        "samples_path": summary_dir / "samples.csv",
        "status_path": summary_dir / "run_status.csv",
        "meta_path": summary_dir / "run_meta.csv",
        "importance_detail_path": summary_dir / "importance_detail.csv",
        "importance_group_path": summary_dir / "importance_by_variable.csv",
    }


def _param_groups_from_columns(columns: Iterable[str]) -> Dict[str, str]:
    groups: Dict[str, str] = {}
    for col in columns:
        text = str(col or "").strip()
        if not text:
            continue
        kind = text.split("|", 1)[0]
        groups[text] = _normalize_kind(kind)
    return groups


def _write_importance_outputs(
    *,
    samples_df: pd.DataFrame,
    summary_dir: Path,
    targets: Optional[List[float]],
    method: str,
    window: Optional[float],
    sigma: Optional[float],
    min_samples: int,
) -> Tuple[Path, Path]:
    if samples_df.empty:
        raise RuntimeError("No valid MC samples available for importance calculation.")

    samples_df = samples_df.copy()
    samples_df["emissions_2080_gt"] = pd.to_numeric(samples_df["emissions_2080_gt"], errors="coerce")
    samples_df = samples_df.dropna(subset=["emissions_2080_gt"]).reset_index(drop=True)
    if samples_df.empty:
        raise RuntimeError("All MC samples have non-numeric emissions_2080_gt.")
    samples_df = _filter_samples_to_importance_range(samples_df)
    if samples_df["emissions_2080_gt"].nunique() < 2:
        raise RuntimeError(
            "MC sensitivity produced fewer than 2 distinct emissions outcomes. "
            "Importance is not identifiable."
        )

    param_cols = [
        c for c in samples_df.columns
        if c not in {"sample_id", "scenario_id", "emissions_2080_gt"}
    ]
    param_groups = _param_groups_from_columns(param_cols)
    targets = _resolve_importance_targets(samples_df, targets)
    targets_path = summary_dir / "importance_targets.csv"
    cfg = _importance_cfg()
    pd.DataFrame(
        {
            "target_emission_gt": targets,
            "target_range_gt": str(cfg.get("target_range_gt")),
            "emissions_filter_range_gt": str(cfg.get("emissions_filter_range_gt", cfg.get("target_range_gt"))),
            "target_bin_width_gt": _configured_target_step_gt(),
            "retained_samples_for_importance": int(len(samples_df)),
        }
    ).to_csv(targets_path, index=False, encoding="utf-8-sig")

    detail_rows: List[Dict[str, object]] = []
    group_rows: List[Dict[str, object]] = []
    for target in targets:
        detail, group = _compute_importance(
            samples_df,
            param_cols,
            param_groups,
            target,
            method=method,
            window=window,
            sigma=sigma,
            min_samples=min_samples,
        )
        detail_rows.extend(detail)
        group_rows.extend(group)

    detail_path = summary_dir / "importance_detail.csv"
    group_path = summary_dir / "importance_by_variable.csv"
    pd.DataFrame(detail_rows).to_csv(detail_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(group_rows).to_csv(group_path, index=False, encoding="utf-8-sig")
    return detail_path, group_path


def _merge_batch_outputs(
    *,
    root_output_dir: Path,
    batch_state: Dict[str, object],
    targets: Optional[List[float]],
    method: str,
    window: Optional[float],
    sigma: Optional[float],
    min_samples: int,
) -> Dict[str, Path]:
    batch_root = root_output_dir / str(batch_state["batches_subdir"])
    root_summary_dir = root_output_dir / "summary"
    _ensure_dir(root_summary_dir)

    samples_frames: List[pd.DataFrame] = []
    status_frames: List[pd.DataFrame] = []
    meta_frames: List[pd.DataFrame] = []

    expected_tags = [
        _batch_tag(i, int(batch_state["count"]))
        for i in range(1, int(batch_state["count"]) + 1)
    ]
    for tag in expected_tags:
        batch_summary_dir = batch_root / tag / "summary"
        meta_path = batch_summary_dir / "run_meta.csv"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing batch meta file: {meta_path}")
        meta_df = pd.read_csv(meta_path)
        if meta_df.empty:
            raise RuntimeError(f"Empty batch meta file: {meta_path}")
        meta_df["batch_tag"] = tag
        meta_frames.append(meta_df)

        samples_path = batch_summary_dir / "samples.csv"
        if samples_path.exists():
            batch_samples_df = pd.read_csv(samples_path)
            if not batch_samples_df.empty:
                samples_frames.append(batch_samples_df)

        status_path = batch_summary_dir / "run_status.csv"
        if status_path.exists():
            batch_status_df = pd.read_csv(status_path)
            if not batch_status_df.empty:
                status_frames.append(batch_status_df)

    merged_samples_df = pd.concat(samples_frames, ignore_index=True) if samples_frames else pd.DataFrame()
    if not merged_samples_df.empty and "sample_id" in merged_samples_df.columns:
        merged_samples_df = (
            merged_samples_df.sort_values(["sample_id", "scenario_id"])
            .drop_duplicates(subset=["sample_id"], keep="last")
            .reset_index(drop=True)
        )
    merged_status_df = pd.concat(status_frames, ignore_index=True) if status_frames else pd.DataFrame()
    if not merged_status_df.empty:
        order_cols = [c for c in ["sample_id", "scenario_id"] if c in merged_status_df.columns]
        if order_cols:
            merged_status_df = merged_status_df.sort_values(order_cols).reset_index(drop=True)
    merged_meta_df = pd.concat(meta_frames, ignore_index=True).reset_index(drop=True)

    invalid_fast_sids: Dict[str, str] = {}
    invalid_market_sids: Dict[str, str] = {}
    market_gap_max_rate = float((CONFIG.get("override_cfg") or {}).get("market_gap_max_rate", 0.05) or 0.05)
    if not merged_status_df.empty and "scenario_id" in merged_status_df.columns:
        status_col = "status" if "status" in merged_status_df.columns else (
            "run_status" if "run_status" in merged_status_df.columns else None
        )
        if "message" not in merged_status_df.columns:
            merged_status_df["message"] = ""
        for idx, row in merged_status_df.iterrows():
            if status_col and str(row.get(status_col, "")).strip().lower() not in {"valid", "ok"}:
                continue
            scenario_id = str(row.get("scenario_id", "")).strip()
            if not scenario_id:
                continue
            batch_tag = str(row.get("batch_tag", "") or "").strip()
            if not batch_tag and "batch_index" in merged_status_df.columns:
                try:
                    batch_tag = _batch_tag(int(row.get("batch_index")), int(batch_state["count"]))
                except Exception:
                    batch_tag = ""
            if not batch_tag:
                continue
            run_dir = batch_root / batch_tag / "runs" / scenario_id
            if not run_dir.exists():
                continue
            fast_msg = _validate_nonluc_fast_emissions(run_dir)
            if fast_msg:
                invalid_fast_sids[scenario_id] = fast_msg
                if status_col:
                    merged_status_df.at[idx, status_col] = "invalid_fast_emissions"
                merged_status_df.at[idx, "message"] = fast_msg
                continue
            gap_msg = _validate_market_balance_gap(run_dir, max_gap_rate=market_gap_max_rate)
            if gap_msg:
                invalid_market_sids[scenario_id] = gap_msg
                if status_col:
                    merged_status_df.at[idx, status_col] = "invalid_market_balance"
                merged_status_df.at[idx, "message"] = gap_msg
        invalid_sids = set(invalid_fast_sids) | set(invalid_market_sids)
        if invalid_sids and not merged_samples_df.empty and "scenario_id" in merged_samples_df.columns:
            merged_samples_df = (
                merged_samples_df[
                    ~merged_samples_df["scenario_id"].astype(str).isin(invalid_sids)
                ]
                .reset_index(drop=True)
            )
        if invalid_fast_sids:
            print(f"[S5_0][merge] removed invalid fast-emission samples: {len(invalid_fast_sids)}")
        if invalid_market_sids:
            print(f"[S5_0][merge] removed invalid market-balance samples: {len(invalid_market_sids)}")

    samples_path = root_summary_dir / "samples.csv"
    status_path = root_summary_dir / "run_status.csv"
    meta_path = root_summary_dir / "run_meta.csv"
    merged_samples_df.to_csv(samples_path, index=False, encoding="utf-8-sig")
    if not merged_status_df.empty:
        merged_status_df.to_csv(status_path, index=False, encoding="utf-8-sig")
    cost_paths = write_sensitivity_cost_summaries(
        merged_status_df,
        output_dir=root_summary_dir,
        run_search_root=root_output_dir,
    )

    first_meta = merged_meta_df.iloc[0].to_dict() if not merged_meta_df.empty else {}
    requested_samples = int(pd.to_numeric(merged_meta_df.get("requested_samples"), errors="coerce").dropna().max())
    assigned_samples = int(pd.to_numeric(merged_meta_df.get("assigned_samples"), errors="coerce").fillna(0).sum())
    attempted_draws = int(pd.to_numeric(merged_meta_df.get("attempted_draws"), errors="coerce").fillna(0).sum())
    invalid_draws = int(pd.to_numeric(merged_meta_df.get("invalid_draws"), errors="coerce").fillna(0).sum())
    valid_samples = int(len(merged_samples_df))
    sampling_cfg = {
        "method": first_meta.get("sampling_method", "lhs_antithetic"),
        "scope": first_meta.get("sampling_scope", "element"),
        "discrete_levels_enabled": first_meta.get("sampling_discrete_levels_enabled", True),
        "discrete_levels_use_quantile_bounds": first_meta.get("sampling_discrete_use_quantile_bounds", False),
        "mix_ratio": first_meta.get("sampling_mix_ratio"),
        "shuffle": first_meta.get("sampling_shuffle", True),
        "scramble": first_meta.get("sampling_scramble", True),
    }
    q_bounds = (
        float(first_meta.get("sampling_q_low", 0.0) or 0.0),
        float(first_meta.get("sampling_q_high", 1.0) or 1.0),
    )
    run_cfg = {
        "use_linear": first_meta.get("use_linear", True),
        "future_last_only": first_meta.get("future_last_only", True),
        "use_regional": first_meta.get("use_regional", False),
        "pre_macc_e0": first_meta.get("pre_macc_e0", False),
        "use_fao_modules": first_meta.get("use_fao_modules", True),
        "fast_emis_only": first_meta.get("fast_emis_only", True),
        "market_gap_max_rate": first_meta.get("market_gap_max_rate", market_gap_max_rate),
        "mc_sheet": first_meta.get("mc_sheet", CONFIG.get("mc_sheet", "MC_effect_low_land_new")),
        "mc_non_ef_mode": first_meta.get("mc_non_ef_mode", "shared"),
        "mc_ef_mode": first_meta.get("mc_ef_mode", "shared"),
        "ef_process_mode": first_meta.get("ef_process_mode", "all"),
    }
    _write_run_meta(
        meta_path,
        requested_samples=requested_samples,
        assigned_samples=assigned_samples,
        valid_samples=valid_samples,
        attempted_draws=attempted_draws,
        invalid_draws=invalid_draws,
        seed=int(float(first_meta.get("seed", CONFIG["seed"]) or CONFIG["seed"])),
        year=int(float(first_meta.get("year", CONFIG["year"]) or CONFIG["year"])),
        unit_scale=float(first_meta.get("unit_scale", CONFIG["unit_scale"]) or CONFIG["unit_scale"]),
        sampling_cfg=sampling_cfg,
        q_bounds=q_bounds,
        run_cfg=run_cfg,
        extra_meta={
            "merge_source": "batches",
            "completed_batches": int(len(expected_tags)),
            "expected_batches": int(batch_state["count"]),
            "batch_assignment": str(batch_state["assignment"]),
        },
    )
    detail_path, group_path = _write_importance_outputs(
        samples_df=merged_samples_df,
        summary_dir=root_summary_dir,
        targets=targets,
        method=method,
        window=window,
        sigma=sigma,
        min_samples=min_samples,
    )
    return {
        "samples": samples_path,
        "run_status": status_path,
        "run_meta": meta_path,
        "importance_targets": root_summary_dir / "importance_targets.csv",
        "importance_detail": detail_path,
        "importance_by_variable": group_path,
        "country_measure_cost": cost_paths["country_measure"],
        "global_measure_cost": cost_paths["global_measure"],
        "cost_summary_audit": cost_paths["audit"],
    }


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


def _normalize_mc_element_name(val: object) -> str:
    """Translate new Element names to legacy ones for draw_mc_effects compatibility."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    raw = str(val or "").strip()
    if not raw:
        return raw
    return raw


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


def _build_region_members(universe) -> Dict[str, List[str]]:
    members: Dict[str, List[str]] = {}
    for c, r in (universe.region_aggMC_by_country or {}).items():
        if c and r:
            members.setdefault(str(r), []).append(str(c))
    return members


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
    def _spec_u() -> float:
        if spec.get("u_is_rescaled", False):
            try:
                u = float(base_u)
            except Exception:
                u = 0.5
            return max(0.0, min(1.0, u))
        return _rescale_u(base_u, spec.get("q_low"), spec.get("q_high"))

    if spec.get("pre_sampled", False) or mode == "shared":
        return _spec_u()
    import hashlib
    if mode == "per_country_commodity":
        key = f"{eff.scenario_id}|{eff.kind}|{eff.country_sel}|{eff.commodity_sel}|{eff.process_sel}|{country}|{commodity}"
    else:
        key = f"{eff.scenario_id}|{eff.kind}|{eff.country_sel}|{eff.commodity_sel}|{eff.process_sel}|{country}"
    seed = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    return _rescale_u(float(rng.random()), spec.get("q_low"), spec.get("q_high"))


def _u_for_ef(spec: dict, eff, *, country: str, commodity: str, mode: str) -> float:
    base_u = spec.get("u", 0.5)
    def _spec_u() -> float:
        if spec.get("u_is_rescaled", False):
            try:
                u = float(base_u)
            except Exception:
                u = 0.5
            return max(0.0, min(1.0, u))
        return _rescale_u(base_u, spec.get("q_low"), spec.get("q_high"))

    if spec.get("pre_sampled", False) or mode == "shared":
        return _spec_u()
    import hashlib
    if mode == "per_country_commodity":
        key = f"{eff.scenario_id}|{eff.kind}|{eff.country_sel}|{eff.commodity_sel}|{eff.process_sel}|{country}|{commodity}"
    else:
        key = f"{eff.scenario_id}|{eff.kind}|{eff.country_sel}|{eff.commodity_sel}|{eff.process_sel}|{country}"
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
        _, demand_item_map = _load_item_demand_extra_and_map(str(Path(get_src_base()) / "dict_v3.xlsx"))
    except Exception:
        demand_item_map = {}
    item_to_comms: Dict[str, List[str]] = {}
    for comm, items in (demand_item_map or {}).items():
        if not comm or not items:
            continue
        for item in items:
            item_norm = _normalize_comp_item_name(item)
            if item_norm:
                item_to_comms.setdefault(item_norm, []).append(comm)
    acc: Dict[Tuple[str, str], float] = {}
    cnt: Dict[Tuple[str, str], int] = {}
    for (m49, item), val in (losses_raw or {}).items():
        try:
            val_f = float(val)
        except Exception:
            continue
        m49_norm = _normalize_m49(m49)
        item_norm = _normalize_comp_item_name(item)
        if not m49_norm or not item_norm:
            continue
        losses_base[(m49_norm, item_norm)] = val_f
        country = universe.country_by_m49.get(m49) or universe.country_by_m49.get(m49_norm)
        if country:
            losses_base[(country, item_norm)] = val_f
        for comm in item_to_comms.get(item_norm, []):
            for country_key in ([m49_norm] + ([country] if country else [])):
                key = (country_key, comm)
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
        for r in grouped.itertuples(index=False):
            try:
                ef_base[(str(r.m49), str(r.Item), str(r.Process), str(r.ghg))] = float(r.__base)
            except Exception:
                continue
        if "__unit" in df.columns:
            unit_rows = df.dropna(subset=["__unit"])[["m49", "Item", "Process", "ghg", "__unit"]]
            for r in unit_rows.itertuples(index=False):
                key = (str(r.m49), str(r.Item), str(r.Process), str(r.ghg))
                if key not in ef_unit:
                    try:
                        ef_unit[key] = str(r.__unit)
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


def _lookup_base_value(kind: str,
                       country: str,
                       commodity: str,
                       baselines: Dict[str, object],
                       region_members: Dict[str, List[str]]) -> Optional[float]:
    if kind == "yield_rate":
        val = _lookup_with_region(baselines["livestock_yield"], country, commodity, region_members)
        if val is None:
            val = _lookup_with_region(baselines["yield"], country, commodity, region_members)
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
        return 1.0
    if kind == "crop_soil_management_ratio":
        return 1.0
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
        val = _lookup_with_region(baselines["livestock_yield"], country, commodity, region_members)
        if val is not None and np.isfinite(val):
            return "t/head"
        val = _lookup_with_region(baselines["yield"], country, commodity, region_members)
        if val is not None and np.isfinite(val):
            return "t/ha"
        return None
    if kind == "fertilizer_rate":
        return "kgN/ha"
    if kind in ("manure_management_ratio", "ruminant_reduction", "losses_ratio", "crop_soil_management_ratio"):
        return "ratio"
    if kind in ("feed_intensity", "feed_efficiency"):
        return "kg/head"
    if kind == "emission_factor":
        if m49 is None:
            m49 = ""
        key = (str(m49), str(commodity), str(process), str(ghg))
        return baselines.get("ef_unit", {}).get(key)
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
        if base_val is None or not np.isfinite(base_val):
            return None
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
                          mc_mode_default: str) -> Dict[str, List[Dict[str, object]]]:
    region_members = _build_region_members(universe)
    out: Dict[str, List[Dict[str, object]]] = {}
    for eff in effects:
        kind = _normalize_kind(getattr(eff, "kind", ""))
        unit = getattr(eff, "unit", "")
        mc_unit = unit
        spec = getattr(eff, "mc_bounds_raw", None) or getattr(eff, "mc_bounds", None)
        mode = str((spec or {}).get("mode") or mc_mode_default or "shared").strip().lower()
        if mode in ("a", "country", "per_country", "independent", "per-country"):
            mode = "per_country"
        elif mode in ("per_country_commodity", "country_commodity", "per-country-commodity", "per_country_item", "per-item"):
            mode = "per_country_commodity"
        else:
            mode = "shared"

        countries = getattr(eff, "countries", None) or []
        commodities = getattr(eff, "commodities", None) or []
        processes = getattr(eff, "processes", None) or []
        ghg_sel = getattr(eff, "ghg_sel", "All") or "All"
        for country in countries:
            for commodity in commodities:
                if kind == "emission_factor":
                    for process in processes:
                        m49 = universe.m49_by_country.get(country)
                        m49_norm = _normalize_m49(m49) if m49 else ""
                        base_val = baselines["ef"].get((m49_norm, commodity, process, ghg_sel))
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
                        if spec and is_y2020:
                            u_val = _u_for_ef(spec, eff, country=country, commodity=commodity, mode=mode)
                        u_used = u_val if u_val is not None else (spec or {}).get("u")
                        value_draw_out = eff.value_2080
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
                            ratio, sample_val = _calc_ratio(kind, unit, eff.value_2080, base_val, spec=spec, u_val=u_val)
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
                            "mc_u": u_val if u_val is not None else (spec or {}).get("u"),
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
                    if spec and is_y2020:
                        u_val = _u_for_country(spec, eff, country=country, commodity=commodity, mode=mode)
                    u_used = u_val if u_val is not None else (spec or {}).get("u")
                    value_draw_out = eff.value_2080
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
                        ratio, sample_val = _calc_ratio(kind, unit, eff.value_2080, base_val, spec=spec, u_val=u_val)
                    if kind == "losses_ratio":
                        base_loss = base_val if base_val is not None and np.isfinite(base_val) else 0.0
                        sample_val = _final_loss_ratio_from_delta(base_loss, eff.value_2080)
                        ratio = _loss_multiplier_from_delta(base_loss, eff.value_2080)
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
                        "mc_u": u_val if u_val is not None else (spec or {}).get("u"),
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
        meta = pd.DataFrame([{
            "scenario_id": scenario_id,
            "sample_id": sample_id,
            "attempt": attempt,
        }])
        meta.to_excel(writer, sheet_name="meta", index=False)
        for kind, rows in rows_by_kind.items():
            if not rows:
                continue
            df = pd.DataFrame(rows)
            sheet = str(kind)[:31] if kind else "data"
            df.to_excel(writer, sheet_name=sheet, index=False)
    return out_path


def _read_validated_fast_csv(
    path: Path,
    validation: ResumeValidation,
) -> pd.DataFrame:
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


def _validated_fast_outputs_ready(
    run_dir: Path,
    validation: ResumeValidation,
) -> bool:
    emis_dir = run_dir / "Emis"
    summary = _read_validated_fast_csv(
        emis_dir / "emissions_fast_summary.csv",
        validation,
    )
    detail = _read_validated_fast_csv(
        emis_dir / "emissions_fast_global_detail.csv",
        validation,
    )
    market_diag = run_dir / "Diagnostics" / "commodity_balance_by_commodity.csv"
    return bool(
        not summary.empty
        and not detail.empty
        and artifact_matches_validated_run(market_diag, validation)
    )


def _sample_resume_fingerprint(
    *,
    scenario_id: str,
    param_rows: List[Dict[str, object]],
    run_cfg: Dict[str, object],
    year: int,
) -> str:
    return build_resume_fingerprint(
        {
            "schema": 1,
            "runner": "S5_0_1_sensitivity_mc_levels",
            "scenario_id": str(scenario_id),
            "param_rows": [dict(row) for row in param_rows],
            "model_options": {
                "year": int(year),
                "fast_emis_only": bool(run_cfg.get("fast_emis_only", True)),
                "use_linear": bool(CFG.get("use_linear_model", True)),
                "use_fao_modules": bool(CFG.get("use_fao_modules", True)),
                "future_last_only": bool(CFG.get("future_last_only", True)),
                "demand_method": str(CFG.get("demand_method", "") or ""),
                "nutrition_profile_sheet": str(
                    CFG.get("nutrition_profile_sheet", "") or ""
                ),
            },
        }
    )


def _read_global_emissions_2080_gt(
    emis_xlsx: Path,
    *,
    year: int,
    unit_scale: float,
    validation: Optional[ResumeValidation] = None,
) -> float:
    fast_csv = emis_xlsx.parent / "emissions_fast_summary.csv"
    if fast_csv.exists():
        if validation is None:
            df = pd.read_csv(fast_csv)
        else:
            df = _read_validated_fast_csv(fast_csv, validation)
            if df.empty:
                return float("nan")
        if "year" in df.columns:
            df = df[df["year"].astype(int) == int(year)]
        if "total_co2eq_gt" in df.columns:
            return float(pd.to_numeric(df["total_co2eq_gt"], errors="coerce").sum())
        if "total_co2eq_kt" in df.columns:
            return float(pd.to_numeric(df["total_co2eq_kt"], errors="coerce").sum()) * unit_scale
    if not emis_xlsx.exists():
        return float("nan")
    if validation is not None and not artifact_matches_validated_run(
        emis_xlsx,
        validation,
    ):
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


def _effects_to_param_values(effects) -> Tuple[Dict[str, float], Dict[str, str]]:
    counts: Dict[str, int] = {}
    params: Dict[str, float] = {}
    groups: Dict[str, str] = {}
    for eff in effects:
        kind = _normalize_kind(getattr(eff, "kind", ""))
        base_id = f"{kind}|{eff.country_sel}|{eff.commodity_sel}|{eff.process_sel}"
        counts[base_id] = counts.get(base_id, 0) + 1
        param_id = base_id if counts[base_id] == 1 else f"{base_id}#{counts[base_id]}"
        params[param_id] = float(eff.value_2080)
        groups[param_id] = kind
    return params, groups


def _weighted_mean_std(x: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    wsum = np.sum(w)
    if wsum <= 0:
        return np.full(x.shape[1], np.nan), np.full(x.shape[1], np.nan)
    mean = np.sum(x * w[:, None], axis=0) / wsum
    var = np.sum(w[:, None] * (x - mean) ** 2, axis=0) / wsum
    std = np.sqrt(var)
    return mean, std


def _standardize(x: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean, std = _weighted_mean_std(x, w)
    std_safe = np.where(std == 0, np.nan, std)
    return (x - mean) / std_safe, std_safe


def _compute_importance(
    df: pd.DataFrame,
    param_cols: List[str],
    param_groups: Dict[str, str],
    target: float,
    *,
    method: str,
    window: Optional[float],
    sigma: Optional[float],
    min_samples: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    work = df.copy()
    work = work[np.isfinite(work["emissions_2080_gt"])]
    if work.empty:
        return [], []

    y = work["emissions_2080_gt"].to_numpy(dtype=float)
    x = work[param_cols].to_numpy(dtype=float)

    if method == "rank":
        x = pd.DataFrame(x, columns=param_cols).rank().to_numpy(dtype=float)
        y = pd.Series(y).rank().to_numpy(dtype=float)

    if window is not None:
        mask = np.abs(y - target) <= window
        if mask.sum() >= min_samples:
            x_use = x[mask]
            y_use = y[mask]
            w = np.ones(len(y_use), dtype=float)
            sigma_use = None
            weight_mode = "window"
        else:
            mask = np.ones(len(y), dtype=bool)
            x_use = x
            y_use = y
            sigma_use = sigma if sigma is not None else max(np.nanstd(y), 1.0)
            w = np.exp(-0.5 * ((y_use - target) / sigma_use) ** 2)
            weight_mode = "gaussian_fallback"
    else:
        x_use = x
        y_use = y
        sigma_use = sigma if sigma is not None else max(np.nanstd(y), 1.0)
        w = np.exp(-0.5 * ((y_use - target) / sigma_use) ** 2)
        weight_mode = "gaussian"

    w = np.where(np.isfinite(w), w, 0.0)
    if np.sum(w) <= 0:
        return [], []
    ess = float((np.sum(w) ** 2) / max(np.sum(w ** 2), 1e-12))

    x_std, x_scale = _standardize(x_use, w)
    y_std = _standardize(y_use.reshape(-1, 1), w)[0].reshape(-1)
    keep_cols = np.isfinite(x_scale) & (x_scale > 0)
    if not keep_cols.any():
        return [], []
    x_std = x_std[:, keep_cols]
    kept_cols = [c for c, keep in zip(param_cols, keep_cols) if keep]

    sqrt_w = np.sqrt(w)
    xw = x_std * sqrt_w[:, None]
    yw = y_std * sqrt_w
    try:
        beta, *_ = np.linalg.lstsq(xw, yw, rcond=None)
    except np.linalg.LinAlgError:
        return [], []

    abs_beta = np.abs(beta)
    total = abs_beta.sum()
    if total <= 0:
        return [], []
    imp = abs_beta / total

    detail_rows: List[Dict[str, object]] = []
    group_totals: Dict[str, float] = {}
    for col, val in zip(kept_cols, imp):
        kind = param_groups.get(col, col)
        group_totals[kind] = group_totals.get(kind, 0.0) + float(val)
        detail_rows.append({
            "target_emission_gt": target,
            "parameter": col,
            "group": kind,
            "importance": float(val),
            "method": method,
            "n_samples": int(len(y_use)),
            "effective_n_samples": ess,
            "weight_mode": weight_mode,
            "weight_sigma_gt": float(sigma_use) if sigma_use is not None else np.nan,
        })

    group_rows = [
        {
            "target_emission_gt": target,
            "parameter": key,
            "importance": float(val),
            "method": method,
            "n_samples": int(len(y_use)),
            "effective_n_samples": ess,
            "weight_mode": weight_mode,
            "weight_sigma_gt": float(sigma_use) if sigma_use is not None else np.nan,
        }
        for key, val in group_totals.items()
    ]
    group_sum = sum(r["importance"] for r in group_rows)
    if group_sum > 0:
        for row in group_rows:
            row["importance"] = row["importance"] / group_sum
    return detail_rows, group_rows


def _configure_cfg(run_cfg: Dict[str, object]) -> None:
    CFG["solve"] = True
    CFG["use_linear_model"] = bool(run_cfg.get("use_linear", True))
    CFG["future_last_only"] = bool(run_cfg.get("future_last_only", True))
    CFG["use_regional_aggregation"] = bool(run_cfg.get("use_regional", False))
    CFG["premacc_e0"] = bool(run_cfg.get("pre_macc_e0", False))
    CFG["use_fao_modules"] = bool(run_cfg.get("use_fao_modules", True))
    CFG["nutrition_profile_sheet"] = str(run_cfg.get("nutrition_profile_sheet", "low_land_new") or "low_land_new")
    CFG["domestic_supply_simulation_mode"] = str(
        run_cfg.get("domestic_supply_simulation_mode", "hard_equation") or "hard_equation"
    )
    CFG["supply_curtailment_enabled"] = bool(run_cfg.get("supply_curtailment_enabled", False))
    CFG["supply_curtailment_penalty"] = float(run_cfg.get("supply_curtailment_penalty", 1e10) or 1e10)
    soft_cfg = run_cfg.get("nutrition_soft_constraints", {}) or {}
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
    land_soft = run_cfg.get("land_soft_constraints", {}) or {}
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
    CFG["mc_y2020_non_ef_mode"] = str(
        run_cfg.get("mc_non_ef_mode", "shared")
    ).strip().lower() or "shared"


def main() -> None:
    parser = argparse.ArgumentParser(description="MC sensitivity for emissions targets.")
    parser.add_argument("--seed", type=int, default=int(CONFIG["seed"]))
    parser.add_argument("--targets", type=str, default="",
                        help=(
                            "Optional comma list. Empty uses CONFIG['importance'] "
                            "target_range_gt and target_bin_width_gt; if target_range_gt "
                            "is None, derives targets from the retained sample range."
                        ))
    parser.add_argument("--year", type=int, default=int(CONFIG["year"]))
    parser.add_argument("--unit-scale", type=float, default=float(CONFIG["unit_scale"]),
                        help="Scale to convert emissions summary units to Gt (default 1e-6 for kt->Gt).")
    parser.add_argument("--samples", type=int, default=int(CONFIG["samples"]),
                        help="Total number of MC draws.")
    parser.add_argument("--mc-sheet", type=str, default=str(CONFIG.get("mc_sheet", "MC_effect_low_land_new")),
                        help="Scenario_config_new.xlsx sheet for MC specs. Default: MC_effect_low_land_new.")
    parser.add_argument(
        "--nutrition-profile-sheet",
        type=str,
        default=str(CONFIG.get("nutrition_profile_sheet", "low_land_new")),
        help="Nutrition profile sheet. Default: low_land_new.",
    )
    parser.add_argument("--output-dir", type=str, default="",
                        help="Root output dir. Default: <NZF_OUTPUT_DIR>/MC_Sensitivity")
    parser.add_argument("--resume", action="store_true", default=bool(CONFIG["resume"]),
                        help="Skip runs if emissions_summary.xlsx already exists.")
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--use-linear", action="store_true", default=bool(CONFIG["use_linear"]))
    parser.add_argument("--no-linear", action="store_false", dest="use_linear")
    parser.add_argument("--future-last-only", action="store_true", default=bool(CONFIG["future_last_only"]))
    parser.add_argument("--all-future", action="store_false", dest="future_last_only")
    parser.add_argument("--use-regional", action="store_true", default=bool(CONFIG["use_regional"]))
    parser.add_argument("--pre-macc-e0", action="store_true", default=bool(CONFIG["pre_macc_e0"]))
    parser.add_argument("--use-fao-modules", action="store_true", default=bool(CONFIG["use_fao_modules"]))
    parser.add_argument("--no-fao-modules", action="store_false", dest="use_fao_modules")
    parser.add_argument(
        "--market-gap-max-rate",
        type=float,
        default=float((CONFIG.get("override_cfg") or {}).get("market_gap_max_rate", 0.05) or 0.05),
        help="Post-solve global market shortage threshold. Default: 0.05.",
    )
    parser.add_argument("--method", type=str, default=str(CONFIG["importance"]["method"]), choices=["linear", "rank"])
    parser.add_argument("--window", type=float, default=CONFIG["importance"]["window"],
                        help="Hard window around target (Gt). If too few samples, fallback to Gaussian weights.")
    parser.add_argument("--sigma", type=float, default=CONFIG["importance"]["sigma"],
                        help="Gaussian sigma for target-local weighting (Gt). Use None in CONFIG to fall back to sample std.")
    parser.add_argument("--min-samples", type=int, default=int(CONFIG["importance"]["min_samples"]))
    parser.add_argument("--batch-enabled", action="store_true", default=bool(CONFIG["batch"]["enabled"]))
    parser.add_argument("--batch-index", type=int, default=int(CONFIG["batch"]["batch_index"]))
    parser.add_argument("--total-batches", type=int, default=int(CONFIG["batch"]["total_batches"]))
    parser.add_argument(
        "--batch-assignment",
        type=str,
        default=str(CONFIG["batch"]["assignment"]),
        choices=["round_robin", "contiguous"],
    )
    parser.add_argument("--merge-only", action="store_true", default=False)
    args = parser.parse_args()

    targets = _parse_float_list(args.targets) if args.targets else None
    run_cfg = {
        "use_linear": args.use_linear,
        "future_last_only": args.future_last_only,
        "use_regional": args.use_regional,
        "pre_macc_e0": args.pre_macc_e0,
        "use_fao_modules": args.use_fao_modules,
        "domestic_supply_simulation_mode": CONFIG.get("domestic_supply_simulation_mode", "hard_equation"),
        "supply_curtailment_enabled": CONFIG.get("supply_curtailment_enabled", False),
        "supply_curtailment_penalty": CONFIG.get("supply_curtailment_penalty", 1e10),
        "fast_emis_only": bool(CONFIG.get("fast_emis_only", True)),
        "market_gap_max_rate": float(args.market_gap_max_rate),
        "mc_sheet": str(args.mc_sheet or CONFIG.get("mc_sheet", "MC_effect_low_land_new")),
        "nutrition_profile_sheet": str(args.nutrition_profile_sheet or CONFIG.get("nutrition_profile_sheet", "low_land_new")),
        "nutrition_soft_constraints": CONFIG.get("nutrition_soft_constraints", {}) or {},
        "land_soft_constraints": CONFIG.get("land_soft_constraints", {}) or {},
        "mc_non_ef_mode": CONFIG.get("mc_non_ef_mode", "shared"),
        "mc_ef_mode": CONFIG.get("mc_ef_mode", "shared"),
        "ef_process_mode": CONFIG.get("ef_process_mode", "all"),
        "batch": {
            "enabled": args.batch_enabled,
            "batch_index": args.batch_index,
            "total_batches": args.total_batches,
            "assignment": args.batch_assignment,
            "batches_subdir": CONFIG.get("batch", {}).get("batches_subdir", "batches"),
        },
    }
    (CONFIG.get("override_cfg") or {})["market_gap_max_rate"] = float(args.market_gap_max_rate)
    _configure_cfg(run_cfg)
    batch_state = _resolve_batch_settings(run_cfg)

    base_out = Path(args.output_dir) if args.output_dir else Path(get_results_base()) / "MC_Sensitivity"
    output_paths = _resolve_output_paths(base_out, batch_state)
    runs_dir = output_paths["runs_dir"]
    summary_dir = output_paths["summary_dir"]
    mc_samples_dir = output_paths["mc_samples_dir"]
    samples_path = output_paths["samples_path"]
    status_path = output_paths["status_path"]
    meta_path = output_paths["meta_path"]

    if args.merge_only:
        merged_outputs = _merge_batch_outputs(
            root_output_dir=base_out,
            batch_state=batch_state,
            targets=targets,
            method=args.method,
            window=args.window,
            sigma=args.sigma,
            min_samples=args.min_samples,
        )
        for key, path in merged_outputs.items():
            print(f"[DONE] {key}: {path}")
        return

    _ensure_dir(runs_dir)
    _ensure_dir(summary_dir)
    _ensure_dir(mc_samples_dir)

    paths = DataPaths()
    cfg = ScenarioConfig()
    universe = build_universe_from_dict_v3(paths.dict_v3_path, cfg)
    base_year = int(cfg.years_hist_end or 2020)
    mc_sheet = resolve_mc_effect_sheet(
        run_cfg.get("mc_sheet", CONFIG.get("mc_sheet", "MC_effect_low_land_new")),
        nutrition_profile_sheet=run_cfg.get("nutrition_profile_sheet", "low_land_new"),
    )
    run_cfg["mc_sheet"] = mc_sheet
    specs_df = _load_mc_specs_effect(
        paths.scenario_config_xlsx,
        prefer_sheet=mc_sheet,
        nutrition_profile_sheet=run_cfg.get("nutrition_profile_sheet", "low_land_new"),
    )
    if specs_df is None or specs_df.empty:
        raise RuntimeError(f"MC specs sheet is empty or missing: {mc_sheet}")
    specs_df = _normalize_mc_specs(specs_df)

    cfg_backup: Dict[str, object] = {}
    for key, value in (CONFIG.get("override_cfg") or {}).items():
        cfg_backup[key] = CFG.get(key)
        CFG[key] = value

    shared_run_cache = build_run_baseline_cache(
        paths,
        cfg,
        universe,
        future_last_only=bool(run_cfg.get("future_last_only", True)),
    )
    baselines = _load_mc_baselines(paths, universe, base_year=base_year)
    mc_mode_default = str(CFG.get("mc_y2020_non_ef_mode", "shared")).strip().lower() or "shared"
    mc_mode_non_ef = str(run_cfg.get("mc_non_ef_mode", "shared")).strip().lower() or "shared"
    mc_mode_ef = str(run_cfg.get("mc_ef_mode", "shared")).strip().lower() or "shared"
    ef_process_mode = str(run_cfg.get("ef_process_mode", "all")).strip().lower() or "all"
    sampling_cfg = dict(CONFIG.get("sampling", {}) or {})
    q_bounds = sampling_cfg.get("quantile_bounds", (0.0, 1.0))
    try:
        q_bounds = (float(q_bounds[0]), float(q_bounds[1]))
    except Exception:
        q_bounds = (0.0, 1.0)
    requested_samples = max(1, int(args.samples))
    sample_indices = _select_batch_sample_indices(
        total_samples=requested_samples,
        batch_count=int(batch_state["count"]) if bool(batch_state.get("enabled")) else 1,
        batch_index=int(batch_state["index"]) if bool(batch_state.get("enabled")) else 1,
        assignment=str(batch_state["assignment"]),
    )
    assigned_samples = len(sample_indices)
    print(f"[S5_0] requested samples={requested_samples}")
    print(f"[S5_0] output_dir={output_paths['active_output_dir']}")
    print(
        f"[S5_0] mc_sheet={mc_sheet} "
        f"fast_emis_only={bool(run_cfg.get('fast_emis_only', True))} "
        f"shared_cache_nodes={len(shared_run_cache.get('node_blueprint') or [])}"
    )
    if bool(batch_state.get("enabled")):
        print(
            f"[S5_0] batch={batch_state['tag']} assignment={batch_state['assignment']} "
            f"assigned_samples={assigned_samples}/{requested_samples}"
        )
    unit_matrix = _sample_unit_matrix_for_specs(
        specs_df,
        requested_samples,
        seed=int(args.seed),
        config=sampling_cfg,
    )

    param_groups: Dict[str, str] = {}
    _reset_output_file(samples_path)
    _reset_output_file(status_path)
    _reset_output_file(meta_path)

    valid_samples = 0
    invalid_draws = 0
    try:
        for local_idx, sample_idx in enumerate(sample_indices, start=1):
            sample_id = sample_idx + 1
            scenario_id = _scenario_id(sample_id)
            unit_row = None
            param_rows = None
            effects = None
            params = None
            groups = None
            rows_by_kind = None
            emis_gt = np.nan

            unit_row = (
                unit_matrix[sample_idx]
                if sample_idx < len(unit_matrix)
                else _draw_unit_row_for_specs(
                    specs_df,
                    seed=int(args.seed) + sample_id * 1000,
                    config=sampling_cfg,
                )
            )
            param_rows = _draw_mc_param_rows(
                specs_df,
                unit_row=unit_row,
                quantile_bounds=q_bounds,
                sampling_cfg=sampling_cfg,
            )
            effects = _build_scenario_effects(
                param_rows,
                universe,
                scenario_id=scenario_id,
                mc_y2020_mode=mc_mode_non_ef,
                mc_mode_non_ef=mc_mode_non_ef,
                mc_mode_ef=mc_mode_ef,
                ef_process_mode=ef_process_mode,
            )
            params, groups = _effects_to_param_values(effects)
            param_groups.update(groups)
            scenario_resume_fingerprint = _sample_resume_fingerprint(
                scenario_id=scenario_id,
                param_rows=param_rows,
                run_cfg=run_cfg,
                year=int(args.year),
            )

            rows_by_kind = _build_mc_sample_rows(
                effects,
                universe=universe,
                baselines=baselines,
                scenario_id=scenario_id,
                sample_id=sample_id,
                attempt=1,
                mc_mode_default=mc_mode_default,
            )
            _write_mc_sample_xlsx(
                mc_samples_dir,
                scenario_id=scenario_id,
                sample_id=sample_id,
                attempt=1,
                rows_by_kind=rows_by_kind,
            )

            run_dir = runs_dir / scenario_id
            emis_dir = run_dir / "Emis"
            emis_path = emis_dir / "emissions_summary.xlsx"
            fast_path = emis_dir / "emissions_fast_summary.csv"
            status_row = _apply_batch_meta({
                "sample_id": sample_id,
                "scenario_id": scenario_id,
                "scenario_dir": str(run_dir),
                "status": "unknown",
                "solver_status": np.nan,
                "emissions_2080_gt": np.nan,
                "valid_sample_count_after_sample": int(valid_samples),
                "message": "",
            }, batch_state)
            try:
                resume_validation: Optional[ResumeValidation] = None
                reuse_allowed = False
                if args.resume:
                    resume_validation = validate_run_for_resume(
                        run_dir,
                        expected_scenario_id=scenario_id,
                        expected_resume_fingerprint=scenario_resume_fingerprint,
                    )
                    reuse_allowed = bool(
                        resume_validation.allowed
                        and _validated_fast_outputs_ready(run_dir, resume_validation)
                    )
                    if not reuse_allowed:
                        reason = (
                            resume_validation.reason
                            if not resume_validation.allowed
                            else "fast_outputs_missing_stale_or_wrong_identity"
                        )
                        print(f"[S5_0][RESUME-RERUN] {scenario_id}: {reason}")

                if not reuse_allowed:
                    run_one_pipeline(
                        paths,
                        pre_macc_e0=CFG["premacc_e0"],
                        scenario_id=scenario_id,
                        scenario_params=None,
                        scenario_effects=effects,
                        solve=CFG["solve"],
                        use_fao_modules=CFG["use_fao_modules"],
                        future_last_only=CFG["future_last_only"],
                        use_linear=CFG["use_linear_model"],
                        fast_emis_only=bool(run_cfg.get("fast_emis_only", True)),
                        fast_emis_year=int(args.year),
                        resume_fingerprint=scenario_resume_fingerprint,
                        save_root=str(runs_dir),
                        prebuilt_config=cfg,
                        prebuilt_universe=universe,
                        prebuilt_run_cache=shared_run_cache,
                    )

                    resume_validation = validate_run_for_resume(
                        run_dir,
                        expected_scenario_id=scenario_id,
                        expected_resume_fingerprint=scenario_resume_fingerprint,
                    )

                if resume_validation is None:
                    resume_validation = validate_run_for_resume(
                        run_dir,
                        expected_scenario_id=scenario_id,
                        expected_resume_fingerprint=scenario_resume_fingerprint,
                    )
                solver_meta = (resume_validation.payload or {}).get("solver") or {}
                solver_status = solver_meta.get("status_code")
                solver_message = (
                    "validated structured run status"
                    if resume_validation.allowed
                    else f"resume validation failed: {resume_validation.reason}"
                )
                status_row["solver_status"] = solver_status if solver_status is not None else np.nan
                if not resume_validation.allowed or solver_status != 2:
                    invalid_draws += 1
                    status_row["status"] = "invalid_run_status"
                    status_row["message"] = solver_message
                    print(
                        f"[MC] skip non-optimal draw {scenario_id}: "
                        f"{solver_message}; emissions not read"
                    )
                    continue

                neg_msg = _validate_nonluc_fast_emissions(run_dir)
                if neg_msg:
                    invalid_draws += 1
                    status_row["status"] = "invalid_fast_emissions"
                    status_row["message"] = neg_msg
                    print(f"[MC] skip invalid fast emissions {scenario_id}: {neg_msg}")
                    continue

                gap_msg = _validate_market_balance_gap(
                    run_dir,
                    max_gap_rate=float(run_cfg.get("market_gap_max_rate", 0.05) or 0.05),
                )
                if gap_msg:
                    invalid_draws += 1
                    status_row["status"] = "invalid_market_balance"
                    status_row["message"] = gap_msg
                    print(f"[MC] skip invalid market balance {scenario_id}: {gap_msg}")
                    continue

                emis_gt = _read_global_emissions_2080_gt(
                    emis_path,
                    year=args.year,
                    unit_scale=args.unit_scale,
                    validation=resume_validation,
                )
                status_row["emissions_2080_gt"] = emis_gt
                if np.isfinite(emis_gt):
                    row = {
                        "sample_id": sample_id,
                        "scenario_id": scenario_id,
                        "emissions_2080_gt": emis_gt,
                    }
                    row.update(params)
                    _append_rows_csv([row], samples_path)
                    valid_samples += 1
                    status_row["status"] = "valid"
                    status_row["valid_sample_count_after_sample"] = int(valid_samples)
                    if valid_samples == 1 or valid_samples % 25 == 0 or local_idx == assigned_samples:
                        print(
                            f"[S5_0] valid {valid_samples}/{assigned_samples} in active run "
                            f"(sample {sample_id}/{requested_samples}, {scenario_id})"
                        )
                else:
                    invalid_draws += 1
                    status_row["status"] = "invalid_emissions"
                    status_row["message"] = "emissions_2080_gt is NaN or non-finite"
                    print(f"[MC] skip invalid output {scenario_id}: emissions_2080_gt is NaN")
            except Exception as exc:
                invalid_draws += 1
                status_row["status"] = "run_error"
                status_row["message"] = str(exc)
                print(f"[MC] skip failed draw {scenario_id}: {exc}")
            finally:
                _append_rows_csv([status_row], status_path)
                unit_row = None
                param_rows = None
                effects = None
                params = None
                groups = None
                rows_by_kind = None
                gc.collect()
    finally:
        for key, value in cfg_backup.items():
            CFG[key] = value

    _write_run_meta(
        meta_path,
        requested_samples=requested_samples,
        assigned_samples=assigned_samples,
        valid_samples=valid_samples,
        attempted_draws=assigned_samples,
        invalid_draws=invalid_draws,
        seed=int(args.seed),
        year=int(args.year),
        unit_scale=float(args.unit_scale),
        sampling_cfg=sampling_cfg,
        q_bounds=q_bounds,
        run_cfg=run_cfg,
        extra_meta=_apply_batch_meta({}, batch_state),
    )
    status_df = pd.read_csv(status_path) if status_path.exists() else pd.DataFrame()
    write_sensitivity_cost_summaries(
        status_df,
        output_dir=summary_dir,
        run_search_root=output_paths["active_output_dir"],
    )

    print(f"[DONE] samples: {samples_path}")
    print(f"[DONE] run_meta: {meta_path}")
    print(f"[DONE] run_status: {status_path}")
    if not bool(batch_state.get("enabled")):
        if not samples_path.exists():
            raise RuntimeError(f"No valid MC samples were written: {samples_path}")
        samples_df = pd.read_csv(samples_path)
        detail_path, group_path = _write_importance_outputs(
            samples_df=samples_df,
            summary_dir=summary_dir,
            targets=targets,
            method=args.method,
            window=args.window,
            sigma=args.sigma,
            min_samples=args.min_samples,
        )
        print(f"[DONE] importance_targets: {summary_dir / 'importance_targets.csv'}")
        print(f"[DONE] importance_detail: {detail_path}")
        print(f"[DONE] importance_by_variable: {group_path}")
    else:
        print(
            "[DONE] batch outputs written. Run the merge step after all batches finish: "
            "python S5_0_2_merge_sensitivity_mc_levels_batches.py "
            f"--total-batches {batch_state['count']}"
        )


if __name__ == "__main__":
    main()
