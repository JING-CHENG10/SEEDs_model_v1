# -*- coding: utf-8 -*-
"""Greedy feasibility search for the maximum-reduction endpoint.

The search starts from the global all-lever maximum-reduction endpoint used by
S5_6_1. If that endpoint is infeasible/nonoptimal, it relaxes one scenario
measure at a time by CONFIG["step_u"] (default 0.01 in normalized MC space)
away from its reduction endpoint and reruns the model. The first feasible run
is recorded as the feasible maximum-reduction incumbent.
"""
from __future__ import annotations

import argparse
import copy
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from S1_0_schema import ScenarioConfig
from S2_0_load_data import DataPaths, build_universe_from_dict_v3
from S4_0_main import CFG, MCPrecheckFailed, build_run_baseline_cache, run_one_pipeline
from S5_cost_summary_outputs import write_sensitivity_cost_summaries
import S5_4_1_monte_carlo_full_variables as fullmc
import S5_6_1_max_emission_reduction_potential as maxred


CONFIG: Dict[str, object] = {
    "year": 2080,
    "output_dir": "",  # empty -> <NZF_OUTPUT_DIR>/Max_Emission_Reduction_Impend
    "runs_subdir": "runs",
    "scenario_prefix": "S5_6_4_IMPEND",
    "nutrition_profile_sheet": "low_land_new",
    "mc_sheet_prefer": "MC_effect_low_land_new",
    "aggregate_non_ef": False,
    "eligible_kinds": list(maxred.MAX_REDUCTION_KIND_ORDER),
    "endpoint_u_by_kind": copy.deepcopy(maxred.CONFIG.get("endpoint_u_by_kind", {}) or {}),
    "unknown_kind_u": 0.5,
    "step_u": 0.01,
    "max_attempts": 500,
    "relax_order": "sheet",  # sheet | kind_order
    "resume": False,
    "clear_existing_run_dirs_when_no_resume": False,
    "stop_on_error": False,
    "continue_after_statuses": [
        "infeasible",
        "nonoptimal",
        "precheck_failed",
        "invalid_fast_emissions",
        "invalid_market_balance",
    ],
    "dry_run": False,
    "fast_emis_only": False,
    "validate_fast_nonluc_emissions": True,
    "validate_market_balance": True,
    "market_gap_max_rate": 0.10,
    "baseline_summary_csv": "",
    "baseline_scenario_id": "S5_6_BASE",
    "sampling": {
        **copy.deepcopy(fullmc.CONFIG.get("sampling", {}) or {}),
        "method": "endpoint",
        "scope": "row",
        "shuffle": False,
        "quantile_bounds": (0.0, 1.0),
    },
    "override_cfg": {
        **copy.deepcopy(maxred.CONFIG.get("override_cfg", {}) or {}),
        "nutrition_profile_sheet": "low_land_new",
        "cost_calculation_method": "unit_cost",
        "debug_level": 0,
        "batch_mode": False,
        "linear_enable_infeasible_iis": False,
        "linear_enable_violation_iis": False,
        "linear_enable_output_diagnostics": True,
        "linear_enable_verbose_logging": False,
        "max_slack_rate": 0.1,
        "max_shortage_slack_rate": 0.1,
        "max_excess_slack_rate": 0.1,
        "market_gap_max_rate": 0.10,
        "domestic_supply_simulation_mode": "hard_equation",
        "supply_curtailment_enabled": False,
    },
}


def _default_output_dir() -> Path:
    return Path(maxred._default_output_dir()).parent / "Max_Emission_Reduction_Impend"


def _output_dir(cfg: Mapping[str, object]) -> Path:
    raw = str(cfg.get("output_dir", "") or "").strip()
    return Path(raw) if raw else _default_output_dir()


def _runs_dir(cfg: Mapping[str, object]) -> Path:
    return _output_dir(cfg) / str(cfg.get("runs_subdir", "runs") or "runs")


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _endpoint_u(kind: str, cfg: Mapping[str, object]) -> float:
    endpoint_map = cfg.get("endpoint_u_by_kind") or {}
    raw = endpoint_map.get(kind, cfg.get("unknown_kind_u", 0.5))
    try:
        val = float(raw)
    except Exception:
        val = 0.5
    return max(0.0, min(1.0, val))


def _load_specs(cfg: Mapping[str, object]) -> pd.DataFrame:
    specs = maxred._load_normalized_specs(cfg).copy()
    eligible = set(maxred._eligible_kinds(cfg, specs))
    specs = specs[specs["__kind"].astype(str).isin(eligible)].copy().reset_index(drop=True)
    specs["spec_row_id"] = np.arange(1, len(specs) + 1)
    if specs.empty:
        raise RuntimeError("No eligible S5.6 measure rows found.")
    return specs


def _initial_u_state(specs: pd.DataFrame, cfg: Mapping[str, object]) -> Dict[int, float]:
    return {
        int(row_id): _endpoint_u(str(kind), cfg)
        for row_id, kind in specs[["spec_row_id", "__kind"]].itertuples(index=False, name=None)
    }


def _relax_direction(endpoint_u: float) -> float:
    return -1.0 if endpoint_u >= 0.5 else 1.0


def _relax_limit(endpoint_u: float) -> float:
    return 0.0 if endpoint_u >= 0.5 else 1.0


def _relax_order(specs: pd.DataFrame, cfg: Mapping[str, object]) -> List[int]:
    mode = str(cfg.get("relax_order", "sheet") or "sheet").strip().lower()
    if mode == "kind_order":
        order = {k: i for i, k in enumerate(maxred.MAX_REDUCTION_KIND_ORDER)}
        work = specs.copy()
        work["_kind_order"] = work["__kind"].astype(str).map(order).fillna(9999).astype(int)
        work = work.sort_values(["_kind_order", "spec_row_id"])
        return [int(v) for v in work["spec_row_id"].tolist()]
    return [int(v) for v in specs["spec_row_id"].tolist()]


def _advance_one_step(
    u_state: Dict[int, float],
    *,
    specs_by_row: Mapping[int, Mapping[str, object]],
    order: Sequence[int],
    cursor: int,
    step: float,
) -> Tuple[bool, int, Optional[int]]:
    n = len(order)
    if n <= 0:
        return False, cursor, None
    step = max(0.0, float(step))
    if step <= 0:
        return False, cursor, None

    for offset in range(n):
        pos = (cursor + offset) % n
        row_id = int(order[pos])
        row = specs_by_row.get(row_id)
        if not row:
            continue
        endpoint = float(row.get("endpoint_u", 0.5))
        direction = _relax_direction(endpoint)
        limit = _relax_limit(endpoint)
        cur = float(u_state.get(row_id, endpoint))
        if direction < 0 and cur <= limit + 1e-12:
            continue
        if direction > 0 and cur >= limit - 1e-12:
            continue
        nxt = cur + direction * step
        if direction < 0:
            nxt = max(limit, nxt)
        else:
            nxt = min(limit, nxt)
        u_state[row_id] = round(float(nxt), 10)
        return True, (pos + 1) % n, row_id
    return False, cursor, None


def _param_rows_from_state(
    specs: pd.DataFrame,
    cfg: Mapping[str, object],
    *,
    u_state: Mapping[int, float],
) -> List[Dict[str, object]]:
    work = specs.copy().reset_index(drop=True)
    unit_row = np.array([float(u_state[int(v)]) for v in work["spec_row_id"]], dtype=float)
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
        rid = int(row.get("spec_row_id"))
        kind = str(row.get("kind", "") or "")
        endpoint = _endpoint_u(kind, cfg)
        row["strategy_endpoint_u"] = endpoint
        row["strategy_current_u"] = float(u_state[rid])
        row["strategy_relax_delta_u"] = abs(float(u_state[rid]) - endpoint)
    return rows


def _design_rows(scenario_id: str, attempt: int, rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        out.append(
            {
                "attempt": attempt,
                "scenario_id": scenario_id,
                "spec_row_id": row.get("spec_row_id"),
                "kind": row.get("kind"),
                "element_name": row.get("element_name"),
                "element_unit": row.get("element_unit"),
                "item_selector": row.get("item_selector"),
                "process_selector": row.get("process_selector"),
                "ghg_selector": row.get("ghg_selector"),
                "region_selector": row.get("region_selector"),
                "endpoint_u": row.get("strategy_endpoint_u"),
                "current_u": row.get("strategy_current_u"),
                "relax_delta_u": row.get("strategy_relax_delta_u"),
                "value_2080": row.get("abs_value"),
                "min_bound": row.get("min_bound"),
                "max_bound": row.get("max_bound"),
            }
        )
    return out


def _numeric_or_none(value: object) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _status_error_text(status: Mapping[str, object], *, max_len: int = 500) -> str:
    err_type = str(status.get("error_type", "") or "").strip()
    err_msg = str(status.get("error_message", "") or "").strip()
    if not err_type and not err_msg and str(status.get("run_status", "") or "") not in {"ok", "resumed"}:
        err_msg = str(status.get("model_status_text", "") or "").strip()
    text = f"{err_type}: {err_msg}" if err_type and err_msg else (err_type or err_msg)
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text


def _should_continue_search(status: Mapping[str, object], cfg: Mapping[str, object]) -> bool:
    run_status = str(status.get("run_status", "") or "").strip()
    continue_statuses = {
        str(v).strip()
        for v in (cfg.get("continue_after_statuses") or [])
        if str(v).strip()
    }
    return run_status in continue_statuses


def _implied_multiplier_2080(kind: object, unit: object, value_2080: object) -> object:
    """Return the 2080 multiplier implied by rate/multiplier-style effects."""
    kind_l = str(kind or "").strip().lower()
    unit_l = str(unit or "").strip().lower()
    value = _numeric_or_none(value_2080)
    if value is None:
        return ""
    multiplier_kinds = {
        "yield_rate",
        "yield_multiplier",
        "yield_improvement",
        "feed_intensity",
        "feed_efficiency",
        "emission_factor",
        "ef_multiplier",
        "emission_factor_multiplier",
        "fertilizer_rate",
        "fertlizer_rate",
        "fertilizer_efficiency",
        "manure_management_ratio",
        "mm_ratio",
        "manure_ratio",
        "crop_soil_management_ratio",
        "crop_soil_ratio",
    }
    if kind_l not in multiplier_kinds:
        return ""
    if unit_l == "rate":
        return max(0.0, 1.0 + value)
    if unit_l == "multiplier":
        return max(0.0, value)
    return ""


def _effect_setting_rows(
    *,
    scenario_id: str,
    attempt: int,
    param_rows: Sequence[Mapping[str, object]],
    shared_universe,
    cfg: Mapping[str, object],
) -> List[Dict[str, object]]:
    effects = maxred._build_effects(list(param_rows), shared_universe, cfg, scenario_id=scenario_id)
    rows: List[Dict[str, object]] = []
    for eff, row in zip(effects, param_rows):
        rows.append(
            {
                "attempt": int(attempt),
                "scenario_id": scenario_id,
                "spec_row_id": getattr(eff, "spec_row_id", row.get("spec_row_id")),
                "kind": getattr(eff, "kind", row.get("kind")),
                "unit": getattr(eff, "unit", row.get("unit")),
                "value_2080": getattr(eff, "value_2080", row.get("abs_value")),
                "implied_multiplier_2080": _implied_multiplier_2080(
                    getattr(eff, "kind", row.get("kind")),
                    getattr(eff, "unit", row.get("unit")),
                    getattr(eff, "value_2080", row.get("abs_value")),
                ),
                "country_selector": getattr(eff, "country_sel", row.get("region")),
                "commodity_selector": getattr(eff, "commodity_sel", row.get("item")),
                "process_selector": getattr(eff, "process_sel", row.get("process")),
                "ghg_selector": getattr(eff, "ghg_sel", row.get("ghg", "All")),
                "country_count": len(getattr(eff, "countries", None) or []),
                "commodity_count": len(getattr(eff, "commodities", None) or []),
                "process_count": len(getattr(eff, "processes", None) or []),
                "element_name": row.get("element_name"),
                "element_unit": row.get("element_unit"),
                "item_selector": row.get("item_selector"),
                "region_selector": row.get("region_selector"),
                "endpoint_u": row.get("strategy_endpoint_u"),
                "current_u": row.get("strategy_current_u"),
                "relax_delta_u": row.get("strategy_relax_delta_u"),
                "min_bound": row.get("min_bound"),
                "max_bound": row.get("max_bound"),
                "mc_u": row.get("mc_u"),
                "q_low": row.get("q_low"),
                "q_high": row.get("q_high"),
            }
        )
    return rows


def _settings_by_kind_rows(effect_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    if not effect_rows:
        return []
    df = pd.DataFrame(effect_rows).copy()
    for col in ("value_2080", "implied_multiplier_2080", "current_u", "endpoint_u", "relax_delta_u"):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    rows: List[Dict[str, object]] = []
    for kind, grp in df.groupby("kind", dropna=False):
        rows.append(
            {
                "kind": kind,
                "rows": int(len(grp)),
                "value_2080_min": float(grp["value_2080"].min()) if grp["value_2080"].notna().any() else np.nan,
                "value_2080_max": float(grp["value_2080"].max()) if grp["value_2080"].notna().any() else np.nan,
                "implied_multiplier_min": (
                    float(grp["implied_multiplier_2080"].min())
                    if grp["implied_multiplier_2080"].notna().any()
                    else np.nan
                ),
                "implied_multiplier_max": (
                    float(grp["implied_multiplier_2080"].max())
                    if grp["implied_multiplier_2080"].notna().any()
                    else np.nan
                ),
                "current_u_min": float(grp["current_u"].min()) if grp["current_u"].notna().any() else np.nan,
                "current_u_max": float(grp["current_u"].max()) if grp["current_u"].notna().any() else np.nan,
                "relax_delta_u_max": (
                    float(grp["relax_delta_u"].max()) if grp["relax_delta_u"].notna().any() else np.nan
                ),
            }
        )
    return rows


def _baseline_summary_candidates(cfg: Mapping[str, object], out_dir: Path) -> List[Path]:
    paths: List[Path] = []
    raw = str(cfg.get("baseline_summary_csv", "") or "").strip()
    if raw:
        paths.append(Path(raw))
    parent = out_dir.parent
    paths.extend(
        [
            parent / "Max_Emission_Reduction_Potential" / "merged" / "scenario_reduction_summary.csv",
            parent / "Max_Emission_Reduction_Potential" / "scenario_reduction_summary.csv",
            parent / "Max_Emission_Reduction_Potential" / "global_max_reduction_summary.csv",
        ]
    )
    seen = set()
    unique: List[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _read_baseline_emissions_gt(cfg: Mapping[str, object], out_dir: Path) -> Tuple[Optional[float], str]:
    scenario_id = str(cfg.get("baseline_scenario_id", "S5_6_BASE") or "S5_6_BASE")
    for path in _baseline_summary_candidates(cfg, out_dir):
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        work = df.copy()
        if "scenario_id" in work.columns:
            work = work[work["scenario_id"].astype(str).eq(scenario_id)]
        if work.empty:
            continue
        for col in ("afolu_emissions_gt_co2eq_yr", "total_co2eq_gt", "baseline_total_co2eq_gt"):
            if col not in work.columns:
                continue
            vals = pd.to_numeric(work[col], errors="coerce").dropna()
            if not vals.empty:
                return float(vals.iloc[0]), str(path)
    return None, ""


def _write_success_emission_diagnostics(
    *,
    status: Mapping[str, object],
    out_dir: Path,
    cfg: Mapping[str, object],
) -> None:
    scenario_dir_raw = str(status.get("scenario_dir", "") or "").strip()
    if not scenario_dir_raw:
        return
    scenario_dir = Path(scenario_dir_raw)
    detail_path = scenario_dir / "Emis" / "emissions_fast_global_detail.csv"
    if not detail_path.exists():
        return
    try:
        detail = pd.read_csv(detail_path)
    except Exception:
        return
    if detail.empty or "co2eq_kt" not in detail.columns:
        return
    detail = detail.copy()
    detail["co2eq_kt"] = pd.to_numeric(detail["co2eq_kt"], errors="coerce")
    detail = detail.dropna(subset=["co2eq_kt"])
    detail["co2eq_gt"] = detail["co2eq_kt"] * 1e-6
    if "emissions_kt" in detail.columns:
        detail["emissions_kt"] = pd.to_numeric(detail["emissions_kt"], errors="coerce")
    if "Y2020_co2eq_kt" in detail.columns:
        detail["Y2020_co2eq_kt"] = pd.to_numeric(detail["Y2020_co2eq_kt"], errors="coerce")
        detail["Y2020_co2eq_gt"] = detail["Y2020_co2eq_kt"] * 1e-6
    if "Y2020_emissions_kt" in detail.columns:
        detail["Y2020_emissions_kt"] = pd.to_numeric(detail["Y2020_emissions_kt"], errors="coerce")

    def _source_rows(df: pd.DataFrame) -> pd.DataFrame:
        if "row_type" not in df.columns:
            return df
        return df[df["row_type"].astype(str).str.strip().ne("co2eq_summary")].copy()

    top = detail.sort_values("co2eq_kt", ascending=False).head(50)
    _write_csv(top, out_dir / "first_feasible_emissions_top50.csv")

    source_detail = _source_rows(detail)
    group_cols = [c for c in ("year", "source_module", "Process") if c in detail.columns]
    if group_cols and not source_detail.empty:
        value_cols = ["co2eq_kt"] + [c for c in ("Y2020_co2eq_kt",) if c in source_detail.columns]
        by_process = source_detail.groupby(group_cols, dropna=False, as_index=False)[value_cols].sum()
        by_process["co2eq_gt"] = by_process["co2eq_kt"] * 1e-6
        if "Y2020_co2eq_kt" in by_process.columns:
            by_process["Y2020_co2eq_gt"] = by_process["Y2020_co2eq_kt"] * 1e-6
            by_process["delta_vs_Y2020_co2eq_gt"] = by_process["co2eq_gt"] - by_process["Y2020_co2eq_gt"]
        by_process = by_process.sort_values("co2eq_kt", ascending=False)
        _write_csv(by_process, out_dir / "first_feasible_emissions_by_process.csv")

    item_cols = [c for c in ("year", "source_module", "Process", "Item", "GHG", "row_type") if c in detail.columns]
    if item_cols:
        value_cols = ["co2eq_kt"] + [c for c in ("Y2020_co2eq_kt",) if c in detail.columns]
        by_item = detail.groupby(item_cols, dropna=False, as_index=False)[value_cols].sum()
        by_item["co2eq_gt"] = by_item["co2eq_kt"] * 1e-6
        if "Y2020_co2eq_kt" in by_item.columns:
            by_item["Y2020_co2eq_gt"] = by_item["Y2020_co2eq_kt"] * 1e-6
            by_item["delta_vs_Y2020_co2eq_gt"] = by_item["co2eq_gt"] - by_item["Y2020_co2eq_gt"]
        by_item = by_item.sort_values("co2eq_kt", ascending=False)
        _write_csv(by_item.head(500), out_dir / "first_feasible_emissions_by_source_process_item.csv")

    baseline_gt, baseline_source = _read_baseline_emissions_gt(cfg, out_dir)
    total_gt = _numeric_or_none(status.get("afolu_emissions_gt_co2eq_yr"))
    comparison = {
        "scenario_id": status.get("scenario_id"),
        "scenario_dir": scenario_dir_raw,
        "afolu_emissions_gt_co2eq_yr": total_gt,
        "baseline_scenario_id": cfg.get("baseline_scenario_id", "S5_6_BASE"),
        "baseline_total_co2eq_gt": baseline_gt,
        "baseline_source": baseline_source,
        "delta_vs_baseline_gt": None,
        "reduction_vs_baseline_pct": None,
        "higher_than_baseline": "",
        "top_source_module": "",
        "top_process": "",
        "top_item": "",
        "top_ghg": "",
        "top_co2eq_gt": None,
    }
    if baseline_gt is not None and total_gt is not None:
        comparison["delta_vs_baseline_gt"] = total_gt - baseline_gt
        comparison["reduction_vs_baseline_pct"] = (
            (baseline_gt - total_gt) / baseline_gt * 100.0 if baseline_gt != 0 else np.nan
        )
        comparison["higher_than_baseline"] = bool(total_gt > baseline_gt)
    if not top.empty:
        first = top.iloc[0]
        comparison["top_source_module"] = first.get("source_module", "")
        comparison["top_process"] = first.get("Process", "")
        comparison["top_item"] = first.get("Item", "")
        comparison["top_ghg"] = first.get("GHG", "")
        comparison["top_co2eq_gt"] = float(first.get("co2eq_gt", np.nan))
    _write_csv(pd.DataFrame([comparison]), out_dir / "first_feasible_emissions_comparison.csv")


def _write_success_cost_diagnostics(
    *,
    status: Mapping[str, object],
    out_dir: Path,
) -> None:
    scenario_dir_raw = str(status.get("scenario_dir", "") or "").strip()
    if not scenario_dir_raw:
        return
    scenario_dir = Path(scenario_dir_raw)
    cost_path = scenario_dir / "cost_summary.csv"
    summary: Dict[str, object] = {
        "scenario_id": status.get("scenario_id"),
        "scenario_dir": scenario_dir_raw,
        "cost_summary_source": str(cost_path),
        "cost_summary_found": bool(cost_path.exists()),
        "cost_rows": 0,
        "year_min": None,
        "year_max": None,
        "total_abatement_tco2eq": None,
        "total_cost_usd": None,
        "avg_unit_cost_usd_per_tco2eq": None,
    }
    if not cost_path.exists():
        _write_csv(pd.DataFrame([summary]), out_dir / "first_feasible_cost_summary.csv")
        return
    try:
        cost_df = pd.read_csv(cost_path)
    except Exception as exc:
        summary["cost_summary_found"] = False
        summary["error_type"] = type(exc).__name__
        summary["error_message"] = str(exc)
        _write_csv(pd.DataFrame([summary]), out_dir / "first_feasible_cost_summary.csv")
        return

    detail = cost_df.copy()
    detail.insert(0, "scenario_id", status.get("scenario_id"))
    detail.insert(1, "scenario_dir", scenario_dir_raw)
    for col in ("year", "abatement_tco2eq", "unit_cost_usd_per_tco2eq", "total_cost_usd"):
        if col in detail.columns:
            detail[col] = pd.to_numeric(detail[col], errors="coerce")
    _write_csv(detail, out_dir / "first_feasible_cost_detail.csv")

    summary["cost_rows"] = int(len(detail))
    if "year" in detail.columns and detail["year"].notna().any():
        summary["year_min"] = int(detail["year"].min())
        summary["year_max"] = int(detail["year"].max())
    total_abatement = (
        float(detail["abatement_tco2eq"].sum()) if "abatement_tco2eq" in detail.columns else np.nan
    )
    total_cost = float(detail["total_cost_usd"].sum()) if "total_cost_usd" in detail.columns else np.nan
    summary["total_abatement_tco2eq"] = total_abatement
    summary["total_cost_usd"] = total_cost
    summary["avg_unit_cost_usd_per_tco2eq"] = (
        total_cost / total_abatement if np.isfinite(total_cost) and total_abatement > 0 else np.nan
    )
    _write_csv(pd.DataFrame([summary]), out_dir / "first_feasible_cost_summary.csv")

    group_cols = [c for c in ("year", "process") if c in detail.columns]
    value_cols = [c for c in ("abatement_tco2eq", "total_cost_usd") if c in detail.columns]
    if group_cols and value_cols:
        by_process = detail.groupby(group_cols, dropna=False, as_index=False)[value_cols].sum()
        if {"abatement_tco2eq", "total_cost_usd"}.issubset(by_process.columns):
            by_process["avg_unit_cost_usd_per_tco2eq"] = np.where(
                by_process["abatement_tco2eq"] > 0,
                by_process["total_cost_usd"] / by_process["abatement_tco2eq"],
                np.nan,
            )
        by_process = by_process.sort_values(
            [c for c in ("year", "total_cost_usd", "process") if c in by_process.columns],
            ascending=[True, False, True][: len([c for c in ("year", "total_cost_usd", "process") if c in by_process.columns])],
        )
        _write_csv(by_process, out_dir / "first_feasible_cost_by_process.csv")


def _write_first_feasible_detail_outputs(
    *,
    status: Mapping[str, object],
    design_rows: Sequence[Mapping[str, object]],
    param_rows: Sequence[Mapping[str, object]],
    shared_universe,
    cfg: Mapping[str, object],
    out_dir: Path,
) -> None:
    scenario_id = str(status.get("scenario_id", "") or "")
    attempt = int(status.get("attempt", 0) or 0)
    _write_csv(pd.DataFrame([status]), out_dir / "first_feasible_summary.csv")
    _write_csv(pd.DataFrame(design_rows), out_dir / "first_feasible_strategy_long.csv")

    if param_rows:
        effect_rows = _effect_setting_rows(
            scenario_id=scenario_id,
            attempt=attempt,
            param_rows=param_rows,
            shared_universe=shared_universe,
            cfg=cfg,
        )
        _write_csv(pd.DataFrame(effect_rows), out_dir / "first_feasible_effects_applied.csv")
        _write_csv(pd.DataFrame(_settings_by_kind_rows(effect_rows)), out_dir / "first_feasible_settings_by_kind.csv")

    _write_success_emission_diagnostics(status=status, out_dir=out_dir, cfg=cfg)
    _write_success_cost_diagnostics(status=status, out_dir=out_dir)


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


def _can_resume_attempt(plan: maxred.ScenarioPlan, scenario_dir: Path) -> bool:
    if not maxred._can_resume(plan, scenario_dir):
        return False
    if not bool(getattr(plan, "fast_emis_only", True)) and not (scenario_dir / "cost_summary.csv").exists():
        return False
    return True


def _run_attempt(
    *,
    scenario_id: str,
    attempt: int,
    param_rows: List[Dict[str, object]],
    paths: DataPaths,
    shared_cfg: ScenarioConfig,
    shared_universe,
    shared_run_cache: Dict[str, object],
    cfg: Mapping[str, object],
) -> Dict[str, object]:
    runs_dir = _runs_dir(cfg)
    scenario_dir = runs_dir / scenario_id
    plan = maxred.ScenarioPlan(
        scenario_id=scenario_id,
        scope="global",
        strategy_name="impend_relaxed_max_reduction",
        include_kinds=tuple(sorted({str(r.get("kind", "")) for r in param_rows if str(r.get("kind", "")).strip()})),
        fast_emis_only=bool(cfg.get("fast_emis_only", True)),
        require_country_detail=False,
    )
    status = maxred._status_base(plan, scenario_dir, shared_universe)
    status["attempt"] = int(attempt)

    if bool(cfg.get("dry_run", False)):
        status["run_status"] = "dry_run"
        return status

    if bool(cfg.get("resume", True)) and _can_resume_attempt(plan, scenario_dir):
        status["run_status"] = "resumed"
        status = maxred._finalize_status_from_outputs(status, scenario_dir, cfg)
        if status["run_status"] == "ok":
            status["run_status"] = "resumed"
        return status

    if not bool(cfg.get("resume", True)) and bool(cfg.get("clear_existing_run_dirs_when_no_resume", False)):
        _clear_scenario_dir(scenario_dir, runs_dir)

    try:
        effects = maxred._build_effects(param_rows, shared_universe, cfg, scenario_id=scenario_id)
        outdir = run_one_pipeline(
            paths,
            pre_macc_e0=False,
            scenario_id=scenario_id,
            scenario_effects=effects,
            solve=True,
            use_fao_modules=True,
            save_root=str(runs_dir),
            future_last_only=True,
            use_linear=True,
            fast_emis_only=bool(cfg.get("fast_emis_only", True)),
            fast_emis_year=int(cfg.get("year", 2080) or 2080),
            prebuilt_config=shared_cfg,
            prebuilt_universe=shared_universe,
            prebuilt_run_cache=shared_run_cache,
        )
        status["scenario_dir"] = str(outdir)
        status = maxred._finalize_status_from_outputs(status, Path(outdir), cfg)
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
    return status


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search first feasible S5.6 maximum-reduction incumbent.")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--step-u", type=float, default=None)
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--clear-existing-runs", action="store_true", default=None)
    parser.add_argument("--keep-existing-runs", action="store_false", dest="clear_existing_runs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--iis", action="store_true")
    parser.add_argument("--iis-timeout", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    return parser.parse_args(argv)


def _effective_config(args: argparse.Namespace) -> Dict[str, object]:
    cfg = copy.deepcopy(CONFIG)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.max_attempts is not None:
        cfg["max_attempts"] = int(args.max_attempts)
    if args.step_u is not None:
        cfg["step_u"] = float(args.step_u)
    if args.resume is not None:
        cfg["resume"] = bool(args.resume)
    if args.clear_existing_runs is not None:
        cfg["clear_existing_run_dirs_when_no_resume"] = bool(args.clear_existing_runs)
    if args.dry_run:
        cfg["dry_run"] = True
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
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    paths = DataPaths()
    shared_cfg = ScenarioConfig()
    shared_universe = build_universe_from_dict_v3(paths.dict_v3_path, shared_cfg)
    specs = _load_specs(cfg)

    endpoint_by_row = {
        int(row_id): _endpoint_u(str(kind), cfg)
        for row_id, kind in specs[["spec_row_id", "__kind"]].itertuples(index=False, name=None)
    }
    specs_by_row = {
        int(row_id): {
            "kind": str(kind),
            "endpoint_u": endpoint_by_row[int(row_id)],
        }
        for row_id, kind in specs[["spec_row_id", "__kind"]].itertuples(index=False, name=None)
    }
    order = _relax_order(specs, cfg)
    u_state = _initial_u_state(specs, cfg)

    print(f"[S5_6_4] output_dir={out_dir}")
    print(f"[S5_6_4] specs={len(specs)} step_u={cfg.get('step_u')} max_attempts={cfg.get('max_attempts')}")
    print(
        f"[S5_6_4] resume={bool(cfg.get('resume', False))} "
        f"clear_existing_runs={bool(cfg.get('clear_existing_run_dirs_when_no_resume', False))}"
    )

    backup = maxred._apply_cfg_overrides(cfg)
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
        design_rows_all: List[Dict[str, object]] = []
        cursor = 0
        relaxed_last: Optional[int] = None
        max_attempts_raw = cfg.get("max_attempts", 500)
        max_attempts = max(0, int(500 if max_attempts_raw is None else max_attempts_raw))
        step_raw = cfg.get("step_u", 0.01)
        step = float(0.01 if step_raw is None else step_raw)
        feasible_status: Optional[Dict[str, object]] = None
        feasible_design: List[Dict[str, object]] = []
        feasible_param_rows: List[Dict[str, object]] = []

        for attempt in range(max_attempts + 1):
            scenario_id = f"{cfg.get('scenario_prefix', 'S5_6_4_IMPEND')}_{attempt:04d}"
            rows = _param_rows_from_state(specs, cfg, u_state=u_state)
            design_rows = _design_rows(scenario_id, attempt, rows)
            for drow in design_rows:
                drow["relaxed_last_spec_row_id"] = relaxed_last if relaxed_last is not None else ""

            print(f"[S5_6_4] attempt={attempt}/{max_attempts} scenario={scenario_id} relaxed_last={relaxed_last}")
            status = _run_attempt(
                scenario_id=scenario_id,
                attempt=attempt,
                param_rows=rows,
                paths=paths,
                shared_cfg=shared_cfg,
                shared_universe=shared_universe,
                shared_run_cache=shared_run_cache,
                cfg=cfg,
            )
            status["relaxed_last_spec_row_id"] = relaxed_last if relaxed_last is not None else ""
            status_rows.append(status)
            design_rows_all.extend(design_rows)

            _write_csv(pd.DataFrame(status_rows), out_dir / "impend_attempt_status.csv")
            _write_csv(pd.DataFrame(design_rows_all), out_dir / "impend_strategy_design_long.csv")
            _write_csv(
                pd.DataFrame(
                    [{"spec_row_id": rid, "current_u": val, "endpoint_u": endpoint_by_row[rid]} for rid, val in sorted(u_state.items())]
                ),
                out_dir / "impend_current_u_state.csv",
            )

            print(
                f"[S5_6_4] -> {status.get('run_status')} "
                f"emissions={status.get('afolu_emissions_gt_co2eq_yr')}"
            )
            error_text = _status_error_text(status)
            if error_text:
                print(f"[S5_6_4] error: {error_text}")
            if str(status.get("run_status")) in {"ok", "resumed"}:
                feasible_status = dict(status)
                feasible_design = list(design_rows)
                feasible_param_rows = list(rows)
                break

            if not _should_continue_search(status, cfg):
                print(
                    f"[S5_6_4] stop: status={status.get('run_status')} is not in "
                    f"continue_after_statuses={cfg.get('continue_after_statuses')}"
                )
                break

            advanced, cursor, relaxed_last = _advance_one_step(
                u_state,
                specs_by_row=specs_by_row,
                order=order,
                cursor=cursor,
                step=step,
            )
            if not advanced:
                print("[S5_6_4] no more relaxable measure variables; stop")
                break

        if feasible_status is not None:
            _write_first_feasible_detail_outputs(
                status=feasible_status,
                design_rows=feasible_design,
                param_rows=feasible_param_rows,
                shared_universe=shared_universe,
                cfg=cfg,
                out_dir=out_dir,
            )
            print(f"[S5_6_4] first feasible attempt={feasible_status.get('attempt')}")
            print(f"[S5_6_4] emissions={feasible_status.get('afolu_emissions_gt_co2eq_yr')}")
        else:
            print("[S5_6_4] no feasible attempt found")
        write_sensitivity_cost_summaries(
            pd.DataFrame(status_rows),
            output_dir=out_dir,
            run_search_root=out_dir,
        )
    finally:
        maxred._restore_cfg_overrides(backup)


if __name__ == "__main__":
    main()
