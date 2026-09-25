# -*- coding: utf-8 -*-
"""Calculate land carbon price requirements along the S5.7 strategy MACC.

This is the second stage of the S5.7 MACC workflow.

Stage 1 runs the nine-strategy maximum-reduction package, excludes land carbon
price, calculates exact Shapley contributions, and derives cost-level strategy
implementation fractions from singleton cost summaries.

Stage 2 reads those implementation fractions. For every configured mitigation
cost level, it applies the corresponding partial strategy package and evaluates
an ordered grid of land carbon prices. The first price bracket that reduces
2080 AFOLU emissions to the configured target is used to estimate the required
price for reforestation CDR.

The land carbon price is not included in the strategy-package contribution
allocation. It is reported separately as the CDR price required to remove the
remaining emissions after agricultural and demand-side mitigation.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import S5_7_1_strategy_endpoint_rerun_max_reduction_potential as s57
from S5_cost_summary_outputs import write_sensitivity_cost_summaries


CONFIG: Dict[str, object] = {
    "source_dir": "",
    "output_dir": "",
    "runs_subdir": "runs",
    "year": 2080,
    "target_emissions_gt": float(
        (s57.CONFIG.get("cdr_price", {}) or {}).get("target_emissions_gt", 0.0) or 0.0
    ),
    "land_price_grid_usd_per_tco2eq": copy.deepcopy(
        (s57.CONFIG.get("cdr_price", {}) or {}).get("land_price_grid_usd_per_tco2eq", [])
    ),
    "partial_baseline_value_by_kind": copy.deepcopy(
        (s57.CONFIG.get("cdr_price", {}) or {}).get("partial_baseline_value_by_kind", {})
    ),
    "strict_strategy_fractions": True,
    "resume": True,
    "dry_run": False,
    "stop_on_error": False,
    "max_cost_levels": None,
    "validate_fast_nonluc_emissions": True,
    "validate_market_balance": True,
    "market_gap_max_rate": 0.05,
    "override_cfg": {
        **copy.deepcopy(s57.CONFIG.get("override_cfg", {}) or {}),
        "cost_calculation_method": "unit_cost",
        "debug_level": 0,
        "batch_mode": False,
        "linear_enable_infeasible_iis": False,
        "linear_enable_violation_iis": False,
        "linear_enable_output_diagnostics": False,
        "linear_enable_verbose_logging": False,
    },
}


def _source_dir(cfg: Mapping[str, object]) -> Path:
    raw = str(cfg.get("source_dir", "") or "").strip()
    if raw:
        return Path(raw)
    root = s57._default_output_dir()
    merged = root / "merged"
    return merged if merged.exists() else root


def _output_dir(cfg: Mapping[str, object]) -> Path:
    raw = str(cfg.get("output_dir", "") or "").strip()
    if raw:
        return Path(raw)
    return s57._default_output_dir() / "cdr_price_curve"


def _runs_dir(cfg: Mapping[str, object]) -> Path:
    return _output_dir(cfg) / str(cfg.get("runs_subdir", "runs") or "runs")


def _safe_float_list(raw_values: Sequence[object]) -> List[float]:
    values: List[float] = []
    for raw in raw_values:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value >= 0:
            values.append(value)
    return sorted(set(values))


def _load_strategy_profiles(cfg: Mapping[str, object]) -> pd.DataFrame:
    path = _source_dir(cfg) / "macc_strategy_cost_profiles.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing S5.7 strategy cost profiles: {path}")
    profiles = pd.read_csv(path)
    required = {
        "cost_level_usd_per_tco2eq",
        "kind",
        "implementation_fraction",
    }
    missing = sorted(required.difference(profiles.columns))
    if missing:
        raise ValueError(f"Strategy cost profile is missing columns: {missing}")
    profiles["cost_level_usd_per_tco2eq"] = pd.to_numeric(
        profiles["cost_level_usd_per_tco2eq"],
        errors="coerce",
    )
    profiles["implementation_fraction"] = pd.to_numeric(
        profiles["implementation_fraction"],
        errors="coerce",
    )
    profiles = profiles.dropna(subset=["cost_level_usd_per_tco2eq", "kind"])
    profiles = profiles[
        profiles["kind"].astype(str).isin(set(s57.STRATEGY_KIND_ORDER))
    ].copy()
    return profiles


def _cost_level_fraction_map(
    profiles: pd.DataFrame,
    cost_level: float,
    cfg: Mapping[str, object],
) -> Dict[str, float]:
    rows = profiles[
        np.isclose(
            profiles["cost_level_usd_per_tco2eq"].to_numpy(dtype=float),
            float(cost_level),
            rtol=0.0,
            atol=1e-9,
        )
    ].copy()
    by_kind = {
        str(row.kind): float(row.implementation_fraction)
        for row in rows[["kind", "implementation_fraction"]].itertuples(index=False)
        if np.isfinite(row.implementation_fraction)
    }
    package_kinds = [
        kind
        for kind in s57.STRATEGY_KIND_ORDER
        if kind in set(profiles["kind"].astype(str))
    ]
    missing = [kind for kind in package_kinds if kind not in by_kind]
    if missing and bool(cfg.get("strict_strategy_fractions", True)):
        raise ValueError(
            f"Missing strategy implementation fractions at cost level {cost_level}: {missing}"
        )
    return {
        kind: min(1.0, max(0.0, float(by_kind.get(kind, 0.0))))
        for kind in s57.STRATEGY_KIND_ORDER
    }


def _neutral_value(
    row: Mapping[str, object],
    kind: str,
    cfg: Mapping[str, object],
) -> float:
    row_neutral = row.get("strategy_neutral_value")
    if row_neutral is not None and not pd.isna(row_neutral):
        if isinstance(row_neutral, str):
            row_key = row_neutral.strip().lower()
            if row_key == "max_bound":
                return float(row.get("max_bound"))
            if row_key == "min_bound":
                return float(row.get("min_bound"))
        return float(row_neutral)
    neutral_map = cfg.get("partial_baseline_value_by_kind") or {}
    raw = neutral_map.get(kind, 0.0) if isinstance(neutral_map, Mapping) else 0.0
    if isinstance(raw, str):
        key = raw.strip().lower()
        if key == "max_bound":
            return float(row.get("max_bound"))
        if key == "min_bound":
            return float(row.get("min_bound"))
    return float(raw)


def _set_row_absolute_value(row: Dict[str, object], value: float) -> None:
    row["abs_value"] = float(value)
    row["continuous_draw"] = float(value)
    lo = row.get("min_bound")
    hi = row.get("max_bound")
    try:
        lo_f = float(lo)
        hi_f = float(hi)
        row["mc_u"] = (float(value) - lo_f) / (hi_f - lo_f) if hi_f != lo_f else np.nan
    except (TypeError, ValueError):
        row["mc_u"] = np.nan


def _build_partial_package_rows(
    specs: pd.DataFrame,
    fractions: Mapping[str, float],
    land_price: float,
    cfg: Mapping[str, object],
) -> List[Dict[str, object]]:
    package_rows = s57._build_param_rows(
        specs,
        s57.CONFIG,
        include_kinds=s57.STRATEGY_KIND_ORDER,
        country=None,
    )
    for row in package_rows:
        kind = str(row.get("strategy_kind", "") or "")
        endpoint = float(row.get("abs_value"))
        neutral = _neutral_value(row, kind, cfg)
        fraction = min(1.0, max(0.0, float(fractions.get(kind, 0.0))))
        partial_value = neutral + fraction * (endpoint - neutral)
        _set_row_absolute_value(row, partial_value)
        row["macc_implementation_fraction"] = fraction
        row["macc_endpoint_value"] = endpoint
        row["macc_neutral_value"] = neutral

    land_rows = s57._build_param_rows(
        specs,
        s57.CONFIG,
        include_kinds=(s57.CDR_PRICE_KIND,),
        country=None,
    )
    for row in land_rows:
        _set_row_absolute_value(row, float(land_price))
        row["strategy_endpoint"] = "CDR_land_price_grid"
        row["strategy_endpoint_source"] = "S5_7_3_price_grid"
        row["configured_endpoint_value"] = float(land_price)
    return package_rows + land_rows


def _scenario_id(cost_level: float, land_price: float) -> str:
    cost_token = s57.s56._safe_token(f"{cost_level:g}")
    price_token = s57.s56._safe_token(f"{land_price:g}")
    return f"S5_7_MACC_C{cost_token}_LCP{price_token}"


def _response_status_base(
    scenario_id: str,
    scenario_dir: Path,
    cost_level: float,
    land_price: float,
    fractions: Mapping[str, float],
) -> Dict[str, object]:
    return {
        "scenario_id": scenario_id,
        "scenario_dir": str(scenario_dir),
        "cost_level_usd_per_tco2eq": float(cost_level),
        "land_carbon_price_usd_per_tco2eq": float(land_price),
        "strategy_fraction_count": int(len(fractions)),
        "strategy_fractions": ";".join(
            f"{kind}={float(fractions.get(kind, 0.0)):.10g}"
            for kind in s57.STRATEGY_KIND_ORDER
        ),
        "run_status": "pending",
        "afolu_emissions_gt_co2eq_yr": np.nan,
        "error_type": "",
        "error_message": "",
    }


def _run_response_scenario(
    *,
    cost_level: float,
    land_price: float,
    fractions: Mapping[str, float],
    specs: pd.DataFrame,
    paths,
    shared_cfg,
    shared_universe,
    shared_run_cache: Dict[str, object],
    cfg: Mapping[str, object],
) -> Dict[str, object]:
    scenario_id = _scenario_id(cost_level, land_price)
    runs_dir = _runs_dir(cfg)
    scenario_dir = runs_dir / scenario_id
    status = _response_status_base(
        scenario_id,
        scenario_dir,
        cost_level,
        land_price,
        fractions,
    )

    if bool(cfg.get("dry_run", False)):
        status["run_status"] = "dry_run"
        return status

    if bool(cfg.get("resume", True)) and s57.s56._can_resume(
        s57.s56.ScenarioPlan(
            scenario_id=scenario_id,
            scope="global",
            strategy_name="macc_package_with_cdr_price",
            include_kinds=tuple(s57.STRATEGY_KIND_ORDER),
            fast_emis_only=True,
        ),
        scenario_dir,
    ):
        status = s57.s56._finalize_status_from_outputs(status, scenario_dir, cfg)
        if status.get("run_status") == "ok":
            status["run_status"] = "resumed"
        return status

    try:
        param_rows = _build_partial_package_rows(specs, fractions, land_price, cfg)
        effects = s57._build_effects(
            param_rows,
            shared_universe,
            cfg,
            scenario_id=scenario_id,
        )
        outdir = s57.s56.run_one_pipeline(
            paths,
            pre_macc_e0=False,
            scenario_id=scenario_id,
            scenario_effects=effects,
            solve=True,
            use_fao_modules=True,
            save_root=str(runs_dir),
            future_last_only=True,
            use_linear=True,
            fast_emis_only=True,
            fast_emis_year=int(cfg.get("year", 2080) or 2080),
            prebuilt_config=shared_cfg,
            prebuilt_universe=shared_universe,
            prebuilt_run_cache=shared_run_cache,
        )
        scenario_dir = Path(outdir)
        status["scenario_dir"] = str(scenario_dir)
        status = s57.s56._finalize_status_from_outputs(status, scenario_dir, cfg)
    except Exception as exc:
        status["run_status"] = "failed"
        status["error_type"] = type(exc).__name__
        status["error_message"] = str(exc)
        if bool(cfg.get("stop_on_error", False)):
            raise
    return status


def _required_price_row(
    cost_level: float,
    response_rows: Sequence[Mapping[str, object]],
    target_emissions_gt: float,
) -> Dict[str, object]:
    response = pd.DataFrame(response_rows)
    if response.empty:
        return {
            "cost_level_usd_per_tco2eq": cost_level,
            "price_status": "missing_response",
        }
    response["land_carbon_price_usd_per_tco2eq"] = pd.to_numeric(
        response["land_carbon_price_usd_per_tco2eq"],
        errors="coerce",
    )
    response["afolu_emissions_gt_co2eq_yr"] = pd.to_numeric(
        response["afolu_emissions_gt_co2eq_yr"],
        errors="coerce",
    )
    valid = response[
        response["run_status"].astype(str).isin(["ok", "resumed"])
        & response["land_carbon_price_usd_per_tco2eq"].notna()
        & response["afolu_emissions_gt_co2eq_yr"].notna()
    ].sort_values("land_carbon_price_usd_per_tco2eq")
    if valid.empty:
        return {
            "cost_level_usd_per_tco2eq": cost_level,
            "price_status": "no_valid_model_runs",
        }

    zero_rows = valid[np.isclose(valid["land_carbon_price_usd_per_tco2eq"], 0.0)]
    emissions_at_zero = (
        float(zero_rows.iloc[0]["afolu_emissions_gt_co2eq_yr"])
        if not zero_rows.empty
        else np.nan
    )
    crossing = valid[valid["afolu_emissions_gt_co2eq_yr"].le(target_emissions_gt)]
    if crossing.empty:
        last = valid.iloc[-1]
        return {
            "cost_level_usd_per_tco2eq": cost_level,
            "target_emissions_gt": target_emissions_gt,
            "emissions_at_zero_land_price_gt": emissions_at_zero,
            "residual_cdr_requirement_gt": max(0.0, emissions_at_zero - target_emissions_gt)
            if np.isfinite(emissions_at_zero)
            else np.nan,
            "required_land_carbon_price_usd_per_tco2eq": np.nan,
            "lower_bracket_price": float(last["land_carbon_price_usd_per_tco2eq"]),
            "upper_bracket_price": np.nan,
            "price_status": "target_not_reached_within_grid",
        }

    upper = crossing.iloc[0]
    upper_price = float(upper["land_carbon_price_usd_per_tco2eq"])
    upper_emissions = float(upper["afolu_emissions_gt_co2eq_yr"])
    lower_candidates = valid[
        valid["land_carbon_price_usd_per_tco2eq"].lt(upper_price)
        & valid["afolu_emissions_gt_co2eq_yr"].gt(target_emissions_gt)
    ]
    if upper_price == 0 or lower_candidates.empty:
        required_price = upper_price
        lower_price = upper_price
        lower_emissions = upper_emissions
        status = "target_reached_at_grid_price"
    else:
        lower = lower_candidates.iloc[-1]
        lower_price = float(lower["land_carbon_price_usd_per_tco2eq"])
        lower_emissions = float(lower["afolu_emissions_gt_co2eq_yr"])
        denominator = lower_emissions - upper_emissions
        if denominator > 0:
            share = (lower_emissions - target_emissions_gt) / denominator
            required_price = lower_price + share * (upper_price - lower_price)
            status = "interpolated_within_price_bracket"
        else:
            required_price = upper_price
            status = "nonmonotonic_bracket_use_upper_price"

    return {
        "cost_level_usd_per_tco2eq": cost_level,
        "target_emissions_gt": target_emissions_gt,
        "emissions_at_zero_land_price_gt": emissions_at_zero,
        "residual_cdr_requirement_gt": max(0.0, emissions_at_zero - target_emissions_gt)
        if np.isfinite(emissions_at_zero)
        else np.nan,
        "required_land_carbon_price_usd_per_tco2eq": required_price,
        "lower_bracket_price": lower_price,
        "lower_bracket_emissions_gt": lower_emissions,
        "upper_bracket_price": upper_price,
        "upper_bracket_emissions_gt": upper_emissions,
        "price_status": status,
    }


def _write_progress(
    response_rows: Sequence[Mapping[str, object]],
    required_rows: Sequence[Mapping[str, object]],
    cfg: Mapping[str, object],
) -> None:
    out_dir = _output_dir(cfg)
    s57.s56._ensure_dir(out_dir)
    s57.s56._write_csv(
        pd.DataFrame(response_rows),
        out_dir / "cdr_land_price_response.csv",
    )
    s57.s56._write_csv(
        pd.DataFrame(required_rows),
        out_dir / "cdr_required_land_carbon_price.csv",
    )


def _write_curve_with_cdr(required_df: pd.DataFrame, cfg: Mapping[str, object]) -> None:
    source_totals = _source_dir(cfg) / "macc_strategy_curve_totals.csv"
    if not source_totals.exists() or required_df.empty:
        return
    totals = pd.read_csv(source_totals)
    totals["cost_level_usd_per_tco2eq"] = pd.to_numeric(
        totals["cost_level_usd_per_tco2eq"],
        errors="coerce",
    )
    merged = totals.merge(
        required_df,
        on="cost_level_usd_per_tco2eq",
        how="left",
    )
    s57.s56._write_csv(
        merged,
        _output_dir(cfg) / "macc_strategy_curve_with_cdr_price.csv",
    )


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S5.7 MACC land-carbon-price response.")
    parser.add_argument("--source-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-cost-levels", type=int, default=None)
    parser.add_argument("--target-emissions-gt", type=float, default=None)
    parser.add_argument("--land-price-grid", type=str, default=None)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--threads", type=int, default=None)
    return parser.parse_args(list(argv) if argv is not None else None)


def _effective_config(args: argparse.Namespace) -> Dict[str, object]:
    cfg = copy.deepcopy(CONFIG)
    if args.source_dir:
        cfg["source_dir"] = args.source_dir
    if args.out_dir:
        cfg["output_dir"] = args.out_dir
    if args.dry_run:
        cfg["dry_run"] = True
    if args.no_resume:
        cfg["resume"] = False
    if args.max_cost_levels is not None:
        cfg["max_cost_levels"] = int(args.max_cost_levels)
    if args.target_emissions_gt is not None:
        cfg["target_emissions_gt"] = float(args.target_emissions_gt)
    if args.land_price_grid:
        cfg["land_price_grid_usd_per_tco2eq"] = [
            part.strip() for part in str(args.land_price_grid).split(",") if part.strip()
        ]
    if args.stop_on_error:
        cfg["stop_on_error"] = True
    if args.threads is not None:
        override = copy.deepcopy(cfg.get("override_cfg", {}) or {})
        override["linear_solver_threads"] = int(args.threads)
        cfg["override_cfg"] = override
    return cfg


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    cfg = _effective_config(args)
    profiles = _load_strategy_profiles(cfg)
    cost_levels = sorted(
        profiles["cost_level_usd_per_tco2eq"].dropna().astype(float).unique().tolist()
    )
    max_levels = cfg.get("max_cost_levels")
    if max_levels is not None:
        max_n = int(max_levels)
        if max_n <= 0:
            raise ValueError("max_cost_levels must be positive or None")
        cost_levels = cost_levels[:max_n]
    price_grid = _safe_float_list(cfg.get("land_price_grid_usd_per_tco2eq") or [])
    if not price_grid:
        raise ValueError("land_price_grid_usd_per_tco2eq is empty")

    out_dir = _output_dir(cfg)
    runs_dir = _runs_dir(cfg)
    s57.s56._ensure_dir(out_dir)
    s57.s56._ensure_dir(runs_dir)

    paths = s57.s56.DataPaths()
    shared_cfg = s57.s56.ScenarioConfig()
    shared_universe = s57.s56.build_universe_from_dict_v3(paths.dict_v3_path, shared_cfg)
    specs = s57._prepare_strategy_specs(
        s57.s56._load_normalized_specs(s57.CONFIG),
        s57.CONFIG,
    )

    print(f"[S5_7_CDR] source_dir={_source_dir(cfg)}")
    print(f"[S5_7_CDR] output_dir={out_dir}")
    print(f"[S5_7_CDR] cost_levels={len(cost_levels)} price_grid={len(price_grid)}")

    backup = s57.s56._apply_cfg_overrides(cfg)
    try:
        shared_run_cache = (
            {}
            if bool(cfg.get("dry_run", False))
            else s57.s56.build_run_baseline_cache(
                paths,
                shared_cfg,
                shared_universe,
                future_last_only=True,
            )
        )
        response_rows: List[Dict[str, object]] = []
        required_rows: List[Dict[str, object]] = []
        target = float(cfg.get("target_emissions_gt", 0.0) or 0.0)

        for level_index, cost_level in enumerate(cost_levels, start=1):
            fractions = _cost_level_fraction_map(profiles, cost_level, cfg)
            level_rows: List[Dict[str, object]] = []
            print(
                f"[S5_7_CDR] cost_level={cost_level:g} "
                f"index={level_index}/{len(cost_levels)}"
            )
            for land_price in price_grid:
                status = _run_response_scenario(
                    cost_level=cost_level,
                    land_price=land_price,
                    fractions=fractions,
                    specs=specs,
                    paths=paths,
                    shared_cfg=shared_cfg,
                    shared_universe=shared_universe,
                    shared_run_cache=shared_run_cache,
                    cfg=cfg,
                )
                level_rows.append(status)
                response_rows.append(status)
                print(
                    f"[S5_7_CDR] cost={cost_level:g} land_price={land_price:g} "
                    f"status={status.get('run_status')} "
                    f"emissions={status.get('afolu_emissions_gt_co2eq_yr')}"
                )
                emissions = pd.to_numeric(
                    pd.Series([status.get("afolu_emissions_gt_co2eq_yr")]),
                    errors="coerce",
                ).iloc[0]
                if (
                    status.get("run_status") in {"ok", "resumed"}
                    and np.isfinite(emissions)
                    and emissions <= target
                ):
                    break

            required_rows.append(_required_price_row(cost_level, level_rows, target))
            _write_progress(response_rows, required_rows, cfg)

        required_df = pd.DataFrame(required_rows)
        _write_curve_with_cdr(required_df, cfg)
        write_sensitivity_cost_summaries(
            pd.DataFrame(response_rows),
            output_dir=out_dir,
            run_search_root=out_dir,
        )
    finally:
        s57.s56._restore_cfg_overrides(backup)


if __name__ == "__main__":
    main()
