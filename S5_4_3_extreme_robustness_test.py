# -*- coding: utf-8 -*-
"""Run five endpoint stress scenarios through S5_4.

This is a diagnostic harness for S5_4_1_monte_carlo_full_variables.py. It
keeps the production Monte Carlo sampler untouched and temporarily injects a
fixed 5-row U matrix that maps directly to Scenario_config_new.xlsx
MC_effect_low_land_new Min_bound/Max_bound values.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

from config_paths import get_results_base
import S5_4_1_monte_carlo_full_variables as fullmc


EXTREME_KINDS = [
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


EXTREME_SCENARIOS: List[Dict[str, object]] = [
    {
        "sample_id": 1,
        "scenario_id": "MC_FULL_00001",
        "extreme_name": "E01_all_min",
        "description": "Every MC_effect kind uses Min_bound.",
        "default_u": 0.0,
        "u_by_kind": {},
    },
    {
        "sample_id": 2,
        "scenario_id": "MC_FULL_00002",
        "extreme_name": "E02_all_max",
        "description": "Every MC_effect kind uses Max_bound.",
        "default_u": 1.0,
        "u_by_kind": {},
    },
    {
        "sample_id": 3,
        "scenario_id": "MC_FULL_00003",
        "extreme_name": "E03_supply_stress",
        "description": "Low yield plus high feed, losses, ruminant share, and high emissions/input pressure.",
        "default_u": 1.0,
        "u_by_kind": {
            "yield_rate": 0.0,
            "feed_intensity": 1.0,
            "losses_ratio": 1.0,
            "ruminant_reduction": 1.0,
            "emission_factor": 1.0,
            "fertilizer_rate": 1.0,
            "manure_management_ratio": 1.0,
            "crop_soil_management_ratio": 1.0,
            "land_carbon_price": 0.0,
        },
    },
    {
        "sample_id": 4,
        "scenario_id": "MC_FULL_00004",
        "extreme_name": "E04_transition_relief",
        "description": "High yield with low feed, losses, ruminant share, emissions, fertilizer, manure, and soil pressure.",
        "default_u": 0.0,
        "u_by_kind": {
            "yield_rate": 1.0,
            "feed_intensity": 0.0,
            "losses_ratio": 0.0,
            "ruminant_reduction": 0.0,
            "emission_factor": 0.0,
            "fertilizer_rate": 0.0,
            "manure_management_ratio": 0.0,
            "crop_soil_management_ratio": 0.0,
            "land_carbon_price": 1.0,
        },
    },
    {
        "sample_id": 5,
        "scenario_id": "MC_FULL_00005",
        "extreme_name": "E05_high_emis_feasible_supply",
        "description": "Supply-side favorable bounds with adverse emissions and input-management bounds.",
        "default_u": 1.0,
        "u_by_kind": {
            "yield_rate": 1.0,
            "feed_intensity": 0.0,
            "losses_ratio": 0.0,
            "ruminant_reduction": 1.0,
            "emission_factor": 1.0,
            "fertilizer_rate": 1.0,
            "manure_management_ratio": 1.0,
            "crop_soil_management_ratio": 1.0,
            "land_carbon_price": 0.0,
        },
    },
]


def _scenario_u(scenario: Mapping[str, object], kind: str) -> float:
    u_by_kind = scenario.get("u_by_kind") or {}
    if isinstance(u_by_kind, Mapping) and kind in u_by_kind:
        return float(u_by_kind[kind])
    return float(scenario.get("default_u", 0.5))


def build_extreme_unit_matrix(specs_df: pd.DataFrame) -> np.ndarray:
    kinds = specs_df["__kind"].astype(str).tolist() if "__kind" in specs_df.columns else [
        str(v) for v in specs_df.get("Element", [])
    ]
    matrix = np.zeros((len(EXTREME_SCENARIOS), len(kinds)), dtype=float)
    for row_idx, scenario in enumerate(EXTREME_SCENARIOS):
        for col_idx, kind in enumerate(kinds):
            matrix[row_idx, col_idx] = max(0.0, min(1.0, _scenario_u(scenario, kind)))
    return matrix


def _load_normalized_specs(cfg: Mapping[str, object]) -> pd.DataFrame:
    paths = fullmc.DataPaths()
    mc_sheet = fullmc.resolve_mc_effect_sheet(
        cfg.get("mc_sheet_prefer", "MC_effect_low_land_new"),
        nutrition_profile_sheet=cfg.get("nutrition_profile_sheet", "low_land_new"),
    )
    specs_df = fullmc._load_mc_specs_effect(
        paths.scenario_config_xlsx,
        prefer_sheet=mc_sheet,
        nutrition_profile_sheet=cfg.get("nutrition_profile_sheet", "low_land_new"),
    )
    specs_df = fullmc._normalize_mc_specs(
        specs_df,
        aggregate_non_ef=bool(cfg.get("aggregate_non_ef", False)),
    ).reset_index(drop=True)
    specs_df["spec_row_id"] = np.arange(1, len(specs_df) + 1)
    return specs_df


def _design_frame() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for scenario in EXTREME_SCENARIOS:
        row = {
            "sample_id": scenario["sample_id"],
            "scenario_id": scenario["scenario_id"],
            "extreme_name": scenario["extreme_name"],
            "description": scenario["description"],
            "default_u": scenario["default_u"],
        }
        for kind in EXTREME_KINDS:
            row[f"u_{kind}"] = _scenario_u(scenario, kind)
        rows.append(row)
    return pd.DataFrame(rows)


def _preview_draws_frame(
    specs_df: pd.DataFrame,
    unit_matrix: np.ndarray,
    sampling_cfg: Mapping[str, object],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    q_bounds = tuple(sampling_cfg.get("quantile_bounds", (0.0, 1.0)))
    for sample_idx, scenario in enumerate(EXTREME_SCENARIOS):
        param_rows = fullmc._draw_mc_param_rows(
            specs_df,
            unit_row=unit_matrix[sample_idx],
            quantile_bounds=(float(q_bounds[0]), float(q_bounds[1])),
            sampling_cfg=dict(sampling_cfg),
        )
        for spec_idx, param in enumerate(param_rows):
            spec = specs_df.iloc[spec_idx]
            rows.append(
                {
                    "sample_id": scenario["sample_id"],
                    "scenario_id": scenario["scenario_id"],
                    "extreme_name": scenario["extreme_name"],
                    "spec_row_id": int(spec.get("spec_row_id", spec_idx + 1)),
                    "element_name": str(spec.get("Element", "")),
                    "kind": str(param.get("kind", "")),
                    "element_unit": str(param.get("element_unit", "")),
                    "process_selector": str(param.get("process", "All")),
                    "item_selector": str(param.get("item", "All")),
                    "ghg_selector": str(param.get("ghg", "All")),
                    "region_selector": str(param.get("region", "All")),
                    "min_bound": param.get("min_bound"),
                    "max_bound": param.get("max_bound"),
                    "mc_u": param.get("mc_u"),
                    "value_draw": param.get("abs_value"),
                    "continuous_draw": param.get("continuous_draw"),
                    "discrete_level_applied": param.get("discrete_level_applied"),
                    "q_low": param.get("q_low"),
                    "q_high": param.get("q_high"),
                }
            )
    return pd.DataFrame(rows)


def _configure_fullmc(
    *,
    output_dir: Path,
    max_runs: Optional[int],
    resume: bool,
    enable_iis: bool,
    iis_timeout: int,
    save_sample_workbook: bool,
    market_gap_max_rate: Optional[float],
    threads: Optional[int],
) -> Dict[str, object]:
    cfg = copy.deepcopy(fullmc.CONFIG)
    cfg["samples"] = len(EXTREME_SCENARIOS)
    cfg["output_dir"] = str(output_dir)
    cfg["resume"] = bool(resume)
    cfg["clear_existing_run_dirs_when_no_resume"] = not bool(resume)
    cfg["save_sample_workbook"] = bool(save_sample_workbook)
    cfg["write_every_n_runs"] = 1
    cfg["max_runs"] = max_runs
    cfg["batch"] = {
        "enabled": False,
        "total_batches": 1,
        "batch_index": 1,
        "assignment": "contiguous",
        "batches_subdir": "batches",
    }

    sampling = copy.deepcopy(cfg.get("sampling", {}) or {})
    sampling["method"] = "extreme5"
    sampling["scope"] = "row"
    sampling["shuffle"] = False
    sampling["quantile_bounds"] = (0.0, 1.0)
    cfg["sampling"] = sampling

    override = copy.deepcopy(cfg.get("override_cfg", {}) or {})
    override["batch_mode"] = False
    override["linear_enable_infeasible_iis"] = bool(enable_iis)
    override["linear_enable_output_diagnostics"] = True
    override["linear_enable_violation_iis"] = False
    override["linear_enable_verbose_logging"] = False
    override["iis_timeout"] = int(iis_timeout)
    if market_gap_max_rate is not None:
        override["market_gap_max_rate"] = float(market_gap_max_rate)
    if threads is not None:
        override["linear_solver_threads"] = int(threads)
    cfg["override_cfg"] = override
    return cfg


def _write_design_outputs(output_dir: Path, cfg: Mapping[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    specs_df = _load_normalized_specs(cfg)
    unit_matrix = build_extreme_unit_matrix(specs_df)
    _design_frame().to_csv(output_dir / "extreme_scenario_design.csv", index=False, encoding="utf-8-sig")
    _preview_draws_frame(specs_df, unit_matrix, cfg.get("sampling", {}) or {}).to_csv(
        output_dir / "extreme_draws_preview_long.csv",
        index=False,
        encoding="utf-8-sig",
    )


def _write_joined_status(output_dir: Path) -> None:
    status_path = output_dir / str(fullmc.CONFIG.get("status_csv") or "mc_sample_status.csv")
    if not status_path.exists():
        return
    status = pd.read_csv(status_path)
    design = _design_frame()
    merged = design.merge(status, on=["sample_id", "scenario_id"], how="left")
    merged.to_csv(output_dir / "extreme_scenario_status.csv", index=False, encoding="utf-8-sig")
    keep_cols = [c for c in ["sample_id", "scenario_id", "extreme_name", "run_status", "model_status_code", "afolu_emissions_gt_co2eq_yr", "error_message"] if c in merged.columns]
    if keep_cols:
        print("[S5_4_EXTREME] status summary:")
        print(merged[keep_cols].to_string(index=False))


def _run_with_injected_sampler(cfg: Dict[str, object]) -> None:
    original_config = fullmc.CONFIG
    original_sampler = fullmc._sample_unit_matrix_for_specs

    def _extreme_sampler(specs_df: pd.DataFrame, n_samples: int, *, seed: int, config: Dict[str, object]) -> np.ndarray:
        matrix = build_extreme_unit_matrix(specs_df)
        if int(n_samples) != len(matrix):
            raise ValueError(f"extreme5 expects {len(matrix)} samples, got {n_samples}")
        return matrix

    fullmc.CONFIG = cfg
    fullmc._sample_unit_matrix_for_specs = _extreme_sampler
    try:
        fullmc.main()
    finally:
        fullmc._sample_unit_matrix_for_specs = original_sampler
        fullmc.CONFIG = original_config


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run five endpoint stress scenarios through S5_4.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Default: <output>/MC_Full_Variables_Extreme_Robustness",
    )
    parser.add_argument("--max-runs", type=int, default=None, help="Limit runs for a quick smoke test.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing run directories.")
    parser.add_argument("--dry-run", action="store_true", help="Only write the design/preview CSV files.")
    parser.add_argument("--no-iis", action="store_true", help="Disable IIS diagnostics for infeasible runs.")
    parser.add_argument("--iis-timeout", type=int, default=600, help="IIS timeout in seconds.")
    parser.add_argument("--no-sample-workbook", action="store_true", help="Do not save the five sampled Excel workbooks.")
    parser.add_argument("--market-gap-max-rate", type=float, default=0.10, help="Override S4 market_gap_max_rate.")
    parser.add_argument("--threads", type=int, default=None, help="Override linear_solver_threads.")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir) if args.output_dir else Path(get_results_base()) / "MC_Full_Variables_Extreme_Robustness"
    cfg = _configure_fullmc(
        output_dir=output_dir,
        max_runs=args.max_runs,
        resume=bool(args.resume),
        enable_iis=not bool(args.no_iis),
        iis_timeout=int(args.iis_timeout),
        save_sample_workbook=not bool(args.no_sample_workbook),
        market_gap_max_rate=args.market_gap_max_rate,
        threads=args.threads,
    )

    _write_design_outputs(output_dir, cfg)
    print(f"[S5_4_EXTREME] design -> {output_dir / 'extreme_scenario_design.csv'}")
    print(f"[S5_4_EXTREME] preview -> {output_dir / 'extreme_draws_preview_long.csv'}")
    if args.dry_run:
        print("[S5_4_EXTREME] dry run only; S5_4 was not executed.")
        return

    _run_with_injected_sampler(cfg)
    _write_joined_status(output_dir)


if __name__ == "__main__":
    main()
