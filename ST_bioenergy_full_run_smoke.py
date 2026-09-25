# -*- coding: utf-8 -*-
"""Layered test runner for the Bioenergy integration.

Layers:
1. unit: focused in-memory MVP tests.
2. module_integration: run S4 data/bioenergy handoff without solving Gurobi.
3. solver_micro: tiny Gurobi solve for explicit bioenergy market balance.
4. full_regression: current end-to-end S4 smoke.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

import S4_0_main as main_model
from config_paths import get_results_base
from market_balance_diagnostics import summarize_market_balance_frame
from model_run_status import (
    emission_modules_complete,
    read_run_status,
    solver_status_name,
)
from S1_0_schema import ScenarioConfig
from S2_0_load_data import DataPaths, build_universe_from_dict_v3
from S3_3_bioenergy import build_bioenergy_bundle, write_bioenergy_bundle


DEFAULT_SCENARIOS = ["baseline_off", "medium_bioenergy"]
TEST_LAYERS = {"unit", "module_integration", "solver_micro", "full_regression"}
SCENARIO_FEASIBILITY_CHECKS = frozenset({"bioenergy_postsolve_feasible"})


def _refresh_check_rollup(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Separate pipeline integrity from scenario feasibility."""
    checks = dict(summary.get("checks", {}) or {})
    integration_checks = {
        name: bool(value)
        for name, value in checks.items()
        if name not in SCENARIO_FEASIBILITY_CHECKS
    }
    scenario_checks = {
        name: bool(value)
        for name, value in checks.items()
        if name in SCENARIO_FEASIBILITY_CHECKS
    }
    integration_passed = all(integration_checks.values()) if integration_checks else True
    scenario_assessed = bool(scenario_checks)
    scenario_feasible = all(scenario_checks.values()) if scenario_assessed else None
    summary["checks"] = checks
    summary["integration_checks_passed"] = bool(integration_passed)
    summary["scenario_feasibility_assessed"] = scenario_assessed
    summary["scenario_feasible"] = (
        bool(scenario_feasible) if scenario_assessed else None
    )
    summary["all_checks_passed"] = bool(
        integration_passed
        and (not scenario_assessed or bool(scenario_feasible))
    )
    return summary


def _normalize_bioenergy_scenario_name(name: str) -> str:
    text = str(name or "").strip()
    aliases = {
        "low": "low_bioenergy",
        "medium": "medium_bioenergy",
        "high": "high_bioenergy",
    }
    return aliases.get(text, text)


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _find_first(root: Path, names: Iterable[str]) -> Optional[Path]:
    relative_names = [Path(name) for name in names]
    for name in relative_names:
        candidate = root / name
        if candidate.is_file():
            return candidate
    for name in relative_names:
        hits = list(root.rglob(name.name))
        if hits:
            return sorted(hits, key=lambda path: (len(path.parts), str(path)))[0]
    return None


def _read_log_tail(path: Path, max_chars: int = 1_000_000) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    if len(text) > max_chars:
        return text[-max_chars:]
    return text


def _detect_solver_status_info(outdir: Path) -> tuple[str, str, Optional[Dict[str, Any]]]:
    """Infer the solver status from durable run artifacts.

    Structured status is authoritative. Log inspection remains only for
    backwards compatibility with pre-status-artifact runs.
    """
    structured = read_run_status(outdir)
    if isinstance(structured, dict):
        solver = structured.get("solver")
        if isinstance(solver, dict):
            raw_status = (
                solver.get("status_name")
                if solver.get("status_name") is not None
                else solver.get("status_code", solver.get("status"))
            )
        else:
            raw_status = structured.get("solver_status", structured.get("status"))
        return solver_status_name(raw_status), "run_status.json", structured

    log_dir = outdir / "Log"
    if list(log_dir.glob("*.ilp")) or list(log_dir.glob("*iis*")):
        return "infeasible", "log_fallback", None
    text = "\n".join(
        _read_log_tail(path)
        for path in [log_dir / "gurobi.log", log_dir / "model.log"]
    ).lower()
    if not text.strip():
        return "missing_log", "log_fallback", None
    if "infeasible model" in text or "model is infeasible" in text or "non-optimal status" in text:
        return "infeasible", "log_fallback", None
    if "optimal objective" in text or "optimization complete" in text or " status: 2" in text:
        return "optimal", "log_fallback", None
    return "unknown", "log_fallback", None


def _detect_solver_status(outdir: Path) -> str:
    return _detect_solver_status_info(outdir)[0]


def _summarize_output(
    outdir: Path,
    scenario_label: str,
    *,
    expect_market_outputs: bool,
    expect_solver_status_artifact: bool = True,
    expect_postsolve_assessment: bool = False,
    expect_fast_emissions: bool = False,
) -> Dict[str, Any]:
    market_path = _find_first(
        outdir,
        [
            "Diagnostics/commodity_balance_by_commodity.csv",
            "DS/market_summary.csv",
            "market_summary.csv",
            "market.csv",
        ],
    )
    bioenergy_detail_path = outdir / "bioenergy_feedstock_use.csv"
    bioenergy_crop_path = outdir / "bioenergy_crop_demand.csv"
    land_path = outdir / "bioenergy_land_handoff.csv"
    diag_path = outdir / "bioenergy_diagnostics.csv"
    feasibility_path = outdir / "bioenergy_target_feasibility.csv"
    postsolve_path = outdir / "bioenergy_postsolve_assessment.csv"
    fast_summary_path = outdir / "Emis" / "emissions_fast_summary.csv"
    fast_detail_path = outdir / "Emis" / "emissions_fast_global_detail.csv"

    market = _read_csv_if_exists(market_path) if market_path else pd.DataFrame()
    detail = _read_csv_if_exists(bioenergy_detail_path)
    crop = _read_csv_if_exists(bioenergy_crop_path)
    land = _read_csv_if_exists(land_path)
    diag = _read_csv_if_exists(diag_path)
    feasibility = _read_csv_if_exists(feasibility_path)
    postsolve = _read_csv_if_exists(postsolve_path)
    fast_summary = _read_csv_if_exists(fast_summary_path)
    fast_detail = _read_csv_if_exists(fast_detail_path)
    solver_status, status_source, structured_status = _detect_solver_status_info(outdir)
    modules_complete = emission_modules_complete(structured_status)
    market_validation_error = ""
    if expect_market_outputs:
        _, market_error = summarize_market_balance_frame(market)
        market_validation_error = str(market_error or "")
    postsolve_status = ""
    if not postsolve.empty and "assessment_status" in postsolve.columns:
        values = postsolve["assessment_status"].dropna().astype(str).str.strip().str.lower()
        if not values.empty:
            postsolve_status = str(values.iloc[0])

    checks: Dict[str, bool] = {
        "outdir_exists": outdir.exists(),
        "solver_status_optimal": solver_status == "optimal",
    }
    if expect_solver_status_artifact:
        checks["structured_run_status_present"] = status_source == "run_status.json"
        checks["pipeline_status_completed"] = bool(
            isinstance(structured_status, dict)
            and str(structured_status.get("pipeline_status", "")).strip().lower()
            == "completed"
        )
        checks["emission_module_completeness_reported"] = modules_complete is not None
        if modules_complete is not None:
            checks["emission_modules_complete"] = bool(modules_complete)
    if expect_market_outputs:
        checks["market_rows_present"] = len(market) > 0
        checks["market_solver_diagnostic_valid"] = not bool(
            market_validation_error
        )
    if expect_fast_emissions:
        fast_summary_values = pd.to_numeric(
            fast_summary.get(
                "total_co2eq_kt",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        )
        fast_detail_values = pd.to_numeric(
            fast_detail.get(
                "co2eq_kt",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        )
        checks["fast_emissions_summary_written"] = fast_summary_path.exists()
        checks["fast_emissions_detail_written"] = fast_detail_path.exists()
        checks["fast_emissions_summary_nonempty_finite"] = bool(
            not fast_summary.empty
            and "total_co2eq_kt" in fast_summary.columns
            and np.isfinite(fast_summary_values.to_numpy(dtype=float)).all()
        )
        checks["fast_emissions_detail_nonempty_finite"] = bool(
            not fast_detail.empty
            and "co2eq_kt" in fast_detail.columns
            and np.isfinite(fast_detail_values.to_numpy(dtype=float)).all()
        )
    if scenario_label == "baseline_off":
        checks["baseline_has_no_bioenergy_files"] = (
            not bioenergy_detail_path.exists()
            and not bioenergy_crop_path.exists()
            and not land_path.exists()
        )
        checks["baseline_has_no_bioenergy_detail"] = detail.empty
    else:
        checks["bioenergy_detail_written"] = bioenergy_detail_path.exists()
        checks["bioenergy_crop_demand_written"] = bioenergy_crop_path.exists()
        checks["bioenergy_land_handoff_written"] = land_path.exists()
        checks["bioenergy_target_feasibility_written"] = feasibility_path.exists()
        checks["enabled_has_bioenergy_detail_rows"] = len(detail) > 0
        checks["enabled_has_crop_demand_rows"] = len(crop) > 0
        checks["enabled_has_land_handoff_rows"] = len(land) > 0
        if expect_postsolve_assessment:
            checks["bioenergy_postsolve_assessment_written"] = postsolve_path.exists()
            checks["bioenergy_postsolve_assessment_rows"] = len(postsolve) > 0
            checks["bioenergy_postsolve_status_recognized"] = postsolve_status in {
                "valid",
                "stress_only",
                "failed",
                "not_applicable",
            }
            checks["bioenergy_postsolve_assessment_succeeded"] = postsolve_status in {
                "valid",
                "stress_only",
                "not_applicable",
            }
            checks["bioenergy_postsolve_feasible"] = postsolve_status == "valid"

    return _refresh_check_rollup({
        "scenario_label": scenario_label,
        "outdir": str(outdir),
        "market_path": str(market_path) if market_path else "",
        "market_rows": int(len(market)),
        "market_validation_error": market_validation_error,
        "expect_market_outputs": bool(expect_market_outputs),
        "bioenergy_detail_rows": int(len(detail)),
        "bioenergy_crop_rows": int(len(crop)),
        "bioenergy_land_rows": int(len(land)),
        "bioenergy_target_feasibility_rows": int(len(feasibility)),
        "bioenergy_postsolve_assessment_rows": int(len(postsolve)),
        "fast_emissions_summary_rows": int(len(fast_summary)),
        "fast_emissions_detail_rows": int(len(fast_detail)),
        "bioenergy_energy_tj": float(pd.to_numeric(detail.get("energy_target_tj"), errors="coerce").fillna(0.0).sum()) if not detail.empty else 0.0,
        "bioenergy_target_ej": float(pd.to_numeric(feasibility.get("target_energy_ej"), errors="coerce").fillna(0.0).sum()) if not feasibility.empty else 0.0,
        "bioenergy_feasible_ej": float(pd.to_numeric(feasibility.get("feasible_supplied_ej"), errors="coerce").fillna(0.0).sum()) if not feasibility.empty else 0.0,
        "bioenergy_unmet_ej": float(pd.to_numeric(feasibility.get("unmet_energy_ej"), errors="coerce").fillna(0.0).sum()) if not feasibility.empty else 0.0,
        "bioenergy_non_crop_cap_unmet_ej": float(pd.to_numeric(feasibility.get("non_crop_cap_unmet_ej"), errors="coerce").fillna(0.0).sum()) if not feasibility.empty else 0.0,
        "bioenergy_crop_residue_cap_unmet_ej": float(pd.to_numeric(feasibility.get("crop_residue_cap_unmet_ej"), errors="coerce").fillna(0.0).sum()) if not feasibility.empty else 0.0,
        "bioenergy_eligible_land_cap_unmet_ej": float(pd.to_numeric(feasibility.get("eligible_land_cap_unmet_ej"), errors="coerce").fillna(0.0).sum()) if not feasibility.empty else 0.0,
        "bioenergy_unattributed_pre_solver_unmet_ej": float(pd.to_numeric(feasibility.get("unattributed_pre_solver_unmet_ej"), errors="coerce").fillna(0.0).sum()) if not feasibility.empty and "unattributed_pre_solver_unmet_ej" in feasibility.columns else 0.0,
        "bioenergy_market_land_solver_unmet_ej": float(pd.to_numeric(feasibility.get("market_land_solver_unmet_ej"), errors="coerce").fillna(0.0).sum()) if not feasibility.empty else 0.0,
        "bioenergy_crop_demand_t": float(pd.to_numeric(crop.get("bioenergy_demand_t"), errors="coerce").fillna(0.0).sum()) if not crop.empty else 0.0,
        "energy_crop_land_requirement_ha": float(pd.to_numeric(land.get("land_requirement_ha"), errors="coerce").fillna(0.0).sum()) if not land.empty else 0.0,
        "solver_status": solver_status,
        "solver_status_source": status_source,
        "pipeline_status": (
            str(structured_status.get("pipeline_status", ""))
            if isinstance(structured_status, dict)
            else ""
        ),
        "emission_modules_complete": modules_complete,
        "bioenergy_postsolve_status": postsolve_status,
        "diagnostics": diag.to_dict(orient="records") if not diag.empty else [],
        "checks": checks,
    })


def _run_case(
    *,
    scenario_label: str,
    bioenergy_enabled: bool,
    bioenergy_scenario: str,
    output_root: Path,
    fast_emis_only: bool,
    use_fao_modules: bool,
    qty_scale: Optional[float],
    land_scale: Optional[float],
    solve_model: bool,
    expect_solver: bool,
) -> Dict[str, Any]:
    paths = DataPaths()
    original_cfg = copy.deepcopy(main_model.CFG)
    scenario_id = scenario_label
    try:
        main_model.CFG.update({
            "bioenergy_enabled": bool(bioenergy_enabled),
            "bioenergy_scenario": bioenergy_scenario,
            "bioenergy_require_historical_feedstock_bridge": True,
            "demand_method": "nutrition",
            "feed_crop_link_mode": "dynamic_constraints",
            "use_linear_model": True,
            "future_last_only": True,
            "solve": bool(solve_model),
            "use_fao_modules": bool(use_fao_modules),
            "run_mode": "single",
            "clean_output_dir": True,
            "iis_timeout": 0,
        })
        if qty_scale is not None:
            main_model.CFG["qty_scale"] = float(qty_scale)
        if land_scale is not None:
            main_model.CFG["land_scale"] = float(land_scale)
        outdir = Path(main_model.run_one_pipeline(
            paths,
            pre_macc_e0=bool(main_model.CFG.get("premacc_e0", False)),
            scenario_id=scenario_id,
            scenario_params={"land_carbon_price_by_year": {y: 0.0 for y in range(2010, 2081, 10)}},
            scenario_effects=None,
            solve=bool(solve_model),
            use_fao_modules=bool(use_fao_modules),
            save_root=str(output_root),
            future_last_only=True,
            use_linear=True,
            fast_emis_only=bool(fast_emis_only),
        ))
        summary = _summarize_output(
            outdir,
            scenario_label,
            expect_market_outputs=bool(expect_solver),
            expect_solver_status_artifact=bool(expect_solver),
            expect_postsolve_assessment=bool(bioenergy_enabled and expect_solver),
            expect_fast_emissions=bool(fast_emis_only and expect_solver),
        )
        if not expect_solver:
            checks = dict(summary.get("checks", {}) or {})
            checks.pop("solver_status_optimal", None)
            summary["checks"] = checks
            summary["solver_status"] = "not_run"
            _refresh_check_rollup(summary)
        summary["run_status"] = "completed"
        summary["qty_scale"] = float(qty_scale) if qty_scale is not None else ""
        summary["land_scale"] = float(land_scale) if land_scale is not None else ""
        summary["test_layer"] = "full_regression" if solve_model else "module_integration"
        return summary
    except Exception as exc:
        return {
            "scenario_label": scenario_label,
            "run_status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "outdir": str(output_root / scenario_id),
            "checks": {"run_completed": False},
            "integration_checks_passed": False,
            "scenario_feasible": False,
            "all_checks_passed": False,
            "test_layer": "full_regression" if solve_model else "module_integration",
        }
    finally:
        main_model.CFG.clear()
        main_model.CFG.update(original_cfg)


def _run_unit_layer(output_root: Path) -> Dict[str, Any]:
    import ST_bioenergy_mvp_test as mvp

    try:
        mvp.test_bundle_and_residual_reconciliation()
        mvp.test_scenario_profile_effect()
        mvp.test_carrier_feedstock_bridge_fallback()
        mvp.test_market_summary_separates_food_and_bioenergy()
        mvp.test_solver_source_contains_explicit_bioenergy_balance()
        mvp.test_resource_constraints_and_handoffs()
        mvp.test_solver_micro_bioenergy_balance_solves()
        mvp.test_enabled_bioenergy_requires_matching_scenario_rows()
        mvp.test_pre_solver_feasibility_keeps_unattributed_gap_out_of_solver_gap()
        mvp.test_postsolve_assessment_distinguishes_valid_stress_and_failed()
        return {
            "test_layer": "unit",
            "scenario_label": "unit",
            "outdir": str(output_root),
            "run_status": "completed",
            "checks": {"unit_tests_passed": True},
            "integration_checks_passed": True,
            "scenario_feasible": True,
            "all_checks_passed": True,
        }
    except Exception as exc:
        return {
            "test_layer": "unit",
            "scenario_label": "unit",
            "outdir": str(output_root),
            "run_status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "checks": {"unit_tests_passed": False},
            "integration_checks_passed": False,
            "scenario_feasible": False,
            "all_checks_passed": False,
        }


def _run_solver_micro_layer(output_root: Path) -> Dict[str, Any]:
    import ST_bioenergy_mvp_test as mvp

    try:
        mvp.test_solver_micro_bioenergy_balance_solves()
        return {
            "test_layer": "solver_micro",
            "scenario_label": "solver_micro",
            "outdir": str(output_root),
            "run_status": "completed",
            "checks": {"solver_micro_passed": True},
            "integration_checks_passed": True,
            "scenario_feasible": True,
            "all_checks_passed": True,
        }
    except Exception as exc:
        return {
            "test_layer": "solver_micro",
            "scenario_label": "solver_micro",
            "outdir": str(output_root),
            "run_status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "checks": {"solver_micro_passed": False},
            "integration_checks_passed": False,
            "scenario_feasible": False,
            "all_checks_passed": False,
        }


def _run_module_integration_case(
    *,
    scenario_label: str,
    bioenergy_enabled: bool,
    bioenergy_scenario: str,
    output_root: Path,
) -> Dict[str, Any]:
    """Run the Bioenergy module against real input files without the full S4 pipeline."""
    scenario_id = _normalize_bioenergy_scenario_name(scenario_label)
    outdir = output_root / scenario_id
    try:
        if outdir.exists():
            try:
                shutil.rmtree(outdir)
            except PermissionError:
                if not bioenergy_enabled:
                    outdir = output_root / f"{scenario_id}_fresh"
                    if outdir.exists():
                        shutil.rmtree(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        if not bioenergy_enabled:
            summary = _summarize_output(
                outdir,
                scenario_label,
                expect_market_outputs=False,
                expect_solver_status_artifact=False,
            )
            checks = dict(summary.get("checks", {}) or {})
            checks.pop("solver_status_optimal", None)
            summary["checks"] = checks
            summary["solver_status"] = "not_run"
            _refresh_check_rollup(summary)
            summary["run_status"] = "completed"
            summary["test_layer"] = "module_integration"
            return summary

        paths = DataPaths()
        config = ScenarioConfig(years_future=[2030, 2050, 2080])
        universe = build_universe_from_dict_v3(paths.dict_v3_path, config)
        scenario_name = _normalize_bioenergy_scenario_name(bioenergy_scenario)
        bundle = build_bioenergy_bundle(
            universe=universe,
            active_years=universe.years,
            hist_end_year=config.years_hist_end,
            scenario=scenario_name,
            historical_feedstock_path=(
                main_model.CFG.get("bioenergy_historical_feedstock_path")
                or getattr(paths, "bioenergy_historical_feedstock_csv", None)
            ),
            scenario_path=(
                main_model.CFG.get("bioenergy_scenario_path")
                or getattr(paths, "bioenergy_scenario_csv", None)
            ),
            parameter_path=(
                main_model.CFG.get("bioenergy_feedstock_parameters_path")
                or getattr(paths, "bioenergy_feedstock_parameters_csv", None)
            ),
            resource_path=(
                main_model.CFG.get("bioenergy_resource_constraints_path")
                or getattr(paths, "bioenergy_resource_constraints_csv", None)
            ),
            require_historical_bridge=bool(
                main_model.CFG.get("bioenergy_require_historical_feedstock_bridge", True)
            ),
            require_scenario_rows=True,
        )
        write_bioenergy_bundle(bundle, str(outdir))
        summary = _summarize_output(
            outdir,
            scenario_id,
            expect_market_outputs=False,
            expect_solver_status_artifact=False,
        )
        checks = dict(summary.get("checks", {}) or {})
        checks.pop("solver_status_optimal", None)
        summary["checks"] = checks
        summary["solver_status"] = "not_run"
        _refresh_check_rollup(summary)
        summary["run_status"] = "completed"
        summary["test_layer"] = "module_integration"
        return summary
    except Exception as exc:
        return {
            "test_layer": "module_integration",
            "scenario_label": scenario_label,
            "outdir": str(outdir),
            "run_status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "checks": {"module_integration_passed": False},
            "integration_checks_passed": False,
            "scenario_feasible": False,
            "all_checks_passed": False,
        }


def _run_module_integration_layer(args: argparse.Namespace, *, output_root: Path) -> List[Dict[str, Any]]:
    output_root.mkdir(parents=True, exist_ok=True)
    requested = [s.strip() for s in str(args.scenarios).split(",") if s.strip()]
    rows: List[Dict[str, Any]] = []
    for name in requested:
        scenario_name = _normalize_bioenergy_scenario_name(name)
        if scenario_name == "baseline_off":
            rows.append(_run_module_integration_case(
                scenario_label=name,
                bioenergy_enabled=False,
                bioenergy_scenario="current_policy",
                output_root=output_root,
            ))
        elif scenario_name in {"low_bioenergy", "medium_bioenergy", "high_bioenergy"}:
            rows.append(_run_module_integration_case(
                scenario_label=scenario_name,
                bioenergy_enabled=True,
                bioenergy_scenario=scenario_name,
                output_root=output_root,
            ))
        else:
            raise ValueError(f"Unsupported module integration scenario: {name}")
    return rows


def _run_pipeline_layer(args: argparse.Namespace, *, output_root: Path, solve_model: bool, layer: str) -> List[Dict[str, Any]]:
    output_root.mkdir(parents=True, exist_ok=True)
    requested = [s.strip() for s in str(args.scenarios).split(",") if s.strip()]
    rows: List[Dict[str, Any]] = []
    for name in requested:
        scenario_name = _normalize_bioenergy_scenario_name(name)
        if scenario_name == "baseline_off":
            rows.append(_run_case(
                scenario_label=name,
                bioenergy_enabled=False,
                bioenergy_scenario="current_policy",
                output_root=output_root,
                fast_emis_only=bool(args.fast_emis_only),
                use_fao_modules=bool(args.use_fao_modules),
                qty_scale=args.qty_scale,
                land_scale=args.land_scale,
                solve_model=solve_model,
                expect_solver=solve_model,
            ))
        elif scenario_name in {"low_bioenergy", "medium_bioenergy", "high_bioenergy"}:
            rows.append(_run_case(
                scenario_label=scenario_name,
                bioenergy_enabled=True,
                bioenergy_scenario=scenario_name,
                output_root=output_root,
                fast_emis_only=bool(args.fast_emis_only),
                use_fao_modules=bool(args.use_fao_modules),
                qty_scale=args.qty_scale,
                land_scale=args.land_scale,
                solve_model=solve_model,
                expect_solver=solve_model,
            ))
        else:
            raise ValueError(f"Unsupported smoke scenario: {name}")
        rows[-1]["test_layer"] = layer
    return rows


def run_smoke(args: argparse.Namespace) -> pd.DataFrame:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    layer_arg = str(args.layer or main_model.CFG.get("bioenergy_test_layer", "full_regression")).strip().lower()
    if layer_arg in {"full", "full-run", "full_smoke"}:
        layer_arg = "full_regression"
    if layer_arg in {"module", "integration"}:
        layer_arg = "module_integration"
    if layer_arg not in TEST_LAYERS and layer_arg != "all":
        raise ValueError(f"Unsupported bioenergy test layer: {layer_arg}")
    layers = ["unit", "module_integration", "solver_micro", "full_regression"] if layer_arg == "all" else [layer_arg]

    rows: List[Dict[str, Any]] = []
    for layer in layers:
        layer_root = output_root / layer if layer_arg == "all" else output_root
        if layer == "unit":
            rows.append(_run_unit_layer(layer_root))
        elif layer == "solver_micro":
            rows.append(_run_solver_micro_layer(layer_root))
        elif layer == "module_integration":
            rows.extend(_run_module_integration_layer(args, output_root=layer_root))
        elif layer == "full_regression":
            rows.extend(_run_pipeline_layer(args, output_root=layer_root, solve_model=True, layer=layer))
    summary = pd.DataFrame(rows)
    summary_path = output_root / "bioenergy_full_run_smoke_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    json_path = output_root / "bioenergy_full_run_smoke_summary.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(Path(get_results_base()) / "Bioenergy_Full_Run_Smoke"))
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument(
        "--layer",
        default=str(main_model.CFG.get("bioenergy_test_layer", "full_regression")),
        help="Bioenergy test layer: unit, module_integration, solver_micro, full_regression, or all.",
    )
    parser.add_argument("--fast-emis-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-fao-modules", action="store_true", default=False)
    parser.add_argument(
        "--qty-scale",
        type=float,
        default=None,
        help="Optional linear solver quantity scale, e.g. 1e6 to solve Q variables in Mt instead of t.",
    )
    parser.add_argument(
        "--land-scale",
        type=float,
        default=None,
        help="Optional linear solver land scale, e.g. 1e6 to solve land/LUC variables in Mha instead of ha.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    summary = run_smoke(parse_args(argv))
    if not bool(summary["all_checks_passed"].all()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
