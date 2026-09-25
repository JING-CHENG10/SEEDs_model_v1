# -*- coding: utf-8 -*-
"""Run global marginal abatement cost curves under bioenergy scenarios.

The runner supports three execution modes:

``preflight``
    Rebuild and audit the current bioenergy input tables, run module-level
    bioenergy integration checks, and dry-run the three S5.7 designs. No
    optimization model is solved.

``quick``
    Run one matched BASE, one nine-measure package, and nine strict singleton
    measures for each bioenergy panel. The package contribution method is the
    positive-singleton-scaled fallback (11 evaluations per panel).

``exact``
    Run the full nine-measure Shapley design (512 evaluations per panel).

Persistent runtime output is rejected unless it is a child of ``Code/output``.
The source tree is used only for code and is never a result destination.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_OUTPUT_ROOT = (CODE_ROOT / "output").resolve()
DEFAULT_OUTPUT_ROOT = ALLOWED_OUTPUT_ROOT / "Bioenergy_Scenario_MACC"
PLOT_SCRIPT = Path(__file__).resolve().with_name("SP_M3a_Figure_macc_stock_v3.1.py")

COST_DATABASE_VERSION = "v2.0-2026-09-04"
COST_DATABASE_SHA256 = "7ad8e7545cfeeb43ed5255c3410dab6ef116190077c44e6a3350eb536f7b0102"

EXPECTED_DATABASE_STRATEGIES = frozenset(
    {
        "RuminantReduction",
        "LossWaste",
        "YieldRate",
        "FeedEfficiency",
        "EntericF",
        "Manure",
        "Residue",
        "Rice",
        "Fertilizer",
    }
)
EXPECTED_STRATEGY_KINDS = frozenset(
    {
        "ruminant_reduction",
        "losses_ratio",
        "yield_rate",
        "feed_intensity",
        "enteric_fermentation_management",
        "manure_management",
        "crop_residue_soil_management",
        "rice_cultivation",
        "fertilizer_efficiency",
    }
)


@dataclass(frozen=True)
class PanelCase:
    scenario: str
    sheet: str
    title: str
    token: str


PANEL_CASES: Tuple[PanelCase, ...] = (
    PanelCase("low_bioenergy", "low_Bio", "a. Low bioenergy", "LOW"),
    PanelCase("medium_bioenergy", "medium_Bio", "b. Medium bioenergy", "MEDIUM"),
    PanelCase("high_bioenergy", "high_Bio", "c. High bioenergy", "HIGH"),
)

CASE_ALIASES: Mapping[str, str] = {
    "low": "low_bioenergy",
    "low_bio": "low_bioenergy",
    "low_bioenergy": "low_bioenergy",
    "medium": "medium_bioenergy",
    "medium_bio": "medium_bioenergy",
    "medium_bioenergy": "medium_bioenergy",
    "high": "high_bioenergy",
    "high_bio": "high_bioenergy",
    "high_bioenergy": "high_bioenergy",
}

ValidationRow = Dict[str, object]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_output_root(raw: object) -> Path:
    """Return a safe experiment directory strictly below ``Code/output``."""

    output_root = Path(str(raw or DEFAULT_OUTPUT_ROOT)).expanduser().resolve()
    if output_root == ALLOWED_OUTPUT_ROOT or not _is_within(output_root, ALLOWED_OUTPUT_ROOT):
        raise ValueError(
            "Figure 3 full-chain output must be a child directory of "
            f"{ALLOWED_OUTPUT_ROOT}; got {output_root}"
        )
    return output_root


def parse_cases(raw: object) -> Tuple[PanelCase, ...]:
    requested = [part.strip().lower() for part in str(raw or "").split(",") if part.strip()]
    if not requested or requested == ["all"]:
        return PANEL_CASES
    canonical: List[str] = []
    for name in requested:
        if name not in CASE_ALIASES:
            raise ValueError(f"Unsupported bioenergy case: {name}")
        value = CASE_ALIASES[name]
        if value not in canonical:
            canonical.append(value)
    by_name = {case.scenario: case for case in PANEL_CASES}
    return tuple(by_name[name] for name in canonical)


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _add_check(
    rows: List[ValidationRow],
    *,
    component: str,
    check: str,
    passed: Optional[bool],
    case: str = "global",
    observed: object = "",
    expected: object = "",
    message: str = "",
) -> None:
    rows.append(
        {
            "component": component,
            "case": case,
            "check": check,
            "status": "SKIP" if passed is None else ("PASS" if passed else "FAIL"),
            "observed": observed,
            "expected": expected,
            "message": message,
        }
    )


def _has_failures(rows: Sequence[ValidationRow]) -> bool:
    return any(str(row.get("status")) == "FAIL" for row in rows)


def _write_validation(
    rows: Sequence[ValidationRow],
    *,
    output_root: Path,
    mode: str,
    started_at: str,
    error: str = "",
) -> Tuple[Path, Path]:
    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    csv_path = analysis_dir / f"full_chain_validation_{mode}.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    status_counts = frame.get("status", pd.Series(dtype=str)).value_counts().to_dict()
    payload = {
        "mode": mode,
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "output_root": str(output_root),
        "overall_passed": not _has_failures(rows) and not error,
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "error": error,
        "validation_csv": str(csv_path),
    }
    json_path = analysis_dir / f"full_chain_status_{mode}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path


def _validate_input_audit(
    audit_dir: Path,
    cases: Sequence[PanelCase],
    rows: List[ValidationRow],
) -> None:
    path = audit_dir / "bioenergy_scenario_allocation_audit.csv"
    _add_check(
        rows,
        component="bioenergy_input",
        check="allocation_audit_exists",
        passed=path.exists(),
        observed=str(path),
        expected="existing CSV",
    )
    if not path.exists():
        return
    frame = pd.read_csv(path)
    selected = frame[frame.get("scenario", "").astype(str).isin({case.scenario for case in cases})]
    failed = selected[~selected.get("status", "").astype(str).str.lower().eq("pass")]
    _add_check(
        rows,
        component="bioenergy_input",
        check="selected_allocation_checks_pass",
        passed=not selected.empty and failed.empty,
        observed=f"rows={len(selected)} failures={len(failed)}",
        expected="all selected rows pass",
    )

    global_path = audit_dir / "bioenergy_scenario_global_targets.csv"
    _add_check(
        rows,
        component="bioenergy_input",
        check="global_targets_exist",
        passed=global_path.exists(),
        observed=str(global_path),
        expected="existing CSV",
    )
    if not global_path.exists():
        return
    targets = pd.read_csv(global_path)
    targets = targets[targets.get("scenario", "").astype(str).isin({case.scenario for case in cases})]
    pairs = set(zip(targets.get("scenario", []), pd.to_numeric(targets.get("year"), errors="coerce")))
    expected_pairs = {(case.scenario, year) for case in cases for year in (2030, 2050, 2080)}
    _add_check(
        rows,
        component="bioenergy_input",
        check="scenario_year_coverage",
        passed=pairs == expected_pairs,
        observed=len(pairs),
        expected=len(expected_pairs),
        message=f"missing={sorted(expected_pairs - pairs)} extra={sorted(pairs - expected_pairs)}",
    )


def _run_input_audit(
    output_root: Path,
    cases: Sequence[PanelCase],
    rows: List[ValidationRow],
) -> None:
    import S0_53_prepare_bioenergy_scenarios as prepare

    audit_dir = output_root / "00_bioenergy_input_audit"
    prepare.main(["--out-dir", str(audit_dir)])
    _validate_input_audit(audit_dir, cases, rows)


def _run_bioenergy_smoke(
    output_root: Path,
    cases: Sequence[PanelCase],
    rows: List[ValidationRow],
    *,
    layer: str,
) -> None:
    import ST_bioenergy_full_run_smoke as smoke

    subdir = "01_bioenergy_module_smoke" if layer == "module_integration" else "01_bioenergy_solver_smoke"
    args = smoke.parse_args(
        [
            "--output-root",
            str(output_root / subdir),
            "--scenarios",
            ",".join(case.scenario for case in cases),
            "--layer",
            layer,
            "--fast-emis-only",
        ]
    )
    summary = smoke.run_smoke(args)
    for case in cases:
        selected = summary[summary.get("scenario_label", "").astype(str).eq(case.scenario)]
        passed = len(selected) == 1 and _as_bool(selected.iloc[0].get("all_checks_passed"))
        _add_check(
            rows,
            component=f"bioenergy_{layer}",
            case=case.scenario,
            check="all_checks_passed",
            passed=passed,
            observed=(selected.iloc[0].get("run_status", "") if len(selected) else "missing"),
            expected="all_checks_passed=True",
        )


def _panel_stage(output_root: Path, mode: str) -> Path:
    return output_root / f"02_s5_7_{mode}"


def _configure_panel(
    base_config: Mapping[str, object],
    case: PanelCase,
) -> Dict[str, object]:
    config = copy.deepcopy(dict(base_config))
    config["scenario_prefix"] = f"FIG3_{case.token}"
    config["baseline_scenario_id"] = f"FIG3_{case.token}_BASE"
    config["global_all_scenario_id"] = f"FIG3_{case.token}_GLOBAL_ALL"
    override = copy.deepcopy(config.get("override_cfg", {}) or {})
    override.update(
        {
            "bioenergy_enabled": True,
            "bioenergy_scenario": case.scenario,
        }
    )
    config["override_cfg"] = override
    cost_curve = copy.deepcopy(config.get("cost_curve", {}) or {})
    cost_curve.update(
        {
            "bioenergy_case": case.sheet,
            "year": 2080,
            "write_excel": True,
            "excel_filename": "macc_strategy_curve_sp_m3a.xlsx",
        }
    )
    config["cost_curve"] = cost_curve
    return config


def _run_s57_panels(
    output_root: Path,
    cases: Sequence[PanelCase],
    *,
    mode: str,
    resume: bool,
    stop_on_error: bool,
    threads: int,
) -> None:
    import S5_7_1_strategy_endpoint_rerun_max_reduction_potential as s57

    original = copy.deepcopy(s57.CONFIG)
    try:
        for case in cases:
            configured = _configure_panel(original, case)
            s57.CONFIG.clear()
            s57.CONFIG.update(configured)
            argv = [
                "--out-dir",
                str(_panel_stage(output_root, mode) / case.scenario),
                "--threads",
                str(threads),
                "--shapley" if mode == "exact" else "--no-shapley",
            ]
            if mode == "preflight":
                argv.append("--dry-run")
            if resume:
                argv.append("--resume")
            if stop_on_error:
                argv.append("--stop-on-error")
            s57.main(argv)
    finally:
        s57.CONFIG.clear()
        s57.CONFIG.update(original)


def _split_kinds(raw: object) -> Tuple[str, ...]:
    return tuple(part.strip() for part in str(raw or "").split(";") if part.strip())


def _validate_panel(
    panel_dir: Path,
    case: PanelCase,
    *,
    mode: str,
    rows: List[ValidationRow],
) -> None:
    component = "s5_7"
    status_path = panel_dir / "scenario_status.csv"
    _add_check(
        rows,
        component=component,
        case=case.scenario,
        check="scenario_status_exists",
        passed=status_path.exists(),
        observed=str(status_path),
        expected="existing CSV",
    )
    if not status_path.exists():
        return
    status = pd.read_csv(status_path)
    expected_count = 512 if mode == "exact" else 11
    _add_check(
        rows,
        component=component,
        case=case.scenario,
        check="scenario_count",
        passed=len(status) == expected_count,
        observed=len(status),
        expected=expected_count,
    )
    expected_statuses = {"dry_run"} if mode == "preflight" else {"ok", "resumed"}
    actual_statuses = set(status.get("run_status", pd.Series(dtype=str)).dropna().astype(str))
    _add_check(
        rows,
        component=component,
        case=case.scenario,
        check="run_statuses",
        passed=bool(actual_statuses) and actual_statuses.issubset(expected_statuses),
        observed=";".join(sorted(actual_statuses)),
        expected=";".join(sorted(expected_statuses)),
    )
    versions = set(status.get("cost_database_version", pd.Series(dtype=str)).dropna().astype(str))
    hashes = set(status.get("cost_database_sha256", pd.Series(dtype=str)).dropna().astype(str))
    _add_check(
        rows,
        component=component,
        case=case.scenario,
        check="cost_database_identity",
        passed=versions == {COST_DATABASE_VERSION} and hashes == {COST_DATABASE_SHA256},
        observed=f"versions={sorted(versions)} hashes={sorted(hashes)}",
        expected=f"{COST_DATABASE_VERSION};{COST_DATABASE_SHA256}",
    )
    kinds = {
        kind
        for raw in status.get("include_kinds", pd.Series(dtype=str)).dropna()
        for kind in _split_kinds(raw)
    }
    _add_check(
        rows,
        component=component,
        case=case.scenario,
        check="nine_strategy_kinds_planned",
        passed=kinds == EXPECTED_STRATEGY_KINDS,
        observed=";".join(sorted(kinds)),
        expected=";".join(sorted(EXPECTED_STRATEGY_KINDS)),
    )

    if mode == "preflight":
        readiness_path = panel_dir / "macc_generation_status.csv"
        _add_check(
            rows,
            component=component,
            case=case.scenario,
            check="dry_run_macc_status_written",
            passed=readiness_path.exists(),
            observed=str(readiness_path),
            expected="diagnostic CSV",
        )
        return

    readiness_path = panel_dir / "macc_generation_status.csv"
    ready = pd.read_csv(readiness_path).iloc[0] if readiness_path.exists() else pd.Series(dtype=object)
    _add_check(
        rows,
        component=component,
        case=case.scenario,
        check="macc_ready",
        passed=not ready.empty and _as_bool(ready.get("macc_ready")),
        observed=ready.get("message", "missing"),
        expected="macc_ready=True",
    )

    detail_path = panel_dir / "macc_singleton_cost_detail.csv"
    detail = pd.read_csv(detail_path) if detail_path.exists() else pd.DataFrame()
    strategies = set(detail.get("database_strategy", pd.Series(dtype=str)).dropna().astype(str))
    _add_check(
        rows,
        component="cost",
        case=case.scenario,
        check="nine_database_strategies_priced",
        passed=not detail.empty and strategies == EXPECTED_DATABASE_STRATEGIES,
        observed=";".join(sorted(strategies)),
        expected=";".join(sorted(EXPECTED_DATABASE_STRATEGIES)),
    )
    attribution = set(detail.get("attribution_method", pd.Series(dtype=str)).dropna().astype(str))
    references = set(detail.get("reference_scenario_id", pd.Series(dtype=str)).dropna().astype(str))
    _add_check(
        rows,
        component="cost",
        case=case.scenario,
        check="strict_singleton_reference",
        passed=(
            attribution == {"singleton_incremental_vs_reference"}
            and references == {f"FIG3_{case.token}_BASE"}
        ),
        observed=f"attribution={sorted(attribution)} references={sorted(references)}",
        expected=f"singleton_incremental_vs_reference;FIG3_{case.token}_BASE",
    )

    country_path = panel_dir / "sensitivity_cost_summary_by_country_measure.csv"
    global_path = panel_dir / "sensitivity_cost_summary_by_global_measure.csv"
    country = pd.read_csv(country_path) if country_path.exists() else pd.DataFrame()
    global_summary = pd.read_csv(global_path) if global_path.exists() else pd.DataFrame()
    country_strategies = set(country.get("database_strategy", pd.Series(dtype=str)).dropna().astype(str))
    global_strategies = set(global_summary.get("database_strategy", pd.Series(dtype=str)).dropna().astype(str))
    _add_check(
        rows,
        component="cost",
        case=case.scenario,
        check="country_and_global_measure_summaries",
        passed=(
            not country.empty
            and not global_summary.empty
            and country_strategies == EXPECTED_DATABASE_STRATEGIES
            and global_strategies == EXPECTED_DATABASE_STRATEGIES
        ),
        observed=f"country={len(country)} global={len(global_summary)}",
        expected="nonempty summaries covering all nine strategies",
    )
    gap = pd.to_numeric(
        global_summary.get("cost_identity_difference_usd", pd.Series(dtype=float)),
        errors="coerce",
    ).abs()
    max_gap = float(gap.max()) if not gap.empty and gap.notna().any() else math.nan
    _add_check(
        rows,
        component="cost",
        case=case.scenario,
        check="global_cost_identity",
        passed=math.isfinite(max_gap) and max_gap <= 1e-4,
        observed=max_gap,
        expected="<=1e-4 USD",
    )

    singleton_rows = status[
        status.get("include_kinds", pd.Series(dtype=str)).map(lambda raw: len(_split_kinds(raw)) == 1)
    ]
    bio_failures: List[str] = []
    result_failures: List[str] = []
    for item in singleton_rows.itertuples(index=False):
        scenario_id = str(getattr(item, "scenario_id"))
        scenario_dir = Path(str(getattr(item, "scenario_dir")))
        feedstock_path = scenario_dir / "bioenergy_feedstock_use.csv"
        postsolve_path = scenario_dir / "bioenergy_postsolve_assessment.csv"
        if not feedstock_path.exists() or not postsolve_path.exists():
            bio_failures.append(scenario_id)
        else:
            feedstock = pd.read_csv(feedstock_path, usecols=lambda col: col == "scenario")
            scenario_values = set(feedstock.get("scenario", pd.Series(dtype=str)).dropna().astype(str))
            postsolve = pd.read_csv(postsolve_path)
            postsolve_values = set(
                postsolve.get("assessment_status", pd.Series(dtype=str)).dropna().astype(str).str.lower()
            )
            if scenario_values != {case.scenario} or postsolve_values != {"valid"}:
                bio_failures.append(scenario_id)
        for filename in (
            "cost_summary.csv",
            "cost_summary_by_country_measure.csv",
            "cost_summary_by_global_measure.csv",
        ):
            if not (scenario_dir / filename).exists():
                result_failures.append(f"{scenario_id}:{filename}")
    _add_check(
        rows,
        component="bioenergy",
        case=case.scenario,
        check="singleton_bioenergy_outputs_valid",
        passed=len(singleton_rows) == 9 and not bio_failures,
        observed=f"singletons={len(singleton_rows)} failures={bio_failures[:10]}",
        expected="9 singleton runs with matching scenario and valid postsolve assessment",
    )
    _add_check(
        rows,
        component="cost",
        case=case.scenario,
        check="per_run_cost_outputs_exist",
        passed=len(singleton_rows) == 9 and not result_failures,
        observed=f"missing={result_failures[:10]}",
        expected="three cost files for every singleton",
    )

    if mode == "exact":
        shapley_path = panel_dir / "shapley_decomposition_status.csv"
        shapley = pd.read_csv(shapley_path).iloc[0] if shapley_path.exists() else pd.Series(dtype=object)
        complete = not shapley.empty and _as_bool(shapley.get("complete"))
        try:
            efficiency_gap = abs(float(shapley.get("efficiency_gap_gt")))
        except (TypeError, ValueError):
            efficiency_gap = math.nan
        _add_check(
            rows,
            component="shapley",
            case=case.scenario,
            check="complete_and_efficient",
            passed=complete and math.isfinite(efficiency_gap) and efficiency_gap <= 1e-8,
            observed=f"complete={complete} efficiency_gap_gt={efficiency_gap}",
            expected="complete=True and abs(gap)<=1e-8 Gt",
        )


def _combine_workbooks(
    output_root: Path,
    cases: Sequence[PanelCase],
    *,
    mode: str,
    panel_dirs: Optional[Mapping[str, Path]] = None,
) -> Tuple[Path, float]:
    target = output_root / "03_figure_input" / f"Figure3_abc_{mode}_v2.xlsx"
    target.parent.mkdir(parents=True, exist_ok=True)
    provenance: List[Dict[str, object]] = []
    max_baseline = 0.0
    with pd.ExcelWriter(target, engine="openpyxl") as writer:
        for case in cases:
            panel = (
                Path(panel_dirs[case.scenario])
                if panel_dirs is not None
                else _panel_stage(output_root, mode) / case.scenario
            )
            source = panel / "macc_strategy_curve_sp_m3a.xlsx"
            frame = pd.read_excel(source, sheet_name=case.sheet)
            frame.to_excel(writer, sheet_name=case.sheet, index=False)
            status = pd.read_csv(panel / "scenario_status.csv").iloc[0]
            readiness = pd.read_csv(panel / "macc_generation_status.csv").iloc[0]
            baseline = float(readiness["baseline_total_co2eq_gt"])
            max_baseline = max(max_baseline, baseline)
            provenance.append(
                {
                    "bioenergy_case": case.scenario,
                    "sheet": case.sheet,
                    "run_mode": mode,
                    "baseline_total_co2eq_gt": baseline,
                    "contribution_method": readiness.get("contribution_method", ""),
                    "cost_database_version": status.get("cost_database_version", ""),
                    "cost_database_sha256": status.get("cost_database_sha256", ""),
                    "source_workbook": str(source),
                }
            )
        pd.DataFrame(provenance).to_excel(writer, sheet_name="_provenance", index=False)
    return target, max_baseline


def _run_plot(
    output_root: Path,
    workbook: Path,
    cases: Sequence[PanelCase],
    *,
    mode: str,
    max_baseline: float,
) -> Tuple[Path, Path]:
    y_max = max(22.0, 3.0 * math.ceil((max_baseline + 0.25) / 3.0))
    stem = f"Figure3_abc_{mode}_v2"
    command = [
        sys.executable,
        "-B",
        str(PLOT_SCRIPT),
        "--input-xlsx",
        str(workbook),
        "--sheets",
        ",".join(case.sheet for case in cases),
        "--titles",
        ",".join(case.title for case in cases),
        "--output-stem",
        stem,
        "--y-max",
        str(y_max),
    ]
    env = os.environ.copy()
    env["NZF_OUTPUT_DIR"] = str(output_root)
    completed = subprocess.run(
        command,
        cwd=str(Path(__file__).resolve().parent),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"plot_{mode}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (log_dir / f"plot_{mode}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Figure 3 plot failed with exit code {completed.returncode}: {completed.stderr}")
    plot_dir = output_root / "Plot" / "Fig4"
    return plot_dir / f"{stem}.png", plot_dir / f"{stem}.svg"


def _target_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _run_cdr(
    output_root: Path,
    cases: Sequence[PanelCase],
    rows: List[ValidationRow],
    *,
    mode: str,
    targets: Sequence[float],
    land_price_grid: str,
    threads: int,
    stop_on_error: bool,
) -> None:
    import S5_7_3_strategy_macc_cdr_price as cdr

    original = copy.deepcopy(cdr.CONFIG)
    try:
        for case in cases:
            for target in targets:
                configured = copy.deepcopy(original)
                override = copy.deepcopy(configured.get("override_cfg", {}) or {})
                override.update(
                    {
                        "bioenergy_enabled": True,
                        "bioenergy_scenario": case.scenario,
                    }
                )
                configured["override_cfg"] = override
                cdr.CONFIG.clear()
                cdr.CONFIG.update(configured)
                out_dir = output_root / "04_cdr" / case.scenario / f"target_{_target_token(target)}_gt"
                argv = [
                    "--source-dir",
                    str(_panel_stage(output_root, mode) / case.scenario),
                    "--out-dir",
                    str(out_dir),
                    "--target-emissions-gt",
                    str(target),
                    "--threads",
                    str(threads),
                ]
                if land_price_grid:
                    argv.extend(["--land-price-grid", land_price_grid])
                if stop_on_error:
                    argv.append("--stop-on-error")
                cdr.main(argv)
                response_path = out_dir / "cdr_land_price_response.csv"
                required_path = out_dir / "cdr_required_land_carbon_price.csv"
                response = pd.read_csv(response_path) if response_path.exists() else pd.DataFrame()
                failed = response[
                    ~response.get("run_status", pd.Series(dtype=str)).astype(str).isin({"ok", "resumed"})
                ]
                _add_check(
                    rows,
                    component="cdr",
                    case=case.scenario,
                    check=f"target_{target:g}_gt",
                    passed=response_path.exists() and required_path.exists() and not response.empty and failed.empty,
                    observed=f"runs={len(response)} failed={len(failed)}",
                    expected="nonempty response and required-price tables with no failed run",
                )
    finally:
        cdr.CONFIG.clear()
        cdr.CONFIG.update(original)


def _write_run_manifest(
    output_root: Path,
    cases: Sequence[PanelCase],
    *,
    mode: str,
    args: argparse.Namespace,
) -> Path:
    manifest = {
        "generated_at_utc": _utc_now(),
        "mode": mode,
        "cases": [case.scenario for case in cases],
        "output_root": str(output_root),
        "allowed_output_root": str(ALLOWED_OUTPUT_ROOT),
        "threads": int(args.threads),
        "resume": bool(args.resume),
        "input_audit": bool(args.input_audit),
        "module_smoke": bool(args.module_smoke),
        "solver_smoke": bool(args.solver_smoke),
        "plot": bool(args.plot),
        "cdr_targets_gt": list(args.cdr_target or []),
        "cost_database_version": COST_DATABASE_VERSION,
        "cost_database_sha256": COST_DATABASE_SHA256,
        "source_files": {
            str(Path(__file__).name): _sha256(Path(__file__).resolve()),
            str(PLOT_SCRIPT.name): _sha256(PLOT_SCRIPT),
        },
    }
    path = output_root / "run_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--mode", choices=("preflight", "quick", "exact"), default="preflight")
    parser.add_argument("--cases", default="low,medium,high")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--input-audit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--module-smoke", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--solver-smoke",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run an additional three-case full-regression smoke before S5.7.",
    )
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cdr-target",
        action="append",
        type=float,
        default=[],
        help="Optional residual-emissions target in Gt CO2e/yr. Repeat for multiple targets.",
    )
    parser.add_argument(
        "--cdr-land-price-grid",
        default="",
        help="Optional comma-separated land carbon price grid passed to S5_7_3.",
    )
    return parser


def run_full_chain(args: argparse.Namespace) -> int:
    started_at = _utc_now()
    rows: List[ValidationRow] = []
    output_root = resolve_output_root(args.output_root)
    cases = parse_cases(args.cases)
    if int(args.threads) <= 0:
        raise ValueError("--threads must be positive")
    if any(not math.isfinite(float(value)) for value in args.cdr_target):
        raise ValueError("--cdr-target values must be finite")
    output_root.mkdir(parents=True, exist_ok=True)
    previous_output = os.environ.get("NZF_OUTPUT_DIR")
    os.environ["NZF_OUTPUT_DIR"] = str(output_root)
    error = ""
    try:
        _write_run_manifest(output_root, cases, mode=args.mode, args=args)

        if args.input_audit:
            _run_input_audit(output_root, cases, rows)
        else:
            _add_check(rows, component="bioenergy_input", check="input_audit", passed=None)

        if args.module_smoke:
            _run_bioenergy_smoke(
                output_root,
                cases,
                rows,
                layer="module_integration",
            )
        else:
            _add_check(rows, component="bioenergy_module_integration", check="module_smoke", passed=None)

        if args.solver_smoke:
            _run_bioenergy_smoke(
                output_root,
                cases,
                rows,
                layer="full_regression",
            )
        else:
            _add_check(rows, component="bioenergy_full_regression", check="solver_smoke", passed=None)

        if _has_failures(rows) and args.stop_on_error:
            raise RuntimeError("A pre-S5.7 validation gate failed")

        _run_s57_panels(
            output_root,
            cases,
            mode=args.mode,
            resume=bool(args.resume),
            stop_on_error=bool(args.stop_on_error),
            threads=int(args.threads),
        )
        for case in cases:
            _validate_panel(
                _panel_stage(output_root, args.mode) / case.scenario,
                case,
                mode=args.mode,
                rows=rows,
            )

        if args.mode == "preflight":
            _add_check(
                rows,
                component="figure",
                check="combine_and_plot",
                passed=None,
                message="preflight has no solved MACC workbook",
            )
            if args.cdr_target:
                _add_check(
                    rows,
                    component="cdr",
                    check="cdr_targets",
                    passed=False,
                    observed=args.cdr_target,
                    expected="quick or exact mode",
                )
        elif not _has_failures(rows):
            workbook, max_baseline = _combine_workbooks(
                output_root,
                cases,
                mode=args.mode,
            )
            _add_check(
                rows,
                component="figure",
                check="combined_workbook",
                passed=workbook.exists(),
                observed=str(workbook),
                expected="existing workbook",
            )
            if args.plot:
                png_path, svg_path = _run_plot(
                    output_root,
                    workbook,
                    cases,
                    mode=args.mode,
                    max_baseline=max_baseline,
                )
                _add_check(
                    rows,
                    component="figure",
                    check="png_and_svg",
                    passed=png_path.exists() and svg_path.exists(),
                    observed=f"{png_path};{svg_path}",
                    expected="PNG and SVG",
                )
            else:
                _add_check(rows, component="figure", check="plot", passed=None)

            if args.cdr_target:
                _run_cdr(
                    output_root,
                    cases,
                    rows,
                    mode=args.mode,
                    targets=args.cdr_target,
                    land_price_grid=str(args.cdr_land_price_grid or ""),
                    threads=int(args.threads),
                    stop_on_error=bool(args.stop_on_error),
                )
        else:
            _add_check(
                rows,
                component="figure",
                check="combine_and_plot",
                passed=False,
                message="skipped because upstream validation failed",
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        _add_check(
            rows,
            component="pipeline",
            check="uncaught_exception",
            passed=False,
            observed=error,
            expected="no exception",
        )
    finally:
        if previous_output is None:
            os.environ.pop("NZF_OUTPUT_DIR", None)
        else:
            os.environ["NZF_OUTPUT_DIR"] = previous_output

    csv_path, json_path = _write_validation(
        rows,
        output_root=output_root,
        mode=args.mode,
        started_at=started_at,
        error=error,
    )
    print(f"[BIOENERGY_MACC] validation={csv_path}")
    print(f"[BIOENERGY_MACC] status={json_path}")
    print(f"[BIOENERGY_MACC] overall_passed={not _has_failures(rows) and not error}")
    return 0 if not _has_failures(rows) and not error else 1


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    return run_full_chain(args)


if __name__ == "__main__":
    raise SystemExit(main())
