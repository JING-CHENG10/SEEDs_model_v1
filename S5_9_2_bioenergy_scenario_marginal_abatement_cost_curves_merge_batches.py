# -*- coding: utf-8 -*-
"""Merge bioenergy-scenario marginal-abatement-cost batches.

For every selected bioenergy panel this program delegates the scenario-level
deduplication and strict cost-provenance checks to the S5.7 merger, validates
the completed v2.0 chain, combines the panel workbooks, and regenerates the
PNG/SVG figure.  All persistent output must remain below ``Code/output``.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

import S5_7_2_strategy_endpoint_rerun_merge_batches as s57merge
import S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves as chain
import S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves_batches as batch_runner


DEFAULT_OUTPUT_ROOT = batch_runner.DEFAULT_OUTPUT_ROOT


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_bool(raw: object) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int(raw: object, *, name: str) -> int:
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer for {name}: {raw}") from exc


def _optional_bool(cli_value: Optional[bool], env_name: str, default: bool) -> bool:
    if cli_value is not None:
        return bool(cli_value)
    raw = str(os.environ.get(env_name, "") or "").strip()
    return _parse_bool(raw) if raw else bool(default)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--mode", choices=("quick", "exact"), default=None)
    parser.add_argument("--cases", default=None)
    parser.add_argument("--total-batches", type=int, default=None)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--input-audit", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=None)
    return parser


def _resolve_cli(
    args: argparse.Namespace,
) -> tuple[Path, str, tuple[chain.PanelCase, ...], int, bool, bool, bool]:
    mode = str(args.mode or os.environ.get("S59_MODE", "") or "exact").strip().lower()
    if mode not in {"quick", "exact"}:
        raise ValueError(f"mode must be quick or exact, got {mode!r}")
    output_raw = args.output_root or os.environ.get("S59_OUTPUT_ROOT", "") or DEFAULT_OUTPUT_ROOT
    output_root = chain.resolve_output_root(output_raw)
    case_raw = args.cases or os.environ.get("S59_CASES", "") or "low,medium,high"
    cases = chain.parse_cases(case_raw)
    default_batches = 32 if mode == "exact" else 3
    total_raw = (
        args.total_batches
        if args.total_batches is not None
        else os.environ.get("S59_TOTAL_BATCHES", "") or default_batches
    )
    total_batches = _parse_int(total_raw, name="total_batches")
    if total_batches <= 0:
        raise ValueError("total_batches must be positive")
    unique_plan_count = 512 if mode == "exact" else 11
    if total_batches > unique_plan_count:
        raise ValueError(
            f"total_batches={total_batches} exceeds the {unique_plan_count} unique {mode} plans"
        )
    strict = _optional_bool(args.strict, "S59_MERGE_STRICT", True)
    input_audit = _optional_bool(args.input_audit, "S59_INPUT_AUDIT", True)
    plot = _optional_bool(args.plot, "S59_PLOT", True)
    return output_root, mode, cases, total_batches, strict, input_audit, plot


def _manifest_checks(
    output_root: Path,
    cases: Sequence[chain.PanelCase],
    *,
    mode: str,
    total_batches: int,
    rows: list[chain.ValidationRow],
) -> None:
    expected_cases = {case.scenario for case in cases}
    for index in range(1, total_batches + 1):
        tag = batch_runner._batch_tag(index, total_batches)
        path = output_root / "batch_manifests" / f"{tag}.json"
        if not path.exists():
            chain._add_check(
                rows,
                component="batch_manifest",
                case=tag,
                check="manifest_exists",
                passed=False,
                observed=str(path),
                expected="existing completed manifest",
            )
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            chain._add_check(
                rows,
                component="batch_manifest",
                case=tag,
                check="manifest_readable",
                passed=False,
                observed=f"{type(exc).__name__}: {exc}",
                expected="valid JSON",
            )
            continue
        observed_cases = set(map(str, payload.get("cases", []) or []))
        passed = (
            str(payload.get("status", "")) == "completed"
            and str(payload.get("mode", "")) == mode
            and int(payload.get("batch_index", 0) or 0) == index
            and int(payload.get("total_batches", 0) or 0) == total_batches
            and expected_cases.issubset(observed_cases)
        )
        chain._add_check(
            rows,
            component="batch_manifest",
            case=tag,
            check="manifest_contract",
            passed=passed,
            observed=(
                f"status={payload.get('status')} mode={payload.get('mode')} "
                f"batch={payload.get('batch_index')}/{payload.get('total_batches')} "
                f"cases={sorted(observed_cases)}"
            ),
            expected=(
                f"completed {mode} batch={index}/{total_batches} "
                f"cases={sorted(expected_cases)}"
            ),
        )


def _merge_panel(
    output_root: Path,
    case: chain.PanelCase,
    *,
    mode: str,
    total_batches: int,
    strict: bool,
) -> Path:
    cfg = dict(s57merge.CONFIG)
    cfg.update(
        {
            "output_dir": str(output_root / case.scenario),
            "total_batches": total_batches,
            "batches_subdir": "batches",
            "merged_subdir": "merged",
            "strict": strict,
            "shapley": mode == "exact",
            "baseline_scenario_id": f"FIG3_{case.token}_BASE",
        }
    )
    return s57merge.run_merge(cfg)


def _write_merge_manifest(
    output_root: Path,
    cases: Sequence[chain.PanelCase],
    *,
    mode: str,
    total_batches: int,
    strict: bool,
    input_audit: bool,
    plot: bool,
    status: str,
    error: str = "",
) -> Path:
    payload = {
        "generated_at_utc": _utc_now(),
        "status": status,
        "error": error,
        "output_root": str(output_root),
        "mode": mode,
        "cases": [case.scenario for case in cases],
        "total_batches": total_batches,
        "strict": strict,
        "input_audit": input_audit,
        "plot": plot,
        "cost_database_version": chain.COST_DATABASE_VERSION,
        "cost_database_sha256": chain.COST_DATABASE_SHA256,
        "source_sha256": chain._sha256(Path(__file__).resolve()),
    }
    path = output_root / "batch_merge_manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_merge(args: argparse.Namespace) -> int:
    started_at = _utc_now()
    rows: list[chain.ValidationRow] = []
    output_root, mode, cases, total_batches, strict, input_audit, plot = _resolve_cli(args)
    output_root.mkdir(parents=True, exist_ok=True)
    error = ""
    _write_merge_manifest(
        output_root,
        cases,
        mode=mode,
        total_batches=total_batches,
        strict=strict,
        input_audit=input_audit,
        plot=plot,
        status="running",
    )
    previous_output = os.environ.get("NZF_OUTPUT_DIR")
    os.environ["NZF_OUTPUT_DIR"] = str(output_root)
    try:
        if input_audit:
            chain._run_input_audit(output_root, cases, rows)
        else:
            chain._add_check(rows, component="bioenergy_input", check="input_audit", passed=None)

        _manifest_checks(
            output_root,
            cases,
            mode=mode,
            total_batches=total_batches,
            rows=rows,
        )
        if strict and chain._has_failures(rows):
            raise RuntimeError("Batch-manifest validation failed before merge")

        panel_dirs: dict[str, Path] = {}
        for case in cases:
            merged = _merge_panel(
                output_root,
                case,
                mode=mode,
                total_batches=total_batches,
                strict=strict,
            )
            panel_dirs[case.scenario] = merged
            chain._add_check(
                rows,
                component="batch_merge",
                case=case.scenario,
                check="merged_directory",
                passed=merged.exists(),
                observed=str(merged),
                expected="existing directory",
            )
            chain._validate_panel(merged, case, mode=mode, rows=rows)

        if chain._has_failures(rows):
            raise RuntimeError("Merged panel validation failed")

        workbook, max_baseline = chain._combine_workbooks(
            output_root,
            cases,
            mode=mode,
            panel_dirs=panel_dirs,
        )
        chain._add_check(
            rows,
            component="figure",
            check="combined_workbook",
            passed=workbook.exists(),
            observed=str(workbook),
            expected="existing workbook",
        )
        if plot:
            png_path, svg_path = chain._run_plot(
                output_root,
                workbook,
                cases,
                mode=mode,
                max_baseline=max_baseline,
            )
            chain._add_check(
                rows,
                component="figure",
                check="png_and_svg",
                passed=png_path.exists() and svg_path.exists(),
                observed=f"{png_path};{svg_path}",
                expected="PNG and SVG",
            )
        else:
            chain._add_check(rows, component="figure", check="plot", passed=None)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        chain._add_check(
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

    validation_csv, status_json = chain._write_validation(
        rows,
        output_root=output_root,
        mode=f"batch_merge_{mode}",
        started_at=started_at,
        error=error,
    )
    passed = not chain._has_failures(rows) and not error
    manifest = _write_merge_manifest(
        output_root,
        cases,
        mode=mode,
        total_batches=total_batches,
        strict=strict,
        input_audit=input_audit,
        plot=plot,
        status="completed" if passed else "failed",
        error=error,
    )
    print(f"[BIOENERGY_MACC_MERGE] validation={validation_csv}")
    print(f"[BIOENERGY_MACC_MERGE] status={status_json}")
    print(f"[BIOENERGY_MACC_MERGE] manifest={manifest}")
    print(f"[BIOENERGY_MACC_MERGE] overall_passed={passed}")
    return 0 if passed else 1


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    return run_merge(args)


if __name__ == "__main__":
    raise SystemExit(main())
