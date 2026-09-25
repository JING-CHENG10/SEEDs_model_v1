# -*- coding: utf-8 -*-
"""Run one bioenergy-scenario marginal-abatement-cost batch.

Each job applies the same S5.7 batch index to the low, medium, and high
bioenergy panels.  S5.7 automatically adds a local matched BASE to every
isolated batch, which keeps strict-singleton physical abatement and cost
attribution self-contained.

Output layout::

    <output_root>/<bioenergy_case>/batches/batch_XX_of_YY/

Merge completed batches with
``S5_9_2_bioenergy_scenario_marginal_abatement_cost_curves_merge_batches``.
All persistent output is rejected unless it is below ``Code/output``.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd

import S5_7_1_strategy_endpoint_rerun_max_reduction_potential as s57
import S5_9_1_bioenergy_scenario_marginal_abatement_cost_curves as chain


DEFAULT_OUTPUT_ROOT = chain.ALLOWED_OUTPUT_ROOT / "Bioenergy_Scenario_MACC_Batches"


@dataclass(frozen=True)
class BatchSettings:
    output_root: Path
    cases: Tuple[chain.PanelCase, ...]
    mode: str
    batch_index: int
    total_batches: int
    assignment: str
    threads: int
    dry_run: bool
    resume: bool
    clear_existing_runs: bool
    stop_on_error: bool
    batch_index_source: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_bool(raw: object) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int(raw: object, *, name: str) -> int:
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer for {name}: {raw}") from exc


def _batch_tag(batch_index: int, total_batches: int) -> str:
    return f"batch_{int(batch_index):02d}_of_{int(total_batches):02d}"


def _infer_batch_index() -> Tuple[int, str]:
    for env_name in ("S59_BATCH_INDEX", "SLURM_ARRAY_TASK_ID"):
        raw = str(os.environ.get(env_name, "") or "").strip()
        if raw:
            return _parse_int(raw, name=env_name), env_name
    for env_name in ("SLURM_JOB_NAME", "JOB_NAME"):
        job_name = str(os.environ.get(env_name, "") or "").strip()
        match = re.search(r"(\d+)$", job_name) if job_name else None
        if match:
            return int(match.group(1)), env_name
    return 1, "default"


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
    parser.add_argument("--batch-index", type=int, default=None)
    parser.add_argument("--total-batches", type=int, default=None)
    parser.add_argument("--assignment", choices=("round_robin", "contiguous"), default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--clear-existing-runs",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--stop-on-error", action=argparse.BooleanOptionalAction, default=None)
    return parser


def resolve_settings(args: argparse.Namespace) -> BatchSettings:
    mode = str(args.mode or os.environ.get("S59_MODE", "") or "exact").strip().lower()
    if mode not in {"quick", "exact"}:
        raise ValueError(f"mode must be quick or exact, got {mode!r}")

    output_raw = args.output_root or os.environ.get("S59_OUTPUT_ROOT", "") or DEFAULT_OUTPUT_ROOT
    output_root = chain.resolve_output_root(output_raw)
    case_raw = args.cases or os.environ.get("S59_CASES", "") or "low,medium,high"
    cases = chain.parse_cases(case_raw)

    inferred_index, inferred_source = _infer_batch_index()
    batch_index = int(args.batch_index) if args.batch_index is not None else inferred_index
    batch_index_source = "--batch-index" if args.batch_index is not None else inferred_source
    default_batches = 32 if mode == "exact" else 3
    total_raw = (
        args.total_batches
        if args.total_batches is not None
        else os.environ.get("S59_TOTAL_BATCHES", "") or default_batches
    )
    total_batches = _parse_int(total_raw, name="total_batches")
    assignment = str(
        args.assignment or os.environ.get("S59_BATCH_ASSIGNMENT", "") or "round_robin"
    ).strip().lower()
    if assignment not in {"round_robin", "contiguous"}:
        raise ValueError("assignment must be round_robin or contiguous")

    threads_raw = args.threads if args.threads is not None else os.environ.get("S59_THREADS", "") or 3
    threads = _parse_int(threads_raw, name="threads")
    if total_batches <= 0:
        raise ValueError("total_batches must be positive")
    if batch_index < 1 or batch_index > total_batches:
        raise ValueError(f"batch_index must be within 1..{total_batches}, got {batch_index}")
    unique_plan_count = 512 if mode == "exact" else 11
    if total_batches > unique_plan_count:
        raise ValueError(
            f"total_batches={total_batches} exceeds the {unique_plan_count} unique {mode} plans"
        )
    if threads <= 0:
        raise ValueError("threads must be positive")

    return BatchSettings(
        output_root=output_root,
        cases=cases,
        mode=mode,
        batch_index=batch_index,
        total_batches=total_batches,
        assignment=assignment,
        threads=threads,
        dry_run=_optional_bool(args.dry_run, "S59_DRY_RUN", False),
        resume=_optional_bool(args.resume, "S59_RESUME", True),
        clear_existing_runs=_optional_bool(
            args.clear_existing_runs,
            "S59_CLEAR_EXISTING_RUNS",
            False,
        ),
        stop_on_error=_optional_bool(args.stop_on_error, "S59_STOP_ON_ERROR", True),
        batch_index_source=batch_index_source,
    )


def _panel_config(
    base_config: dict[str, object],
    case: chain.PanelCase,
    settings: BatchSettings,
) -> dict[str, object]:
    cfg = chain._configure_panel(base_config, case)
    tag = _batch_tag(settings.batch_index, settings.total_batches)
    panel_root = settings.output_root / case.scenario
    cfg["output_dir"] = str(panel_root / "batches" / tag)
    cfg["dry_run"] = settings.dry_run
    cfg["resume"] = settings.resume
    cfg["clear_existing_run_dirs_when_no_resume"] = settings.clear_existing_runs
    cfg["stop_on_error"] = settings.stop_on_error
    cfg["write_summary_outputs"] = False
    cfg["batch"] = {
        "enabled": settings.total_batches > 1,
        "batch_index": settings.batch_index,
        "total_batches": settings.total_batches,
        "assignment": settings.assignment,
    }
    decomposition = copy.deepcopy(cfg.get("decomposition", {}) or {})
    decomposition["method"] = "shapley" if settings.mode == "exact" else "single"
    decomposition["run_shapley"] = settings.mode == "exact"
    cfg["decomposition"] = decomposition
    override = copy.deepcopy(cfg.get("override_cfg", {}) or {})
    override["linear_solver_threads"] = settings.threads
    cfg["override_cfg"] = override
    return cfg


def _manifest_payload(settings: BatchSettings, *, status: str) -> dict[str, object]:
    payload = asdict(settings)
    payload["output_root"] = str(settings.output_root)
    payload["cases"] = [case.scenario for case in settings.cases]
    payload["batch_tag"] = _batch_tag(settings.batch_index, settings.total_batches)
    payload["status"] = status
    payload["updated_at_utc"] = _utc_now()
    payload["unique_plans_per_panel"] = 512 if settings.mode == "exact" else 11
    payload["estimated_total_evaluations_all_panels"] = (
        (512 if settings.mode == "exact" else 11) + settings.total_batches - 1
    ) * len(settings.cases)
    payload["source_sha256"] = chain._sha256(Path(__file__).resolve())
    return payload


def _write_manifest(settings: BatchSettings, payload: dict[str, object]) -> Path:
    manifest_dir = settings.output_root / "batch_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / f"{_batch_tag(settings.batch_index, settings.total_batches)}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_batch(settings: BatchSettings) -> Path:
    settings.output_root.mkdir(parents=True, exist_ok=True)
    payload = _manifest_payload(settings, status="running")
    manifest_path = _write_manifest(settings, payload)
    original = copy.deepcopy(s57.CONFIG)
    previous_output = os.environ.get("NZF_OUTPUT_DIR")
    os.environ["NZF_OUTPUT_DIR"] = str(settings.output_root)
    panel_rows: list[dict[str, object]] = []
    try:
        for case in settings.cases:
            cfg = _panel_config(original, case, settings)
            s57.CONFIG.clear()
            s57.CONFIG.update(cfg)
            s57.main([])
            batch_dir = Path(str(cfg["output_dir"]))
            plan_path = batch_dir / "batch_plan_manifest.csv"
            status_path = batch_dir / "scenario_status.csv"
            plan = pd.read_csv(plan_path) if plan_path.exists() else pd.DataFrame()
            status = pd.read_csv(status_path) if status_path.exists() else pd.DataFrame()
            panel_rows.append(
                {
                    "case": case.scenario,
                    "batch_dir": str(batch_dir),
                    "selected_plan_rows": int(len(plan)),
                    "status_rows": int(len(status)),
                    "run_statuses": sorted(
                        status.get("run_status", pd.Series(dtype=str)).dropna().astype(str).unique()
                    ),
                }
            )
        payload.update(
            {
                "status": "completed",
                "updated_at_utc": _utc_now(),
                "panels": panel_rows,
            }
        )
        return _write_manifest(settings, payload)
    except Exception as exc:
        payload.update(
            {
                "status": "failed",
                "updated_at_utc": _utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
                "panels": panel_rows,
            }
        )
        _write_manifest(settings, payload)
        raise
    finally:
        s57.CONFIG.clear()
        s57.CONFIG.update(original)
        if previous_output is None:
            os.environ.pop("NZF_OUTPUT_DIR", None)
        else:
            os.environ["NZF_OUTPUT_DIR"] = previous_output


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    settings = resolve_settings(args)
    print(
        "[BIOENERGY_MACC_BATCH] "
        f"mode={settings.mode} "
        f"batch={settings.batch_index}/{settings.total_batches} "
        f"assignment={settings.assignment} "
        f"cases={','.join(case.scenario for case in settings.cases)} "
        f"dry_run={settings.dry_run} "
        f"source={settings.batch_index_source}"
    )
    manifest = run_batch(settings)
    print(f"[BIOENERGY_MACC_BATCH] manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
