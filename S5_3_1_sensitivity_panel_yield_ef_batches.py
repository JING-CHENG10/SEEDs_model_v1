from __future__ import annotations

import argparse
import copy
import os
import re
from typing import Dict, Optional, Tuple

import S5_3_1_sensitivity_panel_yield_ef as panel


def _parse_int(raw: object, *, name: str) -> int:
    try:
        return int(str(raw).strip())
    except Exception as exc:
        raise ValueError(f"Invalid integer for {name}: {raw}") from exc


def _parse_bool(raw: object) -> bool:
    text = str(raw).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _infer_batch_index(default: int) -> Tuple[int, str]:
    env_value = str(os.environ.get("PANEL_BATCH_INDEX", "") or "").strip()
    if env_value:
        return _parse_int(env_value, name="PANEL_BATCH_INDEX"), "PANEL_BATCH_INDEX"

    for env_name in ("SLURM_JOB_NAME", "JOB_NAME"):
        job_name = str(os.environ.get(env_name, "") or "").strip()
        if not job_name:
            continue
        match = re.search(r"(\d+)$", job_name)
        if match:
            return int(match.group(1)), env_name

    return int(default), "CONFIG"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one S5_3_1 panel batch.")
    parser.add_argument("--batch-index", type=int, default=None)
    parser.add_argument("--total-batches", type=int, default=None)
    parser.add_argument("--assignment", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-enabled", action="store_true", default=None)
    parser.add_argument("--no-batch-enabled", action="store_false", dest="batch_enabled")
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--clear-existing-runs", action="store_true", default=None)
    parser.add_argument("--keep-existing-runs", action="store_false", dest="clear_existing_runs")
    parser.add_argument("--max-runs", type=int, default=None)
    return parser


def _apply_overrides(
    cfg: Dict[str, object],
    args: Optional[argparse.Namespace] = None,
) -> Dict[str, object]:
    resolved = copy.deepcopy(cfg)
    batch_cfg = resolved.setdefault("batch", {})

    default_batch_index = int(batch_cfg.get("batch_index", 1) or 1)
    batch_index, batch_index_source = _infer_batch_index(default_batch_index)
    batch_cfg["batch_index"] = batch_index

    total_batches_raw = str(os.environ.get("PANEL_TOTAL_BATCHES", "") or "").strip()
    if total_batches_raw:
        batch_cfg["total_batches"] = _parse_int(total_batches_raw, name="PANEL_TOTAL_BATCHES")

    assignment_raw = str(os.environ.get("PANEL_BATCH_ASSIGNMENT", "") or "").strip()
    if assignment_raw:
        batch_cfg["assignment"] = assignment_raw

    enabled_raw = str(os.environ.get("PANEL_BATCH_ENABLED", "") or "").strip()
    if enabled_raw:
        batch_cfg["enabled"] = _parse_bool(enabled_raw)

    output_dir_raw = str(os.environ.get("PANEL_OUTPUT_DIR", "") or "").strip()
    if output_dir_raw:
        resolved["output_dir"] = output_dir_raw

    resume_raw = str(os.environ.get("PANEL_RESUME", "") or "").strip()
    if resume_raw:
        resolved["resume"] = _parse_bool(resume_raw)

    clear_raw = str(os.environ.get("PANEL_CLEAR_EXISTING_RUNS", "") or "").strip()
    if clear_raw:
        resolved["clear_existing_run_dirs_when_no_resume"] = _parse_bool(clear_raw)

    max_runs_raw = str(os.environ.get("PANEL_MAX_RUNS", "") or "").strip()
    if max_runs_raw:
        resolved["max_runs"] = _parse_int(max_runs_raw, name="PANEL_MAX_RUNS")

    if args is not None:
        if args.batch_index is not None:
            batch_cfg["batch_index"] = int(args.batch_index)
            batch_index_source = "--batch-index"
        if args.total_batches is not None:
            batch_cfg["total_batches"] = int(args.total_batches)
        if args.assignment:
            batch_cfg["assignment"] = str(args.assignment)
        if args.batch_enabled is not None:
            batch_cfg["enabled"] = bool(args.batch_enabled)
        if args.output_dir:
            resolved["output_dir"] = str(args.output_dir)
        if args.resume is not None:
            resolved["resume"] = bool(args.resume)
        if args.clear_existing_runs is not None:
            resolved["clear_existing_run_dirs_when_no_resume"] = bool(args.clear_existing_runs)
        if args.max_runs is not None:
            resolved["max_runs"] = int(args.max_runs)

    output_dir_text = str(resolved.get("output_dir", "") or "").strip()
    output_root = panel._resolve_panel_output_root(output_dir_text)
    resolved["output_dir"] = str(output_root)
    panel._sync_panel_output_environment(output_root)

    resolved["_batch_index_source"] = batch_index_source
    return resolved


def main() -> None:
    args = _build_arg_parser().parse_args()
    cfg = _apply_overrides(panel.CONFIG, args)
    panel.CONFIG.clear()
    panel.CONFIG.update(cfg)

    batch_cfg = panel.CONFIG.get("batch", {}) or {}
    print(
        "[S5_3_1_BATCH] "
        f"batch_index={batch_cfg.get('batch_index')} "
        f"total_batches={batch_cfg.get('total_batches')} "
        f"assignment={batch_cfg.get('assignment')} "
        f"enabled={batch_cfg.get('enabled')} "
        f"resume={panel.CONFIG.get('resume')} "
        f"clear_existing_runs={panel.CONFIG.get('clear_existing_run_dirs_when_no_resume')} "
        f"source={panel.CONFIG.get('_batch_index_source')}"
    )
    print(
        "[S5_3_1_BATCH] "
        f"slurm_job_name={os.environ.get('SLURM_JOB_NAME', '') or '<empty>'}"
    )
    if str(panel.CONFIG.get("output_dir", "") or "").strip():
        print(f"[S5_3_1_BATCH] output_dir={panel.CONFIG['output_dir']}")
    print(f"[S5_3_1_BATCH] NZF_OUTPUT_DIR={os.environ.get('NZF_OUTPUT_DIR', '')}")
    print(f"[S5_3_1_BATCH] PANEL_OUTPUT_DIR={os.environ.get('PANEL_OUTPUT_DIR', '')}")

    try:
        panel.main()
    finally:
        panel.CONFIG.pop("_batch_index_source", None)


if __name__ == "__main__":
    main()
