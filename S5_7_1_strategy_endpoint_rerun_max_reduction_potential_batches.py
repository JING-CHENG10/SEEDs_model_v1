# -*- coding: utf-8 -*-
"""Run one S5.7 Strategy endpoint-rerun batch.

Batching is scenario-plan based, not country based. With the current
nine-strategy package, Shapley creates 512 model evaluations including the
baseline, and this wrapper splits those scenario plans across batch jobs.

Each batch writes to:
  <output_dir>/batches/batch_XX_of_YY/

Merge later with:
  python S5_7_2_strategy_endpoint_rerun_merge_batches.py --total-batches YY --shapley
"""
from __future__ import annotations

import argparse
import copy
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import S5_7_1_strategy_endpoint_rerun_max_reduction_potential as s57


def _parse_int(raw: object, *, name: str) -> int:
    try:
        return int(str(raw).strip())
    except Exception as exc:
        raise ValueError(f"Invalid integer for {name}: {raw}") from exc


def _parse_bool(raw: object) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _split_csv(raw: object) -> List[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def _batch_tag(batch_index: int, total_batches: int) -> str:
    return f"batch_{int(batch_index):02d}_of_{int(total_batches):02d}"


def _infer_batch_index(default: int) -> Tuple[int, str]:
    for env_name in ("S57_BATCH_INDEX", "SLURM_ARRAY_TASK_ID"):
        raw = str(os.environ.get(env_name, "") or "").strip()
        if raw:
            return _parse_int(raw, name=env_name), env_name

    for env_name in ("SLURM_JOB_NAME", "JOB_NAME"):
        job_name = str(os.environ.get(env_name, "") or "").strip()
        if not job_name:
            continue
        match = re.search(r"(\d+)$", job_name)
        if match:
            return int(match.group(1)), env_name

    return int(default), "CONFIG"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one S5.7 Strategy endpoint-rerun batch.")
    parser.add_argument("--batch-index", type=int, default=None)
    parser.add_argument("--total-batches", type=int, default=None)
    parser.add_argument("--assignment", type=str, default=None, help="round_robin or contiguous")
    parser.add_argument("--out-dir", type=str, default=None, help="Root S5.7 output dir; batch subdir is added.")
    parser.add_argument("--dry-run", action="store_true", default=None)
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--clear-existing-run-dirs", action="store_true", default=None)
    parser.add_argument("--keep-existing-run-dirs", action="store_false", dest="clear_existing_run_dirs")
    parser.add_argument("--shapley", action="store_true", default=None)
    parser.add_argument("--no-shapley", action="store_false", dest="shapley")
    parser.add_argument("--shapley-keep-standard", action="store_true", default=None)
    parser.add_argument("--shapley-max-strategies", type=int, default=None)
    parser.add_argument("--country-runs", action="store_true", default=None)
    parser.add_argument("--country-individual-levers", action="store_true", default=None)
    parser.add_argument("--countries", type=str, default=None, help="Comma-separated country selectors.")
    parser.add_argument("--country", action="append", default=None, help="Repeatable country selector.")
    parser.add_argument("--max-countries", type=int, default=None)
    parser.add_argument("--detailed-country-accounting", action="store_true", default=None)
    parser.add_argument("--write-summary", action="store_true", default=None)
    parser.add_argument("--no-write-summary", action="store_false", dest="write_summary")
    parser.add_argument("--stop-on-error", action="store_true", default=None)
    parser.add_argument("--threads", type=int, default=None)
    return parser


def _env_bool(name: str, cfg: Dict[str, object], key: str) -> None:
    raw = str(os.environ.get(name, "") or "").strip()
    if raw:
        cfg[key] = _parse_bool(raw)


def _apply_overrides(base_cfg: Dict[str, object], args: Optional[argparse.Namespace]) -> Tuple[Dict[str, object], str]:
    cfg = copy.deepcopy(base_cfg)
    base_batch = cfg.get("batch", {}) or {}

    default_batch_index = int(base_batch.get("batch_index", 1) or 1)
    batch_index, batch_index_source = _infer_batch_index(default_batch_index)
    total_batches = int(os.environ.get("S57_TOTAL_BATCHES", "") or base_batch.get("total_batches", 1) or 1)
    assignment = str(os.environ.get("S57_BATCH_ASSIGNMENT", "") or base_batch.get("assignment", "round_robin"))

    root_output_raw = str(os.environ.get("S57_OUTPUT_DIR", "") or cfg.get("output_dir", "") or "").strip()
    root_output = Path(root_output_raw) if root_output_raw else s57._default_output_dir()

    _env_bool("S57_DRY_RUN", cfg, "dry_run")
    _env_bool("S57_RESUME", cfg, "resume")
    _env_bool("S57_CLEAR_EXISTING_RUNS", cfg, "clear_existing_run_dirs_when_no_resume")
    _env_bool("S57_COUNTRY_RUNS", cfg, "run_country_one_at_a_time")
    _env_bool("S57_COUNTRY_INDIVIDUAL_LEVERS", cfg, "run_country_individual_levers")
    _env_bool("S57_DETAILED_COUNTRY_ACCOUNTING", cfg, "detailed_country_accounting")
    _env_bool("S57_STOP_ON_ERROR", cfg, "stop_on_error")
    _env_bool("S57_WRITE_SUMMARY_OUTPUTS", cfg, "write_summary_outputs")

    countries_env = str(os.environ.get("S57_COUNTRIES", "") or "").strip()
    if countries_env:
        cfg["country_filter"] = _split_csv(countries_env)

    max_c_env = str(os.environ.get("S57_MAX_COUNTRIES", "") or "").strip()
    if max_c_env:
        cfg["max_countries"] = _parse_int(max_c_env, name="S57_MAX_COUNTRIES")

    decomp = copy.deepcopy(cfg.get("decomposition", {}) or {})
    shapley_env = str(os.environ.get("S57_SHAPLEY", "") or "").strip()
    if shapley_env:
        if _parse_bool(shapley_env):
            decomp["method"] = "shapley"
            decomp["run_shapley"] = True
        else:
            decomp["method"] = "single"
            decomp["run_shapley"] = False
    keep_env = str(os.environ.get("S57_SHAPLEY_KEEP_STANDARD", "") or "").strip()
    if keep_env:
        decomp["keep_standard_global_scenarios"] = _parse_bool(keep_env)
    max_s_env = str(os.environ.get("S57_SHAPLEY_MAX_STRATEGIES", "") or "").strip()
    if max_s_env:
        decomp["max_strategies"] = _parse_int(max_s_env, name="S57_SHAPLEY_MAX_STRATEGIES")
    cfg["decomposition"] = decomp

    threads_env = str(os.environ.get("S57_THREADS", "") or "").strip()
    if threads_env:
        override = copy.deepcopy(cfg.get("override_cfg", {}) or {})
        override["linear_solver_threads"] = _parse_int(threads_env, name="S57_THREADS")
        cfg["override_cfg"] = override

    if args is not None:
        if args.batch_index is not None:
            batch_index = int(args.batch_index)
            batch_index_source = "--batch-index"
        if args.total_batches is not None:
            total_batches = int(args.total_batches)
        if args.assignment:
            assignment = str(args.assignment)
        if args.out_dir:
            root_output = Path(args.out_dir)
        if args.dry_run is not None:
            cfg["dry_run"] = bool(args.dry_run)
        if args.resume is not None:
            cfg["resume"] = bool(args.resume)
        if args.clear_existing_run_dirs is not None:
            cfg["clear_existing_run_dirs_when_no_resume"] = bool(args.clear_existing_run_dirs)
        if args.shapley is not None:
            decomp = copy.deepcopy(cfg.get("decomposition", {}) or {})
            if bool(args.shapley):
                decomp["method"] = "shapley"
                decomp["run_shapley"] = True
            else:
                decomp["method"] = "single"
                decomp["run_shapley"] = False
            cfg["decomposition"] = decomp
        if args.shapley_keep_standard is not None:
            decomp = copy.deepcopy(cfg.get("decomposition", {}) or {})
            decomp["keep_standard_global_scenarios"] = bool(args.shapley_keep_standard)
            cfg["decomposition"] = decomp
        if args.shapley_max_strategies is not None:
            decomp = copy.deepcopy(cfg.get("decomposition", {}) or {})
            decomp["max_strategies"] = int(args.shapley_max_strategies)
            cfg["decomposition"] = decomp
        if args.country_runs is not None:
            cfg["run_country_one_at_a_time"] = bool(args.country_runs)
        if args.country_individual_levers is not None:
            cfg["run_country_individual_levers"] = bool(args.country_individual_levers)
        selectors = []
        if args.countries:
            selectors.extend(_split_csv(args.countries))
        if args.country:
            selectors.extend(args.country)
        if selectors:
            cfg["country_filter"] = selectors
        if args.max_countries is not None:
            cfg["max_countries"] = int(args.max_countries)
        if args.detailed_country_accounting is not None:
            cfg["detailed_country_accounting"] = bool(args.detailed_country_accounting)
        if args.write_summary is not None:
            cfg["write_summary_outputs"] = bool(args.write_summary)
        if args.stop_on_error is not None:
            cfg["stop_on_error"] = bool(args.stop_on_error)
        if args.threads is not None:
            override = copy.deepcopy(cfg.get("override_cfg", {}) or {})
            override["linear_solver_threads"] = int(args.threads)
            cfg["override_cfg"] = override

    if total_batches <= 0:
        raise ValueError("total_batches must be positive")
    if batch_index < 1 or batch_index > total_batches:
        raise ValueError(f"batch_index must be within 1..{total_batches}, got {batch_index}")

    tag = _batch_tag(batch_index, total_batches)
    cfg["output_dir"] = str(root_output / "batches" / tag)
    cfg["batch"] = {
        "enabled": True,
        "batch_index": int(batch_index),
        "total_batches": int(total_batches),
        "assignment": assignment,
    }
    cfg["_s57_batch"] = {
        "batch_index": int(batch_index),
        "total_batches": int(total_batches),
        "batch_tag": tag,
        "assignment": assignment,
        "batch_index_source": batch_index_source,
        "root_output_dir": str(root_output),
    }
    return cfg, batch_index_source


def main() -> None:
    args = _build_arg_parser().parse_args()
    cfg, source = _apply_overrides(s57.CONFIG, args)
    batch = cfg.get("_s57_batch", {}) or {}
    decomp = cfg.get("decomposition", {}) or {}
    print(
        "[S5_7_BATCH] "
        f"batch_index={batch.get('batch_index')} "
        f"total_batches={batch.get('total_batches')} "
        f"tag={batch.get('batch_tag')} "
        f"assignment={batch.get('assignment')} "
        f"shapley={bool(decomp.get('run_shapley', False)) or str(decomp.get('method', '')).lower() == 'shapley'} "
        f"resume={cfg.get('resume')} "
        f"clear_existing_runs={cfg.get('clear_existing_run_dirs_when_no_resume')} "
        f"write_summary={cfg.get('write_summary_outputs')} "
        f"source={source}"
    )
    print(f"[S5_7_BATCH] output_dir={cfg.get('output_dir')}")
    print(f"[S5_7_BATCH] slurm_job_name={os.environ.get('SLURM_JOB_NAME', '') or '<empty>'}")

    previous = copy.deepcopy(s57.CONFIG)
    s57.CONFIG.clear()
    s57.CONFIG.update(cfg)
    try:
        s57.main([])
    finally:
        s57.CONFIG.clear()
        s57.CONFIG.update(previous)


if __name__ == "__main__":
    main()
