# -*- coding: utf-8 -*-
"""Run one S5.6 maximum emission-reduction batch.

Batch design:
  - batch_01 runs baseline, global max strategy, global single-lever
    diagnostics, and its assigned country subset.
  - batch_02..N run only their assigned country subsets.

Each batch writes to:
  <output_dir>/batches/batch_XX_of_YY/

Merge later with:
  python S5_6_2_merge_batches.py
"""
from __future__ import annotations

import argparse
import copy
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from S1_0_schema import ScenarioConfig
from S2_0_load_data import DataPaths, build_universe_from_dict_v3
import S5_6_1_max_emission_reduction_potential as maxred


def _parse_int(raw: object, *, name: str) -> int:
    try:
        return int(str(raw).strip())
    except Exception as exc:
        raise ValueError(f"Invalid integer for {name}: {raw}") from exc


def _parse_bool(raw: object) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _batch_tag(batch_index: int, batch_count: int) -> str:
    return f"batch_{int(batch_index):02d}_of_{int(batch_count):02d}"


def _infer_batch_index(default: int) -> Tuple[int, str]:
    for env_name in ("S56_BATCH_INDEX", "SLURM_ARRAY_TASK_ID"):
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


def _select_batch_items(
    items: Sequence[str],
    *,
    batch_count: int,
    batch_index: int,
    assignment: str,
) -> List[str]:
    if batch_count <= 1:
        return list(items)
    if batch_index < 1 or batch_index > batch_count:
        raise ValueError(f"batch_index must be within 1..{batch_count}, got {batch_index}")
    assignment_l = str(assignment or "round_robin").strip().lower()
    n = len(items)
    if n == 0:
        return []
    if assignment_l in {"contiguous", "block", "blocks"}:
        start = (n * (batch_index - 1)) // batch_count
        end = (n * batch_index) // batch_count
        return list(items[start:end])
    if assignment_l in {"round_robin", "round-robin", "rr"}:
        return [item for idx, item in enumerate(items, start=1) if ((idx - 1) % batch_count) == (batch_index - 1)]
    raise ValueError("assignment must be 'round_robin' or 'contiguous'")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one S5.6 maximum-reduction batch.")
    parser.add_argument("--batch-index", type=int, default=None)
    parser.add_argument("--total-batches", type=int, default=None)
    parser.add_argument("--assignment", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--clear-existing-runs", action="store_true", default=None)
    parser.add_argument("--keep-existing-runs", action="store_false", dest="clear_existing_runs")
    parser.add_argument("--dry-run", action="store_true", default=None)
    parser.add_argument("--max-countries", type=int, default=None)
    parser.add_argument("--countries", type=str, default=None)
    parser.add_argument("--country-individual-levers", action="store_true", default=None)
    parser.add_argument("--no-global-individual-levers", action="store_true", default=None)
    parser.add_argument("--no-detailed-country-accounting", action="store_true", default=None)
    parser.add_argument("--iis", action="store_true", default=None)
    parser.add_argument("--iis-timeout", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    return parser


def _root_output_dir(cfg: Dict[str, object]) -> Path:
    raw = str(cfg.get("output_dir", "") or "").strip()
    return Path(raw) if raw else maxred._default_output_dir()


def _apply_overrides(base_cfg: Dict[str, object], args: Optional[argparse.Namespace]) -> Tuple[Dict[str, object], str]:
    cfg = copy.deepcopy(base_cfg)

    batch_index, source = _infer_batch_index(1)
    total_batches = int(os.environ.get("S56_TOTAL_BATCHES", "") or 1)
    assignment = str(os.environ.get("S56_BATCH_ASSIGNMENT", "") or "round_robin")

    output_dir_env = str(os.environ.get("S56_OUTPUT_DIR", "") or "").strip()
    if output_dir_env:
        cfg["output_dir"] = output_dir_env

    resume_env = str(os.environ.get("S56_RESUME", "") or "").strip()
    if resume_env:
        cfg["resume"] = _parse_bool(resume_env)

    clear_env = str(os.environ.get("S56_CLEAR_EXISTING_RUNS", "") or "").strip()
    if clear_env:
        cfg["clear_existing_run_dirs_when_no_resume"] = _parse_bool(clear_env)

    dry_env = str(os.environ.get("S56_DRY_RUN", "") or "").strip()
    if dry_env:
        cfg["dry_run"] = _parse_bool(dry_env)

    max_c_env = str(os.environ.get("S56_MAX_COUNTRIES", "") or "").strip()
    if max_c_env:
        cfg["max_countries"] = _parse_int(max_c_env, name="S56_MAX_COUNTRIES")

    countries_env = str(os.environ.get("S56_COUNTRIES", "") or "").strip()
    if countries_env:
        cfg["country_filter"] = [x.strip() for x in countries_env.split(",") if x.strip()]

    ctry_levers_env = str(os.environ.get("S56_COUNTRY_INDIVIDUAL_LEVERS", "") or "").strip()
    if ctry_levers_env:
        cfg["run_country_individual_levers"] = _parse_bool(ctry_levers_env)

    global_levers_env = str(os.environ.get("S56_GLOBAL_INDIVIDUAL_LEVERS", "") or "").strip()
    if global_levers_env:
        cfg["run_global_individual_levers"] = _parse_bool(global_levers_env)

    detailed_env = str(os.environ.get("S56_DETAILED_COUNTRY_ACCOUNTING", "") or "").strip()
    if detailed_env:
        cfg["detailed_country_accounting"] = _parse_bool(detailed_env)

    if args is not None:
        if args.batch_index is not None:
            batch_index = int(args.batch_index)
            source = "--batch-index"
        if args.total_batches is not None:
            total_batches = int(args.total_batches)
        if args.assignment:
            assignment = str(args.assignment)
        if args.output_dir:
            cfg["output_dir"] = str(args.output_dir)
        if args.resume is not None:
            cfg["resume"] = bool(args.resume)
        if args.clear_existing_runs is not None:
            cfg["clear_existing_run_dirs_when_no_resume"] = bool(args.clear_existing_runs)
        if args.dry_run is not None:
            cfg["dry_run"] = bool(args.dry_run)
        if args.max_countries is not None:
            cfg["max_countries"] = int(args.max_countries)
        if args.countries is not None:
            cfg["country_filter"] = [x.strip() for x in str(args.countries).split(",") if x.strip()]
        if args.country_individual_levers is not None:
            cfg["run_country_individual_levers"] = bool(args.country_individual_levers)
        if args.no_global_individual_levers is not None:
            cfg["run_global_individual_levers"] = not bool(args.no_global_individual_levers)
        if args.no_detailed_country_accounting is not None:
            cfg["detailed_country_accounting"] = not bool(args.no_detailed_country_accounting)
        override = copy.deepcopy(cfg.get("override_cfg", {}) or {})
        if args.iis:
            override["linear_enable_infeasible_iis"] = True
        if args.iis_timeout is not None:
            override["iis_timeout"] = int(args.iis_timeout)
        if args.threads is not None:
            override["linear_solver_threads"] = int(args.threads)
        cfg["override_cfg"] = override

    if total_batches <= 0:
        raise ValueError("total_batches must be positive")
    if batch_index < 1 or batch_index > total_batches:
        raise ValueError(f"batch_index must be within 1..{total_batches}, got {batch_index}")

    paths = DataPaths()
    scfg = ScenarioConfig()
    universe = build_universe_from_dict_v3(paths.dict_v3_path, scfg)
    all_countries = maxred._active_countries(cfg, universe)
    batch_countries = _select_batch_items(
        all_countries,
        batch_count=total_batches,
        batch_index=batch_index,
        assignment=assignment,
    )

    root_output_dir = _root_output_dir(cfg)
    tag = _batch_tag(batch_index, total_batches)
    cfg["output_dir"] = str(root_output_dir / "batches" / tag)
    cfg["country_filter"] = list(batch_countries)
    cfg["_s56_batch"] = {
        "batch_index": int(batch_index),
        "total_batches": int(total_batches),
        "batch_tag": tag,
        "assignment": assignment,
        "batch_index_source": source,
        "assigned_countries": len(batch_countries),
        "total_countries": len(all_countries),
        "root_output_dir": str(root_output_dir),
    }

    if batch_index != 1:
        cfg["run_baseline"] = False
        cfg["run_global_all_levers"] = False
        cfg["run_global_individual_levers"] = False
        cfg["detailed_country_accounting"] = False

    # In batch mode every job should run country scenarios for its assigned
    # subset. Empty subsets are allowed when total_batches > number of countries.
    cfg["run_country_one_at_a_time"] = True
    return cfg, source


def main() -> None:
    args = _build_arg_parser().parse_args()
    cfg, source = _apply_overrides(maxred.CONFIG, args)
    batch = cfg.get("_s56_batch", {}) or {}
    print(
        "[S5_6_BATCH] "
        f"batch_index={batch.get('batch_index')} "
        f"total_batches={batch.get('total_batches')} "
        f"tag={batch.get('batch_tag')} "
        f"assignment={batch.get('assignment')} "
        f"assigned_countries={batch.get('assigned_countries')}/{batch.get('total_countries')} "
        f"resume={cfg.get('resume')} "
        f"clear_existing_runs={cfg.get('clear_existing_run_dirs_when_no_resume')} "
        f"source={source}"
    )
    print(f"[S5_6_BATCH] output_dir={cfg.get('output_dir')}")
    print(f"[S5_6_BATCH] slurm_job_name={os.environ.get('SLURM_JOB_NAME', '') or '<empty>'}")

    previous = copy.deepcopy(maxred.CONFIG)
    maxred.CONFIG.clear()
    maxred.CONFIG.update(cfg)
    try:
        maxred.main([])
    finally:
        maxred.CONFIG.clear()
        maxred.CONFIG.update(previous)


if __name__ == "__main__":
    main()
