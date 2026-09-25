# -*- coding: utf-8 -*-
"""Run one S5.8 country-by-strategy batch.

Countries are partitioned across batches. Every batch runs its own detailed
reference scenario, followed by nine single-strategy scenarios for each
assigned country.
"""
from __future__ import annotations

import argparse
import copy
import os
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import S5_8_1_country_strategy_map_sensitivity as s58


def _parse_int(raw: object, *, name: str) -> int:
    try:
        return int(str(raw).strip())
    except Exception as exc:
        raise ValueError(f"Invalid integer for {name}: {raw}") from exc


def _parse_bool(raw: object) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _split_csv(raw: object) -> List[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def _batch_tag(index: int, count: int) -> str:
    return f"batch_{int(index):02d}_of_{int(count):02d}"


def _infer_batch_index(default: int) -> Tuple[int, str]:
    for env_name in ("S58_BATCH_INDEX", "SLURM_ARRAY_TASK_ID"):
        raw = str(os.environ.get(env_name, "") or "").strip()
        if raw:
            return _parse_int(raw, name=env_name), env_name
    for env_name in ("SLURM_JOB_NAME", "JOB_NAME"):
        job_name = str(os.environ.get(env_name, "") or "").strip()
        match = re.search(r"(\d+)$", job_name) if job_name else None
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
    mode = str(assignment or "round_robin").strip().lower()
    if mode in {"contiguous", "block", "blocks"}:
        start = (len(items) * (batch_index - 1)) // batch_count
        end = (len(items) * batch_index) // batch_count
        return list(items[start:end])
    if mode in {"round_robin", "round-robin", "rr"}:
        return [
            item
            for index, item in enumerate(items, start=1)
            if ((index - 1) % batch_count) == (batch_index - 1)
        ]
    raise ValueError("assignment must be round_robin or contiguous")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one S5.8 batch.")
    parser.add_argument("--batch-index", type=int, default=None)
    parser.add_argument("--total-batches", type=int, default=None)
    parser.add_argument("--assignment", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--countries", type=str, default=None)
    parser.add_argument("--max-countries", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", default=None)
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--clear-existing-runs", action="store_true", default=None)
    parser.add_argument(
        "--keep-existing-runs",
        action="store_false",
        dest="clear_existing_runs",
    )
    parser.add_argument("--stop-on-error", action="store_true", default=None)
    parser.add_argument("--threads", type=int, default=None)
    return parser


def _root_output_dir(cfg: Mapping[str, object]) -> Path:
    raw = str(cfg.get("output_dir", "") or "").strip()
    return Path(raw) if raw else s58._default_output_dir()


def _apply_overrides(
    base_cfg: Dict[str, object],
    args: Optional[argparse.Namespace],
) -> Dict[str, object]:
    cfg = copy.deepcopy(base_cfg)
    batch_index, source = _infer_batch_index(1)
    total_batches = int(os.environ.get("S58_TOTAL_BATCHES", "") or 1)
    assignment = str(
        os.environ.get("S58_BATCH_ASSIGNMENT", "") or "round_robin"
    )

    output_env = str(os.environ.get("S58_OUTPUT_DIR", "") or "").strip()
    if output_env:
        cfg["output_dir"] = output_env
    countries_env = str(os.environ.get("S58_COUNTRIES", "") or "").strip()
    if countries_env:
        cfg["country_filter"] = _split_csv(countries_env)
    max_countries_env = str(
        os.environ.get("S58_MAX_COUNTRIES", "") or ""
    ).strip()
    if max_countries_env:
        cfg["max_countries"] = _parse_int(
            max_countries_env,
            name="S58_MAX_COUNTRIES",
        )
    for env_name, cfg_key in (
        ("S58_DRY_RUN", "dry_run"),
        ("S58_RESUME", "resume"),
        ("S58_CLEAR_EXISTING_RUNS", "clear_existing_run_dirs_when_no_resume"),
        ("S58_STOP_ON_ERROR", "stop_on_error"),
    ):
        raw = str(os.environ.get(env_name, "") or "").strip()
        if raw:
            cfg[cfg_key] = _parse_bool(raw)

    threads_env = str(os.environ.get("S58_THREADS", "") or "").strip()
    if threads_env:
        override = copy.deepcopy(cfg.get("override_cfg", {}) or {})
        override["linear_solver_threads"] = _parse_int(
            threads_env,
            name="S58_THREADS",
        )
        cfg["override_cfg"] = override

    if args is not None:
        if args.batch_index is not None:
            batch_index = int(args.batch_index)
            source = "--batch-index"
        if args.total_batches is not None:
            total_batches = int(args.total_batches)
        if args.assignment:
            assignment = str(args.assignment)
        if args.output_dir:
            cfg["output_dir"] = args.output_dir
        if args.countries:
            cfg["country_filter"] = _split_csv(args.countries)
        if args.max_countries is not None:
            cfg["max_countries"] = int(args.max_countries)
        if args.dry_run is not None:
            cfg["dry_run"] = bool(args.dry_run)
        if args.resume is not None:
            cfg["resume"] = bool(args.resume)
        if args.clear_existing_runs is not None:
            cfg["clear_existing_run_dirs_when_no_resume"] = bool(
                args.clear_existing_runs
            )
        if args.stop_on_error is not None:
            cfg["stop_on_error"] = bool(args.stop_on_error)
        if args.threads is not None:
            override = copy.deepcopy(cfg.get("override_cfg", {}) or {})
            override["linear_solver_threads"] = int(args.threads)
            cfg["override_cfg"] = override

    if total_batches <= 0:
        raise ValueError("total_batches must be positive")
    if batch_index < 1 or batch_index > total_batches:
        raise ValueError(
            f"batch_index must be within 1..{total_batches}, got {batch_index}"
        )

    paths = s58.s56.DataPaths()
    shared_cfg = s58.s56.ScenarioConfig()
    universe = s58.s56.build_universe_from_dict_v3(
        paths.dict_v3_path,
        shared_cfg,
    )
    all_countries = s58.s56._active_countries(cfg, universe)
    assigned = _select_batch_items(
        all_countries,
        batch_count=total_batches,
        batch_index=batch_index,
        assignment=assignment,
    )
    root = _root_output_dir(cfg)
    tag = _batch_tag(batch_index, total_batches)
    cfg["output_dir"] = str(root / "batches" / tag)
    cfg["country_filter"] = assigned
    cfg["run_reference"] = True
    cfg["_s58_batch"] = {
        "batch_index": batch_index,
        "total_batches": total_batches,
        "batch_tag": tag,
        "assignment": assignment,
        "batch_index_source": source,
        "assigned_countries": len(assigned),
        "total_countries": len(all_countries),
        "root_output_dir": str(root),
    }
    return cfg


def main() -> None:
    args = _build_arg_parser().parse_args()
    cfg = _apply_overrides(s58.CONFIG, args)
    batch = cfg.get("_s58_batch", {}) or {}
    print(
        "[S5_8_BATCH] "
        f"batch_index={batch.get('batch_index')} "
        f"total_batches={batch.get('total_batches')} "
        f"assignment={batch.get('assignment')} "
        f"assigned_countries={batch.get('assigned_countries')}/"
        f"{batch.get('total_countries')} "
        f"resume={cfg.get('resume')} "
        f"output_dir={cfg.get('output_dir')}"
    )

    previous = copy.deepcopy(s58.CONFIG)
    s58.CONFIG.clear()
    s58.CONFIG.update(cfg)
    try:
        s58.main([])
    finally:
        s58.CONFIG.clear()
        s58.CONFIG.update(previous)


if __name__ == "__main__":
    main()
