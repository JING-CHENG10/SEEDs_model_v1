# -*- coding: utf-8 -*-
"""Run one S5_5 Region/Item/Process MC batch.

This is a thin batch wrapper around
S5_5_1_Region-Item-Process_Importance_Gen.py. It keeps cluster submission
consistent with the S5_4 batch workflow while leaving the core MC and
postprocess logic in the Gen script.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple


def _load_gen_module():
    module_path = Path(__file__).with_name("S5_5_1_Region-Item-Process_Importance_Gen.py")
    spec = importlib.util.spec_from_file_location("s5_5_region_item_process_importance_gen", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load S5_5 Gen module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_int(raw: object, *, name: str) -> int:
    try:
        return int(str(raw).strip())
    except Exception as exc:
        raise ValueError(f"Invalid integer for {name}: {raw}") from exc


def _parse_bool(raw: object) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _infer_batch_index(default: int) -> Tuple[int, str]:
    for env_name in ("RIPMC_BATCH_INDEX", "SLURM_ARRAY_TASK_ID"):
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
    parser = argparse.ArgumentParser(description="Run one S5_5 Region/Item/Process MC batch.")
    parser.add_argument("--batch-index", type=int, default=None)
    parser.add_argument("--total-batches", type=int, default=None)
    parser.add_argument("--assignment", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--postprocess", action="store_true", default=None)
    parser.add_argument("--no-postprocess", action="store_false", dest="postprocess")
    return parser


def _apply_overrides(base_cfg: Dict[str, object], args: Optional[argparse.Namespace]) -> Tuple[Dict[str, object], str]:
    cfg = copy.deepcopy(base_cfg)
    batch_cfg = cfg.setdefault("batch", {})

    default_batch_index = int(batch_cfg.get("batch_index", 1) or 1)
    batch_index, batch_index_source = _infer_batch_index(default_batch_index)
    batch_cfg["batch_index"] = batch_index

    env_total_batches = str(os.environ.get("RIPMC_TOTAL_BATCHES", "") or "").strip()
    if env_total_batches:
        batch_cfg["total_batches"] = _parse_int(env_total_batches, name="RIPMC_TOTAL_BATCHES")

    env_assignment = str(os.environ.get("RIPMC_BATCH_ASSIGNMENT", "") or "").strip()
    if env_assignment:
        batch_cfg["assignment"] = env_assignment

    env_output_dir = str(os.environ.get("RIPMC_OUTPUT_DIR", "") or "").strip()
    if env_output_dir:
        cfg["output_dir"] = env_output_dir

    env_samples = str(os.environ.get("RIPMC_SAMPLES", "") or "").strip()
    if env_samples:
        cfg["samples"] = _parse_int(env_samples, name="RIPMC_SAMPLES")

    env_max_runs = str(os.environ.get("RIPMC_MAX_RUNS", "") or "").strip()
    if env_max_runs:
        cfg["max_runs"] = _parse_int(env_max_runs, name="RIPMC_MAX_RUNS")

    env_resume = str(os.environ.get("RIPMC_RESUME", "") or "").strip()
    if env_resume:
        cfg["resume"] = _parse_bool(env_resume)

    env_postprocess = str(os.environ.get("RIPMC_POSTPROCESS", "") or "").strip()
    if env_postprocess:
        cfg["postprocess"] = _parse_bool(env_postprocess)

    if args is not None:
        if args.batch_index is not None:
            batch_cfg["batch_index"] = int(args.batch_index)
            batch_index_source = "--batch-index"
        if args.total_batches is not None:
            batch_cfg["total_batches"] = int(args.total_batches)
        if args.assignment:
            batch_cfg["assignment"] = str(args.assignment)
        if args.output_dir:
            cfg["output_dir"] = str(args.output_dir)
        if args.samples is not None:
            cfg["samples"] = int(args.samples)
        if args.max_runs is not None:
            cfg["max_runs"] = int(args.max_runs)
        if args.resume is not None:
            cfg["resume"] = bool(args.resume)
        if args.postprocess is not None:
            cfg["postprocess"] = bool(args.postprocess)

    cfg["run_mc"] = True
    cfg["postprocess_scope"] = "active"
    return cfg, batch_index_source


def main() -> None:
    args = _build_arg_parser().parse_args()
    gen = _load_gen_module()
    cfg, batch_index_source = _apply_overrides(gen.CONFIG, args)
    batch_cfg = cfg.get("batch", {}) or {}

    print(
        "[S5_5_BATCH] "
        f"batch_index={batch_cfg.get('batch_index')} "
        f"total_batches={batch_cfg.get('total_batches')} "
        f"assignment={batch_cfg.get('assignment')} "
        f"resume={cfg.get('resume')} "
        f"postprocess={cfg.get('postprocess')} "
        f"source={batch_index_source}"
    )
    if str(cfg.get("output_dir", "") or "").strip():
        print(f"[S5_5_BATCH] output_dir={cfg['output_dir']}")

    gen._run_mc(cfg)
    if bool(cfg.get("postprocess", True)):
        gen._postprocess(cfg)


if __name__ == "__main__":
    main()
