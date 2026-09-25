# -*- coding: utf-8 -*-
"""Extract aggregate Region/Item/Process tables from all S5_5 MC batches.

The per-batch runner writes detailed model outputs under
<output_dir>/batches/batch_XX_of_YY. This script scans and validates the
complete batch folder set before building merged structure tables under
<output_dir>/merged_structure.

It intentionally does not copy batch run folders or raw model output files.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import os
from pathlib import Path
from typing import Dict, Optional


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


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract aggregate Region/Item/Process tables from all S5_5 MC batches."
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument(
        "--total-batches",
        type=int,
        default=None,
        help=(
            "Compatibility only. Merge auto-detects and validates the complete "
            "batch_XX_of_YY directory set from the directory names."
        ),
    )
    parser.add_argument("--target-nearest-n", type=int, default=None)
    parser.add_argument("--target-window-gt", type=float, default=None)
    return parser


def _apply_overrides(base_cfg: Dict[str, object], args: Optional[argparse.Namespace]) -> Dict[str, object]:
    cfg = copy.deepcopy(base_cfg)
    batch_cfg = cfg.setdefault("batch", {})

    env_output_dir = str(os.environ.get("RIPMC_OUTPUT_DIR", "") or "").strip()
    if env_output_dir:
        cfg["output_dir"] = env_output_dir

    env_samples = str(os.environ.get("RIPMC_SAMPLES", "") or "").strip()
    if env_samples:
        cfg["samples"] = _parse_int(env_samples, name="RIPMC_SAMPLES")

    env_total_batches = str(os.environ.get("RIPMC_TOTAL_BATCHES", "") or "").strip()
    if env_total_batches:
        batch_cfg["total_batches"] = _parse_int(env_total_batches, name="RIPMC_TOTAL_BATCHES")

    if args is not None:
        if args.output_dir:
            cfg["output_dir"] = str(args.output_dir)
        if args.samples is not None:
            cfg["samples"] = int(args.samples)
        if args.total_batches is not None:
            batch_cfg["total_batches"] = int(args.total_batches)
        if args.target_nearest_n is not None:
            cfg["target_nearest_n"] = int(args.target_nearest_n)
        if args.target_window_gt is not None:
            cfg["target_window_gt"] = float(args.target_window_gt)

    cfg["run_mc"] = False
    cfg["postprocess"] = True
    cfg["postprocess_scope"] = "all_batches"
    return cfg


def main() -> None:
    args = _build_arg_parser().parse_args()
    gen = _load_gen_module()
    cfg = _apply_overrides(gen.CONFIG, args)
    root = gen._root_output_dir(cfg)
    out_dir = root / str(cfg.get("merged_structure_subdir", "merged_structure") or "merged_structure")
    batch_dirs = gen._batch_dirs_for_postprocess(root, cfg)

    print(
        "[S5_5_MERGE] "
        "mode=extract_aggregate_only no_raw_file_copy=True "
        f"detected_batches={len(batch_dirs)} "
        f"first_batch={batch_dirs[0].name} "
        f"last_batch={batch_dirs[-1].name} "
        f"output_root={root} "
        f"merged_structure={out_dir} "
        f"samples={cfg.get('samples')}"
    )
    gen._postprocess(cfg)


if __name__ == "__main__":
    main()
