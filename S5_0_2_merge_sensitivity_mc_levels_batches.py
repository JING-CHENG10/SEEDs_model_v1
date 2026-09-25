from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict

from config_paths import get_results_base
import S5_0_1_sensitivity_mc_levels as mclevels


def _detect_total_batches(root_output_dir: Path, batches_subdir: str) -> int:
    batch_root = root_output_dir / str(batches_subdir)
    if not batch_root.exists():
        raise FileNotFoundError(f"Batch root not found: {batch_root}")
    pattern = re.compile(r"^batch_(\d+)_of_(\d+)$")
    counts = set()
    indices = set()
    for child in batch_root.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if not match:
            continue
        indices.add(int(match.group(1)))
        counts.add(int(match.group(2)))
    if not counts:
        raise RuntimeError(f"No batch directories matching batch_XX_of_YY under: {batch_root}")
    if len(counts) > 1:
        raise RuntimeError(
            f"Inconsistent batch count suffixes under {batch_root}: {sorted(counts)}"
        )
    total_batches = counts.pop()
    if indices:
        missing = [i for i in range(1, total_batches + 1) if i not in indices]
        if missing:
            preview = ", ".join(str(i) for i in missing[:10])
            suffix = f" ... (+{len(missing) - 10} more)" if len(missing) > 10 else ""
            raise FileNotFoundError(
                f"Detected total_batches={total_batches}, but missing batch dirs: {preview}{suffix}"
            )
    return int(total_batches)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge batched S5_0 MC sensitivity outputs.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Root output dir. Default: <NZF_OUTPUT_DIR>/MC_Sensitivity",
    )
    parser.add_argument(
        "--total-batches",
        type=int,
        default=0,
        help=(
            "Number of batch dirs to merge. Default 0 means auto-detect from "
            "<output-dir>/batches/batch_XX_of_YY."
        ),
    )
    parser.add_argument(
        "--batch-assignment",
        type=str,
        default=str(mclevels.CONFIG.get("batch", {}).get("assignment", "round_robin") or "round_robin"),
        choices=["round_robin", "contiguous"],
    )
    parser.add_argument(
        "--method",
        type=str,
        default=str(mclevels.CONFIG.get("importance", {}).get("method", "linear") or "linear"),
        choices=["linear", "rank"],
    )
    parser.add_argument(
        "--window",
        type=float,
        default=mclevels.CONFIG.get("importance", {}).get("window"),
        help="Hard window around target (Gt). If too few samples, fallback to Gaussian weights.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=mclevels.CONFIG.get("importance", {}).get("sigma"),
        help="Gaussian sigma for target-local weighting (Gt). CONFIG None falls back to sample std.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=int(mclevels.CONFIG.get("importance", {}).get("min_samples", 30) or 30),
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="",
        help=(
            "Optional comma list. Empty uses S5_0_1 CONFIG['importance'] "
            "target_range_gt and target_bin_width_gt; if target_range_gt is None, "
            "derives targets from the retained merged sample range."
        ),
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    targets = mclevels._parse_float_list(args.targets) if args.targets else None
    base_out = Path(args.output_dir) if args.output_dir else Path(get_results_base()) / "MC_Sensitivity"
    batches_subdir = str(mclevels.CONFIG.get("batch", {}).get("batches_subdir", "batches"))
    total_batches = (
        int(args.total_batches)
        if int(args.total_batches or 0) > 0
        else _detect_total_batches(base_out, batches_subdir)
    )
    print(f"[S5_0_MERGE] output_dir={base_out}")
    print(f"[S5_0_MERGE] total_batches={total_batches}")

    run_cfg: Dict[str, object] = {
        "batch": {
            "enabled": True,
            "batch_index": 1,
            "total_batches": int(total_batches),
            "assignment": str(args.batch_assignment),
            "batches_subdir": batches_subdir,
        }
    }
    batch_state = mclevels._resolve_batch_settings(run_cfg)

    merged_outputs = mclevels._merge_batch_outputs(
        root_output_dir=base_out,
        batch_state=batch_state,
        targets=targets,
        method=args.method,
        window=args.window,
        sigma=args.sigma,
        min_samples=args.min_samples,
    )
    for key, path in merged_outputs.items():
        print(f"[DONE] {key}: {path}")


if __name__ == "__main__":
    main()
