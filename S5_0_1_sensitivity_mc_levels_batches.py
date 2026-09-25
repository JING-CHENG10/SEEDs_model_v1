from __future__ import annotations

import copy
import os
import re
from typing import Dict, Tuple

import S5_0_1_sensitivity_mc_levels as mclevels


def _parse_int(raw: object, *, name: str) -> int:
    try:
        return int(str(raw).strip())
    except Exception as exc:
        raise ValueError(f"Invalid integer for {name}: {raw}") from exc


def _parse_bool(raw: object) -> bool:
    text = str(raw).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _infer_batch_index(default: int) -> Tuple[int, str]:
    env_value = str(os.environ.get("MCLEVELS_BATCH_INDEX", "") or "").strip()
    if env_value:
        return _parse_int(env_value, name="MCLEVELS_BATCH_INDEX"), "MCLEVELS_BATCH_INDEX"

    for env_name in ("SLURM_JOB_NAME", "JOB_NAME"):
        job_name = str(os.environ.get(env_name, "") or "").strip()
        if not job_name:
            continue
        match = re.search(r"(\d+)$", job_name)
        if match:
            return int(match.group(1)), env_name

    return int(default), "CONFIG"


def _apply_env_overrides(cfg: Dict[str, object]) -> Dict[str, object]:
    resolved = copy.deepcopy(cfg)
    batch_cfg = resolved.setdefault("batch", {})

    default_batch_index = int(batch_cfg.get("batch_index", 1) or 1)
    batch_index, batch_index_source = _infer_batch_index(default_batch_index)
    batch_cfg["batch_index"] = batch_index

    total_batches_raw = str(os.environ.get("MCLEVELS_TOTAL_BATCHES", "") or "").strip()
    if total_batches_raw:
        batch_cfg["total_batches"] = _parse_int(total_batches_raw, name="MCLEVELS_TOTAL_BATCHES")

    assignment_raw = str(os.environ.get("MCLEVELS_BATCH_ASSIGNMENT", "") or "").strip()
    if assignment_raw:
        batch_cfg["assignment"] = assignment_raw

    enabled_raw = str(os.environ.get("MCLEVELS_BATCH_ENABLED", "") or "").strip()
    if enabled_raw:
        batch_cfg["enabled"] = _parse_bool(enabled_raw)

    output_dir_raw = str(os.environ.get("MCLEVELS_OUTPUT_DIR", "") or "").strip()
    if output_dir_raw:
        resolved["output_dir"] = output_dir_raw

    resume_raw = str(os.environ.get("MCLEVELS_RESUME", "") or "").strip()
    if resume_raw:
        resolved["resume"] = _parse_bool(resume_raw)

    samples_raw = str(os.environ.get("MCLEVELS_SAMPLES", "") or "").strip()
    if samples_raw:
        resolved["samples"] = _parse_int(samples_raw, name="MCLEVELS_SAMPLES")

    resolved["_batch_index_source"] = batch_index_source
    return resolved


def main() -> None:
    cfg = _apply_env_overrides(mclevels.CONFIG)
    mclevels.CONFIG.clear()
    mclevels.CONFIG.update(cfg)

    batch_cfg = mclevels.CONFIG.get("batch", {}) or {}
    print(
        "[S5_0_BATCH] "
        f"batch_index={batch_cfg.get('batch_index')} "
        f"total_batches={batch_cfg.get('total_batches')} "
        f"assignment={batch_cfg.get('assignment')} "
        f"enabled={batch_cfg.get('enabled')} "
        f"source={mclevels.CONFIG.get('_batch_index_source')}"
    )
    print(
        "[S5_0_BATCH] "
        f"slurm_job_name={os.environ.get('SLURM_JOB_NAME', '') or '<empty>'}"
    )
    if str(mclevels.CONFIG.get("output_dir", "") or "").strip():
        print(f"[S5_0_BATCH] output_dir={mclevels.CONFIG['output_dir']}")

    try:
        mclevels.main()
    finally:
        mclevels.CONFIG.pop("_batch_index_source", None)


if __name__ == "__main__":
    main()
