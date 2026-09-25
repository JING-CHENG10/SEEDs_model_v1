from __future__ import annotations

import argparse
import copy
import os
import re
from typing import Dict, Optional, Tuple

import S5_4_1_monte_carlo_full_variables as fullmc


def _parse_int(raw: object, *, name: str) -> int:
    try:
        return int(str(raw).strip())
    except Exception as exc:
        raise ValueError(f"Invalid integer for {name}: {raw}") from exc


def _parse_bool(raw: object) -> bool:
    text = str(raw).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _infer_batch_index(default: int) -> Tuple[int, str]:
    env_value = str(os.environ.get("FULLMC_BATCH_INDEX", "") or "").strip()
    if env_value:
        return _parse_int(env_value, name="FULLMC_BATCH_INDEX"), "FULLMC_BATCH_INDEX"

    for env_name in ("SLURM_JOB_NAME", "JOB_NAME"):
        job_name = str(os.environ.get(env_name, "") or "").strip()
        if not job_name:
            continue
        match = re.search(r"(\d+)$", job_name)
        if match:
            return int(match.group(1)), env_name

    return int(default), "CONFIG"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one S5_4_1 full-MC batch.")
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
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--ef-intensity-baseline-emissions-csv", type=str, default=None)
    parser.add_argument("--require-ef-co2eq-intensity", action="store_true", default=None)
    parser.add_argument("--no-require-ef-co2eq-intensity", action="store_false", dest="require_ef_co2eq_intensity")
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

    total_batches_raw = str(os.environ.get("FULLMC_TOTAL_BATCHES", "") or "").strip()
    if total_batches_raw:
        batch_cfg["total_batches"] = _parse_int(total_batches_raw, name="FULLMC_TOTAL_BATCHES")

    assignment_raw = str(os.environ.get("FULLMC_BATCH_ASSIGNMENT", "") or "").strip()
    if assignment_raw:
        batch_cfg["assignment"] = assignment_raw

    enabled_raw = str(os.environ.get("FULLMC_BATCH_ENABLED", "") or "").strip()
    if enabled_raw:
        batch_cfg["enabled"] = _parse_bool(enabled_raw)

    output_dir_raw = str(os.environ.get("FULLMC_OUTPUT_DIR", "") or "").strip()
    if output_dir_raw:
        resolved["output_dir"] = output_dir_raw

    resume_raw = str(os.environ.get("FULLMC_RESUME", "") or "").strip()
    if resume_raw:
        resolved["resume"] = _parse_bool(resume_raw)

    clear_raw = str(os.environ.get("FULLMC_CLEAR_EXISTING_RUNS", "") or "").strip()
    if clear_raw:
        resolved["clear_existing_run_dirs_when_no_resume"] = _parse_bool(clear_raw)

    max_runs_raw = str(os.environ.get("FULLMC_MAX_RUNS", "") or "").strip()
    if max_runs_raw:
        resolved["max_runs"] = _parse_int(max_runs_raw, name="FULLMC_MAX_RUNS")

    samples_raw = str(os.environ.get("FULLMC_SAMPLES", "") or "").strip()
    if samples_raw:
        resolved["samples"] = _parse_int(samples_raw, name="FULLMC_SAMPLES")

    ef_path_raw = str(os.environ.get("FULLMC_EF_INTENSITY_BASELINE_EMISSIONS_CSV", "") or "").strip()
    if ef_path_raw:
        resolved["ef_intensity_baseline_emissions_csv"] = ef_path_raw

    require_ef_raw = str(os.environ.get("FULLMC_REQUIRE_EF_CO2EQ_INTENSITY", "") or "").strip()
    if require_ef_raw:
        resolved["require_ef_co2eq_intensity"] = _parse_bool(require_ef_raw)

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
        if args.samples is not None:
            resolved["samples"] = int(args.samples)
        if args.ef_intensity_baseline_emissions_csv:
            resolved["ef_intensity_baseline_emissions_csv"] = str(args.ef_intensity_baseline_emissions_csv)
        if args.require_ef_co2eq_intensity is not None:
            resolved["require_ef_co2eq_intensity"] = bool(args.require_ef_co2eq_intensity)

    resolved["_batch_index_source"] = batch_index_source
    return resolved


def main() -> None:
    args = _build_arg_parser().parse_args()
    cfg = _apply_overrides(fullmc.CONFIG, args)
    fullmc.CONFIG.clear()
    fullmc.CONFIG.update(cfg)

    batch_cfg = fullmc.CONFIG.get("batch", {}) or {}
    print(
        "[S5_4_1_BATCH] "
        f"batch_index={batch_cfg.get('batch_index')} "
        f"total_batches={batch_cfg.get('total_batches')} "
        f"assignment={batch_cfg.get('assignment')} "
        f"enabled={batch_cfg.get('enabled')} "
        f"resume={fullmc.CONFIG.get('resume')} "
        f"clear_existing_runs={fullmc.CONFIG.get('clear_existing_run_dirs_when_no_resume')} "
        f"source={fullmc.CONFIG.get('_batch_index_source')}"
    )
    print(
        "[S5_4_1_BATCH] "
        f"slurm_job_name={os.environ.get('SLURM_JOB_NAME', '') or '<empty>'}"
    )
    if str(fullmc.CONFIG.get("output_dir", "") or "").strip():
        print(f"[S5_4_1_BATCH] output_dir={fullmc.CONFIG['output_dir']}")
    ef_path = fullmc._resolve_ef_intensity_emissions_path(fullmc.CONFIG)
    print(f"[S5_4_1_BATCH] ef_intensity_baseline_emissions_csv={ef_path or '<not found>'}")

    try:
        fullmc.main()
    finally:
        fullmc.CONFIG.pop("_batch_index_source", None)


if __name__ == "__main__":
    main()
