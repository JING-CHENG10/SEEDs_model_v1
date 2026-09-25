# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from config_paths import get_results_base
import S5_1_1_sensitivity_mc_variable_effect as vareffect
from S5_cost_summary_outputs import write_sensitivity_cost_summaries


CONFIG = {
    "output_dir": "",  # empty -> <NZF_OUTPUT_DIR>/MC_Sensitivity_Variable_Effect
    "batch": {
        "total_batches": 0,  # 0 -> auto-detect under batches/
        "batch_tags": [],
        "batches_subdir": "batches",
        "strict": True,
    },
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _batch_tag(batch_index: int, batch_count: int) -> str:
    return f"batch_{int(batch_index):02d}_of_{int(batch_count):02d}"


def _resolve_batch_tags(root_output_dir: Path, cfg: Dict[str, object]) -> Tuple[Path, List[str], bool]:
    batch_cfg = cfg.get("batch", {}) or {}
    batches_root = root_output_dir / str(batch_cfg.get("batches_subdir", "batches") or "batches")
    strict = bool(batch_cfg.get("strict", True))
    explicit_tags = [str(x).strip() for x in (batch_cfg.get("batch_tags") or []) if str(x).strip()]

    if explicit_tags:
        tags = explicit_tags
    else:
        total_batches = batch_cfg.get("total_batches", 0)
        if total_batches in (None, "", 0):
            if not batches_root.exists():
                raise FileNotFoundError(f"batch directory not found: {batches_root}")
            tags = sorted([p.name for p in batches_root.iterdir() if p.is_dir()])
        else:
            batch_count = int(total_batches)
            if batch_count <= 0:
                raise ValueError("batch.total_batches must be a positive integer, 0, or None.")
            tags = [_batch_tag(i, batch_count) for i in range(1, batch_count + 1)]
    if not tags:
        raise FileNotFoundError(f"No batch directories found under {batches_root}")
    return batches_root, tags, strict


def _read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[S5_1_3][WARN] failed to read {path}: {type(exc).__name__}: {exc}")
        return pd.DataFrame()


def _merge_frames(
    frames: Sequence[pd.DataFrame],
    *,
    dedup_cols: Sequence[str],
    sort_cols: Sequence[str],
) -> pd.DataFrame:
    valid_frames = [df for df in frames if df is not None and not df.empty]
    if not valid_frames:
        return pd.DataFrame()
    merged = pd.concat(valid_frames, ignore_index=True)
    keep_cols = [c for c in dedup_cols if c in merged.columns]
    if keep_cols:
        merged = merged.drop_duplicates(subset=keep_cols, keep="last")
    order_cols = [c for c in sort_cols if c in merged.columns]
    if order_cols:
        merged = merged.sort_values(order_cols, kind="stable").reset_index(drop=True)
    return merged


def _reject_conflicting_scenario_provenance(
    frames: Sequence[pd.DataFrame],
    *,
    label: str,
) -> None:
    valid = [frame for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not valid:
        return
    combined = pd.concat(valid, ignore_index=True)
    required = {"scenario_id", "run_id", "resume_fingerprint"}
    missing = sorted(required.difference(combined.columns))
    if missing:
        raise RuntimeError(f"{label} lacks run provenance columns: {missing}")
    duplicate_rows = combined[combined.duplicated("scenario_id", keep=False)]
    if duplicate_rows.empty:
        return
    conflicts = duplicate_rows.groupby("scenario_id", dropna=False)[
        ["run_id", "resume_fingerprint"]
    ].nunique(dropna=False)
    if bool((conflicts > 1).to_numpy().any()):
        raise RuntimeError(
            f"{label} contains duplicate scenario_id values from conflicting runs/draws"
        )


def _meta_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _max_numeric_in_frames(frames: Sequence[pd.DataFrame], column: str) -> float:
    series = [
        pd.to_numeric(df[column], errors="coerce")
        for df in frames
        if df is not None and not df.empty and column in df.columns
    ]
    if not series:
        return float("nan")
    merged = pd.concat(series, ignore_index=True).dropna()
    return float(merged.max()) if not merged.empty else float("nan")


def _infer_batch_meta_from_summary(
    *,
    tag: str,
    batch_samples: pd.DataFrame,
    batch_status: pd.DataFrame,
) -> pd.DataFrame:
    if batch_samples.empty and batch_status.empty:
        return pd.DataFrame()

    status_col = "status" if "status" in batch_status.columns else ""
    attempted_runs = int(len(batch_status)) if not batch_status.empty else int(len(batch_samples))
    if status_col:
        status_values = batch_status[status_col].astype(str).str.strip().str.lower()
        valid_from_status = int(status_values.isin({"valid", "ok", "resumed"}).sum())
        invalid_runs = int((~status_values.isin({"valid", "ok", "resumed"})).sum())
    else:
        valid_from_status = 0
        invalid_runs = 0

    valid_runs = max(int(len(batch_samples)), valid_from_status)
    if attempted_runs > 0 and invalid_runs == 0:
        invalid_runs = max(0, attempted_runs - valid_runs)

    sample_id_max = _max_numeric_in_frames([batch_samples, batch_status], "sample_id")
    samples_per_level = int(sample_id_max) if pd.notna(sample_id_max) else 0
    success_rate = (float(valid_runs) / float(attempted_runs)) if attempted_runs > 0 else float("nan")

    row = {
        "requested_tasks": 0,
        "assigned_tasks": attempted_runs,
        "valid_runs": valid_runs,
        "attempted_runs": attempted_runs,
        "invalid_runs": invalid_runs,
        "success_rate": success_rate,
        "samples_per_level": samples_per_level,
        "seed": int(vareffect.CONFIG["seed"]),
        "year": int(vareffect.CONFIG["year"]),
        "sampling_method": "lhs_antithetic",
        "sampling_scope": "element",
        "sampling_discrete_levels_enabled": True,
        "sampling_discrete_use_quantile_bounds": False,
        "sampling_mix_ratio": None,
        "sampling_shuffle": True,
        "sampling_scramble": True,
        "sampling_q_low": 0.0,
        "sampling_q_high": 1.0,
        "use_linear": True,
        "future_last_only": True,
        "use_regional": False,
        "pre_macc_e0": False,
        "use_fao_modules": True,
        "fast_emis_only": False,
        "market_gap_max_rate": float(
            (vareffect.CONFIG.get("variable_effect", {}) or {}).get("market_gap_max_rate", 0.05) or 0.05
        ),
        "mc_non_ef_mode": "shared",
        "mc_ef_mode": "shared",
        "ef_process_mode": "all",
        "batch_tag": tag,
        "meta_source": "inferred_from_summary",
    }
    return pd.DataFrame([row])


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge batched S5_1_1 variable-effect outputs.")
    parser.add_argument("--output-dir", type=str, default=str(CONFIG["output_dir"] or ""))
    parser.add_argument("--total-batches", type=int, default=int(CONFIG["batch"]["total_batches"] or 0))
    parser.add_argument("--strict", action="store_true", default=bool(CONFIG["batch"]["strict"]))
    parser.add_argument("--no-strict", action="store_false", dest="strict")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    cfg = {
        "output_dir": args.output_dir,
        "batch": {
            "total_batches": int(args.total_batches),
            "batch_tags": [],
            "batches_subdir": CONFIG["batch"]["batches_subdir"],
            "strict": bool(args.strict),
        },
    }

    root_output_dir = (
        Path(str(cfg.get("output_dir")).strip())
        if str(cfg.get("output_dir", "")).strip()
        else Path(get_results_base()) / "MC_Sensitivity_Variable_Effect"
    )
    batches_root, tags, strict = _resolve_batch_tags(root_output_dir, cfg)

    print(f"[S5_1_3] root_output_dir={root_output_dir}")
    print(f"[S5_1_3] batches_root={batches_root}")
    print(f"[S5_1_3] batch_tags={', '.join(tags)}")

    _ensure_dir(root_output_dir)

    sample_frames: List[pd.DataFrame] = []
    status_frames: List[pd.DataFrame] = []
    meta_frames: List[pd.DataFrame] = []

    for tag in tags:
        batch_dir = batches_root / tag
        summary_dir = batch_dir / "summary"
        meta_path = summary_dir / "run_meta.csv"

        batch_samples = _read_csv_safe(summary_dir / "samples.csv")
        if not batch_samples.empty:
            sample_frames.append(batch_samples)

        batch_status = _read_csv_safe(summary_dir / "run_status.csv")
        if not batch_status.empty:
            status_frames.append(batch_status)

        if not meta_path.exists():
            msg = f"[S5_1_3][WARN] missing batch meta: {meta_path}"
            meta_df = _infer_batch_meta_from_summary(
                tag=tag,
                batch_samples=batch_samples,
                batch_status=batch_status,
            )
            if meta_df.empty and strict:
                raise FileNotFoundError(msg)
            print(msg)
            if not meta_df.empty:
                print(f"[S5_1_3][WARN] inferred batch meta from summary outputs: {tag}")
                meta_frames.append(meta_df)
        else:
            meta_df = _read_csv_safe(meta_path)
            if not meta_df.empty:
                meta_df["batch_tag"] = tag
                meta_frames.append(meta_df)

        if meta_path.exists() and meta_df.empty:
            msg = f"[S5_1_3][WARN] empty batch meta: {meta_path}"
            inferred_meta = _infer_batch_meta_from_summary(
                tag=tag,
                batch_samples=batch_samples,
                batch_status=batch_status,
            )
            if inferred_meta.empty and strict:
                raise RuntimeError(msg)
            print(msg)
            if not inferred_meta.empty:
                print(f"[S5_1_3][WARN] inferred batch meta from summary outputs: {tag}")
                meta_frames.append(inferred_meta)

    _reject_conflicting_scenario_provenance(sample_frames, label="samples.csv")
    _reject_conflicting_scenario_provenance(status_frames, label="run_status.csv")

    merged_samples = _merge_frames(
        sample_frames,
        dedup_cols=["scenario_id"],
        sort_cols=["variable", "rate_value", "sample_id", "scenario_id"],
    )
    merged_status = _merge_frames(
        status_frames,
        dedup_cols=["scenario_id"],
        sort_cols=["variable", "rate_value", "sample_id", "scenario_id"],
    )
    merged_meta = _merge_frames(meta_frames, dedup_cols=["batch_tag"], sort_cols=["batch_tag"])
    market_gap_max_rate = float(
        (vareffect.CONFIG.get("variable_effect", {}) or {}).get("market_gap_max_rate", 0.05) or 0.05
    )
    if not merged_status.empty and {"scenario_id", "status"}.issubset(merged_status.columns):
        if "message" not in merged_status.columns:
            merged_status["message"] = ""
        status_norm = merged_status["status"].astype(str).str.strip().str.lower()
        valid_mask = status_norm.isin({"valid", "ok", "resumed"})
        for idx, row in merged_status.loc[valid_mask].iterrows():
            scenario_id = str(row.get("scenario_id", "") or "").strip()
            batch_tag = str(row.get("batch_tag", "") or "").strip()
            expected_run_id = str(row.get("run_id", "") or "").strip()
            expected_fingerprint = str(
                row.get("resume_fingerprint", "") or ""
            ).strip()
            if not scenario_id or not batch_tag or not expected_run_id or not expected_fingerprint:
                merged_status.at[idx, "status"] = "invalid_run_provenance"
                merged_status.at[idx, "message"] = (
                    "missing scenario_id/batch_tag/run_id/resume_fingerprint"
                )
                continue
            run_dir = batches_root / batch_tag / "runs" / scenario_id
            if not run_dir.exists():
                merged_status.at[idx, "status"] = "missing_run_directory"
                merged_status.at[idx, "message"] = f"missing run directory: {run_dir}"
                continue
            validation, summary_df, detail_df = vareffect._validated_resume_artifacts(
                run_dir,
                expected_scenario_id=scenario_id,
                expected_resume_fingerprint=expected_fingerprint,
            )
            if (
                not vareffect._resume_artifacts_ready(validation, summary_df, detail_df)
                or validation.run_id != expected_run_id
            ):
                merged_status.at[idx, "status"] = "invalid_run_provenance"
                merged_status.at[idx, "message"] = (
                    validation.reason
                    if not validation.allowed
                    else "status/fast artifact run identity mismatch"
                )
                continue
            fast_msg = vareffect._validate_nonluc_fast_emissions(
                run_dir,
                validation=validation,
            )
            if fast_msg:
                merged_status.at[idx, "status"] = "invalid_fast_emissions"
                merged_status.at[idx, "message"] = fast_msg
                continue
            gap_msg = vareffect._validate_market_balance_gap(
                run_dir,
                max_gap_rate=market_gap_max_rate,
                validation=validation,
            )
            if gap_msg:
                merged_status.at[idx, "status"] = "invalid_market_balance"
                merged_status.at[idx, "message"] = gap_msg
        status_norm = merged_status["status"].astype(str).str.strip().str.lower()
        if not merged_samples.empty:
            provenance_cols = ["scenario_id", "run_id", "resume_fingerprint"]
            missing_sample_cols = sorted(set(provenance_cols).difference(merged_samples.columns))
            if missing_sample_cols:
                raise RuntimeError(
                    f"merged samples lack run provenance columns: {missing_sample_cols}"
                )
            valid_keys = merged_status.loc[
                status_norm.isin({"valid", "ok", "resumed"}),
                provenance_cols,
            ].drop_duplicates()
            before = len(merged_samples)
            merged_samples = merged_samples.merge(
                valid_keys,
                on=provenance_cols,
                how="inner",
            ).reset_index(drop=True)
            removed = before - len(merged_samples)
            if removed:
                print(f"[S5_1_3] removed nonvalidated samples: {removed}")

    samples_path = root_output_dir / "samples.csv"
    status_path = root_output_dir / "run_status.csv"
    meta_path = root_output_dir / "run_meta.csv"
    merged_samples.to_csv(samples_path, index=False, encoding="utf-8-sig")
    merged_status.to_csv(status_path, index=False, encoding="utf-8-sig")
    write_sensitivity_cost_summaries(
        merged_status,
        output_dir=root_output_dir,
        run_search_root=root_output_dir,
    )

    if not merged_meta.empty and "meta_source" in merged_meta.columns:
        real_meta = merged_meta[merged_meta["meta_source"].astype(str) != "inferred_from_summary"]
    else:
        real_meta = merged_meta
    first_meta = (
        real_meta.iloc[0].to_dict()
        if not real_meta.empty
        else (merged_meta.iloc[0].to_dict() if not merged_meta.empty else {})
    )
    requested_series = _meta_numeric_series(merged_meta, "requested_tasks")
    assigned_series = _meta_numeric_series(merged_meta, "assigned_tasks")
    attempted_series = _meta_numeric_series(merged_meta, "attempted_runs")
    invalid_series = _meta_numeric_series(merged_meta, "invalid_runs")
    samples_per_level_series = _meta_numeric_series(merged_meta, "samples_per_level")
    requested_tasks = int(requested_series.dropna().max()) if not requested_series.dropna().empty else 0
    assigned_tasks = int(assigned_series.fillna(0).sum()) if not assigned_series.empty else 0
    attempted_runs = int(attempted_series.fillna(0).sum()) if not attempted_series.empty else 0
    invalid_runs = int(invalid_series.fillna(0).sum()) if not invalid_series.empty else 0
    samples_per_level = int(samples_per_level_series.dropna().max()) if not samples_per_level_series.dropna().empty else 0
    if requested_tasks <= 0:
        requested_tasks = int(
            pd.concat(
                [
                    merged_status["scenario_id"] if "scenario_id" in merged_status.columns else pd.Series(dtype=object),
                    merged_samples["scenario_id"] if "scenario_id" in merged_samples.columns else pd.Series(dtype=object),
                ],
                ignore_index=True,
            ).dropna().nunique()
        )
    if assigned_tasks <= 0:
        assigned_tasks = attempted_runs or requested_tasks
    if attempted_runs <= 0:
        attempted_runs = int(len(merged_status)) if not merged_status.empty else int(len(merged_samples))
    if invalid_runs <= 0 and attempted_runs > 0:
        invalid_runs = max(0, attempted_runs - int(len(merged_samples)))
    if samples_per_level <= 0:
        sample_id_max = _max_numeric_in_frames([merged_samples, merged_status], "sample_id")
        samples_per_level = int(sample_id_max) if pd.notna(sample_id_max) else 0
    sampling_cfg = {
        "method": first_meta.get("sampling_method", "lhs_antithetic"),
        "scope": first_meta.get("sampling_scope", "element"),
        "discrete_levels_enabled": first_meta.get("sampling_discrete_levels_enabled", True),
        "discrete_levels_use_quantile_bounds": first_meta.get("sampling_discrete_use_quantile_bounds", False),
        "mix_ratio": first_meta.get("sampling_mix_ratio"),
        "shuffle": first_meta.get("sampling_shuffle", True),
        "scramble": first_meta.get("sampling_scramble", True),
    }
    q_bounds = (
        float(first_meta.get("sampling_q_low", 0.0) or 0.0),
        float(first_meta.get("sampling_q_high", 1.0) or 1.0),
    )
    run_cfg = {
        "use_linear": first_meta.get("use_linear", True),
        "future_last_only": first_meta.get("future_last_only", True),
        "use_regional": first_meta.get("use_regional", False),
        "pre_macc_e0": first_meta.get("pre_macc_e0", False),
        "use_fao_modules": first_meta.get("use_fao_modules", True),
        "fast_emis_only": first_meta.get("fast_emis_only", False),
        "market_gap_max_rate": first_meta.get("market_gap_max_rate", market_gap_max_rate),
        "mc_non_ef_mode": first_meta.get("mc_non_ef_mode", "shared"),
        "mc_ef_mode": first_meta.get("mc_ef_mode", "shared"),
        "ef_process_mode": first_meta.get("ef_process_mode", "all"),
    }
    vareffect._write_variable_effect_run_meta(
        meta_path,
        requested_tasks=requested_tasks,
        assigned_tasks=assigned_tasks,
        valid_runs=int(len(merged_samples)),
        attempted_runs=attempted_runs,
        invalid_runs=invalid_runs,
        samples_per_level=samples_per_level,
        seed=int(float(first_meta.get("seed", vareffect.CONFIG["seed"]) or vareffect.CONFIG["seed"])) if first_meta else int(vareffect.CONFIG["seed"]),
        year=int(float(first_meta.get("year", vareffect.CONFIG["year"]) or vareffect.CONFIG["year"])) if first_meta else int(vareffect.CONFIG["year"]),
        sampling_cfg=sampling_cfg,
        q_bounds=q_bounds,
        run_cfg=run_cfg,
        extra_meta={
            "merge_source": "batches",
            "completed_batches": int(len(merged_meta)),
            "expected_batches": int(len(tags)),
        },
    )

    ve_cfg = vareffect.CONFIG.get("variable_effect", {}) or {}
    fig5_cfg = ve_cfg.get("output_fig5", {}) or {}
    if bool(fig5_cfg.get("enabled", False)) and not merged_samples.empty:
        targets = vareffect._parse_targets(ve_cfg.get("targets", []))
        vareffect._write_fig5_outputs(
            merged_samples.to_dict("records"),
            year=int(first_meta.get("year", vareffect.CONFIG["year"]) if first_meta else vareffect.CONFIG["year"]),
            targets=targets,
            write_targets=bool(fig5_cfg.get("write_targets", True)),
        )

    print(f"[DONE] merged samples: {samples_path}")
    print(f"[DONE] merged run_status: {status_path}")
    print(f"[DONE] merged run_meta: {meta_path}")


if __name__ == "__main__":
    main()
