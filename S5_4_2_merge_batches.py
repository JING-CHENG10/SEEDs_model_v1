# -*- coding: utf-8 -*-
"""
S5.4.1: lightweight merging of S5.4 batch Monte Carlo outputs.

Notes:
- Read existing batch CSVs only; do not call the main model or import heavy S4/S5_4 dependencies.
- Combine multiple parallel batch directories into a complete sample set.
- Default input directory structure:
  <output_dir>/batches/batch_01_of_05/
  <output_dir>/batches/batch_02_of_05/
  ...
- Default output directory:
  <output_dir>/merged/
"""
from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from config_paths import get_results_base
from S5_cost_summary_outputs import write_sensitivity_cost_summaries


CONFIG = {
    "output_dir": "",  # empty -> <NZF_OUTPUT_DIR>/MC_Full_Variables
    "status_csv": "mc_sample_status.csv",
    "draws_csv": "mc_draws_long.csv",
    "success_summary_csv": "mc_success_fast_summary.csv",
    "success_process_csv": "mc_success_global_process_co2eq.csv",
    "success_weighted_elements_csv": "mc_success_weighted_elements.csv",
    "require_ef_co2eq_intensity": True,
    "success_realized_ruminant_csv": "mc_success_realized_ruminant_share.csv",
    "success_land_balance_csv": "mc_success_crop_pasture_land_balance.csv",
    "batch": {
        "total_batches": 0,  # 0/None -> auto-detect existing batch dirs under batches/
        "batch_tags": [],  # optional explicit tags, e.g. ["batch_01_of_10", ...]
        "batches_subdir": "batches",
        "merged_subdir": "merged",
        "strict": True,  # True -> expected missing batch dirs raise error
    },
}

INVALID_TOTAL_CO2EQ_GT_VALUES = (1.264874,)

# Example configuration for merging 5 batches:
# Method 1: standard batch naming (recommended).
# CONFIG["output_dir"] = str(Path(get_results_base()) / "MC_Full_Variables")
# CONFIG["batch"] = {
# "total_batches": 10,
# "batch_tags": [],
# "batches_subdir": "batches",
# "merged_subdir": "merged",
# "strict": True,
# }

# The configuration above automatically searches for:
# batches/batch_01_of_10/
# batches/batch_02_of_10/
# ...
# batches/batch_10_of_10/

# Method 2: explicitly specify batch directory names.
# CONFIG["batch"] = {
# "total_batches": 0,
# "batch_tags": [
# "batch_01_of_10",
# "batch_02_of_10",
# "...",
# "batch_10_of_10",
# ],
# "batches_subdir": "batches",
# "merged_subdir": "merged",
# "strict": True,
# }

# With strict=True, any missing batch directory or result file raises an error.
# To merge the available results as far as possible, use:
# CONFIG["batch"]["strict"] = False


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _batch_tag(batch_index: int, batch_count: int) -> str:
    return f"batch_{int(batch_index):02d}_of_{int(batch_count):02d}"


def _resolve_batch_tags(root_output_dir: Path, cfg: Dict[str, object]) -> Tuple[Path, Path, List[str], bool]:
    batch_cfg = cfg.get("batch", {}) or {}
    batches_root = root_output_dir / str(batch_cfg.get("batches_subdir", "batches") or "batches")
    merged_dir = root_output_dir / str(batch_cfg.get("merged_subdir", "merged") or "merged")
    explicit_tags = [str(x).strip() for x in (batch_cfg.get("batch_tags") or []) if str(x).strip()]
    strict = bool(batch_cfg.get("strict", True))

    if explicit_tags:
        tags = explicit_tags
    else:
        total_batches = batch_cfg.get("total_batches", None)
        if total_batches in (None, "", 0):
            if not batches_root.exists():
                raise FileNotFoundError(f"批次目录不存在：{batches_root}")
            tags = sorted([p.name for p in batches_root.iterdir() if p.is_dir()])
        else:
            batch_count = int(total_batches)
            if batch_count <= 0:
                raise ValueError("batch.total_batches 必须为正整数、0/None 或显式 batch_tags。")
            tags = [_batch_tag(i, batch_count) for i in range(1, batch_count + 1)]
    if not tags:
        raise FileNotFoundError(f"No batch directories found under {batches_root}")
    return batches_root, merged_dir, tags, strict


def _read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[S5_4_2][WARN] failed to read {path}: {type(exc).__name__}: {exc}")
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                rows = list(csv.reader(f))
            if not rows:
                return pd.DataFrame()
            header = list(rows[0])
            data_rows = rows[1:]
            max_len = max((len(r) for r in rows), default=len(header))
            if max_len > len(header):
                extra_count = max_len - len(header)
                if path.name == "mc_sample_status.csv" and extra_count == 1:
                    header = header + ["afolu_emissions_gt_co2eq_yr"]
                else:
                    header = header + [f"extra_col_{i}" for i in range(1, extra_count + 1)]
            normalized_rows = [
                list(r[: len(header)]) + [""] * max(0, len(header) - len(r))
                for r in data_rows
            ]
            recovered = pd.DataFrame(normalized_rows, columns=header)
            recovered = _coerce_numeric_like_columns(recovered, recovered.columns)
            print(f"[S5_4_2][WARN] lenient CSV parser recovered {path} rows={len(recovered)}")
            return recovered
        except Exception as fallback_exc:
            print(
                f"[S5_4_2][WARN] lenient parser also failed for {path}: "
                f"{type(fallback_exc).__name__}: {fallback_exc}"
            )
            return pd.DataFrame()


def _coerce_numeric_like_series(series: pd.Series) -> pd.Series:
    if series.empty or pd.api.types.is_numeric_dtype(series):
        return series

    text = series.astype(str).str.strip()
    missing_mask = series.isna() | text.isin(["", "nan", "None", "<NA>"])
    non_missing = text[~missing_mask]
    if non_missing.empty:
        return series

    numeric_non_missing = pd.to_numeric(non_missing, errors="coerce")
    if not numeric_non_missing.notna().all():
        return series

    numeric_full = pd.to_numeric(text.where(~missing_mask, pd.NA), errors="coerce")
    if ((numeric_non_missing % 1) == 0).all():
        return numeric_full.astype("Int64")
    return numeric_full


def _coerce_numeric_like_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = _coerce_numeric_like_series(df[col])
    return df


def _sentinel_mask(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    rounded = vals.round(6)
    bad_vals = [round(float(x), 6) for x in INVALID_TOTAL_CO2EQ_GT_VALUES]
    return rounded.isin(bad_vals)


def _filter_invalid_emission_sentinel(df: pd.DataFrame, *, file_name: str) -> pd.DataFrame:
    sentinel_cols = [
        c
        for c in ("total_co2eq_gt", "afolu_emissions_gt_co2eq_yr")
        if c in df.columns
    ]
    if not sentinel_cols:
        return df
    bad = pd.Series(False, index=df.index)
    for col in sentinel_cols:
        bad = bad | _sentinel_mask(df[col])
    if not bool(bad.any()):
        return df
    before = int(len(df))
    out = df.loc[~bad].copy()
    print(
        f"[S5_4_2][FILTER] {file_name}: dropped {before - len(out)} rows "
        f"with invalid total_co2eq_gt sentinel {INVALID_TOTAL_CO2EQ_GT_VALUES}"
    )
    return out


def _validate_ef_intensity_present(df: pd.DataFrame, *, file_name: str) -> None:
    if "kind" not in df.columns:
        return
    ef_mask = df["kind"].astype(str).str.strip().str.lower().eq("emission_factor")
    ef_rows = int(ef_mask.sum())
    if ef_rows == 0:
        return
    col = "weighted_co2eq_intensity_sample_kg_per_kcal"
    if col not in df.columns:
        raise RuntimeError(
            f"{file_name}: emission_factor rows exist ({ef_rows}), but {col} is missing. "
            "Rerun S5_4_1 with the updated code."
        )
    nonnull = int(pd.to_numeric(df.loc[ef_mask, col], errors="coerce").notna().sum())
    if nonnull == 0:
        batch_hint = ""
        if "batch_tag" in df.columns:
            tags = df.loc[ef_mask, "batch_tag"].dropna().astype(str).unique()
            if len(tags):
                batch_hint = f" Example batch_tag={tags[0]}."
        raise RuntimeError(
            f"{file_name}: emission_factor rows exist ({ef_rows}), but {col} is empty for all of them."
            f"{batch_hint} Check S5_4_1 logs for 'EF CO2eq/kcal baseline intensities=0' or "
            "'EF CO2eq/kcal baseline emissions file not found'."
        )
    print(f"[S5_4_2][CHECK] {file_name}: EF intensity rows {nonnull}/{ef_rows}")


def _merge_one_csv(
    *,
    file_name: str,
    dedup_cols: Sequence[str],
    sort_cols: Sequence[str],
    batches_root: Path,
    merged_dir: Path,
    tags: Sequence[str],
    strict: bool,
    valid_success_keys: Optional[pd.DataFrame] = None,
    require_ef_intensity: bool = False,
) -> int:
    frames: List[pd.DataFrame] = []
    missing_tags: List[str] = []
    for tag in tags:
        batch_file = batches_root / tag / file_name
        if not batch_file.exists():
            missing_tags.append(tag)
            continue
        df = _read_csv_safe(batch_file)
        if not df.empty:
            frames.append(df)

    if missing_tags:
        msg = f"[S5_4_2][WARN] {file_name} missing in batches: {', '.join(missing_tags)}"
        if strict:
            raise FileNotFoundError(msg)
        print(msg)

    if not frames:
        print(f"[S5_4_2][WARN] no rows found for {file_name}")
        return 0

    merged = pd.concat(frames, ignore_index=True)
    merged = _filter_invalid_emission_sentinel(merged, file_name=file_name)
    if valid_success_keys is not None:
        base_filter_cols = [
            "scenario_id",
            "sample_id",
            "experiment_fingerprint",
            "resume_fingerprint",
        ]
        missing_filter_cols = [col for col in base_filter_cols if col not in merged.columns]
        if missing_filter_cols:
            raise RuntimeError(
                f"{file_name}: cannot apply success filter because "
                f"provenance columns are missing: {missing_filter_cols}"
            )
        filter_cols = list(base_filter_cols)
        if "run_id" in merged.columns:
            filter_cols.append("run_id")
        before_filter = int(len(merged))
        merged["scenario_id"] = merged["scenario_id"].astype(str)
        merged["sample_id"] = pd.to_numeric(merged["sample_id"], errors="coerce").astype("Int64")
        merged = merged.dropna(subset=["sample_id"]).copy()
        merged["sample_id"] = merged["sample_id"].astype(int)
        keys = valid_success_keys[filter_cols].copy()
        keys["scenario_id"] = keys["scenario_id"].astype(str)
        keys["sample_id"] = pd.to_numeric(keys["sample_id"], errors="coerce").astype("Int64")
        keys = keys.dropna(subset=["sample_id"]).copy()
        keys["sample_id"] = keys["sample_id"].astype(int)
        keys = keys.drop_duplicates()
        merged = merged.merge(keys, on=filter_cols, how="inner")
        print(
            f"[S5_4_2][FILTER] {file_name}: kept {len(merged)}/{before_filter} rows "
            "with OPTIMAL success status"
        )
    keep_cols = [c for c in dedup_cols if c in merged.columns]
    keep_sort = [c for c in sort_cols if c in merged.columns]
    merged = _coerce_numeric_like_columns(merged, list(dict.fromkeys([*keep_cols, *keep_sort])))
    if keep_cols:
        duplicate_rows = merged[merged.duplicated(subset=keep_cols, keep=False)]
        provenance_cols = [
            col
            for col in ("run_id", "experiment_fingerprint", "resume_fingerprint")
            if col in duplicate_rows.columns
        ]
        if not duplicate_rows.empty and provenance_cols:
            conflicts = duplicate_rows.groupby(keep_cols, dropna=False)[provenance_cols].nunique(
                dropna=False
            )
            if bool((conflicts > 1).to_numpy().any()):
                raise RuntimeError(
                    f"{file_name}: duplicate business keys have conflicting run provenance"
                )
        merged = merged.drop_duplicates(subset=keep_cols, keep="last")
    if keep_sort:
        merged = merged.sort_values(keep_sort).reset_index(drop=True)
    if require_ef_intensity:
        _validate_ef_intensity_present(merged, file_name=file_name)

    out_path = merged_dir / file_name
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[S5_4_2][MERGED] {out_path} rows={len(merged)}")
    return int(len(merged))


def _load_valid_success_keys(status_path: Path) -> pd.DataFrame:
    key_columns = [
        "scenario_id",
        "sample_id",
        "run_id",
        "experiment_fingerprint",
        "resume_fingerprint",
    ]
    if not status_path.exists():
        print(f"[S5_4_2][WARN] status file not found for success filtering: {status_path}")
        return pd.DataFrame(columns=key_columns)
    status = pd.read_csv(status_path)
    missing_keys = [col for col in key_columns if col not in status.columns]
    if missing_keys:
        print(
            f"[S5_4_2][WARN] status file missing provenance columns "
            f"{missing_keys}: {status_path}"
        )
        return pd.DataFrame(columns=key_columns)

    run_status = status.get("run_status", pd.Series("", index=status.index)).astype(str).str.lower()
    run_ok = run_status.isin({"ok", "resumed"})
    model_code = pd.to_numeric(status.get("model_status_code", pd.Series(pd.NA, index=status.index)), errors="coerce")
    model_text = status.get("model_status_text", pd.Series("", index=status.index)).astype(str).str.strip().str.upper()
    model_ok = model_code.eq(2) | model_text.eq("OPTIMAL")
    sentinel_ok = pd.Series(True, index=status.index)
    for col in ("afolu_emissions_gt_co2eq_yr", "total_co2eq_gt"):
        if col in status.columns:
            sentinel_ok = sentinel_ok & ~_sentinel_mask(status[col])
    # Missing solver evidence is never equivalent to OPTIMAL.  Old status CSVs
    # therefore yield no reusable success keys instead of passing through.
    mask = run_ok & model_ok & sentinel_ok

    keys = status.loc[mask, key_columns].copy()
    keys["scenario_id"] = keys["scenario_id"].astype(str)
    keys["sample_id"] = pd.to_numeric(keys["sample_id"], errors="coerce").astype("Int64")
    keys = keys.dropna(subset=["sample_id"]).copy()
    keys["sample_id"] = keys["sample_id"].astype(int)
    for col in ("run_id", "experiment_fingerprint", "resume_fingerprint"):
        keys[col] = keys[col].fillna("").astype(str).str.strip()
    keys = keys[
        keys["run_id"].ne("")
        & keys["experiment_fingerprint"].ne("")
        & keys["resume_fingerprint"].ne("")
    ].copy()
    keys = keys.drop_duplicates().reset_index(drop=True)
    print(
        f"[S5_4_2][FILTER] valid OPTIMAL success samples: {len(keys)}/"
        f"{status['sample_id'].nunique() if 'sample_id' in status.columns else len(status)}"
    )
    return keys


def _validate_merged_experiment_identity(status_path: Path) -> str:
    status = pd.read_csv(status_path)
    required = {"experiment_fingerprint", "batch_count", "batch_tag"}
    missing = sorted(required.difference(status.columns))
    if missing:
        raise RuntimeError(
            f"merged status lacks experiment/batch provenance columns: {missing}"
        )
    fingerprints = set(
        status["experiment_fingerprint"].dropna().astype(str).str.strip()
    )
    fingerprints.discard("")
    if len(fingerprints) != 1:
        raise RuntimeError(
            "merged batches do not belong to exactly one experiment fingerprint"
        )
    batch_counts = set(pd.to_numeric(status["batch_count"], errors="coerce").dropna().astype(int))
    if len(batch_counts) != 1:
        raise RuntimeError("merged batches disagree on batch_count")
    return next(iter(fingerprints))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge batched S5_4 full-variable Monte Carlo outputs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="S5_4 root output directory. Default: <NZF_OUTPUT_DIR>/MC_Full_Variables.",
    )
    parser.add_argument(
        "--total-batches",
        type=int,
        default=None,
        help="Expected batch count. Default 0 auto-detects batch directories.",
    )
    parser.add_argument(
        "--batch-tags",
        default=None,
        help="Optional comma-separated explicit batch directory names.",
    )
    parser.add_argument("--batches-subdir", default=None)
    parser.add_argument("--merged-subdir", default=None)
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Require every expected batch and required CSV (default: true).",
    )
    parser.add_argument(
        "--require-ef-co2eq-intensity",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Require the production-EF CO2eq intensity fields used by Figure 4f.",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> Dict[str, object]:
    cfg = copy.deepcopy(CONFIG)
    batch_cfg = cfg.setdefault("batch", {})
    if not isinstance(batch_cfg, dict):
        raise TypeError("CONFIG['batch'] must be a dictionary")
    if args.output_dir is not None:
        cfg["output_dir"] = str(args.output_dir)
    if args.total_batches is not None:
        if int(args.total_batches) < 0:
            raise ValueError("--total-batches must be >= 0")
        batch_cfg["total_batches"] = int(args.total_batches)
    if args.batch_tags is not None:
        batch_cfg["batch_tags"] = [
            part.strip() for part in str(args.batch_tags).split(",") if part.strip()
        ]
    if args.batches_subdir is not None:
        batch_cfg["batches_subdir"] = str(args.batches_subdir)
    if args.merged_subdir is not None:
        batch_cfg["merged_subdir"] = str(args.merged_subdir)
    if args.strict is not None:
        batch_cfg["strict"] = bool(args.strict)
    if args.require_ef_co2eq_intensity is not None:
        cfg["require_ef_co2eq_intensity"] = bool(args.require_ef_co2eq_intensity)
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    cfg = _config_from_args(args)
    root_output_dir = (
        Path(str(cfg.get("output_dir")).strip())
        if str(cfg.get("output_dir", "")).strip()
        else Path(get_results_base()) / "MC_Full_Variables"
    )
    batches_root, merged_dir, tags, strict = _resolve_batch_tags(root_output_dir, cfg)
    _ensure_dir(merged_dir)

    print(f"[S5_4_2] root_output_dir={root_output_dir}")
    print(f"[S5_4_2] batches_root={batches_root}")
    print(f"[S5_4_2] merged_dir={merged_dir}")
    print(f"[S5_4_2] batch_tags={', '.join(tags)}")

    status_file = str(cfg.get("status_csv") or "mc_sample_status.csv")
    status_spec = (status_file, ["scenario_id"], ["sample_id"])
    success_file_names = {
        str(cfg.get("draws_csv") or "mc_draws_long.csv"),
        str(cfg.get("success_summary_csv") or "mc_success_fast_summary.csv"),
        str(cfg.get("success_process_csv") or "mc_success_global_process_co2eq.csv"),
        str(cfg.get("success_weighted_elements_csv") or "mc_success_weighted_elements.csv"),
        str(cfg.get("success_realized_ruminant_csv") or "mc_success_realized_ruminant_share.csv"),
        str(cfg.get("success_land_balance_csv") or "mc_success_crop_pasture_land_balance.csv"),
    }
    optional_file_names = {
        str(cfg.get("success_realized_ruminant_csv") or "mc_success_realized_ruminant_share.csv"),
        str(cfg.get("success_land_balance_csv") or "mc_success_crop_pasture_land_balance.csv"),
    }

    merge_specs = [
        status_spec,
        (
            str(cfg.get("draws_csv") or "mc_draws_long.csv"),
            ["scenario_id", "spec_row_id"],
            ["sample_id", "spec_row_id"],
        ),
        (
            str(cfg.get("success_summary_csv") or "mc_success_fast_summary.csv"),
            ["scenario_id", "year"],
            ["sample_id", "year"],
        ),
        (
            str(cfg.get("success_process_csv") or "mc_success_global_process_co2eq.csv"),
            ["scenario_id", "year", "source_module", "Process"],
            ["sample_id", "year", "source_module", "Process"],
        ),
        (
            str(cfg.get("success_weighted_elements_csv") or "mc_success_weighted_elements.csv"),
            ["scenario_id", "sample_id", "spec_row_id", "value_unit"],
            ["sample_id", "spec_row_id"],
        ),
        (
            str(cfg.get("success_realized_ruminant_csv") or "mc_success_realized_ruminant_share.csv"),
            ["scenario_id", "sample_id", "year"],
            ["sample_id", "year"],
        ),
        (
            str(cfg.get("success_land_balance_csv") or "mc_success_crop_pasture_land_balance.csv"),
            ["scenario_id", "sample_id", "year", "source"],
            ["sample_id", "source", "year"],
        ),
    ]

    merged_counts: Dict[str, int] = {}
    status_file_name, status_dedup_cols, status_sort_cols = status_spec
    print(f"[S5_4_2][START] merging {status_file_name}")
    merged_counts[status_file_name] = _merge_one_csv(
        file_name=status_file_name,
        dedup_cols=status_dedup_cols,
        sort_cols=status_sort_cols,
        batches_root=batches_root,
        merged_dir=merged_dir,
        tags=tags,
        strict=strict,
    )
    merged_experiment_fingerprint = _validate_merged_experiment_identity(
        merged_dir / status_file_name
    )
    print(
        "[S5_4_2][CHECK] experiment_fingerprint="
        f"{merged_experiment_fingerprint[:16]}..."
    )
    valid_success_keys = _load_valid_success_keys(merged_dir / status_file_name)

    for file_name, dedup_cols, sort_cols in merge_specs[1:]:
        print(f"[S5_4_2][START] merging {file_name}")
        merged_counts[file_name] = _merge_one_csv(
            file_name=file_name,
            dedup_cols=dedup_cols,
            sort_cols=sort_cols,
            batches_root=batches_root,
            merged_dir=merged_dir,
            tags=tags,
            strict=bool(strict and file_name not in optional_file_names),
            valid_success_keys=valid_success_keys if file_name in success_file_names else None,
            require_ef_intensity=bool(
                cfg.get("require_ef_co2eq_intensity", False)
                and file_name == str(cfg.get("success_weighted_elements_csv") or "mc_success_weighted_elements.csv")
            ),
        )

    merged_status_df = pd.read_csv(merged_dir / status_file_name)
    write_sensitivity_cost_summaries(
        merged_status_df,
        output_dir=merged_dir,
        run_search_root=root_output_dir,
    )

    print("[S5_4_2] done")
    for file_name, row_count in merged_counts.items():
        print(f"[S5_4_2] {file_name}: rows={row_count}")


if __name__ == "__main__":
    main()
