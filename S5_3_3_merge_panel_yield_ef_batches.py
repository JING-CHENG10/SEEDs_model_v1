"""
Merge S5_3 batch CSV outputs only.

This script intentionally does not copy per-run directories such as runs/ or
runs_diag/. Batch-level summary CSV files are the source of truth for the merged
outputs; keeping run directories in place avoids large duplicate result trees.
"""
from __future__ import annotations

import argparse
import gc
import os
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

import S5_3_1_sensitivity_panel_yield_ef as panel
from S5_cost_summary_outputs import write_sensitivity_cost_summaries

INVALID_TOTAL_CO2EQ_GT_VALUES = (1.264874,)


CONFIG = {
    "output_dir": "",  # empty -> <NZF_OUTPUT_DIR>/Panel_Yield_EF
    "resubmit_command": "bash submit_sbatch_S5_3_1_panel_yield_ef.sh",
    "data_summary_command": "python S5_3_2_panel_data_summary.py",
    "merge_chunksize": 50000,
    "batch_tag_log_limit": 12,
    "stream_detail_dedup": False,
    "batch": {
        "total_batches": 0,  # 0 -> auto-detect existing batch dirs under batches/
        "batch_tags": [],
        "batches_subdir": "batches",
        "strict": True,
    },
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _batch_tag(batch_index: int, batch_count: int) -> str:
    return f"batch_{int(batch_index):02d}_of_{int(batch_count):02d}"


def _batch_sort_key(tag: str) -> Tuple[int, int, str]:
    match = re.fullmatch(r"batch_(\d+)_of_(\d+)", str(tag).strip())
    if not match:
        return (10**9, 10**9, str(tag))
    return (int(match.group(2)), int(match.group(1)), str(tag))


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
            tags = sorted([p.name for p in batches_root.iterdir() if p.is_dir()], key=_batch_sort_key)
        else:
            batch_count = int(total_batches)
            if batch_count <= 0:
                raise ValueError("batch.total_batches must be a positive integer, 0, or None.")
            tags = [_batch_tag(i, batch_count) for i in range(1, batch_count + 1)]
    if not tags:
        raise FileNotFoundError(f"No batch directories found under {batches_root}")
    if strict:
        # Directory names encode the expected batch count, even when a whole
        # directory is absent. Never infer completion from existing dirs alone.
        existing = [p.name for p in batches_root.iterdir() if p.is_dir()] if batches_root.exists() else []
        parsed = []
        for tag in set(tags + existing):
            match = re.fullmatch(r"batch_(\d+)_of_(\d+)", tag)
            if not match:
                raise ValueError(f"Invalid batch directory/tag: {tag}")
            index, count = map(int, match.groups())
            if not 1 <= index <= count or tag != _batch_tag(index, count):
                raise ValueError(f"Invalid batch index or noncanonical tag: {tag}")
            parsed.append(count)
        if len(set(parsed)) != 1:
            raise ValueError(f"Mixed batch counts under {batches_root}: {sorted(set(parsed))}")
        count = parsed[0]
        expected = [_batch_tag(i, count) for i in range(1, count + 1)]
        if explicit_tags and set(tags) != set(expected):
            raise ValueError("Strict merge requires the complete batch tag set.")
        tags = expected
    return batches_root, tags, strict


def _read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[S5_3_3][WARN] failed to read {path}: {type(exc).__name__}: {exc}")
        return pd.DataFrame()


def _sentinel_mask(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    bad_vals = [round(float(x), 6) for x in INVALID_TOTAL_CO2EQ_GT_VALUES]
    return vals.round(6).isin(bad_vals)


def _drop_invalid_emission_sentinel(df: pd.DataFrame) -> pd.DataFrame:
    sentinel_cols = [
        c
        for c in ("afolu_emissions_gt_co2eq_yr", "total_co2eq_gt")
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
        f"[S5_3_3][FILTER] dropped {before - len(out)} rows with invalid "
        f"emission sentinel {INVALID_TOTAL_CO2EQ_GT_VALUES}"
    )
    return out


def _normalise_ruminant_cap_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    new_col = panel.RUMINANT_CAP_COL
    old_col = panel.DEPRECATED_RUMINANT_CAP_COL
    if new_col not in out.columns and old_col in out.columns:
        out = out.rename(columns={old_col: new_col})
    elif new_col in out.columns and old_col in out.columns:
        out[new_col] = out[new_col].where(out[new_col].notna(), out[old_col])
        out = out.drop(columns=[old_col])
    if "ruminant_intake_multiplier" in out.columns:
        out = out.drop(columns=["ruminant_intake_multiplier"])
    return out


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
    merged = _drop_invalid_emission_sentinel(merged)
    keep_cols = [c for c in dedup_cols if c in merged.columns]
    if keep_cols:
        merged = merged.drop_duplicates(subset=keep_cols, keep="last")
    order_cols = [c for c in sort_cols if c in merged.columns]
    if order_cols:
        merged = merged.sort_values(order_cols, kind="stable").reset_index(drop=True)
    return merged


def _meta_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            return max(sum(1 for _ in handle) - 1, 0)
    except Exception:
        return 0


def _expected_rows_for_batch(tag: str, total_runs: int) -> int:
    match = re.fullmatch(r"batch_(\d+)_of_(\d+)", str(tag).strip())
    if not match:
        return 0
    batch_index = int(match.group(1))
    batch_count = int(match.group(2))
    if batch_count <= 0 or batch_index <= 0 or batch_index > batch_count:
        return 0
    return sum(1 for idx in range(total_runs) if idx % batch_count == batch_index - 1)


def _format_missing_batch_lines(items: Sequence[Tuple[str, Path, int, int]]) -> str:
    lines = []
    for tag, path, rows, expected in items[:20]:
        expected_text = f"/{expected}" if expected else ""
        lines.append(f"  - {tag}: rows={rows}{expected_text}, path={path}")
    if len(items) > 20:
        lines.append(f"  ... {len(items) - 20} more")
    return "\n".join(lines)


def _format_batch_tag_summary(tags: Sequence[str], *, limit: int) -> str:
    if len(tags) <= limit:
        return ", ".join(tags)
    head_n = max(1, limit // 2)
    tail_n = max(1, limit - head_n)
    head = ", ".join(tags[:head_n])
    tail = ", ".join(tags[-tail_n:])
    return f"{head}, ... , {tail} (total={len(tags)})"


def _iter_csv_chunks_safe(path: Path, *, chunksize: int, strict: bool = False):
    if not path.exists():
        if strict:
            raise FileNotFoundError(path)
        return
    try:
        if chunksize > 0:
            reader = pd.read_csv(path, chunksize=chunksize)
            for chunk in reader:
                yield chunk
        else:
            yield pd.read_csv(path)
    except Exception as exc:
        if strict:
            raise
        print(f"[S5_3_3][WARN] failed to stream {path}: {type(exc).__name__}: {exc}")


def _validated_result_ids(df: pd.DataFrame) -> List[str]:
    """A completed attempt has one nonempty ID and a terminal status."""
    required = {"scenario_id", "run_status"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing result columns: {sorted(required - set(df.columns))}")
    ids = df["scenario_id"].astype("string").str.strip()
    if ids.isna().any() or ids.eq("").any():
        raise ValueError("Result contains an empty scenario_id.")
    statuses = df["run_status"].astype("string").str.strip().str.lower()
    terminal = {"ok", "resumed", "infeasible", "nonoptimal", "failed",
                "invalid_fast_emissions", "invalid_market_balance",
                "invalid_forest_target", "missing_fast_summary"}
    if not statuses.isin(terminal).all():
        raise ValueError("Result contains a missing, unknown or unfinished run_status.")
    return ids.tolist()


def _deduplicate_chunk_by_hash(
    df: pd.DataFrame,
    *,
    dedup_cols: Sequence[str],
    seen_hashes: set,
) -> Tuple[pd.DataFrame, int]:
    keep_cols = [c for c in dedup_cols if c in df.columns]
    if not keep_cols or df.empty:
        return df, 0

    key_frame = df.loc[:, keep_cols].astype("string").fillna("<NA>")
    hashes = pd.util.hash_pandas_object(key_frame, index=False).to_numpy(dtype="uint64")
    keep_mask: List[bool] = []
    skipped = 0
    for raw_hash in hashes:
        key = int(raw_hash)
        if key in seen_hashes:
            keep_mask.append(False)
            skipped += 1
        else:
            seen_hashes.add(key)
            keep_mask.append(True)
    if not skipped:
        return df, 0
    return df.loc[keep_mask].copy(), skipped


def _stream_merge_csv_files(
    paths: Sequence[Path],
    out_path: Path,
    *,
    dedup_cols: Sequence[str],
    sort_cols: Sequence[str],
    chunksize: int,
    normalise_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    count_ok_column: Optional[str] = None,
    expected_scenario_ids: Optional[set] = None,
) -> Dict[str, int]:
    """Merge CSVs without holding all batch data in memory.

    The old implementation concatenated every batch DataFrame and then sorted.
    That is exact but memory-heavy for global-emissions detail files. This
    streamed merge preserves filtering/normalisation and duplicate-key removal,
    but writes rows in batch order instead of doing a global in-memory sort.
    """
    _ensure_dir(out_path.parent)
    tmp_path = out_path.with_name(f".{out_path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    if sort_cols:
        print(
            f"[S5_3_3][STREAM] {out_path.name}: global sort skipped in low-memory mode; "
            "rows are written in batch/chunk order."
        )

    rows_written = 0
    ok_count = 0
    duplicate_skipped = 0
    chunks_seen = 0
    header_written = False
    first_columns: Optional[List[str]] = None
    seen_hashes: set = set()
    seen_ids: set = set()

    for path in paths:
        for chunk in _iter_csv_chunks_safe(path, chunksize=chunksize,
                                           strict=expected_scenario_ids is not None):
            chunks_seen += 1
            if expected_scenario_ids is not None:
                ids = _validated_result_ids(chunk)
                repeated = seen_ids.intersection(ids)
                if repeated or len(ids) != len(set(ids)):
                    raise ValueError(f"Duplicate scenario_id in strict merge: {path}")
                if set(ids) - expected_scenario_ids:
                    raise ValueError(f"Unexpected scenario_id in strict merge: {path}")
            if first_columns is None:
                first_columns = list(chunk.columns)
            if normalise_fn is not None and not chunk.empty:
                chunk = normalise_fn(chunk)
                if first_columns is None:
                    first_columns = list(chunk.columns)
            if chunk.empty:
                continue

            chunk = _drop_invalid_emission_sentinel(chunk)
            if expected_scenario_ids is not None:
                seen_ids.update(_validated_result_ids(chunk))
            chunk, skipped = _deduplicate_chunk_by_hash(
                chunk,
                dedup_cols=dedup_cols,
                seen_hashes=seen_hashes,
            )
            duplicate_skipped += skipped
            if chunk.empty:
                continue

            if count_ok_column and count_ok_column in chunk.columns:
                ok_count += int(
                    chunk[count_ok_column]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .eq("ok")
                    .sum()
                )

            encoding = "utf-8-sig" if not header_written else "utf-8"
            chunk.to_csv(
                tmp_path,
                mode="a",
                header=not header_written,
                index=False,
                encoding=encoding,
            )
            header_written = True
            rows_written += int(len(chunk))
            del chunk
        gc.collect()

    if not header_written:
        pd.DataFrame(columns=first_columns or []).to_csv(tmp_path, index=False, encoding="utf-8-sig")

    if expected_scenario_ids is not None and seen_ids != expected_scenario_ids:
        missing = sorted(expected_scenario_ids - seen_ids)
        raise ValueError(f"Incomplete scenario coverage: missing {len(missing)} IDs; examples={missing[:5]}")
    os.replace(tmp_path, out_path)
    print(
        f"[S5_3_3][STREAM] wrote {out_path} rows={rows_written} "
        f"chunks={chunks_seen} duplicate_keys_skipped={duplicate_skipped}"
    )
    return {
        "rows": int(rows_written),
        "ok_count": int(ok_count),
        "duplicate_keys_skipped": int(duplicate_skipped),
        "chunks": int(chunks_seen),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge batched S5_3_1 panel outputs.")
    parser.add_argument("--output-dir", type=str, default=str(CONFIG["output_dir"] or ""))
    parser.add_argument("--total-batches", type=int, default=int(CONFIG["batch"]["total_batches"] or 0))
    parser.add_argument("--strict", action="store_true", default=bool(CONFIG["batch"]["strict"]))
    parser.add_argument("--no-strict", action="store_false", dest="strict")
    parser.add_argument(
        "--merge-chunksize",
        type=int,
        default=int(os.environ.get("PANEL_MERGE_CHUNKSIZE") or CONFIG.get("merge_chunksize") or 50000),
        help="Rows per CSV chunk for low-memory batch merging. Use 0 to read each file in one chunk.",
    )
    parser.add_argument(
        "--dedup-detail",
        action="store_true",
        default=bool(CONFIG.get("stream_detail_dedup", False)),
        help=(
            "Deduplicate global-emissions detail rows while streaming. "
            "Disabled by default to avoid a large in-memory key set; batch outputs should be disjoint."
        ),
    )
    parser.add_argument(
        "--no-merge-runs",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-merge-diag-runs",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    output_dir_raw = (
        str(args.output_dir or "").strip()
        or str(os.environ.get("PANEL_OUTPUT_DIR", "") or "").strip()
    )
    root_output_dir = panel._resolve_panel_output_root(output_dir_raw)
    panel._sync_panel_output_environment(root_output_dir)
    cfg = {
        "resubmit_command": CONFIG.get("resubmit_command", "bash submit_sbatch_S5_3_1_panel_yield_ef.sh"),
        "data_summary_command": CONFIG.get("data_summary_command", "python S5_3_2_panel_data_summary.py"),
        "batch": {
            "total_batches": int(args.total_batches),
            "batch_tags": [],
            "batches_subdir": CONFIG["batch"]["batches_subdir"],
            "strict": bool(args.strict),
        }
    }
    batches_root, tags, strict = _resolve_batch_tags(root_output_dir, cfg)
    _ensure_dir(root_output_dir)
    resubmit_command = str(cfg.get("resubmit_command") or "bash submit_sbatch_S5_3_1_panel_yield_ef.sh")
    data_summary_command = str(cfg.get("data_summary_command") or "python S5_3_2_panel_data_summary.py")

    print(f"[S5_3_3] root_output_dir={root_output_dir}")
    print(f"[S5_3_3] NZF_OUTPUT_DIR={os.environ.get('NZF_OUTPUT_DIR', '')}")
    print(f"[S5_3_3] PANEL_OUTPUT_DIR={os.environ.get('PANEL_OUTPUT_DIR', '')}")
    print(f"[S5_3_3] batches_root={batches_root}")
    print(
        "[S5_3_3] batch_tags="
        f"{_format_batch_tag_summary(tags, limit=int(CONFIG.get('batch_tag_log_limit', 12) or 12))}"
    )
    merge_chunksize = max(0, int(args.merge_chunksize or 0))
    print(f"[S5_3_3] low-memory stream merge chunksize={merge_chunksize or 'whole-file'}")
    print(f"[S5_3_3] detail row deduplication={'enabled' if bool(args.dedup_detail) else 'disabled'}")

    results_name = str(panel.CONFIG.get("results_csv") or "figure_panel_dataset_long.csv")
    detail_name = str(panel.CONFIG.get("global_emissions_detail_csv") or "figure_panel_global_emissions_detail_long.csv")
    try:
        tasks = (
            panel._build_panel_tasks(
                forest_values=[float(v) for v in panel.CONFIG.get("forest_area_change_pct_values") or []],
                ruminant_values=panel._ruminant_cap_values_from_config(panel.CONFIG),
                yield_values=panel._expand_axis_values(
                    panel.CONFIG,
                    "yield_change_pct_values",
                    "yield_change_pct_range",
                ),
                ef_values=panel._expand_axis_values(
                    panel.CONFIG,
                    "emission_factor_change_pct_values",
                    "emission_factor_change_pct_range",
                ),
            )
        )
    except Exception:
        if strict:
            raise
        tasks = []
    total_runs = len(tasks)
    expected_ids = {str(task["scenario_id"]) for task in tasks}
    if strict and (not expected_ids or len(expected_ids) != total_runs):
        raise ValueError("Strict merge requires a nonempty design with unique scenario IDs.")
    result_paths: List[Path] = []
    detail_paths: List[Path] = []
    meta_frames: List[pd.DataFrame] = []
    missing_meta: List[Tuple[str, Path, int, int]] = []
    empty_meta: List[Tuple[str, Path, int, int]] = []
    missing_results: List[Tuple[str, Path, int, int]] = []
    incomplete_results: List[Tuple[str, Path, int, int]] = []
    audit_rows: List[Dict[str, object]] = []
    ids_by_batch: Dict[str, List[str]] = {}

    for tag in tags:
        batch_dir = batches_root / tag
        meta_path = batch_dir / "run_meta.csv"
        result_path = batch_dir / results_name
        expected_rows = _expected_rows_for_batch(tag, total_runs)
        result_df = _read_csv_safe(result_path)
        result_rows = len(result_df)
        result_error = ""
        try:
            result_ids = _validated_result_ids(result_df)
            filtered_ids = _validated_result_ids(_drop_invalid_emission_sentinel(result_df))
            if set(result_ids) - expected_ids:
                result_error = "unexpected scenario IDs"
            elif len(filtered_ids) != len(result_ids):
                result_error = "invalid emission sentinel"
        except ValueError as exc:
            result_ids = []
            result_error = str(exc)
        ids_by_batch[tag] = result_ids
        unique_rows = len(set(result_ids))
        meta_exists = meta_path.exists()
        meta_empty = False
        meta_attempted = None
        meta_success = None
        if not meta_path.exists():
            missing_meta.append((tag, meta_path, result_rows, expected_rows))
        else:
            meta_df = _read_csv_safe(meta_path)
            if meta_df.empty:
                meta_empty = True
                empty_meta.append((tag, meta_path, result_rows, expected_rows))
            else:
                meta_df["batch_tag"] = tag
                meta_frames.append(meta_df)
                if "attempted_runs" in meta_df.columns:
                    vals = pd.to_numeric(meta_df["attempted_runs"], errors="coerce").dropna()
                    if not vals.empty:
                        meta_attempted = int(vals.iloc[-1])
                if "success_count" in meta_df.columns:
                    vals = pd.to_numeric(meta_df["success_count"], errors="coerce").dropna()
                    if not vals.empty:
                        meta_success = int(vals.iloc[-1])

        if result_path.exists():
            result_paths.append(result_path)
        elif strict:
            missing_results.append((tag, result_path, 0, expected_rows))

        detail_path = batch_dir / detail_name
        if detail_path.exists():
            detail_paths.append(detail_path)

        if (
            meta_exists
            and not meta_empty
            and result_path.exists()
            and (unique_rows != expected_rows or result_rows != unique_rows or result_error)
        ):
            incomplete_results.append((tag, result_path, result_rows, expected_rows))
        audit_rows.append(
            {
                "batch_tag": tag,
                "batch_dir": str(batch_dir),
                "has_meta": bool(meta_exists),
                "meta_empty": bool(meta_empty),
                "has_results": bool(result_path.exists()),
                "result_rows": int(result_rows),
                "unique_scenarios": int(unique_rows),
                "result_error": result_error,
                "expected_rows": int(expected_rows),
                "meta_attempted_runs": meta_attempted,
                "meta_success_count": meta_success,
                "is_complete": bool(
                    meta_exists
                    and not meta_empty
                    and result_path.exists()
                    and unique_rows == expected_rows
                    and result_rows == unique_rows
                    and not result_error
                ),
            }
        )

    id_counts = Counter(sid for ids in ids_by_batch.values() for sid in ids)
    duplicate_ids = {sid for sid, count in id_counts.items() if count > 1}
    for row in audit_rows:
        duplicate_count = len(set(ids_by_batch[row["batch_tag"]]) & duplicate_ids)
        row["duplicate_scenarios"] = duplicate_count
        if duplicate_count:
            row["is_complete"] = False
    coverage_complete = bool(expected_ids) and set(id_counts) == expected_ids and not duplicate_ids
    merge_complete = coverage_complete and all(row["is_complete"] for row in audit_rows)
    audit_out = root_output_dir / "batch_completion_audit.csv"
    pd.DataFrame(audit_rows).to_csv(audit_out, index=False, encoding="utf-8-sig")
    print(f"[S5_3_3] batch audit={audit_out} rows={len(audit_rows)}")

    if missing_meta:
        msg = (
            f"[S5_3_3][WARN] missing batch meta in {len(missing_meta)}/{len(tags)} batches. "
            "run_meta.csv is written only after a batch reaches the end, so these batches are not "
            "proven complete even if their result CSV has rows.\n"
            + _format_missing_batch_lines(missing_meta)
            + f"\nResubmit unfinished batches with: {resubmit_command}"
        )
        if strict:
            raise FileNotFoundError(msg)
        print(msg)
    if empty_meta:
        msg = (
            f"[S5_3_3][WARN] empty batch meta in {len(empty_meta)}/{len(tags)} batches.\n"
            + _format_missing_batch_lines(empty_meta)
        )
        if strict:
            raise RuntimeError(msg)
        print(msg)
    if missing_results:
        msg = (
            f"[S5_3_3][WARN] missing batch result CSV in {len(missing_results)}/{len(tags)} batches.\n"
            + _format_missing_batch_lines(missing_results)
        )
        if strict:
            raise FileNotFoundError(msg)
        print(msg)
    if incomplete_results:
        msg = (
            f"[S5_3_3][WARN] incomplete batch result CSV in {len(incomplete_results)}/{len(tags)} batches.\n"
            + _format_missing_batch_lines(incomplete_results)
            + f"\nResubmit unfinished batches with: {resubmit_command}"
        )
        if strict:
            raise RuntimeError(msg)
        print(msg)

    if strict and not merge_complete:
        raise ValueError(
            f"Incomplete unique scenario coverage: expected={len(expected_ids)}, "
            f"unique={len(id_counts)}, missing={len(expected_ids - set(id_counts))}, "
            f"duplicates={len(duplicate_ids)}. See {audit_out}"
        )

    result_out = root_output_dir / results_name
    detail_out = root_output_dir / detail_name
    meta_out = root_output_dir / "run_meta.csv"

    result_stats = _stream_merge_csv_files(
        result_paths,
        result_out,
        dedup_cols=["scenario_id"],
        sort_cols=[
            "panel_row",
            "panel_col",
            "yield_change_pct",
            "emission_factor_change_pct",
            "scenario_id",
        ],
        chunksize=merge_chunksize,
        normalise_fn=_normalise_ruminant_cap_columns,
        count_ok_column="run_status",
        expected_scenario_ids=expected_ids if strict else None,
    )
    detail_stats = _stream_merge_csv_files(
        detail_paths,
        detail_out,
        dedup_cols=(
            [
                "scenario_id",
                "year",
                "source_module",
                "Process",
                "Item",
                "GHG",
            ]
            if bool(args.dedup_detail)
            else []
        ),
        sort_cols=[
            "scenario_id",
            "year",
            "source_module",
            "Process",
            "Item",
            "GHG",
        ],
        chunksize=merge_chunksize,
        normalise_fn=_normalise_ruminant_cap_columns,
    )
    # Result rows are authoritative; batch metadata can be stale after retries.
    merged_status_df = pd.read_csv(result_out) if result_out.exists() else pd.DataFrame()
    statuses = merged_status_df.get("run_status", pd.Series(dtype=str)).astype(str).str.strip().str.lower()
    merged_ids = set(merged_status_df.get("scenario_id", pd.Series(dtype=str)).dropna().astype(str).str.strip())
    merge_complete = merge_complete and merged_ids == expected_ids and len(merged_status_df) == len(expected_ids)
    meta_row = {
        "merge_source": "batches",
        "merge_complete": bool(merge_complete),
        "completed_batches": sum(bool(row["is_complete"]) for row in audit_rows),
        "expected_batches": max(_batch_sort_key(tag)[0] for tag in tags),
        "requested_runs": int(total_runs),
        "assigned_runs": sum(int(row["expected_rows"]) for row in audit_rows),
        "attempted_runs": int(len(merged_status_df)),
        "success_count": int(statuses.isin(["ok", "resumed"]).sum()),
        "resume_count": int(statuses.eq("resumed").sum()),
        "infeasible_count": int(statuses.eq("infeasible").sum()),
        "nonoptimal_count": int(statuses.eq("nonoptimal").sum()),
        "failed_count": int((~statuses.isin(["ok", "resumed", "infeasible", "nonoptimal"])).sum()),
        "detail_rows": int(detail_stats["rows"]),
        "results_csv": results_name,
        "global_emissions_detail_csv": detail_name,
    }
    pd.DataFrame([meta_row]).to_csv(meta_out, index=False, encoding="utf-8-sig")
    write_sensitivity_cost_summaries(
        merged_status_df,
        output_dir=root_output_dir,
        run_search_root=root_output_dir,
    )

    print(f"[DONE] merged results: {result_out} rows={result_stats['rows']}")
    print(f"[DONE] merged detail: {detail_out} rows={detail_stats['rows']}")
    print(f"[DONE] merged run_meta: {meta_out}")
    print("[NOTE] No run directories are copied; merge only extracts batch CSV data.")
    print(f"[NOTE] To rebuild plot-ready data, run: {data_summary_command}")


if __name__ == "__main__":
    main()
