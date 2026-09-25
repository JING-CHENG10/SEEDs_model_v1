# -*- coding: utf-8 -*-
"""Merge S5.6 maximum emission-reduction batch outputs.

Reads:
  <output_dir>/batches/batch_XX_of_YY/

Writes:
  <output_dir>/merged/

The merge recomputes reduction columns against the merged baseline row, so
country-only batches do not need to carry their own baseline scenario.
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

import S5_6_1_max_emission_reduction_potential as maxred
from S1_0_schema import ScenarioConfig
from S2_0_load_data import DataPaths, build_universe_from_dict_v3


CONFIG = {
    "output_dir": "",  # empty -> <NZF_OUTPUT_DIR>/Max_Emission_Reduction_Potential
    "batches_subdir": "batches",
    "merged_subdir": "merged",
    "total_batches": 0,  # 0/None -> auto-detect
    "batch_tags": [],
    "strict": True,
}


def _parse_int(raw: object, *, name: str) -> int:
    try:
        return int(str(raw).strip())
    except Exception as exc:
        raise ValueError(f"Invalid integer for {name}: {raw}") from exc


def _parse_bool(raw: object) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _batch_tag(batch_index: int, batch_count: int) -> str:
    return f"batch_{int(batch_index):02d}_of_{int(batch_count):02d}"


def _root_output_dir(raw: object = "") -> Path:
    text = str(raw or "").strip()
    return Path(text) if text else maxred._default_output_dir()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[S5_6_MERGE][WARN] pandas failed to read {path}: {type(exc).__name__}: {exc}")
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                rows = list(csv.reader(f))
            if not rows:
                return pd.DataFrame()
            header = rows[0]
            body = rows[1:]
            width = max((len(r) for r in rows), default=len(header))
            if width > len(header):
                header = header + [f"extra_col_{i}" for i in range(1, width - len(header) + 1)]
            normalized = [r[: len(header)] + [""] * max(0, len(header) - len(r)) for r in body]
            return pd.DataFrame(normalized, columns=header)
        except Exception as fallback_exc:
            print(
                f"[S5_6_MERGE][WARN] fallback parser failed for {path}: "
                f"{type(fallback_exc).__name__}: {fallback_exc}"
            )
            return pd.DataFrame()


def _resolve_batch_tags(root: Path, cfg: Dict[str, object]) -> Tuple[Path, List[str]]:
    batches_root = root / str(cfg.get("batches_subdir", "batches") or "batches")
    explicit = [str(x).strip() for x in (cfg.get("batch_tags") or []) if str(x).strip()]
    if explicit:
        return batches_root, explicit
    total = cfg.get("total_batches")
    if total not in (None, "", 0):
        count = int(total)
        if count <= 0:
            raise ValueError("total_batches must be positive, 0, or None")
        return batches_root, [_batch_tag(i, count) for i in range(1, count + 1)]
    if not batches_root.exists():
        raise FileNotFoundError(f"No S5.6 batches directory found: {batches_root}")
    tags = sorted(p.name for p in batches_root.iterdir() if p.is_dir())
    if not tags:
        raise FileNotFoundError(f"No S5.6 batch directories found under: {batches_root}")
    return batches_root, tags


def _with_batch_meta(df: pd.DataFrame, *, tag: str, index: int, count: int, batch_dir: Path) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["batch_index"] = int(index)
    out["batch_count"] = int(count)
    out["batch_tag"] = str(tag)
    out["batch_dir"] = str(batch_dir)
    return out


def _status_priority(series: pd.Series) -> pd.Series:
    order = {
        "ok": 0,
        "resumed": 1,
        "dry_run": 2,
        "invalid_market_balance": 3,
        "invalid_fast_emissions": 4,
        "missing_fast_summary": 5,
        "infeasible": 6,
        "nonoptimal": 7,
        "failed": 8,
        "precheck_failed": 9,
    }
    return series.astype(str).str.strip().map(order).fillna(99).astype(int)


def _deduplicate_status(status: pd.DataFrame) -> pd.DataFrame:
    if status.empty or "scenario_id" not in status.columns:
        return status
    work = status.copy()
    work["_status_priority"] = _status_priority(work.get("run_status", pd.Series("", index=work.index)))
    if "batch_index" in work.columns:
        work["_batch_sort"] = pd.to_numeric(work["batch_index"], errors="coerce").fillna(999999)
    else:
        work["_batch_sort"] = 999999
    work = work.sort_values(["scenario_id", "_status_priority", "_batch_sort"])
    work = work.drop_duplicates(subset=["scenario_id"], keep="first")
    return work.drop(columns=["_status_priority", "_batch_sort"], errors="ignore")


def _deduplicate_design(design: pd.DataFrame, valid_scenario_ids: Sequence[str]) -> pd.DataFrame:
    if design.empty or "scenario_id" not in design.columns:
        return design
    out = design[design["scenario_id"].astype(str).isin(set(map(str, valid_scenario_ids)))].copy()
    key_cols = [
        c
        for c in (
            "scenario_id",
            "spec_row_id",
            "kind",
            "country",
            "item_selector",
            "process_selector",
            "ghg_selector",
            "region_selector",
        )
        if c in out.columns
    ]
    if key_cols:
        out = out.drop_duplicates(subset=key_cols, keep="first")
    sort_cols = [c for c in ("scope", "country", "scenario_id", "spec_row_id") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols)
    return out


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    _ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge S5.6 maximum-reduction batch outputs.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--total-batches", type=int, default=None)
    parser.add_argument("--batches-subdir", type=str, default=None)
    parser.add_argument("--merged-subdir", type=str, default=None)
    parser.add_argument("--strict", action="store_true", default=None)
    parser.add_argument("--no-strict", action="store_false", dest="strict")
    return parser


def _effective_config(args: argparse.Namespace) -> Dict[str, object]:
    cfg = dict(CONFIG)
    env_output = str(os.environ.get("S56_OUTPUT_DIR", "") or "").strip()
    if env_output:
        cfg["output_dir"] = env_output
    env_total = str(os.environ.get("S56_TOTAL_BATCHES", "") or "").strip()
    if env_total:
        cfg["total_batches"] = _parse_int(env_total, name="S56_TOTAL_BATCHES")
    env_strict = str(os.environ.get("S56_MERGE_STRICT", "") or "").strip()
    if env_strict:
        cfg["strict"] = _parse_bool(env_strict)

    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.total_batches is not None:
        cfg["total_batches"] = int(args.total_batches)
    if args.batches_subdir:
        cfg["batches_subdir"] = str(args.batches_subdir)
    if args.merged_subdir:
        cfg["merged_subdir"] = str(args.merged_subdir)
    if args.strict is not None:
        cfg["strict"] = bool(args.strict)
    return cfg


def main() -> None:
    args = _build_arg_parser().parse_args()
    cfg = _effective_config(args)
    root = _root_output_dir(cfg.get("output_dir", ""))
    merged_dir = root / str(cfg.get("merged_subdir", "merged") or "merged")
    batches_root, tags = _resolve_batch_tags(root, cfg)
    strict = bool(cfg.get("strict", True))

    status_frames: List[pd.DataFrame] = []
    design_frames: List[pd.DataFrame] = []
    manifest_rows: List[Dict[str, object]] = []
    count = len(tags)
    for idx, tag in enumerate(tags, start=1):
        batch_dir = batches_root / tag
        status_path = batch_dir / "scenario_status.csv"
        design_path = batch_dir / "strategy_design_long.csv"
        if not batch_dir.exists():
            msg = f"missing batch dir: {batch_dir}"
            if strict:
                raise FileNotFoundError(msg)
            print(f"[S5_6_MERGE][WARN] {msg}")
            continue
        status = _read_csv_safe(status_path)
        design = _read_csv_safe(design_path)
        if status.empty:
            msg = f"missing or empty scenario_status.csv in {batch_dir}"
            if strict:
                raise FileNotFoundError(msg)
            print(f"[S5_6_MERGE][WARN] {msg}")
        else:
            status_frames.append(_with_batch_meta(status, tag=tag, index=idx, count=count, batch_dir=batch_dir))
        if design.empty:
            msg = f"missing or empty strategy_design_long.csv in {batch_dir}"
            if strict:
                raise FileNotFoundError(msg)
            print(f"[S5_6_MERGE][WARN] {msg}")
        else:
            design_frames.append(_with_batch_meta(design, tag=tag, index=idx, count=count, batch_dir=batch_dir))
        manifest_rows.append(
            {
                "batch_index": idx,
                "batch_count": count,
                "batch_tag": tag,
                "batch_dir": str(batch_dir),
                "status_rows": int(len(status)),
                "design_rows": int(len(design)),
            }
        )

    if not status_frames:
        raise RuntimeError("No S5.6 batch status rows found.")

    status_all = pd.concat(status_frames, ignore_index=True)
    status_merged = _deduplicate_status(status_all)
    valid_ids = status_merged["scenario_id"].astype(str).tolist() if "scenario_id" in status_merged.columns else []
    design_all = pd.concat(design_frames, ignore_index=True) if design_frames else pd.DataFrame()
    design_merged = _deduplicate_design(design_all, valid_ids)

    _ensure_dir(merged_dir)
    _write_csv(pd.DataFrame(manifest_rows), merged_dir / "batch_merge_manifest.csv")
    _write_csv(status_merged, merged_dir / "scenario_status.csv")
    _write_csv(design_merged, merged_dir / "strategy_design_long.csv")

    paths = DataPaths()
    scfg = ScenarioConfig()
    universe = build_universe_from_dict_v3(paths.dict_v3_path, scfg)
    summary_cfg = copy.deepcopy(maxred.CONFIG)
    summary_cfg["output_dir"] = str(merged_dir)
    maxred._write_summary_outputs(
        status_df=status_merged,
        design_df=design_merged,
        cfg=summary_cfg,
        universe=universe,
    )

    print(
        "[S5_6_MERGE] "
        f"root={root} batches={len(tags)} "
        f"status_rows={len(status_merged)} design_rows={len(design_merged)} "
        f"merged_dir={merged_dir}"
    )


if __name__ == "__main__":
    main()
