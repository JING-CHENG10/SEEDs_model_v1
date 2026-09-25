# -*- coding: utf-8 -*-
"""Merge S5.7 Strategy endpoint-rerun batch outputs.

Reads:
  <output_dir>/batches/batch_XX_of_YY/

Writes:
  <output_dir>/merged/

The merge recomputes normalized single-strategy potentials and, when Shapley
coalitions are present or ``--shapley`` is passed, the exact Shapley
decomposition from the merged full-model reruns.
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

import S5_7_1_strategy_endpoint_rerun_max_reduction_potential as s57


CONFIG = {
    "output_dir": "",  # empty -> <NZF_OUTPUT_DIR>/Strategy_Endpoint_Max_Reduction_Potential
    "batches_subdir": "batches",
    "merged_subdir": "merged",
    "total_batches": 0,  # 0/None -> auto-detect
    "batch_tags": [],
    "strict": True,
    "shapley": None,  # None -> auto-detect from merged rows
    "baseline_scenario_id": str(s57.CONFIG.get("baseline_scenario_id") or "S5_7_BASE"),
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
    return Path(text) if text else s57._default_output_dir()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception as exc:
        print(f"[S5_7_MERGE][WARN] pandas failed to read {path}: {type(exc).__name__}: {exc}")
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
                f"[S5_7_MERGE][WARN] fallback parser failed for {path}: "
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
        raise FileNotFoundError(f"No S5.7 batches directory found: {batches_root}")
    tags = sorted(p.name for p in batches_root.iterdir() if p.is_dir())
    if not tags:
        raise FileNotFoundError(f"No S5.7 batch directories found under: {batches_root}")
    return batches_root, tags


def _with_batch_meta(df: pd.DataFrame, *, tag: str, index: int, count: int, batch_dir: Path) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "batch_index" not in out.columns:
        out["batch_index"] = int(index)
    if "batch_count" not in out.columns:
        out["batch_count"] = int(count)
    if "batch_tag" not in out.columns:
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
    if "plan_index" in work.columns:
        work["_plan_sort"] = pd.to_numeric(work["plan_index"], errors="coerce").fillna(999999)
    else:
        work["_plan_sort"] = 999999
    work = work.sort_values(["scenario_id", "_status_priority", "_batch_sort", "_plan_sort"])
    work = work.drop_duplicates(subset=["scenario_id"], keep="first")
    return work.drop(columns=["_status_priority", "_batch_sort", "_plan_sort"], errors="ignore")


def _deduplicate_design(design: pd.DataFrame, valid_scenario_ids: Sequence[str]) -> pd.DataFrame:
    if design.empty or "scenario_id" not in design.columns:
        return pd.DataFrame(columns=["scenario_id"])
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


def _detect_shapley(status: pd.DataFrame) -> bool:
    if status.empty or "strategy_name" not in status.columns:
        return False
    return status["strategy_name"].astype(str).str.startswith("shapley_coalition_").any()


def _collect_batch_singleton_costs(
    status: pd.DataFrame,
    batch_dir: Path,
    *,
    expected_identity: Optional[Mapping[str, object]] = None,
    reference_scenario_id: str = "",
    strict: bool = True,
) -> pd.DataFrame:
    if status.empty or "scenario_id" not in status.columns:
        return pd.DataFrame()
    work = status.copy()
    work["kind"] = work.get("include_kinds", "").map(s57._single_kind_from_include)
    work = work[
        work.get("scope", "").astype(str).eq("global")
        & work["kind"].astype(str).ne("")
        & work.get("run_status", "").astype(str).isin(["ok", "resumed"])
    ].copy()
    frames: List[pd.DataFrame] = []
    identity = dict(expected_identity or {})
    base_expected_status = {
        "cost_database_version": str(identity.get("cost_database_version", "") or ""),
        "cost_database_sha256": str(identity.get("cost_database_sha256", "") or ""),
        "cost_reference_scenario_id": str(reference_scenario_id or ""),
    }
    for _, row in work.iterrows():
        scenario_id = str(row.get("scenario_id", "") or "")
        kind = str(row.get("kind", "") or "")
        scenario_dir = batch_dir / "runs" / scenario_id
        cost_path = scenario_dir / "cost_summary.csv"
        cost_df = _read_csv_safe(cost_path)
        database_strategy = s57._database_strategy_for_kind(kind)
        design_signature = str(row.get("cost_design_signature", "") or "").strip()
        expected_status = {
            **base_expected_status,
            "active_strategy_cost_keys": database_strategy,
            "cost_attribution_method": "strict_singleton",
            "cost_strategy_regions": "",
            "cost_resume_fingerprint": s57._cost_resume_fingerprint(
                (database_strategy,) if database_strategy else (),
                identity,
                reference_scenario_id,
                design_signature=design_signature,
            ),
        }
        mismatches = []
        if not design_signature:
            mismatches.append("cost_design_signature is empty")
        for column, expected in expected_status.items():
            actual = "" if pd.isna(row.get(column)) else str(row.get(column, "")).strip()
            if not expected or actual != expected:
                mismatches.append(f"{column}={actual!r}, expected={expected!r}")
        cost_matches = bool(database_strategy) and s57._cost_summary_matches_singleton(
            scenario_dir,
            database_strategy=database_strategy,
            identity=identity,
            reference_scenario_id=reference_scenario_id,
        )
        if cost_df.empty or mismatches or not cost_matches:
            message = (
                f"singleton cost provenance mismatch for {scenario_id}: "
                + ("; ".join(mismatches) if mismatches else "cost_summary contract mismatch")
            )
            if strict:
                raise RuntimeError(message)
            print(f"[S5_7_MERGE][WARN] {message}")
            continue
        detail = s57._strict_singleton_cost_rows(cost_df, kind)
        detail = s57._current_database_cost_rows(
            detail,
            {
                **identity,
                "baseline_scenario_id": str(reference_scenario_id or ""),
            },
        )
        if detail.empty:
            if strict:
                raise RuntimeError(
                    f"no current-database singleton cost rows for {scenario_id}"
                )
            continue
        detail.insert(0, "scenario_id", scenario_id)
        detail.insert(1, "kind", kind)
        detail.insert(2, "scenario_dir", str(scenario_dir))
        detail.insert(
            3,
            "cost_resume_fingerprint",
            str(row.get("cost_resume_fingerprint", "") or ""),
        )
        frames.append(detail)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge S5.7 Strategy endpoint-rerun batch outputs.")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--total-batches", type=int, default=None)
    parser.add_argument("--batches-subdir", type=str, default=None)
    parser.add_argument("--merged-subdir", type=str, default=None)
    parser.add_argument("--baseline-scenario-id", type=str, default=None)
    parser.add_argument("--strict", action="store_true", default=None)
    parser.add_argument("--no-strict", action="store_false", dest="strict")
    parser.add_argument("--shapley", action="store_true", default=None)
    parser.add_argument("--no-shapley", action="store_false", dest="shapley")
    return parser


def _effective_config(args: argparse.Namespace) -> Dict[str, object]:
    cfg = dict(CONFIG)
    env_output = str(os.environ.get("S57_OUTPUT_DIR", "") or "").strip()
    if env_output:
        cfg["output_dir"] = env_output
    env_total = str(os.environ.get("S57_TOTAL_BATCHES", "") or "").strip()
    if env_total:
        cfg["total_batches"] = _parse_int(env_total, name="S57_TOTAL_BATCHES")
    env_strict = str(os.environ.get("S57_MERGE_STRICT", "") or "").strip()
    if env_strict:
        cfg["strict"] = _parse_bool(env_strict)
    env_shapley = str(os.environ.get("S57_SHAPLEY", "") or "").strip()
    if env_shapley:
        cfg["shapley"] = _parse_bool(env_shapley)

    if args.out_dir:
        cfg["output_dir"] = args.out_dir
    if args.total_batches is not None:
        cfg["total_batches"] = int(args.total_batches)
    if args.batches_subdir:
        cfg["batches_subdir"] = str(args.batches_subdir)
    if args.merged_subdir:
        cfg["merged_subdir"] = str(args.merged_subdir)
    if args.baseline_scenario_id:
        cfg["baseline_scenario_id"] = str(args.baseline_scenario_id)
    if args.strict is not None:
        cfg["strict"] = bool(args.strict)
    if args.shapley is not None:
        cfg["shapley"] = bool(args.shapley)
    return cfg


def _summary_config(
    merged_dir: Path,
    status_df: pd.DataFrame,
    cfg: Dict[str, object],
    *,
    database_identity: Optional[Mapping[str, object]] = None,
    reference_scenario_id: str = "",
) -> Dict[str, object]:
    summary_cfg = copy.deepcopy(s57.CONFIG)
    summary_cfg["output_dir"] = str(merged_dir)
    summary_cfg["write_summary_outputs"] = True
    summary_cfg.update(dict(database_identity or {}))
    summary_cfg["baseline_scenario_id"] = str(
        reference_scenario_id
        or cfg.get("baseline_scenario_id")
        or summary_cfg.get("baseline_scenario_id")
        or "S5_7_BASE"
    )

    shapley_raw = cfg.get("shapley")
    shapley = _detect_shapley(status_df) if shapley_raw is None else bool(shapley_raw)
    decomp = copy.deepcopy(summary_cfg.get("decomposition", {}) or {})
    if shapley:
        decomp["method"] = "shapley"
        decomp["run_shapley"] = True
    else:
        decomp["method"] = "single"
        decomp["run_shapley"] = False
    summary_cfg["decomposition"] = decomp
    return summary_cfg


def run_merge(cfg: Mapping[str, object]) -> Path:
    """Merge one configured S5.7 batch root and return its merged directory."""

    cfg = dict(cfg)
    root = _root_output_dir(cfg.get("output_dir", ""))
    merged_dir = root / str(cfg.get("merged_subdir", "merged") or "merged")
    batches_root, tags = _resolve_batch_tags(root, cfg)
    strict = bool(cfg.get("strict", True))
    paths = s57.s56.DataPaths()
    database_identity = s57._cost_database_identity(paths)
    reference_scenario_id = str(
        cfg.get("baseline_scenario_id")
        or s57.CONFIG.get("baseline_scenario_id")
        or "S5_7_BASE"
    )
    if strict and (
        str(database_identity.get("cost_database_error", "") or "").strip()
        or not str(database_identity.get("cost_database_version", "") or "").strip()
        or not str(database_identity.get("cost_database_sha256", "") or "").strip()
    ):
        raise RuntimeError(
            "Cannot identify the current v2 cost database for strict batch merge: "
            f"{database_identity}"
        )

    status_frames: List[pd.DataFrame] = []
    design_frames: List[pd.DataFrame] = []
    plan_frames: List[pd.DataFrame] = []
    singleton_cost_frames: List[pd.DataFrame] = []
    manifest_rows: List[Dict[str, object]] = []
    count = len(tags)
    for idx, tag in enumerate(tags, start=1):
        batch_dir = batches_root / tag
        status_path = batch_dir / "scenario_status.csv"
        design_path = batch_dir / "strategy_design_long.csv"
        plan_path = batch_dir / "batch_plan_manifest.csv"
        if not batch_dir.exists():
            msg = f"missing batch dir: {batch_dir}"
            if strict:
                raise FileNotFoundError(msg)
            print(f"[S5_7_MERGE][WARN] {msg}")
            continue

        status = _read_csv_safe(status_path)
        design = _read_csv_safe(design_path)
        plans = _read_csv_safe(plan_path)
        if status.empty:
            msg = f"missing or empty scenario_status.csv in {batch_dir}"
            if strict:
                raise FileNotFoundError(msg)
            print(f"[S5_7_MERGE][WARN] {msg}")
        else:
            status_frames.append(_with_batch_meta(status, tag=tag, index=idx, count=count, batch_dir=batch_dir))
            batch_cost = _collect_batch_singleton_costs(
                status,
                batch_dir,
                expected_identity=database_identity,
                reference_scenario_id=reference_scenario_id,
                strict=strict,
            )
            if not batch_cost.empty:
                singleton_cost_frames.append(
                    _with_batch_meta(batch_cost, tag=tag, index=idx, count=count, batch_dir=batch_dir)
                )
        if not design.empty:
            design_frames.append(_with_batch_meta(design, tag=tag, index=idx, count=count, batch_dir=batch_dir))
        if not plans.empty:
            plan_frames.append(_with_batch_meta(plans, tag=tag, index=idx, count=count, batch_dir=batch_dir))

        manifest_rows.append(
            {
                "batch_index": idx,
                "batch_count": count,
                "batch_tag": tag,
                "batch_dir": str(batch_dir),
                "status_rows": int(len(status)),
                "design_rows": int(len(design)),
                "plan_rows": int(len(plans)),
            }
        )

    if not status_frames:
        raise RuntimeError("No S5.7 batch status rows found.")

    status_all = pd.concat(status_frames, ignore_index=True)
    status_merged = _deduplicate_status(status_all)
    valid_ids = status_merged["scenario_id"].astype(str).tolist() if "scenario_id" in status_merged.columns else []
    design_all = pd.concat(design_frames, ignore_index=True) if design_frames else pd.DataFrame(columns=["scenario_id"])
    design_merged = _deduplicate_design(design_all, valid_ids)
    plan_all = pd.concat(plan_frames, ignore_index=True) if plan_frames else pd.DataFrame()

    _ensure_dir(merged_dir)
    _write_csv(pd.DataFrame(manifest_rows), merged_dir / "batch_merge_manifest.csv")
    _write_csv(status_merged, merged_dir / "scenario_status.csv")
    _write_csv(design_merged, merged_dir / "strategy_design_long.csv")
    if not plan_all.empty:
        _write_csv(plan_all, merged_dir / "batch_plan_manifest.csv")
    if singleton_cost_frames:
        singleton_cost_all = pd.concat(singleton_cost_frames, ignore_index=True)
        dedup_cols = [
            col
            for col in (
                "scenario_id",
                "kind",
                "region",
                "commodity",
                "year",
                "process",
                "segment",
            )
            if col in singleton_cost_all.columns
        ]
        if dedup_cols:
            singleton_cost_all = singleton_cost_all.drop_duplicates(
                subset=dedup_cols,
                keep="first",
            )
        _write_csv(singleton_cost_all, merged_dir / "macc_singleton_cost_detail.csv")

    shared_cfg = s57.s56.ScenarioConfig()
    universe = s57.s56.build_universe_from_dict_v3(paths.dict_v3_path, shared_cfg)
    summary_cfg = _summary_config(
        merged_dir,
        status_merged,
        cfg,
        database_identity=database_identity,
        reference_scenario_id=reference_scenario_id,
    )
    s57._write_summary_outputs(
        status_df=status_merged,
        design_df=design_merged,
        cfg=summary_cfg,
        universe=universe,
    )

    print(
        "[S5_7_MERGE] "
        f"root={root} batches={len(tags)} "
        f"status_rows={len(status_merged)} design_rows={len(design_merged)} "
        f"shapley={s57._shapley_enabled(summary_cfg)} "
        f"merged_dir={merged_dir}"
    )
    return merged_dir


def main(argv: Optional[Iterable[str]] = None) -> Path:
    args = _build_arg_parser().parse_args(list(argv) if argv is not None else None)
    return run_merge(_effective_config(args))


if __name__ == "__main__":
    main()
