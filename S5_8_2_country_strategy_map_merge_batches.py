# -*- coding: utf-8 -*-
"""Merge S5.8 country-by-strategy batch outputs."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import S5_8_1_country_strategy_map_sensitivity as s58
from S5_cost_summary_outputs import write_sensitivity_cost_summaries


CONFIG = {
    "output_dir": "",
    "batches_subdir": "batches",
    "merged_subdir": "merged",
    "total_batches": 0,
    "batch_tags": [],
    # Normal merge writes all available outputs and diagnostics.
    # Use --strict when an incomplete or unusable grid must stop the command.
    "strict": False,
    "reference_tolerance_gt": 1e-9,
    "baseline_scenario_id": str(
        s58.CONFIG.get("baseline_scenario_id") or "BASE_S5_8_REFERENCE"
    ),
}


def _parse_int(raw: object, *, name: str) -> int:
    try:
        return int(str(raw).strip())
    except Exception as exc:
        raise ValueError(f"Invalid integer for {name}: {raw}") from exc


def _parse_bool(raw: object) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _batch_tag(index: int, count: int) -> str:
    return f"batch_{int(index):02d}_of_{int(count):02d}"


def _root_output_dir(raw: object = "") -> Path:
    text = str(raw or "").strip()
    return s58.ensure_output_child(
        Path(text) if text else s58._default_output_dir()
    )


def _read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(
            f"[S5_8_MERGE] pandas read failed for {path}: "
            f"{type(exc).__name__}: {exc}"
        )
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.reader(handle))
            if not rows:
                return pd.DataFrame()
            header = rows[0]
            width = max(len(row) for row in rows)
            if width > len(header):
                header = header + [
                    f"extra_col_{index}"
                    for index in range(1, width - len(header) + 1)
                ]
            body = [
                row[: len(header)] + [""] * max(0, len(header) - len(row))
                for row in rows[1:]
            ]
            return pd.DataFrame(body, columns=header)
        except Exception as fallback_exc:
            print(
                f"[S5_8_MERGE] fallback read failed for {path}: "
                f"{type(fallback_exc).__name__}: {fallback_exc}"
            )
            return pd.DataFrame()


def _resolve_batch_tags(
    root: Path,
    cfg: Mapping[str, object],
) -> Tuple[Path, List[str]]:
    batches_root = root / str(cfg.get("batches_subdir", "batches") or "batches")
    explicit = [
        str(value).strip()
        for value in (cfg.get("batch_tags") or [])
        if str(value).strip()
    ]
    if explicit:
        return batches_root, explicit
    total = cfg.get("total_batches")
    if total not in (None, "", 0):
        count = int(total)
        return batches_root, [_batch_tag(index, count) for index in range(1, count + 1)]
    if not batches_root.exists():
        raise FileNotFoundError(f"Missing S5.8 batches directory: {batches_root}")
    tags = sorted(path.name for path in batches_root.iterdir() if path.is_dir())
    if not tags:
        raise FileNotFoundError(f"No S5.8 batch directories under: {batches_root}")
    return batches_root, tags


def _with_batch_meta(
    df: pd.DataFrame,
    *,
    tag: str,
    index: int,
    count: int,
    batch_dir: Path,
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["batch_index"] = int(index)
    out["batch_count"] = int(count)
    out["batch_tag"] = tag
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
        "reference_unavailable": 6,
        "infeasible": 7,
        "nonoptimal": 8,
        "failed": 9,
        "precheck_failed": 10,
    }
    return series.astype(str).str.strip().map(order).fillna(99).astype(int)


def _deduplicate_status(status: pd.DataFrame) -> pd.DataFrame:
    if status.empty or "scenario_id" not in status.columns:
        return status
    work = status.copy()
    work["_priority"] = _status_priority(
        work.get("run_status", pd.Series("", index=work.index))
    )
    work["_batch_sort"] = pd.to_numeric(
        work.get("batch_index", pd.Series(np.nan, index=work.index)),
        errors="coerce",
    ).fillna(999999)
    work = work.sort_values(["scenario_id", "_priority", "_batch_sort"])
    work = work.drop_duplicates("scenario_id", keep="first")
    return work.drop(columns=["_priority", "_batch_sort"], errors="ignore")


def _deduplicate_design(
    design: pd.DataFrame,
    valid_scenario_ids: Sequence[str],
) -> pd.DataFrame:
    if design.empty or "scenario_id" not in design.columns:
        return design
    out = design[
        design["scenario_id"].astype(str).isin(set(map(str, valid_scenario_ids)))
    ].copy()
    keys = [
        column
        for column in (
            "scenario_id",
            "spec_row_id",
            "kind",
            "source_kind",
            "country",
            "item_selector",
            "process_selector",
            "ghg_selector",
        )
        if column in out.columns
    ]
    if keys:
        out = out.drop_duplicates(keys, keep="first")
    return out.sort_values(
        [
            column
            for column in ("country", "scenario_id", "spec_row_id")
            if column in out.columns
        ]
    )


def _deduplicate_long(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return long_df
    work = long_df.copy()
    work["_priority"] = _status_priority(
        work.get("run_status", pd.Series("", index=work.index))
    )
    work["_batch_sort"] = pd.to_numeric(
        work.get("batch_index", pd.Series(np.nan, index=work.index)),
        errors="coerce",
    ).fillna(999999)
    work = work.sort_values(
        ["M49_Country_Code", "strategy_kind", "_priority", "_batch_sort"]
    )
    work = work.drop_duplicates(
        ["M49_Country_Code", "strategy_kind"],
        keep="first",
    )
    return work.drop(columns=["_priority", "_batch_sort"], errors="ignore")


def _database_provenance_mismatches(
    status: pd.DataFrame,
    *,
    database_identity: Mapping[str, object],
    reference_scenario_id: str,
) -> Dict[str, str]:
    """Return successful country singletons that are not from the current DB."""

    if status.empty:
        return {}
    expected_version = str(
        database_identity.get("cost_database_version", "") or ""
    ).strip()
    expected_sha256 = str(
        database_identity.get("cost_database_sha256", "") or ""
    ).strip()
    mismatches: Dict[str, str] = {}
    for _, row in status.iterrows():
        if str(row.get("scope", "") or "").strip() != "country_strategy":
            continue
        if str(row.get("run_status", "") or "").strip() not in {"ok", "resumed"}:
            continue
        scenario_id = str(row.get("scenario_id", "") or "").strip()
        kind = str(row.get("strategy_kind", "") or "").strip()
        expected_key = s58.s57._database_strategy_for_kind(kind)
        country = str(row.get("country", "") or "").strip()
        strategy_regions = (country,) if country else ()
        design_signature = str(row.get("cost_design_signature", "") or "").strip()
        expected_fingerprint = s58.s57._cost_resume_fingerprint(
            (expected_key,) if expected_key else (),
            database_identity,
            reference_scenario_id,
            strategy_cost_regions=strategy_regions,
            design_signature=design_signature,
        )
        expected = {
            "cost_database_version": expected_version,
            "cost_database_sha256": expected_sha256,
            "cost_reference_scenario_id": str(reference_scenario_id or ""),
            "database_strategy": expected_key,
            "active_strategy_cost_keys": expected_key,
            "cost_attribution_method": "strict_singleton",
            "cost_strategy_regions": "|".join(strategy_regions),
            "cost_resume_fingerprint": expected_fingerprint,
        }
        row_errors = []
        if not design_signature:
            row_errors.append("cost_design_signature is empty")
        for column, expected_value in expected.items():
            actual = "" if pd.isna(row.get(column)) else str(row.get(column, "")).strip()
            if not expected_value or actual != expected_value:
                row_errors.append(
                    f"{column}={actual!r}, expected={expected_value!r}"
                )
        if row_errors:
            mismatches[scenario_id] = "; ".join(row_errors)
    return mismatches


def _mark_provenance_mismatches(
    frame: pd.DataFrame,
    mismatch_ids: Sequence[str],
) -> pd.DataFrame:
    if frame.empty or "scenario_id" not in frame.columns or not mismatch_ids:
        return frame
    out = frame.copy()
    mask = out["scenario_id"].astype(str).isin(set(map(str, mismatch_ids)))
    if "run_status" in out.columns:
        out.loc[mask, "run_status"] = "database_provenance_mismatch"
    for column in ("max_reduction_eligible", "min_cost_eligible"):
        if column in out.columns:
            out.loc[mask, column] = False
    if "error_type" in out.columns:
        out.loc[mask, "error_type"] = "CostDatabaseProvenanceMismatch"
    return out


def _reference_consistency(
    reference_df: pd.DataFrame,
    *,
    tolerance_gt: float,
) -> pd.DataFrame:
    if reference_df.empty:
        return pd.DataFrame()
    work = reference_df.copy()
    work["reference_emissions_gt"] = pd.to_numeric(
        work["reference_emissions_gt"],
        errors="coerce",
    )
    grouped = (
        work.groupby("M49_Country_Code", as_index=False)
        .agg(
            reference_batch_count=("batch_tag", "nunique"),
            reference_emissions_min_gt=("reference_emissions_gt", "min"),
            reference_emissions_max_gt=("reference_emissions_gt", "max"),
            reference_emissions_mean_gt=("reference_emissions_gt", "mean"),
        )
    )
    grouped["reference_emissions_spread_gt"] = (
        grouped["reference_emissions_max_gt"]
        - grouped["reference_emissions_min_gt"]
    )
    grouped["within_tolerance"] = grouped["reference_emissions_spread_gt"].le(
        float(tolerance_gt)
    )
    return grouped


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge S5.8 batch outputs.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--total-batches", type=int, default=None)
    parser.add_argument("--batches-subdir", type=str, default=None)
    parser.add_argument("--merged-subdir", type=str, default=None)
    parser.add_argument("--strict", action="store_true", default=None)
    parser.add_argument("--no-strict", action="store_false", dest="strict")
    parser.add_argument("--reference-tolerance-gt", type=float, default=None)
    return parser


def _effective_config(args: argparse.Namespace) -> Dict[str, object]:
    cfg = dict(CONFIG)
    output_env = str(os.environ.get("S58_OUTPUT_DIR", "") or "").strip()
    if output_env:
        cfg["output_dir"] = output_env
    total_env = str(os.environ.get("S58_TOTAL_BATCHES", "") or "").strip()
    if total_env:
        cfg["total_batches"] = _parse_int(total_env, name="S58_TOTAL_BATCHES")
    strict_env = str(os.environ.get("S58_MERGE_STRICT", "") or "").strip()
    if strict_env:
        cfg["strict"] = _parse_bool(strict_env)

    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.total_batches is not None:
        cfg["total_batches"] = int(args.total_batches)
    if args.batches_subdir:
        cfg["batches_subdir"] = args.batches_subdir
    if args.merged_subdir:
        cfg["merged_subdir"] = args.merged_subdir
    if args.strict is not None:
        cfg["strict"] = bool(args.strict)
    if args.reference_tolerance_gt is not None:
        cfg["reference_tolerance_gt"] = float(args.reference_tolerance_gt)
    return cfg


def main() -> None:
    args = _build_arg_parser().parse_args()
    cfg = _effective_config(args)
    root = _root_output_dir(cfg.get("output_dir", ""))
    merged_dir = root / str(cfg.get("merged_subdir", "merged") or "merged")
    batches_root, tags = _resolve_batch_tags(root, cfg)
    strict = bool(cfg.get("strict", True))
    paths = s58.s56.DataPaths()
    database_identity = s58.s57._cost_database_identity(paths)
    reference_scenario_id = str(
        cfg.get("baseline_scenario_id")
        or s58.CONFIG.get("baseline_scenario_id")
        or "BASE_S5_8_REFERENCE"
    )
    if (
        str(database_identity.get("cost_database_error", "") or "").strip()
        or not str(database_identity.get("cost_database_version", "") or "").strip()
        or not str(database_identity.get("cost_database_sha256", "") or "").strip()
    ):
        raise RuntimeError(
            "Cannot identify the current v2 cost database for S5.8 batch merge: "
            f"{database_identity}"
        )

    status_frames: List[pd.DataFrame] = []
    design_frames: List[pd.DataFrame] = []
    long_frames: List[pd.DataFrame] = []
    reference_frames: List[pd.DataFrame] = []
    manifest_rows: List[Dict[str, object]] = []
    count = len(tags)

    for index, tag in enumerate(tags, start=1):
        batch_dir = batches_root / tag
        if not batch_dir.exists():
            if strict:
                raise FileNotFoundError(f"Missing batch directory: {batch_dir}")
            print(f"[S5_8_MERGE] missing batch directory: {batch_dir}")
            continue
        frames = {
            "status": _read_csv_safe(batch_dir / "scenario_status.csv"),
            "design": _read_csv_safe(batch_dir / "strategy_design_long.csv"),
            "long": _read_csv_safe(batch_dir / "country_strategy_long.csv"),
            "reference": _read_csv_safe(
                batch_dir / "reference_country_emissions.csv"
            ),
        }
        if strict:
            for name, frame in frames.items():
                if frame.empty:
                    raise FileNotFoundError(
                        f"Missing or empty {name} output in {batch_dir}"
                    )
        provenance_mismatches = _database_provenance_mismatches(
            frames["status"],
            database_identity=database_identity,
            reference_scenario_id=reference_scenario_id,
        )
        if provenance_mismatches:
            message = (
                f"{len(provenance_mismatches)} current-status singleton(s) in "
                f"{batch_dir} do not match the active v2 database"
            )
            if strict:
                sample = next(iter(provenance_mismatches.items()))
                raise RuntimeError(f"{message}: {sample[0]}: {sample[1]}")
            print(f"[S5_8_MERGE] warning: {message}")
            mismatch_ids = list(provenance_mismatches)
            frames["status"] = _mark_provenance_mismatches(
                frames["status"], mismatch_ids
            )
            frames["long"] = _mark_provenance_mismatches(
                frames["long"], mismatch_ids
            )
        if not frames["status"].empty:
            status_frames.append(
                _with_batch_meta(
                    frames["status"],
                    tag=tag,
                    index=index,
                    count=count,
                    batch_dir=batch_dir,
                )
            )
        if not frames["design"].empty:
            design_frames.append(
                _with_batch_meta(
                    frames["design"],
                    tag=tag,
                    index=index,
                    count=count,
                    batch_dir=batch_dir,
                )
            )
        if not frames["long"].empty:
            long_frames.append(
                _with_batch_meta(
                    frames["long"],
                    tag=tag,
                    index=index,
                    count=count,
                    batch_dir=batch_dir,
                )
            )
        if not frames["reference"].empty:
            reference_frames.append(
                _with_batch_meta(
                    frames["reference"],
                    tag=tag,
                    index=index,
                    count=count,
                    batch_dir=batch_dir,
                )
            )
        manifest_rows.append(
            {
                "batch_index": index,
                "batch_count": count,
                "batch_tag": tag,
                "batch_dir": str(batch_dir),
                "status_rows": len(frames["status"]),
                "design_rows": len(frames["design"]),
                "country_strategy_rows": len(frames["long"]),
                "reference_rows": len(frames["reference"]),
            }
        )

    if not status_frames or not long_frames:
        raise RuntimeError("No usable S5.8 batch outputs were found.")

    status_all = pd.concat(status_frames, ignore_index=True)
    status_merged = _deduplicate_status(status_all)
    valid_ids = status_merged["scenario_id"].astype(str).tolist()
    design_all = (
        pd.concat(design_frames, ignore_index=True)
        if design_frames
        else pd.DataFrame()
    )
    design_merged = _deduplicate_design(design_all, valid_ids)
    long_all = pd.concat(long_frames, ignore_index=True)
    long_merged = _deduplicate_long(long_all)
    reference_all = (
        pd.concat(reference_frames, ignore_index=True)
        if reference_frames
        else pd.DataFrame()
    )
    consistency = _reference_consistency(
        reference_all,
        tolerance_gt=float(cfg.get("reference_tolerance_gt", 1e-9) or 1e-9),
    )

    max_selected, cost_selected = s58._select_country_strategies(long_merged)
    map_df = s58._build_map_data(long_merged, max_selected, cost_selected)

    reference_country_count = (
        int(reference_all["M49_Country_Code"].astype(str).nunique())
        if not reference_all.empty and "M49_Country_Code" in reference_all.columns
        else 0
    )
    merged_country_count = (
        int(long_merged["M49_Country_Code"].astype(str).nunique())
        if not long_merged.empty
        else 0
    )
    expected_country_count = reference_country_count or merged_country_count
    expected_rows = expected_country_count * len(s58.s57.STRATEGY_KIND_ORDER)
    usable_mask = long_merged["run_status"].astype(str).isin(["ok", "resumed"])
    usable_rows = int(usable_mask.sum())
    unusable = long_merged.loc[~usable_mask].copy()
    status_summary = (
        long_merged.assign(
            run_status=long_merged["run_status"].astype(str).replace("", "missing")
        )
        .groupby("run_status", dropna=False)
        .size()
        .rename("scenario_count")
        .reset_index()
        .sort_values(["scenario_count", "run_status"], ascending=[False, True])
    )
    selected_country_codes = set(
        max_selected.get("M49_Country_Code", pd.Series(dtype=str)).astype(str)
    )
    cost_country_codes = set(
        cost_selected.get("M49_Country_Code", pd.Series(dtype=str)).astype(str)
    )
    country_selection = (
        long_merged[
            [
                "M49_Country_Code",
                "ISO3",
                "country_name",
                "Region_aggMC",
            ]
        ]
        .drop_duplicates("M49_Country_Code")
        .copy()
    )
    country_selection["has_max_reduction_strategy"] = country_selection[
        "M49_Country_Code"
    ].astype(str).isin(selected_country_codes)
    country_selection["has_min_cost_strategy"] = country_selection[
        "M49_Country_Code"
    ].astype(str).isin(cost_country_codes)
    countries_without_selection = country_selection.loc[
        ~country_selection["has_max_reduction_strategy"]
        | ~country_selection["has_min_cost_strategy"]
    ].copy()
    completeness = pd.DataFrame(
        [
            {
                "batch_count": count,
                "reference_country_count": reference_country_count,
                "country_count": merged_country_count,
                "expected_country_count": expected_country_count,
                "strategy_count": len(s58.s57.STRATEGY_KIND_ORDER),
                "expected_country_strategy_rows": int(expected_rows),
                "actual_country_strategy_rows": int(len(long_merged)),
                "complete_country_strategy_grid": bool(
                    len(long_merged) == expected_rows
                ),
                "usable_country_strategy_rows": usable_rows,
                "complete_usable_country_strategy_grid": bool(
                    usable_rows == expected_rows
                ),
                "countries_with_max_reduction_strategy": int(len(max_selected)),
                "countries_with_min_cost_strategy": int(len(cost_selected)),
                "reference_consistency_failures": int(
                    (~consistency.get("within_tolerance", pd.Series(dtype=bool))).sum()
                )
                if not consistency.empty
                else 0,
            }
        ]
    )

    write_sensitivity_cost_summaries(
        status_merged,
        output_dir=merged_dir,
        run_search_root=root,
    )
    _write_csv(pd.DataFrame(manifest_rows), merged_dir / "batch_merge_manifest.csv")
    _write_csv(status_merged, merged_dir / "scenario_status.csv")
    _write_csv(design_merged, merged_dir / "strategy_design_long.csv")
    _write_csv(reference_all, merged_dir / "reference_country_emissions_all_batches.csv")
    _write_csv(consistency, merged_dir / "reference_consistency.csv")
    _write_csv(long_merged, merged_dir / "country_strategy_long.csv")
    _write_csv(max_selected, merged_dir / "country_max_reduction_strategy.csv")
    _write_csv(max_selected, merged_dir / "country_strategy_argmax.csv")
    _write_csv(cost_selected, merged_dir / "country_min_unit_cost_strategy.csv")
    _write_csv(cost_selected, merged_dir / "country_strategy_argmin_cost.csv")
    _write_csv(map_df, merged_dir / "sp_m3b_map_data.csv")
    _write_csv(map_df, merged_dir / "Figure3.csv")
    _write_csv(completeness, merged_dir / "merge_completeness.csv")
    _write_csv(status_summary, merged_dir / "run_status_summary.csv")
    _write_csv(
        unusable,
        merged_dir / "unusable_country_strategy_scenarios.csv",
    )
    _write_csv(
        countries_without_selection,
        merged_dir / "countries_without_selected_strategy.csv",
    )
    import S5_8_3_prepare_country_dominant_mitigation_intervention as figure3d

    figure3d.write_figure3d_outputs(
        long_merged,
        status_merged,
        output_dir=merged_dir / "figure3d",
        settings=figure3d.Figure3dSettings(
            input_dir=merged_dir,
            output_dir=merged_dir / "figure3d",
            metric="domestic",
            require_v2_provenance=True,
            strict=False,
            plot=False,
        ),
        source_files=(
            merged_dir / "country_strategy_long.csv",
            merged_dir / "scenario_status.csv",
        ),
    )

    grid_complete = bool(completeness.iloc[0]["complete_country_strategy_grid"])
    usable_complete = bool(
        completeness.iloc[0]["complete_usable_country_strategy_grid"]
    )
    reference_complete = (
        int(completeness.iloc[0]["reference_consistency_failures"]) == 0
    )
    status_text = ", ".join(
        f"{row.run_status}={int(row.scenario_count)}"
        for row in status_summary.itertuples(index=False)
    )
    if strict:
        if not grid_complete:
            raise RuntimeError(
                "Merged S5.8 country-strategy grid is incomplete. "
                f"Expected {expected_rows} rows but found {len(long_merged)}. "
                f"See {merged_dir / 'merge_completeness.csv'}"
            )
        if not usable_complete:
            raise RuntimeError(
                f"{len(unusable)} S5.8 country-strategy scenarios are not usable. "
                f"Status counts: {status_text}. See "
                f"{merged_dir / 'unusable_country_strategy_scenarios.csv'}"
            )
        if not reference_complete:
            raise RuntimeError(
                "Repeated reference scenarios differ beyond tolerance. "
                f"See {merged_dir / 'reference_consistency.csv'}"
            )
    elif not grid_complete or not usable_complete or not reference_complete:
        print(
            "[S5_8_MERGE] warning: merged outputs contain incomplete or unusable "
            f"scenarios. Status counts: {status_text}. See merge_completeness.csv, "
            "run_status_summary.csv, and unusable_country_strategy_scenarios.csv. "
            "Rerun with --strict after repairing failed batches."
        )

    print(
        "[S5_8_MERGE] "
        f"batches={count} country_strategy_rows={len(long_merged)} "
        f"max_selected={len(max_selected)} cost_selected={len(cost_selected)} "
        f"figure3d_source={merged_dir / 'figure3d'} merged_dir={merged_dir}"
    )


if __name__ == "__main__":
    main()
