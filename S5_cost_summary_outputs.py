# -*- coding: utf-8 -*-
"""Shared S5 collection of per-scenario mitigation-cost summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import pandas as pd

from S4_1_results import (
    COUNTRY_MEASURE_COST_SUMMARY_COLUMNS,
    COUNTRY_MEASURE_COST_SUMMARY_FILENAME,
    GLOBAL_MEASURE_COST_SUMMARY_COLUMNS,
    GLOBAL_MEASURE_COST_SUMMARY_FILENAME,
    write_measure_cost_summaries,
)


SENSITIVITY_COUNTRY_MEASURE_COST_FILENAME = (
    "sensitivity_cost_summary_by_country_measure.csv"
)
SENSITIVITY_GLOBAL_MEASURE_COST_FILENAME = (
    "sensitivity_cost_summary_by_global_measure.csv"
)
SENSITIVITY_COST_AUDIT_FILENAME = "sensitivity_cost_summary_audit.csv"

_COMPLETED_STATUSES = {"ok", "resumed", "valid", "completed"}
_STATUS_METADATA_COLUMNS = (
    "sample_id",
    "task_id",
    "scope",
    "strategy_name",
    "include_kinds",
    "country",
    "variable",
    "rate_value",
    "level_tag",
    "panel_row",
    "panel_col",
    "forest_area_change_pct",
    "ruminant_intake_cap_pct",
    "ruminant_intake_change_pct",
    "yield_change_pct",
    "emission_factor_change_pct",
    "target_year",
    "batch_tag",
    "batch_index",
    "batch_count",
)


def _status_column(status_df: pd.DataFrame) -> Optional[str]:
    for column in ("run_status", "status"):
        if column in status_df.columns:
            return column
    return None


def _candidate_scenario_dirs(
    row: Mapping[str, object],
    *,
    output_dir: Path,
    run_search_root: Path,
) -> Tuple[Path, ...]:
    scenario_id = str(row.get("scenario_id", "") or "").strip()
    batch_tag = str(row.get("batch_tag", "") or "").strip()
    candidates = []
    explicit = str(row.get("scenario_dir", "") or "").strip()
    if explicit:
        candidates.append(Path(explicit))
    if scenario_id:
        candidates.extend(
            [
                run_search_root / "runs" / scenario_id,
                output_dir / "runs" / scenario_id,
            ]
        )
        if batch_tag:
            candidates.append(
                run_search_root / "batches" / batch_tag / "runs" / scenario_id
            )
    unique = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return tuple(unique)


def _resolve_scenario_dir(
    row: Mapping[str, object],
    *,
    output_dir: Path,
    run_search_root: Path,
) -> Optional[Path]:
    for candidate in _candidate_scenario_dirs(
        row,
        output_dir=output_dir,
        run_search_root=run_search_root,
    ):
        if (candidate / "cost_summary.csv").exists():
            return candidate
    return None


def _read_or_build_run_summaries(
    scenario_dir: Path,
    *,
    scenario_id: str,
    dict_v3_path: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    country_path = scenario_dir / COUNTRY_MEASURE_COST_SUMMARY_FILENAME
    global_path = scenario_dir / GLOBAL_MEASURE_COST_SUMMARY_FILENAME
    if not country_path.exists() or not global_path.exists():
        detail_path = scenario_dir / "cost_summary.csv"
        detail = pd.read_csv(detail_path) if detail_path.exists() else pd.DataFrame()
        write_measure_cost_summaries(
            detail,
            output_dir=scenario_dir,
            scenario_id=scenario_id,
            dict_v3_path=dict_v3_path,
        )
    return pd.read_csv(country_path), pd.read_csv(global_path)


def _stamp_sensitivity_metadata(
    summary: pd.DataFrame,
    row: Mapping[str, object],
    *,
    scenario_dir: Path,
    status_column: Optional[str],
) -> pd.DataFrame:
    out = summary.copy()
    scenario_id = str(row.get("scenario_id", "") or "").strip()
    if "scenario_id" not in out.columns:
        out.insert(0, "scenario_id", scenario_id)
    else:
        out["scenario_id"] = out["scenario_id"].fillna("").astype(str)
        out.loc[out["scenario_id"].str.strip() == "", "scenario_id"] = scenario_id
    out.insert(1, "source_scenario_dir", str(scenario_dir))
    status = str(row.get(status_column, "") or "") if status_column else ""
    out.insert(2, "sensitivity_status", status)
    insert_at = 3
    for column in _STATUS_METADATA_COLUMNS:
        if column not in row:
            continue
        output_column = f"sensitivity_{column}"
        out.insert(insert_at, output_column, row.get(column))
        insert_at += 1
    return out


def _deduplicate_sensitivity_rows(df: pd.DataFrame, *, country_level: bool) -> pd.DataFrame:
    if df.empty:
        return df
    key_columns = [
        "scenario_id",
        "year",
        "database_strategy",
        "strategy_kind",
        "cost_database_version",
        "cost_database_sha256",
        "reference_scenario_id",
    ]
    if country_level:
        key_columns.insert(1, "region")
    existing_keys = [column for column in key_columns if column in df.columns]
    if existing_keys:
        df = df.drop_duplicates(subset=existing_keys, keep="last")
    sort_columns = [
        column
        for column in ("scenario_id", "year", "database_strategy", "region")
        if column in df.columns
    ]
    return df.sort_values(sort_columns, kind="stable").reset_index(drop=True)


def write_sensitivity_cost_summaries(
    status_df: pd.DataFrame,
    *,
    output_dir: str | Path,
    run_search_root: str | Path | None = None,
    dict_v3_path: Optional[str] = None,
) -> Dict[str, Path]:
    """Collect valid S5 scenario summaries into experiment-level CSV files."""
    target_dir = Path(output_dir)
    search_root = Path(run_search_root) if run_search_root else target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    status_column = _status_column(status_df)
    country_frames = []
    global_frames = []
    audit_rows = []

    for row in status_df.to_dict("records"):
        scenario_id = str(row.get("scenario_id", "") or "").strip()
        status = str(row.get(status_column, "") or "").strip().lower() if status_column else ""
        if status_column and status not in _COMPLETED_STATUSES:
            audit_rows.append(
                {
                    "scenario_id": scenario_id,
                    "status": status,
                    "scenario_dir": str(row.get("scenario_dir", "") or ""),
                    "cost_summary_found": False,
                    "country_measure_rows": 0,
                    "global_measure_rows": 0,
                    "reason": "status_not_completed",
                }
            )
            continue
        scenario_dir = _resolve_scenario_dir(
            row,
            output_dir=target_dir,
            run_search_root=search_root,
        )
        if scenario_dir is None:
            audit_rows.append(
                {
                    "scenario_id": scenario_id,
                    "status": status,
                    "scenario_dir": str(row.get("scenario_dir", "") or ""),
                    "cost_summary_found": False,
                    "country_measure_rows": 0,
                    "global_measure_rows": 0,
                    "reason": "cost_summary_missing",
                }
            )
            continue

        country, global_summary = _read_or_build_run_summaries(
            scenario_dir,
            scenario_id=scenario_id,
            dict_v3_path=dict_v3_path,
        )
        if not country.empty:
            country_frames.append(
                _stamp_sensitivity_metadata(
                    country,
                    row,
                    scenario_dir=scenario_dir,
                    status_column=status_column,
                )
            )
        if not global_summary.empty:
            global_frames.append(
                _stamp_sensitivity_metadata(
                    global_summary,
                    row,
                    scenario_dir=scenario_dir,
                    status_column=status_column,
                )
            )
        audit_rows.append(
            {
                "scenario_id": scenario_id,
                "status": status,
                "scenario_dir": str(scenario_dir),
                "cost_summary_found": True,
                "country_measure_rows": int(len(country)),
                "global_measure_rows": int(len(global_summary)),
                "reason": "" if not country.empty or not global_summary.empty else "no_measure_rows",
            }
        )

    country_columns = [
        "source_scenario_dir",
        "sensitivity_status",
        *COUNTRY_MEASURE_COST_SUMMARY_COLUMNS,
    ]
    global_columns = [
        "source_scenario_dir",
        "sensitivity_status",
        *GLOBAL_MEASURE_COST_SUMMARY_COLUMNS,
    ]
    country_all = (
        pd.concat(country_frames, ignore_index=True, sort=False)
        if country_frames
        else pd.DataFrame(columns=country_columns)
    )
    global_all = (
        pd.concat(global_frames, ignore_index=True, sort=False)
        if global_frames
        else pd.DataFrame(columns=global_columns)
    )
    country_all = _deduplicate_sensitivity_rows(country_all, country_level=True)
    global_all = _deduplicate_sensitivity_rows(global_all, country_level=False)

    country_path = target_dir / SENSITIVITY_COUNTRY_MEASURE_COST_FILENAME
    global_path = target_dir / SENSITIVITY_GLOBAL_MEASURE_COST_FILENAME
    audit_path = target_dir / SENSITIVITY_COST_AUDIT_FILENAME
    country_all.to_csv(country_path, index=False, encoding="utf-8-sig")
    global_all.to_csv(global_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        audit_rows,
        columns=[
            "scenario_id",
            "status",
            "scenario_dir",
            "cost_summary_found",
            "country_measure_rows",
            "global_measure_rows",
            "reason",
        ],
    ).to_csv(audit_path, index=False, encoding="utf-8-sig")
    return {
        "country_measure": country_path,
        "global_measure": global_path,
        "audit": audit_path,
    }

