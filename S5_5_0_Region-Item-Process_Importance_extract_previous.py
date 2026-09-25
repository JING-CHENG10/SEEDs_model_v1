# -*- coding: utf-8 -*-
"""
Quantify which Region / Item / Process structures matter most for GHG
reduction under different emission-intensity levels.

This script first audits the designed outputs of S5_1 / S5_3 / S5_4 and
then decides whether existing result folders are sufficient for the target
task:
  - S5_1 (MC_Sensitivity_Variable_Effect): usually usable directly because
    runs keep emissions_summary_By_Country_Process_Item when
    fast_emis_only=False.
  - S5_3 (Panel_Yield_EF): designed outputs only keep global fast summaries
    and global process/item detail, so Region / Region-Item /
    Region-Process cannot be rebuilt directly.
  - S5_4 (MC_Full_Variables): designed outputs only keep global totals,
    global process totals, and sampled input-weight tables; Region / Item
    output structures are not preserved when fast_emis_only=True.

If an eligible S5_1 source is found, the script:
1) Reads VE_emission_factor_* runs.
2) Rebuilds the same Region / Item / Process aggregation used in
   SP_M1a_Figure_pie_structure_pre.py.
3) Pairs each emission-intensity level with the same-sample p00 run.
4) Defines group reduction as:
      reduction = emissions_current - emissions_level
   and importance share within each paired run as:
      max(reduction, 0) / sum(max(reduction, 0))

Outputs:
- source_assessment.csv
- scenario_manifest.csv
- aggregated_scenario_emissions.csv
- scenario_totals.csv
- pair_summary.csv
- importance_detail.csv
- EmissionIntensity_Importance.xlsx
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from config_paths import get_results_base
from SP_M1a_Figure_pie_structure_pre import (
    AGGREGATE_M49_CODES,
    AGGREGATE_REGION_LABELS,
    _clean_group_col,
    _load_emis_item_maps,
    _load_region_emis_sum_map,
    _normalize_code,
    _pick_year_col,
)


CONFIG = {
    "results_base_dir": "",
    "input_root": "",
    "output_root": "",
    "target_variable": "emission_factor",
    "reference_level_tag": "p00",
    "reference_rate_value": 0.0,
    "target_year": 2080,
    "unit_scale_gt": 1e-6,  # emissions_summary_By_Country_Process_Item is in kt
    "reuse_aggregate_cache": False,
    "source_mode": "auto",  # auto | S5_1 | S5_3 | S5_4
    "source_priority": ["S5_1", "S5_3", "S5_4"],
    "source_dirs": {
        "S5_1": "",
        "S5_3": "",
        "S5_4": "",
    },
    "batches_subdir": "batches",
    "allow_partial_pairs": True,
}

INPUT_CANDIDATES = (
    "emissions_summary_By_Country_Process_Item.csv",
    "emissions_summary_By_Country_Process_Item.xlsx",
)
SCENARIO_PATTERN = re.compile(
    r"^VE_(?P<variable>.+)_(?P<level_tag>[mp]\d+)_(?P<sample_id>\d+)$",
    flags=re.IGNORECASE,
)
GROUPINGS: Dict[str, Tuple[str, ...]] = {
    "Region": ("Region_emisSum",),
    "Item": ("Item",),
    "Process": ("Process",),
    "Region-Item": ("Region_emisSum", "Item"),
    "Region-Process": ("Region_emisSum", "Process"),
}
GROUP_LABELS = {
    "Region_emisSum": "Region",
    "Item": "Item",
    "Process": "Process",
}
SOURCE_DEFAULT_DIRS = {
    "S5_1": "MC_Sensitivity_Variable_Effect",
    "S5_3": "Panel_Yield_EF",
    "S5_4": "MC_Full_Variables",
}


def _input_root() -> Path:
    if CONFIG.get("input_root"):
        return Path(str(CONFIG["input_root"]))
    return Path(get_results_base()) / "MC_Sensitivity_Variable_Effect"


def _output_root() -> Path:
    if CONFIG.get("output_root"):
        return Path(str(CONFIG["output_root"]))
    return _input_root() / "Structure_Importance"


def _results_base_dir() -> Path:
    if CONFIG.get("results_base_dir"):
        return Path(str(CONFIG["results_base_dir"]))
    return Path(get_results_base())


def _source_root(source_name: str) -> Path:
    source_key = str(source_name or "").strip()
    source_dirs = CONFIG.get("source_dirs", {}) or {}
    override = str(source_dirs.get(source_key, "") or "").strip()
    if override:
        return Path(override)
    if source_key == "S5_1" and CONFIG.get("input_root"):
        return Path(str(CONFIG["input_root"]))
    default_dir = SOURCE_DEFAULT_DIRS.get(source_key, source_key)
    return _results_base_dir() / default_dir


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _first_valid(values: Iterable[object], default: object = "") -> object:
    for val in values:
        if val is None:
            continue
        if isinstance(val, float) and pd.isna(val):
            continue
        text = str(val).strip()
        if text == "" or text.lower() == "nan":
            continue
        return val
    return default


def _parse_level_tag(level_tag: object) -> Optional[float]:
    text = str(level_tag or "").strip().lower()
    if not text:
        return None
    match = re.fullmatch(r"([mp])(\d+)", text)
    if not match:
        return None
    sign = 1.0 if match.group(1) == "p" else -1.0
    return sign * (float(match.group(2)) / 100.0)


def _parse_scenario_name(name: str) -> Dict[str, object]:
    match = SCENARIO_PATTERN.match(str(name or "").strip())
    if not match:
        return {}
    level_tag = str(match.group("level_tag")).lower()
    return {
        "scenario_id": str(name).strip(),
        "variable": str(match.group("variable")).strip(),
        "level_tag": level_tag,
        "rate_value": _parse_level_tag(level_tag),
        "sample_id": int(match.group("sample_id")),
    }


def _find_summary_path(run_dir: Path) -> Optional[Path]:
    emis_dir = run_dir / "Emis"
    for name in INPUT_CANDIDATES:
        path = emis_dir / name
        if path.exists():
            return path
    return None


def _batches_root(root: Path) -> Path:
    batches_subdir = str(CONFIG.get("batches_subdir", "batches") or "batches").strip() or "batches"
    return root / batches_subdir


def _iter_batch_dirs(root: Path) -> List[Path]:
    batches_root = _batches_root(root)
    if not batches_root.exists():
        return []
    return sorted(p for p in batches_root.iterdir() if p.is_dir())


def _batch_tag_for_path(batch_dir: Path) -> str:
    return batch_dir.name


def _load_samples_manifest(samples_path: Path) -> pd.DataFrame:
    columns = ["scenario_id", "variable", "level_tag", "rate_value", "sample_id", "emissions_2080_gt"]
    if not samples_path.exists():
        return pd.DataFrame(columns=columns)

    df = pd.read_csv(samples_path)
    if df.empty:
        return pd.DataFrame(columns=columns)

    df.columns = [str(c).strip() for c in df.columns]
    if "scenario_id" not in df.columns:
        raise KeyError(f"Missing scenario_id in samples manifest: {samples_path}")

    parsed = pd.DataFrame([_parse_scenario_name(v) for v in df["scenario_id"]], index=df.index)
    for col in ("variable", "level_tag", "rate_value", "sample_id"):
        if col not in df.columns:
            df[col] = parsed.get(col)

    df["scenario_id"] = df["scenario_id"].astype("string").str.strip()
    df["variable"] = df["variable"].fillna(parsed.get("variable")).astype("string").str.strip()
    df["level_tag"] = df["level_tag"].fillna(parsed.get("level_tag")).astype("string").str.strip().str.lower()
    df["rate_value"] = pd.to_numeric(df["rate_value"], errors="coerce")
    df["sample_id"] = pd.to_numeric(df["sample_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["scenario_id", "variable", "level_tag", "rate_value", "sample_id"]).copy()
    df["sample_id"] = df["sample_id"].astype(int)

    if "emissions_2080_gt" not in df.columns:
        df["emissions_2080_gt"] = np.nan
    df["emissions_2080_gt"] = pd.to_numeric(df["emissions_2080_gt"], errors="coerce")

    return df[columns].drop_duplicates().reset_index(drop=True)


def _load_all_samples_manifests(root: Path) -> pd.DataFrame:
    columns = [
        "scenario_id",
        "variable",
        "level_tag",
        "rate_value",
        "sample_id",
        "emissions_2080_gt",
        "batch_tag",
        "manifest_source",
    ]
    frames: List[pd.DataFrame] = []

    root_samples_path = root / "samples.csv"
    root_samples = _load_samples_manifest(root_samples_path)
    if not root_samples.empty:
        root_samples["batch_tag"] = ""
        root_samples["manifest_source"] = str(root_samples_path)
        frames.append(root_samples)

    for batch_dir in _iter_batch_dirs(root):
        batch_tag = _batch_tag_for_path(batch_dir)
        for samples_path in (batch_dir / "summary" / "samples.csv", batch_dir / "samples.csv"):
            samples_df = _load_samples_manifest(samples_path)
            if samples_df.empty:
                continue
            samples_df["batch_tag"] = batch_tag
            samples_df["manifest_source"] = str(samples_path)
            frames.append(samples_df)
            break

    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True, sort=False)[columns]


def _scan_runs_manifest(runs_dir: Path, batch_tag: str = "") -> pd.DataFrame:
    columns = [
        "scenario_id",
        "variable",
        "level_tag",
        "rate_value",
        "sample_id",
        "run_dir",
        "summary_path",
        "emissions_2080_gt",
        "batch_tag",
        "manifest_source",
    ]
    rows: List[Dict[str, object]] = []
    if not runs_dir.exists():
        return pd.DataFrame(columns=columns)

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        parsed = _parse_scenario_name(run_dir.name)
        if not parsed:
            continue
        summary_path = _find_summary_path(run_dir)
        parsed["run_dir"] = str(run_dir)
        parsed["summary_path"] = str(summary_path) if summary_path else ""
        parsed["emissions_2080_gt"] = np.nan
        parsed["batch_tag"] = str(batch_tag or "")
        parsed["manifest_source"] = ""
        rows.append(parsed)

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns]


def _scan_all_runs_manifest(root: Path) -> pd.DataFrame:
    frames = [_scan_runs_manifest(root / "runs", batch_tag="")]
    for batch_dir in _iter_batch_dirs(root):
        frames.append(_scan_runs_manifest(batch_dir / "runs", batch_tag=_batch_tag_for_path(batch_dir)))
    return pd.concat(frames, ignore_index=True, sort=False)


def _default_run_dir_for_manifest_row(root: Path, row: pd.Series) -> str:
    scenario_id = str(row.get("scenario_id", "") or "").strip()
    batch_tag = str(row.get("batch_tag", "") or "").strip()
    if batch_tag:
        return str(_batches_root(root) / batch_tag / "runs" / scenario_id)
    return str(root / "runs" / scenario_id)


def _build_scenario_manifest(root: Path, variable: str) -> pd.DataFrame:
    samples_df = _load_all_samples_manifests(root)
    scans_df = _scan_all_runs_manifest(root)

    if samples_df.empty and scans_df.empty:
        raise FileNotFoundError(f"No samples.csv or VE run directories found under root/batches: {root}")

    combined = pd.concat([samples_df, scans_df], ignore_index=True, sort=False)
    if combined.empty:
        return combined

    if "run_dir" not in combined.columns:
        combined["run_dir"] = ""
    if "summary_path" not in combined.columns:
        combined["summary_path"] = ""
    if "emissions_2080_gt" not in combined.columns:
        combined["emissions_2080_gt"] = np.nan
    if "batch_tag" not in combined.columns:
        combined["batch_tag"] = ""
    if "manifest_source" not in combined.columns:
        combined["manifest_source"] = ""

    combined["scenario_id"] = combined["scenario_id"].astype("string").str.strip()
    combined["variable"] = combined["variable"].astype("string").str.strip()
    combined["level_tag"] = combined["level_tag"].astype("string").str.strip().str.lower()
    combined["rate_value"] = pd.to_numeric(combined["rate_value"], errors="coerce")
    combined["sample_id"] = pd.to_numeric(combined["sample_id"], errors="coerce").astype("Int64")
    combined["emissions_2080_gt"] = pd.to_numeric(combined["emissions_2080_gt"], errors="coerce")
    combined["batch_tag"] = combined["batch_tag"].astype("string").fillna("").str.strip()
    combined["manifest_source"] = combined["manifest_source"].astype("string").fillna("").str.strip()
    combined = combined.dropna(subset=["scenario_id", "variable", "level_tag", "rate_value", "sample_id"]).copy()
    combined["sample_id"] = combined["sample_id"].astype(int)

    missing_run_dir = combined["run_dir"].astype("string").str.strip().isin({"", "nan"})
    combined.loc[missing_run_dir, "run_dir"] = combined.loc[missing_run_dir].apply(
        lambda row: _default_run_dir_for_manifest_row(root, row),
        axis=1,
    )

    missing_summary = combined["summary_path"].astype("string").str.strip().isin({"", "nan"})
    if missing_summary.any():
        combined.loc[missing_summary, "summary_path"] = combined.loc[missing_summary, "run_dir"].map(
            lambda p: str(_find_summary_path(Path(p)) or "")
        )

    combined["has_summary"] = combined["summary_path"].astype("string").fillna("").str.strip().ne("")
    combined["priority"] = combined["has_summary"].astype(int)
    combined = combined.sort_values(
        ["scenario_id", "priority", "sample_id"],
        ascending=[True, True, True],
        kind="mergesort",
    )
    manifest = combined.drop_duplicates(subset=["scenario_id"], keep="last").copy()
    manifest = manifest[manifest["variable"].str.casefold() == str(variable).casefold()].copy()
    manifest["summary_path"] = manifest["summary_path"].astype("string").fillna("").str.strip()
    manifest["run_dir"] = manifest["run_dir"].astype("string").fillna("").str.strip()
    manifest["has_summary"] = manifest["summary_path"].fillna("").ne("")

    manifest = _attach_run_diagnostics(manifest)

    ordered_cols = [
        "scenario_id",
        "variable",
        "level_tag",
        "rate_value",
        "sample_id",
        "batch_tag",
        "emissions_2080_gt",
        "has_summary",
        "run_status_guess",
        "model_status_code",
        "model_status_text",
        "error_message_hint",
        "manifest_source",
        "run_dir",
        "summary_path",
    ]
    return manifest[ordered_cols].sort_values(
        ["rate_value", "sample_id", "scenario_id"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _read_model_log_diagnostics(run_dir: Path) -> Dict[str, object]:
    diag: Dict[str, object] = {
        "run_status_guess": "",
        "model_status_code": np.nan,
        "model_status_text": "",
        "error_message_hint": "",
    }
    log_path = run_dir / "Log" / "model.log"
    if not log_path.exists():
        return diag
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            text = log_path.read_text(encoding="gbk", errors="ignore")
        except Exception:
            return diag

    status_matches = re.findall(r"status=(\d+)", text)
    status_code = int(status_matches[-1]) if status_matches else None
    if status_code is not None:
        diag["model_status_code"] = status_code
    status_map = {
        1: "LOADED",
        2: "OPTIMAL",
        3: "INFEASIBLE",
        4: "INF_OR_UNBD",
        5: "UNBOUNDED",
        7: "ITERATION_LIMIT",
        8: "NODE_LIMIT",
        9: "TIME_LIMIT",
        10: "SOLUTION_LIMIT",
        11: "INTERRUPTED",
        12: "NUMERIC",
        13: "SUBOPTIMAL",
        16: "WORK_LIMIT",
        17: "MEM_LIMIT",
    }
    infeasible_hit = bool(re.search(r"\bInfeasible model\b", text, flags=re.IGNORECASE))
    if infeasible_hit or status_code == 3:
        diag["run_status_guess"] = "infeasible"
        diag["model_status_text"] = "INFEASIBLE"
        diag["error_message_hint"] = "Infeasible model"
        return diag
    if status_code is not None and status_code != 2:
        diag["run_status_guess"] = "nonoptimal"
        diag["model_status_text"] = status_map.get(status_code, f"STATUS_{status_code}")
        diag["error_message_hint"] = diag["model_status_text"]
        return diag
    if status_code == 2:
        diag["run_status_guess"] = "optimal"
        diag["model_status_text"] = "OPTIMAL"
    return diag


def _attach_run_diagnostics(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        out = manifest.copy()
        for col in ("run_status_guess", "model_status_code", "model_status_text", "error_message_hint"):
            out[col] = pd.Series(dtype="object")
        return out

    out = manifest.copy()
    diag_rows: List[Dict[str, object]] = []
    for row in out.itertuples(index=False):
        run_dir_text = str(getattr(row, "run_dir", "") or "").strip()
        run_dir = Path(run_dir_text) if run_dir_text else None
        diag = _read_model_log_diagnostics(run_dir) if run_dir is not None else {}
        emissions_val = getattr(row, "emissions_2080_gt", np.nan)
        has_summary = bool(getattr(row, "has_summary", False))
        if has_summary:
            run_status_guess = "ok"
        elif pd.notna(emissions_val):
            run_status_guess = "has_total_only"
        else:
            run_status_guess = str(diag.get("run_status_guess", "") or "missing_emis")
        diag_rows.append(
            {
                "scenario_id": getattr(row, "scenario_id"),
                "run_status_guess": run_status_guess,
                "model_status_code": diag.get("model_status_code", np.nan),
                "model_status_text": diag.get("model_status_text", ""),
                "error_message_hint": diag.get("error_message_hint", ""),
            }
        )
    diag_df = pd.DataFrame(diag_rows)
    out = out.merge(diag_df, on="scenario_id", how="left")
    return out


def _count_country_process_item_files(root: Path, limit: Optional[int] = None) -> int:
    count = 0
    run_roots = [root / "runs"] + [batch_dir / "runs" for batch_dir in _iter_batch_dirs(root)]
    for runs_dir in run_roots:
        if not runs_dir.exists():
            continue
        for path in runs_dir.rglob("emissions_summary_By_Country_Process_Item.csv"):
            count += 1
            if limit is not None and count >= limit:
                return count
        for path in runs_dir.rglob("emissions_summary_By_Country_Process_Item.xlsx"):
            count += 1
            if limit is not None and count >= limit:
                return count
    return count


def _pair_coverage_stats(manifest: pd.DataFrame) -> Dict[str, object]:
    if manifest.empty:
        return {
            "expected_pairs": 0,
            "available_pairs": 0,
            "missing_pairs": 0,
            "levels_in_design": "",
            "levels_with_summary": "",
            "n_reference_runs": 0,
            "n_target_runs": 0,
        }

    work = manifest.copy()
    work["has_summary"] = work["has_summary"].fillna(False).astype(bool)
    work["is_reference"] = work.apply(_is_reference_row, axis=1)
    work["pair_key"] = list(zip(work["variable"].astype(str), work["sample_id"].astype(int)))

    levels_in_design = ",".join(
        sorted({str(x).strip() for x in work["level_tag"].dropna().astype(str) if str(x).strip()})
    )
    levels_with_summary = ",".join(
        sorted(
            {
                str(x).strip()
                for x in work.loc[work["has_summary"], "level_tag"].dropna().astype(str)
                if str(x).strip()
            }
        )
    )

    ref_keys_design = set(work.loc[work["is_reference"], "pair_key"])
    ref_keys_summary = set(work.loc[work["is_reference"] & work["has_summary"], "pair_key"])
    target_design = work.loc[~work["is_reference"]].copy()
    target_design["has_ref_design"] = target_design["pair_key"].isin(ref_keys_design)
    target_design["has_ref_summary"] = target_design["pair_key"].isin(ref_keys_summary)
    target_design["pair_available"] = target_design["has_summary"] & target_design["has_ref_summary"]

    expected_pairs = int(target_design["has_ref_design"].sum())
    available_pairs = int(target_design["pair_available"].sum())
    missing_pairs = max(0, expected_pairs - available_pairs)

    return {
        "expected_pairs": expected_pairs,
        "available_pairs": available_pairs,
        "missing_pairs": missing_pairs,
        "levels_in_design": levels_in_design,
        "levels_with_summary": levels_with_summary,
        "n_reference_runs": int(work["is_reference"].sum()),
        "n_target_runs": int((~work["is_reference"]).sum()),
    }


def _build_coverage_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return pd.DataFrame()

    work = manifest.copy()
    work["has_summary"] = work["has_summary"].fillna(False).astype(bool)
    work["is_reference"] = work.apply(_is_reference_row, axis=1)
    work["pair_key"] = list(zip(work["variable"].astype(str), work["sample_id"].astype(int)))
    ref_summary_keys = set(work.loc[work["is_reference"] & work["has_summary"], "pair_key"])
    work["has_reference_summary"] = work["pair_key"].isin(ref_summary_keys)
    work["pair_available"] = (~work["is_reference"]) & work["has_summary"] & work["has_reference_summary"]

    def _count_status(df: pd.DataFrame, status: str) -> int:
        if "run_status_guess" not in df.columns:
            return 0
        return int(df["run_status_guess"].astype("string").str.strip().str.lower().eq(status).sum())

    summary = (
        work.groupby(["level_tag", "rate_value"], as_index=False)
        .agg(
            designed_runs=("scenario_id", "count"),
            runs_with_detail=("has_summary", "sum"),
            paired_runs=("pair_available", "sum"),
            reference_success_available=("has_reference_summary", "sum"),
        )
    )
    summary["designed_runs"] = summary["designed_runs"].astype(int)
    summary["runs_with_detail"] = summary["runs_with_detail"].astype(int)
    summary["paired_runs"] = summary["paired_runs"].astype(int)
    summary["reference_success_available"] = summary["reference_success_available"].astype(int)
    summary["detail_success_rate"] = np.where(
        summary["designed_runs"] > 0,
        summary["runs_with_detail"] / summary["designed_runs"],
        np.nan,
    )
    summary["paired_rate"] = np.where(
        summary["designed_runs"] > 0,
        summary["paired_runs"] / summary["designed_runs"],
        np.nan,
    )

    extras: List[Dict[str, object]] = []
    for (level_tag, rate_value), grp in work.groupby(["level_tag", "rate_value"], dropna=False):
        extras.append(
            {
                "level_tag": level_tag,
                "rate_value": rate_value,
                "ok_runs": _count_status(grp, "ok"),
                "optimal_no_detail_runs": _count_status(grp, "optimal"),
                "infeasible_runs": _count_status(grp, "infeasible"),
                "nonoptimal_runs": _count_status(grp, "nonoptimal"),
                "missing_emis_runs": _count_status(grp, "missing_emis"),
                "has_total_only_runs": _count_status(grp, "has_total_only"),
            }
        )
    extras_df = pd.DataFrame(extras)
    if not extras_df.empty:
        summary = summary.merge(extras_df, on=["level_tag", "rate_value"], how="left")
    return summary.sort_values(["rate_value", "level_tag"], kind="mergesort").reset_index(drop=True)


def _build_failed_runs_audit(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return pd.DataFrame()
    work = manifest.copy()
    work["has_summary"] = work["has_summary"].fillna(False).astype(bool)
    failed = work.loc[~work["has_summary"]].copy()
    if failed.empty:
        return failed
    ordered_cols = [
        "scenario_id",
        "variable",
        "level_tag",
        "rate_value",
        "sample_id",
        "emissions_2080_gt",
        "run_status_guess",
        "model_status_code",
        "model_status_text",
        "error_message_hint",
        "run_dir",
        "summary_path",
    ]
    existing_cols = [c for c in ordered_cols if c in failed.columns]
    return failed[existing_cols].sort_values(
        ["rate_value", "sample_id", "scenario_id"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _assess_s5_1_source(root: Path, variable: str) -> Dict[str, object]:
    row: Dict[str, object] = {
        "source_name": "S5_1",
        "root": str(root),
        "root_exists": root.exists(),
        "designed_outputs_sufficient": True,
        "supports_target_directly": False,
        "direct_use_recommended": False,
        "needs_code_change": False,
        "needs_rerun_for_full_result": False,
        "reason": "",
        "expected_scenarios": 0,
        "scenarios_with_summary": 0,
        "missing_summary_scenarios": 0,
        "expected_pairs": 0,
        "available_pairs": 0,
        "missing_pairs": 0,
        "levels_in_design": "",
        "levels_with_summary": "",
        "country_process_item_files": 0,
    }
    if not root.exists():
        row["reason"] = "结果目录不存在。"
        return row

    try:
        manifest = _build_scenario_manifest(root, variable)
    except Exception as exc:
        row["reason"] = f"读取 VE manifest 失败: {type(exc).__name__}: {exc}"
        return row

    if manifest.empty:
        row["reason"] = "未找到 VE_emission_factor_* 结果。"
        return row

    row["expected_scenarios"] = int(len(manifest))
    row["scenarios_with_summary"] = int(manifest["has_summary"].sum())
    row["missing_summary_scenarios"] = int((~manifest["has_summary"]).sum())
    row["country_process_item_files"] = int(manifest["has_summary"].sum())

    stats = _pair_coverage_stats(manifest)
    row.update(stats)

    row["supports_target_directly"] = bool(row["available_pairs"] > 0)
    row["direct_use_recommended"] = bool(row["available_pairs"] > 0)
    row["needs_rerun_for_full_result"] = bool(row["missing_pairs"] > 0 or row["missing_summary_scenarios"] > 0)

    if row["available_pairs"] <= 0:
        row["reason"] = (
            "S5_1 设计上足够，但当前缺少可配对的 p00 与目标 emission_factor 场景，"
            "或缺少对应的 emissions_summary_By_Country_Process_Item。"
        )
    elif row["needs_rerun_for_full_result"]:
        row["reason"] = (
            "S5_1 可直接用于部分分析，但要得到完整结果仍需补跑缺失的 VE 场景或缺失的详细 Emis 输出。"
        )
    else:
        row["reason"] = (
            "S5_1 结果可直接使用：按 sample_id 可与 p00 配对，且保留了 By_Country_Process_Item 详细排放。"
        )
    return row


def _assess_s5_3_source(root: Path) -> Dict[str, object]:
    results_csv = root / "figure_panel_dataset_long.csv"
    detail_csv = root / "figure_panel_global_emissions_detail_long.csv"
    row: Dict[str, object] = {
        "source_name": "S5_3",
        "root": str(root),
        "root_exists": root.exists(),
        "designed_outputs_sufficient": False,
        "supports_target_directly": False,
        "direct_use_recommended": False,
        "needs_code_change": True,
        "needs_rerun_for_full_result": root.exists(),
        "reason": "",
        "expected_scenarios": 0,
        "scenarios_with_summary": 0,
        "missing_summary_scenarios": 0,
        "expected_pairs": 0,
        "available_pairs": 0,
        "missing_pairs": 0,
        "levels_in_design": "",
        "levels_with_summary": "",
        "country_process_item_files": _count_country_process_item_files(root, limit=1000000),
    }
    if not root.exists():
        row["reason"] = "结果目录不存在。"
        return row

    run_count = len([p for p in (root / "runs").iterdir()]) if (root / "runs").exists() else 0
    row["expected_scenarios"] = int(run_count)
    row["scenarios_with_summary"] = int(run_count)

    has_panel = results_csv.exists()
    has_detail = detail_csv.exists()
    if row["country_process_item_files"] > 0:
        row["reason"] = (
            "S5_3 默认设计输出只有全球长表；若服务器运行中额外保留了 By_Country_Process_Item，"
            "可另写专门后处理，但当前 S5_5 不以 S5_3 为主源。"
        )
    elif has_panel and has_detail:
        row["reason"] = (
            "S5_3 只能直接提供全球总排放和全球 Process/Item 明细，缺少 Region 层，"
            "也缺少 Region-Item / Region-Process 所需的国家级详细输出。若坚持用 S5_3，需改代码并重跑。"
        )
    else:
        row["reason"] = "S5_3 输出不完整，且即便完整也不足以直接支撑 Region 相关重要性分解。"
    return row


def _assess_s5_4_source(root: Path) -> Dict[str, object]:
    merged_dir = root / "merged"
    summary_csv = merged_dir / "mc_success_fast_summary.csv"
    process_csv = merged_dir / "mc_success_global_process_co2eq.csv"
    weighted_csv = merged_dir / "mc_success_weighted_elements.csv"
    if not merged_dir.exists():
        summary_csv = root / "mc_success_fast_summary.csv"
        process_csv = root / "mc_success_global_process_co2eq.csv"
        weighted_csv = root / "mc_success_weighted_elements.csv"

    row: Dict[str, object] = {
        "source_name": "S5_4",
        "root": str(root),
        "root_exists": root.exists(),
        "designed_outputs_sufficient": False,
        "supports_target_directly": False,
        "direct_use_recommended": False,
        "needs_code_change": True,
        "needs_rerun_for_full_result": root.exists(),
        "reason": "",
        "expected_scenarios": 0,
        "scenarios_with_summary": 0,
        "missing_summary_scenarios": 0,
        "expected_pairs": 0,
        "available_pairs": 0,
        "missing_pairs": 0,
        "levels_in_design": "",
        "levels_with_summary": "",
        "country_process_item_files": _count_country_process_item_files(root, limit=1000000),
    }
    if not root.exists():
        row["reason"] = "结果目录不存在。"
        return row

    status_candidates = [
        root / "mc_sample_status.csv",
        merged_dir / "mc_sample_status.csv",
    ]
    status_path = next((p for p in status_candidates if p.exists()), None)
    if status_path is not None:
        try:
            status_df = pd.read_csv(status_path)
            row["expected_scenarios"] = int(len(status_df))
            if "run_status" in status_df.columns:
                status_text = status_df["run_status"].astype("string").str.strip().str.lower()
                row["scenarios_with_summary"] = int(status_text.isin({"ok", "resumed"}).sum())
                row["missing_summary_scenarios"] = int(len(status_df) - row["scenarios_with_summary"])
        except Exception:
            pass

    has_merged = summary_csv.exists() and process_csv.exists() and weighted_csv.exists()
    if row["country_process_item_files"] > 0:
        row["reason"] = (
            "S5_4 默认 merged 输出不够，但如果服务器运行中额外保留了 By_Country_Process_Item，"
            "可以另写专门后处理。当前代码设计下 fast_emis_only=True，通常仍需要重跑才能系统化导出。"
        )
    elif has_merged:
        row["reason"] = (
            "S5_4 merged 输出只有全球总排放、全球 Process 排放和输入变量加权值；"
            "没有 Region、Item、Region-Item、Region-Process 所需的输出侧结构明细。"
            "若坚持用 S5_4，需先改 S5_4 导出 country-process-item 排放并重跑。"
        )
    else:
        row["reason"] = "S5_4 输出不完整，且按现有设计也不足以直接支撑 Region/Item/Process 结构重要性分解。"
    return row


def _assess_sources() -> pd.DataFrame:
    variable = str(CONFIG.get("target_variable", "emission_factor") or "emission_factor")
    rows = [
        _assess_s5_1_source(_source_root("S5_1"), variable),
        _assess_s5_3_source(_source_root("S5_3")),
        _assess_s5_4_source(_source_root("S5_4")),
    ]
    return pd.DataFrame(rows)


def _select_source(assessment_df: pd.DataFrame) -> Optional[Dict[str, object]]:
    if assessment_df.empty:
        return None
    source_mode = str(CONFIG.get("source_mode", "auto") or "auto").strip()
    if source_mode and source_mode.lower() != "auto":
        forced = assessment_df.loc[assessment_df["source_name"] == source_mode].copy()
        if forced.empty:
            raise ValueError(f"Unknown source_mode: {source_mode}")
        row = forced.iloc[0].to_dict()
        if not bool(row.get("supports_target_directly", False)):
            raise RuntimeError(
                f"{source_mode} cannot directly support the target analysis. Reason: {row.get('reason', '')}"
            )
        return row

    priorities = [str(x).strip() for x in (CONFIG.get("source_priority") or []) if str(x).strip()]
    if not priorities:
        priorities = ["S5_1", "S5_3", "S5_4"]
    for source_name in priorities:
        one = assessment_df.loc[assessment_df["source_name"] == source_name]
        if one.empty:
            continue
        row = one.iloc[0].to_dict()
        if bool(row.get("supports_target_directly", False)):
            return row
    return None


def _read_emissions_source(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def _prepare_groupable_emissions(
    path: Path,
    *,
    year: int,
    process_map: Dict[str, str],
    item_map: Dict[str, str],
    region_map: Dict[str, str],
) -> Tuple[pd.DataFrame, str]:
    df = _read_emissions_source(path)
    required = {"M49_Country_Code", "Region_label_new", "Process", "Item", "GHG"}
    missing = required.difference(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise KeyError(f"{path} is missing required columns: {missing_text}")

    year_col = _pick_year_col(df, year)
    region_label = _clean_group_col(df, "Region_label_new")
    m49_code = _normalize_code(df["M49_Country_Code"])
    ghg = df["GHG"].astype("string").str.strip().str.casefold()

    is_aggregate = region_label.str.casefold().isin(AGGREGATE_REGION_LABELS) | m49_code.isin(AGGREGATE_M49_CODES)
    prepared = df.loc[ghg.eq("co2eq") & ~is_aggregate].copy()
    prepared[year_col] = pd.to_numeric(prepared[year_col], errors="coerce")
    prepared = prepared.loc[prepared[year_col].notna()].copy()

    if prepared.empty:
        return prepared, year_col

    prepared["Region_label_new"] = region_label.loc[prepared.index]
    prepared["Region_emisSum"] = m49_code.loc[prepared.index].map(region_map)
    prepared["Region_emisSum"] = prepared["Region_emisSum"].fillna(prepared["Region_label_new"])
    prepared["Region_emisSum"] = _clean_group_col(prepared, "Region_emisSum")
    prepared["Process"] = _clean_group_col(prepared, "Process")
    prepared["Item"] = _clean_group_col(prepared, "Item")
    prepared["Process"] = prepared["Process"].map(process_map).fillna(prepared["Process"])
    prepared["Item"] = prepared["Item"].map(item_map).fillna(prepared["Item"])

    return prepared, year_col


def _aggregate_scenario(
    prepared: pd.DataFrame,
    *,
    year_col: str,
    scenario_id: str,
    variable: str,
    level_tag: str,
    rate_value: float,
    sample_id: int,
    unit_scale_gt: float,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    total_kt = float(pd.to_numeric(prepared[year_col], errors="coerce").sum()) if not prepared.empty else 0.0

    for grouping, group_cols in GROUPINGS.items():
        if prepared.empty:
            continue
        valid_mask = pd.Series(True, index=prepared.index)
        for col in group_cols:
            values = _clean_group_col(prepared, col)
            valid_mask &= values.notna() & values.ne("")

        grouped = (
            prepared.loc[valid_mask, list(group_cols) + [year_col]]
            .groupby(list(group_cols), as_index=False, dropna=False)[year_col]
            .sum()
        )
        if grouped.empty:
            continue

        for record in grouped.to_dict("records"):
            raw_val = float(record.pop(year_col))
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "variable": variable,
                    "level_tag": level_tag,
                    "rate_value": float(rate_value),
                    "sample_id": int(sample_id),
                    "grouping": grouping,
                    "group_1": str(record.get(group_cols[0], "") or "").strip(),
                    "group_2": (
                        str(record.get(group_cols[1], "") or "").strip()
                        if len(group_cols) > 1
                        else ""
                    ),
                    "emissions_kt": raw_val,
                    "emissions_gt": raw_val * unit_scale_gt,
                }
            )

    total_row = {
        "scenario_id": scenario_id,
        "variable": variable,
        "level_tag": level_tag,
        "rate_value": float(rate_value),
        "sample_id": int(sample_id),
        "total_emissions_kt": total_kt,
        "total_emissions_gt": total_kt * unit_scale_gt,
    }
    return rows, total_row


def _cache_paths(out_root: Path) -> Tuple[Path, Path]:
    return out_root / "aggregated_scenario_emissions.csv", out_root / "scenario_totals.csv"


def _build_aggregates(manifest: pd.DataFrame, *, out_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    agg_cache_path, total_cache_path = _cache_paths(out_root)
    if (
        bool(CONFIG.get("reuse_aggregate_cache", False))
        and agg_cache_path.exists()
        and total_cache_path.exists()
    ):
        return pd.read_csv(agg_cache_path), pd.read_csv(total_cache_path)

    process_map, item_map = _load_emis_item_maps()
    region_map = _load_region_emis_sum_map()
    unit_scale_gt = float(CONFIG.get("unit_scale_gt", 1e-6) or 1e-6)
    target_year = int(CONFIG.get("target_year", 2080) or 2080)

    agg_rows: List[Dict[str, object]] = []
    total_rows: List[Dict[str, object]] = []
    available = manifest[manifest["has_summary"]].copy()

    for row in available.itertuples(index=False):
        scenario_id = str(row.scenario_id)
        path = Path(str(row.summary_path))
        print(f"[LOAD] {scenario_id}")
        prepared, year_col = _prepare_groupable_emissions(
            path,
            year=target_year,
            process_map=process_map,
            item_map=item_map,
            region_map=region_map,
        )
        scenario_rows, total_row = _aggregate_scenario(
            prepared,
            year_col=year_col,
            scenario_id=scenario_id,
            variable=str(row.variable),
            level_tag=str(row.level_tag),
            rate_value=float(row.rate_value),
            sample_id=int(row.sample_id),
            unit_scale_gt=unit_scale_gt,
        )
        agg_rows.extend(scenario_rows)
        total_rows.append(total_row)

    agg_df = pd.DataFrame(agg_rows)
    total_df = pd.DataFrame(total_rows)
    agg_df.to_csv(agg_cache_path, index=False, encoding="utf-8-sig")
    total_df.to_csv(total_cache_path, index=False, encoding="utf-8-sig")
    return agg_df, total_df


def _is_reference_row(row: pd.Series) -> bool:
    ref_tag = str(CONFIG.get("reference_level_tag", "p00") or "p00").strip().lower()
    ref_rate = float(CONFIG.get("reference_rate_value", 0.0) or 0.0)
    level_tag = str(row.get("level_tag", "") or "").strip().lower()
    rate_value = pd.to_numeric(pd.Series([row.get("rate_value")]), errors="coerce").iloc[0]
    if level_tag == ref_tag:
        return True
    return bool(pd.notna(rate_value) and np.isclose(float(rate_value), ref_rate))


def _build_pair_tables(
    manifest: pd.DataFrame,
    agg_df: pd.DataFrame,
    total_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    available = manifest[manifest["has_summary"]].copy()
    if available.empty or agg_df.empty or total_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    available["is_reference"] = available.apply(_is_reference_row, axis=1)
    ref_manifest = available[available["is_reference"]].copy()
    target_manifest = available[~available["is_reference"]].copy()

    ref_lookup: Dict[Tuple[str, int], pd.Series] = {}
    for _, row in ref_manifest.sort_values(["sample_id", "scenario_id"]).iterrows():
        key = (str(row["variable"]), int(row["sample_id"]))
        if key not in ref_lookup:
            ref_lookup[key] = row

    total_lookup = total_df.set_index("scenario_id").to_dict("index")
    detail_rows: List[Dict[str, object]] = []
    pair_rows: List[Dict[str, object]] = []

    for _, scenario_meta in target_manifest.sort_values(
        ["rate_value", "sample_id", "scenario_id"],
        ascending=[True, True, True],
        kind="mergesort",
    ).iterrows():
        key = (str(scenario_meta["variable"]), int(scenario_meta["sample_id"]))
        ref_meta = ref_lookup.get(key)
        if ref_meta is None:
            continue

        scenario_id = str(scenario_meta["scenario_id"])
        ref_scenario_id = str(ref_meta["scenario_id"])
        scenario_total = total_lookup.get(scenario_id, {})
        ref_total = total_lookup.get(ref_scenario_id, {})

        pair_rows.append(
            {
                "variable": str(scenario_meta["variable"]),
                "sample_id": int(scenario_meta["sample_id"]),
                "scenario_id": scenario_id,
                "reference_scenario_id": ref_scenario_id,
                "level_tag": str(scenario_meta["level_tag"]),
                "rate_value": float(scenario_meta["rate_value"]),
                "reference_level_tag": str(ref_meta["level_tag"]),
                "reference_rate_value": float(ref_meta["rate_value"]),
                "scenario_total_gt": float(scenario_total.get("total_emissions_gt", np.nan)),
                "reference_total_gt": float(ref_total.get("total_emissions_gt", np.nan)),
                "net_reduction_gt": float(ref_total.get("total_emissions_gt", np.nan))
                - float(scenario_total.get("total_emissions_gt", np.nan)),
            }
        )

        ref_groups = agg_df[agg_df["scenario_id"] == ref_scenario_id].copy()
        cur_groups = agg_df[agg_df["scenario_id"] == scenario_id].copy()
        if ref_groups.empty or cur_groups.empty:
            continue

        for grouping in GROUPINGS:
            ref_one = ref_groups.loc[
                ref_groups["grouping"] == grouping,
                ["group_1", "group_2", "emissions_kt", "emissions_gt"],
            ].rename(
                columns={
                    "emissions_kt": "reference_emissions_kt",
                    "emissions_gt": "reference_emissions_gt",
                }
            )
            cur_one = cur_groups.loc[
                cur_groups["grouping"] == grouping,
                ["group_1", "group_2", "emissions_kt", "emissions_gt"],
            ].rename(
                columns={
                    "emissions_kt": "scenario_emissions_kt",
                    "emissions_gt": "scenario_emissions_gt",
                }
            )

            merged = ref_one.merge(cur_one, on=["group_1", "group_2"], how="outer")
            if merged.empty:
                continue

            for col in (
                "reference_emissions_kt",
                "reference_emissions_gt",
                "scenario_emissions_kt",
                "scenario_emissions_gt",
            ):
                merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

            merged["reduction_kt"] = merged["reference_emissions_kt"] - merged["scenario_emissions_kt"]
            merged["reduction_gt"] = merged["reference_emissions_gt"] - merged["scenario_emissions_gt"]
            merged["positive_reduction_gt"] = merged["reduction_gt"].clip(lower=0.0)
            total_positive_gt = float(merged["positive_reduction_gt"].sum())
            total_net_gt = float(merged["reduction_gt"].sum())
            merged["importance_share"] = (
                merged["positive_reduction_gt"] / total_positive_gt if total_positive_gt > 0 else np.nan
            )
            merged["net_share"] = merged["reduction_gt"] / total_net_gt if total_net_gt != 0 else np.nan

            for item in merged.to_dict("records"):
                detail_rows.append(
                    {
                        "variable": str(scenario_meta["variable"]),
                        "sample_id": int(scenario_meta["sample_id"]),
                        "scenario_id": scenario_id,
                        "reference_scenario_id": ref_scenario_id,
                        "level_tag": str(scenario_meta["level_tag"]),
                        "rate_value": float(scenario_meta["rate_value"]),
                        "reference_level_tag": str(ref_meta["level_tag"]),
                        "reference_rate_value": float(ref_meta["rate_value"]),
                        "grouping": grouping,
                        "group_1": str(item.get("group_1", "") or "").strip(),
                        "group_2": str(item.get("group_2", "") or "").strip(),
                        "reference_emissions_gt": float(item["reference_emissions_gt"]),
                        "scenario_emissions_gt": float(item["scenario_emissions_gt"]),
                        "reduction_gt": float(item["reduction_gt"]),
                        "positive_reduction_gt": float(item["positive_reduction_gt"]),
                        "importance_share": float(item["importance_share"])
                        if pd.notna(item["importance_share"])
                        else np.nan,
                        "net_share": float(item["net_share"]) if pd.notna(item["net_share"]) else np.nan,
                        "pair_total_positive_reduction_gt": total_positive_gt,
                        "pair_total_net_reduction_gt": total_net_gt,
                    }
                )

    return pd.DataFrame(pair_rows), pd.DataFrame(detail_rows)


def _summarize_importance(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame()

    work = detail_df.copy()
    numeric_cols = [
        "reference_emissions_gt",
        "scenario_emissions_gt",
        "reduction_gt",
        "positive_reduction_gt",
        "importance_share",
        "net_share",
    ]
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    summary = (
        work.groupby(["grouping", "rate_value", "level_tag", "group_1", "group_2"], as_index=False)
        .agg(
            n_pairs=("sample_id", "nunique"),
            mean_reference_gt=("reference_emissions_gt", "mean"),
            mean_scenario_gt=("scenario_emissions_gt", "mean"),
            mean_reduction_gt=("reduction_gt", "mean"),
            median_reduction_gt=("reduction_gt", "median"),
            std_reduction_gt=("reduction_gt", "std"),
            mean_positive_reduction_gt=("positive_reduction_gt", "mean"),
            mean_importance_share=("importance_share", "mean"),
            median_importance_share=("importance_share", "median"),
            mean_net_share=("net_share", "mean"),
            positive_pair_fraction=("reduction_gt", lambda s: float((pd.Series(s) > 0).mean())),
            counteracting_pair_fraction=("reduction_gt", lambda s: float((pd.Series(s) < 0).mean())),
        )
    )

    summary["std_reduction_gt"] = summary["std_reduction_gt"].fillna(0.0)
    summary["mean_importance_pct"] = summary["mean_importance_share"] * 100.0
    summary["rank_by_importance"] = (
        summary.groupby(["grouping", "rate_value"])["mean_importance_share"]
        .rank(method="dense", ascending=False)
        .astype("Int64")
    )
    summary["rank_by_reduction"] = (
        summary.groupby(["grouping", "rate_value"])["mean_reduction_gt"]
        .rank(method="dense", ascending=False)
        .astype("Int64")
    )

    return summary.sort_values(
        ["grouping", "rate_value", "rank_by_importance", "group_1", "group_2"],
        ascending=[True, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _format_summary_sheet(summary_df: pd.DataFrame, grouping: str) -> pd.DataFrame:
    if summary_df is None or summary_df.empty or "grouping" not in summary_df.columns:
        return pd.DataFrame()
    work = summary_df[summary_df["grouping"] == grouping].copy()
    if work.empty:
        return work

    if grouping == "Region":
        work = work.rename(columns={"group_1": "Region"})
        work = work.drop(columns=["group_2"])
    elif grouping == "Item":
        work = work.rename(columns={"group_1": "Item"})
        work = work.drop(columns=["group_2"])
    elif grouping == "Process":
        work = work.rename(columns={"group_1": "Process"})
        work = work.drop(columns=["group_2"])
    elif grouping == "Region-Item":
        work = work.rename(columns={"group_1": "Region", "group_2": "Item"})
    elif grouping == "Region-Process":
        work = work.rename(columns={"group_1": "Region", "group_2": "Process"})

    cols_front = []
    for col in ("rate_value", "level_tag", "rank_by_importance"):
        if col in work.columns:
            cols_front.append(col)
    label_cols = [c for c in ("Region", "Item", "Process") if c in work.columns]
    metric_cols = [c for c in work.columns if c not in {"grouping", *cols_front, *label_cols}]
    ordered = cols_front + label_cols + metric_cols
    return work[ordered]


def build_region_item_process_importance() -> Dict[str, Path]:
    assessment_df = _assess_sources()
    selected_source = _select_source(assessment_df)

    if CONFIG.get("output_root"):
        out_root = Path(str(CONFIG["output_root"]))
    elif selected_source is not None:
        out_root = Path(str(selected_source["root"])) / "Structure_Importance"
    else:
        out_root = _results_base_dir() / "S5_5_Region-Item-Process_Importance"
    _ensure_dir(out_root)
    assessment_path = out_root / "source_assessment.csv"
    assessment_df.to_csv(assessment_path, index=False, encoding="utf-8-sig")

    if selected_source is None:
        reason_lines = []
        if not assessment_df.empty:
            for row in assessment_df.to_dict("records"):
                reason_lines.append(f"{row.get('source_name')}: {row.get('reason', '')}")
        reason_text = " | ".join(reason_lines) if reason_lines else "No usable source found."
        raise RuntimeError(
            "No existing result source can directly support the target Region/Item/Process importance analysis. "
            f"{reason_text}"
        )

    if (
        bool(selected_source.get("needs_rerun_for_full_result", False))
        and not bool(CONFIG.get("allow_partial_pairs", True))
    ):
        raise RuntimeError(
            f"{selected_source.get('source_name')} is only partially usable. "
            f"Rerun is required for full results. Reason: {selected_source.get('reason', '')}"
        )

    root = Path(str(selected_source["root"]))

    variable = str(CONFIG.get("target_variable", "emission_factor") or "emission_factor")
    manifest_df = _build_scenario_manifest(root, variable)
    manifest_path = out_root / "scenario_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    coverage_df = _build_coverage_summary(manifest_df)
    failed_df = _build_failed_runs_audit(manifest_df)
    coverage_path = out_root / "coverage_by_level.csv"
    failed_path = out_root / "failed_runs_audit.csv"
    coverage_df.to_csv(coverage_path, index=False, encoding="utf-8-sig")
    failed_df.to_csv(failed_path, index=False, encoding="utf-8-sig")

    agg_df, total_df = _build_aggregates(manifest_df, out_root=out_root)
    pair_df, detail_df = _build_pair_tables(manifest_df, agg_df, total_df)
    summary_df = _summarize_importance(detail_df)

    pair_path = out_root / "pair_summary.csv"
    detail_path = out_root / "importance_detail.csv"
    excel_path = out_root / "EmissionIntensity_Importance.xlsx"

    pair_df.to_csv(pair_path, index=False, encoding="utf-8-sig")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    metadata_df = pd.DataFrame(
        [
            {"key": "selected_source", "value": str(selected_source.get("source_name", ""))},
            {"key": "selected_source_root", "value": str(root)},
            {"key": "selected_source_reason", "value": str(selected_source.get("reason", ""))},
            {
                "key": "selected_source_needs_rerun_for_full_result",
                "value": bool(selected_source.get("needs_rerun_for_full_result", False)),
            },
            {"key": "input_root", "value": str(root)},
            {"key": "target_variable", "value": variable},
            {"key": "reference_level_tag", "value": str(CONFIG.get("reference_level_tag", "p00"))},
            {"key": "reference_rate_value", "value": str(CONFIG.get("reference_rate_value", 0.0))},
            {"key": "target_year", "value": str(CONFIG.get("target_year", 2080))},
            {"key": "available_scenarios", "value": int(manifest_df["has_summary"].sum())},
            {"key": "matched_pairs", "value": int(len(pair_df))},
            {
                "key": "coverage_note",
                "value": "importance uses paired complete cases only: both p00 and target level must have detailed Emis output",
            },
            {
                "key": "importance_definition",
                "value": "importance_share = max(reduction,0) / sum(max(reduction,0)) within paired sample and grouping",
            },
        ]
    )

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        metadata_df.to_excel(writer, sheet_name="Metadata", index=False)
        assessment_df.to_excel(writer, sheet_name="SourceAssessment", index=False)
        manifest_df.to_excel(writer, sheet_name="Manifest", index=False)
        coverage_df.to_excel(writer, sheet_name="CoverageByLevel", index=False)
        failed_df.to_excel(writer, sheet_name="FailedRuns", index=False)
        pair_df.to_excel(writer, sheet_name="PairSummary", index=False)
        for grouping in GROUPINGS:
            _format_summary_sheet(summary_df, grouping).to_excel(writer, sheet_name=grouping[:31], index=False)

    return {
        "source_assessment": assessment_path,
        "manifest": manifest_path,
        "coverage": coverage_path,
        "failed_runs": failed_path,
        "aggregated": out_root / "aggregated_scenario_emissions.csv",
        "totals": out_root / "scenario_totals.csv",
        "pairs": pair_path,
        "detail": detail_path,
        "excel": excel_path,
    }


def main() -> None:
    outputs = build_region_item_process_importance()
    for key, path in outputs.items():
        print(f"[DONE] {key}: {path}")


if __name__ == "__main__":
    main()
