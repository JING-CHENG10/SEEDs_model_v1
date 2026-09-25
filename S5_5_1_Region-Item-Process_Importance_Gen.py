# -*- coding: utf-8 -*-
"""
Generate new MC outputs for Region / Item / Process structure-importance plots.

This is the production-style S5.5 generator. It runs the same full-variable MC
sampling path as S5_4, but forces detailed emissions output so each successful
sample can be aggregated with the same Region / Item / Process definitions used
by SP_M1a_Figure_pie_structure_pre.py.

Main outputs under <output_dir>/structure or <output_dir>/merged_structure:
  - mc_success_structure_emissions.csv
  - mc_success_structure_totals.csv
  - structure_stack_long.csv
  - structure_stack_top_long.csv
  - structure_quartile_panel_samples.csv
  - structure_quartile_summary.csv
  - structure_quartile_histogram_top.csv
"""
from __future__ import annotations

import argparse
import copy
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import S5_4_1_monte_carlo_full_variables as fullmc
from config_paths import get_results_base
from S5_cost_summary_outputs import write_sensitivity_cost_summaries
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
    "run_mc": True,
    "postprocess": True,
    "postprocess_scope": "active",  # active | all_batches | root
    "seed": fullmc.CONFIG.get("seed", 42),
    "samples": fullmc.CONFIG.get("samples", 20000),
    "year": 2080,
    "output_dir": "",  # empty -> <NZF_OUTPUT_DIR>/MC_Region_Item_Process_Importance
    "resume": False,
    "clear_existing_run_dirs_when_no_resume": True,
    "max_runs": None,
    "batch": {
        "enabled": True,
        "total_batches": 200,
        "batch_index": 1,  # 1-based; usually overridden by S5_5_2 wrapper or RIPMC_BATCH_INDEX.
        "assignment": "round_robin",  # 'round_robin' | 'contiguous'
        "batches_subdir": "batches",
    },
    "sampling": copy.deepcopy(fullmc.CONFIG.get("sampling", {})),
    "nutrition_profile_sheet": "low_land_new",
    "write_every_n_runs": 1,
    "unit_scale_gt": 1e-6,  # emissions_summary_By_Country_Process_Item is in kt CO2eq
    "invalid_total_co2eq_gt_values": (1.264874,),
    "structure_subdir": "structure",
    "merged_structure_subdir": "merged_structure",
    "flush_every_n_runs": 10,
    "target_emissions_gt": [float(x) for x in range(-3, 19)],
    "target_nearest_n": 200,
    "target_window_gt": 0.25,
    "top_n_for_stack": {
        "Region": 10,
        "Item": 15,
        "Process": 12,
        "Region-Item": 25,
        "Region-Process": 25,
        "Process-Item": 25,
        "Region-Item-Process": 40,
    },
    "quartile_rank_metric_mode": "absolute",  # ratio | absolute
    "quartile_rank_metric_mode_by_panel": {
        "ruminate_intake": "absolute",
    },
    "quartile_panels": ["yield", "emission_factor", "ruminate_intake", "luc_land_intensity"],
    "histogram_bins": 120,
    "top_n_for_quartile_histogram": {
        "Region": 8,
        "Item": 10,
        "Process": 10,
        "Region-Item": 12,
        "Region-Process": 12,
        "Process-Item": 12,
        "Region-Item-Process": 15,
    },
    "luc_land_intensity_processes": (
        "De/Reforestation_crop",
        "De/Reforestation_pasture",
        "Forest",
        "Wood harvest",
    ),
    "override_cfg": {
        "nutrition_profile_sheet": "low_land_new",
        "cost_calculation_method": "off",
        "debug_level": 0,
        "domestic_supply_simulation_mode": "hard_equation",
        "supply_curtailment_enabled": False,
        "supply_curtailment_penalty": 1e10,
        "trade_cap_exempt_all_pairs": True,
        "batch_mode": True,
        "linear_enable_infeasible_iis": False,
        "linear_enable_violation_iis": False,
        "linear_enable_output_diagnostics": False,
        "linear_enable_verbose_logging": False,
        "linear_solver_method": 1,
        "linear_solver_threads": 4,
    },
}

GROUPINGS: Dict[str, Tuple[str, ...]] = {
    "Region": ("Region_emisSum",),
    "Item": ("Item",),
    "Process": ("Process",),
    "Region-Item": ("Region_emisSum", "Item"),
    "Region-Process": ("Region_emisSum", "Process"),
    "Process-Item": ("Process", "Item"),
    "Region-Item-Process": ("Region_emisSum", "Item", "Process"),
}

GROUP_LABELS = {
    "Region_emisSum": "Region",
    "Item": "Item",
    "Process": "Process",
}

LUC_AREA_BASE_YEAR = 2020
LUC_CROP_ITEM_ALLOCATION_PROCESSES = {
    "De/Reforestation_crop",
    "Ag land abandonment_crop",
    "Grassland conversion_crop",
}
LUC_PASTURE_ITEM_ALLOCATION_PROCESSES = {
    "De/Reforestation_pasture",
    "Ag land abandonment_pasture",
}
LUC_ITEM_ALLOCATION_PROCESSES = (
    LUC_CROP_ITEM_ALLOCATION_PROCESSES | LUC_PASTURE_ITEM_ALLOCATION_PROCESSES
)
ORGANIC_SOIL_PROCESS = "Drained organic soils"
ORGANIC_SOIL_CROPLAND_ITEMS = {
    "Organic soils",
    "Cropland organic soils",
}
ORGANIC_SOIL_GRASSLAND_ITEMS = {
    "Grassland organic soils",
}
ORGANIC_SOIL_SOURCE_ITEMS = ORGANIC_SOIL_CROPLAND_ITEMS | ORGANIC_SOIL_GRASSLAND_ITEMS

PANEL_SPECS = {
    "yield": {
        "kind": "yield_rate",
        "metric_label": "Yield",
        "value_mode": "change_percent",
    },
    "emission_factor": {
        "kind": "emission_factor",
        "metric_label": "Emission factor",
        "value_mode": "change_percent",
    },
    "ruminate_intake": {
        "kind": "ruminant_reduction",
        "metric_label": "Ruminant kcal share",
        "value_mode": "share_percent",
    },
    "luc_land_intensity": {
        "kind": "luc_land_intensity",
        "metric_label": "LUC intensity per crop + pasture land",
        "value_mode": "t_per_ha",
    },
}

QUARTILE_KEYS = ("q1", "q2", "q3", "q4")
QUARTILE_RANGES = {
    "q1": "0-25%",
    "q2": "25-50%",
    "q3": "50-75%",
    "q4": "75-100%",
}

COUNTRY_PROCESS_ITEM_CANDIDATES = (
    "emissions_summary_By_Country_Process_Item.csv",
    "emissions_summary_By_Country_Process_Item.xlsx",
)

BATCH_DIR_PATTERN = re.compile(r"^batch_(\d+)_of_(\d+)$")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _reset_output_file(path: Path) -> None:
    if path.exists():
        path.unlink()


def _append_rows_csv(rows: List[Dict[str, object]], out_path: Path) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    header = (not out_path.exists()) or out_path.stat().st_size == 0
    df.to_csv(out_path, mode="a", header=header, index=False, encoding="utf-8-sig")
    return int(len(df))


def _root_output_dir(cfg: Dict[str, object]) -> Path:
    raw = str(cfg.get("output_dir", "") or "").strip()
    if raw:
        return Path(raw)
    return Path(get_results_base()) / "MC_Region_Item_Process_Importance"


def _fullmc_cfg(cfg: Dict[str, object]) -> Dict[str, object]:
    out = copy.deepcopy(fullmc.CONFIG)
    out["seed"] = int(cfg.get("seed", out.get("seed", 42)) or 42)
    out["samples"] = int(cfg.get("samples", out.get("samples", 0)) or 0)
    out["year"] = int(cfg.get("year", 2080) or 2080)
    out["fast_emis_year"] = int(cfg.get("year", 2080) or 2080)
    out["output_dir"] = str(_root_output_dir(cfg))
    out["resume"] = bool(cfg.get("resume", False))
    out["clear_existing_run_dirs_when_no_resume"] = bool(
        cfg.get("clear_existing_run_dirs_when_no_resume", True)
    )
    out["max_runs"] = cfg.get("max_runs")
    out["batch"] = copy.deepcopy(cfg.get("batch", {}) or {})
    out["sampling"] = copy.deepcopy(cfg.get("sampling", {}) or {})
    out["nutrition_profile_sheet"] = str(cfg.get("nutrition_profile_sheet", "low_land_new") or "low_land_new")
    out["write_every_n_runs"] = int(cfg.get("write_every_n_runs", 1) or 1)
    out["save_per_run_dirs"] = True
    out["fast_emis_only"] = False
    out["override_cfg"] = {
        **copy.deepcopy(fullmc.CONFIG.get("override_cfg", {}) or {}),
        **copy.deepcopy(cfg.get("override_cfg", {}) or {}),
    }
    out["override_cfg"]["nutrition_profile_sheet"] = out["nutrition_profile_sheet"]
    return out


def _batch_state(mc_cfg: Dict[str, object]) -> Dict[str, object]:
    return fullmc._resolve_batch_settings(mc_cfg)


def _batch_dirs_for_postprocess(root: Path, cfg: Dict[str, object]) -> List[Path]:
    batches_subdir = str((cfg.get("batch") or {}).get("batches_subdir", "batches") or "batches")
    batches_root = root / batches_subdir
    if not batches_root.exists():
        raise FileNotFoundError(f"No S5_5 batch directory found: {batches_root}")

    counts = set()
    dirs_by_index: Dict[int, Path] = {}
    for child in batches_root.iterdir():
        if not child.is_dir():
            continue
        match = BATCH_DIR_PATTERN.match(child.name)
        if not match:
            continue
        idx = int(match.group(1))
        total = int(match.group(2))
        counts.add(total)
        dirs_by_index[idx] = child

    if not dirs_by_index:
        raise RuntimeError(f"No batch_XX_of_YY directories under: {batches_root}")
    if len(counts) != 1:
        raise RuntimeError(
            f"Inconsistent batch count suffixes under {batches_root}: {sorted(counts)}"
        )

    total_batches = counts.pop()
    missing = [idx for idx in range(1, total_batches + 1) if idx not in dirs_by_index]
    if missing:
        preview = ", ".join(str(idx) for idx in missing[:10])
        suffix = f" ... (+{len(missing) - 10} more)" if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"Detected total_batches={total_batches}, but missing S5_5 batch dirs: {preview}{suffix}"
        )
    return [dirs_by_index[idx] for idx in range(1, total_batches + 1)]


def _active_output_dir(cfg: Dict[str, object]) -> Path:
    mc_cfg = _fullmc_cfg(cfg)
    root = _root_output_dir(cfg)
    batch_state = _batch_state(mc_cfg)
    if bool(batch_state.get("enabled")):
        return root / str(batch_state.get("batches_subdir", "batches")) / str(batch_state["tag"])
    return root


def _postprocess_dirs(cfg: Dict[str, object]) -> Tuple[List[Path], Path]:
    root = _root_output_dir(cfg)
    scope = str(cfg.get("postprocess_scope", "active") or "active").strip().lower()
    if scope == "active":
        active = _active_output_dir(cfg)
        return [active], active / str(cfg.get("structure_subdir", "structure") or "structure")
    if scope == "all_batches":
        dirs = _batch_dirs_for_postprocess(root, cfg)
        return dirs, root / str(cfg.get("merged_structure_subdir", "merged_structure") or "merged_structure")
    if scope == "root":
        return [root], root / str(cfg.get("structure_subdir", "structure") or "structure")
    raise ValueError("postprocess_scope must be active, all_batches, or root.")


def _run_mc(cfg: Dict[str, object]) -> None:
    mc_cfg = _fullmc_cfg(cfg)
    previous = copy.deepcopy(fullmc.CONFIG)
    fullmc.CONFIG.clear()
    fullmc.CONFIG.update(mc_cfg)
    try:
        fullmc.main()
    finally:
        fullmc.CONFIG.clear()
        fullmc.CONFIG.update(previous)


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_concat(input_dirs: Sequence[Path], filename: str) -> pd.DataFrame:
    frames = []
    for input_dir in input_dirs:
        path = input_dir / filename
        df = _read_csv_if_exists(path)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _load_success_status(input_dirs: Sequence[Path]) -> pd.DataFrame:
    status = _read_concat(input_dirs, "mc_sample_status.csv")
    if status.empty:
        return status
    status.columns = [str(c).strip() for c in status.columns]
    required = {"scenario_id", "sample_id", "run_status", "scenario_dir"}
    missing = required.difference(status.columns)
    if missing:
        raise KeyError(f"mc_sample_status.csv missing columns: {sorted(missing)}")
    status["run_status"] = status["run_status"].astype("string").str.strip().str.lower()
    status = status[status["run_status"].isin({"ok", "resumed"})].copy()
    status["scenario_id"] = status["scenario_id"].astype("string").str.strip()
    status["sample_id"] = pd.to_numeric(status["sample_id"], errors="coerce")
    status = status.dropna(subset=["scenario_id", "sample_id"]).copy()
    status["sample_id"] = status["sample_id"].astype(int)
    return status.drop_duplicates(subset=["scenario_id", "sample_id"], keep="last").reset_index(drop=True)


def _find_country_process_item_path(scenario_dir: Path) -> Optional[Path]:
    emis_dir = scenario_dir / "Emis"
    for name in COUNTRY_PROCESS_ITEM_CANDIDATES:
        path = emis_dir / name
        if path.exists():
            return path
    return None


def _read_emissions_source(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def _scenario_root_from_emissions_path(path: Path) -> Path:
    parent = path.parent
    if parent.name.strip().lower() == "emis":
        return parent.parent
    return parent


def _load_production_area(scenario_dir: Path, year: int) -> pd.DataFrame:
    path = scenario_dir / "DS" / "production_summary.csv"
    cols = ["M49_Country_Code", "year", "commodity", "crop_area_ha", "pasture_area_ha"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(path, usecols=lambda c: str(c).strip() in set(cols))
    except Exception:
        return pd.DataFrame(columns=cols)
    df.columns = [str(c).strip() for c in df.columns]
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    work = df[cols].copy()
    work["M49_Country_Code"] = _normalize_code(work["M49_Country_Code"])
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work["commodity"] = work["commodity"].astype("string").str.strip()
    for col in ("crop_area_ha", "pasture_area_ha"):
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0).clip(lower=0.0)
    work = work.dropna(subset=["M49_Country_Code", "year", "commodity"]).copy()
    work = work.loc[
        work["M49_Country_Code"].astype(str).str.strip().ne("")
        & work["commodity"].astype(str).str.strip().ne("")
        & work["year"].isin([LUC_AREA_BASE_YEAR, int(year)])
    ].copy()
    if work.empty:
        return work
    work["year"] = work["year"].astype(int)
    return (
        work.groupby(["M49_Country_Code", "year", "commodity"], as_index=False)[
            ["crop_area_ha", "pasture_area_ha"]
        ]
        .sum()
        .reset_index(drop=True)
    )


def _build_luc_area_lookup(area_df: pd.DataFrame, year: int, area_col: str) -> Dict[str, pd.DataFrame]:
    if area_df.empty or area_col not in area_df.columns:
        return {}
    base = (
        area_df.loc[
            area_df["year"].eq(LUC_AREA_BASE_YEAR),
            ["M49_Country_Code", "commodity", area_col],
        ]
        .rename(columns={area_col: "base_area_ha"})
        .groupby(["M49_Country_Code", "commodity"], as_index=False)["base_area_ha"]
        .sum()
    )
    future = (
        area_df.loc[
            area_df["year"].eq(int(year)),
            ["M49_Country_Code", "commodity", area_col],
        ]
        .rename(columns={area_col: "future_area_ha"})
        .groupby(["M49_Country_Code", "commodity"], as_index=False)["future_area_ha"]
        .sum()
    )
    merged = future.merge(base, on=["M49_Country_Code", "commodity"], how="outer")
    merged[["base_area_ha", "future_area_ha"]] = merged[["base_area_ha", "future_area_ha"]].fillna(0.0)
    merged["delta_area_ha"] = merged["future_area_ha"] - merged["base_area_ha"]
    lookup: Dict[str, pd.DataFrame] = {}
    for m49, group in merged.groupby("M49_Country_Code", sort=False):
        sub = group.loc[
            group["commodity"].notna()
            & group["commodity"].astype("string").str.strip().ne("")
        ].copy()
        if not sub.empty:
            lookup[str(m49)] = sub.reset_index(drop=True)
    return lookup


def _luc_item_weights(area_group: pd.DataFrame, value: float) -> Tuple[pd.Series, str]:
    if value > 0:
        weights = pd.to_numeric(area_group["delta_area_ha"], errors="coerce").clip(lower=0.0)
        if float(weights.sum()) > 0:
            return weights, "positive_delta"
    elif value < 0:
        weights = -pd.to_numeric(area_group["delta_area_ha"], errors="coerce").clip(upper=0.0)
        if float(weights.sum()) > 0:
            return weights, "negative_delta"
    weights = pd.to_numeric(area_group["future_area_ha"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return weights, "future_area"


def _empty_luc_allocation_diag() -> Dict[str, object]:
    return {
        "luc_allocation_input_rows": 0,
        "luc_allocation_split_rows": 0,
        "luc_allocation_output_rows": 0,
        "luc_allocation_crop_split_rows": 0,
        "luc_allocation_pasture_split_rows": 0,
        "luc_allocation_ag_abandonment_input_rows": 0,
        "luc_allocation_ag_abandonment_split_rows": 0,
        "luc_allocation_no_lookup_rows": 0,
        "luc_allocation_no_weight_rows": 0,
        "luc_allocation_bad_value_rows": 0,
        "luc_allocation_missing_m49_rows": 0,
        "luc_allocation_weight_positive_delta_rows": 0,
        "luc_allocation_weight_negative_delta_rows": 0,
        "luc_allocation_weight_future_area_rows": 0,
    }


def _empty_organic_soil_allocation_diag() -> Dict[str, object]:
    return {
        "organic_soil_allocation_input_rows": 0,
        "organic_soil_allocation_split_rows": 0,
        "organic_soil_allocation_output_rows": 0,
        "organic_soil_allocation_cropland_split_rows": 0,
        "organic_soil_allocation_grassland_split_rows": 0,
        "organic_soil_allocation_no_lookup_rows": 0,
        "organic_soil_allocation_no_weight_rows": 0,
        "organic_soil_allocation_bad_value_rows": 0,
        "organic_soil_allocation_missing_m49_rows": 0,
    }


def _allocate_luc_emissions_to_items(
    df: pd.DataFrame,
    *,
    scenario_dir: Path,
    year: int,
    value_col: str,
) -> pd.DataFrame:
    diag = _empty_luc_allocation_diag()
    if df.empty:
        out = df.copy()
        out.attrs["luc_item_allocation"] = diag
        return out

    area_df = _load_production_area(scenario_dir, year)
    crop_lookup = _build_luc_area_lookup(area_df, year, "crop_area_ha")
    pasture_lookup = _build_luc_area_lookup(area_df, year, "pasture_area_ha")
    if not crop_lookup and not pasture_lookup:
        out = df.copy()
        out.attrs["luc_item_allocation"] = diag
        return out

    expanded_rows: List[Dict[str, object]] = []
    for row in df.to_dict(orient="records"):
        process = str(row.get("Process_raw", row.get("Process", "")) or "").strip()
        if process not in LUC_ITEM_ALLOCATION_PROCESSES:
            expanded_rows.append(row)
            continue

        diag["luc_allocation_input_rows"] += 1
        if process.startswith("Ag land abandonment"):
            diag["luc_allocation_ag_abandonment_input_rows"] += 1

        m49 = str(row.get("M49_Country_Code", "") or "").strip()
        if not m49:
            expanded_rows.append(row)
            diag["luc_allocation_missing_m49_rows"] += 1
            continue

        value = pd.to_numeric(row.get(value_col), errors="coerce")
        if pd.isna(value):
            expanded_rows.append(row)
            diag["luc_allocation_bad_value_rows"] += 1
            continue
        value_f = float(value)

        is_crop = process in LUC_CROP_ITEM_ALLOCATION_PROCESSES
        area_group = (crop_lookup if is_crop else pasture_lookup).get(str(m49))
        if area_group is None or area_group.empty:
            expanded_rows.append(row)
            diag["luc_allocation_no_lookup_rows"] += 1
            continue

        weights, weight_mode = _luc_item_weights(area_group, value_f)
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            expanded_rows.append(row)
            diag["luc_allocation_no_weight_rows"] += 1
            continue

        allocated = 0.0
        used = False
        for commodity, weight in zip(area_group["commodity"], weights):
            weight_f = float(weight)
            if not np.isfinite(weight_f) or weight_f <= 0:
                continue
            commodity_s = str(commodity or "").strip()
            if not commodity_s:
                continue
            new_row = dict(row)
            new_row["Item"] = commodity_s
            new_value = value_f * weight_f / weight_sum
            new_row[value_col] = new_value
            expanded_rows.append(new_row)
            allocated += new_value
            used = True

        if not used:
            expanded_rows.append(row)
            diag["luc_allocation_no_weight_rows"] += 1
            continue

        diff = value_f - allocated
        if abs(diff) > 1e-9:
            expanded_rows[-1][value_col] = float(expanded_rows[-1][value_col]) + diff

        diag["luc_allocation_split_rows"] += 1
        if is_crop:
            diag["luc_allocation_crop_split_rows"] += 1
        else:
            diag["luc_allocation_pasture_split_rows"] += 1
        if process.startswith("Ag land abandonment"):
            diag["luc_allocation_ag_abandonment_split_rows"] += 1
        diag[f"luc_allocation_weight_{weight_mode}_rows"] += 1

    out = pd.DataFrame(expanded_rows, columns=df.columns)
    diag["luc_allocation_output_rows"] = int(len(out))
    out.attrs["luc_item_allocation"] = diag
    return out


def _allocate_organic_soil_emissions_to_items(
    df: pd.DataFrame,
    *,
    scenario_dir: Path,
    year: int,
    value_col: str,
) -> pd.DataFrame:
    diag = _empty_organic_soil_allocation_diag()
    if df.empty:
        out = df.copy()
        out.attrs["organic_soil_item_allocation"] = diag
        return out

    area_df = _load_production_area(scenario_dir, year)
    crop_lookup = _build_luc_area_lookup(area_df, year, "crop_area_ha")
    pasture_lookup = _build_luc_area_lookup(area_df, year, "pasture_area_ha")
    if not crop_lookup and not pasture_lookup:
        out = df.copy()
        out.attrs["organic_soil_item_allocation"] = diag
        return out

    expanded_rows: List[Dict[str, object]] = []
    for row in df.to_dict(orient="records"):
        process = str(row.get("Process_raw", row.get("Process", "")) or "").strip()
        item = str(row.get("Item", "") or "").strip()
        if process != ORGANIC_SOIL_PROCESS or item not in ORGANIC_SOIL_SOURCE_ITEMS:
            expanded_rows.append(row)
            continue

        diag["organic_soil_allocation_input_rows"] += 1
        m49 = str(row.get("M49_Country_Code", "") or "").strip()
        if not m49:
            expanded_rows.append(row)
            diag["organic_soil_allocation_missing_m49_rows"] += 1
            continue

        value = pd.to_numeric(row.get(value_col), errors="coerce")
        if pd.isna(value):
            expanded_rows.append(row)
            diag["organic_soil_allocation_bad_value_rows"] += 1
            continue
        value_f = float(value)

        is_cropland = item in ORGANIC_SOIL_CROPLAND_ITEMS
        area_group = (crop_lookup if is_cropland else pasture_lookup).get(str(m49))
        if area_group is None or area_group.empty:
            expanded_rows.append(row)
            diag["organic_soil_allocation_no_lookup_rows"] += 1
            continue

        weights = pd.to_numeric(area_group["future_area_ha"], errors="coerce").fillna(0.0).clip(lower=0.0)
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            expanded_rows.append(row)
            diag["organic_soil_allocation_no_weight_rows"] += 1
            continue

        allocated = 0.0
        used = False
        for commodity, weight in zip(area_group["commodity"], weights):
            weight_f = float(weight)
            if not np.isfinite(weight_f) or weight_f <= 0:
                continue
            commodity_s = str(commodity or "").strip()
            if not commodity_s:
                continue
            new_row = dict(row)
            new_row["Item"] = commodity_s
            new_value = value_f * weight_f / weight_sum
            new_row[value_col] = new_value
            expanded_rows.append(new_row)
            allocated += new_value
            used = True

        if not used:
            expanded_rows.append(row)
            diag["organic_soil_allocation_no_weight_rows"] += 1
            continue

        diff = value_f - allocated
        if abs(diff) > 1e-9:
            expanded_rows[-1][value_col] = float(expanded_rows[-1][value_col]) + diff

        diag["organic_soil_allocation_split_rows"] += 1
        if is_cropland:
            diag["organic_soil_allocation_cropland_split_rows"] += 1
        else:
            diag["organic_soil_allocation_grassland_split_rows"] += 1

    out = pd.DataFrame(expanded_rows, columns=df.columns)
    diag["organic_soil_allocation_output_rows"] = int(len(out))
    out.attrs["organic_soil_item_allocation"] = diag
    return out


def _prepare_groupable_emissions(
    path: Path,
    *,
    year: int,
    process_map: Dict[str, str],
    item_map: Dict[str, str],
    region_map: Dict[str, str],
) -> Tuple[pd.DataFrame, str]:
    df = _read_emissions_source(path)
    df.columns = [str(c).strip() for c in df.columns]
    if "M49_Country_Code" not in df.columns and "M49" in df.columns:
        df["M49_Country_Code"] = df["M49"]
    required = {"M49_Country_Code", "Process", "Item", "GHG"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"{path} missing required columns: {sorted(missing)}")

    if "Region_label_new" not in df.columns:
        df["Region_label_new"] = ""

    try:
        value_col = _pick_year_col(df, int(year))
        work = df.copy()
    except Exception:
        if "year" not in df.columns:
            raise
        value_candidates = [c for c in ("co2eq_kt", "emissions_kt", "value") if c in df.columns]
        if not value_candidates:
            raise KeyError(f"{path} has no year column Y{year} and no co2eq_kt/emissions_kt/value column.")
        value_col = value_candidates[0]
        work = df[pd.to_numeric(df["year"], errors="coerce").eq(int(year))].copy()

    region_label = _clean_group_col(work, "Region_label_new")
    m49_code = _normalize_code(work["M49_Country_Code"])
    ghg = work["GHG"].astype("string").str.strip().str.casefold()
    is_aggregate = region_label.str.casefold().isin(AGGREGATE_REGION_LABELS) | m49_code.isin(AGGREGATE_M49_CODES)

    prepared = work.loc[ghg.isin({"co2eq", "co2e"}) & ~is_aggregate].copy()
    prepared[value_col] = pd.to_numeric(prepared[value_col], errors="coerce")
    prepared = prepared.loc[prepared[value_col].notna()].copy()
    if prepared.empty:
        return prepared, value_col

    prepared["M49_Country_Code"] = m49_code.loc[prepared.index]
    prepared["Region_label_new"] = region_label.loc[prepared.index]
    prepared["Region_emisSum"] = m49_code.loc[prepared.index].map(region_map)
    prepared["Region_emisSum"] = prepared["Region_emisSum"].fillna(prepared["Region_label_new"])
    prepared["Region_emisSum"] = _clean_group_col(prepared, "Region_emisSum")
    prepared["Process_raw"] = _clean_group_col(prepared, "Process")
    prepared["Item"] = _clean_group_col(prepared, "Item")
    scenario_dir = _scenario_root_from_emissions_path(path)
    prepared = _allocate_luc_emissions_to_items(
        prepared,
        scenario_dir=scenario_dir,
        year=int(year),
        value_col=value_col,
    )
    allocation_attrs = dict(prepared.attrs.get("luc_item_allocation", {}) or {})
    prepared = _allocate_organic_soil_emissions_to_items(
        prepared,
        scenario_dir=scenario_dir,
        year=int(year),
        value_col=value_col,
    )
    allocation_attrs.update(prepared.attrs.get("organic_soil_item_allocation", {}) or {})
    prepared.attrs["item_allocation"] = allocation_attrs
    process_raw = _clean_group_col(prepared, "Process_raw")
    prepared["Process"] = process_raw.map(process_map).fillna(process_raw)
    item_raw = _clean_group_col(prepared, "Item")
    prepared["Item"] = item_raw.map(item_map).fillna(item_raw)
    return prepared, value_col


def _group_key(record: Dict[str, object], group_cols: Sequence[str]) -> str:
    parts = []
    for col in group_cols:
        label = GROUP_LABELS.get(col, col)
        parts.append(f"{label}={str(record.get(col, '') or '').strip()}")
    return " | ".join(parts)


def _aggregate_one_scenario(
    prepared: pd.DataFrame,
    *,
    year_col: str,
    meta: Dict[str, object],
    unit_scale_gt: float,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    value = pd.to_numeric(prepared[year_col], errors="coerce") if not prepared.empty else pd.Series(dtype=float)
    total_kt = float(value.sum()) if not value.empty else 0.0
    total_gt = total_kt * unit_scale_gt
    base = {
        "scenario_id": str(meta.get("scenario_id", "") or ""),
        "sample_id": int(meta.get("sample_id", 0) or 0),
        "batch_index": meta.get("batch_index", ""),
        "batch_count": meta.get("batch_count", ""),
        "batch_tag": meta.get("batch_tag", ""),
        "scenario_dir": meta.get("scenario_dir", ""),
        "year": int(meta.get("target_year", CONFIG.get("year", 2080)) or CONFIG.get("year", 2080)),
    }
    rows: List[Dict[str, object]] = []
    for grouping, group_cols in GROUPINGS.items():
        if prepared.empty:
            continue
        valid_mask = pd.Series(True, index=prepared.index)
        for col in group_cols:
            vals = _clean_group_col(prepared, col)
            valid_mask &= vals.notna() & vals.ne("")
        grouped = (
            prepared.loc[valid_mask, list(group_cols) + [year_col]]
            .groupby(list(group_cols), as_index=False, dropna=False)[year_col]
            .sum()
        )
        if grouped.empty:
            continue
        for record in grouped.to_dict("records"):
            emissions_kt = float(record.get(year_col, 0.0) or 0.0)
            out = dict(base)
            out.update(
                {
                    "grouping": grouping,
                    "group_key": _group_key(record, group_cols),
                    "group_1": str(record.get(group_cols[0], "") or "").strip(),
                    "group_2": str(record.get(group_cols[1], "") or "").strip() if len(group_cols) > 1 else "",
                    "group_3": str(record.get(group_cols[2], "") or "").strip() if len(group_cols) > 2 else "",
                    "emissions_kt": emissions_kt,
                    "emissions_gt": emissions_kt * unit_scale_gt,
                    "total_emissions_gt": total_gt,
                    "emissions_share": (emissions_kt / total_kt) if total_kt else np.nan,
                }
            )
            rows.append(out)

    total_row = dict(base)
    total_row.update({"total_emissions_kt": total_kt, "total_emissions_gt": total_gt})
    return rows, total_row


def _is_invalid_total_gt(value: object, cfg: Dict[str, object]) -> bool:
    try:
        val = float(value)
    except Exception:
        return True
    if not np.isfinite(val):
        return True
    bad_values = cfg.get("invalid_total_co2eq_gt_values", ()) or ()
    rounded = round(val, 6)
    return any(rounded == round(float(bad), 6) for bad in bad_values)


def _build_structure_outputs(cfg: Dict[str, object], input_dirs: Sequence[Path], out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _ensure_dir(out_dir)
    structure_path = out_dir / "mc_success_structure_emissions.csv"
    totals_path = out_dir / "mc_success_structure_totals.csv"
    missing_path = out_dir / "missing_structure_detail.csv"
    item_allocation_path = out_dir / "item_allocation_diagnostics.csv"
    luc_allocation_path = out_dir / "luc_item_allocation_diagnostics.csv"
    for path in (structure_path, totals_path, missing_path, item_allocation_path, luc_allocation_path):
        _reset_output_file(path)

    status = _load_success_status(input_dirs)
    if status.empty:
        raise RuntimeError("No successful MC runs found in mc_sample_status.csv.")

    process_map, item_map = _load_emis_item_maps()
    region_map = _load_region_emis_sum_map()
    unit_scale_gt = float(cfg.get("unit_scale_gt", 1e-6) or 1e-6)
    target_year = int(cfg.get("year", 2080) or 2080)
    flush_n = int(cfg.get("flush_every_n_runs", 10) or 10)
    if flush_n <= 0:
        flush_n = 10

    structure_rows: List[Dict[str, object]] = []
    total_rows: List[Dict[str, object]] = []
    missing_rows: List[Dict[str, object]] = []
    luc_allocation_rows: List[Dict[str, object]] = []
    processed = 0

    for row in status.to_dict("records"):
        scenario_dir = Path(str(row.get("scenario_dir", "") or "").strip())
        scenario_id = str(row.get("scenario_id", "") or "").strip()
        summary_path = _find_country_process_item_path(scenario_dir)
        if summary_path is None:
            missing_rows.append({**row, "reason": "missing emissions_summary_By_Country_Process_Item"})
            continue
        try:
            prepared, year_col = _prepare_groupable_emissions(
                summary_path,
                year=target_year,
                process_map=process_map,
                item_map=item_map,
                region_map=region_map,
            )
            luc_diag = dict(
                prepared.attrs.get("item_allocation", {})
                or prepared.attrs.get("luc_item_allocation", {})
                or {}
            )
            if luc_diag:
                luc_allocation_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "sample_id": int(row.get("sample_id", 0) or 0),
                        "scenario_dir": str(scenario_dir),
                        **luc_diag,
                    }
                )
            rows, total = _aggregate_one_scenario(
                prepared,
                year_col=year_col,
                meta=row,
                unit_scale_gt=unit_scale_gt,
            )
            if _is_invalid_total_gt(total.get("total_emissions_gt"), cfg):
                missing_rows.append({**row, "reason": f"invalid total {total.get('total_emissions_gt')}"})
                continue
            structure_rows.extend(rows)
            total_rows.append(total)
            processed += 1
        except Exception as exc:
            missing_rows.append({**row, "reason": f"{type(exc).__name__}: {exc}"})
            continue

        if processed % flush_n == 0:
            _append_rows_csv(structure_rows, structure_path)
            _append_rows_csv(total_rows, totals_path)
            _append_rows_csv(missing_rows, missing_path)
            _append_rows_csv(luc_allocation_rows, item_allocation_path)
            _append_rows_csv(luc_allocation_rows, luc_allocation_path)
            structure_rows = []
            total_rows = []
            missing_rows = []
            luc_allocation_rows = []
            print(f"[S5_5_GEN] processed structure runs={processed}")

    _append_rows_csv(structure_rows, structure_path)
    _append_rows_csv(total_rows, totals_path)
    _append_rows_csv(missing_rows, missing_path)
    _append_rows_csv(luc_allocation_rows, item_allocation_path)
    _append_rows_csv(luc_allocation_rows, luc_allocation_path)

    structure_df = pd.read_csv(structure_path) if structure_path.exists() else pd.DataFrame()
    totals_df = pd.read_csv(totals_path) if totals_path.exists() else pd.DataFrame()
    if structure_df.empty or totals_df.empty:
        raise RuntimeError(f"No structure rows were produced. See {missing_path}")
    print(f"[S5_5_GEN] structure rows -> {structure_path}")
    print(f"[S5_5_GEN] totals -> {totals_path}")
    return structure_df, totals_df


def _valid_totals(totals_df: pd.DataFrame, cfg: Dict[str, object]) -> pd.DataFrame:
    out = totals_df.copy()
    out["total_emissions_gt"] = pd.to_numeric(out["total_emissions_gt"], errors="coerce")
    out = out.dropna(subset=["scenario_id", "sample_id", "total_emissions_gt"]).copy()
    out["sample_id"] = pd.to_numeric(out["sample_id"], errors="coerce").astype(int)
    out = out[~out["total_emissions_gt"].map(lambda x: _is_invalid_total_gt(x, cfg))].copy()
    return out.drop_duplicates(subset=["scenario_id", "sample_id"], keep="last").reset_index(drop=True)


def _target_values(totals_df: pd.DataFrame, cfg: Dict[str, object]) -> List[float]:
    configured = cfg.get("target_emissions_gt", None)
    if configured:
        return [float(x) for x in configured]
    vals = pd.to_numeric(totals_df["total_emissions_gt"], errors="coerce").dropna()
    if vals.empty:
        return []
    quantiles = np.linspace(0.05, 0.95, 10)
    return [float(vals.quantile(q)) for q in quantiles]


def _select_samples_near_target(totals_df: pd.DataFrame, target: float, cfg: Dict[str, object]) -> Tuple[pd.DataFrame, str]:
    work = totals_df.copy()
    work["distance_gt"] = (work["total_emissions_gt"] - float(target)).abs()
    window = float(cfg.get("target_window_gt", 0.25) or 0.25)
    nearest_n = int(cfg.get("target_nearest_n", 200) or 200)
    within = work[work["distance_gt"] <= window].copy()
    if len(within) >= max(10, nearest_n // 4):
        return within.sort_values(["distance_gt", "scenario_id"], kind="mergesort"), f"window_{window:g}gt"
    return work.sort_values(["distance_gt", "scenario_id"], kind="mergesort").head(nearest_n), f"nearest_{nearest_n}"


def _build_stack_tables(structure_df: pd.DataFrame, totals_df: pd.DataFrame, cfg: Dict[str, object], out_dir: Path) -> None:
    totals = _valid_totals(totals_df, cfg)
    if totals.empty:
        return
    structure = structure_df.copy()
    structure["emissions_gt"] = pd.to_numeric(structure["emissions_gt"], errors="coerce")
    structure["emissions_share"] = pd.to_numeric(structure["emissions_share"], errors="coerce")
    structure = structure.dropna(subset=["scenario_id", "sample_id", "grouping", "group_key", "emissions_share"])

    rows: List[Dict[str, object]] = []
    for target in _target_values(totals, cfg):
        selected, method = _select_samples_near_target(totals, target, cfg)
        keys = selected[["scenario_id", "sample_id"]].drop_duplicates()
        sub = structure.merge(keys, on=["scenario_id", "sample_id"], how="inner")
        if sub.empty:
            continue
        grouped = (
            sub.groupby(["grouping", "group_key", "group_1", "group_2", "group_3"], as_index=False, dropna=False)
            .agg(
                importance=("emissions_share", "mean"),
                mean_emissions_gt=("emissions_gt", "mean"),
                n_samples=("sample_id", "nunique"),
            )
        )
        for rec in grouped.to_dict("records"):
            rec["target_emission_gt"] = float(target)
            rec["selection_method"] = method
            rec["importance_pct"] = float(rec["importance"]) * 100.0
            rows.append(rec)

    long_df = pd.DataFrame(rows)
    if not long_df.empty:
        long_df["raw_importance"] = pd.to_numeric(long_df["importance"], errors="coerce").fillna(0.0)
        long_df["raw_importance_pct"] = long_df["raw_importance"] * 100.0
        long_df["positive_importance"] = long_df["raw_importance"].clip(lower=0.0)
        denom = (
            long_df.groupby(["target_emission_gt", "grouping"], dropna=False)["positive_importance"]
            .transform("sum")
            .replace(0.0, np.nan)
        )
        long_df["importance"] = long_df["positive_importance"].div(denom).fillna(0.0)
        long_df["importance_pct"] = long_df["importance"] * 100.0
    long_path = out_dir / "structure_stack_long.csv"
    long_df.to_csv(long_path, index=False, encoding="utf-8-sig")

    top_rows: List[Dict[str, object]] = []
    top_n_cfg = cfg.get("top_n_for_stack", {}) or {}
    for (target, grouping), sub in long_df.groupby(["target_emission_gt", "grouping"], sort=False):
        top_n = int(top_n_cfg.get(grouping, 15) or 15)
        work = sub.sort_values("importance", ascending=False, kind="mergesort").reset_index(drop=True)
        top = work.head(top_n).copy()
        for rec in top.to_dict("records"):
            rec["is_other"] = False
            top_rows.append(rec)
        other = work.iloc[top_n:].copy()
        if not other.empty:
            first = other.iloc[0].to_dict()
            top_rows.append(
                {
                    "target_emission_gt": float(target),
                    "selection_method": str(first.get("selection_method", "")),
                    "grouping": grouping,
                    "group_key": "Other",
                    "group_1": "Other",
                    "group_2": "",
                    "group_3": "",
                    "importance": float(other["importance"].sum()),
                    "importance_pct": float(other["importance"].sum() * 100.0),
                    "raw_importance": float(other["raw_importance"].sum()) if "raw_importance" in other.columns else np.nan,
                    "raw_importance_pct": (
                        float(other["raw_importance"].sum() * 100.0)
                        if "raw_importance" in other.columns
                        else np.nan
                    ),
                    "positive_importance": (
                        float(other["positive_importance"].sum())
                        if "positive_importance" in other.columns
                        else np.nan
                    ),
                    "mean_emissions_gt": float(other["mean_emissions_gt"].sum()),
                    "n_samples": int(other["n_samples"].max()),
                    "is_other": True,
                }
            )

    top_df = pd.DataFrame(top_rows)
    top_path = out_dir / "structure_stack_top_long.csv"
    top_df.to_csv(top_path, index=False, encoding="utf-8-sig")
    print(f"[S5_5_GEN] stack long -> {long_path}")
    print(f"[S5_5_GEN] stack top -> {top_path}")


def _rank_mode_for_panel(panel: str, cfg: Dict[str, object]) -> str:
    by_panel = cfg.get("quartile_rank_metric_mode_by_panel", {}) or {}
    mode = str(by_panel.get(panel, cfg.get("quartile_rank_metric_mode", "absolute")) or "absolute").strip().lower()
    if mode not in {"ratio", "absolute"}:
        raise ValueError("quartile rank metric mode must be ratio or absolute.")
    return mode


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _weighted_metric(weighted_df: pd.DataFrame, kind: str, mode: str) -> pd.DataFrame:
    if weighted_df.empty or "kind" not in weighted_df.columns:
        return pd.DataFrame()
    sub = weighted_df[weighted_df["kind"].astype(str).str.strip().str.lower().eq(kind.lower())].copy()
    if sub.empty:
        return pd.DataFrame()
    for col in (
        "sample_id",
        "weighted_value_sample",
        "weighted_value_y2020",
        "weighted_ratio",
        "weight_kcal_total",
        "weighted_co2eq_intensity_sample_kg_per_kcal",
    ):
        if col in sub.columns:
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
    if mode == "ratio":
        sub["metric_value"] = sub["weighted_ratio"]
    elif kind.lower() == "emission_factor" and "weighted_co2eq_intensity_sample_kg_per_kcal" in sub.columns:
        sub["metric_value"] = sub["weighted_co2eq_intensity_sample_kg_per_kcal"]
    else:
        sub["metric_value"] = sub["weighted_value_sample"]
        missing = sub["metric_value"].isna() & sub.get("weighted_value_y2020", pd.Series(index=sub.index)).notna()
        if missing.any() and "weighted_ratio" in sub.columns:
            sub.loc[missing, "metric_value"] = sub.loc[missing, "weighted_value_y2020"] * sub.loc[missing, "weighted_ratio"]
    if "weight_kcal_total" not in sub.columns:
        sub["weight_kcal_total"] = 0.0
    rows = []
    for keys, grp in sub.groupby(["scenario_id", "sample_id"], dropna=False):
        vals = pd.to_numeric(grp["metric_value"], errors="coerce")
        valid = vals.notna()
        if not valid.any():
            continue
        weights = pd.to_numeric(grp.loc[valid, "weight_kcal_total"], errors="coerce").fillna(0.0)
        if float(weights.sum()) > 0:
            metric = float(np.average(vals.loc[valid], weights=weights))
        else:
            metric = float(vals.loc[valid].mean())
        rows.append({"scenario_id": keys[0], "sample_id": int(keys[1]), "metric_value": metric})
    return pd.DataFrame(rows)


def _ruminant_metric(realized_df: pd.DataFrame, weighted_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "absolute" and not realized_df.empty and "realized_ruminant_share_kcal" in realized_df.columns:
        sub = realized_df.copy()
        if "scope" in sub.columns:
            sub = sub[sub["scope"].astype(str).str.strip().eq("global_summary")].copy()
        if {"scenario_id", "sample_id"}.issubset(sub.columns):
            sub["sample_id"] = pd.to_numeric(sub["sample_id"], errors="coerce")
            sub["metric_value"] = pd.to_numeric(sub["realized_ruminant_share_kcal"], errors="coerce")
            sub = sub.dropna(subset=["scenario_id", "sample_id", "metric_value"]).copy()
            sub["sample_id"] = sub["sample_id"].astype(int)
            return sub[["scenario_id", "sample_id", "metric_value"]].drop_duplicates()
    return _weighted_metric(weighted_df, "ruminant_reduction", mode)


def _luc_land_intensity_metric(structure_df: pd.DataFrame, land_df: pd.DataFrame, cfg: Dict[str, object]) -> pd.DataFrame:
    if structure_df.empty or land_df.empty:
        return pd.DataFrame()
    processes = {str(x).strip() for x in (cfg.get("luc_land_intensity_processes") or ()) if str(x).strip()}
    if not processes:
        return pd.DataFrame()
    struct = structure_df[structure_df["grouping"].astype(str).eq("Process")].copy()
    struct = struct[struct["group_1"].astype(str).isin(processes)].copy()
    if struct.empty:
        return pd.DataFrame()
    luc = (
        struct.groupby(["scenario_id", "sample_id"], as_index=False)["emissions_gt"].sum()
        .rename(columns={"emissions_gt": "luc_emissions_gt"})
    )

    land = land_df.copy()
    land["sample_id"] = pd.to_numeric(land.get("sample_id"), errors="coerce")
    value_cols = [c for c in ("crop_area_ha", "pasture_area_ha", "area_ha", "land_area_ha") if c in land.columns]
    if not value_cols:
        return pd.DataFrame()
    for col in value_cols:
        land[col] = pd.to_numeric(land[col], errors="coerce").fillna(0.0)
    land["crop_pasture_area_ha"] = 0.0
    if "crop_area_ha" in land.columns or "pasture_area_ha" in land.columns:
        land["crop_pasture_area_ha"] = land.get("crop_area_ha", 0.0) + land.get("pasture_area_ha", 0.0)
    else:
        land["crop_pasture_area_ha"] = land[value_cols].sum(axis=1)
    land_sum = land.groupby(["scenario_id", "sample_id"], as_index=False)["crop_pasture_area_ha"].sum()
    metric = luc.merge(land_sum, on=["scenario_id", "sample_id"], how="inner")
    metric["metric_value"] = metric["luc_emissions_gt"] * 1e9 / metric["crop_pasture_area_ha"].replace(0.0, np.nan)
    metric = metric.dropna(subset=["metric_value"]).copy()
    metric["sample_id"] = metric["sample_id"].astype(int)
    return metric[["scenario_id", "sample_id", "metric_value"]]


def _metric_display(metric_value: pd.Series, panel: str, mode: str) -> pd.Series:
    spec = PANEL_SPECS[panel]
    if spec["value_mode"] == "share_percent":
        return metric_value * 100.0
    if spec["value_mode"] == "change_percent" and mode == "ratio":
        return (metric_value - 1.0) * 100.0
    return metric_value


def _build_metric_samples(
    cfg: Dict[str, object],
    input_dirs: Sequence[Path],
    structure_df: pd.DataFrame,
    totals_df: pd.DataFrame,
) -> pd.DataFrame:
    weighted_df = _read_concat(input_dirs, "mc_success_weighted_elements.csv")
    realized_df = _read_concat(input_dirs, "mc_success_realized_ruminant_share.csv")
    land_df = _read_concat(input_dirs, "mc_success_crop_pasture_land_balance.csv")
    totals = _valid_totals(totals_df, cfg)[["scenario_id", "sample_id", "total_emissions_gt"]]
    panel_frames = []
    for panel in cfg.get("quartile_panels", []) or []:
        if panel not in PANEL_SPECS:
            continue
        mode = _rank_mode_for_panel(panel, cfg)
        spec = PANEL_SPECS[panel]
        if panel == "ruminate_intake":
            metric = _ruminant_metric(realized_df, weighted_df, mode)
        elif panel == "luc_land_intensity":
            metric = _luc_land_intensity_metric(structure_df, land_df, cfg)
        else:
            metric = _weighted_metric(weighted_df, str(spec["kind"]), mode)
        if metric.empty:
            print(f"[S5_5_GEN][WARN] skip quartile panel {panel}: no metric rows")
            continue
        metric["panel"] = panel
        metric["rank_metric_mode"] = mode
        metric["metric_label"] = str(spec["metric_label"])
        metric["metric_display"] = _metric_display(pd.to_numeric(metric["metric_value"], errors="coerce"), panel, mode)
        metric = metric.merge(totals, on=["scenario_id", "sample_id"], how="inner")
        panel_frames.append(metric)
    if not panel_frames:
        return pd.DataFrame()
    return pd.concat(panel_frames, ignore_index=True, sort=False)


def _assign_quartiles(panel_samples: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for panel, sub in panel_samples.groupby("panel", sort=False):
        work = sub.dropna(subset=["metric_value", "metric_display", "total_emissions_gt"]).copy()
        work = work.sort_values("metric_value", kind="mergesort").reset_index(drop=True)
        n = len(work)
        if n == 0:
            continue
        idx = np.minimum((np.arange(n) * 4) // n, 3)
        work["quartile_index"] = idx
        work["quartile_key"] = [QUARTILE_KEYS[int(i)] for i in idx]
        work["quartile_range"] = work["quartile_key"].map(QUARTILE_RANGES)
        frames.append(work)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _build_histogram(values: pd.Series, bins: int) -> List[Dict[str, object]]:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return []
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    if np.isclose(lo, hi):
        lo -= 0.5
        hi += 0.5
    counts, edges = np.histogram(vals, bins=int(bins), range=(lo, hi))
    return [
        {"bin_left": float(edges[i]), "bin_right": float(edges[i + 1]), "count": int(counts[i])}
        for i in range(len(counts))
    ]


def _build_quartile_tables(
    structure_df: pd.DataFrame,
    totals_df: pd.DataFrame,
    cfg: Dict[str, object],
    input_dirs: Sequence[Path],
    out_dir: Path,
) -> None:
    panel_samples = _build_metric_samples(cfg, input_dirs, structure_df, totals_df)
    if panel_samples.empty:
        return
    panel_samples = _assign_quartiles(panel_samples)
    if panel_samples.empty:
        return
    panel_samples_path = out_dir / "structure_quartile_panel_samples.csv"
    panel_samples.to_csv(panel_samples_path, index=False, encoding="utf-8-sig")

    structure = structure_df.copy()
    # structure rows already carry the same scenario-level total. Drop it here
    # so the merge below keeps the quartile/totals column name unsuffixed.
    structure = structure.drop(columns=["total_emissions_gt"], errors="ignore")
    for col in ("sample_id", "emissions_gt", "emissions_share"):
        structure[col] = pd.to_numeric(structure[col], errors="coerce")
    joined = structure.merge(
        panel_samples[
            [
                "scenario_id",
                "sample_id",
                "panel",
                "quartile_key",
                "quartile_range",
                "metric_value",
                "metric_display",
                "total_emissions_gt",
                "rank_metric_mode",
                "metric_label",
            ]
        ],
        on=["scenario_id", "sample_id"],
        how="inner",
    )
    if joined.empty:
        return

    summary = (
        joined.groupby(
            ["panel", "quartile_key", "quartile_range", "grouping", "group_key", "group_1", "group_2", "group_3"],
            as_index=False,
            dropna=False,
        )
        .agg(
            n_samples=("sample_id", "nunique"),
            metric_mean=("metric_display", "mean"),
            total_emissions_gt_mean=("total_emissions_gt", "mean"),
            total_emissions_gt_median=("total_emissions_gt", "median"),
            group_emissions_gt_mean=("emissions_gt", "mean"),
            group_emissions_gt_median=("emissions_gt", "median"),
            group_emissions_share_mean=("emissions_share", "mean"),
            group_emissions_share_median=("emissions_share", "median"),
        )
        .sort_values(["panel", "grouping", "quartile_key", "group_emissions_gt_mean"], ascending=[True, True, True, False])
        .reset_index(drop=True)
    )
    summary["group_emissions_share_pct_mean"] = summary["group_emissions_share_mean"] * 100.0
    summary_path = out_dir / "structure_quartile_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    hist_rows: List[Dict[str, object]] = []
    bins = int(cfg.get("histogram_bins", 120) or 120)
    top_n_cfg = cfg.get("top_n_for_quartile_histogram", {}) or {}
    top_lookup = set()
    overall = (
        structure.groupby(["grouping", "group_key"], as_index=False)["emissions_gt"]
        .mean()
        .rename(columns={"emissions_gt": "overall_mean_emissions_gt"})
    )
    for grouping, sub in overall.groupby("grouping", sort=False):
        top_n = int(top_n_cfg.get(grouping, 10) or 10)
        for rec in sub.sort_values("overall_mean_emissions_gt", ascending=False).head(top_n).to_dict("records"):
            top_lookup.add((str(rec["grouping"]), str(rec["group_key"])))

    for (panel, qkey, grouping, group_key), sub in joined.groupby(
        ["panel", "quartile_key", "grouping", "group_key"],
        sort=False,
    ):
        if (str(grouping), str(group_key)) not in top_lookup:
            continue
        for hist_rec in _build_histogram(sub["emissions_gt"], bins=bins):
            hist_rec.update(
                {
                    "panel": panel,
                    "quartile_key": qkey,
                    "quartile_range": QUARTILE_RANGES.get(str(qkey), ""),
                    "grouping": grouping,
                    "group_key": group_key,
                }
            )
            hist_rows.append(hist_rec)

    hist_path = out_dir / "structure_quartile_histogram_top.csv"
    pd.DataFrame(hist_rows).to_csv(hist_path, index=False, encoding="utf-8-sig")
    print(f"[S5_5_GEN] quartile samples -> {panel_samples_path}")
    print(f"[S5_5_GEN] quartile summary -> {summary_path}")
    print(f"[S5_5_GEN] quartile histogram top -> {hist_path}")


def _postprocess(cfg: Dict[str, object]) -> None:
    input_dirs, out_dir = _postprocess_dirs(cfg)
    _ensure_dir(out_dir)
    success_status = _load_success_status(input_dirs)
    write_sensitivity_cost_summaries(
        success_status,
        output_dir=out_dir,
        run_search_root=_root_output_dir(cfg),
    )
    structure_df, totals_df = _build_structure_outputs(cfg, input_dirs, out_dir)
    _build_stack_tables(structure_df, totals_df, cfg, out_dir)
    _build_quartile_tables(structure_df, totals_df, cfg, input_dirs, out_dir)
    meta_rows = [
        {"key": "input_dirs", "value": " | ".join(str(p) for p in input_dirs)},
        {"key": "output_dir", "value": str(out_dir)},
        {"key": "postprocess_scope", "value": str(cfg.get("postprocess_scope", "active"))},
        {"key": "year", "value": int(cfg.get("year", 2080) or 2080)},
        {"key": "groupings", "value": " | ".join(GROUPINGS)},
        {"key": "aggregation_reference", "value": "SP_M1a_Figure_pie_structure_pre.py"},
        {
            "key": "item_allocation_reference",
            "value": "Production-area split for LUC placeholders and Drained organic soils before Item mapping",
        },
    ]
    pd.DataFrame(meta_rows).to_csv(out_dir / "run_meta.csv", index=False, encoding="utf-8-sig")


def _parse_int(raw: object, *, name: str) -> int:
    try:
        return int(str(raw).strip())
    except Exception as exc:
        raise ValueError(f"Invalid integer for {name}: {raw}") from exc


def _parse_bool(raw: object) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env(prefix: str, name: str) -> str:
    return str(os.environ.get(f"{prefix}{name}", "") or "").strip()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run S5_5 Region/Item/Process importance MC generator.")
    parser.add_argument("--batch-index", type=int, default=None)
    parser.add_argument("--total-batches", type=int, default=None)
    parser.add_argument("--assignment", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--skip-mc", action="store_true", default=False)
    parser.add_argument("--postprocess-only", action="store_true", default=False)
    parser.add_argument("--no-postprocess", action="store_true", default=False)
    parser.add_argument("--postprocess-scope", choices=["active", "all_batches", "root"], default=None)
    return parser


def _apply_runtime_overrides(cfg: Dict[str, object], args: argparse.Namespace) -> Dict[str, object]:
    out = copy.deepcopy(cfg)
    batch_cfg = out.setdefault("batch", {})

    env_batch_index = _env("RIPMC_", "BATCH_INDEX")
    if env_batch_index:
        batch_cfg["batch_index"] = _parse_int(env_batch_index, name="RIPMC_BATCH_INDEX")
    env_total_batches = _env("RIPMC_", "TOTAL_BATCHES")
    if env_total_batches:
        batch_cfg["total_batches"] = _parse_int(env_total_batches, name="RIPMC_TOTAL_BATCHES")
    env_assignment = _env("RIPMC_", "BATCH_ASSIGNMENT")
    if env_assignment:
        batch_cfg["assignment"] = env_assignment
    env_output_dir = _env("RIPMC_", "OUTPUT_DIR")
    if env_output_dir:
        out["output_dir"] = env_output_dir
    env_samples = _env("RIPMC_", "SAMPLES")
    if env_samples:
        out["samples"] = _parse_int(env_samples, name="RIPMC_SAMPLES")
    env_max_runs = _env("RIPMC_", "MAX_RUNS")
    if env_max_runs:
        out["max_runs"] = _parse_int(env_max_runs, name="RIPMC_MAX_RUNS")
    env_resume = _env("RIPMC_", "RESUME")
    if env_resume:
        out["resume"] = _parse_bool(env_resume)
    env_scope = _env("RIPMC_", "POSTPROCESS_SCOPE")
    if env_scope:
        out["postprocess_scope"] = env_scope

    if args.batch_index is not None:
        batch_cfg["batch_index"] = int(args.batch_index)
    if args.total_batches is not None:
        batch_cfg["total_batches"] = int(args.total_batches)
    if args.assignment:
        batch_cfg["assignment"] = str(args.assignment)
    if args.output_dir:
        out["output_dir"] = str(args.output_dir)
    if args.samples is not None:
        out["samples"] = int(args.samples)
    if args.max_runs is not None:
        out["max_runs"] = int(args.max_runs)
    if args.resume is not None:
        out["resume"] = bool(args.resume)
    if args.postprocess_scope:
        out["postprocess_scope"] = str(args.postprocess_scope)
    if args.skip_mc or args.postprocess_only:
        out["run_mc"] = False
    if args.no_postprocess:
        out["postprocess"] = False
    if args.postprocess_only:
        out["postprocess"] = True
    return out


def main() -> None:
    args = _build_arg_parser().parse_args()
    cfg = _apply_runtime_overrides(CONFIG, args)
    mc_cfg = _fullmc_cfg(cfg)
    batch_state = _batch_state(mc_cfg)
    print(
        "[S5_5_GEN] "
        f"output_root={_root_output_dir(cfg)} "
        f"batch={batch_state.get('tag')} "
        f"enabled={batch_state.get('enabled')} "
        f"samples={cfg.get('samples')} "
        f"run_mc={cfg.get('run_mc')} "
        f"postprocess={cfg.get('postprocess')} "
        f"scope={cfg.get('postprocess_scope')}"
    )
    if bool(cfg.get("run_mc", True)):
        _run_mc(cfg)
    if bool(cfg.get("postprocess", True)):
        _postprocess(cfg)


if __name__ == "__main__":
    main()
