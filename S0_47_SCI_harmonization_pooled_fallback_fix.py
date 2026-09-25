# -*- coding: utf-8 -*-
"""
Post-fix pooled fallback targets for the existing SCI harmonization workbook.

This script does NOT rerun S0_44 on the raw SCI workbook.
Instead, it reads the current harmonized sheets plus diagnostics from:
  output/Plot/Fig6/SCI/SCI_Database_harmonization.xlsx

For rows whose historical targets came from trimmed-mean fallback rather than His,
it rebuilds a pooled fallback target across both scenario groups (1.5D + 2D)
using the current harmonized data, then re-targets the current harmonized series
to those pooled anchor values. The result is written to a new workbook so the
original file stays untouched.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from config_paths import get_results_base


FIG_DIR = Path(get_results_base()) / "Plot" / "Fig6" / "SCI"
INPUT_NAME = "SCI_Database_harmonization.xlsx"
OUTPUT_NAME = "SCI_Database_harmonization_pooled_fallback_fix.xlsx"

ANCHOR_YEAR = 2010
HARMONIZE_YEAR = 2020
CONVERGE_YEAR = 2100
MIN_COUNT_FOR_TRIM = 3
CLIP_SHARE_TO_01 = True

SHEET_MAP = {
    "1.5D_harmonized": "1.5D_diagnostics",
    "2D_harmonized": "2D_diagnostics",
}
VALID_METHODS = {"reduce_ratio", "reduce_offset", "constant_ratio", "constant_offset"}


def _year_cols(df: pd.DataFrame) -> Tuple[List[str], List[int]]:
    cols = [c for c in df.columns if re.fullmatch(r"Y\d{4}", str(c))]
    years = [int(str(c)[1:]) for c in cols]
    order = np.argsort(years)
    cols = [cols[i] for i in order]
    years = [years[i] for i in order]
    return cols, years


def _beta(year: int, base_year: int, converge_year: int) -> float:
    if converge_year == base_year:
        return 0.0
    if year <= base_year:
        return 1.0
    if year <= converge_year:
        return 1.0 - (year - base_year) / (converge_year - base_year)
    return 0.0


def _build_fallback_targets_from_scenario(df: pd.DataFrame, year: int, out_col: str) -> pd.DataFrame:
    base_col = f"Y{year}"
    group_cols = ["Region", "Variable", "Unit"]
    d = df[group_cols + [base_col]].copy()
    d[base_col] = pd.to_numeric(d[base_col], errors="coerce")

    def _agg(vals: pd.Series) -> float:
        arr = vals.dropna().to_numpy(dtype=float)
        if arr.size == 0:
            return np.nan
        if arr.size >= MIN_COUNT_FOR_TRIM:
            arr = np.sort(arr)[1:-1]
            if arr.size == 0:
                arr = vals.dropna().to_numpy(dtype=float)
        return float(np.mean(arr))

    return d.groupby(group_cols, as_index=False)[base_col].agg(_agg).rename(columns={base_col: out_col})


def _read_existing_workbook(path: Path) -> Dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(path, engine="openpyxl")
    return {sheet: pd.read_excel(path, sheet_name=sheet, engine="openpyxl") for sheet in xls.sheet_names}


def _is_share_like(df: pd.DataFrame) -> np.ndarray:
    variable = df["Variable"].astype(str)
    unit = df["Unit"].astype(str).str.lower()
    return unit.eq("share").to_numpy() | variable.str.contains("share", case=False, regex=False).to_numpy()


def _merge_diag(df_h: pd.DataFrame, df_diag: pd.DataFrame) -> pd.DataFrame:
    keys = ["Model", "Scenario", "Region", "Variable", "Unit"]
    use_cols = keys + [
        "target_2010",
        "target_2020",
        "target_source_2010",
        "target_source",
        "method",
    ]
    diag = df_diag[use_cols].copy()
    diag["__diag_row_id"] = diag.index
    out = df_h.merge(diag, on=keys, how="left", validate="many_to_one")
    return out


def _retarget_current_harmonized(
    df_h: pd.DataFrame,
    df_diag: pd.DataFrame,
    pooled_anchor: pd.DataFrame,
    pooled_base: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    year_cols, years = _year_cols(df_h)
    anchor_col = f"Y{ANCHOR_YEAR}"
    base_col = f"Y{HARMONIZE_YEAR}"

    if anchor_col not in year_cols or base_col not in year_cols:
        raise ValueError(f"Missing required year columns: {anchor_col}, {base_col}")

    work = _merge_diag(df_h, df_diag)
    work = work.merge(pooled_anchor, on=["Region", "Variable", "Unit"], how="left")
    work = work.merge(pooled_base, on=["Region", "Variable", "Unit"], how="left")

    work["target_source_2010"] = work["target_source_2010"].fillna("").astype(str)
    work["target_source"] = work["target_source"].fillna("").astype(str)
    work["method"] = work["method"].fillna("").astype(str)

    use_pooled_anchor = work["target_source_2010"].str.contains("trimmed_mean", na=False)
    use_pooled_base = work["target_source"].str.contains("trimmed_mean", na=False)
    valid_method = work["method"].isin(VALID_METHODS)

    model_anchor = pd.to_numeric(work[anchor_col], errors="coerce").to_numpy(dtype=float)
    model_base = pd.to_numeric(work[base_col], errors="coerce").to_numpy(dtype=float)
    pooled_anchor_vals = pd.to_numeric(work[f"target_{ANCHOR_YEAR}_pooled"], errors="coerce").to_numpy(dtype=float)
    pooled_base_vals = pd.to_numeric(work[f"target_{HARMONIZE_YEAR}_pooled"], errors="coerce").to_numpy(dtype=float)

    target_anchor = np.where(use_pooled_anchor, pooled_anchor_vals, model_anchor)
    target_base = np.where(use_pooled_base, pooled_base_vals, model_base)

    can_fix = (
        valid_method.to_numpy()
        & (use_pooled_anchor.to_numpy() | use_pooled_base.to_numpy())
        & np.isfinite(model_anchor)
        & np.isfinite(model_base)
        & np.isfinite(target_anchor)
        & np.isfinite(target_base)
    )

    ratio_anchor = np.full_like(target_anchor, np.nan, dtype=float)
    ratio_base = np.full_like(target_base, np.nan, dtype=float)
    np.divide(target_anchor, model_anchor, out=ratio_anchor, where=model_anchor != 0)
    np.divide(target_base, model_base, out=ratio_base, where=model_base != 0)
    offset_anchor = target_anchor - model_anchor
    offset_base = target_base - model_base
    method = work["method"].to_numpy(dtype=object)
    is_share = _is_share_like(work)

    df_out = df_h.copy()
    for col, year in zip(year_cols, years):
        m = pd.to_numeric(df_h[col], errors="coerce").to_numpy(dtype=float)

        if year <= HARMONIZE_YEAR:
            if HARMONIZE_YEAR == ANCHOR_YEAR:
                w = 1.0
            else:
                w = (year - ANCHOR_YEAR) / (HARMONIZE_YEAR - ANCHOR_YEAR)
                w = min(1.0, max(0.0, w))
            ratio_w = ratio_anchor + w * (ratio_base - ratio_anchor)
            offset_w = offset_anchor + w * (offset_base - offset_anchor)
            out_reduce_ratio = m * ratio_w
            out_reduce_offset = m + offset_w
        else:
            b = _beta(year, HARMONIZE_YEAR, CONVERGE_YEAR)
            out_reduce_ratio = m * (1.0 + b * (ratio_base - 1.0))
            out_reduce_offset = m + b * offset_base

        out_const_ratio = m * ratio_base
        out_const_offset = m + offset_base

        out = m.copy()
        out = np.where(can_fix & (method == "reduce_ratio"), out_reduce_ratio, out)
        out = np.where(can_fix & (method == "reduce_offset"), out_reduce_offset, out)
        out = np.where(can_fix & (method == "constant_ratio"), out_const_ratio, out)
        out = np.where(can_fix & (method == "constant_offset"), out_const_offset, out)

        if CLIP_SHARE_TO_01:
            out = np.where(can_fix & is_share, np.clip(out, 0.0, 1.0), out)

        if year == ANCHOR_YEAR:
            out = np.where(can_fix, target_anchor, out)
        elif year == HARMONIZE_YEAR:
            out = np.where(can_fix, target_base, out)

        df_out[col] = out

    diag_out = df_diag.copy()
    diag_row_id = pd.to_numeric(work["__diag_row_id"], errors="coerce")
    diag_row_id = diag_row_id[can_fix & diag_row_id.notna().to_numpy()].astype(int).to_numpy()
    if "target_2010" in diag_out.columns:
        diag_out.loc[diag_row_id, "target_2010"] = target_anchor[can_fix & pd.notna(work["__diag_row_id"]).to_numpy()]
    if "target_2020" in diag_out.columns:
        diag_out.loc[diag_row_id, "target_2020"] = target_base[can_fix & pd.notna(work["__diag_row_id"]).to_numpy()]
    if "target_source_2010" in diag_out.columns:
        mask = can_fix & use_pooled_anchor.to_numpy() & pd.notna(work["__diag_row_id"]).to_numpy()
        diag_out.loc[pd.to_numeric(work.loc[mask, "__diag_row_id"], errors="coerce").astype(int), "target_source_2010"] = (
            "postfix_pooled_trimmed_mean"
        )
    if "target_source" in diag_out.columns:
        mask = can_fix & use_pooled_base.to_numpy() & pd.notna(work["__diag_row_id"]).to_numpy()
        diag_out.loc[pd.to_numeric(work.loc[mask, "__diag_row_id"], errors="coerce").astype(int), "target_source"] = (
            "postfix_pooled_trimmed_mean"
        )

    fix_summary = (
        work.loc[can_fix, ["Region", "Variable", "Unit"]]
        .groupby(["Region", "Variable", "Unit"], as_index=False)
        .size()
        .rename(columns={"size": "rows_fixed"})
    )
    return df_out, diag_out, fix_summary


def main() -> None:
    input_path = FIG_DIR / INPUT_NAME
    output_path = FIG_DIR / OUTPUT_NAME
    workbook = _read_existing_workbook(input_path)

    combined = pd.concat([workbook[sheet].copy() for sheet in SHEET_MAP], ignore_index=True, sort=False)
    pooled_anchor = _build_fallback_targets_from_scenario(
        combined, ANCHOR_YEAR, f"target_{ANCHOR_YEAR}_pooled"
    )
    pooled_base = _build_fallback_targets_from_scenario(
        combined, HARMONIZE_YEAR, f"target_{HARMONIZE_YEAR}_pooled"
    )

    summary_rows: List[pd.DataFrame] = []
    for harm_sheet, diag_sheet in SHEET_MAP.items():
        fixed_h, fixed_d, fixed_summary = _retarget_current_harmonized(
            workbook[harm_sheet],
            workbook[diag_sheet],
            pooled_anchor,
            pooled_base,
        )
        workbook[harm_sheet] = fixed_h
        workbook[diag_sheet] = fixed_d
        if not fixed_summary.empty:
            fixed_summary.insert(0, "sheet", harm_sheet)
            summary_rows.append(fixed_summary)

    workbook["pooled_fallback_fix_summary"] = (
        pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet, df in workbook.items():
            df.to_excel(writer, sheet_name=sheet, index=False)

    print(f"[DONE] {output_path}")


if __name__ == "__main__":
    main()
