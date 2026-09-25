#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibrate GCE_parameters Synthetic fertilizers Emission factor for Y2020
using historical 2020 intensity from Fertilizer_efficiency.xlsx.
Only updates the 2020 column for paramName='Emission factor'.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional, List

import pandas as pd

MIN_POS = 1e-6


def _norm_m49(val: object) -> str:
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return ""
    if s.startswith("'"):
        s = s[1:]
    if s.count(".") == 1:
        left, right = s.split(".", 1)
        if left.isdigit() and right.strip("0") == "":
            s = left
    if s.isdigit():
        return f"'{s.zfill(3)}"
    return f"'{s}"


def _parse_year_col(col: object) -> Optional[int]:
    s = str(col).strip()
    if s.isdigit():
        return int(s)
    if s.startswith("Y") and s[1:].isdigit():
        return int(s[1:])
    return None


def _year_cols_in_range(df: pd.DataFrame, start_year: int, end_year: int) -> List[str]:
    cols: List[str] = []
    for col in df.columns:
        year = _parse_year_col(col)
        if year is not None and start_year <= year <= end_year:
            cols.append(col)
    return cols


def _row_mean_nonzero(df: pd.DataFrame, cols: List[str], min_value: float) -> pd.Series:
    if not cols:
        return pd.Series(index=df.index, dtype=float)
    vals = df[cols].apply(pd.to_numeric, errors="coerce")
    vals = vals.where(vals > min_value)
    return vals.mean(axis=1)


def _fill_from_history(
    df: pd.DataFrame,
    year_col: str,
    hist_cols: List[str],
    *,
    min_value: float,
    negative_only: bool,
) -> int:
    if not hist_cols:
        return 0
    y = pd.to_numeric(df[year_col], errors="coerce")
    if negative_only:
        mask = y < 0
    else:
        mask = y.isna() | (y <= min_value)
    row_mean = _row_mean_nonzero(df, hist_cols, min_value)
    fill_mask = mask & row_mean.notna()
    if fill_mask.any():
        df.loc[fill_mask, year_col] = row_mean[fill_mask]
    return int(fill_mask.sum())


def _fill_region_global(
    df: pd.DataFrame,
    year_col: str,
    region_map: Dict[str, str],
    *,
    min_value: float,
) -> Tuple[int, int, int]:
    df["_region_tmp"] = df["M49_norm"].map(region_map)
    y = pd.to_numeric(df[year_col], errors="coerce")
    base = df[y > min_value].copy()
    region_mean = {}
    if not base.empty:
        region_mean = (
            base.dropna(subset=["_region_tmp"])
            .groupby(["_region_tmp", "Item"])[year_col]
            .mean()
            .to_dict()
        )

    mask = y.isna() | (y <= min_value)
    filled_region = 0
    if region_mean and mask.any():
        for idx in df[mask].index:
            region = df.at[idx, "_region_tmp"]
            item = df.at[idx, "Item"]
            val = region_mean.get((region, item))
            if val is not None and pd.notna(val):
                df.at[idx, year_col] = val
                filled_region += 1

    y = pd.to_numeric(df[year_col], errors="coerce")
    mask = y.isna() | (y <= min_value)
    base = df[pd.to_numeric(df[year_col], errors="coerce") > min_value]
    global_item_mean = {}
    if not base.empty:
        global_item_mean = base.groupby("Item")[year_col].mean().to_dict()

    filled_global = 0
    if global_item_mean and mask.any():
        for idx in df[mask].index:
            item = df.at[idx, "Item"]
            val = global_item_mean.get(item)
            if val is not None and pd.notna(val):
                df.at[idx, year_col] = val
                filled_global += 1

    y = pd.to_numeric(df[year_col], errors="coerce")
    mask = y.isna() | (y <= min_value)
    filled_global_all = 0
    if not base.empty:
        global_mean_all = float(base[year_col].mean())
        if pd.notna(global_mean_all):
            df.loc[mask, year_col] = global_mean_all
            filled_global_all = int(mask.sum())

    df.drop(columns=["_region_tmp"], inplace=True)
    return filled_region, filled_global, filled_global_all


def _load_item_fert_map(dict_v3_path: Path) -> Dict[str, str]:
    if not dict_v3_path.exists():
        raise FileNotFoundError(f"dict_v3 not found: {dict_v3_path}")
    df = pd.read_excel(dict_v3_path, sheet_name="Emis_item")
    df.columns = [str(c).strip() for c in df.columns]
    need_cols = {"Process", "Item_Fertilizer_Map", "Item_Emis"}
    if not need_cols.issubset(df.columns):
        raise RuntimeError(f"dict_v3 Emis_item missing columns: {sorted(need_cols)}")
    df = df[df["Process"] == "Synthetic fertilizers"]
    mapping: Dict[str, str] = {}
    dup = {}
    for _, row in df.iterrows():
        k = str(row.get("Item_Fertilizer_Map", "")).strip()
        v = str(row.get("Item_Emis", "")).strip()
        if not k or not v or k.lower() == "nan" or v.lower() == "nan":
            continue
        if k in mapping and mapping[k] != v:
            dup.setdefault(k, set()).update({mapping[k], v})
        else:
            mapping[k] = v
    if dup:
        raise ValueError(f"Duplicate Item_Fertilizer_Map entries: {dup}")
    return mapping


def _load_region_mapping(dict_v3_path: Path) -> Tuple[Dict[str, str], str, str]:
    if not dict_v3_path.exists():
        raise FileNotFoundError(f"dict_v3 not found: {dict_v3_path}")
    sheet_name = "region"
    df = pd.read_excel(dict_v3_path, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]
    if "M49_Country_Code" not in df.columns:
        raise RuntimeError(f"{sheet_name} missing column: M49_Country_Code")
    if "Region_market_full" not in df.columns:
        raise RuntimeError(f"{sheet_name} missing region column Region_market_full")
    region_col = "Region_market_full"
    df = df[["M49_Country_Code", region_col]].copy()
    df["M49_norm"] = df["M49_Country_Code"].apply(_norm_m49)
    df[region_col] = df[region_col].astype(str).str.strip()
    df = df.dropna(subset=["M49_norm", region_col])
    mapping = dict(zip(df["M49_norm"], df[region_col]))
    return mapping, sheet_name, region_col


def main() -> None:
    here = Path(__file__).resolve().parent
    code_root = here.parent.parent
    src_base = (code_root / "src").resolve()
    input_base = (code_root / "input").resolve()
    dict_v3_path = src_base / "dict_v3.xlsx"
    gce_path = src_base / "GCE_parameters.xlsx"
    fert_path = input_base / "Fertilizer" / "Fertilizer_efficiency.xlsx"

    if not gce_path.exists():
        raise FileNotFoundError(f"GCE_parameters not found: {gce_path}")
    if not fert_path.exists():
        raise FileNotFoundError(f"Fertilizer_efficiency not found: {fert_path}")

    item_map = _load_item_fert_map(dict_v3_path)
    region_map, region_sheet, region_col = _load_region_mapping(dict_v3_path)

    gce = pd.read_excel(gce_path, sheet_name="GCE_parameters")
    gce.columns = [str(c).strip() for c in gce.columns]
    year_col = None
    if "Y2020" in gce.columns:
        year_col = "Y2020"
    elif "2020" in gce.columns:
        year_col = "2020"
    if year_col is None:
        raise RuntimeError("GCE_parameters missing year column 'Y2020'/'2020'")
    hist_cols = _year_cols_in_range(gce, 2000, 2019)

    gce["M49_norm"] = gce["M49_Country_Code"].apply(_norm_m49)
    rate_mask = (gce["Process"] == "Synthetic fertilizers") & (gce["paramName"] == "Fertlizer rate")
    rate_rows = gce[rate_mask].copy()
    rate_rows = rate_rows[rate_rows["M49_norm"] != ""]
    if rate_rows.empty:
        raise RuntimeError("No Synthetic fertilizers / Fertlizer rate rows found in GCE_parameters")
    if "Item" not in rate_rows.columns:
        raise RuntimeError("GCE_parameters missing Item column for Fertlizer rate")

    filled_hist_rate = _fill_from_history(
        rate_rows,
        year_col,
        hist_cols,
        min_value=MIN_POS,
        negative_only=False,
    )
    filled_region_rate, filled_global_rate, filled_global_all_rate = _fill_region_global(
        rate_rows,
        year_col,
        region_map,
        min_value=MIN_POS,
    )
    gce.loc[rate_rows.index, year_col] = rate_rows[year_col]

    rate_rows = gce.loc[rate_rows.index].copy()
    rate_rows[year_col] = pd.to_numeric(rate_rows[year_col], errors="coerce")
    valid_rate = rate_rows[rate_rows[year_col] > MIN_POS]
    rate_map: Dict[Tuple[str, str], float] = {}
    for r in valid_rate.itertuples(index=False):
        m49 = getattr(r, "M49_norm")
        item = str(getattr(r, "Item")).strip()
        if not m49:
            continue
        rate_map[(m49, item)] = float(getattr(r, year_col))

    fert = pd.read_excel(fert_path, sheet_name="data")
    fert.columns = [str(c).strip() for c in fert.columns]
    need = {"M49_Country_Code", "Item", "EmisN2O_Y2020", "Area_Y2020", "N_FertEffi_Y2020"}
    if not need.issubset(fert.columns):
        raise RuntimeError(f"Fertilizer_efficiency missing columns: {sorted(need)}")

    fert["M49_norm"] = fert["M49_Country_Code"].apply(_norm_m49)
    fert["Item_Emis"] = fert["Item"].astype(str).map(item_map)
    fert = fert.dropna(subset=["M49_norm", "Item_Emis"])
    fert = fert[fert["M49_norm"] != ""]

    fert["EmisN2O_kg"] = pd.to_numeric(fert["EmisN2O_Y2020"], errors="coerce") * 1e6
    fert["Area_ha"] = pd.to_numeric(fert["Area_Y2020"], errors="coerce")
    fert["N_rate_kg_per_ha"] = fert.apply(
        lambda r: rate_map.get((r["M49_norm"], r["Item_Emis"])),
        axis=1,
    )
    fert = fert[(fert["EmisN2O_kg"] > 0) & (fert["Area_ha"] > 0) & (fert["N_rate_kg_per_ha"] > MIN_POS)]
    if fert.empty:
        raise RuntimeError("No valid rows for 2020 calibration after Fertlizer rate fix")

    fert["N_input_kg"] = fert["Area_ha"] * fert["N_rate_kg_per_ha"]
    grouped = fert.groupby(["M49_norm", "Item_Emis"], as_index=False).agg({
        "EmisN2O_kg": "sum",
        "N_input_kg": "sum",
    })
    grouped = grouped[grouped["N_input_kg"] > 0]
    grouped["EF_2020"] = grouped["EmisN2O_kg"] / grouped["N_input_kg"]

    ef_map: Dict[Tuple[str, str], float] = {}
    for r in grouped.itertuples(index=False):
        ef_map[(r.M49_norm, r.Item_Emis)] = float(r.EF_2020)

    mask = (gce["Process"] == "Synthetic fertilizers") & (gce["paramName"] == "Emission factor")
    target = gce[mask]
    if target.empty:
        raise RuntimeError("No Synthetic fertilizers / Emission factor rows found in GCE_parameters")

    updated = 0
    for idx, row in target.iterrows():
        key = (row["M49_norm"], str(row.get("Item", "")).strip())
        if key in ef_map:
            gce.at[idx, year_col] = ef_map[key]
            updated += 1
    target = gce[mask].copy()
    filled_hist_ef = _fill_from_history(
        target,
        year_col,
        hist_cols,
        min_value=MIN_POS,
        negative_only=True,
    )
    filled_region_ef, filled_global_ef, filled_global_all_ef = _fill_region_global(
        target,
        year_col,
        region_map,
        min_value=MIN_POS,
    )
    gce.loc[target.index, year_col] = target[year_col]

    with pd.ExcelWriter(gce_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        gce.drop(columns=["M49_norm"]).to_excel(writer, sheet_name="GCE_parameters", index=False)

    print(
        "[OK] Fixed Fertlizer rate Y2020: hist={0}, region={1}, global_item={2}, global_all={3}.".format(
            filled_hist_rate, filled_region_rate, filled_global_rate, filled_global_all_rate
        )
    )
    print(
        "[OK] Updated {0} rows in GCE_parameters (Synthetic fertilizers Emission factor, 2020).".format(updated)
    )
    print(
        "[OK] Filled EF Y2020 using history(neg)={0}, region={1}, global_item={2}, global_all={3}. Region source: {4}.{5}".format(
            filled_hist_ef, filled_region_ef, filled_global_ef, filled_global_all_ef, region_sheet, region_col
        )
    )


if __name__ == "__main__":
    main()
