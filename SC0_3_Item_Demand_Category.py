# -*- coding: utf-8 -*-
"""
SC0_3_Item_Demand_Category.py
Build demand composition amount/ratio from FAOSTAT Food Balance Sheets.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from config_paths import get_input_base, get_src_base


ELEMENTS = [
    "Feed",
    "Food",
    "Losses",
    "Other uses (non-food)",
    "Seed",
]


def _norm_m49(val: Any) -> Optional[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if not s:
        return None
    if s.startswith("'"):
        s = s[1:]
    s = s.strip()
    if not s:
        return None
    if s.count(".") == 1:
        left, right = s.split(".", 1)
        if left.isdigit() and right.strip("0") == "":
            s = left
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return None
    return f"'{digits.zfill(3)}"


def _find_col(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    col_map = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        key = str(name).strip().lower()
        if key in col_map:
            return col_map[key]
    for c in df.columns:
        c_str = str(c).strip().lower()
        for name in names:
            if str(name).strip().lower() in c_str:
                return c
    return None


def _load_valid_m49(dict_path: str) -> Set[str]:
    df = pd.read_excel(dict_path, sheet_name="region")
    m49_col = _find_col(df, ["M49_Country_Code", "M49 Code", "M49"])
    if not m49_col:
        raise RuntimeError("dict_v3 region sheet missing M49 column")
    label_col = _find_col(df, ["Region_label_new"])
    if label_col:
        mask = df[label_col].astype(str).str.strip().str.lower() != "no"
        df = df[mask]
    m49_vals = df[m49_col].apply(_norm_m49)
    return {m for m in m49_vals if m}


def _load_demand_items(dict_path: str) -> Set[str]:
    df = pd.read_excel(dict_path, sheet_name="Emis_item")
    col = _find_col(df, ["Item_Demand_Map"])
    if not col:
        raise RuntimeError("dict_v3 Emis_item missing Item_Demand_Map column")
    out: Set[str] = set()
    for raw in df[col].dropna().astype(str):
        s = raw.strip()
        if not s or s.lower() == "no":
            continue
        parts = re.split(r"[;|]+", s)
        for part in parts:
            p = part.strip()
            if p:
                out.add(p)
    return out


def _read_fbs(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def _ensure_wide_years(df: pd.DataFrame) -> pd.DataFrame:
    if "Year" in df.columns and "Value" in df.columns:
        tmp = df.copy()
        tmp["Year"] = pd.to_numeric(tmp["Year"], errors="coerce").astype("Int64")
        tmp = tmp.dropna(subset=["Year"])
        tmp["Year"] = "Y" + tmp["Year"].astype(int).astype(str)
        id_cols = [c for c in tmp.columns if c not in ("Year", "Value")]
        wide = tmp.pivot_table(index=id_cols, columns="Year", values="Value", aggfunc="sum").reset_index()
        wide.columns = [str(c) for c in wide.columns]
        return wide
    year_cols = [
        c for c in df.columns
        if isinstance(c, str) and re.match(r"^Y\d{4}$", c.strip())
    ]
    if not year_cols:
        raise RuntimeError("No year columns found in FBS file")
    return df


def main() -> None:
    input_dir = os.path.join(get_input_base(), "Production_Trade")
    fbs_path = os.path.join(input_dir, "FoodBalanceSheets_E_All_Data_NOFLAG_demand_refilled.xlsx")
    if not os.path.exists(fbs_path):
        alt = fbs_path.replace(".xlsx", ".csv")
        if os.path.exists(alt):
            fbs_path = alt
        else:
            raise FileNotFoundError(f"FBS file not found: {fbs_path}")

    dict_path = os.path.join(get_src_base(), "dict_v3.xlsx")
    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"dict_v3 not found: {dict_path}")

    valid_m49 = _load_valid_m49(dict_path)
    valid_items = _load_demand_items(dict_path)

    df = _read_fbs(fbs_path)
    df.columns = [str(c).strip() for c in df.columns]
    m49_col = _find_col(df, ["Area Code (M49)", "M49_Country_Code", "M49 Code", "M49"])
    item_col = _find_col(df, ["Item"])
    element_col = _find_col(df, ["Element"])
    unit_col = _find_col(df, ["Unit"])
    area_col = _find_col(df, ["Area", "Country"])
    if not m49_col or not item_col or not element_col:
        raise RuntimeError(f"Missing required columns: {m49_col=}, {item_col=}, {element_col=}")

    df = _ensure_wide_years(df)
    df[m49_col] = df[m49_col].apply(_norm_m49)
    df[item_col] = df[item_col].astype(str).str.strip()
    df[element_col] = df[element_col].astype(str).str.strip()
    df = df[df[m49_col].isin(valid_m49)]
    df = df[df[item_col].isin(valid_items)]
    df = df[df[element_col].isin(ELEMENTS)]
    year_cols = [
        c for c in df.columns
        if isinstance(c, str) and re.match(r"^Y\d{4}$", c.strip())
    ]
    df[year_cols] = df[year_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    # FBS sometimes repeats identical rows with different item codes; drop exact duplicates after filtering.
    dedup_cols = [m49_col, item_col, element_col] + year_cols
    if area_col:
        dedup_cols.insert(1, area_col)
    if unit_col:
        dedup_cols.append(unit_col)
    before = len(df)
    df = df.drop_duplicates(subset=dedup_cols)
    removed = before - len(df)
    if removed > 0:
        print(f"[WARN] dropped {removed} exact duplicate FBS rows after filtering")
    if "Y2020" in year_cols:
        mask = (df[element_col] == "Food") & (df["Y2020"] <= 0)
        df.loc[mask, "Y2020"] = 0.01
        key_cols = [m49_col, item_col]
        all_keys = df.set_index(key_cols).index.unique()
        food_keys = df[df[element_col] == "Food"].set_index(key_cols).index.unique()
        missing_keys = all_keys.difference(food_keys)
        if len(missing_keys) > 0:
            area_map = df.groupby(key_cols)[area_col].first() if area_col else {}
            unit_map = df.groupby(key_cols)[unit_col].first() if unit_col else {}
            add_rows = []
            for key in missing_keys:
                row = {
                    m49_col: key[0],
                    item_col: key[1],
                    element_col: "Food",
                }
                if area_col:
                    row[area_col] = area_map.get(key)
                if unit_col:
                    row[unit_col] = unit_map.get(key)
                for yc in year_cols:
                    row[yc] = 0.0
                row["Y2020"] = 0.01
                add_rows.append(row)
            df = pd.concat([df, pd.DataFrame(add_rows)], ignore_index=True)

    cols = [m49_col, item_col, element_col] + year_cols
    if area_col and area_col not in cols:
        cols.insert(1, area_col)
    if unit_col and unit_col not in cols:
        cols.append(unit_col)
    amount = df[cols].rename(columns={
        m49_col: "M49_Country_Code",
        item_col: "Item",
        element_col: "Element",
        area_col: "Area",
        unit_col: "Unit",
    })

    ratio = amount.copy()
    group_cols = ["M49_Country_Code", "Item"]
    totals = ratio.groupby(group_cols, dropna=False)[year_cols].transform("sum")
    ratio_vals = ratio[year_cols].to_numpy(dtype=float)
    totals_vals = totals.to_numpy(dtype=float)
    ratio[year_cols] = np.divide(
        ratio_vals,
        totals_vals,
        out=np.zeros_like(ratio_vals),
        where=totals_vals > 0,
    )

    ratio["M49_Country_Code"] = ratio["M49_Country_Code"].apply(_norm_m49)
    ratio["Item"] = ratio["Item"].astype(str).str.strip()
    if "Y2020" in year_cols:
        full_keys = {(m49, item) for m49 in valid_m49 for item in valid_items}
        existing_keys = set(zip(ratio["M49_Country_Code"], ratio["Item"]))
        missing_keys = sorted(full_keys - existing_keys)
        if missing_keys:
            add_rows = []
            for m49, item in missing_keys:
                row = {
                    "M49_Country_Code": m49,
                    "Item": item,
                    "Element": "Food",
                }
                if "Area" in ratio.columns:
                    row["Area"] = None
                if "Unit" in ratio.columns:
                    row["Unit"] = None
                for yc in year_cols:
                    row[yc] = 0.0
                row["Y2020"] = 1.0
                add_rows.append(row)
            ratio = pd.concat([ratio, pd.DataFrame(add_rows)], ignore_index=True)

    out_path = os.path.join(input_dir, "Demand_composition.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        amount.to_excel(writer, sheet_name="amount", index=False)
        ratio.to_excel(writer, sheet_name="ratio", index=False)

    print(f"[OK] amount rows={len(amount)} ratio rows={len(ratio)}")
    print(f"[OK] output: {out_path}")


if __name__ == "__main__":
    main()
