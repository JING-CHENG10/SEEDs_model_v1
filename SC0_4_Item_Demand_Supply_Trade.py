# -*- coding: utf-8 -*-
"""
SC0_4_Item_Demand_Supply_Trade.py
Summarize 2010-2020 demand/supply/import/export by M49 and dict_v3 item.
"""
from __future__ import annotations

import os
from typing import Any, Iterable, Optional, Set, Tuple, Dict

import numpy as np
import pandas as pd

from config_paths import get_input_base, get_src_base
from S1_0_schema import ScenarioConfig
from S2_0_load_data import (
    DataPaths,
    build_universe_from_dict_v3,
    build_demand_total_from_fbs_domestic_supply,
    build_production_from_faostat,
    load_trade_import_export,
)

HIST_START = 2010
HIST_END = 2020
TOL_ABS = 1e-3
TOL_REL = 1e-4


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


def _load_valid_items(dict_path: str) -> Set[str]:
    df = pd.read_excel(dict_path, sheet_name="Emis_item")
    item_col = _find_col(df, ["Item_Emis", "Item Emis", "Item"])
    if not item_col:
        raise RuntimeError("dict_v3 Emis_item missing Item_Emis column")
    items = (
        df[item_col]
        .dropna()
        .astype(str)
        .str.strip()
    )
    out = {
        s for s in items
        if s and s.lower() not in {"nan", "no"}
    }
    return out


def _add_balance_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["balance_gap_t"] = df["supply_t"] + df["import_t"] - df["export_t"] - df["demand_t"]
    df["balance_gap_abs_t"] = df["balance_gap_t"].abs()
    scale = np.maximum(
        1.0,
        np.maximum.reduce([
            df["demand_t"].abs(),
            df["supply_t"].abs(),
            df["import_t"].abs(),
            df["export_t"].abs(),
        ]),
    )
    df["balance_gap_rel"] = df["balance_gap_abs_t"] / scale
    tol = np.maximum(TOL_ABS, TOL_REL * scale)
    df["balance_ok"] = df["balance_gap_abs_t"] <= tol
    return df


def _prep_demand(paths: DataPaths, dict_path: str, universe, valid_m49: Set[str], valid_items: Set[str]) -> pd.DataFrame:
    # Demand uses FBS Domestic Supply Quantity mapped via Item_Demand_Map.
    df = build_demand_total_from_fbs_domestic_supply(paths.fbs_csv, dict_path, universe)
    if df is None or df.empty:
        return pd.DataFrame(columns=["M49_Country_Code", "Item", "year", "demand_t"])
    df = df.copy()
    df["country"] = df["country"].apply(_norm_m49)
    df["commodity"] = df["commodity"].astype(str).str.strip()
    df = df[df["country"].isin(valid_m49)]
    df = df[df["commodity"].isin(valid_items)]
    df = df[(df["year"] >= HIST_START) & (df["year"] <= HIST_END)]
    df = df.rename(columns={"country": "M49_Country_Code", "commodity": "Item", "demand_total_t": "demand_t"})
    return df[["M49_Country_Code", "Item", "year", "demand_t"]]


def _prep_supply(paths: DataPaths, universe, valid_m49: Set[str], valid_items: Set[str]) -> pd.DataFrame:
    df = build_production_from_faostat(paths.production_faostat_csv, universe, fbs_csv=paths.fbs_csv)
    if df is None or df.empty:
        return pd.DataFrame(columns=["M49_Country_Code", "Item", "year", "supply_t"])
    df = df.copy()
    if "M49_Country_Code" in df.columns:
        df["M49_Country_Code"] = df["M49_Country_Code"].apply(_norm_m49)
    else:
        df["M49_Country_Code"] = df["country"].apply(_norm_m49)
    df["commodity"] = df["commodity"].astype(str).str.strip()
    df = df[df["M49_Country_Code"].isin(valid_m49)]
    df = df[df["commodity"].isin(valid_items)]
    df = df[(df["year"] >= HIST_START) & (df["year"] <= HIST_END)]
    df = df.rename(columns={"commodity": "Item", "production_t": "supply_t"})
    return df[["M49_Country_Code", "Item", "year", "supply_t"]]


def _prep_trade(paths: DataPaths, universe, valid_m49: Set[str], valid_items: Set[str]) -> pd.DataFrame:
    imports_map, exports_map = load_trade_import_export(paths, universe)
    if not imports_map and not exports_map:
        return pd.DataFrame(columns=["M49_Country_Code", "Item", "year", "import_t", "export_t"])
    keys = set(imports_map) | set(exports_map)
    rows = []
    for country, item, year in keys:
        m49 = _norm_m49(country)
        if not m49 or m49 not in valid_m49:
            continue
        item_str = str(item).strip()
        if item_str not in valid_items:
            continue
        try:
            year_int = int(year)
        except Exception:
            continue
        if year_int < HIST_START or year_int > HIST_END:
            continue
        rows.append({
            "M49_Country_Code": m49,
            "Item": item_str,
            "year": year_int,
            "import_t": float(imports_map.get((country, item, year), 0.0) or 0.0),
            "export_t": float(exports_map.get((country, item, year), 0.0) or 0.0),
        })
    if not rows:
        return pd.DataFrame(columns=["M49_Country_Code", "Item", "year", "import_t", "export_t"])
    df = pd.DataFrame(rows)
    return df


def main() -> None:
    dict_path = os.path.join(get_src_base(), "dict_v3.xlsx")
    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"dict_v3 not found: {dict_path}")

    valid_m49 = _load_valid_m49(dict_path)
    valid_items = _load_valid_items(dict_path)

    config = ScenarioConfig(
        years_hist_start=HIST_START,
        years_hist_end=HIST_END,
        years_future=[],
    )
    universe = build_universe_from_dict_v3(dict_path, config)
    paths = DataPaths()

    demand_df = _prep_demand(paths, dict_path, universe, valid_m49, valid_items)
    supply_df = _prep_supply(paths, universe, valid_m49, valid_items)
    trade_df = _prep_trade(paths, universe, valid_m49, valid_items)

    years = list(range(HIST_START, HIST_END + 1))
    base = pd.MultiIndex.from_product(
        [sorted(valid_m49), sorted(valid_items), years],
        names=["M49_Country_Code", "Item", "year"],
    ).to_frame(index=False)

    yearly = base.merge(demand_df, how="left", on=["M49_Country_Code", "Item", "year"])
    yearly = yearly.merge(supply_df, how="left", on=["M49_Country_Code", "Item", "year"])
    yearly = yearly.merge(trade_df, how="left", on=["M49_Country_Code", "Item", "year"])

    for col in ["demand_t", "supply_t", "import_t", "export_t"]:
        if col not in yearly.columns:
            yearly[col] = 0.0
        yearly[col] = pd.to_numeric(yearly[col], errors="coerce").fillna(0.0)

    yearly = _add_balance_cols(yearly)
    yearly["iso3"] = yearly["M49_Country_Code"].map(universe.iso3_by_country)
    yearly["country_name"] = yearly["M49_Country_Code"].map(universe.country_by_m49)

    summary = yearly.groupby(["M49_Country_Code", "Item"], as_index=False)[
        ["demand_t", "supply_t", "import_t", "export_t"]
    ].sum()
    summary = _add_balance_cols(summary)
    summary["iso3"] = summary["M49_Country_Code"].map(universe.iso3_by_country)
    summary["country_name"] = summary["M49_Country_Code"].map(universe.country_by_m49)

    imbalance_yearly = yearly[~yearly["balance_ok"]].copy()
    imbalance_yearly = imbalance_yearly.sort_values("balance_gap_abs_t", ascending=False)
    imbalance_summary = summary[~summary["balance_ok"]].copy()
    imbalance_summary = imbalance_summary.sort_values("balance_gap_abs_t", ascending=False)

    out_dir = os.path.join(get_input_base(), "Production_Trade")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"SC0_4_Demand_Supply_Trade_{HIST_START}_{HIST_END}.xlsx")
    yearly_csv_path = os.path.join(out_dir, f"SC0_4_Demand_Supply_Trade_yearly_{HIST_START}_{HIST_END}.csv")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary_2010_2020", index=False)
        imbalance_summary.to_excel(writer, sheet_name="imbalance_summary", index=False)
        if len(yearly) <= 1_000_000:
            yearly.to_excel(writer, sheet_name="yearly", index=False)
            imbalance_yearly.to_excel(writer, sheet_name="imbalance_yearly", index=False)
        else:
            imbalance_yearly.to_excel(writer, sheet_name="imbalance_yearly", index=False)

    if len(yearly) > 1_000_000:
        yearly.to_csv(yearly_csv_path, index=False)
        print(f"[WARN] yearly rows={len(yearly)} exceed Excel row limit; wrote CSV instead: {yearly_csv_path}")

    print(f"[OK] yearly rows={len(yearly)} summary rows={len(summary)}")
    print(f"[OK] output: {out_path}")


if __name__ == "__main__":
    main()
