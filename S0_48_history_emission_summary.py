from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from config_paths import get_input_base, get_src_base
from S1_0_schema import ScenarioConfig, Universe
from S2_0_load_data import (
    DataPaths,
    EmisItemMappings,
    _convert_production_unit,
    _convert_yield_unit,
    build_livestock_stock_from_env,
    build_universe_from_dict_v3,
    load_emis_item_mappings,
)
from S3_2_feed_demand import build_feed_demand_from_stock
from S4_1_results import summarize_emissions_from_detail
from gsoil_emission_complete import _load_historical_drained_organic_emissions


START_YEAR = 1961
END_YEAR = 2020
YEARS = list(range(START_YEAR, END_YEAR + 1))
YEAR_COLS = [f"Y{year}" for year in YEARS]

OUTPUT_NAME = "Emission_history_1961-2020_summary.xlsx"
EXCEL_MAX_ROWS = 1_048_576
EXCEL_MAX_DATA_ROWS = EXCEL_MAX_ROWS - 1
FISH_ITEM = "Fish, Seafood"
FISH_PROCESS = "Fish farming"
LULUCF_SHEET = "LULUCF_updated"
FISH_SHEET = "country_year_panel"
DRAINED_ORGANIC_SOILS_PROCESS = "Drained organic soils"
DRAINED_ORGANIC_SOILS_CROPLAND_ITEM = "Cropland organic soils"
DRAINED_ORGANIC_SOILS_PASTURE_ITEM = "Grassland organic soils"

AGGREGATE_REGION_LABELS = {"global", "world", "row"}
AGGREGATE_M49_CODES = {"000", "001"}

LUC_ALLOC_BASE_YEAR = 2020
LUC_BASE_SPLITS: Dict[str, Tuple[str, str]] = {
    "Buffalo": ("Buffalo, dairy", "Buffalo, non-dairy"),
    "Camels": ("Camel, dairy", "Camel, non-dairy"),
    "Cattle": ("Cattle, dairy", "Cattle, non-dairy"),
    "Chickens": ("Chickens, broilers", "Chickens, layers"),
    "Goats": ("Goats, dairy", "Goats, non-dairy"),
    "Sheep": ("Sheep, dairy", "Sheep, non-dairy"),
}


def _norm_m49(val) -> str:
    if val is None or pd.isna(val):
        return ""
    text = str(val).strip()
    if text.startswith("'"):
        text = text[1:]
    text = text.strip()
    if not text:
        return ""
    if text.count(".") == 1:
        left, right = text.split(".", 1)
        if left.isdigit() and right.strip("0") == "":
            text = left
    if text.isdigit():
        return f"'{text.zfill(3)}"
    return f"'{text}"


def _extract_year(col_name: str) -> int | None:
    text = str(col_name).strip()
    if text.startswith("Y") and text[1:].isdigit():
        return int(text[1:])
    if text.isdigit():
        return int(text)
    return None


def _clean_string(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def _normalize_code_series(series: pd.Series) -> pd.Series:
    return series.apply(_norm_m49)


def _available_year_cols(df: pd.DataFrame) -> List[str]:
    return [col for col in YEAR_COLS if col in df.columns]


def _load_region_maps(dict_v3_path: Path) -> tuple[Dict[str, str], Dict[str, str]]:
    region_df = pd.read_excel(dict_v3_path, sheet_name="region")
    region_df.columns = [str(col).strip() for col in region_df.columns]
    region_df["M49_Country_Code"] = _normalize_code_series(region_df["M49_Country_Code"])

    valid_region_df = region_df.dropna(subset=["M49_Country_Code", "Region_label_new"]).copy()
    valid_region_df["Region_label_new"] = _clean_string(valid_region_df["Region_label_new"])
    valid_region_df = valid_region_df[
        valid_region_df["M49_Country_Code"].ne("")
        & valid_region_df["Region_label_new"].ne("")
        & valid_region_df["Region_label_new"].str.casefold().ne("no")
    ].drop_duplicates(subset=["M49_Country_Code"], keep="first")

    region_label_map = dict(
        zip(valid_region_df["M49_Country_Code"], valid_region_df["Region_label_new"])
    )

    region_emis_df = None
    for sheet_name in ("region_map", "region"):
        try:
            region_emis_df = pd.read_excel(
                dict_v3_path,
                sheet_name=sheet_name,
                usecols=["M49_Country_Code", "Region_emisSum"],
            )
            break
        except ValueError:
            continue

    if region_emis_df is None:
        region_emis_map = {}
    else:
        region_emis_df["M49_Country_Code"] = _normalize_code_series(region_emis_df["M49_Country_Code"])
        region_emis_df["Region_emisSum"] = _clean_string(region_emis_df["Region_emisSum"])
        region_emis_df = region_emis_df.dropna(subset=["M49_Country_Code", "Region_emisSum"])
        region_emis_df = region_emis_df[
            region_emis_df["M49_Country_Code"].ne("")
            & region_emis_df["Region_emisSum"].ne("")
            & ~region_emis_df["Region_emisSum"].str.casefold().isin({"no", "nan"})
        ].drop_duplicates(subset=["M49_Country_Code"], keep="first")
        region_emis_map = dict(
            zip(region_emis_df["M49_Country_Code"], region_emis_df["Region_emisSum"])
        )

    return region_label_map, region_emis_map


def _load_emis_item_maps(
    dict_v3_path: Path,
) -> tuple[pd.DataFrame, Dict[str, set], Dict[str, str], Dict[str, str], Dict[str, str]]:
    emis_item_df = pd.read_excel(dict_v3_path, sheet_name="Emis_item")
    emis_item_df.columns = [str(col).strip() for col in emis_item_df.columns]

    for col in ["Process", "Process_map", "Item_Emis", "Item_EmisSum_map", "Item_Fertilizer_Map"]:
        if col in emis_item_df.columns:
            emis_item_df[col] = _clean_string(emis_item_df[col])

    valid_items_by_process: Dict[str, set] = {}
    valid_items = emis_item_df[["Process", "Item_Emis"]].dropna().copy()
    valid_items = valid_items[
        valid_items["Process"].ne("") & valid_items["Item_Emis"].ne("")
    ]
    for process, group_df in valid_items.groupby("Process", sort=False):
        valid_items_by_process[str(process)] = set(group_df["Item_Emis"].tolist())

    process_map_df = emis_item_df[["Process", "Process_map"]].dropna().copy()
    process_map_df = process_map_df[
        process_map_df["Process"].ne("") & process_map_df["Process_map"].ne("")
    ].drop_duplicates(subset=["Process"], keep="first")
    process_map = dict(zip(process_map_df["Process"], process_map_df["Process_map"]))

    item_sum_df = emis_item_df[["Item_Emis", "Item_EmisSum_map"]].dropna().copy()
    item_sum_df = item_sum_df[
        item_sum_df["Item_Emis"].ne("") & item_sum_df["Item_EmisSum_map"].ne("")
    ].drop_duplicates(subset=["Item_Emis"], keep="first")
    item_sum_map = dict(zip(item_sum_df["Item_Emis"], item_sum_df["Item_EmisSum_map"]))

    fertilizer_item_df = emis_item_df[["Item_Fertilizer_Map", "Item_Emis"]].dropna().copy()
    fertilizer_item_df = fertilizer_item_df[
        fertilizer_item_df["Item_Fertilizer_Map"].ne("") & fertilizer_item_df["Item_Emis"].ne("")
    ].drop_duplicates(subset=["Item_Fertilizer_Map"], keep="first")
    fertilizer_item_map = dict(
        zip(fertilizer_item_df["Item_Fertilizer_Map"], fertilizer_item_df["Item_Emis"])
    )

    return emis_item_df, valid_items_by_process, process_map, item_sum_map, fertilizer_item_map


def _alias_item_name(text: str) -> str:
    normalized = str(text).strip().casefold()
    normalized = normalized.replace(",", "").replace(" ", "")
    aliases = {
        "groundnuts": "Groundnut",
        "maize(corn)": "Maize (corn)",
        "potatoes": "Potatoes",
        "soyabeans": "Soya beans",
        "sugarbeet": "Sugarbeet",
        "sugarcane": "Sugar cane",
        "sweetpotatoes": "Sweetpotato",
        "treenutstotal": "Treenuts, Total",
    }
    return aliases.get(normalized, "")


def _read_faostat_history_wide(csv_path: Path) -> pd.DataFrame:
    header_df = pd.read_csv(csv_path, nrows=0)
    header_cols = [str(col).strip() for col in header_df.columns]

    m49_col = next(
        (col for col in ["M49_Country_Code", "Area Code (M49)"] if col in header_cols),
        None,
    )
    if m49_col is None:
        raise KeyError(f"Missing M49 column in {csv_path}")

    usecols = [m49_col, "Item", "Element"]
    if "Unit" in header_cols:
        usecols.append("Unit")
    if "Select" in header_cols:
        usecols.append("Select")
    usecols.extend([col for col in YEAR_COLS if col in header_cols])

    df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    df.columns = [str(col).strip() for col in df.columns]
    df["M49_Country_Code"] = _normalize_code_series(df[m49_col])
    if "Select" in df.columns:
        df["Select"] = pd.to_numeric(df["Select"], errors="coerce")
        df = df[df["Select"] == 1].copy()
    if "Unit" not in df.columns:
        df["Unit"] = ""
    df["Item"] = _clean_string(df["Item"])
    df["Element"] = _clean_string(df["Element"])
    df["Unit"] = _clean_string(df["Unit"])
    return df


def _melt_history_values(
    wide_df: pd.DataFrame,
    id_vars: List[str],
) -> pd.DataFrame:
    if wide_df.empty:
        return pd.DataFrame(columns=id_vars + ["year", "raw_value"])

    year_cols = _available_year_cols(wide_df)
    if not year_cols:
        return pd.DataFrame(columns=id_vars + ["year", "raw_value"])

    long_df = wide_df.melt(
        id_vars=id_vars,
        value_vars=year_cols,
        var_name="year_col",
        value_name="raw_value",
    )
    long_df["year"] = pd.to_numeric(long_df["year_col"].str.lstrip("Y"), errors="coerce")
    long_df["raw_value"] = pd.to_numeric(long_df["raw_value"], errors="coerce")
    long_df = long_df.dropna(subset=["year", "raw_value"]).copy()
    long_df["year"] = long_df["year"].astype(int)
    long_df = long_df[long_df["year"].between(START_YEAR, END_YEAR)].copy()
    return long_df.drop(columns=["year_col"]).reset_index(drop=True)


def _combine_preferred_history(
    frames: List[Tuple[int, pd.DataFrame]],
    key_cols: List[str],
    value_col: str,
) -> pd.DataFrame:
    prepared: List[pd.DataFrame] = []
    for priority, df in frames:
        if df is None or df.empty:
            continue
        tmp = df.copy()
        tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
        tmp = tmp.dropna(subset=key_cols + [value_col]).copy()
        if tmp.empty:
            continue
        tmp["_source_priority"] = int(priority)
        prepared.append(tmp)

    if not prepared:
        return pd.DataFrame(columns=key_cols + [value_col])

    combined = pd.concat(prepared, ignore_index=True)
    combined = combined.sort_values(key_cols + ["_source_priority"], kind="mergesort")
    combined = combined.drop_duplicates(subset=key_cols, keep="last")
    combined = combined.drop(columns="_source_priority")
    return combined.sort_values(key_cols, kind="mergesort").reset_index(drop=True)


def _load_crop_item_set(emis_item_df: pd.DataFrame) -> set:
    if emis_item_df.empty or "Item_Cat2" not in emis_item_df.columns:
        return set()
    item_cat2 = emis_item_df["Item_Cat2"].astype("string").str.strip().str.casefold()
    item_emis = emis_item_df["Item_Emis"].astype("string").str.strip()
    return set(item_emis[item_cat2 == "crop"].dropna().tolist())


def _extract_crop_production_from_wide(
    wide_df: pd.DataFrame,
    maps: EmisItemMappings,
    crop_items: set,
) -> pd.DataFrame:
    if wide_df.empty or not maps.production_by_item:
        return pd.DataFrame(columns=["M49_Country_Code", "year", "commodity", "production_t"])

    prod_wide = wide_df[
        wide_df["Item"].isin(maps.production_by_item.keys())
        & wide_df["Element"].astype(str).str.contains("Production", case=False, na=False)
    ].copy()
    if prod_wide.empty:
        return pd.DataFrame(columns=["M49_Country_Code", "year", "commodity", "production_t"])

    prod_long = _melt_history_values(prod_wide, ["M49_Country_Code", "Item", "Unit"])
    if prod_long.empty:
        return pd.DataFrame(columns=["M49_Country_Code", "year", "commodity", "production_t"])

    prod_long["commodity"] = prod_long["Item"].map(maps.production_by_item)
    prod_long = prod_long.dropna(subset=["commodity"]).copy()
    if crop_items:
        prod_long = prod_long[prod_long["commodity"].isin(crop_items)].copy()
    if prod_long.empty:
        return pd.DataFrame(columns=["M49_Country_Code", "year", "commodity", "production_t"])

    prod_long["production_t"] = prod_long.apply(
        lambda row: _convert_production_unit(float(row["raw_value"]), row["Unit"]),
        axis=1,
    )
    prod_long = prod_long.replace([np.inf, -np.inf], np.nan).dropna(subset=["production_t"])
    return (
        prod_long.groupby(["M49_Country_Code", "year", "commodity"], as_index=False)["production_t"]
        .sum()
        .sort_values(["M49_Country_Code", "year", "commodity"], kind="mergesort")
        .reset_index(drop=True)
    )


def _extract_crop_yield_from_wide(
    wide_df: pd.DataFrame,
    maps: EmisItemMappings,
    crop_items: set,
) -> pd.DataFrame:
    if wide_df.empty or not maps.yield_item_to_comm:
        return pd.DataFrame(columns=["M49_Country_Code", "year", "commodity", "yield_t_per_ha"])

    yield_wide = wide_df[wide_df["Item"].isin(maps.yield_item_to_comm.keys())].copy()
    if yield_wide.empty:
        return pd.DataFrame(columns=["M49_Country_Code", "year", "commodity", "yield_t_per_ha"])

    yield_wide["target_elem"] = yield_wide["Item"].map(maps.yield_element_by_item)
    yield_wide["element_norm"] = yield_wide["Element"].astype("string").str.strip().str.casefold()
    yield_wide["target_norm"] = yield_wide["target_elem"].astype("string").str.strip().str.casefold()
    mask_specific = yield_wide["target_elem"].notna() & (
        yield_wide["element_norm"] == yield_wide["target_norm"]
    )
    mask_default = yield_wide["target_elem"].isna() & yield_wide["element_norm"].str.contains(
        "yield",
        na=False,
    )
    yield_wide = yield_wide[mask_specific | mask_default].copy()
    if yield_wide.empty:
        return pd.DataFrame(columns=["M49_Country_Code", "year", "commodity", "yield_t_per_ha"])

    yield_long = _melt_history_values(yield_wide, ["M49_Country_Code", "Item", "Unit"])
    if yield_long.empty:
        return pd.DataFrame(columns=["M49_Country_Code", "year", "commodity", "yield_t_per_ha"])

    yield_long["commodity"] = yield_long["Item"].map(maps.yield_item_to_comm)
    yield_long = yield_long.dropna(subset=["commodity"]).copy()
    if crop_items:
        yield_long = yield_long[yield_long["commodity"].isin(crop_items)].copy()
    if yield_long.empty:
        return pd.DataFrame(columns=["M49_Country_Code", "year", "commodity", "yield_t_per_ha"])

    yield_long["yield_t_per_ha"] = yield_long.apply(
        lambda row: _convert_yield_unit(float(row["raw_value"]), row["Unit"]),
        axis=1,
    )
    yield_long = yield_long.replace([np.inf, -np.inf], np.nan).dropna(subset=["yield_t_per_ha"])
    yield_long = yield_long[yield_long["yield_t_per_ha"] > 0].copy()
    return (
        yield_long.groupby(["M49_Country_Code", "year", "commodity"], as_index=False)["yield_t_per_ha"]
        .mean()
        .sort_values(["M49_Country_Code", "year", "commodity"], kind="mergesort")
        .reset_index(drop=True)
    )


def _build_luc_crop_history_inputs(
    retired_production_path: Path,
    current_production_path: Path,
    maps: EmisItemMappings,
    crop_items: set,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    production_frames: List[Tuple[int, pd.DataFrame]] = []
    yield_frames: List[Tuple[int, pd.DataFrame]] = []
    for priority, csv_path in [(0, retired_production_path), (1, current_production_path)]:
        if not csv_path.exists():
            continue
        wide_df = _read_faostat_history_wide(csv_path)
        production_frames.append(
            (priority, _extract_crop_production_from_wide(wide_df, maps, crop_items))
        )
        yield_frames.append((priority, _extract_crop_yield_from_wide(wide_df, maps, crop_items)))

    production_df = _combine_preferred_history(
        production_frames,
        ["M49_Country_Code", "year", "commodity"],
        "production_t",
    )
    yield_df = _combine_preferred_history(
        yield_frames,
        ["M49_Country_Code", "year", "commodity"],
        "yield_t_per_ha",
    )
    return production_df, yield_df


def _build_stock_split_ratio_tables(
    stock_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stock_df = pd.read_csv(stock_path)
    stock_df.columns = [str(col).strip() for col in stock_df.columns]
    stock_df = stock_df[stock_df["Element"] == "Stocks"].copy()
    stock_df["M49_Country_Code"] = _normalize_code_series(stock_df["M49_Country_Code"])

    available_stock_years = [
        col
        for col in stock_df.columns
        if str(col).startswith("Y")
        and _extract_year(col) is not None
        and 2000 <= _extract_year(col) <= 2022
    ]
    if not available_stock_years:
        return (
            pd.DataFrame(columns=["M49_Country_Code", "base_item", "year", "commodity", "split_ratio"]),
            pd.DataFrame(columns=["base_item", "year", "commodity", "split_ratio_default"]),
        )

    specific_frames: List[pd.DataFrame] = []
    default_frames: List[pd.DataFrame] = []
    for base_item, split_items in LUC_BASE_SPLITS.items():
        aligned_parts: List[pd.DataFrame] = []
        for split_item in split_items:
            part_df = (
                stock_df[stock_df["Item"] == split_item]
                .set_index("M49_Country_Code")[available_stock_years]
            )
            aligned_parts.append(part_df)
        aligned_parts[0], aligned_parts[1] = aligned_parts[0].align(
            aligned_parts[1],
            fill_value=0,
        )
        total_df = aligned_parts[0] + aligned_parts[1]

        for split_idx, split_item in enumerate(split_items):
            fallback_default = 0.5 if base_item == "Chickens" else (0.0 if split_idx == 0 else 1.0)
            ratio_df = aligned_parts[split_idx].div(total_df).replace([np.inf, -np.inf], np.nan)
            default_series = ratio_df.mean(axis=0, skipna=True).fillna(fallback_default)
            ratio_df = ratio_df.fillna(default_series)

            if "Y2000" in ratio_df.columns:
                pre2000_series = ratio_df["Y2000"]
                default_pre2000 = float(default_series.get("Y2000", fallback_default))
            else:
                pre2000_series = pd.Series(fallback_default, index=ratio_df.index, dtype=float)
                default_pre2000 = fallback_default

            for year in range(START_YEAR, 2000):
                ratio_df[f"Y{year}"] = pre2000_series
                default_series[f"Y{year}"] = default_pre2000

            ratio_df = ratio_df.reindex(columns=YEAR_COLS)
            specific_long = ratio_df.reset_index().melt(
                id_vars=["M49_Country_Code"],
                value_vars=YEAR_COLS,
                var_name="year_col",
                value_name="split_ratio",
            )
            specific_long["year"] = specific_long["year_col"].str.lstrip("Y").astype(int)
            specific_long["base_item"] = base_item
            specific_long["commodity"] = split_item
            specific_frames.append(
                specific_long[
                    ["M49_Country_Code", "base_item", "year", "commodity", "split_ratio"]
                ]
            )

            default_frames.append(
                pd.DataFrame(
                    {
                        "base_item": base_item,
                        "year": YEARS,
                        "commodity": split_item,
                        "split_ratio_default": [
                            float(default_series.get(f"Y{year}", fallback_default)) for year in YEARS
                        ],
                    }
                )
            )

    specific_df = pd.concat(specific_frames, ignore_index=True) if specific_frames else pd.DataFrame()
    default_df = pd.concat(default_frames, ignore_index=True) if default_frames else pd.DataFrame()
    return specific_df, default_df


def _load_raw_stock_history(
    production_path: Path,
    universe: Universe,
    maps: EmisItemMappings,
    specific_ratio_df: pd.DataFrame,
    default_ratio_df: pd.DataFrame,
) -> pd.DataFrame:
    wide_df = _read_faostat_history_wide(production_path)
    stock_wide = wide_df[wide_df["Element"] == "Stocks"].copy()
    if stock_wide.empty:
        return pd.DataFrame(
            columns=["M49_Country_Code", "country", "iso3", "year", "commodity", "stock_head"]
        )

    stock_long = _melt_history_values(stock_wide, ["M49_Country_Code", "Item"])
    if stock_long.empty:
        return pd.DataFrame(
            columns=["M49_Country_Code", "country", "iso3", "year", "commodity", "stock_head"]
        )

    stock_long = stock_long[stock_long["raw_value"] > 0].copy()
    stock_long["country"] = stock_long["M49_Country_Code"]
    stock_long["iso3"] = stock_long["country"].map(universe.iso3_by_country)
    stock_long = stock_long.dropna(subset=["iso3"]).copy()

    direct_df = stock_long[stock_long["Item"].isin(maps.stock_item_to_comm.keys())].copy()
    direct_df["commodity"] = direct_df["Item"].map(maps.stock_item_to_comm)
    direct_df = direct_df.rename(columns={"raw_value": "stock_head"})
    direct_df = direct_df[
        ["M49_Country_Code", "country", "iso3", "year", "commodity", "stock_head"]
    ].copy()

    split_source = stock_long[stock_long["Item"].isin(LUC_BASE_SPLITS.keys())].copy()
    split_frames: List[pd.DataFrame] = []
    if not split_source.empty and not default_ratio_df.empty:
        split_df = split_source.merge(
            default_ratio_df,
            left_on=["Item", "year"],
            right_on=["base_item", "year"],
            how="left",
        )
        if not specific_ratio_df.empty:
            split_df = split_df.merge(
                specific_ratio_df,
                left_on=["M49_Country_Code", "Item", "year", "commodity"],
                right_on=["M49_Country_Code", "base_item", "year", "commodity"],
                how="left",
                suffixes=("_default", ""),
            )
        else:
            split_df["split_ratio"] = np.nan
        split_df["split_ratio"] = pd.to_numeric(split_df["split_ratio"], errors="coerce")
        split_df["split_ratio_default"] = pd.to_numeric(
            split_df["split_ratio_default"],
            errors="coerce",
        )
        split_df["ratio_final"] = split_df["split_ratio"].fillna(split_df["split_ratio_default"])
        split_df["stock_head"] = pd.to_numeric(split_df["raw_value"], errors="coerce") * split_df["ratio_final"]
        split_df = split_df.dropna(subset=["commodity", "stock_head"])
        split_df = split_df[split_df["stock_head"] > 0].copy()
        split_frames.append(
            split_df[
                ["M49_Country_Code", "country", "iso3", "year", "commodity", "stock_head"]
            ]
        )

    out_df = pd.concat([direct_df] + split_frames, ignore_index=True) if split_frames else direct_df
    return (
        out_df.groupby(
            ["M49_Country_Code", "country", "iso3", "year", "commodity"],
            as_index=False,
        )["stock_head"]
        .sum()
        .sort_values(["M49_Country_Code", "year", "commodity"], kind="mergesort")
        .reset_index(drop=True)
    )


def _build_luc_stock_history(
    retired_production_path: Path,
    stock_ratio_path: Path,
    universe: Universe,
    maps: EmisItemMappings,
) -> pd.DataFrame:
    specific_ratio_df, default_ratio_df = _build_stock_split_ratio_tables(stock_ratio_path)
    raw_stock_df = _load_raw_stock_history(
        retired_production_path,
        universe,
        maps,
        specific_ratio_df,
        default_ratio_df,
    )
    env_stock_df = build_livestock_stock_from_env(str(stock_ratio_path), universe)
    env_stock_df = env_stock_df[env_stock_df["year"].between(START_YEAR, END_YEAR)].copy()

    raw_stock_df["_source_priority"] = 0
    env_stock_df["_source_priority"] = 1
    combined = pd.concat([raw_stock_df, env_stock_df], ignore_index=True, sort=False)
    combined["stock_head"] = pd.to_numeric(combined["stock_head"], errors="coerce")
    combined = combined.dropna(
        subset=["M49_Country_Code", "country", "iso3", "year", "commodity", "stock_head"]
    )
    combined = combined.sort_values(
        ["M49_Country_Code", "year", "commodity", "_source_priority"],
        kind="mergesort",
    )
    combined = combined.drop_duplicates(
        subset=["M49_Country_Code", "year", "commodity"],
        keep="last",
    )
    return combined.drop(columns="_source_priority").reset_index(drop=True)


def _build_grass_alloc_df(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df is None or detail_df.empty:
        return pd.DataFrame(columns=["M49_Country_Code", "year", "commodity", "grass_dm_share"])

    detail_tmp = detail_df.copy()
    if "m49_code" in detail_tmp.columns and "M49_Country_Code" not in detail_tmp.columns:
        detail_tmp["M49_Country_Code"] = detail_tmp["m49_code"]
    required_cols = {"M49_Country_Code", "year", "commodity", "grass_dm_kg"}
    if not required_cols.issubset(detail_tmp.columns):
        return pd.DataFrame(columns=["M49_Country_Code", "year", "commodity", "grass_dm_share"])

    detail_tmp["M49_Country_Code"] = _normalize_code_series(detail_tmp["M49_Country_Code"])
    detail_tmp["year"] = pd.to_numeric(detail_tmp["year"], errors="coerce")
    detail_tmp["grass_dm_kg"] = pd.to_numeric(detail_tmp["grass_dm_kg"], errors="coerce")
    detail_tmp["commodity"] = _clean_string(detail_tmp["commodity"])
    detail_tmp = detail_tmp.dropna(subset=["M49_Country_Code", "year", "commodity", "grass_dm_kg"])
    detail_tmp = detail_tmp[detail_tmp["commodity"].ne("")].copy()
    detail_tmp["year"] = detail_tmp["year"].astype(int)

    alloc_df = (
        detail_tmp.groupby(["M49_Country_Code", "year", "commodity"], as_index=False)["grass_dm_kg"]
        .sum()
        .sort_values(["M49_Country_Code", "year", "commodity"], kind="mergesort")
        .reset_index(drop=True)
    )
    total = alloc_df.groupby(["M49_Country_Code", "year"])["grass_dm_kg"].transform("sum")
    alloc_df["grass_dm_share"] = np.where(total > 0, alloc_df["grass_dm_kg"] / total, np.nan)
    return alloc_df[["M49_Country_Code", "year", "commodity", "grass_dm_share"]]


def _build_feed_crop_area_df(
    crop_feed_df: pd.DataFrame,
    yield_df: pd.DataFrame,
    crop_items: set,
) -> pd.DataFrame:
    if crop_feed_df is None or crop_feed_df.empty or yield_df is None or yield_df.empty:
        return pd.DataFrame(columns=["M49_Country_Code", "year", "commodity", "feed_area_need_ha"])

    feed_df = crop_feed_df[["M49_Country_Code", "year", "commodity", "feed_t"]].copy()
    feed_df["M49_Country_Code"] = _normalize_code_series(feed_df["M49_Country_Code"])
    feed_df["year"] = pd.to_numeric(feed_df["year"], errors="coerce")
    feed_df["feed_t"] = pd.to_numeric(feed_df["feed_t"], errors="coerce")
    feed_df["commodity"] = _clean_string(feed_df["commodity"])
    feed_df = feed_df.dropna(subset=["M49_Country_Code", "year", "commodity", "feed_t"])
    if crop_items:
        feed_df = feed_df[feed_df["commodity"].isin(crop_items)].copy()
    if feed_df.empty:
        return pd.DataFrame(columns=["M49_Country_Code", "year", "commodity", "feed_area_need_ha"])

    yield_map = yield_df[["M49_Country_Code", "year", "commodity", "yield_t_per_ha"]].copy()
    yield_map["M49_Country_Code"] = _normalize_code_series(yield_map["M49_Country_Code"])
    yield_map["year"] = pd.to_numeric(yield_map["year"], errors="coerce")
    yield_map["yield_t_per_ha"] = pd.to_numeric(yield_map["yield_t_per_ha"], errors="coerce")
    yield_map["commodity"] = _clean_string(yield_map["commodity"])
    yield_map = yield_map.dropna(subset=["M49_Country_Code", "year", "commodity", "yield_t_per_ha"])

    feed_df["year"] = feed_df["year"].astype(int)
    yield_map["year"] = yield_map["year"].astype(int)
    feed_df = feed_df.merge(
        yield_map,
        on=["M49_Country_Code", "year", "commodity"],
        how="left",
    )
    feed_df = feed_df.dropna(subset=["yield_t_per_ha"])
    feed_df = feed_df[feed_df["yield_t_per_ha"] > 0].copy()
    if feed_df.empty:
        return pd.DataFrame(columns=["M49_Country_Code", "year", "commodity", "feed_area_need_ha"])

    feed_df["feed_area_need_ha"] = feed_df["feed_t"] / feed_df["yield_t_per_ha"]
    return (
        feed_df.groupby(["M49_Country_Code", "year", "commodity"], as_index=False)["feed_area_need_ha"]
        .sum()
        .sort_values(["M49_Country_Code", "year", "commodity"], kind="mergesort")
        .reset_index(drop=True)
    )


def _finalize_luc_alloc_area_df(
    area_df: Optional[pd.DataFrame],
    *,
    area_col: str,
    prefix: str,
    base_year: int = LUC_ALLOC_BASE_YEAR,
) -> pd.DataFrame:
    out_cols = [
        "M49_Country_Code",
        "year",
        "commodity",
        area_col,
        f"{prefix}_delta_cum_ha",
        f"{prefix}_delta_inc_ha",
    ]
    if area_df is None or area_df.empty:
        return pd.DataFrame(columns=out_cols)
    required_cols = {"M49_Country_Code", "year", "commodity", area_col}
    if not required_cols.issubset(area_df.columns):
        return pd.DataFrame(columns=out_cols)

    work = area_df[["M49_Country_Code", "year", "commodity", area_col]].copy()
    work["M49_Country_Code"] = _normalize_code_series(work["M49_Country_Code"])
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work["commodity"] = _clean_string(work["commodity"])
    work[area_col] = pd.to_numeric(work[area_col], errors="coerce")
    work = work.dropna(subset=["M49_Country_Code", "year", "commodity", area_col])
    work = work[work["commodity"].ne("")].copy()
    if work.empty:
        return pd.DataFrame(columns=out_cols)

    work["year"] = work["year"].astype(int)
    work[area_col] = work[area_col].clip(lower=0.0)
    work = (
        work.groupby(["M49_Country_Code", "year", "commodity"], as_index=False)[area_col]
        .sum()
        .sort_values(["M49_Country_Code", "commodity", "year"], kind="mergesort")
        .reset_index(drop=True)
    )

    base = (
        work[work["year"] == int(base_year)][["M49_Country_Code", "commodity", area_col]]
        .rename(columns={area_col: f"base_{prefix}_area_ha"})
        .drop_duplicates(subset=["M49_Country_Code", "commodity"], keep="last")
    )
    work = work.merge(base, on=["M49_Country_Code", "commodity"], how="left")
    work[f"base_{prefix}_area_ha"] = pd.to_numeric(
        work[f"base_{prefix}_area_ha"],
        errors="coerce",
    ).fillna(0.0)
    work[f"{prefix}_delta_cum_ha"] = work[area_col] - work[f"base_{prefix}_area_ha"]
    work[f"{prefix}_delta_inc_ha"] = (
        work.groupby(["M49_Country_Code", "commodity"])[f"{prefix}_delta_cum_ha"]
        .diff()
        .fillna(work[f"{prefix}_delta_cum_ha"])
    )
    return work[out_cols].reset_index(drop=True)


def _build_luc_crop_alloc_df(
    *,
    hist_production_df: Optional[pd.DataFrame],
    feed_crop_area_df: Optional[pd.DataFrame],
    yield_df: Optional[pd.DataFrame],
    crop_items: Optional[List[str]] = None,
    base_year: int = LUC_ALLOC_BASE_YEAR,
) -> pd.DataFrame:
    crop_items_set = {str(val).strip() for val in (crop_items or []) if str(val).strip()}
    yield_map = pd.DataFrame(columns=["M49_Country_Code", "year", "commodity", "yield_t_per_ha"])
    if isinstance(yield_df, pd.DataFrame) and not yield_df.empty:
        yield_map = yield_df[["M49_Country_Code", "year", "commodity", "yield_t_per_ha"]].copy()
        yield_map["M49_Country_Code"] = _normalize_code_series(yield_map["M49_Country_Code"])
        yield_map["year"] = pd.to_numeric(yield_map["year"], errors="coerce")
        yield_map["commodity"] = _clean_string(yield_map["commodity"])
        yield_map["yield_t_per_ha"] = pd.to_numeric(yield_map["yield_t_per_ha"], errors="coerce")
        yield_map = yield_map.dropna(
            subset=["M49_Country_Code", "year", "commodity", "yield_t_per_ha"]
        )
        if not yield_map.empty:
            yield_map["year"] = yield_map["year"].astype(int)
            yield_map = yield_map[yield_map["yield_t_per_ha"] > 0].copy()
            yield_map = (
                yield_map.groupby(["M49_Country_Code", "year", "commodity"], as_index=False)[
                    "yield_t_per_ha"
                ]
                .mean()
            )

    food_frames: List[pd.DataFrame] = []
    if isinstance(hist_production_df, pd.DataFrame) and not hist_production_df.empty and not yield_map.empty:
        hist = hist_production_df[["M49_Country_Code", "year", "commodity", "production_t"]].copy()
        hist["M49_Country_Code"] = _normalize_code_series(hist["M49_Country_Code"])
        hist["year"] = pd.to_numeric(hist["year"], errors="coerce")
        hist["commodity"] = _clean_string(hist["commodity"])
        hist["production_t"] = pd.to_numeric(hist["production_t"], errors="coerce")
        hist = hist.dropna(subset=["M49_Country_Code", "year", "commodity", "production_t"])
        if not hist.empty:
            hist["year"] = hist["year"].astype(int)
            hist = hist[hist["year"] <= int(base_year)].copy()
            if crop_items_set:
                hist = hist[hist["commodity"].isin(crop_items_set)].copy()
            if not hist.empty:
                hist = hist.merge(
                    yield_map,
                    on=["M49_Country_Code", "year", "commodity"],
                    how="left",
                )
                hist["crop_area_ha"] = hist["production_t"] / hist["yield_t_per_ha"]
                hist = hist.replace([np.inf, -np.inf], np.nan)
                hist = hist.dropna(subset=["crop_area_ha"])
                if not hist.empty:
                    hist = hist[["M49_Country_Code", "year", "commodity", "crop_area_ha"]].copy()
                    hist["source_priority"] = 0
                    food_frames.append(hist)

    food_area = pd.DataFrame(columns=["M49_Country_Code", "year", "commodity", "crop_area_ha"])
    if food_frames:
        food_area = pd.concat(food_frames, ignore_index=True)
        food_area = food_area.sort_values(
            ["M49_Country_Code", "year", "commodity", "source_priority"],
            kind="mergesort",
        ).drop_duplicates(
            subset=["M49_Country_Code", "year", "commodity"],
            keep="last",
        )
        food_area = food_area[["M49_Country_Code", "year", "commodity", "crop_area_ha"]].copy()

    area_parts: List[pd.DataFrame] = []
    if not food_area.empty:
        area_parts.append(food_area.rename(columns={"crop_area_ha": "crop_area_need_ha"}))
    if isinstance(feed_crop_area_df, pd.DataFrame) and not feed_crop_area_df.empty:
        feed_df = feed_crop_area_df[["M49_Country_Code", "year", "commodity", "feed_area_need_ha"]].copy()
        feed_df["M49_Country_Code"] = _normalize_code_series(feed_df["M49_Country_Code"])
        feed_df["year"] = pd.to_numeric(feed_df["year"], errors="coerce")
        feed_df["commodity"] = _clean_string(feed_df["commodity"])
        feed_df["feed_area_need_ha"] = pd.to_numeric(feed_df["feed_area_need_ha"], errors="coerce")
        feed_df = feed_df.dropna(subset=["M49_Country_Code", "year", "commodity", "feed_area_need_ha"])
        if not feed_df.empty:
            feed_df["year"] = feed_df["year"].astype(int)
            if crop_items_set:
                feed_df = feed_df[feed_df["commodity"].isin(crop_items_set)].copy()
            area_parts.append(feed_df.rename(columns={"feed_area_need_ha": "crop_area_need_ha"}))

    if not area_parts:
        return pd.DataFrame(
            columns=[
                "M49_Country_Code",
                "year",
                "commodity",
                "crop_area_need_ha",
                "crop_delta_cum_ha",
                "crop_delta_inc_ha",
            ]
        )

    area_df = pd.concat(area_parts, ignore_index=True)
    return _finalize_luc_alloc_area_df(
        area_df,
        area_col="crop_area_need_ha",
        prefix="crop",
        base_year=base_year,
    )


def _build_luc_pasture_alloc_df(
    *,
    grass_requirement_df: Optional[pd.DataFrame],
    grass_share_df: Optional[pd.DataFrame],
    base_year: int = LUC_ALLOC_BASE_YEAR,
) -> pd.DataFrame:
    out_cols = [
        "M49_Country_Code",
        "year",
        "commodity",
        "pasture_area_need_ha",
        "pasture_delta_cum_ha",
        "pasture_delta_inc_ha",
    ]
    if grass_requirement_df is None or grass_requirement_df.empty:
        return pd.DataFrame(columns=out_cols)
    if grass_share_df is None or grass_share_df.empty:
        return pd.DataFrame(columns=out_cols)

    req = grass_requirement_df[["M49_Country_Code", "year", "grass_area_need_ha"]].copy()
    req["M49_Country_Code"] = _normalize_code_series(req["M49_Country_Code"])
    req["year"] = pd.to_numeric(req["year"], errors="coerce")
    req["grass_area_need_ha"] = pd.to_numeric(req["grass_area_need_ha"], errors="coerce")
    req = req.dropna(subset=["M49_Country_Code", "year", "grass_area_need_ha"])
    if req.empty:
        return pd.DataFrame(columns=out_cols)
    req["year"] = req["year"].astype(int)
    req = req.groupby(["M49_Country_Code", "year"], as_index=False)["grass_area_need_ha"].mean()

    share = grass_share_df[["M49_Country_Code", "year", "commodity", "grass_dm_share"]].copy()
    share["M49_Country_Code"] = _normalize_code_series(share["M49_Country_Code"])
    share["year"] = pd.to_numeric(share["year"], errors="coerce")
    share["commodity"] = _clean_string(share["commodity"])
    share["grass_dm_share"] = pd.to_numeric(share["grass_dm_share"], errors="coerce")
    share = share.dropna(subset=["M49_Country_Code", "year", "commodity", "grass_dm_share"])
    if share.empty:
        return pd.DataFrame(columns=out_cols)
    share["year"] = share["year"].astype(int)
    share = (
        share.groupby(["M49_Country_Code", "year", "commodity"], as_index=False)["grass_dm_share"]
        .mean()
    )
    alloc = share.merge(req, on=["M49_Country_Code", "year"], how="left")
    alloc["pasture_area_need_ha"] = alloc["grass_area_need_ha"] * alloc["grass_dm_share"]
    alloc = alloc[["M49_Country_Code", "year", "commodity", "pasture_area_need_ha"]].copy()
    return _finalize_luc_alloc_area_df(
        alloc,
        area_col="pasture_area_need_ha",
        prefix="pasture",
        base_year=base_year,
    )


def _prepare_luc_alloc_lookup(
    alloc_df: Optional[pd.DataFrame],
    *,
    area_col: str,
    delta_col: str,
) -> Dict[Tuple[str, int], Dict[str, Any]]:
    lookup: Dict[Tuple[str, int], Dict[str, Any]] = {}
    if alloc_df is None or alloc_df.empty:
        return lookup
    required_cols = {"M49_Country_Code", "year", "commodity", area_col, delta_col}
    if not required_cols.issubset(alloc_df.columns):
        return lookup

    work = alloc_df[["M49_Country_Code", "year", "commodity", area_col, delta_col]].copy()
    work["M49_Country_Code"] = _normalize_code_series(work["M49_Country_Code"])
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work["commodity"] = _clean_string(work["commodity"])
    work[area_col] = pd.to_numeric(work[area_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    work[delta_col] = pd.to_numeric(work[delta_col], errors="coerce").fillna(0.0)
    work = work.dropna(subset=["M49_Country_Code", "year"])
    work = work[work["commodity"].ne("")].copy()
    if work.empty:
        return lookup

    work["year"] = work["year"].astype(int)
    work = (
        work.groupby(["M49_Country_Code", "year", "commodity"], as_index=False)
        .agg({area_col: "sum", delta_col: "sum"})
        .sort_values(["M49_Country_Code", "year", "commodity"], kind="mergesort")
        .reset_index(drop=True)
    )
    for (m49, year), group_df in work.groupby(["M49_Country_Code", "year"], sort=False):
        lookup[(str(m49), int(year))] = {
            "commodity": group_df["commodity"].astype(str).to_numpy(dtype=object),
            "area": group_df[area_col].to_numpy(dtype=float),
            "delta": group_df[delta_col].to_numpy(dtype=float),
        }
    return lookup


def _prepare_alloc_year_default_lookup(
    alloc_df: Optional[pd.DataFrame],
    *,
    area_col: str,
) -> Dict[int, Dict[str, Any]]:
    lookup: Dict[int, Dict[str, Any]] = {}
    if alloc_df is None or alloc_df.empty:
        return lookup
    required_cols = {"year", "commodity", area_col}
    if not required_cols.issubset(alloc_df.columns):
        return lookup

    work = alloc_df[["year", "commodity", area_col]].copy()
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work["commodity"] = _clean_string(work["commodity"])
    work[area_col] = pd.to_numeric(work[area_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    work = work.dropna(subset=["year", "commodity"])
    work = work[work["commodity"].ne("")].copy()
    if work.empty:
        return lookup

    work["year"] = work["year"].astype(int)
    work = (
        work.groupby(["year", "commodity"], as_index=False)[area_col]
        .sum()
        .sort_values(["year", "commodity"], kind="mergesort")
        .reset_index(drop=True)
    )
    for year, group_df in work.groupby("year", sort=False):
        lookup[int(year)] = {
            "commodity": group_df["commodity"].astype(str).to_numpy(dtype=object),
            "area": group_df[area_col].to_numpy(dtype=float),
        }
    return lookup


def _index_alloc_years_by_country(
    lookup: Dict[Tuple[str, int], Dict[str, Any]],
) -> Dict[str, List[int]]:
    years_by_country: Dict[str, List[int]] = {}
    for m49, year in lookup.keys():
        years_by_country.setdefault(str(m49), []).append(int(year))
    for m49 in list(years_by_country.keys()):
        years_by_country[m49] = sorted(set(years_by_country[m49]))
    return years_by_country


def _nearest_available_year(years: Iterable[int], target_year: int) -> Optional[int]:
    year_list = sorted(set(int(year) for year in years))
    if not year_list:
        return None
    target = int(target_year)
    return min(year_list, key=lambda year: (abs(year - target), year))


def _get_alloc_with_year_fallback(
    country_lookup: Dict[Tuple[str, int], Dict[str, Any]],
    country_years: Dict[str, List[int]],
    year_lookup: Dict[int, Dict[str, Any]],
    year_defaults: List[int],
    *,
    m49: str,
    year: int,
) -> Optional[Dict[str, Any]]:
    m49_key = str(m49)
    year_int = int(year)

    alloc = country_lookup.get((m49_key, year_int))
    if alloc is not None:
        return alloc

    nearest_country_year = _nearest_available_year(country_years.get(m49_key, []), year_int)
    if nearest_country_year is not None:
        alloc = country_lookup.get((m49_key, nearest_country_year))
        if alloc is not None:
            return alloc

    alloc = year_lookup.get(year_int)
    if alloc is not None:
        return alloc

    nearest_default_year = _nearest_available_year(year_defaults, year_int)
    if nearest_default_year is not None:
        return year_lookup.get(nearest_default_year)
    return None


def _validate_luc_group_conservation(
    before_df: Optional[pd.DataFrame],
    after_df: Optional[pd.DataFrame],
    *,
    group_cols: Optional[List[str]] = None,
    value_col: str = "value",
    atol: float = 1e-8,
) -> None:
    group_cols = group_cols or ["M49_Country_Code", "year", "Process", "GHG"]
    before = pd.DataFrame() if before_df is None else before_df.copy()
    after = pd.DataFrame() if after_df is None else after_df.copy()

    for col in group_cols + [value_col]:
        if col not in before.columns:
            before[col] = np.nan if col != value_col else 0.0
        if col not in after.columns:
            after[col] = np.nan if col != value_col else 0.0

    before[value_col] = pd.to_numeric(before[value_col], errors="coerce").fillna(0.0)
    after[value_col] = pd.to_numeric(after[value_col], errors="coerce").fillna(0.0)
    before_grp = before.groupby(group_cols, as_index=False, dropna=False)[value_col].sum()
    after_grp = after.groupby(group_cols, as_index=False, dropna=False)[value_col].sum()
    merged = before_grp.merge(
        after_grp,
        on=group_cols,
        how="outer",
        suffixes=("_before", "_after"),
    )
    if merged.empty:
        return
    merged[f"{value_col}_before"] = pd.to_numeric(
        merged[f"{value_col}_before"],
        errors="coerce",
    ).fillna(0.0)
    merged[f"{value_col}_after"] = pd.to_numeric(
        merged[f"{value_col}_after"],
        errors="coerce",
    ).fillna(0.0)
    merged["abs_diff"] = (merged[f"{value_col}_after"] - merged[f"{value_col}_before"]).abs()
    failing = merged[merged["abs_diff"] > float(atol)].copy()
    if not failing.empty:
        max_abs_diff = float(failing["abs_diff"].max())
        raise ValueError(
            "LUC allocation conservation failed "
            f"for {len(failing)} groups (max_abs_diff={max_abs_diff:.12f})"
        )


def _expand_luc_country_to_commodity(
    luc_df: Optional[pd.DataFrame],
    *,
    crop_alloc_df: Optional[pd.DataFrame],
    pasture_alloc_df: Optional[pd.DataFrame],
    base_year: int = LUC_ALLOC_BASE_YEAR,
) -> pd.DataFrame:
    if luc_df is None or luc_df.empty:
        return pd.DataFrame() if luc_df is None else luc_df.copy()

    work = luc_df.copy()
    work["M49_Country_Code"] = _normalize_code_series(work["M49_Country_Code"])
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work = work.dropna(subset=["year"]).copy()
    work["year"] = work["year"].astype(int)

    crop_lookup = _prepare_luc_alloc_lookup(
        crop_alloc_df,
        area_col="crop_area_need_ha",
        delta_col="crop_delta_inc_ha",
    )
    pasture_lookup = _prepare_luc_alloc_lookup(
        pasture_alloc_df,
        area_col="pasture_area_need_ha",
        delta_col="pasture_delta_inc_ha",
    )

    expanded_rows: List[Dict[str, Any]] = []
    for row in work.to_dict(orient="records"):
        process = str(row.get("Process", "") or "").strip()
        m49 = _norm_m49(row.get("M49_Country_Code"))
        year_val = row.get("year")
        try:
            year_int = int(year_val)
        except Exception:
            expanded_rows.append(row)
            continue

        if process == "De/Reforestation_crop":
            alloc = crop_lookup.get((str(m49), year_int))
        elif process == "De/Reforestation_pasture":
            alloc = pasture_lookup.get((str(m49), year_int))
        else:
            expanded_rows.append(row)
            continue

        if alloc is None or not m49:
            expanded_rows.append(row)
            continue

        row_value = pd.to_numeric(row.get("value"), errors="coerce")
        if pd.isna(row_value):
            expanded_rows.append(row)
            continue

        row_value_float = float(row_value)
        commodities = alloc["commodity"]
        area_weights = np.clip(np.nan_to_num(alloc["area"], nan=0.0), 0.0, None)
        delta_weights = np.nan_to_num(alloc["delta"], nan=0.0)

        if year_int <= int(base_year):
            weights = area_weights
        elif row_value_float > 0:
            weights = np.clip(delta_weights, 0.0, None)
            if float(weights.sum()) <= 0:
                weights = area_weights
        elif row_value_float < 0:
            weights = np.clip(-delta_weights, 0.0, None)
            if float(weights.sum()) <= 0:
                expanded_rows.append(row)
                continue
        else:
            expanded_rows.append(row)
            continue

        weight_sum = float(np.sum(weights))
        if weight_sum <= 0:
            expanded_rows.append(row)
            continue

        allocated_value = 0.0
        used_any = False
        for commodity, weight in zip(commodities, np.asarray(weights, dtype=float)):
            if not np.isfinite(weight) or weight <= 0:
                continue
            item_name = str(commodity or "").strip()
            if not item_name:
                continue
            new_row = dict(row)
            new_row["Item"] = item_name
            new_value = row_value_float * float(weight) / weight_sum
            new_row["value"] = new_value
            expanded_rows.append(new_row)
            allocated_value += new_value
            used_any = True

        if used_any:
            diff = row_value_float - allocated_value
            if abs(diff) > 1e-12:
                expanded_rows[-1]["value"] = float(expanded_rows[-1]["value"]) + diff
            continue

        expanded_rows.append(row)

    return pd.DataFrame(expanded_rows)


def _expand_drained_organic_soils_to_commodity(
    soil_df: Optional[pd.DataFrame],
    *,
    crop_alloc_df: Optional[pd.DataFrame],
    pasture_alloc_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if soil_df is None or soil_df.empty:
        return pd.DataFrame() if soil_df is None else soil_df.copy()

    work = soil_df.copy()
    required_cols = {"M49_Country_Code", "Process", "Item", "GHG", "year", "value"}
    if not required_cols.issubset(work.columns):
        return work

    work["M49_Country_Code"] = _normalize_code_series(work["M49_Country_Code"])
    work["Process"] = _clean_string(work["Process"])
    work["Item"] = _clean_string(work["Item"])
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work = work.dropna(subset=["year"]).copy()
    work["year"] = work["year"].astype(int)

    crop_lookup = _prepare_luc_alloc_lookup(
        crop_alloc_df,
        area_col="crop_area_need_ha",
        delta_col="crop_delta_inc_ha",
    )
    crop_default_lookup = _prepare_alloc_year_default_lookup(
        crop_alloc_df,
        area_col="crop_area_need_ha",
    )
    crop_country_years = _index_alloc_years_by_country(crop_lookup)
    crop_default_years = sorted(crop_default_lookup.keys())
    pasture_lookup = _prepare_luc_alloc_lookup(
        pasture_alloc_df,
        area_col="pasture_area_need_ha",
        delta_col="pasture_delta_inc_ha",
    )
    pasture_default_lookup = _prepare_alloc_year_default_lookup(
        pasture_alloc_df,
        area_col="pasture_area_need_ha",
    )
    pasture_country_years = _index_alloc_years_by_country(pasture_lookup)
    pasture_default_years = sorted(pasture_default_lookup.keys())

    expanded_rows: List[Dict[str, Any]] = []
    for row in work.to_dict(orient="records"):
        process = "" if pd.isna(row.get("Process")) else str(row.get("Process")).strip()
        item = "" if pd.isna(row.get("Item")) else str(row.get("Item")).strip()
        m49 = _norm_m49(row.get("M49_Country_Code"))
        year_val = row.get("year")
        try:
            year_int = int(year_val)
        except Exception:
            expanded_rows.append(row)
            continue

        if process != DRAINED_ORGANIC_SOILS_PROCESS:
            expanded_rows.append(row)
            continue
        if item == DRAINED_ORGANIC_SOILS_CROPLAND_ITEM:
            alloc = _get_alloc_with_year_fallback(
                crop_lookup,
                crop_country_years,
                crop_default_lookup,
                crop_default_years,
                m49=str(m49),
                year=year_int,
            )
        elif item == DRAINED_ORGANIC_SOILS_PASTURE_ITEM:
            alloc = _get_alloc_with_year_fallback(
                pasture_lookup,
                pasture_country_years,
                pasture_default_lookup,
                pasture_default_years,
                m49=str(m49),
                year=year_int,
            )
        else:
            expanded_rows.append(row)
            continue

        if alloc is None or not m49:
            expanded_rows.append(row)
            continue

        row_value = pd.to_numeric(row.get("value"), errors="coerce")
        if pd.isna(row_value):
            expanded_rows.append(row)
            continue

        row_value_float = float(row_value)
        commodities = alloc["commodity"]
        weights = np.clip(np.nan_to_num(alloc["area"], nan=0.0), 0.0, None)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0:
            expanded_rows.append(row)
            continue

        allocated_value = 0.0
        used_any = False
        for commodity, weight in zip(commodities, np.asarray(weights, dtype=float)):
            if not np.isfinite(weight) or weight <= 0:
                continue
            item_name = str(commodity or "").strip()
            if not item_name:
                continue
            new_row = dict(row)
            new_row["Item"] = item_name
            new_value = row_value_float * float(weight) / weight_sum
            new_row["value"] = new_value
            expanded_rows.append(new_row)
            allocated_value += new_value
            used_any = True

        if used_any:
            diff = row_value_float - allocated_value
            if abs(diff) > 1e-12:
                expanded_rows[-1]["value"] = float(expanded_rows[-1]["value"]) + diff
            continue

        expanded_rows.append(row)

    return pd.DataFrame(expanded_rows)


def _build_luc_alloc_inputs(
    dict_v3_path: Path,
    retired_production_path: Path,
    current_production_path: Path,
    stock_ratio_path: Path,
    emis_item_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    config = ScenarioConfig(years_hist_start=START_YEAR, years_hist_end=END_YEAR, years_future=[])
    universe = build_universe_from_dict_v3(str(dict_v3_path), config)
    maps = load_emis_item_mappings(str(dict_v3_path))
    crop_items = sorted(_load_crop_item_set(emis_item_df))

    production_df, yield_df = _build_luc_crop_history_inputs(
        retired_production_path,
        current_production_path,
        maps,
        set(crop_items),
    )
    stock_history_df = _build_luc_stock_history(
        retired_production_path,
        stock_ratio_path,
        universe,
        maps,
    )
    paths = DataPaths()
    feed_outputs = build_feed_demand_from_stock(
        stock_df=stock_history_df,
        universe=universe,
        maps=maps,
        paths=paths,
        years=YEARS,
    )
    grass_share_df = _build_grass_alloc_df(feed_outputs.species_dm_detail)
    feed_crop_area_df = _build_feed_crop_area_df(
        feed_outputs.crop_feed_demand,
        yield_df,
        set(crop_items),
    )
    crop_alloc_df = _build_luc_crop_alloc_df(
        hist_production_df=production_df,
        feed_crop_area_df=feed_crop_area_df,
        yield_df=yield_df,
        crop_items=crop_items,
        base_year=LUC_ALLOC_BASE_YEAR,
    )
    pasture_alloc_df = _build_luc_pasture_alloc_df(
        grass_requirement_df=feed_outputs.grass_requirement,
        grass_share_df=grass_share_df,
        base_year=LUC_ALLOC_BASE_YEAR,
    )

    diagnostics: Dict[str, object] = {
        "crop_alloc_rows": len(crop_alloc_df),
        "pasture_alloc_rows": len(pasture_alloc_df),
        "crop_items_count": len(crop_items),
        "stock_history_rows": len(stock_history_df),
        "feed_crop_rows": len(feed_outputs.crop_feed_demand),
        "grass_requirement_rows": len(feed_outputs.grass_requirement),
        "grass_alloc_rows": len(grass_share_df),
    }
    return crop_alloc_df, pasture_alloc_df, diagnostics


def _finalize_detail(
    detail_df: pd.DataFrame,
    region_label_map: Dict[str, str],
) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame(
            columns=[
                "M49_Country_Code",
                "Region_label_new",
                "Process",
                "Item",
                "GHG",
                "year",
                "value",
                "Source_Module",
            ]
        )

    out = detail_df.copy()
    out["M49_Country_Code"] = _normalize_code_series(out["M49_Country_Code"])
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["Process"] = _clean_string(out["Process"])
    out["Item"] = _clean_string(out["Item"])
    out["GHG"] = _clean_string(out["GHG"]).str.upper()

    out = out.dropna(subset=["year", "value", "Process", "Item", "GHG"])
    out["year"] = out["year"].astype(int)
    out = out[
        out["M49_Country_Code"].ne("")
        & out["Process"].ne("")
        & out["Item"].ne("")
        & out["GHG"].ne("")
        & out["year"].between(START_YEAR, END_YEAR)
    ].copy()
    out["Region_label_new"] = out["M49_Country_Code"].map(region_label_map)
    out = out[out["Region_label_new"].notna()].copy()
    out = out[out["Region_label_new"].astype(str).str.casefold() != "no"].copy()
    out = out[out["value"].notna() & (out["value"] != 0)].copy()

    group_cols = [
        "M49_Country_Code",
        "Region_label_new",
        "Process",
        "Item",
        "GHG",
        "year",
        "Source_Module",
    ]
    out = (
        out.groupby(group_cols, as_index=False, dropna=False)["value"]
        .sum()
        .sort_values(group_cols, kind="mergesort")
        .reset_index(drop=True)
    )
    return out


def _load_fertilizer_share_tables(
    fertilizer_path: Path,
    fertilizer_item_map: Dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fert_df = pd.read_excel(fertilizer_path, sheet_name=0)
    fert_df.columns = [str(col).strip() for col in fert_df.columns]

    m49_col = "M49_Country_Code" if "M49_Country_Code" in fert_df.columns else "M49 Code"
    fert_df["M49_Country_Code"] = _normalize_code_series(fert_df[m49_col])
    fert_df["Item_raw"] = _clean_string(fert_df["Item"])
    fert_df["Item"] = fert_df["Item_raw"].map(fertilizer_item_map)
    if "EmisN2O_Item" in fert_df.columns:
        fert_df["Item"] = fert_df["Item"].fillna(
            _clean_string(fert_df["EmisN2O_Item"]).map(_alias_item_name)
        )

    value_cols = []
    for col in fert_df.columns:
        text = str(col)
        if not text.startswith("EmisN2O_Y"):
            continue
        year = _extract_year(text.replace("EmisN2O_", "", 1))
        if year is not None and START_YEAR <= year <= END_YEAR:
            value_cols.append(col)
    if not value_cols:
        raise RuntimeError(f"Missing EmisN2O_Y#### columns in {fertilizer_path}")

    long_df = fert_df.melt(
        id_vars=["M49_Country_Code", "Item"],
        value_vars=value_cols,
        var_name="year",
        value_name="weight",
    )
    long_df["year"] = pd.to_numeric(long_df["year"].str.replace("EmisN2O_Y", "", regex=False), errors="coerce")
    long_df["weight"] = pd.to_numeric(long_df["weight"], errors="coerce")
    long_df = long_df.dropna(subset=["M49_Country_Code", "Item", "year", "weight"])
    long_df = long_df[
        long_df["M49_Country_Code"].ne("")
        & long_df["Item"].astype(str).str.strip().ne("")
        & (long_df["weight"] > 0)
    ].copy()
    long_df["year"] = long_df["year"].astype(int)

    country_shares = long_df.groupby(
        ["M49_Country_Code", "year", "Item"], as_index=False
    )["weight"].sum()
    country_shares["total_weight"] = country_shares.groupby(
        ["M49_Country_Code", "year"]
    )["weight"].transform("sum")
    country_shares = country_shares[country_shares["total_weight"] > 0].copy()
    country_shares["share"] = country_shares["weight"] / country_shares["total_weight"]
    country_shares = country_shares[["M49_Country_Code", "year", "Item", "share"]]

    global_year_shares = long_df.groupby(["year", "Item"], as_index=False)["weight"].sum()
    global_year_shares["total_weight"] = global_year_shares.groupby("year")["weight"].transform("sum")
    global_year_shares = global_year_shares[global_year_shares["total_weight"] > 0].copy()
    global_year_shares["share"] = global_year_shares["weight"] / global_year_shares["total_weight"]
    global_year_shares = global_year_shares[["year", "Item", "share"]]

    overall_shares = long_df.groupby(["Item"], as_index=False)["weight"].sum()
    total_weight = overall_shares["weight"].sum()
    overall_shares["share"] = overall_shares["weight"] / total_weight if total_weight > 0 else np.nan
    overall_shares = overall_shares[["Item", "share"]]

    return country_shares, global_year_shares, overall_shares


def _allocate_synthetic_fertilizer_items(
    total_long: pd.DataFrame,
    country_shares: pd.DataFrame,
    global_year_shares: pd.DataFrame,
    overall_shares: pd.DataFrame,
) -> pd.DataFrame:
    merged_country = total_long.merge(
        country_shares,
        on=["M49_Country_Code", "year"],
        how="left",
    )
    country_rows = merged_country[merged_country["Item"].notna()].copy()

    missing_country = merged_country[merged_country["Item"].isna()][
        ["M49_Country_Code", "year", "total_value"]
    ].drop_duplicates()
    merged_global = missing_country.merge(global_year_shares, on="year", how="left")
    global_rows = merged_global[merged_global["Item"].notna()].copy()

    missing_global = merged_global[merged_global["Item"].isna()][
        ["M49_Country_Code", "year", "total_value"]
    ].drop_duplicates()
    if not missing_global.empty:
        overall_rows = (
            missing_global.assign(_join_key=1)
            .merge(overall_shares.assign(_join_key=1), on="_join_key", how="left")
            .drop(columns="_join_key")
        )
        overall_rows = overall_rows[overall_rows["Item"].notna()].copy()
    else:
        overall_rows = pd.DataFrame(columns=["M49_Country_Code", "year", "total_value", "Item", "share"])

    non_empty_frames = [df for df in [country_rows, global_rows, overall_rows] if not df.empty]
    allocated = pd.concat(non_empty_frames, ignore_index=True) if non_empty_frames else pd.DataFrame()
    if allocated.empty:
        return pd.DataFrame(columns=["M49_Country_Code", "Item", "year", "value"])

    allocated["value"] = pd.to_numeric(allocated["total_value"], errors="coerce") * pd.to_numeric(
        allocated["share"], errors="coerce"
    )
    allocated = allocated.dropna(subset=["value"])
    allocated = allocated[allocated["value"] != 0].copy()
    return allocated[["M49_Country_Code", "Item", "year", "value"]]


def _build_crop_history_detail(
    crop_path: Path,
    fertilizer_path: Path,
    valid_items_by_process: Dict[str, set],
    fertilizer_item_map: Dict[str, str],
) -> pd.DataFrame:
    crop_df = pd.read_csv(crop_path)
    crop_df.columns = [str(col).strip() for col in crop_df.columns]
    crop_df["M49_Country_Code"] = _normalize_code_series(crop_df["M49_Country_Code"])

    year_cols = _available_year_cols(crop_df)
    element_map = {
        "Crop residues (Emissions N2O)": ("Crop residues", "N2O"),
        "Burning crop residues (Emissions CH4)": ("Burning crop residues", "CH4"),
        "Burning crop residues (Emissions N2O)": ("Burning crop residues", "N2O"),
        "Rice cultivation (Emissions CH4)": ("Rice cultivation", "CH4"),
    }

    frames: List[pd.DataFrame] = []

    for element, (process, ghg) in element_map.items():
        subset = crop_df[crop_df["Element"] == element][["M49_Country_Code", "Item"] + year_cols].copy()
        if subset.empty:
            continue
        long_df = subset.melt(
            id_vars=["M49_Country_Code", "Item"],
            value_vars=year_cols,
            var_name="year",
            value_name="value",
        )
        long_df["year"] = pd.to_numeric(long_df["year"].str.lstrip("Y"), errors="coerce")
        long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
        long_df["Process"] = process
        long_df["GHG"] = ghg
        long_df["Source_Module"] = "GCE"
        valid_items = valid_items_by_process.get(process, set())
        if valid_items:
            long_df = long_df[long_df["Item"].isin(valid_items)].copy()
        frames.append(long_df[["M49_Country_Code", "Process", "Item", "GHG", "year", "value", "Source_Module"]])

    synth_df = crop_df[crop_df["Element"] == "Synthetic fertilizers (Emissions N2O)"].copy()
    if not synth_df.empty:
        synth_totals = synth_df.groupby("M49_Country_Code", as_index=False)[year_cols].sum(min_count=1)
        synth_totals = synth_totals.melt(
            id_vars=["M49_Country_Code"],
            value_vars=year_cols,
            var_name="year",
            value_name="total_value",
        )
        synth_totals["year"] = pd.to_numeric(synth_totals["year"].str.lstrip("Y"), errors="coerce")
        synth_totals["total_value"] = pd.to_numeric(synth_totals["total_value"], errors="coerce")
        synth_totals = synth_totals.dropna(subset=["year", "total_value"])
        synth_totals["year"] = synth_totals["year"].astype(int)
        synth_totals = synth_totals[synth_totals["year"].between(START_YEAR, END_YEAR)].copy()

        country_shares, global_year_shares, overall_shares = _load_fertilizer_share_tables(
            fertilizer_path,
            fertilizer_item_map,
        )
        synth_alloc = _allocate_synthetic_fertilizer_items(
            synth_totals,
            country_shares,
            global_year_shares,
            overall_shares,
        )
        if not synth_alloc.empty:
            synth_alloc["Process"] = "Synthetic fertilizers"
            synth_alloc["GHG"] = "N2O"
            synth_alloc["Source_Module"] = "GCE"
            valid_items = valid_items_by_process.get("Synthetic fertilizers", set())
            if valid_items:
                synth_alloc = synth_alloc[synth_alloc["Item"].isin(valid_items)].copy()
            frames.append(
                synth_alloc[
                    ["M49_Country_Code", "Process", "Item", "GHG", "year", "value", "Source_Module"]
                ]
            )

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _build_livestock_ratio_lookup(stock_path: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    stock_df = pd.read_csv(stock_path)
    stock_df.columns = [str(col).strip() for col in stock_df.columns]
    stock_df = stock_df[stock_df["Element"] == "Stocks"].copy()
    stock_df["M49_Country_Code"] = _normalize_code_series(stock_df["M49_Country_Code"])

    available_stock_years = [
        col for col in stock_df.columns
        if str(col).startswith("Y") and _extract_year(col) is not None and 2000 <= _extract_year(col) <= 2022
    ]
    if not available_stock_years:
        return {}

    base_to_split = {
        "Buffalo": ("Buffalo, dairy", "Buffalo, non-dairy"),
        "Camels": ("Camel, dairy", "Camel, non-dairy"),
        "Goats": ("Goats, dairy", "Goats, non-dairy"),
        "Sheep": ("Sheep, dairy", "Sheep, non-dairy"),
    }

    ratio_lookup: Dict[str, Dict[str, pd.DataFrame]] = {}
    for base_item, (dairy_item, non_dairy_item) in base_to_split.items():
        dairy_df = (
            stock_df[stock_df["Item"] == dairy_item]
            .set_index("M49_Country_Code")[available_stock_years]
        )
        non_dairy_df = (
            stock_df[stock_df["Item"] == non_dairy_item]
            .set_index("M49_Country_Code")[available_stock_years]
        )
        dairy_df, non_dairy_df = dairy_df.align(non_dairy_df, fill_value=0)
        total_df = dairy_df + non_dairy_df

        dairy_ratio = dairy_df.div(total_df).replace([np.inf, -np.inf], np.nan).fillna(0)
        non_dairy_ratio = non_dairy_df.div(total_df).replace([np.inf, -np.inf], np.nan).fillna(0)
        no_ratio_mask = total_df.isna() | total_df.eq(0)
        dairy_ratio = dairy_ratio.mask(no_ratio_mask, 0)
        non_dairy_ratio = non_dairy_ratio.mask(no_ratio_mask, 1)

        if "Y2000" in dairy_ratio.columns:
            for year in range(START_YEAR, 2000):
                dairy_ratio[f"Y{year}"] = dairy_ratio["Y2000"]
                non_dairy_ratio[f"Y{year}"] = non_dairy_ratio["Y2000"]

        ordered_cols = [f"Y{year}" for year in YEARS]
        ratio_lookup[base_item] = {
            dairy_item: dairy_ratio.reindex(columns=ordered_cols),
            non_dairy_item: non_dairy_ratio.reindex(columns=ordered_cols),
        }

    return ratio_lookup


def _split_livestock_rows(
    livestock_df: pd.DataFrame,
    ratio_lookup: Dict[str, Dict[str, pd.DataFrame]],
    year_cols: List[str],
) -> pd.DataFrame:
    base_to_split = {
        "Buffalo": ("Buffalo, dairy", "Buffalo, non-dairy"),
        "Camels": ("Camel, dairy", "Camel, non-dairy"),
        "Goats": ("Goats, dairy", "Goats, non-dairy"),
        "Sheep": ("Sheep, dairy", "Sheep, non-dairy"),
    }

    rows_to_split = livestock_df["Item"].isin(base_to_split.keys())
    split_source = livestock_df[rows_to_split].copy()
    keep_df = livestock_df[~rows_to_split].copy()

    if split_source.empty:
        return livestock_df

    split_rows: List[dict] = []
    for _, row in split_source.iterrows():
        base_item = str(row["Item"]).strip()
        country_code = _norm_m49(row["M49_Country_Code"])
        dairy_item, non_dairy_item = base_to_split[base_item]
        ratio_frames = ratio_lookup.get(base_item, {})
        dairy_ratios = ratio_frames.get(dairy_item)
        non_dairy_ratios = ratio_frames.get(non_dairy_item)

        if dairy_ratios is not None and country_code in dairy_ratios.index:
            dairy_ratio_values = dairy_ratios.loc[country_code]
            non_dairy_ratio_values = non_dairy_ratios.loc[country_code]
        else:
            dairy_ratio_values = pd.Series(0, index=year_cols, dtype=float)
            non_dairy_ratio_values = pd.Series(1, index=year_cols, dtype=float)

        dairy_row = row.to_dict()
        dairy_row["Item"] = dairy_item
        non_dairy_row = row.to_dict()
        non_dairy_row["Item"] = non_dairy_item

        for year_col in year_cols:
            raw_value = pd.to_numeric(row.get(year_col), errors="coerce")
            if pd.isna(raw_value):
                dairy_row[year_col] = np.nan
                non_dairy_row[year_col] = np.nan
                continue
            dairy_ratio = pd.to_numeric(dairy_ratio_values.get(year_col), errors="coerce")
            non_dairy_ratio = pd.to_numeric(non_dairy_ratio_values.get(year_col), errors="coerce")
            dairy_ratio = 0.0 if pd.isna(dairy_ratio) else float(dairy_ratio)
            non_dairy_ratio = 1.0 if pd.isna(non_dairy_ratio) else float(non_dairy_ratio)
            dairy_row[year_col] = raw_value * dairy_ratio
            non_dairy_row[year_col] = raw_value * non_dairy_ratio

        split_rows.append(dairy_row)
        split_rows.append(non_dairy_row)

    split_df = pd.DataFrame(split_rows)
    return pd.concat([keep_df, split_df], ignore_index=True)


def _build_livestock_history_detail(
    livestock_path: Path,
    stock_ratio_path: Path,
    valid_items_by_process: Dict[str, set],
) -> pd.DataFrame:
    livestock_df = pd.read_csv(livestock_path)
    livestock_df.columns = [str(col).strip() for col in livestock_df.columns]
    livestock_df["M49_Country_Code"] = _normalize_code_series(livestock_df["M49_Country_Code"])

    element_map = {
        "Enteric fermentation (Emissions CH4)": ("Enteric fermentation", "CH4"),
        "Manure management (Emissions CH4)": ("Manure management", "CH4"),
        "Manure management (Emissions N2O)": ("Manure management", "N2O"),
        "Manure applied to soils (Emissions N2O)": ("Manure applied to soils", "N2O"),
        "Manure left on pasture (Emissions N2O)": ("Manure left on pasture", "N2O"),
    }

    livestock_df = livestock_df[livestock_df["Element"].isin(element_map.keys())].copy()
    year_cols = _available_year_cols(livestock_df)
    ratio_lookup = _build_livestock_ratio_lookup(stock_ratio_path)
    livestock_df = _split_livestock_rows(livestock_df, ratio_lookup, year_cols)

    long_df = livestock_df.melt(
        id_vars=["M49_Country_Code", "Item", "Element"],
        value_vars=year_cols,
        var_name="year",
        value_name="value",
    )
    long_df["year"] = pd.to_numeric(long_df["year"].str.lstrip("Y"), errors="coerce")
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df["Process"] = long_df["Element"].map(lambda text: element_map[str(text).strip()][0])
    long_df["GHG"] = long_df["Element"].map(lambda text: element_map[str(text).strip()][1])
    long_df["Source_Module"] = "GLE"

    valid_items = set().union(
        valid_items_by_process.get("Enteric fermentation", set()),
        valid_items_by_process.get("Manure management", set()),
        valid_items_by_process.get("Manure applied to soils", set()),
        valid_items_by_process.get("Manure left on pasture", set()),
    )
    if valid_items:
        long_df = long_df[long_df["Item"].isin(valid_items)].copy()

    return long_df[["M49_Country_Code", "Process", "Item", "GHG", "year", "value", "Source_Module"]]


def _wide_gas_to_long(
    wide_df: pd.DataFrame,
    source_module: str,
) -> pd.DataFrame:
    if wide_df.empty:
        return pd.DataFrame()

    df = wide_df.copy()
    if "process" in df.columns and "Process" not in df.columns:
        df["Process"] = df["process"]
    if "item" in df.columns and "Item" not in df.columns:
        df["Item"] = df["item"]

    gas_col_map = {
        "CH4": next((col for col in ["CH4_kt", "ch4_kt", "CH4", "ch4"] if col in df.columns), None),
        "N2O": next((col for col in ["N2O_kt", "n2o_kt", "N2O", "n2o"] if col in df.columns), None),
        "CO2": next((col for col in ["CO2_kt", "co2_kt", "CO2", "co2"] if col in df.columns), None),
    }

    base_cols = [col for col in ["M49_Country_Code", "Process", "Item", "year"] if col in df.columns]
    frames = []
    for ghg, value_col in gas_col_map.items():
        if value_col is None:
            continue
        part = df[base_cols + [value_col]].copy().rename(columns={value_col: "value"})
        part["GHG"] = ghg
        part["Source_Module"] = source_module
        frames.append(part)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _build_soil_history_detail(
    soil_path: Path,
    valid_items_by_process: Dict[str, set],
    crop_alloc_df: pd.DataFrame,
    pasture_alloc_df: pd.DataFrame,
) -> pd.DataFrame:
    soil_wide = _load_historical_drained_organic_emissions(
        str(soil_path),
        historical_years=YEARS,
        universe=None,
    )
    if soil_wide.empty:
        return pd.DataFrame()

    soil_long = _wide_gas_to_long(soil_wide, "GSOIL")
    valid_items = valid_items_by_process.get(DRAINED_ORGANIC_SOILS_PROCESS, set())
    if valid_items:
        soil_long = soil_long[soil_long["Item"].isin(valid_items)].copy()
    before_alloc = soil_long.copy()
    soil_long = _expand_drained_organic_soils_to_commodity(
        soil_long,
        crop_alloc_df=crop_alloc_df,
        pasture_alloc_df=pasture_alloc_df,
    )
    _validate_luc_group_conservation(
        before_alloc,
        soil_long,
        group_cols=["M49_Country_Code", "year", "Process", "GHG"],
    )
    return soil_long[["M49_Country_Code", "Process", "Item", "GHG", "year", "value", "Source_Module"]]


def _build_lulucf_history_detail(
    lulucf_path: Path,
    valid_items_by_process: Dict[str, set],
    crop_alloc_df: pd.DataFrame,
    pasture_alloc_df: pd.DataFrame,
) -> pd.DataFrame:
    lulucf_df = pd.read_excel(lulucf_path, sheet_name=LULUCF_SHEET)
    lulucf_df.columns = [str(col).strip() for col in lulucf_df.columns]
    lulucf_df["M49_Country_Code"] = _normalize_code_series(lulucf_df["M49_Country_Code"])
    if "Select" in lulucf_df.columns:
        lulucf_df = lulucf_df[lulucf_df["Select"] == 1].copy()

    year_cols = _available_year_cols(lulucf_df)
    lulucf_long = lulucf_df.melt(
        id_vars=["M49_Country_Code", "Land Category", "Item", "GHG"],
        value_vars=year_cols,
        var_name="year",
        value_name="value",
    )
    lulucf_long["year"] = pd.to_numeric(lulucf_long["year"].str.lstrip("Y"), errors="coerce")
    lulucf_long["value"] = pd.to_numeric(lulucf_long["value"], errors="coerce")
    # Align with the main historical LUC loader: source values are stored in Mt/yr and
    # must be converted to kt/yr before entering the common emissions summary pipeline.
    lulucf_long["value"] = lulucf_long["value"] * 1e3
    lulucf_long["Process"] = _clean_string(lulucf_long["Land Category"])
    lulucf_long["GHG"] = _clean_string(lulucf_long["GHG"]).str.upper()
    lulucf_long["Source_Module"] = "LULUCF"

    valid_processes = {
        "Wood harvest",
        "Forest",
        "De/Reforestation_crop",
        "De/Reforestation_pasture",
        "Savanna fire",
        "Peatlands fire",
    }
    lulucf_long = lulucf_long[lulucf_long["Process"].isin(valid_processes)].copy()
    before_alloc = lulucf_long.copy()
    lulucf_long = _expand_luc_country_to_commodity(
        lulucf_long,
        crop_alloc_df=crop_alloc_df,
        pasture_alloc_df=pasture_alloc_df,
        base_year=LUC_ALLOC_BASE_YEAR,
    )
    _validate_luc_group_conservation(before_alloc, lulucf_long)

    valid_mask = pd.Series(True, index=lulucf_long.index)
    split_processes = {"De/Reforestation_crop", "De/Reforestation_pasture"}
    for process in valid_processes - split_processes:
        valid_items = valid_items_by_process.get(process, set())
        if valid_items:
            process_mask = lulucf_long["Process"] == process
            valid_mask &= (~process_mask) | lulucf_long["Item"].isin(valid_items)
    lulucf_long = lulucf_long[valid_mask].copy()

    return lulucf_long[["M49_Country_Code", "Process", "Item", "GHG", "year", "value", "Source_Module"]]


def _build_fish_history_detail(panel_path: Path) -> pd.DataFrame:
    fish_df = pd.read_excel(panel_path, sheet_name=FISH_SHEET)
    fish_df.columns = [str(col).strip() for col in fish_df.columns]
    fish_df["M49_Country_Code"] = _normalize_code_series(fish_df["M49_Country_Code"])
    fish_df["Year"] = pd.to_numeric(fish_df["Year"], errors="coerce")
    fish_df = fish_df.dropna(subset=["M49_Country_Code", "Year"])
    fish_df["Year"] = fish_df["Year"].astype(int)

    ch4_col = "Aquaculture_CH4_emissions_kg_yr"
    n2o_col = "Aquaculture_N2O_emissions_kg_yr"
    required_cols = {"M49_Country_Code", "Year", ch4_col, n2o_col}
    missing = required_cols.difference(fish_df.columns)
    if missing:
        raise KeyError(f"Fish panel is missing required columns: {', '.join(sorted(missing))}")

    full_years = pd.DataFrame({"Year": YEARS})
    rebuilt_frames: List[pd.DataFrame] = []
    for country_code, group_df in fish_df.groupby("M49_Country_Code", sort=False):
        merged = full_years.merge(group_df, on="Year", how="left", sort=True)
        merged["M49_Country_Code"] = country_code
        for col in [ch4_col, n2o_col]:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").bfill().fillna(0)
        rebuilt_frames.append(merged[["M49_Country_Code", "Year", ch4_col, n2o_col]])

    rebuilt_df = pd.concat(rebuilt_frames, ignore_index=True) if rebuilt_frames else pd.DataFrame()
    if rebuilt_df.empty:
        return pd.DataFrame()

    fish_long = rebuilt_df.melt(
        id_vars=["M49_Country_Code", "Year"],
        value_vars=[ch4_col, n2o_col],
        var_name="GHG_source",
        value_name="value_kg",
    )
    fish_long["GHG"] = fish_long["GHG_source"].map({
        ch4_col: "CH4",
        n2o_col: "N2O",
    })
    fish_long["Process"] = FISH_PROCESS
    fish_long["Item"] = FISH_ITEM
    fish_long["year"] = fish_long["Year"].astype(int)
    fish_long["value"] = pd.to_numeric(fish_long["value_kg"], errors="coerce") / 1e6
    fish_long["Source_Module"] = "GFISH"
    return fish_long[["M49_Country_Code", "Process", "Item", "GHG", "year", "value", "Source_Module"]]


def _prepare_pie_source(
    summary_by_ctry_proc_item: pd.DataFrame,
    process_map: Dict[str, str],
    item_sum_map: Dict[str, str],
    region_emis_map: Dict[str, str],
) -> pd.DataFrame:
    if summary_by_ctry_proc_item.empty:
        return pd.DataFrame()

    df = summary_by_ctry_proc_item.copy()
    df["M49_norm"] = (
        df["M49_Country_Code"]
        .astype("string")
        .str.strip()
        .str.replace(r"^'+", "", regex=True)
        .str.replace(r"\.0$", "", regex=True)
    )
    df["Region_label_new"] = _clean_string(df["Region_label_new"])
    df["Process"] = _clean_string(df["Process"])
    df["Item"] = _clean_string(df["Item"])
    df["GHG"] = _clean_string(df["GHG"]).str.upper()

    is_aggregate = (
        df["Region_label_new"].str.casefold().isin(AGGREGATE_REGION_LABELS)
        | df["M49_norm"].isin(AGGREGATE_M49_CODES)
    )
    df = df[df["GHG"].eq("CO2EQ") & ~is_aggregate].copy()
    if df.empty:
        return df

    for col in [year_col for year_col in YEAR_COLS if year_col in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["Region_emisSum"] = df["M49_Country_Code"].map(region_emis_map)
    df["Region_emisSum"] = df["Region_emisSum"].fillna(df["Region_label_new"])
    df["Region_emisSum"] = _clean_string(df["Region_emisSum"])
    df["Process"] = df["Process"].map(process_map).fillna(df["Process"])
    df["Item"] = df["Item"].map(item_sum_map).fillna(df["Item"])

    return df[["Region_emisSum", "Process", "Item"] + [col for col in YEAR_COLS if col in df.columns]].copy()


def _summarize_pie_history(
    df: pd.DataFrame,
    group_cols: Iterable[str],
) -> pd.DataFrame:
    labels = {"Region_emisSum": "Country", "Process": "Process", "Item": "Item"}
    group_cols = list(group_cols)
    if df.empty:
        return pd.DataFrame(columns=[labels.get(col, col) for col in group_cols] + ["Total_1961_2020_CO2eq"] + YEAR_COLS)

    work = df.copy()
    valid_mask = pd.Series(True, index=work.index)
    for col in group_cols:
        values = _clean_string(work[col])
        valid_mask &= values.notna() & values.ne("")
    work = work[valid_mask].copy()

    year_cols = [col for col in YEAR_COLS if col in work.columns]
    grouped = work.groupby(group_cols, as_index=False, dropna=False)[year_cols].sum()
    grouped["Total_1961_2020_CO2eq"] = grouped[year_cols].sum(axis=1)
    grouped = grouped.rename(columns=labels)
    sort_cols = ["Total_1961_2020_CO2eq"] + [labels.get(col, col) for col in group_cols]
    ascending = [False] + [True] * len(group_cols)
    return grouped.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)


def _build_luc_validation_2020(
    summary_by_ctry_proc_item: pd.DataFrame,
    main_summary_path: Path,
) -> pd.DataFrame:
    split_processes = {"De/Reforestation_crop", "De/Reforestation_pasture"}
    if summary_by_ctry_proc_item.empty or not main_summary_path.exists():
        return pd.DataFrame()

    required_cols = {"M49_Country_Code", "Process", "Item", "GHG", "Y2020"}
    if not required_cols.issubset(summary_by_ctry_proc_item.columns):
        return pd.DataFrame()

    our_df = summary_by_ctry_proc_item[
        summary_by_ctry_proc_item["M49_Country_Code"].astype(str).str.strip().eq("'000")
        & summary_by_ctry_proc_item["Process"].isin(split_processes)
        & summary_by_ctry_proc_item["GHG"].astype(str).str.strip().eq("CO2eq")
    ][["Process", "Item", "Y2020"]].copy()
    our_df = our_df.rename(columns={"Y2020": "Our_Y2020"})

    main_df = pd.read_csv(
        main_summary_path,
        usecols=["M49_Country_Code", "Process", "Item", "GHG", "Y2020"],
        low_memory=False,
    )
    main_df = main_df[
        main_df["M49_Country_Code"].astype(str).str.strip().eq("'000")
        & main_df["Process"].isin(split_processes)
        & main_df["GHG"].astype(str).str.strip().eq("CO2eq")
    ][["Process", "Item", "Y2020"]].copy()
    main_df = main_df.rename(columns={"Y2020": "Main_Y2020"})

    merged = our_df.merge(main_df, on=["Process", "Item"], how="outer")
    merged["Our_Y2020"] = pd.to_numeric(merged["Our_Y2020"], errors="coerce").fillna(0.0)
    merged["Main_Y2020"] = pd.to_numeric(merged["Main_Y2020"], errors="coerce").fillna(0.0)
    merged["Diff"] = merged["Our_Y2020"] - merged["Main_Y2020"]
    merged["Abs_Diff"] = merged["Diff"].abs()
    merged["Rel_Diff_vs_Main"] = np.where(
        merged["Main_Y2020"].abs() > 0,
        merged["Diff"] / merged["Main_Y2020"],
        np.nan,
    )
    merged["Check_Level"] = "Global_Item"

    process_totals = (
        merged.groupby("Process", as_index=False)[["Our_Y2020", "Main_Y2020"]]
        .sum()
        .sort_values("Process", kind="mergesort")
        .reset_index(drop=True)
    )
    process_totals["Item"] = "__TOTAL__"
    process_totals["Diff"] = process_totals["Our_Y2020"] - process_totals["Main_Y2020"]
    process_totals["Abs_Diff"] = process_totals["Diff"].abs()
    process_totals["Rel_Diff_vs_Main"] = np.where(
        process_totals["Main_Y2020"].abs() > 0,
        process_totals["Diff"] / process_totals["Main_Y2020"],
        np.nan,
    )
    process_totals["Check_Level"] = "Global_Process_Total"

    result = pd.concat([merged, process_totals], ignore_index=True, sort=False)
    sort_cols = ["Check_Level", "Process", "Abs_Diff", "Item"]
    ascending = [True, True, False, True]
    return result.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)


def _excel_sheet_name(base_name: str, chunk_index: Optional[int] = None) -> str:
    base = str(base_name).strip() or "Sheet"
    if chunk_index is None:
        return base[:31]
    suffix = f"_{int(chunk_index)}"
    return f"{base[:31 - len(suffix)]}{suffix}"


def _write_excel_df(writer: pd.ExcelWriter, df: pd.DataFrame, sheet_name: str) -> List[str]:
    if len(df) <= EXCEL_MAX_DATA_ROWS:
        name = _excel_sheet_name(sheet_name)
        df.to_excel(writer, sheet_name=name, index=False)
        return [name]

    written_names: List[str] = []
    for chunk_index, start in enumerate(range(0, len(df), EXCEL_MAX_DATA_ROWS), start=1):
        name = _excel_sheet_name(sheet_name, chunk_index)
        chunk = df.iloc[start : start + EXCEL_MAX_DATA_ROWS]
        chunk.to_excel(writer, sheet_name=name, index=False)
        written_names.append(name)
    return written_names


def build_history_emission_summary() -> Path:
    input_base = Path(get_input_base())
    src_base = Path(get_src_base())
    retired_base = input_base / "Emission" / "retired-unused-raw"

    dict_v3_path = src_base / "dict_v3.xlsx"
    output_path = input_base / "Emission" / OUTPUT_NAME
    main_summary_path = input_base.parent / "output" / "BASE" / "Emis" / "emissions_summary_By_Country_Process_Item.csv"

    crop_path = retired_base / "Emissions_crops_E_All_Data_NOFLAG.csv"
    livestock_path = retired_base / "Emissions_livestock_E_All_Data_NOFLAG.csv"
    soil_path = retired_base / "Emissions_Drained_Organic_Soils_E_All_Data_NOFLAG.csv"
    lulucf_path = retired_base / "Emission_LULUCF_Historical_updated_Select1_Y1961_Y1999_backfilled.xlsx"
    fish_path = retired_base / "fish_seafood_country_panel_2000_present.xlsx"
    fertilizer_path = retired_base / "Fertilizer_efficiency.xlsx"
    stock_ratio_path = input_base / "Manure_Stock" / "Environment_LivestockManure_with_ratio.csv"
    current_production_path = (
        input_base / "Production_Trade" / "Production_Crops_Livestock_E_All_Data_NOFLAG_yield_refilled_baseYearFilled.csv"
    )
    retired_production_path = (
        input_base / "Production_Trade" / "retired-unused-raw" / "Production_Crops_Livestock_E_All_Data_NOFLAG.csv"
    )

    region_label_map, region_emis_map = _load_region_maps(dict_v3_path)
    emis_item_df, valid_items_by_process, process_map, item_sum_map, fertilizer_item_map = _load_emis_item_maps(dict_v3_path)
    crop_alloc_df, pasture_alloc_df, luc_alloc_diag = _build_luc_alloc_inputs(
        dict_v3_path,
        retired_production_path,
        current_production_path,
        stock_ratio_path,
        emis_item_df,
    )

    detail_frames = [
        _build_crop_history_detail(crop_path, fertilizer_path, valid_items_by_process, fertilizer_item_map),
        _build_livestock_history_detail(livestock_path, stock_ratio_path, valid_items_by_process),
        _build_soil_history_detail(
            soil_path,
            valid_items_by_process,
            crop_alloc_df,
            pasture_alloc_df,
        ),
        _build_lulucf_history_detail(
            lulucf_path,
            valid_items_by_process,
            crop_alloc_df,
            pasture_alloc_df,
        ),
        _build_fish_history_detail(fish_path),
    ]
    detail_long = _finalize_detail(pd.concat(detail_frames, ignore_index=True), region_label_map)

    summaries = summarize_emissions_from_detail(
        detail_long,
        allowed_years=YEARS,
        dict_v3_path=str(dict_v3_path),
    )

    pie_source = _prepare_pie_source(
        summaries["by_ctry_proc_comm"],
        process_map,
        item_sum_map,
        region_emis_map,
    )
    pie_sheets = {
        "Pie_Country": _summarize_pie_history(pie_source, ["Region_emisSum"]),
        "Pie_Process": _summarize_pie_history(pie_source, ["Process"]),
        "Pie_Item": _summarize_pie_history(pie_source, ["Item"]),
        "Pie_Country_Item": _summarize_pie_history(pie_source, ["Region_emisSum", "Item"]),
        "Pie_Process_Item": _summarize_pie_history(pie_source, ["Process", "Item"]),
        "Pie_Process_Country": _summarize_pie_history(pie_source, ["Process", "Region_emisSum"]),
    }

    source_summary = (
        detail_long.groupby(["Source_Module", "Process", "GHG"], as_index=False)
        .agg(rows=("value", "size"), countries=("M49_Country_Code", "nunique"), total_value=("value", "sum"))
        .sort_values(["Source_Module", "Process", "GHG"], kind="mergesort")
        .reset_index(drop=True)
    )
    validation_2020 = _build_luc_validation_2020(
        summaries["by_ctry_proc_comm"],
        main_summary_path,
    )

    meta = pd.DataFrame(
        [
            {"key": "script", "value": str(Path(__file__).name)},
            {"key": "year_range", "value": f"{START_YEAR}-{END_YEAR}"},
            {"key": "output_path", "value": str(output_path)},
            {"key": "crop_source", "value": str(crop_path)},
            {"key": "livestock_source", "value": str(livestock_path)},
            {"key": "soil_source", "value": str(soil_path)},
            {"key": "lulucf_source", "value": str(lulucf_path)},
            {"key": "fish_source", "value": str(fish_path)},
            {"key": "fertilizer_source", "value": str(fertilizer_path)},
            {"key": "livestock_ratio_source", "value": str(stock_ratio_path)},
            {"key": "luc_production_source_retired", "value": str(retired_production_path)},
            {"key": "luc_production_source_current", "value": str(current_production_path)},
            {"key": "luc_main_summary_validation_source", "value": str(main_summary_path)},
            {
                "key": "livestock_pre2000_ratio_rule",
                "value": "Use Y2000 dairy/non-dairy stock ratios to backfill Y1961-Y1999.",
            },
            {
                "key": "luc_stock_pre2000_split_rule",
                "value": "Split pre-2000 raw livestock stocks with country Y2000 ratios; fall back to year global mean when country ratios are missing.",
            },
            {
                "key": "fish_leading_gap_rule",
                "value": "Backfill missing leading historical emissions with the next available year value (bfill).",
            },
            {
                "key": "luc_alloc_rule",
                "value": "Allocate De/Reforestation_crop and De/Reforestation_pasture to commodity items using historical crop area + feed crop area and grass-area shares, matching the main-flow area-based split for years <= 2020.",
            },
            {
                "key": "soil_alloc_rule",
                "value": "Allocate Cropland organic soils to crop commodity items and Grassland organic soils to livestock commodity items using the same historical crop and pasture area shares as LUC; fall back to nearest available country-year or year-level shares when an exact country-year share is unavailable.",
            },
            {
                "key": "note",
                "value": "Workbook output uses .xlsx because the requested result contains multiple sheets.",
            },
            {"key": "excel_max_data_rows_per_sheet", "value": EXCEL_MAX_DATA_ROWS},
        ]
    )
    for diag_key, diag_value in luc_alloc_diag.items():
        meta.loc[len(meta)] = {"key": f"luc_{diag_key}", "value": diag_value}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        _write_excel_df(writer, detail_long, "Detail_Long")
        _write_excel_df(writer, summaries["by_ctry_proc_comm"], "By_Country_Process_Item")
        _write_excel_df(writer, summaries["by_ctry_proc"], "By_Country_Process")
        _write_excel_df(writer, summaries["by_ctry"], "By_Country")
        for sheet_name, sheet_df in pie_sheets.items():
            _write_excel_df(writer, sheet_df, sheet_name)
        if not validation_2020.empty:
            _write_excel_df(writer, validation_2020, "Validation_2020_DeRef")
        _write_excel_df(writer, source_summary, "Source_Process_Summary")
        _write_excel_df(writer, meta, "meta")

    return output_path


def main() -> None:
    output_path = build_history_emission_summary()
    print(f"[DONE] {output_path}")


if __name__ == "__main__":
    main()
