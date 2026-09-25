from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import SP_M1a_Figure_pie_structure_pre as pie_base
from config_paths import get_results_base, get_src_base
from S2_0_load_data import load_nutrient_factors_from_dict_v3


DEFAULT_SCENARIO = "BASE"
TARGET_YEAR = 2080
OUTPUT_NAME = "Y2080_Emission_result_summary.xlsx"
EMISSION_INPUT_CANDIDATES = (
    "Emis/emissions_summary_By_Country_Process_Item.csv",
    "Emis/emissions_summary_By_Country_Process_Item.xlsx",
)
LUC_CROP_PROCESSES = {
    "De/Reforestation_crop",
    "Ag land abandonment_crop",
    "Grassland conversion_crop",
}
LUC_PASTURE_PROCESSES = {"De/Reforestation_pasture", "Ag land abandonment_pasture"}
LUC_TO_DEFORESTATION_PROCESSES = LUC_CROP_PROCESSES | LUC_PASTURE_PROCESSES
FOREST_PROCESS = "Forest"
DEFORESTATION_PROCESS = "De/Reforestation"
ITEM_PLACEHOLDER_EXCLUDE = {
    "",
    "no",
    "nan",
    "none",
}
ITEM_EXCLUDE_FOR_INTENSITY = {
    *ITEM_PLACEHOLDER_EXCLUDE,
    "Fires",
    "Roundwood",
    "Forest",
    "Forestland",
    "Existing forestland",
    "Fore set",
}
GROUP_LABELS_2080 = {
    **pie_base.GROUP_LABELS,
    "Region_emisSum": "Region",
}
VALUE_COL = f"Y{TARGET_YEAR}_CO2eq"


def _scenario_dir(scenario: str, scenario_dir: Optional[str]) -> Path:
    if scenario_dir:
        return Path(scenario_dir).expanduser().resolve()
    return Path(get_results_base(str(scenario))).resolve()


def _resolve_emission_input(scenario_path: Path) -> Path:
    for rel in EMISSION_INPUT_CANDIDATES:
        path = scenario_path / rel
        if path.exists():
            return path
    tried = ", ".join(str(scenario_path / rel) for rel in EMISSION_INPUT_CANDIDATES)
    raise FileNotFoundError(f"Missing scenario emission input. Tried: {tried}")


def _read_table(path: Path, **kwargs) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, **kwargs)
    return pd.read_excel(path, **kwargs)


def _norm_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _year_col(df: pd.DataFrame, year: int) -> str:
    return pie_base._pick_year_col(df, year)


def _clean_code(series: pd.Series) -> pd.Series:
    return pie_base._normalize_code(series)


def _load_production_area(scenario_path: Path, year: int) -> pd.DataFrame:
    path = scenario_path / "DS" / "production_summary.csv"
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "M49_Country_Code",
                "year",
                "commodity",
                "crop_area_ha",
                "pasture_area_ha",
            ]
        )
    cols = ["M49_Country_Code", "year", "commodity", "crop_area_ha", "pasture_area_ha"]
    df = _read_table(path, usecols=lambda c: str(c) in set(cols))
    missing = set(cols).difference(df.columns)
    for col in missing:
        df[col] = np.nan
    work = df[cols].copy()
    work["M49_Country_Code"] = _clean_code(work["M49_Country_Code"])
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work["commodity"] = work["commodity"].astype("string").str.strip()
    for col in ["crop_area_ha", "pasture_area_ha"]:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0).clip(lower=0.0)
    work = work.dropna(subset=["M49_Country_Code", "year", "commodity"])
    work = work.loc[work["commodity"].ne("") & work["year"].isin([2020, int(year)])].copy()
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


def _build_area_lookup(area_df: pd.DataFrame, year: int, area_col: str) -> Dict[str, pd.DataFrame]:
    if area_df.empty or area_col not in area_df.columns:
        return {}
    base = (
        area_df.loc[area_df["year"].eq(2020), ["M49_Country_Code", "commodity", area_col]]
        .rename(columns={area_col: "base_area_ha"})
        .groupby(["M49_Country_Code", "commodity"], as_index=False)["base_area_ha"]
        .sum()
    )
    future = (
        area_df.loc[area_df["year"].eq(int(year)), ["M49_Country_Code", "commodity", area_col]]
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


def _weights_for_luc(area_group: pd.DataFrame, value: float) -> pd.Series:
    if value > 0:
        weights = pd.to_numeric(area_group["delta_area_ha"], errors="coerce").clip(lower=0.0)
        if float(weights.sum()) > 0:
            return weights
    elif value < 0:
        weights = -pd.to_numeric(area_group["delta_area_ha"], errors="coerce").clip(upper=0.0)
        if float(weights.sum()) > 0:
            return weights
    return pd.to_numeric(area_group["future_area_ha"], errors="coerce").fillna(0.0).clip(lower=0.0)


def _allocate_luc_to_items(df: pd.DataFrame, scenario_path: Path, year_col: str) -> pd.DataFrame:
    area_df = _load_production_area(scenario_path, TARGET_YEAR)
    crop_lookup = _build_area_lookup(area_df, TARGET_YEAR, "crop_area_ha")
    pasture_lookup = _build_area_lookup(area_df, TARGET_YEAR, "pasture_area_ha")
    if not crop_lookup and not pasture_lookup:
        return df.copy()

    expanded_rows: List[dict] = []
    for row in df.to_dict(orient="records"):
        process = _norm_text(row.get("Process"))
        if process not in LUC_TO_DEFORESTATION_PROCESSES:
            expanded_rows.append(row)
            continue

        m49 = _clean_code(pd.Series([row.get("M49_Country_Code")])).iloc[0]
        value = pd.to_numeric(row.get(year_col), errors="coerce")
        if pd.isna(value) or not m49:
            expanded_rows.append(row)
            continue

        lookup = crop_lookup if process in LUC_CROP_PROCESSES else pasture_lookup
        area_group = lookup.get(str(m49))
        if area_group is None or area_group.empty:
            expanded_rows.append(row)
            continue

        weights = _weights_for_luc(area_group, float(value))
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            expanded_rows.append(row)
            continue

        allocated = 0.0
        used = False
        for commodity, weight in zip(area_group["commodity"], weights):
            weight_f = float(weight)
            if not np.isfinite(weight_f) or weight_f <= 0:
                continue
            commodity_s = _norm_text(commodity)
            if not commodity_s:
                continue
            new_row = dict(row)
            new_row["Item"] = commodity_s
            new_value = float(value) * weight_f / weight_sum
            new_row[year_col] = new_value
            expanded_rows.append(new_row)
            allocated += new_value
            used = True

        if used:
            diff = float(value) - allocated
            if abs(diff) > 1e-9:
                expanded_rows[-1][year_col] = float(expanded_rows[-1][year_col]) + diff
        else:
            expanded_rows.append(row)

    return pd.DataFrame(expanded_rows, columns=df.columns)


def _prepare_emission_source(scenario_path: Path) -> Tuple[pd.DataFrame, str]:
    source_path = _resolve_emission_input(scenario_path)
    df = _read_table(source_path)
    process_map, item_map = pie_base._load_emis_item_maps()
    region_map = pie_base._load_region_emis_sum_map()

    required = {"M49_Country_Code", "Region_label_new", "Process", "Item", "GHG"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Emission input missing required columns: {', '.join(sorted(missing))}")

    year_col = _year_col(df, TARGET_YEAR)
    ghg = df["GHG"].astype("string").str.strip().str.casefold()
    region_label = pie_base._clean_group_col(df, "Region_label_new")
    m49 = _clean_code(df["M49_Country_Code"])
    is_aggregate = region_label.str.casefold().isin(pie_base.AGGREGATE_REGION_LABELS) | m49.isin(
        pie_base.AGGREGATE_M49_CODES
    )

    prepared = df.loc[ghg.eq("co2eq") & ~is_aggregate].copy()
    prepared[year_col] = pd.to_numeric(prepared[year_col], errors="coerce")
    prepared = prepared.loc[prepared[year_col].notna()].copy()
    prepared["M49_Country_Code"] = m49.loc[prepared.index]
    prepared["Region_label_new"] = region_label.loc[prepared.index]
    prepared["Region_emisSum"] = prepared["M49_Country_Code"].map(region_map)
    prepared["Region_emisSum"] = prepared["Region_emisSum"].fillna(prepared["Region_label_new"])
    prepared["Region_emisSum"] = pie_base._clean_group_col(prepared, "Region_emisSum")
    prepared["Process_raw"] = pie_base._clean_group_col(prepared, "Process")
    prepared["Item"] = pie_base._clean_group_col(prepared, "Item")

    prepared = _allocate_luc_to_items(prepared, scenario_path, year_col)
    prepared["Process"] = prepared["Process_raw"].map(process_map).fillna(prepared["Process_raw"])
    prepared.loc[prepared["Process_raw"].isin(LUC_TO_DEFORESTATION_PROCESSES), "Process"] = (
        DEFORESTATION_PROCESS
    )
    prepared.loc[prepared["Process_raw"].eq(FOREST_PROCESS), "Process"] = FOREST_PROCESS
    prepared = pie_base._expand_organic_soil_aggregate_items(prepared, year_col, item_map)
    prepared["Item"] = pie_base._map_item_to_summary(prepared["Item"], item_map)
    return prepared, year_col


def _summarize_2080(df: pd.DataFrame, group_cols: Iterable[str], year_col: str) -> pd.DataFrame:
    group_cols = list(group_cols)
    valid_mask = pd.Series(True, index=df.index)
    for col in group_cols:
        values = pie_base._clean_group_col(df, col)
        valid_mask &= values.notna() & values.ne("")

    grouped = (
        df.loc[valid_mask, group_cols + [year_col]]
        .groupby(group_cols, as_index=False, dropna=False)[year_col]
        .sum()
        .rename(columns={year_col: VALUE_COL, **{col: GROUP_LABELS_2080.get(col, col) for col in group_cols}})
    )
    grouped = grouped.loc[pd.to_numeric(grouped[VALUE_COL], errors="coerce").fillna(0.0).ne(0.0)].copy()
    sort_cols = [VALUE_COL] + [GROUP_LABELS_2080.get(col, col) for col in group_cols]
    ascending = [False] + [True] * len(group_cols)
    return grouped.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)


def _item_summary_input(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    item = work["Item"].astype("string").str.strip()
    exclude_item = item.isin(ITEM_PLACEHOLDER_EXCLUDE) | item.str.casefold().isin(
        {x.casefold() for x in ITEM_PLACEHOLDER_EXCLUDE}
    )
    return work.loc[~exclude_item].copy()


def _filter_item_intensity_scope(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    item = work["Item"].astype("string").str.strip()
    exclude_item = item.isin(ITEM_EXCLUDE_FOR_INTENSITY) | item.str.casefold().isin(
        {x.casefold() for x in ITEM_EXCLUDE_FOR_INTENSITY}
    )
    return work.loc[~exclude_item].copy()


def _load_food_kcal_by_item(scenario_path: Path, item_map: Dict[str, str]) -> pd.DataFrame:
    candidates = [
        scenario_path / "DS" / "market_summary.csv",
        scenario_path / "Diagnostics" / "commodity_balance_by_commodity.csv",
    ]
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        return pd.DataFrame(columns=["Item", "food_t", "kcal_per_ton_weighted", "kcal"])

    df = _read_table(source)
    required = {"year", "commodity", "food_t"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=["Item", "food_t", "kcal_per_ton_weighted", "kcal"])

    work = df[["year", "commodity", "food_t"]].copy()
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work = work.loc[work["year"].eq(TARGET_YEAR)].copy()
    work["commodity"] = work["commodity"].astype("string").str.strip()
    work["food_t"] = pd.to_numeric(work["food_t"], errors="coerce").fillna(0.0).clip(lower=0.0)
    work = work.loc[work["commodity"].notna() & work["commodity"].ne("") & work["food_t"].gt(0)].copy()
    if work.empty:
        return pd.DataFrame(columns=["Item", "food_t", "kcal_per_ton_weighted", "kcal"])

    kcal_map = load_nutrient_factors_from_dict_v3(str(Path(get_src_base()) / "dict_v3.xlsx"), "energy")
    work["kcal_per_ton"] = work["commodity"].map(kcal_map).fillna(0.0)
    work["kcal"] = work["food_t"] * work["kcal_per_ton"]
    work["Item"] = pie_base._map_item_to_summary(work["commodity"], item_map)
    work = _filter_item_intensity_scope(work)
    grouped = work.groupby("Item", as_index=False).agg(food_t=("food_t", "sum"), kcal=("kcal", "sum"))
    grouped["kcal_per_ton_weighted"] = np.where(
        grouped["food_t"] > 0,
        grouped["kcal"] / grouped["food_t"],
        np.nan,
    )
    return grouped[["Item", "food_t", "kcal_per_ton_weighted", "kcal"]]


def _load_area_by_item(scenario_path: Path, item_map: Dict[str, str]) -> pd.DataFrame:
    path = scenario_path / "DS" / "production_summary.csv"
    if not path.exists():
        return pd.DataFrame(columns=["Item", "crop_area_ha", "pasture_area_ha", "ag_area_ha"])
    cols = ["year", "commodity", "crop_area_ha", "pasture_area_ha"]
    df = _read_table(path, usecols=lambda c: str(c) in set(cols))
    missing = set(cols).difference(df.columns)
    for col in missing:
        df[col] = np.nan
    work = df[cols].copy()
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work = work.loc[work["year"].eq(TARGET_YEAR)].copy()
    work["commodity"] = work["commodity"].astype("string").str.strip()
    for col in ["crop_area_ha", "pasture_area_ha"]:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0).clip(lower=0.0)
    work["Item"] = pie_base._map_item_to_summary(work["commodity"], item_map)
    work = _filter_item_intensity_scope(work)
    grouped = work.groupby("Item", as_index=False)[["crop_area_ha", "pasture_area_ha"]].sum()
    grouped["ag_area_ha"] = grouped["crop_area_ha"] + grouped["pasture_area_ha"]
    return grouped[["Item", "crop_area_ha", "pasture_area_ha", "ag_area_ha"]]


def _build_item_intensity_sheets(
    item_emissions: pd.DataFrame,
    scenario_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _, item_map = pie_base._load_emis_item_maps()
    emis = _filter_item_intensity_scope(item_emissions)[["Item", VALUE_COL]].copy()
    kcal = _load_food_kcal_by_item(scenario_path, item_map)
    area = _load_area_by_item(scenario_path, item_map)

    item_kcal = emis.merge(kcal, on="Item", how="left")
    item_kcal["Emission/kcal"] = np.where(
        pd.to_numeric(item_kcal["kcal"], errors="coerce").fillna(0.0) > 0,
        item_kcal[VALUE_COL] / item_kcal["kcal"],
        np.nan,
    )
    item_kcal = item_kcal.sort_values(VALUE_COL, ascending=False, kind="mergesort").reset_index(drop=True)

    item_area = emis.merge(area, on="Item", how="left")
    item_area["Emission/ha"] = np.where(
        pd.to_numeric(item_area["ag_area_ha"], errors="coerce").fillna(0.0) > 0,
        item_area[VALUE_COL] / item_area["ag_area_ha"],
        np.nan,
    )
    item_area = item_area.sort_values(VALUE_COL, ascending=False, kind="mergesort").reset_index(drop=True)
    return item_kcal, item_area


def build_emission_result_summary(
    scenario: str = DEFAULT_SCENARIO,
    scenario_dir: Optional[str] = None,
    output_name: str = OUTPUT_NAME,
) -> Path:
    scenario_path = _scenario_dir(scenario, scenario_dir)
    df, year_col = _prepare_emission_source(scenario_path)
    item_df = _item_summary_input(df)

    sheets: Dict[str, pd.DataFrame] = {
        "Country": _summarize_2080(df, ["Region_emisSum"], year_col),
        "Process": _summarize_2080(df, ["Process"], year_col),
        "Item": _summarize_2080(item_df, ["Item"], year_col),
        "Country-Item": _summarize_2080(item_df, ["Region_emisSum", "Item"], year_col),
        "Process-Item": _summarize_2080(item_df, ["Process", "Item"], year_col),
        "Process-Country": _summarize_2080(df, ["Process", "Region_emisSum"], year_col),
    }
    sheets["Item-kcal"], sheets["Item-area"] = _build_item_intensity_sheets(
        sheets["Item"],
        scenario_path,
    )

    output_path = scenario_path / output_name
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, sheet_df in sheets.items():
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize 2080 scenario emissions by region/process/item and item intensities."
    )
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO, help="Scenario folder under output/.")
    parser.add_argument("--scenario-dir", default=None, help="Explicit scenario result directory.")
    parser.add_argument("--output-name", default=OUTPUT_NAME, help="Workbook name written under scenario dir.")
    args = parser.parse_args()
    out = build_emission_result_summary(
        scenario=args.scenario,
        scenario_dir=args.scenario_dir,
        output_name=args.output_name,
    )
    print(f"[DONE] {out}")


if __name__ == "__main__":
    main()
