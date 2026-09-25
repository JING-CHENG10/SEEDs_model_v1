import os
import pandas as pd
import numpy as np

from S1_0_schema import ScenarioConfig
from S2_0_load_data import (
    DataPaths,
    build_universe_from_dict_v3,
    build_demand_total_from_fbs_domestic_supply,
    load_population_wpp,
    _norm_m49,
    _read_fbs_table,
    _faostat_wide_to_long,
    _filter_select_rows,
    _find_col,
    _maybe_find_col,
    _load_demand_item_map,
    _attach_country_from_m49,
    _lc,
)


ELEMENT_META = {
    "energy": {
        "element": "Food supply (kcal/capita/day)",
        "unit": "kcal/cap/d",
        "element_code": 664,
        "value_col": "kcal_per_100g",
    },
    "protein": {
        "element": "Protein supply quantity (g/capita/day)",
        "unit": "g/cap/d",
        "element_code": 674,
        "value_col": "g_protein_per_100g",
    },
    "fat": {
        "element": "Fat supply quantity (g/capita/day)",
        "unit": "g/cap/d",
        "element_code": 684,
        "value_col": "g_fat_per_100g",
    },
}


def _load_item_nutrition_map(dict_v3_path: str) -> tuple[dict, dict]:
    df = pd.read_excel(dict_v3_path, sheet_name="Emis_item")
    df.columns = [str(c).strip() for c in df.columns]
    required = {"Item_Emis", "Item_Nutrition_Map"} | {
        meta["value_col"] for meta in ELEMENT_META.values()
    }
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"dict_v3 Emis_item missing columns: {missing}")

    comm_to_item: dict[str, str] = {}
    item_to_per_ton: dict[str, dict[str, float]] = {}

    for _, row in df.iterrows():
        comm = str(row["Item_Emis"]).strip()
        item = str(row["Item_Nutrition_Map"]).strip()
        if not comm or not item:
            continue
        if item.lower() in {"no", "nan", "non food"}:
            continue
        prev = comm_to_item.get(comm)
        if prev and prev != item:
            raise ValueError(f"Item_Nutrition_Map conflict for {comm}: {prev} vs {item}")
        comm_to_item[comm] = item
        per_ton = item_to_per_ton.setdefault(item, {})
        for key, meta in ELEMENT_META.items():
            val = pd.to_numeric(row.get(meta["value_col"]), errors="coerce")
            if pd.isna(val) or val <= 0:
                per_ton[key] = 0.0
            else:
                per_ton[key] = float(val) * 10000.0
    return comm_to_item, item_to_per_ton


def _load_area_map(dict_v3_path: str) -> dict:
    df = pd.read_excel(dict_v3_path, sheet_name="region")
    df.columns = [str(c).strip() for c in df.columns]
    if "M49_Country_Code" not in df.columns or "NAME" not in df.columns:
        return {}
    df["m49"] = df["M49_Country_Code"].apply(_norm_m49)
    return {r.m49: str(r.NAME).strip() for r in df.itertuples(index=False) if r.m49}


def _build_food_from_fbs(fbs_xlsx: str, dict_v3_path: str, universe) -> pd.DataFrame:
    if not fbs_xlsx or not os.path.exists(fbs_xlsx):
        return pd.DataFrame(columns=["country", "year", "commodity", "food_t"])
    item_map = _load_demand_item_map(dict_v3_path)
    if not item_map:
        return pd.DataFrame(columns=["country", "year", "commodity", "food_t"])
    df_raw = _read_fbs_table(fbs_xlsx)
    df_raw = _filter_select_rows(df_raw)
    df = _lc(_faostat_wide_to_long(df_raw))
    c_area = _find_col(df, ["Area"])
    c_year = _find_col(df, ["Year"])
    c_item = _find_col(df, ["Item"])
    c_elem = _find_col(df, ["Element"])
    c_val = _find_col(df, ["Value"])
    c_unit = _maybe_find_col(df, ["Unit"])
    keep_cols = [c_area, c_year, c_item, c_elem, c_val]
    if c_unit:
        keep_cols.append(c_unit)
    z = df[keep_cols].copy()
    rename_map = {c_area: "area", c_year: "year", c_item: "item_raw", c_elem: "element", c_val: "value"}
    if c_unit:
        rename_map[c_unit] = "unit"
    z = z.rename(columns=rename_map)
    if "M49_Country_Code" in df.columns:
        z["M49_Country_Code"] = df["M49_Country_Code"]
    z = _attach_country_from_m49(df, z, universe, context=f"FBS food ({fbs_xlsx})")
    z["item_raw"] = z["item_raw"].astype(str).str.strip()
    z["commodity"] = z["item_raw"].map(item_map)
    z = z.dropna(subset=["commodity"])
    z = z[z["element"].astype(str).str.strip().str.lower() == "food"]
    if z.empty:
        return pd.DataFrame(columns=["country", "year", "commodity", "food_t"])
    z["value"] = pd.to_numeric(z["value"], errors="coerce").fillna(0.0)
    if "unit" in z.columns:
        factors = z["unit"].astype(str).str.contains("1000", case=False, na=False).replace(
            {True: 1000.0, False: 1.0}
        )
        z["value"] = z["value"] * factors
    z["year"] = pd.to_numeric(z["year"], errors="coerce").astype(int)
    result = z.groupby(["country", "year", "commodity"], as_index=False)["value"].sum()
    result = result.rename(columns={"value": "food_t"})
    return result[["country", "year", "commodity", "food_t"]]


def run() -> None:
    cfg = ScenarioConfig(years_hist_start=2010, years_hist_end=2020, years_future=[])
    paths = DataPaths()
    universe = build_universe_from_dict_v3(paths.dict_v3_path, cfg)

    years = list(range(cfg.years_hist_start, cfg.years_hist_end + 1))

    demand_df = build_demand_total_from_fbs_domestic_supply(
        paths.fbs_csv, paths.dict_v3_path, universe
    )
    if demand_df is None or demand_df.empty:
        raise ValueError("No demand data returned from FBS DSQ.")
    food_df = _build_food_from_fbs(paths.fbs_csv, paths.dict_v3_path, universe)
    food_key = {
        (r.country, int(r.year), r.commodity): float(r.food_t)
        for r in food_df.itertuples(index=False)
    } if isinstance(food_df, pd.DataFrame) and not food_df.empty else {}
    demand_df = demand_df[demand_df["year"].isin(years)].copy()
    demand_df["country"] = demand_df["country"].apply(_norm_m49)
    demand_df["demand_total_t"] = pd.to_numeric(demand_df["demand_total_t"], errors="coerce")
    neg_mask = demand_df["demand_total_t"] < 0
    if neg_mask.any():
        def _replace_neg(row: pd.Series) -> float:
            val = food_key.get((row["country"], int(row["year"]), row["commodity"]))
            if val is None or not np.isfinite(val):
                return 0.0
            return float(val) if val > 0 else 0.0
        demand_df.loc[neg_mask, "demand_total_t"] = demand_df.loc[neg_mask].apply(
            _replace_neg, axis=1
        )
    demand_df["demand_total_t"] = (
        demand_df["demand_total_t"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    comm_to_item, item_to_per_ton = _load_item_nutrition_map(paths.dict_v3_path)
    if not comm_to_item:
        raise ValueError("Empty Item_Nutrition_Map from dict_v3.")

    demand_df["Item"] = demand_df["commodity"].map(comm_to_item)
    missing_map = demand_df["Item"].isna().sum()
    if missing_map:
        missing_items = sorted(set(demand_df.loc[demand_df["Item"].isna(), "commodity"]))
        print(f"[WARN] Missing Item_Nutrition_Map for {missing_map} rows. Example: {missing_items[:10]}")
    demand_df = demand_df.dropna(subset=["Item"])

    base = (
        demand_df.groupby(["country", "Item", "year"], as_index=False)["demand_total_t"]
        .sum()
        .rename(columns={"country": "M49_Country_Code", "demand_total_t": "demand_t"})
    )

    pop_map = load_population_wpp(paths.population_wpp_csv, universe)
    pop_rows = [
        {"M49_Country_Code": k[0], "year": k[1], "population": v}
        for k, v in pop_map.items()
    ]
    pop_df = pd.DataFrame(pop_rows)
    if pop_df.empty:
        raise ValueError("Population map is empty.")
    pop_df = pop_df[pop_df["year"].isin(years)].copy()

    pop_pivot = pop_df.pivot_table(
        index="M49_Country_Code", columns="year", values="population", aggfunc="last"
    )
    for y in years:
        if y not in pop_pivot.columns:
            pop_pivot[y] = np.nan
    pop_pivot = pop_pivot[sorted(years)]
    pop_pivot = pop_pivot.ffill(axis=1).bfill(axis=1)
    missing_countries = pop_pivot.index[pop_pivot.isna().all(axis=1)].tolist()
    if missing_countries:
        sample = missing_countries[:10]
        raise ValueError(
            f"Missing population for {len(missing_countries)} countries (all years empty). "
            f"Sample: {sample}"
        )
    pop_df = pop_pivot.reset_index().melt(
        id_vars=["M49_Country_Code"], var_name="year", value_name="population"
    )
    pop_df["year"] = pop_df["year"].astype(int)

    all_countries = sorted(set(pop_df["M49_Country_Code"]))
    all_items = sorted(set(base["Item"]))
    grid = pd.MultiIndex.from_product(
        [all_countries, all_items, years],
        names=["M49_Country_Code", "Item", "year"],
    ).to_frame(index=False)
    grid = grid.merge(base, on=["M49_Country_Code", "Item", "year"], how="left")
    grid["demand_t"] = pd.to_numeric(grid["demand_t"], errors="coerce").fillna(0.0)

    grid = grid.merge(pop_df, on=["M49_Country_Code", "year"], how="left")
    missing_pop = grid["population"].isna().sum()
    if missing_pop:
        miss_sample = grid.loc[grid["population"].isna(), "M49_Country_Code"].head(10).tolist()
        raise ValueError(
            f"Missing population for {missing_pop} rows after fill. Sample: {miss_sample}"
        )

    area_map = _load_area_map(paths.dict_v3_path)
    grid["Area"] = grid["M49_Country_Code"].map(area_map).fillna(grid["M49_Country_Code"])

    out_rows = []
    for key, meta in ELEMENT_META.items():
        per_ton_map = {item: vals.get(key, 0.0) for item, vals in item_to_per_ton.items()}
        work = grid.copy()
        work["nutrient_per_ton"] = work["Item"].map(per_ton_map).fillna(0.0)
        work["value"] = (
            work["demand_t"] * work["nutrient_per_ton"]
        ) / (work["population"] * 365.0)
        work["value"] = work["value"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        work["Element"] = meta["element"]
        work["Unit"] = meta["unit"]
        work["Element Code"] = meta["element_code"]
        work["nutrient_key"] = key
        out_rows.append(
            work[
                [
                    "M49_Country_Code",
                    "Area",
                    "Item",
                    "Element Code",
                    "Element",
                    "Unit",
                    "year",
                    "value",
                    "nutrient_key",
                ]
            ]
        )

    out_long = pd.concat(out_rows, ignore_index=True)
    out_long["Area Code (M49)"] = out_long["M49_Country_Code"]

    wide = out_long.pivot_table(
        index=[
            "Area Code (M49)",
            "M49_Country_Code",
            "Area",
            "Item",
            "Element Code",
            "Element",
            "Unit",
            "nutrient_key",
        ],
        columns="year",
        values="value",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()

    rename_year = {}
    for col in wide.columns:
        if isinstance(col, (int, np.integer)):
            rename_year[col] = f"Y{int(col)}"
        elif isinstance(col, str) and col.isdigit():
            rename_year[col] = f"Y{int(col)}"
    if rename_year:
        wide = wide.rename(columns=rename_year)

    year_cols = [f"Y{y}" for y in years]
    for y in years:
        col = f"Y{y}"
        if col not in wide.columns:
            wide[col] = 0.0
    wide = wide[
        [
            "Area Code (M49)",
            "M49_Country_Code",
            "Area",
            "Item",
            "Element Code",
            "Element",
            "Unit",
        ]
        + year_cols
        + ["nutrient_key"]
    ]

    output_dir = os.path.join(paths.base, "Driver", "Nutrition")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "Nutrition_profile_recalculated_fromD0.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        wide.to_excel(writer, sheet_name="nutrition_profile", index=False)

    print(f"Wrote {len(wide)} rows to {out_path}")


if __name__ == "__main__":
    run()
