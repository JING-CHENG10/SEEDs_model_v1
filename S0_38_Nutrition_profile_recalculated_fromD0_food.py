import os
import pandas as pd
import numpy as np

from S1_0_schema import ScenarioConfig
from S2_0_load_data import (
    DataPaths,
    build_universe_from_dict_v3,
    load_population_wpp,
    _norm_m49,
)
from S0_38_Nutrition_profile_recalculated_fromD0 import (
    ELEMENT_META,
    _load_item_nutrition_map,
    _load_area_map,
    _build_food_from_fbs,
)


def run() -> None:
    cfg = ScenarioConfig(years_hist_start=2010, years_hist_end=2020, years_future=[])
    paths = DataPaths()
    universe = build_universe_from_dict_v3(paths.dict_v3_path, cfg)

    years = list(range(cfg.years_hist_start, cfg.years_hist_end + 1))

    food_df = _build_food_from_fbs(paths.fbs_csv, paths.dict_v3_path, universe)
    if food_df is None or food_df.empty:
        raise ValueError("No food data returned from FBS.")
    food_df = food_df[food_df["year"].isin(years)].copy()
    food_df["country"] = food_df["country"].apply(_norm_m49)
    food_df["food_t"] = pd.to_numeric(food_df["food_t"], errors="coerce").fillna(0.0)
    food_df.loc[food_df["food_t"] < 0, "food_t"] = 0.0

    comm_to_item, item_to_per_ton = _load_item_nutrition_map(paths.dict_v3_path)
    if not comm_to_item:
        raise ValueError("Empty Item_Nutrition_Map from dict_v3.")

    food_df["Item"] = food_df["commodity"].map(comm_to_item)
    missing_map = food_df["Item"].isna().sum()
    if missing_map:
        missing_items = sorted(set(food_df.loc[food_df["Item"].isna(), "commodity"]))
        print(f"[WARN] Missing Item_Nutrition_Map for {missing_map} rows. Example: {missing_items[:10]}")
    food_df = food_df.dropna(subset=["Item"])

    base = (
        food_df.groupby(["country", "Item", "year"], as_index=False)["food_t"]
        .sum()
        .rename(columns={"country": "M49_Country_Code", "food_t": "demand_t"})
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
    out_path = os.path.join(output_dir, "Nutrition_profile_recalculated_fromD0_food_demand.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        wide.to_excel(writer, sheet_name="nutrition_profile", index=False)

    print(f"Wrote {len(wide)} rows to {out_path}")


if __name__ == "__main__":
    run()
