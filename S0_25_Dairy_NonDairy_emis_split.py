
import os

import numpy as np
import pandas as pd

from config_paths import get_input_base, get_results_base


SPLIT_ELEMENT_SET = {
    'Enteric fermentation (Emissions CH4)',
    'Manure management (Emissions CH4)',
    'Manure management (Emissions N2O)',
    'Manure left on pasture (Emissions N2O)',
    'Manure applied to soils (Emissions N2O)',
}

BASE_ITEM_ALIASES = {
    'Buffalo': 'Buffalo',
    'Buffaloes': 'Buffalo',
    'Camels': 'Camels',
    'Goats': 'Goats',
    'Sheep': 'Sheep',
}

SPLIT_TARGETS = {
    'Buffalo': ('Buffalo, dairy', 'Buffalo, non-dairy'),
    'Camels': ('Camel, dairy', 'Camel, non-dairy'),
    'Goats': ('Goats, dairy', 'Goats, non-dairy'),
    'Sheep': ('Sheep, dairy', 'Sheep, non-dairy'),
}


def _extract_year(col_name: str) -> int | None:
    text = str(col_name).strip()
    if text.startswith('Y') and text[1:].isdigit():
        return int(text[1:])
    return None


def _year_columns(df: pd.DataFrame) -> list[str]:
    return [
        col for col in df.columns
        if isinstance(col, str) and col.startswith('Y') and col[1:].isdigit()
    ]


def _normalize_ratio_columns(ratio_df: pd.DataFrame, emission_year_cols: list[str]) -> pd.DataFrame:
    available_year_cols = _year_columns(ratio_df)
    if not available_year_cols:
        return pd.DataFrame(index=ratio_df.index, columns=emission_year_cols, dtype=float)

    available_years = sorted(_extract_year(col) for col in available_year_cols if _extract_year(col) is not None)
    if not available_years:
        return pd.DataFrame(index=ratio_df.index, columns=emission_year_cols, dtype=float)

    min_year = available_years[0]
    max_year = available_years[-1]
    normalized = pd.DataFrame(index=ratio_df.index)

    for year_col in emission_year_cols:
        year = _extract_year(year_col)
        if year_col in ratio_df.columns:
            normalized[year_col] = pd.to_numeric(ratio_df[year_col], errors='coerce')
            continue
        if year is None:
            normalized[year_col] = np.nan
            continue
        if year < min_year:
            normalized[year_col] = pd.to_numeric(ratio_df[f'Y{min_year}'], errors='coerce')
        elif year > max_year:
            normalized[year_col] = pd.to_numeric(ratio_df[f'Y{max_year}'], errors='coerce')
        else:
            normalized[year_col] = np.nan

    return normalized


def _build_ratio_lookup(stock_df: pd.DataFrame, emission_year_cols: list[str]) -> dict[str, dict[str, pd.DataFrame]]:
    stock_year_cols = _year_columns(stock_df)
    ratio_lookup: dict[str, dict[str, pd.DataFrame]] = {}

    for base_item, (dairy_item, non_dairy_item) in SPLIT_TARGETS.items():
        dairy_stock = (
            stock_df[stock_df['Item'] == dairy_item]
            .set_index('M49_Country_Code')[stock_year_cols]
        )
        non_dairy_stock = (
            stock_df[stock_df['Item'] == non_dairy_item]
            .set_index('M49_Country_Code')[stock_year_cols]
        )

        dairy_stock, non_dairy_stock = dairy_stock.align(non_dairy_stock, fill_value=0)
        total_stock = dairy_stock + non_dairy_stock

        dairy_ratio = dairy_stock.div(total_stock).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        non_dairy_ratio = non_dairy_stock.div(total_stock).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        no_ratio_mask = total_stock.isna() | total_stock.eq(0)
        dairy_ratio = dairy_ratio.mask(no_ratio_mask, 0.0)
        non_dairy_ratio = non_dairy_ratio.mask(no_ratio_mask, 1.0)

        ratio_lookup[base_item] = {
            dairy_item: _normalize_ratio_columns(dairy_ratio, emission_year_cols),
            non_dairy_item: _normalize_ratio_columns(non_dairy_ratio, emission_year_cols),
        }

    return ratio_lookup


def split_emissions():
    """
    Split livestock historical emissions for Buffalo/Camels/Goats/Sheep into
    dairy and non-dairy items using stock ratios. Years before 2000 inherit
    the Y2000 ratio because stock splits start in 2000.
    """
    input_base = get_input_base()
    output_base = get_results_base()

    manure_stock_path = os.path.join(input_base, 'Manure_Stock', 'Environment_LivestockManure_with_ratio.csv')
    emissions_path = os.path.join(input_base, 'Emission', 'Emissions_livestock_E_All_Data_NOFLAG.csv')

    input_output_path = os.path.join(input_base, 'Emission', 'Emissions_livestock_dairy_split.csv')
    intermediate_dir = os.path.join(output_base, 'intermediate')
    os.makedirs(intermediate_dir, exist_ok=True)
    intermediate_output_path = os.path.join(intermediate_dir, 'Emissions_livestock_dairy_split.csv')

    print('Loading and processing stock data...')
    stock_df = pd.read_csv(manure_stock_path, encoding='utf-8')
    stock_df = stock_df[stock_df['Element'] == 'Stocks'].copy()

    print('Loading emissions data...')
    emis_df = pd.read_csv(emissions_path, encoding='utf-8')
    emission_year_cols = _year_columns(emis_df)
    ratio_lookup = _build_ratio_lookup(stock_df, emission_year_cols)

    items_to_split = set(BASE_ITEM_ALIASES.keys())
    split_mask = emis_df['Element'].isin(SPLIT_ELEMENT_SET) & emis_df['Item'].isin(items_to_split)
    emis_to_split = emis_df[split_mask].copy()
    emis_to_keep = emis_df[~split_mask].copy()

    print(f'Splitting {len(emis_to_split)} rows across {len(emission_year_cols)} year columns...')
    new_rows = []

    for _, row in emis_to_split.iterrows():
        raw_item = str(row['Item']).strip()
        base_item = BASE_ITEM_ALIASES[raw_item]
        dairy_item, non_dairy_item = SPLIT_TARGETS[base_item]
        ratio_frames = ratio_lookup.get(base_item, {})
        dairy_ratios = ratio_frames.get(dairy_item)
        non_dairy_ratios = ratio_frames.get(non_dairy_item)
        country_code = row['M49_Country_Code']

        if dairy_ratios is not None and country_code in dairy_ratios.index:
            dairy_ratio_values = dairy_ratios.loc[country_code]
            non_dairy_ratio_values = non_dairy_ratios.loc[country_code]
        else:
            dairy_ratio_values = pd.Series(0.0, index=emission_year_cols, dtype=float)
            non_dairy_ratio_values = pd.Series(1.0, index=emission_year_cols, dtype=float)

        dairy_row = row.to_dict()
        dairy_row['Item'] = dairy_item
        non_dairy_row = row.to_dict()
        non_dairy_row['Item'] = non_dairy_item

        for year_col in emission_year_cols:
            raw_value = pd.to_numeric(row.get(year_col), errors='coerce')
            if pd.isna(raw_value):
                dairy_row[year_col] = np.nan
                non_dairy_row[year_col] = np.nan
                continue

            dairy_ratio = pd.to_numeric(dairy_ratio_values.get(year_col), errors='coerce')
            non_dairy_ratio = pd.to_numeric(non_dairy_ratio_values.get(year_col), errors='coerce')
            dairy_ratio = 0.0 if pd.isna(dairy_ratio) else float(dairy_ratio)
            non_dairy_ratio = 1.0 if pd.isna(non_dairy_ratio) else float(non_dairy_ratio)

            dairy_row[year_col] = raw_value * dairy_ratio
            non_dairy_row[year_col] = raw_value * non_dairy_ratio

        new_rows.append(dairy_row)
        new_rows.append(non_dairy_row)

    split_emis_df = pd.DataFrame(new_rows, columns=emis_df.columns)
    final_df = pd.concat([emis_to_keep, split_emis_df], ignore_index=True)
    final_df = final_df.sort_values(by=['M49_Country_Code', 'Item', 'Element']).reset_index(drop=True)

    final_df.to_csv(input_output_path, index=False, encoding='utf-8')
    final_df.to_csv(intermediate_output_path, index=False, encoding='utf-8')

    print(f'Processing complete. Output saved to {input_output_path}')
    print(f'Legacy copy saved to {intermediate_output_path}')

if __name__ == '__main__':
    split_emissions()
