# -*- coding: utf-8 -*-
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


def _norm_m49(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ''
    s = str(val).strip()
    if s.startswith("'"):
        s = s[1:]
    s = s.strip()
    if not s:
        return ''
    if s.count('.') == 1:
        left, right = s.split('.', 1)
        if left.isdigit() and right.strip('0') == '':
            s = left
    if s.isdigit():
        return f"'{s.zfill(3)}"
    return f"'{s}"


def _year_cols(cols: Iterable[str]) -> List[str]:
    out = []
    for c in cols:
        if isinstance(c, str) and c.startswith('Y') and c[1:].isdigit():
            out.append(c)
    return out


def _load_region_m49(dict_path: Path) -> List[str]:
    df = pd.read_excel(dict_path, sheet_name='region')
    df.columns = [str(c).strip() for c in df.columns]
    if 'M49_Country_Code' not in df.columns or 'Region_label_new' not in df.columns:
        return []
    label = df['Region_label_new'].astype(str).str.strip().str.lower()
    df = df[(label != 'no') & (label != '')]
    codes = {_norm_m49(v) for v in df['M49_Country_Code'].tolist()}
    return sorted(c for c in codes if c)


def _load_production_long(prod_path: Path,
                          country_set: set,
                          item_set: set) -> Tuple[Dict[Tuple[str, int], Dict[str, float]], List[str]]:
    with prod_path.open('r', encoding='utf-8', errors='ignore') as f:
        df = pd.read_csv(f)
    df.columns = [str(c).replace('\ufeff', '').strip() for c in df.columns]
    if 'Element' not in df.columns or 'Item' not in df.columns:
        return {}, []
    df['Element'] = df['Element'].astype(str).str.strip()
    df = df[df['Element'].str.lower() == 'production']
    if 'M49_Country_Code' not in df.columns:
        return {}, []
    df['m49_norm'] = df['M49_Country_Code'].apply(_norm_m49)
    df = df[df['m49_norm'].isin(country_set)]
    df['Item'] = df['Item'].astype(str).str.strip()
    df = df[df['Item'].isin(item_set)]
    if df.empty:
        return {}, []
    year_cols = _year_cols(df.columns)
    df = df[['m49_norm', 'Item'] + year_cols]
    for c in year_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
    long_df = df.melt(id_vars=['m49_norm', 'Item'],
                      value_vars=year_cols,
                      var_name='year',
                      value_name='value')
    long_df['year'] = pd.to_numeric(long_df['year'].astype(str).str.lstrip('Y'), errors='coerce')
    long_df = long_df.dropna(subset=['year'])
    long_df['year'] = long_df['year'].astype(int)
    long_df['value'] = pd.to_numeric(long_df['value'], errors='coerce').fillna(0.0)
    prod_by: Dict[Tuple[str, int], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in long_df.itertuples(index=False):
        prod_by[(r.m49_norm, int(r.year))][str(r.Item)] += float(r.value)
    return prod_by, year_cols


def _split_rows(demand_df: pd.DataFrame,
                year_cols: List[str],
                prod_by: Dict[Tuple[str, int], Dict[str, float]],
                split_rules: Dict[str, List[Tuple[str, str]]]) -> List[Dict[str, object]]:
    new_rows: List[Dict[str, object]] = []
    target_items = set(split_rules.keys())
    if demand_df.empty or not target_items:
        return new_rows
    for _, row in demand_df[demand_df['Item'].isin(target_items)].iterrows():
        row_dict = row.to_dict()
        base_item = str(row_dict.get('Item', '')).strip()
        m49_norm = row_dict.get('m49_norm', '')
        splits = split_rules.get(base_item, [])
        if not splits:
            continue
        for new_item, prod_item in splits:
            new_row = dict(row_dict)
            new_row['Item'] = new_item
            for ycol in year_cols:
                year = int(ycol[1:])
                try:
                    base_val = float(new_row.get(ycol, 0.0) or 0.0)
                except Exception:
                    base_val = 0.0
                prod_map = prod_by.get((m49_norm, year), {})
                total_prod = sum(float(prod_map.get(p_item, 0.0) or 0.0) for _, p_item in splits)
                if total_prod > 0 and base_val != 0:
                    share = float(prod_map.get(prod_item, 0.0) or 0.0) / total_prod
                    new_row[ycol] = base_val * share
                else:
                    new_row[ycol] = 0.0
            if any(float(new_row.get(ycol, 0.0) or 0.0) != 0.0 for ycol in year_cols):
                new_rows.append(new_row)
    return new_rows


def main() -> None:
    base = Path(__file__).resolve().parents[2]
    fbs_path = base / 'input' / 'Driver' / 'retired_unused_raw' / 'FoodBalanceSheets_E_All_Data_NOFLAG.csv'
    prod_path = base / 'input' / 'Production_Trade' / 'Production_Crops_Livestock_E_All_Data_NOFLAG_yield_refilled_baseYearFilled.csv'
    dict_path = base / 'src' / 'dict_v3.xlsx'
    out_path = base / 'input' / 'Production_Trade' / 'FoodBalanceSheets_E_All_Data_NOFLAG_demand_refilled.xlsx'

    country_codes = _load_region_m49(dict_path)
    country_set = set(country_codes)

    with fbs_path.open('r', encoding='utf-8', errors='ignore') as f:
        full_df = pd.read_csv(f)
    full_df.columns = [str(c).replace('\ufeff', '').strip() for c in full_df.columns]
    if 'Element' not in full_df.columns or 'Item' not in full_df.columns:
        raise ValueError("FBS file missing Element/Item columns")
    if 'M49_Country_Code' not in full_df.columns:
        raise ValueError("FBS file missing M49_Country_Code column")

    year_cols = _year_cols(full_df.columns)
    m49_norm = full_df['M49_Country_Code'].apply(_norm_m49)
    elem_norm = full_df['Element'].astype(str).str.strip().str.lower()
    demand_mask = (
        m49_norm.isin(country_set)
        & elem_norm.isin({'domestic supply quantity', 'seed', 'feed', 'food', 'losses'})
    )
    demand_df = full_df.loc[demand_mask].copy()
    demand_df['m49_norm'] = m49_norm[demand_mask].values
    demand_df['Item'] = demand_df['Item'].astype(str).str.strip()
    for c in year_cols:
        demand_df[c] = pd.to_numeric(demand_df[c], errors='coerce').fillna(0.0)

    split_rules = {
        'Meat, Other': [
            ('Meat, Other-asses', 'Meat of asses, fresh or chilled'),
            ('Meat, Other-camels', 'Meat of camels, fresh or chilled'),
            ('Meat, Other-horse', 'Horse meat, fresh or chilled'),
            ('Meat, Other-other domestic camelids', 'Meat of other domestic camelids, fresh or chilled'),
            ('Meat, Other-mules', 'Meat of mules, fresh or chilled'),
        ],
        'Milk - Excluding Butter': [
            ('Milk-buffalo', 'Raw milk of buffalo'),
            ('Milk-camel', 'Raw milk of camel'),
            ('Milk-cattle', 'Raw milk of cattle'),
            ('Milk-goats', 'Raw milk of goats'),
            ('Milk-sheep', 'Raw milk of sheep'),
        ],
        'Bovine Meat': [
            ('Bovine Meat-buffalo', 'Meat of buffalo, fresh or chilled'),
            ('Bovine Meat-cattle', 'Meat of cattle with the bone, fresh or chilled'),
        ],
        'Poultry Meat': [
            ('Poultry Meat-chickens', 'Meat of chickens, fresh or chilled'),
            ('Poultry Meat-ducks', 'Meat of ducks, fresh or chilled'),
            ('Poultry Meat-turkeys', 'Meat of turkeys, fresh or chilled'),
        ],
        'Mutton & Goat Meat': [
            ('Mutton & Goat Meat-goat', 'Meat of goat, fresh or chilled'),
            ('Mutton & Goat Meat-sheep', 'Meat of sheep, fresh or chilled'),
        ],
    }
    prod_items = sorted({prod for pairs in split_rules.values() for _, prod in pairs})
    prod_by, _ = _load_production_long(prod_path, country_set, set(prod_items))

    new_rows = _split_rows(demand_df, year_cols, prod_by, split_rules)
    extra_df = pd.DataFrame(new_rows) if new_rows else pd.DataFrame(columns=full_df.columns)
    if 'm49_norm' in extra_df.columns:
        extra_df = extra_df.drop(columns=['m49_norm'])
    out_df = pd.concat([full_df, extra_df], ignore_index=True)
    out_df = out_df[full_df.columns]
    out_df = out_df.sort_values(['M49_Country_Code', 'Item', 'Element'], kind='mergesort')
    out_df.to_excel(out_path, index=False)
    print(f"output_rows={len(out_df)} new_rows={len(extra_df)} output={out_path}")


if __name__ == '__main__':
    main()
