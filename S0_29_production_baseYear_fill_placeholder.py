import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product

BASE_YEAR_COL = 'Y2020'
PRODUCTION_FILE = 'Production_Crops_Livestock_E_All_Data_NOFLAG.csv'
TARGET_ELEMENTS = {
    'Yield': 'crop',
    'Yield/Carcass Weight': 'livestock',
}


def format_m49(val: str) -> str:
    s = str(val).strip()
    if not s:
        return s
    if s.startswith("'"):
        return s
    if s.isdigit():
        return f"'{int(s):03d}"
    try:
        return f"'{int(float(s)) :03d}"
    except Exception:
        return s


def load_dict(base_dir: Path):
    dict_path = (base_dir / '../../src/dict_v3.xlsx').resolve()
    region_df = pd.read_excel(dict_path, sheet_name='region')
    region_df['Region_label_new'] = region_df['Region_label_new'].astype(str)
    region_df = region_df[region_df['Region_label_new'].str.lower() != 'no'].copy()
    region_df['M49_fmt'] = region_df['M49_Country_Code'].apply(format_m49)
    emis_df = pd.read_excel(dict_path, sheet_name='Emis_item')
    mask = emis_df['Production_file_source'].astype(str).str.strip() == PRODUCTION_FILE
    emis_filtered = emis_df[mask].copy()
    emis_filtered['Item_Production_Map'] = emis_filtered['Item_Production_Map'].astype(str).str.strip()
    emis_filtered = emis_filtered[emis_filtered['Item_Production_Map'] != '']
    return region_df, emis_filtered


def build_combos(region_df: pd.DataFrame, emis_df: pd.DataFrame):
    m49_list = sorted(region_df['M49_fmt'].dropna().unique())
    items = sorted(emis_df['Item_Production_Map'].unique())
    combos = pd.DataFrame(product(m49_list, items), columns=['M49', 'Item'])
    item_cat_map = {
        row['Item_Production_Map']: str(row['Item_Cat2']).strip().lower()
        for _, row in emis_df[['Item_Production_Map', 'Item_Cat2']].iterrows()
    }
    combos['cat'] = combos['Item'].map(item_cat_map).fillna('')
    return combos, item_cat_map


def build_templates(df: pd.DataFrame):
    templates = {}
    cols = df.columns.tolist()
    for _, row in df.iterrows():
        key = (row['Item'], row['Element'])
        if key not in templates:
            templates[key] = row.to_dict()
    return templates, cols


def create_row(template, m49, area, element, value):
    row = template.copy()
    row['M49_Country_Code'] = m49
    row['Area'] = area
    row['Element'] = element
    row[BASE_YEAR_COL] = value
    return row


def ensure_production(df: pd.DataFrame,
                      combos: pd.DataFrame,
                      templates: dict,
                      region_lookup: dict,
                      country_lookup: dict):
    changed_keys = set()
    rows_to_add = []
    prod_mask = df['Element'].astype(str).str.strip() == 'Production'
    for m49, item in combos[['M49','Item']].itertuples(index=False):
        mask = prod_mask & (df['M49_Country_Code'] == m49) & (df['Item'] == item)
        if mask.any():
            val = df.loc[mask, BASE_YEAR_COL].iloc[0]
            if pd.isna(val) or val <= 0:
                df.loc[mask, BASE_YEAR_COL] = 1.0 # 0.001
                changed_keys.add((m49, item))
        else:
            tpl = templates.get((item, 'Production'))
            if tpl is None:
                continue
            area = country_lookup.get(m49, 'World')
           # row = create_row(tpl, m49, area, 'Production', 0.001)
            row = create_row(tpl, m49, area, 'Production', 1.0)
            rows_to_add.append(row)
            changed_keys.add((m49, item))
    if rows_to_add:
        df = pd.concat([df, pd.DataFrame(rows_to_add)], ignore_index=True)
    return df, changed_keys


def fill_elements(
    df: pd.DataFrame,
    combos: pd.DataFrame,
    templates: dict,
    region_lookup: dict,
    country_lookup: dict,
    item_cat_map: dict,
    changed_keys: set,
):
    df['_Region'] = df['M49_Country_Code'].map(region_lookup)
    crop_items = {item for item, cat in item_cat_map.items() if cat == 'crop'}
    livestock_items = {item for item, cat in item_cat_map.items() if cat in {'meat', 'dairy', 'other'}}

    for element, group in TARGET_ELEMENTS.items():
        items = crop_items if group == 'crop' else livestock_items
        elem_df = df[df['Element'] == element].copy()
        valid = elem_df[elem_df[BASE_YEAR_COL].notna() & (elem_df[BASE_YEAR_COL] > 0)]
        region_means = valid.groupby(['_Region', 'Item'])[BASE_YEAR_COL].mean().to_dict()
        world_means = valid.groupby('Item')[BASE_YEAR_COL].mean().to_dict()

        rows_to_add = []
        for m49, item in combos[['M49', 'Item']].itertuples(index=False):
            if item not in items:
                continue
            mask = (
                (df['M49_Country_Code'] == m49)
                & (df['Item'] == item)
                & (df['Element'] == element)
            )
            needs_fill = False
            if mask.any():
                val = df.loc[mask, BASE_YEAR_COL].iloc[0]
                if pd.isna(val) or val <= 0:
                    needs_fill = True
            else:
                needs_fill = True
            if not needs_fill:
                continue

            region = region_lookup.get(m49)
            fill_value = region_means.get((region, item))
            if fill_value is None or np.isnan(fill_value):
                fill_value = world_means.get(item)
            if fill_value is None or np.isnan(fill_value):
                fill_value = 0.0

            if mask.any():
                df.loc[mask, BASE_YEAR_COL] = float(fill_value)
            else:
                tpl = templates.get((item, element))
                if tpl is None:
                    continue
                area = country_lookup.get(m49, 'World')
                row = create_row(tpl, m49, area, element, float(fill_value))
                rows_to_add.append(row)

        if rows_to_add:
            df = pd.concat([df, pd.DataFrame(rows_to_add)], ignore_index=True)

    df = df.drop(columns=['_Region'], errors='ignore')
    return df


def main():
    base_dir = Path(__file__).resolve().parent
    region_df, emis_df = load_dict(base_dir)
    combos, item_cat_map = build_combos(region_df, emis_df)
    region_lookup = dict(zip(region_df['M49_fmt'], region_df['Region_agg2']))
    country_lookup = dict(zip(region_df['M49_fmt'], region_df['Region_label_new']))

    prod_path = (base_dir / '../../input/Production_Trade/Production_Crops_Livestock_E_All_Data_NOFLAG_yield_refilled.csv').resolve()
    df = pd.read_csv(prod_path)
    df['M49_Country_Code'] = df['M49_Country_Code'].apply(format_m49)
    templates, cols = build_templates(df)

    df, changed_keys = ensure_production(df, combos, templates, region_lookup, country_lookup)
    df = fill_elements(df, combos, templates, region_lookup, country_lookup, item_cat_map, changed_keys)

    out_path = prod_path.with_name(prod_path.stem + '_baseYearFilled.csv')
    df[cols].to_csv(out_path, index=False)
    print(f"Filled data written to: {out_path}")

if __name__ == '__main__':
    main()
