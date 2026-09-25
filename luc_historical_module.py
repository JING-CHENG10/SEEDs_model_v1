#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Historical LUC emissions loading and processing module.
Read historical emissions from Emission_LULUCF_Historical_updated.xlsx,
select Wood harvest, Forest, De/Reforestation_crop, and De/Reforestation_pasture,
and format outputs consistently with future emissions.
"""
from __future__ import annotations
from typing import Optional, Dict
import pandas as pd
import numpy as np


TARGET_PROCESSES = ['Wood harvest', 'Forest', 'De/Reforestation_crop', 'De/Reforestation_pasture']


def read_luc_historical_emissions(
    hist_file: str,
    dict_v3_path: Optional[str] = None,
    years: Optional[list] = None,
) -> pd.DataFrame:
    """
    Read historical LUC emissions from Emission_LULUCF_Historical_updated.xlsx.
    
    Parameters
    ----
    hist_file : str
        Path to Emission_LULUCF_Historical_updated.xlsx.
    dict_v3_path : str, optional
        Path to dict_v3.xlsx for M49 and region_label_new mappings.
    years : list, optional
        Years to retain, usually 2000-2020; None reads all years.
    
    Returns
    ----
    DataFrame
        Columns: M49_Country_Code, Region_label_new, year, Process, GHG, value (tCO2/year).
        Include only target-process records with Select=1.
    """
    # Read historical emissions.
    df = pd.read_excel(hist_file, sheet_name='LULUCF_updated')
    df.columns = [str(c).strip() for c in df.columns]
    
    # Retain selected rows with Select=1.
    if 'Select' in df.columns:
        df = df[df['Select'] == 1]
    
    # Filter target processes.
    if 'Land Category' in df.columns:
        df = df[df['Land Category'].isin(TARGET_PROCESSES)]
    elif 'Process' in df.columns:
        df = df[df['Process'].isin(TARGET_PROCESSES)]
    else:
        print("[WARN] 无法找到Land Category或Process列，将不过滤过程")
    
    if df.empty:
        return pd.DataFrame(columns=[
            'M49_Country_Code', 'Region_label_new', 'year', 'Process', 'GHG', 'value'
        ])
    
    # Ensure M49_Country_Code exists and is standardized.
    if 'M49_Country_Code' not in df.columns:
        raise KeyError("M49_Country_Code列缺失")
    
    # Exclude global aggregate rows with M49='000' or Region_label_new='World'.
    # Historical World totals would cause double counting.
    df = df[~df['M49_Country_Code'].astype(str).str.strip().str.lstrip("'\"").isin(['0', '00', '000', '1'])]
    if 'Region_label_new' in df.columns:
        df = df[df['Region_label_new'] != 'World']
    print(f"[LUC历史] 过滤World汇总行后: {len(df)} 行")
    
    # Standardize M49 to an apostrophe followed by three digits.
    def _normalize_m49(val):
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

    df['M49_Country_Code'] = df['M49_Country_Code'].apply(_normalize_m49)
    
    # Identify year and process columns.
    year_cols = [c for c in df.columns if c.startswith('Y') and len(c) == 5]  # Y2000, Y2020, etc.
    process_col = 'Land Category' if 'Land Category' in df.columns else 'Process'
    
    # Identify GHG from Species or directly from process names.
    if 'Species' in df.columns:
        ghg_col = 'Species'
    else:
        ghg_col = None
    
    # Convert wide to long format.
    id_cols = ['M49_Country_Code', process_col]
    if ghg_col:
        id_cols.append(ghg_col)
    if 'Region_label_new' in df.columns:
        id_cols.append('Region_label_new')
    
    # Melt year columns.
    df_long = df[id_cols + year_cols].melt(
        id_vars=id_cols,
        value_vars=year_cols,
        var_name='year_str',
        value_name='value'
    )
    
    # Parse years: Y2000 -> 2000.
    df_long['year'] = df_long['year_str'].str.replace('Y', '').astype(int)
    
    # Rename columns.
    df_long = df_long.rename(columns={
        process_col: 'Process',
        ghg_col: 'GHG' if ghg_col else None
    })
    
    # Remove NaN and zero values for simplicity.
    df_long = df_long.dropna(subset=['value'])
    df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')
    df_long = df_long.dropna(subset=['value'])
    
    # Unit conversion
    # Convert historical MtCO2/year (million tonnes) to kt CO2/year (kilotonnes) for aggregation.
    # MtCO2 -> kt: multiply by 1e3; 1 MtCO2 = 1000 ktCO2.
    df_long['value'] = df_long['value'] * 1e3  # MtCO2/yr ?? kt CO2/yr
    
    # Default to CO2 if no GHG column exists.
    if 'GHG' not in df_long.columns or df_long['GHG'].isna().all():
        df_long['GHG'] = 'CO2'
    
    # Load Region_label_new from dict_v3 if absent.
    if 'Region_label_new' not in df_long.columns or df_long['Region_label_new'].isna().all():
        if dict_v3_path:
            try:
                region_df = pd.read_excel(dict_v3_path, sheet_name='region',
                                         usecols=['M49_Country_Code', 'Region_label_new'])
                region_df['M49_Country_Code'] = region_df['M49_Country_Code'].astype(str).str.strip()
                region_map = dict(zip(region_df['M49_Country_Code'], region_df['Region_label_new']))
                df_long['Region_label_new'] = df_long['M49_Country_Code'].map(region_map)
            except Exception as e:
                print(f"[WARN] 无法从dict_v3加载Region_label_new: {e}")
                df_long['Region_label_new'] = 'Unknown'
        else:
            df_long['Region_label_new'] = 'Unknown'
    
    # Add Item by mapping Process to its corresponding Item.
    process_to_item_map = {
        'Wood harvest': 'Roundwood',
        'Forest': 'Forestland',
        'De/Reforestation_crop': 'De/Reforestation_crop area',
        'De/Reforestation_pasture': 'De/Reforestation_pasture area'
    }
    df_long['Item'] = df_long['Process'].map(process_to_item_map)
    
    # Filter years if supplied.
    if years:
        years_set = set(int(y) for y in years)
        df_long = df_long[df_long['year'].isin(years_set)]
    
    # Select and order columns.
    cols_out = ['M49_Country_Code', 'Region_label_new', 'year', 'Process', 'Item', 'GHG', 'value']
    df_long = df_long[cols_out].reset_index(drop=True)
    
    # Aggregate possible duplicates.
    df_long = df_long.groupby(cols_out[:-1], as_index=False)['value'].sum()
    
    return df_long
