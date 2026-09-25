# -*- coding: utf-8 -*-
"""
S3.5_land_use_change: land-use changes with a land-carbon-price intensification proxy.

Period-to-period comparison:
- Calculate land deltas relative to the previous period.
- For example, compare 2020 with LUH2 2020, 2040 with 2020, and 2080 with 2040.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

@dataclass
class LUCConfig:
    yield_t_per_ha_default: float = 3.0
    grass_intensity_tdm_per_ha: float = 5.0
    cropland_restore_share: float = 0.5
    land_carbon_price_per_tco2: float = 0.0
    # Land carbon price scenario ($/tCO2e)
    land_carbon_price_per_tco2: float = 0.0
    carbon_stock_tco2_per_ha: dict = None  # {'forest':150,'cropland':10,'grassland':30}
    # Defaults must not change physical land demand. Yield/feed efficiency
    # changes should come only from explicit scenario variables.
    intensification_per_usd: float = 0.0
    intensification_cap: float = 0.0

def _lc(df: pd.DataFrame) -> pd.DataFrame:
    z = df.copy(); z.columns = [str(c).strip() for c in z.columns]; return z

def compute_luc_areas(*, demand_df: pd.DataFrame, production_df: Optional[pd.DataFrame]=None,
                      crop_yield_df: Optional[pd.DataFrame]=None, grass_requirement_df: Optional[pd.DataFrame]=None,
                      base_area_df: Optional[pd.DataFrame]=None, cfg: LUCConfig = LUCConfig()) -> Dict[str, pd.DataFrame]:
    """
    Calculate land-use-change areas.
    
    Period-to-period comparison:
    - Compare the base year, e.g. 2020, with LUH2 baseline data.
    - Compare later years with previous modeled new_*_ha results.
    - For example, compare 2040 with 2020 new_*_ha, and 2080 with 2040.
    """
    d = _lc(demand_df)
    cn = [c for c in d.columns if 'country' in c.lower() and 'm49' not in c.lower()][0]
    yn = [c for c in d.columns if 'year' in c.lower()][0]
    in_ = [c for c in d.columns if 'comm' in c.lower() or 'item' in c.lower()][0]
    dn = [c for c in d.columns if 'demand' in c.lower() or 'qty' in c.lower()][0]
    d = d.rename(columns={cn:'country', yn:'year', in_:'commodity', dn:'demand_t'})
    # Retain M49_Country_Code if present.
    select_cols = ['country','year','commodity','demand_t']
    if 'M49_Country_Code' in d.columns:
        select_cols = ['M49_Country_Code'] + select_cols
    d = d[select_cols]

    if crop_yield_df is not None and len(crop_yield_df):
        y = _lc(crop_yield_df)
        cn = [c for c in y.columns if 'country' in c.lower() and 'm49' not in c.lower()][0]
        yn = [c for c in y.columns if 'year' in c.lower()][0]
        in_ = [c for c in y.columns if 'comm' in c.lower() or 'item' in c.lower()][0]
        yv = [c for c in y.columns if 'yield' in c.lower()][0]
        y = y.rename(columns={cn:'country', yn:'year', in_:'commodity', yv:'yield_t_per_ha'})
        # Retain M49_Country_Code if present.
        select_cols = ['country','year','commodity','yield_t_per_ha']
        if 'M49_Country_Code' in y.columns:
            select_cols = ['M49_Country_Code'] + select_cols
        y = y[select_cols]
    else:
        # Copy all columns from d, including M49.
        base_cols = ['country','year','commodity']
        if 'M49_Country_Code' in d.columns:
            base_cols = ['M49_Country_Code'] + base_cols
        y = d[base_cols].drop_duplicates()
        y['yield_t_per_ha'] = cfg.yield_t_per_ha_default

    # Merge using every shared key column.
    merge_keys = ['country','year','commodity']
    if 'M49_Country_Code' in d.columns and 'M49_Country_Code' in y.columns:
        merge_keys = ['M49_Country_Code'] + merge_keys
    z = d.merge(y, on=merge_keys, how='left')
    z['yield_t_per_ha'] = z['yield_t_per_ha'].replace(0, np.nan).fillna(cfg.yield_t_per_ha_default)
    
    # Check yield data sources.
    yield_na_count = z['yield_t_per_ha'].isna().sum()
    yield_default_count = (z['yield_t_per_ha'] == cfg.yield_t_per_ha_default).sum()
    print(f"[S3_5 DEBUG] yield数据统计: 总行数={len(z)}, NA后填默认值={yield_na_count}, 使用默认值{cfg.yield_t_per_ha_default}的行数={yield_default_count}")
    if len(z) > 0:
        sample_yields = z[['country', 'year', 'commodity', 'yield_t_per_ha']].drop_duplicates().head(5)
        print(f"[S3_5 DEBUG] yield样本:\n{sample_yields.to_string(index=False)}")
    
    z['crop_area_need_ha'] = z['demand_t'] / z['yield_t_per_ha']
    # Include M49 in groupby if present.
    group_cols = ['country','year']
    if 'M49_Country_Code' in z.columns:
        group_cols = ['M49_Country_Code'] + group_cols
    crop_need = z.groupby(group_cols, as_index=False)['crop_area_need_ha'].sum()
    
    # Check global total area.
    for year in [2020, 2080]:
        year_total = crop_need[crop_need['year'] == year]['crop_area_need_ha'].sum()
        print(f"[S3_5 DEBUG] {year}年全球耕地面积需求: {year_total:,.0f} ha")

    # Land-carbon-price intensification reduces land demand without changing physical quantities.
    if cfg.land_carbon_price_per_tco2 and cfg.land_carbon_price_per_tco2>0:
        red = min(cfg.intensification_per_usd * cfg.land_carbon_price_per_tco2, cfg.intensification_cap)
        crop_need['crop_area_need_ha'] *= (1.0 - red)

    # Grassland demand
    if grass_requirement_df is not None and len(grass_requirement_df):
        print(f"[S3_5 DEBUG] grass_requirement_df 输入: {len(grass_requirement_df)} 行, 列={list(grass_requirement_df.columns)}")
        g = _lc(grass_requirement_df)
        cn = [c for c in g.columns if 'country' in c.lower() and 'm49' not in c.lower()][0]
        yn = [c for c in g.columns if 'year' in c.lower()][0]
        area_col = next((c for c in g.columns if 'area' in c.lower()), None)
        dm_col = next((c for c in g.columns if 'grass_t' in c.lower() or 'dem' in c.lower()), None)
        rename_map = {cn: 'country', yn: 'year'}
        if area_col:
            rename_map[area_col] = 'grass_area_need_ha'
        if dm_col:
            rename_map[dm_col] = 'grass_tdm'
        g = g.rename(columns=rename_map)
        # Retain M49_Country_Code.
        keep_cols = ['country','year']
        if 'M49_Country_Code' in g.columns:
            keep_cols = ['M49_Country_Code'] + keep_cols
        if 'grass_tdm' in g.columns:
            keep_cols.append('grass_tdm')
        if 'grass_area_need_ha' in g.columns:
            keep_cols.append('grass_area_need_ha')
        g = g[keep_cols]
        if 'grass_area_need_ha' not in g.columns and 'grass_tdm' in g.columns:
            g['grass_area_need_ha'] = g['grass_tdm'] / max(cfg.grass_intensity_tdm_per_ha, 1e-9)
        if 'grass_tdm' not in g.columns and 'grass_area_need_ha' in g.columns:
            g['grass_tdm'] = g['grass_area_need_ha'] * max(cfg.grass_intensity_tdm_per_ha, 1e-9)
        # Check grassland demand.
        for year in [2020, 2080]:
            year_total = g[g['year'] == year]['grass_area_need_ha'].sum() if 'grass_area_need_ha' in g.columns else 0
            print(f"[S3_5 DEBUG] {year}年全球草地面积需求: {year_total:,.0f} ha")
    else:
        print(f"[S3_5 DEBUG]  grass_requirement_df 为空或None！草地需求将设为0")
        # Copy all crop_need keys, including M49.
        base_cols = ['country','year']
        if 'M49_Country_Code' in crop_need.columns:
            base_cols = ['M49_Country_Code'] + base_cols
        g = crop_need[base_cols].copy()
        g['grass_area_need_ha'] = 0.0

    # Merge on all shared keys, including M49.
    merge_keys = ['country','year']
    if 'M49_Country_Code' in crop_need.columns and 'M49_Country_Code' in g.columns:
        merge_keys = ['M49_Country_Code'] + merge_keys
    need = crop_need.merge(g, on=merge_keys, how='outer').fillna(0.0)
    need['target_cropland_ha'] = need['crop_area_need_ha']
    need['target_grassland_ha'] = need['grass_area_need_ha']

    # Initialize baseline stocks from base-year data only.
    has_m49 = False
    if base_area_df is not None and len(base_area_df):
        b = _lc(base_area_df)
        cn = [c for c in b.columns if 'country' in c.lower() and 'm49' not in c.lower()][0]
        yn = [c for c in b.columns if 'year' in c.lower()][0]
        cc = [c for c in b.columns if 'crop' in c.lower()][0]
        fc = [c for c in b.columns if 'forest' in c.lower()][0]
        gc = [c for c in b.columns if ('grass' in c.lower() or 'pasture' in c.lower())][0]
        b = b.rename(columns={cn:'country', yn:'year', cc:'cropland_ha', fc:'forest_ha', gc:'grassland_ha'})
        # Retain M49_Country_Code.
        select_cols = ['country','year','cropland_ha','forest_ha','grassland_ha']
        if 'M49_Country_Code' in b.columns:
            select_cols = ['M49_Country_Code'] + select_cols
            has_m49 = True
        b = b[select_cols]
    else:
        # Copy need keys, including M49.
        base_cols = ['country','year']
        if 'M49_Country_Code' in need.columns:
            base_cols = ['M49_Country_Code'] + base_cols
            has_m49 = True
        b = need[base_cols].copy()
        b['cropland_ha'] = need['target_cropland_ha'].values
        b['grassland_ha'] = need['target_grassland_ha'].values
        b['forest_ha'] = 0.0

    # Period-to-period comparisons
    # Get and sort all years.
    all_years = sorted(need['year'].unique())
    print(f"[S3_5 DEBUG] 所有年份: {all_years}")
    print(f"[S3_5 DEBUG] 年份数量: {len(all_years)}, 最小年份: {min(all_years)}, 最大年份: {max(all_years)}")
    
    # Use 2020 as the base year rather than the earliest year.
    # The earliest year may be historical 2010, but LUC changes should use 2020,
    # since future scenarios and optimized Qs begin from 2020.
    base_year = 2020  # Fix the base year to 2020.
    if base_year not in all_years:
        # Fall back to the earliest year only if 2020 is absent.
        base_year = min(all_years)
        print(f"[S3_5 WARNING] 2020年不在数据中，使用最小年份 {base_year} 作为基准")
    print(f"[S3_5 DEBUG] 基准年: {base_year}")
    
    # Get all countries.
    countries = need['country'].unique()
    
    # Prepare output data.
    out_records: List[Dict] = []
    delta_records: List[Dict] = []
    
    # Carbon-pool parameters
    cs = (cfg.carbon_stock_tco2_per_ha or {'forest':150.0,'cropland':10.0,'grassland':30.0})
    
    for country in countries:
        # Get country baseline areas from base_area_df for the base year.
        country_base = b[(b['country'] == country) & (b['year'] == base_year)]
        if country_base.empty:
            # If absent, try the country's earliest year.
            country_base = b[b['country'] == country].sort_values('year').head(1)
            if country_base.empty:
                continue
        
        # Initialize previous-period areas from baseline LUH2 data.
        prev_cropland_ha = float(country_base['cropland_ha'].iloc[0])
        prev_grassland_ha = float(country_base['grassland_ha'].iloc[0])
        prev_forest_ha = float(country_base['forest_ha'].iloc[0])
        total_land = prev_cropland_ha + prev_grassland_ha + prev_forest_ha
        
        # Get M49 if present.
        m49_val = None
        if has_m49 and 'M49_Country_Code' in country_base.columns:
            m49_val = country_base['M49_Country_Code'].iloc[0]
        
        # Process years in order.
        for year in all_years:
            # Get this year's demand targets.
            country_year_need = need[(need['country'] == country) & (need['year'] == year)]
            if country_year_need.empty:
                continue
            
            target_cropland = float(country_year_need['target_cropland_ha'].iloc[0])
            target_grassland_raw = float(country_year_need['target_grassland_ha'].iloc[0])
            
            # If target grassland demand is zero or missing, indicating no projection,
            # preserve the previous area instead of setting zero and spuriously expanding forests.
            if target_grassland_raw <= 0 and prev_grassland_ha > 0:
                # Preserve previous grassland area when no demand projection exists.
                target_grassland = prev_grassland_ha
                print(f"[S3_5 DEBUG] {country} {year}: 草地需求缺失，保持上一期面积={prev_grassland_ha:,.0f} ha")
            else:
                target_grassland = target_grassland_raw
            
            # Correct land-transition logic
            # Land-transition rules:
            # 1. Increased crop/grass demand converts forest first (deforestation).
            # 2. Decreased crop/grass demand releases land for forest recovery (afforestation).
            # 3. forest + cropland + grassland <= total_land, not equality.
            # 4. Forest area must be nonnegative.
            
            # Calculate crop and grass demand changes.
            d_cropland_demand = target_cropland - prev_cropland_ha
            d_grassland_demand = target_grassland - prev_grassland_ha
            
            # Calculate total land-demand change.
            total_demand_change = d_cropland_demand + d_grassland_demand
            
            # New crop/grass areas directly equal target demand.
            new_cropland_ha = target_cropland
            new_grassland_ha = target_grassland
            
            # Forest-area change logic:
            # Increased total demand reduces forests through deforestation.
            # Decreased demand increases forests through afforestation/recovery.
            # Forest area = previous forest area - net deforestation.
            # Net deforestation = cropland expansion + grassland expansion; negative means afforestation.
            
            new_forest_ha = prev_forest_ha - total_demand_change
            
            # Constraint 1: nonnegative forest area.
            if new_forest_ha < 0:
                # Limit expansion when insufficient forest remains.
                available_forest = prev_forest_ha
                # Proportionally reduce cropland and grassland expansion.
                if total_demand_change > 0 and total_demand_change > available_forest:
                    scale_factor = available_forest / total_demand_change if total_demand_change > 0 else 1.0
                    # Reduce only expansion, preserving existing areas.
                    if d_cropland_demand > 0:
                        d_cropland_demand = d_cropland_demand * scale_factor
                    if d_grassland_demand > 0:
                        d_grassland_demand = d_grassland_demand * scale_factor
                    new_cropland_ha = prev_cropland_ha + d_cropland_demand
                    new_grassland_ha = prev_grassland_ha + d_grassland_demand
                    print(f"[S3_5 WARN] {country} {year}: 森林不足，限制扩张 scale={scale_factor:.2%}")
                new_forest_ha = 0.0
            
            # Constraint 2: total area cannot exceed initial total land.
            new_total = new_cropland_ha + new_grassland_ha + new_forest_ha
            if new_total > total_land * 1.001:  # Allow 0.1% tolerance.
                print(f"[S3_5 WARN] {country} {year}: 总面积超限 {new_total:,.0f} > {total_land:,.0f}")
            
            # Calculate actual period-to-period changes.
            d_cropland = new_cropland_ha - prev_cropland_ha
            d_grassland = new_grassland_ha - prev_grassland_ha
            d_forest = new_forest_ha - prev_forest_ha
            
            # Calculate carbon-stock changes.
            before_carbon = prev_cropland_ha * cs['cropland'] + prev_grassland_ha * cs['grassland'] + prev_forest_ha * cs['forest']
            after_carbon = new_cropland_ha * cs['cropland'] + new_grassland_ha * cs['grassland'] + new_forest_ha * cs['forest']
            d_carbon_stock = after_carbon - before_carbon
            carbon_price_cost = -d_carbon_stock * float(cfg.land_carbon_price_per_tco2 or 0.0)
            
            # Record outputs.
            out_row = {
                'country': country,
                'year': year,
                'cropland_ha': prev_cropland_ha,  # Initial period areas from previous results
                'forest_ha': prev_forest_ha,
                'grassland_ha': prev_grassland_ha,
                'target_cropland_ha': target_cropland,
                'target_grassland_ha': target_grassland,
                'new_cropland_ha': new_cropland_ha,
                'new_forest_ha': new_forest_ha,
                'new_grassland_ha': new_grassland_ha,
            }
            if has_m49:
                out_row['M49_Country_Code'] = m49_val
            out_records.append(out_row)
            
            delta_row = {
                'country': country,
                'year': year,
                'd_cropland_ha': d_cropland,
                'd_grassland_ha': d_grassland,
                'd_forest_ha': d_forest,
                'd_carbon_stock_tco2': d_carbon_stock,
                'carbon_price_cost_$': carbon_price_cost,
            }
            if has_m49:
                delta_row['M49_Country_Code'] = m49_val
            delta_records.append(delta_row)
            
            # Carry new areas forward as the previous-period state.
            prev_cropland_ha = new_cropland_ha
            prev_grassland_ha = new_grassland_ha
            prev_forest_ha = new_forest_ha
    
    # Build the output DataFrame.
    out = pd.DataFrame(out_records)
    deltas = pd.DataFrame(delta_records)
    
    # Check delta magnitudes.
    if not deltas.empty:
        us_deltas = deltas[deltas['country'] == 'United States of America']
        if not us_deltas.empty:
            for year in all_years:
                year_data = us_deltas[us_deltas['year'] == year]
                if not year_data.empty:
                    d_crop = year_data['d_cropland_ha'].iloc[0]
                    d_grass = year_data['d_grassland_ha'].iloc[0]
                    d_forest = year_data['d_forest_ha'].iloc[0]
                    print(f"[S3_5 DEBUG] U.S. {year}: d_cropland={d_crop:,.0f} ha, d_grassland={d_grass:,.0f} ha, d_forest={d_forest:,.0f} ha")
            # Check original data in out.
            us_out = out[out['country'] == 'United States of America']
            for year in all_years:
                year_data = us_out[us_out['year'] == year]
                if not year_data.empty:
                    crop_ha = year_data['cropland_ha'].iloc[0]
                    new_crop_ha = year_data['new_cropland_ha'].iloc[0]
                    target_crop_ha = year_data['target_cropland_ha'].iloc[0] if 'target_cropland_ha' in year_data.columns else 0
                    print(f"[S3_5 DEBUG] U.S. {year}: 期初cropland={crop_ha:,.0f}, 期末new_cropland={new_crop_ha:,.0f}, target={target_crop_ha:,.0f}")

    # Build output DataFrames.
    if not out.empty:
        # Initial period area (period_start)
        period_start_cols = ['country', 'year', 'cropland_ha', 'forest_ha', 'grassland_ha']
        if has_m49:
            period_start_cols = ['M49_Country_Code'] + period_start_cols
        period_start = out[period_start_cols].copy()
        
        # Final period area (period_end)
        period_end_cols = ['country', 'year', 'new_cropland_ha', 'new_forest_ha', 'new_grassland_ha']
        if has_m49:
            period_end_cols = ['M49_Country_Code'] + period_end_cols
        period_end = out[period_end_cols].rename(
            columns={'new_cropland_ha': 'cropland_ha', 'new_forest_ha': 'forest_ha', 'new_grassland_ha': 'grassland_ha'})
        
        # Legacy compatibility: luc_area = period_end.
        luc_area = period_end.copy()
    else:
        empty_cols = ['country', 'year', 'cropland_ha', 'forest_ha', 'grassland_ha']
        period_start = pd.DataFrame(columns=empty_cols)
        period_end = pd.DataFrame(columns=empty_cols)
        luc_area = pd.DataFrame(columns=empty_cols)

    return {
        'luc_area': luc_area,           # Final period area for legacy compatibility
        'period_start': period_start,   # Initial period area
        'period_end': period_end,       # Final period area
        'deltas': deltas                # Period-to-period changes
    }

def get_luc_emis():
    """
    This is a placeholder function to help diagnose a circular import.
    """
    pass
