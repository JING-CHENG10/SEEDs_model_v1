# -*- coding: utf-8 -*-
"""
S4_1_results.py: Emissions aggregation and export module.
=====================================
Generate three summary levels strictly following dict_v3:
- Sheet1: Detailed M49_Country_Code, Region_label_new, Emis Process, Emis Item, Emis GHG, and Yxxxx columns.
- Sheet2: Process+GHG by M49_Country_Code, Region_label_new, Emis Process, Emis GHG, and Yxxxx.
- Sheet3: GHG by M49_Country_Code, Region_label_new, Emis GHG, and Yxxxx.

Support GCE (crops), GLE (livestock), GOS (soil), GFE (fisheries), and LUC (land-use change).

Cost outputs:
- cost_summary.csv: Country-commodity-process/strategy cost details.
- cost_summary_by_country_measure.csv: Country-measure cost summaries.
- cost_summary_by_global_measure.csv: Global-measure cost summaries.

Year handling:
- Historical years <=2020: summarize 2010-2020 only.
- Future years >2020: summarize configured model years.
"""

from __future__ import annotations
from typing import Dict, Any, Mapping, Optional, List, Tuple
import os
import tempfile
import builtins
import threading
import numpy as np
import pandas as pd
from pathlib import Path
from config_paths import get_results_base, get_src_base
from runtime_data_cache import read_excel_cached

# AR6 GWP100
GWP100_AR6 = {'CO2': 1.0, 'CH4': 27.2, 'N2O': 273.0}
KT_CO2E_TO_T_CO2E = 1000.0
ALWAYS_KEEP_ZERO_PROCESSES = {
    'Ag land abandonment_crop',
    'Ag land abandonment_pasture',
}

_DEBUG_SAMPLE_LOCK = threading.Lock()
_DEBUG_SAMPLE_STATE = {"seen": 0, "written": 0, "truncated": False}


def _get_debug_level() -> int:
    raw = os.environ.get("NZF_DEBUG_LEVEL", "0").strip()
    try:
        return max(0, int(float(raw)))
    except Exception:
        return 0


def _get_debug_sample_file() -> str:
    sample_file = os.environ.get("NZF_DEBUG_SAMPLE_FILE", "").strip()
    if sample_file:
        return sample_file
    sample_dir = os.environ.get("NZF_DEBUG_SAMPLE_DIR", "").strip()
    if sample_dir:
        os.makedirs(sample_dir, exist_ok=True)
        return os.path.join(sample_dir, "debug_sample.log")
    return os.path.join(tempfile.gettempdir(), "s4_1_debug_sample.log")


def _write_sample_line(text: str) -> None:
    max_lines = 600
    sample_every = 20
    with _DEBUG_SAMPLE_LOCK:
        _DEBUG_SAMPLE_STATE["seen"] += 1
        seen = _DEBUG_SAMPLE_STATE["seen"]
        should_write = (seen <= 80) or (seen % sample_every == 0)
        if not should_write:
            return
        if _DEBUG_SAMPLE_STATE["written"] >= max_lines:
            if not _DEBUG_SAMPLE_STATE["truncated"]:
                path = _get_debug_sample_file()
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write("[debug-sample] ... truncated ...\n")
                _DEBUG_SAMPLE_STATE["truncated"] = True
            return
        path = _get_debug_sample_file()
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text.rstrip() + "\n")
        _DEBUG_SAMPLE_STATE["written"] += 1


def print(*args, **kwargs):  # type: ignore[override]
    """Module-local print gate controlled by NZF_DEBUG_LEVEL."""
    text = " ".join(str(a) for a in args)
    level = _get_debug_level()
    msg_upper = text.upper()
    is_warning = ("[WARN" in msg_upper) or ("WARNING" in msg_upper) or ("[ERROR" in msg_upper) or ("ERROR" in msg_upper)
    if level >= 2 or is_warning:
        return builtins.print(*args, **kwargs)
    if level >= 1 and text:
        _write_sample_line(text)
    return None

def _norm_m49_code(code: Any) -> str:
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return ''
    s = str(code).strip()
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

def _get_column_series(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Return a Series for the named column even if pandas exposes it as a DataFrame because of duplicate labels.
    """
    if column not in df.columns:
        return pd.Series([], dtype=object)
    column_data = df[column]
    if isinstance(column_data, pd.DataFrame):
        if column_data.shape[1] == 0:
            return pd.Series([], dtype=object)
        column_data = column_data.iloc[:, 0]
    if isinstance(column_data, pd.Series):
        return column_data
    return pd.Series(column_data, index=df.index if hasattr(df, 'index') else None)


def _unique_column_values(df: pd.DataFrame, column: str) -> np.ndarray:
    """
    Safely get the unique values for a column, even when duplicate labels force pandas to return a DataFrame.
    """
    column_series = _get_column_series(df, column)
    return column_series.dropna().unique()


def filter_years_for_aggregation(all_years: List[int]) -> List[int]:
    """
    Select summary years according to historical/future period.
    
    Rules:
    - Historical years <=2020: retain 2010-2020 only.
    - Future years >2020: retain all configured future years.
    
    Args:
        all_years: List of all available years.
        
    Returns:
        List of years to summarize.
    """
    if not all_years:
        return []
    
    hist_years = [y for y in all_years if y <= 2020]
    future_years = [y for y in all_years if y > 2020]
    
    # Retain historical 2010-2020 only.
    hist_filtered = [y for y in hist_years if 2010 <= y <= 2020]
    
    # Retain all future years.
    return sorted(hist_filtered + future_years)


class EmissionsAggregator:
    """Generate standardized emissions summaries from individual module results."""
    
    def __init__(self, dict_v3_path: str):
        """
        Initialize the summarizer.
        
        Args:
            dict_v3_path: Path to dict_v3.xlsx.
        """
        self.dict_v3_path = dict_v3_path
        self._load_dict_v3()
        
    def _load_dict_v3(self):
        """Load mapping tables from dict_v3."""
        # Load the region table.
        self.region_df = read_excel_cached(self.dict_v3_path, sheet_name='region')
        
        # Select valid countries with Region_label_new != 'no'.
        valid_regions = self.region_df[self.region_df['Region_label_new'] != 'no'].copy()
        print(f"[INFO] dict_v3 中总共 {len(self.region_df)} 行，有效国家 {len(valid_regions)} 个")
        
        # Map standardized apostrophe-prefixed M49_Country_Code to Region_label_new.
        self.m49_to_region = {}
        for _, row in valid_regions.iterrows():
            m49 = _norm_m49_code(row['M49_Country_Code'])
            region = row['Region_label_new']
            if m49 and pd.notna(region):
                self.m49_to_region[m49] = region
        
        print(f"[INFO] 构建了 {len(self.m49_to_region)} 个国家的 Region 映射（仅有效国家）")
        
        # Load Emis_item.
        self.emis_item_df = read_excel_cached(self.dict_v3_path, sheet_name='Emis_item')
        
        # Build Process-to-GHG mappings, allowing several gases per process.
        self.process_ghg_map = {}
        for _, row in self.emis_item_df.iterrows():
            process = str(row['Process']).strip()
            ghg = str(row['GHG']).upper() if pd.notna(row['GHG']) else None
            
            if process not in self.process_ghg_map:
                self.process_ghg_map[process] = set()
            if ghg and ghg in ['CH4', 'N2O', 'CO2']:
                self.process_ghg_map[process].add(ghg)
        
        # Build Process-to-Item_Emis mappings.
        self.process_item_map = {}
        for _, row in self.emis_item_df.iterrows():
            process = str(row['Process']).strip()
            item_emis = row['Item_Emis']
            if pd.notna(item_emis):
                if process not in self.process_item_map:
                    self.process_item_map[process] = []
                if item_emis not in self.process_item_map[process]:
                    self.process_item_map[process].append(item_emis)
    
    def _normalize_emissions_data(self, 
                                  fao_results: Dict[str, Any],
                                  extra_emis: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Standardize emissions from all modules to one format.
        
        Returns:
            DataFrame with columns: M49_Country_Code, Item, Process, year, CH4_kt, N2O_kt, CO2_kt
        """
        all_records = []
        
        # Process GCE crop emissions.
        gce = fao_results.get('GCE', {})
        if isinstance(gce, dict):
            for process_key, df in gce.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    records = self._extract_emissions(df, f'GCE:{process_key}')
                    all_records.extend(records)
        
        # Process GLE livestock emissions.
        gle = fao_results.get('GLE', [])
        if isinstance(gle, list):
            for item in gle:
                if isinstance(item, pd.DataFrame) and not item.empty:
                    records = self._extract_emissions(item, 'GLE')
                    all_records.extend(records)
                elif isinstance(item, dict):
                    for process_key, df in item.items():
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            records = self._extract_emissions(df, f'GLE:{process_key}')
                            all_records.extend(records)
        
        # Process GOS soil emissions.
        gos = fao_results.get('GOS', [])
        if isinstance(gos, list):
            for item in gos:
                if isinstance(item, dict):
                    for process_key, df in item.items():
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            records = self._extract_emissions(df, f'GOS:{process_key}')
                            all_records.extend(records)
        
        # Process GFE fisheries emissions.
        gfe = fao_results.get('GFE', {})
        if isinstance(gfe, dict):
            for process_key, df in gfe.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    records = self._extract_emissions(df, f'GFE:{process_key}')
                    all_records.extend(records)
        
        # Process LUC land-use-change emissions.
        luc = fao_results.get('LUC')
        if isinstance(luc, pd.DataFrame) and not luc.empty:
            records = self._extract_emissions(luc, 'LUC')
            all_records.extend(records)
        
        # Process additional emissions.
        if extra_emis is not None and not extra_emis.empty:
            records = self._extract_emissions(extra_emis, None)
            all_records.extend(records)
        
        if not all_records:
            return pd.DataFrame(columns=['M49_Country_Code', 'Item', 'Process', 'year', 
                                        'CH4_kt', 'N2O_kt', 'CO2_kt'])

        df = pd.DataFrame(all_records)

        # Defensive de-duplication: exact duplicate rows (same country,item,process,year,ghg values)
        try:
            dup_subset = ['M49_Country_Code', 'Item', 'Process', 'year', 'CH4_kt', 'N2O_kt', 'CO2_kt']
            if set(dup_subset).issubset(df.columns):
                dup_mask = df.duplicated(subset=dup_subset, keep='first')
                n_dup = int(dup_mask.sum())
                if n_dup > 0:
                    # outputs disabled
                    if False:
                        try:
                            dbg_dir = Path(get_results_base()) / 'debug_outputs' / 'Results'
                            dbg_dir.mkdir(parents=True, exist_ok=True)
                            df[dup_mask].to_csv(dbg_dir / 'duplicate_emission_rows_raw.csv', index=False)
                        except Exception:
                            pass
                    print(f"[WARN] Detected {n_dup} exact duplicate emission rows; dropping duplicates before aggregation")
                    df = df[~dup_mask].copy()
        except Exception as e:
            print(f"[ERROR] De-duplication check failed: {e}")

        return df
    
    def _extract_emissions(self, df: pd.DataFrame, default_process: Optional[str]) -> List[Dict]:
        """
        Extract emissions from one DataFrame.
        
        Returns:
            List of dicts with keys: M49_Country_Code, Item, Process, year, CH4_kt, N2O_kt, CO2_kt
        """
        # Identify columns tolerantly across case and naming variants.
        col_mapping = {}
        for col in df.columns:
            col_lower = str(col).lower().strip()
            
            # Country codes
            if 'm49' in col_lower or 'country_code' in col_lower:
                col_mapping['m49'] = col
            elif 'country' in col_lower and 'code' not in col_lower:
                col_mapping['country'] = col
            
            # Item/Commodity
            if 'item' in col_lower or 'commodity' in col_lower:
                col_mapping['item'] = col
            
            # Process
            if 'process' in col_lower:
                col_mapping['process'] = col
            
            # Year
            if col_lower in ['year', 'time']:
                col_mapping['year'] = col
            
            # GHG emissions
            if 'ch4' in col_lower and 'kt' in col_lower:
                col_mapping['ch4'] = col
            elif 'ch4' in col_lower and '_t' not in col_lower:
                col_mapping['ch4'] = col
                
            if 'n2o' in col_lower and 'kt' in col_lower:
                col_mapping['n2o'] = col
            elif 'n2o' in col_lower and '_t' not in col_lower:
                col_mapping['n2o'] = col
                
            if 'co2' in col_lower and 'kt' in col_lower and 'eq' not in col_lower:
                col_mapping['co2'] = col
            elif 'co2' in col_lower and '_t' not in col_lower and 'eq' not in col_lower:
                col_mapping['co2'] = col

        year_col = col_mapping.get('year')
        if year_col is None:
            return []

        year_series = pd.to_numeric(df[year_col], errors='coerce')
        valid_mask = year_series.notna()
        if not valid_mask.any():
            return []

        base_df = df.loc[valid_mask].copy()
        years = year_series.loc[valid_mask].astype(int)

        if 'm49' in col_mapping:
            m49_series = base_df[col_mapping['m49']].map(_norm_m49_code)
        elif 'country' in col_mapping:
            m49_series = base_df[col_mapping['country']].map(_norm_m49_code)
        else:
            m49_series = pd.Series([None] * len(base_df), index=base_df.index, dtype=object)

        item_col = col_mapping.get('item')
        if item_col is not None:
            item_series = base_df[item_col].where(base_df[item_col].notna(), 'ALL').astype(str)
        else:
            item_series = pd.Series('ALL', index=base_df.index, dtype=object)

        process_fallback = default_process or 'Unknown'
        process_col = col_mapping.get('process')
        if process_col is not None:
            process_series = base_df[process_col].where(base_df[process_col].notna(), process_fallback).astype(str)
        else:
            process_series = pd.Series(process_fallback, index=base_df.index, dtype=object)
        process_series = process_series.map(self._clean_process_name)

        def _numeric_series(column_name: Optional[str]) -> pd.Series:
            if column_name is None:
                return pd.Series(0.0, index=base_df.index)
            return pd.to_numeric(base_df[column_name], errors='coerce').fillna(0.0)

        extracted = pd.DataFrame({
            'M49_Country_Code': m49_series,
            'Item': item_series,
            'Process': process_series,
            'year': years,
            'CH4_kt': _numeric_series(col_mapping.get('ch4')),
            'N2O_kt': _numeric_series(col_mapping.get('n2o')),
            'CO2_kt': _numeric_series(col_mapping.get('co2')),
        })
        nonzero_mask = extracted[['CH4_kt', 'N2O_kt', 'CO2_kt']].ne(0).any(axis=1)
        if not nonzero_mask.any():
            return []
        return extracted.loc[nonzero_mask].to_dict('records')
    
    def _clean_process_name(self, process: str) -> str:
        """Remove module prefixes from Process and match standard dict_v3 names."""
        cache = getattr(self, '_clean_process_cache', None)
        if cache is None:
            cache = {}
            self._clean_process_cache = cache
        if process in cache:
            return cache[process]

        original_process = process
        # Remove prefixes such as GCE: and GLE:.
        if ':' in process:
            process = process.split(':', 1)[1]
        
        process = process.strip()
        
        # Try standard dict_v3 Process names.
        for standard_process in self.emis_item_df['Process'].unique():
            if pd.notna(standard_process):
                standard = str(standard_process).strip()
                # Fuzzy matching
                if standard.lower() in process.lower() or process.lower() in standard.lower():
                    cache[original_process] = standard
                    return standard
        
        cache[original_process] = process
        return process
    
    def _safe_float(self, value, default=0.0) -> float:
        """Convert safely to float."""
        try:
            if pd.isna(value):
                return default
            return float(value)
        except:
            return default

    def _prepare_summary_emissions(self, emissions_df: pd.DataFrame) -> pd.DataFrame:
        """Standardize M49/Region while preserving row order for stable aggregation."""
        prepared = emissions_df.copy()
        prepared['M49_Country_Code'] = prepared['M49_Country_Code'].map(_norm_m49_code)
        prepared['Region_label_new'] = prepared['M49_Country_Code'].map(self.m49_to_region)
        prepared['_row_order'] = np.arange(len(prepared), dtype=np.int64)

        unmatched = prepared[prepared['Region_label_new'].isna()]
        if not unmatched.empty:
            unmatch_countries = unmatched['M49_Country_Code'].dropna().unique()
            print(f"[WARN] {len(unmatch_countries)} 个国家未能匹配到Region: {list(unmatch_countries)[:5]}...")

        # Explicitly preserve existing groupby behavior that drops missing Region records.
        return prepared.dropna(subset=['Region_label_new']).copy()

    def _melt_emissions_long(self, prepared_df: pd.DataFrame) -> pd.DataFrame:
        """Melt CH4/N2O/CO2 columns and filter process-allowed gases using dict_v3."""
        ghg_cols = [col for col in ['CH4_kt', 'N2O_kt', 'CO2_kt'] if col in prepared_df.columns]
        if not ghg_cols:
            return pd.DataFrame(
                columns=['M49_Country_Code', 'Region_label_new', 'Process', 'Item', 'year', 'Emis GHG', 'value', '_row_order']
            )

        long_df = prepared_df.melt(
            id_vars=['M49_Country_Code', 'Region_label_new', 'Process', 'Item', 'year', '_row_order'],
            value_vars=ghg_cols,
            var_name='ghg_col',
            value_name='value',
        )
        long_df['Emis GHG'] = long_df['ghg_col'].map({
            'CH4_kt': 'CH4',
            'N2O_kt': 'N2O',
            'CO2_kt': 'CO2',
        })
        long_df['value'] = pd.to_numeric(long_df['value'], errors='coerce').fillna(0.0)

        if self.process_ghg_map:
            allowed_pairs = pd.DataFrame(
                [(process, ghg) for process, ghgs in self.process_ghg_map.items() for ghg in ghgs],
                columns=['Process', 'Emis GHG'],
            )
            if not allowed_pairs.empty:
                allowed_pairs['_allowed_ghg'] = True
                long_df = long_df.merge(allowed_pairs, on=['Process', 'Emis GHG'], how='left')
                mapped_processes = set(self.process_ghg_map)
                long_df = long_df[
                    (~long_df['Process'].isin(mapped_processes)) | long_df['_allowed_ghg'].eq(True)
                ].copy()
                long_df = long_df.drop(columns=['_allowed_ghg'])

        return long_df.drop(columns=['ghg_col'])

    def _finalize_summary_output(
        self,
        df: pd.DataFrame,
        id_cols: List[str],
        years: List[int],
        sort_cols: List[str],
    ) -> pd.DataFrame:
        """Complete year columns, calculate CO2eq, and order export columns."""
        if df.empty:
            return pd.DataFrame()

        year_cols = [f'Y{year}' for year in years]
        df = df.copy()
        for year_col in year_cols:
            if year_col not in df.columns:
                df[year_col] = 0.0
        df[year_cols] = df[year_cols].fillna(0.0)

        for ghg, gwp in GWP100_AR6.items():
            mask = df['Emis GHG'].eq(ghg)
            if not mask.any():
                continue
            co2eq_cols = [f'Y{year}_CO2eq' for year in years]
            df.loc[mask, co2eq_cols] = df.loc[mask, year_cols].to_numpy(dtype=float) * gwp

        ordered_cols = ['M49_Country_Code'] + [col for col in id_cols if col != 'M49_Country_Code']
        ordered_cols.extend(year_cols)
        ordered_cols.extend(f'Y{year}_CO2eq' for year in years)
        df = df[ordered_cols]
        df['M49_Country_Code'] = df['M49_Country_Code'].map(_norm_m49_code)
        return df.sort_values(sort_cols).reset_index(drop=True)

    def generate_detailed_summary(self, emissions_df: pd.DataFrame, years: List[int]) -> pd.DataFrame:
        """
        Generate Sheet1, the most detailed summary.
        
        Columns: M49_Country_Code, Region_label_new, Emis Process, Emis Item, Emis GHG, Y2000, Y2001, ...
        
        Args:
            emissions_df: Standardized emissions data.
            years: Output years.
        
        Returns:
            Detailed summary DataFrame.
        """
        if emissions_df.empty:
            return pd.DataFrame()

        prepared = self._prepare_summary_emissions(emissions_df)
        long_df = self._melt_emissions_long(prepared)
        if long_df.empty:
            print(f"[WARN] Sheet1 没有生成任何记录")
            return pd.DataFrame()

        detail_keys = ['M49_Country_Code', 'Region_label_new', 'Process', 'Item', 'Emis GHG', 'year']
        long_df = long_df.sort_values('_row_order')
        long_df = long_df.drop_duplicates(subset=detail_keys, keep='last')

        group_keys = ['M49_Country_Code', 'Region_label_new', 'Process', 'Item', 'Emis GHG']
        has_data = long_df.groupby(group_keys, sort=False)['value'].transform(lambda s: (s != 0).any())
        keep_zero_process = long_df['Process'].astype(str).isin(ALWAYS_KEEP_ZERO_PROCESSES)
        has_data = has_data | keep_zero_process
        long_df = long_df[has_data].copy()
        if long_df.empty:
            print(f"[WARN] Sheet1 没有生成任何记录")
            return pd.DataFrame()

        df = (
            long_df
            .assign(year_col=long_df['year'].map(lambda year: f'Y{int(year)}'))
            .pivot(
                index=group_keys,
                columns='year_col',
                values='value',
            )
            .reset_index()
        )
        df.columns.name = None
        df = df.rename(columns={'Process': 'Emis Process', 'Item': 'Emis Item'})
        df = self._finalize_summary_output(
            df=df,
            id_cols=['M49_Country_Code', 'Region_label_new', 'Emis Process', 'Emis Item', 'Emis GHG'],
            years=years,
            sort_cols=['M49_Country_Code', 'Emis Process', 'Emis Item', 'Emis GHG'],
        )

        # Statistics
        n_countries = df['M49_Country_Code'].nunique()
        n_processes = df['Emis Process'].nunique()
        n_items = df['Emis Item'].nunique()
        print(f"[INFO] Sheet1 统计: {len(df)} 行, {n_countries} 个国家, {n_processes} 个 Process, {n_items} 个 Item")
        
        # Remove dairy/non-dairy merging.
        # Preserve the original dairy/non-dairy split defined by dict_v3.
        print(f"[INFO] 保留dairy/non-dairy原始分拆形式（按dict_v3定义）")
        return df
    
    def generate_process_ghg_summary(self, emissions_df: pd.DataFrame, years: List[int]) -> pd.DataFrame:
        """
        Generate Sheet2 at Process+GHG level.
        
        Columns: M49_Country_Code, Region_label_new, Emis Process, Emis GHG, Y2000, Y2001, ...
        
        Aggregate away the Item dimension from Sheet1.
        
         Exclude dairy/non-dairy entries to avoid double counting during aggregation.
        """
        if emissions_df.empty:
            return pd.DataFrame()

        prepared = self._prepare_summary_emissions(emissions_df)
        long_df = self._melt_emissions_long(prepared)
        if long_df.empty:
            print(f"[WARN] Sheet2 没有生成任何记录")
            return pd.DataFrame()

        grouped = (
            long_df
            .groupby(['M49_Country_Code', 'Region_label_new', 'Process', 'Emis GHG', 'year'], as_index=False, sort=False)['value']
            .sum()
        )
        grouped = grouped.groupby(
            ['M49_Country_Code', 'Region_label_new', 'Process', 'Emis GHG'],
            sort=False,
        ).filter(
            lambda frame: frame['Process'].astype(str).isin(ALWAYS_KEEP_ZERO_PROCESSES).any()
            or (frame['value'] != 0).any()
        )
        if grouped.empty:
            print(f"[WARN] Sheet2 没有生成任何记录")
            return pd.DataFrame()

        df = (
            grouped
            .assign(year_col=grouped['year'].map(lambda year: f'Y{int(year)}'))
            .pivot(
                index=['M49_Country_Code', 'Region_label_new', 'Process', 'Emis GHG'],
                columns='year_col',
                values='value',
            )
            .reset_index()
        )
        df.columns.name = None
        df = df.rename(columns={'Process': 'Emis Process'})
        df = self._finalize_summary_output(
            df=df,
            id_cols=['M49_Country_Code', 'Region_label_new', 'Emis Process', 'Emis GHG'],
            years=years,
            sort_cols=['M49_Country_Code', 'Emis Process', 'Emis GHG'],
        )

        n_countries = df['M49_Country_Code'].nunique()
        n_processes = df['Emis Process'].nunique()
        print(f"[INFO] Sheet2 统计: {len(df)} 行, {n_countries} 个国家, {n_processes} 个 Process")
        return df
    
    def generate_ghg_summary(self, emissions_df: pd.DataFrame, years: List[int]) -> pd.DataFrame:
        """
        Generate Sheet3 at GHG level.
        
        Columns: M49_Country_Code, Region_label_new, Emis GHG, Y2000, Y2001, ...
        
        Aggregate away Process from Sheet2.
        
         Do not exclude dairy/non-dairy; allow normal aggregation and handle later.
        """
        if emissions_df.empty:
            return pd.DataFrame()

        prepared = self._prepare_summary_emissions(emissions_df)
        long_df = self._melt_emissions_long(prepared)
        if long_df.empty:
            return pd.DataFrame()

        grouped = (
            long_df
            .groupby(['M49_Country_Code', 'Region_label_new', 'Emis GHG', 'year'], as_index=False, sort=False)['value']
            .sum()
        )
        grouped = grouped.groupby(
            ['M49_Country_Code', 'Region_label_new', 'Emis GHG'],
            sort=False,
        ).filter(lambda frame: (frame['value'] != 0).any())
        if grouped.empty:
            return pd.DataFrame()

        df = (
            grouped
            .assign(year_col=grouped['year'].map(lambda year: f'Y{int(year)}'))
            .pivot(
                index=['M49_Country_Code', 'Region_label_new', 'Emis GHG'],
                columns='year_col',
                values='value',
            )
            .reset_index()
        )
        df.columns.name = None
        return self._finalize_summary_output(
            df=df,
            id_cols=['M49_Country_Code', 'Region_label_new', 'Emis GHG'],
            years=years,
            sort_cols=['M49_Country_Code', 'Emis GHG'],
        )
    
    def aggregate_and_export(self, 
                            fao_results: Dict[str, Any],
                            output_path: str,
                            years: List[int],
                            extra_emis: Optional[pd.DataFrame] = None):
        """
        Complete aggregation and export workflow.
        
        Args:
            fao_results: FAO module result dictionary.
            output_path: Output Excel path.
            years: Historical and future output years.
            extra_emis: Optional additional emissions data.
            
        Notes:
            - Historical years <=2020: automatically select 2010-2020.
            - Future years >2020: use all configured model years.
        """
        print("开始排放数据汇总...")
        
        # 0. Filter years: historical 2010-2020 and all future years.
        filtered_years = filter_years_for_aggregation(years)
        print(f"  - 年份过滤: {len(years)} ?? {len(filtered_years)} 年")
        print(f"    原始年份范围: {min(years) if years else 'N/A'} - {max(years) if years else 'N/A'}")
        print(f"    筛选后范围: {min(filtered_years) if filtered_years else 'N/A'} - {max(filtered_years) if filtered_years else 'N/A'}")
        
        if not filtered_years:
            print("  WARNING: no years remain after filtering")
            return
        
        # 1. Standardize emissions.
        print("  - 标准化排放数据...")
        emissions_df = self._normalize_emissions_data(fao_results, extra_emis)
        
        if emissions_df.empty:
            print("  WARNING: no emissions data to aggregate")
            return
        
        print(f"  [诊断] 标准化后: {len(emissions_df)} 行, {emissions_df['M49_Country_Code'].nunique()} 个国家")
        if 'year' in emissions_df.columns:
            print(f"       年份范围: {emissions_df['year'].min()} - {emissions_df['year'].max()}")
        
        # 2. Filter by year.
        print("  - 按年份筛选排放数据...")
        emissions_df_filtered = emissions_df[emissions_df['year'].isin(filtered_years)].copy()
        
        print(f"  [诊断] 年份筛选后: {len(emissions_df_filtered)} 行, {emissions_df_filtered['M49_Country_Code'].nunique()} 个国家")
        print(f"  - 共收集 {len(emissions_df_filtered)} 条排放记录")
        
        # 3. Generate three summary levels.
        print("  - 生成详细汇总表（Sheet1）...")
        sheet1 = self.generate_detailed_summary(emissions_df_filtered, filtered_years)
        
        print("  - 生成Process+GHG汇总表（Sheet2）...")
        sheet2 = self.generate_process_ghg_summary(emissions_df_filtered, filtered_years)
        
        print("  - 生成GHG汇总表（Sheet3）...")
        sheet3 = self.generate_ghg_summary(emissions_df_filtered, filtered_years)
        
        # 4. Export Excel.
        print(f"  - 导出到文件: {output_path}")
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            if not sheet1.empty:
                sheet1.to_excel(writer, sheet_name='Detail_Summary', index=False)
            if not sheet2.empty:
                sheet2.to_excel(writer, sheet_name='Process_GHG_Summary', index=False)
            if not sheet3.empty:
                sheet3.to_excel(writer, sheet_name='GHG_Summary', index=False)
        
        print(f" 排放汇总完成！")
        print(f"  - 汇总年份数: {len(filtered_years)} ({min(filtered_years) if filtered_years else 'N/A'}-{max(filtered_years) if filtered_years else 'N/A'})")
        print(f"  - Sheet1 (详细): {len(sheet1)} 行")
        print(f"  - Sheet2 (Process+GHG): {len(sheet2)} 行")
        print(f"  - Sheet3 (GHG): {len(sheet3)} 行")


# Backward-compatible functions preserving original interfaces
def summarize_emissions_from_detail(emission_detail_df: pd.DataFrame,
                                    process_meta_map: Optional[dict]=None,
                                    allowed_years: Optional[List[int]]=None,
                                    dict_v3_path: Optional[str]=None,
                                    production_df: Optional[pd.DataFrame]=None) -> Dict[str, pd.DataFrame]:
    """
    Generate multiple summary levels from detailed emissions.
    WARNING: rely on Process and Item lists defined in dict_v3 Emis_item
    
    Parameters
    ----------
    emission_detail_df : pd.DataFrame
        Detailed DataFrame with M49_Country_Code, Process, Item, GHG, year, value, etc.
    process_meta_map : dict, optional
        Process metadata mappings for classification.
    allowed_years : List[int], optional
        Allowed years for filtering.
    dict_v3_path : str, optional
        Path to dict_v3.xlsx for Region mappings and Process/Item definitions.
    production_df : pd.DataFrame, optional
        Production data including stock, used to split merged historical livestock emissions.
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        Summaries at multiple levels.
    """

    
    print(f"[DEBUG] summarize_emissions_from_detail called:")
    print(f"  - emission_detail_df length: {len(emission_detail_df) if isinstance(emission_detail_df, pd.DataFrame) else 'Not a DataFrame'}")
    print(f"  - allowed_years: {allowed_years}")
    print(f"  - allowed_years type: {type(allowed_years)}")
    debug_log = os.path.join(tempfile.gettempdir(), "s4_1_debug.log")
    with open(debug_log, "a", encoding="utf-8") as f:
        # Write a single debug line to the temp log
        f.write(f"summarize_emissions_from_detail called: allowed_years={allowed_years}\n")
    
    
    if not isinstance(emission_detail_df, pd.DataFrame) or emission_detail_df.empty:
        return {
            'by_ctry_proc_comm': pd.DataFrame(),
            'by_ctry_proc': pd.DataFrame(),
            'by_ctry': pd.DataFrame(),
            'by_year': pd.DataFrame(),
            'long': emission_detail_df.copy() if isinstance(emission_detail_df, pd.DataFrame) else pd.DataFrame()
        }
    
    df = emission_detail_df.copy()
    
    # Normalize years to integers to avoid filtering mismatches (e.g., strings vs ints)
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df = df.dropna(subset=['year'])
        df['year'] = df['year'].astype(int)

    allowed_years_set = {int(y) for y in allowed_years} if allowed_years else None

    # WARNING: load allowed Process/Item lists from dict_v3
    allowed_processes = set()
    allowed_items = set()
    if dict_v3_path:
        try:
            emis_item_df = read_excel_cached(dict_v3_path, sheet_name='Emis_item')
            allowed_processes = set(emis_item_df['Process'].dropna().unique())
            allowed_items = set(emis_item_df['Item_Emis'].dropna().unique())
            
            # Do not add merged names; split historical emissions into dairy/non-dairy using stock shares.
            # Summarize strictly using dict_v3's dairy/non-dairy definitions.
            
            print(f"[INFO] 从dict_v3加载: {len(allowed_processes)} 个Process, {len(allowed_items)} 个Item_Emis (含合并动物名称)")
            print(f"[INFO] 允许的Process: {sorted(allowed_processes)}")
        except Exception as e:
            print(f"WARNING: failed to load dict_v3 Emis_item: {e}")
    
    # WARNING: filter to Processes defined in dict_v3 that actually appear in data
    if allowed_processes and 'Process' in df.columns:
        process_series = _get_column_series(df, 'Process')
        process_values = _unique_column_values(df, 'Process')
        print(f"[DEBUG] Process过滤前: {len(df)} 行")
        print(f"  - 数据中的Process: {process_values}")
        print(f"  - dict_v3中的Process: {sorted(list(allowed_processes))}")
        
        # Check whether any data match dict_v3 Processes.
        process_mask = process_series.isin(allowed_processes)
        print(f"  - 匹配的Process行数: {process_mask.sum()}/{len(df)}")
        
        if process_mask.any():
            unmatched = df[~process_mask]
            matched = df[process_mask]
            if len(unmatched):
                print(f"[INFO] 发现 {len(unmatched)} 行 Process 未在 dict_v3 中定义，保留并继续处理")
                print(f"  - 非dict_v3 Process 样例: {_unique_column_values(unmatched, 'Process')[:5]}")
                df = pd.concat([matched, unmatched], ignore_index=True)
            else:
                rows_before = len(df)
                df = matched.copy()
                rows_after = len(df)
                if rows_before > rows_after:
                    dropped_processes = set(_unique_column_values(emission_detail_df, 'Process')) - allowed_processes
                    print(f"[INFO] 过滤掉 {rows_before - rows_after} 行非dict_v3定义的Process: {dropped_processes}")
        else:
            # No matches imply Process names are absent from dict_v3.
            # In that case, retain all data without filtering.
            print(f"[INFO] 数据中的Process名称与dict_v3不匹配，不进行Process过滤")
            print(f"  - 数据中的Process: {process_values}")
            print(f"  - dict_v3中的Process: {sorted(list(allowed_processes))[:5]}...")
        
        print(f"[DEBUG] Process过滤后: {len(df)} 行")
    
    # WARNING: filter to Items defined in dict_v3 that actually appear in data
    # Filter true emissions Items only, skipping GLE/GCE commodity names.
    # Check _is_commodity_based, set by summarize_emissions.
    has_commodity_flag = '_is_commodity_based' in df.columns
    if has_commodity_flag:
        # Separate commodity-name rows from emissions-Item rows.
        commodity_rows = df[df['_is_commodity_based'] == True]
        emission_rows = df[df['_is_commodity_based'] == False]
        print(f"\n[DEBUG] 检测到_is_commodity_based标记:")
        print(f"  - Commodity-based行（商品名，不过滤）: {len(commodity_rows)}")
        print(f"  - Emission-based行（排放Item，需过滤）: {len(emission_rows)}")
        df_to_filter = emission_rows
        df_to_keep = commodity_rows
    else:
        # Without the flag, treat every row as an emissions Item.
        df_to_filter = df.copy()
        df_to_keep = pd.DataFrame()
    
    if allowed_items and 'Item' in df_to_filter.columns and not df_to_filter.empty:
        item_series = _get_column_series(df_to_filter, 'Item')
        item_values = _unique_column_values(df_to_filter, 'Item')
        print(f"[DEBUG] Item过滤前（仅emission-based行）: {len(df_to_filter)} 行")
        print(f"  - 数据中的Item: {item_values}")
        print(f"  - dict_v3中的Item_Emis: {sorted(list(allowed_items))[:10]}...")
        
        # Check whether any Items match dict_v3.
        item_mask = item_series.isin(allowed_items)
        print(f"  - 匹配的Item行数: {item_mask.sum()}/{len(df_to_filter)}")
        
        if item_mask.any():
            # Strict filtering: retain only dict_v3 Items.
            unmatched_items = df_to_filter[~item_mask]
            if len(unmatched_items):
                dropped_item_names = _unique_column_values(unmatched_items, 'Item')
                print(f"[INFO] 过滤掉 {len(unmatched_items)} 行非dict_v3定义的排放Item")
                print(f"  - 被过滤的排放Item: {dropped_item_names[:10]}")
            df_to_filter = df_to_filter[item_mask].copy()
        else:
            # No matches imply Item names are absent from dict_v3.
            # In that unexpected case, retain all data without filtering.
            print(f"[WARNING] 数据中的Item名称与dict_v3完全不匹配，保留所有数据")
            print(f"  - 数据中的Item: {item_values[:10]}")
            print(f"  - dict_v3中的Item: {sorted(list(allowed_items))[:5]}...")
        
        # Combine filtered emissions rows with unfiltered commodity rows.
        if not df_to_keep.empty:
            df = pd.concat([df_to_filter, df_to_keep], ignore_index=True)
            print(f"[DEBUG] Item过滤后: {len(df_to_filter)} 排放行 + {len(df_to_keep)} 商品行 = {len(df)} 总行数")
        else:
            df = df_to_filter
            print(f"[DEBUG] Item过滤后: {len(df)} 行")
    elif has_commodity_flag and not df_to_keep.empty:
        # No Item filtering occurred, but commodity rows still need combining.
        df = pd.concat([df_to_filter, df_to_keep], ignore_index=True)
        print(f"\n[DEBUG] 跳过Item过滤（没有allowed_items），保留所有数据: {len(df)} 行")
    
    if df.empty:
        print("WARNING: result after filtering is empty")
        return {
            'by_ctry_proc_comm': pd.DataFrame(),
            'by_ctry_proc': pd.DataFrame(),
            'by_ctry': pd.DataFrame(),
            'by_year': pd.DataFrame(),
            'long': pd.DataFrame()
        }
    
    # Ensure M49_Country_Code exists.
    if 'M49_Country_Code' not in df.columns:
        if 'M49' in df.columns:
            df = df.rename(columns={'M49': 'M49_Country_Code'})
        else:
            print("WARNING: row missing M49_Country_Code; cannot identify country")
            return {
                'by_ctry_proc_comm': pd.DataFrame(),
                'by_ctry_proc': pd.DataFrame(),
                'by_ctry': pd.DataFrame(),
                'by_year': pd.DataFrame(),
                'long': df
            }
    
    # Load dict_v3 M49-to-Region_label_new mappings.
    region_mapping = {}
    valid_countries_only = set()  # M49 codes for valid countries only
    if dict_v3_path:
        try:
            region_df = read_excel_cached(dict_v3_path, sheet_name='region')
            
            # Normalize M49 and load valid countries only.
            for _, row in region_df.iterrows():
                m49_normalized = _norm_m49_code(row['M49_Country_Code'])
                if not m49_normalized:
                    continue
                region = row['Region_label_new']
                
                # Load countries with Region_label_new != 'no'.
                if pd.notna(region) and region != 'no':
                    region_mapping[m49_normalized] = region
                    valid_countries_only.add(m49_normalized)
            
            print(f"[DEBUG] 从dict_v3加载了 {len(region_mapping)} 个有效国家的region映射（已规范化M49代码）")
        except Exception as e:
            print(f"WARNING: failed to map dict_v3 regions: {e}")
    
    # Inspect input structure.
    print(f"\n[DEBUG] summarize_emissions_from_detail 输入数据:")
    print(f"  - 形状: {df.shape}")
    print(f"  - 列名: {list(df.columns)}")
    if 'year' in df.columns:
        years_in_data = sorted(df['year'].unique())
        print(f"  - 数据中的年份: {years_in_data} (共{len(years_in_data)}个)")
        print(f"  - 年份范围: {df['year'].min()}-{df['year'].max()}")
    print(f"  - 总行数: {len(df)}")
    if 'value' in df.columns:
        print(f"  - value非零行: {(df['value'] > 0).sum()}, 总计: {df['value'].sum():.2f}")
    if 'M49_Country_Code' in df.columns:
        m49_null_count = df['M49_Country_Code'].isna().sum()
        print(f"  - M49_Country_Code空值: {m49_null_count}/{len(df)}")
    if 'Item' in df.columns:
        item_null_count = df['Item'].isna().sum()
        print(f"  - Item空值: {item_null_count}/{len(df)}")
    
    # 1. Ensure integer years, then filter allowed_years.
    year_cols = [col for col in df.columns if col.lower() in ['year', 'yr']]
    for col in year_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df[df[col].notna()]  # Drop unparseable years.
        df[col] = df[col].astype(int)
    if allowed_years_set:
        if year_cols:
            year_col = year_cols[0]
            years_in_data = sorted(df[year_col].unique())
            removed_years = [y for y in years_in_data if y not in allowed_years_set]
            before_len = len(df)
            df = df[df[year_col].isin(allowed_years_set)].reset_index(drop=True)
            after_len = len(df)
            print(f"[DEBUG] Year filter: {before_len} -> {after_len} rows (allowed_years={sorted(allowed_years_set)})")
            if removed_years:
                print(f"[DEBUG] removed years: {removed_years}")
        else:
            print(f"[DEBUG] year column not found; skip year filtering")
    
    # 1.4 Disabled: historical dairy/non-dairy splitting, already done in the new input.
    # Emissions_livestock_dairy_split.csv now performs the split at source.
    # Buffalo/Camel/Sheep/Goats dairy/non-dairy emissions are already separate.
    if False and 'Item' in df.columns and 'year' in df.columns and 'Process' in df.columns and production_df is not None:
        print(f"\n[INFO] 开始拆分历史阶段的merged livestock排放...")
        
        # Merged animal names in historical Items that need splitting
        # Include only individual animals defined in dict_v3: Buffalo, Camels, Goats, Sheep.
        # Combined names such as Sheep and Goats are not defined and should be filtered at source.
        merged_items_to_split = {
            'Buffalo', 'Buffaloes',  # Buffaloes is the plural of Buffalo.
            'Camels', 
            'Goats', 
            'Sheep'
            # Exclude Sheep and Goats because dict_v3 does not define it.
        }
        
        # Only livestock processes need splitting.
        livestock_processes = {
            'Enteric fermentation',
            'Manure management',
            'Manure applied to soils',
            'Manure left on pasture'
        }
        
        # Split merged livestock data in historical years <=2020 only.
        historical_threshold = 2020
        merged_mask = (
            df['Item'].isin(merged_items_to_split) &
            (df['year'] <= historical_threshold) &
            df['Process'].isin(livestock_processes)
        )
        merged_rows = df[merged_mask]
        non_merged_rows = df[~merged_mask]
        
        if len(merged_rows) > 0:
            print(f"  - 发现 {len(merged_rows)} 行merged livestock数据需要拆分")
            print(f"  - 涉及Item: {merged_rows['Item'].unique()}")
            print(f"  - 涉及年份: {sorted(merged_rows['year'].unique())}")
            
            # Get stocks from production_df.
            # production_df: M49_Country_Code, Item, Item_Emis, year, stock, ...
            if 'stock' in production_df.columns and 'Item_Emis' in production_df.columns:
                # Do not handle undefined Items such as Sheep and Goats.
                # These should have been removed during loading; their presence indicates a source-data issue.
                # Standardize only individual animal names defined in dict_v3.
                item_mapping = {
                    'Buffalo': 'Buffalo',
                    'Buffaloes': 'Buffalo',  # Normalize to Buffalo.
                    'Camels': 'Camels',
                    'Goats': 'Goats',
                    'Sheep': 'Sheep'
                    # Exclude Sheep and Goats; it is undefined in dict_v3 and should be filtered.
                }
                
                split_rows = []
                skipped_invalid_items = []
                
                for idx, row in merged_rows.iterrows():
                    m49 = row['M49_Country_Code']
                    item = row['Item']
                    year = row['year']
                    value = row['value']
                    
                    # Check for a valid individual animal name.
                    if item not in item_mapping:
                        skipped_invalid_items.append(item)
                        continue  # Skip unmapped entries such as Sheep and Goats.
                    
                    # Standardize Item names.
                    std_item = item_mapping[item]
                    
                    # Find dairy/non-dairy stocks for this country-year-animal.
                    stock_data = production_df[
                        (production_df['M49_Country_Code'] == m49) &
                        (production_df['Item'] == std_item) &
                        (production_df['year'] == year)
                    ]
                    
                    # Retrieve dairy and non-dairy stocks separately.
                    dairy_stock_rows = stock_data[stock_data['Item_Emis'] == f'{std_item}, dairy']
                    nondairy_stock_rows = stock_data[stock_data['Item_Emis'] == f'{std_item}, non-dairy']
                    
                    dairy_stock = dairy_stock_rows['stock'].sum() if not dairy_stock_rows.empty and 'stock' in dairy_stock_rows.columns else 0
                    nondairy_stock = nondairy_stock_rows['stock'].sum() if not nondairy_stock_rows.empty and 'stock' in nondairy_stock_rows.columns else 0
                    total_stock = dairy_stock + nondairy_stock
                    
                    if total_stock > 0:
                        # Split emissions by stock shares.
                        dairy_ratio = dairy_stock / total_stock
                        nondairy_ratio = nondairy_stock / total_stock
                        
                        # Create the dairy row.
                        if dairy_stock > 0:
                            dairy_row = row.copy()
                            dairy_row['Item'] = f'{std_item}, dairy'
                            dairy_row['value'] = value * dairy_ratio
                            split_rows.append(dairy_row)
                        
                        # Create the non-dairy row.
                        if nondairy_stock > 0:
                            nondairy_row = row.copy()
                            nondairy_row['Item'] = f'{std_item}, non-dairy'
                            nondairy_row['value'] = value * nondairy_ratio
                            split_rows.append(nondairy_row)
                    else:
                        # Without stock data, allocate all emissions to non-dairy meat animals.
                        nondairy_row = row.copy()
                        nondairy_row['Item'] = f'{std_item}, non-dairy'
                        split_rows.append(nondairy_row)
                
                # Report skipped invalid Items.
                if skipped_invalid_items:
                    unique_skipped = set(skipped_invalid_items)
                    print(f"   跳过 {len(skipped_invalid_items)} 行不在dict_v3中的项: {unique_skipped}")
                    print(f"     这些项不应该出现在历史数据中，建议检查数据源过滤逻辑")
                
                if split_rows:
                    split_df = pd.DataFrame(split_rows)
                    # Remove original merged Items and retain dairy/non-dairy splits only.
                    df = pd.concat([non_merged_rows, split_df], ignore_index=True)
                    print(f"   拆分完成: {len(merged_rows) - len(skipped_invalid_items)} 行有效merged数据 ?? {len(split_df)} 行dairy/non-dairy数据")
                    print(f"   原始merged项已删除，仅保留dairy/non-dairy拆分结果")
                else:
                    # If splitting failed, remove merged Items to avoid double counting.
                    df = non_merged_rows.copy()
                    print(f"   未能拆分（缺少stock数据），已删除原始merged项以避免重复")
            else:
                # If required columns are absent, remove merged Items to avoid duplication with future dairy/non-dairy data.
                df = non_merged_rows.copy()
                print(f"   production_df缺少stock或Item_Emis列，已删除merged项以避免重复")
        else:
            print(f"  - 未发现需要拆分的merged livestock数据")
    
    # 1.4.1 Apply strict filtering only to merged livestock Items.
    # Do not globally filter all Items against dict_v3, which would also remove commodity-level LUC details,
    # hiding LUC Item breakdowns in By_Country_Process_Item.
    if 'Item' in df.columns and allowed_items:
        strict_livestock_processes = {
            'Enteric fermentation',
            'Manure management',
            'Manure applied to soils',
            'Manure left on pasture'
        }
        strict_filter_mask = pd.Series(False, index=df.index)
        if 'module' in df.columns:
            module_series = _get_column_series(df, 'module').astype(str)
            strict_filter_mask = strict_filter_mask | module_series.eq('GLE')
        if 'Process' in df.columns:
            process_series = _get_column_series(df, 'Process').astype(str)
            strict_filter_mask = strict_filter_mask | process_series.isin(strict_livestock_processes)
        if strict_filter_mask.any():
            df_strict = df[strict_filter_mask].copy()
            df_other = df[~strict_filter_mask].copy()
            before_filter = len(df_strict)
            df_strict = df_strict[df_strict['Item'].isin(allowed_items)].copy()
            after_filter = len(df_strict)
            df = pd.concat([df_other, df_strict], ignore_index=True)
            if before_filter > after_filter:
                print(f"\n[INFO]  严格过滤：删除 {before_filter - after_filter} 行不在dict_v3中的livestock merged Items")
                print(f"  LUC等非livestock commodity Item保留进入最细汇总层")
        else:
            print(f"\n[DEBUG] 跳过全局严格Item过滤：未检测到需要约束的livestock merged项")
    
    # 1.5 Normalize M49, add Region_label_new, and filter invalid countries.
    if region_mapping and 'M49_Country_Code' in df.columns:
        # Remove M49 quotes and pad to three digits.
        def normalize_m49_code(code):
            if pd.isna(code):
                return None
            m49 = _norm_m49_code(code)
            return m49 or None
        df['M49_Country_Code'] = df['M49_Country_Code'].apply(normalize_m49_code)
        
        # Map to Region.
        df['Region_label_new'] = df['M49_Country_Code'].map(region_mapping)
        
        # Count unmatched countries.
        unmatched = df[df['Region_label_new'].isna()]['M49_Country_Code'].unique()
        if len(unmatched) > 0:
            print(f"[WARNING] 未匹配的M49代码（未在dict_v3中找到）: {list(unmatched)[:10]}")
        
        # Exclude Region_label_new values that are NaN or 'no'.
        rows_before = len(df)
        df = df[df['Region_label_new'].notna() & (df['Region_label_new'] != 'no')].reset_index(drop=True)
        rows_after = len(df)
        if rows_before > rows_after:
            print(f"[INFO] 过滤掉 {rows_before - rows_after} 行无效国家数据")
        
        print(f"[INFO] 规范化和过滤后保留 {len(df)} 行，涉及 {df['M49_Country_Code'].nunique()} 个国家")
    
    # 1.9 Remove helper columns.
    if '_is_commodity_based' in df.columns:
        df = df.drop(columns=['_is_commodity_based'])
    
    # 2. Generate long-format original-data results.
    result_long = df.copy()
    
    # As requested, Long omits Country and iso3, retaining M49_Country_Code only.
    # Drop unused empty CH4_kt/N2O_kt/CO2_kt columns; value already contains emissions.
    cols_to_drop = [c for c in ['Country', 'iso3', 'country', 'CH4_kt', 'N2O_kt', 'CO2_kt'] if c in result_long.columns]
    if cols_to_drop:
        result_long = result_long.drop(columns=cols_to_drop)
    
    # Identify available columns.
    has_m49 = 'M49_Country_Code' in df.columns
    has_year = 'year' in df.columns
    has_process = 'Process' in df.columns
    has_item = 'Item' in df.columns
    has_ghg = 'GHG' in df.columns
    
    # Ensure Process is string-typed.
    if has_process:
        df['Process'] = df['Process'].astype(str)

    # Do not sum process totals together with Item details, which would inflate Sheet2/3.
    # Confirmed duplicate sources:
    # 1. GCE All Crops / All Animals.
    # 2. GSOIL aggregate Item 'Drained organic soils'.
    df_agg = df.copy()
    if has_item and has_process:
        process_series = df_agg['Process'].astype(str).str.strip()
        item_series = df_agg['Item'].astype(str).str.strip()
        aggregate_item_mask = item_series.isin({'All Animals', 'All Crops'})
        aggregate_item_mask = aggregate_item_mask | (
            process_series.eq('Drained organic soils') &
            item_series.eq('Drained organic soils')
        )
        removed_aggregate_rows = int(aggregate_item_mask.sum())
        if removed_aggregate_rows > 0:
            df_agg = df_agg.loc[~aggregate_item_mask].copy()
            print(
                f"[INFO] 汇总层排除 {removed_aggregate_rows} 行aggregate item，"
                f"避免By_Country_Process/By_Country重复计数"
            )
    elif has_process:
        df_agg['Process'] = df_agg['Process'].astype(str)

    # 2.5 Remove dairy/non-dairy merging.
    # dict_v3 already defines separate dairy/non-dairy Item_Emis entries; do not create merged ones.
    # Sheet1 directly displays dict_v3 dairy/non-dairy Items.
    # Sheet2/3 use normal groupby aggregation without special exclusions.
    print(f"[INFO] 保留dairy/non-dairy原始分拆形式（按dict_v3定义）")
    
    # 3. Country-process-commodity summary using M49_Country_Code, not Country.
    groupby_cols_cpc = []
    if has_m49:
        groupby_cols_cpc.append('M49_Country_Code')
    if has_process:
        groupby_cols_cpc.append('Process')
    if has_item:
        groupby_cols_cpc.append('Item')
    if has_year:
        groupby_cols_cpc.append('year')
    if has_ghg:
        groupby_cols_cpc.append('GHG')
    
    if len(groupby_cols_cpc) >= 3 and 'value' in df_agg.columns:
        result_by_ctry_proc_comm = df_agg.groupby(by=groupby_cols_cpc, as_index=False, dropna=False).agg({
            'value': 'sum'
        }).rename(columns={'value': 'total_emissions'})
        # Region_label_new already exists and is preserved by groupby.
        if 'Region_label_new' not in result_by_ctry_proc_comm.columns and region_mapping:
            result_by_ctry_proc_comm['Region_label_new'] = result_by_ctry_proc_comm['M49_Country_Code'].map(region_mapping)
        
        # Add Global summary rows.
        if has_process and has_item and has_year and has_ghg:
            global_groupby_cols = ['Process', 'Item', 'year', 'GHG']
            global_agg = df_agg.groupby(by=global_groupby_cols, as_index=False).agg({'value': 'sum'})
            global_agg['M49_Country_Code'] = "'000"
            global_agg['Region_label_new'] = 'Global'
            global_agg = global_agg.rename(columns={'value': 'total_emissions'})
            # Combine.
            result_by_ctry_proc_comm = pd.concat([result_by_ctry_proc_comm, global_agg], ignore_index=True)
        
        # Exclude All Animals and All Crops.
        if 'Item' in result_by_ctry_proc_comm.columns:
            result_by_ctry_proc_comm = result_by_ctry_proc_comm[
                ~result_by_ctry_proc_comm['Item'].isin(['All Animals', 'All Crops'])
            ]
        
        # Column order: M49_Country_Code, Region_label_new, Process, Item, year, GHG, total_emissions.
        cols_order = ['M49_Country_Code', 'Region_label_new', 'Process', 'Item', 'year', 'GHG', 'total_emissions']
        existing_cols = [col for col in cols_order if col in result_by_ctry_proc_comm.columns]
        result_by_ctry_proc_comm = result_by_ctry_proc_comm[existing_cols]
        
        # Sort Region_label_new, Process, Item, and GHG ascending.
        sort_cols = ['Region_label_new']
        if 'Process' in result_by_ctry_proc_comm.columns:
            sort_cols.append('Process')
        if 'Item' in result_by_ctry_proc_comm.columns:
            sort_cols.append('Item')
        if 'GHG' in result_by_ctry_proc_comm.columns:
            sort_cols.append('GHG')
        result_by_ctry_proc_comm = result_by_ctry_proc_comm.sort_values(by=sort_cols, ascending=True).reset_index(drop=True)
        
        print(f"[DEBUG] by_ctry_proc_comm结果形状: {result_by_ctry_proc_comm.shape}")
    else:
        print(f"[DEBUG] 跳过by_ctry_proc_comm groupby (条件不满足: len={len(groupby_cols_cpc)}, value_exists={'value' in df_agg.columns})")
        result_by_ctry_proc_comm = pd.DataFrame()
    
    # 4. Country-process summary using M49_Country_Code only.
    groupby_cols_cp = []
    if has_m49:
        groupby_cols_cp.append('M49_Country_Code')
    if has_process:
        groupby_cols_cp.append('Process')
    if has_year:
        groupby_cols_cp.append('year')
    if has_ghg:
        groupby_cols_cp.append('GHG')
    
    if len(groupby_cols_cp) >= 2 and 'value' in df_agg.columns:
        # Check input data BEFORE groupby
        import logging
        logging.info(f"\n[DEBUG-GROUPBY-INPUT] by_ctry_proc groupby INPUT:")
        logging.info(f"  - df shape: {df_agg.shape}")
        logging.info(f"  - years in df: {sorted(df_agg['year'].unique().tolist()) if 'year' in df_agg.columns else 'N/A'}")
        logging.info(f"  - 2080 data rows in df: {len(df_agg[df_agg['year']==2080]) if 'year' in df_agg.columns else 0}")
        if 'year' in df_agg.columns and 'Process' in df_agg.columns:
            livestock_procs = ['Enteric fermentation', 'Manure management', 'Manure applied to soils', 'Manure left on pasture']
            for proc in livestock_procs:
                df_2080_proc = df_agg[(df_agg['year']==2080) & (df_agg['Process']==proc)]
                logging.info(f"  - {proc} 2080年: {len(df_2080_proc)}行, value非零={sum(df_2080_proc['value'].notna() & (df_2080_proc['value']!=0))}, 总计={df_2080_proc['value'].sum():.2f}")

        # Sheet2 groups all data directly without excluding dairy/non-dairy.
        # groupby aggregates by Process without double counting.
        result_by_ctry_proc = df_agg.groupby(by=groupby_cols_cp, as_index=False, dropna=False).agg({
            'value': 'sum'
        }).rename(columns={'value': 'total_emissions'})
        
        # Check output data AFTER groupby
        logging.info(f"\n[DEBUG-GROUPBY-OUTPUT] by_ctry_proc groupby OUTPUT:")
        logging.info(f"  - result shape: {result_by_ctry_proc.shape}")
        logging.info(f"  - years in result: {sorted(result_by_ctry_proc['year'].unique().tolist()) if 'year' in result_by_ctry_proc.columns else 'N/A'}")
        logging.info(f"  - 2080 data rows in result: {len(result_by_ctry_proc[result_by_ctry_proc['year']==2080]) if 'year' in result_by_ctry_proc.columns else 0}")
        if 'year' in result_by_ctry_proc.columns and 'Process' in result_by_ctry_proc.columns:
            for proc in livestock_procs:
                res_2080_proc = result_by_ctry_proc[(result_by_ctry_proc['year']==2080) & (result_by_ctry_proc['Process']==proc)]
                logging.info(f"  - {proc} 2080年: {len(res_2080_proc)}行, total非零={sum(res_2080_proc['total_emissions'].notna() & (res_2080_proc['total_emissions']!=0))}, 总计={res_2080_proc['total_emissions'].sum():.2f}")
        
        # Region_label_new already exists in df.
        if 'Region_label_new' not in result_by_ctry_proc.columns and region_mapping:
            result_by_ctry_proc['Region_label_new'] = result_by_ctry_proc['M49_Country_Code'].map(region_mapping)
        
        # Add Global summary rows.
        if has_process and has_year and has_ghg:
            global_groupby_cols = ['Process', 'year', 'GHG']
            global_agg = df_agg.groupby(by=global_groupby_cols, as_index=False).agg({'value': 'sum'})
            global_agg['M49_Country_Code'] = "'000"
            global_agg['Region_label_new'] = 'Global'
            global_agg = global_agg.rename(columns={'value': 'total_emissions'})
            # Combine.
            result_by_ctry_proc = pd.concat([result_by_ctry_proc, global_agg], ignore_index=True)
        
        # Column order: M49_Country_Code, Region_label_new, Process, year, GHG, total_emissions.
        cols_order = ['M49_Country_Code', 'Region_label_new', 'Process', 'year', 'GHG', 'total_emissions']
        existing_cols = [col for col in cols_order if col in result_by_ctry_proc.columns]
        result_by_ctry_proc = result_by_ctry_proc[existing_cols]
        
        # Sort Region_label_new, Process, and GHG ascending.
        sort_cols = ['Region_label_new']
        if 'Process' in result_by_ctry_proc.columns:
            sort_cols.append('Process')
        if 'GHG' in result_by_ctry_proc.columns:
            sort_cols.append('GHG')
        result_by_ctry_proc = result_by_ctry_proc.sort_values(by=sort_cols, ascending=True).reset_index(drop=True)
        
        print(f"[DEBUG] by_ctry_proc结果形状: {result_by_ctry_proc.shape}")
    else:
        print(f"[DEBUG] 跳过by_ctry_proc groupby (条件不满足)")
        result_by_ctry_proc = pd.DataFrame()
    
    # 5. Country summary using M49_Country_Code only.
    # Use the dairy/non-dairy-excluded data consistently with by_ctry_proc.
    groupby_cols_c = []
    if has_m49:
        groupby_cols_c.append('M49_Country_Code')
    if has_year:
        groupby_cols_c.append('year')
    if has_ghg:
        groupby_cols_c.append('GHG')
    
    if len(groupby_cols_c) >= 2 and 'value' in df_agg.columns:
        # Sheet3 groups all data directly.
        result_by_ctry = df_agg.groupby(by=groupby_cols_c, as_index=False, dropna=False).agg({
            'value': 'sum'
        }).rename(columns={'value': 'total_emissions'})
        # Region_label_new already exists in df.
        if 'Region_label_new' not in result_by_ctry.columns and region_mapping:
            result_by_ctry['Region_label_new'] = result_by_ctry['M49_Country_Code'].map(region_mapping)
        
        # Add Global summary rows.
        if has_year and has_ghg:
            global_groupby_cols = ['year', 'GHG']
            global_agg = df_agg.groupby(by=global_groupby_cols, as_index=False).agg({'value': 'sum'})
            global_agg['M49_Country_Code'] = "'000"
            global_agg['Region_label_new'] = 'Global'
            global_agg = global_agg.rename(columns={'value': 'total_emissions'})
            # Combine.
            result_by_ctry = pd.concat([result_by_ctry, global_agg], ignore_index=True)
        
        # Column order: M49_Country_Code, Region_label_new, year, GHG, total_emissions.
        cols_order = ['M49_Country_Code', 'Region_label_new', 'year', 'GHG', 'total_emissions']
        existing_cols = [col for col in cols_order if col in result_by_ctry.columns]
        result_by_ctry = result_by_ctry[existing_cols]
        
        # Sort Region_label_new and GHG ascending.
        sort_cols = ['Region_label_new']
        if 'GHG' in result_by_ctry.columns:
            sort_cols.append('GHG')
        result_by_ctry = result_by_ctry.sort_values(by=sort_cols, ascending=True).reset_index(drop=True)
        
        print(f"[DEBUG] by_ctry结果形状: {result_by_ctry.shape}")
    else:
        print(f"[DEBUG] 跳过by_ctry groupby (条件不满足)")
        result_by_ctry = pd.DataFrame()
    
    # Pivot years to wide format, replacing the previous format.
    # Pivot every summary level, replacing long-format results.
    import logging
    logging.info(f"\n[DEBUG-PIVOT-LOOP] 开始pivot循环")
    logging.info(f"  - result_by_ctry_proc_comm: {type(result_by_ctry_proc_comm)}, empty={result_by_ctry_proc_comm.empty if isinstance(result_by_ctry_proc_comm, pd.DataFrame) else 'N/A'}")
    logging.info(f"  - result_by_ctry_proc: {type(result_by_ctry_proc)}, empty={result_by_ctry_proc.empty if isinstance(result_by_ctry_proc, pd.DataFrame) else 'N/A'}")
    logging.info(f"  - result_by_ctry: {type(result_by_ctry)}, empty={result_by_ctry.empty if isinstance(result_by_ctry, pd.DataFrame) else 'N/A'}")
    
    for key, df_result in [('by_ctry_proc_comm', result_by_ctry_proc_comm),
                           ('by_ctry_proc', result_by_ctry_proc),
                           ('by_ctry', result_by_ctry)]:
        logging.info(f"\n[DEBUG-PIVOT-LOOP] 检查{key}: type={type(df_result)}, empty={df_result.empty if isinstance(df_result, pd.DataFrame) else 'N/A'}, has_year={'year' in df_result.columns if isinstance(df_result, pd.DataFrame) else 'N/A'}")
        if isinstance(df_result, pd.DataFrame) and not df_result.empty and 'year' in df_result.columns:
            
            print(f"[DEBUG] 透视前{key}: 行={len(df_result)}, M49值={sorted(df_result['M49_Country_Code'].unique().tolist()) if 'M49_Country_Code' in df_result.columns else 'N/A'}")
            print(f"[DEBUG] 透视前{key}完整数据:\n{df_result[['M49_Country_Code', 'year', 'total_emissions']].head(10) if 'M49_Country_Code' in df_result.columns else 'N/A'}")
            
            
            # Index columns exclude year and total_emissions.
            index_cols = [col for col in df_result.columns 
                         if col not in ['year', 'total_emissions']]
            
            print(f"[DEBUG] {key} 的index_cols: {index_cols}")
            
            if index_cols:
                try:
                    # Pivot years into columns.
                    pivot_df = df_result.pivot_table(
                        index=index_cols,
                        columns='year',
                        values='total_emissions',
                        aggfunc='sum',
                        fill_value=0
                    ).reset_index()
                    
                    import logging
                    logging.info(f"[DEBUG] 透视后{key}: 行={len(pivot_df)}, M49值={sorted(pivot_df['M49_Country_Code'].unique().tolist()) if 'M49_Country_Code' in pivot_df.columns else 'N/A'}")
                    
                    # Check pivoted year columns.
                    year_cols_before_rename = [col for col in pivot_df.columns if isinstance(col, (int, float)) and not pd.isna(col)]
                    logging.info(f"[DEBUG-PIVOT] {key}透视后的年份列: {sorted(year_cols_before_rename)}")
                    if 2080 in year_cols_before_rename:
                        livestock_procs = ['Enteric fermentation', 'Manure management', 'Manure applied to soils', 'Manure left on pasture']
                        if key == 'by_ctry_proc' and 'Process' in pivot_df.columns:
                            for proc in livestock_procs:
                                proc_rows = pivot_df[pivot_df['Process'] == proc]
                                if len(proc_rows) > 0:
                                    # Exclude Global rows with M49=apostrophe-prefixed 000 to prevent double counting.
                                    if 'M49_Country_Code' in pivot_df.columns:
                                        proc_rows_no_global = proc_rows[proc_rows['M49_Country_Code'] != "'000"]
                                        y2080_nonzero = (proc_rows_no_global[2080].notna() & (proc_rows_no_global[2080] != 0)).sum()
                                        y2080_sum = proc_rows_no_global[2080].sum()
                                        y2080_global = proc_rows[proc_rows['M49_Country_Code'] == "'000"][2080].sum() if "'000" in proc_rows['M49_Country_Code'].values else 0
                                        logging.info(f"[DEBUG-PIVOT] {proc}: {len(proc_rows)}行(含Global), Y2080非零={y2080_nonzero}(不含Global), 总计={y2080_sum:.2f}(不含Global), Global={y2080_global:.2f}")
                                    else:
                                        y2080_nonzero = (proc_rows[2080].notna() & (proc_rows[2080] != 0)).sum()
                                        y2080_sum = proc_rows[2080].sum()
                                        logging.info(f"[DEBUG-PIVOT] {proc}: {len(proc_rows)}行, Y2080非零={y2080_nonzero}, 总计={y2080_sum:.2f}")
                    
                    # Rename years to Y2010, Y2020, etc.
                    year_columns = {col: f'Y{col}' for col in pivot_df.columns 
                                   if isinstance(col, (int, float)) and not pd.isna(col)}
                    pivot_df = pivot_df.rename(columns=year_columns)
                    
                    # Keep non-year columns first and year columns last after pivoting.
                    non_year_cols = [col for col in pivot_df.columns if not col.startswith('Y')]
                    year_cols = sorted([col for col in pivot_df.columns if col.startswith('Y')])
                    pivot_df = pivot_df[non_year_cols + year_cols]
                    
                    # Replace the original variable with the pivoted table.
                    if key == 'by_ctry_proc_comm':
                        result_by_ctry_proc_comm = pivot_df
                    elif key == 'by_ctry_proc':
                        result_by_ctry_proc = pivot_df
                    elif key == 'by_ctry':
                        result_by_ctry = pivot_df
                except Exception as e:
                    # Skip if pivoting fails.
                    print(f"[ERROR] {key} 透视失败: {e}")
                    pass
    
    # Append CO2eq rows to each pivot table through GHG values.
    def add_ghg_co2eq_rows(pivot_df: pd.DataFrame) -> pd.DataFrame:
        """Append vectorized CH4_CO2eq/N2O_CO2eq/CO2_CO2eq/CO2eq conversion rows."""
        if pivot_df.empty or 'GHG' not in pivot_df.columns:
            return pivot_df
        year_cols = sorted([col for col in pivot_df.columns if col.startswith('Y')])
        if not year_cols:
            return pivot_df
        non_year_cols = [col for col in pivot_df.columns if not col.startswith('Y')]
        groupby_cols = [col for col in non_year_cols if col != 'GHG']
        gas_rows = pivot_df[pivot_df['GHG'].isin(['CH4', 'N2O', 'CO2'])].copy()
        if gas_rows.empty:
            return pivot_df

        gwp_map = {'CO2': 1.0, 'CH4': 27.2, 'N2O': 273.0}
        gas_rows['_gwp'] = gas_rows['GHG'].map(gwp_map).fillna(0.0)
        gas_rows[year_cols] = gas_rows[year_cols].mul(gas_rows['_gwp'], axis=0)

        ghg_co2eq_rows = gas_rows.copy()
        ghg_co2eq_rows['GHG'] = ghg_co2eq_rows['GHG'].astype(str) + '_CO2eq'
        ghg_co2eq_rows = ghg_co2eq_rows.drop(columns=['_gwp'], errors='ignore')

        total_rows = gas_rows.copy()
        total_rows['GHG'] = 'CO2eq'
        total_rows = total_rows.drop(columns=['_gwp'], errors='ignore')
        if groupby_cols:
            total_rows = total_rows.groupby(groupby_cols + ['GHG'], as_index=False, dropna=False)[year_cols].sum()
        else:
            total_sum = total_rows[year_cols].sum().to_frame().T
            total_sum['GHG'] = 'CO2eq'
            total_rows = total_sum

        result_df = pd.concat([pivot_df, ghg_co2eq_rows, total_rows], ignore_index=True, sort=False)
        return result_df[non_year_cols + year_cols]
    
    # Apply to all three summary tables.
    if isinstance(result_by_ctry_proc_comm, pd.DataFrame) and not result_by_ctry_proc_comm.empty:
        result_by_ctry_proc_comm = add_ghg_co2eq_rows(result_by_ctry_proc_comm)
    
    if isinstance(result_by_ctry_proc, pd.DataFrame) and not result_by_ctry_proc.empty:
        result_by_ctry_proc = add_ghg_co2eq_rows(result_by_ctry_proc)
    
    if isinstance(result_by_ctry, pd.DataFrame) and not result_by_ctry.empty:
        result_by_ctry = add_ghg_co2eq_rows(result_by_ctry)
    
    # Sort all three wide tables by M49_Country_Code, Process, Item, and GHG ascending.
    def sort_wide_table(df: pd.DataFrame) -> pd.DataFrame:
        """Sort M49_Country_Code, Process, Item, and GHG ascending."""
        if df.empty:
            return df
        
        sort_cols = []
        for col in ['M49_Country_Code', 'Process', 'Item', 'GHG']:
            if col in df.columns:
                sort_cols.append(col)
        
        if sort_cols:
            df = df.sort_values(by=sort_cols, ascending=True).reset_index(drop=True)
        
        return df
    
    result_by_ctry_proc_comm = sort_wide_table(result_by_ctry_proc_comm)
    result_by_ctry_proc = sort_wide_table(result_by_ctry_proc)
    result_by_ctry = sort_wide_table(result_by_ctry)
    
    # Convert M49 051 to the leading-apostrophe format.
    def format_m49_with_quote(df_in: pd.DataFrame) -> pd.DataFrame:
        """Format M49_Country_Code as three digits with a leading apostrophe."""
        if df_in is None or df_in.empty or 'M49_Country_Code' not in df_in.columns:
            return df_in
        df_out = df_in.copy()
        def _fmt(x):
            if pd.isna(x) or x is None or str(x).strip() == '':
                return ''
            return _norm_m49_code(x)
        df_out['M49_Country_Code'] = df_out['M49_Country_Code'].apply(_fmt)
        return df_out
    
    result_by_ctry_proc_comm = format_m49_with_quote(result_by_ctry_proc_comm)
    result_by_ctry_proc = format_m49_with_quote(result_by_ctry_proc)
    result_by_ctry = format_m49_with_quote(result_by_ctry)
    result_long = format_m49_with_quote(result_long)
    
    # Return without by_year, removed as requested; pivots replaced the original variables.
    return {
        'by_ctry_proc_comm': result_by_ctry_proc_comm,
        'by_ctry_proc': result_by_ctry_proc,
        'by_ctry': result_by_ctry,
        'long': result_long,  # Retain Detail_Long.
    }


def summarize_saved_detail_long(detail_long_path: str,
                                dict_v3_path: Optional[str] = None,
                                allowed_years: Optional[List[int]] = None,
                                production_df: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
    """Read an existing emissions_summary_Detail_Long.csv and rebuild the three wide summary tables."""
    detail_path = Path(detail_long_path)
    if not detail_path.exists():
        raise FileNotFoundError(f"Detail_Long file not found: {detail_path}")

    detail_df = pd.read_csv(detail_path)
    inferred_years: Optional[List[int]] = None
    if allowed_years is None and 'year' in detail_df.columns:
        years = pd.to_numeric(detail_df['year'], errors='coerce').dropna()
        if not years.empty:
            inferred_years = sorted(years.astype(int).unique().tolist())

    resolved_dict_v3 = dict_v3_path or str(Path(get_src_base()) / 'dict_v3.xlsx')
    return summarize_emissions_from_detail(
        detail_df,
        allowed_years=allowed_years or inferred_years,
        dict_v3_path=resolved_dict_v3,
        production_df=production_df,
    )


def write_summary_tables_from_detail_long(detail_long_path: str,
                                          output_dir: Optional[str] = None,
                                          dict_v3_path: Optional[str] = None,
                                          allowed_years: Optional[List[int]] = None,
                                          production_df: Optional[pd.DataFrame] = None,
                                          filename_suffix: str = '') -> Dict[str, Path]:
    """Rebuild and write the three emissions summary CSVs from an existing Detail_Long file."""
    summaries = summarize_saved_detail_long(
        detail_long_path=detail_long_path,
        dict_v3_path=dict_v3_path,
        allowed_years=allowed_years,
        production_df=production_df,
    )

    detail_path = Path(detail_long_path)
    outdir = Path(output_dir) if output_dir else detail_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    file_map = {
        'by_ctry_proc_comm': f'emissions_summary_By_Country_Process_Item{filename_suffix}.csv',
        'by_ctry_proc': f'emissions_summary_By_Country_Process{filename_suffix}.csv',
        'by_ctry': f'emissions_summary_By_Country{filename_suffix}.csv',
    }
    written: Dict[str, Path] = {}
    for key, filename in file_map.items():
        df = summaries.get(key)
        if not isinstance(df, pd.DataFrame):
            continue
        csv_path = outdir / filename
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        written[key] = csv_path
    return written


def summarize_emissions(fao_results: Dict[str, Any],
                        extra_emis: Optional[pd.DataFrame]=None,
                        process_meta_map: Optional[dict]=None,
                        dict_v3_path: Optional[str]=None,
                        allowed_years: Optional[set]=None,
                        production_df: Optional[pd.DataFrame]=None) -> Dict[str, pd.DataFrame]:
    """
    Generate summaries from FAO module results through the backward-compatible interface.
    
    Args:
        fao_results: Module dictionary, e.g. {'GCE': [df1, df2, ...], 'GLE': [...], ...}.
        extra_emis: Additional emissions data.
        process_meta_map: Process metadata mappings.
        dict_v3_path: Path to dict_v3.xlsx for Region mappings.
        allowed_years: Optional set of permitted years, excluding other years.
        production_df: Production and stocks for splitting merged historical livestock emissions.
        
    Returns:
        Dictionary of summaries at multiple levels.
    """
    # log allowed_years details
    import logging
    logging.info(f"\n[DEBUG-CRITICAL] summarize_emissions arguments:")
    logging.info(f"  - allowed_years type: {type(allowed_years)}")
    allowed_years_set = {int(y) for y in allowed_years} if allowed_years else None
    logging.info(f"  - allowed_years values: {sorted(list(allowed_years_set)) if allowed_years_set else None}")
    logging.info(f"  - contains 2080: {2080 in allowed_years_set if allowed_years_set else 'N/A'}")
    if not fao_results or not isinstance(fao_results, dict):
        return {
            'by_ctry_proc_comm': pd.DataFrame(),
            'by_ctry_proc': pd.DataFrame(),
            'by_ctry': pd.DataFrame(),
            'by_year': pd.DataFrame(),
            'long': pd.DataFrame()
        }
    
    # Collect all emissions.
    all_emis_dfs = []

    def ensure_scalar_process(df, process_value=None):
        """Ensure Process and Item contain scalar values, not Series or other objects."""
        df = df.copy()
        
        # Overwrite Process only when process_value is explicitly supplied.
        # Preserve the original GLE process column otherwise.
        if process_value is not None:
            # Remove existing process/Process columns that may contain Series.
            for col in ['process', 'Process', 'process_old']:
                if col in df.columns:
                    df = df.drop(columns=[col])
            # Add Process as a scalar value.
            df['Process'] = str(process_value)
        else:
            # Without process_value, ensure existing Process values are strings.
            if 'Process' in df.columns:
                # Check for Series values.
                if df['Process'].apply(lambda x: isinstance(x, pd.Series)).any():
                    # Flatten Series when present.
                    df['Process'] = df['Process'].apply(
                        lambda x: str(x.iloc[0]) if isinstance(x, pd.Series) and len(x) > 0 else str(x)
                    )
                else:
                    # Convert directly to strings.
                    df['Process'] = df['Process'].astype(str)
            elif 'process' in df.columns:
                # Rename to Process and convert to strings.
                df = df.rename(columns={'process': 'Process'})
                df['Process'] = df['Process'].astype(str)
        
        # Ensure Item also contains scalar values.
        for item_col in ['Item', 'item', 'commodity']:
            if item_col in df.columns:
                # Check for Series or list values.
                def flatten_value(x):
                    if isinstance(x, pd.Series):
                        return str(x.iloc[0]) if len(x) > 0 else ''
                    elif isinstance(x, (list, tuple)):
                        return str(x[0]) if len(x) > 0 else ''
                    else:
                        return str(x)
                
                if df[item_col].apply(lambda x: isinstance(x, (pd.Series, list, tuple))).any():
                    df[item_col] = df[item_col].apply(flatten_value)
                else:
                    df[item_col] = df[item_col].astype(str)
                
                # Rename to Item if needed.
                if item_col != 'Item':
                    df = df.rename(columns={item_col: 'Item'})
                break
        
        return df
    
    def collect_extra_frames(obj) -> List[pd.DataFrame]:
        frames: List[pd.DataFrame] = []
        if obj is None:
            return frames
        if isinstance(obj, pd.DataFrame):
            if not obj.empty:
                frames.append(obj.copy())
            return frames
        if isinstance(obj, (list, tuple, set)):
            for entry in obj:
                frames.extend(collect_extra_frames(entry))
            return frames
        if isinstance(obj, dict):
            for entry in obj.values():
                frames.extend(collect_extra_frames(entry))
        return frames

    for module_name, module_results in fao_results.items():
        if module_results is None:
            continue
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[DEBUG-S4_1] 处理模块 {module_name}, 类型: {type(module_results)}")
            
        # module_results may be a DataFrame, list, or dictionary.
        if isinstance(module_results, pd.DataFrame):
            if not module_results.empty:
                # Preserve valid existing Process columns in LUC/GSOIL.
                # LUC includes De/Reforestation_crop, De/Reforestation_pasture, Forest, and Wood harvest.
                # GSOIL includes Drained organic soils.
                # Set process_value=module_name only for data without Process, such as certain GCE/GFIRE results.
                process_val = None if module_name in ['LUC', 'GSOIL'] else module_name
                df = ensure_scalar_process(module_results, process_value=process_val)
                if 'module' not in df.columns:
                    df['module'] = module_name
                all_emis_dfs.append(df)
                print(f"  添加DataFrame: {len(df)} 行")
        elif isinstance(module_results, list):
            print(f"  List包含 {len(module_results)} 个元素")
            for idx, item in enumerate(module_results):
                if isinstance(item, pd.DataFrame) and not item.empty:
                    # Preserve process columns in list-contained DataFrames.
                    print(f"  [List项{idx}] 原始数据 - 形状: {item.shape}, 列: {list(item.columns)}")
                    if 'process' in item.columns:
                        print(f"    process列值: {_unique_column_values(item, 'process')}")
                    
                    df = ensure_scalar_process(item, process_value=None)
                    
                    # Standardize gas column names to uppercase.
                    rename_map = {
                        'ch4_kt': 'CH4_kt',
                        'n2o_kt': 'N2O_kt',
                        'co2_kt': 'CO2_kt',
                        'process': 'Process',
                        'country': 'Country'
                    }
                    df = df.rename(columns=rename_map)
                    
                    print(f"  [List项{idx}] ensure_scalar_process后 - 形状: {df.shape}, 列: {list(df.columns)}")
                    if 'Process' in df.columns:
                        print(f"    Process列值: {_unique_column_values(df, 'Process')}")
                    
                    if 'module' not in df.columns:
                        df['module'] = module_name
                    
                    # WARNING: debug: inspect emission values
                    print(f"  [List项{idx}] 最终形状: {df.shape}")
                    for gas_col in ['ch4_kt', 'n2o_kt', 'co2_kt', 'CH4_kt', 'N2O_kt', 'CO2_kt']:
                        if gas_col in df.columns:
                            non_zero = (df[gas_col] > 0).sum()
                            total = df[gas_col].sum()
                            print(f"    {gas_col}: {non_zero}个非零值, 总计={total:.2f}")
                    
                    print(f"  [List项{idx}] 添加到all_emis_dfs前 - 行数: {len(df)}")
                    # Check year distribution.
                    if 'year' in df.columns:
                        years_in_df = sorted(df['year'].unique())
                        print(f"    年份: {years_in_df}")
                        print(f"    包含2080: {2080 in years_in_df}")
                    all_emis_dfs.append(df)
                    print(f"  [List项{idx}] 已添加到all_emis_dfs (总计现在有{len(all_emis_dfs)}个DataFrame)")
        elif isinstance(module_results, dict):
            print(f"  Dict包含 {len(module_results)} 个process")
            # FAO format: {'GCE': {'Residues': df, 'Burning': df, ...}}.
            for process_name, process_df in module_results.items():
                if isinstance(process_df, pd.DataFrame) and not process_df.empty:
                    print(f"  [Dict项 {process_name}] 原始数据 - 形状: {process_df.shape}, 列: {list(process_df.columns)}")
                    
                    # Preserve Process in dictionary-contained DataFrames; do not overwrite with dictionary keys.
                    df = ensure_scalar_process(process_df, process_value=None)
                    
                    # Standardize gas column names to uppercase.
                    rename_map = {
                        'ch4_kt': 'CH4_kt',
                        'n2o_kt': 'N2O_kt',
                        'co2_kt': 'CO2_kt',
                        'process': 'Process'
                    }
                    df = df.rename(columns=rename_map)
                    
                    if 'module' not in df.columns:
                        df['module'] = module_name
                    
                    print(f"  [Dict项 {process_name}] 标准化后 - 形状: {df.shape}, 列: {list(df.columns)}")
                    all_emis_dfs.append(df)
                    print(f"  [Dict项 {process_name}] 已添加到all_emis_dfs (总计现在有{len(all_emis_dfs)}个DataFrame)")

    # Add extra emissions data (DataFrame, list, or dict).
    extra_frames = collect_extra_frames(extra_emis)
    if extra_frames:
        print(f"[DEBUG] 额外排放输入共 {len(extra_frames)} 个DataFrame")
    for idx, extra_df in enumerate(extra_frames):
        df = ensure_scalar_process(extra_df, process_value=None)
        if 'module' not in df.columns:
            df['module'] = 'extra_emis'
        all_emis_dfs.append(df)
        print(f"  [EXTRA {idx}] 行数: {len(df)}, 列: {list(df.columns)}")
    
    if not all_emis_dfs:
        return {
            'by_ctry_proc_comm': pd.DataFrame(),
            'by_ctry_proc': pd.DataFrame(),
            'by_ctry': pd.DataFrame(),
            'by_year': pd.DataFrame(),
            'long': pd.DataFrame()
        }
    
    # Combine all data.
    combined_df = pd.concat(all_emis_dfs, ignore_index=True)
    # Normalize year column to integers for reliable filtering
    if 'year' in combined_df.columns:
        combined_df['year'] = pd.to_numeric(combined_df['year'], errors='coerce')
        combined_df = combined_df.dropna(subset=['year'])
        combined_df['year'] = combined_df['year'].astype(int)
    
    # Remove only exact duplicate rows. Rows sharing the same
    # (M49, Process, Item, GHG, year) can be valid additive components
    # (for example LUC carbon-pool response plus historical-background
    # direct emissions) and must be aggregated downstream, not dropped here.
    key_cols = ['M49_Country_Code', 'Process', 'Item', 'GHG', 'year']
    existing_key_cols = [c for c in key_cols if c in combined_df.columns]
    before_exact_dedup = len(combined_df)
    combined_df = combined_df.drop_duplicates(keep='first').copy()
    after_exact_dedup = len(combined_df)
    if before_exact_dedup > after_exact_dedup:
        print(
            f" [WARN] 发现并移除 {before_exact_dedup - after_exact_dedup} 行完全重复排放记录 "
            f"({before_exact_dedup} ?? {after_exact_dedup})"
        )
    if len(existing_key_cols) >= 4:
        same_key_rows = int(combined_df.duplicated(subset=existing_key_cols, keep=False).sum())
        if same_key_rows > 0:
            same_key_groups = int(
                combined_df.loc[
                    combined_df.duplicated(subset=existing_key_cols, keep=False),
                    existing_key_cols,
                ].drop_duplicates().shape[0]
            )
            print(
                f"[INFO] 发现 {same_key_rows} 行共享排放键、覆盖 {same_key_groups} 个键；"
                "这些行将保留并在后续汇总中求和。"
            )
    
    
    import logging
    logging.info(f"\n[DEBUG] Combined后的DataFrame:")
    logging.info(f"  - 总行数: {len(combined_df)}")
    logging.info(f"  - 列: {list(combined_df.columns)}")
    if 'year' in combined_df.columns:
        years_in_combined = sorted(combined_df['year'].unique())
        logging.info(f"  - 年份范围: {years_in_combined}")
        logging.info(f"  - 2080是否在combined_df: {2080 in years_in_combined}")
        if 2080 in years_in_combined:
            livestock_2080 = combined_df[(combined_df['year'] == 2080) & (combined_df['module'] == 'GLE')]
            logging.info(f"  - GLE模块2080年数据: {len(livestock_2080)} 行")
    if 'module' in combined_df.columns:
        logging.info(f"  - module分布: {combined_df['module'].value_counts().to_dict()}")
    if 'Process' in combined_df.columns:
        logging.info(f"  - Process分布: {combined_df['Process'].value_counts().to_dict()}")
    
    # WARNING: filter allowed years to prevent stray years
    import logging
    if allowed_years_set and 'year' in combined_df.columns:
        years_in_data = sorted(combined_df['year'].unique())
        removed_years = [y for y in years_in_data if y not in allowed_years_set]
        logging.info(f"[DEBUG-CRITICAL] Before year filter:")
        logging.info(f"  - years in combined_df: {years_in_data}")
        logging.info(f"  - allowed_years: {sorted(list(allowed_years_set)) if isinstance(allowed_years_set, set) else allowed_years_set}")
        logging.info(f"  - contains 2080 in combined_df: {2080 in years_in_data}")
        logging.info(f"  - contains 2080 in allowed_years: {2080 in allowed_years_set}")
        
        before_len = len(combined_df)
        combined_df = combined_df[combined_df['year'].isin(allowed_years_set)].copy()
        after_len = len(combined_df)
        if before_len != after_len:
            print(f"[DEBUG] Year filter: {before_len} -> {after_len} rows (removed {before_len-after_len})")
            print(f"  - removed years: {removed_years}")
    
    # Mark commodity-based rows that should bypass dict_v3 Item_Emis filtering.
    # Rules:
    # 1. A commodity column identifies commodity-based data (FAO direct).
    # 2. Otherwise, a Process column identifies emissions-module data: retain these emissions Items as well.
    # 3. Apply strict emissions-Item filtering in all other cases.
    if 'commodity' in combined_df.columns:
        combined_df['_is_commodity_based'] = True
        n_marked = len(combined_df)
        print(f"[DEBUG] 标记了 {n_marked} 行为commodity-based（FAO商品数据）")
    elif 'process' in combined_df.columns or 'Process' in combined_df.columns:
        # Emissions modules such as GLE/GCE/GOS have a process column and emissions Items (e.g., 'Cattle, dairy').
        # Retain these rows by marking them True to bypass dict_v3 Item_Emis filtering.
        combined_df['_is_commodity_based'] = True
        n_marked = len(combined_df)
        print(f"[DEBUG] 标记了 {n_marked} 行为commodity-based（排放模块数据，包含Process列）")
    else:
        combined_df['_is_commodity_based'] = False
        print(f"[DEBUG] 标记了 {len(combined_df)} 行为emission-based（需要dict_v3 Item_Emis过滤）")
    
    # Standardize column names.
    column_mapping = {
        'country': 'Country',
        'commodity': 'Item',  # Preserve the original logic: rename commodity to Item.
        'process': 'Process',
        'ch4_kt': 'CH4_kt',
        'n2o_kt': 'N2O_kt',
        'co2_kt': 'CO2_kt',
        'CH4_kt': 'CH4_kt',  # GLE returns the uppercase column CH4_kt.
        'N2O_kt': 'N2O_kt',  # Support uppercase column names.
        'CO2_kt': 'CO2_kt',  # Support uppercase column names.
        'emissions_kt': 'value'
    }
    combined_df = combined_df.rename(columns={k: v for k, v in column_mapping.items() if k in combined_df.columns})
    
    # Ensure Process values are scalar strings, not Series objects.
    if 'Process' in combined_df.columns:
        # Check for any Series-valued cells.
        mask = combined_df['Process'].apply(lambda x: isinstance(x, pd.Series))
        if mask.any():
            # Flatten Series when present.
            combined_df['Process'] = combined_df['Process'].apply(
                lambda x: str(x.iloc[0]) if isinstance(x, pd.Series) and len(x) > 0 else str(x)
            )
        else:
            # Convert directly to strings.
            combined_df['Process'] = combined_df['Process'].astype(str)
    
    # Ensure Item values are scalars, not Series objects or lists.
    if 'Item' in combined_df.columns:
        # Check for any Series- or list-valued cells.
        def flatten_item(x):
            if isinstance(x, pd.Series):
                return str(x.iloc[0]) if len(x) > 0 else ''
            elif isinstance(x, (list, tuple)):
                return str(x[0]) if len(x) > 0 else ''
            else:
                return str(x)
        
        has_non_scalar = combined_df['Item'].apply(lambda x: isinstance(x, (pd.Series, list, tuple))).any()
        if has_non_scalar:
            combined_df['Item'] = combined_df['Item'].apply(flatten_item)
        else:
            combined_df['Item'] = combined_df['Item'].astype(str)
    
    # Ensure other key columns (Country, M49_Country_Code, GHG, etc.) also contain scalars.
    for col in ['Country', 'M49_Country_Code', 'GHG', 'year']:
        if col in combined_df.columns:
            def flatten_col(x):
                if isinstance(x, pd.Series):
                    return x.iloc[0] if len(x) > 0 else (None if col == 'year' else '')
                elif isinstance(x, (list, tuple)):
                    return x[0] if len(x) > 0 else (None if col == 'year' else '')
                else:
                    return x
            
            has_non_scalar = combined_df[col].apply(lambda x: isinstance(x, (pd.Series, list, tuple))).any()
            if has_non_scalar:
                combined_df[col] = combined_df[col].apply(flatten_col)
            elif col != 'year':  # year is usually numeric.
                try:
                    combined_df[col] = combined_df[col].astype(str)
                except:
                    pass
    
    # Diagnostic: inspect the combined_df structure.
    print(f"\n[DEBUG] Combined DataFrame 结构:")
    print(f"  - 形状: {combined_df.shape}")
    print(f"  - 列名: {list(combined_df.columns)}")
    process_values = _unique_column_values(combined_df, 'Process')
    if not combined_df.empty:
        print(f"  - Process唯一值: {process_values[:10] if process_values.size else '无Process列'}")
        print(f"  - 年份范围: {combined_df['year'].min()}-{combined_df['year'].max() if 'year' in combined_df.columns else '无year列'}")
        # Inspect gas-column values.
        for gas_col in ['CH4_kt', 'N2O_kt', 'CO2_kt', 'ch4_kt', 'n2o_kt', 'co2_kt']:
            if gas_col in combined_df.columns:
                non_zero = int((combined_df[gas_col] > 0).sum())
                total_val = float(combined_df[gas_col].sum())
                print(f"  - {gas_col}: {non_zero}个非零值, 总计={total_val:.2f}")
        # Inspect null values in Country and Item.
        if 'Country' in combined_df.columns:
            country_null_count = combined_df['Country'].isna().sum()
            print(f"  - Country空值: {country_null_count}/{len(combined_df)}")
        if 'Item' in combined_df.columns:
            item_null_count = combined_df['Item'].isna().sum()
            print(f"  - Item空值: {item_null_count}/{len(combined_df)}")
    
    # Standardize column names: rename lowercase gas columns to uppercase format, avoiding duplicates.
    col_rename_map = {
        'ch4_kt': 'CH4_kt',
        'n2o_kt': 'N2O_kt', 
        'co2_kt': 'CO2_kt',
        'country': 'Country'
    }
    combined_df = combined_df.rename(columns=col_rename_map)
    
    # Drop fully duplicated columns, if any.
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
    
    # Handle mixed formats (wide: CH4_kt/N2O_kt/CO2_kt; long: GHG/value).
    has_gas_cols = any(col in combined_df.columns for col in ['CH4_kt', 'N2O_kt', 'CO2_kt'])
    has_ghg_value = ('GHG' in combined_df.columns and 'value' in combined_df.columns)
    if has_gas_cols and has_ghg_value:
        # Mixed formats: handle wide-format GLE/GCE and long-format LUC data separately.
        import logging
        logging.info(f"[INFO] 检测到混合格式数据（wide+long），分别处理...")
        
        # 1. Identify rows already in long format.
        # Use module to help identify long-format rows so future LUC rows with value=NaN are not classified as wide format.
        if 'module' in combined_df.columns:
            # Always treat LUC-module data as long format.
            is_luc = combined_df['module'].astype(str).isin(['LUC'])
            # Treat other modules as long format when both GHG and value are populated.
            is_other_long = combined_df['GHG'].notna() & combined_df['value'].notna()
            is_long = is_luc | is_other_long
            logging.info(f"  - 使用module列识别Long格式: LUC={is_luc.sum()}, Other_Long={is_other_long.sum()}, Total_Long={is_long.sum()}")
        else:
            is_long = combined_df['GHG'].notna() & combined_df['value'].notna()
            logging.info(f"  - 使用GHG/value列识别Long格式: Total_Long={is_long.sum()}")
            
        long_data = combined_df[is_long].copy()
        wide_data = combined_df[~is_long].copy()
        logging.info(f"  - Long格式（LUC等）: {len(long_data)} 行")
        logging.info(f"  - Wide格式（GLE/GCE等）: {len(wide_data)} 行")
        # Inspect the year distribution in long_data.
        if 'year' in long_data.columns and not long_data.empty:
            long_years = sorted(long_data['year'].unique())
            logging.info(f"  - Long格式年份: {long_years}")
            logging.info(f"  - Long格式包含2080: {2080 in long_years}")
            if 'Process' in long_data.columns:
                long_processes = long_data['Process'].value_counts().to_dict()
                logging.info(f"  - Long格式Process分布: {long_processes}")
        # 2. Convert wide format to long format.
        # Retain the original units (kt); do not convert to CO2eq here.
        # Apply GWP conversion centrally in generate_detailed_summary to avoid double conversion.
        gas_dfs = []
        if len(wide_data) > 0:
            non_gas_cols = [col for col in wide_data.columns if col not in ['CH4_kt', 'N2O_kt', 'CO2_kt', 'value', 'GHG']]
            if 'CH4_kt' in wide_data.columns:
                ch4_df = wide_data[non_gas_cols + ['CH4_kt']].copy()
                ch4_df['GHG'] = 'CH4'
                ch4_df['value'] = ch4_df['CH4_kt']  # Retain kt units; do not multiply by GWP.
                ch4_df = ch4_df.drop(columns=['CH4_kt'])
                gas_dfs.append(ch4_df)
            if 'N2O_kt' in wide_data.columns:
                n2o_df = wide_data[non_gas_cols + ['N2O_kt']].copy()
                n2o_df['GHG'] = 'N2O'
                n2o_df['value'] = n2o_df['N2O_kt']  # Retain kt units; do not multiply by GWP.
                n2o_df = n2o_df.drop(columns=['N2O_kt'])
                gas_dfs.append(n2o_df)
            if 'CO2_kt' in wide_data.columns:
                co2_df = wide_data[non_gas_cols + ['CO2_kt']].copy()
                co2_df['GHG'] = 'CO2'
                co2_df['value'] = co2_df['CO2_kt']  # Retain kt units.
                co2_df = co2_df.drop(columns=['CO2_kt'])
                gas_dfs.append(co2_df)
        # 3. Combine existing long-format data with converted wide-format data.
        all_dfs = [long_data] + gas_dfs
        combined_df = pd.concat(all_dfs, ignore_index=True)
        logging.info(f"  - 合并后总行数: {len(combined_df)}")
        if 'Process' in combined_df.columns:
            logging.info(f"  - 合并后Process分布: {combined_df['Process'].value_counts().to_dict()}")
    elif has_gas_cols:
        # For wide-format data only, follow the original logic.
        # Retain the original units (kt); do not convert to CO2eq here.
        gas_dfs = []
        non_gas_cols = [col for col in combined_df.columns if col not in ['CH4_kt', 'N2O_kt', 'CO2_kt', 'value', 'GHG']]
        if 'CH4_kt' in combined_df.columns:
            ch4_df = combined_df[non_gas_cols + ['CH4_kt']].copy()
            ch4_df['GHG'] = 'CH4'
            ch4_df['value'] = ch4_df['CH4_kt']  # Retain kt units; do not multiply by GWP.
            ch4_df = ch4_df.drop(columns=['CH4_kt'])
            gas_dfs.append(ch4_df)
        if 'N2O_kt' in combined_df.columns:
            n2o_df = combined_df[non_gas_cols + ['N2O_kt']].copy()
            n2o_df['GHG'] = 'N2O'
            n2o_df['value'] = n2o_df['N2O_kt']  # Retain kt units; do not multiply by GWP.
            n2o_df = n2o_df.drop(columns=['N2O_kt'])
            gas_dfs.append(n2o_df)
        if 'CO2_kt' in combined_df.columns:
            co2_df = combined_df[non_gas_cols + ['CO2_kt']].copy()
            co2_df['GHG'] = 'CO2'
            co2_df['value'] = co2_df['CO2_kt']  # Retain kt units.
            co2_df = co2_df.drop(columns=['CO2_kt'])
            gas_dfs.append(co2_df)
        if gas_dfs:
            combined_df = pd.concat(gas_dfs, ignore_index=True)
            print(f"\n[DEBUG] 气体拆分后:")
            print(f"  - 形状: {combined_df.shape}")
            if 'value' in combined_df.columns:
                non_zero = (combined_df['value'] > 0).sum()
                total_val = combined_df['value'].sum()
                print(f"  - value: {non_zero}个非零值, 总计={total_val:.2f}")
            if 'GHG' in combined_df.columns:
                print(f"  - GHG分布: {combined_df['GHG'].value_counts().to_dict()}")
        else:
            if 'GHG' not in combined_df.columns:
                combined_df['GHG'] = 'Mixed'
            if 'value' not in combined_df.columns:
                combined_df['value'] = 0.0
    elif 'value' not in combined_df.columns:
        if 'GHG' not in combined_df.columns:
            combined_df['GHG'] = 'CO2eq'
    if 'value' not in combined_df.columns:
        combined_df['value'] = 0.0
    
    # Call the unified summary function with dict_v3_path and allowed_years.
    # Convert a set to a list if needed.
    allowed_years_list = sorted(allowed_years_set) if allowed_years_set else None
    return summarize_emissions_from_detail(combined_df, 
                                          process_meta_map=process_meta_map,
                                          allowed_years=allowed_years_list,
                                          dict_v3_path=dict_v3_path,
                                          production_df=production_df)


def summarize_market(model,
                     var,
                     universe,
                     data=None,
                     price_df: Optional[pd.DataFrame] = None,
                     demand_components_hist: Optional[pd.DataFrame] = None,
                     feed_sim_df: Optional[pd.DataFrame] = None,
                     residual_sim_df: Optional[pd.DataFrame] = None,
                     bioenergy_sim_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Extract supply and demand data from model results, including M49_Country_Code.
    """
    rows = []
    try:
        Qs = var['Qs']; Qd = var['Qd']
    except Exception:
        return pd.DataFrame(columns=['M49_Country_Code','Region_label_new','year','commodity','Qs','Qd','bioenergy_demand_t','market_total_use_t','food_t','feed_t','residual_t','net_import_t','net_export_t','price_global'])

    Pc = var.get('Pc', {})
    Pnet = var.get('Pnet', {})
    W = var.get('W', {})
    Cij = var.get('C', {})
    Eij = var.get('E', {})
    
    # Build the country-to-M49 mapping.
    country_to_m49 = {}
    if data and hasattr(data, 'nodes'):
        for node in data.nodes:
            if hasattr(node, 'country') and hasattr(node, 'm49'):
                country = str(node.country).strip()
                m49 = node.m49
                if m49:
                    # Standardize M49 codes to the 'xxx format.
                    m49_norm = _norm_m49_code(m49)
                    if m49_norm:
                        country_to_m49[country] = m49_norm

    # Build the M49 -> Region_label_new mapping.
    region_mapping: Dict[str, str] = {}
    dict_v3_path = os.path.join(get_src_base(), 'dict_v3.xlsx')
    if os.path.exists(dict_v3_path):
        try:
            region_df = read_excel_cached(dict_v3_path, sheet_name='region',
                                      usecols=['M49_Country_Code', 'Region_label_new'])
            region_df = region_df.dropna(subset=['M49_Country_Code'])
            region_df['M49_Country_Code'] = region_df['M49_Country_Code'].apply(_norm_m49_code)
            region_mapping = dict(zip(region_df['M49_Country_Code'], region_df['Region_label_new']))
        except Exception:
            region_mapping = {}
    
    def _safe_float(val, default=np.nan):
        try:
            if val is None:
                return default
            out = float(val)
            if np.isnan(out):
                return default
            return out
        except Exception:
            return default

    def _get_x(v):
        try:
            return float(v.X)
        except Exception:
            return np.nan

    hist_end_year = 2020
    if data is not None and hasattr(data, 'config'):
        try:
            hist_end_year = int(getattr(data.config, 'years_hist_end', 2020) or 2020)
        except Exception:
            hist_end_year = 2020

    # Track added rows with a set to prevent duplicates.
    seen = set()
    bioenergy_lookup: Dict[Tuple[str, str, int], float] = {}
    if isinstance(bioenergy_sim_df, pd.DataFrame) and not bioenergy_sim_df.empty:
        bio = bioenergy_sim_df.copy()
        required = {'country', 'year', 'commodity', 'bioenergy_demand_t'}
        if required.issubset(bio.columns):
            bio['year'] = pd.to_numeric(bio['year'], errors='coerce')
            bio['bioenergy_demand_t'] = pd.to_numeric(
                bio['bioenergy_demand_t'],
                errors='coerce',
            ).fillna(0.0)
            bio = bio.dropna(subset=['country', 'year', 'commodity'])
            bio['year'] = bio['year'].astype(int)
            bio = bio.groupby(
                ['country', 'commodity', 'year'],
                as_index=False,
            )['bioenergy_demand_t'].sum()
            bioenergy_lookup = {
                (str(r.country), str(r.commodity), int(r.year)): float(r.bioenergy_demand_t)
                for r in bio.itertuples(index=False)
            }
    
    for (i, j, t), svar in Qs.items():
        q_supply = _get_x(svar)
        dvar = Qd.get((i, j, t))
        q_demand = _get_x(dvar)
        q_bioenergy = float(bioenergy_lookup.get((str(i), str(j), int(t)), 0.0) or 0.0)
        # Historical Qd is observed domestic supply and already contains
        # bioenergy. Future Qd excludes explicit crop-bioenergy demand.
        q_total_use = q_demand if int(t) <= hist_end_year else q_demand + q_bioenergy
        imp = max(q_total_use - q_supply, 0.0) if np.isfinite(q_supply) and np.isfinite(q_total_use) else np.nan
        exp = max(q_supply - q_total_use, 0.0) if np.isfinite(q_supply) and np.isfinite(q_total_use) else np.nan
        price = Pc.get((i, j, t))
        if price is None:
            price = Pc.get((j, t))
        price_val = _get_x(price)
        price_net = Pnet.get((i, j, t))
        price_net_val = _get_x(price_net)
        
        m49 = country_to_m49.get(i, '')
        key = (i, t, j)  # Used for deduplication.
        if key not in seen:
            rows.append((m49, i, t, j, q_supply, q_demand, q_bioenergy, q_total_use, imp, exp, price_val, price_net_val))
            seen.add(key)

    import_slack = var.get('Import') or {}
    export_slack = var.get('Export') or {}
    slack_keys = set()
    if isinstance(import_slack, dict):
        slack_keys.update(import_slack.keys())
    if isinstance(export_slack, dict):
        slack_keys.update(export_slack.keys())
    for key in sorted(slack_keys):
        j, t = key
        imp_var = import_slack.get(key)
        exp_var = export_slack.get(key)
        imp_val = _get_x(imp_var)
        exp_val = _get_x(exp_var)
        if not (np.isfinite(imp_val) or np.isfinite(exp_val)):
            continue
        slack_key = ('ROW', t, j)
        if slack_key not in seen:
            rows.append(('', 'ROW', t, j, np.nan, np.nan, np.nan, np.nan,
                         imp_val if np.isfinite(imp_val) else np.nan,
                         exp_val if np.isfinite(exp_val) else np.nan,
                         np.nan, np.nan))
            seen.add(slack_key)

    # Do not append Q0/D0 data from nodes: the model has already been solved.

    out = pd.DataFrame(rows, columns=['M49_Country_Code','country','year','commodity','Qs','Qd','bioenergy_demand_t','market_total_use_t','net_import_t','net_export_t',
                                      'price_global','price_net'])
    out['food_t'] = np.nan
    out['feed_t'] = np.nan
    out['residual_t'] = np.nan
    if isinstance(demand_components_hist, pd.DataFrame) and not demand_components_hist.empty:
        comp = demand_components_hist.copy()
        required_cols = {'country', 'year', 'commodity'}
        if required_cols.issubset(comp.columns):
            for col in ['food_t', 'feed_t']:
                if col not in comp.columns:
                    comp[col] = 0.0
            comp = comp[['country', 'year', 'commodity', 'food_t', 'feed_t']].copy()
            comp['year'] = pd.to_numeric(comp['year'], errors='coerce')
            comp['food_t'] = pd.to_numeric(comp['food_t'], errors='coerce').fillna(0.0)
            comp['feed_t'] = pd.to_numeric(comp['feed_t'], errors='coerce').fillna(0.0)
            comp = comp.dropna(subset=['country', 'year', 'commodity'])
            comp['year'] = comp['year'].astype(int)
            comp = comp.groupby(['country', 'year', 'commodity'], as_index=False)[['food_t', 'feed_t']].sum()
            if not comp.empty:
                out = out.merge(comp, on=['country', 'year', 'commodity'], how='left', suffixes=('', '_hist'))
                hist_mask = out['year'] <= hist_end_year
                out.loc[hist_mask, 'food_t'] = out.loc[hist_mask, 'food_t_hist']
                out.loc[hist_mask, 'feed_t'] = out.loc[hist_mask, 'feed_t_hist']
                out = out.drop(columns=['food_t_hist', 'feed_t_hist'], errors='ignore')
                hist_qd = pd.to_numeric(out.loc[hist_mask, 'Qd'], errors='coerce')
                hist_food = pd.to_numeric(out.loc[hist_mask, 'food_t'], errors='coerce').fillna(0.0)
                hist_feed = pd.to_numeric(out.loc[hist_mask, 'feed_t'], errors='coerce').fillna(0.0)
                out.loc[hist_mask, 'residual_t'] = (hist_qd - hist_food - hist_feed).clip(lower=0.0)
    if isinstance(feed_sim_df, pd.DataFrame) and not feed_sim_df.empty:
        feed_sim = feed_sim_df.copy()
        required_cols = {'country', 'year', 'commodity', 'feed_t'}
        if required_cols.issubset(feed_sim.columns):
            feed_sim = feed_sim[['country', 'year', 'commodity', 'feed_t']].copy()
            feed_sim['year'] = pd.to_numeric(feed_sim['year'], errors='coerce')
            feed_sim['feed_t'] = pd.to_numeric(feed_sim['feed_t'], errors='coerce')
            feed_sim = feed_sim.dropna(subset=['country', 'year', 'commodity', 'feed_t'])
            feed_sim['year'] = feed_sim['year'].astype(int)
            feed_sim = feed_sim.groupby(['country', 'year', 'commodity'], as_index=False)['feed_t'].sum()
            if not feed_sim.empty:
                out = out.merge(feed_sim, on=['country', 'year', 'commodity'], how='left', suffixes=('', '_sim'))
                future_mask = out['year'] > hist_end_year
                out.loc[future_mask, 'feed_t'] = out.loc[future_mask, 'feed_t_sim']
                out = out.drop(columns=['feed_t_sim'], errors='ignore')
    if isinstance(residual_sim_df, pd.DataFrame) and not residual_sim_df.empty:
        residual_sim = residual_sim_df.copy()
        required_cols = {'country', 'year', 'commodity', 'residual_t'}
        if required_cols.issubset(residual_sim.columns):
            residual_sim = residual_sim[['country', 'year', 'commodity', 'residual_t']].copy()
            residual_sim['year'] = pd.to_numeric(residual_sim['year'], errors='coerce')
            residual_sim['residual_t'] = pd.to_numeric(residual_sim['residual_t'], errors='coerce')
            residual_sim = residual_sim.dropna(subset=['country', 'year', 'commodity', 'residual_t'])
            residual_sim['year'] = residual_sim['year'].astype(int)
            residual_sim = residual_sim.groupby(['country', 'year', 'commodity'], as_index=False)['residual_t'].sum()
            if not residual_sim.empty:
                out = out.merge(residual_sim, on=['country', 'year', 'commodity'], how='left', suffixes=('', '_sim'))
                future_mask = out['year'] > hist_end_year
                out.loc[future_mask, 'residual_t'] = out.loc[future_mask, 'residual_t_sim']
                out = out.drop(columns=['residual_t_sim'], errors='ignore')
    future_mask = out['year'] > hist_end_year
    if future_mask.any():
        out.loc[future_mask, 'feed_t'] = out.loc[future_mask, 'feed_t'].fillna(0.0)
        out.loc[future_mask, 'residual_t'] = out.loc[future_mask, 'residual_t'].fillna(0.0)
        out.loc[future_mask, 'food_t'] = np.where(
            np.isfinite(out.loc[future_mask, 'Qd']),
            out.loc[future_mask, 'Qd'] - out.loc[future_mask, 'feed_t'] - out.loc[future_mask, 'residual_t'],
            np.nan
        )
    if price_df is not None and len(price_df):
        out = out.merge(price_df[['country','year','commodity','price']],
                        on=['country','year','commodity'], how='left')
    if not out.empty:
        mask = out['commodity'].astype(str).str.strip().str.lower().isin({'1', '2', 'nan'})
        out = out[~mask].reset_index(drop=True)
        out['M49_Country_Code'] = out['M49_Country_Code'].apply(_norm_m49_code)
        out['Region_label_new'] = out['M49_Country_Code'].map(region_mapping)
        out.loc[out['country'] == 'ROW', 'Region_label_new'] = 'ROW'
        out['Region_label_new'] = out['Region_label_new'].fillna('Unknown')
        out = out.drop(columns=['country'])
    return out


_DATABASE_STRATEGY_TO_KIND = {
    'RuminantReduction': 'ruminant_reduction',
    'LossWaste': 'losses_ratio',
    'YieldRate': 'yield_rate',
    'FeedEfficiency': 'feed_intensity',
    'EntericF': 'enteric_fermentation_management',
    'Manure': 'manure_management',
    'Residue': 'crop_residue_soil_management',
    'Rice': 'rice_cultivation',
    'Fertilizer': 'fertilizer_efficiency',
}


_COST_SUMMARY_COLUMNS = [
    'region', 'commodity', 'year', 'process', 'segment',
    'cost_component_type', 'database_strategy', 'strategy_kind',
    'cost_database_version', 'cost_database_sha256', 'cost_basis',
    'is_priced', 'unpriced_reason', 'attribution_method',
    'reference_scenario_id',
    'abatement_native', 'abatement_native_unit', 'abatement_ktco2eq',
    'priced_reduction_native', 'priced_reduction_native_unit',
    'priced_reduction_ktco2eq', 'priced_reduction_tco2eq',
    'abatement_tco2eq', 'unit_cost_usd_per_tco2eq', 'total_cost_usd',
]

COUNTRY_MEASURE_COST_SUMMARY_FILENAME = 'cost_summary_by_country_measure.csv'
GLOBAL_MEASURE_COST_SUMMARY_FILENAME = 'cost_summary_by_global_measure.csv'

_MEASURE_COST_ID_COLUMNS = [
    'scenario_id', 'year', 'database_strategy', 'strategy_kind',
    'cost_database_version', 'cost_database_sha256', 'reference_scenario_id',
]
_MEASURE_COST_VALUE_COLUMNS = [
    'detail_row_count', 'priced_row_count', 'unpriced_row_count',
    'is_fully_priced', 'cost_component_types', 'cost_basis',
    'attribution_method', 'unpriced_reasons', 'abatement_tco2eq',
    'priced_reduction_tco2eq', 'unpriced_abatement_tco2eq',
    'pricing_coverage_ratio', 'unit_cost_usd_per_tco2eq',
    'total_cost_usd', 'recomputed_total_cost_usd',
    'cost_identity_difference_usd',
]
COUNTRY_MEASURE_COST_SUMMARY_COLUMNS = [
    'scenario_id', 'region', 'M49_Country_Code', 'ISO3', 'Country',
    'Region_label_new', 'year', 'database_strategy', 'strategy_kind',
    'cost_database_version', 'cost_database_sha256', 'reference_scenario_id',
    *_MEASURE_COST_VALUE_COLUMNS,
]
GLOBAL_MEASURE_COST_SUMMARY_COLUMNS = [
    'scenario_id', 'M49_Country_Code', 'ISO3', 'Country',
    'Region_label_new', 'year', 'database_strategy', 'strategy_kind',
    'cost_database_version', 'cost_database_sha256', 'reference_scenario_id',
    'country_count', *_MEASURE_COST_VALUE_COLUMNS,
]


def _cost_summary_bool_value(raw: Any) -> bool:
    if raw is None:
        return False
    try:
        if bool(pd.isna(raw)):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(raw, str):
        return raw.strip().lower() in {'1', 'true', 'yes', 'y'}
    return bool(raw)


def _cost_summary_join_unique(values: pd.Series) -> str:
    unique_values = set()
    for value in values:
        if value is None:
            continue
        try:
            if bool(pd.isna(value)):
                continue
        except (TypeError, ValueError):
            pass
        text = str(value).strip()
        if text:
            unique_values.add(text)
    unique = sorted(unique_values)
    return ';'.join(unique)


def _cost_summary_region_value(raw: Any) -> str:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ''
    text = str(raw).strip()
    digits = text[1:] if text.startswith("'") else text
    if digits.count('.') == 1:
        left, right = digits.split('.', 1)
        if left.isdigit() and not right.strip('0'):
            digits = left
    return _norm_m49_code(digits) if digits.isdigit() else text


def _cost_summary_region_to_m49(raw: Any) -> str:
    region = _cost_summary_region_value(raw)
    digits = region[1:] if region.startswith("'") else region
    return _norm_m49_code(digits) if digits.isdigit() else ''


def _load_cost_country_metadata(dict_v3_path: Optional[str]) -> pd.DataFrame:
    path = Path(dict_v3_path) if dict_v3_path else Path(get_src_base()) / 'dict_v3.xlsx'
    columns = ['M49_Country_Code', 'ISO3', 'Country', 'Region_label_new']
    if not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        metadata = read_excel_cached(
            str(path),
            sheet_name='region',
            usecols=['M49_Country_Code', 'ISO3 Code', 'NAME', 'Region_label_new'],
        ).rename(columns={'ISO3 Code': 'ISO3', 'NAME': 'Country'})
    except (FileNotFoundError, KeyError, ValueError):
        return pd.DataFrame(columns=columns)
    metadata['M49_Country_Code'] = metadata['M49_Country_Code'].map(_norm_m49_code)
    return (
        metadata[columns]
        .dropna(subset=['M49_Country_Code'])
        .drop_duplicates(subset=['M49_Country_Code'], keep='first')
    )


def _prepare_measure_cost_detail(
    cost_summary_df: pd.DataFrame,
    *,
    scenario_id: str,
) -> pd.DataFrame:
    required = {
        'region', 'year', 'database_strategy', 'strategy_kind',
        'cost_component_type', 'cost_database_version',
        'cost_database_sha256', 'cost_basis', 'is_priced',
        'unpriced_reason', 'attribution_method', 'reference_scenario_id',
        'abatement_tco2eq', 'priced_reduction_tco2eq',
        'unit_cost_usd_per_tco2eq', 'total_cost_usd',
    }
    work = cost_summary_df.copy()
    for column in required.difference(work.columns):
        work[column] = np.nan if column.endswith(('_tco2eq', '_usd')) else ''
    if work.empty:
        return work

    work['scenario_id'] = str(scenario_id or '')
    work['region'] = work['region'].map(_cost_summary_region_value)
    work['database_strategy'] = work['database_strategy'].fillna('').astype(str).str.strip()
    work = work[work['database_strategy'] != ''].copy()
    if work.empty:
        return work

    inferred_kinds = work['database_strategy'].map(_DATABASE_STRATEGY_TO_KIND).fillna('')
    work['strategy_kind'] = work['strategy_kind'].fillna('').astype(str).str.strip()
    work.loc[work['strategy_kind'] == '', 'strategy_kind'] = inferred_kinds
    for column in (
        'cost_component_type', 'cost_database_version',
        'cost_database_sha256', 'cost_basis', 'unpriced_reason',
        'attribution_method', 'reference_scenario_id',
    ):
        work[column] = work[column].fillna('').astype(str).str.strip()
    work['year'] = pd.to_numeric(work['year'], errors='coerce').astype('Int64')
    for column in (
        'abatement_tco2eq', 'priced_reduction_tco2eq',
        'unit_cost_usd_per_tco2eq', 'total_cost_usd',
    ):
        work[column] = pd.to_numeric(work[column], errors='coerce').fillna(0.0)

    work['_is_priced'] = work['is_priced'].map(_cost_summary_bool_value)
    work['_priced_reduction_tco2eq'] = np.where(
        work['_is_priced'], work['priced_reduction_tco2eq'], 0.0
    )
    work['_unpriced_abatement_tco2eq'] = np.where(
        work['_is_priced'], 0.0, work['abatement_tco2eq']
    )
    work['_reported_total_cost_usd'] = np.where(
        work['_is_priced'], work['total_cost_usd'], 0.0
    )
    work['_recomputed_total_cost_usd'] = np.where(
        work['_is_priced'],
        work['priced_reduction_tco2eq'] * work['unit_cost_usd_per_tco2eq'],
        0.0,
    )
    return work


def _assert_measure_cost_component_exclusivity(work: pd.DataFrame) -> None:
    if work.empty:
        return
    key_columns = [
        'scenario_id', 'region', 'year', 'database_strategy', 'strategy_kind',
        'cost_database_version', 'cost_database_sha256', 'reference_scenario_id',
    ]
    component_counts = (
        work[work['cost_component_type'] != '']
        .groupby(key_columns, dropna=False)['cost_component_type']
        .nunique()
    )
    invalid = component_counts[component_counts > 1]
    if invalid.empty:
        return
    examples = [
        '|'.join(str(part) for part in key)
        for key in list(invalid.index[:5])
    ]
    raise ValueError(
        'Country-year-measure cost rows mix process and strategy components; '
        'refusing to create a double-counted summary. Examples: '
        + ', '.join(examples)
    )


def _aggregate_measure_cost_rows(
    work: pd.DataFrame,
    *,
    group_columns: List[str],
    include_country_count: bool,
) -> pd.DataFrame:
    if work.empty:
        return pd.DataFrame()
    named_aggregations: Dict[str, Any] = {
        'detail_row_count': ('database_strategy', 'size'),
        'priced_row_count': ('_is_priced', 'sum'),
        'cost_component_types': ('cost_component_type', _cost_summary_join_unique),
        'cost_basis': ('cost_basis', _cost_summary_join_unique),
        'attribution_method': ('attribution_method', _cost_summary_join_unique),
        'unpriced_reasons': ('unpriced_reason', _cost_summary_join_unique),
        'abatement_tco2eq': ('abatement_tco2eq', 'sum'),
        'priced_reduction_tco2eq': ('_priced_reduction_tco2eq', 'sum'),
        'unpriced_abatement_tco2eq': ('_unpriced_abatement_tco2eq', 'sum'),
        'total_cost_usd': ('_reported_total_cost_usd', 'sum'),
        'recomputed_total_cost_usd': ('_recomputed_total_cost_usd', 'sum'),
    }
    if include_country_count:
        named_aggregations['country_count'] = ('region', 'nunique')
    summary = (
        work.groupby(group_columns, as_index=False, dropna=False)
        .agg(**named_aggregations)
        .reset_index(drop=True)
    )
    summary['detail_row_count'] = summary['detail_row_count'].astype(int)
    summary['priced_row_count'] = summary['priced_row_count'].astype(int)
    summary['unpriced_row_count'] = (
        summary['detail_row_count'] - summary['priced_row_count']
    ).astype(int)
    summary['is_fully_priced'] = summary['unpriced_row_count'].eq(0)
    positive_physical = summary['abatement_tco2eq'] > 0.0
    summary['pricing_coverage_ratio'] = np.where(
        positive_physical,
        summary['priced_reduction_tco2eq'] / summary['abatement_tco2eq'],
        np.nan,
    )
    positive_priced = summary['priced_reduction_tco2eq'] > 0.0
    summary['unit_cost_usd_per_tco2eq'] = np.where(
        positive_priced,
        summary['total_cost_usd'] / summary['priced_reduction_tco2eq'],
        np.nan,
    )
    summary['cost_identity_difference_usd'] = (
        summary['total_cost_usd'] - summary['recomputed_total_cost_usd']
    )
    identity_tolerance = np.maximum(
        1e-6,
        summary['recomputed_total_cost_usd'].abs() * 1e-12,
    )
    summary.loc[
        summary['cost_identity_difference_usd'].abs() <= identity_tolerance,
        'cost_identity_difference_usd',
    ] = 0.0
    return summary


def build_measure_cost_summaries(
    cost_summary_df: pd.DataFrame,
    *,
    scenario_id: str = '',
    dict_v3_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate the priced-cost ledger to country-measure and global-measure rows."""
    work = _prepare_measure_cost_detail(cost_summary_df, scenario_id=scenario_id)
    if work.empty:
        return (
            pd.DataFrame(columns=COUNTRY_MEASURE_COST_SUMMARY_COLUMNS),
            pd.DataFrame(columns=GLOBAL_MEASURE_COST_SUMMARY_COLUMNS),
        )
    _assert_measure_cost_component_exclusivity(work)

    country_group_columns = ['scenario_id', 'region', *_MEASURE_COST_ID_COLUMNS[1:]]
    country = _aggregate_measure_cost_rows(
        work,
        group_columns=country_group_columns,
        include_country_count=False,
    )
    country['M49_Country_Code'] = country['region'].map(_cost_summary_region_to_m49)
    metadata = _load_cost_country_metadata(dict_v3_path)
    if not metadata.empty:
        country = country.merge(metadata, on='M49_Country_Code', how='left')
    else:
        country['ISO3'] = ''
        country['Country'] = ''
        country['Region_label_new'] = ''
    for column in ('ISO3', 'Country', 'Region_label_new'):
        country[column] = country[column].fillna('')
    country = country.reindex(columns=COUNTRY_MEASURE_COST_SUMMARY_COLUMNS)
    country = country.sort_values(
        ['scenario_id', 'year', 'database_strategy', 'M49_Country_Code', 'region'],
        kind='stable',
    ).reset_index(drop=True)

    global_group_columns = list(_MEASURE_COST_ID_COLUMNS)
    global_summary = _aggregate_measure_cost_rows(
        work,
        group_columns=global_group_columns,
        include_country_count=True,
    )
    global_summary['M49_Country_Code'] = "'000"
    global_summary['ISO3'] = 'WLD'
    global_summary['Country'] = 'Global'
    global_summary['Region_label_new'] = 'Global'
    global_summary = global_summary.reindex(columns=GLOBAL_MEASURE_COST_SUMMARY_COLUMNS)
    global_summary = global_summary.sort_values(
        ['scenario_id', 'year', 'database_strategy'], kind='stable'
    ).reset_index(drop=True)
    return country, global_summary


def write_measure_cost_summaries(
    cost_summary_df: pd.DataFrame,
    *,
    output_dir: str | Path,
    scenario_id: str = '',
    dict_v3_path: Optional[str] = None,
) -> Dict[str, Path]:
    """Write country/global measure-cost summaries beside the detailed ledger."""
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    country, global_summary = build_measure_cost_summaries(
        cost_summary_df,
        scenario_id=scenario_id,
        dict_v3_path=dict_v3_path,
    )
    country_path = target_dir / COUNTRY_MEASURE_COST_SUMMARY_FILENAME
    global_path = target_dir / GLOBAL_MEASURE_COST_SUMMARY_FILENAME
    country.to_csv(country_path, index=False, encoding='utf-8-sig')
    global_summary.to_csv(global_path, index=False, encoding='utf-8-sig')
    return {'country_measure': country_path, 'global_measure': global_path}


def _cost_summary_solver_value(raw: Any, *, default: float = 0.0) -> float:
    """Read a solved scalar from a Gurobi variable, mapping, or numeric value."""
    if isinstance(raw, Mapping):
        for field in (
            'abatement_tco2eq', 'abatement', 'value', 'var',
            'abatement_var', 'strategy_abatement_var',
        ):
            if field in raw:
                return _cost_summary_solver_value(raw[field], default=default)
        return float(default)
    if raw is None:
        return float(default)
    try:
        if hasattr(raw, 'X'):
            return float(raw.X)
        return float(raw)
    except (AttributeError, TypeError, ValueError):
        return float(default)


def _cost_summary_lookup(mapping: Any, key: Any, *fallback_keys: Any) -> Any:
    if not isinstance(mapping, Mapping):
        return None
    for candidate in (key, *fallback_keys):
        try:
            if candidate in mapping:
                return mapping[candidate]
        except TypeError:
            continue
    return None


def _cost_summary_metadata_value(
    primary: Any,
    secondary: Any,
    fields: Tuple[str, ...],
    default: Any = '',
) -> Any:
    for source in (primary, secondary):
        if not isinstance(source, Mapping):
            continue
        for field in fields:
            value = source.get(field)
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                return value
    return default


def _cost_summary_unit_cost(raw: Any) -> Tuple[float, bool]:
    if isinstance(raw, Mapping):
        for field in (
            'unit_cost_usd_per_tco2eq', 'unit_cost_usd_tco2e',
            'unit_cost', 'cost_usd_per_tco2eq', 'cost',
        ):
            if field in raw:
                return _cost_summary_unit_cost(raw[field])
        return 0.0, False
    if raw is None:
        return 0.0, False
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0, False
    return (value, bool(np.isfinite(value) and value >= 0.0))


def _cost_summary_bool(raw: Any, default: bool) -> bool:
    if raw is None:
        return bool(default)
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in {'1', 'true', 'yes', 'y'}:
            return True
        if text in {'0', 'false', 'no', 'n'}:
            return False
    return bool(raw)


def generate_cost_summary(var: Dict, unit_cost_data: Dict, baseline_scenario_result: Dict,
                         output_path: str, regions: List[str], commodities: List[str],
                         years: List[int], *,
                         cost_database_version: Optional[str] = None,
                         cost_database_sha256: Optional[str] = None,
                         attribution_method: Optional[str] = None,
                         reference_scenario_id: Optional[str] = None,
                         scenario_id: Optional[str] = None,
                         dict_v3_path: Optional[str] = None) -> None:
    """
    Generate an abatement-cost summary table.
    
    Extract abatement quantities and costs from model solutions and write a summary CSV.
    
    Args:
        var: Model-variable dictionary containing Qs (supply), abatement_vars (abatement quantities), and abatement_costs (unit costs).
        unit_cost_data: Unit-cost data {(region, process): cost_per_tco2eq}.
        baseline_scenario_result: BASE scenario results {'Qs': {...}}.
        output_path: Output CSV file path.
        regions: List of regions.
        commodities: List of commodities.
        years: List of years.
    """
    import pandas as pd
    
    print("\n" + "=" * 80)
    print("生成减排成本汇总表")
    print("=" * 80)
    
    # Extract variables. strategy_abatement_blocks uses three-part keys
    # (region, year, database_strategy), with tCO2e as the common unit.
    abatement_vars = var.get('abatement_vars', {})
    abatement_cost_vars = var.get('abatement_cost_vars', {})
    abatement_costs_map = var.get('abatement_costs', {})
    abatement_database_keys = var.get('abatement_database_keys', {})
    strategy_abatement_blocks = (
        var.get('strategy_abatement_blocks')
        or var.get('strategy_abatement_vars')
        or var.get('strategy_abatement')
        or {}
    )
    strategy_abatement_cost_vars = var.get('strategy_abatement_cost_vars', {})
    strategy_abatement_costs = var.get(
        'strategy_abatement_costs', var.get('strategy_costs', {})
    )
    strategy_abatement_metadata = var.get(
        'strategy_abatement_metadata', var.get('strategy_cost_metadata', {})
    )
    database_metadata = var.get('cost_database_metadata', {})
    database_version = str(
        cost_database_version
        or var.get('cost_database_version')
        or _cost_summary_metadata_value(
            database_metadata, None, ('database_version', 'version'), ''
        )
        or ''
    )
    database_sha256 = str(
        cost_database_sha256
        or var.get('cost_database_sha256')
        or _cost_summary_metadata_value(
            database_metadata, None, ('cost_database_sha256', 'sha256'), ''
        )
        or ''
    )
    default_attribution = str(
        attribution_method
        or var.get('attribution_method')
        or _cost_summary_metadata_value(
            database_metadata, None, ('attribution_method',), ''
        )
        or ''
    )
    default_reference = str(
        reference_scenario_id
        or var.get('reference_scenario_id')
        or _cost_summary_metadata_value(
            database_metadata, None, ('reference_scenario_id',), ''
        )
        or ''
    )
    
    if not abatement_vars and not abatement_cost_vars and not strategy_abatement_blocks:
        print("[WARN] 未找到减排量变量，导出空成本汇总文件")
        empty_df = pd.DataFrame(columns=_COST_SUMMARY_COLUMNS)
        empty_df.to_csv(output_path, index=False)
        write_measure_cost_summaries(
            empty_df,
            output_dir=Path(output_path).parent,
            scenario_id=str(scenario_id or ''),
            dict_v3_path=dict_v3_path,
        )
        print(f"  [OK] 空成本汇总已导出: {output_path}")
        return
    
    print(f"  - 减排量变量数: {len(abatement_vars)}")
    print(f"  - MACC计价减排变量数: {len(abatement_cost_vars)}")
    print(f"  - 单位成本记录数: {len(abatement_costs_map)}")
    print(f"  - 策略减排块数: {len(strategy_abatement_blocks)}")
    
    # Build result records.
    records = []
    
    all_keys = list(dict.fromkeys([*abatement_vars.keys(), *abatement_cost_vars.keys()]))
    for key in all_keys:
        try:
            if not isinstance(key, tuple):
                continue
            if len(key) == 4:
                region, commodity, year, process = key
                segment = None
            elif len(key) == 5:
                region, commodity, year, process, segment = key
            else:
                continue

            database_strategy = _cost_summary_lookup(
                abatement_database_keys,
                key,
                (region, process),
                process,
            )
            database_strategy = str(database_strategy or '')
            process_meta = _cost_summary_lookup(
                strategy_abatement_metadata,
                database_strategy,
                (region, database_strategy),
            )
            if not isinstance(process_meta, Mapping):
                process_meta = {}
            strategy_kind = str(
                process_meta.get(
                    'strategy_kind', _DATABASE_STRATEGY_TO_KIND.get(database_strategy, '')
                )
                or ''
            )
            process_cost_basis = str(
                process_meta.get(
                    'abatement_quantity_basis', 'process_baseline_minus_current'
                )
                or 'process_baseline_minus_current'
            )
            
            # Four-part unit-cost keys store tCO2e. Five-part MACC segment
            # keys store ktCO2e because they are also used in the emissions
            # equation; convert only the latter before pricing/export.
            abat_var = abatement_vars.get(key, 0.0)
            if hasattr(abat_var, 'X'):
                abatement_native = float(abat_var.X)
            else:
                abatement_native = float(abat_var or 0.0)
            if len(key) == 5:
                abatement_ktco2eq = abatement_native
                abatement = abatement_native * KT_CO2E_TO_T_CO2E
                abatement_unit = 'ktCO2e'
            else:
                abatement_ktco2eq = abatement_native / KT_CO2E_TO_T_CO2E
                abatement = abatement_native
                abatement_unit = 'tCO2e'

            priced_var = abatement_cost_vars.get(key)
            if priced_var is None:
                priced_reduction_native = abatement_native
            elif hasattr(priced_var, 'X'):
                priced_reduction_native = float(priced_var.X)
            else:
                priced_reduction_native = float(priced_var or 0.0)
            if len(key) == 5:
                priced_reduction_ktco2eq = priced_reduction_native
                priced_reduction_tco2eq = (
                    priced_reduction_native * KT_CO2E_TO_T_CO2E
                )
                priced_reduction_native_unit = 'ktCO2e'
            else:
                priced_reduction_ktco2eq = (
                    priced_reduction_native / KT_CO2E_TO_T_CO2E
                )
                priced_reduction_tco2eq = priced_reduction_native
                priced_reduction_native_unit = 'tCO2e'
            
            # Get the unit cost (USD/tCO2eq).
            unit_cost_raw = abatement_costs_map.get(key)
            if unit_cost_raw is None and len(key) == 5:
                unit_cost_raw = abatement_costs_map.get((region, commodity, year, process))
            if unit_cost_raw is None and database_strategy:
                unit_cost_raw = _cost_summary_lookup(
                    unit_cost_data,
                    (region, database_strategy),
                    database_strategy,
                )
            unit_cost, is_priced = _cost_summary_unit_cost(unit_cost_raw)
            
            # Calculate total cost (USD).
            total_cost = priced_reduction_tco2eq * unit_cost
            
            # Preserve an explicitly mapped and priced zero-abatement row.
            # It is part of the strict singleton cost contract and prevents a
            # valid completed run from being treated as permanently resumable-
            # false merely because its physical response is exactly zero.
            has_priced_database_record = bool(database_strategy) and bool(is_priced)
            if (
                max(abs(abatement), abs(priced_reduction_tco2eq)) > 1e-6
                or has_priced_database_record
            ):
                records.append({
                    'region': region,
                    'commodity': commodity,
                    'year': year,
                    'process': process,
                    'segment': segment,
                    'cost_component_type': 'process',
                    'database_strategy': database_strategy,
                    'strategy_kind': strategy_kind,
                    'cost_database_version': database_version,
                    'cost_database_sha256': database_sha256,
                    'cost_basis': process_cost_basis,
                    'is_priced': is_priced,
                    'unpriced_reason': '' if is_priced else 'missing_or_invalid_unit_cost',
                    'attribution_method': default_attribution or 'direct_process',
                    'reference_scenario_id': default_reference,
                    'abatement_native': abatement_native,
                    'abatement_native_unit': abatement_unit,
                    'abatement_ktco2eq': abatement_ktco2eq,
                    'abatement_tco2eq': abatement,
                    'priced_reduction_native': priced_reduction_native,
                    'priced_reduction_native_unit': priced_reduction_native_unit,
                    'priced_reduction_ktco2eq': priced_reduction_ktco2eq,
                    'priced_reduction_tco2eq': priced_reduction_tco2eq,
                    'unit_cost_usd_per_tco2eq': unit_cost,
                    'total_cost_usd': total_cost
                })
        except Exception as e:
            print(f"    [WARN] 提取过程减排数据失败: {key}: {e}")
            continue


    # Strategy-level blocks are deliberately distinct from process rows.  A
    # strict singleton row must never inherit the entire process cost summary.
    if isinstance(strategy_abatement_blocks, Mapping):
        for key, raw_block in strategy_abatement_blocks.items():
            try:
                if not isinstance(key, tuple) or len(key) != 3:
                    continue
                region, year, database_strategy_raw = key
                database_strategy = str(database_strategy_raw or '').strip()
                if not database_strategy:
                    continue
                block_meta = raw_block if isinstance(raw_block, Mapping) else {}
                extra_meta = _cost_summary_lookup(
                    strategy_abatement_metadata,
                    key,
                    (region, database_strategy),
                    database_strategy,
                )
                if not isinstance(extra_meta, Mapping):
                    extra_meta = {}

                abatement = _cost_summary_solver_value(raw_block)
                priced_raw = _cost_summary_lookup(
                    strategy_abatement_cost_vars,
                    key,
                    (region, database_strategy),
                    database_strategy,
                )
                if priced_raw is None:
                    priced_raw = _cost_summary_metadata_value(
                        block_meta,
                        extra_meta,
                        ('priced_reduction_tco2eq', 'priced_reduction', 'costed_abatement_tco2eq'),
                        None,
                    )

                unit_cost_raw = _cost_summary_metadata_value(
                    block_meta,
                    extra_meta,
                    (
                        'unit_cost_usd_per_tco2eq', 'unit_cost_usd_tco2e',
                        'unit_cost', 'cost_usd_per_tco2eq', 'cost',
                    ),
                    None,
                )
                if unit_cost_raw is None:
                    unit_cost_raw = _cost_summary_lookup(
                        strategy_abatement_costs,
                        key,
                        (region, database_strategy),
                        database_strategy,
                    )
                if unit_cost_raw is None:
                    unit_cost_raw = _cost_summary_lookup(
                        unit_cost_data,
                        (region, database_strategy),
                        database_strategy,
                    )
                unit_cost, cost_found = _cost_summary_unit_cost(unit_cost_raw)
                explicit_is_priced = _cost_summary_metadata_value(
                    block_meta, extra_meta, ('is_priced',), None
                )
                is_priced = _cost_summary_bool(explicit_is_priced, cost_found) and cost_found
                priced_reduction = (
                    _cost_summary_solver_value(priced_raw, default=abatement)
                    if is_priced
                    else 0.0
                )
                strategy_kind = str(
                    _cost_summary_metadata_value(
                        block_meta,
                        extra_meta,
                        ('strategy_kind',),
                        _DATABASE_STRATEGY_TO_KIND.get(database_strategy, ''),
                    )
                    or ''
                )
                row_version = str(
                    _cost_summary_metadata_value(
                        block_meta,
                        extra_meta,
                        ('cost_database_version', 'database_version'),
                        database_version,
                    )
                    or ''
                )
                row_sha256 = str(
                    _cost_summary_metadata_value(
                        block_meta,
                        extra_meta,
                        ('cost_database_sha256', 'database_sha256', 'sha256'),
                        database_sha256,
                    )
                    or ''
                )
                row_cost_basis = str(
                    _cost_summary_metadata_value(
                        block_meta,
                        extra_meta,
                        ('cost_basis', 'abatement_quantity_basis'),
                        'strict_singleton_incremental_abatement',
                    )
                    or ''
                )
                row_attribution = str(
                    _cost_summary_metadata_value(
                        block_meta,
                        extra_meta,
                        ('attribution_method',),
                        default_attribution or 'strict_singleton',
                    )
                    or ''
                )
                row_reference = str(
                    _cost_summary_metadata_value(
                        block_meta,
                        extra_meta,
                        ('reference_scenario_id', 'reference'),
                        default_reference,
                    )
                    or ''
                )
                unpriced_reason = str(
                    _cost_summary_metadata_value(
                        block_meta,
                        extra_meta,
                        ('unpriced_reason',),
                        '' if is_priced else 'missing_or_invalid_unit_cost',
                    )
                    or ''
                )
                records.append({
                    'region': region,
                    'commodity': '',
                    'year': year,
                    'process': '',
                    'segment': None,
                    'cost_component_type': 'strategy',
                    'database_strategy': database_strategy,
                    'strategy_kind': strategy_kind,
                    'cost_database_version': row_version,
                    'cost_database_sha256': row_sha256,
                    'cost_basis': row_cost_basis,
                    'is_priced': is_priced,
                    'unpriced_reason': unpriced_reason,
                    'attribution_method': row_attribution,
                    'reference_scenario_id': row_reference,
                    'abatement_native': abatement,
                    'abatement_native_unit': 'tCO2e',
                    'abatement_ktco2eq': abatement / KT_CO2E_TO_T_CO2E,
                    'abatement_tco2eq': abatement,
                    'priced_reduction_native': priced_reduction,
                    'priced_reduction_native_unit': 'tCO2e',
                    'priced_reduction_ktco2eq': priced_reduction / KT_CO2E_TO_T_CO2E,
                    'priced_reduction_tco2eq': priced_reduction,
                    'unit_cost_usd_per_tco2eq': unit_cost,
                    'total_cost_usd': priced_reduction * unit_cost if is_priced else 0.0,
                })
            except Exception as e:
                print(f"    [WARN] 提取策略减排数据失败: {key}: {e}")
                continue
    
    if not records:
        print("[WARN] 未找到有效的减排记录，生成空文件")
        df = pd.DataFrame(columns=_COST_SUMMARY_COLUMNS)
    else:
        df = pd.DataFrame(records, columns=_COST_SUMMARY_COLUMNS)
        
        # Sort by region, year, process, and commodity.
        df = df.sort_values(
            ['cost_component_type', 'region', 'year', 'database_strategy', 'process', 'commodity']
        )
        
        # Statistics
        total_abatement = df['abatement_tco2eq'].sum()
        priced_mask = df['is_priced'].map(_cost_summary_bool_value)
        total_priced_reduction = df.loc[priced_mask, 'priced_reduction_tco2eq'].sum()
        total_cost = df.loc[priced_mask, 'total_cost_usd'].sum()
        
        print(f"\n  [OK] 成本汇总统计:")
        print(f"    - 记录数: {len(df):,}")
        print(f"    - 区域数: {df['region'].nunique()}")
        print(f"    - 商品数: {df['commodity'].nunique()}")
        print(f"    - 过程数: {df['process'].nunique()}")
        print(f"    - 年份范围: {df['year'].min()}-{df['year'].max()}")
        print(f"    - 总物理减排量: {total_abatement:,.2f} tCO2eq")
        print(f"    - 总计价减排量: {total_priced_reduction:,.2f} tCO2eq")
        print(f"    - 总成本: ${total_cost:,.2f} USD")
        print(
            f"    - 平均单位成本: "
            f"${total_cost/total_priced_reduction:.2f} USD/tCO2eq"
            if total_priced_reduction > 0 else ""
        )
    
    # Export CSV.
    df.to_csv(output_path, index=False)
    measure_paths = write_measure_cost_summaries(
        df,
        output_dir=Path(output_path).parent,
        scenario_id=str(scenario_id or ''),
        dict_v3_path=dict_v3_path,
    )
    print(f"\n  [OK] 成本汇总已导出: {output_path}")
    print(f"  [OK] 国家-措施成本汇总已导出: {measure_paths['country_measure']}")
    print(f"  [OK] 全球-措施成本汇总已导出: {measure_paths['global_measure']}")
    print("=" * 80 + "\n")
