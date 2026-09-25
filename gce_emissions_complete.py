# -*- coding: utf-8 -*-
"""
gce_emissions_complete.py - Complete crop emissions calculation module aligned with FAOSTAT.

Module purpose
--------
This module implements the complete crop-sector (GCE) emissions engine, analogous to the livestock module,
providing stateless, composable calculation functions.

Four main processes are covered:
1. Crop residues (direct N2O): N2O from returning crop residues to soil.
2. Burning crop residues (CH4/N2O): CH4 and N2O from residue burning.
3. Rice cultivation (CH4): CH4 from paddy fields.
4. Synthetic fertilizers (N2O): N2O from synthetic fertilizer application.

Data flow:
  Production (production_t) plus parameters (residue N content, EF) 
  -> Residue nitrogen or dry matter quantity 
  -> Multiply by emission factors 
  -> Emissions (N2O_kt, CH4_kt).

Key data sources:
  - Historical production: Production_Crops_Livestock_E_All_Data_NOFLAG.csv.
  - Historical emissions: Emissions_crops_E_All_Data_NOFLAG.csv.
  - Parameters: Code/src/GCE_parameters.xlsx (GCE_parameters sheet).
  - dict_v3: Item-name mapping and M49 country filtering.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple, Set
import pandas as pd
import numpy as np
import os
import sys
import builtins
from runtime_data_cache import read_excel_cached, read_csv_cached

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from config_paths import get_input_base, get_src_base


def _get_debug_level() -> int:
    raw = os.environ.get("NZF_DEBUG_LEVEL", "0").strip()
    try:
        return max(0, int(float(raw)))
    except Exception:
        return 0


def print(*args, **kwargs):  # type: ignore[override]
    text = " ".join(str(a) for a in args)
    upper = text.upper()
    is_warning = ("WARN" in upper) or ("ERROR" in upper)
    if _get_debug_level() >= 2 or is_warning:
        return builtins.print(*args, **kwargs)
    return None


def _norm_m49(val: str) -> str:
    """Normalize M49 to 'xxx format."""
    if val is None or pd.isna(val):
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


def _lookup_ef_multiplier(ef_mult_dict: Optional[Dict], m49: str, item: str, process: str, year: int, ghg: Optional[str] = None) -> float:
    if not ef_mult_dict:
        return 1.0
    m49_key = _norm_m49(m49)
    if not m49_key:
        return 1.0
    item_key = str(item).strip()
    proc_key = str(process).strip()
    ghg_key = str(ghg).strip() if ghg else None
    try:
        year_key = int(year)
    except Exception:
        year_key = year
    keys = []
    if ghg_key:
        keys.extend([
            (m49_key, item_key, proc_key, ghg_key, year_key),
            (m49_key, item_key, 'All', ghg_key, year_key),
            (m49_key, 'All', proc_key, ghg_key, year_key),
            (m49_key, 'All', 'All', ghg_key, year_key),
            (m49_key, item_key, proc_key, 'All', year_key),
            (m49_key, item_key, 'All', 'All', year_key),
            (m49_key, 'All', proc_key, 'All', year_key),
            (m49_key, 'All', 'All', 'All', year_key),
        ])
    keys.extend([
        (m49_key, item_key, proc_key, year_key),
        (m49_key, item_key, 'All', year_key),
        (m49_key, 'All', proc_key, year_key),
        (m49_key, 'All', 'All', year_key),
    ])
    for key in keys:
        if key in ef_mult_dict:
            try:
                return float(ef_mult_dict.get(key, 1.0))
            except Exception:
                return 1.0
    return 1.0


def _lookup_crop_soil_management_multiplier(mult_dict: Optional[Dict], m49: str, item: str, process: str, year: int) -> float:
    """Lookup multiplier for crop_soil_management_ratio by country-item-process-year."""
    return _lookup_ef_multiplier(mult_dict, m49, item, process, year, ghg=None)


def _lookup_ef_absolute(ef_abs_dict: Optional[Dict], m49: str, item: str, process: str, ghg: str, year: int) -> Optional[float]:
    if not ef_abs_dict:
        return None
    m49_key = _norm_m49(m49)
    if not m49_key:
        return None
    item_key = str(item).strip()
    proc_key = str(process).strip()
    ghg_key = str(ghg).strip() if ghg else 'All'
    try:
        year_key = int(year)
    except Exception:
        year_key = year
    keys = [
        (m49_key, item_key, proc_key, ghg_key, year_key),
        (m49_key, item_key, 'All', ghg_key, year_key),
        (m49_key, 'All', proc_key, ghg_key, year_key),
        (m49_key, 'All', 'All', ghg_key, year_key),
        (m49_key, item_key, proc_key, 'All', year_key),
        (m49_key, item_key, 'All', 'All', year_key),
        (m49_key, 'All', proc_key, 'All', year_key),
        (m49_key, 'All', 'All', 'All', year_key),
        (m49_key, item_key, proc_key, year_key),
        (m49_key, item_key, 'All', year_key),
        (m49_key, 'All', proc_key, year_key),
        (m49_key, 'All', 'All', year_key),
    ]
    for key in keys:
        if key in ef_abs_dict:
            try:
                return float(ef_abs_dict.get(key))
            except Exception:
                return None
    return None


def _lookup_ef_bound(ef_bound_dict: Optional[Dict], m49: str, item: str, process: str, ghg: str, year: int) -> Optional[Tuple[float, float, bool, bool, float]]:
    if not ef_bound_dict:
        return None
    m49_key = _norm_m49(m49)
    if not m49_key:
        return None
    item_key = str(item).strip()
    proc_key = str(process).strip()
    ghg_key = str(ghg).strip() if ghg else 'All'
    try:
        year_key = int(year)
    except Exception:
        year_key = year
    keys = [
        (m49_key, item_key, proc_key, ghg_key, year_key),
        (m49_key, item_key, 'All', ghg_key, year_key),
        (m49_key, 'All', proc_key, ghg_key, year_key),
        (m49_key, 'All', 'All', ghg_key, year_key),
        (m49_key, item_key, proc_key, 'All', year_key),
        (m49_key, item_key, 'All', 'All', year_key),
        (m49_key, 'All', proc_key, 'All', year_key),
        (m49_key, 'All', 'All', 'All', year_key),
        (m49_key, item_key, proc_key, year_key),
        (m49_key, item_key, 'All', year_key),
        (m49_key, 'All', proc_key, year_key),
        (m49_key, 'All', 'All', year_key),
    ]
    for key in keys:
        if key in ef_bound_dict:
            try:
                return ef_bound_dict.get(key)
            except Exception:
                return None
    return None


def _apply_ef_bound(base_ef: float, bound: Tuple[float, float, bool, bool, float]) -> float:
    lo, hi, lo_is_y2020, hi_is_y2020, u = bound
    try:
        lo = float(lo)
        hi = float(hi)
        u = float(u)
    except Exception:
        return base_ef
    lo_val = base_ef * lo if lo_is_y2020 else lo
    hi_val = base_ef * hi if hi_is_y2020 else hi
    if hi_val < lo_val:
        lo_val, hi_val = hi_val, lo_val
    if u < 0.0:
        u = 0.0
    elif u > 1.0:
        u = 1.0
    return lo_val + (hi_val - lo_val) * u
class CropEmissionsCalculator:
    """
    Crop emissions calculator.
    
    Responsibilities:
    1. Read parameters from GCE_parameters.xlsx.
    2. Read historical data from the emissions CSV.
    3. Calculate future emissions for each process.
    4. Support scenario and MC simulations.
    """
    
    def __init__(self, 
                 gle_params_path: str,
                 dict_v3_path: str,
                 hist_emissions_crop_path: str,
                 fertilizer_efficiency_path: Optional[str] = None):
        """
        Initialize the Crop Emissions Calculator.
        
        Args:
            gle_params_path: Path to GCE_parameters.xlsx.
            dict_v3_path: Path to dict_v3.xlsx.
            hist_emissions_crop_path: Path to Emissions_crops_E_All_Data_NOFLAG.csv.
            fertilizer_efficiency_path: Path to Fertilizer_efficiency.xlsx for historical synthetic-fertilizer allocation.
        """
        self.gle_params_path = gle_params_path
        self.dict_v3_path = dict_v3_path
        self.hist_emissions_crop_path = hist_emissions_crop_path
        self.fertilizer_efficiency_path = fertilizer_efficiency_path
        
        # Load the GCE_parameters table.
        self._load_gce_parameters()
        
        # Load historical emissions for direct retrieval in historical years.
        self._load_historical_emissions()
        
        # Load fertilizer efficiency data for historical synthetic-fertilizer allocation.
        self._load_fertilizer_efficiency()
        self._parameter_matrix_cache: Dict[Tuple[str, str, int, str], pd.DataFrame] = {}
        
    def _load_gce_parameters(self) -> None:
        """Load parameters from GCE_parameters.xlsx."""
        if not os.path.exists(self.gle_params_path):
            print(f"WARNING: parameter file not found: {self.gle_params_path}")
            self.gce_params = pd.DataFrame()
            return
        
        try:
            # Read the GCE_parameters sheet.
            self.gce_params = read_excel_cached(self.gle_params_path, sheet_name='GCE_parameters')
            print(f"[INFO] Loaded GCE_parameters: {len(self.gce_params)} rows")
            print(f"[DEBUG] Columns: {list(self.gce_params.columns)}")
            
            # Retain only rows with Select=1.
            if 'Select' in self.gce_params.columns:
                before = len(self.gce_params)
                self.gce_params = self.gce_params[self.gce_params['Select'] == 1].copy()
                after = len(self.gce_params)
                print(f"[INFO] Filter Select=1: {before} -> {after} rows")
            
            # Normalize M49 codes if present.
            if 'M49_Country_Code' in self.gce_params.columns:
                self.gce_params['M49_Country_Code'] = self.gce_params['M49_Country_Code'].apply(_norm_m49)
                
        except Exception as e:
            print(f"[ERROR] Failed to load GCE_parameters: {e}")
            self.gce_params = pd.DataFrame()
    
    def _load_historical_emissions(self) -> None:
        """Load historical emissions from Emissions_crops_E_All_Data_NOFLAG.csv."""
        if not os.path.exists(self.hist_emissions_crop_path):
            print(f"WARNING: historical emissions file not found: {self.hist_emissions_crop_path}")
            self.hist_emissions_crop = pd.DataFrame()
            return
        
        try:
            self.hist_emissions_crop = read_csv_cached(self.hist_emissions_crop_path, encoding='utf-8')
            print(f"[INFO] Loaded historical crop emissions: {len(self.hist_emissions_crop)} rows")
            
            # Normalize M49 codes.
            if 'M49_Country_Code' in self.hist_emissions_crop.columns:
                self.hist_emissions_crop['M49_Country_Code'] = self.hist_emissions_crop['M49_Country_Code'].apply(_norm_m49)
                
        except Exception as e:
            print(f"[ERROR] Failed to load historical crop emissions: {e}")
            self.hist_emissions_crop = pd.DataFrame()
    
    def _load_fertilizer_efficiency(self) -> None:
        """Load fertilizer efficiency data from Fertilizer_efficiency.xlsx."""
        if not self.fertilizer_efficiency_path or not os.path.exists(self.fertilizer_efficiency_path):
            print("[INFO] Fertilizer_efficiency.xlsx not provided or missing; using defaults")
            self.fertilizer_efficiency = pd.DataFrame()
            return
        
        try:
            self.fertilizer_efficiency = read_excel_cached(self.fertilizer_efficiency_path, sheet_name="data")
            print(f"[INFO] Loaded Fertilizer_efficiency: {len(self.fertilizer_efficiency)} rows")
            
            # Normalize M49 codes.
            if 'M49_Country_Code' in self.fertilizer_efficiency.columns:
                self.fertilizer_efficiency['M49_Country_Code'] = self.fertilizer_efficiency['M49_Country_Code'].apply(_norm_m49)
                
        except Exception as e:
            print(f"WARNING: Failed to load Fertilizer_efficiency: {e}")
            self.fertilizer_efficiency = pd.DataFrame()
    
    def _standardize_results(self, results: List[Dict]) -> pd.DataFrame:
        """
        Standardize calculation results to a common column structure.
        Returns: M49_Country_Code, Item, year, process, CH4_kt, N2O_kt, CO2_kt.
        """
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        
        # Ensure all required columns exist; initialize missing columns to zero.
        for col in ['M49_Country_Code', 'Item', 'year', 'process']:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        for gas_col in ['CH4_kt', 'N2O_kt', 'CO2_kt']:
            if gas_col not in df.columns:
                df[gas_col] = 0.0
        
        # Handle multiple rows when a 'gas' column exists.
        if 'gas' in df.columns:
            gas_series = df['gas'].astype(str)
            for gas in ['CH4', 'N2O', 'CO2']:
                alt_cols = [col for col in df.columns if col.startswith(gas) and col.endswith('_kt') and col != f'{gas}_kt']
                if not alt_cols:
                    continue
                source = df[alt_cols].bfill(axis=1).iloc[:, 0].fillna(0.0)
                mask = gas_series.eq(gas)
                if mask.any():
                    df.loc[mask, f'{gas}_kt'] = source.loc[mask].to_numpy()
        
        return df[['M49_Country_Code', 'Item', 'year', 'process', 'CH4_kt', 'N2O_kt', 'CO2_kt']]
    
    def get_parameter(self, 
                     m49_code: str, 
                     item: str, 
                     process: str, 
                     param_name: str,
                     year: int) -> Optional[float]:
        """
        Get a single parameter value from the parameter table.
        
        Lookup logic:
        1. Exact M49 + Item + Process + ParamName match.
        2. Otherwise fall back to M49='000' (Global).
        3. If the target year is absent or NaN, search backward for the latest available year.
        """
        # Normalize M49.
        m49 = _norm_m49(m49_code)
        
        # Identify and sort all year columns.
        year_map = {}
        for col in self.gce_params.columns:
            s = str(col).strip()
            if s.isdigit():
                year_map.setdefault(int(s), []).append(col)
            elif s.startswith('Y') and s[1:].isdigit():
                year_map.setdefault(int(s[1:]), []).append(col)
        year_cols = sorted(year_map.keys())
        if not year_cols:
            return None
        
        def _find_value_in_row(row: pd.Series, target_year: int) -> Optional[float]:
            """
            Look up a parameter in a row, supporting year extrapolation and NaN handling.
            
            Strategy:
            1. Try the target year.
            2. If absent or NaN, search backward for the latest available year.
            """
            # Year column names are strings such as '2000' and '2020', without a Y prefix.
            for col in year_map.get(target_year, []):
                if col in row.index and pd.notna(row[col]):
                    return float(row[col])
            available_years = [y for y in year_cols if y <= target_year]
            if not available_years:
                # Use the earliest year if the target year precedes all data.
                available_years = [min(year_cols)]
            
            # Search backward from the latest year for the first non-NaN value.
            for search_year in reversed(available_years):
                for col in year_map.get(search_year, []):
                    if col in row.index and pd.notna(row[col]):
                        return float(row[col])
            
            return None
        
        # 1. Exact match on M49 + Item + Process + ParamName.
        mask = (
            (self.gce_params['M49_Country_Code'].astype(str) == m49) &
            (self.gce_params['Item'].astype(str) == item) &
            (self.gce_params['Process'].astype(str) == process) &
            (self.gce_params['paramName'].astype(str) == param_name)
        )
        
        if mask.any():
            row = self.gce_params[mask].iloc[0]
            val = _find_value_in_row(row, year)
            if val is not None:
                return val
        
        # 2. Global fallback (M49='000' or '0').
        for global_m49 in ['000', '0']:
            mask = (
                (self.gce_params['M49_Country_Code'].astype(str) == global_m49) &
                (self.gce_params['Item'].astype(str) == item) &
                (self.gce_params['Process'].astype(str) == process) &
                (self.gce_params['paramName'].astype(str) == param_name)
            )
            if mask.any():
                row = self.gce_params[mask].iloc[0]
                val = _find_value_in_row(row, year)
                if val is not None:
                    return val
        
        return None

    def _get_parameter_matrix(self, process: str, param_name: str, year: int, value_col: str) -> pd.DataFrame:
        """Build parameter table for vectorized join with fallback-to-earlier-year logic."""
        cache = getattr(self, '_parameter_matrix_cache', None)
        if cache is None:
            cache = {}
            self._parameter_matrix_cache = cache
        cache_key = (str(process), str(param_name), int(year), str(value_col))
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        if self.gce_params.empty:
            return pd.DataFrame(columns=['M49_norm', 'Item', value_col])
        required = {'M49_Country_Code', 'Item', 'Process', 'paramName'}
        if not required.issubset(self.gce_params.columns):
            return pd.DataFrame(columns=['M49_norm', 'Item', value_col])

        params = self.gce_params[
            (self.gce_params['Process'].astype(str) == str(process)) &
            (self.gce_params['paramName'].astype(str) == str(param_name))
        ].copy()
        if params.empty:
            return pd.DataFrame(columns=['M49_norm', 'Item', value_col])

        year_map: Dict[int, List[str]] = {}
        for col in params.columns:
            col_str = str(col).strip()
            if col_str.isdigit():
                year_map.setdefault(int(col_str), []).append(col)
            elif col_str.startswith('Y') and col_str[1:].isdigit():
                year_map.setdefault(int(col_str[1:]), []).append(col)
        if not year_map:
            return pd.DataFrame(columns=['M49_norm', 'Item', value_col])

        available_years = sorted(year_map.keys())
        target_years = [y for y in available_years if y <= int(year)]
        if not target_years:
            target_years = [available_years[0]]
        ordered_cols: List[str] = []
        for y in sorted(target_years, reverse=True):
            ordered_cols.extend(year_map[y])
        params['_value'] = params[ordered_cols].bfill(axis=1).iloc[:, 0]
        params = params.dropna(subset=['_value']).copy()
        params['M49_norm'] = params['M49_Country_Code'].apply(_norm_m49)
        params['Item'] = params['Item'].astype(str)
        params[value_col] = pd.to_numeric(params['_value'], errors='coerce')
        params = params.dropna(subset=[value_col])
        result = params[['M49_norm', 'Item', value_col]].drop_duplicates(subset=['M49_norm', 'Item'])
        cache[cache_key] = result
        return result

    def _merge_parameter(self, df: pd.DataFrame, param_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        if param_df.empty:
            out = df.copy()
            out[value_col] = np.nan
            return out
        out = df.merge(param_df, on=['M49_norm', 'Item'], how='left')
        global_mask = param_df['M49_norm'].isin({"'000", "000", "'0", "0"})
        if global_mask.any():
            global_map = param_df[global_mask].drop_duplicates(subset=['Item']).set_index('Item')[value_col]
            out[value_col] = out[value_col].fillna(out['Item'].map(global_map))
        return out

    @staticmethod
    def _series_or_default(df: pd.DataFrame, column: str, default: Any) -> pd.Series:
        if column in df.columns:
            return df[column]
        if isinstance(default, pd.Series):
            return default.reindex(df.index)
        return pd.Series(default, index=df.index)
    
    def compute_crop_residues_n2o(self,
                                  production_df: pd.DataFrame,
                                  year: int,
                                  scenario_params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Calculate direct N2O emissions from crop residues.
        
        Units from GCE_parameters.xlsx:
        - Residue N content: kg N / tonne product
        - Emission factor: kg N2O / kg N; this is N2O directly, not N2O-N.
        
        Calculation formulas:
        1. residue_n_kg = production_t * Residue_N_content [tonne * kg/tonne = kg N].
        2. n2o_kg = residue_n_kg * EF [kg N * kg N2O/kg N = kg N2O].
        3. n2o_kt = n2o_kg / 1e6 [kg -> kt].
        """
        process = "Crop residues"
        if not isinstance(production_df, pd.DataFrame) or production_df.empty:
            return self._standardize_results([])

        work = production_df.copy()
        work['M49_Country_Code'] = self._series_or_default(work, 'M49_Country_Code', '').astype(str)
        work['Item'] = self._series_or_default(work, 'Item', '').astype(str)
        if 'Item_Emis' not in work.columns:
            work['Item_Emis'] = work['Item']
        work['Item_Emis'] = work['Item_Emis'].astype(str)
        work['production_t'] = pd.to_numeric(
            self._series_or_default(work, 'production_t', 0.0),
            errors='coerce'
        ).fillna(0.0)
        work = work[work['production_t'] > 0].copy()
        if work.empty:
            return self._standardize_results([])
        work['M49_norm'] = work['M49_Country_Code'].apply(_norm_m49)

        residue_n_df = self._get_parameter_matrix(process, "Residue N content", year, value_col='residue_n')
        ef_df = self._get_parameter_matrix(process, "Emission factor", year, value_col='ef')
        work = self._merge_parameter(work, residue_n_df, 'residue_n')
        work = self._merge_parameter(work, ef_df, 'ef')
        work = work.dropna(subset=['residue_n', 'ef'])
        if work.empty:
            return self._standardize_results([])

        if scenario_params:
            adjusted_ef = []
            for m49, item_emis, ef in zip(work['M49_Country_Code'], work['Item_Emis'], work['ef']):
                ef_val = float(ef)
                ef_bound = _lookup_ef_bound(
                    scenario_params.get('emission_factor_bound_by'),
                    m49,
                    item_emis,
                    process,
                    'N2O',
                    year
                ) if 'emission_factor_bound_by' in scenario_params else None
                if ef_bound is not None:
                    ef_val = _apply_ef_bound(ef_val, ef_bound)
                else:
                    ef_abs = _lookup_ef_absolute(
                        scenario_params.get('emission_factor_absolute_by'),
                        m49,
                        item_emis,
                        process,
                        'N2O',
                        year
                    ) if 'emission_factor_absolute_by' in scenario_params else None
                    if ef_abs is not None:
                        ef_val = ef_abs
                    elif 'emission_factor_multiplier' in scenario_params:
                        ef_val *= _lookup_ef_multiplier(
                            scenario_params.get('emission_factor_multiplier'),
                            m49,
                            item_emis,
                            process,
                            year,
                            ghg='N2O'
                        )
                adjusted_ef.append(ef_val)
            work['ef'] = np.asarray(adjusted_ef, dtype=float)

        work['N2O_kt'] = (work['production_t'] * work['residue_n'] * work['ef']) / 1e6
        if scenario_params and 'crop_soil_management_multiplier' in scenario_params:
            csm = [
                _lookup_crop_soil_management_multiplier(
                    scenario_params.get('crop_soil_management_multiplier'),
                    m49,
                    item_emis,
                    process,
                    year,
                )
                for m49, item_emis in zip(work['M49_Country_Code'], work['Item_Emis'])
            ]
            work['N2O_kt'] = work['N2O_kt'] * np.asarray(csm, dtype=float)

        result_df = pd.DataFrame({
            'M49_Country_Code': work['M49_Country_Code'],
            'Item': work['Item'],
            'year': int(year),
            'process': process,
            'CH4_kt': 0.0,
            'N2O_kt': work['N2O_kt'],
            'CO2_kt': 0.0,
        })
        return self._standardize_results(result_df.to_dict('records'))
    
    def compute_burning_ch4_n2o(self,
                               production_df: pd.DataFrame,
                               year: int,
                               scenario_params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Calculate CH4 and N2O emissions from burning crop residues.
        
        Units from the parameter table:
        - Biomass burning DM content: kg DM / tonne product
        - Emission factor (CH4): kg CH4/kg DM
        - Emission factor (N2O): kg N2O/kg DM
        
        Calculation formulas:
        1. biomass_dm_kg = production_t * dm_content [tonne * kg/tonne = kg DM].
        2. ch4_kg = biomass_dm_kg * ef_ch4 [kg DM * kg CH4/kg DM = kg CH4].
        3. n2o_kg = biomass_dm_kg * ef_n2o [kg DM * kg N2O/kg DM = kg N2O].
        
        The current parameter table may not distinguish CH4/N2O factors; use the IPCC default ratio for now.
        Typical IPCC values: EF_CH4 approximately 0.0027 kg/kg DM; EF_N2O approximately 0.00007 kg/kg DM.
        """
        process_display = "Burning crop residues"
        if not isinstance(production_df, pd.DataFrame) or production_df.empty:
            return self._standardize_results([])

        work = production_df.copy()
        work['M49_Country_Code'] = self._series_or_default(work, 'M49_Country_Code', '').astype(str)
        work['Item'] = self._series_or_default(work, 'Item', '').astype(str)
        if 'Item_Emis' not in work.columns:
            work['Item_Emis'] = work['Item']
        work['Item_Emis'] = work['Item_Emis'].astype(str)
        work['production_t'] = pd.to_numeric(
            self._series_or_default(work, 'production_t', 0.0),
            errors='coerce'
        ).fillna(0.0)
        work = work[work['production_t'] > 0].copy()
        if work.empty:
            return self._standardize_results([])
        work['M49_norm'] = work['M49_Country_Code'].apply(_norm_m49)

        dm_df = self._get_parameter_matrix(process_display, "Biomass burning DM content", year, value_col='dm_content')
        ef_df = self._get_parameter_matrix(process_display, "Emission factor", year, value_col='ef')
        work = self._merge_parameter(work, dm_df, 'dm_content')
        work = self._merge_parameter(work, ef_df, 'ef')
        work = work.dropna(subset=['dm_content', 'ef'])
        work = work[(work['dm_content'] > 0) & (work['ef'] > 0)].copy()
        if work.empty:
            return self._standardize_results([])

        ef_ch4_arr: List[float] = []
        ef_n2o_arr: List[float] = []
        for m49, item_emis, ef in zip(work['M49_Country_Code'], work['Item_Emis'], work['ef']):
            ef_val = float(ef)
            ef_abs_ch4 = None
            ef_abs_n2o = None
            if scenario_params and 'emission_factor_bound_by' in scenario_params:
                bound_ch4 = _lookup_ef_bound(
                    scenario_params.get('emission_factor_bound_by'),
                    m49, item_emis, process_display, 'CH4', year
                )
                if bound_ch4 is not None:
                    ef_abs_ch4 = _apply_ef_bound(ef_val, bound_ch4)
                bound_n2o = _lookup_ef_bound(
                    scenario_params.get('emission_factor_bound_by'),
                    m49, item_emis, process_display, 'N2O', year
                )
                if bound_n2o is not None:
                    ef_abs_n2o = _apply_ef_bound(ef_val, bound_n2o)
            if scenario_params and 'emission_factor_absolute_by' in scenario_params:
                if ef_abs_ch4 is None:
                    ef_abs_ch4 = _lookup_ef_absolute(
                        scenario_params.get('emission_factor_absolute_by'),
                        m49, item_emis, process_display, 'CH4', year
                    )
                if ef_abs_n2o is None:
                    ef_abs_n2o = _lookup_ef_absolute(
                        scenario_params.get('emission_factor_absolute_by'),
                        m49, item_emis, process_display, 'N2O', year
                    )
            ef_final = ef_val
            if ef_abs_ch4 is None and ef_abs_n2o is None and scenario_params and 'emission_factor_multiplier' in scenario_params:
                ef_final = ef_final * _lookup_ef_multiplier(
                    scenario_params.get('emission_factor_multiplier'),
                    m49, item_emis, process_display, year, ghg='CH4'
                )
            if ef_abs_ch4 is None and ef_abs_n2o is None:
                if ef_final > 0.001:
                    ef_ch4 = ef_final
                    ef_n2o = ef_final / 39.0
                else:
                    ef_n2o = ef_final
                    ef_ch4 = ef_final * 39.0
            elif ef_abs_ch4 is None and ef_abs_n2o is not None:
                ef_n2o = float(ef_abs_n2o)
                ef_ch4 = ef_n2o * 39.0
            elif ef_abs_n2o is None and ef_abs_ch4 is not None:
                ef_ch4 = float(ef_abs_ch4)
                ef_n2o = ef_ch4 / 39.0
            else:
                ef_ch4 = float(ef_abs_ch4)
                ef_n2o = float(ef_abs_n2o)
            ef_ch4_arr.append(ef_ch4)
            ef_n2o_arr.append(ef_n2o)

        work['ef_ch4'] = np.asarray(ef_ch4_arr, dtype=float)
        work['ef_n2o'] = np.asarray(ef_n2o_arr, dtype=float)
        biomass_dm_kg = work['production_t'] * work['dm_content']
        work['CH4_kt'] = (biomass_dm_kg * work['ef_ch4']) / 1e6
        work['N2O_kt'] = (biomass_dm_kg * work['ef_n2o']) / 1e6
        if scenario_params and 'crop_soil_management_multiplier' in scenario_params:
            csm = [
                _lookup_crop_soil_management_multiplier(
                    scenario_params.get('crop_soil_management_multiplier'),
                    m49,
                    item_emis,
                    process_display,
                    year,
                )
                for m49, item_emis in zip(work['M49_Country_Code'], work['Item_Emis'])
            ]
            csm_arr = np.asarray(csm, dtype=float)
            work['CH4_kt'] = work['CH4_kt'] * csm_arr
            work['N2O_kt'] = work['N2O_kt'] * csm_arr

        out = work[(work['CH4_kt'] > 0) | (work['N2O_kt'] > 0)].copy()
        result_df = pd.DataFrame({
            'M49_Country_Code': out['M49_Country_Code'],
            'Item': out['Item'],
            'year': int(year),
            'process': process_display,
            'CH4_kt': out['CH4_kt'],
            'N2O_kt': out['N2O_kt'],
            'CO2_kt': 0.0,
        })
        return self._standardize_results(result_df.to_dict('records'))
    
    def compute_rice_ch4(self,
                        harvest_area_df: pd.DataFrame,
                        year: int,
                        scenario_params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Calculate CH4 emissions from rice cultivation.
        
        Units from GCE_parameters.xlsx:
        - Emission factor: kg CH4/ha
        
        Calculation formulas:
        1. ch4_kg = area_ha * EF [ha * kg/ha = kg CH4].
        2. ch4_kt = ch4_kg / 1e6 [kg -> kt].
        """
        process_display = "Rice cultivation"
        if not isinstance(harvest_area_df, pd.DataFrame) or harvest_area_df.empty:
            return self._standardize_results([])

        work = harvest_area_df.copy()
        work['M49_Country_Code'] = self._series_or_default(work, 'M49_Country_Code', '').astype(str)
        work['Item'] = self._series_or_default(work, 'Item', '').astype(str)
        if 'Item_Emis' not in work.columns:
            work['Item_Emis'] = work['Item']
        work['Item_Emis'] = work['Item_Emis'].astype(str)
        commodity_series = self._series_or_default(work, 'commodity', work['Item']).astype(str)
        work['area_ha'] = pd.to_numeric(
            self._series_or_default(work, 'harvest_area_ha', 0.0),
            errors='coerce'
        ).fillna(0.0)
        if 'harvested_area_ha' in work.columns:
            alt_area = pd.to_numeric(work['harvested_area_ha'], errors='coerce').fillna(0.0)
        else:
            alt_area = pd.Series(0.0, index=work.index)
        work.loc[work['area_ha'] <= 0, 'area_ha'] = alt_area.loc[work['area_ha'] <= 0]
        is_rice = (
            work['Item'].str.contains('rice', case=False, na=False) |
            commodity_series.str.contains('rice', case=False, na=False)
        )
        work = work[(work['area_ha'] > 0) & is_rice].copy()
        if work.empty:
            return self._standardize_results([])

        work['M49_norm'] = work['M49_Country_Code'].apply(_norm_m49)
        ef_df = self._get_parameter_matrix(process_display, "Emission factor", year, value_col='ef')
        ef_rice = ef_df[ef_df['Item'].astype(str) == 'Rice'].copy().drop_duplicates(subset=['M49_norm'])
        work = work.merge(ef_rice[['M49_norm', 'ef']], on='M49_norm', how='left')
        global_ef = ef_rice[ef_rice['M49_norm'].isin({"'000", "000", "'0", "0"})]
        if not global_ef.empty:
            fallback = float(global_ef['ef'].iloc[0])
            work['ef'] = work['ef'].fillna(fallback)
        work = work.dropna(subset=['ef'])
        if work.empty:
            return self._standardize_results([])

        if scenario_params:
            adjusted_ef = []
            for m49, item_emis, ef_val in zip(work['M49_Country_Code'], work['Item_Emis'], work['ef']):
                ef = float(ef_val)
                ef_bound = _lookup_ef_bound(
                    scenario_params.get('emission_factor_bound_by'),
                    m49,
                    item_emis,
                    process_display,
                    'CH4',
                    year
                ) if 'emission_factor_bound_by' in scenario_params else None
                if ef_bound is not None:
                    ef = _apply_ef_bound(ef, ef_bound)
                else:
                    ef_abs = _lookup_ef_absolute(
                        scenario_params.get('emission_factor_absolute_by'),
                        m49,
                        item_emis,
                        process_display,
                        'CH4',
                        year
                    ) if 'emission_factor_absolute_by' in scenario_params else None
                    if ef_abs is not None:
                        ef = ef_abs
                    elif 'emission_factor_multiplier' in scenario_params:
                        ef *= _lookup_ef_multiplier(
                            scenario_params.get('emission_factor_multiplier'),
                            m49,
                            item_emis,
                            process_display,
                            year,
                            ghg='CH4'
                        )
                adjusted_ef.append(ef)
            work['ef'] = np.asarray(adjusted_ef, dtype=float)

        work['CH4_kt'] = (work['ef'] * work['area_ha']) / 1e6
        result_df = pd.DataFrame({
            'M49_Country_Code': work['M49_Country_Code'],
            'Item': work['Item'],
            'year': int(year),
            'process': process_display,
            'CH4_kt': work['CH4_kt'],
            'N2O_kt': 0.0,
            'CO2_kt': 0.0
        })
        return self._standardize_results(result_df.to_dict('records'))
    
    def compute_synthetic_fert_n2o(self,
                                  fert_df: pd.DataFrame,
                                  year: int,
                                  scenario_params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Calculate future N2O emissions from synthetic fertilizers.
        
        Logic:
        1. Use parameters for the synthetic_fertilizer_direct_N2O process.
        2. Read fertilizer application rates and emission factors for each M49-Item pair.
        3. Calculate direct N2O emissions.
        """
        process_display = "Synthetic fertilizers"
        if not isinstance(fert_df, pd.DataFrame) or fert_df.empty:
            return self._standardize_results([])

        work = fert_df.copy()
        work['M49_Country_Code'] = self._series_or_default(work, 'M49_Country_Code', '').astype(str)
        work['Item'] = self._series_or_default(work, 'Item', '').astype(str)
        if 'Item_Emis' not in work.columns:
            work['Item_Emis'] = work['Item']
        work['Item_Emis'] = work['Item_Emis'].astype(str)
        work['country'] = self._series_or_default(work, 'country', '').astype(str)
        work['area_ha'] = pd.to_numeric(
            self._series_or_default(work, 'harvest_area_ha', 0.0),
            errors='coerce'
        ).fillna(0.0)
        if 'harvested_area_ha' in work.columns:
            alt_area = pd.to_numeric(work['harvested_area_ha'], errors='coerce').fillna(0.0)
            work.loc[work['area_ha'] <= 0, 'area_ha'] = alt_area.loc[work['area_ha'] <= 0]
        work = work[work['area_ha'] > 0].copy()
        if work.empty:
            return self._standardize_results([])

        work['M49_norm'] = work['M49_Country_Code'].apply(_norm_m49)
        fert_rate_df = self._get_parameter_matrix(process_display, "Fertlizer rate", year, value_col='fert_rate')
        ef_df = self._get_parameter_matrix(process_display, "Emission factor", year, value_col='ef')
        work = self._merge_parameter(work, fert_rate_df, 'fert_rate')
        work = self._merge_parameter(work, ef_df, 'ef')
        work = work.dropna(subset=['fert_rate', 'ef'])
        if work.empty:
            return self._standardize_results([])

        if scenario_params and 'fertilizer_rate_multiplier' in scenario_params:
            fert_mult_dict = scenario_params['fertilizer_rate_multiplier']
            fert_mults = []
            for m49, item_emis, country in zip(work['M49_Country_Code'], work['Item_Emis'], work['country']):
                item_key = str(item_emis)
                fert_mult = 1.0
                m49_key = _norm_m49(m49)
                if m49_key:
                    fert_mult = fert_mult_dict.get((m49_key, item_key, year), 1.0)
                    if fert_mult == 1.0:
                        fert_mult = fert_mult_dict.get((m49_key, 'All', year), 1.0)
                if fert_mult == 1.0 and country:
                    fert_mult = fert_mult_dict.get((country, item_key, year), 1.0)
                    if fert_mult == 1.0:
                        fert_mult = fert_mult_dict.get((country, 'All', year), 1.0)
                fert_mults.append(fert_mult)
            work['fert_rate'] = work['fert_rate'].to_numpy(dtype=float) * np.asarray(fert_mults, dtype=float)

        if scenario_params:
            adjusted_ef = []
            for m49, item_emis, ef in zip(work['M49_Country_Code'], work['Item_Emis'], work['ef']):
                ef_val = float(ef)
                ef_bound = _lookup_ef_bound(
                    scenario_params.get('emission_factor_bound_by'),
                    m49,
                    item_emis,
                    process_display,
                    'N2O',
                    year
                ) if 'emission_factor_bound_by' in scenario_params else None
                if ef_bound is not None:
                    ef_val = _apply_ef_bound(ef_val, ef_bound)
                else:
                    ef_abs = _lookup_ef_absolute(
                        scenario_params.get('emission_factor_absolute_by'),
                        m49,
                        item_emis,
                        process_display,
                        'N2O',
                        year
                    ) if 'emission_factor_absolute_by' in scenario_params else None
                    if ef_abs is not None:
                        ef_val = ef_abs
                    elif 'emission_factor_multiplier' in scenario_params:
                        ef_val *= _lookup_ef_multiplier(
                            scenario_params['emission_factor_multiplier'],
                            m49,
                            item_emis,
                            process_display,
                            year,
                            ghg='N2O'
                        )
                adjusted_ef.append(ef_val)
            work['ef'] = np.asarray(adjusted_ef, dtype=float)

        work['N2O_kt'] = (work['area_ha'] * work['fert_rate'] * work['ef']) / 1e6
        result_df = pd.DataFrame({
            'M49_Country_Code': work['M49_Country_Code'],
            'Item': work['Item'],
            'year': int(year),
            'process': process_display,
            'CH4_kt': 0.0,
            'N2O_kt': work['N2O_kt'],
            'CO2_kt': 0.0
        })
        return self._standardize_results(result_df.to_dict('records'))
    
    def _get_historical_emissions(self, year: int) -> Dict[str, pd.DataFrame]:
        """
        Read emissions for a specified year from the historical emissions file.
        
        File format: wide, with one column per year (Y2000, Y2001, ...).
        The Element column contains 'Crop residues (Emissions N2O)', 'Burning crop residues (Emissions CH4)', etc.
        
        Returns: {process_name: DataFrame}, standardized to (M49, Item, year, process, CH4_kt, N2O_kt, CO2_kt).
        """
        if self.hist_emissions_crop.empty:
            return {}
        
        results = {}
        
        # Check for wide format with year columns such as Y2000 and Y2001.
        year_cols = [col for col in self.hist_emissions_crop.columns if col.startswith('Y')]
        year_col_name = f'Y{year}'
        
        if year_col_name not in self.hist_emissions_crop.columns:
            print(f"WARNING: year column {year_col_name} not found in historical file")
            return {}
        
        # Extract data for this year.
        # Map Element to (process, gas_type).
        # Only use Total (Emissions N2O); Direct emissions is already included.
        element_map = {
            'Crop residues (Emissions N2O)': ('Crop residues', 'N2O'),
            # 'Crop residues (Direct emissions N2O)': ('Crop residues', 'N2O'), # Already included in Total; do not duplicate.
            'Burning crop residues (Emissions N2O)': ('Burning crop residues', 'N2O'),
            'Burning crop residues (Emissions CH4)': ('Burning crop residues', 'CH4'),
            'Rice cultivation (Emissions CH4)': ('Rice cultivation', 'CH4'),
            'Synthetic fertilizers (Emissions N2O)': ('Synthetic fertilizers', 'N2O'),
        }
        
        # Collect emissions by process.
        process_data = {
            'Crop residues': [],
            'Burning crop residues': [],
            'Rice cultivation': [],
            'Synthetic fertilizers': []
        }
        
        for element, (process, gas_type) in element_map.items():
            # Filter all data for this Element.
            elem_mask = self.hist_emissions_crop['Element'] == element
            if not elem_mask.any():
                continue
            
            elem_df = self.hist_emissions_crop[elem_mask].copy()
            
            # Extract this year's values.
            elem_df['value'] = elem_df[year_col_name]
            
            # Keep required columns.
            elem_df = elem_df[['M49_Country_Code', 'Item', 'value']].copy()
            elem_df['year'] = year
            elem_df['process'] = process
            elem_df['gas'] = gas_type
            
            # Remove NaN values.
            elem_df = elem_df.dropna(subset=['value'])
            
            if not elem_df.empty:
                process_data[process].append(elem_df)
        
        # Merge and pivot each process into CH4_kt, N2O_kt, and CO2_kt columns.
        for process_name in ['Crop residues', 'Burning crop residues', 'Rice cultivation', 'Synthetic fertilizers']:
            if not process_data[process_name]:
                continue
            
            # Merge all data for this process.
            df = pd.concat(process_data[process_name], ignore_index=True)
            
            # Special handling: allocate synthetic fertilizer emissions across 19 Items.
            if process_name == 'Synthetic fertilizers' and not self.fertilizer_efficiency.empty:
                df = self._allocate_synthetic_fertilizers_by_items(df, year)
            
            # Convert to wide format, with one column per gas.
            pivot_df = df.pivot_table(
                index=['M49_Country_Code', 'Item', 'year', 'process'],
                columns='gas',
                values='value',
                aggfunc='sum'
            ).reset_index()
            
            # Clear the column index name.
            pivot_df.columns.name = None
            
            # Ensure all gas columns exist; fill missing columns with zero.
            for gas in ['CH4', 'N2O', 'CO2']:
                if gas not in pivot_df.columns:
                    pivot_df[f'{gas}_kt'] = 0.0
                else:
                    pivot_df.rename(columns={gas: f'{gas}_kt'}, inplace=True)
            
            # Standardize column names and order.
            if 'CH4_kt' not in pivot_df.columns:
                pivot_df['CH4_kt'] = 0.0
            if 'N2O_kt' not in pivot_df.columns:
                pivot_df['N2O_kt'] = 0.0
            if 'CO2_kt' not in pivot_df.columns:
                pivot_df['CO2_kt'] = 0.0
            
            results[process_name] = pivot_df[['M49_Country_Code', 'Item', 'year', 'process', 'CH4_kt', 'N2O_kt', 'CO2_kt']].reset_index(drop=True)
        
        return results
    
    def _allocate_synthetic_fertilizers_by_items(self, df: pd.DataFrame, year: int) -> pd.DataFrame:
        """
        Allocate historical synthetic fertilizer emissions across 19 Items.
        
        Logic:
        1. Read N_contentModi_Yxxxx columns from Fertilizer_efficiency.xlsx.
        2. Use dict_v3 Item_Fertilizer_Map to standardize Item names to Item_Emis.
        3. Allocate N applied to unmapped Items, such as Others_crops, proportionally among mapped Items.
        4. Calculate each Item's share of N application within each M49_Country_Code.
        5. Allocate total emissions to standardized Items using these shares.
        
        Args:
            df: Original emissions data with Item='Nutrient nitrogen N (total)'.
            year: Year.
        
        Returns:
            Allocated emissions data using standardized Item_Emis names.
        """
        if self.fertilizer_efficiency.empty:
            return df
        
        # Year column name
        n_content_col = f'N_contentModi_Y{year}'
        if n_content_col not in self.fertilizer_efficiency.columns:
            print(f"WARNING: {n_content_col} not found in Fertilizer_efficiency")
            return df
        
        # Extract N application data for this year.
        fert_eff = self.fertilizer_efficiency[['M49_Country_Code', 'Item', n_content_col]].copy()
        fert_eff = fert_eff.rename(columns={n_content_col: 'n_content'})
        fert_eff = fert_eff.dropna(subset=['n_content'])
        
        if fert_eff.empty:
            return df
        
        # Read the Item_Fertilizer_Map -> Item_Emis mapping from dict_v3.
        # This is the official standardized mapping.
        item_name_mapping = self._get_fertilizer_item_mapping()
        
        # Flag Items that can be mapped.
        fert_eff['Item_Emis'] = fert_eff['Item'].map(item_name_mapping)
        fert_eff['is_mappable'] = fert_eff['Item_Emis'].notna()
        
        # Drop unmapped items such as Others_crops instead of redistributing them.
        # Original logic: allocate N from unmapped Items proportionally among mapped Items.
        # Current logic: drop unmapped Items directly and retain only mapped entries.
        fert_eff_redistributed = (
            fert_eff[fert_eff['is_mappable']]
            .groupby(['M49_Country_Code', 'Item_Emis'], as_index=False)['n_content']
            .sum()
            .rename(columns={'Item_Emis': 'Item'})
        )

        if fert_eff_redistributed.empty:
            return df

        # Calculate Item shares within each M49 code.
        fert_eff_redistributed['total_n'] = fert_eff_redistributed.groupby('M49_Country_Code')['n_content'].transform('sum')
        fert_eff_redistributed['share'] = fert_eff_redistributed['n_content'] / fert_eff_redistributed['total_n']
        fert_eff_redistributed = fert_eff_redistributed[fert_eff_redistributed['share'] > 0]  # Remove zero shares.
        
        # Merge emissions data, usually with Item='Nutrient nitrogen N (total)'.
        # Extract total emissions.
        total_emis = df[df['Item'].str.contains('Nutrient nitrogen N', na=False, case=False)].copy()
        
        if total_emis.empty:
            # Return original data if no total is available.
            return df
        
        allocated = total_emis.merge(
            fert_eff_redistributed[['M49_Country_Code', 'Item', 'share']],
            on='M49_Country_Code',
            how='left',
            suffixes=('', '_alloc')
        )
        share_mask = allocated['share'].notna()
        allocated_with_share = allocated[share_mask].copy()
        allocated_without_share = allocated[~share_mask].copy()

        if not allocated_with_share.empty:
            allocated_with_share['Item'] = allocated_with_share['Item_alloc']
            allocated_with_share['value'] = pd.to_numeric(allocated_with_share['value'], errors='coerce').fillna(0.0) * allocated_with_share['share']
        allocated_without_share = allocated_without_share[total_emis.columns]
        allocated_frames = [frame for frame in [allocated_with_share[total_emis.columns] if not allocated_with_share.empty else None, allocated_without_share] if frame is not None and not frame.empty]

        if allocated_frames:
            df_allocated = pd.concat(allocated_frames, ignore_index=True)
            df_no_total = df[~df['Item'].str.contains('Nutrient nitrogen N', na=False, case=False)]
            df = pd.concat([df_no_total, df_allocated], ignore_index=True)
        
        return df
    
    def _get_fertilizer_item_mapping(self) -> dict:
        """
        Get the Item_Fertilizer_Map -> Item_Emis mapping from the Emis_item sheet in dict_v3.
        
        Returns:
            dict: {Item_Fertilizer_Map: Item_Emis}
        """
        cached = getattr(self, '_fertilizer_item_mapping_cache', None)
        if cached is not None:
            return cached

        if not os.path.exists(self.dict_v3_path):
            # Fall back to the hardcoded mapping.
            mapping = {
                'Maize': 'Maize (corn)',
                'Potato': 'Potatoes', 
                'Soybean': 'Soya beans',
                'Sugarcane': 'Sugar cane',
                'Barley': 'Barley',
                'Cassava': 'Cassava',
                'Cotton': 'Cotton',
                'Fruits': 'Fruits',
                'Groundnut': 'Groundnut',
                'Oilpalm': 'Oilpalm',
                'Rapeseed': 'Rapeseed',
                'Rice': 'Rice',
                'Rye': 'Rye',
                'Sorghum': 'Sorghum',
                'Sugarbeet': 'Sugarbeet',
                'Sweetpotato': 'Sweetpotato',
                'Vegetables': 'Vegetables',
                'Wheat': 'Wheat',
                'sunflower': 'sunflower',
            }
            self._fertilizer_item_mapping_cache = mapping
            return mapping
        
        try:
            emis_item_df = read_excel_cached(self.dict_v3_path, sheet_name='Emis_item')
            synth_items = emis_item_df[emis_item_df['Process'] == 'Synthetic fertilizers']
            
            mapping = {}
            valid_items = synth_items[['Item_Fertilizer_Map', 'Item_Emis']].dropna()
            if not valid_items.empty:
                mapping = dict(zip(valid_items['Item_Fertilizer_Map'], valid_items['Item_Emis']))

            self._fertilizer_item_mapping_cache = mapping
            return mapping
        except Exception as e:
            print(f"WARNING: Failed to read mapping from dict_v3: {e}")
            # Fall back to the hardcoded mapping.
            mapping = {
                'Maize': 'Maize (corn)',
                'Potato': 'Potatoes', 
                'Soybean': 'Soya beans',
                'Sugarcane': 'Sugar cane',
            }
            self._fertilizer_item_mapping_cache = mapping
            return mapping
    
    def run_full_calculation(self,
                            production_df: pd.DataFrame,
                            harvest_area_df: pd.DataFrame,
                            years: List[int],
                            scenario_params: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
        """
        Run the full crop emissions calculation.
        
        Logic:
        1. Historical years (<=2020): read directly from Emissions_crops_E_All_Data_NOFLAG.csv.
        2. Future years (>2020): calculate from parameters.
        
        Args:
            production_df: Production data (M49_Country_Code, Item, year, production_t).
            harvest_area_df: Harvest area data (M49_Country_Code, Item, year, harvested_area_ha).
            years: List of calculation years.
            scenario_params: Scenario parameters.
        
        Returns:
            {'Crop residues': df, 'Burning crop residues': df, 'Rice cultivation': df, 'Synthetic fertilizers': df}
        """
        all_results = {}
        
        print(f"\n{'='*60}")
        print("Crop Emissions Calculation")
        print(f"{'='*60}")
        
        # Separate historical and future years.
        historical_years = [y for y in years if y <= 2020]
        future_years = [y for y in years if y > 2020]
        
        # 1. Historical years: read directly from the emissions file.
        if historical_years:
            print(f"[Historical years] Read directly from emissions file: {historical_years}")
            for year in historical_years:
                hist_data = self._get_historical_emissions(year)
                for process, df in hist_data.items():
                    if not df.empty:
                        all_results.setdefault(process, []).append(df)
        
        # 2. Future years: calculate using parameters.
        if future_years:
            print(f"[Future years] Calculated from parameters: {future_years}")
            for year in future_years:
                print(f"Processing year: {year}")
                
                # Filter data for this year.
                prod_year = production_df[production_df['year'] == year]
                area_year = harvest_area_df[harvest_area_df['year'] == year]
                
                if not prod_year.empty:
                    # 1. Crop residues N2O
                    res = self.compute_crop_residues_n2o(prod_year, year, scenario_params)
                    if not res.empty:
                        all_results.setdefault('Crop residues', []).append(res)
                    
                    # 2. Burning crop residues
                    burn = self.compute_burning_ch4_n2o(prod_year, year, scenario_params)
                    if not burn.empty:
                        all_results.setdefault('Burning crop residues', []).append(burn)
                
                if not area_year.empty:
                    # 3. Rice cultivation CH4
                    rice = self.compute_rice_ch4(area_year, year, scenario_params)
                    if not rice.empty:
                        all_results.setdefault('Rice cultivation', []).append(rice)
                    
                    # 4. Synthetic fertilizers N2O
                    fert = self.compute_synthetic_fert_n2o(area_year, year, scenario_params)
                    if not fert.empty:
                        all_results.setdefault('Synthetic fertilizers', []).append(fert)
        
        # Combine results from all processes.
        final_results = {}
        for process, dfs in all_results.items():
            if dfs:
                combined = pd.concat(dfs, ignore_index=True)
                final_results[process] = combined
                print(f"[OK] {process}: {len(combined)} rows")
        
        return final_results


def run_crop_emissions(production_df: pd.DataFrame,
                      harvest_area_df: pd.DataFrame,
                      years: List[int],
                      gle_params_path: str,
                      dict_v3_path: str,
                      hist_emissions_crop_path: str,
                      fertilizer_efficiency_path: Optional[str] = None,
                      scenario_params: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
    """
    Main function: run crop emissions calculations.
    
    Args:
        production_df: Production DataFrame.
        harvest_area_df: Harvest area DataFrame.
        years: Calculation years.
        gle_params_path: Parameter file path.
        dict_v3_path: Path to the dict_v3 file.
        hist_emissions_crop_path: Historical emissions CSV path.
        fertilizer_efficiency_path: Path to Fertilizer_efficiency.xlsx for historical synthetic-fertilizer allocation.
        scenario_params: Scenario parameters.
    
    Returns:
        Dictionary {process_name: DataFrame}.
    """
    calculator = CropEmissionsCalculator(
        gle_params_path=gle_params_path,
        dict_v3_path=dict_v3_path,
        hist_emissions_crop_path=hist_emissions_crop_path,
        fertilizer_efficiency_path=fertilizer_efficiency_path
    )
    
    return calculator.run_full_calculation(
        production_df=production_df,
        harvest_area_df=harvest_area_df,
        years=years,
        scenario_params=scenario_params
    )


__all__ = [
    'CropEmissionsCalculator',
    'run_crop_emissions',
]
