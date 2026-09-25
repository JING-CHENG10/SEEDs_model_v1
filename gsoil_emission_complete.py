# -*- coding: utf-8 -*-
"""
Global Soil Emissions Module.
Calculate N2O and CO2 emissions from drained organic soils.

Two components:
1. Historical emissions read from FAO CSV files.
2. Future emissions calculated from areas after land-use change.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


def normalize_m49(val) -> str:
    """Normalize M49 to 'xxx format."""
    if val is None or pd.isna(val):
        return ""
    s = str(val).strip()
    if s.startswith("'"):
        s = s[1:]
    s = s.strip()
    if not s:
        return ""
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
    m49_key = normalize_m49(m49)
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
    m49_key = normalize_m49(m49)
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
    m49_key = normalize_m49(m49)
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
class DrainedOrganicSoilsEmissions:
    """Drained organic soil emissions calculator."""
    
    def __init__(self, dict_v3_path: str, soil_params_path: str):
        """
        Initialize the drained organic soil emissions calculator.
        
        Args:
            dict_v3_path: Path to dict_v3.xlsx.
            soil_params_path: Path to Soil_parameters.xlsx.
        """
        self.dict_v3_path = dict_v3_path
        self.soil_params_path = soil_params_path
        
        # Read Emis_item mappings from dict_v3.
        self.emis_item_map = self._load_emis_item_map()
        
        # Read Soil_parameters.
        self.soil_params = self._load_soil_parameters()
        
    def _load_emis_item_map(self) -> Dict:
        """Read Process-Item-GHG mappings from the Emis_item sheet in dict_v3."""
        try:
            df = pd.read_excel(self.dict_v3_path, sheet_name='Emis_item')
            
            # Select drained organic soil rows.
            mask = df['Process'].str.contains('Drained organic soils', case=False, na=False)
            emis_item_subset = df[mask]
            
            # Build {process: {item: [ghg_list]}} mappings.
            emis_map = {}
            for _, row in emis_item_subset.iterrows():
                process = row['Process']
                item = row['Item_Emis']
                ghg = row['GHG']
                
                if process not in emis_map:
                    emis_map[process] = {}
                if item not in emis_map[process]:
                    emis_map[process][item] = []
                if ghg not in emis_map[process][item]:
                    emis_map[process][item].append(ghg)
            
            logger.info(f"已加载Emis_item映射: {len(emis_map)}个排放过程")
            return emis_map
            
        except Exception as e:
            logger.warning(f"加载Emis_item映射失败: {e}")
            return {}
    
    def _load_soil_parameters(self) -> pd.DataFrame:
        """Read soil parameters from Soil_parameters.xlsx."""
        try:
            df = pd.read_excel(self.soil_params_path)
            logger.info(f"已加载Soil_parameters: {len(df)}行")
            return df
        except Exception as e:
            logger.warning(f"加载Soil_parameters失败: {e}")
            return pd.DataFrame()
    
    def calculate_future_emissions(self,
                                  cropland_area_ha: Dict[Tuple[str, str, int], float],
                                  grassland_area_ha: Dict[Tuple[str, str, int], float],
                                  scenario_params: Optional[Dict] = None) -> Dict:
        """
        Calculate future drained organic soil emissions.
        
        Args:
            cropland_area_ha: {(m49, country, year): area_ha}
            grassland_area_ha: {(m49, country, year): area_ha}
        
        Returns:
            Emissions results as {process_name: DataFrame}.
        """
        
        logger.info(f"[GSOIL] calculate_future_emissions: cropland输入{len(cropland_area_ha)}条, grassland输入{len(grassland_area_ha)}条")
        
        results = {}
        
        # Process cropland organic soils.
        cropland_results = self._calculate_organic_soil_emissions(
            area_dict=cropland_area_ha,
            soil_type='Cropland organic soils',
            item_name='Cropland organic soils',
            scenario_params=scenario_params
        )
        logger.info(f"[GSOIL] Cropland计算完成: {len(cropland_results)}行")
        if not cropland_results.empty:
            results['Drained organic soils (Cropland)'] = cropland_results
        else:
            logger.warning(f"[GSOIL] Cropland结果为空")
        
        # Process grassland organic soils.
        grassland_results = self._calculate_organic_soil_emissions(
            area_dict=grassland_area_ha,
            soil_type='Grassland organic soils',
            item_name='Grassland organic soils',
            scenario_params=scenario_params
        )
        logger.info(f"[GSOIL] Grassland计算完成: {len(grassland_results)}行")
        if not grassland_results.empty:
            results['Drained organic soils (Grassland)'] = grassland_results
        else:
            logger.warning(f"[GSOIL] Grassland结果为空")
        
        return results
    
    def _calculate_organic_soil_emissions(self,
                                         area_dict: Dict[Tuple[str, str, int], float],
                                         soil_type: str,
                                         item_name: str,
                                         scenario_params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Calculate organic soil emissions for cropland or grassland.
        
        Args:
            area_dict: Area dictionary {(m49, country, year): area_ha}.
            soil_type: Parameter lookup category, e.g. 'Cropland organic soils'.
            item_name: Item name for output.
        
        Returns:
            Emissions DataFrame.
        """
        
        if self.soil_params.empty:
            logger.warning(f"[GSOIL] Soil_parameters为空，无法计算{soil_type}排放")
            return pd.DataFrame()
        
        logger.info(f"[GSOIL] 开始计算{soil_type}，输入{len(area_dict)}条面积记录")
        
        # Standardize M49 in soil_params.
        self.soil_params['M49_normalized'] = self.soil_params['M49_Country_Code'].apply(normalize_m49)
        
        # Ensure column names are strings.
        self.soil_params.columns = self.soil_params.columns.astype(str)
        
        # Get available year columns (2000-2022).
        # Check for a Y prefix.
        has_y_prefix = any(col.startswith('Y') and col[1:].isdigit() for col in self.soil_params.columns)
        
        if has_y_prefix:
            year_cols = [f"Y{y}" for y in range(2000, 2023)]
        else:
            year_cols = [str(y) for y in range(2000, 2023)]
            
        available_years = [y for y in year_cols if y in self.soil_params.columns]
        
        if not available_years:
            logger.error(f"[GSOIL] Soil_parameters 中没有年份列！可用列: {list(self.soil_params.columns)}")
            return pd.DataFrame()
        
        logger.info(f"[GSOIL] Soil_parameters 可用年份: {available_years[0]}-{available_years[-1]}")
        
        emissions_list = []
        skipped_no_area_coeff = 0
        skipped_no_ef = 0
        skipped_zero_area = 0
        processed_count = 0
        ef_checked = 0
        ef_applied = 0
        ef_no_match = 0
        ef_log_samples = 0
        
        for (m49, country, year), area_ha in area_dict.items():
            if area_ha <= 0:  # Skip zero or negative values.
                skipped_zero_area += 1
                continue
            
            processed_count += 1
            
            # Standardize input M49 codes.
            m49_norm = normalize_m49(m49)
            
            # Select the parameter year:
            # Historical years (<=2020): use the corresponding year.
            # Future years (>2020): use 2020 as the baseline.
            if year <= 2020:
                param_year = min(max(year, 2000), 2020)  # Restrict to 2000-2020.
            else:
                param_year = 2020  # Use the 2020 baseline for future years.
            param_year_col = f"Y{param_year}" if has_y_prefix else str(param_year)
            
            # Look up Area correlation.
            area_coeff_df = self.soil_params[
                (self.soil_params['M49_normalized'] == m49_norm) &
                (self.soil_params['Item'] == soil_type) &
                (self.soil_params['paramName'] == 'Area correlation')
            ]
            
            if area_coeff_df.empty:
                if processed_count <= 3:  # Log only the first few failures.
                    logger.debug(f"[GSOIL]   跳过 {country}({m49_norm}, 原始{m49}) {year}: 无Area correlation")
                skipped_no_area_coeff += 1
                continue
            
            # Read the area coefficient from the year column.
            area_coeff = area_coeff_df.iloc[0].get(param_year_col, 0)
            if pd.isna(area_coeff) or area_coeff == 0:
                skipped_no_area_coeff += 1
                continue
                
            organic_area_ha = area_ha * area_coeff  # Organic soil area = total area * coefficient.
            
            if organic_area_ha <= 0:
                continue
            
            # Look up N2O and CO2 emission factors.
            ef_df = self.soil_params[
                (self.soil_params['M49_normalized'] == m49_norm) &
                (self.soil_params['Item'] == soil_type) &
                (self.soil_params['paramName'] == 'Emission factor')
            ]
            
            if ef_df.empty:
                if processed_count <= 3:
                    logger.debug(f"[GSOIL]   跳过 {country}({m49_norm}) {year}: 无Emission factor")
                skipped_no_ef += 1
                continue
            
            # Iterate over emission-factor rows, potentially including both N2O and CO2.
            n2o_kt_total = 0.0
            co2_kt_total = 0.0
            has_valid_ef = False
            
            for _, ef_row in ef_df.iterrows():
                emission_factor_t_ha = ef_row.get(param_year_col, np.nan)  # Unit: t/ha/year.
                param_mms = str(ef_row.get('paramMMS', '')).upper()
                ghg_key = param_mms if param_mms else 'All'
                ef_bound = None
                if scenario_params and 'emission_factor_bound_by' in scenario_params:
                    ef_bound = _lookup_ef_bound(
                        scenario_params.get('emission_factor_bound_by'),
                        m49_norm,
                        item_name,
                        'Drained organic soils',
                        ghg_key,
                        year
                    )
                if ef_bound is not None:
                    if (ef_bound[2] or ef_bound[3]) and not np.isfinite(emission_factor_t_ha):
                        raise ValueError(f"GSOIL relative EF bound requires a finite baseline: {m49_norm}/{item_name}/{year}/{ghg_key}")
                    emission_factor_t_ha = _apply_ef_bound(float(emission_factor_t_ha), ef_bound)
                else:
                    ef_abs = None
                    if scenario_params and 'emission_factor_absolute_by' in scenario_params:
                        ef_abs = _lookup_ef_absolute(
                            scenario_params.get('emission_factor_absolute_by'),
                            m49_norm,
                            item_name,
                            'Drained organic soils',
                            ghg_key,
                            year
                        )
                    if ef_abs is not None:
                        emission_factor_t_ha = ef_abs
                    elif not np.isfinite(emission_factor_t_ha):
                        skipped_no_ef += 1
                        continue
                    elif scenario_params and 'emission_factor_multiplier' in scenario_params:
                        ef_mult = _lookup_ef_multiplier(
                            scenario_params['emission_factor_multiplier'],
                            m49_norm,
                            item_name,
                            'Drained organic soils',
                            year,
                            ghg=ghg_key
                        )
                        ef_checked += 1
                        if ef_mult != 1.0:
                            ef_applied += 1
                            if ef_log_samples < 3:
                                logger.info(
                                    f"[GSOIL][EF_MULT] {m49_norm} {item_name} Y{year} mult={ef_mult:.4f}"
                                )
                                ef_log_samples += 1
                        else:
                            ef_no_match += 1
                        emission_factor_t_ha *= ef_mult
                
                if not np.isfinite(emission_factor_t_ha) or emission_factor_t_ha < 0:
                    raise ValueError(f"GSOIL effective EF must be finite and nonnegative: {m49_norm}/{item_name}/{year}/{ghg_key}")
                # Emissions = organic soil area (ha) * EF (t/ha/year), converted to kt/year using 1 t = 0.001 kt.
                emission_kt = organic_area_ha * emission_factor_t_ha / 1000

                # crop_soil_management_ratio: apply directly on process-item emissions.
                if scenario_params and 'crop_soil_management_multiplier' in scenario_params:
                    csm_mult = _lookup_crop_soil_management_multiplier(
                        scenario_params.get('crop_soil_management_multiplier'),
                        m49_norm,
                        item_name,
                        'Drained organic soils',
                        year
                    )
                    emission_kt *= csm_mult

                if not np.isfinite(emission_kt) or emission_kt < 0:
                    raise ValueError(f"GSOIL emissions must be finite and nonnegative: {m49_norm}/{item_name}/{year}/{ghg_key}")
                if param_mms in {'N2O', 'CO2'}:
                    has_valid_ef = True
                    param_mms = str(ef_row.get('paramMMS', '')).upper()
                    
                    if param_mms == 'N2O':
                        n2o_kt_total += emission_kt
                    elif param_mms == 'CO2':
                        co2_kt_total += emission_kt
                    # Other types such as N2O_CO2 need interpretation; currently assume factors represent only N2O or CO2.
            
            if has_valid_ef:
                emissions_list.append({
                    'M49_Country_Code': m49_norm,
                    'Country': country,
                    'Year': year,
                    'Item': item_name,
                    'Process': 'Drained organic soils',
                    'GHG': 'N2O' if n2o_kt_total > 0 else 'CO2', # Flag the primary GHG; actual output contains both columns.
                    'n2o_kt': n2o_kt_total,
                    'co2_kt': co2_kt_total,
                    'ch4_kt': 0.0,
                    'Organic_soil_area_ha': organic_area_ha,
                    'Emission_factor_t_ha': 0.0, # A single EF no longer applies after aggregation.
                    'Param_year_used': param_year
                })
        
        logger.info(f"[GSOIL] {soil_type}计算完成: 处理{processed_count}条, 跳过零面积{skipped_zero_area}条, 无面积系数{skipped_no_area_coeff}条, 无排放因子{skipped_no_ef}条, 生成{len(emissions_list)}条排放记录")
        return pd.DataFrame(emissions_list)


def _load_historical_drained_organic_emissions(emissions_csv_path: Optional[str],
                                               historical_years: Optional[List[int]],
                                               universe=None) -> pd.DataFrame:
    if not emissions_csv_path:
        return pd.DataFrame()
    path = Path(emissions_csv_path)
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.warning(f"[GSOIL] ????CSV????: {exc}")
        return pd.DataFrame()
    df.columns = [str(c).strip() for c in df.columns]
    if "Source" in df.columns:
        df = df[df["Source"].astype(str).str.contains("TIER 1", case=False, na=False)]
    if "Element" in df.columns:
        elem = df["Element"].astype(str)
        mask = elem.str.contains(r"N2O|CO2", case=False, na=False) & ~elem.str.contains("Area", case=False, na=False)
        df = df.loc[mask].copy()
        if df.empty:
            return pd.DataFrame()
        df["GHG"] = df["Element"].astype(str).str.extract(r"(?i)(N2O|CO2)", expand=False).str.upper()
    elif "Process" in df.columns:
        df = df[df["Process"].astype(str).str.contains("Drained organic soils", case=False, na=False)]
        if df.empty:
            return pd.DataFrame()
    else:
        return pd.DataFrame()

    if "M49_Country_Code" in df.columns:
        df["M49_Country_Code"] = df["M49_Country_Code"].apply(normalize_m49)
    elif "Area" in df.columns:
        df["M49_Country_Code"] = df["Area"].apply(normalize_m49)

    if universe is not None and hasattr(universe, "country_by_m49") and "M49_Country_Code" in df.columns:
        df["Country"] = df["M49_Country_Code"].map(universe.country_by_m49)

    year_cols = [c for c in df.columns if str(c).startswith("Y") and str(c)[1:].isdigit()]
    if year_cols:
        id_cols = [c for c in df.columns if c not in year_cols]
        long_df = df.melt(id_vars=id_cols, value_vars=year_cols, var_name="year", value_name="value")
        long_df["year"] = pd.to_numeric(long_df["year"].astype(str).str.lstrip("Y"), errors="coerce")
    elif "year" in df.columns or "Year" in df.columns:
        long_df = df.copy()
        if "Year" in long_df.columns and "year" not in long_df.columns:
            long_df = long_df.rename(columns={"Year": "year"})
        val_col = "value" if "value" in long_df.columns else ("Value" if "Value" in long_df.columns else None)
        if val_col and val_col != "value":
            long_df = long_df.rename(columns={val_col: "value"})
        if "value" not in long_df.columns:
            return pd.DataFrame()
        long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce")
    else:
        return pd.DataFrame()

    long_df["value"] = pd.to_numeric(long_df.get("value"), errors="coerce")
    long_df = long_df.dropna(subset=["year", "value"])
    long_df = long_df[long_df["value"] != 0]
    if historical_years:
        years_set = set(int(y) for y in historical_years)
        long_df = long_df[long_df["year"].astype(int).isin(years_set)]
    else:
        long_df = long_df[long_df["year"].astype(int) <= 2020]
    if long_df.empty:
        return pd.DataFrame()

    if "GHG" not in long_df.columns:
        def _infer_ghg_row(row) -> str:
            for key in ("GHG", "gas"):
                if key in row and str(row.get(key)).strip():
                    return str(row.get(key)).strip().upper()
            for key in ("Element", "Item", "Process"):
                if key in row and str(row.get(key)).strip():
                    s = str(row.get(key)).upper()
                    if "N2O" in s:
                        return "N2O"
                    if "CO2" in s:
                        return "CO2"
                    if "CH4" in s:
                        return "CH4"
            return "N2O"
        long_df["GHG"] = long_df.apply(_infer_ghg_row, axis=1)
    else:
        long_df["GHG"] = long_df["GHG"].astype(str).str.upper()

    long_df["Process"] = "Drained organic soils"

    group_cols = [c for c in ["M49_Country_Code", "Country", "Item", "Process", "year", "GHG"] if c in long_df.columns]
    long_df = long_df.groupby(group_cols, as_index=False)["value"].sum()

    base_cols = [c for c in ["M49_Country_Code", "Country", "Item", "Process", "year"] if c in long_df.columns]
    out = long_df.pivot_table(index=base_cols, columns="GHG", values="value", aggfunc="sum").reset_index()
    for col in ("N2O", "CO2", "CH4"):
        if col not in out.columns:
            out[col] = 0.0
    out = out.rename(columns={"N2O": "n2o_kt", "CO2": "co2_kt", "CH4": "ch4_kt"})
    return out


def run_drained_organic_soils_emissions(*,
                                        universe,
                                        dict_v3_path: str,
                                        soil_params_path: str,
                                        emissions_csv_path: Optional[str] = None,
                                        cropland_area_future: Optional[Dict[Tuple[str, str, int], float]] = None,
                                        grassland_area_future: Optional[Dict[Tuple[str, str, int], float]] = None,
                                        historical_years: Optional[List[int]] = None,
                                        scenario_params: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
    """Wrapper expected by S4_0_main. Returns a dict of emission DataFrames."""
    results: Dict[str, pd.DataFrame] = {}
    hist_df = _load_historical_drained_organic_emissions(emissions_csv_path, historical_years, universe=universe)
    if isinstance(hist_df, pd.DataFrame) and not hist_df.empty:
        results["Historical Drained organic soils"] = hist_df

    engine = DrainedOrganicSoilsEmissions(dict_v3_path=dict_v3_path, soil_params_path=soil_params_path)
    cropland_area_future = cropland_area_future or {}
    grassland_area_future = grassland_area_future or {}
    future_results = engine.calculate_future_emissions(
        cropland_area_ha=cropland_area_future,
        grassland_area_ha=grassland_area_future,
        scenario_params=scenario_params
    )
    if isinstance(future_results, dict):
        results.update(future_results)
    return results
