
# -*- coding: utf-8 -*-
"""S3.6_scenarios.py - read/apply scenario config (Scenario_config_new.xlsx).
Dimensions supported: Country (All/Region_aggMC/specific), Commodity (All/crop/meat/dairy/other/specific), Emis process (All/specific).

Element unit:
  - rate: 2080 change relative to 2020 (linear ramp to future years); yield/fertilizer/EF use 1+rate, feed_intensity uses 1+rate
  - multiplier: direct multiplier (linear ramp to future years)
  - amount: absolute level (e.g., $/tCO2e), constant for future years
  - absolute: absolute value (e.g., t/ha, t/head, kgN/ha), overwrites the parameter itself (MC uses this)
  - best_value: read the specified column from Scenario_variable_historical_range.xlsx; applied like absolute (same effect as absolute)
  - profile: used by nutrition_profile_xlsx/nutrition_profile; Value is a profile column or xlsx path (Unit itself is ignored)

English examples (Element unit usage):
  - rate:
      Example: Yield rate = 0.30 -> treated as 1+rate (1.30 multiplier) by 2080, linearly ramped from 2020.
      Example: Feed intensity rate = -0.30 -> treated as 1+rate (0.70) for feed requirement by 2080.
      Example: Emission factor rate = -0.20 -> treated as 1+rate (0.80) by 2080.
  - multiplier:
      Example: Yield multiplier = 1.15 -> directly scales yield to 1.15 by 2080 (linear ramp from 2020).
      Example: EF multiplier = 0.75 -> directly scales EF to 0.75 by 2080.
  - amount:
      Example: Land carbon price = 50 ($/tCO2e) -> fixed 50 for all future years.
  - absolute:
      Example: Yield absolute = 6.5 (t/ha) -> overwrite yield_t_per_ha to 6.5 for future years.
      Example: Livestock yield absolute = 0.25 (t/head) -> overwrite yield_t_per_head to 0.25.
      Example: Fertilizer efficiency absolute = 120 (kgN/ha) -> overwrite fertilizer_efficiency_kgN_per_ha.
  - best_value:
      Example: Element unit = best_value, Value = mean_value -> read that column from Scenario_variable_historical_range.xlsx
      and apply as an absolute value (overwrite the parameter).
  - profile:
      Example: Scenario Element = nutrition_profile_xlsx, Unit = profile, Value = <path to profile xlsx>.
      Example: Scenario Element = nutrition_profile, Unit = profile, Value = <profile column name>.

Examples:
- Feed intensity (rate): 2080 vs 2020 change of -30% -> future years linearly go from 0% to 30% reduction in feed requirement.
- Land carbon price (amount, $/tCO2e): used by LUC module as a per-unit tax (tax_unit).
- Ruminant intake decreasing ratio (rate): sets ruminant demand cap Qd <= (1 - r(t)) * Qd_2020 (by country, year).
"""


from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import re
import pandas as pd
import numpy as np

from S1_0_schema import Node, Universe
from config_paths import get_src_base

RUMINANT_COMMS = [
    'Cattle, non-dairy',
    'Buffalo, non-dairy',
    'Sheep, non-dairy',
    'Goats, non-dairy'
]

CROP_SOIL_MANAGEMENT_PROCESSES = (
    'Crop residues',
    'Burning crop residues',
    'Drained organic soils',
)

_NODE_SCOPED_SCENARIO_MAPS = (
    'feed_reduction_by',
    'ruminant_intake_cap',
    'tax_unit_adder',
    'dm_conversion_multiplier',
    'dm_conversion_multiplier_ipcc',
    'dm_conversion_multiplier_gleam',
    'feed_reduction_by_ipcc',
    'feed_reduction_by_gleam',
    'emission_factor_multiplier',
    'emission_factor_absolute_by',
    'emission_factor_bound_by',
    'fertilizer_rate_multiplier',
    'fertilizer_efficiency_absolute_by',
    'yield_multiplier',
    'yield_absolute_by',
    'manure_management_ratio_multiplier',
    'manure_management_ratio_absolute_by',
    'aquaculture_share_multiplier',
    'aquaculture_share_absolute_by',
    'waste_reduction_rate_by',
    'losses_ratio_by',
    'crop_soil_management_multiplier',
)


def _restrict_scenario_maps_to_active_nodes(
    scenario_params: Dict[str, Any],
    nodes: List[Node],
    *,
    excluded_commodities: Optional[List[str]] = None,
) -> None:
    """Drop country/commodity/year effects outside the actual solver nodes."""
    excluded = {
        str(commodity).strip().lower()
        for commodity in (excluded_commodities or [])
        if str(commodity).strip()
    }
    active_node_keys = {
        (str(n.country), str(n.commodity), int(n.year))
        for n in nodes
        if str(n.commodity).strip().lower() not in excluded
    }
    for map_name in _NODE_SCOPED_SCENARIO_MAPS:
        values = scenario_params.get(map_name)
        if not isinstance(values, dict) or not values:
            continue
        restricted: Dict[Tuple, Any] = {}
        for key, value in values.items():
            if not isinstance(key, tuple) or len(key) < 3:
                continue
            try:
                node_key = (str(key[0]), str(key[1]), int(key[-1]))
            except (TypeError, ValueError):
                continue
            if node_key in active_node_keys:
                restricted[key] = value
        scenario_params[map_name] = restricted

@dataclass
class ScenarioEffect:
    scenario_id: str
    kind: str                 # 'feed_intensity', 'land_carbon_price', 'ruminant_intake_ratio'
    unit: str                 # 'rate' or 'amount'
    value_2080: Any
    country_sel: str          # 'All' or Region_aggMC or specific country (name)
    commodity_sel: str        # 'All' or Cat2 bucket ('crop'/'meat'/'dairy'/'other') or exact commodity
    process_sel: str          # 'All' or exact process
    ghg_sel: str = 'All'      # only for emission_factor
    # Parsed sets
    countries: List[str] = None
    commodities: List[str] = None
    processes: List[str] = None

def _linear_path_2020_2080(value_2080: float, unit: str, years: List[int]) -> Dict[int, float]:
    # Return {year: f(year)}; start at zero for rates or baseline for amounts in 2020, varying linearly to 2080.
    out = {}
    for y in years:
        if y <= 2020:
            out[y] = 0.0 if unit=='rate' else np.nan
        else:
            frac = (y - 2020) / (2080 - 2020)
            out[y] = value_2080 * frac if unit=='rate' else value_2080  # Use constant amounts.
    return out


@lru_cache(maxsize=8)
def _read_scenario_range_sheet(sheet: str) -> pd.DataFrame:
    path = Path(get_src_base()) / "Scenario_variable_historical_range.xlsx"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_excel(path, sheet_name=sheet)
    except Exception:
        return pd.DataFrame()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _best_value_map_for_sheet(sheet: str, value_col: str) -> pd.DataFrame:
    df = _read_scenario_range_sheet(sheet)
    if df.empty:
        return df
    if value_col not in df.columns:
        raise ValueError(f"best_value column '{value_col}' not found in {sheet}")
    return df


def _parse_scenario_value(kind: str, raw: Any) -> Any:
    kind_l = str(kind or '').lower()
    if kind_l in (
        'nutrition_profile_xlsx', 'nutrition_profile_path', 'nutrition_profile',
        'bioenergy_scenario', 'biomass_scenario', 'bioenergy_profile',
    ):
        return str(raw).strip()
    try:
        return float(raw)
    except Exception:
        return raw


from mc_bound_utils import _parse_mc_bound_value, draw_mc_multiplier


def _select_countries(universe: Universe, key: str) -> List[str]:
    if not key or str(key).lower()=='all':
        return list(universe.countries)
    # Region_aggMC
    inv = {}
    for c, r in (universe.region_aggMC_by_country or {}).items():
        inv.setdefault(str(r), []).append(c)
    if key in inv:
        return inv[key]
    # Single country
    return [key] if key in universe.countries else []

def _select_commodities(universe: Universe, key: str) -> List[str]:
    if not key or str(key).lower()=='all':
        return list(universe.commodities)
    k = str(key).lower()
    if k in ('crop','meat','dairy','other'):
        return [c for c in universe.commodities if (universe.item_cat2_by_commodity or {}).get(c, '').lower()==k]
    # Single commodity name
    return [key] if key in universe.commodities else []

def _select_processes(universe: Universe, key: str) -> List[str]:
    if not key or str(key).lower()=='all':
        return list(universe.processes)
    return [key] if key in universe.processes else []


def _canonical_crop_soil_process(raw: Any) -> Optional[str]:
    s = str(raw or '').strip().lower()
    if not s:
        return None
    if 'burning crop residues' in s:
        return 'Burning crop residues'
    if 'crop residues' in s:
        return 'Crop residues'
    if 'drained organic soils' in s:
        return 'Drained organic soils'
    return None


def _resolve_crop_soil_processes(processes: Optional[List[str]], *, process_sel: Any) -> List[str]:
    out: List[str] = []
    seen = set()
    for p in (processes or []):
        canon = _canonical_crop_soil_process(p)
        if canon and canon not in seen:
            out.append(canon)
            seen.add(canon)

    sel_raw = str(process_sel or '').strip()
    sel_l = sel_raw.lower()
    if not out and (not sel_l or sel_l == 'all'):
        return list(CROP_SOIL_MANAGEMENT_PROCESSES)
    if not out:
        canon = _canonical_crop_soil_process(sel_raw)
        if canon:
            return [canon]
    return out

def load_scenario_config(xlsx_path: str, universe: Universe) -> List[ScenarioEffect]:
    try:
        df = pd.read_excel(xlsx_path, sheet_name=0)
    except Exception:
        return []
    df.columns = [str(c).strip() for c in df.columns]
    reqs = ['Scenario ID','Scenario Type','Scenario Unit','Value','Country','Commodity','Emis process']
    # Support Scenario Element and Element unit column names.
    cols_lower = {c.lower(): c for c in df.columns}
    if 'scenario element' in cols_lower and 'scenario type' not in cols_lower:
        df = df.rename(columns={cols_lower['scenario element']: 'Scenario Type'})
        cols_lower = {c.lower(): c for c in df.columns}
    if 'element unit' in cols_lower and 'scenario unit' not in cols_lower:
        df = df.rename(columns={cols_lower['element unit']: 'Scenario Unit'})
        cols_lower = {c.lower(): c for c in df.columns}
    miss = [c for c in reqs if c not in df.columns]
    if miss:
        # Tolerate case differences where possible.
        cols = {c.lower(): c for c in df.columns}
        reqs2 = [cols.get(c.lower(), c) for c in reqs]
        df = df.rename(columns={cols.get(c.lower(), c): c for c in df.columns})
    effects: List[ScenarioEffect] = []
    for r in df.itertuples(index=False):
        sid  = getattr(r, 'Scenario ID')
        kind = str(getattr(r, 'Scenario Type')).strip().lower().replace(' ', '_')
        unit = str(getattr(r, 'Scenario Unit')).strip().lower()
        val_raw = getattr(r, 'Value')
        cty  = getattr(r, 'Country')
        comm = getattr(r, 'Commodity')
        proc = getattr(r, 'Emis process', 'All')
        val = _parse_scenario_value(kind, val_raw)
        eff = ScenarioEffect(sid, kind, unit, val, cty, comm, proc)
        eff.countries   = _select_countries(universe, cty)
        eff.commodities = _select_commodities(universe, comm)
        eff.processes   = _select_processes(universe, proc)
        if eff.kind in ('crop_soil_management_ratio', 'crop_soil_ratio'):
            eff.processes = _resolve_crop_soil_processes(eff.processes, process_sel=proc)
        effects.append(eff)
    return effects

def apply_scenario_to_data(effects: List[ScenarioEffect], scenario_id: str, universe: Universe, nodes: List[Node],
                           *, base_year:int=2020,
                           excluded_commodities: Optional[List[str]] = None) -> Dict[str, Dict]:
    """Return year-specific scenario parameters for downstream S4.0_main modules.
    keys:
      - feed_reduction_by[(country,commodity,year)] = signed rate in [-1, 1] (dm_conversion_multiplier = 1 + rate)
      - land_carbon_price_by_year[year] = $/tCO2e
      - ruminant_intake_cap[(country,commodity,year)] = cap_in_t, based on 2020 commodity demand * (1 + rate_t).
      - tax_unit_adder[(country,commodity,year)] = $/t, used for supply-side minimum producer prices.
    """
    out = {
        'feed_reduction_by': {},  # (country, commodity, year) -> signed rate
        'land_carbon_price_by_year': {},
        'ruminant_intake_cap': {},  # (country, commodity, year) -> cap_in_t
        'tax_unit_adder': {},
        'dm_conversion_multiplier': {},
        'dm_conversion_multiplier_ipcc': {},
        'dm_conversion_multiplier_gleam': {},
        'feed_reduction_by_ipcc': {},
        'feed_reduction_by_gleam': {},
        'emission_factor_multiplier': {},      # (country, commodity, process, year) -> multiplier (1.0 + rate)
        'emission_factor_absolute_by': {},     # (country, commodity, process, ghg, year) -> absolute value
        'emission_factor_bound_by': {},        # (country, commodity, process, ghg, year) -> (lo, hi, lo_is_y2020, hi_is_y2020, u)
        'fertilizer_rate_multiplier': {},      # (country, commodity, year) -> multiplier (1.0 + rate)
        'fertilizer_efficiency_absolute_by': {},  # (country, commodity, year) -> absolute value
        'yield_multiplier': {},                 # (country, commodity, year) -> multiplier (1.0 + rate)
        'yield_absolute_by': {},                # (country, commodity, year) -> absolute value
        'manure_management_ratio_multiplier': {},  # (country, commodity, year) -> multiplier (1.0 + rate)
        'manure_management_ratio_absolute_by': {},  # (country, commodity, year) -> absolute value
        'aquaculture_share_multiplier': {},    # (country, commodity, year) -> multiplier (1.0 + rate)
        'aquaculture_share_absolute_by': {},   # (country, commodity, year) -> absolute value
        'waste_reduction_rate_by': {},         # legacy (country, commodity, year) -> additive losses-ratio delta in [-1, 1]
        'losses_ratio_by': {},                 # (country, commodity, year) -> additive losses-ratio delta in [-1, 1]
        'crop_soil_management_multiplier': {}, # (country, commodity, process, year) -> multiplier (1.0 + rate)
        'bioenergy_scenario': None,              # profile name in bioenergy_scenario_targets.csv
    }
    # Precompute 2020 ruminant baseline demand separately by country and commodity.
    base_rumi = {}  # {(country, commodity): base_demand}
    for n in nodes:
        if n.year==base_year and n.commodity in RUMINANT_COMMS:
            key = (n.country, n.commodity)
            base_rumi[key] = base_rumi.get(key, 0.0) + float(n.D0)

    future_years = [y for y in universe.years if y > base_year]
    def _rescale_u(u_raw: float, q_low: Optional[float], q_high: Optional[float]) -> float:
        try:
            u = float(u_raw)
        except Exception:
            u = 0.5
        if u < 0.0:
            u = 0.0
        elif u > 1.0:
            u = 1.0
        if q_low is None or q_high is None:
            return u
        try:
            ql = float(q_low)
            qh = float(q_high)
        except Exception:
            return u
        ql = max(0.0, min(1.0, ql))
        qh = max(0.0, min(1.0, qh))
        if qh < ql:
            ql, qh = qh, ql
        return ql + (qh - ql) * u

    def _mc_mode(spec: dict) -> str:
        mode = str(spec.get('mode') or 'shared').strip().lower()
        if mode in ('a', 'country', 'per_country', 'independent', 'per-country'):
            return 'per_country'
        if mode in ('per_commodity', 'commodity', 'per-item', 'per_item'):
            return 'per_commodity'
        if mode in ('per_country_commodity', 'country_commodity', 'per-country-commodity', 'per_country_item', 'per-item'):
            return 'per_country_commodity'
        return 'shared'

    def _spec_u(spec: dict) -> float:
        base_u = spec.get('u', 0.5)
        if spec.get('u_is_rescaled', False):
            try:
                u = float(base_u)
            except Exception:
                u = 0.5
            return max(0.0, min(1.0, u))
        return _rescale_u(base_u, spec.get('q_low'), spec.get('q_high'))

    def _u_for_ef(spec: dict, eff: ScenarioEffect, country: str, commodity: str, process: str) -> float:
        mode = _mc_mode(spec)
        if spec.get('pre_sampled', False):
            return _spec_u(spec)
        ef_process_mode = str(spec.get('ef_process_mode') or 'all').strip().lower() or 'all'
        process_key = process if ef_process_mode == 'by_process' else 'All'
        if mode == 'shared' and 'u' in spec:
            return _spec_u(spec)
        import hashlib
        if mode == 'per_country_commodity':
            key = f"{eff.scenario_id}|{eff.kind}|{process_key}|{country}|{commodity}"
        elif mode == 'per_commodity':
            key = f"{eff.scenario_id}|{eff.kind}|{process_key}|{commodity}"
        elif mode == 'per_country':
            key = f"{eff.scenario_id}|{eff.kind}|{process_key}|{country}"
        else:
            key = f"{eff.scenario_id}|{eff.kind}|{process_key}"
        seed = int(hashlib.md5(key.encode('utf-8')).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        u = float(rng.random())
        return _rescale_u(u, spec.get('q_low'), spec.get('q_high'))
    for eff in [e for e in effects if e.scenario_id==scenario_id]:
        mc_bounds = getattr(eff, 'mc_bounds', None)
        if mc_bounds and eff.kind in ('emission_factor', 'ef_multiplier', 'emission_factor_multiplier'):
            if not future_years:
                continue
            ghg_sel = getattr(eff, 'ghg_sel', 'All') or 'All'
            for y in future_years:
                for i in eff.countries:
                    for j in eff.commodities:
                        for p in eff.processes:
                            u_val = _u_for_ef(mc_bounds, eff, i, j, p)
                            out['emission_factor_bound_by'][(i, j, p, ghg_sel, y)] = (
                                float(mc_bounds.get('lo', 0.0)),
                                float(mc_bounds.get('hi', 0.0)),
                                bool(mc_bounds.get('lo_is_y2020', False)),
                                bool(mc_bounds.get('hi_is_y2020', False)),
                                float(u_val),
                            )
            continue
        if eff.kind in ('nutrition_profile_xlsx', 'nutrition_profile_path', 'nutrition_profile'):
            val = str(eff.value_2080).strip()
            if val and val.lower() not in ('nan', 'none', 'no'):
                out['nutrition_profile'] = val
                out['nutrition_profile_xlsx'] = val
            continue
        if eff.kind in ('bioenergy_scenario', 'biomass_scenario', 'bioenergy_profile'):
            val = str(eff.value_2080).strip()
            if val and val.lower() not in ('nan', 'none', 'no'):
                out['bioenergy_scenario'] = val
            continue
        if eff.unit == 'best_value':
            value_col = str(eff.value_2080).strip()
            if not value_col:
                continue
            if not future_years:
                continue
            if eff.kind in ('yield_rate', 'yield_multiplier', 'yield_improvement'):
                df = _best_value_map_for_sheet("yield_multiplier", value_col)
                if "Item_Emis" not in df.columns:
                    continue
                val_map = {
                    str(r.Item_Emis).strip(): float(getattr(r, value_col))
                    for r in df[['Item_Emis', value_col]].dropna(subset=['Item_Emis', value_col]).itertuples(index=False)
                }
                for y in future_years:
                    for i in eff.countries:
                        for j in eff.commodities:
                            v = val_map.get(j)
                            if v is None or not np.isfinite(v):
                                continue
                            out['yield_absolute_by'][(i, j, y)] = float(v)
                continue
            if eff.kind in ('feed_intensity', 'feed_intensity_rate', 'feed_intensity_improve',
                            'feed_efficiency', 'feed_efficiency_rate', 'feed_efficiency_improve'):
                df_ipcc = _best_value_map_for_sheet("dm_conversion_multiplier_IPCC", value_col)
                df_gleam = _best_value_map_for_sheet("dm_conversion_multiplier_GLEAM", value_col)
                if df_ipcc.empty and df_gleam.empty:
                    df = _best_value_map_for_sheet("dm_conversion_multiplier", value_col)
                    if "Item_Emis" not in df.columns:
                        continue
                    val_map = {
                        str(r.Item_Emis).strip(): float(getattr(r, value_col))
                        for r in df[['Item_Emis', value_col]].dropna(subset=['Item_Emis', value_col]).itertuples(index=False)
                    }
                    for y in future_years:
                        for i in eff.countries:
                            for j in eff.commodities:
                                v = val_map.get(j)
                                if v is None or not np.isfinite(v):
                                    continue
                                key = (i, j, y)
                                out['dm_conversion_multiplier'][key] = float(v)
                                # best_value is absolute dm_conversion_multiplier; derive signed rate (mult - 1)
                                out['feed_reduction_by'][key] = float(v) - 1.0
                    continue
                if not df_ipcc.empty and "Item_Emis" in df_ipcc.columns:
                    val_map = {
                        str(r.Item_Emis).strip(): float(getattr(r, value_col))
                        for r in df_ipcc[['Item_Emis', value_col]].dropna(subset=['Item_Emis', value_col]).itertuples(index=False)
                    }
                    for y in future_years:
                        for i in eff.countries:
                            for j in eff.commodities:
                                v = val_map.get(j)
                                if v is None or not np.isfinite(v):
                                    continue
                                key = (i, j, y)
                                out['dm_conversion_multiplier_ipcc'][key] = float(v)
                                out['feed_reduction_by_ipcc'][key] = float(v) - 1.0
                if not df_gleam.empty and "Item_Emis" in df_gleam.columns:
                    val_map = {
                        str(r.Item_Emis).strip(): float(getattr(r, value_col))
                        for r in df_gleam[['Item_Emis', value_col]].dropna(subset=['Item_Emis', value_col]).itertuples(index=False)
                    }
                    for y in future_years:
                        for i in eff.countries:
                            for j in eff.commodities:
                                v = val_map.get(j)
                                if v is None or not np.isfinite(v):
                                    continue
                                key = (i, j, y)
                                out['dm_conversion_multiplier_gleam'][key] = float(v)
                                out['feed_reduction_by_gleam'][key] = float(v) - 1.0
                continue
            if eff.kind in ('fertilizer_rate', 'fertlizer_rate', 'fertilizer_efficiency'):
                df = _best_value_map_for_sheet("fertilizer_rate_multiplier", value_col)
                if "Item_Emis" not in df.columns:
                    continue
                val_map = {
                    str(r.Item_Emis).strip(): float(getattr(r, value_col))
                    for r in df[['Item_Emis', value_col]].dropna(subset=['Item_Emis', value_col]).itertuples(index=False)
                }
                for y in future_years:
                    for i in eff.countries:
                        for j in eff.commodities:
                            v = val_map.get(j)
                            if v is None or not np.isfinite(v):
                                continue
                            out['fertilizer_efficiency_absolute_by'][(i, j, y)] = float(v)
                continue
            if eff.kind in ('emission_factor', 'ef_multiplier', 'emission_factor_multiplier'):
                df = _best_value_map_for_sheet("emission_factor_multiplier", value_col)
                if "Item_Emis" not in df.columns or "Process" not in df.columns:
                    continue
                use_ghg = "GHG" in df.columns
                if use_ghg:
                    val_map = {
                        (str(r.Item_Emis).strip(), str(r.Process).strip(), str(r.GHG).strip()): float(getattr(r, value_col))
                        for r in df[['Item_Emis', 'Process', 'GHG', value_col]].dropna(subset=['Item_Emis', 'Process', 'GHG', value_col]).itertuples(index=False)
                    }
                else:
                    val_map = {
                        (str(r.Item_Emis).strip(), str(r.Process).strip()): float(getattr(r, value_col))
                        for r in df[['Item_Emis', 'Process', value_col]].dropna(subset=['Item_Emis', 'Process', value_col]).itertuples(index=False)
                    }
                for y in future_years:
                    for i in eff.countries:
                        for j in eff.commodities:
                            for p in eff.processes:
                                if use_ghg:
                                    for ghg in df['GHG'].dropna().astype(str).str.strip().unique().tolist():
                                        v = val_map.get((j, p, ghg))
                                        if v is None or not np.isfinite(v):
                                            continue
                                        out['emission_factor_absolute_by'][(i, j, p, ghg, y)] = float(v)
                                else:
                                    v = val_map.get((j, p))
                                    if v is None or not np.isfinite(v):
                                        continue
                                    out['emission_factor_absolute_by'][(i, j, p, y)] = float(v)
                continue
            if eff.kind in ('manure_management_ratio', 'mm_ratio', 'manure_ratio'):
                df = _best_value_map_for_sheet("manure_mgmt_ratio_mult", value_col)
                if "Item_Emis" not in df.columns:
                    continue
                val_map = {
                    str(r.Item_Emis).strip(): float(getattr(r, value_col))
                    for r in df[['Item_Emis', value_col]].dropna(subset=['Item_Emis', value_col]).itertuples(index=False)
                }
                for y in future_years:
                    for i in eff.countries:
                        for j in eff.commodities:
                            v = val_map.get(j)
                            if v is None or not np.isfinite(v):
                                continue
                            out['manure_management_ratio_absolute_by'][(i, j, y)] = float(v)
                continue
            if eff.kind in ('losses_ratio', 'loss_ratio', 'losses_rate', 'loss_rate',
                            'waste_reduction', 'waste_reduction_rate', 'waste_rate', 'waste'):
                df = _best_value_map_for_sheet("waste_reduction_multiplier", value_col)
                if "Item_Emis" not in df.columns:
                    continue
                val_map = {
                    str(r.Item_Emis).strip(): float(getattr(r, value_col))
                    for r in df[['Item_Emis', value_col]].dropna(subset=['Item_Emis', value_col]).itertuples(index=False)
                }
                for y in future_years:
                    for i in eff.countries:
                        for j in eff.commodities:
                            v = val_map.get(j)
                            if v is None or not np.isfinite(v):
                                continue
                            delta = max(-1.0, min(1.0, float(v)))
                            out['losses_ratio_by'][(i, j, y)] = delta
                continue
            # Unsupported best_value kind -> skip
            continue
        timeline = _linear_path_2020_2080(float(eff.value_2080), eff.unit, universe.years)
        abs_unit = str(eff.unit).lower() in ('absolute', 'abs', 'value', 'level')
        if eff.kind in ('feed_intensity','feed_intensity_rate','feed_intensity_improve',
                        'feed_efficiency','feed_efficiency_rate','feed_efficiency_improve'):
            for y,val in timeline.items():
                if eff.unit=='rate':
                    rate = float(val)
                    rate = max(-1.0, min(1.0, rate))
                    for i in eff.countries:
                        for j in eff.commodities:
                            key = (i, j, y)
                            prev = out['dm_conversion_multiplier'].get(key, 1.0)
                            mult = max(0.0, 1.0 + rate)
                            eff_mult = max(0.0, prev * mult)
                            out['dm_conversion_multiplier'][key] = eff_mult
                            out['feed_reduction_by'][i, j, y] = eff_mult - 1.0
                elif eff.unit=='multiplier':
                    mult = max(0.0, float(val))
                    for i in eff.countries:
                        for j in eff.commodities:
                            key = (i, j, y)
                            prev = out['dm_conversion_multiplier'].get(key, 1.0)
                            eff_mult = max(0.0, prev * mult)
                            out['dm_conversion_multiplier'][key] = eff_mult
                            out['feed_reduction_by'][i, j, y] = eff_mult - 1.0
                elif abs_unit:
                    if y <= base_year: continue
                    mult = max(0.0, float(val))
                    for i in eff.countries:
                        for j in eff.commodities:
                            key = (i, j, y)
                            out['dm_conversion_multiplier'][key] = mult
                            out['feed_reduction_by'][i, j, y] = mult - 1.0
        elif eff.kind in ('land_carbon_price','land_co2_price'):
            for y,val in timeline.items():
                if str(eff.unit).strip().lower() in ('amount', 'price', 'value', 'absolute', 'abs', 'level'):
                    try:
                        price_val = float(val)
                    except Exception:
                        continue
                    if not np.isfinite(price_val):
                        continue
                    out['land_carbon_price_by_year'][y] = price_val
                    out['land_carbon_price'] = price_val
        elif eff.kind in ('ruminant_intake_decreasing_ratio','ruminant_intake_ratio','ruminant_reduction'):
            for y,val in timeline.items():
                # Apply only in future years; historical demand is fixed by demand_fixed.
                if y <= 2020:
                    continue
                if eff.unit=='rate':
                    for i in eff.countries:
                        for j in eff.commodities:
                            # Apply to ruminants only.
                            if j in RUMINANT_COMMS:
                                cap = (1.0 + val) * base_rumi.get((i, j), 0.0)
                                out['ruminant_intake_cap'][(i, j, y)] = max(0.0, cap)
        
        # Adjust emission factors downward.
        
        elif eff.kind in ('emission_factor', 'ef_multiplier', 'emission_factor_multiplier'):
            for y, val in timeline.items():
                if eff.unit == 'rate':
                    # rate????????????multiplier = 1.0 + rate (??ate=-0.3, mult=0.7)
                    mult = max(0.0, 1.0 + float(val))
                    ghg_sel = getattr(eff, 'ghg_sel', 'All') or 'All'
                    for i in eff.countries:
                        for j in eff.commodities:
                            for p in eff.processes:
                                key = (i, j, p, ghg_sel, y) if ghg_sel != 'All' else (i, j, p, y)
                                out['emission_factor_multiplier'][key] = mult

                elif eff.unit == 'multiplier':
                    mult = max(0.0, float(val))
                    ghg_sel = getattr(eff, 'ghg_sel', 'All') or 'All'
                    for i in eff.countries:
                        for j in eff.commodities:
                            for p in eff.processes:
                                key = (i, j, p, ghg_sel, y) if ghg_sel != 'All' else (i, j, p, y)
                                out['emission_factor_multiplier'][key] = mult
                elif abs_unit:
                    if y <= base_year: continue
                    ghg_sel = getattr(eff, 'ghg_sel', 'All') or 'All'
                    for i in eff.countries:
                        for j in eff.commodities:
                            for p in eff.processes:
                                out['emission_factor_absolute_by'][(i, j, p, ghg_sel, y)] = float(val)
        
        elif eff.kind in ('fertilizer_rate', 'fertlizer_rate', 'fertilizer_efficiency'):
            for y, val in timeline.items():
                if eff.unit == 'rate':
                    # rate????????????multiplier = 1.0 + rate
                    mult = max(0.0, 1.0 + float(val))
                    for i in eff.countries:
                        for j in eff.commodities:
                            out['fertilizer_rate_multiplier'][(i, j, y)] = mult

                elif eff.unit == 'multiplier':
                    mult = max(0.0, float(val))
                    for i in eff.countries:
                        for j in eff.commodities:
                            out['fertilizer_rate_multiplier'][(i, j, y)] = mult
                elif abs_unit:
                    if y <= base_year: continue
                    for i in eff.countries:
                        for j in eff.commodities:
                            out['fertilizer_efficiency_absolute_by'][(i, j, y)] = float(val)
        # yield??
        elif eff.kind in ('yield_rate', 'yield_multiplier', 'yield_improvement'):
            for y, val in timeline.items():
                if eff.unit == 'rate':
                    # rate????????????multiplier = 1.0 + rate (??ate=0.3, mult=1.3)
                    mult = max(0.0, 1.0 + float(val))
                    for i in eff.countries:
                        for j in eff.commodities:
                            out['yield_multiplier'][(i, j, y)] = mult

                elif eff.unit == 'multiplier':
                    mult = max(0.0, float(val))
                    for i in eff.countries:
                        for j in eff.commodities:
                            out['yield_multiplier'][(i, j, y)] = mult
                elif abs_unit:
                    if y <= base_year: continue
                    for i in eff.countries:
                        for j in eff.commodities:
                            out['yield_absolute_by'][(i, j, y)] = float(val)
        # New: aquaculture share adjustment (multiplier)
        elif eff.kind in ('aquaculture_share', 'aquaculture_share_multiplier', 'aquaculture_share_rate', 'aquaculture_ratio'):
            for y, val in timeline.items():
                if eff.unit == 'rate':
                    mult = max(0.0, 1.0 + float(val))
                    for i in eff.countries:
                        for j in eff.commodities:
                            out['aquaculture_share_multiplier'][(i, j, y)] = mult

                elif eff.unit == 'multiplier':
                    mult = max(0.0, float(val))
                    for i in eff.countries:
                        for j in eff.commodities:
                            out['aquaculture_share_multiplier'][(i, j, y)] = mult
                elif abs_unit:
                    if y <= base_year: continue
                    for i in eff.countries:
                        for j in eff.commodities:
                            out['aquaculture_share_absolute_by'][(i, j, y)] = float(val)
        
        elif eff.kind in ('manure_management_ratio', 'mm_ratio', 'manure_ratio'):
            for y, val in timeline.items():
                if eff.unit == 'rate':
                    # rate????????????multiplier = 1.0 + rate
                    mult = max(0.0, 1.0 + float(val))
                    for i in eff.countries:
                        for j in eff.commodities:
                            out['manure_management_ratio_multiplier'][(i, j, y)] = mult

                elif eff.unit == 'multiplier':
                    mult = max(0.0, float(val))
                    for i in eff.countries:
                        for j in eff.commodities:
                            out['manure_management_ratio_multiplier'][(i, j, y)] = mult
                elif abs_unit:
                    if y <= base_year: continue
                    for i in eff.countries:
                        for j in eff.commodities:
                            out['manure_management_ratio_absolute_by'][(i, j, y)] = float(val)
        # losses_ratio is an additive delta to the baseline losses ratio.
        # The demand model clips final_loss_ratio = baseline + delta to [0, 1).
        elif eff.kind in ('losses_ratio', 'loss_ratio', 'losses_rate', 'loss_rate'):
            for y, val in timeline.items():
                if y <= base_year:
                    continue
                try:
                    delta = float(val)
                except Exception:
                    continue
                delta = max(-1.0, min(1.0, delta))
                if eff.unit == 'multiplier':
                    delta = max(-1.0, min(1.0, delta - 1.0))
                for i in eff.countries:
                    for j in eff.commodities:
                        out['losses_ratio_by'][(i, j, y)] = delta
        # Legacy waste_reduction names now use the same additive delta semantics,
        # allowing positive values for waste increases.
        elif eff.kind in ('waste_reduction', 'waste_reduction_rate', 'waste_rate', 'waste'):
            for y, val in timeline.items():
                if y <= base_year: continue
                if eff.unit == 'rate':
                    try:
                        rate = float(val)
                    except Exception:
                        continue
                    rate = max(-1.0, min(1.0, rate))
                    for i in eff.countries:
                        for j in eff.commodities:
                            out['waste_reduction_rate_by'][(i, j, y)] = rate
                elif eff.unit == 'multiplier':
                    mult = max(0.0, float(val))
                    rate = max(-1.0, min(1.0, mult - 1.0))
                    for i in eff.countries:
                        for j in eff.commodities:
                            out['waste_reduction_rate_by'][(i, j, y)] = rate
                elif abs_unit:
                    if y <= base_year: continue
                    try:
                        rate = float(val)
                    except Exception:
                        continue
                    rate = max(-1.0, min(1.0, rate))
                    for i in eff.countries:
                        for j in eff.commodities:
                            out['waste_reduction_rate_by'][(i, j, y)] = rate
        elif eff.kind in ('crop_soil_management_ratio', 'crop_soil_ratio'):
            proc_list = _resolve_crop_soil_processes(
                getattr(eff, 'processes', None),
                process_sel=getattr(eff, 'process_sel', 'All')
            )
            if not proc_list:
                continue
            for y, val in timeline.items():
                if y <= base_year:
                    continue
                if eff.unit == 'rate':
                    mult = max(0.0, 1.0 + float(val))
                elif eff.unit == 'multiplier' or abs_unit:
                    mult = max(0.0, float(val))
                else:
                    continue
                for i in eff.countries:
                    for j in eff.commodities:
                        for p in proc_list:
                            out['crop_soil_management_multiplier'][(i, j, p, y)] = mult
    _restrict_scenario_maps_to_active_nodes(
        out,
        nodes,
        excluded_commodities=excluded_commodities,
    )
    return out
def load_scenarios(xlsx_path: str, universe: Universe, sheet: str='Scenario') -> List[ScenarioEffect]:
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    # Adapt column names.
    col_id   = 'Scenario ID' if 'Scenario ID' in df.columns else 'ScenarioID'
    col_type = 'Scenario Element' if 'Scenario Element' in df.columns else 'Scenario Type'
    col_unit = (
        'Scenario Unit' if 'Scenario Unit' in df.columns
        else ('Element unit' if 'Element unit' in df.columns else 'Unit')
    )
    col_proc = 'Emis Process' if 'Emis Process' in df.columns else 'Emis process'
    col_reg  = 'Region' if 'Region' in df.columns else 'Country'
    col_comm = 'Commodity' if 'Commodity' in df.columns else 'Item'
    col_val  = 'Value'
    
    effects = []
    # Use iterrows, not itertuples, because spaces in column names cause attribute-name issues.
    for idx, row in df.iterrows():
        sid  = row[col_id]
        kind = str(row[col_type]).strip().lower().replace(' ', '_')
        unit = str(row[col_unit]).strip().lower()
        val_raw = row[col_val]
        reg  = row.get(col_reg, 'All')
        com  = row.get(col_comm, 'All')
        proc = row.get(col_proc, 'All')
        val = _parse_scenario_value(kind, val_raw)
        eff = ScenarioEffect(sid, kind, unit, val, reg, com, proc)
        eff.countries   = _select_countries(universe, reg)
        eff.commodities = _select_commodities(universe, com)
        eff.processes   = _select_processes(universe, proc)
        if eff.kind in ('crop_soil_management_ratio', 'crop_soil_ratio'):
            eff.processes = _resolve_crop_soil_processes(eff.processes, process_sel=proc)
        effects.append(eff)
    return effects

def load_mc_specs(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name='MC')
    df.columns = [str(c).strip() for c in df.columns]
    # Required: Element, Element unit, Process, Item, GHG, Min_bound, Max_bound, Region_cat.
    return df


def _mc_draw_to_multiplier(draw: float, unit: str) -> float:
    """Legacy MC dictionaries store multipliers; rate draws are relative changes."""
    unit_l = str(unit or '').strip().lower()
    if unit_l in {'rate', 'ratio', 'pct', 'percent', 'percentage'}:
        return max(0.0, 1.0 + float(draw))
    return max(0.0, float(draw))


def draw_mc_to_params(df_specs: pd.DataFrame, universe: Universe, *, seed: int, draw_idx: int) -> dict:
    rng = np.random.default_rng(seed + draw_idx)
    ef_mult = {}   # ((country, commodity, process, year) -> factor)
    feed_conv_mult: Dict[Tuple[str, str, int], float] = {}
    yield_mult: Dict[Tuple[str, str, int], float] = {}
    aquaculture_share_mult: Dict[Tuple[str, str, int], float] = {}
    cols = [str(c).strip() for c in df_specs.columns]
    for r in df_specs.itertuples(index=False, name=None):
        row = dict(zip(cols, r))
        elem = str(row.get('Element', '')).lower()
        unit = str(row.get('Element unit', '')).lower()
        proc = str(row.get('Process', 'All')) if 'Process' in row else 'All'
        item = str(row.get('Item', 'All')) if 'Item' in row else 'All'
        ghg  = str(row.get('GHG', '')) if 'GHG' in row else ''
        regc = str(row.get('Region_cat', 'All'))
        mult = draw_mc_multiplier(rng, row.get('Min_bound', 1.0), row.get('Max_bound', 1.0), unit)

        if 'feed efficiency' in elem or 'feed intensity' in elem:
            if item and item.lower()!='all':
                comms = _select_commodities(universe, item)
            else:
                comms = list(universe.commodities)
            if not comms:
                continue
            if regc and regc.lower()!='all':
                inv = {}
                for c, ragg in (universe.region_aggMC_by_country or {}).items():
                    inv.setdefault(str(ragg), []).append(c)
                countries = inv.get(regc, [])
            else:
                countries = list(universe.countries)
            if not countries:
                continue
            for i in countries:
                for j in comms:
                    for y in universe.years:
                        feed_conv_mult[(i, j, y)] = mult
        # Yield multiplier (applies to all commodities unless filtered)
        elif 'yield' in elem and 'feed' not in elem:
            if item and item.lower()!='all':
                comms = _select_commodities(universe, item)
            else:
                comms = list(universe.commodities)
            if not comms:
                continue
            if regc and regc.lower()!='all':
                inv = {}
                for c, ragg in (universe.region_aggMC_by_country or {}).items():
                    inv.setdefault(str(ragg), []).append(c)
                countries = inv.get(regc, [])
            else:
                countries = list(universe.countries)
            if not countries:
                continue
            for i in countries:
                for j in comms:
                    for y in universe.years:
                        yield_mult[(i, j, y)] = mult
        # Aquaculture share multiplier (fish-specific)
        elif 'share' in elem and ('aqua' in elem or 'fish' in elem):
            if item and item.lower()!='all':
                comms = _select_commodities(universe, item)
            else:
                comms = list(universe.commodities)
            if not comms:
                continue
            if regc and regc.lower()!='all':
                inv = {}
                for c, ragg in (universe.region_aggMC_by_country or {}).items():
                    inv.setdefault(str(ragg), []).append(c)
                countries = inv.get(regc, [])
            else:
                countries = list(universe.countries)
            if not countries:
                continue
            for i in countries:
                for j in comms:
                    for y in universe.years:
                        aquaculture_share_mult[(i, j, y)] = mult
        # Support emission-factor/intensity uncertainty only, identified by keywords.
        elif 'ef' in elem or 'emission' in elem:
            procs = _select_processes(universe, proc)
            # Item selection
            if item and item.lower()!='all':
                comms = _select_commodities(universe, item)
            else:
                comms = list(universe.commodities)
            # Region selection
            if regc and regc.lower()!='all':
                # Region_aggMC classification
                inv = {}
                for c, ragg in (universe.region_aggMC_by_country or {}).items():
                    inv.setdefault(str(ragg), []).append(c)
                countries = inv.get(regc, [])
            else:
                countries = list(universe.countries)
            # Sampling
            for i in countries:
                for j in comms:
                    for p in procs:
                        for y in universe.years:
                            ef_mult[i,j,p,y] = mult
    return {
        'ef_multiplier_by': ef_mult,
        'dm_conversion_multiplier': feed_conv_mult,
        'yield_multiplier': yield_mult,
        'aquaculture_share_multiplier': aquaculture_share_mult
    }


def _mc_element_to_kind(elem: str) -> Optional[str]:
    elem_l = str(elem or '').strip().lower()
    if not elem_l:
        return None
    if 'land' in elem_l and ('carbon' in elem_l or 'co2' in elem_l) and 'price' in elem_l:
        return 'land_carbon_price'
    if 'ruminant' in elem_l or 'ruminate' in elem_l or 'intake' in elem_l:
        return 'ruminant_intake_ratio'
    if 'feed' in elem_l and ('intensity' in elem_l or 'eff' in elem_l):
        return 'feed_intensity'
    if 'fertilizer' in elem_l or 'fertlizer' in elem_l:
        return 'fertilizer_rate'
    if 'manure' in elem_l and ('ratio' in elem_l or 'management' in elem_l):
        return 'manure_management_ratio'
    if 'yield' in elem_l and 'feed' not in elem_l:
        return 'yield_rate'
    if 'loss' in elem_l:
        return 'losses_ratio'
    if 'waste' in elem_l:
        return 'waste_reduction'
    if 'crop_soil_management' in elem_l or ('soil' in elem_l and 'management' in elem_l and 'ratio' in elem_l):
        return 'crop_soil_management_ratio'
    if 'ef' in elem_l or 'emission' in elem_l:
        return 'emission_factor'
    return None


def _normalize_mc_unit(raw: Any, default: str) -> str:
    val = str(raw or '').strip().lower()
    if val in ('rate', 'ratio', 'pct', 'percent', 'percentage'):
        return 'rate'
    if val in ('amount', 'price', 'usd', '$/t', '$/tco2e', 'value'):
        return 'amount'
    if val in ('multiplier', 'factor'):
        return 'multiplier'
    if val in ('absolute', 'abs', 'level', 'value'):
        return 'absolute'
    if val in ('best_value', 'bestvalue', 'best'):
        return 'best_value'
    if val in ('profile', 'path', 'file', 'xlsx'):
        return 'profile'
    return default


def draw_mc_effects(specs_df: pd.DataFrame,
                    universe: Universe,
                    *,
                    seed: int,
                    draw_idx: int,
                    scenario_id: str = 'MC') -> List[ScenarioEffect]:
    """Draw MC samples as ScenarioEffect list for use with apply_scenario_to_data()."""
    rng = np.random.default_rng(seed + draw_idx)
    df = specs_df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    effects: List[ScenarioEffect] = []
    for row in df.to_dict(orient="records"):
        elem_raw = row.get('Element', '')
        kind = _mc_element_to_kind(elem_raw)
        if not kind:
            continue
        lo, lo_y2020 = _parse_mc_bound_value(row.get('Min_bound', 0.0))
        hi, hi_y2020 = _parse_mc_bound_value(row.get('Max_bound', 0.0))
        if lo is None or hi is None:
            raise ValueError(f"Invalid MC bounds for {elem_raw!r}")
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(f"Non-finite MC bounds for {elem_raw!r}")
        if hi < lo and lo_y2020 == hi_y2020:
            lo, hi = hi, lo
        u = float(rng.uniform(0.0, 1.0))
        mixed_bounds = bool(lo_y2020 != hi_y2020)
        if mixed_bounds:
            draw = u
        elif hi == lo:
            draw = lo
        else:
            draw = lo + (hi - lo) * u
        unit_raw = row.get('Element unit', '') if 'Element unit' in df.columns else ''
        if kind in ('land_carbon_price', 'land_co2_price'):
            default_unit = 'amount'
        elif kind in ('ruminant_intake_ratio', 'waste_reduction', 'losses_ratio', 'crop_soil_management_ratio'):
            default_unit = 'rate'
        else:
            default_unit = 'absolute'
        unit = _normalize_mc_unit(unit_raw, default_unit)
        if (lo_y2020 or hi_y2020) and kind != 'emission_factor':
            if mixed_bounds:
                raise ValueError(f'Mixed absolute/Y2020 bounds are unsupported for {kind}')
            unit = 'multiplier'
        value = draw
        regc = str(row.get('Region_cat', 'All') or 'All')
        item = str(row.get('Item', 'All') or 'All')
        proc = str(row.get('Process', 'All') or 'All')
        eff = ScenarioEffect(scenario_id, kind, unit, value, regc, item, proc)
        eff.countries = _select_countries(universe, regc)
        eff.commodities = _select_commodities(universe, item)
        eff.processes = _select_processes(universe, proc)
        if kind == 'crop_soil_management_ratio':
            eff.processes = _resolve_crop_soil_processes(eff.processes, process_sel=proc)
        if kind == 'emission_factor' and 'GHG' in df.columns:
            eff.ghg_sel = str(row.get('GHG', 'All') or 'All')
        if (lo_y2020 or hi_y2020) and kind == 'emission_factor':
            eff.mc_bounds = {
                'lo': float(lo),
                'hi': float(hi),
                'lo_is_y2020': bool(lo_y2020),
                'hi_is_y2020': bool(hi_y2020),
                'u': float(u),
            }
        effects.append(eff)
    return effects
