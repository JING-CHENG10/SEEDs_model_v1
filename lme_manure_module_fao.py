
# -*- coding: utf-8 -*-
"""
lme_manure_module_fao_wide_cn.py
================================
Livestock Manure: calculations using a unified WIDE parameter table, with detailed documentation.

Implement Tier 1 logic for the FAOSTAT Livestock Manure domain,
reading all coefficients, including N.excretion.rate, MMS shares MS, system losses Frac.loss,
and volatilization/leaching fractions Frac.GASM/Frac.LEACH, from one WIDE table
for consistency with crop and livestock modules.

1. Input data: WIDE parameter table
--------------------------
- CSV or XLSX; CSV is recommended for I/O performance. Default Excel sheet: LIV_parameters_WIDE.
- Required columns:
  AreaCode, IPCC_AreaCode, ItemCode, ItemName, Process, ParamName, ParamMMS, ParamCode, Units, Source, 1990..2022
  * Year columns have four-digit names, automatically recognized as strings or numbers.
  * Prepared from Parameters_Livestock.xlsx::parameters_AreaCode; see the upstream script.

- Key ParamName values are case-insensitive and support aliases:
  * N.excretion.rate: kg N/head/year; multiplying by headcount gives total manure N excretion.
  * MS (ParamMMS=system): management pathway/system shares, including Pasture and Burned for fuel.
  * Frac.loss (ParamMMS=system): total system loss fraction, combining NH3/NOx/N2O/N2/leakage.
  * Frac.GASM: volatilization fraction for NH3+NOx, applied to the total.
  * Frac.LEACH: leaching/runoff fraction.

- System names (ParamMMS):
  * Special pathways excluded from system losses: Pasture and Burned for fuel.
  * Management systems subject to system losses:
    ["Lagoon","Slurry","Solid storage","Drylot","Daily spread",
     "Anaerobic digester","Pit < 1 month","Pit ≥ 1 month","Other"]

2. External animal population inputs (stocks)
--------------------------
- Caller supplies a tidy DataFrame containing at least:
  AreaCode, year, ItemCode (or ItemName), head (standing stock in heads).

3. Outputs
--------
- Tidy table: ['AreaCode','year','ItemCode','ItemName','element','value_ktN'].
- Units: kt N/year; kg-to-kt conversion is internal.
- Elements aligned with FAOSTAT:
  1) Amount excreted in manure (N content)
  2) Manure left on pasture (N content)
  3) Manure left on pasture that volatilises (N content)
  4) Manure left on pasture that leaches (N content)
  5) Manure treated (N content)
  6) Losses from manure treated (N content)
  7) Manure applied to soils (N content)
  8) Manure applied to soils that volatilises (N content)
  9) Manure applied to soils that leaches (N content)

4. Core formulas, consistent with FAOSTAT documentation
----------------------------------
- Total manure excretion: N_total = heads * N.excretion.rate (kg N/year).
- Left on pasture: N_pasture = N_total * (MS_Pasture + 0.5 * MS_BurnedForFuel).
- Pasture volatilization/leaching: N_pasture multiplied by Frac.GASM or Frac.LEACH, respectively.
- Entering managed systems: N_treated = N_total * sum(MMS shares excluding Pasture/Burned).
- System losses: Loss = sum(N_total * MS_sys * Frac.loss_sys).
- Applied to soil: N_applied = max(N_treated - Loss, 0).
- Post-application volatilization/leaching: N_applied multiplied by Frac.GASM or Frac.LEACH, respectively.

5. Usage example
------------
    from lme_manure_module_fao_wide_cn import load_parameters_wide, run_lme_from_wide
    P = load_parameters_wide("LIV_parameters_WIDE_1990_2022.csv")
    populations = ... # DataFrame(AreaCode, year, ItemCode/ItemName, head)
    out = run_lme_from_wide(P, populations, years=[2020])
    print(out.head())

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd

# Constant for kg-to-kt conversion
KG_TO_KT = 1e-6

# MMS systems subject to system losses, commonly used in IPCC/FAOSTAT accounting
MMS_CANON = [
    "Lagoon", "Slurry", "Solid storage", "Drylot", "Daily spread",
    "Anaerobic digester", "Pit < 1 month", "Pit ≥ 1 month", "Other"
]

# Special pathways bypass system losses: include Pasture plus 50% of Burned for fuel as pasture urinary N.
SPECIAL_PATHS = ["Pasture", "Burned for fuel"]

# 
# 1) Load the WIDE table.
# 

def load_parameters_wide(path: str, sheet: str = "LIV_parameters_WIDE") -> pd.DataFrame:
    """
    Read CSV/XLSX WIDE data and standardize fields:
    - Convert ID fields to nullable Int64 for joining external stock tables.
    - Retain all year columns (1990..2022) for on-demand access.
    - Add lowercase matching fields _param / _mms / _item / _process.
    """
    # Support CSV and Excel.
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path, dtype=str)
    else:
        df = pd.read_excel(path, sheet_name=sheet, dtype=str)

    # Trim column-name whitespace.
    df.columns = [str(c).strip() for c in df.columns]

    # Convert ID columns to nullable Int64.
    for c in ["AreaCode","IPCC_AreaCode","ItemCode","ParamCode"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # Add helper columns for case-insensitive matching.
    df["_param"] = df.get("ParamName","").astype(str).str.strip().str.lower()
    df["_mms"]   = df.get("ParamMMS","").astype(str).str.strip()
    df["_item"]  = df.get("ItemName","").astype(str).str.strip()
    df["_process"] = df.get("Process","").astype(str).str.strip()

    # Identify year columns, including string names.
    years = [int(c) for c in df.columns if c.isdigit()]
    df["_years"] = [years]*len(df)
    return df

# 
# 2) Parameter lookup by key and year, with aliases and temporal fallback
# 

# Case-insensitive ParamName aliases
ALIASES = {
    "n.excretion.rate": ["n.excretion.rate", "nex", "n_excretion_rate"],
    "ms": ["ms","share","share.ms","mms.share"],
    "frac.loss": ["frac.loss","loss.frac","loss_fraction"],
    "frac.gasm": ["frac.gasm","frac_nh3_nox","frac.volatilisation"],
    "frac.leach": ["frac.leach","frac_leaching","frac.runoff"],
}

def _match_param(df: pd.DataFrame, pname: str) -> pd.Series:
    """
    Check whether rows match the requested ParamName, including aliases.
    Return a Boolean Series indicating matches.
    """
    key = pname.strip().lower()
    cands = ALIASES.get(key, [key])
    return df["_param"].isin(cands)

def _row_year_value(row: pd.Series, year: int) -> Optional[float]:
    """
    Get a record's target-year value; if empty:
    1) Search the nearest earlier year with a value.
    2) If still empty, search the nearest later year with a value.
    Return float on success, otherwise None.
    """
    years = [int(c) for c in row.index if str(c).isdigit()]
    years.sort()
    # Current year
    if str(year) in row.index and pd.notna(row[str(year)]):
        return float(row[str(year)])
    # Search earlier years.
    for y in sorted([y for y in years if y < year], reverse=True):
        v = row[str(y)]
        if pd.notna(v): return float(v)
    # Search later years.
    for y in sorted([y for y in years if y > year]):
        v = row[str(y)]
        if pd.notna(v): return float(v)
    return None

def get_param(
    P: pd.DataFrame, *, year: int, areacode: int,
    itemcode: Optional[int] = None, itemname: Optional[str] = None,
    param: str, mms: Optional[str] = None, default: Optional[float] = None
) -> Optional[float]:
    """
    Retrieve parameter values from the WIDE table.
    Matching priority:
      (1) AreaCode + ItemCode or ItemName + ParamName (+ParamMMS).
      (2) AreaCode + ParamName (+ParamMMS), unrestricted by Item.
      (3) Global AreaCode==0 + ParamName (+ParamMMS).

    Parameters
    ----
    P: DataFrame returned by load_parameters_wide().
    year: Year (int).
    areacode: FAOSTAT country/area code.
    itemcode: Optional livestock species code.
    itemname: Optional species name; unnecessary when itemcode is supplied.
    param: Case-insensitive ParamName with alias support.
    mms: ParamMMS management system/pathway, e.g. Slurry or Pasture.
    default: Value returned when lookup fails.

    Returns
    ----
    float or None.
    """
    sub = P.copy()

    # 1) Filter by AreaCode first.
    sub = sub[(sub["AreaCode"].astype("Int64")==pd.Series([areacode]*len(sub), dtype="Int64"))]

    # 2) Prefer ItemCode, then ItemName; leave unrestricted if neither is supplied.
    if itemcode is not None and "ItemCode" in sub.columns:
        sub_item = sub[sub["ItemCode"].astype("Int64")==pd.Series([itemcode]*len(sub), dtype="Int64")]
    else:
        sub_item = pd.DataFrame(columns=sub.columns)
    if itemname:
        sub_name = sub[sub["_item"].str.lower()==str(itemname).strip().lower()]
    else:
        sub_name = pd.DataFrame(columns=sub.columns)
    sub_any = pd.concat([sub_item, sub_name]).drop_duplicates() if not sub_item.empty or not sub_name.empty else sub

    # 3) Match ParamName and optional ParamMMS within candidates.
    m = _match_param(sub_any, param)
    if mms is not None:
        m = m & (sub_any["_mms"].str.lower()==str(mms).strip().lower())
    cand = sub_any[m]
    if not cand.empty:
        r = cand.iloc[0]  # Take the first matching row; additional Source/Units rules can be added.
        return _row_year_value(r, year)

    # 4) Fall back to AreaCode + ParamName (+ParamMMS), without Item restrictions.
    m = _match_param(sub, param)
    if mms is not None:
        m = m & (sub["_mms"].str.lower()==str(mms).strip().lower())
    cand = sub[m]
    if not cand.empty:
        r = cand.iloc[0]
        return _row_year_value(r, year)

    # 5) Global fallback: AreaCode==0.
    g = P[P["AreaCode"].fillna(0).astype(int)==0]
    if not g.empty:
        m = _match_param(g, param)
        if mms is not None:
            m = m & (g["_mms"].str.lower()==str(mms).strip().lower())
        cand = g[m]
        if not cand.empty:
            r = cand.iloc[0]
            return _row_year_value(r, year)

    return default

def get_share(P: pd.DataFrame, year: int, areacode: int, itemcode: Optional[int], itemname: Optional[str], system: str) -> float:
    """Read management system/pathway share MS; return zero if missing."""
    v = get_param(P, year=year, areacode=areacode, itemcode=itemcode, itemname=itemname, param="MS", mms=system, default=0.0)
    return float(v) if v is not None else 0.0

def get_loss_frac(P: pd.DataFrame, year: int, areacode: int, itemcode: Optional[int], itemname: Optional[str], system: str) -> float:
    """Read system loss fraction Frac.loss; return zero if missing."""
    v = get_param(P, year=year, areacode=areacode, itemcode=itemcode, itemname=itemname, param="Frac.loss", mms=system, default=0.0)
    return float(v) if v is not None else 0.0

def get_frac_scalar(P: pd.DataFrame, year: int, areacode: int, itemcode: Optional[int], itemname: Optional[str], key: str, default: float) -> float:
    """Read Frac.GASM/Frac.LEACH with regional or global defaults."""
    v = get_param(P, year=year, areacode=areacode, itemcode=itemcode, itemname=itemname, param=key, default=default)
    return float(v) if v is not None else default

# 
# 3) Calculate one AreaCode-year-species record.
# 

def compute_record(
    P: pd.DataFrame, *, year: int, areacode: int,
    itemcode: Optional[int], itemname: Optional[str],
    head: float
) -> List[Dict[str, object]]:
    """
    Calculate all elements for one country-year-species combination.
    Return a list of dictionaries, each containing one element's kt N value.
    """
    # 1. Nex: kg N/head/year.
    Nex = get_param(P, year=year, areacode=areacode, itemcode=itemcode, itemname=itemname, param="N.excretion.rate")
    if Nex is None or not np.isfinite(Nex):
        # Return NaN if Nex is missing, signaling an upstream parameter gap.
        return [{
            "AreaCode": areacode, "year": year, "ItemCode": itemcode, "ItemName": itemname,
            "element": "Amount excreted in manure (N content)", "value_ktN": np.nan
        }]

    # 2. Total manure excretion (kg N)
    N_total_kg = float(head) * float(Nex)

    # 3. MS shares for Pasture, Burned, and managed MMS systems
    ms = {sys: get_share(P, year, areacode, itemcode, itemname, sys) for sys in (SPECIAL_PATHS + MMS_CANON)}
    # Clamp negatives to zero; if the sum is positive, normalize shares to one for physical consistency.
    ms = {k: max(0.0, float(v or 0.0)) for k,v in ms.items()}
    ssum = sum(ms.values())
    if ssum > 0:
        ms = {k: v/ssum for k, v in ms.items()}

    # 4. Volatilization/leaching fractions; default if missing, but explicit WIDE values are recommended.
    FracGASM  = get_frac_scalar(P, year, areacode, itemcode, itemname, "Frac.GASM", default=0.10)
    FracLEACH = get_frac_scalar(P, year, areacode, itemcode, itemname, "Frac.LEACH", default=0.30)

    # 5. Pasture plus half of fuel-burning manure as urinary N
    MS_past = ms.get("Pasture", 0.0)
    MS_burn = ms.get("Burned for fuel", 0.0)
    N_pasture_kg = N_total_kg * (MS_past + 0.5*MS_burn)
    N_past_vol_kg   = N_pasture_kg * FracGASM
    N_past_leach_kg = N_pasture_kg * FracLEACH

    # 6. Entering managed systems, excluding Pasture/Burned
    MS_treated = sum(v for k,v in ms.items() if k not in SPECIAL_PATHS)
    N_treated_kg = N_total_kg * MS_treated

    # 7. System losses: N_total * MS_sys * Frac.loss_sys for each system.
    loss_kg = 0.0
    for sys in MMS_CANON:
        share = ms.get(sys, 0.0)
        if share <= 0: 
            continue
        L = get_loss_frac(P, year, areacode, itemcode, itemname, sys)
        loss_kg += N_total_kg * share * float(L or 0.0)

    # 8. Soil application: treat all remaining N after losses as applied.
    N_applied_kg = max(N_treated_kg - loss_kg, 0.0)
    N_appl_vol_kg   = N_applied_kg * FracGASM
    N_appl_leach_kg = N_applied_kg * FracLEACH

    # Helper: express all outputs in kt N.
    def row(elem, kg): 
        return {
            "AreaCode": areacode, "year": year,
            "ItemCode": itemcode, "ItemName": itemname,
            "element": elem, "value_ktN": float(kg) * KG_TO_KT
        }

    out = [
        row("Amount excreted in manure (N content)", N_total_kg),
        row("Manure left on pasture (N content)", N_pasture_kg),
        row("Manure left on pasture that volatilises (N content)", N_past_vol_kg),
        row("Manure left on pasture that leaches (N content)", N_past_leach_kg),
        row("Manure treated (N content)", N_treated_kg),
        row("Losses from manure treated (N content)", loss_kg),
        row("Manure applied to soils (N content)", N_applied_kg),
        row("Manure applied to soils that volatilises (N content)", N_appl_vol_kg),
        row("Manure applied to soils that leaches (N content)", N_appl_leach_kg),
    ]
    return out

# 
# 4) Batch orchestration by country-year-species
# 

def run_lme_from_wide(
    params_wide: pd.DataFrame,
    populations: pd.DataFrame,
    years: Optional[Iterable[int]] = None,
    itemcode_col: str = "ItemCode",
    itemname_col: str = "ItemName",
    head_col: str = "head"
) -> pd.DataFrame:
    """
    Calculate livestock manure N flows in batches, reading all coefficients from WIDE.

    Parameters
    ----
    params_wide: DataFrame read by load_parameters_wide().
    populations: Caller-supplied stock table with at least:
                  AreaCode, year, ItemCode (or ItemName), head.
    years: Calculation years; defaults to all years in populations.
    itemcode_col: Species code column (default ItemCode).
    itemname_col: Species name column (default ItemName).
    head_col: Stock column (default head).

    Returns
    ----
    Tidy output: AreaCode, year, ItemCode, ItemName, element, value_ktN.
    """
    P = params_wide.copy()

    # Standardize stock-table fields.
    pop = populations.copy()
    pop.columns = [c.strip() for c in pop.columns]
    pop["AreaCode"] = pd.to_numeric(pop["AreaCode"], errors="coerce").astype("Int64")
    pop["year"] = pd.to_numeric(pop["year"], errors="coerce").astype(int)
    pop[head_col] = pd.to_numeric(pop[head_col], errors="coerce").fillna(0.0)

    # Year range
    if years is None:
        years = sorted(pop["year"].unique().tolist())
    years = [int(y) for y in years]

    rows = []
    # Aggregate by country-year-species before calculation, summing head across duplicate species records.
    gcols = ["AreaCode","year", itemcode_col if itemcode_col in pop.columns else "ItemCode", itemname_col if itemname_col in pop.columns else "ItemName"]
    for (ac, y, ic, iname), g in pop.groupby(gcols, dropna=False):
        head = float(g[head_col].sum())
        recs = compute_record(
            P, year=int(y),
            areacode=int(ac) if pd.notna(ac) else 0,
            itemcode=int(ic) if pd.notna(ic) else None,
            itemname=str(iname) if pd.notna(iname) else None,
            head=head
        )
        rows.extend(recs)

    out = pd.DataFrame(rows)
    out["value_ktN"] = out["value_ktN"].astype(float)
    return out

__all__ = ["load_parameters_wide","get_param","run_lme_from_wide","compute_record","MMS_CANON","SPECIAL_PATHS"]
