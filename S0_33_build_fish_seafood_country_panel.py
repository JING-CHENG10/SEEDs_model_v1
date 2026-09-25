# -*- coding: utf-8 -*-
"""
S0_32_build_fish_seafood_country_panel_local.py

Build a country-year (2010–2020) panel for fish/seafood production (capture vs aquaculture),
aquaculture yield proxy, and CH4/N2O emission factors.

Key change vs. earlier versions
-------------------------------
- NO network download inside this script.
- All required source files must be pre-downloaded into: ../../input/Aquaculture/raw_unused_retired (default)
- Outputs are written to OUT_DIR.

Required local inputs (place in DATA_DIR)
-----------------------------------------
1) Our World in Data (OWID) Grapher "full" CSV exports (Entity, Code, Year, Value):
   - capture-fishery-production.csv
   - aquaculture-farmed-fish-production.csv

2) Food balance sheet production (required for scaling):
   - ../../input/Production_Trade/FoodBalanceSheets_E_All_Data_NOFLAG_demand_refilled*.csv / .xlsx

3) FAOSTAT Land Use CSV (required for yield proxy):
   - ../../input/Land/Inputs_LandUse_E_All_Data_NOFLAG_with_Pasture.csv
     Used to extract (sum): Land used for aquaculture + Inland waters used for aquac. or holding facilities.
     Also used to flag saline aquaculture (coastal/EEZ items) for CH4 EF selection.

4) IPCC default EFs (required):
   - ../../input/Aquaculture/raw_unused_retired/default_nonenergy_fish_CH4_N2O_EFs_IPCC.xlsx
     Used to populate EF columns (CH4/N2O, no CO2e conversion).

Local mapping input (NOT in DATA_DIR by default)
------------------------------------------------
- ../../src/dict_v3.xlsx (sheet: region)
  Provides Country ? ISO3 mapping and M49_Country_Code.
  Valid countries are those with Region_label_new != 'no'.

Outputs (written to OUT_DIR)
-----------------------------
- fish_seafood_country_panel_2000_present.xlsx

Notes
-----
- Emission factors are populated from IPCC defaults (CH4/N2O); no CO2e conversion is applied.
"""
from __future__ import annotations

import argparse
import json
import datetime
import re
import os
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import numpy as np
import pandas as pd



# User-configurable knobs

YEAR_MIN = 2010
YEAR_MAX = 2020
LIVE_TO_PRODUCT_YIELD_DEFAULT = 0.52

# OWID slugs (also expected as local file names: "<slug>.csv")
OWID_SLUG_CAPTURE = "capture-fishery-production"
OWID_SLUG_AQUA = "aquaculture-farmed-fish-production"
OWID_SLUG_DEMAND = "fish-and-seafood-consumption-per-capita"

# FishStatJ local exports (deprecated for this panel; kept for reference)
# Place these files in DATA_DIR (same folder as OWID files) unless specified otherwise.
FSJ_AQUA_QTY_FILE = "Aquaculture_Quantity.csv"          # long format with COUNTRY.UN_CODE, PERIOD, VALUE, MEASURE=Q_tlw
FSJ_AQUA_VAL_FILE = "Aquaculture_Value.csv"             # long format with MEASURE=V_USD_1000 (thousand USD)
# Alternative FishStatJ exports (wide format with year columns like "[2020]")
FSJ_AQUA_QTY_WIDE_FILE = "FAO_global_aquatic_production_quantity.csv"
FSJ_AQUA_VAL_WIDE_FILE = "FAO_global_aquatic_production_value.csv"
# FishStatJ capture production exports (optional; if present, will override OWID capture)
FSJ_CAPTURE_QTY_WIDE_FILE = "FAO_global_capture_production_quantity.csv"
FSJ_CAPTURE_VAL_WIDE_FILE = "FAO_global_capture_production_value.csv"
FSJ_SPECIES_GROUPS_FILE = "CL_FI_SPECIES_GROUPS.csv"    # mapping for SPECIES.ALPHA_3_CODE -> ISSCAAP/Yearbook groups (optional)

# FAO Food Balance Sheet aquatic products (deprecated for this panel)
FBS_AQUATIC_FILE = "FAO_FBS_aquatic_products.csv"
FBS_ELEMENT_DEMAND = "Total food supply"  # apparent consumption proxy (tonnes live weight)

# Optional trade/processed exports (not used in main panel)
TRADE_QTY_FILE = "FAO_global_aquatic_trade_quantity_aggregated_partner.csv"
TRADE_VAL_FILE = "FAO_global_aquatic_trade_value_aggregated_partner.csv"
PROC_PROD_FILE  = "FAO_global_aquatic_processed_production_statistics.csv"


POPULATION_CANDIDATES: List[str] = [
    "population",
    "population-with-un-projections",
    "population-long-run-with-projections",
]

# FAOSTAT bulk file (required)
FAOSTAT_LANDUSE_CSV = "Inputs_LandUse_E_All_Data_NOFLAG_with_Pasture.csv"
FAOSTAT_AQUA_ITEMS = [
    "Land used for aquaculture",
    "Inland waters used for aquac. or holding facilities",
]
FAOSTAT_SALINE_ITEMS = [
    "Coastal waters used for aquac. or holding facilities",
    "EEZ used for aquac. or holding facilities",
]

# Food balance sheet production file (Fish, Seafood total production; 1000 t)
FBS_PROD_FILE_STEM = "FoodBalanceSheets_E_All_Data_NOFLAG_demand_refilled"

# IPCC defaults (CH4/N2O) file
IPCC_EF_FILE = "default_nonenergy_fish_CH4_N2O_EFs_IPCC.xlsx"



RUN_LOG_TXT = "fish_seafood_build_log.txt"
RUN_LOG_JSON = "fish_seafood_build_log.json"


# Helpers

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _fmt_m49(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits == "":
        return None
    code = digits.zfill(3) if len(digits) <= 3 else digits
    return f"'{code}"


def _read_region_mapping_from_dict(dict_xlsx: str, sheet_name: str = "region") -> pd.DataFrame:
    """
    Read valid countries and their ISO3 + M49 mapping from dict_v3.xlsx (sheet: region).

    Valid countries: Region_label_new != 'no' (case-insensitive).
    Required columns:
      - Country
      - Region_label_new
      - Region_market_full
      - M49_Country_Code
      - an ISO3 column (any column whose name contains 'ISO3', case-insensitive)
    """
    df = pd.read_excel(dict_xlsx, sheet_name=sheet_name)

    required = ["Country", "Region_label_new", "Region_market_full", "M49_Country_Code"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"dict_v3.xlsx sheet '{sheet_name}' missing required columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    iso_cols = [c for c in df.columns if "iso3" in str(c).lower()]
    if not iso_cols:
        raise RuntimeError(
            f"dict_v3.xlsx sheet '{sheet_name}' does not contain an ISO3 mapping column (name must contain 'ISO3'). "
            f"Available columns: {list(df.columns)}"
        )
    iso_col = iso_cols[0]

    valid = df["Region_label_new"].astype(str).str.strip().str.lower().ne("no")
    df = df.loc[valid, ["Country", iso_col, "M49_Country_Code", "Region_label_new", "Region_market_full"]].copy()

    df["Country"] = df["Country"].astype(str).str.strip()
    df["ISO3"] = df[iso_col].astype(str).str.strip().str.upper()
    df["M49_Country_Code"] = df["M49_Country_Code"].apply(_fmt_m49)
    df["Region_label_new"] = df["Region_label_new"].astype(str).str.strip()
    df["Region_market_full"] = df["Region_market_full"].astype(str).str.strip()

    df = df.loc[df["Country"].notna() & (df["Country"] != "")]
    df = df.loc[df["ISO3"].notna() & (df["ISO3"] != "")]
    df = df.loc[df["M49_Country_Code"].notna() & (df["M49_Country_Code"] != "")]
    df = df.loc[df["Region_label_new"].notna() & (df["Region_label_new"] != "") & (df["Region_label_new"].str.lower() != "nan")]
    df = df.loc[df["Region_market_full"].notna() & (df["Region_market_full"] != "") & (df["Region_market_full"].str.lower() != "nan")]
    df = df.drop_duplicates(subset=["ISO3"]).reset_index(drop=True)

    bad_iso = df.loc[df["ISO3"].astype(str).str.len().ne(3), ["Country", "ISO3"]]
    if len(bad_iso) > 0:
        raise RuntimeError(
            "Found ISO3 codes not length 3 in dict mapping (please fix dict_v3.xlsx):\n"
            + bad_iso.to_string(index=False)
        )
    return df[["Country", "ISO3", "M49_Country_Code", "Region_label_new", "Region_market_full"]]


def _pick_value_column(df: pd.DataFrame) -> str:
    core = {"Entity", "Code", "Year"}
    value_cols = [c for c in df.columns if c not in core]
    if not value_cols:
        raise RuntimeError(f"OWID file has no value column. Columns={list(df.columns)}")
    if len(value_cols) == 1:
        return value_cols[0]

    # prefer numeric-like
    num_like = []
    for c in value_cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            num_like.append(c)
        else:
            frac_num = s.astype(str).str.match(r"^-?\d+(\.\d+)?$").mean()
            if frac_num > 0.8:
                num_like.append(c)
    return num_like[0] if num_like else value_cols[0]


def _read_owid_local_csv(slug: str, data_dir: str) -> pd.DataFrame:
    fp = Path(data_dir) / f"{slug}.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing required OWID file: {fp}")
    df = pd.read_csv(fp)

    # Normalize to Entity, Code, Year, Value
    if "Entity" not in df.columns or "Year" not in df.columns:
        raise RuntimeError(f"Unexpected OWID schema for {fp.name}. Columns={list(df.columns)}")

    if "Code" not in df.columns:
        # sometimes Code may be absent; but we need ISO3-like codes
        raise RuntimeError(f"OWID file {fp.name} must contain 'Code' column for ISO3.")

    vcol = _pick_value_column(df)
    df = df.rename(columns={vcol: "Value"})
    return df[["Entity", "Code", "Year", "Value"]]


def _load_population_local(data_dir: str) -> Tuple[pd.DataFrame, str]:
    """
    Return population DF and which slug was used, by checking local files in order.
    """
    for slug in POPULATION_CANDIDATES:
        fp = Path(data_dir) / f"{slug}.csv"
        if fp.exists():
            return _read_owid_local_csv(slug, data_dir), slug
    raise FileNotFoundError(
        f"Missing population file in {data_dir}. Expected one of: "
        + ", ".join([f"{s}.csv" for s in POPULATION_CANDIDATES])
    )



def _read_csv_fallback(path: Path, usecols=None, nrows=None) -> pd.DataFrame:
    """
    Read CSV with robust encoding fallback (FishStatJ/FAO exports can be latin1/cp1252).
    """
    for enc in ("utf-8", "utf-8-sig", "latin1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, usecols=usecols, nrows=nrows, low_memory=False)
        except Exception:
            continue
    # final fallback
    return pd.read_csv(path, encoding="latin1", usecols=usecols, nrows=nrows, low_memory=False)


def _read_fsj_aquaculture_quantity(data_dir: str) -> pd.DataFrame:
    """
    Read FishStatJ aquaculture quantity (long) and aggregate to country-year.

    Required columns:
      COUNTRY.UN_CODE (M49 numeric), PERIOD (year), VALUE, MEASURE (Q_tlw)
    Returns:
      M49_Country_Code (xxx str), Year, Aquaculture_prod_t
    """
    fp = Path(data_dir) / FSJ_AQUA_QTY_FILE
    if not fp.exists():
        raise FileNotFoundError(f"Missing required FishStatJ aquaculture quantity file: {fp}")

    df = _read_csv_fallback(fp, usecols=["COUNTRY.UN_CODE", "PERIOD", "VALUE", "MEASURE"])
    df = df.loc[df["MEASURE"].astype(str).str.strip() == "Q_tlw"].copy()
    df["COUNTRY.UN_CODE"] = pd.to_numeric(df["COUNTRY.UN_CODE"], errors="coerce")
    df["PERIOD"] = pd.to_numeric(df["PERIOD"], errors="coerce")
    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")

    df = df.dropna(subset=["COUNTRY.UN_CODE", "PERIOD"])
    df["M49_Country_Code"] = df["COUNTRY.UN_CODE"].apply(_fmt_m49)
    df["Year"] = df["PERIOD"].astype(int)

    out = (
        df.groupby(["M49_Country_Code", "Year"], as_index=False)["VALUE"]
          .sum()
          .rename(columns={"VALUE": "Aquaculture_prod_t"})
    )
    return out


def _read_fsj_aquaculture_value(data_dir: str) -> pd.DataFrame:
    """
    Read FishStatJ aquaculture value (long) and aggregate to country-year.

    MEASURE=V_USD_1000 (thousand USD).
    Returns:
      M49_Country_Code (xxx str), Year, Aquaculture_value_1000USD
    """
    fp = Path(data_dir) / FSJ_AQUA_VAL_FILE
    if not fp.exists():
        raise FileNotFoundError(f"Missing required FishStatJ aquaculture value file: {fp}")

    df = _read_csv_fallback(fp, usecols=["COUNTRY.UN_CODE", "PERIOD", "VALUE", "MEASURE"])
    df = df.loc[df["MEASURE"].astype(str).str.strip() == "V_USD_1000"].copy()
    df["COUNTRY.UN_CODE"] = pd.to_numeric(df["COUNTRY.UN_CODE"], errors="coerce")
    df["PERIOD"] = pd.to_numeric(df["PERIOD"], errors="coerce")
    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")

    df = df.dropna(subset=["COUNTRY.UN_CODE", "PERIOD"])
    df["M49_Country_Code"] = df["COUNTRY.UN_CODE"].apply(_fmt_m49)
    df["Year"] = df["PERIOD"].astype(int)

    out = (
        df.groupby(["M49_Country_Code", "Year"], as_index=False)["VALUE"]
          .sum()
          .rename(columns={"VALUE": "Aquaculture_value_1000USD"})
    )
    return out



def _read_fsj_aquaculture_quantity_wide(data_dir: str, year_min: int) -> pd.DataFrame:
    """
    Read FishStatJ *wide* aquaculture quantity export (year columns like "[2020]") and aggregate to country-year.

    Expected columns include:
      - Country (Name)
      - year columns "[YYYY]"
      - (optional) Unit / Unit (Name)

    Returns:
      Country, Year, Aquaculture_prod_t
    """
    fp = Path(data_dir) / FSJ_AQUA_QTY_WIDE_FILE
    if not fp.exists():
        raise FileNotFoundError(f"Missing required FishStatJ aquaculture quantity wide file: {fp}")

    header = _read_csv_fallback(fp, nrows=0)
    id_cols = ["Country (Name)"]
    for c in ["Unit", "Unit (Name)"]:
        if c in header.columns:
            id_cols.append(c)

    years = _list_year_columns_from_header(fp)
    if not years:
        return pd.DataFrame(columns=["Country", "Year", "Aquaculture_prod_t"])
    year_max = max(int(c.strip("[]")) for c in years)

    df_long = _read_fao_wide_years(fp, id_cols=id_cols, year_min=year_min, year_max=year_max)

    # If unit column exists, keep TLW-like rows
    if "Unit" in df_long.columns:
        df_long = df_long.loc[df_long["Unit"].astype(str).str.strip().isin(["TLW", "Tonnes - live weight"])].copy()

    out = (
        df_long.groupby(["Country (Name)", "Year"], as_index=False)["Value"]
              .sum()
              .rename(columns={"Country (Name)": "Country", "Value": "Aquaculture_prod_t"})
    )
    return out


def _read_fsj_aquaculture_value_wide(data_dir: str, year_min: int) -> pd.DataFrame:
    """
    Read FishStatJ *wide* aquaculture value export (year columns like "[2020]") and aggregate to country-year.

    Returns:
      Country, Year, Aquaculture_value_1000USD
    """
    fp = Path(data_dir) / FSJ_AQUA_VAL_WIDE_FILE
    if not fp.exists():
        raise FileNotFoundError(f"Missing required FishStatJ aquaculture value wide file: {fp}")

    header = _read_csv_fallback(fp, nrows=0)
    id_cols = ["Country (Name)"]
    for c in ["Unit", "Unit (Name)"]:
        if c in header.columns:
            id_cols.append(c)

    years = _list_year_columns_from_header(fp)
    if not years:
        return pd.DataFrame(columns=["Country", "Year", "Aquaculture_value_1000USD"])
    year_max = max(int(c.strip("[]")) for c in years)

    df_long = _read_fao_wide_years(fp, id_cols=id_cols, year_min=year_min, year_max=year_max)

    if "Unit" in df_long.columns:
        df_long = df_long.loc[df_long["Unit"].astype(str).str.contains("USD", case=False, na=False)].copy()

    out = (
        df_long.groupby(["Country (Name)", "Year"], as_index=False)["Value"]
              .sum()
              .rename(columns={"Country (Name)": "Country", "Value": "Aquaculture_value_1000USD"})
    )
    return out



def _read_fsj_capture_quantity_wide(data_dir: str, year_min: int) -> pd.DataFrame:
    """
    Read FishStatJ *wide* capture production quantity export and aggregate to country-year.

    Expected columns include:
      - Country (Name)
      - year columns "[YYYY]"
      - (optional) Unit / Unit (Name)

    Returns:
      Country, Year, Capture_prod_t
    """
    fp = Path(data_dir) / FSJ_CAPTURE_QTY_WIDE_FILE
    if not fp.exists():
        raise FileNotFoundError(f"Missing required FishStatJ capture quantity wide file: {fp}")

    header = _read_csv_fallback(fp, nrows=0)
    id_cols = ["Country (Name)"]
    for c in ["Unit", "Unit (Name)"]:
        if c in header.columns:
            id_cols.append(c)

    years = _list_year_columns_from_header(fp)
    if not years:
        return pd.DataFrame(columns=["Country", "Year", "Capture_prod_t"])
    year_max = max(int(c.strip("[]")) for c in years)

    df_long = _read_fao_wide_years(fp, id_cols=id_cols, year_min=year_min, year_max=year_max)

    if "Unit" in df_long.columns:
        df_long = df_long.loc[df_long["Unit"].astype(str).str.strip().isin(["TLW", "Tonnes - live weight"])].copy()

    out = (
        df_long.groupby(["Country (Name)", "Year"], as_index=False)["Value"]
              .sum()
              .rename(columns={"Country (Name)": "Country", "Value": "Capture_prod_t"})
    )
    return out


def _read_fsj_capture_value_wide(data_dir: str, year_min: int) -> pd.DataFrame:
    """
    Read FishStatJ *wide* capture production value export and aggregate to country-year.

    Returns:
      Country, Year, Capture_value_1000USD
    """
    fp = Path(data_dir) / FSJ_CAPTURE_VAL_WIDE_FILE
    if not fp.exists():
        raise FileNotFoundError(f"Missing required FishStatJ capture value wide file: {fp}")

    header = _read_csv_fallback(fp, nrows=0)
    id_cols = ["Country (Name)"]
    for c in ["Unit", "Unit (Name)"]:
        if c in header.columns:
            id_cols.append(c)

    years = _list_year_columns_from_header(fp)
    if not years:
        return pd.DataFrame(columns=["Country", "Year", "Capture_value_1000USD"])
    year_max = max(int(c.strip("[]")) for c in years)

    df_long = _read_fao_wide_years(fp, id_cols=id_cols, year_min=year_min, year_max=year_max)

    if "Unit" in df_long.columns:
        df_long = df_long.loc[df_long["Unit"].astype(str).str.contains("USD", case=False, na=False)].copy()

    out = (
        df_long.groupby(["Country (Name)", "Year"], as_index=False)["Value"]
              .sum()
              .rename(columns={"Country (Name)": "Country", "Value": "Capture_value_1000USD"})
    )
    return out


def _list_year_columns_from_header(csv_path: Path) -> list:
    """
    Return list of year columns that look like "[YYYY]" in the CSV header.
    """
    df0 = _read_csv_fallback(csv_path, nrows=0)
    cols = [str(c).strip() for c in df0.columns]
    years = [c for c in cols if re.fullmatch(r"\[\d{4}\]", c)]
    return years


def _read_fao_wide_years(csv_path: Path, id_cols: list, year_min: int, year_max: int) -> pd.DataFrame:
    """
    Read FAO/FishStatJ wide export with year columns "[YYYY]" and melt to long.

    Only reads year columns within [year_min, year_max] to save memory.
    """
    year_cols_all = _list_year_columns_from_header(csv_path)
    year_cols = [c for c in year_cols_all if year_min <= int(c.strip("[]")) <= year_max]

    usecols = id_cols + year_cols
    df = _read_csv_fallback(csv_path, usecols=usecols)

    df_long = df.melt(id_vars=id_cols, value_vars=year_cols, var_name="Year", value_name="Value")
    df_long["Year"] = df_long["Year"].astype(str).str.strip().str.strip("[]")
    df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce").astype("Int64")
    df_long["Value"] = pd.to_numeric(df_long["Value"], errors="coerce")
    df_long = df_long.dropna(subset=["Year"])
    return df_long


def _read_fbs_aquatic_total_food_supply(data_dir: str, year_min: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read FAO FBS aquatic products (wide) and compute Total food supply by country-year.

    Caveat: this export uses country names (no codes). We attempt exact name matching to dict_v3 'Country'.

    Returns:
      (demand_df, unmapped_countries_df)
      demand_df columns: Country, Year, Demand_total_t_FBS
    """
    fp = Path(data_dir) / FBS_AQUATIC_FILE
    if not fp.exists():
        raise FileNotFoundError(f"Missing required FAO FBS aquatic products file: {fp}")

    id_cols = ["Country (Name)", "FAOSTAT group (Name)", "Element (Name)", "Unit"]
    # Determine max year from header (your file ends at 2019)
    years = _list_year_columns_from_header(fp)
    year_max = max(int(c.strip("[]")) for c in years) if years else 2019

    df_long = _read_fao_wide_years(fp, id_cols=id_cols, year_min=year_min, year_max=year_max)

    # Drop obvious non-country lines (these exports sometimes embed totals/footnotes into Country column)
    c = df_long["Country (Name)"].astype(str)
    bad = (
        c.str.startswith("Totals", na=False)
        | c.str.contains("Fishery and Aquaculture Statistics", na=False)
        | c.str.contains("www\\.fao\\.org", na=False)
        | c.str.contains("^FAO\\.", regex=True, na=False)
    )
    df_long = df_long.loc[~bad].copy()

    # Keep only Total food supply and aggregate across groups (fish/shellfish/etc.)
    df_long = df_long.loc[df_long["Element (Name)"].astype(str).str.strip() == FBS_ELEMENT_DEMAND].copy()
    out = (
        df_long.groupby(["Country (Name)", "Year"], as_index=False)["Value"]
              .sum()
              .rename(columns={"Country (Name)": "Country", "Value": "Demand_total_t_FBS"})
    )

    # collect country list for mapping QC
    unmapped = pd.DataFrame({"Country (Name)": sorted(df_long["Country (Name)"].dropna().unique())})
    return out, unmapped


def _read_faostat_aquaculture_area_from_landcsv(land_dir: str,
                                               year_min: int,
                                               year_max: int) -> pd.DataFrame:
    """
    Read FAOSTAT Land Use CSV and extract aquaculture area (ha).

    Expected file in land_dir:
      - Inputs_LandUse_E_All_Data_NOFLAG_with_Pasture.csv

    Uses items:
      - Land used for aquaculture
      - Inland waters used for aquac. or holding facilities
      - Coastal waters used for aquac. or holding facilities (saline flag)
      - EEZ used for aquac. or holding facilities (saline flag)

    Units in source: 1000 ha (converted to ha).
    Returns:
      M49_Country_Code (3-digit string), Year (int),
      Aquaculture_area_ha (float), Aquaculture_saline_area_ha (float),
      Aquaculture_saline_flag (bool)
    """
    fpath = Path(land_dir) / FAOSTAT_LANDUSE_CSV
    if not fpath.exists():
        raise FileNotFoundError(f"Missing required FAOSTAT land-use CSV: {fpath}")

    header = _read_csv_fallback(fpath, nrows=0)
    cols = list(header.columns)
    cols = [str(c).strip() for c in cols]

    def _pick(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in cols:
                return c
        return None

    m49_col = _pick(["M49_Country_Code", "Area Code (M49)", "Area Code (M49)_x", "Area Code (M49)_y", "M49 Code", "Area Code M49"])
    item_col = _pick(["Item", "Item (Name)"])
    elem_col = _pick(["Element", "Element (Name)"])
    unit_col = _pick(["Unit"])
    year_cols = [c for c in cols if re.fullmatch(r"Y\d{4}", str(c).strip())]

    missing = [name for name, col in [
        ("M49_Country_Code", m49_col),
        ("Item", item_col),
        ("Element", elem_col),
    ] if col is None]
    if not year_cols:
        missing.append("Y#### year columns")
    if missing:
        raise RuntimeError(
            f"FAOSTAT land-use CSV missing columns: {missing}. Columns: {cols}"
        )

    usecols = [m49_col, item_col, elem_col] + year_cols
    if unit_col:
        usecols.append(unit_col)
    df = _read_csv_fallback(fpath, usecols=usecols)
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    df = df.rename(columns={
        m49_col: "M49_Country_Code",
        item_col: "Item",
        elem_col: "Element",
    })
    if unit_col:
        df = df.rename(columns={unit_col: "Unit"})

    df["Item"] = df["Item"].astype(str).str.strip()
    df["Element"] = df["Element"].astype(str).str.strip()
    wanted_items = set(FAOSTAT_AQUA_ITEMS + FAOSTAT_SALINE_ITEMS)
    df = df[df["Item"].isin(wanted_items)].copy()
    df = df[df["Element"].str.lower() == "area"].copy()
    if df.empty:
        raise ValueError(f"No aquaculture land-use rows found in {fpath}")

    df_long = df.melt(id_vars=["M49_Country_Code", "Item", "Element"],
                      value_vars=year_cols,
                      var_name="Year",
                      value_name="Value")
    df_long["Year"] = df_long["Year"].astype(str).str.replace("Y", "", regex=False)
    df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce")
    df_long["Value"] = pd.to_numeric(df_long["Value"], errors="coerce")
    df_long = df_long.dropna(subset=["M49_Country_Code", "Year"])
    df_long = df_long[(df_long["Year"] >= year_min) & (df_long["Year"] <= year_max)]

    df_long["M49_Country_Code"] = df_long["M49_Country_Code"].apply(_fmt_m49)
    df_long = df_long.dropna(subset=["M49_Country_Code", "Year"])

    aqua = (
        df_long[df_long["Item"].isin(FAOSTAT_AQUA_ITEMS)]
        .groupby(["M49_Country_Code", "Year"], as_index=False)["Value"]
        .sum(min_count=1)
        .rename(columns={"Value": "Aquaculture_area_ha"})
    )
    saline = (
        df_long[df_long["Item"].isin(FAOSTAT_SALINE_ITEMS)]
        .groupby(["M49_Country_Code", "Year"], as_index=False)["Value"]
        .sum(min_count=1)
        .rename(columns={"Value": "Aquaculture_saline_area_ha"})
    )
    out = aqua.merge(saline, on=["M49_Country_Code", "Year"], how="outer")
    if "Aquaculture_area_ha" not in out.columns:
        out["Aquaculture_area_ha"] = np.nan
    if "Aquaculture_saline_area_ha" not in out.columns:
        out["Aquaculture_saline_area_ha"] = np.nan
    out["Aquaculture_area_ha"] = out["Aquaculture_area_ha"] * 1000.0
    out["Aquaculture_saline_area_ha"] = out["Aquaculture_saline_area_ha"] * 1000.0
    out["Aquaculture_saline_flag"] = (
        out.groupby("M49_Country_Code")["Aquaculture_saline_area_ha"]
           .transform(lambda s: (s.fillna(0.0) > 0).any())
    )
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce").astype("Int64")
    return out


def _find_fbs_production_file(prod_trade_dir: str) -> Path:
    base = Path(prod_trade_dir) / FBS_PROD_FILE_STEM
    if base.exists():
        return base
    for ext in (".csv", ".csv.gz", ".txt", ".xlsx", ".xls", ".xlsm"):
        candidate = Path(f"{base}{ext}")
        if candidate.exists():
            return candidate
    matches = sorted(Path(prod_trade_dir).glob(f"{FBS_PROD_FILE_STEM}*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Missing required FBS production file under {prod_trade_dir}: {FBS_PROD_FILE_STEM}*"
    )


def _read_fbs_fish_production_total(prod_trade_dir: str,
                                    year_min: int,
                                    year_max: int) -> pd.DataFrame:
    """
    Read Fish, Seafood total production from FBS file.
    Assumes units are 1000 t, returns Production_total_t in t.
    """
    fpath = _find_fbs_production_file(prod_trade_dir)
    suffix = fpath.suffix.lower()
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        df = pd.read_excel(fpath)
    else:
        df = _read_csv_fallback(fpath)
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    cols = list(df.columns)

    def _pick(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in cols:
                return c
        return None

    m49_col = _pick(["M49_Country_Code", "Area Code (M49)", "Area Code (M49)_x", "Area Code (M49)_y", "M49 Code", "Area Code"])
    item_col = _pick(["Item", "Item (Name)"])
    elem_col = _pick(["Element", "Element (Name)"])
    year_col = _pick(["Year", "year"])
    val_col = _pick(["Value", "value"])
    year_wide_cols = [c for c in cols if re.fullmatch(r"Y\d{4}", str(c).strip())]

    missing = [name for name, col in [
        ("Area Code (M49)", m49_col),
        ("Item", item_col),
        ("Element", elem_col),
    ] if col is None]
    if year_col is None and val_col is None and not year_wide_cols:
        missing.extend(["Year/Value or YYYYY columns"])
    if missing:
        raise RuntimeError(
            f"FBS production file missing columns {missing}. Columns: {cols}"
        )

    df = df.rename(columns={
        m49_col: "M49_Country_Code",
        item_col: "Item",
        elem_col: "Element",
    })
    if year_col is not None:
        df = df.rename(columns={year_col: "Year"})
    if val_col is not None:
        df = df.rename(columns={val_col: "Value"})

    df["Item"] = df["Item"].astype(str).str.strip()
    df["Element"] = df["Element"].astype(str).str.strip()
    mask = df["Item"].str.lower().eq("fish, seafood") & df["Element"].str.lower().eq("production")
    df = df.loc[mask].copy()
    if df.empty:
        raise ValueError(f"No Fish, Seafood Production rows in {fpath}")

    if "Year" in df.columns and "Value" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        df = df.dropna(subset=["M49_Country_Code", "Year", "Value"]).copy()
    else:
        df_long = df.melt(id_vars=["M49_Country_Code", "Item", "Element"],
                          value_vars=year_wide_cols,
                          var_name="Year",
                          value_name="Value")
        df_long["Year"] = df_long["Year"].astype(str).str.replace("Y", "", regex=False)
        df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce")
        df_long["Value"] = pd.to_numeric(df_long["Value"], errors="coerce")
        df = df_long.dropna(subset=["M49_Country_Code", "Year", "Value"]).copy()
    df["M49_Country_Code"] = df["M49_Country_Code"].apply(_fmt_m49)
    df = df.dropna(subset=["M49_Country_Code"])
    df = df[(df["Year"] >= year_min) & (df["Year"] <= year_max)]

    df["Production_total_t"] = df["Value"] * 1000.0
    out = (
        df.groupby(["M49_Country_Code", "Year"], as_index=False)["Production_total_t"]
          .sum()
    )
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce").astype("Int64")
    if out.empty:
        raise ValueError(
            f"No Fish, Seafood Production rows in {fpath} after year filter {year_min}-{year_max}"
        )
    return out


def _load_ipcc_default_efs(data_dir: str) -> Dict[str, Dict[str, float]]:
    fpath = Path(data_dir) / IPCC_EF_FILE
    if not fpath.exists():
        raise FileNotFoundError(f"Missing required IPCC EF defaults file: {fpath}")
    df = pd.read_excel(fpath, sheet_name="defaults")
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    for col in ["Process", "Gas", "Default_EF", "Lower_95CI", "Upper_95CI"]:
        if col not in df.columns:
            raise RuntimeError(f"IPCC EF defaults missing column '{col}'. Columns: {list(df.columns)}")
    df["Process"] = df["Process"].astype(str).str.strip()
    df["Gas"] = df["Gas"].astype(str).str.strip().str.upper()

    def pick_row(process: str, gas: str, applies_contains: Optional[str] = None) -> Dict[str, float]:
        sub = df[(df["Process"].str.lower() == process.lower()) & (df["Gas"] == gas.upper())].copy()
        if applies_contains:
            applies_col = "Applies_to" if "Applies_to" in sub.columns else ("Applies to" if "Applies to" in sub.columns else None)
            if applies_col:
                sub = sub[sub[applies_col].astype(str).str.contains(applies_contains, case=False, na=False)]
            else:
                return {}
        if sub.empty:
            return {}
        row = sub.iloc[0]
        return {
            "median": float(row.get("Default_EF", 0.0) or 0.0),
            "p05": float(row.get("Lower_95CI", 0.0) or 0.0),
            "p95": float(row.get("Upper_95CI", 0.0) or 0.0),
        }

    cap_ch4 = pick_row("Capture (excluding fuel/energy)", "CH4")
    cap_n2o = pick_row("Capture (excluding fuel/energy)", "N2O")

    aqua_n2o = pick_row("Aquaculture", "N2O")
    if not aqua_n2o:
        aqua_n2o_n = pick_row("Aquaculture", "N2O-N")
        if aqua_n2o_n:
            aqua_n2o = {
                "median": aqua_n2o_n["median"] * (44.0 / 28.0),
                "p05": aqua_n2o_n["p05"] * (44.0 / 28.0),
                "p95": aqua_n2o_n["p95"] * (44.0 / 28.0),
            }

    aqua_ch4_ha_fresh = pick_row("Aquaculture (ponds)", "CH4", applies_contains="Freshwater")
    if not aqua_ch4_ha_fresh:
        aqua_ch4_ha_fresh = pick_row("Aquaculture (ponds)", "CH4", applies_contains="Brackish")
    if not aqua_ch4_ha_fresh:
        aqua_ch4_ha_fresh = pick_row("Aquaculture (ponds)", "CH4")

    aqua_ch4_ha_saline = pick_row("Aquaculture (ponds)", "CH4", applies_contains="Saline")
    if not aqua_ch4_ha_saline:
        aqua_ch4_ha_saline = aqua_ch4_ha_fresh

    if not cap_ch4 or not cap_n2o or not aqua_n2o or not aqua_ch4_ha_fresh or not aqua_ch4_ha_saline:
        raise RuntimeError("IPCC EF defaults missing required rows for capture/aquaculture.")

    return {
        "cap_ch4": cap_ch4,
        "cap_n2o": cap_n2o,
        "aqua_n2o": aqua_n2o,
        "aqua_ch4_ha_fresh": aqua_ch4_ha_fresh,
        "aqua_ch4_ha_saline": aqua_ch4_ha_saline,
    }


def _populate_ef_columns(panel: pd.DataFrame, efs: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    if "Aquaculture_saline_flag" in panel.columns:
        is_saline = panel["Aquaculture_saline_flag"].fillna(False).astype(bool)
    else:
        is_saline = pd.Series(False, index=panel.index)

    fresh = efs["aqua_ch4_ha_fresh"]
    saline = efs["aqua_ch4_ha_saline"]
    panel["EF_aqua_CH4_kg_per_ha_yr_median"] = np.where(is_saline, saline["median"], fresh["median"])
    panel["EF_aqua_CH4_kg_per_ha_yr_p05"] = np.where(is_saline, saline["p05"], fresh["p05"])
    panel["EF_aqua_CH4_kg_per_ha_yr_p95"] = np.where(is_saline, saline["p95"], fresh["p95"])

    panel["EF_aqua_N2O_kg_per_kg_median"] = float(efs["aqua_n2o"]["median"])
    panel["EF_aqua_N2O_kg_per_kg_p05"] = float(efs["aqua_n2o"]["p05"])
    panel["EF_aqua_N2O_kg_per_kg_p95"] = float(efs["aqua_n2o"]["p95"])

    panel["EF_wild_CH4_kg_per_kg_median"] = float(efs["cap_ch4"]["median"])
    panel["EF_wild_CH4_kg_per_kg_p05"] = float(efs["cap_ch4"]["p05"])
    panel["EF_wild_CH4_kg_per_kg_p95"] = float(efs["cap_ch4"]["p95"])

    panel["EF_wild_N2O_kg_per_kg_median"] = float(efs["cap_n2o"]["median"])
    panel["EF_wild_N2O_kg_per_kg_p05"] = float(efs["cap_n2o"]["p05"])
    panel["EF_wild_N2O_kg_per_kg_p95"] = float(efs["cap_n2o"]["p95"])

    return panel


def _scale_production_to_fbs(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Scale capture/aquaculture production to match FBS total production.
    Rules:
      - If both capture and aquaculture > 0, scale both by the same factor.
      - If only one is > 0, set it to the FBS total.
      - If both are 0 and FBS total > 0, set aquaculture to FBS total.
    """
    cap = pd.to_numeric(panel["Capture_prod_t"], errors="coerce").fillna(0.0)
    aqua = pd.to_numeric(panel["Aquaculture_prod_t"], errors="coerce").fillna(0.0)
    fbs = pd.to_numeric(panel["Production_total_t"], errors="coerce")

    has_fbs = fbs.notna()
    total = cap + aqua

    both_pos = has_fbs & (cap > 0) & (aqua > 0) & (total > 0)
    scale = fbs / total
    panel.loc[both_pos, "Capture_prod_t"] = cap[both_pos] * scale[both_pos]
    panel.loc[both_pos, "Aquaculture_prod_t"] = aqua[both_pos] * scale[both_pos]

    only_cap = has_fbs & (cap > 0) & (aqua <= 0)
    panel.loc[only_cap, "Capture_prod_t"] = fbs[only_cap]

    only_aqua = has_fbs & (aqua > 0) & (cap <= 0)
    panel.loc[only_aqua, "Aquaculture_prod_t"] = fbs[only_aqua]

    none_but_fbs = has_fbs & (cap <= 0) & (aqua <= 0) & (fbs > 0)
    panel.loc[none_but_fbs, "Aquaculture_prod_t"] = fbs[none_but_fbs]
    return panel


def _fill_aquaculture_yield_and_area(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing aquaculture yield (2010-2020) and back-calculate missing area.
    Fill order for yield:
      1) nearest valid year for the same country
      2) same-year regional mean (Region_market_full)
      3) same-year global mean
    """
    required_cols = {"Year", "ISO3", "Region_market_full", "Aquaculture_yield_t_per_ha",
                     "Aquaculture_area_ha", "Aquaculture_prod_t"}
    missing = [c for c in required_cols if c not in panel.columns]
    if missing:
        raise RuntimeError(f"Missing columns for yield fill: {missing}")

    out = panel.copy()
    yield_col = "Aquaculture_yield_t_per_ha"
    area_col = "Aquaculture_area_ha"
    prod_col = "Aquaculture_prod_t"

    out[yield_col] = out[yield_col].replace([np.inf, -np.inf], np.nan)

    def _fill_nearest(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("Year").copy()
        years = pd.to_numeric(g["Year"], errors="coerce").to_numpy()
        vals = pd.to_numeric(g[yield_col], errors="coerce").to_numpy()
        if np.all(np.isnan(vals)):
            return g
        for i, v in enumerate(vals):
            if np.isnan(v):
                valid_mask = ~np.isnan(vals)
                diffs = np.abs(years[valid_mask] - years[i])
                j = np.where(valid_mask)[0][int(np.argmin(diffs))]
                vals[i] = vals[j]
        g[yield_col] = vals
        return g

    out = out.groupby("ISO3", group_keys=False).apply(_fill_nearest)

    missing_yield = out[yield_col].isna()
    if missing_yield.any():
        region_mean = (
            out.loc[out[yield_col].notna()]
               .groupby(["Region_market_full", "Year"])[yield_col]
               .mean()
        )
        out.loc[missing_yield, yield_col] = (
            out.loc[missing_yield].set_index(["Region_market_full", "Year"]).index.map(region_mean)
        )

    missing_yield = out[yield_col].isna()
    if missing_yield.any():
        global_mean = (
            out.loc[out[yield_col].notna()]
               .groupby("Year")[yield_col]
               .mean()
        )
        out.loc[missing_yield, yield_col] = out.loc[missing_yield, "Year"].map(global_mean)

    area = pd.to_numeric(out[area_col], errors="coerce")
    missing_area = area.isna() | (area <= 0)
    valid_yield = out[yield_col].notna() & (out[yield_col] > 0)
    fill_area = missing_area & valid_yield
    if fill_area.any():
        out.loc[fill_area, area_col] = out.loc[fill_area, prod_col] / out.loc[fill_area, yield_col]

    return out


def _compute_aquaculture_baseline_emissions(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute aquaculture baseline CH4/N2O emissions (kg/yr) using IPCC formulas.
    """
    required = {"Aquaculture_area_ha", "Aquaculture_prod_t",
                "EF_aqua_CH4_kg_per_ha_yr_median", "EF_aqua_N2O_kg_per_kg_median",
                "Live_to_product_yield_kg_per_kg"}
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise RuntimeError(f"Missing columns for baseline emissions: {missing}")

    out = panel.copy()
    area = pd.to_numeric(out["Aquaculture_area_ha"], errors="coerce")
    ef_ch4 = pd.to_numeric(out["EF_aqua_CH4_kg_per_ha_yr_median"], errors="coerce")
    out["Aquaculture_CH4_emissions_kg_yr"] = area * ef_ch4

    prod = pd.to_numeric(out["Aquaculture_prod_t"], errors="coerce")
    ef_n2o = pd.to_numeric(out["EF_aqua_N2O_kg_per_kg_median"], errors="coerce")
    yield_ratio = pd.to_numeric(out["Live_to_product_yield_kg_per_kg"], errors="coerce")
    out["Aquaculture_N2O_emissions_kg_yr"] = np.where(
        yield_ratio > 0,
        (prod * 1000.0) / yield_ratio * ef_n2o,
        np.nan,
    )
    return out


def build_panel(dict_xlsx: str,
                data_dir: str,
                land_dir: str,
                prod_trade_dir: str,
                region_sheet: str = "region") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the fish/seafood country-year panel using:
      - dict_v3.xlsx (region sheet) for ISO3 and M49 mapping + valid country filter
      - local OWID CSV exports in data_dir (capture + aquaculture production)
      - local FBS production file in prod_trade_dir (total production scaling)
      - local FAOSTAT land-use CSV in land_dir for aquaculture area (yield proxy)

    Notes on source priority:
      - Capture_prod_t: OWID capture-fishery-production.csv only.
      - Aquaculture_prod_t: OWID aquaculture-farmed-fish-production.csv only.
    """
    mapping = _read_region_mapping_from_dict(dict_xlsx, sheet_name=region_sheet).copy()
    mapping["ISO3_method"] = "dict_v3"

    keep_iso3 = set(mapping["ISO3"].tolist())
    # OWID capture + aquaculture
    cap = _read_owid_local_csv(OWID_SLUG_CAPTURE, data_dir)
    aqua = _read_owid_local_csv(OWID_SLUG_AQUA, data_dir)

    cap = cap.loc[cap["Code"].isin(keep_iso3)].copy()
    aqua = aqua.loc[aqua["Code"].isin(keep_iso3)].copy()

    years = list(range(YEAR_MIN, YEAR_MAX + 1))

    skel = pd.MultiIndex.from_product([sorted(keep_iso3), years], names=["ISO3", "Year"]).to_frame(index=False)
    skel = skel.merge(mapping, on="ISO3", how="left")

    cap = cap.rename(columns={"Code": "ISO3", "Year": "Year", "Value": "Capture_prod_t"})
    aqua = aqua.rename(columns={"Code": "ISO3", "Year": "Year", "Value": "Aquaculture_prod_t"})

    cap["Year"] = pd.to_numeric(cap["Year"], errors="coerce").astype("Int64")
    aqua["Year"] = pd.to_numeric(aqua["Year"], errors="coerce").astype("Int64")
    cap = cap[(cap["Year"] >= YEAR_MIN) & (cap["Year"] <= YEAR_MAX)].copy()
    aqua = aqua[(aqua["Year"] >= YEAR_MIN) & (aqua["Year"] <= YEAR_MAX)].copy()
    cap["Capture_prod_t"] = pd.to_numeric(cap["Capture_prod_t"], errors="coerce")
    aqua["Aquaculture_prod_t"] = pd.to_numeric(aqua["Aquaculture_prod_t"], errors="coerce")

    cap_source = "OWID"
    aq_source = "OWID"

    # Merge everything onto skeleton
    base = skel.merge(cap[["ISO3", "Year", "Capture_prod_t"]], on=["ISO3", "Year"], how="left")
    base = base.merge(aqua[["ISO3", "Year", "Aquaculture_prod_t"]], on=["ISO3", "Year"], how="left")
    fbs_prod = _read_fbs_fish_production_total(prod_trade_dir, YEAR_MIN, YEAR_MAX)
    base = base.merge(fbs_prod, on=["M49_Country_Code", "Year"], how="left")
    base = _scale_production_to_fbs(base)
    if "Production_total_t" in base.columns:
        base = base.drop(columns=["Production_total_t"])

    # totals
    base["Total_prod_t"] = pd.to_numeric(base["Capture_prod_t"], errors="coerce").fillna(0) + pd.to_numeric(base["Aquaculture_prod_t"], errors="coerce").fillna(0)
    base["Aquaculture_share"] = np.where(base["Total_prod_t"] > 0, base["Aquaculture_prod_t"] / base["Total_prod_t"], np.nan)
    base["Capture_share"] = np.where(base["Total_prod_t"] > 0, base["Capture_prod_t"] / base["Total_prod_t"], np.nan)

    if "Country_x" in base.columns:
        base = base.rename(columns={"Country_x": "Country"})
    if "Country_y" in base.columns:
        base = base.drop(columns=["Country_y"])

    # aquaculture area from local FAOSTAT land-use CSV (merge via M49)
    aqua_area = _read_faostat_aquaculture_area_from_landcsv(land_dir, YEAR_MIN, YEAR_MAX)
    if len(aqua_area) > 0:
        base = base.merge(aqua_area, on=["M49_Country_Code", "Year"], how="left")
        prod = pd.to_numeric(base["Aquaculture_prod_t"], errors="coerce")
        area = pd.to_numeric(base["Aquaculture_area_ha"], errors="coerce")
        base["Aquaculture_yield_t_per_ha"] = np.where(area > 0, prod / area, np.nan)
        if "Aquaculture_saline_flag" not in base.columns:
            base["Aquaculture_saline_flag"] = False
        if "Aquaculture_saline_area_ha" not in base.columns:
            base["Aquaculture_saline_area_ha"] = np.nan
    else:
        base["Aquaculture_area_ha"] = np.nan
        base["Aquaculture_saline_area_ha"] = np.nan
        base["Aquaculture_saline_flag"] = False
        base["Aquaculture_yield_t_per_ha"] = np.nan

    # fill yield gaps and back-calculate area where missing (2010-2020)
    base = _fill_aquaculture_yield_and_area(base)

    # default live-weight to product yield ratio
    base["Live_to_product_yield_kg_per_kg"] = LIVE_TO_PRODUCT_YIELD_DEFAULT

    # emissions (IPCC defaults)
    efs = _load_ipcc_default_efs(data_dir)
    base = _populate_ef_columns(base, efs)
    base = _compute_aquaculture_baseline_emissions(base)

    # notes / metadata
    base["Aquaculture_source"] = aq_source
    base["Capture_source"] = cap_source
    base.attrs["data_dir"] = data_dir
    base.attrs["land_dir"] = land_dir
    base.attrs["prod_trade_dir"] = prod_trade_dir

    return base, mapping


def _collect_input_file_status(dict_xlsx: str,
                               region_sheet: str,
                               data_dir: str,
                               land_dir: str,
                               prod_trade_dir: str,
                               panel: pd.DataFrame) -> pd.DataFrame:
    """
    Build a tidy table describing which files were expected / found / used (with resolved paths).
    """
    rows = []

    def add(role: str, folder: str, fname: str, required: bool, used: object = None, note: str = ""):
        fpath = Path(folder) / fname
        if "*" in fname:
            exists = any(Path(folder).glob(fname))
            path_str = str(fpath)
        else:
            exists = fpath.exists()
            path_str = str(fpath.resolve())
        rows.append({
            "role": role,
            "required": bool(required),
            "used": used,  # True/False/None
            "exists": exists,
            "path": path_str,
            "note": note
        })

    # dict
    dx = Path(dict_xlsx)
    add("dict_v3 (region mapping)", str(dx.parent), dx.name, required=True, used=True, note=f"sheet={region_sheet}")

    # OWID (required)
    add("OWID capture production", data_dir, f"{OWID_SLUG_CAPTURE}.csv", required=True, used=True)
    add("OWID aquaculture production", data_dir, f"{OWID_SLUG_AQUA}.csv", required=True, used=True)

    # IPCC EF defaults (required)
    add("IPCC default non-energy fish EFs", data_dir, IPCC_EF_FILE, required=True, used=True)

    # FBS production (required)
    add("FBS Fish, Seafood production total", prod_trade_dir, f"{FBS_PROD_FILE_STEM}*", required=True, used=True)

    # Land-use (aquaculture area; yield proxy)
    add("FAOSTAT land-use normalized CSV (aquaculture area)", land_dir, FAOSTAT_LANDUSE_CSV, required=True,
        used=("Aquaculture_area_ha" in panel.columns and panel["Aquaculture_area_ha"].notna().any()))

    return pd.DataFrame(rows)


def _write_run_log_files(out_dir: str, panel: pd.DataFrame, inputs_df: pd.DataFrame, out_xlsx: str):
    """
    Write logs (TXT + JSON) to OUT_DIR to record exactly which inputs were found/used.
    """
    stamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")

    aquasrc = panel["Aquaculture_source"].iloc[0] if ("Aquaculture_source" in panel.columns and len(panel)) else ""

    # Text log
    lines = []
    lines.append(f"Fish/Seafood panel build log @ {stamp}")
    lines.append("")
    lines.append("Outputs:")
    lines.append(f"  - Excel: {Path(out_xlsx).resolve()}")
    lines.append("")
    lines.append("Key choices:")
    lines.append(f"  - Capture_source    : {panel['Capture_source'].iloc[0] if ('Capture_source' in panel.columns and len(panel)) else '' }")
    lines.append(f"  - Aquaculture_source: {aquasrc}")
    lines.append("")
    lines.append("Input files (resolved paths):")
    for _, r in inputs_df.iterrows():
        used = r["used"]
        used_s = "YES" if used is True else ("NO" if used is False else "—")
        lines.append(f"- {r['role']}: used={used_s} exists={bool(r['exists'])} required={bool(r['required'])}")
        lines.append(f"    {r['path']}")
        if str(r.get("note","")).strip():
            lines.append(f"    note: {r['note']}")
    lines.append("")

    Path(out_dir, RUN_LOG_TXT).write_text("\n".join(lines), encoding="utf-8")

    # JSON log
    payload = {
        "timestamp": stamp,
        "outputs": {"excel": str(Path(out_xlsx).resolve())},
        "choices": {"Capture_source": (panel["Capture_source"].iloc[0] if ("Capture_source" in panel.columns and len(panel)) else ""), "Aquaculture_source": aquasrc},
        "inputs": inputs_df.to_dict(orient="records"),
    }
    Path(out_dir, RUN_LOG_JSON).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_outputs(panel: pd.DataFrame,
                  mapping: pd.DataFrame,
                  data_dir: str,
                  dict_xlsx: str,
                  region_sheet: str,
                  land_dir: str,
                  prod_trade_dir: str,
                  out_dir: str) -> str:
    out_xlsx = os.path.join(out_dir, "fish_seafood_country_panel_2000_present.xlsx")

    drop_cols = [
        "ISO3_method",
        "Country",
        "Region_market_full",
        "Aquaculture_saline_area_ha",
        "Aquaculture_saline_flag",
        "EF_wild_median_kgCO2e_per_kg",
        "EF_wild_p05",
        "EF_wild_p95",
        "EF_aqua_median_kgCO2e_per_kg",
        "EF_aqua_p05",
        "EF_aqua_p95",
        "GHG_capture_MtCO2e_median",
        "GHG_capture_MtCO2e_p05",
        "GHG_capture_MtCO2e_p95",
        "GHG_aquaculture_MtCO2e_median",
        "GHG_aquaculture_MtCO2e_p05",
        "GHG_aquaculture_MtCO2e_p95",
        "GHG_total_MtCO2e_median",
        "GHG_total_MtCO2e_p05",
        "GHG_total_MtCO2e_p95",
    ]
    panel_out = panel.drop(columns=[c for c in drop_cols if c in panel.columns], errors="ignore")
    mapping_out = mapping.drop(columns=[c for c in ["ISO3_method"] if c in mapping.columns], errors="ignore")

    inputs_df = _collect_input_file_status(dict_xlsx, region_sheet, data_dir, land_dir, prod_trade_dir, panel)

    data_dict = pd.DataFrame([
        ("ISO3", "ISO-3166 alpha-3"),
        ("M49_Country_Code", "UN M49 3-digit code as string 'xxx'"),
        ("Region_label_new", "Region label from dict_v3.xlsx"),
        ("Year", "Calendar year"),
        ("Capture_prod_t", "Capture fishery production (metric tons)"),
        ("Aquaculture_prod_t", "Aquaculture production (metric tons)"),
        ("Total_prod_t", "Capture + aquaculture production (metric tons)"),
        ("Aquaculture_share", "Aquaculture production share of total"),
        ("Capture_share", "Capture production share of total"),
        ("Aquaculture_area_ha", "Aquaculture area (hectares) from FAOSTAT bulk: land used for aquaculture + inland waters used for aquac."),
        ("Aquaculture_yield_t_per_ha", "Aquaculture yield proxy (t/ha)"),
        ("Live_to_product_yield_kg_per_kg", "Live-weight to product yield ratio (kg/kg), default 0.52"),
        ("EF_aqua_CH4_kg_per_ha_yr_*", "Aquaculture CH4 EF (kg CH4/ha/yr) from IPCC defaults; saline vs freshwater by country via FAOSTAT salinity items"),
        ("EF_aqua_N2O_kg_per_kg_*", "Aquaculture N2O EF (kg N2O/kg fish) from IPCC defaults"),
        ("EF_wild_CH4_kg_per_kg_*", "Capture CH4 EF (kg CH4/kg fish) from IPCC defaults"),
        ("EF_wild_N2O_kg_per_kg_*", "Capture N2O EF (kg N2O/kg fish) from IPCC defaults"),
        ("Aquaculture_CH4_emissions_kg_yr", "Aquaculture CH4 baseline emissions (kg/yr), Area_ha * EF_CH4_kg_per_ha_yr"),
        ("Aquaculture_N2O_emissions_kg_yr", "Aquaculture N2O baseline emissions (kg/yr), (Prod_t*1000)/yield_kg_per_kg*EF_N2O_kg_per_kg"),
        ("Aquaculture_source", "Aquaculture source used (OWID)"),
        ("Notes", "Processing notes"),
    ], columns=["Field", "Meaning"])

    meta = pd.DataFrame([
        ("Built_at", pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")),
        ("YEAR_MIN", YEAR_MIN),
        ("YEAR_MAX", YEAR_MAX),
        ("DATA_DIR", str(Path(data_dir).resolve())),
        ("OUT_DIR", str(Path(out_dir).resolve())),
        ("LAND_DIR", str(Path(panel.attrs.get("land_dir","")).resolve()) if panel.attrs.get("land_dir") else ""),
        ("PROD_TRADE_DIR", str(Path(panel.attrs.get("prod_trade_dir","")).resolve()) if panel.attrs.get("prod_trade_dir") else ""),
        ("OWID_capture_file", f"{OWID_SLUG_CAPTURE}.csv"),
        ("OWID_aquaculture_file", f"{OWID_SLUG_AQUA}.csv"),
        ("IPCC_EF_defaults_file", IPCC_EF_FILE),
        ("FBS_production_file", f"{FBS_PROD_FILE_STEM}*"),
        ("FAOSTAT_landuse_csv", FAOSTAT_LANDUSE_CSV),
        ("FAOSTAT_aquaculture_items", "; ".join(FAOSTAT_AQUA_ITEMS)),
        ("FAOSTAT_salinity_items", "; ".join(FAOSTAT_SALINE_ITEMS)),
        ("FAOSTAT_aquaculture_unit", "1000 ha"),
        ("LIVE_TO_PRODUCT_YIELD_DEFAULT", LIVE_TO_PRODUCT_YIELD_DEFAULT),
    ], columns=["Key", "Value"])

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        panel_out.to_excel(writer, index=False, sheet_name="country_year_panel")
        mapping_out.to_excel(writer, index=False, sheet_name="country_iso3_mapping")
        data_dict.to_excel(writer, index=False, sheet_name="data_dictionary")
        meta.to_excel(writer, index=False, sheet_name="run_metadata")
        inputs_df.to_excel(writer, index=False, sheet_name="inputs_used")

    _write_run_log_files(out_dir, panel, inputs_df, out_xlsx)
    return out_xlsx


def main():
    ap = argparse.ArgumentParser()
    default_dict = str((Path(__file__).resolve().parent / "../../src/dict_v3.xlsx").resolve())
    default_data_dir = str((Path(__file__).resolve().parent / "../../input/Aquaculture/raw_unused_retired").resolve())
    default_out_dir = str((Path(__file__).resolve().parent / "../../input/Aquaculture").resolve())
    default_land_dir = str((Path(__file__).resolve().parent / "../../input/Land").resolve())
    default_prod_trade_dir = str((Path(__file__).resolve().parent / "../../input/Production_Trade").resolve())

    ap.add_argument("--dict", dest="dict_xlsx", default=default_dict,
                    help="Path to dict_v3.xlsx (default: ../../src/dict_v3.xlsx relative to this script)")
    ap.add_argument("--sheet", dest="region_sheet", default="region",
                    help="Sheet name in dict_v3.xlsx (default: region)")
    ap.add_argument("--data_dir", default=default_data_dir,
                    help="Folder containing all pre-downloaded input files "
                         "(default: ../../input/Aquaculture/raw_unused_retired relative to this script)")
    ap.add_argument("--out_dir", default=default_out_dir,
                    help="Folder to write outputs (default: ../../input/Aquaculture relative to this script)")
    ap.add_argument("--land_dir", default=default_land_dir,
                    help="Folder containing FAOSTAT land-use CSV (Inputs_LandUse_E_All_Data_(Normalized).csv). "
                         "(default: ../../input/Land relative to this script)")
    ap.add_argument("--prod_trade_dir", default=default_prod_trade_dir,
                    help="Folder containing FoodBalanceSheets_E_All_Data_NOFLAG_demand_refilled files "
                         "(default: ../../input/Production_Trade relative to this script)")
    args = ap.parse_args()

    _ensure_dir(args.data_dir)
    _ensure_dir(args.out_dir)
    _ensure_dir(args.land_dir)

    panel, mapping = build_panel(args.dict_xlsx, args.data_dir, args.land_dir, args.prod_trade_dir, region_sheet=args.region_sheet)
    out_xlsx = write_outputs(panel, mapping, args.data_dir, args.dict_xlsx, args.region_sheet, args.land_dir, args.prod_trade_dir, args.out_dir)

    print("Done.")
    print("Excel:", out_xlsx)


if __name__ == "__main__":
    main()
