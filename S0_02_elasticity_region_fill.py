# -*- coding: utf-8 -*-
"""
Fill regional means using country-to-region and commodity-to-Crop/Meat/Dairy/Other mappings,
preserving all workbook sheets and replacing only six target tables.

(1) Read region_map.xlsx.
    - region sheet: Country or equivalent -> Region or equivalent.
    - item sheet: Commodity -> Commodity_agg in Crop/Meat/Dairy/Other.

(2) Four non-cross tables: Supply-Temperature, Supply-Own-Price, Demand-Own-Price, Demand-Income.
    - Fill Elasticity_mean/min/max with non-NaN Region-Commodity means first.
    - If unavailable, use Region-Commodity_agg means.
    - Leave remaining gaps as NaN.

(3) Two cross tables: Demand_Cross_mean and Supply_Cross_mean.
    - Regional filling only: zero -> NaN, join Region, fill each column with Region-Commodity means, then remaining NaN -> zero.
    - Do not use Commodity_agg-level filling.

Notes:
- Replace only these six tables; copy other sheet data without guaranteeing formatting/formula preservation.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# Paths: defaults work when run in the same directory.

IN_ELA  = Path("../../src/bakup/Elasticity_v3_processed_out3.1.xlsx")
IN_RMAP = Path("../../src/dict_v3.xlsx")
OUT_ELA = Path("../../src/bakup/Elasticity_v3_processed_filled_by_region_.xlsx")

# Target sheet names
NON_CROSS_SHEETS  = ["Supply-Temperature", "Supply-Own-Price", "Demand-Own-Price", "Demand-Income"]
CROSS_MEAN_SHEETS = ["Demand_Cross_mean", "Supply_Cross_mean"]



# Helpers for string normalization and fuzzy column matching

def norm(s):
    """Trim, normalize dashes, and collapse spaces; preserve NaN."""
    if pd.isna(s):
        return np.nan
    s = str(s).strip().replace("–", "-").replace("—", "-")
    s = " ".join(s.split())
    return s

def canon(name: str) -> str:
    """Normalize column names for fuzzy matching."""
    return str(name).strip().lower().replace("_", " ").replace("-", " ")

def find_col(columns, candidates):
    """
    Find candidate columns, ignoring case, underscores, and hyphens.
    candidates: Alternative names, e.g. ['Country', 'Region_label_new'].
    Return the actual matching name, or None.
    """
    cmap = {canon(c): c for c in columns}
    for cand in candidates:
        key = canon(cand)
        if key in cmap:
            return cmap[key]
    return None



# Read Country-to-Region and Commodity-to-Commodity_agg mappings.

def load_mappings(xlsx_path: Path):
    xls = pd.ExcelFile(xlsx_path)
    sheets = set(xls.sheet_names)

    # 1) region sheet：Country??Region
    reg = pd.read_excel(xls, sheet_name="region")
    reg.columns = [str(c).strip() for c in reg.columns]

    reg["Country"] = reg["Region_label_new"].map(norm)
    reg["Region"]  = reg["Region_agg4"].map(norm)
    reg = reg.dropna(subset=["Country", "Region"]).drop_duplicates(subset=["Country"])

    # 2) item sheet：Commodity??Commodity_agg
    itm = pd.read_excel(xls, sheet_name="Emis_item")
    itm.columns = [str(c).strip() for c in itm.columns]

    itm["Commodity"]     = itm["Item_Elasticity_Map"].map(norm)
    itm["Commodity_agg"] = itm["Item_Cat2"].map(norm)

    # Allow only four categories; map blanks or unknown categories to Other.
    allowed = {"crop", "meat", "dairy", "other"}
    itm["Commodity_agg"] = itm["Commodity_agg"].apply(
        lambda x: "Other" if pd.isna(x) or str(x).strip().lower() not in allowed else str(x).strip().title()
    )

    return reg, itm



# Fill non-cross tables by Region-Commodity, then Region-Commodity_agg.

def fill_non_cross_with_region_and_cat(df: pd.DataFrame, reg: pd.DataFrame, itm: pd.DataFrame) -> pd.DataFrame:
    """
    Fill non-cross sheets containing Elasticity_* columns in two stages:
      Stage 1: Region-Commodity means.
      Stage 2: Region-Commodity_agg means.
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    # Normalize key columns.
    for c in ["Country", "Commodity"]:
        if c in df.columns:
            df[c] = df[c].map(norm)

    # Target columns to fill
    raw_target_cols = [c for c in ["Elasticity_mean", "Elasticity_min", "Elasticity_max"] if c in df.columns]
    numeric_targets = [c for c in raw_target_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_targets:
        # Return directly if no target columns exist.
        return df

    # Join Region information.
    df = df.merge(reg, on="Country", how="left")

    # Stage 1: Region-Commodity means.
    # Group only rows with a Region.
    has_region = df["Region"].notna()

    if numeric_targets:
        grp1 = df.loc[has_region].groupby(["Region", "Commodity"])[numeric_targets]
        reg_comm_means = grp1.transform("mean").reindex(df.index)

        # Fill using stage-1 means.
        for col in numeric_targets:
            mask = df[col].isna() & df["Region"].notna() & reg_comm_means[col].notna()
            df.loc[mask, col] = reg_comm_means.loc[mask, col]

    # Stage 2: Region-Commodity_agg means.
    # Join Commodity_agg.
    df = df.merge(itm, on="Commodity", how="left")
    # Unmapped commodities were assigned Other by load_mappings.
    has_region_cat = df["Region"].notna() & df["Commodity_agg"].notna()
    if has_region_cat.any() and numeric_targets:
        grp2 = df.loc[has_region_cat].groupby(["Region", "Commodity_agg"])[numeric_targets]
        reg_cat_means = grp2.transform("mean").reindex(df.index)

        for col in numeric_targets:
            mask = df[col].isna() & has_region_cat & reg_cat_means[col].notna()
            df.loc[mask, col] = reg_cat_means.loc[mask, col]

    # Remove helper columns.
    return df.drop(columns=[c for c in ["Region", "Commodity_agg"] if c in df.columns])



# Cross tables: zero -> NaN -> Region-Commodity mean -> zero for remaining NaNs.

def fill_cross_mean_with_region(df: pd.DataFrame, reg: pd.DataFrame) -> pd.DataFrame:
    """
    For wide cross-mean tables with Country, Commodity, and cross-item columns.
    Logic:
      - Treat every zero as missing (NaN).
      - Join Region and fill each cross column with Region-Commodity means.
      - Replace remaining NaNs with zero.
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    for c in ["Country", "Commodity"]:
        if c in df.columns:
            df[c] = df[c].map(norm)

    # Cross columns are all columns except Country and Commodity.
    cross_cols_all = [c for c in df.columns if c not in ["Country", "Commodity"]]
    num_cols = [c for c in cross_cols_all if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        return df

    # 0 -> NaN
    df[num_cols] = df[num_cols].replace(0, np.nan)

    # Join Region.
    df = df.merge(reg, on="Country", how="left")
    has_region = df["Region"].notna()

    if has_region.any():
     # Compute Region-Commodity means separately for numeric columns.
        grp = df.loc[has_region].groupby(["Region", "Commodity"])[num_cols]
        reg_comm_means = grp.transform("mean").reindex(df.index)
        for col in num_cols:
             mask = df[col].isna() & df["Region"].notna() & reg_comm_means[col].notna()
             df.loc[mask, col] = reg_comm_means.loc[mask, col]
    # Remaining NaN -> zero.
    df[num_cols] = df[num_cols].fillna(0)

    return df.drop(columns=["Region"])



# Main workflow: read -> process -> write, preserving all sheets.

# 1) Load mappings.
reg, itm = load_mappings(IN_RMAP)

# 2) Read all original sheets as {sheet_name: DataFrame}.
xls = pd.ExcelFile(IN_ELA)
all_sheets = {}
for name in xls.sheet_names:
    all_sheets[name] = pd.read_excel(xls, sheet_name=name)

# 3) Process the six target sheets.
processed = {}

# Two-stage filling for four non-cross sheets
for name in NON_CROSS_SHEETS:
    if name in all_sheets:
        df = all_sheets[name]
        processed[name] = fill_non_cross_with_region_and_cat(df, reg, itm)

# Two cross sheets: zero -> NaN -> regional mean -> zero.
for name in CROSS_MEAN_SHEETS:
    if name in all_sheets:
        df = all_sheets[name]
        processed[name] = fill_cross_mean_with_region(df, reg)

# 4) Write all sheets, replacing only the six processed tables.
with pd.ExcelWriter(OUT_ELA, engine="xlsxwriter") as writer:
    for name, df in all_sheets.items():
        if name in processed:           # Use processed data.
            processed[name].to_excel(writer, sheet_name=name[:31], index=False)
        else:                            # Write other sheets unchanged.
            df.to_excel(writer, sheet_name=name[:31], index=False)

print(f"完成：{OUT_ELA.resolve()}")

