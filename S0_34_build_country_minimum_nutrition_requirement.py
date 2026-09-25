# -*- coding: utf-8 -*-
"""
Build country/region nutrition indicators using dict_v3.xlsx (region sheet) as the region info source.

- MDER (kcal/cap/day): FAO ESS "MinimumDietaryEnergyRequirement_en.xls" (3-year windows)
- protein_g_cap_day & fat_g_cap_day (g/cap/day): FAOSTAT bulk (Food Security Data) latest 3-year window

Input (region info): ../../src/dict_v3.xlsx, sheet "region"
Expected columns (flexible):
  - M49 code: one of ["M49_Country_Code", "Area Code (M49)", "Area Code"]
  - ISO3: one of ["ISO3 Code", "ISO3"]
  - Region rows: ISO3 == "no" (case-insensitive) represent region/grouping codes.
Output: /mnt/data/region_with_MDER_protein_fat.xlsx by default
"""

from __future__ import annotations

import argparse
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests


# URLs

MDER_XLS_URL = (
    "https://www.fao.org/fileadmin/templates/ess/documents/"
    "food_security_statistics/MinimumDietaryEnergyRequirement_en.xls"
)

FAOSTAT_FS_ZIP_URLS = [
    "https://bulks-faostat.fao.org/production/Food_Security_Data_E_All_Data_(Normalized).zip",
    "https://fenixservices.fao.org/faostat/static/bulkdownloads/Food_Security_Data_E_All_Data_(Normalized).zip",
]

M49_MAP_JSON_URL = "https://codelists.codeforiati.org/api/json/en/RegionM49.json"



# Helpers

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def download(url: str, out_path: str, timeout: int = 60) -> str:
    _ensure_dir(os.path.dirname(out_path))
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)
    return out_path


def download_first_working(urls: List[str], out_path: str, timeout: int = 60) -> str:
    last_err = None
    for u in urls:
        try:
            return download(u, out_path, timeout=timeout)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"All downloads failed for: {urls}. Last error: {last_err}")


def clean_m49(x: pd.Series) -> pd.Series:
    s = x.astype(str).str.strip()
    s = s.str.replace("'", "", regex=False)
    s = s.replace({"nan": np.nan, "None": np.nan})
    out = pd.to_numeric(s, errors="coerce")
    return out.astype("Int64")


def safe_mean(s: pd.Series) -> float:
    s2 = pd.to_numeric(s, errors="coerce")
    if s2.notna().sum() == 0:
        return np.nan
    return float(s2.mean(skipna=True))


def pick_latest_3yr_window(year_values: Iterable[str]) -> Optional[str]:
    wins = []
    for y in set(map(str, year_values)):
        y2 = y.strip()
        m = re.match(r"^(\d{4})\s*[-–]\s*(\d{4})$", y2)
        if m:
            wins.append((int(m.group(2)), y2))
    if not wins:
        return None
    wins.sort()
    return wins[-1][1]


def standardize_region_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize to required columns:
      - M49_Country_Code
      - ISO3 Code
    """
    df = df.copy()

    # ISO3
    iso3_col = None
    for c in ["ISO3 Code", "ISO3", "iso3 code", "iso3"]:
        if c in df.columns:
            iso3_col = c
            break
    if iso3_col is None:
        raise ValueError("Cannot find ISO3 column in dict_v3.xlsx region sheet (need 'ISO3 Code' or 'ISO3').")
    if iso3_col != "ISO3 Code":
        df = df.rename(columns={iso3_col: "ISO3 Code"})

    # M49
    m49_col = None
    for c in ["M49_Country_Code", "Area Code (M49)", "Area Code", "M49 code", "M49"]:
        if c in df.columns:
            m49_col = c
            break
    if m49_col is None:
        raise ValueError(
            "Cannot find M49/Area code column in dict_v3.xlsx region sheet "
            "(need one of 'M49_Country_Code' / 'Area Code (M49)' / 'Area Code')."
        )
    if m49_col != "M49_Country_Code":
        df = df.rename(columns={m49_col: "M49_Country_Code"})

    # Keep all original columns but ensure key ones exist
    df["ISO3 Code"] = df["ISO3 Code"].astype(str).str.strip()
    return df



# Load M49 mapping & membership

@dataclass
class M49Membership:
    m49_df: pd.DataFrame
    code_to_name: Dict[int, str]

    def members_for_region_code(self, code: int) -> Set[int]:
        df = self.m49_df

        if code in (901, 902):
            return set()

        # groupings via flags
        if code == 199:
            return set(df.loc[df["Least Developed Countries (LDC)"].fillna(0).astype(int) == 1, "Code"].tolist())
        if code == 432:
            return set(df.loc[df["Land Locked Developing Countries (LLDC)"].fillna(0).astype(int) == 1, "Code"].tolist())
        if code == 722:
            return set(df.loc[df["Small Island Developing States (SIDS)"].fillna(0).astype(int) == 1, "Code"].tolist())

        # special SDG-style aggregate
        if code == 97:
            m_eur = (
                df["Region code"].astype(str).str.replace(r"\D", "", regex=True).replace("", np.nan).astype("Float64") == 150
            )
            m_na = (
                df["Sub-region code"].astype(str).str.replace(r"\D", "", regex=True).replace("", np.nan).astype("Float64") == 21
            )
            return set(df.loc[(m_eur | m_na), "Code"].tolist())

        # standard hierarchy match
        cols = ["Global code", "Region code", "Sub-region code", "Intermediate region code"]
        mask = np.zeros(len(df), dtype=bool)
        for c in cols:
            v = df[c].astype(str).str.replace(r"\D", "", regex=True).replace("", np.nan)
            v_num = pd.to_numeric(v, errors="coerce")
            mask |= (v_num == code).fillna(False).to_numpy()

        return set(df.loc[mask, "Code"].tolist())

    def name_for_code(self, code: int) -> Optional[str]:
        if code in self.code_to_name:
            return self.code_to_name[code]
        manual = {
            97: "Europe and Northern America",
            199: "Least Developed Countries (LDC)",
            432: "Land Locked Developing Countries (LLDC)",
            722: "Small Island Developing States (SIDS)",
        }
        return manual.get(code)


def load_m49_membership(cache_dir: str) -> M49Membership:
    fp = os.path.join(cache_dir, "RegionM49.json")
    if not os.path.exists(fp):
        download(M49_MAP_JSON_URL, fp)

    obj = requests.get(M49_MAP_JSON_URL, timeout=60).json()
    data = obj.get("data", obj)
    df = pd.DataFrame(data)

    if "Code" not in df.columns:
        raise ValueError("Unexpected M49 JSON schema: cannot find 'Code'.")
    df["Code"] = pd.to_numeric(df["Code"], errors="coerce").astype("Int64")

    code_to_name: Dict[int, str] = {}
    pairs = [
        ("Global code", "Global name"),
        ("Region code", "Region name"),
        ("Sub-region code", "Sub-region name"),
        ("Intermediate region code", "Intermediate region name"),
    ]
    for c_code, c_name in pairs:
        if c_code in df.columns and c_name in df.columns:
            tmp = df[[c_code, c_name]].dropna()
            for _, row in tmp.iterrows():
                try:
                    k = int(re.sub(r"\D", "", str(row[c_code])))
                    v = str(row[c_name]).strip()
                    if k and v:
                        code_to_name.setdefault(k, v)
                except Exception:
                    continue

    code_to_name.setdefault(1, "World")
    return M49Membership(m49_df=df, code_to_name=code_to_name)



# Load MDER

def load_mder_table(cache_dir: str) -> pd.DataFrame:
    fp = os.path.join(cache_dir, "MinimumDietaryEnergyRequirement_en.xls")
    if not os.path.exists(fp):
        download(MDER_XLS_URL, fp)

    try:
        df = pd.read_excel(fp)
    except Exception:
        df = pd.read_excel(fp, engine="xlrd")

    rename = {}
    for c in df.columns:
        c2 = str(c).strip()
        if c2.lower() in ("iso3 code", "iso3"):
            rename[c] = "ISO3 Code"
        if c2.lower() in ("area code", "area code (m49)", "m49 code", "m49"):
            rename[c] = "Area Code (M49)"
    df = df.rename(columns=rename)

    period_cols = []
    for c in df.columns:
        c2 = str(c).strip()
        if re.match(r"^\d{4}\s*-\s*\d{2}$", c2):
            period_cols.append(c)

    keep_cols = [c for c in ["ISO3 Code", "Area Code (M49)"] if c in df.columns] + period_cols
    df = df[keep_cols].copy()

    for c in period_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Area Code (M49)" in df.columns:
        df["Area Code (M49)"] = pd.to_numeric(df["Area Code (M49)"], errors="coerce").astype("Int64")

    return df


def reshape_mder(df_mder: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    period_cols = [c for c in df_mder.columns if re.match(r"^\d{4}\s*-\s*\d{2}$", str(c).strip())]
    out = df_mder.copy()
    mder_cols: List[str] = []
    for c in period_cols:
        key = str(c).strip().replace(" ", "").replace("-", "_")
        newc = f"MDER_{key}_kcal_cap_day"
        out = out.rename(columns={c: newc})
        mder_cols.append(newc)
    return out, mder_cols



# Load FAOSTAT Food Security protein/fat

def load_faostat_food_security(cache_dir: str) -> pd.DataFrame:
    fp = os.path.join(cache_dir, "Food_Security_Data_E_All_Data_(Normalized).zip")
    if not os.path.exists(fp):
        download_first_working(FAOSTAT_FS_ZIP_URLS, fp)

    with zipfile.ZipFile(fp, "r") as z:
        csv_names = [n for n in z.namelist() if n.lower().endswith(".csv") and "food_security_data_e_all_data" in n.lower()]
        if not csv_names:
            csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError("No CSV found in Food Security bulk ZIP.")
        csv_name = sorted(csv_names, key=len)[0]
        with z.open(csv_name) as f:
            df = pd.read_csv(f, low_memory=False)

    # Normalize columns
    rename = {}
    for c in df.columns:
        c2 = str(c).strip().lower()
        if c2 == "area code (m49)":
            rename[c] = "Area Code (M49)"
        if c2 == "area code":
            rename[c] = "Area Code"
        if c2 == "year":
            rename[c] = "Year"
        if c2 == "value":
            rename[c] = "Value"
    df = df.rename(columns=rename)

    if "Area Code (M49)" not in df.columns:
        if "Area Code" in df.columns:
            df["Area Code (M49)"] = pd.to_numeric(df["Area Code"], errors="coerce").astype("Int64")
        else:
            raise ValueError("Cannot find Area Code (M49) or Area Code in Food Security bulk file.")
    if "Year" not in df.columns:
        raise ValueError("Cannot find Year column in Food Security bulk file.")
    if "Value" not in df.columns:
        raise ValueError("Cannot find Value column in Food Security bulk file.")
    if "Item" not in df.columns or "Element" not in df.columns:
        raise ValueError("Food Security bulk file must contain 'Item' and 'Element' columns.")

    df["Area Code (M49)"] = pd.to_numeric(df["Area Code (M49)"], errors="coerce").astype("Int64")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["Year"] = df["Year"].astype(str).str.strip()

    return df


def extract_indicator_latest_window(df_fs: pd.DataFrame, item_exact: str) -> Tuple[pd.DataFrame, str]:
    m = (df_fs["Item"].astype(str).str.strip() == item_exact) & (df_fs["Element"].astype(str).str.strip() == "Value")
    sub = df_fs.loc[m, ["Area Code (M49)", "Year", "Value"]].copy()

    win = pick_latest_3yr_window(sub["Year"].unique())
    if win is None:
        raise ValueError(f"Cannot detect 3-year window for indicator: {item_exact}")

    sub = sub.loc[sub["Year"] == win, ["Area Code (M49)", "Value"]].copy()
    out = sub.groupby("Area Code (M49)", as_index=False)["Value"].mean()

    return out, win



# Main build

def build(dict_xlsx: str, region_sheet: str, output_xlsx: str, cache_dir: str) -> None:
    _ensure_dir(cache_dir)

    # 1) Read region info from dict_v3.xlsx
    df_region_raw = pd.read_excel(dict_xlsx, sheet_name=region_sheet)
    df_region = standardize_region_df(df_region_raw)

    # Keep all columns, but require ISO3 + M49
    df_region["M49_int"] = clean_m49(df_region["M49_Country_Code"])
    df_region["is_region_row"] = (df_region["ISO3 Code"].astype(str).str.strip().str.lower() == "no")

    # 2) Load MDER and merge
    mder_raw = load_mder_table(cache_dir)
    mder, mder_cols = reshape_mder(mder_raw)

    df_out = df_region.copy()

    # merge by ISO3 first (if MDER has ISO3)
    if "ISO3 Code" in mder.columns:
        df_out = df_out.merge(
            mder.drop(columns=[c for c in ["Area Code (M49)"] if c in mder.columns]),
            on="ISO3 Code",
            how="left",
        )

    # fill missing by M49
    if "Area Code (M49)" in mder.columns:
        mder_m49 = mder[["Area Code (M49)"] + mder_cols].dropna(subset=["Area Code (M49)"]).copy()
        df_out = df_out.merge(
            mder_m49,
            left_on="M49_int",
            right_on="Area Code (M49)",
            how="left",
            suffixes=("", "_m49"),
        )
        for c in mder_cols:
            if f"{c}_m49" in df_out.columns:
                df_out[c] = df_out[c].combine_first(df_out[f"{c}_m49"])
                df_out.drop(columns=[f"{c}_m49"], inplace=True)
        df_out.drop(columns=["Area Code (M49)"], inplace=True)

    # 3) Load protein/fat and merge (latest 3-year window)
    fs = load_faostat_food_security(cache_dir)

    prot_tbl, prot_win = extract_indicator_latest_window(fs, "Average protein supply (g/cap/day)")
    fat_tbl, fat_win = extract_indicator_latest_window(fs, "Average fat supply (g/cap/day)")

    df_out = df_out.merge(prot_tbl.rename(columns={"Value": "protein_g_cap_day"}), left_on="M49_int", right_on="Area Code (M49)", how="left")
    df_out.drop(columns=["Area Code (M49)"], inplace=True)
    df_out = df_out.merge(fat_tbl.rename(columns={"Value": "fat_g_cap_day"}), left_on="M49_int", right_on="Area Code (M49)", how="left")
    df_out.drop(columns=["Area Code (M49)"], inplace=True)

    # 4) Region aggregates for ISO3 == "no" rows
    m49mem = load_m49_membership(cache_dir)

    df_out["RegionName_if_ISO3_no"] = np.nan
    df_out["RegionType"] = np.nan
    df_out["Region_nCountries_MDER"] = np.nan

    country_mask = ~df_out["is_region_row"]
    countries_in_input: Set[int] = set(df_out.loc[country_mask, "M49_int"].dropna().astype(int).tolist())

    mder_cols_nonempty = [c for c in mder_cols if c in df_out.columns]
    mder_n_col = mder_cols_nonempty[-1] if mder_cols_nonempty else None

    region_rows = df_out.loc[df_out["is_region_row"] & df_out["M49_int"].notna()].index.tolist()
    for idx in region_rows:
        code = int(df_out.at[idx, "M49_int"])
        name = m49mem.name_for_code(code)
        if name:
            df_out.at[idx, "RegionName_if_ISO3_no"] = name

        if code in (199, 432, 722):
            df_out.at[idx, "RegionType"] = "grouping"
        elif code == 97:
            df_out.at[idx, "RegionType"] = "special"
        elif code in (901, 902):
            df_out.at[idx, "RegionType"] = "unknown"
        else:
            df_out.at[idx, "RegionType"] = "tree"

        members = m49mem.members_for_region_code(code)
        members = set(int(x) for x in members if pd.notna(x)) & countries_in_input
        if not members:
            continue

        member_mask = df_out["M49_int"].isin(list(members)) & (~df_out["is_region_row"])

        for c in mder_cols_nonempty:
            df_out.at[idx, c] = safe_mean(df_out.loc[member_mask, c])

        if mder_n_col:
            df_out.at[idx, "Region_nCountries_MDER"] = int(df_out.loc[member_mask, mder_n_col].notna().sum())

        df_out.at[idx, "protein_g_cap_day"] = safe_mean(df_out.loc[member_mask, "protein_g_cap_day"])
        df_out.at[idx, "fat_g_cap_day"] = safe_mean(df_out.loc[member_mask, "fat_g_cap_day"])

    # 5) Cleanup helpers
    df_out.drop(columns=["M49_int", "is_region_row"], inplace=True)

    # 6) sources
    sources = pd.DataFrame(
        [
            {
                "variable": "MDER_*_kcal_cap_day",
                "source": "FAO ESS: Minimum Dietary Energy Requirement (MDER)",
                "url": MDER_XLS_URL,
                "note": "Values are reported by 3-year windows (e.g., 1990-92, 1995-97, ...).",
            },
            {
                "variable": "protein_g_cap_day",
                "source": "FAOSTAT Bulk Download: Food Security Data (Normalized)",
                "indicator": "Average protein supply (g/cap/day) (3-year average)",
                "period_used": prot_win,
                "url": FAOSTAT_FS_ZIP_URLS[0],
            },
            {
                "variable": "fat_g_cap_day",
                "source": "FAOSTAT Bulk Download: Food Security Data (Normalized)",
                "indicator": "Average fat supply (g/cap/day) (3-year average)",
                "period_used": fat_win,
                "url": FAOSTAT_FS_ZIP_URLS[0],
            },
            {
                "variable": "Region aggregates (ISO3 == 'no')",
                "source": "UN M49 membership via RegionM49 codelist (IATI mirror)",
                "url": M49_MAP_JSON_URL,
                "note": "Region rows are computed as simple means across member countries present in dict_v3 region sheet.",
            },
        ]
    )

    # 7) Write
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as w:
        df_out.to_excel(w, sheet_name="region", index=False)
        sources.to_excel(w, sheet_name="sources", index=False)

    print(f"Saved: {output_xlsx}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dict",
        default=str((Path(__file__).resolve().parent / "../../src/dict_v3.xlsx").resolve()),
        help="Path to dict_v3.xlsx",
    )
    ap.add_argument("--sheet", default="region", help="Sheet name for region info")
    ap.add_argument("--output", default="/mnt/data/region_with_MDER_protein_fat.xlsx", help="Output Excel path")
    ap.add_argument("--cache", default="/mnt/data/_cache_fao", help="Cache directory for downloads")
    args = ap.parse_args()

    build(args.dict, args.sheet, args.output, args.cache)


if __name__ == "__main__":
    main()
