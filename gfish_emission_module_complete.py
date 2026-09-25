# -*- coding: utf-8 -*-
"""
Compute fish farming emissions (CH4, N2O) using the aquaculture panel.
Historical years are read directly from the panel; future years are computed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

FISH_ITEM = "Fish, Seafood"
FISH_PROCESS = "Fish farming"
PANEL_SHEET = "country_year_panel"

REQUIRED_PANEL_COLS = [
    "M49_Country_Code",
    "Year",
    "Aquaculture_share",
    "Capture_share",
    "Aquaculture_yield_t_per_ha",
    "EF_aqua_CH4_kg_per_ha_yr_median",
    "Live_to_product_yield_kg_per_kg",
    "EF_aqua_N2O_kg_per_kg_median",
    "Aquaculture_CH4_emissions_kg_yr",
    "Aquaculture_N2O_emissions_kg_yr",
]


def _norm_m49(val: Any) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip()
    if s.startswith("'"):
        s = s[1:]
    s = s.strip()
    if not s:
        return ""
    if s.count(".") == 1:
        left, right = s.split(".", 1)
        if left.isdigit() and right.strip("0") == "":
            s = left
    if s.isdigit():
        return f"'{s.zfill(3)}"
    return f"'{s}"


def _pick_col(cols: List[str], cands: List[str]) -> Optional[str]:
    for c in cands:
        if c in cols:
            return c
    return None


def _load_dict_v3_country_maps(dict_v3_path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    df = pd.read_excel(dict_v3_path, sheet_name="region")
    df.columns = [str(c).strip() for c in df.columns]

    m49_col = _pick_col(list(df.columns), ["M49_Country_Code", "M49 Code", "M49"])
    country_col = _pick_col(list(df.columns), ["Country", "country"])
    region_col = _pick_col(list(df.columns), ["Region_label_new"])
    if m49_col is None or country_col is None:
        raise RuntimeError(f"dict_v3 region sheet missing M49/Country columns: {list(df.columns)}")

    if region_col:
        df = df[df[region_col].astype(str).str.strip().str.lower() != "no"].copy()
    df["M49_Country_Code"] = df[m49_col].apply(_norm_m49)
    df["Country"] = df[country_col].astype(str).str.strip()
    df = df[(df["M49_Country_Code"] != "") & (df["Country"] != "")]

    m49_to_country = dict(zip(df["M49_Country_Code"], df["Country"]))
    country_to_m49 = dict(zip(df["Country"], df["M49_Country_Code"]))
    return m49_to_country, country_to_m49


def _load_fish_item_map(dict_v3_path: str) -> List[str]:
    df = pd.read_excel(dict_v3_path, sheet_name="Emis_item")
    df.columns = [str(c).strip() for c in df.columns]
    if "Process" not in df.columns:
        return [FISH_ITEM]
    fish = df[df["Process"].astype(str).str.strip() == FISH_PROCESS].copy()
    if fish.empty:
        return [FISH_ITEM]
    items = set()
    for col in ["Item_Production_Map", "Item_Emis"]:
        if col in fish.columns:
            for v in fish[col].dropna().tolist():
                s = str(v).strip()
                if s and s.lower() != "nan":
                    items.add(s)
    return sorted(items) if items else [FISH_ITEM]


def _load_fish_panel(panel_path: str) -> pd.DataFrame:
    fpath = Path(panel_path)
    if not fpath.exists():
        raise FileNotFoundError(f"Fish panel not found: {fpath}")
    df = pd.read_excel(fpath, sheet_name=PANEL_SHEET)
    df.columns = [str(c).strip() for c in df.columns]
    if "year" in df.columns and "Year" not in df.columns:
        df = df.rename(columns={"year": "Year"})
    missing = [c for c in REQUIRED_PANEL_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Fish panel missing required columns: {missing}. Columns: {list(df.columns)}")

    df["M49_Country_Code"] = df["M49_Country_Code"].apply(_norm_m49)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["M49_Country_Code", "Year"])
    df["Year"] = df["Year"].astype(int)
    return df


def _select_baseline_rows(panel_df: pd.DataFrame, base_year: int) -> pd.DataFrame:
    df = panel_df.sort_values(["M49_Country_Code", "Year"]).copy()

    def _pick(g: pd.DataFrame) -> pd.Series:
        base = g[g["Year"] == base_year]
        if base.empty:
            base = g[g["Year"] <= base_year]
        if base.empty:
            base = g
        return base.iloc[-1]

    base = df.groupby("M49_Country_Code", as_index=False).apply(_pick)
    return base.reset_index(drop=True)


def _lookup_mult_by_country(
    mult_dict: Optional[Dict],
    country: str,
    item: str,
    process: Optional[str],
    year: int,
    *,
    m49: Optional[str] = None,
    ghg: Optional[str] = None
) -> float:
    if not mult_dict:
        return 1.0
    ghg_key = str(ghg).strip() if ghg else None
    keys = []
    if process is None:
        keys = [
            (country, item, year),
            (country, "All", year),
        ]
        if m49:
            keys.extend([(m49, item, year), (m49, "All", year)])
    else:
        if ghg_key:
            keys = [
                (country, item, process, ghg_key, year),
                (country, item, "All", ghg_key, year),
                (country, "All", process, ghg_key, year),
                (country, "All", "All", ghg_key, year),
                (country, item, process, "All", year),
                (country, item, "All", "All", year),
                (country, "All", process, "All", year),
                (country, "All", "All", "All", year),
            ]
            if m49:
                keys.extend([
                    (m49, item, process, ghg_key, year),
                    (m49, item, "All", ghg_key, year),
                    (m49, "All", process, ghg_key, year),
                    (m49, "All", "All", ghg_key, year),
                    (m49, item, process, "All", year),
                    (m49, item, "All", "All", year),
                    (m49, "All", process, "All", year),
                    (m49, "All", "All", "All", year),
                ])
        else:
            keys = [
                (country, item, process, year),
                (country, item, "All", year),
                (country, "All", process, year),
                (country, "All", "All", year),
            ]
            if m49:
                keys.extend([
                    (m49, item, process, year),
                    (m49, item, "All", year),
                    (m49, "All", process, year),
                    (m49, "All", "All", year),
                ])
    for k in keys:
        if k in mult_dict:
            return float(mult_dict.get(k, 1.0))
    return 1.0


def _lookup_abs_by_country(
    abs_dict: Optional[Dict],
    country: str,
    item: str,
    process: Optional[str],
    ghg: str,
    year: int,
    *,
    m49: Optional[str] = None
) -> Optional[float]:
    if not abs_dict:
        return None
    ghg_key = str(ghg).strip() if ghg else 'All'
    keys = []
    if process is None:
        keys = [
            (country, item, ghg_key, year),
            (country, "All", ghg_key, year),
            (country, item, "All", year),
            (country, "All", "All", year),
        ]
        if m49:
            keys.extend([
                (m49, item, ghg_key, year),
                (m49, "All", ghg_key, year),
                (m49, item, "All", year),
                (m49, "All", "All", year),
            ])
    else:
        keys = [
            (country, item, process, ghg_key, year),
            (country, item, "All", ghg_key, year),
            (country, "All", process, ghg_key, year),
            (country, "All", "All", ghg_key, year),
            (country, item, process, "All", year),
            (country, item, "All", "All", year),
            (country, "All", process, "All", year),
            (country, "All", "All", "All", year),
            (country, item, process, year),
            (country, item, "All", year),
            (country, "All", process, year),
            (country, "All", "All", year),
        ]
        if m49:
            keys.extend([
                (m49, item, process, ghg_key, year),
                (m49, item, "All", ghg_key, year),
                (m49, "All", process, ghg_key, year),
                (m49, "All", "All", ghg_key, year),
                (m49, item, process, "All", year),
                (m49, item, "All", "All", year),
                (m49, "All", process, "All", year),
                (m49, "All", "All", "All", year),
                (m49, item, process, year),
                (m49, item, "All", year),
                (m49, "All", process, year),
                (m49, "All", "All", year),
            ])
    for k in keys:
        if k in abs_dict:
            try:
                return float(abs_dict.get(k))
            except Exception:
                return None
    return None


def _lookup_bound_by_country(
    bound_dict: Optional[Dict],
    country: str,
    item: str,
    process: Optional[str],
    ghg: str,
    year: int,
    *,
    m49: Optional[str] = None
) -> Optional[Tuple[float, float, bool, bool, float]]:
    if not bound_dict:
        return None
    ghg_key = str(ghg).strip() if ghg else 'All'
    keys = []
    if process is None:
        keys = [
            (country, item, ghg_key, year),
            (country, "All", ghg_key, year),
            (country, item, "All", year),
            (country, "All", "All", year),
        ]
        if m49:
            keys.extend([
                (m49, item, ghg_key, year),
                (m49, "All", ghg_key, year),
                (m49, item, "All", year),
                (m49, "All", "All", year),
            ])
    else:
        keys = [
            (country, item, process, ghg_key, year),
            (country, item, "All", ghg_key, year),
            (country, "All", process, ghg_key, year),
            (country, "All", "All", ghg_key, year),
            (country, item, process, "All", year),
            (country, item, "All", "All", year),
            (country, "All", process, "All", year),
            (country, "All", "All", "All", year),
            (country, item, process, year),
            (country, item, "All", year),
            (country, "All", process, year),
            (country, "All", "All", year),
        ]
        if m49:
            keys.extend([
                (m49, item, process, ghg_key, year),
                (m49, item, "All", ghg_key, year),
                (m49, "All", process, ghg_key, year),
                (m49, "All", "All", ghg_key, year),
                (m49, item, process, "All", year),
                (m49, item, "All", "All", year),
                (m49, "All", process, "All", year),
                (m49, "All", "All", "All", year),
                (m49, item, process, year),
                (m49, item, "All", year),
                (m49, "All", process, year),
                (m49, "All", "All", year),
            ])
    for k in keys:
        if k in bound_dict:
            return bound_dict.get(k)
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
def run_fish_emissions(
    production_df: pd.DataFrame,
    years: List[int],
    fish_panel_path: str,
    dict_v3_path: str,
    scenario_params: Optional[Dict] = None,
    hist_cutoff_year: int = 2020
) -> Dict[str, Any]:
    scenario_params = scenario_params if isinstance(scenario_params, dict) else {}
    panel = _load_fish_panel(fish_panel_path)
    m49_to_country, country_to_m49 = _load_dict_v3_country_maps(dict_v3_path)
    fish_items = _load_fish_item_map(dict_v3_path)
    fish_items_lower = {str(x).strip().lower() for x in fish_items}

    years = sorted({int(y) for y in years})
    hist_years = [y for y in years if y <= hist_cutoff_year]
    future_years = [y for y in years if y > hist_cutoff_year]

    base_year = hist_cutoff_year
    base_rows = _select_baseline_rows(panel, base_year)

    param_cols = [
        "Aquaculture_share",
        "Capture_share",
        "Aquaculture_yield_t_per_ha",
        "EF_aqua_CH4_kg_per_ha_yr_median",
        "Live_to_product_yield_kg_per_kg",
        "EF_aqua_N2O_kg_per_kg_median",
    ]
    global_means = (
        panel[panel["Year"] == base_year][param_cols]
        .apply(pd.to_numeric, errors="coerce")
        .mean(numeric_only=True)
    )

    base = base_rows[["M49_Country_Code"] + param_cols].copy()
    for col in param_cols:
        base[col] = pd.to_numeric(base[col], errors="coerce")

    # Aquaculture share
    share = base["Aquaculture_share"]
    cap_share = base["Capture_share"]
    share = share.where(share.notna(), 1.0 - cap_share)
    if pd.notna(global_means.get("Aquaculture_share")):
        share = share.fillna(global_means.get("Aquaculture_share"))
    share = share.clip(lower=0.0, upper=1.0)
    base["Aquaculture_share"] = share
    base["Capture_share"] = 1.0 - share

    # Other parameters
    for col in [
        "Aquaculture_yield_t_per_ha",
        "EF_aqua_CH4_kg_per_ha_yr_median",
        "Live_to_product_yield_kg_per_kg",
        "EF_aqua_N2O_kg_per_kg_median",
    ]:
        base.loc[base[col] <= 0, col] = np.nan
        if pd.notna(global_means.get(col)):
            base[col] = base[col].fillna(global_means.get(col))

    base["country"] = base["M49_Country_Code"].map(m49_to_country)

    # Historical emissions from panel
    hist = panel[panel["Year"].isin(hist_years)].copy()
    if not hist.empty:
        hist["ch4_kt"] = pd.to_numeric(hist["Aquaculture_CH4_emissions_kg_yr"], errors="coerce") / 1e6
        hist["n2o_kt"] = pd.to_numeric(hist["Aquaculture_N2O_emissions_kg_yr"], errors="coerce") / 1e6
        hist["co2_kt"] = 0.0
        hist["commodity"] = FISH_ITEM
        hist["process"] = FISH_PROCESS
        hist["country"] = hist["M49_Country_Code"].map(m49_to_country)
        hist = hist.rename(columns={"Year": "year"})
        hist_out = hist[["M49_Country_Code", "country", "year", "commodity", "process", "ch4_kt", "n2o_kt", "co2_kt"]]
    else:
        hist_out = pd.DataFrame(columns=["M49_Country_Code", "country", "year", "commodity", "process",
                                         "ch4_kt", "n2o_kt", "co2_kt"])

    # Future emissions from production + baseline params
    future_out = pd.DataFrame(columns=hist_out.columns)
    if future_years and production_df is not None and not production_df.empty:
        prod = production_df.copy()
        prod.columns = [str(c).strip() for c in prod.columns]

        c_item = _pick_col(list(prod.columns), ["commodity", "Commodity", "Item"])
        c_year = _pick_col(list(prod.columns), ["year", "Year"])
        c_prod = _pick_col(list(prod.columns), ["production_t", "Production_t", "production"])
        if c_item is None or c_year is None or c_prod is None:
            raise RuntimeError(f"Fish production data missing required columns: {list(prod.columns)}")

        if c_item != "commodity":
            prod = prod.rename(columns={c_item: "commodity"})
        if c_year != "year":
            prod = prod.rename(columns={c_year: "year"})
        if c_prod != "production_t":
            prod = prod.rename(columns={c_prod: "production_t"})

        if "M49_Country_Code" not in prod.columns:
            if "country" in prod.columns:
                prod["M49_Country_Code"] = prod["country"].map(country_to_m49)
            else:
                prod["M49_Country_Code"] = ""

        prod["M49_Country_Code"] = prod["M49_Country_Code"].apply(_norm_m49)
        prod = prod[prod["M49_Country_Code"] != ""].copy()
        prod["commodity"] = prod["commodity"].astype(str).str.strip()
        prod = prod[prod["commodity"].str.lower().isin(fish_items_lower)].copy()
        prod["year"] = pd.to_numeric(prod["year"], errors="coerce").astype("Int64")
        prod["production_t"] = pd.to_numeric(prod["production_t"], errors="coerce").fillna(0.0)
        prod = prod[prod["year"].isin(future_years)]

        if not prod.empty:
            prod = prod.merge(base, on="M49_Country_Code", how="left", suffixes=("", "_base"))
            if "country" not in prod.columns:
                prod["country"] = prod["M49_Country_Code"].map(m49_to_country)
            else:
                prod["country"] = prod["country"].fillna(prod["M49_Country_Code"].map(m49_to_country))

            def _row_country(r):
                c = r.get("country")
                return str(c).strip() if pd.notna(c) else ""

            def _apply_params(r):
                year = int(r.get("year", 0))
                if year <= hist_cutoff_year:
                    return r
                country = _row_country(r)
                m49 = r.get("M49_Country_Code")

                # Aquaculture share multiplier
                share_mult = _lookup_mult_by_country(
                    scenario_params.get("aquaculture_share_multiplier") if scenario_params else None,
                    country, FISH_ITEM, None, year, m49=m49
                )
                aq_share = float(r.get("Aquaculture_share", 0.0)) * share_mult
                aq_share = max(0.0, min(1.0, aq_share))

                # Yield multipliers
                ym = _lookup_mult_by_country(
                    scenario_params.get("yield_multiplier") if scenario_params else None,
                    country, FISH_ITEM, None, year, m49=m49
                )
                ym_aqua = _lookup_mult_by_country(
                    scenario_params.get("aquaculture_yield_multiplier") if scenario_params else None,
                    country, FISH_ITEM, None, year, m49=m49
                )
                ym_live = _lookup_mult_by_country(
                    scenario_params.get("live_to_product_yield_multiplier") if scenario_params else None,
                    country, FISH_ITEM, None, year, m49=m49
                )
                if ym_aqua == 1.0:
                    ym_aqua = ym
                if ym_live == 1.0:
                    ym_live = ym

                # EF multipliers / absolutes (GHG-aware)
                ef_mult_ch4 = _lookup_mult_by_country(
                    scenario_params.get("emission_factor_multiplier") if scenario_params else None,
                    country, FISH_ITEM, FISH_PROCESS, year, m49=m49, ghg="CH4"
                )
                ef_mult_n2o = _lookup_mult_by_country(
                    scenario_params.get("emission_factor_multiplier") if scenario_params else None,
                    country, FISH_ITEM, FISH_PROCESS, year, m49=m49, ghg="N2O"
                )
                ef_mc = _lookup_mult_by_country(
                    scenario_params.get("ef_multiplier_by") if scenario_params else None,
                    country, FISH_ITEM, FISH_PROCESS, year, m49=m49
                )
                base_ch4 = float(r.get("EF_aqua_CH4_kg_per_ha_yr_median", np.nan))
                base_n2o = float(r.get("EF_aqua_N2O_kg_per_kg_median", np.nan))
                ef_abs_ch4 = None
                ef_abs_n2o = None
                if scenario_params and scenario_params.get("emission_factor_bound_by"):
                    bound_ch4 = _lookup_bound_by_country(
                        scenario_params.get("emission_factor_bound_by"),
                        country, FISH_ITEM, FISH_PROCESS, "CH4", year, m49=m49
                    )
                    if bound_ch4 is not None and np.isfinite(base_ch4):
                        ef_abs_ch4 = _apply_ef_bound(base_ch4, bound_ch4)
                    bound_n2o = _lookup_bound_by_country(
                        scenario_params.get("emission_factor_bound_by"),
                        country, FISH_ITEM, FISH_PROCESS, "N2O", year, m49=m49
                    )
                    if bound_n2o is not None and np.isfinite(base_n2o):
                        ef_abs_n2o = _apply_ef_bound(base_n2o, bound_n2o)
                if ef_abs_ch4 is None:
                    ef_abs_ch4 = _lookup_abs_by_country(
                        scenario_params.get("emission_factor_absolute_by") if scenario_params else None,
                        country, FISH_ITEM, FISH_PROCESS, "CH4", year, m49=m49
                    )
                if ef_abs_n2o is None:
                    ef_abs_n2o = _lookup_abs_by_country(
                        scenario_params.get("emission_factor_absolute_by") if scenario_params else None,
                        country, FISH_ITEM, FISH_PROCESS, "N2O", year, m49=m49
                    )

                r["Aquaculture_share"] = aq_share
                r["Capture_share"] = 1.0 - aq_share
                r["Aquaculture_yield_t_per_ha"] = float(r.get("Aquaculture_yield_t_per_ha", np.nan)) * ym_aqua
                r["Live_to_product_yield_kg_per_kg"] = float(r.get("Live_to_product_yield_kg_per_kg", np.nan)) * ym_live
                if ef_abs_ch4 is not None:
                    r["EF_aqua_CH4_kg_per_ha_yr_median"] = ef_abs_ch4
                else:
                    r["EF_aqua_CH4_kg_per_ha_yr_median"] = base_ch4 * (ef_mult_ch4 * ef_mc)
                if ef_abs_n2o is not None:
                    r["EF_aqua_N2O_kg_per_kg_median"] = ef_abs_n2o
                else:
                    r["EF_aqua_N2O_kg_per_kg_median"] = base_n2o * (ef_mult_n2o * ef_mc)
                return r

            prod = prod.apply(_apply_params, axis=1)

            # Compute emissions
            aquaculture_prod_t = prod["production_t"] * prod["Aquaculture_share"]
            yield_t_per_ha = pd.to_numeric(prod["Aquaculture_yield_t_per_ha"], errors="coerce")
            area_ha = np.where(yield_t_per_ha > 0, aquaculture_prod_t / yield_t_per_ha, np.nan)
            ef_ch4 = pd.to_numeric(prod["EF_aqua_CH4_kg_per_ha_yr_median"], errors="coerce")
            ch4_kg = area_ha * ef_ch4
            ch4_kt = ch4_kg / 1e6

            live_yield = pd.to_numeric(prod["Live_to_product_yield_kg_per_kg"], errors="coerce")
            live_kg = np.where(live_yield > 0, aquaculture_prod_t * 1000.0 / live_yield, np.nan)
            ef_n2o = pd.to_numeric(prod["EF_aqua_N2O_kg_per_kg_median"], errors="coerce")
            n2o_kg = live_kg * ef_n2o
            n2o_kt = n2o_kg / 1e6

            future_out = pd.DataFrame({
                "M49_Country_Code": prod["M49_Country_Code"],
                "country": prod["country"],
                "year": prod["year"].astype(int),
                "commodity": FISH_ITEM,
                "process": FISH_PROCESS,
                "ch4_kt": ch4_kt.fillna(0.0),
                "n2o_kt": n2o_kt.fillna(0.0),
                "co2_kt": 0.0,
            })

    emissions = pd.concat([hist_out, future_out], ignore_index=True)
    return {
        "emissions": emissions,
        "parameters": {
            "baseline": base
        }
    }
