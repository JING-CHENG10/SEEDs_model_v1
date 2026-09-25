from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config_paths import get_input_base, get_src_base
from S2_0_load_data import DataPaths

BASE_YEAR = 2020
HIST_START = 2010
IQR_MULT = 2.5
FILTER_EXTREME_VALUES = False  # False to disable IQR-based outlier filtering


def _norm_m49(val: object) -> str:
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


def _safe_str(val: object) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def _extract_year(col: str) -> Optional[int]:
    m = re.search(r"Y(\d{4})$", str(col))
    if not m:
        return None
    return int(m.group(1))


def _load_emis_item(dict_path: Path) -> pd.DataFrame:
    df = pd.read_excel(dict_path, sheet_name="Emis_item")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _build_item_maps(emis_df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    map_exact: Dict[str, str] = {}
    map_lower: Dict[str, str] = {}
    feed_map: Dict[str, str] = {}

    cols = [
        "Item_Emis",
        "Item_Production_Map",
        "Item_Fertilizer_Map",
        "Item_Stock_Map",
        "Item_Yield_Map",
        "Item_Demand_Map",
        "Item_Slaughtered_Map",
    ]
    for _, row in emis_df.iterrows():
        item_emis = _safe_str(row.get("Item_Emis"))
        if not item_emis:
            continue
        for col in cols:
            raw = _safe_str(row.get(col))
            if not raw or raw.lower() in {"nan", "none", "no"}:
                continue
            map_exact.setdefault(raw, item_emis)
            map_lower.setdefault(raw.lower(), item_emis)
        feed_raw = _safe_str(row.get("Item_Feed_Map"))
        if feed_raw and feed_raw.lower() not in {"nan", "none", "no"}:
            feed_map.setdefault(feed_raw.lower(), item_emis)
    return map_exact, map_lower, feed_map


def _build_process_item_ghg_map(emis_df: pd.DataFrame) -> Dict[Tuple[str, str], List[str]]:
    if emis_df.empty or not {"Process", "Item_Emis", "GHG"}.issubset(set(emis_df.columns)):
        return {}
    tmp = emis_df[["Process", "Item_Emis", "GHG"]].copy()
    tmp["Process"] = tmp["Process"].apply(_safe_str)
    tmp["Item_Emis"] = tmp["Item_Emis"].apply(_safe_str)
    tmp["GHG"] = tmp["GHG"].apply(_safe_str)
    tmp = tmp[(tmp["Process"] != "") & (tmp["Item_Emis"] != "") & (tmp["GHG"] != "")]
    out: Dict[Tuple[str, str], List[str]] = {}
    for (proc, item), g in tmp.groupby(["Process", "Item_Emis"]):
        ghgs = sorted({_safe_str(x) for x in g["GHG"].tolist() if _safe_str(x)})
        if ghgs:
            out[(proc, item)] = ghgs
    return out


def _infer_ghg_from_text(text: object) -> Optional[str]:
    s = _safe_str(text).upper()
    if not s:
        return None
    if "CH4" in s:
        return "CH4"
    if "N2O" in s:
        return "N2O"
    if "CO2" in s:
        return "CO2"
    return None


def _resolve_ghg_list(process: str,
                      item_emis: str,
                      param_name: object,
                      units: object,
                      ghg_map: Dict[Tuple[str, str], List[str]]) -> List[str]:
    ghg = _infer_ghg_from_text(param_name) or _infer_ghg_from_text(units)
    if ghg:
        return [ghg]
    units_str = _safe_str(units).lower().replace(" ", "")
    if units_str:
        n_like = "kg/kgn" in units_str or units_str.endswith("/kgn") or ("kg/kg" in units_str and "n" in units_str)
        if n_like:
            return ["N2O"]
    key = (_safe_str(process), _safe_str(item_emis))
    ghgs = ghg_map.get(key, [])
    if len(ghgs) == 1:
        return ghgs
    if len(ghgs) > 1:
        return ghgs
    return ["Unknown"]


def _build_yield_unit_map(dict_path: Path, valid_items: set) -> Dict[str, str]:
    emis_df = _load_emis_item(dict_path)
    req = {"Item_Emis", "Item_Yield_Map", "Item_Yield_Unit"}
    if not req.issubset(set(emis_df.columns)):
        return {}
    df = emis_df[list(req)].dropna(subset=["Item_Yield_Map", "Item_Yield_Unit"]).copy()
    df["Item_Emis"] = df["Item_Emis"].astype(str).str.strip()
    if valid_items:
        df = df[df["Item_Emis"].isin(valid_items)]

    def _norm_unit(val: object) -> str:
        u = _safe_str(val).lower()
        if not u:
            return ""
        if "kg" in u and "ha" in u:
            return "t/ha"
        if "kg" in u and ("an" in u or "animal" in u):
            return "t/An"
        if "t" in u and "ha" in u:
            return "t/ha"
        if "t" in u and ("an" in u or "animal" in u):
            return "t/An"
        return _safe_str(val)

    df["unit_norm"] = df["Item_Yield_Unit"].apply(_norm_unit)
    unit_map: Dict[str, str] = {}
    for item, g in df.groupby("Item_Emis"):
        units = sorted({u for u in g["unit_norm"].tolist() if u})
        if not units:
            continue
        unit_map[item] = units[0] if len(units) == 1 else "|".join(units)
    return unit_map


def _map_item(
    raw: object,
    valid_items: set,
    map_exact: Dict[str, str],
    map_lower: Dict[str, str],
) -> Optional[str]:
    s = _safe_str(raw)
    if not s:
        return None
    if s in valid_items:
        return s
    if s in map_exact:
        return map_exact[s]
    s_lower = s.lower()
    if s_lower in map_lower:
        return map_lower[s_lower]
    return None


def _map_item_multi_base(raw: object, valid_items: set, map_exact: Dict[str, str], map_lower: Dict[str, str]) -> List[str]:
    s = _safe_str(raw)
    if not s:
        return []
    mapped = _map_item(s, valid_items, map_exact, map_lower)
    if mapped:
        return [mapped]
    base = s.lower()
    candidates = [item for item in valid_items if item.lower().startswith(base + ",")]
    return candidates


def _compute_stats(
    df: pd.DataFrame,
    group_cols: List[str],
    value_col: str,
    *,
    country_col: str = "M49_Country_Code",
    year_col: str = "year",
    base_group_cols: Optional[List[str]] = None,
    extra_cols: Optional[List[str]] = None,
    filter_outliers: Optional[bool] = None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    if filter_outliers is None:
        filter_outliers = FILTER_EXTREME_VALUES
    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col, year_col])
    df = df[df[value_col] > 0]
    df = df[df[year_col] <= BASE_YEAR]
    df = df[df[year_col] >= HIST_START]
    if df.empty:
        return pd.DataFrame()

    if base_group_cols is None:
        base_group_cols = list(group_cols) + [country_col]

    df = df.sort_values(base_group_cols + [year_col])
    base = df.groupby(base_group_cols, as_index=False).tail(1)
    base = base[base_group_cols + [value_col]].rename(columns={value_col: "base_value"})
    df = df.merge(base, on=base_group_cols, how="left")
    df = df[df["base_value"] > 0]
    df["mult"] = df[value_col] / df["base_value"]
    df = df[df["mult"] > 0]

    rows = []
    extra_cols = extra_cols or []
    for group, g in df.groupby(group_cols):
        if g.empty:
            continue
        g = g.copy()
        g = g[g["mult"] > 0]
        if g.empty:
            continue
        if filter_outliers:
            q1m = g["mult"].quantile(0.25)
            q3m = g["mult"].quantile(0.75)
            iqr_m = q3m - q1m
            if pd.notna(iqr_m) and iqr_m > 0:
                low_m = q1m - IQR_MULT * iqr_m
                high_m = q3m + IQR_MULT * iqr_m
            else:
                low_m = -np.inf
                high_m = np.inf
            if np.isfinite(low_m) or np.isfinite(high_m):
                min_m = g["mult"].min()
                max_m = g["mult"].max()
                if np.isfinite(min_m):
                    low_m = max(low_m, min_m)
                if np.isfinite(max_m):
                    high_m = min(high_m, max_m)

            q1v = g[value_col].quantile(0.25)
            q3v = g[value_col].quantile(0.75)
            iqr_v = q3v - q1v
            if pd.notna(iqr_v) and iqr_v > 0:
                low_v = q1v - IQR_MULT * iqr_v
                high_v = q3v + IQR_MULT * iqr_v
            else:
                low_v = -np.inf
                high_v = np.inf
            if np.isfinite(low_v) or np.isfinite(high_v):
                min_v = g[value_col].min()
                max_v = g[value_col].max()
                if np.isfinite(min_v):
                    low_v = max(low_v, min_v)
                if np.isfinite(max_v):
                    high_v = min(high_v, max_v)

            g = g[(g["mult"] >= low_m) & (g["mult"] <= high_m) & (g[value_col] >= low_v) & (g[value_col] <= high_v)]
            if g.empty:
                continue
        min_idx = g["mult"].idxmin()
        max_idx = g["mult"].idxmax()
        min_row = g.loc[min_idx]
        max_row = g.loc[max_idx]
        min_value = float(g[value_col].min())
        max_value = float(g[value_col].max())
        mean_value = float(g[value_col].mean()) if len(g) else np.nan
        data = {}
        if not isinstance(group, tuple):
            group = (group,)
        data.update(dict(zip(group_cols, group)))
        data.update(
            {
                "n_obs": int(len(g)),
                "min_mult": float(min_row["mult"]),
                "min_country": _safe_str(min_row.get(country_col)),
                "min_year": int(min_row.get(year_col)),
                "max_mult": float(max_row["mult"]),
                "max_country": _safe_str(max_row.get(country_col)),
                "max_year": int(max_row.get(year_col)),
                "mean_mult": float(g["mult"].mean()) if len(g) else np.nan,
                "min_value": min_value,
                "max_value": max_value,
                "mean_value": mean_value,
            }
        )
        for col in extra_cols:
            data[f"min_{col}"] = _safe_str(min_row.get(col))
            data[f"max_{col}"] = _safe_str(max_row.get(col))
        rows.append(data)
    return pd.DataFrame(rows)


def _load_yield_data(paths: DataPaths, dict_path: Path, valid_items: set) -> pd.DataFrame:
    emis_df = _load_emis_item(dict_path)
    req_cols = {"Item_Emis", "Item_Yield_Map", "Item_Yield_Element", "Item_Yield_Unit"}
    if not req_cols.issubset(set(emis_df.columns)):
        return pd.DataFrame()
    map_df = emis_df[list(req_cols)].dropna(subset=["Item_Yield_Map", "Item_Yield_Element", "Item_Yield_Unit"]).copy()
    map_df["Item_Emis"] = map_df["Item_Emis"].astype(str).str.strip()
    if valid_items:
        map_df = map_df[map_df["Item_Emis"].isin(valid_items)]
    map_df["item_key"] = map_df["Item_Yield_Map"].astype(str).str.strip().str.lower()
    map_df["element_key"] = map_df["Item_Yield_Element"].astype(str).str.strip().str.lower()
    map_df["unit_key"] = map_df["Item_Yield_Unit"].astype(str).str.strip().str.lower()
    map_df = map_df.drop_duplicates(subset=["item_key", "element_key", "unit_key", "Item_Emis"])
    if map_df.empty:
        return pd.DataFrame()

    year_cols = [f"Y{y}" for y in range(HIST_START, BASE_YEAR + 1)]
    usecols = ["M49_Country_Code", "Item", "Element", "Unit"] + year_cols
    if Path(paths.production_faostat_csv).exists():
        df = pd.read_csv(paths.production_faostat_csv, usecols=lambda c: c in set(usecols + ["Select"]))
    else:
        return pd.DataFrame()
    df.columns = [str(c).strip() for c in df.columns]
    if "Select" in df.columns:
        sel = pd.to_numeric(df["Select"], errors="coerce")
        df = df[sel == 1]
    if not set(["M49_Country_Code", "Item", "Element", "Unit"]).issubset(set(df.columns)):
        return pd.DataFrame()
    id_cols = [c for c in df.columns if c not in year_cols]
    if not year_cols:
        return pd.DataFrame()
    long_df = df.melt(id_vars=id_cols, value_vars=year_cols, var_name="year", value_name="value")
    long_df["year"] = pd.to_numeric(long_df["year"].astype(str).str.lstrip("Y"), errors="coerce")
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["year", "value"])
    long_df["year"] = long_df["year"].astype(int)
    long_df = long_df[(long_df["year"] >= HIST_START) & (long_df["year"] <= BASE_YEAR)]
    if long_df.empty:
        return pd.DataFrame()

    long_df["item_key"] = long_df["Item"].astype(str).str.strip().str.lower()
    long_df["element_key"] = long_df["Element"].astype(str).str.strip().str.lower()
    long_df["unit_key"] = long_df["Unit"].astype(str).str.strip().str.lower()
    long_df = long_df.merge(
        map_df[["item_key", "element_key", "unit_key", "Item_Emis"]],
        on=["item_key", "element_key", "unit_key"],
        how="inner",
    )
    if long_df.empty:
        return pd.DataFrame()
    long_df["M49_Country_Code"] = long_df["M49_Country_Code"].apply(_norm_m49)

    mask_kg_ha = long_df["unit_key"] == "kg/ha"
    mask_kg_an = long_df["unit_key"] == "kg/an"
    if mask_kg_ha.any() or mask_kg_an.any():
        long_df.loc[mask_kg_ha | mask_kg_an, "value"] = long_df.loc[mask_kg_ha | mask_kg_an, "value"] / 1000.0

    out = long_df.groupby(["M49_Country_Code", "year", "Item_Emis"], as_index=False)["value"].mean()
    return out[["M49_Country_Code", "year", "Item_Emis", "value"]]


def _load_fertilizer_data(paths: DataPaths, valid_items: set, map_exact: Dict[str, str], map_lower: Dict[str, str]) -> pd.DataFrame:
    df = pd.read_excel(paths.fertilizer_efficiency_xlsx, sheet_name="data")
    df.columns = [str(c).strip() for c in df.columns]
    year_cols = [c for c in df.columns if _extract_year(c) is not None and str(c).startswith("N_FertEffi_")]
    if not year_cols:
        return pd.DataFrame()
    records = []
    for col in year_cols:
        year = _extract_year(col)
        if year is None or year < HIST_START or year > BASE_YEAR:
            continue
        tmp = df[["M49_Country_Code", "Item", col]].copy()
        tmp = tmp.rename(columns={col: "value"})
        tmp["year"] = year
        records.append(tmp)
    if not records:
        return pd.DataFrame()
    out = pd.concat(records, ignore_index=True)
    out["M49_Country_Code"] = out["M49_Country_Code"].apply(_norm_m49)
    out["Item_Emis"] = out["Item"].apply(lambda x: _map_item(x, valid_items, map_exact, map_lower))
    out = out.dropna(subset=["Item_Emis"])
    return out[["M49_Country_Code", "year", "Item_Emis", "value"]]


def _load_emission_factor_data(dict_path: Path, valid_items: set, valid_processes: set,
                               map_exact: Dict[str, str], map_lower: Dict[str, str],
                               ghg_map: Dict[Tuple[str, str], List[str]]) -> pd.DataFrame:
    files = [
        (dict_path.parent / "GCE_parameters.xlsx", "GCE_parameters"),
        (dict_path.parent / "GLE_parameters.xlsx", "GLE_para"),
        (dict_path.parent / "Soil_parameters.xlsx", "Soil_parameter"),
    ]
    frames = []
    for path, sheet in files:
        if not path.exists():
            continue
        df = pd.read_excel(path, sheet_name=sheet)
        df.columns = [str(c).strip() for c in df.columns]
        if "paramName" not in df.columns:
            continue
        if "Select" in df.columns:
            sel = pd.to_numeric(df["Select"], errors="coerce")
            df = df[sel == 1]
        df["paramName"] = df["paramName"].astype(str).str.strip()
        mask = df["paramName"].str.contains("emission factor", case=False, na=False) | df["paramName"].str.match(
            r"(?i)^ef\\b|^ef_"
        )
        df = df[mask].copy()
        if df.empty:
            continue
        year_cols = [c for c in df.columns if _extract_year(c) is not None and str(c).startswith("Y")]
        if not year_cols:
            continue
        records = []
        for col in year_cols:
            year = _extract_year(col)
            if year is None or year < HIST_START or year > BASE_YEAR:
                continue
            cols = ["M49_Country_Code", "Item", "Process", "paramName", col]
            if "units" in df.columns:
                cols.append("units")
            tmp = df[cols].copy()
            tmp = tmp.rename(columns={col: "value"})
            tmp["year"] = year
            records.append(tmp)
        if not records:
            continue
        out = pd.concat(records, ignore_index=True)
        out["M49_Country_Code"] = out["M49_Country_Code"].apply(_norm_m49)
        out["Item_Emis"] = out["Item"].apply(lambda x: _map_item(x, valid_items, map_exact, map_lower))
        out["Process"] = out["Process"].astype(str).str.strip()
        out = out.dropna(subset=["Item_Emis"])
        if valid_processes:
            out = out[out["Process"].isin(valid_processes)]
        out["GHG_list"] = out.apply(
            lambda r: _resolve_ghg_list(
                r.get("Process"),
                r.get("Item_Emis"),
                r.get("paramName"),
                r.get("units"),
                ghg_map,
            ),
            axis=1,
        )
        expanded = []
        for _, row in out.iterrows():
            ghgs = row.get("GHG_list") or ["Unknown"]
            for ghg in ghgs:
                expanded.append(
                    {
                        "M49_Country_Code": row.get("M49_Country_Code"),
                        "year": row.get("year"),
                        "Item_Emis": row.get("Item_Emis"),
                        "Process": row.get("Process"),
                        "GHG": ghg,
                        "paramName": row.get("paramName"),
                        "units": row.get("units"),
                        "value": row.get("value"),
                    }
                )
        if expanded:
            frames.append(pd.DataFrame(expanded))
    panel_path = Path(get_input_base()) / "Aquaculture" / "fish_seafood_country_panel_2000_present.xlsx"
    if panel_path.exists():
        panel = pd.read_excel(panel_path, sheet_name="country_year_panel")
        panel.columns = [str(c).strip() for c in panel.columns]
        if "year" in panel.columns and "Year" not in panel.columns:
            panel = panel.rename(columns={"year": "Year"})
        ef_cols = [
            "EF_aqua_CH4_kg_per_ha_yr_median",
            "EF_aqua_N2O_kg_per_kg_median",
        ]
        if {"M49_Country_Code", "Year"}.issubset(set(panel.columns)) and any(c in panel.columns for c in ef_cols):
            panel["M49_Country_Code"] = panel["M49_Country_Code"].apply(_norm_m49)
            panel["year"] = pd.to_numeric(panel["Year"], errors="coerce").astype("Int64")
            panel = panel[(panel["year"] >= HIST_START) & (panel["year"] <= BASE_YEAR)]
            for col in ef_cols:
                if col not in panel.columns:
                    continue
                tmp = panel[["M49_Country_Code", "year", col]].copy()
                tmp = tmp.rename(columns={col: "value"})
                tmp["paramName"] = col
                tmp["Process"] = "Fish farming"
                tmp["units"] = ""
                tmp = tmp.dropna(subset=["value"])
                emis_df = _load_emis_item(dict_path)
                fish_items = emis_df[emis_df["Process"].astype(str).str.strip() == "Fish farming"]["Item_Emis"].dropna().unique()
                fish_items = [i for i in fish_items if _safe_str(i)]
                if not fish_items:
                    fish_items = ["Fish, Seafood"]
                for item in fish_items:
                    if item not in valid_items:
                        continue
                    tmp2 = tmp.copy()
                    tmp2["Item_Emis"] = item
                    tmp2["GHG_list"] = tmp2.apply(
                        lambda r: _resolve_ghg_list(
                            r.get("Process"),
                            r.get("Item_Emis"),
                            r.get("paramName"),
                            r.get("units"),
                            ghg_map,
                        ),
                        axis=1,
                    )
                    expanded = []
                    for _, row in tmp2.iterrows():
                        ghgs = row.get("GHG_list") or ["Unknown"]
                        for ghg in ghgs:
                            expanded.append(
                                {
                                    "M49_Country_Code": row.get("M49_Country_Code"),
                                    "year": row.get("year"),
                                    "Item_Emis": row.get("Item_Emis"),
                                    "Process": row.get("Process"),
                                    "GHG": ghg,
                                    "paramName": row.get("paramName"),
                                    "units": row.get("units"),
                                    "value": row.get("value"),
                                }
                            )
                    if expanded:
                        frames.append(pd.DataFrame(expanded))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_dm_conversion_data(paths: DataPaths,
                             valid_items: set,
                             feed_map: Dict[str, str],
                             *,
                             scheme: str) -> pd.DataFrame:
    xls = pd.ExcelFile(paths.feed_need_xlsx)
    scheme_norm = str(scheme or "").strip().lower()
    if scheme_norm == "ipcc":
        sheet = "total_kgDM_per_head_IPCC"
    else:
        sheet = "total_kgDM_per_head_GLEAM"
    if sheet not in xls.sheet_names:
        return pd.DataFrame()
    df = pd.read_excel(paths.feed_need_xlsx, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    records = []
    if "year" in df.columns and "total_kgDM_per_head" in df.columns:
        tmp = df.rename(columns={"total_kgDM_per_head": "value", "Species": "species"})
        tmp["year"] = pd.to_numeric(tmp["year"], errors="coerce").astype("Int64")
        tmp = tmp.dropna(subset=["year"])
        tmp = tmp[(tmp["year"] >= HIST_START) & (tmp["year"] <= BASE_YEAR)]
        tmp["M49_Country_Code"] = tmp["M49_Country_Code"].apply(_norm_m49)
        tmp["Item_Emis"] = tmp["species"].apply(lambda x: feed_map.get(_safe_str(x).lower(), None))
        tmp = tmp.dropna(subset=["Item_Emis"])
        return tmp[["M49_Country_Code", "year", "Item_Emis", "value"]]

    year_cols = [c for c in df.columns if _extract_year(c) is not None and str(c).startswith("Y")]
    for col in year_cols:
        year = _extract_year(col)
        if year is None or year < HIST_START or year > BASE_YEAR:
            continue
        tmp = df[["M49_Country_Code", "Species", col]].copy()
        tmp = tmp.rename(columns={"Species": "species", col: "value"})
        tmp["year"] = year
        records.append(tmp)
    if not records:
        return pd.DataFrame()
    out = pd.concat(records, ignore_index=True)
    out["M49_Country_Code"] = out["M49_Country_Code"].apply(_norm_m49)
    out["Item_Emis"] = out["species"].apply(lambda x: feed_map.get(_safe_str(x).lower(), None))
    out = out.dropna(subset=["Item_Emis"])
    out = out[out["Item_Emis"].isin(valid_items)]
    return out[["M49_Country_Code", "year", "Item_Emis", "value"]]


def _load_manure_ratio_data(paths: DataPaths, valid_items: set,
                            map_exact: Dict[str, str], map_lower: Dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(paths.manure_stock_with_ratio_csv)
    df.columns = [str(c).strip() for c in df.columns]
    if "Element" not in df.columns:
        return pd.DataFrame()
    df = df[df["Element"].astype(str).str.strip().str.lower() == "manure management ratio"].copy()
    year_cols = [c for c in df.columns if _extract_year(c) is not None and str(c).startswith("Y")]
    records = []
    for col in year_cols:
        year = _extract_year(col)
        if year is None or year < HIST_START or year > BASE_YEAR:
            continue
        tmp = df[["M49_Country_Code", "Item", col]].copy()
        tmp = tmp.rename(columns={col: "value"})
        tmp["year"] = year
        records.append(tmp)
    if not records:
        return pd.DataFrame()
    out = pd.concat(records, ignore_index=True)
    out["M49_Country_Code"] = out["M49_Country_Code"].apply(_norm_m49)
    expanded_rows = []
    for _, row in out.iterrows():
        items = _map_item_multi_base(row.get("Item"), valid_items, map_exact, map_lower)
        for item in items:
            expanded_rows.append(
                {
                    "M49_Country_Code": row.get("M49_Country_Code"),
                    "year": row.get("year"),
                    "Item_Emis": item,
                    "value": row.get("value"),
                }
            )
    if not expanded_rows:
        return pd.DataFrame()
    return pd.DataFrame(expanded_rows)


def _load_aquaculture_share_data(dict_path: Path, valid_items: set) -> pd.DataFrame:
    panel_path = Path(get_input_base()) / "Aquaculture" / "fish_seafood_country_panel_2000_present.xlsx"
    if not panel_path.exists():
        return pd.DataFrame()
    panel = pd.read_excel(panel_path, sheet_name="country_year_panel")
    panel.columns = [str(c).strip() for c in panel.columns]
    if "year" in panel.columns and "Year" not in panel.columns:
        panel = panel.rename(columns={"year": "Year"})
    req = {"M49_Country_Code", "Year", "Aquaculture_share"}
    if not req.issubset(set(panel.columns)):
        return pd.DataFrame()
    panel["M49_Country_Code"] = panel["M49_Country_Code"].apply(_norm_m49)
    panel["year"] = pd.to_numeric(panel["Year"], errors="coerce").astype("Int64")
    panel["value"] = pd.to_numeric(panel["Aquaculture_share"], errors="coerce")
    panel = panel.dropna(subset=["M49_Country_Code", "year", "value"])
    panel = panel[(panel["year"] >= HIST_START) & (panel["year"] <= BASE_YEAR)]

    emis_df = _load_emis_item(dict_path)
    fish_items = emis_df[emis_df["Process"].astype(str).str.strip() == "Fish farming"]["Item_Emis"].dropna().unique()
    fish_items = [i for i in fish_items if _safe_str(i)]
    if not fish_items:
        fish_items = ["Fish, Seafood"]
    rows = []
    for item in fish_items:
        if item not in valid_items:
            continue
        tmp = panel[["M49_Country_Code", "year", "value"]].copy()
        tmp["Item_Emis"] = item
        rows.append(tmp)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _load_waste_ratio_data(dict_path: Path,
                           valid_items: set,
                           map_exact: Dict[str, str],
                           map_lower: Dict[str, str]) -> pd.DataFrame:
    ratio_path = Path(get_input_base()) / "Production_Trade" / "Demand_composition.xlsx"
    if not ratio_path.exists():
        return pd.DataFrame()
    df = pd.read_excel(ratio_path, sheet_name="ratio")
    df.columns = [str(c).strip() for c in df.columns]
    if "Element" not in df.columns or "Item" not in df.columns or "M49_Country_Code" not in df.columns:
        return pd.DataFrame()
    df = df[df["Element"].astype(str).str.strip().str.lower() == "losses"].copy()
    if df.empty:
        return pd.DataFrame()
    year_cols = [c for c in df.columns if _extract_year(c) is not None and str(c).startswith("Y")]
    records = []
    for col in year_cols:
        year = _extract_year(col)
        if year is None or year < HIST_START or year > BASE_YEAR:
            continue
        tmp = df[["M49_Country_Code", "Item", col]].copy()
        tmp = tmp.rename(columns={col: "value"})
        tmp["year"] = year
        records.append(tmp)
    if not records:
        return pd.DataFrame()
    out = pd.concat(records, ignore_index=True)
    out["M49_Country_Code"] = out["M49_Country_Code"].apply(_norm_m49)
    out["Item_Emis"] = out["Item"].apply(lambda x: _map_item(x, valid_items, map_exact, map_lower))
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["Item_Emis", "M49_Country_Code", "year", "value"])
    out = out.groupby(["M49_Country_Code", "year", "Item_Emis"], as_index=False)["value"].mean()
    return out[["M49_Country_Code", "year", "Item_Emis", "value"]]


def main() -> None:
    src_base = Path(get_src_base())
    dict_path = src_base / "dict_v3.xlsx"
    if not dict_path.exists():
        raise FileNotFoundError(f"dict_v3.xlsx not found: {dict_path}")

    emis_df = _load_emis_item(dict_path)
    valid_items = set(emis_df["Item_Emis"].dropna().astype(str).str.strip())
    valid_processes = set(emis_df["Process"].dropna().astype(str).str.strip())
    map_exact, map_lower, feed_map = _build_item_maps(emis_df)
    ghg_map = _build_process_item_ghg_map(emis_df)

    paths = DataPaths()

    yield_df = _load_yield_data(paths, dict_path, valid_items)
    yield_stats = _compute_stats(
        yield_df,
        ["Item_Emis"],
        "value",
        country_col="M49_Country_Code",
        year_col="year",
    )
    if not yield_stats.empty:
        yield_unit_map = _build_yield_unit_map(dict_path, valid_items)
        yield_stats["yield_unit"] = yield_stats["Item_Emis"].map(yield_unit_map)

    fert_df = _load_fertilizer_data(paths, valid_items, map_exact, map_lower)
    fert_stats = _compute_stats(
        fert_df,
        ["Item_Emis"],
        "value",
        country_col="M49_Country_Code",
        year_col="year",
    )

    ef_df = _load_emission_factor_data(dict_path, valid_items, valid_processes, map_exact, map_lower, ghg_map)
    ef_stats = _compute_stats(
        ef_df,
        ["Item_Emis", "Process", "GHG"],
        "value",
        country_col="M49_Country_Code",
        year_col="year",
        base_group_cols=["M49_Country_Code", "Item_Emis", "Process", "GHG", "paramName"],
        extra_cols=["paramName", "units"],
    )

    dm_ipcc_df = _load_dm_conversion_data(paths, valid_items, feed_map, scheme="IPCC")
    dm_ipcc_stats = _compute_stats(
        dm_ipcc_df,
        ["Item_Emis"],
        "value",
        country_col="M49_Country_Code",
        year_col="year",
    )
    dm_gleam_df = _load_dm_conversion_data(paths, valid_items, feed_map, scheme="GLEAM")
    dm_gleam_stats = _compute_stats(
        dm_gleam_df,
        ["Item_Emis"],
        "value",
        country_col="M49_Country_Code",
        year_col="year",
    )

    mm_df = _load_manure_ratio_data(paths, valid_items, map_exact, map_lower)
    mm_stats = _compute_stats(
        mm_df,
        ["Item_Emis"],
        "value",
        country_col="M49_Country_Code",
        year_col="year",
    )

    aq_df = _load_aquaculture_share_data(dict_path, valid_items)
    aq_stats = _compute_stats(
        aq_df,
        ["Item_Emis"],
        "value",
        country_col="M49_Country_Code",
        year_col="year",
    )

    waste_df = _load_waste_ratio_data(dict_path, valid_items, map_exact, map_lower)
    waste_stats = _compute_stats(
        waste_df,
        ["Item_Emis"],
        "value",
        country_col="M49_Country_Code",
        year_col="year",
    )

    out_path = src_base / "Scenario_variable_historical_range.xlsx"
    with pd.ExcelWriter(out_path) as writer:
        if not yield_stats.empty:
            yield_stats.to_excel(writer, sheet_name="yield_multiplier", index=False)
        if not fert_stats.empty:
            fert_stats.to_excel(writer, sheet_name="fertilizer_rate_multiplier", index=False)
        if not ef_stats.empty:
            ef_stats.to_excel(writer, sheet_name="emission_factor_multiplier", index=False)
        if not dm_ipcc_stats.empty:
            dm_ipcc_stats.to_excel(writer, sheet_name="dm_conversion_multiplier_IPCC", index=False)
        if not dm_gleam_stats.empty:
            dm_gleam_stats.to_excel(writer, sheet_name="dm_conversion_multiplier_GLEAM", index=False)
        if not mm_stats.empty:
            mm_stats.to_excel(writer, sheet_name="manure_mgmt_ratio_mult", index=False)
        if not aq_stats.empty:
            aq_stats.to_excel(writer, sheet_name="aquaculture_share_multiplier", index=False)
        if not waste_stats.empty:
            waste_stats.to_excel(writer, sheet_name="waste_reduction_multiplier", index=False)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
