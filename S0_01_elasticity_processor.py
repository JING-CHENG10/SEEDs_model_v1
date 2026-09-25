# -*- coding: utf-8 -*-
"""
S0.5 Elasticity Processor: many-to-many mapping, wide outputs only, fully documented.
================================================================
Objectives:
  Generate wide elasticity tables by country, target commodity (Commodity Map/All), and six Element categories.
  Cross-price columns are fixed to the Commodity All allowlist in Item map;
  initialize every country-commodity cross-price vector to zero, then overwrite with observations.

Sources in Elasticity_v3.xlsx:
  - raw_all: Original entries with Commodity, Cross_price_item, Element Map, D/S, Region, Selected, Elasticity.
  - Item map: Commodity Map, Commodity Raw, Commodity All; optional Cross_price_item Map/Raw.
  - Region map: Region_label_new, Region_Elasticity_v3, Region_Elasticity_keep.

Core rules:
  1) Retain raw_all records with Selected==1 only.
  2) Use many-to-many Map-Raw mappings:
     - forward: map_token_lc -> {original raw strings, ...}.
     - reverse:  raw_lc        -> {map_token_lc, ...}
     Semicolon-separated Map tokens may share one Raw; a Map token may appear in several rows and map to multiple Raw values.
  3) Classify six Elements strictly by Element Map + D/S, without using Cross_price_item.
     Demand-Cross-Price / Demand-Income / Demand-Own-Price /
     Supply-Cross-Price / Supply-Own-Price / Supply-Temperature
  4) Three fallback levels:
     Country (Region_Elasticity_v3) -> region (Region_Elasticity_keep) -> countries in the same keep group (peers).
     - Non-cross-price: aggregate raw values/regions with n-weighted means, minimum min, maximum max, and summed n.
     - Cross-price: aggregate (main_raw, cross_raw) by country/region/peer, then reverse-map cross_raw to target tokens.
             Combine multiple sources per target token using the same n-weighting rules; output Commodity All columns with missing values zero.
  5) Outputs:
     - Demand/Supply Cross: four matrices each (mean/min/max/n), indexed by Country/Commodity, with Commodity All columns.
     - Four non-cross categories: one table each with Elasticity_mean/min/max/n and source_level.

"""

from __future__ import annotations
import pandas as pd
import numpy as np
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

INPUT_XLSX  = Path("../../src/bakup/Elasticity_v3.xlsx")
OUTPUT_XLSX = Path("../../src/bakup/Elasticity_v3_processed_out3.1.xlsx")


# Text normalization helpers

def norm(s):
    """Trim edges, normalize dashes, and collapse repeated spaces; preserve display case."""
    if pd.isna(s):
        return np.nan
    s = str(s).strip().replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    return s

def canon(s):
    """Apply norm, lowercase, replace underscores/hyphens with spaces, and trim.
    Use for matching independent of case, hyphens, and repeated spaces."""
    if pd.isna(s):
        return np.nan
    s0 = norm(s).lower().replace("_", " ").replace("-", " ").strip()
    return s0

def lc(x):
    """Lowercase and trim strings only; return other values unchanged."""
    return x.lower().strip() if isinstance(x, str) else x

def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find candidate columns ignoring case/hyphen/spacing differences; return the actual name."""
    cmap = {canon(c): c for c in df.columns}
    for cand in candidates:
        key = canon(cand)
        if key in cmap:
            return cmap[key]
    return None


# Many-to-many Map-Raw mappings

def build_maps_many_to_many(df_map: pd.DataFrame, map_col: str, raw_col: str):
    """Build from Item map:
       forward: {map_token_lc -> set(original raw strings)}.
       reverse: { raw_lc        -> set(map_token_lc) }
       - Map supports semicolon-separated values.
       - Lowercase keys enable case-insensitive matching.
       - Preserve Raw strings for review and export."""
    forward: Dict[str, Set[str]] = defaultdict(set)
    reverse: Dict[str, Set[str]] = defaultdict(set)
    # Read each mapping row; Commodity Map may contain 'a;b;c'.
    for _, r in df_map[[map_col, raw_col]].dropna(how="any").iterrows():
        raw_v = norm(r[raw_col])     # Clean without changing case.
        raw_lc = lc(raw_v)           # Lowercase key
        for tok in str(r[map_col]).split(";"):
            token = norm(tok)
            if not token:
                continue
            token_lc = lc(token)     # Target token as a lowercase key
            forward[token_lc].add(raw_v)
            reverse[raw_lc].add(token_lc)
    return forward, reverse


# Element classification strictly from Element Map + D/S

def classify_element_ds(element_text: Optional[str], ds_text: Optional[str]) -> Tuple[str, str]:
    """Assign one of six Element categories; Cross_price_item is used only later for cross mappings."""
    # Distinguish Demand/Supply, tolerating abbreviations, case, and spaces.
    ds = (str(ds_text).strip().title() if pd.notna(ds_text) else "Demand")
    ds = "Supply" if ds.lower().startswith("sup") else "Demand"
    # Normalize Element Map text.
    em = canon(element_text)
    # Prefer exact names, allowing case/hyphen differences.
    canonical = {
        "demand cross price":      ("Demand-Cross-Price", "Demand"),
        "demand income":           ("Demand-Income",      "Demand"),
        "demand own price":        ("Demand-Own-Price",   "Demand"),
        "supply cross price":      ("Supply-Cross-Price", "Supply"),
        "supply own price":        ("Supply-Own-Price",   "Supply"),
        "supply temperature":      ("Supply-Temperature", "Supply"),
    }
    if em in canonical:
        return canonical[em]
    # Keyword fallback for nonstandard Element Map values
    if em is not np.nan:
        if "temperature" in em or "temp" in em:
            return ("Supply-Temperature", "Supply")
        if "income" in em:
            return ("Demand-Income", "Demand")
        if "cross" in em:
            return (f"{ds}-Cross-Price", ds)
        if "own" in em or "own price" in em or "price elasticity" in em or "price" in em:
            return (f"{ds}-Own-Price", ds)
    # Final fallback: classify as Own-Price using D/S.
    return (f"{ds}-Own-Price", ds)


# Main workflow

# Read and normalize.
xls = pd.ExcelFile(INPUT_XLSX)
raw_all    = pd.read_excel(xls, sheet_name="raw_all")
item_map   = pd.read_excel(xls, sheet_name="Item map")
region_map = pd.read_excel(xls, sheet_name="Region map")

# Trim column names to avoid hidden mismatches.
raw_all.columns    = [c.strip() for c in raw_all.columns]
item_map.columns   = [c.strip() for c in item_map.columns]
region_map.columns = [c.strip() for c in region_map.columns]

# Standardize columns and convert numeric values.
for c in ["Commodity", "Element", "Element Map", "Cross_price_item", "D/S", "Region", "Selected", "Elasticity"]:
    if c in raw_all.columns:
        if c in ["Selected", "Elasticity"]:
            raw_all[c] = pd.to_numeric(raw_all[c], errors="coerce")
        else:
            raw_all[c] = raw_all[c].map(norm)

# Retain Selected==1 and classify Elements.
raw_sel = raw_all[raw_all["Selected"] == 1].copy()
raw_sel["Element_Class"], raw_sel["DS_Class"] = zip(*raw_sel.apply(
    lambda r: classify_element_ds(r.get("Element Map"), r.get("D/S")), axis=1
))

# Item map column identification and allowlist
cm_map_col  = find_col(item_map, ["Commodity Map"])
cm_raw_col  = find_col(item_map, ["Commodity Raw"])
cm_all_col  = find_col(item_map, ["Commodity All"])
if not (cm_map_col and cm_raw_col and cm_all_col):
    raise ValueError("Item map 缺少 Commodity Map/Raw/All 列")

# Target commodity allowlist supplies column names and target tokens.
target_all: List[str] = [norm(x) for x in item_map[cm_all_col].dropna().unique().tolist()]
target_all_lc = set([x.lower() for x in target_all])

# Build the main Commodity many-to-many mapping.
commodity_fwd_m2m, commodity_rev_m2m = build_maps_many_to_many(item_map, cm_map_col, cm_raw_col)

# Prefer dedicated Cross_price_item Map/Raw for cross mappings; otherwise use commodity_rev_m2m.
cpi_map_col = find_col(item_map, ["Cross_price_item Map", "Cross item Map", "Cross-price item Map", "Cross price item Map"])
cpi_raw_col = find_col(item_map, ["Cross_price_item Raw", "Cross item Raw", "Cross-price item Raw", "Cross price item Raw"])
if cpi_map_col and cpi_raw_col:
    cross_fwd_m2m, cross_rev_m2m = build_maps_many_to_many(item_map, cpi_map_col, cpi_raw_col)
else:
    cross_fwd_m2m, cross_rev_m2m = {}, {}

# Build the Region map fallback chain.
rl_col   = find_col(region_map, ["Region_label_new"])
v3_col   = find_col(region_map, ["Region_Elasticity_v3"])
keep_col = find_col(region_map, ["Region_Elasticity_keep"])
if not rl_col:
    raise ValueError("Region map 缺少 Region_label_new 列")

region_map[rl_col]   = region_map[rl_col].map(norm)
if v3_col:
    region_map[v3_col]   = region_map[v3_col].map(norm)
if keep_col:
    region_map[keep_col] = region_map[keep_col].map(norm)

# Country -> v3 region; country -> keep group.
country_to_region: Dict[str, str] = {}
country_to_keep  : Dict[str, Optional[str]] = {}
for _, r in region_map.iterrows():
    cty = r.get(rl_col)
    v3  = r.get(v3_col) or cty if v3_col else cty
    kp  = r.get(keep_col) if keep_col else None
    if pd.notna(cty):
        country_to_region[cty] = v3
        country_to_keep[cty]   = kp

# Output countries are defined by Region map.
target_countries = [x for x in region_map[rl_col].dropna().unique().tolist()]

# keep group -> unique v3 regions for peer fallback across countries in the same group.
group_to_regionlist: Dict[str, List[str]] = defaultdict(list)
for cty, grp in country_to_keep.items():
    reg = country_to_region.get(cty)
    if grp and reg:
        group_to_regionlist[grp].append(reg)
for grp in list(group_to_regionlist.keys()):
    group_to_regionlist[grp] = sorted(set(group_to_regionlist[grp]))

# Preaggregate raw values by Element, main_raw, and Region.
# Non-cross: dict[(ec, main_raw_lc, region_lc)] -> (mean, min, max, n).
non_cross_by_key: Dict[Tuple[str, str, str], Tuple[float, float, float, int]] = {}
for (ec, com, reg), grp in raw_sel.groupby(["Element_Class", "Commodity", "Region"]):
    vals = pd.to_numeric(grp["Elasticity"], errors="coerce").dropna()
    if len(vals):
        non_cross_by_key[(ec, lc(com), lc(reg))] = (float(vals.mean()), float(vals.min()), float(vals.max()), int(vals.shape[0]))

# Cross: dict[(ec, main_raw_lc, region_lc)] -> {cross_raw_lc: (mean, min, max, n)}.
cross_by_key: Dict[Tuple[str, str, str], Dict[str, Tuple[float, float, float, int]]] = defaultdict(dict)
if "Cross_price_item" in raw_sel.columns:
    for (ec, com, reg, cp), grp in raw_sel.groupby(["Element_Class", "Commodity", "Region", "Cross_price_item"]):
        vals = pd.to_numeric(grp["Elasticity"], errors="coerce").dropna()
        if len(vals):
            cross_by_key[(ec, lc(com), lc(reg))][lc(cp)] = (float(vals.mean()), float(vals.min()), float(vals.max()), int(vals.shape[0]))

# Aggregation and retrieval helpers
def agg_stats(stats_list):
    """Aggregate mean/min/max/n using n-weighted means, minimum min, maximum max, and summed n."""
    if not stats_list:
        return None
    m_sum = 0.0; n_sum = 0; mi = np.inf; ma = -np.inf
    for (mean_v, min_v, max_v, n_v) in stats_list:
        m_sum += mean_v * n_v
        n_sum += n_v
        mi = min(mi, min_v)
        ma = max(ma, max_v)
    if n_sum == 0:
        return None
    mean_w = m_sum / n_sum
    if mi == np.inf:    mi = mean_w
    if ma == -np.inf:   ma = mean_w
    return (mean_w, mi, ma, int(n_sum))

def fetch_non_cross(ec: str, main_raws: Set[str], region: Optional[str] = None,
                    keep_grp: Optional[str] = None, scope: str = "country"):
    """Aggregate non-cross main_raws entries within country/region/peer scope and return statistics."""
    if scope == "country" and region:
        reg_candidates = [lc(region)]
    elif scope == "region" and keep_grp:
        reg_candidates = [lc(keep_grp)]
    elif scope == "peer" and keep_grp:
        reg_candidates = [lc(r) for r in group_to_regionlist.get(keep_grp, [])]
    else:
        reg_candidates = []
    stats = []
    for reg_lc in reg_candidates:
        for raw_name in main_raws:
            st = non_cross_by_key.get((ec, lc(raw_name), reg_lc))
            if st:
                stats.append(st)
    return agg_stats(stats)

def fetch_cross_dict(ec: str, main_raws: Set[str], region: Optional[str] = None,
                     keep_grp: Optional[str] = None, scope: str = "country"):
    """Collect main_raw/cross_raw pairs within country/region/peer scope and aggregate by cross_raw."""
    if scope == "country" and region:
        reg_candidates = [lc(region)]
    elif scope == "region" and keep_grp:
        reg_candidates = [lc(keep_grp)]
    elif scope == "peer" and keep_grp:
        reg_candidates = [lc(r) for r in group_to_regionlist.get(keep_grp, [])]
    else:
        reg_candidates = []
    bucket: Dict[str, List[Tuple[float, float, float, int]]] = defaultdict(list)
    for reg_lc in reg_candidates:
        for raw_name in main_raws:
            dd = cross_by_key.get((ec, lc(raw_name), reg_lc), {})
            for cp_raw_lc, st in dd.items():
                bucket[cp_raw_lc].append(st)
    # Return {cross_raw_lc: (mean, min, max, n)}, already combined over main_raw.
    out: Dict[str, Tuple[float, float, float, int]] = {}
    for cp_raw_lc, lst in bucket.items():
        st = agg_stats(lst)
        if st:
            out[cp_raw_lc] = st
    return out

# Output containers: cross matrices and non-cross tables
# Collect wide rows for mean/min/max/n cross matrices before writing the workbook.
cross_rows_by_stat: Dict[Tuple[str, str], List[Dict[str, object]]] = {
    ("Demand-Cross-Price", "mean"): [],
    ("Demand-Cross-Price", "min"):  [],
    ("Demand-Cross-Price", "max"):  [],
    ("Demand-Cross-Price", "n"):    [],
    ("Supply-Cross-Price", "mean"): [],
    ("Supply-Cross-Price", "min"):  [],
    ("Supply-Cross-Price", "max"):  [],
    ("Supply-Cross-Price", "n"):    [],
}
# One non-cross table per each of four Elements
non_cross_rows: Dict[str, List[Dict[str, object]]] = {
    "Demand-Income": [],
    "Demand-Own-Price": [],
    "Supply-Own-Price": [],
    "Supply-Temperature": [],
}

# Main loop over Element, Country, and Commodity Map token
for ec in ["Demand-Cross-Price", "Demand-Income", "Demand-Own-Price",
           "Supply-Cross-Price", "Supply-Own-Price", "Supply-Temperature"]:
    for country in target_countries:
        reg_name = country_to_region.get(country)     # Country's v3 region
        keep_grp = country_to_keep.get(country)       # Country's keep group for regional aggregation
        for tgt in target_all:
            # One target token may map to multiple raw values.
            main_raws = commodity_fwd_m2m.get(tgt.lower(), set())

            if ec in ["Demand-Cross-Price", "Supply-Cross-Price"]:
                # Cross-price: retrieve cross_raw statistics, then map to target Cross tokens.
                dd = fetch_cross_dict(ec, main_raws, region=reg_name, scope="country")
                if not dd and keep_grp:
                    dd = fetch_cross_dict(ec, main_raws, keep_grp=keep_grp, scope="region")
                if not dd and keep_grp:
                    dd = fetch_cross_dict(ec, main_raws, keep_grp=keep_grp, scope="peer")

                # present stores aggregates by target Cross token; initialize empty, then fill allowlisted gaps with zero.
                present = defaultdict(lambda: {"m": 0.0, "mi": np.inf, "ma": -np.inf, "n": 0})
                if dd:
                    for cp_raw_lc, st in dd.items():
                        # Map cross_raw_lc to target tokens, preferring dedicated Cross mappings over commodity_rev.
                        if cross_rev_m2m and (cp_raw_lc in cross_rev_m2m):
                            targets_lc = list(cross_rev_m2m[cp_raw_lc])
                        else:
                            targets_lc = list(commodity_rev_m2m.get(cp_raw_lc, []))
                        # Retain allowlisted Cross targets only.
                        targets_lc = [t for t in targets_lc if t in target_all_lc]
                        if not targets_lc:
                            continue
                        mean_i, min_i, max_i, n_i = st
                        # Combine multiple cross_raw sources per target into present using n weights.
                        for cp_tgt_lc in targets_lc:
                            present[cp_tgt_lc]["m"]  += mean_i * n_i
                            present[cp_tgt_lc]["n"]  += n_i
                            present[cp_tgt_lc]["mi"]  = min(present[cp_tgt_lc]["mi"], min_i)
                            present[cp_tgt_lc]["ma"]  = max(present[cp_tgt_lc]["ma"], max_i)

                # Build rows for four matrices: initialize all Commodity All columns to zero, then overwrite from present.
                base_mean = {"Country": country, "Commodity": tgt}
                base_min  = {"Country": country, "Commodity": tgt}
                base_max  = {"Country": country, "Commodity": tgt}
                base_n    = {"Country": country, "Commodity": tgt}
                for cp_name in target_all:
                    key_lc = cp_name.lower()
                    d = present.get(key_lc, None)
                    if d and d["n"] > 0:
                        mean_w = d["m"] / d["n"]
                        base_mean[cp_name] = mean_w
                        base_min[cp_name]  = d["mi"] if d["mi"] != np.inf else mean_w
                        base_max[cp_name]  = d["ma"] if d["ma"] != -np.inf else mean_w
                        base_n[cp_name]    = int(d["n"])
                    else:
                        base_mean[cp_name] = 0.0
                        base_min[cp_name]  = 0.0
                        base_max[cp_name]  = 0.0
                        base_n[cp_name]    = 0
                cross_rows_by_stat[(ec, "mean")].append(base_mean)
                cross_rows_by_stat[(ec, "min")].append(base_min)
                cross_rows_by_stat[(ec, "max")].append(base_max)
                cross_rows_by_stat[(ec, "n")].append(base_n)

            else:
                # Non-cross: aggregate main_raws using the three fallback levels.
                st = fetch_non_cross(ec, main_raws, region=reg_name, scope="country")
                if not st and keep_grp:
                    st = fetch_non_cross(ec, main_raws, keep_grp=keep_grp, scope="region")
                if not st and keep_grp:
                    st = fetch_non_cross(ec, main_raws, keep_grp=keep_grp, scope="peer")
                if not st:
                    # If nothing is found, retain the row with NaN/zero placeholders.
                    non_cross_rows[ec].append({
                        "Country": country, "Commodity": tgt,
                        "Elasticity_mean": np.nan, "Elasticity_min": np.nan, "Elasticity_max": np.nan,
                        "n": 0, "source_level": "missing"
                    })
                else:
                    non_cross_rows[ec].append({
                        "Country": country, "Commodity": tgt,
                        "Elasticity_mean": st[0], "Elasticity_min": st[1], "Elasticity_max": st[2],
                        "n": st[3], "source_level": "country/region/peer"
                    })

# Write all tables to one Excel workbook.
with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter") as writer:
    # Cross-price: Demand/Supply x mean/min/max/n.
    for ec in ["Demand-Cross-Price", "Supply-Cross-Price"]:
        for stat in ["mean", "min", "max", "n"]:
            rows = cross_rows_by_stat[(ec, stat)]
            df = pd.DataFrame(rows)
            # Column order: Country, Commodity, then Commodity All.
            ordered_cols = ["Country", "Commodity"] + target_all
            # Ensure every target column exists, including rare absent targets.
            for col in target_all:
                if col not in df.columns:
                    df[col] = 0 if stat in ["mean", "min", "max"] else 0
            df = df[ordered_cols]
            sheet = f"{ec.split('-')[0]}_Cross_{stat}"[:31]  # Excel sheet names allow at most 31 characters.
            df.to_excel(writer, sheet_name=sheet, index=False)

    # Four non-cross categories
    for ec in ["Demand-Income", "Demand-Own-Price", "Supply-Own-Price", "Supply-Temperature"]:
        df = pd.DataFrame(non_cross_rows[ec])
        ordered_cols = ["Country", "Commodity", "Elasticity_mean", "Elasticity_min", "Elasticity_max", "n", "source_level"]
        for col in ordered_cols:
            if col not in df.columns:
                df[col] = pd.Series(dtype="float64")
        df = df[ordered_cols]
        sheet = ec[:31]
        df.to_excel(writer, sheet_name=sheet, index=False)

    # Add two mapping-review tables for many-to-many cases such as Fish/Durum.
    fwd_rows = [[tok_lc, ";".join(sorted(set(raws)))] for tok_lc, raws in commodity_fwd_m2m.items()]
    pd.DataFrame(fwd_rows, columns=["Map_token(lc)", "Raw_list"]).to_excel(writer, sheet_name="commodity_forward_m2m", index=False)
    rev_rows = [[raw_lc, ";".join(sorted(set(toks)))] for raw_lc, toks in commodity_rev_m2m.items()]
    pd.DataFrame(rev_rows, columns=["Raw(lc)", "Map_tokens(lc)"]).to_excel(writer, sheet_name="commodity_reverse_m2m", index=False)

print("Done:", OUTPUT_XLSX)
