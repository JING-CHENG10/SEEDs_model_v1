#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hard-coded pipeline to map each country (from region.xlsx) to:
- climate_domain
- ipcc_ecological_zone (proxy mapping)
- dominant crop_type (Annual/Perennial)
- soil_type (HAC/LAC/Sandy/Spodic/Volcanic; fallback to HAC if outside SOCref types)

Outputs:
1) region_with_Tier1_keys.xlsx (enhanced region table)
2) LUCE_parameter_Tier1_LUH2_mapping_with_country_keys.xlsx (append sheets into workbook)
"""

import os
import re
import json
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.windows import from_bounds
from shapely.geometry import shape, Point
import fiona
import requests
import openpyxl



# CONFIG: change paths here (NO command-line args needed)


CONFIG = {
    # inputs
    "REGION_XLSX": r"/mnt/data/region.xlsx",
    "LUCE_WORKBOOK_XLSX": r"/mnt/data/LUCE_parameter_Tier1_LUH2_mapping.xlsx",

    # You need a country code + centroid table (M49 numeric -> ISO3 + centroid lat/lon).
    # If your region.xlsx already includes ISO3 and centroid lat/lon columns, you can set this to None.
    "COUNTRY_CODES_CSV": r"/mnt/data/countries_codes_and_coordinates.csv",

    # Optional: ADM0 boundary file. If None or missing, all countries use point fallback only.
    # Needs an ISO3 field, commonly ISO_A3 / ADM0_A3 / ISO3 etc.
    "ADM0_BOUNDARY_FILE": None,  # e.g., r"/mnt/data/adm0.shp"

    # ESDAC/IPCC Tier-1 rasters (IDRISI .rst + .RDC legend)
    "ESDAC_CLIMATE_RST": r"/mnt/data/esdac/CLIMATE_ZONE.rst",
    "ESDAC_CLIMATE_RDC": r"/mnt/data/esdac/CLIMATE_ZONE.RDC",
    "ESDAC_SOIL_RST": r"/mnt/data/esdac/SOIL_TYPE.rst",
    "ESDAC_SOIL_RDC": r"/mnt/data/esdac/SOIL_TYPE.RDC",

    # WDI cache (the script downloads from World Bank API if not cached)
    "WDI_CACHE_DIR": r"/mnt/data/wdi_cache",

    # outputs
    "OUT_REGION_XLSX": r"/mnt/data/region_with_Tier1_keys.xlsx",
    "OUT_WORKBOOK_XLSX": r"/mnt/data/LUCE_parameter_Tier1_LUH2_mapping_with_country_keys.xlsx",

    # parameters
    "NODATA_CODE": 0,          # ESDAC rasters use 0 as nodata/ocean
    "MAX_RADIUS_PX": 60,       # nearest-nonzero search radius (pixels)
}



# Helpers


def _clean_m49(x) -> Optional[int]:
    """Convert M49 numeric code like "'051'" or "051" to int 51."""
    if pd.isna(x):
        return None
    s = str(x).strip().strip("'").strip('"').strip()
    s = re.sub(r"[^\d]", "", s)
    return int(s) if s else None


def parse_idrisi_rdc(rdc_path: str) -> Dict[int, str]:
    """Parse IDRISI .RDC legend entries of form: 'code  1 : Name'."""
    mapping: Dict[int, str] = {}
    with open(rdc_path, "r", encoding="latin1") as f:
        for line in f:
            line = line.strip()
            m = re.match(r"code\s+(\d+)\s*:\s*(.+)$", line)
            if m:
                mapping[int(m.group(1))] = m.group(2).strip()
    if not mapping:
        raise ValueError(f"Failed to parse legend from RDC: {rdc_path}")
    return mapping


def load_adm0_geoms(boundary_path: str,
                    iso_field_candidates: List[str] = None) -> Dict[str, object]:
    """
    Load ADM0 geometries into dict[ISO3] = unified geometry.
    No geopandas dependency.
    """
    if iso_field_candidates is None:
        iso_field_candidates = ["ISO_A3", "iso_a3", "ADM0_A3", "adm0_a3", "ISO3", "iso3"]

    geoms: Dict[str, object] = {}

    with fiona.open(boundary_path, "r") as src:
        props = src.schema["properties"].keys()
        iso_field = None
        for cand in iso_field_candidates:
            if cand in props:
                iso_field = cand
                break
        if iso_field is None:
            raise ValueError(
                f"Cannot find an ISO3 field in boundary file. "
                f"Available fields: {list(props)}"
            )

        for feat in src:
            iso = feat["properties"].get(iso_field)
            if iso is None:
                continue
            iso = str(iso).strip().upper()
            if iso in ("", "-99", "99", "NONE", "NAN"):
                continue

            geom = shape(feat["geometry"])
            geoms[iso] = geoms[iso].union(geom) if iso in geoms else geom

    return geoms


def wdi_fetch_indicator(indicator: str, cache_json_path: str, per_page: int = 20000) -> dict:
    """
    Fetch World Bank WDI indicator as raw JSON.
    If cache exists: read cache; else download and write cache.
    """
    if os.path.exists(cache_json_path):
        with open(cache_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator}"
    params = {"format": "json", "per_page": per_page}
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    os.makedirs(os.path.dirname(cache_json_path), exist_ok=True)
    with open(cache_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    return data


def wdi_latest_by_iso3(raw_json: dict) -> Dict[str, Tuple[Optional[float], Optional[int]]]:
    """
    Convert World Bank JSON into mapping iso3 -> (latest_value, latest_year),
    choosing newest year with non-null value.
    """
    if not isinstance(raw_json, list) or len(raw_json) < 2:
        raise ValueError("Unexpected WDI JSON structure.")
    records = raw_json[1]
    out: Dict[str, Tuple[Optional[float], Optional[int]]] = {}

    for rec in records:
        if rec is None:
            continue
        iso3 = rec.get("countryiso3code")
        year = rec.get("date")
        val = rec.get("value")
        if not iso3:
            continue
        iso3 = str(iso3).upper().strip()

        try:
            y = int(year)
        except Exception:
            y = None
        v = None if val is None else float(val)

        if iso3 not in out:
            out[iso3] = (v, y)
        else:
            prev_v, prev_y = out[iso3]
            if y is not None and (prev_y is None or y > prev_y):
                if v is not None:
                    out[iso3] = (v, y)
            elif y == prev_y:
                if prev_v is None and v is not None:
                    out[iso3] = (v, y)

    return out



# Raster mapping (dominant + nearest non-zero fallback)


@dataclass
class DominantResult:
    code: Optional[int]
    method: str                 # "direct" or "nearest_nonzero" or "missing"
    search_radius_px: int
    top3: List[Tuple[int, float]]


def _cos_lat_weights_for_rows(transform, row_indices: np.ndarray) -> np.ndarray:
    """cos(lat) weights for each pixel row (approx area weight in lon/lat grid)."""
    dy = transform.e
    y0 = transform.f
    lat = y0 + (row_indices + 0.5) * dy
    return np.cos(np.deg2rad(lat))


def dominant_code_in_geom(src: rasterio.io.DatasetReader, geom, nodata: int = 0) -> Optional[Tuple[int, List[Tuple[int, float]]]]:
    """Area-weighted dominant code within a polygon; returns (dominant, top3 fractions) or None."""
    minx, miny, maxx, maxy = geom.bounds
    win = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
    if win.width <= 0 or win.height <= 0:
        return None
    win = win.round_offsets().round_lengths()

    data = src.read(1, window=win, boundless=True, fill_value=nodata)
    local_transform = rasterio.windows.transform(win, src.transform)

    mask = features.rasterize(
        [(geom, 1)],
        out_shape=data.shape,
        transform=local_transform,
        fill=0,
        dtype="uint8",
        all_touched=False
    )

    valid = (mask == 1) & (data != nodata)
    if not np.any(valid):
        return None

    vals = data[valid].astype(np.int32)
    rows = np.where(valid)[0].astype(np.int32)
    w = _cos_lat_weights_for_rows(local_transform, rows)

    uniq = np.unique(vals)
    wsum = {int(v): float(w[vals == v].sum()) for v in uniq}
    total = sum(wsum.values())
    ranked = sorted(wsum.items(), key=lambda kv: kv[1], reverse=True)

    dom = ranked[0][0]
    top3 = [(k, v / total) for k, v in ranked[:3]]
    return dom, top3


def nearest_nonzero_code(src: rasterio.io.DatasetReader,
                         lon: float,
                         lat: float,
                         nodata: int = 0,
                         max_radius_px: int = 60) -> Tuple[Optional[int], int]:
    """
    Find nearest non-nodata code around a point by expanding pixel radius.
    Returns (code, radius_px).
    """
    try:
        row0, col0 = src.index(lon, lat)
    except Exception:
        return None, max_radius_px

    arr = src.read(1)  # ESDAC rasters are small (5 arc-min global), full read is OK
    h, w = arr.shape

    for r in range(0, max_radius_px + 1):
        rmin = max(0, row0 - r)
        rmax = min(h - 1, row0 + r)
        cmin = max(0, col0 - r)
        cmax = min(w - 1, col0 + r)

        window = arr[rmin:rmax + 1, cmin:cmax + 1]
        ys, xs = np.where(window != nodata)
        if ys.size == 0:
            continue

        abs_rows = ys + rmin
        abs_cols = xs + cmin
        d2 = (abs_rows - row0) ** 2 + (abs_cols - col0) ** 2
        idx = int(np.argmin(d2))
        code = int(arr[int(abs_rows[idx]), int(abs_cols[idx])])
        return code, r

    return None, max_radius_px


def get_dominant_with_fallback(src: rasterio.io.DatasetReader,
                               geom_or_point,
                               nodata: int = 0,
                               max_radius_px: int = 60) -> DominantResult:
    """Polygon dominant; if empty or missing, fallback to nearest non-zero at representative point/centroid."""
    if geom_or_point is None:
        return DominantResult(code=None, method="missing", search_radius_px=0, top3=[])

    if isinstance(geom_or_point, Point):
        code, radius = nearest_nonzero_code(src, geom_or_point.x, geom_or_point.y, nodata, max_radius_px)
        return DominantResult(code=code, method="nearest_nonzero", search_radius_px=radius, top3=[])

    res = dominant_code_in_geom(src, geom_or_point, nodata=nodata)
    if res is not None:
        dom, top3 = res
        return DominantResult(code=dom, method="direct", search_radius_px=0, top3=top3)

    rp = geom_or_point.representative_point()
    code, radius = nearest_nonzero_code(src, rp.x, rp.y, nodata, max_radius_px)
    return DominantResult(code=code, method="nearest_nonzero", search_radius_px=radius, top3=[])



# Rule mappings using the requested approximate mapping


def climate_domain_from_ipcc_zone(zone: str) -> str:
    if zone.startswith("Tropical"):
        return "Tropical"
    if zone.startswith("Warm Temperate"):
        return "Subtropical"
    if zone.startswith("Cool Temperate"):
        return "Temperate"
    if zone.startswith("Boreal"):
        return "Boreal"
    if zone.startswith("Polar"):
        return "Boreal"  # your earlier convention
    return "Unknown"


ECOZONE_BY_IPCC_ZONE = {
    "Tropical Wet": "rain forest",
    "Tropical Moist": "moist deciduous forest",
    "Tropical Dry": "dry forest",
    "Tropical Montane": "mountain systems",
    "Warm Temperate Moist": "humid forest",
    "Warm Temperate Dry": "steppe",
    "Cool Temperate Moist": "oceanic forest",
    "Cool Temperate Dry": "continental forest",
    "Boreal Moist": "coniferous forest",
    "Boreal Dry": "tundra woodland",
    "Polar Moist": "tundra woodland",
    "Polar Dry": "tundra woodland",
}

SOC_CLIMATE_REGION_BY_IPCC_ZONE = {
    "Tropical Wet": "Tropical, wet",
    "Tropical Moist": "Tropical, moist",
    "Tropical Dry": "Tropical, dry",
    "Tropical Montane": "Tropical montane",
    "Warm Temperate Moist": "Warm temperate, moist",
    "Warm Temperate Dry": "Warm temperate, dry",
    "Cool Temperate Moist": "Cold temperate, moist",
    "Cool Temperate Dry": "Cold temperate, dry",
    "Boreal Moist": "Boreal",
    "Boreal Dry": "Boreal",
    "Polar Moist": "Boreal",
    "Polar Dry": "Boreal",
}


def soil_type_normalize(soil_raw: str) -> Tuple[str, str, bool]:
    """
    SOCref table typically uses HAC/LAC/Sandy/Spodic/Volcanic.
    Others (Organic/Wetland/Other) fallback to HAC (flagged).
    """
    s = soil_raw
    if s == "High Activity Clay Soils":
        return s, "HAC", False
    if s == "Low Activity Clay Soils":
        return s, "LAC", False
    if s == "Sandy Soils":
        return s, "Sandy", False
    if s == "Spodic Soils":
        return s, "Spodic", False
    if s == "Volcanic Soils":
        return s, "Volcanic", False

    if s in ("Organic", "Wetland Soils", "Other Areas"):
        return s, "HAC", True

    return s, "HAC", True


def dominant_crop_type_from_wdi(perm: Optional[float], arable: Optional[float]) -> Tuple[str, str, Optional[float]]:
    """
    Scheme 2: dominant crop type using:
    perennial_share = perm/(perm+arable) > 0.5 -> Perennial else Annual
    Returns (crop_type, method, perennial_share)
    """
    method = "worldbank_ratio_perm_over_perm_plus_arable"
    if perm is None and arable is None:
        return "Annual cropland", "missing_wdi_default_annual", None
    perm = 0.0 if perm is None else float(perm)
    arable = 0.0 if arable is None else float(arable)
    denom = perm + arable
    if denom <= 0:
        return "Annual cropland", "degenerate_wdi_default_annual", None
    share = perm / denom
    if share > 0.5:
        return "Perennial cropland", method, share
    return "Annual cropland", method, share


def crop_climate_region(crop_type: str, ipcc_zone: str) -> str:
    """Crop table keys: Annual -> All; Perennial -> climate-specific bins."""
    if crop_type == "Annual cropland":
        return "All"
    # perennial
    if ipcc_zone.startswith(("Warm Temperate", "Cool Temperate", "Boreal", "Polar")):
        return "Temperate (all moisture regimes)"
    return SOC_CLIMATE_REGION_BY_IPCC_ZONE.get(ipcc_zone, "Tropical, moist")


def grass_climate_region(ipcc_zone: str) -> str:
    return SOC_CLIMATE_REGION_BY_IPCC_ZONE.get(ipcc_zone, "Warm temperate, moist")



# Excel output helpers


def write_df_to_sheet(wb: openpyxl.Workbook, sheet_name: str, df: pd.DataFrame, index: int = 0):
    if sheet_name in wb.sheetnames:
        ws_old = wb[sheet_name]
        wb.remove(ws_old)
    ws = wb.create_sheet(sheet_name, index)
    ws.append(list(df.columns))
    for row in df.itertuples(index=False):
        ws.append(list(row))



# Main


def main():
    # check inputs
    for k in ["REGION_XLSX", "LUCE_WORKBOOK_XLSX", "ESDAC_CLIMATE_RST", "ESDAC_CLIMATE_RDC", "ESDAC_SOIL_RST", "ESDAC_SOIL_RDC"]:
        if not os.path.exists(CONFIG[k]):
            raise FileNotFoundError(f"Missing required file: {k} => {CONFIG[k]}")

    if CONFIG["COUNTRY_CODES_CSV"] and not os.path.exists(CONFIG["COUNTRY_CODES_CSV"]):
        raise FileNotFoundError(f"Missing COUNTRY_CODES_CSV: {CONFIG['COUNTRY_CODES_CSV']}")

    if CONFIG["ADM0_BOUNDARY_FILE"] and (not os.path.exists(CONFIG["ADM0_BOUNDARY_FILE"])):
        raise FileNotFoundError(f"Missing ADM0 boundary: {CONFIG['ADM0_BOUNDARY_FILE']}")

    os.makedirs(CONFIG["WDI_CACHE_DIR"], exist_ok=True)

    # read region.xlsx
    reg = pd.read_excel(CONFIG["REGION_XLSX"]).copy()
    # normalize m49
    if "M49_Country_Code" not in reg.columns:
        raise ValueError("region.xlsx must contain column: M49_Country_Code")
    reg["M49_num"] = reg["M49_Country_Code"].apply(_clean_m49)

    # detect ISO3 column (user said it exists now)
    iso_col = None
    for c in reg.columns:
        if c.lower() in ("iso3", "alpha3", "iso_a3"):
            iso_col = c
            break

    # read codes CSV if available to ensure centroid lat/lon
    if CONFIG["COUNTRY_CODES_CSV"]:
        codes = pd.read_csv(CONFIG["COUNTRY_CODES_CSV"])
        # expected columns (common in UN table); adjust if yours differ
        # Numeric code, Alpha-3 code, Country, Latitude (average), Longitude (average)
        for need in ["Numeric code", "Alpha-3 code", "Latitude (average)", "Longitude (average)"]:
            if need not in codes.columns:
                raise ValueError(f"COUNTRY_CODES_CSV missing column: {need}")

        codes["Numeric code"] = codes["Numeric code"].apply(_clean_m49)
        codes["iso3"] = codes["Alpha-3 code"].astype(str).str.replace('"', "").str.strip().str.upper()
        codes["centroid_lat"] = codes["Latitude (average)"].astype(str).str.replace('"', "").astype(float)
        codes["centroid_lon"] = codes["Longitude (average)"].astype(str).str.replace('"', "").astype(float)

        if iso_col is None:
            # join by M49 to get ISO3
            reg = reg.merge(
                codes[["Numeric code", "iso3", "centroid_lat", "centroid_lon"]],
                left_on="M49_num", right_on="Numeric code", how="left"
            )
        else:
            reg["iso3"] = reg[iso_col].astype(str).str.strip().str.upper()
            reg = reg.merge(
                codes[["Numeric code", "iso3", "centroid_lat", "centroid_lon"]],
                left_on=["M49_num", "iso3"], right_on=["Numeric code", "iso3"], how="left"
            )
    else:
        # require region has centroid columns
        if iso_col is None or "centroid_lat" not in reg.columns or "centroid_lon" not in reg.columns:
            raise ValueError("If COUNTRY_CODES_CSV is None, region.xlsx must include ISO3 + centroid_lat + centroid_lon columns.")
        reg["iso3"] = reg[iso_col].astype(str).str.strip().str.upper()

    # load boundaries (optional)
    iso3_to_geom = {}
    if CONFIG["ADM0_BOUNDARY_FILE"]:
        iso3_to_geom = load_adm0_geoms(CONFIG["ADM0_BOUNDARY_FILE"])

    # open rasters
    clim_legend = parse_idrisi_rdc(CONFIG["ESDAC_CLIMATE_RDC"])
    soil_legend = parse_idrisi_rdc(CONFIG["ESDAC_SOIL_RDC"])
    clim_src = rasterio.open(CONFIG["ESDAC_CLIMATE_RST"])
    soil_src = rasterio.open(CONFIG["ESDAC_SOIL_RST"])

    # fetch WDI data (cached)
    perm_path = os.path.join(CONFIG["WDI_CACHE_DIR"], "AG.LND.CROP.ZS.json")
    arbl_path = os.path.join(CONFIG["WDI_CACHE_DIR"], "AG.LND.ARBL.ZS.json")
    perm_json = wdi_fetch_indicator("AG.LND.CROP.ZS", perm_path)
    arbl_json = wdi_fetch_indicator("AG.LND.ARBL.ZS", arbl_path)
    perm_latest = wdi_latest_by_iso3(perm_json)
    arbl_latest = wdi_latest_by_iso3(arbl_json)

    # per-country mapping
    rows = []
    for _, r in reg.iterrows():
        iso3 = r.get("iso3")
        if pd.isna(iso3) or not iso3:
            continue
        iso3 = str(iso3).upper().strip()

        geom = iso3_to_geom.get(iso3)

        if geom is None:
            lon = r.get("centroid_lon")
            lat = r.get("centroid_lat")
            geom_or_point = Point(float(lon), float(lat)) if pd.notna(lon) and pd.notna(lat) else None
        else:
            geom_or_point = geom

        # climate & soil dominant with scheme A fallback
        clim_res = get_dominant_with_fallback(
            clim_src, geom_or_point, nodata=CONFIG["NODATA_CODE"], max_radius_px=CONFIG["MAX_RADIUS_PX"]
        )
        soil_res = get_dominant_with_fallback(
            soil_src, geom_or_point, nodata=CONFIG["NODATA_CODE"], max_radius_px=CONFIG["MAX_RADIUS_PX"]
        )

        ipcc_zone = clim_legend.get(clim_res.code) if clim_res.code is not None else None
        soil_raw = soil_legend.get(soil_res.code) if soil_res.code is not None else None

        # rule mappings
        if ipcc_zone is None:
            climate_domain = "Unknown"
            ecozone = "Unknown"
            soc_region = "Unknown"
        else:
            climate_domain = climate_domain_from_ipcc_zone(ipcc_zone)
            ecozone = ECOZONE_BY_IPCC_ZONE.get(ipcc_zone, "Unknown")
            soc_region = SOC_CLIMATE_REGION_BY_IPCC_ZONE.get(ipcc_zone, "Unknown")

        soil_type_raw = soil_raw if soil_raw is not None else "Unknown"
        soil_raw_label, soil_type_for_socref, soil_fallback = soil_type_normalize(soil_type_raw)

        # dominant crop_type (scheme 2)
        perm_val, perm_year = perm_latest.get(iso3, (None, None))
        arbl_val, arbl_year = arbl_latest.get(iso3, (None, None))
        crop_type, crop_method, perennial_share = dominant_crop_type_from_wdi(perm_val, arbl_val)

        rows.append({
            "iso3": iso3,
            "M49_num": int(r["M49_num"]) if pd.notna(r["M49_num"]) else None,

            # optional: keep the original country label if present
            "country_label": r.get("Region_label_new") if "Region_label_new" in reg.columns else r.get("country", None),

            "centroid_lat": r.get("centroid_lat", None),
            "centroid_lon": r.get("centroid_lon", None),

            # climate
            "climate_zone_code": clim_res.code,
            "ipcc_climate_zone": ipcc_zone,
            "climate_domain": climate_domain,
            "ipcc_ecological_zone": ecozone,
            "soc_climate_region": soc_region,
            "grass_climate_region": grass_climate_region(ipcc_zone or ""),

            "climate_mapping_method": clim_res.method,
            "climate_mapping_search_radius_px": clim_res.search_radius_px,

            # soil
            "soil_type_code": soil_res.code,
            "soil_type_raw": soil_type_raw,
            "soil_type": soil_type_for_socref,
            "soil_type_is_fallback": bool(soil_fallback),
            "soil_mapping_method": soil_res.method,
            "soil_mapping_search_radius_px": soil_res.search_radius_px,

            # crop
            "crop_type": crop_type,
            "crop_type_method": crop_method,
            "perennial_share_perm_over_perm_plus_arable": perennial_share,
            "crop_climate_region": crop_climate_region(crop_type, ipcc_zone or ""),

            "perm_cropland_pct_land_latest": perm_val,
            "perm_cropland_year": perm_year,
            "arable_land_pct_land_latest": arbl_val,
            "arable_land_year": arbl_year,
        })

    out = pd.DataFrame(rows).sort_values(["iso3"])

    # write enhanced region file
    reg_out = reg.copy()
    if "iso3" not in reg_out.columns:
        reg_out["iso3"] = reg_out[iso_col].astype(str).str.strip().str.upper()

    reg_out = reg_out.merge(out, on=["iso3", "M49_num"], how="left")
    reg_out.to_excel(CONFIG["OUT_REGION_XLSX"], index=False)

    # write into LUCE workbook: add two sheets
    wb = openpyxl.load_workbook(CONFIG["LUCE_WORKBOOK_XLSX"])

    notes = pd.DataFrame([{
        "ecozone_mapping": "proxy via IPCC climate zone -> Table4.12 ecozone dictionary (user approved)",
        "missing_islands": "nearest_nonzero fallback (scheme A), radius in pixels recorded",
        "crop_type_scheme": "dominant crop type = perm/(perm+arable) > 0.5 => Perennial else Annual (one row per country)",
        "boundary_file_used": CONFIG["ADM0_BOUNDARY_FILE"] or "None (centroid point-only)",
        "rasters": f"{CONFIG['ESDAC_CLIMATE_RST']} ; {CONFIG['ESDAC_SOIL_RST']}",
        "wdi_indicators": "AG.LND.CROP.ZS (perm cropland % land) ; AG.LND.ARBL.ZS (arable land % land)",
        "nodata_code": CONFIG["NODATA_CODE"],
        "max_radius_px": CONFIG["MAX_RADIUS_PX"],
    }])

    write_df_to_sheet(wb, "country_to_Tier1_keys", out, index=0)
    write_df_to_sheet(wb, "country_to_Tier1_notes", notes, index=1)
    wb.save(CONFIG["OUT_WORKBOOK_XLSX"])

    print("Done.")
    print(f"Enhanced region saved to: {CONFIG['OUT_REGION_XLSX']}")
    print(f"Workbook saved to:       {CONFIG['OUT_WORKBOOK_XLSX']}")


if __name__ == "__main__":
    main()
