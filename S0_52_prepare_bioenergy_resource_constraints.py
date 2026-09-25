# -*- coding: utf-8 -*-
"""Prepare source-backed bioenergy resource constraints.

This preprocessor builds the P1 `bioenergy_resource_constraints.csv` input from
public residue data and local model land-cover data. It does not overwrite raw
source files. By default it writes derived CSVs directly under
`input/Bioenergy/`.

Main data sources:
- OMD crop residues: https://doi.org/10.5281/zenodo.10450921
- Land-cover base: model `Land_cover_base_refill.xlsx`
- Energy-crop yield defaults: Li et al. 2018, Scientific Data,
  https://doi.org/10.1038/sdata.2018.169
"""

from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from config_paths import get_input_base
from S2_0_load_data import DataPaths, _norm_m49


OMD_RECORD_API = "https://zenodo.org/api/records/10450921"
OMD_DOI = "https://doi.org/10.5281/zenodo.10450921"
OMD_PAPER_DOI = "https://doi.org/10.5194/essd-17-369-2025"
ENERGY_CROP_YIELD_SOURCE = "https://doi.org/10.1038/sdata.2018.169"
LI2020_RECORD_API = "https://zenodo.org/api/records/3274254"
LI2020_ZENODO_DOI = "https://doi.org/10.5281/zenodo.3274254"
LI2020_PAPER_DOI = "https://doi.org/10.5194/essd-12-789-2020"
LI2020_ZIP_NAME = "Bioenergy_crop_yields.zip"


RESIDUE_QUALITY_COLUMNS = [
    "residue_carbon_pct",
    "residue_nitrogen_pct",
    "residue_phosphorus_pct",
    "residue_potassium_pct",
    "residue_calcium_pct",
    "residue_magnesium_pct",
    "residue_sulfur_pct",
    "residue_lignin_pct",
    "residue_polyphenols_pct",
    "residue_cellulose_pct",
    "residue_ash_pct",
    "residue_quality_n_obs",
]


QUALITY_SOURCE_COLUMNS = [
    "residue_quality_match",
]


RESOURCE_COLUMNS = [
    "scenario",
    "M49_Country_Code",
    "country_name",
    "year",
    "feedstock",
    "feedstock_category",
    "parent_commodity",
    "resource_available_t",
    "resource_available_tdm",
    "competing_use_t",
    "competing_use_tdm",
    "sustainable_fraction",
    "yield_tdm_per_ha",
    "eligible_land_area_ha",
    "ghg_direct_kgco2e_per_tdm",
    "ghg_soil_kgco2e_per_tdm",
    "ghg_avoided_kgco2e_per_tdm",
    "fossil_displacement_kgco2e_per_tj",
    "beccs_capture_kgco2_per_tdm",
    *RESIDUE_QUALITY_COLUMNS,
    *QUALITY_SOURCE_COLUMNS,
    "source",
    "notes",
]


def _norm_m49(value: Any) -> str:
    """Normalize M49 codes to the model convention: apostrophe + 3 digits."""
    if pd.isna(value):
        return ""
    text = str(value).strip().lstrip("'\"")
    if not text:
        return ""
    try:
        return f"'{int(float(text)):03d}"
    except Exception:
        digits = "".join(ch for ch in text if ch.isdigit())
        if digits:
            try:
                return f"'{int(digits):03d}"
            except Exception:
                return ""
    return text


OMD_ITEM_MAP: Dict[str, Dict[str, Any]] = {
    "Barley": {"feedstock": "barley_straw", "parent_commodity": "Barley"},
    "Beans, dry": {"feedstock": "dry_bean_residue", "parent_commodity": "Beans, dry"},
    "Groundnuts, with shell": {"feedstock": "groundnut_residue", "parent_commodity": "Groundnut"},
    "Maize": {"feedstock": "maize_stover", "parent_commodity": "Maize (corn)"},
    "Millet": {"feedstock": "millet_stover", "parent_commodity": "Millet"},
    "Oats": {"feedstock": "oat_straw", "parent_commodity": "Oats"},
    "Potatoes": {"feedstock": "potato_residue", "parent_commodity": "Potatoes"},
    "Rice, paddy": {"feedstock": "rice_straw", "parent_commodity": "Rice"},
    "Rye": {"feedstock": "rye_straw", "parent_commodity": "Rye"},
    "Sorghum": {"feedstock": "sorghum_stover", "parent_commodity": "Sorghum"},
    "Soybeans": {"feedstock": "soybean_residue", "parent_commodity": "Soya beans"},
    "Wheat": {"feedstock": "wheat_straw", "parent_commodity": "Wheat"},
}


ENERGY_CROP_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "eucalypt_energy_crop": {
        "yield_tdm_per_ha": 12.0,
        "li2020_var": "Eucalypt",
        "notes": "Fallback only; Li et al. 2020 gridded yield is preferred when available.",
    },
    "miscanthus_energy_crop": {
        "yield_tdm_per_ha": 10.0,
        "li2020_var": "Miscanthus",
        "notes": "Fallback only; Li et al. 2020 gridded yield is preferred when available.",
    },
    "poplar_energy_crop": {
        "yield_tdm_per_ha": 8.0,
        "li2020_var": "Poplar",
        "notes": "Fallback only; Li et al. 2020 gridded yield is preferred when available.",
    },
    "switchgrass_energy_crop": {
        "yield_tdm_per_ha": 8.0,
        "li2020_var": "Switchgrass",
        "notes": "Fallback only; Li et al. 2020 gridded yield is preferred when available.",
    },
    "willow_energy_crop": {
        "yield_tdm_per_ha": 8.0,
        "li2020_var": "Willow",
        "notes": "Fallback only; Li et al. 2020 gridded yield is preferred when available.",
    },
}


NONCROP_FEEDSTOCK_RESOURCE_CAPS: Dict[str, Dict[str, Any]] = {
    "Animal waste": {
        "feedstock": "animal_waste_biogas",
        "feedstock_category": "animal_waste",
        "lhv_gj_per_tdm": 15.0,
    },
    "Biogases": {
        "feedstock": "biogas_waste_feedstock",
        "feedstock_category": "waste_biomass",
        "lhv_gj_per_tdm": 15.0,
    },
    "Black liquor": {
        "feedstock": "black_liquor_forest_industrial_residue",
        "feedstock_category": "forest_industrial_residue",
        "lhv_gj_per_tdm": 14.0,
    },
    "Charcoal": {
        "feedstock": "charcoal_woody_biomass",
        "feedstock_category": "forest_biomass",
        "lhv_gj_per_tdm": 29.0,
    },
    "Fuelwood": {
        "feedstock": "fuelwood_forest_biomass",
        "feedstock_category": "forest_biomass",
        "lhv_gj_per_tdm": 18.0,
    },
    # Carrier-level crop-residue fallbacks. OMD crop-specific residues remain
    # the preferred constraints, but these prevent FAOSTAT bagasse / other
    # vegetal residue demand rows from staying uncapped when no crop-specific
    # residue mix is available for a country.
    "Bagasse": {
        "feedstock": "bagasse_residue",
        "feedstock_category": "crop_residue",
        "lhv_gj_per_tdm": 17.0,
    },
    "Other vegetal material and residues": {
        "feedstock": "mixed_crop_residue",
        "feedstock_category": "crop_residue",
        "lhv_gj_per_tdm": 17.0,
    },
}


OMD_QUALITY_SPECS: Dict[str, Dict[str, Any]] = {
    "barley_straw": {"crop": ["Barley"], "parts": ["straw"], "match": "exact_crop_part"},
    "dry_bean_residue": {"crop": ["Cowpea"], "parts": ["haulm"], "match": "proxy_cowpea_haulm_for_dry_beans"},
    "groundnut_residue": {"crop": ["Groundnut"], "parts": ["straw", "haulm", "shell"], "match": "exact_or_close_crop_part"},
    "maize_stover": {"crop": ["Maize"], "parts": ["stover", "stalk", "leaf", "leaves", "husk", "cob"], "match": "exact_crop_residue_parts"},
    "millet_stover": {"crop": ["Millet"], "parts": ["straw", "stover"], "match": "exact_crop_part"},
    "rice_straw": {"crop": ["Rice"], "parts": ["straw"], "match": "exact_crop_part"},
    "soybean_residue": {"crop": ["Soybean"], "parts": ["straw"], "match": "exact_crop_part"},
    "wheat_straw": {"crop": ["Wheat"], "parts": ["straw"], "match": "exact_crop_part"},
    # OMD quality file does not contain clean residue observations for these
    # feedstocks. They are left blank rather than silently borrowing grain,
    # silage, manure or meal observations.
    "oat_straw": {"crop": ["Oats"], "parts": ["straw"], "match": "missing_exact_quality"},
    "potato_residue": {"crop": ["Potatoes"], "parts": ["straw", "haulm", "vine"], "match": "missing_exact_quality"},
    "rye_straw": {"crop": ["Rye"], "parts": ["straw"], "match": "missing_exact_quality"},
    "sorghum_stover": {"crop": ["Sorghum"], "parts": ["stover", "stalk", "straw"], "match": "missing_exact_quality"},
}


def _download_omd_files(raw_dir: Path, required: Iterable[str]) -> Dict[str, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(OMD_RECORD_API, timeout=60) as response:
        record = json.load(response)
    files = {f.get("key"): f for f in record.get("files", [])}
    out: Dict[str, Path] = {}
    for key in required:
        info = files.get(key)
        if not info:
            raise FileNotFoundError(f"OMD record does not contain {key!r}")
        dest = raw_dir / key
        expected_size = int(info.get("size") or 0)
        if dest.exists() and (expected_size <= 0 or dest.stat().st_size == expected_size):
            out[key] = dest
            continue
        url = info.get("links", {}).get("self")
        if not url:
            raise RuntimeError(f"OMD record file {key!r} has no download URL")
        with urllib.request.urlopen(url, timeout=180) as response, open(dest, "wb") as handle:
            handle.write(response.read())
        out[key] = dest
    return out


def _country_name_to_m49(dict_v3_path: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[int, str]]:
    region = pd.read_excel(dict_v3_path, sheet_name="region")
    region.columns = [str(c).strip() for c in region.columns]
    name_cols = [
        "Region_label_new",
        "Country",
        "NAME",
        "Region_label",
        "Region_label2",
    ]
    name_to_m49: Dict[str, str] = {}
    m49_to_name: Dict[str, str] = {}
    area_code_to_m49: Dict[int, str] = {}
    for _, row in region.iterrows():
        m49 = _norm_m49(row.get("M49_Country_Code") or row.get("M49 Code") or row.get("Country"))
        if not m49:
            continue
        try:
            area_code = int(float(row.get("Area Code")))
            area_code_to_m49[area_code] = m49
        except Exception:
            pass
        label = str(row.get("Region_label_new") or row.get("NAME") or row.get("Country") or "").strip()
        if label:
            m49_to_name[m49] = label
        for col in name_cols:
            val = row.get(col)
            if pd.isna(val):
                continue
            text = str(val).strip()
            if text:
                name_to_m49[text.lower()] = m49
    return name_to_m49, m49_to_name, area_code_to_m49


def _read_omd_crop_residues(raw_dir: Path, *, download: bool) -> pd.DataFrame:
    crop_file = raw_dir / "Crop residues.csv"
    if download or not crop_file.exists():
        _download_omd_files(raw_dir, ["Crop residues.csv", "Readme file.csv"])
    if not crop_file.exists():
        raise FileNotFoundError(
            f"{crop_file} not found. Run with --download-omd or download from {OMD_DOI}."
        )
    df = pd.read_csv(crop_file, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _read_omd_residue_quality(raw_dir: Path, *, download: bool) -> pd.DataFrame:
    quality_file = raw_dir / "Residue quality_12_05_24.csv"
    if download or not quality_file.exists():
        _download_omd_files(raw_dir, ["Residue quality_12_05_24.csv"])
    if not quality_file.exists():
        raise FileNotFoundError(
            f"{quality_file} not found. Run with --download-omd or download from {OMD_DOI}."
        )
    raw = pd.read_csv(quality_file, header=None, low_memory=False)
    header_row = None
    for idx in range(min(5, len(raw))):
        row = raw.iloc[idx].fillna("").astype(str).str.strip().tolist()
        if "Category" in row and "Crop" in row and "Plant part" in row:
            header_row = idx
            break
    if header_row is None:
        raise ValueError(f"Could not locate OMD quality header row in {quality_file}")
    headers = raw.iloc[header_row].fillna("").astype(str).str.strip().tolist()
    headers = [h if h else f"col_{i}" for i, h in enumerate(headers)]
    df = raw.iloc[header_row + 1 :].copy()
    df.columns = headers
    return df.reset_index(drop=True)


def _clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.replace({"na": np.nan, "NA": np.nan, "?": np.nan, "": np.nan}),
        errors="coerce",
    )


def _quality_median(rowset: pd.DataFrame, column: str) -> float:
    if column not in rowset.columns:
        return np.nan
    vals = _clean_numeric(rowset[column])
    return float(vals.median()) if vals.notna().any() else np.nan


def build_residue_quality_lookup(quality_df: Optional[pd.DataFrame]) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame]:
    diagnostics: List[Dict[str, Any]] = []
    if quality_df is None or quality_df.empty:
        return {}, pd.DataFrame([{"issue": "missing_omd_residue_quality_file"}])
    df = quality_df.copy()
    for col in ["Category", "Crop", "Plant part"]:
        if col not in df.columns:
            diagnostics.append({"issue": "missing_omd_quality_column", "column": col})
            return {}, pd.DataFrame(diagnostics)
        df[col] = df[col].fillna("").astype(str).str.strip()
    df = df[df["Category"].str.lower().eq("crop residues")].copy()
    lookup: Dict[str, Dict[str, Any]] = {}
    for feedstock, spec in OMD_QUALITY_SPECS.items():
        crop_names = [str(x).lower() for x in spec.get("crop", [])]
        part_terms = [str(x).lower() for x in spec.get("parts", [])]
        crop_mask = df["Crop"].str.lower().isin(crop_names)
        if not crop_mask.any():
            crop_mask = pd.Series(False, index=df.index)
            for crop in crop_names:
                crop_mask = crop_mask | df["Crop"].str.lower().str.contains(crop, regex=False, na=False)
        part_lower = df["Plant part"].str.lower()
        part_mask = pd.Series(False, index=df.index)
        for term in part_terms:
            part_mask = part_mask | part_lower.str.contains(term, regex=False, na=False)
        subset = df[crop_mask & part_mask].copy()
        if subset.empty:
            diagnostics.append({
                "issue": "missing_residue_quality_match",
                "feedstock": feedstock,
                "crop_terms": ";".join(crop_names),
                "part_terms": ";".join(part_terms),
            })
            continue
        rec = {
            "residue_carbon_pct": _quality_median(subset, "Carbon (%)"),
            "residue_nitrogen_pct": _quality_median(subset, "Nitrogen (%)"),
            "residue_phosphorus_pct": _quality_median(subset, "Phosphorus"),
            "residue_potassium_pct": _quality_median(subset, "Potassium (%)"),
            "residue_calcium_pct": _quality_median(subset, "Calcium (%)"),
            "residue_magnesium_pct": _quality_median(subset, "Magnesium (%)"),
            "residue_sulfur_pct": _quality_median(subset, "Sulfur"),
            "residue_lignin_pct": _quality_median(subset, "Lignin (%)"),
            "residue_polyphenols_pct": _quality_median(subset, "Polyphenols (%)"),
            "residue_cellulose_pct": _quality_median(subset, "Cellulose (%)"),
            "residue_ash_pct": _quality_median(subset, "Ash (%)"),
            "residue_quality_n_obs": int(len(subset)),
            "residue_quality_match": str(spec.get("match") or "matched"),
        }
        lookup[feedstock] = rec
    return lookup, pd.DataFrame(diagnostics)


def build_crop_residue_constraints(
    omd_df: pd.DataFrame,
    *,
    dict_v3_path: str,
    scenario: str,
    sustainable_fraction: float,
    competing_use_fraction: float,
    residue_quality_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    name_to_m49, m49_to_name, area_code_to_m49 = _country_name_to_m49(dict_v3_path)
    rows: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    mapped_items = set(OMD_ITEM_MAP)
    work = omd_df[omd_df["Item"].isin(mapped_items)].copy()
    work["Year"] = pd.to_numeric(work["Year"], errors="coerce")
    work["Resid production (tonnes/year)"] = pd.to_numeric(
        work["Resid production (tonnes/year)"],
        errors="coerce",
    )
    work = work.dropna(subset=["Year", "Resid production (tonnes/year)"])
    work = work[work["Resid production (tonnes/year)"].gt(0.0)]
    for _, row in work.iterrows():
        area = str(row.get("Area")).strip()
        try:
            area_code = int(float(row.get("Area Code")))
        except Exception:
            area_code = None
        m49 = area_code_to_m49.get(area_code) if area_code is not None else None
        if not m49:
            m49 = name_to_m49.get(area.lower())
        if not m49:
            diagnostics.append({
                "issue": "unmapped_omd_country",
                "area": area,
                "item": row.get("Item"),
                "year": int(row.get("Year")),
            })
            continue
        item = str(row.get("Item"))
        spec = OMD_ITEM_MAP[item]
        quality = dict((residue_quality_lookup or {}).get(spec["feedstock"], {}))
        residue_tdm = float(row.get("Resid production (tonnes/year)"))
        competing = residue_tdm * max(float(competing_use_fraction), 0.0)
        notes = (
            "OMD national crop-residue production. Treated as dry-matter-equivalent "
            "P1 resource; sustainable_fraction is a configurable soil/competing-use "
            "screen and should be replaced by crop/country-specific residue-removal "
            "limits when available."
        )
        rows.append({
            "scenario": scenario,
            "M49_Country_Code": m49,
            "country_name": m49_to_name.get(m49, area),
            "year": int(row.get("Year")),
            "feedstock": spec["feedstock"],
            "feedstock_category": "crop_residue",
            "parent_commodity": spec["parent_commodity"],
            "resource_available_t": np.nan,
            "resource_available_tdm": residue_tdm,
            "competing_use_t": np.nan,
            "competing_use_tdm": competing,
            "sustainable_fraction": max(0.0, min(float(sustainable_fraction), 1.0)),
            "yield_tdm_per_ha": np.nan,
            "eligible_land_area_ha": np.nan,
            "ghg_direct_kgco2e_per_tdm": np.nan,
            "ghg_soil_kgco2e_per_tdm": np.nan,
            "ghg_avoided_kgco2e_per_tdm": np.nan,
            "fossil_displacement_kgco2e_per_tj": np.nan,
            "beccs_capture_kgco2_per_tdm": np.nan,
            **{col: quality.get(col, np.nan) for col in RESIDUE_QUALITY_COLUMNS},
            "residue_quality_match": quality.get("residue_quality_match", ""),
            "source": f"{OMD_DOI}; {OMD_PAPER_DOI}",
            "notes": (
                notes
                + (
                    " Residue quality medians from OMD Residue quality_12_05_24.csv."
                    if quality else " No clean OMD residue-quality match for this feedstock."
                )
            ),
        })
    return pd.DataFrame(rows, columns=RESOURCE_COLUMNS), pd.DataFrame(diagnostics)


def _load_luh2_land_cover(land_cover_xlsx: str) -> pd.DataFrame:
    df = pd.read_excel(land_cover_xlsx, sheet_name="LUH2")
    df.columns = [str(c).strip() for c in df.columns]
    if "Land cover" not in df.columns:
        raise ValueError(f"{land_cover_xlsx} sheet LUH2 missing 'Land cover'")
    return df


def _download_li2020_yield_map(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / LI2020_ZIP_NAME
    if not zip_path.exists() or zip_path.stat().st_size <= 0:
        with urllib.request.urlopen(LI2020_RECORD_API, timeout=60) as response:
            record = json.load(response)
        files = {f.get("key"): f for f in record.get("files", [])}
        info = files.get(LI2020_ZIP_NAME)
        if not info:
            raise FileNotFoundError(f"Li 2020 Zenodo record does not contain {LI2020_ZIP_NAME!r}")
        url = info.get("links", {}).get("self")
        if not url:
            raise RuntimeError(f"Li 2020 file {LI2020_ZIP_NAME!r} has no download URL")
        with urllib.request.urlopen(url, timeout=180) as response, open(zip_path, "wb") as handle:
            handle.write(response.read())
    nc_path = raw_dir / "Bioenergy_crop_yields.nc"
    if not nc_path.exists():
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(raw_dir)
    if not nc_path.exists():
        raise FileNotFoundError(f"{zip_path} did not extract Bioenergy_crop_yields.nc")
    return nc_path


def _mask_id_to_m49(dict_v3_path: str) -> Dict[int, str]:
    region = pd.read_excel(dict_v3_path, sheet_name="region")
    region.columns = [str(c).strip() for c in region.columns]
    out: Dict[int, str] = {}
    for _, row in region.iterrows():
        try:
            mask_id = int(float(row.get("Region_maskID")))
        except Exception:
            continue
        m49 = _norm_m49(row.get("M49_Country_Code") or row.get("M49 Code") or row.get("Country"))
        if m49:
            out[mask_id] = m49
    return out


def build_li2020_country_yields(
    *,
    yield_nc_path: str,
    country_mask_nc_path: str,
    dict_v3_path: str,
    feedstocks: Iterable[str],
) -> Tuple[Dict[Tuple[str, str], float], pd.DataFrame, pd.DataFrame]:
    try:
        import xarray as xr
    except Exception as exc:
        return {}, pd.DataFrame(), pd.DataFrame([{
            "issue": "xarray_unavailable_for_li2020",
            "message": str(exc),
        }])

    yield_path = Path(yield_nc_path)
    mask_path = Path(country_mask_nc_path)
    if not yield_path.exists():
        return {}, pd.DataFrame(), pd.DataFrame([{"issue": "missing_li2020_yield_nc", "path": str(yield_path)}])
    if not mask_path.exists():
        return {}, pd.DataFrame(), pd.DataFrame([{"issue": "missing_country_mask_nc", "path": str(mask_path)}])

    id_to_m49 = _mask_id_to_m49(dict_v3_path)
    if not id_to_m49:
        return {}, pd.DataFrame(), pd.DataFrame([{"issue": "empty_region_maskid_mapping"}])

    yds = xr.open_dataset(yield_path)
    mds = xr.open_dataset(mask_path)
    mask_var = None
    for candidate in ["id1", "Band1"]:
        if candidate in mds.data_vars:
            mask_var = candidate
            break
    if mask_var is None:
        for candidate in mds.data_vars:
            if candidate.lower() != "crs":
                mask_var = candidate
                break
    if mask_var is None:
        return {}, pd.DataFrame(), pd.DataFrame([{"issue": "no_country_mask_variable", "path": str(mask_path)}])

    mask_da = mds[mask_var]
    if "lat" not in mask_da.dims or "lon" not in mask_da.dims:
        return {}, pd.DataFrame(), pd.DataFrame([{"issue": "country_mask_missing_lat_lon", "var": mask_var}])
    mask_da = mask_da.sortby("lat")
    id_on_yield = mask_da.interp(lat=yds["lat"], lon=yds["lon"], method="nearest")
    id_raw = np.asarray(id_on_yield.values, dtype=float)
    id_vals = np.where(np.isfinite(id_raw), np.rint(id_raw), 0).astype("int32")
    lat_vals = np.asarray(yds["lat"].values, dtype=float)
    weights_2d = np.cos(np.deg2rad(lat_vals))[:, None] * np.ones((1, len(yds["lon"])))
    max_id = int(np.nanmax(id_vals)) if id_vals.size else 0
    rows: List[Dict[str, Any]] = []
    lookup: Dict[Tuple[str, str], float] = {}
    diagnostics: List[Dict[str, Any]] = []
    for feedstock in feedstocks:
        spec = ENERGY_CROP_DEFAULTS.get(feedstock)
        if not spec:
            diagnostics.append({"issue": "unknown_energy_crop_feedstock", "feedstock": feedstock})
            continue
        var = spec.get("li2020_var")
        if not var or var not in yds.data_vars:
            diagnostics.append({"issue": "missing_li2020_variable", "feedstock": feedstock, "variable": var})
            continue
        vals = np.asarray(yds[var].values, dtype=float)
        valid = np.isfinite(vals) & np.isfinite(weights_2d) & (id_vals > 0)
        if not valid.any():
            diagnostics.append({"issue": "no_valid_li2020_cells", "feedstock": feedstock, "variable": var})
            continue
        flat_id = id_vals[valid]
        flat_w = weights_2d[valid]
        flat_y = vals[valid]
        weighted_sum = np.bincount(flat_id, weights=flat_y * flat_w, minlength=max_id + 1)
        weight_sum = np.bincount(flat_id, weights=flat_w, minlength=max_id + 1)
        for mask_id, m49 in id_to_m49.items():
            if mask_id >= len(weight_sum) or weight_sum[mask_id] <= 0:
                continue
            y_val = float(weighted_sum[mask_id] / weight_sum[mask_id])
            if not np.isfinite(y_val) or y_val <= 0:
                continue
            lookup[(m49, feedstock)] = y_val
            rows.append({
                "M49_Country_Code": m49,
                "feedstock": feedstock,
                "li2020_variable": var,
                "yield_tdm_per_ha": y_val,
                "source": f"{LI2020_ZENODO_DOI}; {LI2020_PAPER_DOI}",
                "notes": "Area-weighted country mean over model LUH2 country mask; weights use cos(latitude).",
            })
    return lookup, pd.DataFrame(rows), pd.DataFrame(diagnostics)


def _load_eligible_land_mask_csv(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Eligible land mask CSV not found: {p}")
    df = pd.read_csv(p, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    aliases = {
        "M49_Country_Code": ["m49", "country_code", "region"],
        "feedstock": ["biomass_type", "feedstock_name"],
        "eligible_land_area_ha": ["eligible_area_ha", "available_land_ha", "land_available_ha"],
        "source": ["source_url", "reference"],
        "notes": ["comment"],
    }
    for target, candidates in aliases.items():
        if target not in df.columns:
            for c in candidates:
                if c in df.columns:
                    df[target] = df[c]
                    break
    if "M49_Country_Code" not in df.columns or "eligible_land_area_ha" not in df.columns:
        raise ValueError("Eligible land mask CSV must contain M49_Country_Code and eligible_land_area_ha")
    if "feedstock" not in df.columns:
        df["feedstock"] = ""
    if "source" not in df.columns:
        df["source"] = ""
    if "notes" not in df.columns:
        df["notes"] = ""
    df["M49_Country_Code"] = df["M49_Country_Code"].apply(_norm_m49)
    df["feedstock"] = df["feedstock"].fillna("").astype(str).str.strip()
    df["eligible_land_area_ha"] = pd.to_numeric(df["eligible_land_area_ha"], errors="coerce").fillna(0.0)
    df = df[df["M49_Country_Code"].ne("") & df["eligible_land_area_ha"].gt(0.0)].copy()
    return df


def _eligible_land_lookup(mask_df: pd.DataFrame) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if mask_df.empty:
        return out
    grouped = mask_df.groupby(["M49_Country_Code", "feedstock"], dropna=False, as_index=False).agg({
        "eligible_land_area_ha": "sum",
        "source": lambda x: "; ".join(sorted({str(v).strip() for v in x if str(v).strip()})),
        "notes": lambda x: "; ".join(sorted({str(v).strip() for v in x if str(v).strip()})),
    })
    for _, row in grouped.iterrows():
        out[(str(row["M49_Country_Code"]), str(row["feedstock"]).strip())] = {
            "eligible_land_area_ha": float(row["eligible_land_area_ha"]),
            "source": str(row.get("source") or ""),
            "notes": str(row.get("notes") or ""),
        }
    return out


def build_energy_crop_constraints(
    *,
    dict_v3_path: str,
    land_cover_xlsx: str,
    scenario: str,
    grassland_share: float,
    base_year: int,
    future_years: Iterable[int],
    feedstocks: Optional[Iterable[str]] = None,
    yield_lookup: Optional[Dict[Tuple[str, str], float]] = None,
    eligible_mask_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    has_external_mask = eligible_mask_df is not None and not eligible_mask_df.empty
    if grassland_share <= 0 and not has_external_mask:
        return pd.DataFrame(columns=RESOURCE_COLUMNS)
    _, m49_to_name, _ = _country_name_to_m49(dict_v3_path)
    land = _load_luh2_land_cover(land_cover_xlsx)
    year_col = f"Y{int(base_year)}"
    if year_col not in land.columns:
        raise ValueError(f"{land_cover_xlsx} sheet LUH2 missing {year_col}")
    grass = land[
        land["Land cover"].fillna("").astype(str).str.strip().str.lower().eq("grassland")
    ].copy()
    grass[year_col] = pd.to_numeric(grass[year_col], errors="coerce").fillna(0.0)
    years = sorted({int(y) for y in future_years if int(y) > int(base_year)})
    feedstock_names = list(feedstocks or ENERGY_CROP_DEFAULTS.keys())
    eligible_lookup = _eligible_land_lookup(eligible_mask_df if eligible_mask_df is not None else pd.DataFrame())
    rows: List[Dict[str, Any]] = []
    for _, row in grass.iterrows():
        m49 = _norm_m49(row.get("M49_Country_Code"))
        if not m49:
            continue
        grassland_ha = max(0.0, float(row.get(year_col) or 0.0))
        for feedstock in feedstock_names:
            spec = ENERGY_CROP_DEFAULTS.get(feedstock, ENERGY_CROP_DEFAULTS["miscanthus_energy_crop"])
            mask_info = (
                eligible_lookup.get((m49, feedstock))
                or eligible_lookup.get((m49, ""))
                or {}
            )
            if mask_info:
                eligible_base = float(mask_info["eligible_land_area_ha"])
                eligible_source = mask_info.get("source") or "ESA WorldCover/WDPA/GAEZ eligible-land mask CSV"
                eligible_notes = mask_info.get("notes") or "Eligible land from preprocessed ESA WorldCover/WDPA/GAEZ mask."
            else:
                eligible_base = grassland_ha * max(float(grassland_share), 0.0)
                eligible_source = "Land_cover_base_refill.xlsx"
                eligible_notes = (
                    f"Fallback eligible land = {grassland_share:.4g} * {base_year} LUH2 grassland. "
                    "Replace with ESA WorldCover/WDPA/GAEZ mask CSV for final runs."
                )
            if eligible_base <= 0:
                continue
            yield_val = None
            if yield_lookup:
                yield_val = yield_lookup.get((m49, feedstock))
            if yield_val is None or not np.isfinite(float(yield_val)) or float(yield_val) <= 0.0:
                yield_val = float(spec["yield_tdm_per_ha"])
                yield_source = ENERGY_CROP_YIELD_SOURCE
                yield_note = spec.get("notes", "")
            else:
                yield_source = f"{LI2020_ZENODO_DOI}; {LI2020_PAPER_DOI}"
                yield_note = "Li 2020 0.5 degree gridded yield aggregated to model country mask."
            for year in years:
                rows.append({
                    "scenario": scenario,
                    "M49_Country_Code": m49,
                    "country_name": m49_to_name.get(m49, str(row.get("Region_label_new") or "")),
                    "year": int(year),
                    "feedstock": feedstock,
                    "feedstock_category": "dedicated_energy_crop",
                    "parent_commodity": "",
                    "resource_available_t": np.nan,
                    "resource_available_tdm": np.nan,
                    "competing_use_t": np.nan,
                    "competing_use_tdm": np.nan,
                    "sustainable_fraction": 1.0,
                    "yield_tdm_per_ha": float(yield_val),
                    "eligible_land_area_ha": eligible_base,
                    "ghg_direct_kgco2e_per_tdm": np.nan,
                    "ghg_soil_kgco2e_per_tdm": np.nan,
                    "ghg_avoided_kgco2e_per_tdm": np.nan,
                    "fossil_displacement_kgco2e_per_tj": np.nan,
                    "beccs_capture_kgco2_per_tdm": np.nan,
                    **{col: np.nan for col in RESIDUE_QUALITY_COLUMNS},
                    "residue_quality_match": "",
                    "source": f"{eligible_source}; {yield_source}",
                    "notes": (
                        f"{eligible_notes} {yield_note}"
                    ),
                })
    return pd.DataFrame(rows, columns=RESOURCE_COLUMNS)


def _parse_year_list(raw: str) -> List[int]:
    years: List[int] = []
    for part in str(raw).replace(";", ",").split(","):
        part = part.strip()
        if part:
            years.append(int(part))
    return sorted(set(years))


def _leaf_country_lookup(dict_v3_path: str) -> Dict[str, str]:
    region = pd.read_excel(dict_v3_path, sheet_name="region")
    region.columns = [str(c).strip() for c in region.columns]
    work = region.copy()
    work["M49_Country_Code"] = work["M49_Country_Code"].apply(_norm_m49)
    iso3 = work.get("ISO3 Code", pd.Series("", index=work.index)).fillna("").astype(str).str.strip()
    agg2 = work.get("Region_agg2", pd.Series("", index=work.index)).fillna("").astype(str).str.strip()
    agg4 = work.get("Region_agg4", pd.Series("", index=work.index)).fillna("").astype(str).str.strip()
    work = work[
        work["M49_Country_Code"].ne("")
        & iso3.str.match(r"^[A-Z]{3}$", na=False)
        & agg2.ne("no")
        & agg4.ne("no")
    ].copy()
    labels = (
        work.get("Region_label_new", pd.Series("", index=work.index))
        .fillna(work.get("NAME", pd.Series("", index=work.index)))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    return dict(zip(work["M49_Country_Code"].astype(str), labels))


def build_noncrop_feedstock_constraints(
    *,
    profile_csv: str,
    dict_v3_path: str,
    scenario: str,
    base_year: int,
    future_years: Iterable[int],
    resource_multiplier: float,
    cap_mode: str = "observed-use",
    technical_potential_csv: str = "",
    technical_potential_multiplier: float = 5.0,
    technical_missing_policy: str = "observed-use",
    unconstrained_cap_tdm: float = 1e15,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build national carrier-level biomass caps from FAOSTAT baseline use.

    These caps are intentionally conservative: they represent country-level
    baseline observed use converted to dry matter, optionally scaled by
    ``resource_multiplier``. They prevent non-market feedstocks from being
    uncapped while leaving room to replace the table with literature-based
    technical potentials later. Crop-residue rows emitted here are fallback
    caps for FAOSTAT carrier rows that are not covered by OMD crop-specific
    residue feedstocks.
    """
    p = Path(profile_csv)
    if not p.exists():
        return pd.DataFrame(columns=RESOURCE_COLUMNS), pd.DataFrame([{
            "issue": "missing_noncrop_profile_csv",
            "path": str(p),
        }])
    country_names = _leaf_country_lookup(dict_v3_path)
    valid_m49 = set(country_names)
    raw = pd.read_csv(p, low_memory=False)
    raw.columns = [str(c).strip() for c in raw.columns]
    if "M49_Country_Code" not in raw.columns or "year" not in raw.columns:
        raise ValueError(f"{profile_csv} must contain M49_Country_Code and year")
    raw["M49_Country_Code"] = raw["M49_Country_Code"].apply(_norm_m49)
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce")
    available_years = sorted(raw["year"].dropna().astype(int).unique().tolist())
    if not available_years:
        raise ValueError(f"{profile_csv} has no usable year values")
    selected_years = [y for y in available_years if y <= int(base_year)]
    selected_year = max(selected_years) if selected_years else min(available_years)
    work = raw[raw["year"].eq(selected_year) & raw["M49_Country_Code"].isin(valid_m49)].copy()
    rows: List[Dict[str, Any]] = []
    years = [int(y) for y in future_years if int(y) > int(base_year)]
    cap_mode_norm = str(cap_mode or "observed-use").strip().lower().replace("_", "-")
    if cap_mode_norm in {"observed", "observed-use", "observeduse"}:
        cap_mode_norm = "observed-use"
    elif cap_mode_norm in {"technical", "technical-potential", "technicalpotential"}:
        cap_mode_norm = "technical-potential"
    elif cap_mode_norm in {"unconstrained", "none", "off"}:
        cap_mode_norm = "unconstrained"
    else:
        raise ValueError(f"Unsupported noncrop cap mode: {cap_mode}")
    multiplier = max(0.0, float(resource_multiplier))
    technical_multiplier = max(0.0, float(technical_potential_multiplier))
    missing_policy = str(technical_missing_policy or "observed-use").strip().lower().replace("_", "-")
    if missing_policy not in {"proxy", "observed-use", "zero"}:
        raise ValueError(f"Unsupported noncrop technical missing policy: {technical_missing_policy}")
    unconstrained_cap = max(0.0, float(unconstrained_cap_tdm))
    technical_lookup: Dict[Tuple[str, str], float] = {}
    technical_status = "not_used"
    tech_path = Path(str(technical_potential_csv or ""))
    if cap_mode_norm == "technical-potential" and str(technical_potential_csv or "").strip():
        if tech_path.exists():
            tech = pd.read_csv(tech_path, low_memory=False)
            tech.columns = [str(c).strip() for c in tech.columns]
            if {"M49_Country_Code", "feedstock", "resource_available_tdm"}.issubset(tech.columns):
                tech["M49_Country_Code"] = tech["M49_Country_Code"].apply(_norm_m49)
                tech["feedstock"] = tech["feedstock"].astype(str).str.strip()
                tech["resource_available_tdm"] = pd.to_numeric(tech["resource_available_tdm"], errors="coerce").fillna(0.0).clip(lower=0.0)
                grouped = tech.groupby(["M49_Country_Code", "feedstock"], as_index=False)["resource_available_tdm"].sum()
                technical_lookup = {
                    (str(r.M49_Country_Code), str(r.feedstock)): float(r.resource_available_tdm)
                    for r in grouped.itertuples(index=False)
                }
                technical_status = "csv_loaded"
            else:
                technical_status = "csv_missing_required_columns"
        else:
            technical_status = "csv_missing"
    elif cap_mode_norm == "technical-potential":
        technical_status = "proxy_multiplier_no_csv"
    for carrier, spec in NONCROP_FEEDSTOCK_RESOURCE_CAPS.items():
        if carrier not in work.columns:
            continue
        carrier_tj = pd.to_numeric(work[carrier], errors="coerce").fillna(0.0).clip(lower=0.0)
        positive = work.copy()
        positive["_carrier_tj"] = carrier_tj.loc[positive.index]
        lhv = float(spec["lhv_gj_per_tdm"])
        for _, record in positive.iterrows():
            observed_dry_tdm = float(record["_carrier_tj"]) * 1000.0 / max(lhv, 1e-9)
            if cap_mode_norm == "unconstrained":
                dry_tdm = unconstrained_cap
                source = "Sensitivity upper bound: unconstrained non-crop resource cap"
                mode_note = f"Unconstrained sensitivity cap={unconstrained_cap:g} tDM; not a resource estimate."
            elif cap_mode_norm == "technical-potential":
                lookup_key = (str(record["M49_Country_Code"]), str(spec["feedstock"]))
                if lookup_key in technical_lookup:
                    dry_tdm = float(technical_lookup[lookup_key])
                    source = f"Technical potential table: {tech_path}"
                    mode_note = "Country-feedstock technical-potential cap loaded from explicit CSV."
                elif missing_policy == "zero":
                    dry_tdm = 0.0
                    source = f"Technical potential table missing country-feedstock row: {tech_path}"
                    mode_note = (
                        "Missing country-feedstock technical-potential row; cap set to zero by "
                        "--noncrop-technical-missing-policy zero."
                    )
                elif missing_policy == "observed-use":
                    dry_tdm = observed_dry_tdm * multiplier
                    source = (
                        "Technical potential table missing country-feedstock row; "
                        "fallback to FAOSTAT observed use"
                    )
                    mode_note = (
                        f"Missing country-feedstock technical-potential row; fallback cap = "
                        f"observed-use x resource_multiplier {multiplier:g}."
                    )
                else:
                    dry_tdm = observed_dry_tdm * technical_multiplier * multiplier
                    source = (
                        "Upper-bound sensitivity proxy from FAOSTAT observed use multiplier; "
                        "not used for main results; replace with country-feedstock technical potential CSV"
                    )
                    mode_note = (
                        f"Upper-bound sensitivity proxy cap = observed-use x {technical_multiplier:g} "
                        f"x resource_multiplier {multiplier:g}; not a literature technical-potential estimate."
                    )
            else:
                dry_tdm = observed_dry_tdm * multiplier
                source = (
                    "FAOSTAT Bioenergy country carrier profile; "
                    "https://www.fao.org/statistics/highlights-archive/highlights-detail/"
                    "bioenergy-statistics-1990-2024/en"
                )
                mode_note = (
                    f"Observed-use availability cap from {selected_year} FAOSTAT {carrier} "
                    f"carrier use converted by LHV={lhv:g} GJ/tDM and multiplier={multiplier:g}."
                )
            for year in years:
                rows.append({
                    "scenario": scenario,
                    "M49_Country_Code": record["M49_Country_Code"],
                    "country_name": country_names.get(
                        record["M49_Country_Code"],
                        str(record.get("country_name", "")),
                    ),
                    "year": int(year),
                    "feedstock": spec["feedstock"],
                    "feedstock_category": spec["feedstock_category"],
                    "parent_commodity": "",
                    "resource_available_t": np.nan,
                    "resource_available_tdm": dry_tdm,
                    "competing_use_t": np.nan,
                    "competing_use_tdm": 0.0,
                    "sustainable_fraction": 1.0,
                    "yield_tdm_per_ha": np.nan,
                    "eligible_land_area_ha": np.nan,
                    "ghg_direct_kgco2e_per_tdm": np.nan,
                    "ghg_soil_kgco2e_per_tdm": np.nan,
                    "ghg_avoided_kgco2e_per_tdm": np.nan,
                    "fossil_displacement_kgco2e_per_tj": np.nan,
                    "beccs_capture_kgco2_per_tdm": np.nan,
                    **{col: np.nan for col in RESIDUE_QUALITY_COLUMNS},
                    "residue_quality_match": "",
                    "source": source,
                    "notes": (
                        f"noncrop_cap_mode={cap_mode_norm}. {mode_note}"
                    ),
                })
    diagnostics = pd.DataFrame([
        {
            "issue": "faostat_carrier_resource_caps_built",
            "profile_csv": str(p),
            "base_year_requested": int(base_year),
            "baseline_profile_year": int(selected_year),
            "resource_multiplier": multiplier,
            "noncrop_cap_mode": cap_mode_norm,
            "technical_potential_csv": str(tech_path) if str(technical_potential_csv or "").strip() else "",
            "technical_potential_status": technical_status,
            "technical_potential_multiplier": technical_multiplier,
            "technical_missing_policy": missing_policy,
            "unconstrained_cap_tdm": unconstrained_cap,
            "rows": len(rows),
            "feedstocks": ",".join(sorted({str(r["feedstock"]) for r in rows})),
        }
    ])
    return pd.DataFrame(rows, columns=RESOURCE_COLUMNS), diagnostics


def write_outputs(
    constraints: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    output_dir: Path,
    write_input: bool,
    overwrite: bool,
    extra_outputs: Optional[Dict[str, pd.DataFrame]] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    constraints_path = output_dir / "bioenergy_resource_constraints.csv"
    diagnostics_path = output_dir / "bioenergy_resource_constraints_diagnostics.csv"
    constraints.to_csv(constraints_path, index=False, encoding="utf-8-sig")
    diagnostics.to_csv(diagnostics_path, index=False, encoding="utf-8-sig")
    print(f"wrote {constraints_path} rows={len(constraints)}")
    print(f"wrote {diagnostics_path} rows={len(diagnostics)}")
    for name, frame in (extra_outputs or {}).items():
        out_path = output_dir / name
        frame.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"wrote {out_path} rows={len(frame)}")
    if write_input:
        input_dir = Path(get_input_base()) / "Bioenergy"
        input_dir.mkdir(parents=True, exist_ok=True)
        input_path = input_dir / "bioenergy_resource_constraints.csv"
        if input_path.resolve() == constraints_path.resolve():
            print(f"input path already written at {input_path} rows={len(constraints)}")
            return
        if input_path.exists() and not overwrite:
            raise FileExistsError(
                f"{input_path} already exists. Re-run with --overwrite-input to replace derived file."
            )
        constraints.to_csv(input_path, index=False, encoding="utf-8-sig")
        print(f"wrote {input_path} rows={len(constraints)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_input_dir = Path(get_input_base()) / "Bioenergy"
    parser.add_argument("--output-dir", default=str(default_input_dir))
    parser.add_argument("--scenario", default="default")
    parser.add_argument("--profile-csv", default=str(default_input_dir / "bioenergy_country_profiles.csv"))
    parser.add_argument("--download-omd", action="store_true")
    parser.add_argument("--residue-sustainable-fraction", type=float, default=0.30)
    parser.add_argument("--residue-competing-use-fraction", type=float, default=0.0)
    parser.add_argument("--include-energy-crops", action="store_true")
    parser.add_argument("--energy-crop-grassland-share", type=float, default=0.01)
    parser.add_argument(
        "--energy-crop-feedstocks",
        default=",".join(ENERGY_CROP_DEFAULTS.keys()),
        help="Comma-separated dedicated energy crop feedstocks to emit.",
    )
    parser.add_argument(
        "--energy-crop-yield-mode",
        choices=["li2020", "defaults"],
        default="li2020",
        help="Use Li 2020 gridded yields aggregated by country, or fixed defaults.",
    )
    parser.add_argument("--li2020-yield-nc", default="")
    parser.add_argument("--country-mask-nc", default="")
    parser.add_argument(
        "--eligible-land-mask-csv",
        default="",
        help=(
            "Optional preprocessed country/feedstock eligible-land mask CSV, "
            "e.g. derived from ESA WorldCover, WDPA and GAEZ."
        ),
    )
    parser.add_argument("--require-eligible-land-mask", action="store_true")
    parser.add_argument("--include-noncrop-feedstocks", action="store_true")
    parser.add_argument("--noncrop-resource-base-year", type=int, default=2023)
    parser.add_argument("--noncrop-resource-multiplier", type=float, default=1.0)
    parser.add_argument("--noncrop-cap-mode", choices=["observed-use", "technical-potential", "unconstrained"], default="observed-use")
    parser.add_argument(
        "--noncrop-technical-potential-csv",
        default=str(default_input_dir / "bioenergy_noncrop_resource_availability_country.csv"),
    )
    parser.add_argument("--noncrop-technical-potential-multiplier", type=float, default=5.0)
    parser.add_argument(
        "--noncrop-technical-missing-policy",
        choices=["proxy", "observed-use", "zero"],
        default="observed-use",
        help=(
            "When --noncrop-cap-mode technical-potential and the explicit CSV lacks a "
            "country-feedstock row: use upper-bound proxy sensitivity, observed-use only, "
            "or zero cap. Main results should use observed-use or zero, not proxy."
        ),
    )
    parser.add_argument("--noncrop-unconstrained-cap-tdm", type=float, default=1e15)
    parser.add_argument("--base-year", type=int, default=2020)
    parser.add_argument("--future-years", default="2030,2050,2080")
    parser.add_argument("--write-input", action="store_true")
    parser.add_argument("--overwrite-input", action="store_true")
    args = parser.parse_args()

    paths = DataPaths()
    out_dir = Path(args.output_dir)
    raw_dir = out_dir / "raw_omd"
    omd = _read_omd_crop_residues(raw_dir, download=bool(args.download_omd))
    quality_df = _read_omd_residue_quality(raw_dir, download=bool(args.download_omd))
    quality_lookup, quality_diag = build_residue_quality_lookup(quality_df)
    residues, residue_diag = build_crop_residue_constraints(
        omd,
        dict_v3_path=paths.dict_v3_path,
        scenario=args.scenario,
        sustainable_fraction=args.residue_sustainable_fraction,
        competing_use_fraction=args.residue_competing_use_fraction,
        residue_quality_lookup=quality_lookup,
    )
    frames = [residues]
    diagnostics_frames = [residue_diag, quality_diag]
    extra_outputs: Dict[str, pd.DataFrame] = {}
    years = _parse_year_list(args.future_years)
    if args.include_energy_crops:
        feedstocks = [x.strip() for x in str(args.energy_crop_feedstocks).split(",") if x.strip()]
        yield_lookup: Dict[Tuple[str, str], float] = {}
        if args.energy_crop_yield_mode == "li2020":
            li_nc = Path(args.li2020_yield_nc) if args.li2020_yield_nc else _download_li2020_yield_map(out_dir / "raw_li2020")
            country_mask = (
                Path(args.country_mask_nc)
                if args.country_mask_nc
                else Path(get_input_base()) / "Land" / "LUH2_GCB2019" / "data" / "mask_LUH2_025d.nc"
            )
            yield_lookup, yield_df, yield_diag = build_li2020_country_yields(
                yield_nc_path=str(li_nc),
                country_mask_nc_path=str(country_mask),
                dict_v3_path=paths.dict_v3_path,
                feedstocks=feedstocks,
            )
            if not yield_df.empty:
                extra_outputs["bioenergy_energy_crop_country_yields.csv"] = yield_df
            diagnostics_frames.append(yield_diag)
        if args.require_eligible_land_mask and not args.eligible_land_mask_csv:
            raise ValueError(
                "--require-eligible-land-mask was set, but --eligible-land-mask-csv was not provided. "
                "Provide the ESA WorldCover/WDPA/GAEZ preprocessed mask before interpreting "
                "high-bioenergy LUC results."
            )
        eligible_mask_df = _load_eligible_land_mask_csv(args.eligible_land_mask_csv)
        if args.eligible_land_mask_csv:
            diagnostics_frames.append(pd.DataFrame([{
                "issue": "eligible_land_mask_csv_loaded",
                "path": args.eligible_land_mask_csv,
                "rows": len(eligible_mask_df),
            }]))
        else:
            diagnostics_frames.append(pd.DataFrame([{
                "issue": "eligible_land_mask_fallback",
                "message": "No ESA WorldCover/WDPA/GAEZ mask CSV provided; using LUH2 grassland share fallback.",
            }]))
        energy = build_energy_crop_constraints(
            dict_v3_path=paths.dict_v3_path,
            land_cover_xlsx=paths.land_cover_base_xlsx,
            scenario=args.scenario,
            grassland_share=float(args.energy_crop_grassland_share),
            base_year=int(args.base_year),
            future_years=years,
            feedstocks=feedstocks,
            yield_lookup=yield_lookup,
            eligible_mask_df=eligible_mask_df,
        )
        frames.append(energy)
    if args.include_noncrop_feedstocks:
        noncrop, noncrop_diag = build_noncrop_feedstock_constraints(
            profile_csv=args.profile_csv,
            dict_v3_path=paths.dict_v3_path,
            scenario=args.scenario,
            base_year=int(args.noncrop_resource_base_year),
            future_years=years,
            resource_multiplier=float(args.noncrop_resource_multiplier),
            cap_mode=str(args.noncrop_cap_mode),
            technical_potential_csv=str(args.noncrop_technical_potential_csv or ""),
            technical_potential_multiplier=float(args.noncrop_technical_potential_multiplier),
            technical_missing_policy=str(args.noncrop_technical_missing_policy),
            unconstrained_cap_tdm=float(args.noncrop_unconstrained_cap_tdm),
        )
        frames.append(noncrop)
        diagnostics_frames.append(noncrop_diag)
    constraints = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RESOURCE_COLUMNS)
    diagnostics = (
        pd.concat([d for d in diagnostics_frames if isinstance(d, pd.DataFrame) and not d.empty], ignore_index=True)
        if diagnostics_frames else pd.DataFrame()
    )
    constraints = constraints[RESOURCE_COLUMNS].sort_values(
        ["scenario", "M49_Country_Code", "year", "feedstock"],
        na_position="last",
    )
    write_outputs(
        constraints,
        diagnostics,
        output_dir=out_dir,
        write_input=bool(args.write_input),
        overwrite=bool(args.overwrite_input),
        extra_outputs=extra_outputs,
    )


if __name__ == "__main__":
    main()
