# -*- coding: utf-8 -*-
"""Build country-level non-crop bioenergy resource availability from OMD.

The output is a technical-potential CSV accepted by
`S0_52_prepare_bioenergy_resource_constraints.py --noncrop-technical-potential-csv`.

Source: Organic Matter Database (OMD), Zenodo record 10450921.
The OMD gives national gross residue/by-product availability.  This script
keeps gross availability and writes a conservative `resource_available_tdm`
column in dry-matter equivalent.  Sustainable collection/removal constraints
should still be imposed downstream through scenario-specific fractions/caps.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config_paths import get_input_base, get_src_base


OMD_RECORD_API = "https://zenodo.org/api/records/10450921"
OMD_DOI = "https://doi.org/10.5281/zenodo.10450921"
OMD_PAPER_DOI = "https://doi.org/10.5194/essd-17-369-2025"

REQUIRED_OMD_FILES = [
    "Wood residues.csv",
    "Manure.csv",
    "Agroprocessing residues.csv",
    "Coffee cocoa and oilpalm residues.csv",
    "Sugarcane bagase.csv",
    "Fish processing byproducts.csv",
    "Meat processing residues.csv",
    "Poultry slaughterhouse residues.csv",
]

WOODY_FEEDSTOCKS = {
    "Fuelwood": ("fuelwood_forest_biomass", "forest_biomass"),
    "Charcoal": ("charcoal_woody_biomass", "forest_biomass"),
    "Black liquor": ("black_liquor_forest_industrial_residue", "forest_industrial_residue"),
}

PROCESSING_FEEDSTOCKS = {
    "Biogases": ("biogas_waste_feedstock", "waste_biomass"),
    "Other vegetal material and residues": ("mixed_crop_residue", "crop_residue"),
}

SINGLE_FEEDSTOCKS = {
    "animal_waste_biogas": "animal_waste",
    "bagasse_residue": "crop_residue",
}


def _norm_m49(value: Any) -> str:
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


def _clean_name(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("&", "and").split())


def _download_omd_files(raw_dir: Path, required: Iterable[str]) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(OMD_RECORD_API, timeout=60) as response:
        record = json.load(response)
    files = {f.get("key"): f for f in record.get("files", [])}
    for name in required:
        dest = raw_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        info = files.get(name)
        if not info:
            raise FileNotFoundError(f"OMD record does not contain {name!r}")
        url = info.get("links", {}).get("self")
        if not url:
            raise RuntimeError(f"OMD file {name!r} has no download URL")
        with urllib.request.urlopen(url, timeout=180) as response, open(dest, "wb") as handle:
            handle.write(response.read())


def _country_maps(dict_xlsx: Path) -> Tuple[Dict[int, str], Dict[str, str], Dict[str, str]]:
    region = pd.read_excel(dict_xlsx, sheet_name="region")
    region.columns = [str(c).strip() for c in region.columns]
    area_to_m49: Dict[int, str] = {}
    name_to_m49: Dict[str, str] = {}
    m49_to_name: Dict[str, str] = {}
    for _, row in region.iterrows():
        m49 = _norm_m49(row.get("M49_Country_Code") or row.get("M49 Code") or row.get("Country"))
        if not m49:
            continue
        try:
            area_to_m49[int(float(row.get("Area Code")))] = m49
        except Exception:
            pass
        names = [
            row.get("Region_label_new"),
            row.get("NAME"),
            row.get("Country"),
            row.get("Region_label"),
            row.get("Region_label2"),
        ]
        for name in names:
            clean = _clean_name(name)
            if clean and clean != "no":
                name_to_m49[clean] = m49
        label = str(row.get("Region_label_new") or row.get("NAME") or row.get("Country") or "").strip()
        if label and label != "no":
            m49_to_name[m49] = label
    return area_to_m49, name_to_m49, m49_to_name


def _assign_m49(df: pd.DataFrame, *, area_to_m49: Mapping[int, str], name_to_m49: Mapping[str, str]) -> pd.DataFrame:
    out = df.copy()
    m49 = pd.Series([""] * len(out), index=out.index, dtype="object")
    for col in ["M49_Country_Code", "M49_Code", "Area Code (M49)", "M49 Code"]:
        if col in out.columns:
            candidate = out[col].map(_norm_m49)
            m49 = m49.where(m49.astype(str).str.len() > 0, candidate)
    for col in ["Area Code", "Area Code (FAO)", "Area code", "Area code (FAO)", "Area code (FAO)"]:
        if col in out.columns:
            numeric = pd.to_numeric(out[col], errors="coerce")
            candidate = numeric.map(lambda x: area_to_m49.get(int(x), "") if pd.notna(x) else "")
            m49 = m49.where(m49.astype(str).str.len() > 0, candidate)
    for col in ["Area", "Area/country", "Name_En"]:
        if col in out.columns:
            candidate = out[col].map(lambda x: name_to_m49.get(_clean_name(x), ""))
            m49 = m49.where(m49.astype(str).str.len() > 0, candidate)
    out["M49_Country_Code"] = m49
    return out[out["M49_Country_Code"].astype(str).str.len() > 0].copy()


def _latest_window_sum(
    df: pd.DataFrame,
    *,
    value_col: str,
    year_col: str,
    source_file: str,
    area_to_m49: Mapping[int, str],
    name_to_m49: Mapping[str, str],
    dry_matter_fraction: float,
    tonnes_multiplier: float = 1.0,
    pool: str,
) -> pd.DataFrame:
    work = _assign_m49(df, area_to_m49=area_to_m49, name_to_m49=name_to_m49)
    if work.empty or value_col not in work.columns or year_col not in work.columns:
        return pd.DataFrame()
    work["year"] = pd.to_numeric(work[year_col], errors="coerce")
    work["value_t"] = pd.to_numeric(work[value_col], errors="coerce") * float(tonnes_multiplier)
    work = work.dropna(subset=["year", "value_t"])
    work = work[work["value_t"] > 0].copy()
    if work.empty:
        return pd.DataFrame()
    work["year"] = work["year"].astype(int)
    latest = work.groupby("M49_Country_Code")["year"].transform("max")
    work = work[work["year"] >= latest - 2].copy()
    grouped = (
        work.groupby("M49_Country_Code", as_index=False)
        .agg(
            gross_availability_t=("value_t", "mean"),
            source_year_start=("year", "min"),
            source_year_end=("year", "max"),
            n_source_rows=("value_t", "size"),
        )
    )
    grouped["resource_available_tdm"] = grouped["gross_availability_t"] * float(dry_matter_fraction)
    grouped["source_file"] = source_file
    grouped["dry_matter_fraction"] = float(dry_matter_fraction)
    grouped["allocation_pool"] = pool
    return grouped


def _read_agroprocessing(path: Path, *, area_to_m49: Mapping[int, str], name_to_m49: Mapping[str, str]) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    if "Area" not in raw.columns:
        header_idx = raw.index[raw.iloc[:, 0].astype(str).str.strip().eq("Area")]
        if len(header_idx) > 0:
            header = int(header_idx[0])
            cols = raw.iloc[header].fillna("").astype(str).str.strip().tolist()
            raw = raw.iloc[header + 1 :].copy()
            raw.columns = cols
    value_col = "Total (tonnes)"
    return _latest_window_sum(
        raw,
        value_col=value_col,
        year_col="Year",
        source_file=path.name,
        area_to_m49=area_to_m49,
        name_to_m49=name_to_m49,
        dry_matter_fraction=0.88,
        pool="processing_residue_pool",
    )


def _pool_tables(raw_dir: Path, *, area_to_m49: Mapping[int, str], name_to_m49: Mapping[str, str]) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    tables["wood_residue_pool"] = _latest_window_sum(
        pd.read_csv(raw_dir / "Wood residues.csv", low_memory=False),
        value_col="Residue production (in tonnes)",
        year_col="Year",
        source_file="Wood residues.csv",
        area_to_m49=area_to_m49,
        name_to_m49=name_to_m49,
        dry_matter_fraction=0.85,
        pool="wood_residue_pool",
    )
    tables["manure_pool"] = _latest_window_sum(
        pd.read_csv(raw_dir / "Manure.csv", low_memory=False),
        value_col="Estimated manure production (tonnes DM/Year)",
        year_col="Year",
        source_file="Manure.csv",
        area_to_m49=area_to_m49,
        name_to_m49=name_to_m49,
        dry_matter_fraction=1.0,
        pool="manure_pool",
    )
    tables["bagasse_pool"] = _latest_window_sum(
        pd.read_csv(raw_dir / "Sugarcane bagase.csv", low_memory=False),
        value_col="Bagasse (1000 tonnes)",
        year_col="Year Code",
        source_file="Sugarcane bagase.csv",
        area_to_m49=area_to_m49,
        name_to_m49=name_to_m49,
        dry_matter_fraction=0.50,
        tonnes_multiplier=1000.0,
        pool="bagasse_pool",
    )
    processing_parts = [
        _read_agroprocessing(raw_dir / "Agroprocessing residues.csv", area_to_m49=area_to_m49, name_to_m49=name_to_m49),
        _latest_window_sum(
            pd.read_csv(raw_dir / "Coffee cocoa and oilpalm residues.csv", low_memory=False),
            value_col="Estimated residue production (tonnes/year)",
            year_col="Year",
            source_file="Coffee cocoa and oilpalm residues.csv",
            area_to_m49=area_to_m49,
            name_to_m49=name_to_m49,
            dry_matter_fraction=0.88,
            pool="processing_residue_pool",
        ),
        _latest_window_sum(
            pd.read_csv(raw_dir / "Fish processing byproducts.csv", low_memory=False),
            value_col="By-product production (tons DM/year)",
            year_col="YEAR",
            source_file="Fish processing byproducts.csv",
            area_to_m49=area_to_m49,
            name_to_m49=name_to_m49,
            dry_matter_fraction=1.0,
            pool="processing_residue_pool",
        ),
        _latest_window_sum(
            pd.read_csv(raw_dir / "Meat processing residues.csv", low_memory=False),
            value_col="Estimated residue production (fresh tonnes/year)",
            year_col="Year",
            source_file="Meat processing residues.csv",
            area_to_m49=area_to_m49,
            name_to_m49=name_to_m49,
            dry_matter_fraction=0.30,
            pool="processing_residue_pool",
        ),
        _latest_window_sum(
            pd.read_csv(raw_dir / "Poultry slaughterhouse residues.csv", low_memory=False),
            value_col="Estimated residue production (fresh in tonness/year)",
            year_col="Year",
            source_file="Poultry slaughterhouse residues.csv",
            area_to_m49=area_to_m49,
            name_to_m49=name_to_m49,
            dry_matter_fraction=0.30,
            pool="processing_residue_pool",
        ),
    ]
    processing = pd.concat([x for x in processing_parts if not x.empty], ignore_index=True)
    if not processing.empty:
        processing = (
            processing.groupby("M49_Country_Code", as_index=False)
            .agg(
                gross_availability_t=("gross_availability_t", "sum"),
                resource_available_tdm=("resource_available_tdm", "sum"),
                source_year_start=("source_year_start", "min"),
                source_year_end=("source_year_end", "max"),
                n_source_rows=("n_source_rows", "sum"),
            )
        )
        processing["source_file"] = "Agroprocessing; coffee/cocoa/oilpalm; fish/meat/poultry processing"
        processing["dry_matter_fraction"] = np.nan
        processing["allocation_pool"] = "processing_residue_pool"
    tables["processing_residue_pool"] = processing
    return tables


def _carrier_shares(profile_csv: Path, carriers: Sequence[str]) -> pd.DataFrame:
    profile = pd.read_csv(profile_csv, low_memory=False)
    profile["M49_Country_Code"] = profile["M49_Country_Code"].map(_norm_m49)
    missing = [c for c in carriers if c not in profile.columns]
    for col in missing:
        profile[col] = 0.0
    latest = pd.to_numeric(profile.get("year"), errors="coerce").max()
    if pd.notna(latest):
        profile = profile[pd.to_numeric(profile["year"], errors="coerce") == latest].copy()
    cols = ["M49_Country_Code", *carriers]
    out = profile[cols].copy()
    for col in carriers:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).clip(lower=0.0)
    return out.groupby("M49_Country_Code", as_index=False)[list(carriers)].sum()


def _allocate_pool(
    pool: pd.DataFrame,
    *,
    feedstocks: Mapping[str, Tuple[str, str]],
    shares: pd.DataFrame,
    default_equal: bool,
    m49_to_name: Mapping[str, str],
    notes: str,
) -> pd.DataFrame:
    if pool.empty:
        return pd.DataFrame()
    carriers = list(feedstocks)
    work = pool.merge(shares, on="M49_Country_Code", how="left")
    for col in carriers:
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0).clip(lower=0.0)
    total = work[carriers].sum(axis=1)
    rows: List[Dict[str, Any]] = []
    for _, row in work.iterrows():
        if float(total.loc[row.name]) > 0:
            fractions = {c: float(row[c]) / float(total.loc[row.name]) for c in carriers}
        elif default_equal:
            fractions = {c: 1.0 / len(carriers) for c in carriers}
        else:
            fractions = {c: 0.0 for c in carriers}
        for carrier, (feedstock, category) in feedstocks.items():
            share = fractions[carrier]
            if share <= 0:
                continue
            rows.append({
                "M49_Country_Code": row["M49_Country_Code"],
                "country_name": m49_to_name.get(str(row["M49_Country_Code"]), ""),
                "feedstock": feedstock,
                "feedstock_category": category,
                "resource_available_tdm": float(row["resource_available_tdm"]) * share,
                "gross_availability_t": float(row["gross_availability_t"]) * share,
                "dry_matter_fraction": row.get("dry_matter_fraction", np.nan),
                "allocation_pool": row.get("allocation_pool", ""),
                "allocation_share": share,
                "source_file": row.get("source_file", ""),
                "source_year_start": int(row.get("source_year_start", 0)),
                "source_year_end": int(row.get("source_year_end", 0)),
                "n_source_rows": int(row.get("n_source_rows", 0)),
                "source": f"{OMD_DOI}; {OMD_PAPER_DOI}",
                "notes": notes,
            })
    return pd.DataFrame(rows)


def build_availability(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    input_base = Path(get_input_base())
    raw_dir = Path(args.raw_omd_dir) if args.raw_omd_dir else input_base / "Bioenergy" / "raw_omd"
    if args.download_omd:
        _download_omd_files(raw_dir, REQUIRED_OMD_FILES)
    dict_xlsx = Path(args.dict_xlsx) if args.dict_xlsx else Path(get_src_base()) / "dict_v3.xlsx"
    profile_csv = Path(args.profile_csv) if args.profile_csv else input_base / "Bioenergy" / "bioenergy_country_profiles.csv"
    area_to_m49, name_to_m49, m49_to_name = _country_maps(dict_xlsx)
    pools = _pool_tables(raw_dir, area_to_m49=area_to_m49, name_to_m49=name_to_m49)

    woody_shares = _carrier_shares(profile_csv, list(WOODY_FEEDSTOCKS))
    processing_shares = _carrier_shares(profile_csv, list(PROCESSING_FEEDSTOCKS))
    frames = [
        _allocate_pool(
            pools["wood_residue_pool"],
            feedstocks=WOODY_FEEDSTOCKS,
            shares=woody_shares,
            default_equal=True,
            m49_to_name=m49_to_name,
            notes=(
                "OMD wood residues allocated to woody bioenergy feedstocks by latest "
                "FAOSTAT Bioenergy carrier profile shares. This is forest/industrial residue "
                "availability, not standing-forest harvest potential."
            ),
        ),
        _allocate_pool(
            pools["processing_residue_pool"],
            feedstocks=PROCESSING_FEEDSTOCKS,
            shares=processing_shares,
            default_equal=True,
            m49_to_name=m49_to_name,
            notes=(
                "OMD agro-processing and animal/fish processing by-products allocated to "
                "biogas and mixed vegetal residue feedstocks by latest FAOSTAT Bioenergy "
                "carrier profile shares. Fresh meat/poultry residues use a conservative "
                "0.30 dry-matter conversion."
            ),
        ),
    ]
    manure = pools["manure_pool"].copy()
    if not manure.empty:
        manure["feedstock"] = "animal_waste_biogas"
        manure["feedstock_category"] = SINGLE_FEEDSTOCKS["animal_waste_biogas"]
        manure["country_name"] = manure["M49_Country_Code"].map(m49_to_name).fillna("")
        manure["allocation_share"] = 1.0
        manure["source"] = f"{OMD_DOI}; {OMD_PAPER_DOI}"
        manure["notes"] = "OMD manure production in tonnes dry matter per year."
        frames.append(manure)
    bagasse = pools["bagasse_pool"].copy()
    if not bagasse.empty:
        bagasse["feedstock"] = "bagasse_residue"
        bagasse["feedstock_category"] = SINGLE_FEEDSTOCKS["bagasse_residue"]
        bagasse["country_name"] = bagasse["M49_Country_Code"].map(m49_to_name).fillna("")
        bagasse["allocation_share"] = 1.0
        bagasse["source"] = f"{OMD_DOI}; {OMD_PAPER_DOI}"
        bagasse["notes"] = "OMD sugarcane bagasse gross availability; converted from 1000 tonnes with dry matter fraction 0.50."
        frames.append(bagasse)

    out = pd.concat([frame for frame in frames if frame is not None and not frame.empty], ignore_index=True)
    if out.empty:
        return out, pd.DataFrame([{"issue": "no_rows_built"}])
    out["resource_available_tdm"] = pd.to_numeric(out["resource_available_tdm"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out = out[out["resource_available_tdm"] > 0].copy()
    ordered = [
        "M49_Country_Code",
        "country_name",
        "feedstock",
        "feedstock_category",
        "resource_available_tdm",
        "gross_availability_t",
        "dry_matter_fraction",
        "allocation_pool",
        "allocation_share",
        "source_file",
        "source_year_start",
        "source_year_end",
        "n_source_rows",
        "source",
        "notes",
    ]
    out = (
        out[ordered]
        .sort_values(["M49_Country_Code", "feedstock"], kind="mergesort")
        .reset_index(drop=True)
    )
    diag = (
        out.groupby(["feedstock", "allocation_pool"], as_index=False)
        .agg(
            countries=("M49_Country_Code", "nunique"),
            resource_available_tdm=("resource_available_tdm", "sum"),
            source_year_start=("source_year_start", "min"),
            source_year_end=("source_year_end", "max"),
        )
        .sort_values(["feedstock", "allocation_pool"], kind="mergesort")
    )
    return out, diag


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_input = Path(get_input_base()) / "Bioenergy"
    parser.add_argument("--raw-omd-dir", default=str(default_input / "raw_omd"))
    parser.add_argument("--output-csv", default=str(default_input / "bioenergy_noncrop_resource_availability_country.csv"))
    parser.add_argument("--diagnostics-csv", default=str(default_input / "bioenergy_noncrop_resource_availability_diagnostics.csv"))
    parser.add_argument("--profile-csv", default=str(default_input / "bioenergy_country_profiles.csv"))
    parser.add_argument("--dict-xlsx", default=str(Path(get_src_base()) / "dict_v3.xlsx"))
    parser.add_argument("--download-omd", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out, diag = build_availability(args)
    output_csv = Path(args.output_csv)
    diagnostics_csv = Path(args.diagnostics_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    diag.to_csv(diagnostics_csv, index=False, encoding="utf-8-sig")
    print(f"wrote {output_csv} rows={len(out)}")
    print(f"wrote {diagnostics_csv} rows={len(diag)}")


if __name__ == "__main__":
    main()
