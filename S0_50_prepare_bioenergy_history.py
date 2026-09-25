# -*- coding: utf-8 -*-
"""Prepare FAOSTAT Bioenergy history and country carrier profiles."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from config_paths import get_input_base
from S2_0_load_data import DataPaths
from S3_3_bioenergy import normalize_m49
from runtime_data_cache import read_csv_cached


LEAF_BIOFUELS = [
    "Fuelwood",
    "Charcoal",
    "Other vegetal material and residues",
    "Bagasse",
    "Black liquor",
    "Animal waste",
    "Biogasoline",
    "Biodiesel",
    "Other liquid biofuels",
    "Bio jet kerosene",
    "Biogases",
]


def _column(df: pd.DataFrame, names: Iterable[str]) -> str:
    by_lower = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        found = by_lower.get(str(name).strip().lower())
        if found is not None:
            return found
    raise KeyError(f"Missing required column; expected one of {list(names)}")


def prepare_history(raw_path: str) -> pd.DataFrame:
    raw = read_csv_cached(raw_path, low_memory=False)
    raw.columns = [str(c).strip() for c in raw.columns]
    area_col = _column(raw, ["Area"])
    m49_col = _column(raw, ["M49_Country_Code", "Area Code (M49)", "M49"])
    item_col = _column(raw, ["Item"])
    element_col = _column(raw, ["Element"])
    unit_col = _column(raw, ["Unit"])
    year_cols = [
        c for c in raw.columns
        if str(c).startswith("Y") and str(c)[1:].isdigit()
    ]
    if not year_cols:
        raise ValueError("No FAOSTAT Yxxxx columns found")
    keep = [area_col, m49_col, item_col, element_col, unit_col, *year_cols]
    long = raw[keep].melt(
        id_vars=[area_col, m49_col, item_col, element_col, unit_col],
        value_vars=year_cols,
        var_name="year",
        value_name="value",
    )
    long = long.rename(
        columns={
            area_col: "country_name",
            m49_col: "M49_Country_Code",
            item_col: "carrier",
            element_col: "element",
            unit_col: "unit",
        }
    )
    long["M49_Country_Code"] = long["M49_Country_Code"].apply(normalize_m49)
    long["year"] = pd.to_numeric(long["year"].str[1:], errors="coerce")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["year", "value"])
    long["year"] = long["year"].astype(int)
    long["carrier"] = long["carrier"].astype(str).str.strip()
    long["element"] = long["element"].astype(str).str.strip().str.lower()
    long["unit"] = long["unit"].astype(str).str.strip()
    long = long[long["M49_Country_Code"].ne("")]
    long = long[long["carrier"].isin(LEAF_BIOFUELS)]
    long["production_tj"] = np.where(
        long["element"].str.contains("production", na=False),
        long["value"],
        0.0,
    )
    long["final_consumption_tj"] = np.where(
        long["element"].str.contains("consumption", na=False),
        long["value"],
        0.0,
    )
    out = long.groupby(
        ["M49_Country_Code", "country_name", "year", "carrier", "unit"],
        as_index=False,
    )[["production_tj", "final_consumption_tj"]].sum()
    out["source"] = "FAOSTAT Bioenergy"
    out["is_leaf_carrier"] = True
    return out


def _profile_class(row: pd.Series) -> str:
    dominant = str(row.get("dominant_carrier", ""))
    if dominant in {"Fuelwood", "Charcoal", "Animal waste"}:
        return "traditional_solid"
    if dominant in {"Black liquor"}:
        return "industrial_forest"
    if dominant in {"Bagasse"}:
        return "sugar_bagasse"
    if dominant in {"Biogasoline"}:
        return "ethanol_liquid"
    if dominant in {"Biodiesel", "Other liquid biofuels", "Bio jet kerosene"}:
        return "biodiesel_liquid"
    if dominant in {"Biogases"}:
        return "biogas"
    if dominant in {"Other vegetal material and residues"}:
        return "agricultural_residue"
    return "mixed"


def build_country_profiles(history: pd.DataFrame, profile_year: int | None = None) -> pd.DataFrame:
    consumption = history.copy()
    if profile_year is None:
        latest = consumption.groupby("M49_Country_Code")["year"].transform("max")
        consumption = consumption[consumption["year"].eq(latest)]
    else:
        consumption = consumption[consumption["year"].eq(int(profile_year))]
    pivot = consumption.pivot_table(
        index=["M49_Country_Code", "country_name", "year"],
        columns="carrier",
        values="final_consumption_tj",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    for carrier in LEAF_BIOFUELS:
        if carrier not in pivot.columns:
            pivot[carrier] = 0.0
    pivot["total_bioenergy_tj"] = pivot[LEAF_BIOFUELS].sum(axis=1)
    shares = pivot[LEAF_BIOFUELS].div(
        pivot["total_bioenergy_tj"].replace(0.0, np.nan),
        axis=0,
    ).fillna(0.0)
    for carrier in LEAF_BIOFUELS:
        pivot[f"share_{carrier}"] = shares[carrier]
    pivot["dominant_carrier"] = shares.idxmax(axis=1)
    pivot["dominant_share"] = shares.max(axis=1)
    pivot["profile_class"] = pivot.apply(_profile_class, axis=1)
    return pivot


def main() -> None:
    paths = DataPaths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=paths.bioenergy_faostat_csv)
    parser.add_argument("--output-dir", default=str(Path(get_input_base()) / "Bioenergy"))
    parser.add_argument("--profile-year", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = prepare_history(args.input)
    profiles = build_country_profiles(history, args.profile_year)
    history.to_csv(
        output_dir / "bioenergy_history_country_carrier.csv",
        index=False,
        encoding="utf-8-sig",
    )
    profiles.to_csv(
        output_dir / "bioenergy_country_profiles.csv",
        index=False,
        encoding="utf-8-sig",
    )
    coverage = history.groupby("year")["M49_Country_Code"].nunique().reset_index(name="countries")
    coverage.to_csv(
        output_dir / "bioenergy_history_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(
        f"prepared rows={len(history)} profiles={len(profiles)} "
        f"years={history['year'].min()}-{history['year'].max()}"
    )


if __name__ == "__main__":
    main()
