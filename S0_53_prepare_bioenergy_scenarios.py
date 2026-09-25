# -*- coding: utf-8 -*-
"""Prepare low/medium/high bioenergy scenario input tables.

The global primary-energy targets come from AR6/SCI scenario-database
percentiles for ``Primary Energy|Biomass``. The default trajectory uses the
AR6 C1-C2 subset: low=P10, medium=P50, high=P90. The energy-crop land share is
the corresponding percentile of
``Land Cover|Cropland|Energy Crops / Land Cover|Cropland``.

FAOSTAT Bioenergy is used only as a historical country/R5 allocation profile:
it does not set the future global total. Within each R5 region, the
non-dedicated-biomass part follows baseline country carrier shares. Dedicated
energy-crop demand is derived from the scenario cropland-share target and
Li et al. country-level energy-crop yields prepared by S0_52.

Outputs are written under Code/input/Bioenergy for S3_3_bioenergy.py.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_INPUT_DIR = CODE_ROOT / "input" / "Bioenergy"

DEFAULT_PROFILE_CSV = DEFAULT_INPUT_DIR / "bioenergy_country_profiles.csv"
DEFAULT_RESOURCE_CSV = DEFAULT_INPUT_DIR / "bioenergy_resource_constraints.csv"
DEFAULT_YIELD_CSV = DEFAULT_INPUT_DIR / "bioenergy_energy_crop_country_yields.csv"
DEFAULT_DICT_XLSX = CODE_ROOT / "src" / "dict_v3.xlsx"
DEFAULT_LAND_XLSX = CODE_ROOT / "input" / "Land" / "Land_cover_base_refill.xlsx"
DEFAULT_OUT_DIR = DEFAULT_INPUT_DIR

EJ_TO_TJ = 1_000_000.0
ENERGY_CROP_LHV_GJ_PER_TDM = 18.0
GENERIC_LHV_GJ_PER_TDM = 17.0
BASE_YEAR_DEFAULT = 2023
TARGET_YEAR_DEFAULT = 2050
DEFAULT_YEARS = (2030, 2050, 2080)

GLOBAL_TRAJECTORY_SOURCE = (
    "AR6 Scenarios Database v1.1, World rows merged with metadata, "
    "Category in C1-C2; cross-checked against SCI-2025 1.5D."
)

TRAJECTORY_PERCENTILES = {
    "low_bioenergy": "p10",
    "medium_bioenergy": "p50",
    "high_bioenergy": "p90",
}

AR6_C1C2_TRAJECTORY = {
    "primary_bioenergy_ej_yr": {
        2030: {"p10": 57.84307531, "p50": 69.34558035, "p90": 92.28613997},
        2050: {"p10": 74.28481046, "p50": 111.76589885, "p90": 194.78514399},
        2080: {"p10": 100.78975467, "p50": 180.01811973, "p90": 264.34115},
        2100: {"p10": 105.81151, "p50": 207.99275754, "p90": 286.58945},
    },
    "energy_crops_pct_cropland": {
        2030: {"p10": 0.0, "p50": 1.67106295, "p90": 7.28707808},
        2050: {"p10": 3.76444808, "p50": 8.82916310, "p90": 20.95791325},
        2080: {"p10": 8.85191188, "p50": 16.81126026, "p90": 30.60448916},
        2100: {"p10": 8.23642225, "p50": 19.40640134, "p90": 35.89364547},
    },
}

AR6_C1C2_PRIMARY_FEEDSTOCK_SPLIT_EJ = {
    "dedicated_energy_crop": {
        2030: {"p10": 0.0, "p50": 8.353, "p90": 20.752},
        2050: {"p10": 9.718, "p50": 37.167, "p90": 86.809},
        2080: {"p10": 31.913, "p50": 59.340, "p90": 121.125},
        2100: {"p10": 37.764, "p50": 75.094, "p90": 144.166},
    },
    "crop_residue": {
        2030: {"p10": 5.748, "p50": 28.190, "p90": 56.692},
        2050: {"p10": 10.371, "p50": 40.968, "p90": 74.634},
        2080: {"p10": 15.910, "p50": 46.496, "p90": 71.205},
        2100: {"p10": 17.833, "p50": 46.961, "p90": 59.433},
    },
    "first_generation_crop": {
        2030: {"p10": 5.726, "p50": 6.188, "p90": 8.729},
        2050: {"p10": 3.457, "p50": 6.132, "p90": 6.929},
        2080: {"p10": 0.510, "p50": 5.965, "p90": 6.319},
        2100: {"p10": 0.451, "p50": 5.968, "p90": 6.307},
    },
}

SCI_AGRICULTURAL_BIOENERGY_DEMAND_MTDM = {
    # SCI 1.5D sheet; units are million t dry matter per year.
    "Agricultural Demand|Crops|Bioenergy": {
        2030: {"p10": 366.187, "p50": 593.970, "p90": 1319.095},
        2050: {"p10": 951.595, "p50": 2070.739, "p90": 5205.186},
        2080: {"p10": 1098.881, "p50": 2894.078, "p90": 9681.477},
        2100: {"p10": 997.342, "p50": 2801.619, "p90": 10640.133},
    },
    "Agricultural Demand|Residues|Bioenergy": {
        2030: {"p10": 0.0, "p50": 336.669, "p90": 705.383},
        2050: {"p10": 0.0, "p50": 417.914, "p90": 1123.840},
        2080: {"p10": 0.0, "p50": 402.724, "p90": 1417.550},
        2100: {"p10": 0.0, "p50": 365.204, "p90": 1504.365},
    },
}

SCENARIO_SETTINGS = {
    "low_bioenergy": {
        "label": "Low bioenergy",
        "percentile": "p10",
        "percentile_note": (
            "Low uses the AR6 C1-C2 P10 trajectory for Primary Energy|Biomass "
            "and the matched P10 energy-crop cropland share."
        ),
    },
    "medium_bioenergy": {
        "label": "Medium bioenergy",
        "percentile": "p50",
        "percentile_note": (
            "Medium uses the AR6 C1-C2 P50 trajectory for Primary Energy|Biomass "
            "and the matched P50 energy-crop cropland share."
        ),
    },
    "high_bioenergy": {
        "label": "High bioenergy",
        "percentile": "p90",
        "percentile_note": (
            "High uses the AR6 C1-C2 P90 trajectory. The previous screenshot "
            "400 EJ/yr value is above the 2100 P90 and is no longer the default."
        ),
    },
}

DATABASE_CHECK_ROWS = [
    {
        "database": "SCI-2025 global ensemble",
        "filter": "all data",
        "variable": "Primary Energy|Biomass",
        "year": 2100,
        "p10": 77.9,
        "p50": 150.0,
        "p90": 277.0,
        "p95": 334.0,
        "p99": 449.0,
        "unit": "EJ/yr",
        "notes": "Computed locally from SCI-2025_v1.0_pathways_ensemble_global.xlsx sheet=data.",
    },
    {
        "database": "SCI-2025 global ensemble",
        "filter": "1.5D cross-check",
        "variable": "Primary Energy|Biomass",
        "year": 2030,
        "p10": 55.4,
        "p50": 67.5,
        "p90": 91.4,
        "p95": np.nan,
        "p99": np.nan,
        "unit": "EJ/yr",
        "notes": "Cross-check from SCI-2025_v1.0_pathways_ensemble_global.xlsx sheet=1.5D.",
    },
    {
        "database": "SCI-2025 global ensemble",
        "filter": "1.5D cross-check",
        "variable": "Primary Energy|Biomass",
        "year": 2050,
        "p10": 69.7,
        "p50": 106.0,
        "p90": 189.0,
        "p95": np.nan,
        "p99": np.nan,
        "unit": "EJ/yr",
        "notes": "Cross-check from SCI-2025_v1.0_pathways_ensemble_global.xlsx sheet=1.5D.",
    },
    {
        "database": "SCI-2025 global ensemble",
        "filter": "1.5D cross-check",
        "variable": "Primary Energy|Biomass",
        "year": 2080,
        "p10": 93.6,
        "p50": 167.0,
        "p90": 257.0,
        "p95": np.nan,
        "p99": np.nan,
        "unit": "EJ/yr",
        "notes": "Cross-check from SCI-2025_v1.0_pathways_ensemble_global.xlsx sheet=1.5D.",
    },
    {
        "database": "SCI-2025 global ensemble",
        "filter": "1.5D cross-check",
        "variable": "Primary Energy|Biomass",
        "year": 2100,
        "p10": 99.3,
        "p50": 189.0,
        "p90": 281.0,
        "p95": 304.0,
        "p99": 493.0,
        "unit": "EJ/yr",
        "notes": "Computed locally from SCI-2025_v1.0_pathways_ensemble_global.xlsx sheet=1.5D.",
    },
    {
        "database": "AR6 Scenarios Database v1.1",
        "filter": "all World rows",
        "variable": "Primary Energy|Biomass",
        "year": 2100,
        "p10": 80.4,
        "p50": 168.0,
        "p90": 309.0,
        "p95": 365.0,
        "p99": 459.0,
        "unit": "EJ/yr",
        "notes": "Computed locally from AR6_Scenarios_Database_World_v1.1.csv.",
    },
    {
        "database": "AR6 Scenarios Database v1.1",
        "filter": "C1-C2 default",
        "variable": "Primary Energy|Biomass",
        "year": 2030,
        "p10": 57.84307531,
        "p50": 69.34558035,
        "p90": 92.28613997,
        "p95": 98.79150620,
        "p99": 130.81151975,
        "unit": "EJ/yr",
        "notes": "Default global total trajectory. Computed after merging AR6 metadata category fields.",
    },
    {
        "database": "AR6 Scenarios Database v1.1",
        "filter": "C1-C2 default",
        "variable": "Primary Energy|Biomass",
        "year": 2050,
        "p10": 74.28481046,
        "p50": 111.76589885,
        "p90": 194.78514399,
        "p95": 211.69306167,
        "p99": 256.716765,
        "unit": "EJ/yr",
        "notes": "Default global total trajectory. Computed after merging AR6 metadata category fields.",
    },
    {
        "database": "AR6 Scenarios Database v1.1",
        "filter": "C1-C2 default",
        "variable": "Primary Energy|Biomass",
        "year": 2080,
        "p10": 100.78975467,
        "p50": 180.01811973,
        "p90": 264.34115,
        "p95": 293.51346507,
        "p99": 391.22319944,
        "unit": "EJ/yr",
        "notes": "Default global total trajectory. Computed after merging AR6 metadata category fields.",
    },
    {
        "database": "AR6 Scenarios Database v1.1",
        "filter": "C1-C2 default",
        "variable": "Primary Energy|Biomass",
        "year": 2100,
        "p10": 105.81151,
        "p50": 207.99275754,
        "p90": 286.58945,
        "p95": 305.10373977,
        "p99": 493.61480285,
        "unit": "EJ/yr",
        "notes": "Default global total trajectory. Computed after merging AR6 metadata category fields.",
    },
    {
        "database": "SCI-2025 global ensemble",
        "filter": "all data",
        "variable": "Land Cover|Cropland|Energy Crops / Land Cover|Cropland",
        "year": 2100,
        "p10": 4.0,
        "p50": 14.3,
        "p90": 31.1,
        "p95": 35.4,
        "p99": 43.7,
        "unit": "% of cropland",
        "notes": "Computed locally from SCI-2025_v1.0_pathways_ensemble_global.xlsx sheet=data.",
    },
    {
        "database": "AR6 Scenarios Database v1.1",
        "filter": "C1-C2 default",
        "variable": "Land Cover|Cropland|Energy Crops / Land Cover|Cropland",
        "year": 2030,
        "p10": 0.0,
        "p50": 1.67106295,
        "p90": 7.28707808,
        "p95": 9.36288104,
        "p99": 18.38907343,
        "unit": "% of cropland",
        "notes": "Default energy-crop share trajectory; ratio matched by Model#Scenario.",
    },
    {
        "database": "AR6 Scenarios Database v1.1",
        "filter": "C1-C2 default",
        "variable": "Land Cover|Cropland|Energy Crops / Land Cover|Cropland",
        "year": 2050,
        "p10": 3.76444808,
        "p50": 8.82916310,
        "p90": 20.95791325,
        "p95": 25.31298894,
        "p99": 29.11545775,
        "unit": "% of cropland",
        "notes": "Default energy-crop share trajectory; ratio matched by Model#Scenario.",
    },
    {
        "database": "AR6 Scenarios Database v1.1",
        "filter": "C1-C2 default",
        "variable": "Land Cover|Cropland|Energy Crops / Land Cover|Cropland",
        "year": 2080,
        "p10": 8.85191188,
        "p50": 16.81126026,
        "p90": 30.60448916,
        "p95": 33.28244641,
        "p99": 40.67330199,
        "unit": "% of cropland",
        "notes": "Default energy-crop share trajectory; ratio matched by Model#Scenario.",
    },
    {
        "database": "AR6 Scenarios Database v1.1",
        "filter": "C1-C2 default",
        "variable": "Land Cover|Cropland|Energy Crops / Land Cover|Cropland",
        "year": 2100,
        "p10": 8.23642225,
        "p50": 19.40640134,
        "p90": 35.89364547,
        "p95": 37.26966434,
        "p99": 42.20469491,
        "unit": "% of cropland",
        "notes": "Default energy-crop share trajectory; ratio matched by Model#Scenario.",
    },
]

SOURCE_URLS = (
    "https://scenariocompass.org; "
    "https://download.scenariocompass.org; "
    "https://iiasa.ac.at/models-tools-data/ar6-scenario-explorer-and-database; "
    "https://www.fao.org/statistics/highlights-archive/highlights-detail/"
    "bioenergy-statistics-1990-2024/en; "
    "https://doi.org/10.5194/essd-12-789-2020; "
    "https://doi.org/10.5281/zenodo.3274254"
)

BIOENERGY_CARRIERS = [
    "Animal waste",
    "Bagasse",
    "Bio jet kerosene",
    "Biodiesel",
    "Biogases",
    "Biogasoline",
    "Black liquor",
    "Charcoal",
    "Fuelwood",
    "Other liquid biofuels",
    "Other vegetal material and residues",
]


@dataclass(frozen=True)
class FeedstockSpec:
    feedstock: str
    feedstock_category: str
    carrier: str
    model_commodity: str = ""
    parent_commodity: str = ""
    market_link: bool = False
    lhv_gj_per_tdm: float = GENERIC_LHV_GJ_PER_TDM
    dry_matter_fraction: float = 1.0
    residue_mix: bool = False


CARRIER_TO_FEEDSTOCK: Mapping[str, FeedstockSpec] = {
    "Animal waste": FeedstockSpec(
        feedstock="animal_waste_biogas",
        feedstock_category="animal_waste",
        carrier="biogas",
        lhv_gj_per_tdm=15.0,
    ),
    "Bagasse": FeedstockSpec(
        feedstock="bagasse_residue",
        feedstock_category="crop_residue",
        carrier="solid biomass",
        parent_commodity="Sugar cane",
        lhv_gj_per_tdm=17.0,
    ),
    "Bio jet kerosene": FeedstockSpec(
        feedstock="biojet_oilcrop_feedstock",
        feedstock_category="first_generation_crop",
        carrier="liquid biofuel",
        model_commodity="Soya beans",
        market_link=True,
        lhv_gj_per_tdm=19.0,
        dry_matter_fraction=0.91,
    ),
    "Biodiesel": FeedstockSpec(
        feedstock="biodiesel_oilcrop_feedstock",
        feedstock_category="first_generation_crop",
        carrier="liquid biofuel",
        model_commodity="Soya beans",
        market_link=True,
        lhv_gj_per_tdm=19.0,
        dry_matter_fraction=0.91,
    ),
    "Biogases": FeedstockSpec(
        feedstock="biogas_waste_feedstock",
        feedstock_category="waste_biomass",
        carrier="biogas",
        lhv_gj_per_tdm=15.0,
    ),
    "Biogasoline": FeedstockSpec(
        feedstock="ethanol_maize_feedstock",
        feedstock_category="first_generation_crop",
        carrier="liquid biofuel",
        model_commodity="Maize (corn)",
        market_link=True,
        lhv_gj_per_tdm=17.0,
        dry_matter_fraction=0.88,
    ),
    "Black liquor": FeedstockSpec(
        feedstock="black_liquor_forest_industrial_residue",
        feedstock_category="forest_industrial_residue",
        carrier="solid biomass",
        lhv_gj_per_tdm=14.0,
    ),
    "Charcoal": FeedstockSpec(
        feedstock="charcoal_woody_biomass",
        feedstock_category="forest_biomass",
        carrier="solid biomass",
        lhv_gj_per_tdm=29.0,
    ),
    "Fuelwood": FeedstockSpec(
        feedstock="fuelwood_forest_biomass",
        feedstock_category="forest_biomass",
        carrier="solid biomass",
        lhv_gj_per_tdm=18.0,
    ),
    "Other liquid biofuels": FeedstockSpec(
        feedstock="other_liquid_crop_feedstock",
        feedstock_category="first_generation_crop",
        carrier="liquid biofuel",
        model_commodity="Maize (corn)",
        market_link=True,
        lhv_gj_per_tdm=17.0,
        dry_matter_fraction=0.88,
    ),
    "Other vegetal material and residues": FeedstockSpec(
        feedstock="mixed_crop_residue",
        feedstock_category="crop_residue",
        carrier="solid biomass",
        lhv_gj_per_tdm=17.0,
        residue_mix=True,
    ),
}

FIRST_GENERATION_CROP_MIX: Tuple[Dict[str, Any], ...] = (
    {
        "share": 0.30,
        "feedstock": "ethanol_maize_feedstock",
        "model_commodity": "Maize (corn)",
        "carrier": "liquid biofuel",
        "lhv_gj_per_tdm": 17.0,
        "dry_matter_fraction": 0.88,
        "coproduct_feed_commodity": "DDGS",
        "coproduct_feed_credit_ratio": 0.30,
    },
    {
        "share": 0.25,
        "feedstock": "ethanol_sugarcane_feedstock",
        "model_commodity": "Sugar cane",
        "carrier": "liquid biofuel",
        "lhv_gj_per_tdm": 17.0,
        "dry_matter_fraction": 0.30,
        "coproduct_feed_commodity": "",
        "coproduct_feed_credit_ratio": 0.0,
    },
    {
        "share": 0.10,
        "feedstock": "ethanol_wheat_feedstock",
        "model_commodity": "Wheat",
        "carrier": "liquid biofuel",
        "lhv_gj_per_tdm": 17.0,
        "dry_matter_fraction": 0.88,
        "coproduct_feed_commodity": "DDGS",
        "coproduct_feed_credit_ratio": 0.30,
    },
    {
        "share": 0.05,
        "feedstock": "ethanol_cassava_feedstock",
        "model_commodity": "Cassava",
        "carrier": "liquid biofuel",
        "lhv_gj_per_tdm": 17.0,
        "dry_matter_fraction": 0.35,
        "coproduct_feed_commodity": "",
        "coproduct_feed_credit_ratio": 0.0,
    },
    {
        "share": 0.10,
        "feedstock": "biodiesel_oilcrop_feedstock",
        "model_commodity": "Soya beans",
        "carrier": "liquid biofuel",
        "lhv_gj_per_tdm": 19.0,
        "dry_matter_fraction": 0.91,
        "coproduct_feed_commodity": "oilseed_meal",
        "coproduct_feed_credit_ratio": 0.75,
    },
    {
        "share": 0.10,
        "feedstock": "biodiesel_rapeseed_feedstock",
        "model_commodity": "Rapeseed",
        "carrier": "liquid biofuel",
        "lhv_gj_per_tdm": 19.0,
        "dry_matter_fraction": 0.91,
        "coproduct_feed_commodity": "oilseed_meal",
        "coproduct_feed_credit_ratio": 0.60,
    },
    {
        "share": 0.10,
        "feedstock": "biodiesel_oilpalm_feedstock",
        "model_commodity": "Oilpalm",
        "carrier": "liquid biofuel",
        "lhv_gj_per_tdm": 19.0,
        "dry_matter_fraction": 0.50,
        "coproduct_feed_commodity": "palm_kernel_meal",
        "coproduct_feed_credit_ratio": 0.10,
    },
)

OECD_EU_ISO3 = {
    # EU-27 + EFTA/UK
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA",
    "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD",
    "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE", "CHE", "ISL", "NOR",
    "GBR",
    # OECD90 / high-income Pacific and North America members commonly grouped
    # in IAM R5 OECD & EU.
    "AUS", "CAN", "JPN", "KOR", "NZL", "USA", "ISR", "TUR",
}

REFORMING_ISO3 = {
    "ALB", "ARM", "AZE", "BIH", "BLR", "GEO", "KAZ", "KGZ", "MDA", "MKD",
    "MNE", "RUS", "SRB", "TJK", "TKM", "UKR", "UZB",
}


def normalize_m49(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "no"}:
        return ""
    if text.lower() in {"world", "global", "all"}:
        return "World"
    if text.startswith("'"):
        text = text[1:]
    if text.count(".") == 1:
        left, right = text.split(".", 1)
        if left.isdigit() and right.strip("0") == "":
            text = left
    if text.isdigit():
        return f"'{text.zfill(3)}"
    return f"'{text}"


def parse_years(raw: str) -> List[int]:
    years = []
    for part in str(raw).replace(";", ",").split(","):
        part = part.strip()
        if part:
            years.append(int(part))
    if not years:
        raise ValueError("At least one future year is required.")
    return sorted(set(years))


def ramp_to_target(
    *,
    base_value: float,
    target_value: float,
    year: int,
    base_year: int,
    target_year: int,
) -> float:
    if year <= base_year:
        return float(base_value)
    if year >= target_year:
        return float(target_value)
    frac = (int(year) - int(base_year)) / (int(target_year) - int(base_year))
    return float(base_value) + (float(target_value) - float(base_value)) * frac


def trajectory_value(metric: str, percentile: str, year: int) -> float:
    """Return an AR6/SCI-backed global trajectory value for an arbitrary year."""
    points = AR6_C1C2_TRAJECTORY.get(metric)
    if not points:
        raise KeyError(f"Unknown bioenergy trajectory metric: {metric}")
    pct = str(percentile).strip().lower()
    years = sorted(int(y) for y in points)
    y = int(year)
    if y in points:
        return float(points[y][pct])
    if y <= years[0]:
        return float(points[years[0]][pct])
    if y >= years[-1]:
        return float(points[years[-1]][pct])
    for left, right in zip(years[:-1], years[1:]):
        if left <= y <= right:
            left_val = float(points[left][pct])
            right_val = float(points[right][pct])
            frac = (y - left) / (right - left)
            return left_val + (right_val - left_val) * frac
    raise RuntimeError(f"Could not interpolate {metric} {pct} for year {year}")


def trajectory_from_points(points: Mapping[int, Mapping[str, float]], percentile: str, year: int) -> float:
    pct = str(percentile).strip().lower()
    years = sorted(int(y) for y in points)
    y = int(year)
    if y in points:
        return float(points[y][pct])
    if y <= years[0]:
        return float(points[years[0]][pct])
    if y >= years[-1]:
        return float(points[years[-1]][pct])
    for left, right in zip(years[:-1], years[1:]):
        if left <= y <= right:
            left_val = float(points[left][pct])
            right_val = float(points[right][pct])
            frac = (y - left) / (right - left)
            return left_val + (right_val - left_val) * frac
    raise RuntimeError(f"Could not interpolate trajectory for {pct} year {year}")


def native_feedstock_split_targets(
    *,
    total_primary_ej: float,
    percentile: str,
    year: int,
) -> Dict[str, float]:
    raw = {
        name: trajectory_from_points(points, percentile, year)
        for name, points in AR6_C1C2_PRIMARY_FEEDSTOCK_SPLIT_EJ.items()
    }
    component_sum = sum(max(0.0, float(v)) for v in raw.values())
    scale = 1.0
    if component_sum > max(float(total_primary_ej), 0.0) and component_sum > 0:
        scale = float(total_primary_ej) / component_sum
    out = {name: max(0.0, float(value)) * scale for name, value in raw.items()}
    out["other_profile"] = max(0.0, float(total_primary_ej) - sum(out.values()))
    out["native_component_scale"] = scale
    return out


def agricultural_bioenergy_demand_value(variable: str, percentile: str, year: int) -> float:
    points = SCI_AGRICULTURAL_BIOENERGY_DEMAND_MTDM.get(variable)
    if not points:
        return np.nan
    return trajectory_from_points(points, percentile, year)


def load_country_map(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="region")
    keep = [
        "M49_Country_Code",
        "NAME",
        "Region_label_new",
        "ISO3 Code",
        "Region_agg2",
        "Region_agg4",
        "Region_agg5",
        "Region_aggMC",
        "Region_RuminateMCScen_new",
    ]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"{path} sheet=region missing columns: {missing}")
    out = df[keep].copy()
    out["M49_Country_Code"] = out["M49_Country_Code"].map(normalize_m49)
    out = out[out["M49_Country_Code"].ne("")].copy()
    out["country_name"] = (
        out["Region_label_new"].fillna(out["NAME"]).fillna("").astype(str).str.strip()
    )
    out["iso3"] = out["ISO3 Code"].fillna("").astype(str).str.strip()
    out = out[
        out["iso3"].str.match(r"^[A-Z]{3}$", na=False)
        & out["Region_agg2"].fillna("").astype(str).str.strip().ne("no")
        & out["Region_agg4"].fillna("").astype(str).str.strip().ne("no")
    ].copy()
    out["r5_region"] = out.apply(derive_r5_region, axis=1)
    out = out.drop_duplicates(subset=["M49_Country_Code"], keep="first")
    return out


def derive_r5_region(row: pd.Series) -> str:
    iso3 = str(row.get("ISO3 Code", "") or "").strip()
    agg2 = str(row.get("Region_agg2", "") or "").strip()
    agg4 = str(row.get("Region_agg4", "") or "").strip()
    if iso3 in OECD_EU_ISO3:
        return "OECD & EU (R5)"
    if iso3 in REFORMING_ISO3:
        return "Reforming Economies (R5)"
    if agg4 == "Latin America & Caribbean":
        return "Latin America (R5)"
    if agg4 in {"Middle East & North Africa", "Sub-Saharan Africa"}:
        return "Middle East & Africa (R5)"
    if agg2 in {"Asia", "Oceania"} or agg4 in {"South Asia", "East Asia & Pacific"}:
        return "Asia (R5)"
    if agg4 == "Europe & Central Asia":
        return "OECD & EU (R5)"
    if agg4 == "North America":
        return "OECD & EU (R5)"
    return "Other (R5)"


def load_leaf_profile(path: Path, countries: pd.DataFrame, base_year: int) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    raw["M49_Country_Code"] = raw["M49_Country_Code"].map(normalize_m49)
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce")
    available_years = sorted(raw["year"].dropna().astype(int).unique().tolist())
    if not available_years:
        raise ValueError(f"{path} has no usable year column.")
    selected_year = base_year if base_year in available_years else max(y for y in available_years if y <= base_year)
    raw = raw[raw["year"].eq(selected_year)].copy()
    for col in BIOENERGY_CARRIERS:
        if col not in raw.columns:
            raw[col] = 0.0
        raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0.0).clip(lower=0.0)

    prof = countries.merge(
        raw[["M49_Country_Code", "year", *BIOENERGY_CARRIERS, "total_bioenergy_tj"]],
        on="M49_Country_Code",
        how="left",
    )
    for col in BIOENERGY_CARRIERS:
        prof[col] = pd.to_numeric(prof[col], errors="coerce").fillna(0.0).clip(lower=0.0)
    prof["baseline_profile_year"] = int(selected_year)
    prof["baseline_bioenergy_tj"] = prof[BIOENERGY_CARRIERS].sum(axis=1)
    total = float(prof["baseline_bioenergy_tj"].sum())
    if total <= 0:
        raise ValueError("Leaf-country baseline bioenergy total is zero; cannot downscale.")
    prof["baseline_global_share"] = prof["baseline_bioenergy_tj"] / total
    for col in BIOENERGY_CARRIERS:
        share_col = f"share_{col}"
        prof[share_col] = np.where(
            prof["baseline_bioenergy_tj"].gt(0.0),
            prof[col] / prof["baseline_bioenergy_tj"],
            0.0,
        )
    return prof


def load_cropland(path: Path, countries: pd.DataFrame, land_source: str = "LUH2") -> pd.DataFrame:
    source = str(land_source).strip().upper()
    if source == "LUH2":
        land = pd.read_excel(path, sheet_name="LUH2")
        land["M49_Country_Code"] = land["M49_Country_Code"].map(normalize_m49)
        land_type = land["Land cover"].fillna("").astype(str).str.strip().str.lower()
        land = land[land_type.eq("cropland")].copy()
        if "Y2020" not in land.columns:
            raise ValueError(f"{path} sheet=LUH2 missing Y2020")
        land["cropland_ha"] = pd.to_numeric(land["Y2020"], errors="coerce").fillna(0.0).clip(lower=0.0)
    elif source == "FAO":
        land = pd.read_excel(path, sheet_name="FAO")
        land["M49_Country_Code"] = land["M49_Country_Code"].map(normalize_m49)
        item = land["Item"].fillna("").astype(str).str.strip().str.lower()
        elem = land["Element"].fillna("").astype(str).str.strip().str.lower()
        land = land[item.isin({"arable land", "permanent crops"}) & elem.eq("area")].copy()
        if "Y2020" not in land.columns:
            raise ValueError(f"{path} sheet=FAO missing Y2020")
        factor = np.where(land["Unit"].fillna("").astype(str).str.lower().str.contains("1000"), 1000.0, 1.0)
        land["cropland_ha"] = pd.to_numeric(land["Y2020"], errors="coerce").fillna(0.0) * factor
        land = land.groupby("M49_Country_Code", as_index=False)["cropland_ha"].sum()
    else:
        raise ValueError("--land-source must be LUH2 or FAO")

    return countries[["M49_Country_Code"]].merge(
        land[["M49_Country_Code", "cropland_ha"]],
        on="M49_Country_Code",
        how="left",
    ).fillna({"cropland_ha": 0.0})


def load_best_energy_crop_yields(path: Path, countries: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    raw["M49_Country_Code"] = raw["M49_Country_Code"].map(normalize_m49)
    raw["yield_tdm_per_ha"] = pd.to_numeric(raw["yield_tdm_per_ha"], errors="coerce")
    raw = raw[raw["yield_tdm_per_ha"].gt(0.0)].copy()
    raw = raw.sort_values(["M49_Country_Code", "yield_tdm_per_ha"])
    best = raw.drop_duplicates(subset=["M49_Country_Code"], keep="last")[
        ["M49_Country_Code", "feedstock", "yield_tdm_per_ha"]
    ].copy()
    out = countries[["M49_Country_Code"]].merge(best, on="M49_Country_Code", how="left")
    global_median = float(pd.to_numeric(best["yield_tdm_per_ha"], errors="coerce").median())
    out["feedstock"] = out["feedstock"].fillna("miscanthus_energy_crop")
    out["yield_tdm_per_ha"] = pd.to_numeric(out["yield_tdm_per_ha"], errors="coerce").fillna(global_median)
    return out


def load_residue_mix(path: Path, countries: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=[
            "M49_Country_Code",
            "feedstock",
            "parent_commodity",
            "residue_weight",
            "country_residue_weight_tdm",
        ])
    raw = pd.read_csv(path, low_memory=False)
    raw["M49_Country_Code"] = raw["M49_Country_Code"].map(normalize_m49)
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce")
    raw = raw[raw["feedstock_category"].fillna("").astype(str).str.lower().eq("crop_residue")].copy()
    raw = raw[
        ~raw["feedstock"].fillna("").astype(str).isin({"bagasse_residue", "mixed_crop_residue"})
    ].copy()
    if raw.empty:
        return pd.DataFrame(columns=[
            "M49_Country_Code",
            "feedstock",
            "parent_commodity",
            "residue_weight",
            "country_residue_weight_tdm",
        ])
    max_year = raw.groupby(["M49_Country_Code", "feedstock"])["year"].transform("max")
    raw = raw[raw["year"].eq(max_year)].copy()
    raw["resource_available_tdm"] = pd.to_numeric(raw["resource_available_tdm"], errors="coerce").fillna(0.0)
    raw["sustainable_fraction"] = pd.to_numeric(raw["sustainable_fraction"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
    raw["residue_weight"] = (raw["resource_available_tdm"] * raw["sustainable_fraction"]).clip(lower=0.0)
    raw = raw[raw["residue_weight"].gt(0.0)].copy()
    raw = countries[["M49_Country_Code"]].merge(
        raw[["M49_Country_Code", "feedstock", "parent_commodity", "residue_weight"]],
        on="M49_Country_Code",
        how="inner",
    )
    totals = raw.groupby("M49_Country_Code")["residue_weight"].transform("sum")
    raw["country_residue_weight_tdm"] = totals
    raw["residue_weight"] = np.where(totals.gt(0.0), raw["residue_weight"] / totals, 0.0)
    return raw


def distribute_non_dedicated_energy(
    *,
    scenario: str,
    year: int,
    country: pd.Series,
    country_energy_tj: float,
    residue_mix: pd.DataFrame,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if country_energy_tj <= 0:
        return rows
    for carrier in BIOENERGY_CARRIERS:
        spec = CARRIER_TO_FEEDSTOCK[carrier]
        carrier_share = float(country.get(f"share_{carrier}", 0.0) or 0.0)
        carrier_tj = country_energy_tj * carrier_share
        if carrier_tj <= 1e-9:
            continue
        if spec.residue_mix:
            mix = residue_mix[residue_mix["M49_Country_Code"].eq(country.M49_Country_Code)]
            if mix.empty:
                rows.append(model_row(scenario, year, country, carrier_tj, spec))
            else:
                for residue in mix.itertuples(index=False):
                    residue_spec = FeedstockSpec(
                        feedstock=str(residue.feedstock),
                        feedstock_category="crop_residue",
                        carrier=spec.carrier,
                        parent_commodity=str(residue.parent_commodity or ""),
                        lhv_gj_per_tdm=spec.lhv_gj_per_tdm,
                    )
                    rows.append(model_row(scenario, year, country, carrier_tj * float(residue.residue_weight), residue_spec))
        else:
            rows.append(model_row(scenario, year, country, carrier_tj, spec))
    return rows


def distribute_profile_other_energy(
    *,
    scenario: str,
    year: int,
    country: pd.Series,
    country_energy_tj: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if country_energy_tj <= 0:
        return rows
    allowed = [
        carrier
        for carrier in BIOENERGY_CARRIERS
        if CARRIER_TO_FEEDSTOCK[carrier].feedstock_category not in {"first_generation_crop", "crop_residue"}
        and not CARRIER_TO_FEEDSTOCK[carrier].residue_mix
    ]
    raw_weights = np.array([float(country.get(f"share_{carrier}", 0.0) or 0.0) for carrier in allowed], dtype=float)
    if not np.isfinite(raw_weights).any() or raw_weights.sum() <= 0:
        raw_weights = np.ones(len(allowed), dtype=float)
    weights = raw_weights / raw_weights.sum()
    for carrier, weight in zip(allowed, weights):
        rows.append(
            model_row(
                scenario,
                year,
                country,
                country_energy_tj * float(weight),
                CARRIER_TO_FEEDSTOCK[carrier],
                notes=(
                    "Other biomass component allocated by FAOSTAT 2023 carrier profile "
                    "after removing native AR6 energy-crops, residues and 1st-generation split."
                ),
            )
        )
    return rows


def distribute_crop_residue_energy(
    *,
    scenario: str,
    year: int,
    country: pd.Series,
    country_energy_tj: float,
    country_demand_tdm: Optional[float],
    residue_mix: pd.DataFrame,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    demand_tdm = None if country_demand_tdm is None else max(0.0, float(country_demand_tdm))
    if country_energy_tj <= 0 and (demand_tdm is None or demand_tdm <= 0):
        return rows
    mix = residue_mix[residue_mix["M49_Country_Code"].eq(country.M49_Country_Code)]
    if mix.empty:
        spec = FeedstockSpec(
            feedstock="mixed_crop_residue",
            feedstock_category="crop_residue",
            carrier="solid biomass",
            lhv_gj_per_tdm=17.0,
        )
        return [
            model_row(
                scenario,
                year,
                country,
                country_energy_tj,
                spec,
                feedstock_demand_tdm=demand_tdm,
                residue_feed_competition_tdm=(
                    demand_tdm
                    if demand_tdm is not None
                    else country_energy_tj * 1000.0 / spec.lhv_gj_per_tdm
                ),
                notes=(
                    "Native AR6 Primary Energy|Biomass|Residues component allocated to mixed residue "
                    "because country-specific OMD residue mix was unavailable. Physical dry-matter "
                    "demand uses SCI Agricultural Demand|Residues|Bioenergy when available."
                ),
            )
        ]
    for residue in mix.itertuples(index=False):
        spec = FeedstockSpec(
            feedstock=str(residue.feedstock),
            feedstock_category="crop_residue",
            carrier="solid biomass",
            parent_commodity=str(residue.parent_commodity or ""),
            lhv_gj_per_tdm=17.0,
        )
        energy = country_energy_tj * float(residue.residue_weight)
        dry = None if demand_tdm is None else demand_tdm * float(residue.residue_weight)
        rows.append(
            model_row(
                scenario,
                year,
                country,
                energy,
                spec,
                feedstock_demand_tdm=dry,
                residue_feed_competition_tdm=(
                    dry
                    if dry is not None
                    else energy * 1000.0 / spec.lhv_gj_per_tdm
                ),
                notes=(
                    "Native AR6 Primary Energy|Biomass|Residues component allocated with OMD "
                    "country crop-residue resource weights. Physical dry-matter demand uses "
                    "SCI Agricultural Demand|Residues|Bioenergy when available."
                ),
            )
        )
    return rows


def distribute_first_generation_crop_energy(
    *,
    scenario: str,
    year: int,
    country: pd.Series,
    country_energy_tj: float,
    country_demand_tdm: Optional[float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    demand_tdm = None if country_demand_tdm is None else max(0.0, float(country_demand_tdm))
    if country_energy_tj <= 0 and (demand_tdm is None or demand_tdm <= 0):
        return rows
    total_share = sum(float(item["share"]) for item in FIRST_GENERATION_CROP_MIX)
    if total_share <= 0:
        return rows
    for item in FIRST_GENERATION_CROP_MIX:
        share = float(item["share"]) / total_share
        spec = FeedstockSpec(
            feedstock=str(item["feedstock"]),
            feedstock_category="first_generation_crop",
            carrier=str(item["carrier"]),
            model_commodity=str(item["model_commodity"]),
            market_link=True,
            lhv_gj_per_tdm=float(item["lhv_gj_per_tdm"]),
            dry_matter_fraction=float(item["dry_matter_fraction"]),
        )
        energy = country_energy_tj * share
        dry = (
            demand_tdm * share
            if demand_tdm is not None
            else energy * 1000.0 / spec.lhv_gj_per_tdm
        )
        coproduct_ratio = float(item.get("coproduct_feed_credit_ratio", 0.0) or 0.0)
        rows.append(
            model_row(
                scenario,
                year,
                country,
                energy,
                spec,
                feedstock_demand_tdm=dry,
                coproduct_feed_commodity=str(item.get("coproduct_feed_commodity", "")),
                coproduct_feed_credit_tdm=dry * coproduct_ratio,
                notes=(
                    "Native AR6 Primary Energy|Biomass|1st Generation component allocated with "
                    "expanded first-generation crop mix. Physical dry-matter demand uses SCI "
                    "Agricultural Demand|Crops|Bioenergy crop total when available; coproduct "
                    "feed credit is handoff-only."
                ),
            )
        )
    return rows


def model_row(
    scenario: str,
    year: int,
    country: pd.Series,
    energy_tj: float,
    spec: FeedstockSpec,
    *,
    feedstock_demand_tdm: Optional[float] = None,
    yield_tdm_per_ha: Optional[float] = None,
    land_target_ha: Optional[float] = None,
    coproduct_feed_commodity: str = "",
    coproduct_feed_credit_tdm: Optional[float] = None,
    residue_feed_competition_tdm: Optional[float] = None,
    notes: str = "",
) -> Dict[str, Any]:
    dry = feedstock_demand_tdm
    if dry is None and spec.lhv_gj_per_tdm > 0:
        dry = energy_tj * 1000.0 / spec.lhv_gj_per_tdm
    wet = None if dry is None else dry / max(spec.dry_matter_fraction, 1e-9)
    return {
        "scenario": scenario,
        "M49_Country_Code": country.M49_Country_Code,
        "country_name": country.country_name,
        "year": int(year),
        "carrier": spec.carrier,
        "feedstock": spec.feedstock,
        "feedstock_category": spec.feedstock_category,
        "model_commodity": spec.model_commodity,
        "market_link": bool(spec.market_link),
        "energy_basis": "primary",
        "energy_target_tj": float(energy_tj),
        "feedstock_demand_t": wet,
        "feedstock_demand_tdm": dry,
        "parent_commodity": spec.parent_commodity,
        "lhv_gj_per_tdm": spec.lhv_gj_per_tdm,
        "conversion_efficiency": 1.0,
        "dry_matter_fraction": spec.dry_matter_fraction,
        "yield_tdm_per_ha": yield_tdm_per_ha,
        "source": SOURCE_URLS,
        "notes": notes or (
            "Generated by S0_53 from AR6 C1-C2 P10/P50/P90 Primary Energy|Biomass "
            "trajectory; FAOSTAT baseline is used only for country/R5 allocation weights."
        ),
        "energy_crop_area_target_ha": land_target_ha,
        "coproduct_feed_commodity": coproduct_feed_commodity,
        "coproduct_feed_credit_tdm": coproduct_feed_credit_tdm,
        "residue_feed_competition_tdm": residue_feed_competition_tdm,
    }


def prepare_scenarios(args: argparse.Namespace) -> Dict[str, pd.DataFrame]:
    years = parse_years(args.years)
    countries = load_country_map(Path(args.dict_xlsx))
    profile = load_leaf_profile(Path(args.profile_csv), countries, int(args.base_year))
    cropland = load_cropland(Path(args.land_cover_xlsx), countries, args.land_source)
    yields = load_best_energy_crop_yields(Path(args.energy_crop_yield_csv), countries)
    residue_mix = load_residue_mix(Path(args.resource_constraints_csv), countries)

    base = (
        profile.merge(cropland, on="M49_Country_Code", how="left")
        .merge(yields.rename(columns={"feedstock": "best_energy_crop_feedstock"}), on="M49_Country_Code", how="left")
    )
    base["cropland_ha"] = pd.to_numeric(base["cropland_ha"], errors="coerce").fillna(0.0).clip(lower=0.0)
    base["yield_tdm_per_ha"] = pd.to_numeric(base["yield_tdm_per_ha"], errors="coerce").fillna(0.0).clip(lower=0.0)
    base["baseline_bioenergy_tj"] = pd.to_numeric(base["baseline_bioenergy_tj"], errors="coerce").fillna(0.0)
    base_global_ej = float(base["baseline_bioenergy_tj"].sum() / EJ_TO_TJ)
    if base_global_ej <= 0:
        raise ValueError("Baseline global leaf-country bioenergy is zero.")

    r5_base = base.groupby("r5_region", as_index=False).agg(
        baseline_bioenergy_tj=("baseline_bioenergy_tj", "sum"),
        cropland_ha=("cropland_ha", "sum"),
    )
    r5_base["r5_baseline_share"] = r5_base["baseline_bioenergy_tj"] / r5_base["baseline_bioenergy_tj"].sum()

    global_rows: List[Dict[str, Any]] = []
    r5_rows: List[Dict[str, Any]] = []
    country_rows: List[Dict[str, Any]] = []
    model_rows: List[Dict[str, Any]] = []

    for scenario, setting in SCENARIO_SETTINGS.items():
        percentile = str(setting["percentile"])
        for year in years:
            target_ej = trajectory_value("primary_bioenergy_ej_yr", percentile, int(year))
            pct_cropland = trajectory_value("energy_crops_pct_cropland", percentile, int(year))
            split = native_feedstock_split_targets(
                total_primary_ej=target_ej,
                percentile=percentile,
                year=int(year),
            )
            ag_crop_mtdm = agricultural_bioenergy_demand_value(
                "Agricultural Demand|Crops|Bioenergy",
                percentile,
                int(year),
            )
            ag_residue_mtdm = agricultural_bioenergy_demand_value(
                "Agricultural Demand|Residues|Bioenergy",
                percentile,
                int(year),
            )
            ag_crop_tdm = (
                float(ag_crop_mtdm) * 1_000_000.0
                if np.isfinite(ag_crop_mtdm) and float(ag_crop_mtdm) >= 0
                else np.nan
            )
            ag_residue_tdm = (
                float(ag_residue_mtdm) * 1_000_000.0
                if np.isfinite(ag_residue_mtdm) and float(ag_residue_mtdm) >= 0
                else np.nan
            )
            crop_primary_ej = (
                float(split["dedicated_energy_crop"])
                + float(split["first_generation_crop"])
            )
            dedicated_crop_physical_share = (
                float(split["dedicated_energy_crop"]) / crop_primary_ej
                if crop_primary_ej > 0
                else (1.0 if pct_cropland > 0 else 0.0)
            )
            firstgen_crop_physical_share = max(0.0, 1.0 - dedicated_crop_physical_share)
            global_energy_crop_tdm_target = (
                float(ag_crop_tdm) * dedicated_crop_physical_share
                if np.isfinite(ag_crop_tdm)
                else np.nan
            )
            global_first_generation_tdm_target = (
                float(ag_crop_tdm) * firstgen_crop_physical_share
                if np.isfinite(ag_crop_tdm)
                else np.nan
            )
            global_residue_tdm_target = float(ag_residue_tdm) if np.isfinite(ag_residue_tdm) else np.nan
            global_crop_area = float(base["cropland_ha"].sum())
            global_energy_crop_area_target = global_crop_area * pct_cropland / 100.0
            global_rows.append(
                {
                    "scenario": scenario,
                    "scenario_label": setting["label"],
                    "year": year,
                    "primary_bioenergy_ej_yr": target_ej,
                    "energy_crops_pct_cropland": pct_cropland,
                    "global_cropland_ha": global_crop_area,
                    "energy_crop_area_target_ha": global_energy_crop_area_target,
                    "native_energy_crop_primary_ej_yr": split["dedicated_energy_crop"],
                    "native_residue_primary_ej_yr": split["crop_residue"],
                    "native_first_generation_primary_ej_yr": split["first_generation_crop"],
                    "profile_other_primary_ej_yr": split["other_profile"],
                    "native_component_scale": split["native_component_scale"],
                    "agricultural_demand_crops_bioenergy_mtdm_yr": ag_crop_mtdm,
                    "agricultural_demand_residues_bioenergy_mtdm_yr": ag_residue_mtdm,
                    "sci_crop_physical_demand_tdm_yr": global_energy_crop_tdm_target + global_first_generation_tdm_target if np.isfinite(ag_crop_tdm) else np.nan,
                    "sci_energy_crop_physical_demand_tdm_yr": global_energy_crop_tdm_target,
                    "sci_first_generation_physical_demand_tdm_yr": global_first_generation_tdm_target,
                    "sci_residue_physical_demand_tdm_yr": global_residue_tdm_target,
                    "baseline_global_bioenergy_ej_yr": base_global_ej,
                    "base_year": int(args.base_year),
                    "target_year": int(args.target_year),
                    "trajectory_source": GLOBAL_TRAJECTORY_SOURCE,
                    "trajectory_percentile": percentile.upper(),
                    "source": SOURCE_URLS,
                    "percentile_note": setting["percentile_note"],
                    "trajectory_note": (
                        f"AR6 C1-C2 {percentile.upper()} value for {year}; "
                        "native AR6 feedstock split uses Primary Energy|Biomass|Energy Crops, "
                        "Primary Energy|Biomass|Residues and Primary Energy|Biomass|1st Generation; "
                        "SCI Agricultural Demand variables replace physical dry-matter feedstock demand "
                        "for crop and residue bioenergy when available."
                    ),
                }
            )

            for r5 in r5_base.itertuples(index=False):
                r5_region = str(r5.r5_region)
                r5_total_tj = target_ej * EJ_TO_TJ * float(r5.r5_baseline_share)
                r5_country = base[base["r5_region"].eq(r5_region)].copy()
                r5_cropland = float(r5.cropland_ha or 0.0)
                r5_crop_share = r5_cropland / global_crop_area if global_crop_area > 0 else float(r5.r5_baseline_share)
                r5_energy_crop_area_target = global_energy_crop_area_target * r5_crop_share
                r5_energy_crop_tj_target = split["dedicated_energy_crop"] * EJ_TO_TJ * float(r5.r5_baseline_share)
                r5_residue_tj_target = split["crop_residue"] * EJ_TO_TJ * float(r5.r5_baseline_share)
                r5_first_generation_tj_target = split["first_generation_crop"] * EJ_TO_TJ * float(r5.r5_baseline_share)
                r5_profile_other_tj_target = split["other_profile"] * EJ_TO_TJ * float(r5.r5_baseline_share)
                r5_crop_energy_share = (
                    (r5_energy_crop_tj_target + r5_first_generation_tj_target)
                    / max(crop_primary_ej * EJ_TO_TJ, 1e-12)
                    if crop_primary_ej > 0
                    else float(r5.r5_baseline_share)
                )
                r5_residue_energy_share = (
                    r5_residue_tj_target / max(float(split["crop_residue"]) * EJ_TO_TJ, 1e-12)
                    if float(split["crop_residue"]) > 0
                    else float(r5.r5_baseline_share)
                )
                r5_crop_tdm_target = float(ag_crop_tdm) * r5_crop_energy_share if np.isfinite(ag_crop_tdm) else np.nan
                r5_residue_tdm_target = float(ag_residue_tdm) * r5_residue_energy_share if np.isfinite(ag_residue_tdm) else np.nan
                r5_dedicated_share = (
                    r5_energy_crop_tj_target
                    / max(r5_energy_crop_tj_target + r5_first_generation_tj_target, 1e-12)
                    if (r5_energy_crop_tj_target + r5_first_generation_tj_target) > 0
                    else dedicated_crop_physical_share
                )
                r5_energy_crop_tdm_target = (
                    r5_crop_tdm_target * r5_dedicated_share
                    if np.isfinite(r5_crop_tdm_target)
                    else np.nan
                )
                r5_first_generation_tdm_target = (
                    r5_crop_tdm_target * max(0.0, 1.0 - r5_dedicated_share)
                    if np.isfinite(r5_crop_tdm_target)
                    else np.nan
                )

                crop_weights = pd.to_numeric(r5_country["cropland_ha"], errors="coerce").fillna(0.0)
                if crop_weights.sum() <= 0:
                    crop_weights = pd.to_numeric(r5_country["baseline_bioenergy_tj"], errors="coerce").fillna(0.0)
                if crop_weights.sum() <= 0:
                    crop_weights = pd.Series(1.0, index=r5_country.index)
                r5_country["energy_crop_area_ha"] = r5_energy_crop_area_target * crop_weights / crop_weights.sum()
                energy_crop_weights = (
                    pd.to_numeric(r5_country["cropland_ha"], errors="coerce").fillna(0.0)
                    * pd.to_numeric(r5_country["yield_tdm_per_ha"], errors="coerce").fillna(0.0)
                )
                if energy_crop_weights.sum() <= 0:
                    energy_crop_weights = crop_weights
                r5_country["energy_crop_tj"] = (
                    r5_energy_crop_tj_target * energy_crop_weights / energy_crop_weights.sum()
                    if energy_crop_weights.sum() > 0 else 0.0
                )
                r5_country["energy_crop_tdm"] = (
                    r5_energy_crop_tdm_target * energy_crop_weights / energy_crop_weights.sum()
                    if np.isfinite(r5_energy_crop_tdm_target) and energy_crop_weights.sum() > 0
                    else r5_country["energy_crop_tj"] * 1000.0 / ENERGY_CROP_LHV_GJ_PER_TDM
                )

                profile_weights = pd.to_numeric(r5_country["baseline_bioenergy_tj"], errors="coerce").fillna(0.0)
                if profile_weights.sum() <= 0:
                    profile_weights = pd.Series(1.0, index=r5_country.index)
                r5_country["profile_other_tj"] = r5_profile_other_tj_target * profile_weights / profile_weights.sum()

                firstgen_weights = (
                    pd.to_numeric(r5_country.get("Biogasoline", 0.0), errors="coerce").fillna(0.0)
                    + pd.to_numeric(r5_country.get("Biodiesel", 0.0), errors="coerce").fillna(0.0)
                    + pd.to_numeric(r5_country.get("Bio jet kerosene", 0.0), errors="coerce").fillna(0.0)
                    + pd.to_numeric(r5_country.get("Other liquid biofuels", 0.0), errors="coerce").fillna(0.0)
                )
                if firstgen_weights.sum() <= 0:
                    firstgen_weights = profile_weights
                r5_country["first_generation_tj"] = (
                    r5_first_generation_tj_target * firstgen_weights / firstgen_weights.sum()
                    if firstgen_weights.sum() > 0 else 0.0
                )
                r5_country["first_generation_tdm"] = (
                    r5_first_generation_tdm_target * firstgen_weights / firstgen_weights.sum()
                    if np.isfinite(r5_first_generation_tdm_target) and firstgen_weights.sum() > 0
                    else np.nan
                )

                residue_country_weight = residue_mix.groupby("M49_Country_Code")[
                    "country_residue_weight_tdm"
                ].max() if not residue_mix.empty else pd.Series(dtype=float)
                r5_country["residue_country_weight_tdm"] = (
                    r5_country["M49_Country_Code"].map(residue_country_weight).fillna(0.0)
                )
                residue_weights = pd.to_numeric(r5_country["residue_country_weight_tdm"], errors="coerce").fillna(0.0)
                if residue_weights.sum() <= 0:
                    residue_weights = profile_weights
                r5_country["residue_tj"] = (
                    r5_residue_tj_target * residue_weights / residue_weights.sum()
                    if residue_weights.sum() > 0 else 0.0
                )
                r5_country["residue_tdm"] = (
                    r5_residue_tdm_target * residue_weights / residue_weights.sum()
                    if np.isfinite(r5_residue_tdm_target) and residue_weights.sum() > 0
                    else np.nan
                )
                r5_country["target_primary_tj"] = (
                    r5_country["energy_crop_tj"]
                    + r5_country["residue_tj"]
                    + r5_country["first_generation_tj"]
                    + r5_country["profile_other_tj"]
                )

                r5_rows.append(
                    {
                        "scenario": scenario,
                        "scenario_label": setting["label"],
                        "year": year,
                        "r5_region": r5_region,
                        "r5_baseline_bioenergy_tj": float(r5.baseline_bioenergy_tj),
                        "r5_baseline_share": float(r5.r5_baseline_share),
                        "target_primary_bioenergy_ej_yr": r5_total_tj / EJ_TO_TJ,
                        "cropland_ha": r5_cropland,
                        "energy_crops_pct_cropland": pct_cropland,
                        "energy_crop_area_target_ha": float(r5_country["energy_crop_area_ha"].sum()),
                        "native_energy_crop_ej_yr": r5_energy_crop_tj_target / EJ_TO_TJ,
                        "native_residue_ej_yr": r5_residue_tj_target / EJ_TO_TJ,
                        "native_first_generation_ej_yr": r5_first_generation_tj_target / EJ_TO_TJ,
                        "profile_other_ej_yr": r5_profile_other_tj_target / EJ_TO_TJ,
                        "sci_crop_physical_demand_tdm_yr": float(r5_crop_tdm_target) if np.isfinite(r5_crop_tdm_target) else np.nan,
                        "sci_energy_crop_physical_demand_tdm_yr": float(r5_energy_crop_tdm_target) if np.isfinite(r5_energy_crop_tdm_target) else np.nan,
                        "sci_first_generation_physical_demand_tdm_yr": float(r5_first_generation_tdm_target) if np.isfinite(r5_first_generation_tdm_target) else np.nan,
                        "sci_residue_physical_demand_tdm_yr": float(r5_residue_tdm_target) if np.isfinite(r5_residue_tdm_target) else np.nan,
                        "target_primary_bioenergy_check_ej_yr": float(r5_country["target_primary_tj"].sum()) / EJ_TO_TJ,
                    }
                )

                for _, c in r5_country.iterrows():
                    country_rows.append(
                        {
                            "scenario": scenario,
                            "scenario_label": setting["label"],
                            "year": year,
                            "r5_region": r5_region,
                            "M49_Country_Code": c.M49_Country_Code,
                            "country_name": c.country_name,
                            "iso3": c.iso3,
                            "baseline_profile_year": int(c.baseline_profile_year),
                            "baseline_bioenergy_tj": float(c.baseline_bioenergy_tj),
                            "baseline_global_share": float(c.baseline_global_share),
                            "cropland_ha": float(c.cropland_ha),
                            "energy_crops_pct_cropland": pct_cropland,
                            "dedicated_energy_crop_area_ha": float(c.energy_crop_area_ha),
                            "dedicated_energy_crop_feedstock": c.best_energy_crop_feedstock,
                            "dedicated_energy_crop_yield_tdm_per_ha": float(c.yield_tdm_per_ha),
                            "dedicated_energy_crop_demand_tdm": float(c.energy_crop_tdm),
                            "dedicated_energy_crop_ej_yr": float(c.energy_crop_tj) / EJ_TO_TJ,
                            "first_generation_crop_demand_tdm": float(c.first_generation_tdm) if np.isfinite(c.first_generation_tdm) else np.nan,
                            "crop_residue_bioenergy_demand_tdm": float(c.residue_tdm) if np.isfinite(c.residue_tdm) else np.nan,
                            "crop_residue_bioenergy_ej_yr": float(c.residue_tj) / EJ_TO_TJ,
                            "first_generation_crop_bioenergy_ej_yr": float(c.first_generation_tj) / EJ_TO_TJ,
                            "profile_other_ej_yr": float(c.profile_other_tj) / EJ_TO_TJ,
                            "target_primary_bioenergy_ej_yr": float(c.target_primary_tj) / EJ_TO_TJ,
                            "feedstock_split_source": "AR6 native primary feedstock split; SCI agricultural demand physical demand",
                        }
                    )
                    if float(c.energy_crop_tj) > 0:
                        spec = FeedstockSpec(
                            feedstock=str(c.best_energy_crop_feedstock),
                            feedstock_category="dedicated_energy_crop",
                            carrier="primary bioenergy",
                            lhv_gj_per_tdm=ENERGY_CROP_LHV_GJ_PER_TDM,
                        )
                        model_rows.append(
                            model_row(
                                scenario,
                                year,
                                c,
                                float(c.energy_crop_tj),
                                spec,
                                feedstock_demand_tdm=float(c.energy_crop_tdm),
                                yield_tdm_per_ha=float(c.yield_tdm_per_ha),
                                land_target_ha=float(c.energy_crop_area_ha),
                                notes=(
                                    "Dedicated energy-crop demand from native AR6 "
                                    "Primary Energy|Biomass|Energy Crops. Physical dry-matter demand "
                                    "uses SCI Agricultural Demand|Crops|Bioenergy allocated across "
                                    "dedicated and first-generation crops. Cropland-share area is carried "
                                    "separately as within-cropland reallocation target."
                                ),
                            )
                        )
                    model_rows.extend(
                        distribute_crop_residue_energy(
                            scenario=scenario,
                            year=year,
                            country=c,
                            country_energy_tj=float(c.residue_tj),
                            country_demand_tdm=(
                                float(c.residue_tdm)
                                if np.isfinite(c.residue_tdm)
                                else None
                            ),
                            residue_mix=residue_mix,
                        )
                    )
                    model_rows.extend(
                        distribute_first_generation_crop_energy(
                            scenario=scenario,
                            year=year,
                            country=c,
                            country_energy_tj=float(c.first_generation_tj),
                            country_demand_tdm=(
                                float(c.first_generation_tdm)
                                if np.isfinite(c.first_generation_tdm)
                                else None
                            ),
                        )
                    )
                    model_rows.extend(
                        distribute_profile_other_energy(
                            scenario=scenario,
                            year=year,
                            country=c,
                            country_energy_tj=float(c.profile_other_tj),
                        )
                    )

    global_df = pd.DataFrame(global_rows)
    r5_df = pd.DataFrame(r5_rows)
    country_df = pd.DataFrame(country_rows)
    model_df = pd.DataFrame(model_rows)
    source_check_df = pd.DataFrame(DATABASE_CHECK_ROWS)

    audit_df = validate_outputs(global_df, r5_df, country_df, model_df)
    return {
        "bioenergy_scenario_global_targets": global_df,
        "bioenergy_scenario_r5_targets": r5_df,
        "bioenergy_scenario_country_targets": country_df,
        "bioenergy_scenario_model_input": model_df,
        "bioenergy_scenario_ar6_sci_percentile_check": source_check_df,
        "bioenergy_scenario_allocation_audit": audit_df,
    }


def validate_outputs(
    global_df: pd.DataFrame,
    r5_df: pd.DataFrame,
    country_df: pd.DataFrame,
    model_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for scenario, year in global_df[["scenario", "year"]].drop_duplicates().itertuples(index=False):
        g = global_df[(global_df["scenario"].eq(scenario)) & (global_df["year"].eq(year))].iloc[0]
        target_tj = float(g.primary_bioenergy_ej_yr) * EJ_TO_TJ
        r5_sum = float(
            r5_df[(r5_df["scenario"].eq(scenario)) & (r5_df["year"].eq(year))][
                "target_primary_bioenergy_ej_yr"
            ].sum()
            * EJ_TO_TJ
        )
        country_sum = float(
            country_df[(country_df["scenario"].eq(scenario)) & (country_df["year"].eq(year))][
                "target_primary_bioenergy_ej_yr"
            ].sum()
            * EJ_TO_TJ
        )
        model_sum = float(
            pd.to_numeric(
                model_df[(model_df["scenario"].eq(scenario)) & (model_df["year"].eq(year))][
                    "energy_target_tj"
                ],
                errors="coerce",
            ).sum()
        )
        energy_crop_area = float(
            country_df[(country_df["scenario"].eq(scenario)) & (country_df["year"].eq(year))][
                "dedicated_energy_crop_area_ha"
            ].sum()
        )
        cropland = float(
            country_df[(country_df["scenario"].eq(scenario)) & (country_df["year"].eq(year))][
                "cropland_ha"
            ].sum()
        )
        area_pct = 100.0 * energy_crop_area / cropland if cropland > 0 else np.nan
        rows.extend(
            [
                {
                    "scenario": scenario,
                    "year": year,
                    "check": "r5_sum_matches_global_target",
                    "expected": target_tj,
                    "actual": r5_sum,
                    "difference": r5_sum - target_tj,
                    "status": "pass" if abs(r5_sum - target_tj) <= max(1e-6, target_tj * 1e-9) else "fail",
                },
                {
                    "scenario": scenario,
                    "year": year,
                    "check": "country_sum_matches_global_target",
                    "expected": target_tj,
                    "actual": country_sum,
                    "difference": country_sum - target_tj,
                    "status": "pass" if abs(country_sum - target_tj) <= max(1e-6, target_tj * 1e-9) else "fail",
                },
                {
                    "scenario": scenario,
                    "year": year,
                    "check": "model_input_sum_matches_global_target",
                    "expected": target_tj,
                    "actual": model_sum,
                    "difference": model_sum - target_tj,
                    "status": "pass" if abs(model_sum - target_tj) <= max(1e-6, target_tj * 1e-9) else "fail",
                },
                {
                    "scenario": scenario,
                    "year": year,
                    "check": "energy_crop_area_pct_cropland",
                    "expected": float(g.energy_crops_pct_cropland),
                    "actual": area_pct,
                    "difference": area_pct - float(g.energy_crops_pct_cropland),
                    "status": (
                        "pass"
                        if abs(area_pct - float(g.energy_crops_pct_cropland)) <= 1e-6
                        else "scaled_or_capped"
                    ),
                },
            ]
        )
    return pd.DataFrame(rows)


def write_outputs(outputs: Mapping[str, pd.DataFrame], out_dir: Path, input_dir: Optional[Path]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in outputs.items():
        df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    if input_dir is not None:
        input_dir.mkdir(parents=True, exist_ok=True)
        outputs["bioenergy_scenario_model_input"].to_csv(
            input_dir / "bioenergy_scenario_targets.csv",
            index=False,
            encoding="utf-8-sig",
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-csv", default=str(DEFAULT_PROFILE_CSV))
    parser.add_argument("--resource-constraints-csv", default=str(DEFAULT_RESOURCE_CSV))
    parser.add_argument("--energy-crop-yield-csv", default=str(DEFAULT_YIELD_CSV))
    parser.add_argument("--dict-xlsx", default=str(DEFAULT_DICT_XLSX))
    parser.add_argument("--land-cover-xlsx", default=str(DEFAULT_LAND_XLSX))
    parser.add_argument("--land-source", default="LUH2", choices=["LUH2", "FAO"])
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--years", default=",".join(str(y) for y in DEFAULT_YEARS))
    parser.add_argument("--base-year", type=int, default=BASE_YEAR_DEFAULT)
    parser.add_argument(
        "--target-year",
        type=int,
        default=TARGET_YEAR_DEFAULT,
        help="Deprecated compatibility option; AR6/SCI year-specific trajectories are used instead.",
    )
    parser.add_argument("--write-input", action="store_true")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    outputs = prepare_scenarios(args)
    write_outputs(
        outputs,
        out_dir=Path(args.out_dir),
        input_dir=Path(args.input_dir) if args.write_input else None,
    )
    audit = outputs["bioenergy_scenario_allocation_audit"]
    failed = audit[~audit["status"].isin(["pass"])]
    print("Generated bioenergy scenario tables:")
    for name, df in outputs.items():
        print(f"  - {name}.csv: {len(df):,} rows")
    if failed.empty:
        print("All allocation checks passed.")
    else:
        print("Non-pass allocation checks:")
        print(failed.to_string(index=False))


if __name__ == "__main__":
    main()
