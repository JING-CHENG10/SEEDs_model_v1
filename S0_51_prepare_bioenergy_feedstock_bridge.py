# -*- coding: utf-8 -*-
"""Allocate FAOSTAT bioenergy carriers to physical feedstocks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config_paths import get_input_base
from S2_0_load_data import DataPaths
from S3_3_bioenergy import normalize_m49
from runtime_data_cache import read_csv_cached
try:
    from S0_53_prepare_bioenergy_scenarios import CARRIER_TO_FEEDSTOCK, FIRST_GENERATION_CROP_MIX, SOURCE_URLS
except Exception:
    CARRIER_TO_FEEDSTOCK = {}
    FIRST_GENERATION_CROP_MIX = ()
    SOURCE_URLS = ""


DEFAULT_BRIDGE_YEAR = 2023


def _default_bridge() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for carrier, spec in CARRIER_TO_FEEDSTOCK.items():
        if carrier in {"Biogasoline", "Other liquid biofuels"}:
            ethanol_mix = [
                item for item in FIRST_GENERATION_CROP_MIX
                if str(item.get("feedstock", "")).startswith("ethanol_")
            ]
            share_sum = sum(float(item.get("share", 0.0) or 0.0) for item in ethanol_mix)
            for item in ethanol_mix:
                if share_sum <= 0:
                    continue
                rows.append(
                    {
                        "M49_Country_Code": "World",
                        "country_name": "World",
                        "year": DEFAULT_BRIDGE_YEAR,
                        "carrier": carrier,
                        "feedstock": item["feedstock"],
                        "feedstock_category": "first_generation_crop",
                        "model_commodity": item["model_commodity"],
                        "market_link": True,
                        "share": float(item["share"]) / share_sum,
                        "lhv_gj_per_tdm": float(item["lhv_gj_per_tdm"]),
                        "conversion_efficiency": 1.0,
                        "dry_matter_fraction": float(item["dry_matter_fraction"]),
                        "source": SOURCE_URLS,
                        "notes": "Default expanded ethanol carrier-to-feedstock proxy from S0_53.",
                    }
                )
            continue
        if carrier in {"Biodiesel", "Bio jet kerosene"}:
            biodiesel_mix = [
                item for item in FIRST_GENERATION_CROP_MIX
                if str(item.get("feedstock", "")).startswith("biodiesel_")
                or str(item.get("feedstock", "")) == "biojet_oilcrop_feedstock"
            ]
            share_sum = sum(float(item.get("share", 0.0) or 0.0) for item in biodiesel_mix)
            for item in biodiesel_mix:
                if share_sum <= 0:
                    continue
                rows.append(
                    {
                        "M49_Country_Code": "World",
                        "country_name": "World",
                        "year": DEFAULT_BRIDGE_YEAR,
                        "carrier": carrier,
                        "feedstock": item["feedstock"],
                        "feedstock_category": "first_generation_crop",
                        "model_commodity": item["model_commodity"],
                        "market_link": True,
                        "share": float(item["share"]) / share_sum,
                        "lhv_gj_per_tdm": float(item["lhv_gj_per_tdm"]),
                        "conversion_efficiency": 1.0,
                        "dry_matter_fraction": float(item["dry_matter_fraction"]),
                        "source": SOURCE_URLS,
                        "notes": "Default expanded biodiesel/biojet carrier-to-feedstock proxy from S0_53.",
                    }
                )
            continue
        rows.append(
            {
                "M49_Country_Code": "World",
                "country_name": "World",
                "year": DEFAULT_BRIDGE_YEAR,
                "carrier": carrier,
                "feedstock": spec.feedstock,
                "feedstock_category": spec.feedstock_category,
                "model_commodity": spec.model_commodity,
                "market_link": bool(spec.market_link),
                "share": 1.0,
                "lhv_gj_per_tdm": float(spec.lhv_gj_per_tdm),
                "conversion_efficiency": 1.0,
                "dry_matter_fraction": float(spec.dry_matter_fraction),
                "source": SOURCE_URLS,
                "notes": (
                    "Default one-to-one carrier-to-feedstock proxy from S0_53. "
                    "Replace with country/feedstock-specific historical bridge when available."
                ),
            }
        )
    return pd.DataFrame(rows)


def _load_bridge(path: str, *, use_default_carrier_map: bool) -> pd.DataFrame:
    p = Path(path) if path else None
    if p is None or not p.exists():
        if use_default_carrier_map:
            return _default_bridge()
        raise FileNotFoundError(f"carrier-feedstock bridge not found: {path}")
    df = read_csv_cached(str(p), low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    required = {"M49_Country_Code", "year", "carrier", "feedstock", "share"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"carrier-feedstock bridge missing columns: {sorted(missing)}")
    if df.empty and use_default_carrier_map:
        return _default_bridge()
    df["M49_Country_Code"] = df["M49_Country_Code"].apply(normalize_m49)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["share"] = pd.to_numeric(df["share"], errors="coerce")
    df = df.dropna(subset=["year", "share"])
    df["year"] = df["year"].astype(int)
    df["carrier"] = df["carrier"].fillna("").astype(str).str.strip()
    df["feedstock"] = df["feedstock"].fillna("").astype(str).str.strip()
    df = df[(df["carrier"] != "") & (df["feedstock"] != "") & df["share"].gt(0.0)]
    for col, default in [
        ("country_name", ""),
        ("feedstock_category", ""),
        ("model_commodity", ""),
        ("market_link", False),
        ("lhv_gj_per_tdm", np.nan),
        ("conversion_efficiency", 1.0),
        ("dry_matter_fraction", 1.0),
        ("source", ""),
        ("notes", ""),
    ]:
        if col not in df.columns:
            df[col] = default
    for col in ["lhv_gj_per_tdm", "conversion_efficiency", "dry_matter_fraction"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _select_shares(
    bridge: pd.DataFrame,
    country: str,
    carrier: str,
    year: int,
) -> Tuple[pd.DataFrame, str]:
    for geography, label in [(country, "country"), ("World", "world")]:
        subset = bridge[
            bridge["M49_Country_Code"].eq(geography)
            & bridge["carrier"].eq(carrier)
        ]
        if subset.empty:
            continue
        exact = subset[subset["year"].eq(int(year))]
        if not exact.empty:
            return exact.copy(), f"{label}_exact_year"
        earlier = subset[subset["year"].le(int(year))]
        if not earlier.empty:
            latest_year = int(earlier["year"].max())
            return earlier[earlier["year"].eq(latest_year)].copy(), f"{label}_latest_prior"
        earliest_year = int(subset["year"].min())
        return subset[subset["year"].eq(earliest_year)].copy(), f"{label}_earliest_future"
    return pd.DataFrame(columns=bridge.columns), "unmapped"


def prepare_historical_feedstocks(
    history: pd.DataFrame,
    bridge: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    history = history.copy()
    bridge = bridge.copy()
    for col, default in [
        ("country_name", ""),
        ("feedstock_category", ""),
        ("model_commodity", ""),
        ("market_link", False),
        ("lhv_gj_per_tdm", np.nan),
        ("conversion_efficiency", 1.0),
        ("dry_matter_fraction", 1.0),
        ("source", ""),
        ("notes", ""),
    ]:
        if col not in bridge.columns:
            bridge[col] = default
    for col in ["lhv_gj_per_tdm", "conversion_efficiency", "dry_matter_fraction"]:
        bridge[col] = pd.to_numeric(bridge[col], errors="coerce")
    required = {
        "M49_Country_Code",
        "country_name",
        "year",
        "carrier",
        "final_consumption_tj",
    }
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"bioenergy history missing columns: {sorted(missing)}")
    history["M49_Country_Code"] = history["M49_Country_Code"].apply(normalize_m49)
    history["year"] = pd.to_numeric(history["year"], errors="coerce")
    history["final_consumption_tj"] = pd.to_numeric(
        history["final_consumption_tj"],
        errors="coerce",
    ).fillna(0.0)
    history = history.dropna(subset=["year"])
    history["year"] = history["year"].astype(int)
    history = history[history["final_consumption_tj"].gt(0.0)]

    rows: List[Dict[str, object]] = []
    diagnostics: List[Dict[str, object]] = []
    for record in history.itertuples(index=False):
        country = str(record.M49_Country_Code)
        carrier = str(record.carrier)
        year = int(record.year)
        shares, method = _select_shares(bridge, country, carrier, year)
        if shares.empty:
            diagnostics.append(
                {
                    "M49_Country_Code": country,
                    "country_name": record.country_name,
                    "year": year,
                    "carrier": carrier,
                    "final_consumption_tj": float(record.final_consumption_tj),
                    "mapped_share": 0.0,
                    "unmapped_energy_tj": float(record.final_consumption_tj),
                    "mapping_method": method,
                }
            )
            continue
        share_sum = float(shares["share"].sum())
        scale = 1.0 / share_sum if share_sum > 1.0 + 1e-9 else 1.0
        mapped_share = min(1.0, share_sum)
        for share_row in shares.itertuples(index=False):
            allocated_share = float(share_row.share) * scale
            rows.append(
                {
                    "scenario": "historical",
                    "M49_Country_Code": country,
                    "country_name": record.country_name,
                    "year": year,
                    "carrier": carrier,
                    "feedstock": share_row.feedstock,
                    "feedstock_category": share_row.feedstock_category,
                    "model_commodity": share_row.model_commodity,
                    "market_link": share_row.market_link,
                    "energy_basis": "final",
                    "energy_target_tj": float(record.final_consumption_tj) * allocated_share,
                    "feedstock_demand_t": np.nan,
                    "feedstock_demand_tdm": np.nan,
                    "share": 1.0,
                    "lhv_gj_per_tdm": share_row.lhv_gj_per_tdm,
                    "conversion_efficiency": share_row.conversion_efficiency,
                    "dry_matter_fraction": share_row.dry_matter_fraction,
                    "source": share_row.source,
                    "notes": (
                        f"carrier bridge={method}; original_share_sum={share_sum:.6g}; "
                        f"{share_row.notes}"
                    ).strip(),
                }
            )
        diagnostics.append(
            {
                "M49_Country_Code": country,
                "country_name": record.country_name,
                "year": year,
                "carrier": carrier,
                "final_consumption_tj": float(record.final_consumption_tj),
                "mapped_share": mapped_share,
                "unmapped_energy_tj": float(record.final_consumption_tj) * max(0.0, 1.0 - mapped_share),
                "mapping_method": method,
            }
        )
    columns = [
        "scenario",
        "M49_Country_Code",
        "country_name",
        "year",
        "carrier",
        "feedstock",
        "feedstock_category",
        "model_commodity",
        "market_link",
        "energy_basis",
        "energy_target_tj",
        "feedstock_demand_t",
        "feedstock_demand_tdm",
        "share",
        "lhv_gj_per_tdm",
        "conversion_efficiency",
        "dry_matter_fraction",
        "source",
        "notes",
    ]
    return pd.DataFrame(rows, columns=columns), pd.DataFrame(diagnostics)


def main() -> None:
    paths = DataPaths()
    input_dir = Path(get_input_base()) / "Bioenergy"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history",
        default=str(input_dir / "bioenergy_history_country_carrier.csv"),
    )
    parser.add_argument(
        "--bridge",
        default=paths.bioenergy_carrier_feedstock_share_csv,
    )
    parser.add_argument("--output-dir", default=str(input_dir))
    parser.add_argument(
        "--no-default-carrier-map",
        dest="use_default_carrier_map",
        action="store_false",
        help="Fail instead of using the S0_53 one-to-one carrier map when the bridge CSV is missing or empty.",
    )
    parser.set_defaults(use_default_carrier_map=True)
    parser.add_argument("--write-input", action="store_true")
    parser.add_argument("--overwrite-input", action="store_true")
    args = parser.parse_args()

    history = read_csv_cached(args.history, low_memory=False)
    bridge = _load_bridge(args.bridge, use_default_carrier_map=bool(args.use_default_carrier_map))
    feedstocks, diagnostics = prepare_historical_feedstocks(history, bridge)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "bioenergy_historical_feedstock.csv"
    feedstocks.to_csv(output_path, index=False, encoding="utf-8-sig")
    diagnostics.to_csv(
        output_dir / "bioenergy_feedstock_bridge_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    total = diagnostics["final_consumption_tj"].sum() if not diagnostics.empty else 0.0
    unmapped = diagnostics["unmapped_energy_tj"].sum() if not diagnostics.empty else 0.0
    print(
        f"feedstock rows={len(feedstocks)} total_energy_tj={total:.6g} "
        f"unmapped_tj={unmapped:.6g}"
    )
    if args.write_input:
        input_dir = Path(get_input_base()) / "Bioenergy"
        input_dir.mkdir(parents=True, exist_ok=True)
        input_path = input_dir / "bioenergy_historical_feedstock.csv"
        if input_path.resolve() == output_path.resolve():
            print(f"input path already written at {input_path} rows={len(feedstocks)}")
            return
        if input_path.exists() and not args.overwrite_input:
            raise FileExistsError(
                f"{input_path} already exists. Re-run with --overwrite-input to replace it."
            )
        feedstocks.to_csv(input_path, index=False, encoding="utf-8-sig")
        print(f"wrote {input_path} rows={len(feedstocks)}")


if __name__ == "__main__":
    main()
