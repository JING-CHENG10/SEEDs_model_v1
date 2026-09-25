# -*- coding: utf-8 -*-
"""Build a raster-derived eligible-land mask for dedicated bioenergy crops.

The output raster is aligned to the local LUH2 0.25-degree country mask.  ESA
WorldCover is sampled from public cloud-optimized GeoTIFF tiles; optional WDPA
and GAEZ inputs can further exclude protected or unsuitable cells.  The country
CSV emitted here is the file consumed by S0_52_prepare_bioenergy_resource_constraints.py.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin, rowcol
import xarray as xr

from config_paths import get_input_base
from S2_0_load_data import DataPaths
from S0_52_prepare_bioenergy_resource_constraints import (
    ENERGY_CROP_DEFAULTS,
    _country_name_to_m49,
    _load_luh2_land_cover,
    _mask_id_to_m49,
)


WORLDCOVER_GRID_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v100/2020/"
    "esa_worldcover_2020_grid.geojson"
)
WORLDCOVER_S3_PREFIX = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
WORLDCOVER_AZURE_PREFIX = "https://ai4edataeuwest.blob.core.windows.net/esa-worldcover/v200/2021/map"
PLANETARY_COMPUTER_SIGN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"
WORLDCOVER_SOURCE = "ESA WorldCover 2021 v200"
WDPA_SOURCE = "WDPA/Protected Planet"
GAEZ_SOURCE = "GAEZ v4"
DEFAULT_ELIGIBLE_CLASSES = (20, 30, 60)  # shrubland, grassland, bare/sparse vegetation
EARTH_RADIUS_M = 6_371_008.8


def _format_worldcover_tile(lat: float, lon: float) -> str:
    lat0 = int(math.floor(lat / 3.0) * 3)
    lon0 = int(math.floor(lon / 3.0) * 3)
    lat0 = max(-90, min(87, lat0))
    lon0 = max(-180, min(177, lon0))
    ns = "N" if lat0 >= 0 else "S"
    ew = "E" if lon0 >= 0 else "W"
    return f"{ns}{abs(lat0):02d}{ew}{abs(lon0):03d}"


def _worldcover_url(tile: str, source: str = "aws") -> str:
    source_norm = str(source or "aws").strip().lower()
    prefix = WORLDCOVER_AZURE_PREFIX if source_norm in {"planetary-computer", "pc", "azure"} else WORLDCOVER_S3_PREFIX
    return f"{prefix}/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"


def _sign_planetary_computer_href(href: str) -> str:
    try:
        resp = requests.get(PLANETARY_COMPUTER_SIGN_URL, params={"href": href}, timeout=30)
        resp.raise_for_status()
        signed = resp.json().get("href")
        return str(signed or href)
    except Exception:
        return href


def _load_country_mask(country_mask_nc: Path, dict_v3_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, str], Dict[str, str]]:
    ds = xr.open_dataset(country_mask_nc)
    mask_var = "id1" if "id1" in ds.data_vars else next(v for v in ds.data_vars if v.lower() != "crs")
    da = ds[mask_var].sortby("lat")
    lat = np.asarray(da["lat"].values, dtype=float)
    lon = np.asarray(da["lon"].values, dtype=float)
    ids = np.where(np.isfinite(da.values), np.rint(da.values), 0).astype("int32")
    id_to_m49 = _mask_id_to_m49(dict_v3_path)
    _, m49_to_name, _ = _country_name_to_m49(dict_v3_path)
    return lat, lon, ids, id_to_m49, m49_to_name


def _cell_area_ha(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    if len(lat) < 2 or len(lon) < 2:
        raise ValueError("lat/lon grid must have at least two cells")
    dlat = abs(float(np.nanmedian(np.diff(lat))))
    dlon = abs(float(np.nanmedian(np.diff(lon))))
    lat1 = np.deg2rad(np.clip(lat - dlat / 2.0, -90.0, 90.0))
    lat2 = np.deg2rad(np.clip(lat + dlat / 2.0, -90.0, 90.0))
    dlon_rad = math.radians(dlon)
    strip = (EARTH_RADIUS_M ** 2) * dlon_rad * (np.sin(lat2) - np.sin(lat1))
    return np.abs(strip)[:, None] * np.ones((1, len(lon))) / 10_000.0


def _load_worldcover_tiles() -> Set[str]:
    grid = gpd.read_file(WORLDCOVER_GRID_URL)
    if "ll_tile" not in grid.columns:
        raise ValueError(f"WorldCover grid lacks ll_tile column: {WORLDCOVER_GRID_URL}")
    return {str(v).strip() for v in grid["ll_tile"].dropna().astype(str)}


def _sample_worldcover_classes(
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    valid_country: np.ndarray,
    eligible_classes: Set[int],
    max_tiles: Optional[int] = None,
    workers: int = 8,
    cache_dir: Optional[Path] = None,
    time_budget_minutes: float = 0.0,
    source: str = "aws",
) -> Tuple[np.ndarray, pd.DataFrame]:
    available_tiles = _load_worldcover_tiles()
    rows, cols = np.where(valid_country)
    tile_to_points: Dict[str, List[Tuple[int, float, float]]] = {}
    for r, c in zip(rows.tolist(), cols.tolist()):
        tile = _format_worldcover_tile(float(lat[r]), float(lon[c]))
        if tile not in available_tiles:
            continue
        flat = r * len(lon) + c
        tile_to_points.setdefault(tile, []).append((flat, float(lon[c]), float(lat[r])))
    if max_tiles is not None and max_tiles > 0:
        tile_to_points = dict(list(sorted(tile_to_points.items()))[: int(max_tiles)])

    sampled = np.zeros(valid_country.shape, dtype="uint8")
    diagnostics: List[Dict[str, Any]] = []
    cache_dir = Path(cache_dir) if cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    env_opts = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
        "AWS_NO_SIGN_REQUEST": "YES",
    }
    def _sample_tile(tile_items: Tuple[str, List[Tuple[int, float, float]]]) -> Tuple[str, np.ndarray, np.ndarray, Dict[str, Any]]:
        tile, items = tile_items
        url_raw = _worldcover_url(tile, source=source)
        url = _sign_planetary_computer_href(url_raw) if str(source).strip().lower() in {"planetary-computer", "pc", "azure"} else url_raw
        try:
            with rasterio.Env(**env_opts):
                with rasterio.open(url) as src:
                    coords = [(x, y) for _, x, y in items]
                    vals = [int(v[0]) if len(v) else 0 for v in src.sample(coords)]
        except Exception as exc:
            return tile, np.asarray([], dtype=np.int64), np.asarray([], dtype=bool), {
                "issue": "worldcover_tile_read_failed",
                "tile": tile,
                "url": url_raw,
                "source": source,
                "message": str(exc)[:500],
                "points": len(items),
            }
        flat_idx = np.asarray([flat for flat, _, _ in items], dtype=np.int64)
        is_eligible = np.isin(np.asarray(vals, dtype=np.int16), list(eligible_classes))
        return tile, flat_idx, is_eligible, {
            "issue": "worldcover_tile_sampled",
            "tile": tile,
            "source": source,
            "points": len(items),
            "eligible_points": int(is_eligible.sum()),
        }

    tile_items_all = list(sorted(tile_to_points.items()))
    tile_items: List[Tuple[str, List[Tuple[int, float, float]]]] = []
    for tile, items in tile_items_all:
        cache_path = cache_dir / f"{tile}.npz" if cache_dir else None
        if cache_path and cache_path.exists():
            try:
                cached = np.load(cache_path, allow_pickle=False)
                flat_idx = cached["flat_idx"].astype(np.int64)
                is_eligible = cached["is_eligible"].astype(bool)
                sampled.reshape(-1)[flat_idx[is_eligible]] = 1
                diagnostics.append({
                    "issue": "worldcover_tile_loaded_from_cache",
                    "tile": tile,
                    "points": int(len(flat_idx)),
                    "eligible_points": int(is_eligible.sum()),
                    "cache_path": str(cache_path),
                })
                continue
            except Exception as exc:
                diagnostics.append({
                    "issue": "worldcover_tile_cache_read_failed",
                    "tile": tile,
                    "cache_path": str(cache_path),
                    "message": str(exc)[:500],
                })
        tile_items.append((tile, items))
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(_sample_tile, item) for item in tile_items]
        for future in as_completed(futures):
            tile, flat_idx, is_eligible, diagnostic = future.result()
            if len(flat_idx):
                sampled.reshape(-1)[flat_idx[is_eligible]] = 1
                if cache_dir:
                    try:
                        np.savez_compressed(
                            cache_dir / f"{tile}.npz",
                            flat_idx=flat_idx,
                            is_eligible=is_eligible.astype("uint8"),
                        )
                    except Exception as exc:
                        diagnostic = dict(diagnostic)
                        diagnostic["cache_write_error"] = str(exc)[:500]
            diagnostics.append(diagnostic)
            if time_budget_minutes and time_budget_minutes > 0:
                elapsed_minutes = (time.time() - start_time) / 60.0
                if elapsed_minutes >= float(time_budget_minutes):
                    for pending in futures:
                        pending.cancel()
                    diagnostics.append({
                        "issue": "worldcover_time_budget_reached",
                        "elapsed_minutes": elapsed_minutes,
                        "remaining_submitted_tiles": sum(1 for f in futures if not f.done()),
                    })
                    break
    return sampled, pd.DataFrame(diagnostics)


def _rasterize_wdpa_exclusions(
    wdpa_vector: Optional[Path],
    *,
    shape: Tuple[int, int],
    transform: rasterio.Affine,
) -> Tuple[np.ndarray, pd.DataFrame]:
    if not wdpa_vector:
        return np.zeros(shape, dtype=bool), pd.DataFrame([{"issue": "wdpa_not_provided"}])
    if not wdpa_vector.exists():
        return np.zeros(shape, dtype=bool), pd.DataFrame([{"issue": "wdpa_missing", "path": str(wdpa_vector)}])
    gdf = gpd.read_file(wdpa_vector)
    if gdf.empty:
        return np.zeros(shape, dtype=bool), pd.DataFrame([{"issue": "wdpa_empty", "path": str(wdpa_vector)}])
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    if "MARINE" in gdf.columns:
        gdf = gdf[gdf["MARINE"].astype(str).isin(["0", "1"])]
    geoms = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not geoms:
        return np.zeros(shape, dtype=bool), pd.DataFrame([{"issue": "wdpa_no_valid_geometry", "path": str(wdpa_vector)}])
    protected_desc = rasterize(geoms, out_shape=shape, transform=transform, fill=0, dtype="uint8", all_touched=True)
    protected_asc = np.flipud(protected_desc).astype(bool)
    return protected_asc, pd.DataFrame([{
        "issue": "wdpa_rasterized",
        "path": str(wdpa_vector),
        "features": len(geoms),
        "excluded_cells": int(protected_asc.sum()),
    }])


def _sample_gaez_suitability(
    gaez_raster: Optional[Path],
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    valid_country: np.ndarray,
    threshold: float,
) -> Tuple[np.ndarray, pd.DataFrame]:
    if not gaez_raster:
        return np.ones(valid_country.shape, dtype=bool), pd.DataFrame([{"issue": "gaez_not_provided"}])
    if not gaez_raster.exists():
        return np.ones(valid_country.shape, dtype=bool), pd.DataFrame([{"issue": "gaez_missing", "path": str(gaez_raster)}])
    rows, cols = np.where(valid_country)
    suitable = np.zeros(valid_country.shape, dtype=bool)
    with rasterio.open(gaez_raster) as src:
        coords = [(float(lon[c]), float(lat[r])) for r, c in zip(rows.tolist(), cols.tolist())]
        vals = np.asarray([float(v[0]) if len(v) else np.nan for v in src.sample(coords)], dtype=float)
    keep = np.isfinite(vals) & (vals >= float(threshold))
    suitable[rows[keep], cols[keep]] = True
    return suitable, pd.DataFrame([{
        "issue": "gaez_sampled",
        "path": str(gaez_raster),
        "threshold": threshold,
        "sampled_cells": len(rows),
        "suitable_cells": int(keep.sum()),
    }])


def _luh2_grassland_fraction_mask(
    *,
    land_cover_xlsx: Path,
    country_ids: np.ndarray,
    id_to_m49: Dict[int, str],
    area_ha: np.ndarray,
    base_year: int,
    grassland_share: float,
) -> Tuple[np.ndarray, Dict[str, float], pd.DataFrame]:
    land = _load_luh2_land_cover(str(land_cover_xlsx))
    year_col = f"Y{int(base_year)}"
    if year_col not in land.columns:
        raise ValueError(f"{land_cover_xlsx} sheet LUH2 missing {year_col}")
    grass = land[
        land["Land cover"].fillna("").astype(str).str.strip().str.lower().eq("grassland")
    ].copy()
    grass["M49_Country_Code"] = grass["M49_Country_Code"].astype(str).str.strip()
    grass[year_col] = pd.to_numeric(grass[year_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    target_by_m49 = {
        str(row["M49_Country_Code"]).strip(): float(row[year_col]) * max(0.0, float(grassland_share))
        for _, row in grass.iterrows()
    }
    fraction = np.zeros(country_ids.shape, dtype="float32")
    flat_id = country_ids.reshape(-1)
    flat_area = area_ha.reshape(-1)
    flat_fraction = fraction.reshape(-1)
    area_by_m49: Dict[str, float] = {}
    for mask_id, m49 in id_to_m49.items():
        idx = flat_id == int(mask_id)
        grid_area = float(flat_area[idx].sum())
        target_area = max(0.0, float(target_by_m49.get(m49, 0.0)))
        area_by_m49[m49] = target_area
        if grid_area > 0 and target_area > 0:
            flat_fraction[idx] = min(1.0, target_area / grid_area)
    diag = pd.DataFrame([{
        "issue": "luh2_grassland_fraction_fallback",
        "land_cover_xlsx": str(land_cover_xlsx),
        "base_year": int(base_year),
        "grassland_share": float(grassland_share),
        "message": (
            "Fallback raster contains country-uniform eligible fractions derived from LUH2 "
            "grassland area. It is not a substitute for ESA WorldCover/WDPA/GAEZ overlay."
        ),
    }])
    return fraction, area_by_m49, diag


def _write_raster(path: Path, data_asc: np.ndarray, *, lat: np.ndarray, lon: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    res_y = abs(float(np.nanmedian(np.diff(lat))))
    res_x = abs(float(np.nanmedian(np.diff(lon))))
    transform = from_origin(float(lon.min() - res_x / 2.0), float(lat.max() + res_y / 2.0), res_x, res_y)
    profile = {
        "driver": "GTiff",
        "height": data_asc.shape[0],
        "width": data_asc.shape[1],
        "count": 1,
        "dtype": str(data_asc.dtype),
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": "deflate",
        "nodata": 0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.flipud(data_asc), 1)


def build_mask(args: argparse.Namespace) -> Dict[str, pd.DataFrame]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = DataPaths()
    country_mask = Path(args.country_mask_nc) if args.country_mask_nc else Path(get_input_base()) / "Land" / "LUH2_GCB2019" / "data" / "mask_LUH2_025d.nc"
    dict_v3 = args.dict_xlsx or paths.dict_v3_path
    lat, lon, country_ids, id_to_m49, m49_to_name = _load_country_mask(country_mask, dict_v3)
    known_ids = np.asarray(sorted(id_to_m49), dtype=np.int32)
    valid_country = np.isin(country_ids, known_ids)
    area_ha = _cell_area_ha(lat, lon)
    fallback_area_by_m49: Dict[str, float] = {}

    eligible_classes = {int(x) for x in str(args.worldcover_eligible_classes).replace(";", ",").split(",") if str(x).strip()}
    if args.worldcover_mode == "s3-sample":
        wc_mask, wc_diag = _sample_worldcover_classes(
            lat=lat,
            lon=lon,
            valid_country=valid_country,
            eligible_classes=eligible_classes,
            max_tiles=args.max_worldcover_tiles if args.max_worldcover_tiles > 0 else None,
            workers=int(args.worldcover_workers),
            cache_dir=Path(args.worldcover_cache_dir) if args.worldcover_cache_dir else None,
            time_budget_minutes=float(args.worldcover_time_budget_minutes or 0.0),
            source=str(args.worldcover_source),
        )
        eligibility_fraction = wc_mask.astype("float32")
    elif args.worldcover_mode == "luh2-grassland-fallback":
        land_cover_xlsx = Path(args.land_cover_xlsx) if args.land_cover_xlsx else Path(paths.land_cover_base_xlsx)
        eligibility_fraction, fallback_area_by_m49, wc_diag = _luh2_grassland_fraction_mask(
            land_cover_xlsx=land_cover_xlsx,
            country_ids=country_ids,
            id_to_m49=id_to_m49,
            area_ha=area_ha,
            base_year=int(args.base_year),
            grassland_share=float(args.fallback_grassland_share),
        )
    elif args.worldcover_mode == "off":
        eligibility_fraction = valid_country.astype("float32")
        wc_diag = pd.DataFrame([{"issue": "worldcover_sampling_off"}])
    else:
        raise ValueError(f"Unsupported --worldcover-mode: {args.worldcover_mode}")

    res_y = abs(float(np.nanmedian(np.diff(lat))))
    res_x = abs(float(np.nanmedian(np.diff(lon))))
    transform = from_origin(float(lon.min() - res_x / 2.0), float(lat.max() + res_y / 2.0), res_x, res_y)
    wdpa_excluded, wdpa_diag = _rasterize_wdpa_exclusions(
        Path(args.wdpa_vector) if args.wdpa_vector else None,
        shape=valid_country.shape,
        transform=transform,
    )
    gaez_suitable, gaez_diag = _sample_gaez_suitability(
        Path(args.gaez_suitability_raster) if args.gaez_suitability_raster else None,
        lat=lat,
        lon=lon,
        valid_country=valid_country,
        threshold=float(args.gaez_threshold),
    )
    eligibility_fraction = np.where(valid_country & ~wdpa_excluded & gaez_suitable, eligibility_fraction, 0.0).astype("float32")
    eligible = eligibility_fraction > 0

    rows: List[Dict[str, Any]] = []
    flat_id = country_ids.reshape(-1)
    flat_fraction = eligibility_fraction.reshape(-1)
    flat_area = area_ha.reshape(-1)
    for mask_id, m49 in sorted(id_to_m49.items(), key=lambda kv: kv[1]):
        idx = flat_id == int(mask_id)
        if args.worldcover_mode == "luh2-grassland-fallback" and not args.wdpa_vector and not args.gaez_suitability_raster:
            eligible_area = float(fallback_area_by_m49.get(m49, 0.0))
        else:
            eligible_area = float((flat_area[idx] * flat_fraction[idx]).sum())
        if eligible_area <= 0:
            continue
        for feedstock in ENERGY_CROP_DEFAULTS:
            land_source = (
                WORLDCOVER_SOURCE
                if args.worldcover_mode == "s3-sample"
                else "LUH2 grassland fallback"
            )
            rows.append({
                "M49_Country_Code": m49,
                "country_name": m49_to_name.get(m49, ""),
                "feedstock": feedstock,
                "eligible_land_area_ha": eligible_area,
                "source": "; ".join([
                    land_source,
                    WDPA_SOURCE if args.wdpa_vector else "WDPA not applied",
                    GAEZ_SOURCE if args.gaez_suitability_raster else "GAEZ not applied",
                ]),
                "notes": (
                    f"Raster-derived eligible land aligned to LUH2 0.25 degree grid; mode={args.worldcover_mode}. "
                    f"WorldCover eligible classes={sorted(eligible_classes)}. "
                    "Default excludes tree cover, cropland, built-up, water, wetlands, mangroves, snow/ice."
                ),
            })
    mask_df = pd.DataFrame(rows)
    csv_path = out_dir / args.output_csv
    raster_path = out_dir / args.output_raster
    diag_path = out_dir / args.diagnostics_csv
    mask_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    _write_raster(raster_path, eligibility_fraction, lat=lat, lon=lon)
    diagnostics = pd.concat([
        wc_diag,
        wdpa_diag,
        gaez_diag,
        pd.DataFrame([{
            "issue": "eligible_mask_written",
            "country_mask_nc": str(country_mask),
            "output_csv": str(csv_path),
            "output_raster": str(raster_path),
            "eligible_cells": int(eligible.sum()),
            "eligible_area_ha": float((area_ha * eligibility_fraction).sum()),
            "rows": len(mask_df),
        }]),
    ], ignore_index=True, sort=False)
    diagnostics.to_csv(diag_path, index=False, encoding="utf-8-sig")
    return {"eligible_mask": mask_df, "diagnostics": diagnostics}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    default_input = Path(get_input_base()) / "Bioenergy"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(default_input))
    parser.add_argument("--output-csv", default="bioenergy_energy_crop_eligible_land_mask.csv")
    parser.add_argument("--output-raster", default="bioenergy_energy_crop_eligible_land_mask.tif")
    parser.add_argument("--diagnostics-csv", default="bioenergy_energy_crop_eligible_land_mask_diagnostics.csv")
    parser.add_argument("--country-mask-nc", default="")
    parser.add_argument("--dict-xlsx", default="")
    parser.add_argument("--worldcover-mode", choices=["s3-sample", "luh2-grassland-fallback", "off"], default="s3-sample")
    parser.add_argument("--worldcover-eligible-classes", default=",".join(str(x) for x in DEFAULT_ELIGIBLE_CLASSES))
    parser.add_argument("--max-worldcover-tiles", type=int, default=0, help="Diagnostic limit; 0 means all available intersecting tiles.")
    parser.add_argument("--worldcover-workers", type=int, default=8)
    parser.add_argument("--worldcover-source", choices=["aws", "planetary-computer"], default="aws")
    parser.add_argument("--worldcover-cache-dir", default="", help="Optional per-tile cache directory for resumable ESA WorldCover sampling.")
    parser.add_argument("--worldcover-time-budget-minutes", type=float, default=0.0, help="Optional soft runtime budget for WorldCover sampling; cached tiles make reruns resumable.")
    parser.add_argument("--land-cover-xlsx", default="")
    parser.add_argument("--base-year", type=int, default=2020)
    parser.add_argument("--fallback-grassland-share", type=float, default=0.01)
    parser.add_argument("--wdpa-vector", default="", help="Optional WDPA polygon vector file readable by GeoPandas.")
    parser.add_argument("--gaez-suitability-raster", default="", help="Optional GAEZ suitability/index raster sampled at LUH2 cell centers.")
    parser.add_argument("--gaez-threshold", type=float, default=40.0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    outputs = build_mask(parse_args(argv))
    print(
        "wrote eligible land mask rows=%d diagnostics=%d"
        % (len(outputs["eligible_mask"]), len(outputs["diagnostics"]))
    )


if __name__ == "__main__":
    main()
