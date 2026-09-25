# -*- coding: utf-8 -*-
"""
Build S5.3 panel plotting data from merged batch CSV outputs.

This script intentionally reads only the merged batch-level CSV files produced
by S5_3_3. It does not scan per-run directories, copy result trees, or reopen
`Emis/` outputs from individual scenarios.

Outputs by default:
  - figure_panel_dataset_long_rebuilt.csv
  - figure_panel_dataset_long_plot_ready.csv
  - figure_panel_global_emissions_detail_long_rebuilt.csv
  - figure_panel_missing_points_audit.csv
  - figure_panel_data_summary.csv
"""
from __future__ import annotations

import argparse
import gc
import itertools
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from config_paths import get_results_base

CONFIG = {
    "panel_root_dir": "",  # empty -> <NZF_OUTPUT_DIR>/Panel_Yield_EF
    "input_results_csv": "figure_panel_dataset_long.csv",
    "input_detail_csv": "figure_panel_global_emissions_detail_long.csv",
    "target_year": 2080,
    "raw_results_csv": "figure_panel_dataset_long_rebuilt.csv",
    "plot_ready_csv": "figure_panel_dataset_long_plot_ready.csv",
    "detail_results_csv": "figure_panel_global_emissions_detail_long_rebuilt.csv",
    "missing_audit_csv": "figure_panel_missing_points_audit.csv",
    "summary_csv": "figure_panel_data_summary.csv",
    "detail_chunksize": 50000,
    "stream_detail_dedup": False,
    "forest_area_change_pct_values": None,
    "ruminant_kcal_share_cap_pct_values": None,
    "yield_change_pct_values": None,
    "emission_factor_change_pct_values": None,
    "interpolation_neighbors_k": 4,
    "interpolation_power": 2.0,
    "allow_cross_panel_fallback": False,
    "quality_gate_enabled": True,
    "min_observed_fraction": 0.50,
    "max_interpolated_fraction": 0.50,
    "min_observed_points_per_panel": 25,
    "min_observed_fraction_per_panel": 0.25,
    "max_interpolated_fraction_per_panel": 0.75,
}


SCENARIO_RE = re.compile(
    r"^FIG_PANEL_F(?P<forest>[mp]\d+)_R(?P<rumi>[mp]\d+)_Y(?P<yield>[mp]\d+)_E(?P<ef>[mp]\d+)$",
    flags=re.IGNORECASE,
)

INVALID_TOTAL_CO2EQ_GT_VALUES = (1.264874,)
RUMINANT_CAP_COL = "ruminant_kcal_share_cap_pct"
RUMINANT_CAP_VALUES_KEY = "ruminant_kcal_share_cap_pct_values"
DEPRECATED_RUMINANT_CAP_COL = "ruminant_intake_change_pct"
DEPRECATED_RUMINANT_CAP_VALUES_KEY = "ruminant_intake_change_pct_values"

RAW_KEEP_COLS = [
    "scenario_id",
    "panel_row",
    "panel_col",
    "forest_area_change_pct",
    "forest_area_multiplier",
    RUMINANT_CAP_COL,
    "yield_change_pct",
    "yield_multiplier",
    "emission_factor_change_pct",
    "emission_factor_multiplier",
    "target_year",
    "run_status",
    "afolu_emissions_gt_co2eq_yr",
    "scenario_dir",
    "diagnostic_scenario_dir",
    "model_status_code",
    "model_status_text",
    "iis_summary",
    "error_type",
    "error_message",
    "has_run_dir",
    "has_fast_summary",
    "has_global_detail",
    "pale_population",
    "pale_ag_output_kcal",
    "pale_land_cropland_ha",
    "pale_land_pasture_ha",
    "pale_land_cropland_pasture_ha",
    "pale_luc_emissions_gt_co2eq_yr",
    "pale_ag_production_emissions_gt_co2eq_yr",
    "pale_total_emissions_gt_co2eq_yr",
    "pale_a_per_p_kcal_cap_yr",
    "pale_l_per_a_ha_per_kcal",
    "pale_luc_intensity_gt_per_ha",
    "pale_ag_intensity_gt_per_kcal",
    "pale_land_source",
    "pale_ag_output_source",
    "pale_emissions_source",
    "pale_missing_reason",
]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_abs_path(raw_path: Path) -> Path:
    path = raw_path.expanduser()
    if not path.is_absolute():
        root = _project_root()
        resolved = (root / path).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Relative S5_3 output path escapes project root: {raw_path}. "
                "Use an absolute path instead."
            ) from exc
        return resolved
    return path.resolve()


def _default_output_base() -> Path:
    return _resolve_abs_path(Path(get_results_base()))


def _resolve_panel_root_dir(raw_panel_root_dir: object = "") -> Path:
    text = str(raw_panel_root_dir or "").strip()
    if text:
        return _resolve_abs_path(Path(text))
    return _default_output_base() / "Panel_Yield_EF"


def _sync_panel_root_environment(panel_root: Path) -> None:
    root = _resolve_abs_path(panel_root)
    os.environ["PANEL_OUTPUT_DIR"] = str(root)
    os.environ["NZF_OUTPUT_DIR"] = str(root.parent)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build S5_3 panel plotting data.")
    parser.add_argument(
        "--output-dir",
        "--panel-root-dir",
        dest="panel_root_dir",
        type=str,
        default=None,
        help="Panel_Yield_EF root directory. Defaults to PANEL_OUTPUT_DIR or <NZF_OUTPUT_DIR>/Panel_Yield_EF.",
    )
    parser.add_argument(
        "--detail-chunksize",
        type=int,
        default=int(os.environ.get("PANEL_SUMMARY_DETAIL_CHUNKSIZE") or CONFIG.get("detail_chunksize") or 50000),
        help="Rows per chunk when rebuilding the global-emissions detail CSV. Use 0 to read whole file.",
    )
    parser.add_argument(
        "--dedup-detail",
        action="store_true",
        default=bool(CONFIG.get("stream_detail_dedup", False)),
        help=(
            "Deduplicate rebuilt detail rows while streaming. Disabled by default to avoid "
            "a large in-memory key set; merged batch detail rows should be disjoint."
        ),
    )
    return parser


def _pct_to_multiplier(pct: float) -> float:
    return 1.0 + float(pct) / 100.0


def _normalise_ruminant_cap_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if RUMINANT_CAP_COL not in out.columns and DEPRECATED_RUMINANT_CAP_COL in out.columns:
        out = out.rename(columns={DEPRECATED_RUMINANT_CAP_COL: RUMINANT_CAP_COL})
    elif RUMINANT_CAP_COL in out.columns and DEPRECATED_RUMINANT_CAP_COL in out.columns:
        out[RUMINANT_CAP_COL] = out[RUMINANT_CAP_COL].where(
            out[RUMINANT_CAP_COL].notna(),
            out[DEPRECATED_RUMINANT_CAP_COL],
        )
        out = out.drop(columns=[DEPRECATED_RUMINANT_CAP_COL])
    if "ruminant_intake_multiplier" in out.columns:
        out = out.drop(columns=["ruminant_intake_multiplier"])
    return out


def _configured_ruminant_cap_values(cfg: Dict[str, object]) -> Optional[Sequence[float]]:
    values = cfg.get(RUMINANT_CAP_VALUES_KEY)
    if not values:
        values = cfg.get(DEPRECATED_RUMINANT_CAP_VALUES_KEY)
    return values


def _fmt_pct(pct: float) -> str:
    val = int(round(float(pct)))
    sign = "p" if val >= 0 else "m"
    return f"{sign}{abs(val)}"


def _scenario_id(
    *,
    forest_pct: float,
    ruminant_pct: float,
    yield_pct: float,
    ef_pct: float,
) -> str:
    return (
        "FIG_PANEL"
        f"_F{_fmt_pct(forest_pct)}"
        f"_R{_fmt_pct(ruminant_pct)}"
        f"_Y{_fmt_pct(yield_pct)}"
        f"_E{_fmt_pct(ef_pct)}"
    )


def _parse_signed_token(token: str) -> float:
    token_s = str(token or "").strip().lower()
    if not token_s:
        raise ValueError("empty signed token")
    sign = token_s[0]
    magnitude = token_s[1:]
    if sign not in {"p", "m"}:
        raise ValueError(f"invalid signed token: {token}")
    try:
        value = float(magnitude)
    except Exception as exc:
        raise ValueError(f"invalid signed token: {token}") from exc
    return value if sign == "p" else -value


def _parse_scenario_id(scenario_id: str) -> Optional[Dict[str, float]]:
    m = SCENARIO_RE.match(str(scenario_id or "").strip())
    if not m:
        return None
    return {
        "forest_area_change_pct": _parse_signed_token(m.group("forest")),
        RUMINANT_CAP_COL: _parse_signed_token(m.group("rumi")),
        "yield_change_pct": _parse_signed_token(m.group("yield")),
        "emission_factor_change_pct": _parse_signed_token(m.group("ef")),
    }


def _read_required_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found: {path}. Run S5_3_3_merge_panel_yield_ef_batches.py first."
        )
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to read {label}: {path}: {type(exc).__name__}: {exc}") from exc
    if df.empty:
        raise RuntimeError(f"{label} is empty: {path}")
    return df


def _read_optional_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        print(f"[S5_3_2][WARN] optional {label} not found: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        print(f"[S5_3_2][WARN] failed to read optional {label}: {path}: {type(exc).__name__}: {exc}")
        return pd.DataFrame()


def _iter_csv_chunks_safe(path: Path, *, chunksize: int, label: str):
    try:
        if chunksize > 0:
            reader = pd.read_csv(path, chunksize=chunksize)
            for chunk in reader:
                yield chunk
        else:
            yield pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to stream {label}: {path}: {type(exc).__name__}: {exc}") from exc


def _deduplicate_chunk_by_hash(
    df: pd.DataFrame,
    *,
    dedup_cols: Sequence[str],
    seen_hashes: Set[int],
) -> Tuple[pd.DataFrame, int]:
    keep_cols = [c for c in dedup_cols if c in df.columns]
    if not keep_cols or df.empty:
        return df, 0
    key_frame = df.loc[:, keep_cols].astype("string").fillna("<NA>")
    hashes = pd.util.hash_pandas_object(key_frame, index=False).to_numpy(dtype="uint64")
    keep_mask: List[bool] = []
    skipped = 0
    for raw_hash in hashes:
        key = int(raw_hash)
        if key in seen_hashes:
            keep_mask.append(False)
            skipped += 1
        else:
            seen_hashes.add(key)
            keep_mask.append(True)
    if not skipped:
        return df, 0
    return df.loc[keep_mask].copy(), skipped


def _not_blank(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def _coerce_bool_value(value: object, default: bool = False) -> bool:
    if value is None or pd.isna(value):
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n", ""}:
        return False
    return bool(default)


def _path_exists_value(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    text = str(value).strip()
    if not text:
        return False
    try:
        return Path(text).expanduser().exists()
    except Exception:
        return False


def _set_text_default(df: pd.DataFrame, column: str, default: str = "") -> None:
    if column not in df.columns:
        df[column] = default
        return
    df[column] = df[column].where(df[column].notna(), default)


def _sentinel_mask(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    bad_vals = [round(float(x), 6) for x in INVALID_TOTAL_CO2EQ_GT_VALUES]
    return vals.round(6).isin(bad_vals)


def _normalise_panel_results(
    input_df: pd.DataFrame,
    *,
    target_year: int,
    detail_scenario_ids: Sequence[str],
) -> pd.DataFrame:
    if "scenario_id" not in input_df.columns:
        raise RuntimeError("Merged panel results must contain scenario_id.")

    df = input_df.copy()
    df = _normalise_ruminant_cap_columns(df)
    df["scenario_id"] = df["scenario_id"].astype(str).str.strip()
    df = df.loc[df["scenario_id"].ne("")].copy()
    if df.empty:
        raise RuntimeError("Merged panel results contain no non-empty scenario_id rows.")

    parsed = pd.DataFrame(
        [_parse_scenario_id(scenario_id) or {} for scenario_id in df["scenario_id"]],
        index=df.index,
    )
    pct_cols = [
        "forest_area_change_pct",
        RUMINANT_CAP_COL,
        "yield_change_pct",
        "emission_factor_change_pct",
    ]
    for col in pct_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if col in parsed.columns:
            df[col] = df[col].where(df[col].notna(), parsed[col])

    missing_axis = df.loc[df[pct_cols].isna().any(axis=1), "scenario_id"].head(10).tolist()
    if missing_axis:
        raise RuntimeError(
            "Cannot infer panel axis values for scenario_id rows: " + ", ".join(missing_axis)
        )

    multiplier_map = {
        "forest_area_change_pct": "forest_area_multiplier",
        "yield_change_pct": "yield_multiplier",
        "emission_factor_change_pct": "emission_factor_multiplier",
    }
    for pct_col, mult_col in multiplier_map.items():
        if mult_col not in df.columns:
            df[mult_col] = np.nan
        df[mult_col] = pd.to_numeric(df[mult_col], errors="coerce")
        df[mult_col] = df[mult_col].where(
            df[mult_col].notna(),
            df[pct_col].map(_pct_to_multiplier),
        )

    if "target_year" not in df.columns:
        df["target_year"] = int(target_year)
    df["target_year"] = pd.to_numeric(df["target_year"], errors="coerce").fillna(int(target_year)).astype(int)

    if "afolu_emissions_gt_co2eq_yr" not in df.columns:
        df["afolu_emissions_gt_co2eq_yr"] = np.nan
    df["afolu_emissions_gt_co2eq_yr"] = pd.to_numeric(
        df["afolu_emissions_gt_co2eq_yr"],
        errors="coerce",
    )

    _set_text_default(df, "run_status", "")
    df["run_status"] = df["run_status"].astype(str).str.strip()
    has_value = df["afolu_emissions_gt_co2eq_yr"].notna()
    df.loc[df["run_status"].eq("") & has_value, "run_status"] = "ok"
    df.loc[df["run_status"].eq(""), "run_status"] = "missing_result_value"

    for col in [
        "scenario_dir",
        "diagnostic_scenario_dir",
        "model_status_text",
        "iis_summary",
        "error_type",
        "error_message",
    ]:
        _set_text_default(df, col, "")
    if "model_status_code" not in df.columns:
        df["model_status_code"] = np.nan
    df["model_status_code"] = pd.to_numeric(df["model_status_code"], errors="coerce")

    detail_sid_set = set(str(x).strip() for x in detail_scenario_ids if str(x).strip())
    bool_defaults = {
        "has_run_dir": df["scenario_dir"].map(_path_exists_value),
        "has_fast_summary": has_value,
        "has_global_detail": df["scenario_id"].isin(detail_sid_set),
    }
    for col, default_series in bool_defaults.items():
        if col == "has_run_dir":
            df[col] = default_series.astype(bool)
            continue
        if col in df.columns:
            df[col] = [
                _coerce_bool_value(value, bool(default))
                for value, default in zip(df[col].tolist(), default_series.tolist())
            ]
        else:
            df[col] = default_series.astype(bool)

    bad_model = df["model_status_code"].eq(3) | df["model_status_text"].astype(str).str.contains(
        "infeasible",
        case=False,
        na=False,
    )
    if bad_model.any():
        df.loc[bad_model, "run_status"] = "infeasible"
        df.loc[bad_model, "afolu_emissions_gt_co2eq_yr"] = np.nan
        df.loc[bad_model, "has_fast_summary"] = False
        df.loc[bad_model & df["error_type"].astype(str).str.strip().eq(""), "error_type"] = "InfeasibleModel"
        df.loc[bad_model & df["error_message"].astype(str).str.strip().eq(""), "error_message"] = (
            "model_status_code=3; discarded partial fast-emission output"
        )
    bad_sentinel = _sentinel_mask(df["afolu_emissions_gt_co2eq_yr"])
    if bad_sentinel.any():
        df.loc[bad_sentinel, "run_status"] = "invalid_fast_emissions"
        df.loc[bad_sentinel, "afolu_emissions_gt_co2eq_yr"] = np.nan
        df.loc[bad_sentinel, "has_fast_summary"] = False
        df.loc[
            bad_sentinel & df["error_type"].astype(str).str.strip().eq(""),
            "error_type",
        ] = "InvalidFastSummarySentinel"
        df.loc[
            bad_sentinel & df["error_message"].astype(str).str.strip().eq(""),
            "error_message",
        ] = f"discarded invalid emission sentinel {INVALID_TOTAL_CO2EQ_GT_VALUES}"

    df = df.drop_duplicates(subset=["scenario_id"], keep="last").reset_index(drop=True)
    return df


def _empty_detail_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "scenario_id",
            "panel_row",
            "panel_col",
            "forest_area_change_pct",
            RUMINANT_CAP_COL,
            "yield_change_pct",
            "emission_factor_change_pct",
            "target_year",
            "run_status",
            "scenario_dir",
            "year",
            "source_module",
            "Process",
            "Item",
            "GHG",
            "emissions_kt",
            "co2eq_kt",
        ]
    )


def _normalise_detail_results(
    input_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    *,
    target_year: int,
) -> pd.DataFrame:
    if input_df is None or input_df.empty:
        return _empty_detail_frame()
    if "scenario_id" not in input_df.columns:
        print("[S5_3_2][WARN] detail CSV has no scenario_id; writing empty detail output.")
        return _empty_detail_frame()

    df = input_df.copy()
    df = _normalise_ruminant_cap_columns(df)
    df["scenario_id"] = df["scenario_id"].astype(str).str.strip()
    df = df.loc[df["scenario_id"].ne("")].copy()
    if df.empty:
        return _empty_detail_frame()

    if "year" in df.columns:
        year_series = pd.to_numeric(df["year"], errors="coerce")
        df = df.loc[year_series.eq(float(target_year))].copy()
        df["year"] = int(target_year)
    else:
        df["year"] = int(target_year)
    if df.empty:
        return _empty_detail_frame()

    valid_raw = raw_df.loc[
        raw_df["run_status"].isin(["ok", "resumed"])
        & pd.to_numeric(raw_df["afolu_emissions_gt_co2eq_yr"], errors="coerce").notna(),
        "scenario_id",
    ].astype(str).str.strip()
    if not valid_raw.empty:
        df = df.loc[df["scenario_id"].isin(set(valid_raw))].copy()
    else:
        df = df.iloc[0:0].copy()
    if df.empty:
        return _empty_detail_frame()

    meta_cols = [
        "panel_row",
        "panel_col",
        "forest_area_change_pct",
        RUMINANT_CAP_COL,
        "yield_change_pct",
        "emission_factor_change_pct",
        "target_year",
        "run_status",
        "scenario_dir",
    ]
    raw_meta_cols = [c for c in meta_cols if c in raw_df.columns]
    if raw_meta_cols:
        raw_meta = raw_df[["scenario_id"] + raw_meta_cols].drop_duplicates("scenario_id", keep="last")
        df = df.merge(raw_meta, on="scenario_id", how="left", suffixes=("", "_from_results"))
        for col in raw_meta_cols:
            helper_col = f"{col}_from_results"
            if helper_col not in df.columns:
                continue
            if col in df.columns:
                df[col] = df[col].where(_not_blank(df[col]), df[helper_col])
            else:
                df[col] = df[helper_col]
            df = df.drop(columns=[helper_col])

    df["target_year"] = pd.to_numeric(df["target_year"], errors="coerce").fillna(int(target_year)).astype(int)
    for col in ["panel_row", "panel_col"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in [
        "forest_area_change_pct",
        RUMINANT_CAP_COL,
        "yield_change_pct",
        "emission_factor_change_pct",
        "emissions_kt",
        "co2eq_kt",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    dedup_cols = [c for c in ["scenario_id", "year", "source_module", "Process", "Item", "GHG"] if c in df.columns]
    if dedup_cols:
        df = df.drop_duplicates(subset=dedup_cols, keep="last")

    detail_cols = ["year", "source_module", "Process", "Item", "GHG", "emissions_kt", "co2eq_kt"]
    ordered = (
        ["scenario_id"]
        + [c for c in meta_cols if c in df.columns]
        + [c for c in detail_cols if c in df.columns]
        + [c for c in df.columns if c not in {"scenario_id", *meta_cols, *detail_cols}]
    )
    sort_cols = [c for c in ["scenario_id", "year", "source_module", "Process", "Item", "GHG"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return df[ordered]


def _normalise_detail_chunk(
    chunk: pd.DataFrame,
    raw_df: pd.DataFrame,
    raw_meta: pd.DataFrame,
    valid_scenario_ids: Set[str],
    *,
    target_year: int,
) -> pd.DataFrame:
    if chunk is None or chunk.empty:
        return _empty_detail_frame().iloc[0:0].copy()
    if "scenario_id" not in chunk.columns:
        return _empty_detail_frame().iloc[0:0].copy()

    df = chunk.copy()
    df = _normalise_ruminant_cap_columns(df)
    df["scenario_id"] = df["scenario_id"].astype(str).str.strip()
    df = df.loc[df["scenario_id"].ne("")].copy()
    if df.empty:
        return _empty_detail_frame().iloc[0:0].copy()

    if "year" in df.columns:
        year_series = pd.to_numeric(df["year"], errors="coerce")
        df = df.loc[year_series.eq(float(target_year))].copy()
        df["year"] = int(target_year)
    else:
        df["year"] = int(target_year)
    if df.empty:
        return _empty_detail_frame().iloc[0:0].copy()

    if valid_scenario_ids:
        df = df.loc[df["scenario_id"].isin(valid_scenario_ids)].copy()
    else:
        df = df.iloc[0:0].copy()
    if df.empty:
        return _empty_detail_frame().iloc[0:0].copy()

    meta_cols = [
        "panel_row",
        "panel_col",
        "forest_area_change_pct",
        RUMINANT_CAP_COL,
        "yield_change_pct",
        "emission_factor_change_pct",
        "target_year",
        "run_status",
        "scenario_dir",
    ]
    raw_meta_cols = [c for c in meta_cols if c in raw_df.columns]
    if raw_meta_cols:
        df = df.merge(raw_meta, on="scenario_id", how="left", suffixes=("", "_from_results"))
        for col in raw_meta_cols:
            helper_col = f"{col}_from_results"
            if helper_col not in df.columns:
                continue
            if col in df.columns:
                df[col] = df[col].where(_not_blank(df[col]), df[helper_col])
            else:
                df[col] = df[helper_col]
            df = df.drop(columns=[helper_col])

    if "target_year" not in df.columns:
        df["target_year"] = int(target_year)
    df["target_year"] = pd.to_numeric(df["target_year"], errors="coerce").fillna(int(target_year)).astype(int)
    for col in ["panel_row", "panel_col"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in [
        "forest_area_change_pct",
        RUMINANT_CAP_COL,
        "yield_change_pct",
        "emission_factor_change_pct",
        "emissions_kt",
        "co2eq_kt",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    detail_cols = ["year", "source_module", "Process", "Item", "GHG", "emissions_kt", "co2eq_kt"]
    ordered = (
        ["scenario_id"]
        + [c for c in meta_cols if c in df.columns]
        + [c for c in detail_cols if c in df.columns]
        + [c for c in df.columns if c not in {"scenario_id", *meta_cols, *detail_cols}]
    )
    return df[ordered]


def _stream_normalise_detail_results(
    input_path: Path,
    output_path: Path,
    raw_df: pd.DataFrame,
    *,
    target_year: int,
    chunksize: int,
    dedup_detail: bool,
) -> Dict[str, object]:
    empty_template = _empty_detail_frame()
    if not input_path.exists():
        print(f"[S5_3_2][WARN] optional merged global emissions detail CSV not found: {input_path}")
        empty_template.to_csv(output_path, index=False, encoding="utf-8-sig")
        return {
            "input_rows": 0,
            "output_rows": 0,
            "scenario_ids": set(),
            "duplicate_keys_skipped": 0,
            "chunks": 0,
        }

    valid_scenario_ids = set(
        raw_df.loc[
            raw_df["run_status"].isin(["ok", "resumed"])
            & pd.to_numeric(raw_df["afolu_emissions_gt_co2eq_yr"], errors="coerce").notna(),
            "scenario_id",
        ]
        .astype(str)
        .str.strip()
        .tolist()
    )
    meta_cols = [
        "panel_row",
        "panel_col",
        "forest_area_change_pct",
        RUMINANT_CAP_COL,
        "yield_change_pct",
        "emission_factor_change_pct",
        "target_year",
        "run_status",
        "scenario_dir",
    ]
    raw_meta_cols = [c for c in meta_cols if c in raw_df.columns]
    raw_meta = raw_df[["scenario_id"] + raw_meta_cols].drop_duplicates("scenario_id", keep="last")

    tmp_path = output_path.with_name(f".{output_path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    scenario_ids: Set[str] = set()
    seen_hashes: Set[int] = set()
    input_rows = 0
    output_rows = 0
    chunks_seen = 0
    duplicate_skipped = 0
    header_written = False
    first_columns: Optional[List[str]] = None

    print(
        f"[S5_3_2] stream detail rebuild chunksize={chunksize or 'whole-file'} "
        f"dedup={'enabled' if dedup_detail else 'disabled'}"
    )

    try:
        for chunk in _iter_csv_chunks_safe(input_path, chunksize=chunksize, label="merged global emissions detail CSV"):
            chunks_seen += 1
            input_rows += int(len(chunk))
            detail_chunk = _normalise_detail_chunk(
                chunk,
                raw_df,
                raw_meta,
                valid_scenario_ids,
                target_year=target_year,
            )
            if first_columns is None:
                first_columns = list(detail_chunk.columns) if not detail_chunk.empty else list(empty_template.columns)
            if detail_chunk.empty:
                del chunk, detail_chunk
                gc.collect()
                continue

            if dedup_detail:
                dedup_cols = [
                    c
                    for c in ["scenario_id", "year", "source_module", "Process", "Item", "GHG"]
                    if c in detail_chunk.columns
                ]
                detail_chunk, skipped = _deduplicate_chunk_by_hash(
                    detail_chunk,
                    dedup_cols=dedup_cols,
                    seen_hashes=seen_hashes,
                )
                duplicate_skipped += skipped
                if detail_chunk.empty:
                    del chunk, detail_chunk
                    gc.collect()
                    continue

            scenario_ids.update(detail_chunk["scenario_id"].dropna().astype(str).str.strip().tolist())
            encoding = "utf-8-sig" if not header_written else "utf-8"
            detail_chunk.to_csv(
                tmp_path,
                mode="a",
                header=not header_written,
                index=False,
                encoding=encoding,
            )
            header_written = True
            output_rows += int(len(detail_chunk))
            del chunk, detail_chunk
            gc.collect()

        if not header_written:
            pd.DataFrame(columns=first_columns or list(empty_template.columns)).to_csv(
                tmp_path,
                index=False,
                encoding="utf-8-sig",
            )
        os.replace(tmp_path, output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return {
        "input_rows": int(input_rows),
        "output_rows": int(output_rows),
        "scenario_ids": scenario_ids,
        "duplicate_keys_skipped": int(duplicate_skipped),
        "chunks": int(chunks_seen),
    }


def _merge_level_values(
    explicit_values: Optional[Sequence[float]],
    observed_values: Sequence[float],
    *,
    existing_df: Optional[pd.DataFrame] = None,
    existing_col: Optional[str] = None,
) -> List[float]:
    merged: List[float] = []
    if explicit_values:
        merged.extend(float(v) for v in explicit_values)
    merged.extend(float(v) for v in observed_values)
    if existing_df is not None and existing_col and existing_col in existing_df.columns:
        existing_vals = pd.to_numeric(existing_df[existing_col], errors="coerce").dropna().tolist()
        merged.extend(float(v) for v in existing_vals)
    merged = sorted({round(float(v), 10) for v in merged})
    return [float(v) for v in merged]


def _median_step(values: Sequence[float]) -> float:
    vals = sorted(float(v) for v in values)
    if len(vals) <= 1:
        return 1.0
    diffs = [b - a for a, b in zip(vals[:-1], vals[1:]) if (b - a) > 0]
    if not diffs:
        return 1.0
    return float(np.median(diffs))


def _linear_interp_1d(target: float, left: Tuple[float, float], right: Tuple[float, float]) -> float:
    x0, y0 = left
    x1, y1 = right
    if x1 == x0:
        return float((y0 + y1) * 0.5)
    frac = (target - x0) / (x1 - x0)
    return float(y0 + frac * (y1 - y0))


def _idw_estimate(
    target: Sequence[float],
    points: Sequence[Tuple[Sequence[float], float]],
    *,
    k: int,
    power: float,
) -> Optional[Tuple[float, int, float]]:
    if not points:
        return None
    weighted: List[Tuple[float, float]] = []
    for coords, value in points:
        dist_sq = 0.0
        for a, b in zip(target, coords):
            diff = float(a) - float(b)
            dist_sq += diff * diff
        dist = math.sqrt(dist_sq)
        if dist <= 1e-12:
            return float(value), 1, 0.0
        weighted.append((dist, float(value)))
    weighted.sort(key=lambda x: x[0])
    use = weighted[: max(1, int(k))]
    weights = np.array([1.0 / max(d, 1e-12) ** power for d, _ in use], dtype=float)
    values = np.array([v for _, v in use], dtype=float)
    estimate = float(np.average(values, weights=weights))
    mean_dist = float(np.mean([d for d, _ in use]))
    return estimate, len(use), mean_dist


def _interpolate_same_panel(
    panel_df: pd.DataFrame,
    *,
    yield_step: float,
    ef_step: float,
    neighbors_k: int,
    power: float,
) -> pd.DataFrame:
    panel_df = panel_df.copy()
    observed = panel_df[panel_df["afolu_emissions_gt_co2eq_yr_raw"].notna()].copy()
    if observed.empty:
        return panel_df

    obs_map = {
        (float(r["yield_change_pct"]), float(r["emission_factor_change_pct"])): float(r["afolu_emissions_gt_co2eq_yr_raw"])
        for _, r in observed.iterrows()
    }
    obs_yield = sorted({float(v) for v in observed["yield_change_pct"].dropna().tolist()})
    obs_ef = sorted({float(v) for v in observed["emission_factor_change_pct"].dropna().tolist()})

    for idx, row in panel_df.loc[panel_df["afolu_emissions_gt_co2eq_yr"].isna()].iterrows():
        y = float(row["yield_change_pct"])
        e = float(row["emission_factor_change_pct"])

        y_lower = [v for v in obs_yield if v < y]
        y_upper = [v for v in obs_yield if v > y]
        e_lower = [v for v in obs_ef if v < e]
        e_upper = [v for v in obs_ef if v > e]

        bilinear_done = False
        if y_lower and y_upper and e_lower and e_upper:
            y0 = max(y_lower)
            y1 = min(y_upper)
            e0 = max(e_lower)
            e1 = min(e_upper)
            corners = [(y0, e0), (y0, e1), (y1, e0), (y1, e1)]
            if all(c in obs_map for c in corners):
                q11 = obs_map[(y0, e0)]
                q12 = obs_map[(y0, e1)]
                q21 = obs_map[(y1, e0)]
                q22 = obs_map[(y1, e1)]
                top = _linear_interp_1d(e, (e0, q11), (e1, q12))
                bot = _linear_interp_1d(e, (e0, q21), (e1, q22))
                est = _linear_interp_1d(y, (y0, top), (y1, bot))
                panel_df.at[idx, "afolu_emissions_gt_co2eq_yr"] = est
                panel_df.at[idx, "plot_data_status"] = "interpolated"
                panel_df.at[idx, "is_interpolated"] = True
                panel_df.at[idx, "interpolation_method"] = "bilinear_4pt"
                panel_df.at[idx, "interpolation_neighbor_count"] = 4
                panel_df.at[idx, "interpolation_distance_mean"] = float(
                    np.mean(
                        [
                            math.sqrt(((y - yy) / max(yield_step, 1e-12)) ** 2 + ((e - ee) / max(ef_step, 1e-12)) ** 2)
                            for yy, ee in corners
                        ]
                    )
                )
                bilinear_done = True
        if bilinear_done:
            continue

        row_points = [(ee, obs_map[(y, ee)]) for ee in obs_ef if (y, ee) in obs_map]
        left = [(ee, vv) for ee, vv in row_points if ee < e]
        right = [(ee, vv) for ee, vv in row_points if ee > e]
        if left and right:
            left_pt = max(left, key=lambda x: x[0])
            right_pt = min(right, key=lambda x: x[0])
            est = _linear_interp_1d(e, left_pt, right_pt)
            panel_df.at[idx, "afolu_emissions_gt_co2eq_yr"] = est
            panel_df.at[idx, "plot_data_status"] = "interpolated"
            panel_df.at[idx, "is_interpolated"] = True
            panel_df.at[idx, "interpolation_method"] = "linear_ef"
            panel_df.at[idx, "interpolation_neighbor_count"] = 2
            panel_df.at[idx, "interpolation_distance_mean"] = float(
                np.mean([abs(e - left_pt[0]), abs(right_pt[0] - e)]) / max(ef_step, 1e-12)
            )
            continue

        col_points = [(yy, obs_map[(yy, e)]) for yy in obs_yield if (yy, e) in obs_map]
        low = [(yy, vv) for yy, vv in col_points if yy < y]
        high = [(yy, vv) for yy, vv in col_points if yy > y]
        if low and high:
            low_pt = max(low, key=lambda x: x[0])
            high_pt = min(high, key=lambda x: x[0])
            est = _linear_interp_1d(y, low_pt, high_pt)
            panel_df.at[idx, "afolu_emissions_gt_co2eq_yr"] = est
            panel_df.at[idx, "plot_data_status"] = "interpolated"
            panel_df.at[idx, "is_interpolated"] = True
            panel_df.at[idx, "interpolation_method"] = "linear_yield"
            panel_df.at[idx, "interpolation_neighbor_count"] = 2
            panel_df.at[idx, "interpolation_distance_mean"] = float(
                np.mean([abs(y - low_pt[0]), abs(high_pt[0] - y)]) / max(yield_step, 1e-12)
            )
            continue

        idw_points = [
            ((yy / max(yield_step, 1e-12), ee / max(ef_step, 1e-12)), vv)
            for (yy, ee), vv in obs_map.items()
        ]
        idw = _idw_estimate(
            (y / max(yield_step, 1e-12), e / max(ef_step, 1e-12)),
            idw_points,
            k=neighbors_k,
            power=power,
        )
        if idw is not None:
            est, count, mean_dist = idw
            panel_df.at[idx, "afolu_emissions_gt_co2eq_yr"] = est
            panel_df.at[idx, "plot_data_status"] = "interpolated"
            panel_df.at[idx, "is_interpolated"] = True
            panel_df.at[idx, "interpolation_method"] = "idw_same_panel"
            panel_df.at[idx, "interpolation_neighbor_count"] = count
            panel_df.at[idx, "interpolation_distance_mean"] = mean_dist

    return panel_df


def _interpolate_cross_panel(
    full_df: pd.DataFrame,
    *,
    forest_step: float,
    rumi_step: float,
    yield_step: float,
    ef_step: float,
    neighbors_k: int,
    power: float,
) -> pd.DataFrame:
    full_df = full_df.copy()
    known = full_df[full_df["afolu_emissions_gt_co2eq_yr"].notna()].copy()
    if known.empty:
        return full_df

    points = [
        (
            (
                float(r["forest_area_change_pct"]) / max(forest_step, 1e-12),
                float(r[RUMINANT_CAP_COL]) / max(rumi_step, 1e-12),
                float(r["yield_change_pct"]) / max(yield_step, 1e-12),
                float(r["emission_factor_change_pct"]) / max(ef_step, 1e-12),
            ),
            float(r["afolu_emissions_gt_co2eq_yr"]),
        )
        for _, r in known.iterrows()
    ]

    for idx, row in full_df.loc[full_df["afolu_emissions_gt_co2eq_yr"].isna()].iterrows():
        idw = _idw_estimate(
            (
                float(row["forest_area_change_pct"]) / max(forest_step, 1e-12),
                float(row[RUMINANT_CAP_COL]) / max(rumi_step, 1e-12),
                float(row["yield_change_pct"]) / max(yield_step, 1e-12),
                float(row["emission_factor_change_pct"]) / max(ef_step, 1e-12),
            ),
            points,
            k=neighbors_k,
            power=power,
        )
        if idw is None:
            continue
        est, count, mean_dist = idw
        full_df.at[idx, "afolu_emissions_gt_co2eq_yr"] = est
        full_df.at[idx, "plot_data_status"] = "interpolated"
        full_df.at[idx, "is_interpolated"] = True
        full_df.at[idx, "interpolation_method"] = "idw_cross_panel"
        full_df.at[idx, "interpolation_neighbor_count"] = count
        full_df.at[idx, "interpolation_distance_mean"] = mean_dist
    return full_df


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "t", "yes", "y"})


def _validate_plot_ready_quality(plot_df: pd.DataFrame, *, cfg: Dict[str, object]) -> None:
    if not bool(cfg.get("quality_gate_enabled", True)):
        return
    total = int(len(plot_df))
    if total <= 0:
        raise RuntimeError("[S5_3_2] plot-ready quality gate failed: no grid rows were built.")

    status = plot_df.get("plot_data_status", pd.Series("", index=plot_df.index)).astype(str).str.strip().str.lower()
    z_raw = pd.to_numeric(plot_df.get("afolu_emissions_gt_co2eq_yr_raw", pd.Series(np.nan, index=plot_df.index)), errors="coerce")
    z_plot = pd.to_numeric(plot_df.get("afolu_emissions_gt_co2eq_yr", pd.Series(np.nan, index=plot_df.index)), errors="coerce")
    interp_flag = _bool_series(plot_df.get("is_interpolated", pd.Series(False, index=plot_df.index)))

    observed = status.eq("observed") & z_raw.notna()
    interpolated = (status.eq("interpolated") | interp_flag) & z_plot.notna()
    observed_fraction = float(observed.mean())
    interpolated_fraction = float(interpolated.mean())

    min_observed_fraction = float(cfg.get("min_observed_fraction", 0.50) or 0.0)
    max_interpolated_fraction = float(cfg.get("max_interpolated_fraction", 0.50) or 1.0)
    min_panel_points = int(cfg.get("min_observed_points_per_panel", 25) or 0)
    min_panel_fraction = float(cfg.get("min_observed_fraction_per_panel", 0.25) or 0.0)
    max_panel_interp = float(cfg.get("max_interpolated_fraction_per_panel", 0.75) or 1.0)

    issues: List[str] = []
    if observed_fraction < min_observed_fraction:
        issues.append(
            f"observed_fraction={observed_fraction:.3f} < {min_observed_fraction:.3f} "
            f"({int(observed.sum())}/{total})"
        )
    if interpolated_fraction > max_interpolated_fraction:
        issues.append(
            f"interpolated_fraction={interpolated_fraction:.3f} > {max_interpolated_fraction:.3f} "
            f"({int(interpolated.sum())}/{total})"
        )

    weak_panels: List[str] = []
    interp_panels: List[str] = []
    for (forest_pct, rumi_pct), idxs in plot_df.groupby(
        ["forest_area_change_pct", RUMINANT_CAP_COL], dropna=False
    ).groups.items():
        panel_total = int(len(idxs))
        if panel_total <= 0:
            continue
        panel_observed = int(observed.loc[idxs].sum())
        panel_observed_fraction = float(panel_observed / panel_total)
        panel_interpolated_fraction = float(interpolated.loc[idxs].sum() / panel_total)
        panel_label = f"F={forest_pct}, R={rumi_pct}"
        if panel_observed < min_panel_points or panel_observed_fraction < min_panel_fraction:
            weak_panels.append(f"{panel_label}: observed={panel_observed}/{panel_total}")
        if panel_interpolated_fraction > max_panel_interp:
            interp_panels.append(f"{panel_label}: interpolated={panel_interpolated_fraction:.3f}")

    if weak_panels:
        issues.append(
            f"{len(weak_panels)} panels below observed support gate; examples: "
            + "; ".join(weak_panels[:6])
        )
    if interp_panels:
        issues.append(
            f"{len(interp_panels)} panels above interpolation gate; examples: "
            + "; ".join(interp_panels[:6])
        )

    if issues:
        raise RuntimeError(
            "[S5_3_2] plot-ready quality gate failed: "
            + " | ".join(issues)
            + ". Rerun S5_3_1/S5_3_3 after fixing model feasibility; disable "
            "CONFIG['quality_gate_enabled'] only for diagnostics."
        )


def main() -> None:
    args = _build_arg_parser().parse_args()
    cfg = dict(CONFIG)
    panel_root_raw = (
        str(args.panel_root_dir or "").strip()
        or str(os.environ.get("PANEL_OUTPUT_DIR", "") or "").strip()
        or str(cfg.get("panel_root_dir", "") or "").strip()
    )
    panel_root = _resolve_panel_root_dir(panel_root_raw)
    _sync_panel_root_environment(panel_root)
    input_results_path = panel_root / str(cfg.get("input_results_csv") or "figure_panel_dataset_long.csv")
    input_detail_path = panel_root / str(cfg.get("input_detail_csv") or "figure_panel_global_emissions_detail_long.csv")
    raw_out = panel_root / str(cfg.get("raw_results_csv") or "figure_panel_dataset_long_rebuilt.csv")
    plot_out = panel_root / str(cfg.get("plot_ready_csv") or "figure_panel_dataset_long_plot_ready.csv")
    detail_out = panel_root / str(cfg.get("detail_results_csv") or "figure_panel_global_emissions_detail_long_rebuilt.csv")
    missing_out = panel_root / str(cfg.get("missing_audit_csv") or "figure_panel_missing_points_audit.csv")
    summary_out = panel_root / str(cfg.get("summary_csv") or "figure_panel_data_summary.csv")
    _ensure_dir(panel_root)

    target_year = int(cfg.get("target_year", 2080) or 2080)
    detail_chunksize = max(0, int(args.detail_chunksize or 0))
    input_results_df = _read_required_csv(input_results_path, "merged panel results CSV")
    raw_df = _normalise_panel_results(
        input_results_df,
        target_year=target_year,
        detail_scenario_ids=[],
    )

    print(f"[S5_3_2] panel_root={panel_root}")
    print(f"[S5_3_2] NZF_OUTPUT_DIR={os.environ.get('NZF_OUTPUT_DIR', '')}")
    print(f"[S5_3_2] PANEL_OUTPUT_DIR={os.environ.get('PANEL_OUTPUT_DIR', '')}")
    print(f"[S5_3_2] input_results={input_results_path} rows={len(input_results_df)}")
    print(f"[S5_3_2] input_detail={input_detail_path} streaming")

    axis_cols = [
        "forest_area_change_pct",
        RUMINANT_CAP_COL,
        "yield_change_pct",
        "emission_factor_change_pct",
    ]
    for col in axis_cols:
        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce").round(10)
    forest_levels = _merge_level_values(
        cfg.get("forest_area_change_pct_values"),
        raw_df["forest_area_change_pct"].dropna().tolist(),
    )
    rumi_levels = _merge_level_values(
        _configured_ruminant_cap_values(cfg),
        raw_df[RUMINANT_CAP_COL].dropna().tolist(),
    )
    yield_levels = _merge_level_values(
        cfg.get("yield_change_pct_values"),
        raw_df["yield_change_pct"].dropna().tolist(),
    )
    ef_levels = _merge_level_values(
        cfg.get("emission_factor_change_pct_values"),
        raw_df["emission_factor_change_pct"].dropna().tolist(),
    )

    forest_index = {v: i + 1 for i, v in enumerate(forest_levels)}
    rumi_index = {v: i + 1 for i, v in enumerate(rumi_levels)}
    raw_df["panel_row"] = raw_df["forest_area_change_pct"].map(forest_index)
    raw_df["panel_col"] = raw_df[RUMINANT_CAP_COL].map(rumi_index)
    if raw_df[["panel_row", "panel_col"]].isna().any().any():
        bad_ids = raw_df.loc[raw_df[["panel_row", "panel_col"]].isna().any(axis=1), "scenario_id"].head(10).tolist()
        raise RuntimeError("Failed to map panel row/col for scenario_id rows: " + ", ".join(bad_ids))
    raw_df["panel_row"] = raw_df["panel_row"].astype(int)
    raw_df["panel_col"] = raw_df["panel_col"].astype(int)

    detail_stats = _stream_normalise_detail_results(
        input_detail_path,
        detail_out,
        raw_df,
        target_year=target_year,
        chunksize=detail_chunksize,
        dedup_detail=bool(args.dedup_detail),
    )
    detail_scenario_ids = detail_stats.get("scenario_ids", set())
    if detail_scenario_ids:
        raw_df["has_global_detail"] = raw_df["has_global_detail"] | raw_df["scenario_id"].isin(detail_scenario_ids)

    for col in RAW_KEEP_COLS:
        if col not in raw_df.columns:
            raw_df[col] = np.nan

    raw_df = raw_df[RAW_KEEP_COLS].sort_values(
        ["panel_row", "panel_col", "yield_change_pct", "emission_factor_change_pct", "scenario_id"],
        ignore_index=True,
    )
    raw_df.to_csv(raw_out, index=False, encoding="utf-8-sig")
    print(f"[S5_3_2] raw_results={raw_out} rows={len(raw_df)}")

    print(
        f"[S5_3_2] detail_results={detail_out} rows={detail_stats['output_rows']} "
        f"input_rows={detail_stats['input_rows']} chunks={detail_stats['chunks']} "
        f"duplicate_keys_skipped={detail_stats['duplicate_keys_skipped']}"
    )

    full_grid_rows: List[Dict[str, object]] = []
    for forest_pct, rumi_pct, yield_pct, ef_pct in itertools.product(
        forest_levels,
        rumi_levels,
        yield_levels,
        ef_levels,
    ):
        full_grid_rows.append(
            {
                "scenario_id": _scenario_id(
                    forest_pct=forest_pct,
                    ruminant_pct=rumi_pct,
                    yield_pct=yield_pct,
                    ef_pct=ef_pct,
                ),
                "panel_row": forest_index[forest_pct],
                "panel_col": rumi_index[rumi_pct],
                "forest_area_change_pct": forest_pct,
                "forest_area_multiplier": _pct_to_multiplier(forest_pct),
                RUMINANT_CAP_COL: rumi_pct,
                "yield_change_pct": yield_pct,
                "yield_multiplier": _pct_to_multiplier(yield_pct),
                "emission_factor_change_pct": ef_pct,
                "emission_factor_multiplier": _pct_to_multiplier(ef_pct),
                "target_year": target_year,
            }
        )

    full_df = pd.DataFrame(full_grid_rows).merge(
        raw_df,
        on=[
            "scenario_id",
            "panel_row",
            "panel_col",
            "forest_area_change_pct",
            "forest_area_multiplier",
            RUMINANT_CAP_COL,
            "yield_change_pct",
            "yield_multiplier",
            "emission_factor_change_pct",
            "emission_factor_multiplier",
            "target_year",
        ],
        how="left",
    )
    full_df["has_run_dir"] = full_df["has_run_dir"].fillna(False).astype(bool)
    full_df["has_fast_summary"] = full_df["has_fast_summary"].fillna(False).astype(bool)
    full_df["has_global_detail"] = full_df["has_global_detail"].fillna(False).astype(bool)
    full_df["raw_run_status"] = full_df["run_status"].fillna("missing_panel_csv_row")
    full_df["afolu_emissions_gt_co2eq_yr_raw"] = pd.to_numeric(full_df["afolu_emissions_gt_co2eq_yr"], errors="coerce")
    full_df["plot_data_status"] = np.where(
        full_df["afolu_emissions_gt_co2eq_yr_raw"].notna(),
        "observed",
        "missing",
    )
    full_df["is_interpolated"] = False
    full_df["interpolation_method"] = ""
    full_df["interpolation_neighbor_count"] = 0
    full_df["interpolation_distance_mean"] = np.nan
    full_df["afolu_emissions_gt_co2eq_yr"] = full_df["afolu_emissions_gt_co2eq_yr_raw"]

    yield_step = _median_step(yield_levels)
    ef_step = _median_step(ef_levels)
    forest_step = _median_step(forest_levels)
    rumi_step = _median_step(rumi_levels)

    panel_frames: List[pd.DataFrame] = []
    for _, panel in full_df.groupby(["forest_area_change_pct", RUMINANT_CAP_COL], sort=False):
        panel_frames.append(
            _interpolate_same_panel(
                panel,
                yield_step=yield_step,
                ef_step=ef_step,
                neighbors_k=int(cfg.get("interpolation_neighbors_k", 4) or 4),
                power=float(cfg.get("interpolation_power", 2.0) or 2.0),
            )
        )
    full_df = pd.concat(panel_frames, ignore_index=True)

    if bool(cfg.get("allow_cross_panel_fallback", True)) and full_df["afolu_emissions_gt_co2eq_yr"].isna().any():
        full_df = _interpolate_cross_panel(
            full_df,
            forest_step=forest_step,
            rumi_step=rumi_step,
            yield_step=yield_step,
            ef_step=ef_step,
            neighbors_k=int(cfg.get("interpolation_neighbors_k", 4) or 4),
            power=float(cfg.get("interpolation_power", 2.0) or 2.0),
        )

    full_df["plot_data_status"] = np.where(
        full_df["afolu_emissions_gt_co2eq_yr_raw"].notna(),
        "observed",
        full_df["plot_data_status"],
    )
    full_df["plot_data_status"] = np.where(
        full_df["afolu_emissions_gt_co2eq_yr"].isna(),
        "missing_unfilled",
        full_df["plot_data_status"],
    )

    plot_df = full_df.sort_values(
        ["panel_row", "panel_col", "yield_change_pct", "emission_factor_change_pct", "scenario_id"],
        ignore_index=True,
    )
    plot_df.to_csv(plot_out, index=False, encoding="utf-8-sig")
    print(f"[S5_3_2] plot_ready={plot_out} rows={len(plot_df)}")

    missing_audit = plot_df.loc[
        plot_df["plot_data_status"].ne("observed"),
        [
            "scenario_id",
            "panel_row",
            "panel_col",
            "forest_area_change_pct",
            RUMINANT_CAP_COL,
            "yield_change_pct",
            "emission_factor_change_pct",
            "raw_run_status",
            "plot_data_status",
            "interpolation_method",
            "interpolation_neighbor_count",
            "interpolation_distance_mean",
            "has_run_dir",
            "has_fast_summary",
            "has_global_detail",
            "afolu_emissions_gt_co2eq_yr_raw",
            "afolu_emissions_gt_co2eq_yr",
            "error_type",
            "error_message",
        ],
    ].copy()
    missing_audit.to_csv(missing_out, index=False, encoding="utf-8-sig")
    print(f"[S5_3_2] missing_audit={missing_out} rows={len(missing_audit)}")

    summary_rows = [
        {"metric": "input_source", "value": "merged_csv"},
        {"metric": "input_results_csv", "value": str(input_results_path)},
        {"metric": "input_detail_csv", "value": str(input_detail_path)},
        {"metric": "input_results_rows_loaded", "value": int(len(input_results_df))},
        {"metric": "input_detail_rows_loaded", "value": int(detail_stats["input_rows"])},
        {"metric": "input_detail_rows_streamed", "value": int(detail_stats["input_rows"])},
        {"metric": "raw_rows_written", "value": int(len(raw_df))},
        {"metric": "detail_rows_written", "value": int(detail_stats["output_rows"])},
        {"metric": "detail_chunks_streamed", "value": int(detail_stats["chunks"])},
        {"metric": "detail_duplicate_keys_skipped", "value": int(detail_stats["duplicate_keys_skipped"])},
        {"metric": "grid_forest_levels", "value": int(len(forest_levels))},
        {"metric": "grid_ruminant_levels", "value": int(len(rumi_levels))},
        {"metric": "grid_yield_levels", "value": int(len(yield_levels))},
        {"metric": "grid_ef_levels", "value": int(len(ef_levels))},
        {"metric": "grid_total_points", "value": int(len(plot_df))},
        {"metric": "observed_points", "value": int((plot_df["plot_data_status"] == "observed").sum())},
        {"metric": "interpolated_points", "value": int((plot_df["plot_data_status"] == "interpolated").sum())},
        {"metric": "missing_unfilled_points", "value": int((plot_df["plot_data_status"] == "missing_unfilled").sum())},
    ]
    run_status_counts = raw_df["run_status"].value_counts(dropna=False)
    for status, count in run_status_counts.items():
        summary_rows.append({"metric": f"raw_run_status::{status}", "value": int(count)})
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_out, index=False, encoding="utf-8-sig")
    print(f"[S5_3_2] summary={summary_out} rows={len(summary_df)}")

    _validate_plot_ready_quality(plot_df, cfg=cfg)

    print(
        "[S5_3_2] done | "
        f"observed={(plot_df['plot_data_status'] == 'observed').sum()} "
        f"interpolated={(plot_df['plot_data_status'] == 'interpolated').sum()} "
        f"missing_unfilled={(plot_df['plot_data_status'] == 'missing_unfilled').sum()}"
    )


if __name__ == "__main__":
    main()
