"""Solver-faithful market-balance diagnostics shared by S3, S4, and S5.

The linear solver uses two linked identities for each future commodity-year:

1. Regional balance:
   ``Qs + net_import = Qd + bioenergy``
2. Global clearing:
   ``net_import + excess - shortage = 0``

Combining them gives the physical global balance:
``Qs + shortage = Qd + bioenergy + excess``.

This module keeps those quantities explicit.  In particular, the raw
``Qd - Qs`` difference is never interpreted as solver shortage because it also
contains the effects of explicit bioenergy demand and trade.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


MARKET_BALANCE_SCHEMA_VERSION = 1
MARKET_BALANCE_DIAGNOSTIC_SOURCE = "linear_solver_postsolve"
MARKET_BALANCE_DIAGNOSTIC_FILENAME = "commodity_balance_by_commodity.csv"

MARKET_BALANCE_REQUIRED_COLUMNS = frozenset(
    {
        "schema_version",
        "diagnostic_source",
        "year",
        "commodity",
        "Qs",
        "Qd",
        "bioenergy_demand_t",
        "market_total_use_t",
        "net_import_t",
        "shortage_t",
        "excess_t",
        "regional_balance_residual_t",
        "regional_balance_max_abs_residual_t",
        "regional_balance_sum_abs_residual_t",
        "global_clearing_residual_t",
        "physical_balance_residual_t",
        "shortage_rate",
    }
)


def _coerce_finite_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _aggregate_regional_mapping(
    mapping: Optional[Mapping[Any, Any]],
) -> Dict[Tuple[str, int], float]:
    """Aggregate ``(region, commodity, year)`` values to commodity-year."""
    out: Dict[Tuple[str, int], float] = {}
    invalid: set[Tuple[str, int]] = set()
    for key, value in (mapping or {}).items():
        if not isinstance(key, tuple) or len(key) < 3:
            continue
        try:
            commodity = str(key[-2])
            year = int(key[-1])
        except (TypeError, ValueError):
            continue
        agg_key = (commodity, year)
        value_f = _coerce_finite_float(value)
        if not np.isfinite(value_f):
            invalid.add(agg_key)
            continue
        out[agg_key] = out.get(agg_key, 0.0) + value_f
    for agg_key in invalid:
        out[agg_key] = float("nan")
    return out


def _normalize_regional_mapping(
    mapping: Optional[Mapping[Any, Any]],
) -> Dict[Tuple[str, str, int], float]:
    """Normalize ``(region, commodity, year)`` mappings without aggregating."""
    out: Dict[Tuple[str, str, int], float] = {}
    for key, value in (mapping or {}).items():
        if not isinstance(key, tuple) or len(key) < 3:
            continue
        try:
            norm_key = (str(key[-3]), str(key[-2]), int(key[-1]))
        except (TypeError, ValueError):
            continue
        value_f = _coerce_finite_float(value)
        if not np.isfinite(value_f):
            out[norm_key] = float("nan")
        elif np.isfinite(out.get(norm_key, 0.0)):
            out[norm_key] = out.get(norm_key, 0.0) + value_f
    return out


def _aggregate_global_mapping(
    mapping: Optional[Mapping[Any, Any]],
) -> Dict[Tuple[str, int], float]:
    """Normalize ``(commodity, year)`` values."""
    out: Dict[Tuple[str, int], float] = {}
    invalid: set[Tuple[str, int]] = set()
    for key, value in (mapping or {}).items():
        if not isinstance(key, tuple) or len(key) < 2:
            continue
        try:
            commodity = str(key[-2])
            year = int(key[-1])
        except (TypeError, ValueError):
            continue
        agg_key = (commodity, year)
        value_f = _coerce_finite_float(value)
        if not np.isfinite(value_f):
            invalid.add(agg_key)
            continue
        out[agg_key] = out.get(agg_key, 0.0) + value_f
    for agg_key in invalid:
        out[agg_key] = float("nan")
    return out


def build_market_balance_diagnostics(
    *,
    qs: Optional[Mapping[Any, Any]],
    qd: Optional[Mapping[Any, Any]],
    net_import: Optional[Mapping[Any, Any]],
    bioenergy: Optional[Mapping[Any, Any]],
    shortage: Optional[Mapping[Any, Any]],
    excess: Optional[Mapping[Any, Any]],
    hist_end_year: int = 2020,
) -> List[Dict[str, Any]]:
    """Build one solver-faithful diagnostic row per future commodity-year."""
    qs_by_key = _aggregate_regional_mapping(qs)
    qd_by_key = _aggregate_regional_mapping(qd)
    net_import_by_key = _aggregate_regional_mapping(net_import)
    bioenergy_by_key = _aggregate_regional_mapping(bioenergy)
    shortage_by_key = _aggregate_global_mapping(shortage)
    excess_by_key = _aggregate_global_mapping(excess)
    qs_by_region = _normalize_regional_mapping(qs)
    qd_by_region = _normalize_regional_mapping(qd)
    net_import_by_region = _normalize_regional_mapping(net_import)
    bioenergy_by_region = _normalize_regional_mapping(bioenergy)

    keys = (
        set(qs_by_key)
        | set(qd_by_key)
        | set(net_import_by_key)
        | set(bioenergy_by_key)
        | set(shortage_by_key)
        | set(excess_by_key)
    )
    rows: List[Dict[str, Any]] = []
    for commodity, year in sorted(keys, key=lambda item: (item[1], item[0])):
        if int(year) <= int(hist_end_year):
            continue
        supply_t = float(qs_by_key.get((commodity, year), 0.0))
        demand_t = float(qd_by_key.get((commodity, year), 0.0))
        net_import_t = float(net_import_by_key.get((commodity, year), 0.0))
        bioenergy_t = float(bioenergy_by_key.get((commodity, year), 0.0))
        shortage_t = float(shortage_by_key.get((commodity, year), 0.0))
        excess_t = float(excess_by_key.get((commodity, year), 0.0))

        market_total_use_t = demand_t + bioenergy_t
        regional_residual_t = supply_t + net_import_t - demand_t - bioenergy_t
        regional_keys = (
            set(qs_by_region)
            | set(qd_by_region)
            | set(net_import_by_region)
            | set(bioenergy_by_region)
        )
        region_residuals = [
            float(qs_by_region.get((region, commodity, year), 0.0))
            + float(net_import_by_region.get((region, commodity, year), 0.0))
            - float(qd_by_region.get((region, commodity, year), 0.0))
            - float(bioenergy_by_region.get((region, commodity, year), 0.0))
            for region, item, item_year in regional_keys
            if item == commodity and item_year == year
        ]
        if region_residuals and np.isfinite(region_residuals).all():
            regional_max_abs_residual_t = max(abs(v) for v in region_residuals)
            regional_sum_abs_residual_t = sum(abs(v) for v in region_residuals)
        elif region_residuals:
            regional_max_abs_residual_t = float("nan")
            regional_sum_abs_residual_t = float("nan")
        else:
            regional_max_abs_residual_t = abs(regional_residual_t)
            regional_sum_abs_residual_t = abs(regional_residual_t)
        clearing_residual_t = net_import_t + excess_t - shortage_t
        physical_residual_t = (
            supply_t + shortage_t - excess_t - demand_t - bioenergy_t
        )
        raw_qd_minus_qs_t = demand_t - supply_t
        if np.isfinite(market_total_use_t) and market_total_use_t > 0.0:
            shortage_rate = shortage_t / market_total_use_t
        elif np.isfinite(shortage_t) and abs(shortage_t) <= 0.0:
            shortage_rate = 0.0
        else:
            shortage_rate = float("nan")

        rows.append(
            {
                "schema_version": MARKET_BALANCE_SCHEMA_VERSION,
                "diagnostic_source": MARKET_BALANCE_DIAGNOSTIC_SOURCE,
                "year": int(year),
                "commodity": commodity,
                "Qs": supply_t,
                "Qd": demand_t,
                "bioenergy_demand_t": bioenergy_t,
                "market_total_use_t": market_total_use_t,
                "net_import_t": net_import_t,
                "shortage_t": shortage_t,
                "excess_t": excess_t,
                "regional_balance_residual_t": regional_residual_t,
                "regional_balance_max_abs_residual_t": regional_max_abs_residual_t,
                "regional_balance_sum_abs_residual_t": regional_sum_abs_residual_t,
                "global_clearing_residual_t": clearing_residual_t,
                "physical_balance_residual_t": physical_residual_t,
                "raw_qd_minus_qs_t": raw_qd_minus_qs_t,
                "shortage_rate": shortage_rate,
            }
        )
    return rows


def _default_market_summary(error: str = "") -> Dict[str, Any]:
    return {
        "market_shortage_t": None,
        "market_excess_t": None,
        "market_gap_rate": None,
        "market_gap_year": None,
        "market_balance_max_abs_residual_t": None,
        "market_balance_diagnostic_error": str(error or ""),
    }


def summarize_market_balance_frame(
    df: pd.DataFrame,
    *,
    max_gap_rate: Optional[float] = None,
    residual_abs_tol_t: float = 1e-4,
    residual_rel_tol: float = 1e-9,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Validate a diagnostic frame and summarize its worst future year."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        error = "market balance diagnostics are empty"
        return _default_market_summary(error), error

    missing = sorted(MARKET_BALANCE_REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        error = f"market balance diagnostics missing solver-faithful columns: {missing}"
        return _default_market_summary(error), error

    work = df.copy()
    sources = set(work["diagnostic_source"].astype(str).str.strip())
    if sources != {MARKET_BALANCE_DIAGNOSTIC_SOURCE}:
        error = (
            "market balance diagnostics are not solver-postsolve data: "
            f"sources={sorted(sources)}"
        )
        return _default_market_summary(error), error

    schema_version = pd.to_numeric(work["schema_version"], errors="coerce")
    if schema_version.isna().any() or (schema_version < MARKET_BALANCE_SCHEMA_VERSION).any():
        error = "market balance diagnostics have an invalid schema_version"
        return _default_market_summary(error), error

    numeric_cols = [
        "year",
        "Qs",
        "Qd",
        "bioenergy_demand_t",
        "market_total_use_t",
        "net_import_t",
        "shortage_t",
        "excess_t",
        "regional_balance_residual_t",
        "regional_balance_max_abs_residual_t",
        "regional_balance_sum_abs_residual_t",
        "global_clearing_residual_t",
        "physical_balance_residual_t",
        "shortage_rate",
    ]
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    finite_mask = np.isfinite(work[numeric_cols].to_numpy(dtype=float))
    if not bool(finite_mask.all()):
        error = "market balance diagnostics contain missing or non-finite numeric values"
        return _default_market_summary(error), error

    work["year"] = work["year"].astype(int)
    future = work[work["year"] > 2020].copy()
    if future.empty:
        error = "market balance diagnostics contain no future-year rows"
        return _default_market_summary(error), error
    work = future

    if work.duplicated(["year", "commodity"]).any():
        error = "market balance diagnostics contain duplicate commodity-year rows"
        return _default_market_summary(error), error

    q_supply = work["Qs"].to_numpy(dtype=float)
    q_demand = work["Qd"].to_numpy(dtype=float)
    q_bioenergy = work["bioenergy_demand_t"].to_numpy(dtype=float)
    q_net_import = work["net_import_t"].to_numpy(dtype=float)
    q_shortage = work["shortage_t"].to_numpy(dtype=float)
    q_excess = work["excess_t"].to_numpy(dtype=float)

    calc_total_use = q_demand + q_bioenergy
    calc_regional = q_supply + q_net_import - q_demand - q_bioenergy
    calc_clearing = q_net_import + q_excess - q_shortage
    calc_physical = q_supply + q_shortage - q_excess - q_demand - q_bioenergy
    scale = np.maximum.reduce(
        [
            np.abs(q_supply),
            np.abs(calc_total_use),
            np.abs(q_net_import),
            np.abs(q_shortage),
            np.abs(q_excess),
            np.ones(len(work), dtype=float),
        ]
    )
    allowed = np.maximum(float(residual_abs_tol_t), float(residual_rel_tol) * scale)

    declared_fields = {
        "market_total_use_t": calc_total_use,
        "regional_balance_residual_t": calc_regional,
        "global_clearing_residual_t": calc_clearing,
        "physical_balance_residual_t": calc_physical,
    }
    for col, calculated in declared_fields.items():
        declared = work[col].to_numpy(dtype=float)
        mismatch = np.abs(declared - calculated) > allowed
        if bool(mismatch.any()):
            idx = int(np.flatnonzero(mismatch)[0])
            row = work.iloc[idx]
            error = (
                f"market balance diagnostic field {col} is inconsistent for "
                f"{row['commodity']} {int(row['year'])}"
            )
            return _default_market_summary(error), error

    negative_slack = (q_shortage < -allowed) | (q_excess < -allowed)
    if bool(negative_slack.any()):
        error = "market balance diagnostics contain negative shortage/excess slack"
        return _default_market_summary(error), error

    regional_max_abs = work["regional_balance_max_abs_residual_t"].to_numpy(dtype=float)
    residual_matrix = np.vstack(
        [
            np.abs(calc_regional),
            np.abs(regional_max_abs),
            np.abs(calc_clearing),
            np.abs(calc_physical),
        ]
    )
    residual_exceeded = (residual_matrix > allowed).any(axis=0)
    max_abs_residual_t = float(residual_matrix.max()) if residual_matrix.size else 0.0
    if bool(residual_exceeded.any()):
        idx = int(np.flatnonzero(residual_exceeded)[0])
        row = work.iloc[idx]
        error = (
            f"market balance residual exceeds tolerance for {row['commodity']} "
            f"{int(row['year'])}: max_abs={float(residual_matrix[:, idx].max()):.6g} t, "
            f"allowed={float(allowed[idx]):.6g} t"
        )
        summary = _default_market_summary(error)
        summary["market_balance_max_abs_residual_t"] = max_abs_residual_t
        return summary, error

    grouped = (
        work.groupby("year", as_index=False)[
            ["shortage_t", "excess_t", "market_total_use_t"]
        ]
        .sum(min_count=1)
        .sort_values("year")
    )
    grouped["gap_rate"] = np.where(
        grouped["market_total_use_t"] > 0.0,
        grouped["shortage_t"] / grouped["market_total_use_t"],
        np.where(grouped["shortage_t"].abs() <= float(residual_abs_tol_t), 0.0, np.nan),
    )
    if not np.isfinite(grouped["gap_rate"].to_numpy(dtype=float)).all():
        error = "market shortage is positive while market total use is non-positive"
        return _default_market_summary(error), error

    worst = grouped.sort_values(["gap_rate", "year"], ascending=[False, True]).iloc[0]
    summary = {
        "market_shortage_t": float(worst["shortage_t"]),
        "market_excess_t": float(worst["excess_t"]),
        "market_gap_rate": float(worst["gap_rate"]),
        "market_gap_year": int(worst["year"]),
        "market_balance_max_abs_residual_t": max_abs_residual_t,
        "market_balance_diagnostic_error": "",
    }
    if max_gap_rate is not None and float(worst["gap_rate"]) > float(max_gap_rate):
        error = (
            f"global market shortage {float(worst['shortage_t']):.6g} t "
            f"({float(worst['gap_rate']) * 100:.2f}% of total market use) "
            f"in {int(worst['year'])}; threshold={float(max_gap_rate) * 100:.2f}%"
        )
        summary["market_balance_diagnostic_error"] = error
        return summary, error
    return summary, None


def read_market_balance_summary(scenario_dir: Path) -> Dict[str, Any]:
    """Read and strictly validate a run's solver-faithful market diagnostics."""
    diag_path = (
        Path(scenario_dir)
        / "Diagnostics"
        / MARKET_BALANCE_DIAGNOSTIC_FILENAME
    )
    if not diag_path.exists():
        return _default_market_summary(
            f"missing market balance diagnostics: {diag_path}"
        )
    try:
        df = pd.read_csv(diag_path)
    except Exception as exc:
        return _default_market_summary(
            f"failed to read market balance diagnostics: {exc}"
        )
    summary, _ = summarize_market_balance_frame(df)
    return summary


def validate_market_balance_gap(
    scenario_dir: Path,
    *,
    max_gap_rate: float = 0.01,
    residual_abs_tol_t: float = 1e-4,
    residual_rel_tol: float = 1e-9,
) -> Optional[str]:
    """Return an error unless solver balance and shortage-rate checks pass."""
    diag_path = (
        Path(scenario_dir)
        / "Diagnostics"
        / MARKET_BALANCE_DIAGNOSTIC_FILENAME
    )
    if not diag_path.exists():
        return f"missing market balance diagnostics: {diag_path}"
    try:
        df = pd.read_csv(diag_path)
    except Exception as exc:
        return f"failed to read market balance diagnostics: {exc}"
    _, error = summarize_market_balance_frame(
        df,
        max_gap_rate=max_gap_rate,
        residual_abs_tol_t=residual_abs_tol_t,
        residual_rel_tol=residual_rel_tol,
    )
    return error
