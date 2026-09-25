# -*- coding: utf-8 -*-
"""Prepare publication-traceable source data for Figure 3d.

The upstream S5.8 design applies each mitigation intervention separately in
one country while holding the other countries at the matched reference.  This
module ranks the nine physical mitigation potentials within each country,
reports exact and near ties, validates the country-by-intervention grid, and
writes the categorical source table consumed by the Figure 3d map plotter.

The default ranking metric is the selected country's own emissions reduction,
which reproduces the scientific definition used by the existing Figure 3d
draft.  A global-system metric is available as an explicit leakage/trade
sensitivity diagnostic; it is never substituted silently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_OUTPUT_ROOT = (CODE_ROOT / "output").resolve()
DEFAULT_INPUT_DIR = ALLOWED_OUTPUT_ROOT / "Country_Strategy_Map_Sensitivity" / "merged"
DEFAULT_OUTPUT_DIR = ALLOWED_OUTPUT_ROOT / "Country_Dominant_Mitigation_Intervention"

USABLE_RUN_STATUSES = frozenset({"ok", "resumed"})
EXPECTED_STRATEGY_KINDS: Tuple[str, ...] = (
    "ruminant_reduction",
    "losses_ratio",
    "yield_rate",
    "feed_intensity",
    "enteric_fermentation_management",
    "manure_management",
    "crop_residue_soil_management",
    "rice_cultivation",
    "fertilizer_efficiency",
)

STRATEGY_METADATA: Dict[str, Dict[str, object]] = {
    "ruminant_reduction": {
        "database_strategy": "RuminantReduction",
        "display_name": "Reduce ruminant",
        "color": "#911c43",
    },
    "losses_ratio": {
        "database_strategy": "LossWaste",
        "display_name": "Reduce waste",
        "color": "#fb9a99",
    },
    "yield_rate": {
        "database_strategy": "YieldRate",
        "display_name": "Improve yield rate",
        "color": "#e4754f",
    },
    "feed_intensity": {
        "database_strategy": "FeedEfficiency",
        "display_name": "Improve feed efficiency",
        "color": "#fdb75c",
    },
    "enteric_fermentation_management": {
        "database_strategy": "EntericF",
        "display_name": "Enteric fermentation management",
        "color": "#5c509d",
    },
    "manure_management": {
        "database_strategy": "Manure",
        "display_name": "Manure management",
        "color": "#0868ac",
    },
    "crop_residue_soil_management": {
        "database_strategy": "Residue",
        "display_name": "Crop residue+soil management",
        "color": "#2bafd7",
    },
    "rice_cultivation": {
        "database_strategy": "Rice",
        "display_name": "Rice management",
        "color": "#7bccc4",
    },
    "fertilizer_efficiency": {
        "database_strategy": "Fertilizer",
        "display_name": "Improve nitrogen efficiency",
        "color": "#ccebc5",
    },
}

METRIC_COLUMNS = {
    "domestic": "domestic_emission_reduction_gt",
    "global_system": "global_system_emission_reduction_gt",
}


@dataclass(frozen=True)
class Figure3dSettings:
    input_dir: Path = DEFAULT_INPUT_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    metric: str = "domestic"
    minimum_positive_gt: float = 1e-12
    tie_absolute_tolerance_gt: float = 1e-9
    tie_relative_tolerance: float = 1e-6
    near_tie_percent: float = 1.0
    expected_country_count: Optional[int] = 190
    require_v2_provenance: bool = True
    strict: bool = True
    plot: bool = True


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_m49(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip().replace("'", "")
    if not text or text.lower() == "nan":
        return ""
    try:
        return f"{int(float(text)):03d}"
    except (TypeError, ValueError, OverflowError):
        return text


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna("").astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes", "y", "on"}
    )


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def ensure_output_child(path: Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved == ALLOWED_OUTPUT_ROOT or not _is_within(
        resolved, ALLOWED_OUTPUT_ROOT
    ):
        raise ValueError(
            "Figure 3d output must be a strict child of Code/output; "
            f"got {resolved}"
        )
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing Figure 3d source input: {path}")
    frame = pd.read_csv(path, dtype={"M49_Country_Code": str})
    if frame.empty:
        raise ValueError(f"Figure 3d source input is empty: {path}")
    return frame


def _extract_reference_global_emissions(
    long_df: pd.DataFrame,
    status_df: pd.DataFrame,
) -> Tuple[float, float]:
    values = pd.Series(dtype=float)
    if "reference_global_afolu_emissions_gt" in long_df.columns:
        values = pd.to_numeric(
            long_df["reference_global_afolu_emissions_gt"], errors="coerce"
        ).dropna()
    if values.empty and not status_df.empty:
        if "scope" in status_df.columns and "afolu_emissions_gt_co2eq_yr" in status_df.columns:
            baseline = status_df[
                status_df["scope"].fillna("").astype(str).str.strip().eq("baseline")
            ]
            values = pd.to_numeric(
                baseline["afolu_emissions_gt_co2eq_yr"], errors="coerce"
            ).dropna()
    if values.empty:
        return np.nan, np.nan
    return float(values.iloc[0]), float(values.max() - values.min())


def _prepare_working_table(
    long_df: pd.DataFrame,
    status_df: pd.DataFrame,
    *,
    metric: str,
    minimum_positive_gt: float,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    metric_key = str(metric).strip().lower()
    if metric_key not in METRIC_COLUMNS:
        raise ValueError(f"metric must be one of {sorted(METRIC_COLUMNS)}")
    required = {
        "M49_Country_Code",
        "strategy_kind",
        "run_status",
        "domestic_emission_reduction_gt",
    }
    missing = sorted(required.difference(long_df.columns))
    if missing:
        raise ValueError(f"country_strategy_long.csv is missing columns: {missing}")

    work = long_df.copy()
    work["M49_Country_Code"] = work["M49_Country_Code"].map(_normalize_m49)
    work["strategy_kind"] = work["strategy_kind"].fillna("").astype(str).str.strip()
    work["run_status"] = work["run_status"].fillna("").astype(str).str.strip()
    work = work[
        work["M49_Country_Code"].ne("") & work["strategy_kind"].ne("")
    ].copy()

    reference_global_gt, reference_spread_gt = _extract_reference_global_emissions(
        work, status_df
    )
    if "global_system_emission_reduction_gt" not in work.columns:
        if "global_afolu_emissions_gt" in work.columns and np.isfinite(reference_global_gt):
            work["global_system_emission_reduction_gt"] = (
                reference_global_gt
                - pd.to_numeric(work["global_afolu_emissions_gt"], errors="coerce")
            )
        else:
            work["global_system_emission_reduction_gt"] = np.nan

    metric_column = METRIC_COLUMNS[metric_key]
    work[metric_column] = pd.to_numeric(work[metric_column], errors="coerce")
    work["figure3d_metric"] = metric_key
    work["figure3d_metric_column"] = metric_column
    work["mitigation_potential_gt"] = work[metric_column]
    work["figure3d_strategy_order"] = work["strategy_kind"].map(
        {kind: index for index, kind in enumerate(EXPECTED_STRATEGY_KINDS)}
    ).fillna(999).astype(int)
    work["figure3d_database_strategy"] = work["strategy_kind"].map(
        {kind: meta["database_strategy"] for kind, meta in STRATEGY_METADATA.items()}
    )
    work["figure3d_intervention"] = work["strategy_kind"].map(
        {kind: meta["display_name"] for kind, meta in STRATEGY_METADATA.items()}
    )
    work["figure3d_color"] = work["strategy_kind"].map(
        {kind: meta["color"] for kind, meta in STRATEGY_METADATA.items()}
    )
    work["figure3d_eligible"] = (
        work["strategy_kind"].isin(EXPECTED_STRATEGY_KINDS)
        & work["run_status"].isin(USABLE_RUN_STATUSES)
        & np.isfinite(work["mitigation_potential_gt"])
        & work["mitigation_potential_gt"].gt(float(minimum_positive_gt))
    )
    metadata = {
        "metric": metric_key,
        "metric_column": metric_column,
        "reference_global_afolu_emissions_gt": reference_global_gt,
        "reference_global_emissions_spread_gt": reference_spread_gt,
    }
    return work, metadata


def rank_country_interventions(
    long_df: pd.DataFrame,
    status_df: Optional[pd.DataFrame] = None,
    *,
    metric: str = "domestic",
    minimum_positive_gt: float = 1e-12,
    tie_absolute_tolerance_gt: float = 1e-9,
    tie_relative_tolerance: float = 1e-6,
    near_tie_percent: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Return ranked rows, one country record, category summary, and diagnostics."""

    status = status_df.copy() if status_df is not None else pd.DataFrame()
    work, metadata = _prepare_working_table(
        long_df,
        status,
        metric=metric,
        minimum_positive_gt=minimum_positive_gt,
    )
    key_columns = ["M49_Country_Code", "strategy_kind"]
    duplicate_mask = work.duplicated(key_columns, keep=False)
    duplicate_key_count = int(work.loc[duplicate_mask, key_columns].drop_duplicates().shape[0])

    work["_status_priority"] = work["run_status"].map(
        {"ok": 0, "resumed": 1}
    ).fillna(9)
    work = (
        work.sort_values(
            key_columns + ["_status_priority", "scenario_id"]
            if "scenario_id" in work.columns
            else key_columns + ["_status_priority"]
        )
        .drop_duplicates(key_columns, keep="first")
        .drop(columns="_status_priority")
        .reset_index(drop=True)
    )
    work["mitigation_rank"] = pd.Series(pd.NA, index=work.index, dtype="Int64")
    work["is_dominant_intervention"] = False
    work["within_numeric_tie_tolerance"] = False

    country_records = []
    country_columns = [
        column
        for column in (
            "ISO3",
            "country_name",
            "Region_aggMC",
            "year",
            "figure3d_reference_context",
            "bioenergy_enabled",
            "bioenergy_scenario",
        )
        if column in work.columns
    ]
    for country, group in work.groupby("M49_Country_Code", sort=True):
        eligible = group[_as_bool(group["figure3d_eligible"])].sort_values(
            ["mitigation_potential_gt", "figure3d_strategy_order"],
            ascending=[False, True],
        )
        if not eligible.empty:
            work.loc[eligible.index, "mitigation_rank"] = list(
                range(1, len(eligible) + 1)
            )
        base = {"M49_Country_Code": country}
        for column in country_columns:
            values = group[column].dropna()
            base[column] = values.iloc[0] if not values.empty else ""
        strategy_count = int(group["strategy_kind"].nunique())
        usable_count = int(group["run_status"].isin(USABLE_RUN_STATUSES).sum())
        positive_count = int(len(eligible))
        base.update(
            {
                "scenario_family": "country_local_single_intervention_endpoint",
                "ranking_metric": metadata["metric"],
                "ranking_metric_column": metadata["metric_column"],
                "strategy_count": strategy_count,
                "expected_strategy_count": len(EXPECTED_STRATEGY_KINDS),
                "complete_strategy_grid": strategy_count == len(EXPECTED_STRATEGY_KINDS),
                "usable_strategy_count": usable_count,
                "positive_strategy_count": positive_count,
                "dominant_strategy_kind": "",
                "dominant_database_strategy": "",
                "dominant_intervention": "",
                "dominant_color": "",
                "mitigation_potential_gt": np.nan,
                "runner_up_strategy_kind": "",
                "runner_up_intervention": "",
                "runner_up_mitigation_potential_gt": np.nan,
                "dominance_margin_gt": np.nan,
                "dominance_margin_percent": np.nan,
                "numeric_tie": False,
                "near_tie": False,
                "tie_tolerance_gt": np.nan,
                "selection_status": "no_positive_potential",
            }
        )
        if not eligible.empty:
            top = eligible.iloc[0]
            runner = eligible.iloc[1] if len(eligible) > 1 else None
            top_value = float(top["mitigation_potential_gt"])
            runner_value = (
                float(runner["mitigation_potential_gt"])
                if runner is not None
                else np.nan
            )
            margin = top_value - runner_value if np.isfinite(runner_value) else np.nan
            margin_percent = (
                margin / top_value * 100.0
                if np.isfinite(margin) and top_value > 0
                else np.nan
            )
            tolerance = max(
                float(tie_absolute_tolerance_gt),
                abs(top_value) * float(tie_relative_tolerance),
            )
            numeric_tie = bool(np.isfinite(margin) and abs(margin) <= tolerance)
            near_tie = bool(
                np.isfinite(margin_percent)
                and margin_percent <= float(near_tie_percent)
            )
            tied = eligible[
                eligible["mitigation_potential_gt"].ge(top_value - tolerance)
            ]
            work.loc[top.name, "is_dominant_intervention"] = True
            work.loc[tied.index, "within_numeric_tie_tolerance"] = True
            base.update(
                {
                    "dominant_strategy_kind": top["strategy_kind"],
                    "dominant_database_strategy": top["figure3d_database_strategy"],
                    "dominant_intervention": top["figure3d_intervention"],
                    "dominant_color": top["figure3d_color"],
                    "mitigation_potential_gt": top_value,
                    "runner_up_strategy_kind": (
                        runner["strategy_kind"] if runner is not None else ""
                    ),
                    "runner_up_intervention": (
                        runner["figure3d_intervention"] if runner is not None else ""
                    ),
                    "runner_up_mitigation_potential_gt": runner_value,
                    "dominance_margin_gt": margin,
                    "dominance_margin_percent": margin_percent,
                    "numeric_tie": numeric_tie,
                    "near_tie": near_tie,
                    "tie_tolerance_gt": tolerance,
                    "selection_status": (
                        "selected_tie_broken_by_fixed_order"
                        if numeric_tie
                        else "selected_unique"
                    ),
                    "source_scenario_id": top.get("scenario_id", ""),
                    "source_scenario_dir": top.get("scenario_dir", ""),
                    "run_status": top.get("run_status", ""),
                    "cost_database_version": top.get("cost_database_version", ""),
                    "cost_database_sha256": top.get("cost_database_sha256", ""),
                    "cost_reference_scenario_id": top.get(
                        "cost_reference_scenario_id", ""
                    ),
                    "cost_attribution_method": top.get(
                        "cost_attribution_method", ""
                    ),
                }
            )
        country_records.append(base)

    ranked = work.sort_values(
        ["M49_Country_Code", "figure3d_strategy_order"]
    ).reset_index(drop=True)
    dominant = pd.DataFrame(country_records).sort_values(
        "M49_Country_Code"
    ).reset_index(drop=True)
    selected = dominant[dominant["dominant_strategy_kind"].ne("")].copy()
    if selected.empty:
        category_summary = pd.DataFrame(
            columns=[
                "strategy_order",
                "dominant_strategy_kind",
                "dominant_intervention",
                "dominant_color",
                "country_count",
                "country_share_percent",
                "total_dominant_mitigation_potential_gt",
                "median_dominance_margin_percent",
            ]
        )
    else:
        category_summary = (
            selected.groupby(
                [
                    "dominant_strategy_kind",
                    "dominant_intervention",
                    "dominant_color",
                ],
                as_index=False,
            )
            .agg(
                country_count=("M49_Country_Code", "nunique"),
                total_dominant_mitigation_potential_gt=(
                    "mitigation_potential_gt",
                    "sum",
                ),
                median_dominance_margin_percent=(
                    "dominance_margin_percent",
                    "median",
                ),
            )
        )
        category_summary["country_share_percent"] = (
            category_summary["country_count"] / len(dominant) * 100.0
        )
        category_summary["strategy_order"] = category_summary[
            "dominant_strategy_kind"
        ].map({kind: i for i, kind in enumerate(EXPECTED_STRATEGY_KINDS)})
        category_summary = category_summary.sort_values("strategy_order")

    diagnostics: Dict[str, object] = {
        **metadata,
        "input_rows": int(len(long_df)),
        "deduplicated_rows": int(len(ranked)),
        "duplicate_country_strategy_keys": duplicate_key_count,
        "country_count": int(dominant["M49_Country_Code"].nunique()),
        "winner_country_count": int(selected["M49_Country_Code"].nunique()),
        "incomplete_country_count": int((~dominant["complete_strategy_grid"]).sum()),
        "unusable_strategy_rows": int(
            (~ranked["run_status"].isin(USABLE_RUN_STATUSES)).sum()
        ),
        "nonfinite_metric_rows": int(
            (~np.isfinite(ranked["mitigation_potential_gt"])).sum()
        ),
        "unexpected_strategy_rows": int(
            (~ranked["strategy_kind"].isin(EXPECTED_STRATEGY_KINDS)).sum()
        ),
        "numeric_tie_country_count": int(dominant["numeric_tie"].sum()),
        "near_tie_country_count": int(dominant["near_tie"].sum()),
    }
    return ranked, dominant, category_summary, diagnostics


def build_validation_table(
    ranked: pd.DataFrame,
    dominant: pd.DataFrame,
    diagnostics: Mapping[str, object],
    *,
    require_v2_provenance: bool,
    expected_country_count: Optional[int] = None,
) -> pd.DataFrame:
    rows = []

    def add(
        check: str,
        passed: bool,
        actual: object,
        expected: object,
        detail: str,
        *,
        failure_status: str = "FAIL",
    ) -> None:
        rows.append(
            {
                "check": check,
                "status": "PASS" if passed else failure_status,
                "actual": actual,
                "expected": expected,
                "detail": detail,
            }
        )

    country_count = int(diagnostics.get("country_count", 0) or 0)
    expected_rows = country_count * len(EXPECTED_STRATEGY_KINDS)
    add("nonempty_country_set", country_count > 0, country_count, ">0", "Modeled countries.")
    if expected_country_count is not None and int(expected_country_count) > 0:
        add(
            "expected_country_coverage",
            country_count == int(expected_country_count),
            country_count,
            int(expected_country_count),
            "Publication maps require the declared full model-country universe.",
        )
    add(
        "unique_country_strategy_key",
        int(diagnostics.get("duplicate_country_strategy_keys", 0) or 0) == 0,
        diagnostics.get("duplicate_country_strategy_keys", 0),
        0,
        "Duplicate M49-strategy keys are not valid ranking evidence.",
    )
    add(
        "complete_nine_strategy_grid",
        int(diagnostics.get("incomplete_country_count", 0) or 0) == 0
        and len(ranked) == expected_rows,
        f"rows={len(ranked)}; incomplete_countries={diagnostics.get('incomplete_country_count', 0)}",
        f"rows={expected_rows}; incomplete_countries=0",
        "Every country must be evaluated for the same nine interventions.",
    )
    add(
        "expected_strategy_vocabulary",
        int(diagnostics.get("unexpected_strategy_rows", 0) or 0) == 0,
        diagnostics.get("unexpected_strategy_rows", 0),
        0,
        "Only the Figure 3 intervention vocabulary may enter the ranking.",
    )
    add(
        "all_scenarios_usable",
        int(diagnostics.get("unusable_strategy_rows", 0) or 0) == 0,
        diagnostics.get("unusable_strategy_rows", 0),
        0,
        "Only ok/resumed optimization results are publication-ready.",
    )
    add(
        "finite_ranking_metric",
        int(diagnostics.get("nonfinite_metric_rows", 0) or 0) == 0,
        diagnostics.get("nonfinite_metric_rows", 0),
        0,
        "Every country-strategy row needs a finite physical mitigation metric.",
    )
    add(
        "winner_for_every_country",
        int(diagnostics.get("winner_country_count", 0) or 0) == country_count,
        diagnostics.get("winner_country_count", 0),
        country_count,
        "Countries without positive mitigation remain explicit no-data records.",
    )

    version_values = set()
    hash_values = set()
    if "cost_database_version" in ranked.columns:
        version_values = {
            str(value).strip()
            for value in ranked["cost_database_version"].dropna()
            if str(value).strip()
        }
    if "cost_database_sha256" in ranked.columns:
        hash_values = {
            str(value).strip()
            for value in ranked["cost_database_sha256"].dropna()
            if str(value).strip()
        }
    provenance_ok = (
        len(version_values) == 1
        and next(iter(version_values), "").startswith("v2.0")
        and len(hash_values) == 1
        and len(next(iter(hash_values), "")) == 64
    )
    add(
        "current_v2_database_provenance",
        provenance_ok,
        f"versions={sorted(version_values)}; hashes={len(hash_values)}",
        "one v2.0 version and one 64-character SHA-256",
        "The ranking is physical, but publication reruns still carry the active model/database identity.",
        failure_status="WARN" if not require_v2_provenance else "FAIL",
    )
    add(
        "reference_global_consistency",
        not np.isfinite(float(diagnostics.get("reference_global_emissions_spread_gt", np.nan)))
        or float(diagnostics.get("reference_global_emissions_spread_gt", 0.0)) <= 1e-9,
        diagnostics.get("reference_global_emissions_spread_gt", np.nan),
        "<=1e-9 Gt",
        "Repeated batch reference totals must agree when present.",
    )
    reference_contexts = set()
    bioenergy_flags = set()
    if "figure3d_reference_context" in ranked.columns:
        reference_contexts = {
            str(value).strip()
            for value in ranked["figure3d_reference_context"].dropna()
            if str(value).strip()
        }
    if "bioenergy_enabled" in ranked.columns:
        bioenergy_flags = set(_as_bool(ranked["bioenergy_enabled"]).tolist())
    explicit_disabled_reference = (
        reference_contexts == {"bioenergy_disabled_reference"}
        and bioenergy_flags == {False}
    )
    add(
        "explicit_bioenergy_disabled_reference",
        explicit_disabled_reference,
        f"contexts={sorted(reference_contexts)}; enabled_flags={sorted(bioenergy_flags)}",
        "contexts=['bioenergy_disabled_reference']; enabled_flags=[False]",
        (
            "Panel d uses one explicitly bioenergy-disabled reference; it is not "
            "silently inherited from panels a-c."
        ),
    )
    rows.append(
        {
            "check": "numeric_ties",
            "status": "INFO",
            "actual": diagnostics.get("numeric_tie_country_count", 0),
            "expected": "reported",
            "detail": "Exact/numerical ties use fixed intervention order only for deterministic display.",
        }
    )
    near_ties = int(diagnostics.get("near_tie_country_count", 0) or 0)
    rows.append(
        {
            "check": "near_ties",
            "status": "WARN" if near_ties else "PASS",
            "actual": near_ties,
            "expected": "reported",
            "detail": "Near-tie countries should be retained in source data for reviewer sensitivity checks.",
        }
    )
    comparable = int(
        diagnostics.get("metric_sensitivity_comparable_countries", 0) or 0
    )
    same = int(
        diagnostics.get("metric_sensitivity_same_winner_countries", 0) or 0
    )
    rows.append(
        {
            "check": "domestic_vs_global_system_metric_sensitivity",
            "status": "INFO",
            "actual": f"same={same}; comparable={comparable}",
            "expected": "reported",
            "detail": (
                "The manuscript map uses domestic potential; this diagnostic "
                "shows how trade/leakage changes the winner under a global-system definition."
            ),
        }
    )
    return pd.DataFrame(rows)


def write_figure3d_outputs(
    long_df: pd.DataFrame,
    status_df: Optional[pd.DataFrame],
    *,
    output_dir: Path,
    settings: Optional[Figure3dSettings] = None,
    source_files: Sequence[Path] = (),
) -> Tuple[Path, pd.DataFrame]:
    cfg = settings or Figure3dSettings(output_dir=Path(output_dir), plot=False)
    out_dir = ensure_output_child(Path(output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    ranked, dominant, category_summary, diagnostics = rank_country_interventions(
        long_df,
        status_df,
        metric=cfg.metric,
        minimum_positive_gt=cfg.minimum_positive_gt,
        tie_absolute_tolerance_gt=cfg.tie_absolute_tolerance_gt,
        tie_relative_tolerance=cfg.tie_relative_tolerance,
        near_tie_percent=cfg.near_tie_percent,
    )
    if cfg.metric == "domestic":
        domestic_dominant = dominant
    else:
        _, domestic_dominant, _, _ = rank_country_interventions(
            long_df,
            status_df,
            metric="domestic",
            minimum_positive_gt=cfg.minimum_positive_gt,
            tie_absolute_tolerance_gt=cfg.tie_absolute_tolerance_gt,
            tie_relative_tolerance=cfg.tie_relative_tolerance,
            near_tie_percent=cfg.near_tie_percent,
        )
    if cfg.metric == "global_system":
        global_dominant = dominant
    else:
        _, global_dominant, _, _ = rank_country_interventions(
            long_df,
            status_df,
            metric="global_system",
            minimum_positive_gt=cfg.minimum_positive_gt,
            tie_absolute_tolerance_gt=cfg.tie_absolute_tolerance_gt,
            tie_relative_tolerance=cfg.tie_relative_tolerance,
            near_tie_percent=cfg.near_tie_percent,
        )
    domestic_compare = domestic_dominant[
        [
            "M49_Country_Code",
            "dominant_strategy_kind",
            "dominant_intervention",
            "mitigation_potential_gt",
        ]
    ].rename(
        columns={
            "dominant_strategy_kind": "domestic_dominant_strategy_kind",
            "dominant_intervention": "domestic_dominant_intervention",
            "mitigation_potential_gt": "domestic_mitigation_potential_gt",
        }
    )
    global_compare = global_dominant[
        [
            "M49_Country_Code",
            "dominant_strategy_kind",
            "dominant_intervention",
            "mitigation_potential_gt",
        ]
    ].rename(
        columns={
            "dominant_strategy_kind": "global_system_dominant_strategy_kind",
            "dominant_intervention": "global_system_dominant_intervention",
            "mitigation_potential_gt": "global_system_mitigation_potential_gt",
        }
    )
    metric_sensitivity = domestic_compare.merge(
        global_compare,
        on="M49_Country_Code",
        how="outer",
        validate="one_to_one",
    )
    metric_sensitivity["same_dominant_intervention"] = metric_sensitivity[
        "domestic_dominant_strategy_kind"
    ].fillna("").eq(
        metric_sensitivity["global_system_dominant_strategy_kind"].fillna("")
    )
    metric_sensitivity["both_metrics_have_winner"] = (
        metric_sensitivity["domestic_dominant_strategy_kind"].fillna("").ne("")
        & metric_sensitivity["global_system_dominant_strategy_kind"].fillna("").ne("")
    )
    comparable = metric_sensitivity["both_metrics_have_winner"]
    diagnostics["metric_sensitivity_comparable_countries"] = int(comparable.sum())
    diagnostics["metric_sensitivity_same_winner_countries"] = int(
        (
            metric_sensitivity["same_dominant_intervention"]
            & metric_sensitivity["both_metrics_have_winner"]
        ).sum()
    )
    validation = build_validation_table(
        ranked,
        dominant,
        diagnostics,
        require_v2_provenance=cfg.require_v2_provenance,
        expected_country_count=cfg.expected_country_count,
    )
    overall_passed = not validation["status"].eq("FAIL").any()
    reference_context_values = sorted(
        {
            str(value).strip()
            for value in ranked.get(
                "figure3d_reference_context", pd.Series(dtype=str)
            ).dropna()
            if str(value).strip()
        }
    )
    reference_context = (
        reference_context_values[0]
        if len(reference_context_values) == 1
        else "mixed_or_missing"
    )

    ranked_path = out_dir / "figure3d_country_strategy_ranked.csv"
    dominant_path = out_dir / "figure3d_country_dominant_intervention.csv"
    category_path = out_dir / "figure3d_category_summary.csv"
    sensitivity_path = out_dir / "figure3d_metric_sensitivity_comparison.csv"
    validation_path = out_dir / "figure3d_data_quality.csv"
    ranked.to_csv(ranked_path, index=False, encoding="utf-8-sig")
    dominant.to_csv(dominant_path, index=False, encoding="utf-8-sig")
    category_summary.to_csv(category_path, index=False, encoding="utf-8-sig")
    metric_sensitivity.to_csv(sensitivity_path, index=False, encoding="utf-8-sig")
    validation.to_csv(validation_path, index=False, encoding="utf-8-sig")

    metadata_rows = [
        {"field": "generated_at_utc", "value": _utc_now()},
        {"field": "scenario_family", "value": "country_local_single_intervention_endpoint"},
        {"field": "reference_context", "value": reference_context},
        {"field": "ranking_metric", "value": cfg.metric},
        {"field": "ranking_metric_column", "value": METRIC_COLUMNS[cfg.metric]},
        {"field": "minimum_positive_gt", "value": cfg.minimum_positive_gt},
        {"field": "tie_absolute_tolerance_gt", "value": cfg.tie_absolute_tolerance_gt},
        {"field": "tie_relative_tolerance", "value": cfg.tie_relative_tolerance},
        {"field": "near_tie_percent", "value": cfg.near_tie_percent},
        {"field": "expected_country_count", "value": cfg.expected_country_count},
        {"field": "overall_passed", "value": overall_passed},
    ]
    metadata = pd.DataFrame(metadata_rows)
    workbook_path = out_dir / "Figure3d_country_dominant_intervention_source_data.xlsx"
    with pd.ExcelWriter(workbook_path) as writer:
        dominant.to_excel(writer, sheet_name="country_dominant", index=False)
        ranked.to_excel(writer, sheet_name="country_strategy_ranked", index=False)
        category_summary.to_excel(writer, sheet_name="category_summary", index=False)
        metric_sensitivity.to_excel(writer, sheet_name="metric_sensitivity", index=False)
        validation.to_excel(writer, sheet_name="data_quality", index=False)
        metadata.to_excel(writer, sheet_name="metadata", index=False)

    manifest = {
        "schema_version": 1,
        "generated_at_utc": _utc_now(),
        "scenario_family": "country_local_single_intervention_endpoint",
        "reference_context": reference_context,
        "reference_context_values": reference_context_values,
        "ranking_metric": cfg.metric,
        "ranking_metric_column": METRIC_COLUMNS[cfg.metric],
        "expected_strategy_kinds": list(EXPECTED_STRATEGY_KINDS),
        "minimum_positive_gt": cfg.minimum_positive_gt,
        "tie_absolute_tolerance_gt": cfg.tie_absolute_tolerance_gt,
        "tie_relative_tolerance": cfg.tie_relative_tolerance,
        "near_tie_percent": cfg.near_tie_percent,
        "expected_country_count": cfg.expected_country_count,
        "require_v2_provenance": cfg.require_v2_provenance,
        "overall_passed": bool(overall_passed),
        "diagnostics": diagnostics,
        "source_files": [
            {
                "path": str(Path(path).resolve()),
                "sha256": _sha256(Path(path)),
            }
            for path in source_files
            if Path(path).exists()
        ],
        "outputs": {
            "ranked_csv": str(ranked_path),
            "dominant_csv": str(dominant_path),
            "category_summary_csv": str(category_path),
            "metric_sensitivity_csv": str(sensitivity_path),
            "validation_csv": str(validation_path),
            "source_workbook": str(workbook_path),
        },
    }
    manifest_path = out_dir / "figure3d_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    if cfg.strict and not overall_passed:
        failed = validation.loc[validation["status"].eq("FAIL"), "check"].tolist()
        raise RuntimeError(
            "Figure 3d source data failed strict validation: "
            f"{failed}. See {validation_path}"
        )
    return dominant_path, validation


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rank country-local mitigation interventions and prepare Figure 3d source data."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metric", choices=sorted(METRIC_COLUMNS), default="domestic")
    parser.add_argument("--minimum-positive-gt", type=float, default=1e-12)
    parser.add_argument("--tie-absolute-tolerance-gt", type=float, default=1e-9)
    parser.add_argument("--tie-relative-tolerance", type=float, default=1e-6)
    parser.add_argument("--near-tie-percent", type=float, default=1.0)
    parser.add_argument(
        "--expected-country-count",
        type=int,
        default=190,
        help="Required model-country count; use 0 only for an explicitly partial diagnostic.",
    )
    parser.add_argument(
        "--require-v2-provenance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> Path:
    args = _build_arg_parser().parse_args(list(argv) if argv is not None else None)
    input_dir = Path(args.input_dir).expanduser().resolve()
    long_path = input_dir / "country_strategy_long.csv"
    status_path = input_dir / "scenario_status.csv"
    long_df = _read_csv(long_path)
    status_df = _read_csv(status_path) if status_path.exists() else pd.DataFrame()
    settings = Figure3dSettings(
        input_dir=input_dir,
        output_dir=Path(args.output_dir),
        metric=str(args.metric),
        minimum_positive_gt=float(args.minimum_positive_gt),
        tie_absolute_tolerance_gt=float(args.tie_absolute_tolerance_gt),
        tie_relative_tolerance=float(args.tie_relative_tolerance),
        near_tie_percent=float(args.near_tie_percent),
        expected_country_count=(
            int(args.expected_country_count)
            if int(args.expected_country_count) > 0
            else None
        ),
        require_v2_provenance=bool(args.require_v2_provenance),
        strict=bool(args.strict),
        plot=bool(args.plot),
    )
    dominant_path, validation = write_figure3d_outputs(
        long_df,
        status_df,
        output_dir=Path(args.output_dir),
        settings=settings,
        source_files=(long_path, status_path),
    )
    out_dir = dominant_path.parent
    if settings.plot:
        import SP_M3d_Figure_country_dominant_mitigation_intervention as plotter

        plotter.main(
            [
                "--input-data",
                str(dominant_path),
                "--output-dir",
                str(out_dir / "figure"),
            ]
        )
    passed = not validation["status"].eq("FAIL").any()
    print(f"[S5_8_3] source_data={dominant_path}")
    print(f"[S5_8_3] validation={out_dir / 'figure3d_data_quality.csv'}")
    print(f"[S5_8_3] overall_passed={passed}")
    return out_dir


if __name__ == "__main__":
    main()
