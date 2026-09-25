# -*- coding: utf-8 -*-
"""
Diagnose whether small Strategy variables in S5_0_1 are active and whether
their attribution is reduced by multivariate regression.

Inputs:
- <results>/MC_Sensitivity/summary/samples.csv
- <results>/MC_Sensitivity/summary/importance_by_variable.csv
- <src>/Scenario_config_new.xlsx, sheet MC_effect_low_land_new

Outputs:
- <results>/Diagnostics/S5_0_1_small_variable_importance_diagnostics.xlsx
- <results>/Diagnostics/S5_0_1_small_variable_importance_diagnostics.md
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from config_paths import get_results_base, get_src_base


SMALL_KINDS = [
    "feed_intensity",
    "fertilizer_rate",
    "crop_soil_management_ratio",
    "land_carbon_price",
    "manure_management_ratio",
]

LARGE_REFERENCE_KINDS = [
    "yield_rate",
    "emission_factor",
    "ruminant_reduction",
    "losses_ratio",
]

DISPLAY_NAMES = {
    "yield_rate": "Improve yield rate",
    "feed_intensity": "Improve feed efficiency",
    "fertilizer_rate": "Improve fertilizer efficiency",
    "losses_ratio": "Reduce waste",
    "ruminant_reduction": "Reduce Ruminate",
    "manure_management_ratio": "Manure management",
    "crop_soil_management_ratio": "Crop residue management",
    "emission_factor": "Emission intensity",
    "land_carbon_price": "Land carbon price",
}

KIND_ALIASES = {
    "feed_efficiency": "feed_intensity",
    "feed_efficiency_rate": "feed_intensity",
    "feed_intensity_rate": "feed_intensity",
    "fertlizer_rate": "fertilizer_rate",
    "fertilizer_efficiency": "fertilizer_rate",
    "manure_ratio": "manure_management_ratio",
    "mm_ratio": "manure_management_ratio",
    "crop_soil_ratio": "crop_soil_management_ratio",
    "waste_rate": "losses_ratio",
    "waste_reduction": "losses_ratio",
    "ruminant_intake_ratio": "ruminant_reduction",
    "ruminant_intake_decreasing_ratio": "ruminant_reduction",
    "land_co2_price": "land_carbon_price",
}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose S5_0_1 small variable importance.")
    parser.add_argument("--summary-dir", type=str, default=None)
    parser.add_argument("--scenario-config", type=str, default=None)
    parser.add_argument("--mc-sheet", type=str, default="MC_effect_low_land_new")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--target-current", type=float, default=12.9)
    return parser


def _summary_dir(args: argparse.Namespace) -> Path:
    return Path(args.summary_dir) if args.summary_dir else Path(get_results_base()) / "MC_Sensitivity" / "summary"


def _scenario_config_path(args: argparse.Namespace) -> Path:
    return Path(args.scenario_config) if args.scenario_config else Path(get_src_base()) / "Scenario_config_new.xlsx"


def _out_dir(args: argparse.Namespace) -> Path:
    out = Path(args.out_dir) if args.out_dir else Path(get_results_base()) / "Diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _canonical_kind(raw: object) -> str:
    text = str(raw or "").strip().lower()
    if "|" in text:
        text = text.split("|", 1)[0]
    return KIND_ALIASES.get(text, text)


def _kind_from_element(elem: object) -> Optional[str]:
    elem_l = str(elem or "").strip().lower()
    if not elem_l:
        return None
    if "land" in elem_l and ("carbon" in elem_l or "co2" in elem_l) and "price" in elem_l:
        return "land_carbon_price"
    if "ruminant" in elem_l or "ruminate" in elem_l or "intake" in elem_l:
        return "ruminant_reduction"
    if "feed" in elem_l and ("intensity" in elem_l or "eff" in elem_l):
        return "feed_intensity"
    if "fertilizer" in elem_l or "fertlizer" in elem_l:
        return "fertilizer_rate"
    if "manure" in elem_l and ("ratio" in elem_l or "management" in elem_l):
        return "manure_management_ratio"
    if "yield" in elem_l and "feed" not in elem_l:
        return "yield_rate"
    if "loss" in elem_l or "waste" in elem_l:
        return "losses_ratio"
    if "crop_soil_management" in elem_l or ("soil" in elem_l and "management" in elem_l and "ratio" in elem_l):
        return "crop_soil_management_ratio"
    if "ef" in elem_l or "emission" in elem_l:
        return "emission_factor"
    return None


def _collapse_values(values: Iterable[object], limit: int = 8) -> str:
    items = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text not in items:
            items.append(text)
        if len(items) >= limit:
            break
    return " | ".join(items)


def _load_spec_summary(path: Path, sheet: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    for col in ("Element", "Element unit", "Process", "Item", "GHG", "Min_bound", "Max_bound", "Region_cat"):
        if col not in df.columns:
            df[col] = np.nan
    df["kind"] = df["Element"].map(_kind_from_element)
    df = df[df["kind"].notna()].copy()
    if df.empty:
        return pd.DataFrame()
    df["Min_bound_num"] = pd.to_numeric(df["Min_bound"], errors="coerce")
    df["Max_bound_num"] = pd.to_numeric(df["Max_bound"], errors="coerce")
    rows: List[Dict[str, object]] = []
    for kind, sub in df.groupby("kind", sort=True):
        rows.append(
            {
                "kind": kind,
                "display_name": DISPLAY_NAMES.get(kind, kind),
                "spec_rows": int(len(sub)),
                "units": _collapse_values(sub["Element unit"].dropna().unique()),
                "min_bound_min": float(sub["Min_bound_num"].min()) if sub["Min_bound_num"].notna().any() else np.nan,
                "max_bound_max": float(sub["Max_bound_num"].max()) if sub["Max_bound_num"].notna().any() else np.nan,
                "items": _collapse_values(sub["Item"].dropna().unique()),
                "processes": _collapse_values(sub["Process"].dropna().unique()),
                "regions": _collapse_values(sub["Region_cat"].dropna().unique()),
            }
        )
    return pd.DataFrame(rows)


def _load_samples(summary_dir: Path) -> pd.DataFrame:
    path = summary_dir / "samples.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing samples.csv: {path}")
    df = pd.read_csv(path)
    if "emissions_2080_gt" not in df.columns:
        raise KeyError(f"{path} missing emissions_2080_gt.")
    return df


def _parameter_columns_by_kind(samples_df: pd.DataFrame) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    excluded = {"sample_id", "scenario_id", "emissions_2080_gt"}
    for col in samples_df.columns:
        if col in excluded:
            continue
        kind = _canonical_kind(col)
        out.setdefault(kind, []).append(col)
    return out


def _group_matrix(samples_df: pd.DataFrame, by_kind: Dict[str, List[str]]) -> pd.DataFrame:
    groups: Dict[str, pd.Series] = {}
    for kind, cols in by_kind.items():
        vals = samples_df[cols].apply(pd.to_numeric, errors="coerce")
        groups[kind] = vals.mean(axis=1)
    return pd.DataFrame(groups)


def _sample_group_summary(samples_df: pd.DataFrame, by_kind: Dict[str, List[str]], groups: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for kind, cols in sorted(by_kind.items()):
        vals = samples_df[cols].apply(pd.to_numeric, errors="coerce")
        std_by_col = vals.std(axis=0, skipna=True)
        nonconstant = int((std_by_col.fillna(0.0) > 1e-12).sum())
        g = pd.to_numeric(groups[kind], errors="coerce")
        rows.append(
            {
                "kind": kind,
                "display_name": DISPLAY_NAMES.get(kind, kind),
                "parameter_columns": int(len(cols)),
                "nonconstant_columns": nonconstant,
                "group_value_min": float(g.min()) if g.notna().any() else np.nan,
                "group_value_max": float(g.max()) if g.notna().any() else np.nan,
                "group_value_std": float(g.std()) if g.notna().any() else np.nan,
                "group_value_unique_rounded": int(g.round(10).nunique(dropna=True)),
                "sample_columns_example": " | ".join(cols[:5]),
            }
        )
    return pd.DataFrame(rows)


def _pearson(x: pd.Series, y: pd.Series) -> float:
    work = pd.concat([x, y], axis=1).dropna()
    if len(work) < 3:
        return np.nan
    if work.iloc[:, 0].std() <= 0 or work.iloc[:, 1].std() <= 0:
        return np.nan
    return float(work.iloc[:, 0].corr(work.iloc[:, 1], method="pearson"))


def _spearman(x: pd.Series, y: pd.Series) -> float:
    work = pd.concat([x, y], axis=1).dropna()
    if len(work) < 3:
        return np.nan
    if work.iloc[:, 0].std() <= 0 or work.iloc[:, 1].std() <= 0:
        return np.nan
    return float(work.iloc[:, 0].corr(work.iloc[:, 1], method="spearman"))


def _global_correlations(groups: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = []
    for kind in groups.columns:
        rows.append(
            {
                "kind": kind,
                "display_name": DISPLAY_NAMES.get(kind, kind),
                "pearson_r": _pearson(groups[kind], y),
                "spearman_r": _spearman(groups[kind], y),
            }
        )
    return pd.DataFrame(rows).sort_values("kind").reset_index(drop=True)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    wsum = float(np.sum(weights))
    if wsum <= 0:
        return np.full(values.shape[1], np.nan)
    return np.sum(values * weights[:, None], axis=0) / wsum


def _weighted_standardize(values: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = _weighted_mean(values, weights)
    var = _weighted_mean((values - mean) ** 2, weights)
    std = np.sqrt(var)
    std_safe = np.where(std > 0, std, np.nan)
    return (values - mean) / std_safe, std_safe


def _weighted_univariate_importance(
    groups: pd.DataFrame,
    y: pd.Series,
    targets: Iterable[float],
    *,
    sigma: float,
) -> pd.DataFrame:
    x = groups.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y_arr = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(y_arr) & np.all(np.isfinite(x), axis=1)
    x = x[finite]
    y_arr = y_arr[finite]
    cols = list(groups.columns)
    rows: List[Dict[str, object]] = []
    if len(y_arr) < 3:
        return pd.DataFrame()
    for target in targets:
        w = np.exp(-0.5 * ((y_arr - float(target)) / float(sigma)) ** 2)
        w = np.where(np.isfinite(w), w, 0.0)
        if w.sum() <= 0:
            continue
        ess = float((w.sum() ** 2) / max(np.sum(w ** 2), 1e-12))
        x_std, x_scale = _weighted_standardize(x, w)
        y_std = _weighted_standardize(y_arr.reshape(-1, 1), w)[0].reshape(-1)
        beta_abs = []
        for idx, col in enumerate(cols):
            if not np.isfinite(x_scale[idx]) or x_scale[idx] <= 0:
                beta_abs.append(np.nan)
                continue
            denom = float(np.sum(w * x_std[:, idx] * x_std[:, idx]))
            if denom <= 0:
                beta_abs.append(np.nan)
                continue
            beta = float(np.sum(w * x_std[:, idx] * y_std) / denom)
            beta_abs.append(abs(beta))
        beta_arr = np.asarray(beta_abs, dtype=float)
        total = np.nansum(beta_arr)
        for col, beta in zip(cols, beta_arr):
            rows.append(
                {
                    "target_emission_gt": float(target),
                    "kind": col,
                    "display_name": DISPLAY_NAMES.get(col, col),
                    "univariate_abs_beta": float(beta) if np.isfinite(beta) else np.nan,
                    "univariate_share": float(beta / total) if total > 0 and np.isfinite(beta) else np.nan,
                    "effective_n_samples": ess,
                }
            )
    return pd.DataFrame(rows)


def _load_multivariate_importance(summary_dir: Path) -> pd.DataFrame:
    path = summary_dir / "importance_by_variable.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = {"target_emission_gt", "parameter", "importance"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    df = df.copy()
    df["kind"] = df["parameter"].map(_canonical_kind)
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce")
    return df[["target_emission_gt", "kind", "importance", *[c for c in ("effective_n_samples", "weight_mode") if c in df.columns]]]


def _target_compare(univar: pd.DataFrame, multi: pd.DataFrame) -> pd.DataFrame:
    if univar.empty or multi.empty:
        return pd.DataFrame()
    m = multi.rename(columns={"importance": "multivariate_share"}).copy()
    out = univar.merge(
        m,
        on=["target_emission_gt", "kind"],
        how="left",
        suffixes=("_univariate", "_multivariate"),
    )
    if "effective_n_samples_univariate" in out.columns:
        out["effective_n_samples"] = out["effective_n_samples_univariate"]
    elif "effective_n_samples" not in out.columns and "effective_n_samples_multivariate" in out.columns:
        out["effective_n_samples"] = out["effective_n_samples_multivariate"]
    out["absorption_ratio_univ_over_multi"] = out["univariate_share"] / out["multivariate_share"].replace(0.0, np.nan)
    out["is_small_kind"] = out["kind"].isin(SMALL_KINDS)
    return out.sort_values(["target_emission_gt", "kind"]).reset_index(drop=True)


def _max_abs_correlations(groups: pd.DataFrame) -> pd.DataFrame:
    corr = groups.corr(method="pearson")
    rows = []
    for kind in groups.columns:
        refs = [r for r in LARGE_REFERENCE_KINDS if r in corr.columns and r != kind]
        if not refs:
            rows.append({"kind": kind, "max_abs_corr_with_large": np.nan, "max_corr_large_kind": ""})
            continue
        vals = corr.loc[kind, refs].abs().dropna()
        if vals.empty:
            rows.append({"kind": kind, "max_abs_corr_with_large": np.nan, "max_corr_large_kind": ""})
        else:
            max_kind = str(vals.idxmax())
            rows.append(
                {
                    "kind": kind,
                    "display_name": DISPLAY_NAMES.get(kind, kind),
                    "max_abs_corr_with_large": float(vals.loc[max_kind]),
                    "max_corr_large_kind": max_kind,
                }
            )
    return pd.DataFrame(rows).sort_values("kind").reset_index(drop=True)


def _nearest_target_rows(df: pd.DataFrame, target: float) -> pd.DataFrame:
    if df.empty or "target_emission_gt" not in df.columns:
        return pd.DataFrame()
    vals = pd.to_numeric(df["target_emission_gt"], errors="coerce")
    idx_target = float(vals.iloc[(vals - float(target)).abs().argsort().iloc[0]])
    return df[np.isclose(vals, idx_target)].copy()


def _write_markdown(
    path: Path,
    *,
    spec_summary: pd.DataFrame,
    sample_summary: pd.DataFrame,
    global_corr: pd.DataFrame,
    target_compare: pd.DataFrame,
    max_corr: pd.DataFrame,
    target_current: float,
) -> None:
    lines: List[str] = []
    lines.append("# S5_0_1 Small Variable Importance Diagnostics")
    lines.append("")
    lines.append("## Bottom Line")
    lines.append("")
    lines.append("- This diagnostic checks whether the small variables are present in MC specs, present and nonconstant in `samples.csv`, and how their univariate weighted signal compares with S5_0_1 multivariate regression importance.")
    lines.append("- A high univariate share but low multivariate share indicates possible attribution absorption by correlated or stronger variables; a low univariate share means the sampled model response itself is weak.")
    lines.append("")
    lines.append("## Code Path Evidence")
    lines.append("")
    lines.append("- `S3_6_scenarios.py` maps `feed_intensity`, `fertilizer_rate`, `manure_management_ratio`, `crop_soil_management_ratio`, and `land_carbon_price` into `scenario_ctx`.")
    lines.append("- `S4_0_main.py` applies feed, fertilizer, and manure multipliers to production/emission input tables, and passes `land_carbon_price_by_year` into the linear land/LUC solver.")
    lines.append("- `gce_emissions_complete.py` and `gsoil_emission_complete.py` apply `crop_soil_management_multiplier`; `gle_emissions_complete.py` applies manure management adjustments.")
    lines.append("")
    lines.append("## Small Variable Coverage")
    lines.append("")
    small_sample = sample_summary[sample_summary["kind"].isin(SMALL_KINDS)].copy()
    if small_sample.empty:
        lines.append("No small variable columns were found in `samples.csv`.")
    else:
        lines.append(small_sample.to_markdown(index=False))
    lines.append("")
    lines.append("## Spec Rows")
    lines.append("")
    small_specs = spec_summary[spec_summary["kind"].isin(SMALL_KINDS)].copy()
    lines.append(small_specs.to_markdown(index=False) if not small_specs.empty else "No matching spec rows found.")
    lines.append("")
    lines.append(f"## Nearest Target To {target_current:g} Gt")
    lines.append("")
    cur = _nearest_target_rows(target_compare[target_compare["kind"].isin(SMALL_KINDS)], target_current)
    cols = [
        "target_emission_gt",
        "kind",
        "univariate_share",
        "multivariate_share",
        "absorption_ratio_univ_over_multi",
        "effective_n_samples",
    ]
    cur_cols = [c for c in cols if c in cur.columns]
    lines.append(cur[cur_cols].to_markdown(index=False) if not cur.empty else "No target comparison available.")
    lines.append("")
    lines.append("## Global Correlation With Emissions")
    lines.append("")
    small_corr = global_corr[global_corr["kind"].isin(SMALL_KINDS)].copy()
    lines.append(small_corr.to_markdown(index=False) if not small_corr.empty else "No correlation table available.")
    lines.append("")
    lines.append("## Correlation With Larger Variables")
    lines.append("")
    small_max_corr = max_corr[max_corr["kind"].isin(SMALL_KINDS)].copy()
    lines.append(small_max_corr.to_markdown(index=False) if not small_max_corr.empty else "No correlation matrix available.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _build_arg_parser().parse_args()
    summary_dir = _summary_dir(args)
    out_dir = _out_dir(args)
    scenario_config = _scenario_config_path(args)

    samples = _load_samples(summary_dir)
    y = pd.to_numeric(samples["emissions_2080_gt"], errors="coerce")
    by_kind = _parameter_columns_by_kind(samples)
    groups = _group_matrix(samples, by_kind)

    spec_summary = _load_spec_summary(scenario_config, args.mc_sheet)
    sample_summary = _sample_group_summary(samples, by_kind, groups)
    global_corr = _global_correlations(groups, y)
    max_corr = _max_abs_correlations(groups)

    multi = _load_multivariate_importance(summary_dir)
    if multi.empty:
        targets = np.array([args.target_current], dtype=float)
    else:
        targets = np.sort(pd.to_numeric(multi["target_emission_gt"], errors="coerce").dropna().unique())
    univar = _weighted_univariate_importance(groups, y, targets, sigma=float(args.sigma))
    target_compare = _target_compare(univar, multi)

    workbook_path = out_dir / "S5_0_1_small_variable_importance_diagnostics.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        spec_summary.to_excel(writer, sheet_name="spec_summary", index=False)
        sample_summary.to_excel(writer, sheet_name="sample_group_summary", index=False)
        global_corr.to_excel(writer, sheet_name="global_correlation", index=False)
        max_corr.to_excel(writer, sheet_name="max_abs_corr_large", index=False)
        univar.to_excel(writer, sheet_name="univariate_weighted", index=False)
        target_compare.to_excel(writer, sheet_name="target_compare", index=False)

    md_path = out_dir / "S5_0_1_small_variable_importance_diagnostics.md"
    _write_markdown(
        md_path,
        spec_summary=spec_summary,
        sample_summary=sample_summary,
        global_corr=global_corr,
        target_compare=target_compare,
        max_corr=max_corr,
        target_current=float(args.target_current),
    )

    print(f"[DONE] workbook: {workbook_path}")
    print(f"[DONE] report: {md_path}")


if __name__ == "__main__":
    main()
