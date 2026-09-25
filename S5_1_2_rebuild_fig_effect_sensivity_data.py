from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config_paths import get_results_base


CONFIG = {
    "input_root": "",
    "fig5_dir": "",
    "year": 2080,
    "unit_scale": 1e-6,  # kt -> Gt
    "n_bins": 120,
    "targets": [
        ("1.5D", 0.9),
        ("2D", 4.2),
        ("Current", 12.6),
        ("RCP4.5", 14.5),
    ],
    "rebuilt_root_samples_csv": "samples_rebuilt_from_runs.csv",
    "rebuilt_fig5_samples_csv": "samples_rebuilt_from_runs.csv",
    "rebuilt_histogram_csv": "histogram_rebuilt_from_runs.csv",
    "rebuilt_summary_csv": "summary_rebuilt_from_runs.csv",
    "rebuilt_targets_csv": "targets_rebuilt_from_runs.csv",
    "audit_csv": "fig5_rebuild_audit.csv",
    # If the legacy files are missing, also write them under the exact names
    # expected by SP1_5_Figure_yield_ef_effect_line*.py.
    "write_legacy_names_if_missing": True,
}


SCENARIO_PATTERN = re.compile(
    r"^VE_(?P<variable>.+)_(?P<level_tag>[mp]\d+)_(?P<sample_id>\d+)$",
    flags=re.IGNORECASE,
)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _input_root() -> Path:
    if CONFIG.get("input_root"):
        return Path(str(CONFIG["input_root"]))
    return Path(get_results_base()) / "MC_Sensitivity_Variable_Effect"


def _runs_dir() -> Path:
    return _input_root() / "runs"


def _fig5_dir() -> Path:
    if CONFIG.get("fig5_dir"):
        return Path(str(CONFIG["fig5_dir"]))
    return Path(get_results_base()) / "Plot" / "Fig5"


def _rate_from_level_tag(level_tag: str) -> float:
    tag = str(level_tag or "").strip().lower()
    if not tag or len(tag) < 2:
        raise ValueError(f"Invalid level_tag: {level_tag}")
    sign = 1.0 if tag[0] == "p" else -1.0
    pct = float(tag[1:])
    return sign * pct / 100.0


def _fig5_key_label(kind: str, rate_value: float) -> Tuple[str, str, str]:
    kind_l = str(kind or "").strip().lower()
    pct = int(round(abs(rate_value) * 100))
    direction = "up" if rate_value > 0 else "down"

    if kind_l == "yield_rate":
        if abs(rate_value) < 1e-9:
            return "yield", "yield_current", "Yield rate current"
        sign = "+" if rate_value > 0 else "-"
        return "yield", f"yield_{direction}_{pct}", f"Yield rate {sign}{pct}%"

    if kind_l == "emission_factor":
        if abs(rate_value) < 1e-9:
            return "emission_factor", "ef_current", "Emission factor current"
        sign = "+" if rate_value > 0 else "-"
        return "emission_factor", f"ef_{direction}_{pct}", f"Emission factor {sign}{pct}%"

    # Keep panel/key names compatible with SP1_5_Figure_yield_ef_effect_line_v2.py.
    if kind_l == "ruminant_reduction":
        if abs(rate_value) < 1e-9:
            return "ruminate_intake", "ruminate_intake_current", "Ruminate intake current"
        sign = "+" if rate_value > 0 else "-"
        return (
            "ruminate_intake",
            f"ruminate_intake_{direction}_{pct}",
            f"Ruminate intake {sign}{pct}%",
        )

    return "", "", ""


def _read_global_emissions_2080_gt(
    run_dir: Path,
    *,
    year: int,
    unit_scale: float,
) -> Tuple[float, str, str]:
    emis_dir = run_dir / "Emis"
    fast_csv = emis_dir / "emissions_fast_summary.csv"
    if fast_csv.exists():
        try:
            df = pd.read_csv(fast_csv)
        except Exception as exc:
            return float("nan"), "fast_csv_error", f"failed to read fast csv: {exc}"

        if df.empty:
            return float("nan"), "fast_csv_empty", "emissions_fast_summary.csv is empty"

        cols = [str(c).strip() for c in df.columns]
        df.columns = cols
        if "year" in df.columns:
            year_series = pd.to_numeric(df["year"], errors="coerce")
            df = df.loc[year_series == int(year)].copy()
        if df.empty:
            return float("nan"), "fast_csv_missing_year", f"target year {year} not found"
        if "total_co2eq_gt" in df.columns:
            val = pd.to_numeric(df["total_co2eq_gt"], errors="coerce").sum()
            if pd.notna(val):
                return float(val), "fast_csv_gt", ""
        if "total_co2eq_kt" in df.columns:
            val = pd.to_numeric(df["total_co2eq_kt"], errors="coerce").sum()
            if pd.notna(val):
                return float(val) * float(unit_scale), "fast_csv_kt", ""
        return float("nan"), "fast_csv_missing_total", "missing total_co2eq_gt/kt column"

    by_country_csv = emis_dir / "emissions_summary_By_Country.csv"
    if by_country_csv.exists():
        try:
            df = pd.read_csv(by_country_csv)
        except Exception as exc:
            return float("nan"), "by_country_csv_error", f"failed to read emissions_summary_By_Country.csv: {exc}"
        if df.empty:
            return float("nan"), "by_country_csv_empty", "emissions_summary_By_Country.csv is empty"
        if "GHG" in df.columns:
            df = df[df["GHG"].astype(str).str.upper() == "CO2EQ"]
        if "Region_label_new" in df.columns:
            df = df[df["Region_label_new"].astype(str).str.lower() == "global"]
        elif "M49_Country_Code" in df.columns:
            df = df[df["M49_Country_Code"].astype(str).str.contains("000", na=False)]
        ycol = f"Y{int(year)}"
        if ycol not in df.columns:
            return float("nan"), "by_country_csv_missing_year", f"{ycol} not found in emissions_summary_By_Country.csv"
        val = pd.to_numeric(df[ycol], errors="coerce").sum()
        if pd.notna(val):
            return float(val) * float(unit_scale), "by_country_csv", ""
        return float("nan"), "by_country_csv_invalid", f"{ycol} is non-numeric"

    by_country_xlsx = emis_dir / "emissions_summary_By_Country.xlsx"
    if by_country_xlsx.exists():
        try:
            df = pd.read_excel(by_country_xlsx)
        except Exception as exc:
            return float("nan"), "by_country_xlsx_error", f"failed to read emissions_summary_By_Country.xlsx: {exc}"
        if df.empty:
            return float("nan"), "by_country_xlsx_empty", "emissions_summary_By_Country.xlsx is empty"
        if "GHG" in df.columns:
            df = df[df["GHG"].astype(str).str.upper() == "CO2EQ"]
        if "Region_label_new" in df.columns:
            df = df[df["Region_label_new"].astype(str).str.lower() == "global"]
        elif "M49_Country_Code" in df.columns:
            df = df[df["M49_Country_Code"].astype(str).str.contains("000", na=False)]
        ycol = f"Y{int(year)}"
        if ycol not in df.columns:
            return float("nan"), "by_country_xlsx_missing_year", f"{ycol} not found in emissions_summary_By_Country.xlsx"
        val = pd.to_numeric(df[ycol], errors="coerce").sum()
        if pd.notna(val):
            return float(val) * float(unit_scale), "by_country_xlsx", ""
        return float("nan"), "by_country_xlsx_invalid", f"{ycol} is non-numeric"

    emis_xlsx = emis_dir / "emissions_summary.xlsx"
    if emis_xlsx.exists():
        try:
            df = pd.read_excel(emis_xlsx, sheet_name="By_Country")
        except Exception as exc:
            return float("nan"), "summary_xlsx_error", f"failed to read emissions_summary.xlsx: {exc}"
        if df.empty:
            return float("nan"), "summary_xlsx_empty", "emissions_summary.xlsx By_Country is empty"
        if "GHG" in df.columns:
            df = df[df["GHG"].astype(str).str.upper() == "CO2EQ"]
        if "Region_label_new" in df.columns:
            df = df[df["Region_label_new"].astype(str).str.lower() == "global"]
        elif "M49_Country_Code" in df.columns:
            df = df[df["M49_Country_Code"].astype(str).str.contains("000", na=False)]
        ycol = f"Y{int(year)}"
        if ycol not in df.columns:
            return float("nan"), "summary_xlsx_missing_year", f"{ycol} not found in emissions_summary.xlsx"
        val = pd.to_numeric(df[ycol], errors="coerce").sum()
        if pd.notna(val):
            return float(val) * float(unit_scale), "summary_xlsx", ""
        return float("nan"), "summary_xlsx_invalid", f"{ycol} is non-numeric"

    return float("nan"), "missing_emis", (
        "Emis/emissions_fast_summary.csv, emissions_summary_By_Country.csv, "
        "emissions_summary_By_Country.xlsx, and emissions_summary.xlsx are all missing"
    )


def _build_histogram_and_summary(
    fig5_samples_df: pd.DataFrame,
    *,
    year: int,
    n_bins: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    observed = fig5_samples_df.copy()
    observed["emissions_2080_gt"] = pd.to_numeric(observed["emissions_2080_gt"], errors="coerce")
    observed = observed.dropna(subset=["emissions_2080_gt"]).copy()

    if observed.empty:
        return pd.DataFrame(), pd.DataFrame()

    x_min = float(observed["emissions_2080_gt"].min())
    x_max = float(observed["emissions_2080_gt"].max())
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        return pd.DataFrame(), pd.DataFrame()
    if abs(x_max - x_min) < 1e-12:
        x_min -= 0.5
        x_max += 0.5

    hist_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for (panel, scenario_key), sub in observed.groupby(["panel", "scenario_key"], dropna=False):
        vals = sub["emissions_2080_gt"].to_numpy(dtype=float)
        label = str(sub["scenario_label"].iloc[0])
        counts, edges = np.histogram(vals, bins=int(n_bins), range=(x_min, x_max))
        for j in range(len(counts)):
            hist_rows.append(
                {
                    "panel": str(panel),
                    "scenario_key": str(scenario_key),
                    "scenario_label": label,
                    "year": int(year),
                    "bin_left": float(edges[j]),
                    "bin_right": float(edges[j + 1]),
                    "count": int(counts[j]),
                }
            )

        summary_rows.append(
            {
                "panel": str(panel),
                "scenario_key": str(scenario_key),
                "scenario_label": label,
                "year": int(year),
                "n_total_runs": int(len(fig5_samples_df[
                    (fig5_samples_df["panel"] == panel) & (fig5_samples_df["scenario_key"] == scenario_key)
                ])),
                "n_with_emis": int(len(vals)),
                "n_missing_emis": int(len(fig5_samples_df[
                    (fig5_samples_df["panel"] == panel) & (fig5_samples_df["scenario_key"] == scenario_key)
                ]) - len(vals)),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=0)),
                "p05": float(np.percentile(vals, 5)),
                "p50": float(np.percentile(vals, 50)),
                "p95": float(np.percentile(vals, 95)),
            }
        )

    return pd.DataFrame(hist_rows), pd.DataFrame(summary_rows)


def main() -> None:
    cfg = CONFIG
    root = _input_root()
    runs_dir = _runs_dir()
    fig5_dir = _fig5_dir()

    if not runs_dir.exists():
        raise FileNotFoundError(f"runs directory not found: {runs_dir}")

    _ensure_dir(fig5_dir)

    year = int(cfg.get("year", 2080))
    unit_scale = float(cfg.get("unit_scale", 1e-6))
    n_bins = int(cfg.get("n_bins", 120))

    root_rows: List[Dict[str, object]] = []
    fig5_rows: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []

    scenario_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("VE_")])
    for scenario_dir in scenario_dirs:
        match = SCENARIO_PATTERN.match(scenario_dir.name)
        if not match:
            audit_rows.append(
                {
                    "scenario_id": scenario_dir.name,
                    "parse_status": "unmatched_name",
                    "variable": "",
                    "level_tag": "",
                    "sample_id": np.nan,
                    "rate_value": np.nan,
                    "run_dir": str(scenario_dir),
                    "emissions_status": "skipped",
                    "emissions_source": "",
                    "error_message": "scenario id does not match VE_<variable>_<level>_<sample>",
                }
            )
            continue

        variable = str(match.group("variable"))
        level_tag = str(match.group("level_tag")).lower()
        sample_id = int(match.group("sample_id"))
        rate_value = _rate_from_level_tag(level_tag)
        scenario_id = scenario_dir.name

        emissions_gt, source_type, err = _read_global_emissions_2080_gt(
            scenario_dir,
            year=year,
            unit_scale=unit_scale,
        )
        emissions_status = "ok" if np.isfinite(emissions_gt) else "missing_emis"

        panel, scenario_key, scenario_label = _fig5_key_label(variable, rate_value)
        if panel:
            fig5_rows.append(
                {
                    "panel": panel,
                    "scenario_key": scenario_key,
                    "scenario_label": scenario_label,
                    "sample_id": sample_id,
                    "year": year,
                    "emissions_2080_gt": emissions_gt if np.isfinite(emissions_gt) else np.nan,
                    "scenario_id": scenario_id,
                    "variable": variable,
                    "rate_value": rate_value,
                    "level_tag": level_tag,
                    "emissions_status": emissions_status,
                    "emissions_source": source_type,
                }
            )

        root_rows.append(
            {
                "variable": variable,
                "rate_value": rate_value,
                "level_tag": level_tag,
                "sample_id": sample_id,
                "scenario_id": scenario_id,
                "emissions_2080_gt": emissions_gt if np.isfinite(emissions_gt) else np.nan,
                "emissions_status": emissions_status,
                "emissions_source": source_type,
                "run_dir": str(scenario_dir),
                "error_message": err,
            }
        )

        audit_rows.append(
            {
                "scenario_id": scenario_id,
                "parse_status": "ok",
                "variable": variable,
                "level_tag": level_tag,
                "sample_id": sample_id,
                "rate_value": rate_value,
                "run_dir": str(scenario_dir),
                "emissions_status": emissions_status,
                "emissions_source": source_type,
                "error_message": err,
            }
        )

    root_cols = [
        "variable",
        "rate_value",
        "level_tag",
        "sample_id",
        "scenario_id",
        "emissions_2080_gt",
        "emissions_status",
        "emissions_source",
        "run_dir",
        "error_message",
    ]
    fig5_cols = [
        "panel",
        "scenario_key",
        "scenario_label",
        "sample_id",
        "year",
        "emissions_2080_gt",
        "scenario_id",
        "variable",
        "rate_value",
        "level_tag",
        "emissions_status",
        "emissions_source",
    ]
    audit_cols = [
        "scenario_id",
        "parse_status",
        "variable",
        "level_tag",
        "sample_id",
        "rate_value",
        "run_dir",
        "emissions_status",
        "emissions_source",
        "error_message",
    ]

    root_df = pd.DataFrame(root_rows)
    if root_df.empty:
        root_df = pd.DataFrame(columns=root_cols)
    else:
        root_df = root_df.sort_values(["variable", "rate_value", "sample_id"], kind="stable").reset_index(drop=True)

    fig5_df = pd.DataFrame(fig5_rows)
    if fig5_df.empty:
        fig5_df = pd.DataFrame(columns=fig5_cols)
    else:
        fig5_df = fig5_df.sort_values(["panel", "scenario_key", "sample_id"], kind="stable").reset_index(drop=True)

    audit_df = pd.DataFrame(audit_rows)
    if audit_df.empty:
        audit_df = pd.DataFrame(columns=audit_cols)
    else:
        audit_df = audit_df.sort_values(["parse_status", "variable", "rate_value", "sample_id"], kind="stable").reset_index(drop=True)

    hist_df, summary_df = _build_histogram_and_summary(fig5_df, year=year, n_bins=n_bins)
    targets_df = pd.DataFrame(
        [{"label": str(label), "emission_gt": float(val)} for label, val in (cfg.get("targets") or [])]
    )

    rebuilt_root_samples = root / str(cfg.get("rebuilt_root_samples_csv") or "samples_rebuilt_from_runs.csv")
    rebuilt_fig5_samples = fig5_dir / str(cfg.get("rebuilt_fig5_samples_csv") or "samples_rebuilt_from_runs.csv")
    rebuilt_hist = fig5_dir / str(cfg.get("rebuilt_histogram_csv") or "histogram_rebuilt_from_runs.csv")
    rebuilt_summary = fig5_dir / str(cfg.get("rebuilt_summary_csv") or "summary_rebuilt_from_runs.csv")
    rebuilt_targets = fig5_dir / str(cfg.get("rebuilt_targets_csv") or "targets_rebuilt_from_runs.csv")
    audit_path = fig5_dir / str(cfg.get("audit_csv") or "fig5_rebuild_audit.csv")

    root_df.to_csv(rebuilt_root_samples, index=False, encoding="utf-8-sig")
    fig5_df.to_csv(rebuilt_fig5_samples, index=False, encoding="utf-8-sig")
    hist_df.to_csv(rebuilt_hist, index=False, encoding="utf-8-sig")
    summary_df.to_csv(rebuilt_summary, index=False, encoding="utf-8-sig")
    targets_df.to_csv(rebuilt_targets, index=False, encoding="utf-8-sig")
    audit_df.to_csv(audit_path, index=False, encoding="utf-8-sig")

    if bool(cfg.get("write_legacy_names_if_missing", True)):
        legacy_root_samples = root / "samples.csv"
        legacy_fig5_samples = fig5_dir / "samples.csv"
        legacy_hist = fig5_dir / "histogram.csv"
        legacy_summary = fig5_dir / "summary.csv"
        legacy_targets = fig5_dir / "targets.csv"

        if not legacy_root_samples.exists():
            root_df.to_csv(legacy_root_samples, index=False, encoding="utf-8-sig")
        if not legacy_fig5_samples.exists():
            fig5_df.to_csv(legacy_fig5_samples, index=False, encoding="utf-8-sig")
        if not legacy_hist.exists():
            hist_df.to_csv(legacy_hist, index=False, encoding="utf-8-sig")
        if not legacy_summary.exists():
            summary_df.to_csv(legacy_summary, index=False, encoding="utf-8-sig")
        if not legacy_targets.exists():
            targets_df.to_csv(legacy_targets, index=False, encoding="utf-8-sig")

    n_total = int(len(audit_df))
    n_ok = int((audit_df["emissions_status"] == "ok").sum()) if not audit_df.empty else 0
    n_missing = int((audit_df["emissions_status"] != "ok").sum()) if not audit_df.empty else 0

    print(f"[DONE] rebuilt root samples: {rebuilt_root_samples}")
    print(f"[DONE] rebuilt fig5 samples: {rebuilt_fig5_samples}")
    print(f"[DONE] rebuilt histogram: {rebuilt_hist}")
    print(f"[DONE] rebuilt summary: {rebuilt_summary}")
    print(f"[DONE] rebuilt targets: {rebuilt_targets}")
    print(f"[DONE] rebuild audit: {audit_path}")
    print(f"[SUMMARY] total_runs={n_total} ok_with_emis={n_ok} missing_emis={n_missing}")


if __name__ == "__main__":
    main()
