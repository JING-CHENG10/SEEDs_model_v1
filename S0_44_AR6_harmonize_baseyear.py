import re
from pathlib import Path

import numpy as np
import pandas as pd
from config_paths import get_results_base


# User settings


# Scenario workbook (contains 1.5D and 2D sheets, and usually His as well)
FIG6_DIR = Path(get_results_base()) / "Plot" / "Fig6"
# INPUT_XLSX = FIG6_DIR / "AR6_scenario_prepared.xlsx"
INPUT_XLSX = FIG6_DIR / "SCI" / "SCI_Database.xlsx"

# Historical target workbook/sheet
HIS_XLSX = INPUT_XLSX
HIS_SHEET = "His"

# Dual anchors: both years are forced to historical targets
ANCHOR_YEAR = 2010
HARMONIZE_YEAR = 2020

# Harmonization effect decays to zero at this year (for years > HARMONIZE_YEAR)
CONVERGE_YEAR = 2100

# Harmonization method for non-share variables by default.
# Options: reduce_ratio, reduce_offset, constant_ratio, constant_offset, auto
# auto: share -> reduce_offset; otherwise reduce_ratio (fallback to offset if ratio invalid)
HARMONIZATION_METHOD = "auto"

# When the model value at harmonize year is small relative to the target
# and the post-harmonize trajectory turns negative, ratio scaling can
# over-amplify large negative values. Force these rows to reduce_offset.
FORCE_OFFSET_FOR_SMALL_BASE_NEGATIVE_FUTURE = True
SMALL_BASE_RELATIVE_TO_TARGET_THRESHOLD = 0.25

# Clip share-like variables to [0, 1]
CLIP_SHARE_TO_01 = True

# Only process these scenario sheets
SHEETS_TO_PROCESS = ["1.5D", "2D"]

# Fallback when His target is missing:
# use trimmed mean of scenario values pooled across all processed scenario sheets
# at that year (drop min/max if sample size >= MIN_COUNT_FOR_TRIM)
USE_TRIMMED_MEAN_FALLBACK = True
MIN_COUNT_FOR_TRIM = 3



# Utilities


def _year_cols(df: pd.DataFrame):
    cols = [c for c in df.columns if re.fullmatch(r"Y\d{4}", str(c))]
    years = [int(str(c)[1:]) for c in cols]
    order = np.argsort(years)
    cols = [cols[i] for i in order]
    years = [years[i] for i in order]
    return cols, years


def _beta(year: int, base_year: int, converge_year: int) -> float:
    """Decay factor for post-base years: base_year->1, converge_year->0, after->0."""
    if converge_year == base_year:
        return 0.0
    if year <= base_year:
        return 1.0
    if year <= converge_year:
        return 1.0 - (year - base_year) / (converge_year - base_year)
    return 0.0


def _choose_method_per_row(df: pd.DataFrame, default_method: str):
    """Return per-row method vector and share-mask."""
    is_share = (
        df["Unit"].astype(str).str.lower().eq("share")
        | df["Variable"].astype(str).str.contains("share", case=False, regex=False)
    )

    if default_method != "auto":
        method = np.full(len(df), default_method, dtype=object)
        return method, is_share

    method = np.where(is_share, "reduce_offset", "reduce_ratio").astype(object)

    bad_ratio = ~np.isfinite(df["ratio"]) | (df["ratio"] <= 0)
    if "ratio_anchor" in df.columns:
        bad_ratio = bad_ratio | ~np.isfinite(df["ratio_anchor"]) | (df["ratio_anchor"] <= 0)
    method = np.where((method == "reduce_ratio") & bad_ratio, "reduce_offset", method)

    return method, is_share


def _read_his_targets(
    his_xlsx: Path,
    his_sheet: str,
    anchor_year: int,
    harmonize_year: int,
) -> pd.DataFrame:
    """Read His sheet to target_{anchor_year} and target_{harmonize_year}."""
    dfh = pd.read_excel(his_xlsx, sheet_name=his_sheet)

    ycol_anchor = f"Y{anchor_year}"
    ycol_base = f"Y{harmonize_year}"

    if "Variable" not in dfh.columns or "Unit" not in dfh.columns:
        raise ValueError("His sheet must include columns: Variable, Unit")
    for ycol in (ycol_anchor, ycol_base):
        if ycol not in dfh.columns:
            raise ValueError(f"His sheet missing required column: {ycol}")

    target_anchor_col = f"target_{anchor_year}"
    target_base_col = f"target_{harmonize_year}"
    out = dfh[["Variable", "Unit", ycol_anchor, ycol_base]].copy()
    out = out.rename(columns={ycol_anchor: target_anchor_col, ycol_base: target_base_col})
    out[target_anchor_col] = pd.to_numeric(out[target_anchor_col], errors="coerce")
    out[target_base_col] = pd.to_numeric(out[target_base_col], errors="coerce")
    out = out[out["Variable"].notna()].copy()
    return out


def _build_fallback_targets_from_scenario(
    df: pd.DataFrame,
    year: int,
    out_col: str,
) -> pd.DataFrame:
    """Build fallback targets by (Region, Variable, Unit) from scenario values at a given year."""
    base_col = f"Y{year}"
    if base_col not in df.columns:
        raise ValueError(f"Scenario sheet missing column: {base_col}")

    group_cols = ["Region", "Variable", "Unit"]
    d = df[group_cols + [base_col]].copy()
    d[base_col] = pd.to_numeric(d[base_col], errors="coerce")

    def _agg(vals: pd.Series) -> float:
        arr = vals.dropna().to_numpy(dtype=float)
        if arr.size == 0:
            return np.nan
        if USE_TRIMMED_MEAN_FALLBACK and arr.size >= MIN_COUNT_FOR_TRIM:
            arr = np.sort(arr)[1:-1]
            if arr.size == 0:
                return float(np.mean(vals.dropna().to_numpy(dtype=float)))
            return float(np.mean(arr))
        return float(np.mean(arr))

    fb = (
        d.groupby(group_cols, as_index=False)[base_col]
        .agg(_agg)
        .rename(columns={base_col: out_col})
    )
    return fb


def _combine_targets(
    df_scenario: pd.DataFrame,
    his_targets: pd.DataFrame,
    anchor_year: int,
    harmonize_year: int,
    fallback_anchor: pd.DataFrame | None = None,
    fallback_base: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combine His targets (preferred) with scenario fallback for both anchors."""
    target_anchor_col = f"target_{anchor_year}"
    target_base_col = f"target_{harmonize_year}"

    fb_anchor_col = f"{target_anchor_col}_fallback"
    fb_base_col = f"{target_base_col}_fallback"

    source_anchor_col = f"target_source_{anchor_year}"
    source_base_col = f"target_source_{harmonize_year}"

    if fallback_anchor is None:
        fb_anchor = _build_fallback_targets_from_scenario(df_scenario, anchor_year, fb_anchor_col)
        anchor_source_label = "trimmed_mean"
    else:
        fb_anchor = fallback_anchor.rename(
            columns={f"target_{anchor_year}_fallback": fb_anchor_col}
        ).copy()
        anchor_source_label = "pooled_trimmed_mean"

    if fallback_base is None:
        fb_base = _build_fallback_targets_from_scenario(df_scenario, harmonize_year, fb_base_col)
        base_source_label = "trimmed_mean"
    else:
        fb_base = fallback_base.rename(
            columns={f"target_{harmonize_year}_fallback": fb_base_col}
        ).copy()
        base_source_label = "pooled_trimmed_mean"

    base = df_scenario[["Region", "Variable", "Unit"]].drop_duplicates().copy()
    m = base.merge(
        his_targets[["Variable", "Unit", target_anchor_col, target_base_col]],
        on=["Variable", "Unit"],
        how="left",
    )
    m = m.merge(fb_anchor, on=["Region", "Variable", "Unit"], how="left")
    m = m.merge(fb_base, on=["Region", "Variable", "Unit"], how="left")

    m[source_anchor_col] = np.select(
        [m[target_anchor_col].notna(), m[fb_anchor_col].notna()],
        ["His", anchor_source_label],
        default="missing",
    )
    m[source_base_col] = np.select(
        [m[target_base_col].notna(), m[fb_base_col].notna()],
        ["His", base_source_label],
        default="missing",
    )

    m[target_anchor_col] = np.where(m[target_anchor_col].notna(), m[target_anchor_col], m[fb_anchor_col])
    m[target_base_col] = np.where(m[target_base_col].notna(), m[target_base_col], m[fb_base_col])

    # Compatibility aliases (2020 anchor)
    m["target_base"] = m[target_base_col]
    m["target_source"] = m[source_base_col]

    return m[
        [
            "Region",
            "Variable",
            "Unit",
            target_anchor_col,
            target_base_col,
            source_anchor_col,
            source_base_col,
            "target_base",
            "target_source",
        ]
    ]


def harmonize_sheet(
    df_in: pd.DataFrame,
    targets: pd.DataFrame,
    anchor_year: int,
    harmonize_year: int,
    converge_year: int,
    method_default: str,
):
    anchor_col = f"Y{anchor_year}"
    base_col = f"Y{harmonize_year}"
    target_anchor_col = f"target_{anchor_year}"
    target_base_col = f"target_{harmonize_year}"

    year_cols, years = _year_cols(df_in)
    if anchor_col not in year_cols or base_col not in year_cols:
        raise ValueError(f"Missing anchor columns: {anchor_col}, {base_col}")

    df = df_in.merge(targets, on=["Region", "Variable", "Unit"], how="left")

    df["model_anchor"] = pd.to_numeric(df[anchor_col], errors="coerce")
    df["model_base"] = pd.to_numeric(df[base_col], errors="coerce")
    can_harmonize = (
        df[target_anchor_col].notna()
        & df[target_base_col].notna()
        & df["model_anchor"].notna()
        & df["model_base"].notna()
    )
    df["harmonize_status"] = np.select(
        [
            df[target_anchor_col].isna() | df[target_base_col].isna(),
            df["model_anchor"].isna() | df["model_base"].isna(),
        ],
        [
            "skip_missing_target",
            "skip_missing_model_anchor_or_base",
        ],
        default="harmonized",
    )

    df["offset_anchor"] = df[target_anchor_col].astype(float) - df["model_anchor"].astype(float)
    df["offset"] = df[target_base_col].astype(float) - df["model_base"].astype(float)

    df["ratio_anchor"] = np.where(
        df["model_anchor"] != 0,
        df[target_anchor_col].astype(float) / df["model_anchor"].astype(float),
        np.nan,
    )
    df["ratio"] = np.where(
        df["model_base"] != 0,
        df[target_base_col].astype(float) / df["model_base"].astype(float),
        np.nan,
    )

    method_vec, is_share = _choose_method_per_row(df, method_default)
    forced_method_reason = np.full(len(df), "", dtype=object)
    if FORCE_OFFSET_FOR_SMALL_BASE_NEGATIVE_FUTURE:
        future_cols = [col for col, y in zip(year_cols, years) if y > harmonize_year]
        if future_cols:
            future_vals = df_in[future_cols].apply(pd.to_numeric, errors="coerce")
            has_negative_future = future_vals.lt(0).any(axis=1).to_numpy(dtype=bool)
        else:
            has_negative_future = np.zeros(len(df), dtype=bool)

        model_base_abs = np.abs(df["model_base"].astype(float).to_numpy())
        target_base_abs = np.abs(df[target_base_col].astype(float).to_numpy())
        small_base_vs_target = (
            np.isfinite(model_base_abs)
            & np.isfinite(target_base_abs)
            & (target_base_abs > 0)
            & (model_base_abs <= SMALL_BASE_RELATIVE_TO_TARGET_THRESHOLD * target_base_abs)
        )
        force_offset_mask = (
            (method_vec == "reduce_ratio")
            & has_negative_future
            & small_base_vs_target
        )
        method_vec = np.where(force_offset_mask, "reduce_offset", method_vec)
        forced_method_reason = np.where(
            force_offset_mask,
            "small_base_negative_future",
            forced_method_reason,
        )
    method_vec = np.where(can_harmonize, method_vec, df["harmonize_status"]).astype(object)
    df_out = df_in.copy()

    ratio_anchor = df["ratio_anchor"].astype(float)
    ratio_base = df["ratio"].astype(float)
    offset_anchor = df["offset_anchor"].astype(float)
    offset_base = df["offset"].astype(float)
    can_harmonize_arr = can_harmonize.to_numpy()

    for col, y in zip(year_cols, years):
        m = pd.to_numeric(df_in[col], errors="coerce").astype(float)

        if y <= harmonize_year:
            if harmonize_year == anchor_year:
                w = 1.0
            else:
                w = (y - anchor_year) / (harmonize_year - anchor_year)
                w = min(1.0, max(0.0, w))

            ratio_w = ratio_anchor + w * (ratio_base - ratio_anchor)
            offset_w = offset_anchor + w * (offset_base - offset_anchor)
            out_reduce_ratio = m * ratio_w
            out_reduce_offset = m + offset_w
        else:
            b = _beta(y, harmonize_year, converge_year)
            out_reduce_ratio = m * (1.0 + b * (ratio_base - 1.0))
            out_reduce_offset = m + b * offset_base

        out_const_ratio = m * ratio_base
        out_const_offset = m + offset_base

        out = m.to_numpy(copy=True)
        out = np.where(method_vec == "reduce_ratio", out_reduce_ratio, out)
        out = np.where(method_vec == "reduce_offset", out_reduce_offset, out)
        out = np.where(method_vec == "constant_ratio", out_const_ratio, out)
        out = np.where(method_vec == "constant_offset", out_const_offset, out)

        if CLIP_SHARE_TO_01:
            out = np.where(is_share & can_harmonize_arr, np.clip(out, 0.0, 1.0), out)

        # Hard anchors only for rows with complete targets and valid model anchors.
        if y == anchor_year:
            out = np.where(can_harmonize_arr, df[target_anchor_col].astype(float).to_numpy(), out)
        elif y == harmonize_year:
            out = np.where(can_harmonize_arr, df[target_base_col].astype(float).to_numpy(), out)

        df_out[col] = out

    diagnostics_cols = [
        "Model",
        "Scenario",
        "Region",
        "Variable",
        "Unit",
        "model_anchor",
        "model_base",
        target_anchor_col,
        target_base_col,
        f"target_source_{anchor_year}",
        "target_source",
        "ratio_anchor",
        "ratio",
        "offset_anchor",
        "offset",
        "harmonize_status",
    ]
    diagnostics = df[diagnostics_cols].copy()
    diagnostics["forced_method_reason"] = forced_method_reason
    diagnostics["method"] = method_vec
    diagnostics["harmonize_year"] = harmonize_year
    diagnostics["converge_year"] = converge_year

    return df_out, diagnostics



# Main


def main() -> None:
    input_path = Path(INPUT_XLSX).resolve()
    his_path = Path(HIS_XLSX).resolve()
    out_path = input_path.with_name(f"{input_path.stem}_harmonization{input_path.suffix}")

    his_targets = _read_his_targets(
        his_xlsx=his_path,
        his_sheet=HIS_SHEET,
        anchor_year=ANCHOR_YEAR,
        harmonize_year=HARMONIZE_YEAR,
    )

    scenario_frames: dict[str, pd.DataFrame] = {
        sheet: pd.read_excel(input_path, sheet_name=sheet) for sheet in SHEETS_TO_PROCESS
    }
    scenario_pool = pd.concat(
        [df.copy() for df in scenario_frames.values()],
        ignore_index=True,
        sort=False,
    )
    fallback_anchor = _build_fallback_targets_from_scenario(
        scenario_pool,
        ANCHOR_YEAR,
        f"target_{ANCHOR_YEAR}_fallback",
    )
    fallback_base = _build_fallback_targets_from_scenario(
        scenario_pool,
        HARMONIZE_YEAR,
        f"target_{HARMONIZE_YEAR}_fallback",
    )

    outputs = []
    for sheet, df in scenario_frames.items():

        targets_final = _combine_targets(
            df_scenario=df,
            his_targets=his_targets,
            anchor_year=ANCHOR_YEAR,
            harmonize_year=HARMONIZE_YEAR,
            fallback_anchor=fallback_anchor,
            fallback_base=fallback_base,
        )

        df_h, diag = harmonize_sheet(
            df_in=df,
            targets=targets_final,
            anchor_year=ANCHOR_YEAR,
            harmonize_year=HARMONIZE_YEAR,
            converge_year=CONVERGE_YEAR,
            method_default=HARMONIZATION_METHOD,
        )

        outputs.append((sheet, df_h, targets_final, diag))

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for sheet, df_h, targets_final, diag in outputs:
            df_h.to_excel(writer, sheet_name=f"{sheet}_harmonized", index=False)
            targets_final.to_excel(writer, sheet_name=f"{sheet}_targets_final", index=False)
            diag.to_excel(writer, sheet_name=f"{sheet}_diagnostics", index=False)

    print("Saved:", out_path)


if __name__ == "__main__":
    main()
