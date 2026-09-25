# -*- coding: utf-8 -*-
"""S5.3 v2 panel data generator.

Compared with S5_3_1_sensitivity_panel_yield_ef.py, this version keeps the
same forest and ruminant-cap panel dimensions, but changes the two scan axes:

* Horizontal E/L axis:
  emission_factor, fertilizer_rate, and crop_soil_management_ratio move
  together; manure_management_ratio moves in the opposite direction by
  scenario meaning.
* Vertical L/A axis:
  yield_rate moves with the axis; feed_intensity moves in the opposite
  direction by scenario meaning.

The implementation delegates the heavy panel loop to the original S5.3 module
and only overrides the scenario effect builder, scenario id, default output
directory, and output metadata columns.  PALE outputs are inherited from the
base S5.3 runner; in v2, L is still reported as realized cropland + pasture.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

import S5_3_1_sensitivity_panel_yield_ef as base_panel
from S3_6_scenarios import ScenarioEffect


RUMINANT_CAP_COL = base_panel.RUMINANT_CAP_COL
RUMINANT_CAP_VALUES_KEY = base_panel.RUMINANT_CAP_VALUES_KEY
DEPRECATED_RUMINANT_CAP_COL = base_panel.DEPRECATED_RUMINANT_CAP_COL
DEPRECATED_RUMINANT_CAP_VALUES_KEY = base_panel.DEPRECATED_RUMINANT_CAP_VALUES_KEY
PALE_RESULT_COLS = base_panel.PALE_RESULT_COLS

EL_AXIS_LABEL = "E/L"
LA_AXIS_LABEL = "L/A"

EL_DIRECT_KINDS = (
    "emission_factor",
    "fertilizer_rate",
    "crop_soil_management_ratio",
)
EL_REVERSE_KINDS = ("manure_management_ratio",)
LA_DIRECT_KINDS = ("yield_rate",)
LA_REVERSE_KINDS = ("feed_intensity",)


def _axis_rate(pct: float, *, reverse: bool = False) -> float:
    val = -float(pct) if reverse else float(pct)
    return val / 100.0


def _make_effect(
    *,
    universe,
    scenario_id: str,
    kind: str,
    value_2080: object,
    unit: str = "rate",
) -> ScenarioEffect:
    return base_panel._make_effect(
        universe=universe,
        scenario_id=scenario_id,
        kind=kind,
        value_2080=value_2080,
        unit=unit,
    )


def _build_effects(
    *,
    universe,
    scenario_id: str,
    yield_change_pct: float,
    ef_change_pct: float,
    ruminant_change_pct: float,
) -> List[ScenarioEffect]:
    """Build grouped-axis scenario effects for the v2 panel.

    The delegated base panel still passes values named yield_change_pct and
    ef_change_pct. In v2 their meanings are:
      - ef_change_pct: horizontal E/L axis.
      - yield_change_pct: vertical L/A axis.
    """
    el_pct = float(ef_change_pct)
    la_pct = float(yield_change_pct)
    effects: List[ScenarioEffect] = []

    for kind in EL_DIRECT_KINDS:
        effects.append(
            _make_effect(
                universe=universe,
                scenario_id=scenario_id,
                kind=kind,
                value_2080=_axis_rate(el_pct),
            )
        )
    for kind in EL_REVERSE_KINDS:
        effects.append(
            _make_effect(
                universe=universe,
                scenario_id=scenario_id,
                kind=kind,
                value_2080=_axis_rate(el_pct, reverse=True),
            )
        )

    for kind in LA_DIRECT_KINDS:
        effects.append(
            _make_effect(
                universe=universe,
                scenario_id=scenario_id,
                kind=kind,
                value_2080=_axis_rate(la_pct),
            )
        )
    for kind in LA_REVERSE_KINDS:
        effects.append(
            _make_effect(
                universe=universe,
                scenario_id=scenario_id,
                kind=kind,
                value_2080=_axis_rate(la_pct, reverse=True),
            )
        )

    ruminant_pct = float(ruminant_change_pct)
    profile_sheet = str(
        (base_panel.CONFIG.get("override_cfg", {}) or {}).get(
            "nutrition_profile_sheet",
            base_panel.CFG.get("nutrition_profile_sheet", "low_land_new"),
        )
        or "low_land_new"
    ).strip().lower()
    if profile_sheet == "low_land_new":
        rumi_pct = int(round(max(0.0, min(100.0, ruminant_pct))))
        profile_name = f"Ruminate_Cap{rumi_pct:02d}"
        effects.append(
            _make_effect(
                universe=universe,
                scenario_id=scenario_id,
                kind="nutrition_profile",
                unit="profile",
                value_2080=profile_name,
            )
        )
    elif ruminant_pct != 0:
        rumi_pct = int(round(abs(ruminant_pct)))
        profile_name = (
            f"Ruminate_Cap{rumi_pct:02d}"
            if ruminant_pct < 0
            else f"Ruminate_Cap-{rumi_pct:02d}"
        )
        effects.append(
            _make_effect(
                universe=universe,
                scenario_id=scenario_id,
                kind="nutrition_profile",
                unit="profile",
                value_2080=profile_name,
            )
        )
    return effects


def _scenario_id(
    *,
    forest_pct: float,
    ruminant_pct: float,
    yield_pct: float,
    ef_pct: float,
) -> str:
    return (
        "FIG_PANEL_V2"
        f"_F{base_panel._fmt_pct(forest_pct)}"
        f"_R{base_panel._fmt_pct(ruminant_pct)}"
        f"_LA{base_panel._fmt_pct(yield_pct)}"
        f"_EL{base_panel._fmt_pct(ef_pct)}"
    )


def _parse_pct_list(raw: str) -> Optional[List[float]]:
    text = str(raw or "").strip()
    if not text:
        return None
    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    return [float(p) for p in parts] if parts else None


def _parse_pct_range(raw: str) -> Optional[tuple]:
    vals = _parse_pct_list(raw)
    if not vals:
        return None
    if len(vals) != 3:
        raise ValueError(f"Expected three values for range, got: {raw!r}")
    return (float(vals[0]), float(vals[1]), float(vals[2]))


def _make_default_config() -> Dict[str, object]:
    cfg = copy.deepcopy(base_panel.CONFIG)
    cfg.setdefault("override_cfg", {})
    cfg["axis_mode"] = "EL_LA_v2"
    cfg["el_change_pct_range"] = cfg.get("emission_factor_change_pct_range")
    cfg["la_change_pct_range"] = cfg.get("yield_change_pct_range")
    cfg["el_change_pct_values"] = cfg.get("emission_factor_change_pct_values")
    cfg["la_change_pct_values"] = cfg.get("yield_change_pct_values")
    if not str(cfg.get("output_dir", "") or "").strip():
        cfg["output_dir"] = str(base_panel._default_output_base() / "Panel_Yield_EF_v2")
    return cfg


CONFIG = _make_default_config()

_default_output_base = base_panel._default_output_base
_resolve_panel_output_root = base_panel._resolve_panel_output_root
_sync_panel_output_environment = base_panel._sync_panel_output_environment
_resolve_batch_settings = base_panel._resolve_batch_settings
_resolve_output_dir = base_panel._resolve_output_dir
_build_panel_tasks = base_panel._build_panel_tasks
_ruminant_cap_values_from_config = base_panel._ruminant_cap_values_from_config
_expand_axis_values = base_panel._expand_axis_values


def apply_v2_axis_aliases(cfg: Dict[str, object]) -> None:
    """Map v2 axis names onto the delegated base panel's axis keys."""
    if cfg.get("el_change_pct_values") is not None:
        cfg["emission_factor_change_pct_values"] = cfg.get("el_change_pct_values")
    if cfg.get("el_change_pct_range") is not None:
        cfg["emission_factor_change_pct_range"] = cfg.get("el_change_pct_range")
    if cfg.get("la_change_pct_values") is not None:
        cfg["yield_change_pct_values"] = cfg.get("la_change_pct_values")
    if cfg.get("la_change_pct_range") is not None:
        cfg["yield_change_pct_range"] = cfg.get("la_change_pct_range")

    el_values_env = _parse_pct_list(os.environ.get("PANEL_EL_CHANGE_PCT_VALUES", ""))
    la_values_env = _parse_pct_list(os.environ.get("PANEL_LA_CHANGE_PCT_VALUES", ""))
    el_range_env = _parse_pct_range(os.environ.get("PANEL_EL_CHANGE_PCT_RANGE", ""))
    la_range_env = _parse_pct_range(os.environ.get("PANEL_LA_CHANGE_PCT_RANGE", ""))
    if el_values_env is not None:
        cfg["emission_factor_change_pct_values"] = el_values_env
    if la_values_env is not None:
        cfg["yield_change_pct_values"] = la_values_env
    if el_range_env is not None:
        cfg["emission_factor_change_pct_range"] = el_range_env
    if la_range_env is not None:
        cfg["yield_change_pct_range"] = la_range_env


def _add_v2_axis_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "emission_factor_change_pct" not in out.columns or "yield_change_pct" not in out.columns:
        return out
    el = pd.to_numeric(out["emission_factor_change_pct"], errors="coerce")
    la = pd.to_numeric(out["yield_change_pct"], errors="coerce")

    out["x_axis_class"] = EL_AXIS_LABEL
    out["y_axis_class"] = LA_AXIS_LABEL
    out["el_change_pct"] = el
    out["la_change_pct"] = la
    out["el_multiplier"] = 1.0 + el / 100.0
    out["la_multiplier"] = 1.0 + la / 100.0

    out["fertilizer_rate_change_pct"] = el
    out["fertilizer_rate_multiplier"] = 1.0 + el / 100.0
    out["crop_soil_management_ratio_change_pct"] = el
    out["crop_soil_management_ratio_multiplier"] = 1.0 + el / 100.0
    out["manure_management_ratio_change_pct"] = -el
    out["manure_management_ratio_multiplier"] = 1.0 - el / 100.0
    out["yield_rate_change_pct"] = la
    out["yield_rate_multiplier"] = 1.0 + la / 100.0
    out["feed_intensity_change_pct"] = -la
    out["feed_intensity_multiplier"] = 1.0 - la / 100.0

    # Keep legacy columns because downstream panel scripts may still read them,
    # but make their v2 meaning explicit.
    out["emission_factor_axis_role"] = "E/L axis direct component"
    out["yield_axis_role"] = "L/A axis direct component"
    return out


def _postprocess_v2_csv_in_chunks(path: Path, *, chunksize: int) -> int:
    tmp_path = path.with_name(f".{path.name}.v2tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    rows_written = 0
    header_written = False
    try:
        reader = pd.read_csv(path, chunksize=chunksize) if chunksize > 0 else [pd.read_csv(path)]
        for chunk in reader:
            chunk = _add_v2_axis_columns(chunk)
            encoding = "utf-8-sig" if not header_written else "utf-8"
            chunk.to_csv(
                tmp_path,
                mode="a",
                header=not header_written,
                index=False,
                encoding=encoding,
            )
            header_written = True
            rows_written += int(len(chunk))
        if not header_written:
            pd.DataFrame().to_csv(tmp_path, index=False, encoding="utf-8-sig")
        os.replace(tmp_path, path)
        return rows_written
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def postprocess_v2_outputs(cfg: Dict[str, object]) -> None:
    root_output_dir = base_panel._resolve_panel_output_root(cfg.get("output_dir", ""))
    batch_state = base_panel._resolve_batch_settings(cfg)
    output_dir = base_panel._resolve_output_dir(root_output_dir, batch_state)
    chunksize = max(0, int(os.environ.get("PANEL_V2_POSTPROCESS_CHUNKSIZE") or 50000))
    names = [
        str(cfg.get("results_csv") or "figure_panel_dataset_long.csv"),
        str(cfg.get("global_emissions_detail_csv") or "figure_panel_global_emissions_detail_long.csv"),
        "run_meta.csv",
    ]
    for name in names:
        path = output_dir / name
        if not path.exists():
            continue
        try:
            rows = _postprocess_v2_csv_in_chunks(path, chunksize=chunksize)
            print(f"[S5_3_1_V2] refreshed v2 axis columns: {path} rows={rows}")
        except Exception as exc:
            print(f"[S5_3_1_V2][WARN] failed to add v2 axis columns to {path}: {exc}")


def main(config: Optional[Dict[str, object]] = None) -> None:
    cfg = copy.deepcopy(CONFIG if config is None else config)
    apply_v2_axis_aliases(cfg)

    CONFIG.clear()
    CONFIG.update(copy.deepcopy(cfg))
    base_panel.CONFIG.clear()
    base_panel.CONFIG.update(cfg)

    orig_build_effects = base_panel._build_effects
    orig_scenario_id = base_panel._scenario_id
    base_panel._build_effects = _build_effects
    base_panel._scenario_id = _scenario_id

    print(
        "[S5_3_1_V2] "
        f"axis_mode={cfg.get('axis_mode')} "
        f"x={EL_AXIS_LABEL}:{EL_DIRECT_KINDS}+{EL_REVERSE_KINDS}(reverse) "
        f"y={LA_AXIS_LABEL}:{LA_DIRECT_KINDS}+{LA_REVERSE_KINDS}(reverse)"
    )
    if str(cfg.get("output_dir", "") or "").strip():
        print(f"[S5_3_1_V2] output_dir={cfg['output_dir']}")

    try:
        base_panel.main()
        postprocess_v2_outputs(base_panel.CONFIG)
    finally:
        base_panel._build_effects = orig_build_effects
        base_panel._scenario_id = orig_scenario_id
        base_panel.CONFIG.pop("_batch_index_source", None)


if __name__ == "__main__":
    main()
