# -*- coding: utf-8 -*-
"""Build S5.3 v2 panel plotting data from merged batch CSV outputs."""
from __future__ import annotations

import copy
import gc
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

import S5_3_1_sensitivity_panel_yield_ef_v2 as panel_v2
import S5_3_2_panel_data_summary as summary_base


SCENARIO_RE_V2 = re.compile(
    r"^FIG_PANEL_V2_F(?P<forest>[mp]\d+)_R(?P<rumi>[mp]\d+)_LA(?P<la>[mp]\d+)_EL(?P<el>[mp]\d+)$",
    flags=re.IGNORECASE,
)

V2_AXIS_COLS = [
    "x_axis_class",
    "y_axis_class",
    "el_change_pct",
    "la_change_pct",
    "el_multiplier",
    "la_multiplier",
    "fertilizer_rate_change_pct",
    "fertilizer_rate_multiplier",
    "manure_management_ratio_change_pct",
    "manure_management_ratio_multiplier",
    "crop_soil_management_ratio_change_pct",
    "crop_soil_management_ratio_multiplier",
    "yield_rate_change_pct",
    "yield_rate_multiplier",
    "feed_intensity_change_pct",
    "feed_intensity_multiplier",
    "emission_factor_axis_role",
    "yield_axis_role",
]

PALE_RESULT_COLS = list(getattr(panel_v2, "PALE_RESULT_COLS", []))


CONFIG = copy.deepcopy(summary_base.CONFIG)
CONFIG["panel_root_dir"] = str(panel_v2.CONFIG.get("output_dir", "") or "")
_ORIG_PARSE_SCENARIO_ID = summary_base._parse_scenario_id
_ORIG_NORMALISE_PANEL_RESULTS = summary_base._normalise_panel_results


def _parse_scenario_id_v2(scenario_id: str) -> Optional[Dict[str, float]]:
    m = SCENARIO_RE_V2.match(str(scenario_id or "").strip())
    if not m:
        return _ORIG_PARSE_SCENARIO_ID(scenario_id)
    return {
        "forest_area_change_pct": summary_base._parse_signed_token(m.group("forest")),
        summary_base.RUMINANT_CAP_COL: summary_base._parse_signed_token(m.group("rumi")),
        "yield_change_pct": summary_base._parse_signed_token(m.group("la")),
        "emission_factor_change_pct": summary_base._parse_signed_token(m.group("el")),
    }


def _scenario_id_v2(
    *,
    forest_pct: float,
    ruminant_pct: float,
    yield_pct: float,
    ef_pct: float,
) -> str:
    return panel_v2._scenario_id(
        forest_pct=forest_pct,
        ruminant_pct=ruminant_pct,
        yield_pct=yield_pct,
        ef_pct=ef_pct,
    )


def _normalise_panel_results_v2(*args, **kwargs) -> pd.DataFrame:
    df = _ORIG_NORMALISE_PANEL_RESULTS(*args, **kwargs)
    return panel_v2._add_v2_axis_columns(df)


def _cli_int_option(names) -> Optional[int]:
    args = list(sys.argv[1:])
    for idx, arg in enumerate(args):
        for name in names:
            if arg == name and idx + 1 < len(args):
                try:
                    return int(args[idx + 1])
                except Exception:
                    return None
            prefix = f"{name}="
            if arg.startswith(prefix):
                try:
                    return int(arg[len(prefix):])
                except Exception:
                    return None
    return None


def _refresh_v2_axis_columns_in_chunks(path: Path, *, chunksize: int) -> int:
    tmp_path = path.with_name(f".{path.name}.v2summarytmp")
    if tmp_path.exists():
        tmp_path.unlink()

    rows_written = 0
    header_written = False
    try:
        reader = pd.read_csv(path, chunksize=chunksize) if chunksize > 0 else [pd.read_csv(path, low_memory=False)]
        for chunk in reader:
            chunk = panel_v2._add_v2_axis_columns(chunk)
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
            del chunk
            gc.collect()
        if not header_written:
            pd.DataFrame().to_csv(tmp_path, index=False, encoding="utf-8-sig")
        os.replace(tmp_path, path)
        return rows_written
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _refresh_v2_axis_columns(panel_root: str, cfg: Dict[str, object]) -> None:
    root = summary_base._resolve_panel_root_dir(panel_root)
    chunksize = max(
        0,
        int(
            _cli_int_option(("--detail-chunksize",))
            or os.environ.get("PANEL_SUMMARY_V2_REFRESH_CHUNKSIZE")
            or os.environ.get("PANEL_SUMMARY_DETAIL_CHUNKSIZE")
            or cfg.get("detail_chunksize")
            or 50000
        ),
    )
    names = [
        str(cfg.get("input_results_csv") or "figure_panel_dataset_long.csv"),
        str(cfg.get("input_detail_csv") or "figure_panel_global_emissions_detail_long.csv"),
        str(cfg.get("raw_results_csv") or "figure_panel_dataset_long_rebuilt.csv"),
        str(cfg.get("plot_ready_csv") or "figure_panel_dataset_long_plot_ready.csv"),
        str(cfg.get("detail_results_csv") or "figure_panel_global_emissions_detail_long_rebuilt.csv"),
        str(cfg.get("missing_audit_csv") or "figure_panel_missing_points_audit.csv"),
    ]
    for name in names:
        path = root / name
        if not path.exists():
            continue
        try:
            rows = _refresh_v2_axis_columns_in_chunks(path, chunksize=chunksize)
            print(f"[S5_3_2_V2] refreshed v2 axis columns: {path} rows={rows}")
        except Exception as exc:
            print(f"[S5_3_2_V2][WARN] failed to refresh v2 axis columns in {path}: {exc}")


def main() -> None:
    old_config = summary_base.CONFIG
    old_raw_keep_cols = list(summary_base.RAW_KEEP_COLS)
    old_parse_scenario_id = summary_base._parse_scenario_id
    old_scenario_id = summary_base._scenario_id
    old_normalise_panel_results = summary_base._normalise_panel_results

    cfg = copy.deepcopy(CONFIG)
    keep_cols = list(dict.fromkeys(list(summary_base.RAW_KEEP_COLS) + PALE_RESULT_COLS + V2_AXIS_COLS))
    summary_base.CONFIG = cfg
    summary_base.RAW_KEEP_COLS = keep_cols
    summary_base._parse_scenario_id = _parse_scenario_id_v2
    summary_base._scenario_id = _scenario_id_v2
    global _ORIG_PARSE_SCENARIO_ID, _ORIG_NORMALISE_PANEL_RESULTS
    _ORIG_PARSE_SCENARIO_ID = old_parse_scenario_id
    _ORIG_NORMALISE_PANEL_RESULTS = old_normalise_panel_results
    summary_base._normalise_panel_results = _normalise_panel_results_v2

    try:
        if not any(arg in ("-h", "--help") for arg in sys.argv[1:]):
            print("[S5_3_2_V2] building Panel_Yield_EF_v2 plot-ready data")
        summary_base.main()
        panel_root = os.environ.get("PANEL_OUTPUT_DIR") or str(cfg.get("panel_root_dir") or "")
        _refresh_v2_axis_columns(panel_root, cfg)
        print("[S5_3_2_V2] v2 axis columns refreshed in summary outputs")
    finally:
        summary_base.CONFIG = old_config
        summary_base.RAW_KEEP_COLS = old_raw_keep_cols
        summary_base._parse_scenario_id = old_parse_scenario_id
        summary_base._scenario_id = old_scenario_id
        summary_base._normalise_panel_results = old_normalise_panel_results


if __name__ == "__main__":
    main()
