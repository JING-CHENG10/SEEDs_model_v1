# -*- coding: utf-8 -*-
"""
Forest-area-only sensitivity runner for S5.3.1.

This wrapper reuses ``S5_3_1_sensitivity_panel_yield_ef`` while fixing yield,
emission factor, and ruminant cap so forest target sensitivity can be checked
without regenerating the full panel grid.
"""
from __future__ import annotations

import argparse
import copy
from typing import List

import S5_3_1_sensitivity_panel_yield_ef as panel


def _parse_number_list(raw: str) -> List[float]:
    values: List[float] = []
    for part in str(raw).split(","):
        text = part.strip()
        if not text:
            continue
        values.append(float(text))
    if not values:
        raise argparse.ArgumentTypeError("value list must contain at least one number")
    return values


def _single_value(raw: str) -> List[float]:
    return [float(raw)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run S5.3.1 forest-area sensitivity with yield, EF, and ruminant cap fixed."
        )
    )
    parser.add_argument(
        "--forest-values",
        type=_parse_number_list,
        default=[0.0, 30.0],
        help="Comma-separated forest_area_change_pct values. Default: 0,30.",
    )
    parser.add_argument(
        "--yield-pct",
        type=_single_value,
        default=[0.0],
        help="Fixed yield_change_pct value. Default: 0.",
    )
    parser.add_argument(
        "--ef-pct",
        type=_single_value,
        default=[0.0],
        help="Fixed emission_factor_change_pct value. Default: 0.",
    )
    parser.add_argument(
        "--ruminant-cap",
        type=_single_value,
        default=[13.0],
        help="Fixed ruminant_kcal_share_cap_pct value. Default: 13.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/Panel_Forest_Area_Fixed_Yield_EF",
        help="Output directory under NZF_OUTPUT_DIR/output base, or absolute path.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing completed per-scenario run directories in the output dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = copy.deepcopy(panel.CONFIG)
    cfg.update(
        {
            "output_dir": args.output_dir,
            "results_csv": "forest_area_fixed_sensitivity.csv",
            "global_emissions_detail_csv": "forest_area_fixed_global_emissions_detail.csv",
            "emission_factor_change_pct_values": args.ef_pct,
            "yield_change_pct_values": args.yield_pct,
            "ruminant_kcal_share_cap_pct_values": args.ruminant_cap,
            "forest_area_change_pct_values": args.forest_values,
            "resume": bool(args.resume),
            "clear_existing_run_dirs_when_no_resume": not bool(args.resume),
            "batch": {
                "enabled": False,
                "total_batches": 1,
                "batch_index": 1,
                "assignment": "round_robin",
                "batches_subdir": "batches",
            },
        }
    )
    panel.CONFIG = cfg
    panel.main()


if __name__ == "__main__":
    main()
