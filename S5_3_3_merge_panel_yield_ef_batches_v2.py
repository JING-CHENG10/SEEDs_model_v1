# -*- coding: utf-8 -*-
"""Merge batched S5_3_1 v2 panel outputs.

This is the v2 companion for S5_3_3_merge_panel_yield_ef_batches.py.  It
reuses the original merge implementation and points it at the v2 panel module,
whose default output root is <NZF_OUTPUT_DIR>/Panel_Yield_EF_v2.
"""
from __future__ import annotations

import copy
import os
import sys

import S5_3_1_sensitivity_panel_yield_ef_v2 as panel_v2
import S5_3_3_merge_panel_yield_ef_batches as merge_base


CONFIG = copy.deepcopy(merge_base.CONFIG)
CONFIG["output_dir"] = str(panel_v2.CONFIG.get("output_dir", "") or "")
CONFIG["resubmit_command"] = "bash submit_sbatch_S5_3_1_panel_yield_ef_v2.sh"
CONFIG["data_summary_command"] = "python S5_3_2_panel_data_summary_v2.py"


def main() -> None:
    old_panel = merge_base.panel
    old_config = merge_base.CONFIG
    panel_v2.apply_v2_axis_aliases(panel_v2.CONFIG)
    merge_base.panel = panel_v2
    merge_base.CONFIG = copy.deepcopy(CONFIG)
    try:
        if not any(arg in ("-h", "--help") for arg in sys.argv[1:]):
            print("[S5_3_3_V2] merging Panel_Yield_EF_v2 batches")
            print("[S5_3_3_V2] low-memory streamed merge/postprocess enabled")
        merge_base.main()
        post_cfg = copy.deepcopy(panel_v2.CONFIG)
        if os.environ.get("PANEL_OUTPUT_DIR"):
            post_cfg["output_dir"] = os.environ["PANEL_OUTPUT_DIR"]
        panel_v2.postprocess_v2_outputs(post_cfg)
        print("[S5_3_3_V2] v2 axis columns refreshed in merged CSV outputs")
        print("[S5_3_3_V2] Next: python S5_3_2_panel_data_summary_v2.py --output-dir <Panel_Yield_EF_v2>")
    finally:
        merge_base.panel = old_panel
        merge_base.CONFIG = old_config


if __name__ == "__main__":
    main()
