from __future__ import annotations

import os

import S5_3_1_sensitivity_panel_yield_ef_batches as batch_base
import S5_3_1_sensitivity_panel_yield_ef_v2 as panel_v2


def main() -> None:
    args = batch_base._build_arg_parser().parse_args()
    cfg = batch_base._apply_overrides(panel_v2.CONFIG, args)
    panel_v2.apply_v2_axis_aliases(cfg)

    batch_cfg = cfg.get("batch", {}) or {}
    print(
        "[S5_3_1_BATCH_V2] "
        f"axis_mode={cfg.get('axis_mode')} "
        f"x={panel_v2.EL_AXIS_LABEL}:{panel_v2.EL_DIRECT_KINDS}+{panel_v2.EL_REVERSE_KINDS}(reverse) "
        f"y={panel_v2.LA_AXIS_LABEL}:{panel_v2.LA_DIRECT_KINDS}+{panel_v2.LA_REVERSE_KINDS}(reverse)"
    )
    print(
        "[S5_3_1_BATCH_V2] "
        f"batch_index={batch_cfg.get('batch_index')} "
        f"total_batches={batch_cfg.get('total_batches')} "
        f"assignment={batch_cfg.get('assignment')} "
        f"enabled={batch_cfg.get('enabled')} "
        f"resume={cfg.get('resume')} "
        f"clear_existing_runs={cfg.get('clear_existing_run_dirs_when_no_resume')} "
        f"source={cfg.get('_batch_index_source')}"
    )
    print(
        "[S5_3_1_BATCH_V2] "
        f"slurm_job_name={os.environ.get('SLURM_JOB_NAME', '') or '<empty>'}"
    )
    if str(cfg.get("output_dir", "") or "").strip():
        print(f"[S5_3_1_BATCH_V2] output_dir={cfg['output_dir']}")
    print(f"[S5_3_1_BATCH_V2] NZF_OUTPUT_DIR={os.environ.get('NZF_OUTPUT_DIR', '')}")
    print(f"[S5_3_1_BATCH_V2] PANEL_OUTPUT_DIR={os.environ.get('PANEL_OUTPUT_DIR', '')}")

    panel_v2.main(cfg)


if __name__ == "__main__":
    main()
