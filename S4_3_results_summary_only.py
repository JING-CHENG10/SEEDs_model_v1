from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from config_paths import get_src_base
from S4_1_results import write_summary_tables_from_detail_long


DEFAULT_DETAIL_FILENAME = "emissions_summary_Detail_Long.csv"
DEFAULT_FAST_SUMMARY_FILENAME = "emissions_fast_summary.csv"
DEFAULT_DICT_V3_PATH = str(Path(get_src_base()) / "dict_v3.xlsx")


CONFIG = {
    # Recommended: set the parent results directory here.
    # The script will recursively scan all descendants under results_root_dir
    # and process every */Emis/emissions_summary_Detail_Long.csv it finds.
    # Example: <Code>/output/MC_Sensitivity_Variable_Effect/runs
    "results_root_dir": "",
    "dict_v3_path": DEFAULT_DICT_V3_PATH,
    "detail_filename": DEFAULT_DETAIL_FILENAME,
    "filename_suffix": "",
    "allowed_years": None,  # Example: [2020, 2030, 2050, 2080]
}


def _discover_detail_long_files(root: Path, detail_filename: str) -> List[Path]:
    if root.is_file():
        if root.name == detail_filename:
            return [root]
        return []

    if root.is_dir() and root.name == "Emis":
        detail_path = root / detail_filename
        return [detail_path] if detail_path.exists() else []

    direct_detail = root / "Emis" / detail_filename
    if direct_detail.exists():
        return [direct_detail]

    matches = []
    for path in root.rglob(detail_filename):
        if path.parent.name == "Emis":
            matches.append(path)
    return sorted(set(matches))


def _discover_emis_files(root: Path, filename: str) -> List[Path]:
    if root.is_file():
        if root.name == filename and root.parent.name == "Emis":
            return [root]
        return []

    matches = []
    for path in root.rglob(filename):
        if path.parent.name == "Emis":
            matches.append(path)
    return sorted(set(matches))


def _format_written_paths(written: Iterable[Path]) -> str:
    return ", ".join(str(path.name) for path in written)


def _resolve_config_target_root() -> Path:
    root = str(CONFIG.get("results_root_dir", "") or "").strip()
    if not root:
        raise ValueError(
            "Missing target path. Pass target_path on the command line, or set CONFIG['results_root_dir'] "
            "to the parent directory whose children contain Emis folders."
        )
    return Path(root)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild emissions summary tables from existing Emis/emissions_summary_Detail_Long.csv "
            "without rerunning the simulation."
        )
    )
    parser.add_argument(
        "target_path",
        nargs="?",
        help=(
            "Optional target path. If omitted, the script uses CONFIG['results_root_dir'] and processes "
            "<results_root_dir>/<scenario>/Emis."
        ),
    )
    parser.add_argument(
        "--dict-v3-path",
        default=str(CONFIG.get("dict_v3_path") or DEFAULT_DICT_V3_PATH),
        help="Path to dict_v3.xlsx. Defaults to src/dict_v3.xlsx.",
    )
    parser.add_argument(
        "--detail-filename",
        default=str(CONFIG.get("detail_filename") or DEFAULT_DETAIL_FILENAME),
        help=f"Detail_Long filename to search for. Default: {DEFAULT_DETAIL_FILENAME}",
    )
    parser.add_argument(
        "--filename-suffix",
        default=str(CONFIG.get("filename_suffix") or ""),
        help="Optional suffix inserted before .csv for regenerated summary files.",
    )
    parser.add_argument(
        "--allowed-years",
        nargs="*",
        type=int,
        default=CONFIG.get("allowed_years"),
        help="Optional explicit list of years. By default the script infers years from each Detail_Long file.",
    )
    args = parser.parse_args()

    using_config_root = not args.target_path
    if using_config_root:
        try:
            target_path = _resolve_config_target_root()
        except ValueError as exc:
            print(f"[ERROR] {exc}")
            return 1
        detail_files = _discover_detail_long_files(target_path, args.detail_filename)
    else:
        target_path = Path(args.target_path)
        detail_files = _discover_detail_long_files(target_path, args.detail_filename)

    if not detail_files:
        print(f"[ERROR] No {args.detail_filename} found under: {target_path}")
        fast_files = _discover_emis_files(target_path, DEFAULT_FAST_SUMMARY_FILENAME)
        if fast_files:
            print(
                f"[HINT] Found {len(fast_files)} {DEFAULT_FAST_SUMMARY_FILENAME} file(s) under the same root. "
                "These runs are likely fast-emissions-only outputs, which do not include Detail_Long."
            )
            print(
                "[HINT] S4_3_results_summary_only.py rebuilds summary tables from "
                "Emis/emissions_summary_Detail_Long.csv, so you need runs that saved full detail outputs."
            )
        return 1

    if using_config_root:
        print(f"[INFO] Using CONFIG['results_root_dir']: {target_path}")
        print(f"[INFO] Recursively scanning result folders under: {target_path}")
    print(f"[INFO] Found {len(detail_files)} Detail_Long file(s) under: {target_path}")
    success = 0
    failed = 0

    for idx, detail_path in enumerate(detail_files, start=1):
        result_dir = detail_path.parent.parent if detail_path.parent.name == "Emis" else detail_path.parent
        output_dir = detail_path.parent
        print(f"\n[INFO] [{idx}/{len(detail_files)}] Rebuilding summaries for: {result_dir}")
        print(f"[START] Reading Detail_Long: {detail_path}")
        print(f"[START] Writing summary files to: {output_dir}")
        try:
            written = write_summary_tables_from_detail_long(
                detail_long_path=str(detail_path),
                output_dir=str(output_dir),
                dict_v3_path=args.dict_v3_path,
                allowed_years=args.allowed_years,
                filename_suffix=args.filename_suffix,
            )
            print(f"[OK] Wrote {len(written)} file(s): {_format_written_paths(written.values())}")
            print(f"[DONE] Finished processing: {detail_path}")
            print(f"[DONE] Output directory: {output_dir}")
            success += 1
        except Exception as exc:
            print(f"[ERROR] Failed for {detail_path}: {exc}")
            failed += 1

    print(f"\n[SUMMARY] success={success}, failed={failed}, total={len(detail_files)}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
