# -*- coding: utf-8 -*-
"""
SC1_1_regression_output_compare.py

Compare two scenario output folders and check whether result tables remain
semantically consistent after code changes. The checker focuses on non-log
result files and tolerates tiny floating-point drift caused by different
aggregation order.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from config_paths import get_results_base


DEFAULT_BASELINE_ROOT = Path(get_results_base()) / "MC_Sensitivity_Variable_Effect" / "runs"
DEFAULT_CANDIDATE_ROOT = Path(get_results_base()) / "MC_Sensitivity_Variable_Effect_after_patch_compare" / "runs"
DEFAULT_EXTENSIONS = {".csv", ".xlsx", ".xls"}


@dataclass
class TableCompareResult:
    name: str
    status: str
    reason: str
    rows_old: int
    rows_new: int
    cols_old: int
    cols_new: int
    missing_keys_old_only: int = 0
    missing_keys_new_only: int = 0
    column_order_equal: bool = True
    column_set_equal: bool = True
    max_abs_diff: float = 0.0
    max_rel_diff: float = 0.0
    numeric_diff_by_col: Dict[str, Dict[str, float]] | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two scenario output folders and validate regression consistency."
    )
    parser.add_argument(
        "--scenario-id",
        required=True,
        help="Scenario/output folder name under both baseline and candidate roots.",
    )
    parser.add_argument(
        "--baseline-root",
        default=str(DEFAULT_BASELINE_ROOT),
        help=f"Baseline runs root (default: {DEFAULT_BASELINE_ROOT}).",
    )
    parser.add_argument(
        "--candidate-root",
        default=str(DEFAULT_CANDIDATE_ROOT),
        help=f"Candidate runs root (default: {DEFAULT_CANDIDATE_ROOT}).",
    )
    parser.add_argument(
        "--include-log",
        action="store_true",
        help="Include files under Log/ when scanning result files.",
    )
    parser.add_argument(
        "--abs-tol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for numeric comparison (default: 1e-6).",
    )
    parser.add_argument(
        "--rel-tol",
        type=float,
        default=1e-8,
        help="Relative tolerance for numeric comparison (default: 1e-8).",
    )
    parser.add_argument(
        "--report",
        default="",
        help="Optional explicit JSON report path. Defaults to candidate scenario folder.",
    )
    return parser.parse_args()


def _read_csv_auto(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except Exception:
            continue
    return pd.read_csv(path, low_memory=False)


def _read_tables(path: Path) -> Dict[str, pd.DataFrame]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return {path.name: _read_csv_auto(path)}
    if suffix in {".xlsx", ".xls"}:
        sheets = pd.read_excel(path, sheet_name=None)
        return {f"{path.name}::{sheet_name}": df for sheet_name, df in sheets.items()}
    raise ValueError(f"Unsupported file type: {path}")


def _normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if not is_numeric_dtype(out[col]):
            out[col] = out[col].replace({np.nan: ""}).astype(str).str.strip()
    return out


def _iter_result_files(root: Path, include_log: bool) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in DEFAULT_EXTENSIONS:
            continue
        if not include_log and "Log" in path.parts:
            continue
        yield path


def _relative_file_set(root: Path, include_log: bool) -> List[str]:
    return sorted(str(path.relative_to(root)) for path in _iter_result_files(root, include_log))


def _safe_rel(abs_diff: float, base_value: float) -> float:
    if base_value == 0.0:
        return 0.0 if abs_diff == 0.0 else float("inf")
    return float(abs_diff / abs(base_value))


def _compare_dataframe(
    name: str,
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
    abs_tol: float,
    rel_tol: float,
) -> TableCompareResult:
    old_df = _normalize_text_columns(old_df)
    new_df = _normalize_text_columns(new_df)
    old_cols = list(old_df.columns)
    new_cols = list(new_df.columns)
    col_set_old = set(old_cols)
    col_set_new = set(new_cols)
    common_cols = [c for c in old_cols if c in col_set_new]
    result = TableCompareResult(
        name=name,
        status="pass",
        reason="",
        rows_old=len(old_df),
        rows_new=len(new_df),
        cols_old=len(old_cols),
        cols_new=len(new_cols),
        column_order_equal=(old_cols == new_cols),
        column_set_equal=(col_set_old == col_set_new),
        numeric_diff_by_col={},
    )
    if not result.column_set_equal:
        result.status = "fail"
        result.reason = "column_set_mismatch"
        return result

    numeric_cols = [c for c in common_cols if is_numeric_dtype(old_df[c]) and is_numeric_dtype(new_df[c])]
    key_cols = [c for c in common_cols if c not in numeric_cols]

    if numeric_cols:
        if key_cols:
            old_cmp = old_df[common_cols].groupby(key_cols, dropna=False, as_index=False)[numeric_cols].sum()
            new_cmp = new_df[common_cols].groupby(key_cols, dropna=False, as_index=False)[numeric_cols].sum()
            merged = old_cmp.merge(new_cmp, on=key_cols, how="outer", suffixes=("_old", "_new"), indicator=True)
            result.missing_keys_old_only = int((merged["_merge"] == "left_only").sum())
            result.missing_keys_new_only = int((merged["_merge"] == "right_only").sum())
            if result.missing_keys_old_only or result.missing_keys_new_only:
                result.status = "fail"
                result.reason = "key_mismatch"
            for col in numeric_cols:
                delta = (merged[f"{col}_old"].fillna(0.0) - merged[f"{col}_new"].fillna(0.0)).abs()
                max_abs = float(delta.max()) if len(delta) else 0.0
                base = merged[f"{col}_old"].fillna(0.0).abs()
                rel = np.where(base > 0, delta / base, np.where(delta == 0, 0.0, np.inf))
                max_rel = float(np.nanmax(rel)) if len(delta) else 0.0
                sum_abs = float(delta.sum()) if len(delta) else 0.0
                result.numeric_diff_by_col[col] = {
                    "max_abs_diff": max_abs,
                    "max_rel_diff": max_rel,
                    "sum_abs_diff": sum_abs,
                }
                result.max_abs_diff = max(result.max_abs_diff, max_abs)
                result.max_rel_diff = max(result.max_rel_diff, max_rel)
                if max_abs > abs_tol and max_rel > rel_tol and result.status != "fail":
                    result.status = "warn"
                    result.reason = "numeric_drift_over_tolerance"
        else:
            for col in numeric_cols:
                old_sum = float(pd.to_numeric(old_df[col], errors="coerce").sum())
                new_sum = float(pd.to_numeric(new_df[col], errors="coerce").sum())
                max_abs = abs(old_sum - new_sum)
                max_rel = _safe_rel(max_abs, old_sum)
                result.numeric_diff_by_col[col] = {
                    "max_abs_diff": max_abs,
                    "max_rel_diff": max_rel,
                    "sum_abs_diff": max_abs,
                }
                result.max_abs_diff = max(result.max_abs_diff, max_abs)
                result.max_rel_diff = max(result.max_rel_diff, max_rel)
                if max_abs > abs_tol and max_rel > rel_tol and result.status != "fail":
                    result.status = "warn"
                    result.reason = "numeric_drift_over_tolerance"
    else:
        old_rows = sorted(map(tuple, old_df[common_cols].itertuples(index=False, name=None)))
        new_rows = sorted(map(tuple, new_df[common_cols].itertuples(index=False, name=None)))
        if old_rows != new_rows:
            result.status = "fail"
            result.reason = "non_numeric_content_mismatch"

    if result.status == "pass" and not result.reason:
        result.reason = "matched"
    return result


def _compare_file(
    rel_path: str,
    old_path: Path,
    new_path: Path,
    abs_tol: float,
    rel_tol: float,
) -> List[TableCompareResult]:
    old_tables = _read_tables(old_path)
    new_tables = _read_tables(new_path)
    results: List[TableCompareResult] = []
    all_names = sorted(set(old_tables) | set(new_tables))
    for name in all_names:
        if name not in old_tables or name not in new_tables:
            status = "fail"
            reason = "sheet_missing"
            results.append(
                TableCompareResult(
                    name=f"{rel_path}::{name}",
                    status=status,
                    reason=reason,
                    rows_old=len(old_tables.get(name, pd.DataFrame())),
                    rows_new=len(new_tables.get(name, pd.DataFrame())),
                    cols_old=len(old_tables.get(name, pd.DataFrame()).columns),
                    cols_new=len(new_tables.get(name, pd.DataFrame()).columns),
                )
            )
            continue
        table_name = rel_path if old_path.suffix.lower() == ".csv" else f"{rel_path}::{name.split('::', 1)[1]}"
        results.append(_compare_dataframe(table_name, old_tables[name], new_tables[name], abs_tol, rel_tol))
    return results


def main() -> None:
    args = _parse_args()
    baseline_dir = Path(args.baseline_root) / args.scenario_id
    candidate_dir = Path(args.candidate_root) / args.scenario_id
    if not baseline_dir.exists():
        raise FileNotFoundError(f"Baseline scenario folder not found: {baseline_dir}")
    if not candidate_dir.exists():
        raise FileNotFoundError(f"Candidate scenario folder not found: {candidate_dir}")

    old_files = set(_relative_file_set(baseline_dir, args.include_log))
    new_files = set(_relative_file_set(candidate_dir, args.include_log))
    common_files = sorted(old_files & new_files)
    old_only = sorted(old_files - new_files)
    new_only = sorted(new_files - old_files)

    table_results: List[TableCompareResult] = []
    for rel_path in common_files:
        table_results.extend(
            _compare_file(
                rel_path,
                baseline_dir / rel_path,
                candidate_dir / rel_path,
                abs_tol=args.abs_tol,
                rel_tol=args.rel_tol,
            )
        )

    failed = [r for r in table_results if r.status == "fail"]
    warned = [r for r in table_results if r.status == "warn"]
    passed = [r for r in table_results if r.status == "pass"]

    report = {
        "scenario_id": args.scenario_id,
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "abs_tol": args.abs_tol,
        "rel_tol": args.rel_tol,
        "old_only_files": old_only,
        "new_only_files": new_only,
        "summary": {
            "common_files": len(common_files),
            "tables_pass": len(passed),
            "tables_warn": len(warned),
            "tables_fail": len(failed),
        },
        "results": [r.__dict__ for r in table_results],
    }

    report_path = Path(args.report) if args.report else (candidate_dir / f"regression_compare_{args.scenario_id}.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 100)
    print(f"Scenario: {args.scenario_id}")
    print(f"Baseline:  {baseline_dir}")
    print(f"Candidate: {candidate_dir}")
    print(f"Common files: {len(common_files)}")
    print(f"Old-only files: {len(old_only)}")
    print(f"New-only files: {len(new_only)}")
    print(f"Tables: pass={len(passed)} warn={len(warned)} fail={len(failed)}")
    if old_only:
        print("[OLD-ONLY]")
        for rel in old_only:
            print(f"  {rel}")
    if new_only:
        print("[NEW-ONLY]")
        for rel in new_only:
            print(f"  {rel}")
    if warned:
        print("[WARN]")
        for item in warned:
            print(
                f"  {item.name} | {item.reason} | "
                f"max_abs={item.max_abs_diff:.12g} | max_rel={item.max_rel_diff:.12g}"
            )
    if failed:
        print("[FAIL]")
        for item in failed:
            print(f"  {item.name} | {item.reason}")
    print(f"Report: {report_path}")
    print("=" * 100)

    if old_only or new_only or failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
