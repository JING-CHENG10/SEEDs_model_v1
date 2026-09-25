"""
Build a workbook of harmonized AR6 rows that are missing from SCI.

Reads:
  - output/Plot/Fig6/AR6_scenario_prepared_harmonization.xlsx
  - output/Plot/Fig6/SCI/SCI_Database_harmonization.xlsx

Compares:
  - 1.5D_harmonized
  - 2D_harmonized

Outputs:
  - output/Plot/Fig6/SCI/SCI_added_harmonization_from_AR6.xlsx

The output keeps only AR6 rows whose Model#Scenario + Region + Variable (+ Unit)
do not already exist in the SCI workbook. Sheet columns are aligned to the
SCI harmonized sheet structure.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from config_paths import get_results_base


FIG6_DIR = Path(get_results_base()) / "Plot" / "Fig6"
AR6_PATH = FIG6_DIR / "AR6_scenario_prepared_harmonization.xlsx"
SCI_PATH = FIG6_DIR / "SCI" / "SCI_Database_harmonization.xlsx"
OUTPUT_PATH = FIG6_DIR / "SCI" / "SCI_added_harmonization_from_AR6.xlsx"
SHEETS = ["1.5D_harmonized", "2D_harmonized"]


def _read_sheet(path: Path, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    df.columns = [str(col).strip() for col in df.columns]
    return df


def _model_scenario_key(df: pd.DataFrame) -> pd.Series:
    if "Model#Scenario" in df.columns:
        key = df["Model#Scenario"]
    else:
        model = df["Model"] if "Model" in df.columns else ""
        scenario = df["Scenario"] if "Scenario" in df.columns else ""
        key = model.astype(str) + "#" + scenario.astype(str)
    return key.fillna("").astype(str).str.strip()


def _key_index(df: pd.DataFrame) -> pd.Index:
    work = pd.DataFrame(
        {
            "Model#Scenario": _model_scenario_key(df),
            "Region": df["Region"].fillna("").astype(str).str.strip() if "Region" in df.columns else "",
            "Variable": df["Variable"].fillna("").astype(str).str.strip() if "Variable" in df.columns else "",
            "Unit": df["Unit"].fillna("").astype(str).str.strip() if "Unit" in df.columns else "",
        }
    )
    return pd.Index(pd.MultiIndex.from_frame(work))


def _align_to_sci_columns(df_ar6: pd.DataFrame, sci_columns: List[str]) -> pd.DataFrame:
    out = df_ar6.copy()
    for col in sci_columns:
        if col not in out.columns:
            out[col] = np.nan
    return out.loc[:, sci_columns].copy()


def _build_added_rows(ar6_df: pd.DataFrame, sci_df: pd.DataFrame) -> pd.DataFrame:
    ar6_keys = _key_index(ar6_df)
    sci_keys = _key_index(sci_df)
    keep_mask = ~ar6_keys.isin(sci_keys)
    return ar6_df.loc[keep_mask].copy()


def main() -> None:
    if not AR6_PATH.exists():
        raise FileNotFoundError(f"Missing AR6 workbook: {AR6_PATH}")
    if not SCI_PATH.exists():
        raise FileNotFoundError(f"Missing SCI workbook: {SCI_PATH}")

    outputs: Dict[str, pd.DataFrame] = {}
    counts = []
    for sheet in SHEETS:
        ar6_df = _read_sheet(AR6_PATH, sheet)
        sci_df = _read_sheet(SCI_PATH, sheet)
        added = _build_added_rows(ar6_df, sci_df)
        added = _align_to_sci_columns(added, list(sci_df.columns))
        outputs[sheet] = added
        counts.append(
            {
                "sheet": sheet,
                "ar6_rows": len(ar6_df),
                "sci_rows": len(sci_df),
                "added_rows": len(added),
            }
        )

    outputs["summary"] = pd.DataFrame(counts)

    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        for sheet, df in outputs.items():
            df.to_excel(writer, sheet_name=sheet, index=False)

    print(f"[DONE] {OUTPUT_PATH}")
    for row in counts:
        print(
            f"[COUNT] {row['sheet']}: "
            f"AR6={row['ar6_rows']}, SCI={row['sci_rows']}, added={row['added_rows']}"
        )


if __name__ == "__main__":
    main()
