import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config_paths import get_input_base, get_src_base, get_results_base


def _format_m49(val: object) -> str | None:
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return None
    s = s.replace("'", "")
    try:
        num = int(float(s))
        return f"'{num:03d}"
    except Exception:
        return None


def _load_mapping(dict_path: Path) -> pd.DataFrame:
    df = pd.read_excel(dict_path, sheet_name="Emis_item")
    cols = ["Item_Nutrition_Map", "Item_Production_Map"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"dict_v3 missing columns: {missing}")
    df = df[cols].copy()
    df["Item_Nutrition_Map"] = df["Item_Nutrition_Map"].astype(str).str.strip()
    df["Item_Production_Map"] = df["Item_Production_Map"].astype(str).str.strip()
    df = df[
        (df["Item_Nutrition_Map"] != "")
        & (df["Item_Production_Map"] != "")
        & (df["Item_Nutrition_Map"].str.lower() != "non food")
    ]
    df = df.drop_duplicates()
    return df


def _load_production(prod_path: Path) -> pd.DataFrame:
    df = pd.read_csv(prod_path)
    need_cols = {"M49_Country_Code", "Item", "Element", "Y2020"}
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"production file missing columns: {missing}")
    df = df[df["Element"].astype(str).str.strip() == "Production"].copy()
    df["M49_fmt"] = df["M49_Country_Code"].apply(_format_m49)
    df["Y2020"] = pd.to_numeric(df["Y2020"], errors="coerce")
    df = df.dropna(subset=["M49_fmt", "Item"])
    prod = (
        df.groupby(["M49_fmt", "Item"], as_index=False)["Y2020"]
        .sum()
        .rename(columns={"Item": "Item_Production_Map", "Y2020": "production_y2020"})
    )
    return prod


def _load_nutrition(nut_path: Path) -> pd.DataFrame:
    df = pd.read_excel(nut_path, sheet_name=0)
    need_cols = {"M49_Country_Code", "Area", "Item", "Element", "Y2020"}
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"nutrition profile missing columns: {missing}")
    df = df.copy()
    elem = "Food supply (kcal/capita/day)"
    df = df[df["Element"].astype(str).str.strip().str.lower() == elem.lower()]
    df["M49_fmt"] = df["M49_Country_Code"].apply(_format_m49)
    df["Y2020"] = pd.to_numeric(df["Y2020"], errors="coerce")
    df = df.dropna(subset=["M49_fmt", "Item"])
    df = df.rename(columns={"Item": "Item_Nutrition_Map", "Y2020": "nutrition_y2020"})
    return df[["M49_fmt", "M49_Country_Code", "Area", "Item_Nutrition_Map", "nutrition_y2020"]]


def main() -> None:
    input_base = Path(get_input_base())
    src_base = Path(get_src_base())
    output_base = Path(get_results_base())

    parser = argparse.ArgumentParser(description="Check nutrition demand vs production for 2020.")
    parser.add_argument(
        "--production",
        default=str(
            input_base
            / "Production_Trade"
            / "Production_Crops_Livestock_E_All_Data_NOFLAG_yield_refilled_baseYearFilled.csv"
        ),
    )
    parser.add_argument(
        "--nutrition",
        default=str(input_base / "Driver" / "Nutrition" / "Nutrition_profile_rescaled.xlsx"),
    )
    parser.add_argument("--dict", default=str(src_base / "dict_v3.xlsx"))
    parser.add_argument(
        "--output",
        default=str(output_base / "Nutrition_production_check_2020.xlsx"),
    )
    args = parser.parse_args()

    map_df = _load_mapping(Path(args.dict))
    prod_df = _load_production(Path(args.production))
    nut_df = _load_nutrition(Path(args.nutrition))

    merged = nut_df.merge(map_df, on="Item_Nutrition_Map", how="inner")
    merged = merged.merge(
        prod_df,
        on=["M49_fmt", "Item_Production_Map"],
        how="left",
    )
    merged["production_y2020"] = merged["production_y2020"].fillna(0.0)

    mask = (merged["nutrition_y2020"] > 0) & (merged["production_y2020"] <= 0)
    out = merged.loc[mask].copy()
    out = out.sort_values(["M49_fmt", "Item_Nutrition_Map", "Item_Production_Map"])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_excel(out_path, index=False)
    print(f"Wrote {len(out)} rows to {out_path}")


if __name__ == "__main__":
    main()
