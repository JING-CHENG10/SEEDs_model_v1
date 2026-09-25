from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


TARGET_ELEMENTS = [
    "Manure applied ratio",
    "Manure left_pasture ratio",
    "Manure management ratio",
]


def _norm_m49(val: object) -> str:
    if pd.isna(val):
        return ""
    s = str(val).strip().lstrip("'\"")
    s = s.strip()
    if not s:
        return ""
    if s.count('.') == 1:
        left, right = s.split('.', 1)
        if left.isdigit() and right.strip('0') == '':
            s = left
    if s.isdigit():
        return f"'{s.zfill(3)}"
    return f"'{s}"


def _load_valid_m49_and_region(dict_v3_path: Path) -> Tuple[List[str], Dict[str, Optional[str]], Dict[str, str]]:
    region_df = pd.read_excel(dict_v3_path, sheet_name="region")
    region_df.columns = [str(c).strip() for c in region_df.columns]
    valid_mask = region_df["Region_label_new"].astype(str).str.lower() != "no"
    region_df = region_df[valid_mask].copy()
    region_df["M49_norm"] = region_df["M49_Country_Code"].apply(_norm_m49)
    region_df["Region_market_full"] = region_df["Region_market_full"].apply(
        lambda x: str(x).strip() if pd.notna(x) else None
    )
    m49_to_region = region_df.set_index("M49_norm")["Region_market_full"].to_dict()
    m49_to_country = region_df.set_index("M49_norm")["Country"].astype(str).to_dict()
    valid_m49 = sorted([m for m in region_df["M49_norm"].unique() if m])
    return valid_m49, m49_to_region, m49_to_country


def _load_items_to_process(dict_v3_path: Path) -> Tuple[List[str], Dict[str, str]]:
    emis_df = pd.read_excel(dict_v3_path, sheet_name="Emis_item")
    emis_df.columns = [str(c).strip() for c in emis_df.columns]
    cat3 = emis_df["Item_Cat3"].astype(str).str.strip()
    keep_mask = ~cat3.str.lower().isin({"non-food", "no", "crop", "other"})
    emis_df = emis_df[keep_mask].copy()
    emis_df = emis_df.dropna(subset=["Item_Emis", "Item_Cat3"])
    emis_df["Item_Emis"] = emis_df["Item_Emis"].astype(str).str.strip()
    emis_df["Item_Cat3"] = emis_df["Item_Cat3"].astype(str).str.strip()
    item_to_cat3 = (
        emis_df[["Item_Emis", "Item_Cat3"]]
        .drop_duplicates()
        .set_index("Item_Emis")["Item_Cat3"]
        .to_dict()
    )
    items = sorted(item_to_cat3.keys())
    return items, item_to_cat3


def _build_means(
    df: pd.DataFrame,
    year_cols: List[str],
    item_to_cat3: Dict[str, str],
    m49_to_region: Dict[str, Optional[str]],
) -> Dict[str, pd.DataFrame]:
    df = df.copy()
    df["Region"] = df["M49_norm"].map(m49_to_region)
    df["Item_Cat3"] = df["Item"].map(item_to_cat3)
    for c in year_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    means = {}
    means["region_item"] = (
        df.groupby(["Region", "Item", "Element"], as_index=False)[year_cols].mean()
    )
    means["country_cat"] = (
        df.groupby(["M49_norm", "Item_Cat3", "Element"], as_index=False)[year_cols].mean()
    )
    means["region_cat"] = (
        df.groupby(["Region", "Item_Cat3", "Element"], as_index=False)[year_cols].mean()
    )
    means["global_item"] = (
        df.groupby(["Item", "Element"], as_index=False)[year_cols].mean()
    )
    return means


def _fill_with_means(
    df: pd.DataFrame,
    mean_df: pd.DataFrame,
    on: List[str],
    year_cols: List[str],
    suffix: str,
) -> pd.DataFrame:
    if mean_df.empty:
        return df
    mean_cols = {c: f"{c}{suffix}" for c in year_cols}
    tmp = mean_df.rename(columns=mean_cols)
    df = df.merge(tmp, how="left", on=on)
    for c in year_cols:
        df[c] = df[c].fillna(df[f"{c}{suffix}"])
    df = df.drop(columns=list(mean_cols.values()))
    return df


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    dict_v3_path = (base_dir / ".." / ".." / "src" / "dict_v3.xlsx").resolve()
    raw_path = (
        base_dir
        / ".."
        / ".."
        / "input"
        / "Manure_Stock"
        / "retired-unused-raw"
        / "Environment_LivestockManure_with_ratio.csv"
    ).resolve()
    out_path = (
        base_dir
        / ".."
        / ".."
        / "input"
        / "Manure_Stock"
        / "Environment_LivestockManure_with_ratio.csv"
    ).resolve()

    if not dict_v3_path.exists():
        raise FileNotFoundError(f"dict_v3 not found: {dict_v3_path}")
    if not raw_path.exists():
        raise FileNotFoundError(f"input file not found: {raw_path}")

    valid_m49, m49_to_region, m49_to_country = _load_valid_m49_and_region(dict_v3_path)
    items, item_to_cat3 = _load_items_to_process(dict_v3_path)

    df = pd.read_csv(raw_path)
    df.columns = [str(c).strip() for c in df.columns]
    year_cols = [c for c in df.columns if c.startswith("Y") and c[1:].isdigit()]
    for col in ["M49_Country_Code", "Item", "Element"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["Element"] = df["Element"].astype(str).str.strip()
    df["Item"] = df["Item"].astype(str).str.strip()

    df_target_all = df[df["Element"].isin(TARGET_ELEMENTS)].copy()
    df_other = df[~df["Element"].isin(TARGET_ELEMENTS)].copy()

    df_target_all["M49_norm"] = df_target_all["M49_Country_Code"].apply(_norm_m49)
    df_target_process = df_target_all[
        df_target_all["M49_norm"].isin(valid_m49) & df_target_all["Item"].isin(items)
    ].copy()
    df_target_keep = df_target_all.drop(df_target_process.index).copy()

    means = _build_means(df_target_process, year_cols, item_to_cat3, m49_to_region)

    meta_cols = [c for c in df.columns if c not in year_cols]
    item_code_map = (
        df.dropna(subset=["Item Code"])
        .groupby("Item")["Item Code"]
        .first()
        .to_dict()
    )
    item_cpc_map = (
        df.dropna(subset=["Item Code (CPC)"])
        .groupby("Item")["Item Code (CPC)"]
        .first()
        .to_dict()
    )
    element_code_map = (
        df.dropna(subset=["Element Code"])
        .groupby("Element")["Element Code"]
        .first()
        .to_dict()
    )
    unit_map = (
        df.dropna(subset=["Unit"])
        .groupby("Element")["Unit"]
        .first()
        .to_dict()
    )
    area_map = (
        df.dropna(subset=["Area"])
        .assign(M49_norm=lambda x: x["M49_Country_Code"].apply(_norm_m49))
        .groupby("M49_norm")["Area"]
        .first()
        .to_dict()
    )

    combos = pd.MultiIndex.from_product([valid_m49, items], names=["M49_norm", "Item"]).to_frame(index=False)
    combos["Region"] = combos["M49_norm"].map(m49_to_region)
    combos["Item_Cat3"] = combos["Item"].map(item_to_cat3)

    filled_list = []
    for elem in TARGET_ELEMENTS:
        elem_existing = df_target_process[df_target_process["Element"] == elem].copy()
        elem_existing["M49_norm"] = elem_existing["M49_Country_Code"].apply(_norm_m49)
        elem_existing = elem_existing[["M49_norm"] + df.columns.tolist()]

        merged = combos.merge(elem_existing, how="left", on=["M49_norm", "Item"])
        merged["Element"] = elem
        merged["M49_Country_Code"] = merged["M49_Country_Code"].fillna(merged["M49_norm"])

        merged["Area"] = merged["Area"].fillna(merged["M49_norm"].map(area_map))
        merged["Area"] = merged["Area"].fillna(merged["M49_norm"].map(m49_to_country))
        merged["Item Code"] = merged["Item Code"].fillna(merged["Item"].map(item_code_map))
        merged["Item Code (CPC)"] = merged["Item Code (CPC)"].fillna(merged["Item"].map(item_cpc_map))
        elem_code = element_code_map.get(elem)
        if elem_code is not None and pd.notna(elem_code):
            merged["Element Code"] = merged["Element Code"].fillna(elem_code)
        unit_val = unit_map.get(elem)
        if unit_val is not None and pd.notna(unit_val):
            merged["Unit"] = merged["Unit"].fillna(unit_val)

        merged = _fill_with_means(
            merged,
            means["region_item"],
            on=["Region", "Item", "Element"],
            year_cols=year_cols,
            suffix="_mean_ri",
        )
        merged = _fill_with_means(
            merged,
            means["country_cat"],
            on=["M49_norm", "Item_Cat3", "Element"],
            year_cols=year_cols,
            suffix="_mean_cc",
        )
        merged = _fill_with_means(
            merged,
            means["region_cat"],
            on=["Region", "Item_Cat3", "Element"],
            year_cols=year_cols,
            suffix="_mean_rc",
        )
        merged = _fill_with_means(
            merged,
            means["global_item"],
            on=["Item", "Element"],
            year_cols=year_cols,
            suffix="_mean_gi",
        )

        merged = merged[meta_cols + year_cols].copy()
        filled_list.append(merged)

    df_filled = pd.concat(filled_list, ignore_index=True)
    out_df = pd.concat([df_other, df_target_keep, df_filled], ignore_index=True)
    out_df = out_df.sort_values(["M49_Country_Code", "Item", "Element"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[OK] wrote {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    main()
