from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from config_paths import get_input_base, get_src_base

BASE_YEAR = 2020
LOW_LAND_NEW_RUMINANT_SHARE_CAPS = list(range(0, 101))
RUMINANT_INCREASE_CAPS = list(range(5, 85, 5))
RUMINANT_REDUCTION_CAPS = list(range(5, 105, 5))
RUMINANT_ITEMS = [
    "Mutton & Goat Meat-sheep",
    "Milk-buffalo",
    "Milk-cattle",
    "Bovine Meat-cattle",
    "Milk-goats",
    "Mutton & Goat Meat-goat",
    "Bovine Meat-buffalo",
    "Milk-camel",
    "Meat, Other-camels",
    "Meat, Other-other domestic camelids",
    "Milk-sheep",
]
FISH_ITEMS = ["Fish, Seafood"]
GENERATED_PREFIXES = ("Ruminate_Cap",)
GENERATED_COLUMNS = {"EAT_LANCET", "EAT_LANCET_recal", "EAT_LANCET_hisBase"}
LOW_LAND_CROP_SUPPLEMENT_CAP = 0.01
LOW_LAND_CROP_CUT_CAP = 1.0


def _pick_country_col(df: pd.DataFrame) -> str:
    for col in ["M49_Country_Code", "Area Code (M49)", "M49"]:
        if col in df.columns:
            return col
    raise ValueError("Missing country column (M49_Country_Code or Area Code (M49)).")


def _resolve_base_col(df: pd.DataFrame, base_year: int) -> str:
    col = f"Y{base_year}"
    if col in df.columns:
        return col
    year_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("Y") and c[1:].isdigit()]
    if not year_cols:
        raise ValueError("No year columns (Yxxxx) found in nutrition profile.")
    year_cols = sorted(year_cols, key=lambda x: int(x[1:]))
    return year_cols[-1]


def _load_eat_lancet_vector(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing EAT-Lancet vector: {path}")
    df = pd.read_excel(path, sheet_name="vector")
    df.columns = [str(c).strip() for c in df.columns]
    needed = [
        "Item_Nutrition_Map",
        "kcal_share_frac",
        "protein_share_frac",
        "fat_share_frac",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"EAT_Lancet vector missing columns: {missing}")
    df["Item_Nutrition_Map"] = df["Item_Nutrition_Map"].astype(str).str.strip()
    df = df.dropna(subset=["Item_Nutrition_Map"]).drop_duplicates(subset=["Item_Nutrition_Map"])
    for col in ["kcal_share_frac", "protein_share_frac", "fat_share_frac"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["kcal_share_frac", "protein_share_frac", "fat_share_frac"]].isna().any().any():
        bad = df[df[["kcal_share_frac", "protein_share_frac", "fat_share_frac"]].isna().any(axis=1)]
        sample = bad["Item_Nutrition_Map"].head(10).tolist()
        raise ValueError(f"EAT_Lancet vector has NaN shares for items: {sample}")
    return df[needed]


def _normalize_column_name(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def _match_required_column(columns: List[str], target: str) -> Optional[str]:
    target_norm = _normalize_column_name(target)
    for col in columns:
        if _normalize_column_name(col) == target_norm:
            return col
    return None


def _find_share_column(columns: List[str], nutrient: str) -> str:
    candidates = [
        f"{nutrient}_share_frac",
        f"{nutrient}_share",
        f"{nutrient}_ratio",
        f"{nutrient}_fraction",
    ]
    for cand in candidates:
        col = _match_required_column(columns, cand)
        if col:
            return col
    matches = []
    for col in columns:
        norm = _normalize_column_name(col)
        if nutrient in norm and any(k in norm for k in ("share", "ratio", "frac")):
            matches.append(col)
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Missing {nutrient} share column in EAT_LANCET_profile: {columns}")


def _load_eat_lancet_group_profile(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Missing EAT-Lancet profile: {path}")

    mapping_df = pd.read_excel(path, sheet_name="Nutrition_mapping")
    mapping_df.columns = [str(c).strip() for c in mapping_df.columns]
    map_item_col = _match_required_column(mapping_df.columns, "Item_Nutrition_Map")
    map_group_col = _match_required_column(mapping_df.columns, "EAT_group")
    if not map_item_col or not map_group_col:
        raise ValueError("Nutrition_mapping missing Item_Nutrition_Map or EAT_group.")
    mapping_df = mapping_df[[map_item_col, map_group_col]].copy()
    mapping_df.columns = ["Item_Nutrition_Map", "EAT_group"]
    mapping_df["Item_Nutrition_Map"] = mapping_df["Item_Nutrition_Map"].astype(str).str.strip()
    mapping_df["EAT_group"] = mapping_df["EAT_group"].astype(str).str.strip()
    mapping_df = mapping_df.dropna(subset=["Item_Nutrition_Map", "EAT_group"])
    mapping_df = mapping_df[mapping_df["Item_Nutrition_Map"] != ""]
    dup = mapping_df.groupby("Item_Nutrition_Map")["EAT_group"].nunique()
    if (dup > 1).any():
        bad = dup[dup > 1].index.tolist()[:10]
        raise ValueError(f"Nutrition_mapping has items mapped to multiple groups: {bad}")
    mapping_df = mapping_df.drop_duplicates(subset=["Item_Nutrition_Map"])

    profile_df = pd.read_excel(path, sheet_name="EAT_LANCET_profile")
    profile_df.columns = [str(c).strip() for c in profile_df.columns]
    prof_group_col = _match_required_column(profile_df.columns, "EAT_group")
    if not prof_group_col:
        raise ValueError("EAT_LANCET_profile missing EAT_group column.")
    kcal_col = _find_share_column(profile_df.columns, "kcal")
    protein_col = _find_share_column(profile_df.columns, "protein")
    fat_col = _find_share_column(profile_df.columns, "fat")
    profile_df = profile_df[[prof_group_col, kcal_col, protein_col, fat_col]].copy()
    profile_df.columns = ["EAT_group", "kcal_share_frac", "protein_share_frac", "fat_share_frac"]
    profile_df["EAT_group"] = profile_df["EAT_group"].astype(str).str.strip()
    profile_df = profile_df.dropna(subset=["EAT_group"])
    if profile_df["EAT_group"].duplicated().any():
        dup_groups = profile_df.loc[profile_df["EAT_group"].duplicated(), "EAT_group"].tolist()[:10]
        raise ValueError(f"EAT_LANCET_profile has duplicated groups: {dup_groups}")
    for col in ["kcal_share_frac", "protein_share_frac", "fat_share_frac"]:
        profile_df[col] = pd.to_numeric(profile_df[col], errors="coerce")
    if profile_df[["kcal_share_frac", "protein_share_frac", "fat_share_frac"]].isna().any().any():
        bad = profile_df[profile_df[["kcal_share_frac", "protein_share_frac", "fat_share_frac"]].isna().any(axis=1)]
        sample = bad["EAT_group"].head(10).tolist()
        raise ValueError(f"EAT_LANCET_profile has NaN shares for groups: {sample}")
    for col in ["kcal_share_frac", "protein_share_frac", "fat_share_frac"]:
        total = float(profile_df[col].sum())
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"EAT_LANCET_profile share sum for {col} is {total:.6f}, expected 1.0")

    return mapping_df, profile_df


def _read_source_profile(path: Path) -> pd.DataFrame:
    """Read the workbook even after generated sheets replace the legacy sheet."""
    errors = []
    for sheet in ["nutrition_profile", "current_mix", "low_land", 0]:
        try:
            return pd.read_excel(path, sheet_name=sheet)
        except Exception as exc:
            errors.append(f"{sheet}: {exc}")
    raise ValueError(f"Could not read nutrition profile from {path}. Tried: {'; '.join(errors)}")


def _strip_generated_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = []
    for col in df.columns:
        name = str(col).strip()
        if name in GENERATED_COLUMNS or any(name.startswith(prefix) for prefix in GENERATED_PREFIXES):
            drop_cols.append(col)
    return df.drop(columns=drop_cols, errors="ignore")


def _is_blank_map_value(value: object) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True
    text = str(value).strip()
    return not text or text.lower() in {"nan", "none", "no"}


def _fallback_land_score(item: str, cat2: str = "", cat3: str = "") -> float:
    item_l = str(item).strip().lower()
    cat2_l = str(cat2).strip().lower()
    cat3_l = str(cat3).strip().lower()
    if item in FISH_ITEMS or "fish" in item_l or "fish" in cat3_l:
        return 0.0
    if cat2_l == "crop" or "crop" in cat3_l:
        return 1.0
    if item_l == "eggs" or "poultry" in item_l or "poultry" in cat3_l:
        return 4.0
    if item_l == "pigmeat" or "swine" in cat3_l:
        return 5.0
    return 8.0


def _load_low_land_item_scores(profile_items: List[str], base_year: int) -> Dict[str, float]:
    """Lower score means the item is used/cut first in low_land profiles.

    For crop items the score is global 2020 harvested area per kcal produced
    from the model's FAOSTAT production table. Fish is treated as zero direct
    crop/pasture land in this profile generator. Non-ruminant animal products
    without item-level area rows fall back behind direct crop foods.
    """
    profile_set = {str(i).strip() for i in profile_items if str(i).strip()}
    scores: Dict[str, float] = {item: _fallback_land_score(item) for item in profile_set}
    for item in FISH_ITEMS:
        if item in profile_set:
            scores[item] = 0.0

    dict_path = Path(get_src_base()) / "dict_v3.xlsx"
    prod_path = (
        Path(get_input_base())
        / "Production_Trade"
        / "Production_Crops_Livestock_E_All_Data_NOFLAG_yield_refilled_baseYearFilled.csv"
    )
    year_col = f"Y{base_year}"
    if not dict_path.exists() or not prod_path.exists():
        return scores

    try:
        map_df = pd.read_excel(
            dict_path,
            sheet_name="Emis_item",
            usecols=[
                "Item_Nutrition_Map",
                "Item_Production_Map",
                "Item_Production_Element",
                "Item_Area_Map",
                "Item_Area_Element",
                "Item_Cat2",
                "Item_Cat3",
                "kcal_per_100g",
            ],
        )
        map_df.columns = [str(c).strip() for c in map_df.columns]
        map_df["Item_Nutrition_Map"] = map_df["Item_Nutrition_Map"].astype(str).str.strip()
        map_df = map_df[map_df["Item_Nutrition_Map"].isin(profile_set)].copy()
        for _, row in map_df.iterrows():
            item = str(row.get("Item_Nutrition_Map", "")).strip()
            if item and scores.get(item, 8.0) >= 1.0:
                scores[item] = _fallback_land_score(
                    item,
                    str(row.get("Item_Cat2", "")),
                    str(row.get("Item_Cat3", "")),
                )

        prod_df = pd.read_csv(prod_path, usecols=["Item", "Element", year_col])
        prod_df["Item"] = prod_df["Item"].astype(str).str.strip()
        prod_df["Element"] = prod_df["Element"].astype(str).str.strip()
        prod_df[year_col] = pd.to_numeric(prod_df[year_col], errors="coerce").fillna(0.0)
        global_vals = prod_df.groupby(["Item", "Element"], dropna=False)[year_col].sum()

        crop_scores: Dict[str, List[float]] = {}
        for _, row in map_df.iterrows():
            item = str(row.get("Item_Nutrition_Map", "")).strip()
            if not item or item in FISH_ITEMS:
                continue
            area_map = row.get("Item_Area_Map")
            area_elem = row.get("Item_Area_Element")
            prod_map = row.get("Item_Production_Map")
            prod_elem = row.get("Item_Production_Element", "Production")
            kcal_per_100g = pd.to_numeric(row.get("kcal_per_100g"), errors="coerce")
            if (
                _is_blank_map_value(area_map)
                or _is_blank_map_value(area_elem)
                or _is_blank_map_value(prod_map)
                or _is_blank_map_value(prod_elem)
                or pd.isna(kcal_per_100g)
                or float(kcal_per_100g) <= 0
            ):
                continue
            area_val = float(global_vals.get((str(area_map).strip(), str(area_elem).strip()), 0.0) or 0.0)
            prod_val = float(global_vals.get((str(prod_map).strip(), str(prod_elem).strip()), 0.0) or 0.0)
            if area_val <= 0 or prod_val <= 0:
                continue
            kcal_total = prod_val * float(kcal_per_100g) * 10000.0
            if kcal_total <= 0:
                continue
            crop_scores.setdefault(item, []).append(area_val / kcal_total)

        positive_crop_scores = [min(v) for v in crop_scores.values() if v]
        crop_scale = float(np.median(positive_crop_scores)) if positive_crop_scores else 1.0
        if crop_scale <= 0 or not np.isfinite(crop_scale):
            crop_scale = 1.0
        for item, vals in crop_scores.items():
            if vals:
                scores[item] = min(vals)
        animal_floor = max(positive_crop_scores) * 1.5 if positive_crop_scores else crop_scale * 4.0
        for item, score in list(scores.items()):
            if item in FISH_ITEMS:
                continue
            if score >= 4.0:
                scores[item] = animal_floor * score
    except Exception as exc:
        print(f"[WARN] low_land score fallback used where needed: {exc}")
    return scores


def _load_low_land_crop_items(profile_items: List[str]) -> Set[str]:
    """Foods eligible for the 1% low_land pre-fish adjustment.

    The pool includes direct crops plus non-ruminant pig/poultry/egg items.
    Fish is handled separately as the residual supplement / first-cut item.
    """
    profile_set = {str(i).strip() for i in profile_items if str(i).strip()}
    dict_path = Path(get_src_base()) / "dict_v3.xlsx"
    if not dict_path.exists():
        return _fallback_crop_items(profile_set)
    try:
        map_df = pd.read_excel(
            dict_path,
            sheet_name="Emis_item",
            usecols=["Item_Nutrition_Map", "Item_Cat2", "Item_Cat3", "Item_Area_Map"],
        )
        map_df.columns = [str(c).strip() for c in map_df.columns]
        map_df["Item_Nutrition_Map"] = map_df["Item_Nutrition_Map"].astype(str).str.strip()
        map_df = map_df[map_df["Item_Nutrition_Map"].isin(profile_set)].copy()
        crop_items: Set[str] = set()
        for _, row in map_df.iterrows():
            item = str(row.get("Item_Nutrition_Map", "")).strip()
            if not item or item in FISH_ITEMS or item in RUMINANT_ITEMS:
                continue
            cat2 = str(row.get("Item_Cat2", "")).strip().lower()
            cat3 = str(row.get("Item_Cat3", "")).strip().lower()
            area_map = row.get("Item_Area_Map")
            if (
                cat2 == "crop"
                or "crop" in cat3
                or not _is_blank_map_value(area_map)
                or _is_pig_poultry_egg_item(item, cat2, cat3)
            ):
                crop_items.add(item)
        return crop_items if crop_items else _fallback_crop_items(profile_set)
    except Exception as exc:
        print(f"[WARN] low_land crop item fallback used: {exc}")
        return _fallback_crop_items(profile_set)


def _is_pig_poultry_egg_item(item: str, cat2: str = "", cat3: str = "") -> bool:
    item_l = str(item).strip().lower()
    cat2_l = str(cat2).strip().lower()
    cat3_l = str(cat3).strip().lower()
    return (
        item_l == "eggs"
        or item_l == "pigmeat"
        or "poultry" in item_l
        or "egg" in item_l
        or "swine" in cat3_l
        or "poultry" in cat3_l
        or (cat2_l == "dairy" and item_l == "eggs")
    )


def _fallback_crop_items(profile_set: Set[str]) -> Set[str]:
    animal_tokens = (
        "meat",
        "milk",
        "eggs",
        "fish",
        "seafood",
        "pig",
        "poultry",
        "cattle",
        "bovine",
        "mutton",
        "goat",
        "buffalo",
        "camel",
        "sheep",
    )
    return {
        item
        for item in profile_set
        if item not in RUMINANT_ITEMS
        and item not in FISH_ITEMS
        and (
            not any(tok in item.lower() for tok in animal_tokens)
            or _is_pig_poultry_egg_item(item)
        )
    }


def _allocated_group_sum(df: pd.DataFrame, group_cols: List[str], values: np.ndarray) -> np.ndarray:
    return (
        pd.Series(values, index=df.index)
        .groupby([df[col] for col in group_cols], dropna=False)
        .transform("sum")
        .to_numpy(dtype=float)
    )


def _priority_allocate(
    df: pd.DataFrame,
    group_cols: List[str],
    need_vals: np.ndarray,
    capacity_vals: np.ndarray,
    eligible_mask: np.ndarray,
    score_vals: np.ndarray,
) -> np.ndarray:
    allocation = np.zeros(len(df), dtype=float)
    alloc_df = df[group_cols].copy()
    alloc_df["_pos"] = np.arange(len(df), dtype=int)
    alloc_df["_need"] = np.nan_to_num(need_vals, nan=0.0, posinf=0.0, neginf=0.0)
    alloc_df["_capacity"] = np.nan_to_num(capacity_vals, nan=0.0, posinf=np.inf, neginf=0.0)
    alloc_df["_eligible"] = eligible_mask
    alloc_df["_score"] = np.nan_to_num(score_vals, nan=np.inf, posinf=np.inf, neginf=0.0)
    for _, group in alloc_df.groupby(group_cols, dropna=False, sort=False):
        need = float(group["_need"].max())
        if need <= 0:
            continue
        remaining = need
        eligible = group[(group["_eligible"]) & (group["_capacity"] > 0)].sort_values(
            ["_score", "_pos"],
            kind="mergesort",
        )
        for _, row in eligible.iterrows():
            take = min(float(row["_capacity"]), remaining)
            if take <= 0:
                continue
            allocation[int(row["_pos"])] = take
            remaining -= take
            if remaining <= 1e-12:
                break
    return allocation


def _build_eat_lancet_hisbase(
    df: pd.DataFrame,
    base_vals: pd.Series,
    total_vals: np.ndarray,
    *,
    country_col: str,
    eat_mapping: pd.DataFrame,
    eat_profile: pd.DataFrame,
) -> np.ndarray:
    item_series = df["Item"].astype(str).str.strip()
    mapping_map = dict(zip(eat_mapping["Item_Nutrition_Map"], eat_mapping["EAT_group"]))
    group_series = item_series.map(mapping_map)
    if group_series.isna().any():
        missing_items = sorted(set(item_series[group_series.isna()]))[:10]
        raise ValueError(f"EAT_LANCET mapping missing items: {missing_items}")
    group_work = pd.DataFrame(
        {
            "_country": df[country_col].to_numpy(),
            "_element": df["Element"].to_numpy(),
            "_eat_group": group_series.to_numpy(),
            "_base": base_vals.to_numpy(dtype=float),
        },
        index=df.index,
    )

    element_series = df["Element"].astype(str).str.strip()
    element_map = {
        "Food supply (kcal/capita/day)": "kcal_share_frac",
        "Protein supply quantity (g/capita/day)": "protein_share_frac",
        "Fat supply quantity (g/capita/day)": "fat_share_frac",
    }
    unknown_elements = sorted(set(element_series.unique()) - set(element_map))
    if unknown_elements:
        raise ValueError(f"Unknown Element values in nutrition profile: {unknown_elements}")

    share_df = eat_profile.set_index("EAT_group")
    group_share_vals = np.zeros(len(df))
    for elem, col in element_map.items():
        mask = element_series == elem
        if not mask.any():
            continue
        groups = group_series[mask]
        try:
            group_share_vals[mask] = share_df.loc[groups, col].to_numpy()
        except KeyError:
            missing_groups = sorted(set(groups) - set(share_df.index))
            raise ValueError(f"EAT_LANCET_profile missing groups: {missing_groups[:10]}")

    group_keys = ["_country", "_element", "_eat_group"]
    group_base = group_work.groupby(group_keys, dropna=False)["_base"].transform("sum")
    group_count = group_work.groupby(group_keys, dropna=False)["_base"].transform("count")
    base_arr = base_vals.to_numpy()
    base_share = np.where(group_base > 0, base_arr / group_base, 0.0)
    fallback_share = np.where(group_count > 0, 1.0 / group_count, 0.0)
    item_share = np.where(group_base > 0, base_share, fallback_share)
    group_share_use = np.where(group_base > 0, group_share_vals, 0.0)
    group_share_series = pd.Series(group_share_use, index=df.index)
    group_share_by_group = group_share_series.groupby(
        [group_work["_country"], group_work["_element"], group_work["_eat_group"]], dropna=False
    ).first()
    group_share_sum = group_share_by_group.groupby(level=[0, 1], dropna=False).sum()
    group_share_sum_vals = group_share_sum.reindex(
        pd.MultiIndex.from_frame(group_work[["_country", "_element"]])
    ).to_numpy()
    group_share_norm = np.where(group_share_sum_vals > 0, group_share_use / group_share_sum_vals, 0.0)
    return total_vals * group_share_norm * item_share


def _build_scenario_columns(
    df: pd.DataFrame,
    base_col: str,
    *,
    ruminate_strategy: str = "current_mix",
    ruminate_cap_mode: str = "relative_change",
    ruminate_share_caps: Optional[List[int]] = None,
    land_intensity_scores: Optional[Dict[str, float]] = None,
    low_land_crop_items: Optional[Set[str]] = None,
    eat_vector: Optional[pd.DataFrame] = None,
    eat_mapping: Optional[pd.DataFrame] = None,
    eat_profile: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "Item" not in df.columns or "Element" not in df.columns:
        raise ValueError("Missing required columns: Item or Element.")

    country_col = _pick_country_col(df)
    group_cols = [country_col, "Element"]

    item_series = df["Item"].astype(str).str.strip()
    base_vals = pd.to_numeric(df[base_col], errors="coerce").fillna(0.0)
    base_arr = base_vals.to_numpy(dtype=float)
    element_series = df["Element"].astype(str).str.strip()
    element_map = {
        "Food supply (kcal/capita/day)": "kcal_share_frac",
        "Protein supply quantity (g/capita/day)": "protein_share_frac",
        "Fat supply quantity (g/capita/day)": "fat_share_frac",
    }

    df["_is_rumi"] = item_series.isin(RUMINANT_ITEMS)
    df["_base"] = base_vals
    strategy = str(ruminate_strategy or "current_mix").strip().lower()
    if strategy not in {"current_mix", "low_land", "fish_supp", "no_supp"}:
        raise ValueError(f"Unknown ruminate_strategy: {ruminate_strategy}")
    cap_mode = str(ruminate_cap_mode or "relative_change").strip().lower()
    if cap_mode not in {"relative_change", "absolute_share"}:
        raise ValueError(f"Unknown ruminate_cap_mode: {ruminate_cap_mode}")
    if cap_mode == "absolute_share" and strategy != "low_land":
        raise ValueError("absolute_share cap mode is currently defined only for low_land strategy.")

    total_by_group = df.groupby(group_cols, dropna=False)["_base"].sum()
    rumi_by_group = df[df["_is_rumi"]].groupby(group_cols, dropna=False)["_base"].sum()

    idx = pd.MultiIndex.from_frame(df[group_cols])
    total_vals = total_by_group.reindex(idx).to_numpy()
    rumi_vals = rumi_by_group.reindex(idx).to_numpy()
    rumi_vals = np.nan_to_num(rumi_vals, nan=0.0)
    other_vals = total_vals - rumi_vals
    other_share = np.zeros(len(df), dtype=float)
    np.divide(base_arr, other_vals, out=other_share, where=other_vals > 0)
    is_rumi = df["_is_rumi"].to_numpy(dtype=bool)
    is_other = ~is_rumi
    fish_mask = item_series.isin(FISH_ITEMS).to_numpy(dtype=bool)
    other_nonfish_mask = is_other & (~fish_mask)
    crop_mask = item_series.isin(low_land_crop_items or set()).to_numpy(dtype=bool) & other_nonfish_mask
    add_capacity_current = np.where(
        fish_mask,
        np.inf,
        np.where(is_other, base_arr * 0.75, 0.0),
    )
    add_capacity_low_land_crop = np.where(
        crop_mask,
        base_arr * LOW_LAND_CROP_SUPPLEMENT_CAP,
        0.0,
    )
    cut_capacity_low_land_crop = np.where(
        crop_mask,
        base_arr * LOW_LAND_CROP_CUT_CAP,
        0.0,
    )
    cut_capacity_fish_supp = np.where(
        fish_mask,
        base_arr,
        np.where(other_nonfish_mask, base_arr * 0.5, 0.0),
    )
    score_default = max((land_intensity_scores or {"_": 8.0}).values()) * 10.0
    score_vals = item_series.map(lambda item: (land_intensity_scores or {}).get(item, score_default)).to_numpy(
        dtype=float
    )

    def _supplement_after_ruminant_reduction(reduction_vals: np.ndarray) -> np.ndarray:
        if strategy == "current_mix":
            return np.minimum(reduction_vals * other_share, add_capacity_current)
        if strategy == "low_land":
            crop_add = _priority_allocate(
                df,
                group_cols,
                reduction_vals,
                add_capacity_low_land_crop,
                crop_mask,
                score_vals,
            )
            fish_need = np.maximum(
                reduction_vals - _allocated_group_sum(df, group_cols, crop_add),
                0.0,
            )
            fish_add = _priority_allocate(
                df,
                group_cols,
                fish_need,
                np.where(fish_mask, np.inf, 0.0),
                fish_mask,
                score_vals,
            )
            return crop_add + fish_add
        if strategy == "fish_supp":
            return _priority_allocate(
                df,
                group_cols,
                reduction_vals,
                np.where(fish_mask, np.inf, 0.0),
                fish_mask,
                score_vals,
            )
        return np.zeros(len(df), dtype=float)

    def _cut_after_ruminant_increase(increase_vals: np.ndarray) -> np.ndarray:
        if strategy == "current_mix":
            return base_arr * np.where(
                other_vals > 0,
                np.minimum(increase_vals / other_vals, np.where(fish_mask, 1.0, 0.5)),
                0.0,
            )
        if strategy == "low_land":
            fish_cut = _priority_allocate(
                df,
                group_cols,
                increase_vals,
                np.where(fish_mask, base_arr, 0.0),
                fish_mask,
                score_vals,
            )
            crop_need = np.maximum(
                increase_vals - _allocated_group_sum(df, group_cols, fish_cut),
                0.0,
            )
            crop_cut = _priority_allocate(
                df,
                group_cols,
                crop_need,
                cut_capacity_low_land_crop,
                crop_mask,
                score_vals,
            )
            return fish_cut + crop_cut
        if strategy == "fish_supp":
            return _priority_allocate(
                df,
                group_cols,
                increase_vals,
                cut_capacity_fish_supp,
                is_other,
                score_vals,
            )
        return np.zeros(len(df), dtype=float)

    def _ruminant_distribution() -> np.ndarray:
        direct_share = np.zeros(len(df), dtype=float)
        np.divide(base_arr, rumi_vals, out=direct_share, where=(is_rumi) & (rumi_vals > 0))
        rumi_df = df.loc[is_rumi, [country_col, "Element", "Item"]].copy()
        if rumi_df.empty:
            return direct_share

        global_item = df.loc[is_rumi].groupby(["Element", "Item"], dropna=False)["_base"].sum()
        global_elem = df.loc[is_rumi].groupby("Element", dropna=False)["_base"].sum()
        fallback_raw = np.zeros(len(df), dtype=float)
        for idx_row in rumi_df.index:
            elem = df.at[idx_row, "Element"]
            item = df.at[idx_row, "Item"]
            elem_total = float(global_elem.get(elem, 0.0) or 0.0)
            if elem_total > 0:
                fallback_raw[idx_row] = float(global_item.get((elem, item), 0.0) or 0.0) / elem_total
        fallback_sum = _allocated_group_sum(df, group_cols, fallback_raw)
        fallback_share = np.zeros(len(df), dtype=float)
        np.divide(
            fallback_raw,
            fallback_sum,
            out=fallback_share,
            where=(is_rumi) & (fallback_sum > 0),
        )
        rumi_count = _allocated_group_sum(df, group_cols, is_rumi.astype(float))
        equal_share = np.zeros(len(df), dtype=float)
        np.divide(1.0, rumi_count, out=equal_share, where=(is_rumi) & (rumi_count > 0))
        fallback_share = np.where(fallback_share > 0, fallback_share, equal_share)
        return np.where(rumi_vals > 0, direct_share, fallback_share)

    scenario_columns: Dict[str, np.ndarray] = {"Ruminate_Cap00": base_arr.copy()}
    if cap_mode == "absolute_share":
        current_share = np.where(total_vals > 0, rumi_vals / total_vals, 0.0)
        cut_capacity_group = _allocated_group_sum(
            df,
            group_cols,
            np.where(fish_mask, base_arr, 0.0) + cut_capacity_low_land_crop,
        )
        rumi_distribution = _ruminant_distribution()
        cap_values = ruminate_share_caps or LOW_LAND_NEW_RUMINANT_SHARE_CAPS
        for cap in cap_values:
            target_share = max(0.0, min(float(cap) / 100.0, 0.999999))
            col_name = f"Ruminate_Cap{int(cap):02d}"
            decrease_mask = (total_vals > 0) & (target_share < current_share - 1e-12)
            increase_mask = (total_vals > 0) & (target_share > current_share + 1e-12)

            target_rumi_if_decrease = target_share * total_vals
            reduction = np.where(decrease_mask, np.maximum(rumi_vals - target_rumi_if_decrease, 0.0), 0.0)
            full_offset_increase = np.maximum(target_share * total_vals - rumi_vals, 0.0)
            increase = np.where(
                increase_mask,
                np.minimum(full_offset_increase, cut_capacity_group),
                0.0,
            )

            other_add = _supplement_after_ruminant_reduction(reduction)
            other_cut = _cut_after_ruminant_increase(increase)
            target_rumi_total = np.where(
                decrease_mask,
                rumi_vals - reduction,
                np.where(increase_mask, rumi_vals + increase, rumi_vals),
            )
            new_vals = np.where(
                is_rumi,
                target_rumi_total * rumi_distribution,
                base_arr + other_add - other_cut,
            )
            scenario_columns[col_name] = np.maximum(new_vals, 0.0)
    else:
        for cap in RUMINANT_REDUCTION_CAPS:
            rate = cap / 100.0
            reduction = rumi_vals * rate
            col_name = f"Ruminate_Cap{cap:02d}"
            # For ruminant-reduction scenarios, replace reduced ruminant intake
            # according to the sheet strategy. In low_land, low-land crops are
            # used first up to the configured Y2020 cap, then remaining intake goes to fish.
            other_add = _supplement_after_ruminant_reduction(reduction)
            new_vals = np.where(
                is_rumi,
                base_arr * (1.0 - rate),
                base_arr + other_add,
            )
            scenario_columns[col_name] = new_vals

        # Positive ruminant_reduction rates in S5 map to negative Cap columns:
        # +0.20 -> Ruminate_Cap-20. For these scenarios, ruminant items increase
        # by the requested rate. The added intake is offset by proportional cuts to
        # non-ruminant items within each country + Element group, but no non-ruminant
        # item is reduced below 50% of its Y2020 level in current_mix. In low_land,
        # fish is cut first to zero, then low-land crops are cut up to the configured Y2020 cap.
        for cap in RUMINANT_INCREASE_CAPS:
            rate = cap / 100.0
            col_name = f"Ruminate_Cap-{cap:02d}"
            rumi_increase = rumi_vals * rate
            other_cut = _cut_after_ruminant_increase(rumi_increase)
            new_vals = np.where(
                is_rumi,
                base_arr * (1.0 + rate),
                base_arr - other_cut,
            )
            scenario_columns[col_name] = new_vals

    eat_columns: Dict[str, np.ndarray] = {}
    if eat_mapping is not None and eat_profile is not None:
        eat_columns["EAT_LANCET_hisBase"] = _build_eat_lancet_hisbase(
            df,
            base_vals,
            total_vals,
            country_col=country_col,
            eat_mapping=eat_mapping,
            eat_profile=eat_profile,
        )

    if eat_vector is not None and not eat_vector.empty:
        share_df = eat_vector.copy()
        share_df["Item_Nutrition_Map"] = share_df["Item_Nutrition_Map"].astype(str).str.strip()
        share_df = share_df.set_index("Item_Nutrition_Map")

        missing_items = sorted(set(item_series) - set(share_df.index))
        if missing_items:
            raise ValueError(f"EAT_Lancet vector missing items: {missing_items[:10]}")

        share_lookup = share_df.reindex(item_series)[
            ["kcal_share_frac", "protein_share_frac", "fat_share_frac"]
        ].copy()
        share_lookup.index = df.index
        if share_lookup.isna().any().any():
            bad_items = share_lookup[share_lookup.isna().any(axis=1)].index.unique().tolist()
            raise ValueError(f"EAT_Lancet vector has NaN shares for items: {bad_items[:10]}")

        unknown_elements = sorted(set(element_series.unique()) - set(element_map))
        if unknown_elements:
            raise ValueError(f"Unknown Element values in nutrition profile: {unknown_elements}")

        share_vals = np.zeros(len(df))
        for elem, col in element_map.items():
            mask = element_series == elem
            share_vals[mask] = share_lookup.loc[mask, col].to_numpy()

        share_sums = {
            col: float(share_df[col].sum())
            for col in ["kcal_share_frac", "protein_share_frac", "fat_share_frac"]
        }
        for col, total in share_sums.items():
            if not np.isclose(total, 1.0, atol=1e-6):
                raise ValueError(f"EAT_Lancet share sum for {col} is {total:.6f}, expected 1.0")

        eat_columns["EAT_LANCET"] = total_vals * share_vals

        nonzero_mask = base_vals > 0
        share_raw = pd.Series(share_vals, index=df.index)
        share_sum = share_raw.where(nonzero_mask, 0.0).groupby(idx).transform("sum")
        share_norm = np.where((nonzero_mask) & (share_sum > 0), share_raw / share_sum, 0.0)
        eat_columns["EAT_LANCET_recal"] = total_vals * share_norm

    drop_cols = ["_is_rumi", "_base"]
    base_df = df.drop(columns=drop_cols)
    extra_frames = [pd.DataFrame(scenario_columns, index=df.index)]
    if eat_columns:
        extra_frames.append(pd.DataFrame(eat_columns, index=df.index))
    return pd.concat([base_df, *extra_frames], axis=1).copy()


def main() -> None:
    base_dir = Path(get_input_base()) / "Driver" / "Nutrition"
    input_path = base_dir / "Nutrition_profile_recalculated_fromD0_food_demand.xlsx"
    eat_path = base_dir / "EAT_Lancet_energy_share.xlsx"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing nutrition profile: {input_path}")

    df = _strip_generated_columns(_read_source_profile(input_path))
    base_col = _resolve_base_col(df, BASE_YEAR)
    eat_vector = _load_eat_lancet_vector(eat_path)
    eat_mapping, eat_profile = _load_eat_lancet_group_profile(eat_path)
    profile_items = sorted(df["Item"].dropna().astype(str).str.strip().unique().tolist())
    land_scores = _load_low_land_item_scores(profile_items, BASE_YEAR)
    low_land_crop_items = _load_low_land_crop_items(profile_items)
    sheets = {
        "low_land": _build_scenario_columns(
            df,
            base_col,
            ruminate_strategy="low_land",
            land_intensity_scores=land_scores,
            low_land_crop_items=low_land_crop_items,
            eat_vector=eat_vector,
            eat_mapping=eat_mapping,
            eat_profile=eat_profile,
        ),
        "low_land_new": _build_scenario_columns(
            df,
            base_col,
            ruminate_strategy="low_land",
            ruminate_cap_mode="absolute_share",
            ruminate_share_caps=LOW_LAND_NEW_RUMINANT_SHARE_CAPS,
            land_intensity_scores=land_scores,
            low_land_crop_items=low_land_crop_items,
            eat_vector=eat_vector,
            eat_mapping=eat_mapping,
            eat_profile=eat_profile,
        ),
        "current_mix": _build_scenario_columns(
            df,
            base_col,
            ruminate_strategy="current_mix",
            land_intensity_scores=land_scores,
            low_land_crop_items=low_land_crop_items,
            eat_vector=eat_vector,
            eat_mapping=eat_mapping,
            eat_profile=eat_profile,
        ),
        "fish_supp": _build_scenario_columns(
            df,
            base_col,
            ruminate_strategy="fish_supp",
            land_intensity_scores=land_scores,
            low_land_crop_items=low_land_crop_items,
            eat_vector=eat_vector,
            eat_mapping=eat_mapping,
            eat_profile=eat_profile,
        ),
        "no_supp": _build_scenario_columns(
            df,
            base_col,
            ruminate_strategy="no_supp",
            land_intensity_scores=land_scores,
            low_land_crop_items=low_land_crop_items,
            eat_vector=eat_vector,
            eat_mapping=eat_mapping,
            eat_profile=eat_profile,
        ),
    }

    tmp_path = input_path.with_name(f"{input_path.stem}__new{input_path.suffix}")
    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except Exception:
        pass

    with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
        for sheet_name, out_df in sheets.items():
            out_df.to_excel(writer, sheet_name=sheet_name, index=False)

    try:
        os.replace(tmp_path, input_path)
        output_path = input_path
    except PermissionError:
        output_path = tmp_path
        print(f"[WARN] Target workbook is locked; generated file kept at: {tmp_path}")

    print(f"[OK] Updated: {output_path}")
    print(f"[INFO] Base column: {base_col}")
    print("[INFO] Sheets: low_land, low_land_new, current_mix, fish_supp, no_supp")


if __name__ == "__main__":
    main()
