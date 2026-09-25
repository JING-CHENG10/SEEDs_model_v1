#!/usr/bin/env python3
"""Build EAT-Lancet (2500 kcal/d) nutrient-share vectors aligned to Emis_item.Item_Nutrition_Map.

Inputs:
  - ../../src/dict_v3.xlsx (sheet: Emis_item)

Outputs:
  - ../../input/Driver/Nutrition/EAT_Lancet_energy_share_vector_via_EmisItem.xlsx
  - ../../input/Driver/Nutrition/EAT_Lancet_energy_share_vector_via_EmisItem.csv
  - ../../input/Driver/Nutrition/EAT_Lancet_energy_share_vector_via_EmisItem.json

Notes:
  - EAT-Lancet Table-1 kcal/day by food group is used (rounded), then scaled to sum to exactly 2500.
  - Within each EAT group, kcal are split evenly across mapped commodities (unless custom weights provided).
  - Grams/day are back-calculated using Emis_item.kcal_per_100g.
  - Protein and fat grams/day computed using Emis_item g_protein_per_100g and g_fat_per_100g.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    
    # Paths (relative to this script's location)
    base = Path(__file__).resolve().parent
    dict_xlsx = (base / "../../src/dict_v3.xlsx").resolve()

    out_dir = (base / "../../input/Driver/Nutrition").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_xlsx = out_dir / "EAT_Lancet_energy_share_vector_via_EmisItem.xlsx"
    out_csv = out_dir / "EAT_Lancet_energy_share_vector_via_EmisItem.csv"
    out_json = out_dir / "EAT_Lancet_energy_share_vector_via_EmisItem.json"

    
    # Read Emis_item and build category order using np.unique(Item_Nutrition_Map)
    emis = pd.read_excel(dict_xlsx, sheet_name="Emis_item")

    if "Item_Nutrition_Map" not in emis.columns:
        raise KeyError("dict_v3.xlsx:Emis_item must contain column 'Item_Nutrition_Map'.")

    cats = np.unique(
        emis["Item_Nutrition_Map"].dropna().astype(str).values
    ).tolist()

    # Coefficients table (dedupe by Item_Nutrition_Map)
    needed_cols = ["Item_Nutrition_Map", "kcal_per_100g", "g_protein_per_100g", "g_fat_per_100g"]
    missing = [c for c in needed_cols if c not in emis.columns]
    if missing:
        raise KeyError(f"dict_v3.xlsx:Emis_item missing columns: {missing}")

    coef = (
        emis.dropna(subset=["Item_Nutrition_Map"])
        .drop_duplicates(subset=["Item_Nutrition_Map"])
        [needed_cols]
        .copy()
    )
    coef["Item_Nutrition_Map"] = coef["Item_Nutrition_Map"].astype(str)
    coef = coef.set_index("Item_Nutrition_Map").reindex(cats)

    
    # EAT-Lancet Table 1 kcal/day by group (rounded); scale to 2500 kcal/day
    group_kcal = {
        "whole_grains": 811,
        "starchy_tubers": 39,
        "vegetables": 78,
        "fruits": 126,
        "dairy": 153,
        "red_meat": 30,
        "poultry": 62,
        "eggs": 19,
        "fish": 40,
        "legumes": 284,
        "nuts": 291,
        "unsat_oils": 354,
        "sat_oils": 96,
        "added_sugars": 120,
    }
    scale = 2500.0 / float(sum(group_kcal.values()))
    group_kcal = {k: v * scale for k, v in group_kcal.items()}

    
    # Mapping from EAT groups -> Item_Nutrition_Map categories.
    # Adjust lists/weights if you want region- or scenario-specific mixes.
    mapping: dict[str, list[str]] = {
        "whole_grains": [
            "Maize and products",
            "Rice and products",
            "Wheat and products",
            "Barley and products",
            "Millet and products",
            "Oats",
            "Rye and products",
            "Sorghum and products",
        ],
        "starchy_tubers": ["Potatoes and products", "Cassava and products", "Sweet potatoes"],
        "vegetables": ["Vegetables"],
        "fruits": ["Fruits-Excluding Wine"],
        "dairy": ["Milk-buffalo", "Milk-camel", "Milk-cattle", "Milk-goats", "Milk-sheep"],
        "red_meat": [
            "Bovine Meat-cattle",
            "Bovine Meat-buffalo",
            "Mutton & Goat Meat-goat",
            "Mutton & Goat Meat-sheep",
            "Pigmeat",
        ],
        "poultry": ["Poultry Meat-chickens", "Poultry Meat-ducks", "Poultry Meat-turkeys"],
        "eggs": ["Eggs"],
        "fish": ["Fish, Seafood"],
        "legumes": ["Beans", "Soyabeans"],
        "nuts": ["Groundnuts", "Treenuts"],
        "unsat_oils": ["Oilcrops", "Rape and Mustardseed", "Sunflower seed"],
        "sat_oils": ["Oilcrops"],
        "added_sugars": ["Sugar cane", "Sugar beet"],
    }

    # Optional: custom within-group energy weights (must sum to 1 within group)
    weights: dict[str, dict[str, float]] = {}

    
    kcal_alloc = pd.Series(0.0, index=cats)
    group_of = pd.Series("0 (not in EAT-Lancet reference basket)", index=cats)

    for g, kcal in group_kcal.items():
        items = mapping[g]
        if g in weights:
            w = weights[g]
            if set(w) != set(items):
                raise ValueError(f"Weights for group {g} must cover exactly: {items}")
            w_sum = sum(w.values())
            if not np.isclose(w_sum, 1.0):
                raise ValueError(f"Weights for group {g} must sum to 1; got {w_sum}")
            for it, wi in w.items():
                kcal_alloc[it] += kcal * wi
                group_of[it] = g
        else:
            for it in items:
                kcal_alloc[it] += kcal / len(items)
                group_of[it] = g

    # Ensure coefficients exist for any non-zero kcal allocation
    missing_coef = coef["kcal_per_100g"].isna() & (kcal_alloc > 0)
    if missing_coef.any():
        raise ValueError(
            "Missing kcal_per_100g for categories with positive kcal allocation: "
            + str(coef.index[missing_coef].tolist())
        )

    # Convert kcal -> grams/day using kcal_per_100g
    grams = pd.Series(0.0, index=cats)
    kcal100 = coef["kcal_per_100g"].astype(float)
    has_energy = kcal_alloc > 0
    grams.loc[has_energy] = 100.0 * kcal_alloc.loc[has_energy] / kcal100.loc[has_energy]

    # Protein/fat grams/day
    protein = pd.Series(0.0, index=cats)
    fat = pd.Series(0.0, index=cats)
    p100 = coef["g_protein_per_100g"].astype(float)
    f100 = coef["g_fat_per_100g"].astype(float)
    protein.loc[has_energy] = (grams.loc[has_energy] / 100.0) * p100.loc[has_energy]
    fat.loc[has_energy] = (grams.loc[has_energy] / 100.0) * f100.loc[has_energy]

    # Totals
    total_kcal = float(kcal_alloc.sum())
    total_protein = float(protein.sum())
    total_fat = float(fat.sum())

    # Shares
    kcal_share = kcal_alloc / total_kcal
    protein_share = protein / total_protein if total_protein > 0 else protein * 0
    fat_share = fat / total_fat if total_fat > 0 else fat * 0

    out = pd.DataFrame(
        {
            "Item_Nutrition_Map": cats,
            "EAT_group": group_of.values,
            "kcal_per_day": kcal_alloc.values,
            "grams_per_day": grams.values,
            "protein_g_per_day": protein.values,
            "fat_g_per_day": fat.values,
            "kcal_share_frac": kcal_share.values,
            "protein_share_frac": protein_share.values,
            "fat_share_frac": fat_share.values,
            "kcal_share_pct": (kcal_share.values * 100.0),
            "protein_share_pct": (protein_share.values * 100.0),
            "fat_share_pct": (fat_share.values * 100.0),
        }
    )

    totals = pd.DataFrame(
        {
            "metric": ["total_kcal_per_day", "total_protein_g_per_day", "total_fat_g_per_day"],
            "value": [total_kcal, total_protein, total_fat],
        }
    )

    map_rows = []
    for g, items in mapping.items():
        for it in items:
            map_rows.append(
                {
                    "EAT_group": g,
                    "Item_Nutrition_Map": it,
                    "within_group_weight": weights.get(g, {}).get(it, 1.0 / len(items)),
                }
            )
    mapping_df = pd.DataFrame(map_rows)

    assumptions = pd.DataFrame(
        {
            "assumption": [
                "EAT-Lancet kcal/day by food group uses rounded Table-1 values, then scaled to sum to 2500 kcal/day.",
                "Within each EAT group, kcal split evenly across mapped commodities (unless weights specified).",
                "Grams/day back-calculated using Emis_item.kcal_per_100g; macros computed using Emis_item g_protein_per_100g and g_fat_per_100g.",
                "Categories not mapped to any EAT group receive zero allocation.",
                "Category order is np.unique(Emis_item['Item_Nutrition_Map']).",
            ]
        }
    )

    # Write outputs
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        out.to_excel(w, sheet_name="vector", index=False)
        totals.to_excel(w, sheet_name="totals", index=False)
        mapping_df.to_excel(w, sheet_name="mapping", index=False)
        assumptions.to_excel(w, sheet_name="assumptions", index=False)

    out.to_csv(out_csv, index=False)

    payload = {
        "order": cats,
        "kcal_share": kcal_share.round(10).tolist(),
        "protein_share": protein_share.round(10).tolist(),
        "fat_share": fat_share.round(10).tolist(),
        "totals": {
            "kcal_per_day": total_kcal,
            "protein_g_per_day": total_protein,
            "fat_g_per_day": total_fat,
        },
        "paths": {
            "dict_xlsx": str(dict_xlsx),
            "out_dir": str(out_dir),
        },
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {out_xlsx}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    print("Totals:", total_kcal, total_protein, total_fat)


if __name__ == "__main__":
    main()
