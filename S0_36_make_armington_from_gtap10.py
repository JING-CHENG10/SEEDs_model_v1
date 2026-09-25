# -*- coding: utf-8 -*-
"""
Build Armington_elasticity.xlsx for NET-ZERO FOOD when market_clearing_mode = "regional_armington",
using GTAP 10 (free archive) elasticities from the GTAP package (.pkg).

Inputs (edit paths below):
  - GTAP10A_GTAP_AY.pkg              : the GTAP 10 archive package (zip-like)
  - dict_v3.xlsx                     : your dictionary (needs sheets: "Emis_item", "region")

Output:
  - Armington_elasticity.xlsx with sheet "armington_sigma"

------------------------------------------------------------
What is Armington_sigma in THIS model?
------------------------------------------------------------
In your S3_0_ds_linear_regional.py, net imports respond linearly to (Pw - Pc):
  net_import_{r,j,t} = base_net_{r,j} + b_m * (Pw_{j,t} - Pc_{r,j,t})
  b_m = armington_trade_scale * trade_volume_{r,j} * Armington_sigma_{r,j} / max(P0_{r,j,t}, eps)

So Armington_sigma here controls the slope of the import response to relative price,
scaled by baseline trade volume and divided by baseline price.

------------------------------------------------------------
Which GTAP elasticities can be used as Armington_sigma?
------------------------------------------------------------
GTAP provides two relevant substitution elasticities in its standard parameter file:

  1) ESUBD (GTAP code: ESBD in default.prm)
     - "Domestic vs Imported" substitution elasticity (Armington top nest)
     - Typically the BEST match to a single-sigma net-import response.
     - Recommended default.

  2) ESUBM (GTAP code: ESBM in default.prm)
     - "Import sources" substitution elasticity (lower nest across exporters)
     - Not the same object as ESUBD. Use only if your trade response is meant to represent
       source-switching rather than domestic-vs-import substitution.

Practical choices you may want (set SIGMA_CHOICE below):
  - "ESUBD"          : Armington_sigma = ESUBD (recommended)
  - "ESUBM"          : Armington_sigma = ESUBM (alternative; interpret carefully)
  - "MIN_ESUBD_ESUBM": conservative; Armington_sigma = min(ESUBD, ESUBM)
  - "GMEAN"          : geometric mean; Armington_sigma = sqrt(ESUBD * ESUBM)
  - "CUSTOM"         : use a user-provided override table (see OVERRIDE_XLSX)

Notes:
  - GTAP elasticities are sector-level and global; we repeat them across Region_market_agg.
  - Items that are not economic commodities in GTAP (e.g., "organic soils", "fires", "forestland")
    will receive a safe DEFAULT_SIGMA but will usually have trade_volume=0 in your model (no trade mapping).

Author: ChatGPT (reproducible helper script)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import re, zipfile, io, struct
from pathlib import Path


# USER SETTINGS

GTAP_PKG = r"GTAP10A_GTAP_AY.pkg"
DICT_V3  = r"dict_v3.xlsx"
OUT_XLSX = r"Armington_elasticity.xlsx"

# Choose which elasticity becomes Armington_sigma
SIGMA_CHOICE = "ESUBD"  # {"ESUBD","ESUBM","MIN_ESUBD_ESUBM","GMEAN","CUSTOM"}

# If SIGMA_CHOICE="CUSTOM", provide an override file with columns:
# Item_Emis, Armington_sigma, (optional) Region_market_agg or M49_Country_Code
OVERRIDE_XLSX = None

# Default sigma for unmapped/non-commodity items (safe fallback; usually irrelevant if trade_volume=0)
DEFAULT_SIGMA = 2.0

# Which year folder to read in the pkg (any available year is fine; in GTAP10A these ESUB values match across years)
GTAP_YEAR_FOLDER = "2014"


# GTAP sector list (65) — we only need the subset used by your food system items

GTAP_SECTOR_NAME = {
    "pdr":"Paddy rice",
    "wht":"Wheat",
    "gro":"Cereal grains nec",
    "v_f":"Vegetables, fruit, nuts (often also pulses & roots/tubers in concordances)",
    "osd":"Oil seeds (incl. oleaginous fruit such as oil palm fruit)",
    "c_b":"Sugar cane, sugar beet",
    "pfb":"Plant-based fibers",
    "ocr":"Crops nec",
    "ctl":"Bovine cattle, sheep and goats, horses (live animals)",
    "oap":"Animal products nec (incl. eggs, poultry, swine, etc.)",
    "rmk":"Raw milk",
    "wol":"Wool, silk-worm cocoons",
    "frs":"Forestry",
    "fsh":"Fishing",
    "cmt":"Bovine meat products",
    "omt":"Meat products nec",
    "vol":"Vegetable oils and fats",
    "mil":"Dairy products",
    "pcr":"Processed rice",
    "sgr":"Sugar",
    "ofd":"Food products nec",
    "b_t":"Beverages and tobacco products",
}


# Low-level helper: read Fortran unformatted records (as used in HAR-like files)

def _read_fortran_records(data: bytes) -> list[bytes]:
    """
    Parse Fortran unformatted sequential records:
      [len][payload][len] repeating (little-endian int32 lengths)
    """
    buf = io.BytesIO(data)
    recs: list[bytes] = []
    while True:
        b = buf.read(4)
        if not b or len(b) < 4:
            break
        (n,) = struct.unpack("<i", b)
        payload = buf.read(n)
        end = buf.read(4)
        if len(payload) != n or len(end) < 4:
            break
        (n2,) = struct.unpack("<i", end)
        if n2 != n:
            break
        recs.append(payload)
    return recs

def _extract_vector_from_default_prm(zf: zipfile.ZipFile, prm_name: str, code4: bytes) -> tuple[list[str], np.ndarray]:
    """
    Extract a 1-D sector vector from default.prm by searching for a 4-byte code record.
    This works for ESBD (ESUBD) and ESBM (ESUBM) in GTAP10A packages.
    """
    with zf.open(prm_name) as f:
        data = f.read()
    recs = _read_fortran_records(data)

    idxs = [i for i, r in enumerate(recs) if len(r) == 4 and r == code4]
    if not idxs:
        raise KeyError(f"Cannot find record code {code4!r} in {prm_name}")
    idx = idxs[0]

    # The sector names are in a nearby text record. For GTAP10A default.prm, the list lives at idx+3.
    # If your package differs, you may need to adjust offsets.
    codes_rec = recs[idx + 3].decode("latin1", errors="ignore")
    codes = re.findall(r"\b[a-z0-9_]{3,4}\b", codes_rec)

    # Numeric vector is in a nearby binary record; for GTAP10A default.prm it's at idx+6.
    data_rec = recs[idx + 6]
    vals = np.frombuffer(data_rec[8:], dtype="<f4").astype(float)  # skip the 8-byte header
    if len(vals) != len(codes):
        # Last-resort: try to parse all floats in record (rare)
        vals = np.frombuffer(data_rec, dtype="<f4").astype(float)
        vals = vals[-len(codes):]
    return codes, vals

def load_gtap_esub(pkg_path: str, year_folder: str) -> pd.DataFrame:
    """
    Read ESUBD (ESBD) and ESUBM (ESBM) from GTAP default.prm inside the pkg.
    """
    with zipfile.ZipFile(pkg_path, "r") as zf:
        prm = f"GTAP10A/GTAP/{year_folder}/default.prm"
        codes_d, esubd = _extract_vector_from_default_prm(zf, prm, b"ESBD")
        codes_m, esubm = _extract_vector_from_default_prm(zf, prm, b"ESBM")
    if codes_d != codes_m:
        raise ValueError("Sector code lists differ between ESBD and ESBM extraction; check parsing offsets.")
    return pd.DataFrame({"GTAP_code": codes_d, "ESUBD": esubd, "ESUBM": esubm})


# Heuristic concordance: Item_Emis -> GTAP_code
# You can replace/override this with your own mapping table if you want.

def _norm(x) -> str:
    return "" if pd.isna(x) else str(x).strip().lower()

def item_to_gtap_code(item_emis: str, item_prod: str, item_trade: str, item_price: str, item_cat2: str) -> tuple[str, str, str, str, str]:
    """
    Returns:
      (GTAP_code_suggested, GTAP_sector_name, Confidence, Alt_GTAP_code, Notes)
    """
    it = _norm(item_emis)
    pm = _norm(item_prod)
    tm = _norm(item_trade)
    pr = _norm(item_price)
    c2 = _norm(item_cat2)

    # Non-market / land / emissions objects (not GTAP commodities)
    if any(k in it for k in ["organic soils", "forestland", "de/reforestation", "savanna fires", "peatlands fire"]):
        return ("NA", "Non-market land/emissions object", "high", "", "Not an economic commodity in GTAP; keep sigma as DEFAULT_SIGMA and trade_volume should be 0.")

    # Forestry & fishing
    if "roundwood" in it or "roundwood" in pm or "roundwood" in pr:
        return ("frs", GTAP_SECTOR_NAME["frs"], "high", "", "Roundwood is primary forestry output.")
    if "fish" in it or "seafood" in it or "fish" in pm or "fish" in pr:
        return ("fsh", GTAP_SECTOR_NAME["fsh"], "high", "", "Fish/seafood.")

    # Eggs (layers)
    if "eggs" in pm or "eggs" in pr or "eggs" in tm or "layers" in it:
        return ("oap", GTAP_SECTOR_NAME["oap"], "high", "", "Eggs are typically in oap (animal products nec).")

    # Oil crops / oil seeds
    if any(k in pr for k in ["oilcrops", "oil crops", "oilseeds", "oil seeds"]) or any(k in pm for k in ["oilcrops", "oil crops", "oilseeds", "oil seeds"]):
        return ("osd", GTAP_SECTOR_NAME["osd"], "high", "vol", "Oil crops mapped to osd; if processed oil, consider vol.")
    oilseed_kw = ["soya", "soy", "sunflower", "rapeseed", "canola", "groundnut", "peanut", "oilpalm", "oil palm", "sesame", "linseed"]
    if any(k in it for k in oilseed_kw) or any(k in pm for k in oilseed_kw) or any(k in pr for k in oilseed_kw):
        return ("osd", GTAP_SECTOR_NAME["osd"], "high", "vol", "Oil seeds/oleaginous fruit mapped to osd; if processed oil, consider vol.")

    # Dairy
    if "raw milk" in pm or "raw milk" in pr or c2 == "dairy":
        return ("rmk", GTAP_SECTOR_NAME["rmk"], "high", "mil", "Raw milk mapped to rmk; processed dairy would be mil.")

    # Meat
    if "meat" in pm or "meat" in pr or "meat of" in pm or c2 == "meat":
        if any(k in it for k in ["cattle", "buffalo"]) or "meat of cattle" in pm or "meat of buffalo" in pm:
            return ("cmt", GTAP_SECTOR_NAME["cmt"], "high", "", "Bovine meat products (cmt).")
        return ("omt", GTAP_SECTOR_NAME["omt"], "high", "cmt", "Non-bovine meat products mapped to omt; some aggregations may group with cmt.")

    # Rice
    if it == "rice" or "rice" in pm or "rice" in pr:
        return ("pdr", GTAP_SECTOR_NAME["pdr"], "high", "pcr", "Rice primary mapped to pdr; processed rice would be pcr.")

    # Wheat
    if it == "wheat" or "wheat" in pm or "wheat" in pr:
        return ("wht", GTAP_SECTOR_NAME["wht"], "high", "", "Wheat (wht).")

    # Sugar crops
    if any(k in it for k in ["sugar cane", "sugarbeet", "sugar beet"]) or any(k in pm for k in ["sugar cane", "sugarbeet", "sugar beet"]) or any(k in pr for k in ["sugar cane", "sugarbeet", "sugar beet"]):
        return ("c_b", GTAP_SECTOR_NAME["c_b"], "high", "sgr", "Sugar crops mapped to c_b; refined sugar would be sgr.")

    # Plant fibers
    if "cotton" in it or "cotton" in pm or "cotton" in pr:
        return ("pfb", GTAP_SECTOR_NAME["pfb"], "high", "", "Cotton/fibre crops mapped to pfb.")

    # Other cereals (maize etc.)
    cereals_kw = ["maize", "corn", "barley", "rye", "oats", "millet", "sorghum"]
    if any(k in it for k in cereals_kw) or any(k in pm for k in cereals_kw) or any(k in pr for k in cereals_kw):
        return ("gro", GTAP_SECTOR_NAME["gro"], "high", "", "Other cereals mapped to gro (cereal grains nec).")

    # Fruits/vegetables/pulses/roots
    vf_kw = ["vegetables", "vegetable", "fruits", "fruit", "beans", "pulses", "cassava", "potatoes", "sweetpotato", "sweet potato"]
    if any(k in it for k in vf_kw) or any(k in pm for k in vf_kw) or any(k in pr for k in vf_kw):
        return ("v_f", GTAP_SECTOR_NAME["v_f"], "medium", "ocr",
                "Mapped to v_f; many concordances include pulses & roots/tubers in v_f. If you prefer, use ocr.")

    # Fallback crops
    if c2 in ["crop", "other"]:
        return ("ocr", GTAP_SECTOR_NAME["ocr"], "low", "", "Fallback crops/other to ocr (crops nec). Verify manually.")

    return ("NA", "Unmapped", "low", "", "No rule matched; please review.")

def choose_sigma(row: pd.Series, sigma_choice: str) -> float:
    if sigma_choice == "ESUBD":
        return float(row["ESUBD"])
    if sigma_choice == "ESUBM":
        return float(row["ESUBM"])
    if sigma_choice == "MIN_ESUBD_ESUBM":
        return float(min(row["ESUBD"], row["ESUBM"]))
    if sigma_choice == "GMEAN":
        return float(np.sqrt(row["ESUBD"] * row["ESUBM"]))
    raise ValueError(f"Unknown SIGMA_CHOICE={sigma_choice}")

def main():
    # Load dict tables
    region = pd.read_excel(DICT_V3, sheet_name="region")
    emis   = pd.read_excel(DICT_V3, sheet_name="Emis_item")

    # Valid countries -> region list
    valid = region[region["Region_label_new"].astype(str).str.strip().str.lower() != "no"].copy()
    regions = sorted([r for r in valid["Region_market_agg"].dropna().astype(str).str.strip().unique().tolist() if r and r.lower() != "nan"])

    # Unique items
    rep_cols = ["Item_Emis","Item_Cat2","Item_Production_Map","Item_Trade_Map","Item_Price_Map"]
    items = (emis[rep_cols].drop_duplicates(subset=["Item_Emis"]).sort_values("Item_Emis").reset_index(drop=True))

    # Load GTAP elasticities
    gtap_esub = load_gtap_esub(GTAP_PKG, GTAP_YEAR_FOLDER)

    # Concordance + merge
    rows=[]
    for _, r in items.iterrows():
        gtap_code, gtap_name, conf, alt, notes = item_to_gtap_code(
            r["Item_Emis"], r["Item_Production_Map"], r["Item_Trade_Map"], r["Item_Price_Map"], r["Item_Cat2"]
        )
        rows.append({
            "Item_Emis": r["Item_Emis"],
            "Item_Cat2": r["Item_Cat2"],
            "Item_Production_Map": r["Item_Production_Map"],
            "Item_Trade_Map": r["Item_Trade_Map"],
            "Item_Price_Map": r["Item_Price_Map"],
            "GTAP_code": gtap_code,
            "GTAP_sector_name": gtap_name,
            "Alt_GTAP_code": alt,
            "Confidence": conf,
            "Notes": notes,
        })
    conc = pd.DataFrame(rows)
    merged = conc.merge(gtap_esub, on="GTAP_code", how="left")

    # Determine sigma
    if SIGMA_CHOICE == "CUSTOM":
        if OVERRIDE_XLSX is None:
            raise ValueError("SIGMA_CHOICE='CUSTOM' requires OVERRIDE_XLSX path.")
        ov = pd.read_excel(OVERRIDE_XLSX, sheet_name="armington_sigma")
        # Normalize
        ov_cols = [c.lower() for c in ov.columns]
        if "item_emis" not in ov_cols or "armington_sigma" not in ov_cols:
            raise ValueError("Override must contain columns: Item_Emis, Armington_sigma")
        # Merge on Item_Emis and optional Region_market_agg etc. (handled later in expanded table)
        merged["Armington_sigma"] = np.nan
        merged = merged.merge(ov[["Item_Emis","Armington_sigma"]], on="Item_Emis", how="left", suffixes=("","_ov"))
        merged["Armington_sigma"] = merged["Armington_sigma_ov"]
        merged.drop(columns=[c for c in merged.columns if c.endswith("_ov")], inplace=True)
    else:
        merged["Armington_sigma"] = merged.apply(lambda x: choose_sigma(x, SIGMA_CHOICE) if pd.notna(x["ESUBD"]) else np.nan, axis=1)

    # Safe fallback for NA / unmapped
    merged["Armington_sigma"] = merged["Armington_sigma"].fillna(DEFAULT_SIGMA)
    merged["Suggested_exclude_from_trade"] = merged["GTAP_code"].eq("NA")

    # Expand to region-market sheet (GTAP elasticities are global by sector)
    expanded = []
    core_cols = [
        "Item_Emis","Armington_sigma","GTAP_code","ESUBD","ESUBM","Confidence",
        "Suggested_exclude_from_trade","GTAP_sector_name","Alt_GTAP_code","Notes"
    ]
    for reg in regions:
        tmp = merged[core_cols].copy()
        tmp.insert(0, "Region_market_agg", reg)
        expanded.append(tmp)
    armington_sigma = pd.concat(expanded, ignore_index=True)

    needs_review = merged[merged["Confidence"].isin(["low","medium"]) | merged["GTAP_code"].eq("NA")].copy()

    meta = pd.DataFrame({
        "key": [
            "gtap_pkg","gtap_prm","gtap_year_folder","sigma_choice","default_sigma_for_unmapped",
            "n_regions","n_items","n_rows_armington_sigma",
            "important_note_1","important_note_2"
        ],
        "value": [
            Path(GTAP_PKG).name,
            f"GTAP10A/GTAP/{GTAP_YEAR_FOLDER}/default.prm",
            GTAP_YEAR_FOLDER,
            SIGMA_CHOICE,
            DEFAULT_SIGMA,
            len(regions),
            len(merged),
            len(armington_sigma),
            "Your model uses ONE sigma per (region,item). ESUBD is typically the best match for 'domestic vs import' substitution.",
            "GTAP elasticities are global by sector; values are repeated across Region_market_agg. Override specific rows if you have region-specific estimates."
        ]
    })

    # Write output
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        armington_sigma.to_excel(w, sheet_name="armington_sigma", index=False)  # model reads this sheet
        merged.to_excel(w, sheet_name="item_level", index=False)
        needs_review.to_excel(w, sheet_name="needs_review", index=False)
        gtap_esub.to_excel(w, sheet_name="gtap_esub", index=False)
        meta.to_excel(w, sheet_name="meta", index=False)

    print(f"Done. Wrote: {OUT_XLSX}")
    print(f"Sigma choice = {SIGMA_CHOICE}; unmapped items -> DEFAULT_SIGMA={DEFAULT_SIGMA}")

if __name__ == "__main__":
    main()
