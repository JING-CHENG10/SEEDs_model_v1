# NOTE: Energy (Food supply kcal/capita/day) scaling uses max(Y2020, MDER_2006_08_kcal_cap_day)
# for year 2020 only; other years follow Intake_constraint YYYYY columns.
import os
from pathlib import Path
import pandas as pd
import numpy as np


def _norm_m49(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    if s.startswith("'"):
        s = s[1:]
    s = s.strip()
    if not s:
        return None
    if s.count('.') == 1:
        left, right = s.split('.', 1)
        if left.isdigit() and right.strip('0') == '':
            s = left
    if s.isdigit():
        return f"'{s.zfill(3)}"
    digits = ''.join(ch for ch in s if ch.isdigit())
    if not digits:
        return None
    return f"'{digits.zfill(3)}"


def _load_intake_totals(intake_path, years):
    df = pd.read_excel(intake_path, sheet_name='extract')
    df.columns = [str(c).strip() for c in df.columns]
    need_cols = ['M49_Country_Code', 'Indicator'] + [f"Y{y}" for y in years]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"Intake_constraint 缺少列: {c}")
    if 2020 in years and 'MDER_2006_08_kcal_cap_day' not in df.columns:
        raise ValueError("Intake_constraint ????????? MDER_2006_08_kcal_cap_day")
    df['m49_norm'] = df['M49_Country_Code'].apply(_norm_m49)
    ind_map = {
        'energy': 'Energy supply',
        'protein': 'Protein supply',
        'fat': 'Fat supply',
    }
    rows = []
    for key, ind_name in ind_map.items():
        sub = df[df['Indicator'].astype(str).str.strip() == ind_name].copy()
        if sub.empty:
            raise ValueError(f"Intake_constraint 中找不到 Indicator={ind_name}")
        for y in years:
            col = f"Y{y}"
            if key == 'energy' and y == 2020:
                tmp = sub[['m49_norm', col, 'MDER_2006_08_kcal_cap_day']].copy()
                tmp = tmp.rename(columns={col: 'intake_y', 'MDER_2006_08_kcal_cap_day': 'intake_mder'})
                tmp['intake_y'] = pd.to_numeric(tmp['intake_y'], errors='coerce')
                tmp['intake_mder'] = pd.to_numeric(tmp['intake_mder'], errors='coerce')
                tmp['intake_val'] = tmp[['intake_y', 'intake_mder']].max(axis=1, skipna=True)
                tmp = tmp.drop(columns=['intake_y', 'intake_mder'])
            else:
                tmp = sub[['m49_norm', col]].copy()
                tmp = tmp.rename(columns={col: 'intake_val'})
            tmp['nutrient_key'] = key
            tmp['year'] = y
            rows.append(tmp)
    out = pd.concat(rows, ignore_index=True)
    out['intake_val'] = pd.to_numeric(out['intake_val'], errors='coerce')
    out = out.dropna(subset=['m49_norm', 'intake_val'])
    return out


def _load_nutrition_profile(nut_path):
    def _read_csv_with_fallback(path):
        encodings = ['utf-8', 'utf-8-sig', 'gb18030', 'gbk', 'cp1252', 'latin1']
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return pd.read_csv(f)

    if str(nut_path).lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(nut_path, sheet_name='select')
    else:
        df = _read_csv_with_fallback(nut_path)
    df.columns = [str(c).strip() for c in df.columns]
    if 'M49_Country_Code' not in df.columns or 'Element' not in df.columns:
        raise ValueError("Nutrition_profile 缺少 M49_Country_Code 或 Element 列")
    df['m49_norm'] = df['M49_Country_Code'].apply(_norm_m49)
    df['M49_Country_Code'] = df['m49_norm']
    
    # Ensure Element column is clean string
    df['Element'] = df['Element'].astype(str).str.strip()
    
    element_map = {
        'Food supply (kcal/capita/day)': 'energy',
        'Protein supply quantity (g/capita/day)': 'protein',
        'Fat supply quantity (g/capita/day)': 'fat',
    }
    df['nutrient_key'] = df['Element'].map(element_map)
    return df


def run():
    intake_path = Path(r"..\..\input\Constraint\Intake_constraint.xlsx")
    nutrition_path = Path(r"..\..\input\Driver\retired_unused_raw\Nutrition_profile_updated2.csv")
    output_path = nutrition_path.with_name("Nutrition_profile_rescaled2.xlsx")

    years = list(range(2010, 2021))
    scale_year_cols = [f"Y{y}" for y in years]

    if not intake_path.exists():
        raise FileNotFoundError(f"找不到 Intake_constraint: {intake_path}")
    if not nutrition_path.exists():
        raise FileNotFoundError(f"找不到 Nutrition_profile: {nutrition_path}")

    intake_long = _load_intake_totals(str(intake_path), years)
    nut_df = _load_nutrition_profile(str(nutrition_path))

    year_cols = [c for c in nut_df.columns if isinstance(c, str) and c.startswith('Y') and c[1:].isdigit()]
    if not year_cols:
        raise ValueError("Nutrition_profile 未找到年份列 (Yxxxx)")
    for col in scale_year_cols:
        if col not in nut_df.columns:
            raise ValueError(f"Nutrition_profile 缺少列: {col}")

    id_cols = [c for c in nut_df.columns if not (isinstance(c, str) and c.startswith('Y') and c[1:].isdigit())]
    nut_long = nut_df.melt(id_vars=id_cols, value_vars=year_cols, var_name='year', value_name='profile_val')
    nut_long['year'] = nut_long['year'].astype(str).str.lstrip('Y').astype(int)
    nut_long['profile_val'] = pd.to_numeric(nut_long['profile_val'], errors='coerce')

    # Deduplicate: Keep only one value per ID+Year (matching final output logic)
    # This prevents double-counting in the sum calculation (prof_tot)
    nut_long = (nut_long.groupby(id_cols + ['year'], dropna=False, as_index=False)
                ['profile_val'].first())

    scale_base = nut_long[nut_long['year'].isin(years)].copy()
    scale_base = scale_base.dropna(subset=['m49_norm', 'nutrient_key', 'profile_val'])
    prof_tot = (scale_base.groupby(['m49_norm', 'nutrient_key', 'year'], as_index=False)
                ['profile_val'].sum()
                .rename(columns={'profile_val': 'profile_total'}))

    scale_df = pd.merge(
        intake_long,
        prof_tot,
        how='left',
        on=['m49_norm', 'nutrient_key', 'year']
    )
    scale_df['scale'] = np.where(
        (scale_df['intake_val'] > 0) & (scale_df['profile_total'] > 0),
        scale_df['intake_val'] / scale_df['profile_total'],
        1.0
    )

    # Merge scale
    nut_long = pd.merge(
        nut_long,
        scale_df[['m49_norm', 'nutrient_key', 'year', 'scale']],
        how='left',
        on=['m49_norm', 'nutrient_key', 'year']
    )
    
    # START
    try:
        debug_path = Path(os.environ.get("NZF_DEBUG_DUMP_PATH", Path(__file__).resolve().with_name("debug_dump.txt")))
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write("=== DEBUG LOG v2 ===\n")
            
            target_m49 = "'728"
            f.write(f"Target M49: {target_m49}\n\n")
            
            # Check Intake for 728
            f.write("--- Intake Long (728, 2020) ---\n")
            intake_sub = intake_long[(intake_long['m49_norm'] == target_m49) & (intake_long['year'] == 2020)]
            if intake_sub.empty:
                f.write("EMPTY! Checking unique M49s in Intake:\n")
                f.write(str(intake_long['m49_norm'].unique()) + "\n")
            else:
                f.write(intake_sub.to_string() + "\n")
            
            # Check Prof Tot for 728
            f.write("\n--- Prof Tot (728, 2020) ---\n")
            prof_sub = prof_tot[(prof_tot['m49_norm'] == target_m49) & (prof_tot['year'] == 2020)]
            if prof_sub.empty:
                f.write("EMPTY! Checking unique M49s in Prof Tot:\n")
                f.write(str(prof_tot['m49_norm'].unique()) + "\n")
            else:
                f.write(prof_sub.to_string() + "\n")
            
            # Check Scale DF for 728
            f.write("\n--- Scale DF (728, 2020) ---\n")
            scale_sub = scale_df[(scale_df['m49_norm'] == target_m49) & (scale_df['year'] == 2020)]
            if scale_sub.empty:
                 f.write("EMPTY!\n")
            else:
                f.write(scale_sub.to_string() + "\n")
            
            f.write("\n--- Dtypes ---\n")
            f.write(f"Intake m49: {intake_long['m49_norm'].dtype}\n")
            f.write(f"Prof m49: {prof_tot['m49_norm'].dtype}\n")
            
        print(f"[DEBUG] Wrote debug info to {debug_path}")
    except Exception as e:
        print(f"[DEBUG] Failed to write debug log: {e}")
    # END

    nut_long['scale'] = nut_long['scale'].fillna(1.0)
    mask_scale = nut_long['year'].isin(years) & nut_long['nutrient_key'].notna()
    nut_long['profile_val'] = np.where(
        mask_scale,
        nut_long['profile_val'] * nut_long['scale'],
        nut_long['profile_val']
    )
    nut_long = nut_long.drop(columns=['scale'])

    # Use groupby + unstack to preserve rows with all-NaNs without triggering
    # the combinatorial explosion (MemoryError) of pivot_table(dropna=False).
    out_df = (nut_long.groupby(id_cols + ['year'], dropna=False)['profile_val']
              .first()
              .unstack('year')
              .reset_index())
    out_df.columns = [f"Y{c}" if isinstance(c, (int, np.integer)) else c for c in out_df.columns]

    out_df = out_df[nut_df.columns]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_excel(output_path, index=False)

    scaled_pairs = scale_df[scale_df['scale'] != 1.0]
    print(f"[INFO] 输出: {output_path}")
    print(f"[INFO] 缩放记录数: {len(scaled_pairs)} (country-nutrient-year)")
    if len(scaled_pairs):
        print(f"[INFO] 缩放系数示例: {scaled_pairs['scale'].dropna().head(5).tolist()}")


if __name__ == "__main__":
    run()
