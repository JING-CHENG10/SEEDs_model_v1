import pandas as pd
import numpy as np
import os

# Configure paths.
base_dict_path = "../../src/dict_v3.xlsx"
input_file_path = "../../input/Land/Feed_pasture/Grass_feed_ratio_by_country_livestock.xlsx"
output_file_path = "../../input/Land/Feed_pasture/Grass_feed_ratio_by_country_livestock_refilled.xlsx"

# Define helpers.

def clean_m49_code(series):
    """Remove apostrophes from M49 codes and pad to three-digit strings."""
    # Remove apostrophes, convert to strings, trim, and pad leading zeros.
    s = series.astype(str).str.replace("'", "", regex=False).str.strip()
    s = s.str.replace(r"\.0+$", "", regex=True)
    return s.str.zfill(3)

def fill_crop_with_means(df, region_map_df):
    """
    Fill Crop by Species-Country using regional means, then world means.
    """
    # 1. Prepare regional mappings.
    region_map_clean = region_map_df[['M49_Country_Code', 'Region_agg2']].drop_duplicates().dropna(subset=['Region_agg2'])
    
    # Join Region_agg2 to the main table.
    df = df.merge(region_map_clean, on='M49_Country_Code', how='left')
    
    # 2. Calculate means from existing data only.
    # Regional means grouped by Species and Region_agg2
    region_means = df.groupby(['Species', 'Region_agg2'])['Crop'].transform('mean')
    
    # World means grouped by Species
    global_means = df.groupby(['Species'])['Crop'].transform('mean')
    
    # 3. Fill values.
    # Record missingness before filling.
    missing_before = df['Crop'].isna().sum()
    
    # First fill with regional means.
    df['Crop'] = df['Crop'].fillna(region_means)
    missing_after_region = df['Crop'].isna().sum()
    print(f"  - 使用区域均值填充了 {missing_before - missing_after_region} 条缺失数据")
    
    # Fill remaining gaps with world means.
    df['Crop'] = df['Crop'].fillna(global_means)
    missing_after_world = df['Crop'].isna().sum()
    print(f"  - 使用世界均值填充了 {missing_after_region - missing_after_world} 条缺失数据")
    
    # Remove helper columns.
    df = df.drop(columns=['Region_agg2'])
    
    return df

# Main program

def main():
    print("开始执行 S0_27_grassfeed_ratio_refill.py ...")

    
    # 1. Read the dictionary and get valid countries.
    
    print(f"步骤 1: 读取字典文件 {base_dict_path} ...")
    if not os.path.exists(base_dict_path):
        print(f"错误: 找不到文件 {base_dict_path}")
        return

    df_region_info = pd.read_excel(base_dict_path, sheet_name='region')
    df_region_info['M49_Country_Code'] = clean_m49_code(df_region_info['M49_Country_Code'])

    # Select valid countries with Region_label_new != 'no'.
    valid_mask = (df_region_info['Region_label_new'].astype(str).str.lower() != 'no') & \
                 (df_region_info['Region_label_new'].notna())
    valid_countries = df_region_info.loc[valid_mask, 'M49_Country_Code'].unique()
    
    print(f"  - 筛选出 {len(valid_countries)} 个有效国家代码")

    
    # 2. Read grass feed ratios.
    
    print(f"步骤 2: 读取输入文件 {input_file_path} ...")
    if not os.path.exists(input_file_path):
        print(f"错误: 找不到输入文件 {input_file_path}")
        return

    # Read country_level_weighted.
    df_input = pd.read_excel(input_file_path, sheet_name='country_level_weighted')
    df_input['M49_Country_Code'] = clean_m49_code(df_input['M49_Country_Code'])
    
    # Optionally retain valid countries only, depending on the intended filtering.
    df_input = df_input[df_input['M49_Country_Code'].isin(valid_countries)]

    
    # 3. Build Species x Valid Countries combinations.
    
    print("步骤 3: 建立 Species 与 所有有效国家的正交组合...")
    
    all_species = df_input['Species'].dropna().unique()
    print(f"  - 检测到 {len(all_species)} 种物种: {all_species}")

    # Create a Species-Country MultiIndex.
    idx_product = pd.MultiIndex.from_product([all_species, valid_countries], names=['Species', 'M49_Country_Code'])
    
    # Set the index, deduplicate, then reindex.
    # Crop is the main target; other columns may be omitted or retained with NaNs.
    # Preserve all original columns for the final output file.
    df_indexed = df_input.set_index(['Species', 'M49_Country_Code'])
    
    # Keep the first row when a country-species pair is duplicated.
    df_indexed = df_indexed[~df_indexed.index.duplicated(keep='first')]
    
    # Reindexing adds dictionary countries absent from the input, with Crop initially NaN.
    df_full = df_indexed.reindex(idx_product).reset_index()
    
    print(f"  - 扩展后的行数: {len(df_full)} (应为 {len(all_species)} * {len(valid_countries)})")

    
    # 4. Fill missing Crop values.
    
    print("步骤 4: 填充 Crop 列缺失值 (区域均值 -> 世界均值)...")
    
    # Call the fill function.
    df_filled = fill_crop_with_means(df_full, df_region_info)
    
    # Check remaining NaNs, which may persist for species without any global data.
    final_nan_count = df_filled['Crop'].isna().sum()
    if final_nan_count > 0:
        print(f"  - 警告: 仍有 {final_nan_count} 行 Crop 数据无法填充 (可能该物种无任何基础数据)，将填充为 0")
        df_filled['Crop'] = df_filled['Crop'].fillna(0)

    
    # 5. Calculate Grass.
    
    print("步骤 5: 计算 Grass 列 (Grass = 1 - Crop)...")
    
    df_filled['Grass'] = 1.0 - df_filled['Crop']
    
    # Optionally clip tiny precision errors below zero or above one, or preserve values as-is.
    # df_filled['Grass'] = df_filled['Grass'].clip(0, 1) # Optional.

    
    # 6. Save results.
    
    print(f"步骤 6: 保存结果到 {output_file_path} ...")
    
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    with pd.ExcelWriter(output_file_path) as writer:
        df_filled.to_excel(writer, sheet_name='country_level_weighted', index=False)
        
    print("处理完成！")

if __name__ == "__main__":
    main()
