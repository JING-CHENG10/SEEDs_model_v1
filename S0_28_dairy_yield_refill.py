import pandas as pd
import numpy as np
import os
import warnings

# Ignore warnings such as division by zero.
warnings.filterwarnings('ignore')

# Configure paths.
base_dict_path = "../../src/dict_v3.xlsx"
input_file_path = "../../input/Production_Trade/Production_Crops_Livestock_E_All_Data_NOFLAG.csv"
output_file_path = "../../input/Production_Trade/Production_Crops_Livestock_E_All_Data_NOFLAG_yield_refilled_baseYearFilled.csv"

# Target Items
target_items = [
    'Raw milk of buffalo',
    'Raw milk of camel',
    'Raw milk of goats',
    'Raw milk of sheep'
]

# Process years.
year_cols = [f'Y{year}' for year in range(2000, 2024)]

# Helper functions

def clean_m49_to_string(series):
    """
    Standardize M49 to three-digit strings, removing apostrophes and padding zeros for matching.
    """
    s = series.astype(str).str.replace("'", "", regex=False).str.strip()
    s = s.str.replace(r"\.0+$", "", regex=True)
    return s.str.zfill(3)

def fill_with_means(df, value_cols, group_cols, region_map_df):
    """
    General fill order: adjacent years -> regional mean -> world mean.
    """
    # 1. Fill within rows from adjacent years.
    df[value_cols] = df[value_cols].replace(0, np.nan)
    df[value_cols] = df[value_cols].ffill(axis=1).bfill(axis=1)
    
    # Prepare the regional join.
    region_map_clean = region_map_df[['M49_Country_Code', 'Region_agg2']].drop_duplicates()
    
    # Temporarily join to calculate means.
    df = df.merge(region_map_clean, on='M49_Country_Code', how='left')
    
    # 2. Fill with regional means.
    for col in value_cols:
        region_means = df.groupby(['Item', 'Region_agg2'])[col].transform('mean')
        df[col] = df[col].fillna(region_means)
        
    # 3. Fill with world means.
    for col in value_cols:
        global_means = df.groupby(['Item'])[col].transform('mean')
        df[col] = df[col].fillna(global_means)
        
    # Remove helper columns.
    if 'Region_agg2' in df.columns:
        df = df.drop(columns=['Region_agg2'])
        
    return df

# Main program

def main():
    print("开始执行 S0_27_dairy_yield_refill.py (无 Select 预筛选版) ...")

    # 1. Read the dictionary and select valid countries.
    print("步骤 1: 读取字典并筛选 Region_label_new != 'no' 的国家...")
    if not os.path.exists(base_dict_path):
        print(f"错误: 找不到文件 {base_dict_path}")
        return

    df_region = pd.read_excel(base_dict_path, sheet_name='region')
    
    # Clean dictionary M49 codes for filtering.
    df_region['clean_m49'] = clean_m49_to_string(df_region['M49_Country_Code'])
    
    # Select valid countries.
    valid_mask = (df_region['Region_label_new'].astype(str).str.lower() != 'no') & (df_region['Region_label_new'].notna())
    valid_countries = df_region.loc[valid_mask, 'clean_m49'].unique()
    
    # Prepare regional mappings for filling.
    region_map = df_region.loc[valid_mask, ['clean_m49', 'Region_agg2']].rename(columns={'clean_m49': 'M49_Country_Code'})
    
    print(f"  - 有效国家数量: {len(valid_countries)}")

    # 2. Read production data without Select filtering.
    print("步骤 2: 读取 FAO 生产数据 (保留所有 Select 状态)...")
    if not os.path.exists(input_file_path):
        print(f"错误: 找不到文件 {input_file_path}")
        return

    df_prod = pd.read_csv(input_file_path, encoding='utf-8', low_memory=False)
    
    # Create standardized M49 codes for internal processing.
    df_prod['clean_m49'] = clean_m49_to_string(df_prod['M49_Country_Code'])
    
    # 3. Calculate Yield.
    print("步骤 3: 反算 Yield (kg/An) ...")
    
    # Select relevant Items.
    # Use original Item names without mapping.
    df_relevant = df_prod[df_prod['Item'].isin(target_items)].copy()
    
    # Calculate only for valid countries, even though Select is not filtered.
    df_relevant = df_relevant[df_relevant['clean_m49'].isin(valid_countries)]
    
    print(f"  - 相关 Item 的数据行数: {len(df_relevant)}")
    
    # Keys
    id_vars = ['clean_m49', 'M49_Country_Code', 'Area', 'Item'] 
    
    # Extract Production (t).
    prod_mask = df_relevant['Element'] == 'Production'
    df_p = df_relevant[prod_mask][id_vars + year_cols].set_index(id_vars)
    df_p = df_p[~df_p.index.duplicated(keep='first')] # Deduplicate.
    
    # Extract Milk Animals (head).
    # Common Element names: 'Milk Animals' or 'Animals milk'.
    anim_mask = df_relevant['Element'].isin(['Milk Animals', 'Animals milk'])
    df_a = df_relevant[anim_mask][id_vars + year_cols].set_index(id_vars)
    df_a = df_a[~df_a.index.duplicated(keep='first')] # Deduplicate.
    
    print(f"  - Production 数据行数: {len(df_p)}")
    print(f"  - Milk Animals 数据行数: {len(df_a)}")
    
    # Align data.
    common_indices = df_p.index.intersection(df_a.index)
    
    if len(common_indices) == 0:
        print("  警告: 仍未找到匹配的 Production 和 Milk Animals 数据。")
        return

    df_p = df_p.loc[common_indices]
    df_a = df_a.loc[common_indices]
    
    # Calculate (Production * 1000) / Milk Animals.
    df_a_values = df_a[year_cols].replace(0, np.nan)
    df_yield_values = (df_p[year_cols] * 1000) / df_a_values
    
    # Remove infinities.
    df_yield_values = df_yield_values.replace([np.inf, -np.inf], np.nan)
    
    # Reconstruct the DataFrame.
    df_yield = df_yield_values.reset_index()
    
    # 4. Fill missing values.
    print("步骤 4: 填充缺失数据 (Time -> Region -> World) ...")
    
    # Prepare filling using clean_m49.
    df_yield_for_fill = df_yield.copy()
    # rename column for merge compatibility with region_map
    df_yield_for_fill = df_yield_for_fill.rename(columns={'clean_m49': 'M49_Country_Code_Temp'})
    df_yield_for_fill['M49_Country_Code'] = df_yield_for_fill['M49_Country_Code_Temp']
    
    df_filled = fill_with_means(df_yield_for_fill, year_cols, ['Item'], region_map)
    
    # 5. Format new rows.
    print("步骤 5: 格式化新生成的行 ...")
    df_final_yield = df_filled.copy()
    
    # Set metadata.
    df_final_yield['Element'] = 'Yield'
    df_final_yield['Unit'] = 'kg/An'
    df_final_yield['Select'] = 1  # Explicitly set to 1.
    df_final_yield['Note'] = 'FAO Recalculated'
    
    # Restore M49_Country_Code using the original apostrophe-prefixed column.
    # M49_Country_Code remains in id_vars after reset_index.
    # Remove temporary fill columns and ensure the original column is used.
    if 'M49_Country_Code_Temp' in df_final_yield.columns:
        # The fill_with_means merge may confuse M49 columns;
        # M49_Country_Code_Temp in id_vars actually contains clean codes.
        # Prefer checking whether the original CSV M49_Country_Code still exists in id_vars.
        # Before step 4, df_yield came from reset_index and contained all id_vars.
        # df_filled modifies year_cols based on df_yield.
        # The original column remains unless explicitly dropped.
        pass

    # df_final_yield should contain clean M49_Country_Code_Temp and original M49_Country_Code,
    # unless fill_with_means overwrote them.
    # Remap clean to original codes for safety.
    m49_map = df_prod[['clean_m49', 'M49_Country_Code']].drop_duplicates().set_index('clean_m49')['M49_Country_Code']
    
    # Restore apostrophes if M49_Country_Code has become clean-format codes.
    # Check the first row.
    sample_m49 = str(df_final_yield['M49_Country_Code'].iloc[0])
    if "'" not in sample_m49 and 'M49_Country_Code_Temp' in df_final_yield.columns:
        # If overwritten or confused, map clean M49 codes back to original values.
        df_final_yield['M49_Country_Code'] = df_final_yield['M49_Country_Code_Temp'].map(m49_map)
    
    # Align column order.
    original_columns = df_prod.columns.tolist()
    if 'clean_m49' in original_columns:
        original_columns.remove('clean_m49')
        
    for col in original_columns:
        if col not in df_final_yield.columns:
            df_final_yield[col] = np.nan
            
    df_final_yield = df_final_yield[original_columns]
    
    print(f"  - 新增 Yield 行数: {len(df_final_yield)}")

    # 6. Combine and save.
    print("步骤 6: 合并并保存结果 ...")
    
    if 'clean_m49' in df_prod.columns:
        df_prod = df_prod.drop(columns=['clean_m49'])
        
    # Append new rows to the original data.
    df_combined = pd.concat([df_prod, df_final_yield], ignore_index=True)
    
    # Sort.
    df_combined = df_combined.sort_values(by=['Area', 'Item', 'Element'])
    
    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
    df_combined.to_csv(output_file_path, index=False, encoding='utf-8')
    print(f"  - 文件已生成: {output_file_path}")
    print("全部完成！")

if __name__ == "__main__":
    main()
