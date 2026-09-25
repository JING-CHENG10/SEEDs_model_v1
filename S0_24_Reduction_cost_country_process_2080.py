import pandas as pd
import numpy as np
import os


# 1. Configuration parameters

# Raw MACC CSV data
INPUT_MACC_FILE = '../../input/Price_Cost/Cost/macc_results_raw_AGRICULTURE.csv'

# Excel dictionary
INPUT_DICT_PATH = '../../src/dict_v3.xlsx'
DICT_SHEET_NAME = 'region'

# Output file
OUTPUT_EXCEL = '../../input/Price_Cost/Cost/MACC_2080_GapFilled_Final_overZero.xlsx'

# Priority years for 2080
PRIORITY_ORDER = [2080, 2085, 2075, 2090, 2070, 2095, 2065, 2060, 2055, 2050, 2045, 2040, 2035, 2030]

# Switch: whether to exclude negative-cost data
# True: drop rows with Unit_Cost < 0.
# False: retain all data.
FILTER_NEGATIVE_COSTS = True 


# 2. Country-name corrections

MANUAL_COUNTRY_FIX = {
    'United States': 'USA', 'United States of America': 'USA', 'US': 'USA',
    'China': 'CHN', 'China, mainland': 'CHN',
    'Viet Nam': 'VNM', 'Vietnam': 'VNM',
    'Bolivia': 'BOL', 'Bolivia (Plurinational State of)': 'BOL',
    'Venezuela': 'VEN', 'Venezuela (Bolivarian Republic of)': 'VEN',
    'Iran': 'IRN', 'Iran (Islamic Republic of)': 'IRN',
    'South Korea': 'KOR', 'Republic of Korea': 'KOR', 'Korea, Republic of': 'KOR',
    'North Korea': 'PRK', 'Dem. People\'s Rep. of Korea': 'PRK',
    'Russia': 'RUS', 'Russian Federation': 'RUS',
    'Tanzania': 'TZA', 'United Republic of Tanzania': 'TZA',
    'Syria': 'SYR', 'Syrian Arab Republic': 'SYR',
    'Laos': 'LAO', 'Lao People\'s Democratic Republic': 'LAO',
    'Moldova': 'MDA', 'Republic of Moldova': 'MDA',
    'Congo': 'COG', 'Democratic Republic of the Congo': 'COD'
}


# 3. Helper functions

def read_csv_robust(file_path):
    """Try multiple encodings when reading a MACC CSV."""
    encodings_to_try = ['utf-8', 'gbk', 'gb18030', 'latin1', 'cp1252']
    for enc in encodings_to_try:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            raise e
    raise ValueError(f" 无法读取文件 {file_path}，尝试了所有常见编码均失败。")

ag_mapping_rules = {
    'enteric': 'Enteric fermentation', '3nop': 'Enteric fermentation', 'asparagopsis': 'Enteric fermentation',
    'red algae': 'Enteric fermentation', 'propionate': 'Enteric fermentation', 'antimethanogen': 'Enteric fermentation',
    'antibiotics': 'Enteric fermentation', 'bst': 'Enteric fermentation', 'feed conversion': 'Enteric fermentation',
    'grazing': 'Enteric fermentation', 'masks': 'Enteric fermentation', 'vaccine': 'Enteric fermentation',
    'breeding': 'Enteric fermentation', 'lipid': 'Enteric fermentation',
    'digester': 'Manure management', 'lagoon': 'Manure management', 'dairy production rng': 'Manure management',
    'swine production rng': 'Manure management', 'manure management': 'Manure management', 'biodigester': 'Manure management',
    'gas': 'Manure management', 'flare': 'Manure management', 'separator': 'Manure management', 'compost': 'Manure management',
    'acidification': 'Manure management',
    'rice': 'Rice cultivation', 'midseason drainage': 'Rice cultivation', 'wetting': 'Rice cultivation',
    'flooding': 'Rice cultivation', 'dry-seeding': 'Rice cultivation', 'sulfate': 'Rice cultivation', 'hybrid': 'Rice cultivation',
    'fertilizer': 'Synthetic fertilizers', 'nitrification': 'Synthetic fertilizers', 'urea': 'Synthetic fertilizers',
    'auto fertilization': 'Synthetic fertilizers', 'synthetic': 'Synthetic fertilizers',
    'residue': 'Crop residues', 'tillage': 'Crop residues', 'till': 'Crop residues',
    'cover crop': 'Crop residues', 'cropping': 'Crop residues', 'legume': 'Crop residues', 'biochar': 'Crop residues',
    'application': 'Manure applied to soils', 'spread': 'Manure applied to soils', 'injection': 'Manure applied to soils',
    'drained': 'Drained organic soils', 'organic soil': 'Drained organic soils', 'water table': 'Drained organic soils',
    'burning': 'Burning crop residues',
    'reforestation': 'De/Reforestation_crop', 'afforestation': 'De/Reforestation_crop', 'forest': 'De/Reforestation_crop',
    'silvopasture': 'De/Reforestation_pasture'
}

def match_ag_tech(tech_name):
    tech_lower = str(tech_name).lower()
    if 'burning' in tech_lower: return 'Burning crop residues'
    for keyword, process in ag_mapping_rules.items():
        if keyword in tech_lower: return process
    if 'manure' in tech_lower: return 'Manure management'
    return 'Other'


# 4. Main processing logic

def process_data():
    print(">>> 1. 读取数据...")
    if not os.path.exists(INPUT_MACC_FILE):
        raise FileNotFoundError(f"未找到 MACC 文件: {INPUT_MACC_FILE}")
    if not os.path.exists(INPUT_DICT_PATH):
        raise FileNotFoundError(f"未找到字典文件: {INPUT_DICT_PATH}")

    # A. Read MACC CSV data.
    df_macc = read_csv_robust(INPUT_MACC_FILE)
    df_macc.columns = df_macc.columns.str.strip()
    rename_dict = {'tech_long': 'Technology', 'q_total': 'Reduction_potential', 'p': 'Unit_Cost'}
    df_macc.rename(columns=rename_dict, inplace=True)
    
    # Filter negative costs.
    if FILTER_NEGATIVE_COSTS:
        print(">>> 1.2. 根据配置过滤负成本 (Unit_Cost < 0)...")
        before_neg_count = len(df_macc)
        df_macc = df_macc[df_macc['Unit_Cost'] >= 0]
        print(f"    已移除 {before_neg_count - len(df_macc)} 行负成本数据。")
    else:
        print(">>> 1.2. 配置为保留负成本数据，跳过过滤。")
    

    # Filter outliers.
    print(">>> 1.5. 过滤成本异常值...")
    if len(df_macc) == 0:
        print("    警告：当前数据为空，跳过异常值过滤。")
    else:
        initial_rows = len(df_macc)
        # Calculate the 1st and 99th percentiles.
        lower_bound = df_macc['Unit_Cost'].quantile(0.01)
        upper_bound = df_macc['Unit_Cost'].quantile(0.99)
        print(f"    成本范围 (1% to 99%): [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        # Filter data.
        df_macc = df_macc[(df_macc['Unit_Cost'] >= lower_bound) & (df_macc['Unit_Cost'] <= upper_bound)]
        filtered_rows = len(df_macc)
        print(f"    已移除 {initial_rows - filtered_rows} 行异常值数据。剩余行数: {filtered_rows}")

    df_macc['Process'] = df_macc['Technology'].apply(match_ag_tech)
    
    # B. Read the Excel Region dictionary.
    print(f"    读取字典 Excel: {INPUT_DICT_PATH} (Sheet: {DICT_SHEET_NAME})")
    df_dict = pd.read_excel(INPUT_DICT_PATH, sheet_name=DICT_SHEET_NAME)
    df_dict.columns = df_dict.columns.str.strip()
    
    # 2. Build country-name matching data.
    print(">>> 2. 构建国家匹配库...")
    
    # Adjust candidate columns to the actual names reported in the error.
    # The actual file column is 'ISO3 Code'.
    possible_iso_cols = ['ISO3 Code', 'ISO3_Code', 'ISO3', 'ISO_Code', 'M49_Code', 'Country Code'] 
    dict_iso_col = next((c for c in possible_iso_cols if c in df_dict.columns), None)
    
    # Detect the Name column automatically.
    possible_name_cols = ['Country', 'Country_Name', 'Name', 'Area']
    dict_name_col = next((c for c in possible_name_cols if c in df_dict.columns), None)
    
    if not dict_iso_col:
        raise KeyError(f"在字典 Sheet '{DICT_SHEET_NAME}' 中未找到 ISO3 代码列，现有列: {df_dict.columns.tolist()}")
    else:
        print(f"    成功识别 ISO3 列名: '{dict_iso_col}'")
        
    # Build {Name -> ISO3} mappings.
    iso_mapper = {str(code).strip(): str(code).strip() for code in df_dict[dict_iso_col].dropna().unique()}
    if dict_name_col:
        for idx, row in df_dict.iterrows():
            name = str(row[dict_name_col]).strip()
            iso = str(row[dict_iso_col]).strip()
            iso_mapper[name] = iso
            iso_mapper[name.lower()] = iso
            
    iso_mapper.update(MANUAL_COUNTRY_FIX)
    
    # 3. Standardize MACC data.
    print(">>> 3. 标准化 MACC 数据国家列...")
    macc_country_raw_col = 'country_code' if 'country_code' in df_macc.columns else 'country'
    
    def standardize_country(val):
        val_str = str(val).strip()
        if val_str in iso_mapper: return iso_mapper[val_str]
        if val_str.lower() in iso_mapper: return iso_mapper[val_str.lower()]
        if len(val_str) == 3 and val_str.isupper(): return val_str
        return None

    df_macc['ISO3_MATCHED'] = df_macc[macc_country_raw_col].apply(standardize_country)
    
    # Drop unidentified countries to avoid errors in later Cartesian expansion.
    df_macc_clean = df_macc.dropna(subset=['ISO3_MATCHED']).copy()
    print(f"    标准化后有效行数: {len(df_macc_clean)}")
    
    # 4. Generate 2080 data using year priorities.
    print(">>> 4. 生成 2080 优先级数据...")
    def fill_priority(df):
        filled = []
        groups = df.groupby(['ISO3_MATCHED', 'Process'])
        for (iso, proc), group in groups:
            avail_years = set(group['year'].unique())
            sel_year = next((y for y in PRIORITY_ORDER if y in avail_years), None)
            if sel_year:
                filled.append(group[group['year'] == sel_year].copy())
        return pd.concat(filled) if filled else pd.DataFrame()

    df_2080 = fill_priority(df_macc_clean)
    
    # Aggregate to obtain Direct Cost.
    def calc_weighted_cost(x):
        pot_sum = x['Reduction_potential'].sum()
        if pot_sum == 0: return np.nan
        cost_sum = (x['Unit_Cost'] * x['Reduction_potential']).sum()
        return cost_sum / pot_sum

    df_2080_agg = df_2080.groupby(['ISO3_MATCHED', 'Process']).apply(calc_weighted_cost).reset_index(name='Direct_Cost')

    # 5. Build a Cartesian skeleton.
    print(">>> 5. 构建正交骨架并计算基准值...")
    
    # Select Region_label_new != 'no'.
    if 'Region_label_new' not in df_dict.columns:
         raise KeyError(f"字典中缺少 'Region_label_new' 列，现有列: {df_dict.columns.tolist()}")
         
    valid_countries = df_dict[df_dict['Region_label_new'] != 'no'].copy()
    
    # Merge region info to calculate benchmarks
    df_2080_geo = df_2080.merge(
        valid_countries[[dict_iso_col, 'Region_agg2']], 
        left_on='ISO3_MATCHED', 
        right_on=dict_iso_col, 
        how='inner'
    )
    
    # Calculate baseline values.
    regional_benchmark = df_2080_geo.groupby(['Region_agg2', 'Process']).apply(
        lambda x: (x['Unit_Cost'] * x['Reduction_potential']).sum() / x['Reduction_potential'].replace(0, np.nan).sum()
    ).to_dict()
    
    global_benchmark = df_2080.groupby('Process').apply(
        lambda x: (x['Unit_Cost'] * x['Reduction_potential']).sum() / x['Reduction_potential'].replace(0, np.nan).sum()
    ).to_dict()

    # Build the skeleton.
    skeleton_rows = []
    all_processes = [p for p in df_macc_clean['Process'].unique() if p != 'Other']
    
    # Use M49_Country_Code, confirmed by the reported column names.
    for _, row in valid_countries.iterrows():
        iso = row[dict_iso_col]
        m49 = row['M49_Country_Code']
        region = row['Region_agg2']
        
        for proc in all_processes:
            skeleton_rows.append({
                'ISO3_Target': iso,
                'M49_Country_Code': m49,
                'Region_agg2': region,
                'Process': proc
            })
    
    df_skeleton = pd.DataFrame(skeleton_rows)
    print(f"    骨架构建完成，共 {len(df_skeleton)} 行")
    
    # 6. Fill missing values.
    print(">>> 6. 执行数据填补...")
    
    df_final = df_skeleton.merge(
        df_2080_agg, 
        left_on=['ISO3_Target', 'Process'], 
        right_on=['ISO3_MATCHED', 'Process'], 
        how='left'
    )
    
    def fill_logic(row):
        cost = row['Direct_Cost']
        source = 'Raw_Data'
        
        # Fill missing, infinite, or zero values.
        # Negative values have already been removed if that filter is enabled.
        if pd.isna(cost) or np.isinf(cost) or cost == 0:
            # Level 2: Region Fill
            reg_key = (row['Region_agg2'], row['Process'])
            if reg_key in regional_benchmark and pd.notna(regional_benchmark[reg_key]):
                cost = regional_benchmark[reg_key]
                source = 'Region_Fill'
            else:
                # Level 3: Global Fill
                if row['Process'] in global_benchmark and pd.notna(global_benchmark[row['Process']]):
                    cost = global_benchmark[row['Process']]
                    source = 'Global_Fill'
                else:
                    cost = np.nan
                    source = 'Missing'
        
        # Final safety check
        if np.isinf(cost): cost = np.nan
        return pd.Series([cost, source], index=['Final_Unit_Cost', 'Data_Source'])

    df_final[['Final_Unit_Cost', 'Data_Source']] = df_final.apply(fill_logic, axis=1)
    
    # 7. Save.
    print(f">>> 7. 保存结果至: {OUTPUT_EXCEL}")
    os.makedirs(os.path.dirname(OUTPUT_EXCEL), exist_ok=True)
    
    out_cols = ['M49_Country_Code', 'Region_agg2', 'Process', 'Final_Unit_Cost', 'Data_Source', 'Direct_Cost', 'ISO3_Target']
    df_export = df_final[out_cols].sort_values(by=['Region_agg2', 'M49_Country_Code', 'Process'])
    
    with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
        df_export.to_excel(writer, sheet_name='Gap_Filled_Data', index=False)
        pd.DataFrame(list(regional_benchmark.items()), columns=['Key', 'Cost']).to_excel(writer, sheet_name='Ref_Region_Benchmark')
        pd.DataFrame(list(global_benchmark.items()), columns=['Key', 'Cost']).to_excel(writer, sheet_name='Ref_Global_Benchmark')
        
    print(">>> 全部完成！")

if __name__ == "__main__":
    try:
        process_data()
    except Exception as e:
        print(f" 运行错误: {e}")