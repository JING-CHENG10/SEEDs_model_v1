import pandas as pd
import numpy as np
import os


# 1. Configuration parameters

INPUT_FILE = '../../input/Price_Cost/Cost/macc_results_raw_AGRICULTURE.csv'
# Use the optimized 2080 output filename.
OUTPUT_EXCEL = '../../input/Price_Cost/Cost/MACC_2080_Optimized_Analysis.xlsx'

# Priority years for filling missing 2080 data
PRIORITY_ORDER = [
    2080, 
    2085, 2075, 
    2090, 2070, 
    2095, 2065, 
    2060, 2055, 2050, 2045, 2040, 2035, 2030
]


# 2. Classification and matching logic, unchanged

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
    'residue': 'Crop residues', 'tillage': 'Crop residues', 'no till': 'Crop residues', 'till': 'Crop residues',
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

def get_species(row):
    tech = str(row['Technology']).lower()
    process = row['Process']
    if process == 'Rice cultivation' or 'rice' in tech: return 'Rice'
    if process in ['Enteric fermentation', 'Manure management', 'Manure applied to soils']:
        if 'dairy' in tech: return 'Dairy Cattle'
        if 'swine' in tech or 'pig' in tech or 'hog' in tech: return 'Swine'
        if 'beef' in tech or 'meat' in tech or 'steer' in tech: return 'Non-Dairy Cattle'
        if 'poultry' in tech or 'chicken' in tech or 'broiler' in tech: return 'Poultry'
        if 'sheep' in tech or 'goat' in tech: return 'Sheep/Goats'
        return 'Other Livestock/General'
    if process in ['Synthetic fertilizers', 'Crop residues', 'Burning crop residues', 'Drained organic soils']:
        return 'Crops (General)'
    if 'forest' in process.lower(): return 'Forest'
    return 'Other'


# 3. Core function: prioritized data filling

def fill_data_with_priority(df, priority_list):
    """
    Group by Country-Process and select the best available year using the priority list.
    """
    # Determine the country column.
    country_col = 'country' if 'country' in df.columns else 'country_code'
    
    # Result container
    filled_frames = []
    
    # Group by country and Process.
    # Add Species for finer filling, such as species-specific gaps,
    # although MACC gaps usually affect an entire Process.
    groups = df.groupby([country_col, 'Process'])
    
    print(f"开始扫描 {len(groups)} 个 Country-Process 组合...")
    
    for (country, process), group in groups:
        # Get all available years in the group.
        available_years = set(group['year'].unique())
        
        selected_year = None
        
        # Select the first available year in priority order.
        for y in priority_list:
            if y in available_years:
                selected_year = y
                break
        
        if selected_year is not None:
            # Extract that year's data.
            subset = group[group['year'] == selected_year].copy()
            # Record source year for traceability.
            subset['Year_Used'] = selected_year
            filled_frames.append(subset)
        else:
            # Edge case: no data in any priority year.
            # print(f"Warning: {country} - {process} has no data in any specified year!")
            pass
            
    if not filled_frames:
        return pd.DataFrame()
        
    return pd.concat(filled_frames, ignore_index=True)


# 4. Main program

if __name__ == "__main__":
    # 0. Check paths.
    output_dir = os.path.dirname(OUTPUT_EXCEL)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Read data.
    print(f"正在读取文件: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError("未找到输入文件！")
        
    df_raw = pd.read_csv(INPUT_FILE)
    df_raw.columns = df_raw.columns.str.strip()
    df_raw.rename(columns={'tech_long': 'Technology', 'q_total': 'Reduction_potential', 'p': 'Unit_Cost'}, inplace=True)
    
    # 2. Preprocess.
    df_raw['Process'] = df_raw['Technology'].apply(match_ag_tech)
    df_raw['Species'] = df_raw.apply(get_species, axis=1)
    
    # Ensure integer years.
    if 'year' in df_raw.columns:
        df_raw['year'] = df_raw['year'].fillna(0).astype(int)
    
    # 3. Fill missing data to build the optimized 2080 dataset.
    print(f"\n>>> 正在构建 2080 优化数据集 (优先级填补)...")
    df_2080_optimized = fill_data_with_priority(df_raw, PRIORITY_ORDER)
    
    if df_2080_optimized.empty:
        print("错误：无法生成任何数据，请检查年份列。")
    else:
        # 4. Calculate weighted average costs.
        country_col = 'country' if 'country' in df_raw.columns else 'country_code'
        
        def weighted_avg_agg(x):
            total_pot = x['Reduction_potential'].sum()
            if total_pot == 0: return 0
            w_avg_cost = (x['Unit_Cost'] * x['Reduction_potential']).sum() / total_pot
            return pd.Series({
                'Weighted_Avg_Unit_Cost_USD_per_tCO2e': w_avg_cost,
                'Total_Reduction_Potential_MtCO2e': total_pot,
                'Tech_Count': x['Technology'].nunique(),
                'Year_Source': int(x['Year_Used'].mode()[0]) # Record the group's main source year.
            })

        # Group and aggregate.
        df_result = df_2080_optimized.groupby([country_col, 'Process', 'Species']).apply(weighted_avg_agg).reset_index()
        df_result = df_result.sort_values(by=[country_col, 'Process'])

        # 5. Save results.
        print(f"\n>>> 正在保存 Excel: {OUTPUT_EXCEL}")
        try:
            with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
                # Sheet 1: final calculated results.
                df_result.to_excel(writer, sheet_name='2080_Optimized', index=False)
                print("    已写入 Sheet: 2080_Optimized")
                
                # Sheet 2: source tracking by Country-Process and year actually used.
                # This table makes country data gaps easy to identify.
                source_stats = df_2080_optimized.groupby([country_col, 'Process', 'Year_Used']).size().reset_index(name='Tech_Rows')
                source_stats = source_stats.sort_values(by=[country_col, 'Process'])
                source_stats.to_excel(writer, sheet_name='Source_Tracking', index=False)
                print("    已写入 Sheet: Source_Tracking (来源年份追踪)")
                
            print(f"\n>>> 全部完成！请查看生成的文件。")
            
        except Exception as e:
            print(f"\n 保存失败: {e}")