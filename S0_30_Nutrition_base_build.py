import pandas as pd
import numpy as np
import os
import warnings

# Ignore possible slice warnings.
warnings.filterwarnings('ignore')

def run():
    
    # 1. Define file paths.
    
    # Input path
    path_nutrition = r"..\..\input\Driver\retired_unused_raw\FoodBalanceSheets_E_All_Data_NOFLAG.csv"
    path_production = r"..\..\input\Production_Trade\Production_Crops_Livestock_E_All_Data_NOFLAG_yield_refilled_baseYearFilled.csv"
    
    # Output path
    output_dir = r"..\..\input\Driver\Nutrition"
    output_file = os.path.join(output_dir, "Nutrition_profile.xlsx")

    # Ensure the output directory exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("正在初始化...")

    
    # 2. Read and process nutrition supply data.
    
    print(f"读取营养数据: {path_nutrition}")
    # FAO data often contain Latin-1 characters; preserve original M49 formatting on read.
    df_nut = pd.read_csv(path_nutrition, encoding='latin-1', dtype={'M49_Country_Code': str})
    
    # Filter Element.
    target_elements = [
        'Food supply (kcal/capita/day)', 
        'Protein supply quantity (g/capita/day)', 
        'Fat supply quantity (g/capita/day)'
    ]
    df_nut = df_nut[df_nut['Element'].isin(target_elements)]
    
    # Retain all countries/regions without country filtering.
    
    # Identify Yxxxx year columns.
    year_cols = [c for c in df_nut.columns if c.startswith('Y') and c[1:].isdigit()]
    
    
    # 3. Read and process production data.
    
    print(f"读取产量数据: {path_production}")
    # Again preserve original M49 formatting.
    df_prod = pd.read_csv(path_production, encoding='latin-1', dtype={'M49_Country_Code': str})
    
    # Select Element = Production.
    df_prod = df_prod[df_prod['Element'] == 'Production']

    
    # 4. Define splitting rules.
    
    split_rules = [
        {
            'target_item': 'Sugar (Raw Equivalent)',
            'ref_items': ['Sugar cane', 'Sugar beet'],
            'suffixes': ['-Sugar cane', '-Sugar beet']
        },
        {
            'target_item': 'Meat, Other',
            'ref_items': [
                'Meat of asses, fresh or chilled', 
                'Meat of camels, fresh or chilled', 
                'Horse meat, fresh or chilled', 
                'Meat of other domestic camelids, fresh or chilled', 
                'Meat of mules, fresh or chilled'
            ],
            'suffixes': ['-asses', '-camels', '-horse', '-other domestic camelids', '-mules']
        },
        {
            'target_item': 'Milk - Excluding Butter',
            'ref_items': [
                'Raw milk of buffalo', 
                'Raw milk of camel', 
                'Raw milk of cattle', 
                'Raw milk of goats', 
                'Raw milk of sheep'
            ],
            'suffixes': ['-buffalo', '-camel', '-cattle', '-goats', '-sheep']
        },
        {
            'target_item': 'Bovine Meat',
            'ref_items': [
                'Meat of buffalo, fresh or chilled', 
                'Meat of cattle with the bone, fresh or chilled'
            ],
            'suffixes': ['-buffalo', '-cattle']
        },
        {
            'target_item': 'Poultry Meat',
            'ref_items': [
                'Meat of chickens, fresh or chilled', 
                'Meat of ducks, fresh or chilled', 
                'Meat of turkeys, fresh or chilled'
            ],
            'suffixes': ['-chickens', '-ducks', '-turkeys']
        },
        {
            'target_item': 'Mutton & Goat Meat',
            'ref_items': [
                'Meat of goat, fresh or chilled', 
                'Meat of sheep, fresh or chilled'
            ],
            'suffixes': ['-goat', '-sheep']
        }
    ]

    new_rows_list = []

    print("开始进行数据拆分与填充...")

    for rule in split_rules:
        target_item = rule['target_item']
        ref_items = rule['ref_items']
        suffixes = rule['suffixes']
        
        print(f"  处理项目: {target_item}")
        
        # 1. Get this Item's nutrition data.
        nut_subset = df_nut[df_nut['Item'] == target_item].copy()
        if nut_subset.empty:
            continue
            
        # 2. Get corresponding production data.
        prod_subset = df_prod[df_prod['Item'].isin(ref_items)].copy()
        
        # 3. Build the production-share matrix by pivoting.
        # Retain shared year columns only.
        prod_years = [y for y in year_cols if y in df_prod.columns]
        
        # Convert to long format for processing.
        prod_long = prod_subset.melt(
            id_vars=['M49_Country_Code', 'Item'], 
            value_vars=prod_years, 
            var_name='Year', 
            value_name='Production'
        )
        
        # Pivot: Index=[Country, Year], Columns=Item
        prod_pivot = prod_long.pivot_table(
            index=['M49_Country_Code', 'Year'], 
            columns='Item', 
            values='Production', 
            fill_value=0
        )
        
        # Ensure all reference Items exist as columns, even if absent in every country.
        for item in ref_items:
            if item not in prod_pivot.columns:
                prod_pivot[item] = 0
                
        # Calculate total production.
        prod_pivot['Total_Prod'] = prod_pivot[ref_items].sum(axis=1)
        
        # Calculate shares, avoiding division by zero.
        ratios = pd.DataFrame(index=prod_pivot.index)
        for item in ref_items:
            # Divide when total production > 0; otherwise use zero.
            ratios[item] = np.where(
                prod_pivot['Total_Prod'] > 0, 
                prod_pivot[item] / prod_pivot['Total_Prod'], 
                0
            )
        
        # Reset the share-table index for merging.
        ratios = ratios.reset_index()
        
        # 4. Convert nutrition data to long format and join shares.
        # id_vars consists of all non-year columns.
        id_vars_nut = [c for c in nut_subset.columns if c not in year_cols]
        nut_long = nut_subset.melt(
            id_vars=id_vars_nut, 
            value_vars=year_cols, 
            var_name='Year', 
            value_name='Value'
        )
        
        # Left join to preserve every nutrition row.
        # M49_Country_Code formatting must match: either all apostrophe-prefixed or all unprefixed.
        # Both source formats were preserved and match, so merge directly.
        merged = pd.merge(nut_long, ratios, on=['M49_Country_Code', 'Year'], how='left')
        
        # Use zero shares for unmatched records without production data.
        for item in ref_items:
            merged[item] = merged[item].fillna(0)
            
        # 5. Generate new rows.
        for i, ref_item in enumerate(ref_items):
            suffix = suffixes[i]
            new_item_name = f"{target_item}{suffix}"
            
            # Calculate split values.
            # Copy merged data for the current subitem.
            temp_df = merged.copy()
            temp_df['Value'] = temp_df['Value'] * temp_df[ref_item]
            
            # Update Item names.
            temp_df['Item'] = new_item_name
            
            # temp_df already preserves original Unit, Area, Element, and other columns.
            
            # Pivot back to wide format.
            # Preserve all id_vars, with Item now changed.
            # Pivot index combinations must be unique.
            pivot_index = [c for c in id_vars_nut if c != 'Item'] + ['Item']
            
            # Some columns may contain NaN; pivot_table defaults to dropna=True.
            temp_wide = temp_df.pivot_table(
                index=pivot_index, 
                columns='Year', 
                values='Value'
            ).reset_index()
            
            new_rows_list.append(temp_wide)

    
    # 5. Combine results and sort.
    
    print("正在合并数据...")
    if new_rows_list:
        df_new_rows = pd.concat(new_rows_list, ignore_index=True)
        # Restore original df_nut column order if pivoting changed it.
        # Add any missing columns and align.
        df_new_rows = df_new_rows.reindex(columns=df_nut.columns)
        
        # Append new rows to original data.
        df_final = pd.concat([df_nut, df_new_rows], ignore_index=True)
    else:
        df_final = df_nut

    print("正在排序...")
    # Sort M49_Country_Code, Item, and Element ascending.
    df_final.sort_values(
        by=['M49_Country_Code', 'Item', 'Element'], 
        ascending=[True, True, True], 
        inplace=True
    )

    
    # 6. Save the file.
    
    print(f"保存文件至: {output_file}")
    df_final.to_excel(output_file, index=False)
    print("脚本执行完毕。")

if __name__ == "__main__":
    run()