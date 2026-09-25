import pandas as pd

# Read the original file.
# Ensure the filename matches the local file.
# df_profile = pd.read_excel("../../input/Driver/retired_unused_raw/Nutrition_profile.xlsx", sheet_name='select')
df_profile = pd.read_csv("../../input/Driver/retired_unused_raw/Nutrition_profile_updated.csv")
df_missing = pd.read_csv("../../input/Driver/retired_unused_raw/nutrition_missing_report.csv")

# 1. Build Area Code and Item Code mappings from existing data.
area_mapping = df_profile[['M49_Country_Code', 'Area']].drop_duplicates('M49_Country_Code').set_index('M49_Country_Code')
item_mapping = df_profile[['Item', 'Item Code', 'Item Code (FBS)']].drop_duplicates('Item').set_index('Item')

# 2. Define nutrition standards to add.
elements_defaults = [
    {'Element Code': 664, 'Element': 'Food supply (kcal/capita/day)', 'Unit': 'kcal/cap/d'},
    {'Element Code': 674, 'Element': 'Protein supply quantity (g/capita/day)', 'Unit': 'g/cap/d'},
    {'Element Code': 684, 'Element': 'Fat supply quantity (g/capita/day)', 'Unit': 'g/cap/d'}
]

# 3. Extract missing entries.
missing_entries = df_missing[['M49_Country_Code', 'country_name', 'item_nutrition_map']].drop_duplicates()

new_rows = []

# 4. Iterate and construct new data.
for _, row in missing_entries.iterrows():
    m49 = row['M49_Country_Code']
    item = row['item_nutrition_map']
    
    # Look up country/area information or use defaults.
    if m49 in area_mapping.index:
        area_name = area_mapping.loc[m49, 'Area']
    else:
        area_name = row['country_name']
        
    # Look up commodity codes or use defaults.
    if item in item_mapping.index:
        item_code = item_mapping.loc[item, 'Item Code']
        item_fbs = item_mapping.loc[item, 'Item Code (FBS)']
    else:
        item_code = None
        item_fbs = None
        
    # Add a row for each nutrient.
    for elem in elements_defaults:
        new_row = {
            'Area Code (M49)': m49,
            'M49_Country_Code': m49,
            'Area': area_name,
            'Item Code': item_code,
            'Item Code (FBS)': item_fbs,
            'Item': item,
            'Element Code': elem['Element Code'],
            'Element': elem['Element'],
            'Unit': elem['Unit'],
            'Y2020': 0.01  # Populate 2020 as requested.
        }
        new_rows.append(new_row)

# 5. Combine and save.
df_new = pd.DataFrame(new_rows)
df_final = pd.concat([df_profile, df_new], ignore_index=True)

df_final.to_csv("../../input/Driver/retired_unused_raw/Nutrition_profile_updated2.csv", index=False)
print("处理完成，文件已保存为 Nutrition_profile_updated2.csv")
