import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as c
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os


# 1. File-path configuration.

emissions_path = '../../output/BASE/Emis/emissions_summary.xlsx'
pop_path = '../../input/Driver/Population/WPP/Population_E_All_Data_NOFLAG.csv'
dict_path = '../../src/dict_v3.xlsx'
shp_map_path = '../../src/World_map/polygon/World_polygon.shp'

output_dir = '../../output/BASE/Plot/'
if not os.path.exists(output_dir):
    try:
        os.makedirs(output_dir)
    except:
        pass

img_name_total = 'Global_GHG_2020_Total_Optimized_Gt.png'
img_name_percapita = 'Global_GHG_2020_PerCapita_Optimized.png'

path_total = os.path.join(output_dir, img_name_total)
path_percapita = os.path.join(output_dir, img_name_percapita)


# 2. Read data.

print("正在读取数据...")
try:
    df_emissions = pd.read_excel(emissions_path, engine='openpyxl')
except:
    try:
        df_emissions = pd.read_csv(emissions_path, encoding='latin1')
    except:
        df_emissions = pd.read_csv(emissions_path)

try:
    df_pop = pd.read_csv(pop_path, encoding='latin1')
except:
    try:
        df_pop = pd.read_csv(pop_path, encoding='ISO-8859-1')
    except:
        df_pop = pd.read_csv(pop_path)

df_dict = pd.read_excel(dict_path, sheet_name='region')
gdf_world = gpd.read_file(shp_map_path)


# 3. Data processing.

print("正在处理数据...")

# 3.1 Emissions.
df_emissions['M49_Country_Code'] = df_emissions['M49_Country_Code'].astype(str).str.replace("'", "")
df_ghg = df_emissions[df_emissions['GHG'] == 'CO2eq'].copy()
df_ghg = df_ghg.groupby(['M49_Country_Code'], as_index=False)['Y2020'].sum()
df_ghg.rename(columns={'Y2020': 'GHG_2020_Kt'}, inplace=True)

# Unit conversion: Kt -> Gt.
df_ghg['GHG_2020_Gt'] = df_ghg['GHG_2020_Kt'] / 1_000_000

# 3.2 Population.
df_pop['M49_Country_Code'] = df_pop['M49_Country_Code'].astype(str).str.replace("'", "")
df_pop_total = df_pop[df_pop['Element'] == 'Total Population - Both sexes'].copy()
df_pop_total = df_pop_total[['M49_Country_Code', 'Y2020']]
df_pop_total.rename(columns={'Y2020': 'Pop_2020_1000s'}, inplace=True)

# 3.3 Merge.
df_data = pd.merge(df_ghg, df_pop_total, on='M49_Country_Code', how='inner')
df_data['Per_Capita_GHG_2020'] = df_data['GHG_2020_Kt'] / df_data['Pop_2020_1000s']


# 4. Mapping (M49 -> NAME -> SHP).

df_dict['M49_Country_Code'] = df_dict['M49_Country_Code'].astype(str).str.replace("'", "")
df_map = df_dict[['M49_Country_Code', 'NAME']].drop_duplicates()
df_data_mapped = pd.merge(df_data, df_map, on='M49_Country_Code', how='left')
df_data_mapped = df_data_mapped.dropna(subset=['NAME'])

# Match SHP records.
shp_columns = gdf_world.columns.tolist()
join_col = 'NAME' if 'NAME' in shp_columns else shp_columns[0]
gdf_world[join_col] = gdf_world[join_col].astype(str).str.strip()
df_data_mapped['NAME'] = df_data_mapped['NAME'].astype(str).str.strip()

gdf_plot = gdf_world.merge(df_data_mapped, left_on=join_col, right_on='NAME', how='left')


# 5. Visualization helper (colors and class breaks).

def get_optimized_norm_cmap(data_series):
    """
    Optimize for one-sided, mostly positive distributions using a sequential YlOrRd color scheme.
    """
    values = data_series.dropna().values
    
    # Exclude zeros to calculate more meaningful quantiles.
    val_positive = values[values > 0]
    
    breaks = []
    
    # Use quantiles for color boundaries to improve contrast.
    # 0, 20%, 40%, 60%, 80%, 90%, 95%, 99%, 100%
    if len(val_positive) > 0:
        percentiles = [0, 20, 40, 60, 80, 90, 95, 99, 100]
        breaks = np.nanpercentile(val_positive, percentiles, method='midpoint').tolist()
    else:
        breaks = np.linspace(values.min(), values.max(), 9).tolist()

    # If rare negative values occur, include their range or simply start at zero.
    if values.min() < 0:
        breaks.insert(0, values.min())
    
    # Ensure breaks are unique and sorted.
    breaks = sorted(list(set(breaks)))
    
    # If the minimum break is above zero, insert zero as the starting point.
    if breaks[0] > 0:
        breaks.insert(0, 0)

    # Color configuration.
    # Use Matplotlib's built-in Yellow-Orange-Red gradient.
    # This color scheme is intuitive for displaying intensity or quantity.
    # Sample the colormap according to the number of breaks.
    cmap_base = plt.get_cmap('YlOrRd')
    
    # Create a discrete colormap.
    # The number of colors should equal the number of intervals: len(breaks) - 1.
    n_bins = len(breaks) - 1
    colors = [cmap_base(i/n_bins) for i in range(n_bins)]
    
    cmap_plot = c.ListedColormap(colors)
    norm = mpl.colors.BoundaryNorm(breaks, cmap_plot.N)
    
    return norm, cmap_plot


# 6. Main plotting routine.

def plot_map(data_col, output_path, label_text):
    print(f"正在绘制: {output_path} ...")
    
    # Get the optimized Norm and Cmap.
    norm, cmap = get_optimized_norm_cmap(gdf_plot[data_col])
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Draw the base map.
    gdf_world.plot(ax=ax, color='#f0f0f0', edgecolor='white', linewidth=0.2)
    
    # Set the colorbar position.
    divider = make_axes_locatable(ax)
    # size="2%" narrows the colorbar.
    cax = divider.append_axes("right", size="2%", pad=0.1) 
    
    gdf_plot.plot(column=data_col, 
                  ax=ax,
                  cmap=cmap,
                  norm=norm,
                  linewidth=0.3,
                  edgecolor='black',
                  legend=True,
                  cax=cax, 
                  legend_kwds={
                      'label': label_text, 
                      'orientation': "vertical",
                      'format': "%.3f" # Main change: force display to three decimal places.
                  },
                  missing_kwds={'color': '#d9d9d9', 'label': 'No Data'})

    # Adjust the colorbar font.
    cax.tick_params(labelsize=12)
    cax.set_ylabel(label_text, fontsize=14, labelpad=15)

    ax.set_axis_off()
    
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"保存成功: {output_path}")

# Generate plots.
# Figure 1: totals, with units changed to Gt.
plot_map('GHG_2020_Gt', path_total, "Total GHG Emissions 2020 (Gt CO2eq)")

# Figure 2: per-capita values, retaining the original units with updated colors and formatting.
plot_map('Per_Capita_GHG_2020', path_percapita, "Per Capita GHG 2020 (Tonnes/capita)")

print("全部绘图完成！")