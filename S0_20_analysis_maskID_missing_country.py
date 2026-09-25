# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 21:54:08 2025

@author: cheng
"""

# -*- coding: utf-8 -*-
"""
Script functionality:
1. Read mask_LUH2_025d.nc.
2. Calculate geometric center coordinates for listed unknown mask IDs.
3. Query geopy online to identify and print country names.
"""

import numpy as np
import netCDF4 as nc
import time
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

# Settings
nc_path = r"..\..\src\mask_LUH2_025d.nc"  # NetCDF file path
# Enter IDs marked #N/A in the Excel screenshot.
target_ids = [2, 5, 8, 16, 21, 33, 36, 49, 79, 
              131, 132, 146, 147, 165, 173, 181, 187, 212]


def get_location_name(geolocator, lat, lon, retries=3):
    """Try to retrieve address information for coordinates."""
    for i in range(retries):
        try:
            # language='en' returns English names for comparison with ISO standards.
            # language='zh-cn' returns Chinese names.
            location = geolocator.reverse((lat, lon), language='en', timeout=10)
            if location:
                return location.raw.get('address', {})
            return None
        except (GeocoderTimedOut, Exception) as e:
            print(f"    (连接超时，正在重试 {i+1}/{retries}...)")
            time.sleep(1)
    return None

def main():
    print(f"正在读取文件: {nc_path} ...")
    ds = nc.Dataset(nc_path)
    
    # Read variables.
    lat_arr = ds.variables['lat'][:]
    lon_arr = ds.variables['lon'][:]
    mask_arr = ds.variables['id1'][:]  # (lat, lon)
    
    # Create an OpenStreetMap geocoder.
    geolocator = Nominatim(user_agent="geo_checker_script_v1")

    print("-" * 95)
    print(f"{'Mask_ID':<8} | {'Center Lat':<10} | {'Center Lon':<10} | {'Identified Country / Territory'}")
    print("-" * 95)

    # Grid longitude/latitude for indexing.
    # mask_arr has shape (lat, lon); lon_grid and lat_grid must match.
    lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr)

    for uid in target_ids:
        # 1. Find all grid cells for this ID.
        rows, cols = np.where(mask_arr == uid)
        
        if len(rows) == 0:
            print(f"{uid:<8} | {'Empty':<10} | {'Empty':<10} | (该 ID 在地图中未找到像素)")
            continue

        # 2. Calculate the center as a simple arithmetic mean.
        # Island-country means may lie at sea, but OpenStreetMap usually identifies the surrounding territory.
        center_lat = np.mean(lat_grid[rows, cols])
        center_lon = np.mean(lon_grid[rows, cols])

        # 3. Query country names online.
        address = get_location_name(geolocator, center_lat, center_lon)
        
        country_name = "Unknown"
        cc = ""
        
        if address:
            # Prefer country, then territory for dependencies, then state.
            country_name = address.get('country', address.get('territory', 'Unknown'))
            cc = address.get('country_code', '').upper()
        
        # 4. Print results.
        print(f"{uid:<8} | {center_lat:<10.4f} | {center_lon:<10.4f} | {country_name} ({cc})")
        
        # Delay requests to respect API limits.
        time.sleep(0.5)

    ds.close()
    print("-" * 95)
    print("完成。请根据上述英文名更新您的 Excel 映射表。")

if __name__ == "__main__":
    main()