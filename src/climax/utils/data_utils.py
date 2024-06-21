# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

NAME_TO_VAR = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "msl",
    "surface_pressure": "sp",
    "toa_incident_solar_radiation": "tisr",
    "total_precipitation": "tp",
    "land_sea_mask": "lsm",
    "orography": "orography",
    "lattitude": "lat2d",
    "geopotential": "z",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "temperature": "t",
    "relative_humidity": "r",
    "specific_humidity": "q",
}

VAR_TO_NAME = {v: k for k, v in NAME_TO_VAR.items()}

SINGLE_LEVEL_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "surface_pressure",
    "toa_incident_solar_radiation",
    "total_precipitation",
    "land_sea_mask",
    "orography",
    "lattitude",
]
PRESSURE_LEVEL_VARS = [
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "relative_humidity",
    "specific_humidity",
]
DEFAULT_PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

NAME_LEVEL_TO_VAR_LEVEL = {}

for var in SINGLE_LEVEL_VARS:
    NAME_LEVEL_TO_VAR_LEVEL[var] = NAME_TO_VAR[var]

for var in PRESSURE_LEVEL_VARS:
    for l in DEFAULT_PRESSURE_LEVELS:
        NAME_LEVEL_TO_VAR_LEVEL[var + "_" + str(l)] = NAME_TO_VAR[var] + "_" + str(l)

VAR_LEVEL_TO_NAME_LEVEL = {v: k for k, v in NAME_LEVEL_TO_VAR_LEVEL.items()}

BOUNDARIES = {
    'NorthAmerica': { # 8x14
        'lat_range': (15, 65),
        'lon_range': (220, 300)
    },
    'SouthAmerica': { # 14x10
        'lat_range': (-55, 20),
        'lon_range': (270, 330)
    },
    'Europe': { # 6x8
        'lat_range': (30, 65),
        'lon_range': (0, 40)
    },
    'SouthAsia': { # 10, 14
        'lat_range': (-15, 45),
        'lon_range': (25, 110)
    },
    'EastAsia': { # 10, 12
        'lat_range': (5, 65),
        'lon_range': (70, 150)
    },
    'Australia': { # 10x14
        'lat_range': (-50, 10),
        'lon_range': (100, 180)
    },
    'PuertoRicoRegion': {
        'lat_range': (10, 25),
        'lon_range': (285, 300)
    },
    'WesternHemisphere': {
        'lat_range': (-90, 90),
        'lon_range': (180, 360)  # 0 to 180°W in 0-360 degree notation
    },
    'Global': { # 32, 64
        'lat_range': (-90, 90),
        'lon_range': (0, 360)
    }
}

def calculate_region_info(region, lat, lon, patch_size):
    region = BOUNDARIES[region] # sets region to equal coordinates of specific region called
    lat_range = region['lat_range'] # sets
    lon_range = region['lon_range']
    lat = lat[::-1] # -90 to 90 from south (bottom) to north (top)
    h, w = len(lat), len(lon)
    lat_matrix = np.expand_dims(lat, axis=1).repeat(w, axis=1)
    lon_matrix = np.expand_dims(lon, axis=0).repeat(h, axis=0)
    valid_cells = (lat_matrix >= lat_range[0]) & (lat_matrix <= lat_range[1]) & (lon_matrix >= lon_range[0]) & (lon_matrix <= lon_range[1])
    h_ids, w_ids = np.nonzero(valid_cells)
    h_from, h_to = h_ids[0], h_ids[-1]
    w_from, w_to = w_ids[0], w_ids[-1]
    patch_idx = -1
    p = patch_size
    valid_patch_ids = []
    min_h, max_h = 1e5, -1e5
    min_w, max_w = 1e5, -1e5
    for i in range(0, h, p):
        for j in range(0, w, p):
            patch_idx += 1
            if (i >= h_from) & (i + p - 1 <= h_to) & (j >= w_from) & (j + p - 1 <= w_to):
                valid_patch_ids.append(patch_idx)
                min_h = min(min_h, i)
                max_h = max(max_h, i + p - 1)
                min_w = min(min_w, j)
                max_w = max(max_w, j + p - 1)
    
    region_info = {
        'patch_ids': valid_patch_ids,
        'min_h': min_h,
        'max_h': max_h,
        'min_w': min_w,
        'max_w': max_w,
    }

    return region_info

def get_region_info(region, lat, lon, patch_size):
    global_info = calculate_region_info('Global', lat, lon, patch_size)
    puerto_rico_info = calculate_region_info('PuertoRicoRegion', lat, lon, patch_size)
    
    global_info['max_h_pr'] = puerto_rico_info['max_h']
    global_info['min_h_pr'] = puerto_rico_info['min_h']
    global_info['max_w_pr'] = puerto_rico_info['max_w']
    global_info['min_w_pr'] = puerto_rico_info['min_w']

    return global_info

# Example usage
lat = np.linspace(-90, 90, 180)
lon = np.linspace(0, 360, 360)
patch_size = 5

region_info = get_region_info('Global', lat, lon, patch_size)
print(region_info)