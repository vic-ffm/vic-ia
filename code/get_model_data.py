import pandas as pd
import numpy as np
from includes import base_include
from includes.base_include import *
from modelling_functions import *

# Groups of modelling features
weather_features = ['T_SFC', 'T_SFC_ishistorical', 'RH_SFC', 'DF_SFC', 'WindMagKmh_SFC', 'KBDI']
dem_features = ['elevation_m', 'ruggedness_average_3km']
urban_density_features = ['building_density_20km', 'building_density_3km',  'road_density_km_in_3km', 'road_distance_m']
vegetation_features = ['grass_density_3km', 'forest_density_3km', 'shrub_density_3km', 'noveg_density_3km', 'distance_to_interface', 'soil_moisture', 'Curing']
model_features = weather_features + dem_features + urban_density_features + vegetation_features
# Modelling outcomes
outcomes = ['uncontrolled_within_2_hrs', 'uncontrolled_within_4_hrs', 'uncontrolled_within_5_ha', 'uncontrolled_within_100_ha']

# Read in data and add features
incidents = pd.read_pickle(MODEL_INPUT)
incidents = (incidents
 .assign(agency=incidents.agency.astype('category'),
         primary_fuel_type=incidents.primary_fuel_type.astype('category'),
         ffmv_region=incidents.ffmv_region.astype('category'),
         T_SFC = incidents.T_SFC.fillna(incidents.T_SFC_historical),
         RH_SFC = incidents.RH_SFC.fillna(incidents.RH_SFC_historical),
         DF_SFC = incidents.DF_SFC.fillna(incidents.DF_SFC_historical),
         WindMagKmh_SFC = incidents.WindMagKmh_SFC.fillna(incidents.WindMagKmh_SFC_historical),
         KBDI = incidents.KBDI.fillna(incidents.KBDI_historical),
         KBDI_nextday = incidents.KBDI_nextday.fillna(incidents.KBDI_nextday_historical),
         Curing = incidents.Curing.fillna(incidents.Curing_historical),
         FFDI = incidents.FFDI.fillna(incidents.FFDI_historical),
         GFDI = incidents.GFDI.fillna(incidents.GFDI_historical),
         T_SFC_ishistorical = np.where(incidents.T_SFC.isna() & ~incidents.T_SFC_historical.isna(), 1, 0),
         RH_SFC_ishistorical = np.where(incidents.RH_SFC.isna() & ~incidents.RH_SFC_historical.isna(), 1, 0),
         DF_SFC_ishistorical = np.where(incidents.DF_SFC.isna() & ~incidents.DF_SFC_historical.isna(), 1, 0),
         WindMagKmh_SFC_ishistorical = np.where(incidents.WindMagKmh_SFC.isna() & ~incidents.WindMagKmh_SFC_historical.isna(), 1, 0),
         KBDI_ishistorical = np.where(incidents.KBDI.isna() & ~incidents.KBDI_historical.isna(), 1, 0),
         Curing_ishistorical = np.where(incidents.Curing.isna() & ~incidents.Curing_historical.isna(), 1, 0),
         grass_density_3km = incidents.grass_density_3km*100, # convert to percentage to make log scaling easier
         forest_density_3km = incidents.forest_density_3km*100,
         shrub_density_3km = incidents.shrub_density_3km*100,
         noveg_density_3km = incidents.noveg_density_3km*100,
         soil_moisture = incidents.soil_moisture*100,
        )
)
incidents = (incidents
             .drop(columns=incidents.columns[incidents.columns.str.endswith('_historical')])
             .query(f'reported_time.dt.date < datetime.date(2024, 4, 1)') # set a "clean" end date for our data
            )[['agency', 'fire_name', 'reported_time', 'point', 'season', 'primary_fuel_type', 'FFDI', 'GFDI'] + model_features + outcomes]
incidents = incidents.dropna(subset = model_features + outcomes)

# split data by primary fuel type and remove the veg_density_3km of the primary fuel type in each group, because that's our reference
grass_incidents = (incidents
                   .query('primary_fuel_type=="grass"')
                   .drop(columns=['grass_density_3km'])
                   )
forest_incidents = (incidents
                   .query('primary_fuel_type=="forest"')
                   .drop(columns=['forest_density_3km'])
                   )
shrub_incidents = (incidents
                   .query('primary_fuel_type=="shrub"')
                   .drop(columns=['shrub_density_3km'])
                   )
noveg_incidents = (incidents
                   .query('primary_fuel_type=="noveg"')
                   .drop(columns=['noveg_density_3km'])
                   )
grass_features = model_features.copy()
grass_features.remove('grass_density_3km')
forest_features = model_features.copy()
forest_features.remove('forest_density_3km')
shrub_features = model_features.copy()
shrub_features.remove('shrub_density_3km')

# add column for test/train split
grass_incidents = add_is_train_column(grass_incidents)
forest_incidents = add_is_train_column(forest_incidents)

# features selected in variable_selection notebook
grass_2_hrs_features = ['T_SFC', 'T_SFC_ishistorical', 'RH_SFC', 'WindMagKmh_SFC', 'KBDI', 'ruggedness_average_3km', 'building_density_3km', 'road_density_km_in_3km', 'road_distance_m', 'forest_density_3km', 'shrub_density_3km', 'soil_moisture', 'Curing']
grass_100_ha_features = ['T_SFC', 'DF_SFC', 'WindMagKmh_SFC', 'ruggedness_average_3km', 'building_density_20km', 'road_density_km_in_3km', 'road_distance_m', 'noveg_density_3km', 'distance_to_interface', 'Curing']
forest_4_hrs_features = ['T_SFC', 'KBDI', 'elevation_m', 'road_density_km_in_3km', 'road_distance_m', 'grass_density_3km', 'shrub_density_3km', 'noveg_density_3km', 'distance_to_interface', 'Curing']
forest_5_ha_features = ['RH_SFC', 'DF_SFC', 'WindMagKmh_SFC', 'elevation_m', 'building_density_20km', 'road_density_km_in_3km', 'road_distance_m', 'grass_density_3km']



