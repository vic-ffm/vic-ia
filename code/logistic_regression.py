# Databricks notebook source
# Refresh input files when they change
%load_ext autoreload
%autoreload 2

# COMMAND ----------

# Import Python packages
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime as dt
from datetime import timedelta

import sklearn.metrics
# from sklearn.model_selection import cross_val_score, train_test_split
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import mlflow
mlflow.sklearn.autolog(disable=True)
mlflow.statsmodels.autolog(disable=True)
from pathlib import Path

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.graphics.api as smg
from scipy import stats

# Import setup file
from base_include import *
from get_model_data import *
from modelling_functions import *

# features selected in variable_selection notebook
grass_2_hrs_features = ['T_SFC', 'T_SFC_ishistorical', 'RH_SFC', 'WindMagKmh_SFC', 'KBDI', 'ruggedness_average_3km', 'building_density_3km', 'road_density_km_in_3km', 'road_distance_m', 'forest_density_3km', 'shrub_density_3km', 'soil_moisture', 'Curing']
grass_100_ha_features = ['T_SFC', 'DF_SFC', 'WindMagKmh_SFC', 'ruggedness_average_3km', 'building_density_20km', 'road_density_km_in_3km', 'road_distance_m', 'noveg_density_3km', 'distance_to_interface', 'Curing']
forest_4_hrs_features = ['T_SFC', 'KBDI', 'elevation_m', 'road_density_km_in_3km', 'road_distance_m', 'grass_density_3km', 'shrub_density_3km', 'noveg_density_3km', 'distance_to_interface', 'Curing']
forest_5_ha_features = ['RH_SFC', 'DF_SFC', 'WindMagKmh_SFC', 'elevation_m', 'building_density_20km', 'road_density_km_in_3km', 'road_distance_m', 'grass_density_3km']

# COMMAND ----------

incidents.shape

# COMMAND ----------

incidents.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data summary for paper

# COMMAND ----------

count = incidents.groupby('primary_fuel_type').count().season
print(count)
print(count/count.sum()*100)

# COMMAND ----------

count = incidents.query('primary_fuel_type=="grass"').groupby('uncontrolled_within_2_hrs').count().season
print(count)
print(count/count.sum()*100)

# COMMAND ----------

count = incidents.query('primary_fuel_type=="grass"').groupby('uncontrolled_within_100_ha').count().season
print(count)
print(count/count.sum()*100)

# COMMAND ----------

count = incidents.query('primary_fuel_type=="forest"').groupby('uncontrolled_within_4_hrs').count().season
print(count)
print(count/count.sum()*100)

# COMMAND ----------

count = incidents.query('primary_fuel_type=="forest"').groupby('uncontrolled_within_5_ha').count().season
print(count)
print(count/count.sum()*100)

# COMMAND ----------

grass_incidents.groupby('season').is_train_data.unique()

# COMMAND ----------

forest_incidents.groupby('season').is_train_data.unique()

# COMMAND ----------

grass_incidents.is_train_data.value_counts()/grass_incidents.shape[0]

# COMMAND ----------

forest_incidents.is_train_data.value_counts()/forest_incidents.shape[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modelling

# COMMAND ----------

# MAGIC %md
# MAGIC ### Grass

# COMMAND ----------

grass_fit = {}
grass_2_hrs_transform_features = ['WindMagKmh_SFC', 'ruggedness_average_3km', 'building_density_3km', 'road_density_km_in_3km', 'road_distance_m', 'shrub_density_3km', 'soil_moisture']
grass_100_ha_transform_features = ['WindMagKmh_SFC', 'ruggedness_average_3km', 'building_density_20km', 'road_density_km_in_3km', 'road_distance_m', 'noveg_density_3km', 'distance_to_interface']
grass_fit['uncontrolled_within_2_hrs'] = fit_logistic_regression(grass_incidents, 'uncontrolled_within_2_hrs', model_features=grass_2_hrs_features, 
                                                                 transform={key: 'np.log1p' for key in grass_2_hrs_transform_features})
get_model_diagnostics(rslt = grass_fit['uncontrolled_within_2_hrs'][0], 
                      incidents_train = grass_fit['uncontrolled_within_2_hrs'][1], 
                      incidents_test = grass_fit['uncontrolled_within_2_hrs'][2],
                      outcome = 'uncontrolled_within_2_hrs',
                      print_diagnostics=True, print_confusion_matrix=True)
grass_fit['uncontrolled_within_100_ha'] = fit_logistic_regression(grass_incidents, 'uncontrolled_within_100_ha', model_features=grass_100_ha_features,
                                                                  transform={key: 'np.log1p' for key in grass_100_ha_transform_features})
get_model_diagnostics(rslt = grass_fit['uncontrolled_within_100_ha'][0], 
                      incidents_train = grass_fit['uncontrolled_within_100_ha'][1], 
                      incidents_test = grass_fit['uncontrolled_within_100_ha'][2],
                      outcome = 'uncontrolled_within_100_ha',
                      print_diagnostics=True, print_confusion_matrix=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Paper plots and model summary

# COMMAND ----------

outcome = 'uncontrolled_within_2_hrs'
get_model_diagnostics(rslt = grass_fit[outcome][0], 
                      incidents_train = grass_fit[outcome][1], 
                      incidents_test = grass_fit[outcome][2],
                      outcome = outcome,
                      print_density=True,
                      model_name_text='Grass: ')

# COMMAND ----------

outcome = 'uncontrolled_within_100_ha'
get_model_diagnostics(rslt = grass_fit[outcome][0], 
                      incidents_train = grass_fit[outcome][1], 
                      incidents_test = grass_fit[outcome][2],
                      outcome = outcome,
                      print_density=True,
                      model_name_text='Grass: ')

# COMMAND ----------

outcome = 'uncontrolled_within_2_hrs'
get_model_diagnostics(rslt = grass_fit[outcome][0], 
                      incidents_train = grass_fit[outcome][1], 
                      incidents_test = grass_fit[outcome][2],
                      outcome = outcome,
                      print_appendix_diagnostics=True)

# COMMAND ----------

outcome = 'uncontrolled_within_100_ha'
get_model_diagnostics(rslt = grass_fit[outcome][0], 
                      incidents_train = grass_fit[outcome][1], 
                      incidents_test = grass_fit[outcome][2],
                      outcome = outcome,
                      print_appendix_diagnostics=True)

# COMMAND ----------

for outcome in ['uncontrolled_within_2_hrs', 'uncontrolled_within_100_ha']:
    print(grass_fit[outcome][0].summary())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Comparison with GFDI

# COMMAND ----------

grass_fit_fdi = {}
for outcome in ['uncontrolled_within_2_hrs', 'uncontrolled_within_100_ha']:
    grass_fit_fdi[outcome] = fit_logistic_regression(grass_incidents.dropna(subset='GFDI'), outcome, model_features=['GFDI'])
    get_model_diagnostics(rslt = grass_fit_fdi[outcome][0], 
                          incidents_train = grass_fit_fdi[outcome][1], 
                          incidents_test = grass_fit_fdi[outcome][2],
                          outcome = outcome,
                          print_diagnostics=True, print_confusion_matrix=True)

# COMMAND ----------

outcome = 'uncontrolled_within_2_hrs'
get_model_diagnostics(rslt = grass_fit_fdi[outcome][0], 
                      incidents_train = grass_fit_fdi[outcome][1], 
                      incidents_test = grass_fit_fdi[outcome][2],
                      outcome = outcome,
                      print_density=True,
                      model_name_text='GFDI: ')

# COMMAND ----------

outcome = 'uncontrolled_within_100_ha'
get_model_diagnostics(rslt = grass_fit_fdi[outcome][0], 
                      incidents_train = grass_fit_fdi[outcome][1], 
                      incidents_test = grass_fit_fdi[outcome][2],
                      outcome = outcome,
                      print_density=True,
                      model_name_text='GFDI: ')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Forest

# COMMAND ----------

forest_fit = {}
forest_4_hrs_transform_features = ['road_density_km_in_3km', 'road_distance_m', 'noveg_density_3km', 'distance_to_interface']
forest_5_ha_transform_features = ['WindMagKmh_SFC', 'building_density_20km', 'road_density_km_in_3km', 'road_distance_m']
forest_fit['uncontrolled_within_4_hrs'] = fit_logistic_regression(forest_incidents, 'uncontrolled_within_4_hrs', model_features=forest_4_hrs_features, 
                                                                  transform={key: 'np.log1p' for key in forest_4_hrs_transform_features})
get_model_diagnostics(rslt = forest_fit['uncontrolled_within_4_hrs'][0], 
                      incidents_train = forest_fit['uncontrolled_within_4_hrs'][1], 
                      incidents_test = forest_fit['uncontrolled_within_4_hrs'][2],
                      outcome = 'uncontrolled_within_4_hrs',
                      print_diagnostics=True, print_confusion_matrix=True)
forest_fit['uncontrolled_within_5_ha'] = fit_logistic_regression(forest_incidents, 'uncontrolled_within_5_ha', model_features=forest_5_ha_features, 
                                                                 transform={key: 'np.log1p' for key in forest_5_ha_transform_features})
get_model_diagnostics(rslt = forest_fit['uncontrolled_within_5_ha'][0], 
                      incidents_train = forest_fit['uncontrolled_within_5_ha'][1], 
                      incidents_test = forest_fit['uncontrolled_within_5_ha'][2],
                      outcome = 'uncontrolled_within_5_ha',
                      print_diagnostics=True, print_confusion_matrix=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Paper plots and model summary

# COMMAND ----------

outcome = 'uncontrolled_within_4_hrs'
get_model_diagnostics(rslt = forest_fit[outcome][0], 
                      incidents_train = forest_fit[outcome][1], 
                      incidents_test = forest_fit[outcome][2],
                      outcome = outcome,
                      print_density=True,
                      model_name_text='Forest: ')

# COMMAND ----------

outcome = 'uncontrolled_within_5_ha'
get_model_diagnostics(rslt = forest_fit[outcome][0], 
                      incidents_train = forest_fit[outcome][1], 
                      incidents_test = forest_fit[outcome][2],
                      outcome = outcome,
                      print_density=True,
                      model_name_text='Forest: ')

# COMMAND ----------

outcome = 'uncontrolled_within_4_hrs'
get_model_diagnostics(rslt = forest_fit[outcome][0], 
                      incidents_train = forest_fit[outcome][1], 
                      incidents_test = forest_fit[outcome][2],
                      outcome = outcome,
                      print_appendix_diagnostics=True)

# COMMAND ----------

outcome = 'uncontrolled_within_5_ha'
get_model_diagnostics(rslt = forest_fit[outcome][0], 
                      incidents_train = forest_fit[outcome][1], 
                      incidents_test = forest_fit[outcome][2],
                      outcome = outcome,
                      print_appendix_diagnostics=True)

# COMMAND ----------

for outcome in ['uncontrolled_within_4_hrs', 'uncontrolled_within_5_ha']:
    print(forest_fit[outcome][0].summary())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Comparison with FFDI

# COMMAND ----------

forest_fit_fdi = {}
for outcome in ['uncontrolled_within_4_hrs', 'uncontrolled_within_5_ha']:
    forest_fit_fdi[outcome] = fit_logistic_regression(forest_incidents, outcome, model_features=['FFDI'])
    get_model_diagnostics(rslt = forest_fit_fdi[outcome][0], 
                          incidents_train = forest_fit_fdi[outcome][1], 
                          incidents_test = forest_fit_fdi[outcome][2],
                          outcome = outcome,
                          print_diagnostics=True, print_confusion_matrix=True)

# COMMAND ----------

outcome = 'uncontrolled_within_4_hrs'
get_model_diagnostics(rslt = forest_fit_fdi[outcome][0], 
                      incidents_train = forest_fit_fdi[outcome][1], 
                      incidents_test = forest_fit_fdi[outcome][2],
                      outcome = outcome,
                      print_density=True,
                      model_name_text='FFDI: ')

# COMMAND ----------

outcome = 'uncontrolled_within_5_ha'
get_model_diagnostics(rslt = forest_fit_fdi[outcome][0], 
                      incidents_train = forest_fit_fdi[outcome][1], 
                      incidents_test = forest_fit_fdi[outcome][2],
                      outcome = outcome,
                      print_density=True,
                      model_name_text='FFDI: ')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model testing

# COMMAND ----------

# MAGIC %md
# MAGIC ### Colinearity

# COMMAND ----------

# MAGIC %md VIF exceeding 5 or 10 suggests a problematic amount of collinearity (from ISLP). All the VIF scores are less than 5, so don't have collinearity issues.

# COMMAND ----------

compute_vif(grass_incidents, grass_features)

# COMMAND ----------

compute_vif(forest_incidents, forest_features)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Residuals

# COMMAND ----------

# MAGIC %md
# MAGIC - Residuals look pretty flat and even above/below. 100 ha outcome possibly exhibiting cone-like structure, but I think that's more skewing than cone.

# COMMAND ----------

outcome = 'uncontrolled_within_2_hrs'
features = [x for x in grass_2_hrs_features if x != 'T_SFC_ishistorical']
plot_quantile_residuals(grass_fit[outcome][2], outcome, outcome + '_p', features, log_transform=grass_2_hrs_transform_features, model_name_text='Grass: ')

# COMMAND ----------

outcome = 'uncontrolled_within_100_ha'
plot_quantile_residuals(grass_fit[outcome][2], outcome, outcome + '_p', grass_100_ha_features, log_transform=grass_100_ha_transform_features, model_name_text='Grass: ')

# COMMAND ----------

outcome = 'uncontrolled_within_4_hrs'
plot_quantile_residuals(forest_fit[outcome][2], outcome, outcome + '_p', forest_4_hrs_features, log_transform=forest_4_hrs_transform_features, model_name_text='Forest: ')

# COMMAND ----------

outcome = 'uncontrolled_within_5_ha'
plot_quantile_residuals(forest_fit[outcome][2], outcome, outcome + '_p', forest_5_ha_features, log_transform=forest_5_ha_transform_features, model_name_text='Forest: ')

# COMMAND ----------

# MAGIC %md
# MAGIC
