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
# MAGIC # Case studies
# MAGIC Retrain models using all data, only for the time-based outcomes. 

# COMMAND ----------

grass_incidents['is_train_data'] = 1
outcome = 'uncontrolled_within_2_hrs'
grass_fit_all = {}
grass_fit_all[outcome] = fit_logistic_regression(grass_incidents, outcome, model_features=grass_2_hrs_features, 
                                                                 transform={key: 'np.log1p' for key in grass_2_hrs_transform_features})
get_model_diagnostics(rslt = grass_fit_all[outcome][0], 
                      incidents_train = grass_fit_all[outcome][1], 
                      incidents_test = grass_fit_all[outcome][1],
                      outcome = outcome,
                      print_diagnostics=True, print_confusion_matrix=True)

# COMMAND ----------

# Retrain models with all data, using only time based outcomes.
forest_incidents['is_train_data'] = 1
outcome = 'uncontrolled_within_4_hrs'
forest_fit_all = {}
forest_fit_all[outcome] = fit_logistic_regression(forest_incidents, outcome, model_features=forest_4_hrs_features, 
                                                                  transform={key: 'np.log1p' for key in forest_4_hrs_transform_features})
get_model_diagnostics(rslt = forest_fit_all[outcome][0], 
                      incidents_train = forest_fit_all[outcome][1], 
                      incidents_test = forest_fit_all[outcome][1],
                      outcome = outcome,
                      print_diagnostics=True, print_confusion_matrix=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test individual day performance
# MAGIC

# COMMAND ----------

day = dt(2009,2,7)
print('Grass performance on day {}'.format(str(day)))
incidents_train = grass_fit_all['uncontrolled_within_2_hrs'][1]
get_model_diagnostics(rslt = grass_fit_all['uncontrolled_within_2_hrs'][0], 
                      incidents_train = incidents_train,
                      incidents_test = incidents_train[(incidents_train['reported_time']>day) & (incidents_train['reported_time']<(day+timedelta(days=1)))],
                      outcome = 'uncontrolled_within_2_hrs',
                      print_diagnostics=True, print_confusion_matrix=True)

print('Forest performance on day {}'.format(str(day)))
incidents_train = forest_fit_all['uncontrolled_within_4_hrs'][1]
get_model_diagnostics(rslt = forest_fit_all['uncontrolled_within_4_hrs'][0], 
                      incidents_train = incidents_train,
                      incidents_test = incidents_train[(incidents_train['reported_time']>day) & (incidents_train['reported_time']<(day+timedelta(days=1)))],
                      outcome = 'uncontrolled_within_4_hrs',
                      print_diagnostics=True, print_confusion_matrix=True)

# COMMAND ----------

day = dt(2019,11,21)
print('Grass performance on day {}'.format(str(day)))
incidents_train = grass_fit_all['uncontrolled_within_2_hrs'][1]
get_model_diagnostics(rslt = grass_fit_all['uncontrolled_within_2_hrs'][0], 
                      incidents_train = incidents_train,
                      incidents_test = incidents_train[(incidents_train['reported_time']>day) & (incidents_train['reported_time']<(day+timedelta(days=1)))],
                      outcome = 'uncontrolled_within_2_hrs',
                      print_diagnostics=True, print_confusion_matrix=True)

print('Forest performance on day {}'.format(str(day)))
incidents_train = forest_fit_all['uncontrolled_within_4_hrs'][1]
get_model_diagnostics(rslt = forest_fit_all['uncontrolled_within_4_hrs'][0], 
                      incidents_train = incidents_train,
                      incidents_test = incidents_train[(incidents_train['reported_time']>day) & (incidents_train['reported_time']<(day+timedelta(days=1)))],
                      outcome = 'uncontrolled_within_4_hrs',
                      print_diagnostics=True, print_confusion_matrix=True)

# COMMAND ----------

day = dt(2023,2,17)
print('Grass performance on day {}'.format(str(day)))
incidents_train = grass_fit_all['uncontrolled_within_2_hrs'][1]
get_model_diagnostics(rslt = grass_fit_all['uncontrolled_within_2_hrs'][0], 
                      incidents_train = incidents_train,
                      incidents_test = incidents_train[(incidents_train['reported_time']>day) & (incidents_train['reported_time']<(day+timedelta(days=1)))],
                      outcome = 'uncontrolled_within_2_hrs',
                      print_diagnostics=True, print_confusion_matrix=True)

print('Forest performance on day {}'.format(str(day)))
incidents_train = forest_fit_all['uncontrolled_within_4_hrs'][1]
get_model_diagnostics(rslt = forest_fit_all['uncontrolled_within_4_hrs'][0], 
                      incidents_train = incidents_train,
                      incidents_test = incidents_train[(incidents_train['reported_time']>day) & (incidents_train['reported_time']<(day+timedelta(days=1)))],
                      outcome = 'uncontrolled_within_4_hrs',
                      print_diagnostics=True, print_confusion_matrix=True)

# COMMAND ----------

day = dt(2018,3,17)
print('Grass performance on day {}'.format(str(day)))
incidents_train = grass_fit_all['uncontrolled_within_2_hrs'][1]
get_model_diagnostics(rslt = grass_fit_all['uncontrolled_within_2_hrs'][0], 
                      incidents_train = incidents_train,
                      incidents_test = incidents_train[(incidents_train['reported_time']>day) & (incidents_train['reported_time']<(day+timedelta(days=1)))],
                      outcome = 'uncontrolled_within_2_hrs',
                      print_diagnostics=True, print_confusion_matrix=True)

print('Forest performance on day {}'.format(str(day)))
incidents_train = forest_fit_all['uncontrolled_within_4_hrs'][1]
get_model_diagnostics(rslt = forest_fit_all['uncontrolled_within_4_hrs'][0], 
                      incidents_train = incidents_train,
                      incidents_test = incidents_train[(incidents_train['reported_time']>day) & (incidents_train['reported_time']<(day+timedelta(days=1)))],
                      outcome = 'uncontrolled_within_4_hrs',
                      print_diagnostics=True, print_confusion_matrix=True)

# COMMAND ----------

day = dt(2024,2,13)
print('Grass performance on day {}'.format(str(day)))
incidents_train = grass_fit_all['uncontrolled_within_2_hrs'][1]
get_model_diagnostics(rslt = grass_fit_all['uncontrolled_within_2_hrs'][0], 
                      incidents_train = incidents_train,
                      incidents_test = incidents_train[(incidents_train['reported_time']>day) & (incidents_train['reported_time']<(day+timedelta(days=1)))],
                      outcome = 'uncontrolled_within_2_hrs',
                      print_diagnostics=True, print_confusion_matrix=True)

print('Forest performance on day {}'.format(str(day)))
incidents_train = forest_fit_all['uncontrolled_within_4_hrs'][1]
get_model_diagnostics(rslt = forest_fit_all['uncontrolled_within_4_hrs'][0], 
                      incidents_train = incidents_train,
                      incidents_test = incidents_train[(incidents_train['reported_time']>day) & (incidents_train['reported_time']<(day+timedelta(days=1)))],
                      outcome = 'uncontrolled_within_4_hrs',
                      print_diagnostics=True, print_confusion_matrix=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test performance at 90th percentile weather

# COMMAND ----------

# Join the 90th percentile FFDI for that cell
rank10_wx = gpd.read_file('/dbfs/mnt/raw/suppression/join_data/Rank10_wx_risk2.0.geojson')
rank10_wx =  rank10_wx.groupby(['geom_ref'], as_index=False).agg({'max_daily_ffdi':'first', 'geometry':'first'})
rank10_wx = gpd.GeoDataFrame(rank10_wx, geometry='geometry', crs='EPSG:3111')
incidents = incidents.set_geometry('point')
rank10_wx = rank10_wx.to_crs(incidents.crs)

incidents_join = gpd.sjoin(incidents, rank10_wx, how='left', predicate='within')
incidents_join = incidents_join.drop(columns=['index_right', 'geom_ref'])
incidents = incidents_join

# COMMAND ----------


print('Grass performance above 90th percentile')
incidents_train = grass_fit_all['uncontrolled_within_2_hrs'][1]
print(len(incidents_train))
incidents_train = incidents_train.merge(incidents[['max_daily_ffdi']], left_index=True, right_index=True, how='left')
print(len(incidents_train))
incidents_test = incidents_train[incidents_train['FFDI']>=incidents_train['max_daily_ffdi']]
get_model_diagnostics(rslt = grass_fit_all['uncontrolled_within_2_hrs'][0], 
                      incidents_train = incidents_train,
                      incidents_test = incidents_test,
                      outcome = 'uncontrolled_within_2_hrs',
                      print_diagnostics=True, print_confusion_matrix=True)

print('Forest performance above 90th perctile')
incidents_train = forest_fit_all['uncontrolled_within_4_hrs'][1]
print(len(incidents_train))
incidents_train = incidents_train.merge(incidents[['max_daily_ffdi']], left_index=True, right_index=True, how='left')
print(len(incidents_train))
incidents_test = incidents_train[incidents_train['FFDI']>=incidents_train['max_daily_ffdi']]
get_model_diagnostics(rslt = forest_fit_all['uncontrolled_within_4_hrs'][0], 
                      incidents_train = incidents_train,
                      incidents_test = incidents_test,
                      outcome = 'uncontrolled_within_4_hrs',
                      print_diagnostics=True, print_confusion_matrix=True)

# COMMAND ----------

incidents_test[incidents_test['uncontrolled_within_4_hrs']==1].sort_values(by=['uncontrolled_within_4_hrs_p'], ascending=False)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Combine into results table

# COMMAND ----------

days = [dt(2009,2,7).date(), dt(2019,11,21).date(), dt(2023,2,17).date(), dt(2024,2,13).date(), dt(2018,3,17).date()]

# Grass
outcome = 'uncontrolled_within_2_hrs'
incidents_train = grass_fit_all[outcome][1]
incidents_train['date'] = incidents_train['reported_time'].dt.date
incidents_test = incidents_train[incidents_train['date'].isin(days)]

outcome_prob = outcome + "_p"
outcome_threshold_roc = outcome + "_threshold_roc"
outcome_threshold_f1 = outcome + "_threshold_f1"

## From Precision-Recall (F1)
precision, recall, thresholds = sklearn.metrics.precision_recall_curve(incidents_train[outcome], incidents_train[outcome_prob])
with warnings.catch_warnings():
    # f1 score is nan when precision and recall are zero, which is the behaviour we want, so no need for a warning
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    f1_scores = 2*recall*precision/(recall+precision)    
threshold_f1 = thresholds[np.nanargmax(f1_scores)]
incidents_test = incidents_test.assign(**{outcome_threshold_f1: np.where(incidents_test[outcome_prob] > threshold_f1, 1, 0)})
results_table = incidents_test.groupby(['date', outcome_threshold_f1, outcome], as_index=False)['agency'].count()
results_table = results_table.pivot(index='date', columns=[outcome_threshold_f1, outcome], values='agency').fillna(0)
results_table.columns = ['TN', 'FN', 'TP', 'FP', ]
results_table['N'] = results_table.sum(axis=1)
results_table = results_table[['N', 'TP','TN', 'FP', 'FN' ]]
results_table['Accuracy'] = (results_table['TP']+results_table['TN'])/(results_table.sum(axis=1,))
results_table['Recall'] = results_table['TP']/(results_table['TP']+results_table['FN'])
results_table['FPR'] = results_table['FP']/(results_table['FP']+results_table['TN'])
results_table['Precision'] = results_table['TP']/(results_table['FP']+results_table['TP'])
results_table

# COMMAND ----------

# Forest
outcome = 'uncontrolled_within_4_hrs'
incidents_train = forest_fit_all[outcome][1]
incidents_train['date'] = incidents_train['reported_time'].dt.date
incidents_test = incidents_train[incidents_train['date'].isin(days)]

outcome_prob = outcome + "_p"
outcome_threshold_roc = outcome + "_threshold_roc"
outcome_threshold_f1 = outcome + "_threshold_f1"

## From Precision-Recall (F1)
precision, recall, thresholds = sklearn.metrics.precision_recall_curve(incidents_train[outcome], incidents_train[outcome_prob])
with warnings.catch_warnings():
    # f1 score is nan when precision and recall are zero, which is the behaviour we want, so no need for a warning
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    f1_scores = 2*recall*precision/(recall+precision)    
threshold_f1 = thresholds[np.nanargmax(f1_scores)]
incidents_test = incidents_test.assign(**{outcome_threshold_f1: np.where(incidents_test[outcome_prob] > threshold_f1, 1, 0)})
results_table = incidents_test.groupby(['date', outcome_threshold_f1, outcome], as_index=False)['agency'].count()
results_table = results_table.pivot(index='date', columns=[outcome_threshold_f1, outcome], values='agency').fillna(0)
results_table.columns = ['TN', 'FN', 'TP', 'FP', ]
results_table['N'] = results_table.sum(axis=1)
results_table = results_table[['N', 'TP','TN', 'FP', 'FN' ]]
results_table['Accuracy'] = (results_table['TP']+results_table['TN'])/(results_table.sum(axis=1,))
results_table['Recall'] = results_table['TP']/(results_table['TP']+results_table['FN'])
results_table['FPR'] = results_table['FP']/(results_table['FP']+results_table['TN'])
results_table['Precision'] = results_table['TP']/(results_table['FP']+results_table['TP'])
results_table

# COMMAND ----------

# MAGIC %md
# MAGIC ### Combine into results plot

# COMMAND ----------

# Grass
outcome = 'uncontrolled_within_2_hrs'
incidents_train = grass_fit_all[outcome][1]
incidents_train['date'] = incidents_train['reported_time'].dt.date
#incidents_test = incidents_train[incidents_train['date'].isin(days)]

outcome_prob = outcome + "_p"
outcome_threshold_roc = outcome + "_threshold_roc"
outcome_threshold_f1 = outcome + "_threshold_f1"

## From Precision-Recall (F1)
precision, recall, thresholds = sklearn.metrics.precision_recall_curve(incidents_train[outcome], incidents_train[outcome_prob])
with warnings.catch_warnings():
    # f1 score is nan when precision and recall are zero, which is the behaviour we want, so no need for a warning
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    f1_scores = 2*recall*precision/(recall+precision)    
threshold_f1 = thresholds[np.nanargmax(f1_scores)]
incidents_test = incidents_test.assign(**{outcome_threshold_f1: np.where(incidents_test[outcome_prob] > threshold_f1, 1, 0)})
results_table = incidents_test.groupby(['date', outcome_threshold_f1, outcome], as_index=False)['agency'].count()
results_table = results_table.pivot(index='date', columns=[outcome_threshold_f1, outcome], values='agency').fillna(0)
results_table.columns = ['TN', 'FN', 'TP', 'FP', ]
results_table['N'] = results_table.sum(axis=1)
results_table = results_table[['N', 'TP','TN', 'FP', 'FN' ]]
results_table['Accuracy'] = (results_table['TP']+results_table['TN'])/(results_table.sum(axis=1,))
results_table['Recall'] = results_table['TP']/(results_table['TP']+results_table['FN'])
results_table['FPR'] = results_table['FP']/(results_table['FP']+results_table['TN'])
results_table['Precision'] = results_table['TP']/(results_table['FP']+results_table['TP'])
results_table

# COMMAND ----------

incidents_train['incidents']

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------


