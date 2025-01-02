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
from includes.base_include import *
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

count = incidents.groupby('primary_fuel_type').count().agency
print(count)
print(count/count.sum()*100)

# COMMAND ----------

count = incidents.query('primary_fuel_type=="grass"').groupby('uncontrolled_within_2_hrs').count().agency
print(count)
print(count/count.sum()*100)

# COMMAND ----------

count = incidents.query('primary_fuel_type=="grass"').groupby('uncontrolled_within_100_ha').count().agency
print(count)
print(count/count.sum()*100)

# COMMAND ----------

count = incidents.query('primary_fuel_type=="forest"').groupby('uncontrolled_within_4_hrs').count().agency
print(count)
print(count/count.sum()*100)

# COMMAND ----------

count = incidents.query('primary_fuel_type=="forest"').groupby('uncontrolled_within_5_ha').count().agency
print(count)
print(count/count.sum()*100)

# COMMAND ----------

grass_incidents.groupby('season').is_train_data.unique()

# COMMAND ----------

forest_incidents.groupby('season').is_train_data.unique()

# COMMAND ----------

(13/17, 4/17)

# COMMAND ----------

grass_incidents.is_train_data.value_counts()/grass_incidents.shape[0]

# COMMAND ----------

forest_incidents.is_train_data.value_counts()/forest_incidents.shape[0]

# COMMAND ----------

"uncontrolled_within_2_hrs_p".replace('_p', '').replace('_', ' ').capitalize()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modelling

# COMMAND ----------

# MAGIC %md
# MAGIC ### All

# COMMAND ----------

all_fit = {}
for outcome in outcomes:
    all_fit[outcome] = fit_logistic_regression(incidents, outcome, model_features=grass_features)
    all_fit

# COMMAND ----------

# MAGIC %md
# MAGIC ### Grass

# COMMAND ----------

grass_2_hrs_features

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

outcome = 'uncontrolled_within_2_hrs'
get_model_diagnostics(rslt = grass_fit[outcome][0], 
                      incidents_train = grass_fit[outcome][1], 
                      incidents_test = grass_fit[outcome][2],
                      outcome = outcome,
                      print_density=True,
                      model_name_text='Grass: ',
                      save_to=DENSITY / (outcome + '_density.eps'))

# COMMAND ----------

outcome = 'uncontrolled_within_100_ha'
get_model_diagnostics(rslt = grass_fit[outcome][0], 
                      incidents_train = grass_fit[outcome][1], 
                      incidents_test = grass_fit[outcome][2],
                      outcome = outcome,
                      print_density=True,
                      model_name_text='Grass: ',
                      save_to=DENSITY / (outcome + '_density.eps'))

# COMMAND ----------

outcome = 'uncontrolled_within_2_hrs'
get_model_diagnostics(rslt = grass_fit[outcome][0], 
                      incidents_train = grass_fit[outcome][1], 
                      incidents_test = grass_fit[outcome][2],
                      outcome = outcome,
                      print_appendix_diagnostics=True,
                      save_to = DIAGNOSTICS / (outcome + '_logistic_diagnostics.eps'))

# COMMAND ----------

outcome = 'uncontrolled_within_100_ha'
get_model_diagnostics(rslt = grass_fit[outcome][0], 
                      incidents_train = grass_fit[outcome][1], 
                      incidents_test = grass_fit[outcome][2],
                      outcome = outcome,
                      print_appendix_diagnostics=True,
                      save_to = DIAGNOSTICS / (outcome + '_logistic_diagnostics.eps'))

# COMMAND ----------

for outcome in ['uncontrolled_within_2_hrs', 'uncontrolled_within_100_ha']:
    print(grass_fit[outcome][0].summary())

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
                      model_name_text='GFDI: ',
                      save_to=DENSITY / (outcome + '_gfdi_density.eps'))

# COMMAND ----------

outcome = 'uncontrolled_within_100_ha'
get_model_diagnostics(rslt = grass_fit_fdi[outcome][0], 
                      incidents_train = grass_fit_fdi[outcome][1], 
                      incidents_test = grass_fit_fdi[outcome][2],
                      outcome = outcome,
                      print_density=True,
                      model_name_text='GFDI: ',
                      save_to=DENSITY / (outcome + '_gfdi_density.eps'))

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

outcome = 'uncontrolled_within_4_hrs'
get_model_diagnostics(rslt = forest_fit[outcome][0], 
                      incidents_train = forest_fit[outcome][1], 
                      incidents_test = forest_fit[outcome][2],
                      outcome = outcome,
                      print_density=True,
                      model_name_text='Forest: ',
                      save_to=DENSITY / (outcome + '_density.eps'))

# COMMAND ----------

outcome = 'uncontrolled_within_5_ha'
get_model_diagnostics(rslt = forest_fit[outcome][0], 
                      incidents_train = forest_fit[outcome][1], 
                      incidents_test = forest_fit[outcome][2],
                      outcome = outcome,
                      print_density=True,
                      model_name_text='Forest: ',
                      save_to=DENSITY / (outcome + '_density.eps'))

# COMMAND ----------

outcome = 'uncontrolled_within_4_hrs'
get_model_diagnostics(rslt = forest_fit[outcome][0], 
                      incidents_train = forest_fit[outcome][1], 
                      incidents_test = forest_fit[outcome][2],
                      outcome = outcome,
                      print_appendix_diagnostics=True,
                      save_to = DIAGNOSTICS / (outcome + '_logistic_diagnostics.eps'))

# COMMAND ----------

outcome = 'uncontrolled_within_5_ha'
get_model_diagnostics(rslt = forest_fit[outcome][0], 
                      incidents_train = forest_fit[outcome][1], 
                      incidents_test = forest_fit[outcome][2],
                      outcome = outcome,
                      print_appendix_diagnostics=True,
                      save_to = DIAGNOSTICS / (outcome + '_logistic_diagnostics.eps'))

# COMMAND ----------

for outcome in ['uncontrolled_within_4_hrs', 'uncontrolled_within_5_ha']:
    print(forest_fit[outcome][0].summary())

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
                      model_name_text='FFDI: ',
                      save_to=DENSITY / (outcome + '_ffdi_density.eps'))

# COMMAND ----------

outcome = 'uncontrolled_within_5_ha'
get_model_diagnostics(rslt = forest_fit_fdi[outcome][0], 
                      incidents_train = forest_fit_fdi[outcome][1], 
                      incidents_test = forest_fit_fdi[outcome][2],
                      outcome = outcome,
                      print_density=True,
                      model_name_text='FFDI: ',
                      save_to=DENSITY / (outcome + '_ffdi_density.eps'))

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

model_features

# COMMAND ----------

outcome = 'uncontrolled_within_2_hrs'
features = [x for x in grass_2_hrs_features if x != 'T_SFC_ishistorical']
plot_quantile_residuals(grass_fit[outcome][2], outcome, outcome + '_p', features, log_transform=grass_2_hrs_transform_features, model_name_text='Grass: ', save_to=RESIDUALS / (outcome + '.eps'))

# COMMAND ----------

outcome = 'uncontrolled_within_100_ha'
plot_quantile_residuals(grass_fit[outcome][2], outcome, outcome + '_p', grass_100_ha_features, log_transform=grass_100_ha_transform_features, model_name_text='Grass: ', save_to=RESIDUALS / (outcome + '.eps'))

# COMMAND ----------

outcome = 'uncontrolled_within_4_hrs'
plot_quantile_residuals(forest_fit[outcome][2], outcome, outcome + '_p', forest_4_hrs_features, log_transform=forest_4_hrs_transform_features, model_name_text='Forest: ', save_to=RESIDUALS / (outcome + '.eps'))

# COMMAND ----------

outcome = 'uncontrolled_within_5_ha'
plot_quantile_residuals(forest_fit[outcome][2], outcome, outcome + '_p', forest_5_ha_features, log_transform=forest_5_ha_transform_features, model_name_text='Forest: ', save_to=RESIDUALS / (outcome + '.eps'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Influence

# COMMAND ----------

# MAGIC %md
# MAGIC - I can't remember what the studentised residuals plots should show.

# COMMAND ----------

potential_outliers = {}
for outcome in ['uncontrolled_within_2_hrs', 'uncontrolled_within_100_ha']:
    potential_outliers['grass', outcome] = check_influence(grass_incidents, grass_fit[outcome][0], outcome)

# COMMAND ----------

for outcome in ['uncontrolled_within_4_hrs', 'uncontrolled_within_5_ha']:
    potential_outliers['forest', outcome] = check_influence(forest_incidents, forest_fit[outcome][0], outcome)

# COMMAND ----------

incidents.loc[potential_outliers[('grass', 'uncontrolled_within_100_ha')][0], ['uncontrolled_within_100_ha'] + model_features]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Linear contribution distribution
# MAGIC - Multiply each data column by its coefficient and plot it as a histogram/kdeplot
# MAGIC - Compare to see which variables give the largest contribution: very positive and very negative

# COMMAND ----------

# MAGIC %md
# MAGIC #### Grass, 2 hrs

# COMMAND ----------

grass_fit['uncontrolled_within_2_hrs'][0].summary()

# COMMAND ----------



plot_feature_distributions(grass_incidents, grass_fit['uncontrolled_within_2_hrs'][0], grass_2_hrs_features, log_transform=grass_2_hrs_transform_features)

# COMMAND ----------

plot_feature_distributions(grass_incidents, grass_fit['uncontrolled_within_2_hrs'][0], ['T_SFC', 'RH_SFC', 'WindMagKmh_SFC'], log_transform=grass_2_hrs_transform_features)

# COMMAND ----------

plot_feature_distributions(grass_incidents, grass_fit['uncontrolled_within_2_hrs'][0],['KBDI',  'Curing'], log_transform=grass_2_hrs_transform_features)

# COMMAND ----------

plot_feature_distributions(grass_incidents, grass_fit['uncontrolled_within_2_hrs'][0], ['soil_moisture'], log_transform=grass_2_hrs_transform_features)

# COMMAND ----------

plot_feature_distributions(grass_incidents, grass_fit['uncontrolled_within_2_hrs'][0], ['ruggedness_average_3km', 'building_density_3km', 'road_density_km_in_3km', 'road_distance_m'], log_transform=grass_2_hrs_transform_features)

# COMMAND ----------

plot_feature_distributions(grass_incidents, grass_fit['uncontrolled_within_2_hrs'][0], ['forest_density_3km', 'shrub_density_3km'], log_transform=grass_2_hrs_transform_features)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Grass, 100 ha

# COMMAND ----------

grass_fit['uncontrolled_within_100_ha'][0].summary()

# COMMAND ----------

plot_feature_distributions(grass_incidents, grass_fit['uncontrolled_within_100_ha'][0], grass_100_ha_features, log_transform=grass_100_ha_transform_features)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Forest, 4 hrs

# COMMAND ----------

forest_fit['uncontrolled_within_4_hrs'][0].summary()

# COMMAND ----------

plot_feature_distributions(forest_incidents, forest_fit['uncontrolled_within_4_hrs'][0], forest_4_hrs_features, log_transform=forest_4_hrs_transform_features)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Forest, 5 ha

# COMMAND ----------

forest_fit['uncontrolled_within_5_ha'][0].summary()

# COMMAND ----------

plot_feature_distributions(forest_incidents, forest_fit['uncontrolled_within_5_ha'][0], forest_5_ha_features, log_transform=forest_5_ha_transform_features)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature distribution as confusion matrix

# COMMAND ----------

# MAGIC %md
# MAGIC TODO: To run this plot, need to pull out the code that does the classification from `get_model_diagnostics()` and put it back into `fit_logistic_regression()`.

# COMMAND ----------

def plot_feature_distribution_as_confusion_matrix(data, outcome, features, n_cols=4):

    tp_df = data.query(f'{outcome}==1 & {outcome}_threshold_roc==1')
    fp_df = data.query(f'{outcome}==0 & {outcome}_threshold_roc==1')
    fn_df = data.query(f'{outcome}==1 & {outcome}_threshold_roc==0')
    tn_df = data.query(f'{outcome}==0 & {outcome}_threshold_roc==0')

    n_rows=int(np.ceil(len(features)/n_cols))
    fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(21, 14))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.suptitle(outcome, fontsize=14, y=0.9)
    for row in range(n_rows):
        if (row<n_rows-1) | (len(features)%n_cols==0):
            col_range=n_cols
        else:
            col_range=len(features)%n_cols
        for col in range(col_range):
            if n_rows==1:
                axis = axs[col]
            else:
                axis = axs[row,col]
            feature = features[row*n_cols + col]
            sns.kdeplot(x=feature, data=tp_df, label='tp_df', color='orange', fill=True, ax=axis)
            sns.kdeplot(x=feature, data=fp_df, label='fp_df', linestyle='--', color='orange', fill=True, ax=axis)
            sns.kdeplot(x=feature, data=fn_df, label='fn_df', linestyle='--', color='lightblue', fill=True, ax=axis)
            sns.kdeplot(x=feature, data=tn_df, label='tn_df', color='lightblue', fill=True, ax=axis)
            axis.legend()
    plt.show()

# COMMAND ----------

# plot_feature_distribution_as_confusion_matrix(forest_fit['uncontrolled_within_5_ha'][2], 'uncontrolled_within_5_ha', forest_5_ha_features)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Correlations in model features

# COMMAND ----------

# MAGIC %md
# MAGIC Looking at combined contributions of correlated features to see if they cancel each other out. (Leaving this here as a reminder of what I did.)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Grass, 2 hrs

# COMMAND ----------

(pd.concat([grass_fit['uncontrolled_within_2_hrs'][0].params, grass_fit['uncontrolled_within_2_hrs'][0].pvalues], axis=1)
 .rename(columns={0: 'coef', 1: 'pvalue'})
 .sort_values('pvalue'))

# COMMAND ----------

smg.plot_corr(np.corrcoef(grass_incidents[grass_2_hrs_features].T), xnames=grass_2_hrs_features, normcolor=True);

# COMMAND ----------

plot_summed_feature_distribution(grass_incidents, grass_fit['uncontrolled_within_2_hrs'][0], ['T_SFC', 'RH_SFC'])

# COMMAND ----------

# TODO: if decide to continue with this analysis, need to implement log1p transforms
# plot_summed_feature_distribution(grass_incidents, grass_fit['uncontrolled_within_2_hrs'][0], ['road_density_km_in_3km', 'building_density_3km'])

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


