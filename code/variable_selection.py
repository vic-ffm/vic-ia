# Databricks notebook source
# MAGIC %md
# MAGIC # Variable selection for the first attack model

# COMMAND ----------

# Import Python packages
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import datetime

import sklearn.metrics
# from sklearn.model_selection import cross_val_score, train_test_split
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

import mlflow
mlflow.sklearn.autolog(disable=True)
mlflow.statsmodels.autolog(disable=True)
from pathlib import Path

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.graphics.api as smg
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Import setup file
from includes.base_include import *
from get_model_data import *
from modelling_functions import *

# COMMAND ----------

from datetime import datetime
datetime.now()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fit models

# COMMAND ----------

# DBTITLE 1,Grass
grass_fit = {}
for outcome in ['uncontrolled_within_2_hrs', 'uncontrolled_within_100_ha']:
    grass_fit[outcome] = fit_logistic_regression(grass_incidents, "grass", outcome, model_features=grass_features, print_diagnostics=True, print_confusion_matrix=True)

# COMMAND ----------

# DBTITLE 1,Forest
forest_fit = {}
for outcome in ['uncontrolled_within_4_hrs', 'uncontrolled_within_5_ha']:
    forest_fit[outcome] = fit_logistic_regression(forest_incidents, "forest", outcome, model_features=forest_features, print_diagnostics=True, print_confusion_matrix=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Variable selection by eye using linear contribution plots

# COMMAND ----------

# MAGIC %md
# MAGIC ### Grass

# COMMAND ----------

outcome = 'uncontrolled_within_2_hrs'
features = ['T_SFC', 'KBDI', 'WindMagKmh_SFC', 'road_density_km_in_3km', 'Curing']
plot_feature_distributions(grass_incidents, grass_fit[outcome][0], features)

# COMMAND ----------

grass_fit_selected = {}
grass_fit_selected[outcome] = fit_logistic_regression(grass_incidents, "grass", outcome, model_features=features, print_diagnostics=True, print_confusion_matrix=True)

# COMMAND ----------

outcome = 'uncontrolled_within_100_ha'
features = ['T_SFC', 'RH_SFC', 'DF_SFC', 'WindMagKmh_SFC', 'road_density_km_in_3km', 'Curing']
plot_feature_distributions(grass_incidents, grass_fit[outcome][0], features)

# COMMAND ----------

grass_fit_selected[outcome] = fit_logistic_regression(grass_incidents, "grass", outcome, model_features=features, print_diagnostics=True, print_confusion_matrix=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Forest

# COMMAND ----------

outcome = 'uncontrolled_within_4_hrs'
features = ['T_SFC', 'road_density_km_in_3km', 'grass_density_3km', 'Curing']
plot_feature_distributions(forest_incidents, forest_fit[outcome][0], features)

# COMMAND ----------

forest_fit_selected = {}
forest_fit_selected[outcome] = fit_logistic_regression(forest_incidents, "forest", outcome, model_features=features, print_diagnostics=True, print_confusion_matrix=True)

# COMMAND ----------

outcome = 'uncontrolled_within_5_ha'
features = ['T_SFC', 'RH_SFC', 'DF_SFC', 'road_density_km_in_3km', 'grass_density_3km', 'Curing']
plot_feature_distributions(forest_incidents, forest_fit[outcome][0], features)

# COMMAND ----------

forest_fit_selected[outcome] = fit_logistic_regression(forest_incidents, "forest", outcome, model_features=features, print_diagnostics=True, print_confusion_matrix=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Variable selection by Lasso regularisation

# COMMAND ----------

def vary_lasso_penalisation(incidents_unstandardised, features, outcome, alpha_range = np.arange(0,400, 20)):

    # rescale data
    scaler = StandardScaler()
    incidents_standardised = pd.DataFrame(scaler.fit_transform(incidents_unstandardised[features]), columns=features, index=incidents_unstandardised.index)
    incidents_standardised = pd.concat([incidents_standardised, incidents_unstandardised[outcome], incidents_unstandardised['season']], axis=1)

    # split data into test/train
    incidents_standardised = add_is_train_column(incidents_standardised)
    incidents_standardised_train = incidents_standardised.query('is_train_data==1')
    incidents_standardised_test = incidents_standardised.query('is_train_data==0')

    # fit model for a range of alpha values
    outcome_prob = outcome + "_p"
    metrics_df = pd.DataFrame(
        {
            'alpha': alpha_range,
            'roc_auc': np.nan,
            'average_precision_score': np.nan,
            'n_features': np.nan,
            'feature_names': np.nan
        }
    )

    for idx in metrics_df.index:
        rslt = smf.logit(formula=F"{outcome} ~ {'+'.join(features)}", data=incidents_standardised_train).fit_regularized(alpha=metrics_df.loc[idx, 'alpha'], disp=0)
        prediction = {outcome_prob: rslt.predict(exog=incidents_standardised_test), outcome: incidents_standardised_test[outcome]}
        # print(roc_auc_score(prediction[outcome], prediction[outcome_prob]))
        metrics_df.loc[idx, 'roc_auc'] = roc_auc_score(prediction[outcome], prediction[outcome_prob])
        metrics_df.loc[idx, 'average_precision_score'] = average_precision_score(prediction[outcome], prediction[outcome_prob])
        metrics_df.loc[idx, 'n_features'] = np.sum(rslt.params!=0)
        metrics_df.loc[idx, 'feature_names'] = ', '.join(rslt.params[rslt.params!=0].index)

    # plot how AUC and average precision decrease as number of parameters decrease
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    fig.suptitle(outcome) 
    sns.lineplot(x='n_features', y='roc_auc', data=metrics_df, ax=axs[0])
    sns.scatterplot(x='n_features', y='roc_auc', data=metrics_df, ax=axs[0])
    axs[0].set_xticks(np.unique(metrics_df.n_features))
    sns.lineplot(x='n_features', y='average_precision_score', data=metrics_df, ax=axs[1])
    sns.scatterplot(x='n_features', y='average_precision_score', data=metrics_df, ax=axs[1])
    axs[1].set_xticks(np.unique(metrics_df.n_features))
    fig.tight_layout()

    return metrics_df


# COMMAND ----------

# MAGIC %md
# MAGIC ### Grass

# COMMAND ----------

parameter_tuning_metrics = {}
parameter_tuning_metrics['grass_2hrs'] = vary_lasso_penalisation(grass_incidents, grass_features, outcomes[0]);
parameter_tuning_metrics['grass_100_ha'] = vary_lasso_penalisation(grass_incidents, grass_features, outcomes[3]);

# COMMAND ----------

parameter_tuning_metrics['grass_2hrs'].query('n_features==7').feature_names[9]

# COMMAND ----------

parameter_tuning_metrics['grass_100_ha'].query('n_features==7').feature_names[2]

# COMMAND ----------

grass_fit_lasso = {}
outcome = 'uncontrolled_within_2_hrs'
features = parameter_tuning_metrics['grass_2hrs'].query('n_features==7').feature_names[9].split(', ')[1:]
grass_fit_lasso[outcome] = fit_logistic_regression(grass_incidents, "grass", outcome, model_features=features, print_diagnostics=True)

# COMMAND ----------

plot_feature_distributions(grass_incidents, grass_fit_lasso[outcome][0], features)

# COMMAND ----------

outcome = 'uncontrolled_within_100_ha'
features = parameter_tuning_metrics['grass_100_ha'].query('n_features==7').feature_names[2].split(', ')[1:]
grass_fit_lasso[outcome] = fit_logistic_regression(grass_incidents, "grass", outcome, model_features=features, print_diagnostics=True)

# COMMAND ----------

plot_feature_distributions(grass_incidents, grass_fit_lasso[outcome][0], features)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Forest

# COMMAND ----------

parameter_tuning_metrics['forest_4_hrs'] = vary_lasso_penalisation(forest_incidents, forest_features, outcomes[1], alpha_range=np.arange(0,600,50));
parameter_tuning_metrics['forest_5_ha'] = vary_lasso_penalisation(forest_incidents, forest_features, outcomes[2], alpha_range=np.arange(0,600,50));

# COMMAND ----------

parameter_tuning_metrics['forest_4_hrs'].query('n_features==2')#.feature_names[9]

# COMMAND ----------

parameter_tuning_metrics['forest_5_ha'].query('n_features==2')#.feature_names[4]

# COMMAND ----------

forest_fit_lasso = {}
outcome = 'uncontrolled_within_4_hrs'
features = parameter_tuning_metrics['forest_4_hrs'].query('n_features==2').feature_names[9].split(', ')[1:]
forest_fit_lasso[outcome] = fit_logistic_regression(forest_incidents, "forest", outcome, model_features=features, print_diagnostics=True)

# COMMAND ----------

plot_feature_distributions(forest_incidents, forest_fit_lasso[outcome][0], features)

# COMMAND ----------

fig, ax = plt.subplots(1,1)
for feature in features:
    sns.kdeplot(data=(forest_fit_lasso[outcome][0].params[features[0]]*forest_incidents[[features[0]]]).join(forest_incidents[[outcome]]), x=feature, label=feature, hue=outcome, common_norm=False)
    ax.legend()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

# COMMAND ----------

outcome = 'uncontrolled_within_5_ha'
features = parameter_tuning_metrics['forest_5_ha'].query('n_features==2').feature_names[4].split(', ')[1:]
forest_fit_lasso[outcome] = fit_logistic_regression(forest_incidents, "forest", outcome, model_features=features, print_diagnostics=True)

# COMMAND ----------

plot_feature_distributions(forest_incidents, forest_fit_lasso[outcome][0], features)

# COMMAND ----------

fig, ax = plt.subplots(1,1)
for feature in features:
    sns.kdeplot(data=(forest_fit_lasso[outcome][0].params[features[0]]*forest_incidents[[features[0]]]).join(forest_incidents[[outcome]]), x=feature, label=feature, hue=outcome, common_norm=False)
    ax.legend()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Variable selection by Adaptive Lasso and BIC

# COMMAND ----------

grass_incidents_train_variable_selection_filepath = MODEL_INPUT.parent / f'grass_incidents_train_variable_selection_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv'
grass_incidents.query('is_train_data==1').to_csv(grass_incidents_train_variable_selection_filepath)
print(grass_incidents_train_variable_selection_filepath)

# COMMAND ----------

forest_incidents_train_variable_selection_filepath = MODEL_INPUT.parent / f'forest_incidents_train_variable_selection_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv'
forest_incidents.query('is_train_data==1').to_csv(forest_incidents_train_variable_selection_filepath)
print(forest_incidents_train_variable_selection_filepath)

# COMMAND ----------

model_features

# COMMAND ----------

# MAGIC %r
# MAGIC install.packages('glmnet')

# COMMAND ----------

# MAGIC %r
# MAGIC library(tidyverse)
# MAGIC library(glmnet)
# MAGIC # library(glmnetUtils)
# MAGIC GRASS_FILEPATH = '/dbfs/mnt/raw/suppression/incidents_clean/release_version_2.3/grass_incidents_train_variable_selection_20240731001124.csv'
# MAGIC FOREST_FILEPATH = '/dbfs/mnt/raw/suppression/incidents_clean/release_version_2.3/forest_incidents_train_variable_selection_20240731001130.csv'

# COMMAND ----------

# MAGIC %r
# MAGIC model_features = c('T_SFC',  'T_SFC_ishistorical',  'RH_SFC',  'DF_SFC',  'WindMagKmh_SFC',  'KBDI',  'elevation_m',  'ruggedness_average_3km',  'building_density_20km',  'building_density_3km',  'road_density_km_in_3km',  'road_distance_m',  'grass_density_3km',  'forest_density_3km',  'shrub_density_3km',  'noveg_density_3km',  'distance_to_interface',  'soil_moisture',  'Curing')
# MAGIC outcomes = c('uncontrolled_within_2_hrs', 'uncontrolled_within_4_hrs', 'uncontrolled_within_5_ha', 'uncontrolled_within_100_ha')
# MAGIC forest_features = model_features[model_features != 'forest_density_3km']
# MAGIC grass_features = model_features[model_features != 'grass_density_3km']
# MAGIC grass_incidents_df = read_csv(GRASS_FILEPATH)
# MAGIC forest_incidents_df = read_csv(FOREST_FILEPATH)

# COMMAND ----------

# MAGIC %r
# MAGIC fit_adaptive_lasso = function(X_df, y_vec){
# MAGIC   X_df = scale(X_df)
# MAGIC   mod_glm = glm(y_vec ~ X_df, family=binomial(link='logit'))
# MAGIC   alasso_weights = 1/abs(mod_glm$coefficients[-1])
# MAGIC   mod_al = glmnet(scale(X_df),
# MAGIC                         y_vec,
# MAGIC                         family=binomial(link='logit'),
# MAGIC                         penalty.factor=alasso_weights,
# MAGIC                         standardize=FALSE
# MAGIC                   )
# MAGIC   bic = deviance(mod_al) + log(nrow(X_df))*mod_al$df
# MAGIC   plot(bic)
# MAGIC   best_lambda = mod_al$lambda[which.min(bic)]
# MAGIC   mod_al_best = glmnet(X_df,
# MAGIC                        y_vec,
# MAGIC                        family=binomial(link='logit'),
# MAGIC                        penalty.factor=alasso_weights,
# MAGIC                        standardize=FALSE,
# MAGIC                        lambda = best_lambda)
# MAGIC   coeff_best = mod_al_best$beta
# MAGIC   selected_vars = coeff_best@Dimnames[[1]][coeff_best@i + 1]
# MAGIC   # output so I can easily copy paste into Python
# MAGIC   return(paste0("['", paste(selected_vars, collapse="', '"), "']"))
# MAGIC }

# COMMAND ----------

# DBTITLE 1,Grass, 2 hrs
# MAGIC %r
# MAGIC fit_adaptive_lasso(grass_incidents_df[grass_features], grass_incidents_df$uncontrolled_within_2_hrs)

# COMMAND ----------

grass_adaptive_lasso_fit = {}
outcome = 'uncontrolled_within_2_hrs'
features = ['T_SFC', 'T_SFC_ishistorical', 'RH_SFC', 'WindMagKmh_SFC', 'KBDI', 'ruggedness_average_3km', 'building_density_3km', 'road_density_km_in_3km', 'road_distance_m', 'forest_density_3km', 'shrub_density_3km', 'soil_moisture', 'Curing'] # copy pasted from R output
grass_adaptive_lasso_fit[outcome] = fit_logistic_regression(grass_incidents, outcome, model_features=features, print_diagnostics=True, print_confusion_matrix=True)

# COMMAND ----------

# DBTITLE 1,Grass, 100 ha
# MAGIC %r
# MAGIC fit_adaptive_lasso(grass_incidents_df[grass_features], grass_incidents_df$uncontrolled_within_100_ha)

# COMMAND ----------

outcome = 'uncontrolled_within_100_ha'
features = ['T_SFC', 'DF_SFC', 'WindMagKmh_SFC', 'ruggedness_average_3km', 'building_density_20km', 'road_density_km_in_3km', 'road_distance_m', 'noveg_density_3km', 'distance_to_interface', 'Curing'] # copy pasted from R output
grass_adaptive_lasso_fit[outcome] = fit_logistic_regression(grass_incidents, outcome, model_features=features, print_diagnostics=True, print_confusion_matrix=True)

# COMMAND ----------

# DBTITLE 1,Forest, 4 hrs
# MAGIC %r
# MAGIC fit_adaptive_lasso(forest_incidents_df[forest_features], forest_incidents_df$uncontrolled_within_4_hrs)

# COMMAND ----------

forest_adaptive_lasso_fit = {}
outcome = 'uncontrolled_within_4_hrs'
features = ['T_SFC', 'KBDI', 'elevation_m', 'road_density_km_in_3km', 'road_distance_m', 'grass_density_3km', 'shrub_density_3km', 'noveg_density_3km', 'distance_to_interface', 'Curing'] # copy pasted from R output
forest_adaptive_lasso_fit[outcome] = fit_logistic_regression(forest_incidents, outcome, model_features=features, print_diagnostics=True, print_confusion_matrix=True)

# COMMAND ----------

# DBTITLE 1,Forest, 5 ha
# MAGIC %r
# MAGIC fit_adaptive_lasso(forest_incidents_df[forest_features], forest_incidents_df$uncontrolled_within_5_ha)

# COMMAND ----------

outcome = 'uncontrolled_within_5_ha'
features = ['RH_SFC', 'DF_SFC', 'WindMagKmh_SFC', 'elevation_m', 'building_density_20km', 'road_density_km_in_3km', 'road_distance_m', 'grass_density_3km'] # copy pasted from R output
forest_adaptive_lasso_fit[outcome] = fit_logistic_regression(forest_incidents, outcome, model_features=features, print_diagnostics=True, print_confusion_matrix=True)
