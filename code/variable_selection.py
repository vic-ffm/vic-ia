# Databricks notebook source
# MAGIC %md
# MAGIC # Variable selection for the first attack model

# COMMAND ----------

# Refresh input files when they change
%load_ext autoreload
%autoreload 2

# COMMAND ----------

# Import Python packages
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
mlflow.sklearn.autolog(disable=True)
mlflow.statsmodels.autolog(disable=True)

# Import setup file
from base_include import *
from get_model_data import *
from modelling_functions import *

# COMMAND ----------

# MAGIC %md
# MAGIC ## Variable selection by Adaptive Lasso and BIC

# COMMAND ----------

# Export training data to csv so it can be read in by R
grass_incidents.query('is_train_data==1').to_csv(GRASS_VARIABLE_SELECTION_INPUT_CSV)
forest_incidents.query('is_train_data==1').to_csv(FOREST_VARIABLE_SELECTION_INPUT_CSV)


# COMMAND ----------

# print model features to paste into R code
model_features

# COMMAND ----------

# MAGIC %r
# MAGIC install.packages('glmnet')

# COMMAND ----------

# MAGIC %r
# MAGIC library(tidyverse)
# MAGIC library(glmnet)
# MAGIC setwd("..") # set file path to be the main repo folder
# MAGIC DATA_DIRECTORY = file.path('data', 'processed')
# MAGIC GRASS_FILEPATH = file.path(DATA_DIRECTORY, 'incidents_train_variable_selection_grass.csv')
# MAGIC FOREST_FILEPATH = file.path(DATA_DIRECTORY, 'incidents_train_variable_selection_forest.csv')

# COMMAND ----------

# MAGIC %r
# MAGIC # model features pasted from Python code
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

# MAGIC %md
# MAGIC Fit the model again with only the selected predictors/input variables

# COMMAND ----------

grass_adaptive_lasso_fit = {}
outcome = 'uncontrolled_within_2_hrs'
features = ['T_SFC', 'T_SFC_ishistorical', 'RH_SFC', 'WindMagKmh_SFC', 'KBDI', 'ruggedness_average_3km', 'building_density_3km', 'road_density_km_in_3km', 'road_distance_m', 'forest_density_3km', 'shrub_density_3km', 'soil_moisture', 'Curing'] # copy pasted from R output
grass_adaptive_lasso_fit[outcome] = fit_logistic_regression(grass_incidents, outcome, model_features=features)
get_model_diagnostics(rslt = grass_adaptive_lasso_fit[outcome][0], 
                      incidents_train = grass_adaptive_lasso_fit[outcome][1], 
                      incidents_test = grass_adaptive_lasso_fit[outcome][2],
                      outcome = outcome,
                      print_diagnostics=True, print_confusion_matrix=True)

# COMMAND ----------

# DBTITLE 1,Grass, 100 ha
# MAGIC %r
# MAGIC fit_adaptive_lasso(grass_incidents_df[grass_features], grass_incidents_df$uncontrolled_within_100_ha)

# COMMAND ----------

outcome = 'uncontrolled_within_100_ha'
features = ['T_SFC', 'DF_SFC', 'WindMagKmh_SFC', 'ruggedness_average_3km', 'building_density_20km', 'road_density_km_in_3km', 'road_distance_m', 'noveg_density_3km', 'distance_to_interface', 'Curing'] # copy pasted from R output
grass_adaptive_lasso_fit[outcome] = fit_logistic_regression(grass_incidents, outcome, model_features=features)
get_model_diagnostics(rslt = grass_adaptive_lasso_fit[outcome][0], 
                      incidents_train = grass_adaptive_lasso_fit[outcome][1], 
                      incidents_test = grass_adaptive_lasso_fit[outcome][2],
                      outcome = outcome,
                      print_diagnostics=True, print_confusion_matrix=True)

# COMMAND ----------

# DBTITLE 1,Forest, 4 hrs
# MAGIC %r
# MAGIC fit_adaptive_lasso(forest_incidents_df[forest_features], forest_incidents_df$uncontrolled_within_4_hrs)

# COMMAND ----------

forest_adaptive_lasso_fit = {}
outcome = 'uncontrolled_within_4_hrs'
features = ['T_SFC', 'KBDI', 'elevation_m', 'road_density_km_in_3km', 'road_distance_m', 'grass_density_3km', 'shrub_density_3km', 'noveg_density_3km', 'distance_to_interface', 'Curing'] # copy pasted from R output
forest_adaptive_lasso_fit[outcome] = fit_logistic_regression(forest_incidents, outcome, model_features=features)
get_model_diagnostics(rslt = forest_adaptive_lasso_fit[outcome][0], 
                      incidents_train = forest_adaptive_lasso_fit[outcome][1], 
                      incidents_test = forest_adaptive_lasso_fit[outcome][2],
                      outcome = outcome,
                      print_diagnostics=True, print_confusion_matrix=True)

# COMMAND ----------

# DBTITLE 1,Forest, 5 ha
# MAGIC %r
# MAGIC fit_adaptive_lasso(forest_incidents_df[forest_features], forest_incidents_df$uncontrolled_within_5_ha)

# COMMAND ----------

outcome = 'uncontrolled_within_5_ha'
features = ['RH_SFC', 'DF_SFC', 'WindMagKmh_SFC', 'elevation_m', 'building_density_20km', 'road_density_km_in_3km', 'road_distance_m', 'grass_density_3km'] # copy pasted from R output
forest_adaptive_lasso_fit[outcome] = fit_logistic_regression(forest_incidents, outcome, model_features=features)
get_model_diagnostics(rslt = forest_adaptive_lasso_fit[outcome][0], 
                      incidents_train = forest_adaptive_lasso_fit[outcome][1], 
                      incidents_test = forest_adaptive_lasso_fit[outcome][2],
                      outcome = outcome,
                      print_diagnostics=True, print_confusion_matrix=True)

# COMMAND ----------

# MAGIC %md
# MAGIC The variables selected for each model in this code have been manually entered into the `logistic_regression` notebook.
