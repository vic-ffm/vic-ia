# Databricks notebook source
# MAGIC %md
# MAGIC # GAMs to estimate probability of first attack failure

# COMMAND ----------

# MAGIC %md
# MAGIC The model fitting is done in R using the `mgcv` package. This notebook contains both R and Python code.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load packages and data

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

# Import setup file
from base_include import *
from get_model_data import *
from modelling_functions import *

# Save the grass and forest model input data to csv so it can be read in by R.
grass_incidents.to_csv(GRASS_MODEL_INPUT_CSV)
forest_incidents.to_csv(FOREST_MODEL_INPUT_CSV)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modelling using mgcv in R

# COMMAND ----------

# MAGIC %r
# MAGIC install.packages("PRROC")
# MAGIC install.packages("yardstick")

# COMMAND ----------

# MAGIC %r
# MAGIC library(tidyverse)
# MAGIC library(mgcv)
# MAGIC library(pROC)
# MAGIC library(PRROC)
# MAGIC library(yardstick)

# COMMAND ----------

# MAGIC %r
# MAGIC setwd("..") # set file path to be the main repo folder
# MAGIC DATA_DIRECTORY = file.path('data', 'processed')
# MAGIC FOREST_DATA = file.path(DATA_DIRECTORY, 'incidents_modelling_forest.csv')
# MAGIC GRASS_DATA = file.path(DATA_DIRECTORY, 'incidents_modelling_grass.csv')
# MAGIC
# MAGIC FOREST_GAM_RESULTS = file.path(DATA_DIRECTORY, 'incidents_forest_test_gam_results.csv')
# MAGIC GRASS_GAM_RESULTS = file.path(DATA_DIRECTORY, 'incidents_grass_test_gam_results.csv')
# MAGIC
# MAGIC grass_incidents_df = read_csv(GRASS_DATA, show_col_types = FALSE)
# MAGIC forest_incidents_df = read_csv(FOREST_DATA, show_col_types = FALSE)
# MAGIC
# MAGIC grass_incidents_train_df = grass_incidents_df %>% filter(is_train_data==1)
# MAGIC forest_incidents_train_df = forest_incidents_df %>% filter(is_train_data==1)
# MAGIC grass_incidents_test_df = grass_incidents_df %>% filter(is_train_data==0)
# MAGIC forest_incidents_test_df = forest_incidents_df %>% filter(is_train_data==0)

# COMMAND ----------

# MAGIC %r
# MAGIC # Function to calculate the average precision
# MAGIC calculate_average_precision = function(df, truth, prediction){
# MAGIC   truth = enquo(truth)
# MAGIC   prediction = enquo(prediction)
# MAGIC   
# MAGIC   if(class(df %>% pull(!!truth)) != "factor"){
# MAGIC     df = df %>%
# MAGIC       mutate(truth_factor = as.factor(!!truth))
# MAGIC   } else{
# MAGIC     df = df %>%
# MAGIC       mutate(truth_factor = !!truth)
# MAGIC   }
# MAGIC   
# MAGIC   curve = df %>%
# MAGIC   pr_curve(truth_factor, !!prediction, event_level = "second")
# MAGIC
# MAGIC   ap = curve %>%
# MAGIC     filter_all(all_vars(!is.infinite(.))) %>%
# MAGIC     mutate(
# MAGIC       pr_summand = (recall - lag(recall))*precision,
# MAGIC       pr_summand = replace_na(pr_summand, 0)
# MAGIC     ) %>%
# MAGIC     pull(pr_summand) %>%
# MAGIC     sum()
# MAGIC   
# MAGIC   return(ap)
# MAGIC }

# COMMAND ----------

# MAGIC %md
# MAGIC ### Grass, 2 hrs

# COMMAND ----------

# MAGIC %r
# MAGIC mod01 = gam(uncontrolled_within_2_hrs ~ s(T_SFC) + T_SFC_ishistorical + s(RH_SFC) + s(DF_SFC) + s(WindMagKmh_SFC) + s(KBDI) + s(elevation_m) + s(ruggedness_average_3km) + s(building_density_3km) + 
# MAGIC             s(building_density_20km) + s(road_density_km_in_3km) + s(forest_density_3km) + s(shrub_density_3km) + s(noveg_density_3km) + s(distance_to_interface) + s(soil_moisture) + s(Curing),
# MAGIC             data = grass_incidents_train_df,
# MAGIC             family = binomial,
# MAGIC             method = "REML")

# COMMAND ----------

# MAGIC %r  
# MAGIC plot(mod01, pages=1, trans=plogis, shift = coef(mod01)[1], seWithMean=TRUE)

# COMMAND ----------

# MAGIC %r
# MAGIC gam.check(mod01)

# COMMAND ----------

# MAGIC %r
# MAGIC grass_incidents_test_df = grass_incidents_test_df %>%
# MAGIC     mutate(uncontrolled_within_2_hrs_p = as.numeric(predict(mod01, grass_incidents_test_df, type="response")),
# MAGIC            uncontrolled_within_2_hrs = as.factor(uncontrolled_within_2_hrs)
# MAGIC     )

# COMMAND ----------

# MAGIC %r
# MAGIC ggplot(grass_incidents_test_df, aes(x=uncontrolled_within_2_hrs_p, colour=uncontrolled_within_2_hrs)) +
# MAGIC   geom_density() + 
# MAGIC   scale_color_manual(values=c("#56B4E9", "#E69F00"))

# COMMAND ----------

# MAGIC %r
# MAGIC auc_mod01 = grass_incidents_test_df %>%
# MAGIC   yardstick::roc_auc(uncontrolled_within_2_hrs, uncontrolled_within_2_hrs_p, event_level = "second")
# MAGIC
# MAGIC grass_incidents_test_df %>%
# MAGIC   roc_curve(uncontrolled_within_2_hrs, uncontrolled_within_2_hrs_p, event_level = "second") %>%
# MAGIC   autoplot() +
# MAGIC   annotate("text", x = 0.9, y = 0.1, label = paste("AUC:", round(auc_mod01['.estimate'], 4))) +
# MAGIC   ggtitle("Model 01: grass, 2 hrs") 

# COMMAND ----------

# MAGIC %r
# MAGIC ap_mod01 = calculate_average_precision(grass_incidents_test_df, uncontrolled_within_2_hrs, uncontrolled_within_2_hrs_p)
# MAGIC
# MAGIC grass_incidents_test_df %>%
# MAGIC   pr_curve(uncontrolled_within_2_hrs, uncontrolled_within_2_hrs_p, event_level = "second") %>%
# MAGIC   autoplot() +
# MAGIC   annotate("text", x = 0.9, y = 0.1, label = paste("AP:", round(ap_mod01, 4))) +
# MAGIC   ggtitle("Model 01: grass, 2 hrs")

# COMMAND ----------

# MAGIC %md
# MAGIC - The `gam.check` p-values aren't that small and the fit degrees are much less than `k'`, so not worth adding higher degree polynomials

# COMMAND ----------

# MAGIC %md
# MAGIC ### Grass, 100 ha

# COMMAND ----------

# MAGIC %r
# MAGIC mod02 = gam(uncontrolled_within_100_ha ~ s(T_SFC) + T_SFC_ishistorical + s(RH_SFC) + s(DF_SFC) + s(WindMagKmh_SFC) + s(KBDI) + s(elevation_m) + s(ruggedness_average_3km) + s(building_density_3km) + 
# MAGIC             s(building_density_20km) + s(road_density_km_in_3km) + s(forest_density_3km) + s(shrub_density_3km) + s(noveg_density_3km) + s(distance_to_interface) + s(soil_moisture) + s(Curing),
# MAGIC             data = grass_incidents_train_df,
# MAGIC             family = binomial,
# MAGIC             method = "REML")

# COMMAND ----------

# MAGIC %r
# MAGIC plot(mod02, pages=1, trans=plogis, shift = coef(mod02)[1], seWithMean=TRUE)

# COMMAND ----------

# MAGIC %r
# MAGIC gam.check(mod02)

# COMMAND ----------

# MAGIC %r
# MAGIC grass_incidents_test_df = grass_incidents_test_df %>%
# MAGIC     mutate(uncontrolled_within_100_ha_p = as.numeric(predict(mod02, grass_incidents_test_df, 
# MAGIC                       type="response")),
# MAGIC            uncontrolled_within_100_ha = as.factor(uncontrolled_within_100_ha)
# MAGIC     )
# MAGIC
# MAGIC ggplot(grass_incidents_test_df, aes(x=uncontrolled_within_100_ha_p, colour=uncontrolled_within_100_ha)) +
# MAGIC   geom_density() + 
# MAGIC   scale_color_manual(values=c("#56B4E9", "#E69F00"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Forest, 4 hrs

# COMMAND ----------

# MAGIC %r
# MAGIC mod03 = gam(uncontrolled_within_4_hrs ~ s(T_SFC) + T_SFC_ishistorical + s(RH_SFC) + s(DF_SFC) + s(WindMagKmh_SFC) + s(KBDI) + s(elevation_m) + s(ruggedness_average_3km) + s(building_density_3km) + 
# MAGIC             s(building_density_20km) + s(road_density_km_in_3km) + s(shrub_density_3km) + s(shrub_density_3km) + s(noveg_density_3km) + s(distance_to_interface) + s(soil_moisture) + s(Curing),
# MAGIC             data = forest_incidents_train_df,
# MAGIC             family = binomial,
# MAGIC             method = "REML")
# MAGIC
# MAGIC gam.check(mod03)

# COMMAND ----------

# MAGIC %r
# MAGIC plot(mod03, pages=1, trans=plogis, shift = coef(mod03)[1], seWithMean=TRUE)

# COMMAND ----------

# MAGIC %r
# MAGIC forest_incidents_test_df = forest_incidents_test_df %>%
# MAGIC     mutate(uncontrolled_within_4_hrs_p = as.numeric(predict(mod03, forest_incidents_test_df, 
# MAGIC                       type="response")),
# MAGIC            uncontrolled_within_4_hrs = as.factor(uncontrolled_within_4_hrs)
# MAGIC     )
# MAGIC
# MAGIC ggplot(forest_incidents_test_df, aes(x=uncontrolled_within_4_hrs_p, colour=uncontrolled_within_4_hrs)) +
# MAGIC   geom_density() + 
# MAGIC   scale_color_manual(values=c("#56B4E9", "#E69F00"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Forest, 5 ha

# COMMAND ----------

# MAGIC %r
# MAGIC mod04 = gam(uncontrolled_within_5_ha ~ s(T_SFC) + T_SFC_ishistorical + s(RH_SFC) + s(DF_SFC) + s(WindMagKmh_SFC) + s(KBDI) + s(elevation_m) + s(ruggedness_average_3km) + s(building_density_3km) + 
# MAGIC             s(building_density_20km) + s(road_density_km_in_3km) + s(shrub_density_3km) + s(shrub_density_3km) + s(noveg_density_3km) + s(distance_to_interface) + s(soil_moisture) + s(Curing),
# MAGIC             data = forest_incidents_train_df,
# MAGIC             family = binomial,
# MAGIC             method = "REML")
# MAGIC
# MAGIC gam.check(mod04)

# COMMAND ----------

# MAGIC %r
# MAGIC forest_incidents_test_df = forest_incidents_test_df %>%
# MAGIC     mutate(uncontrolled_within_5_ha_p = as.numeric(predict(mod04, forest_incidents_test_df, 
# MAGIC                       type="response")),
# MAGIC            uncontrolled_within_5_ha = as.factor(uncontrolled_within_5_ha)
# MAGIC     )
# MAGIC
# MAGIC ggplot(forest_incidents_test_df, aes(x=uncontrolled_within_5_ha_p, colour=uncontrolled_within_5_ha)) +
# MAGIC   geom_density() + 
# MAGIC   scale_color_manual(values=c("#56B4E9", "#E69F00"))

# COMMAND ----------

# MAGIC %r
# MAGIC plot(mod04, pages=1, trans=plogis, shift = coef(mod04)[1], seWithMean=TRUE)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write to csv

# COMMAND ----------

# MAGIC %r
# MAGIC grass_incidents_df = grass_incidents_df %>%
# MAGIC   mutate(uncontrolled_within_2_hrs_p = as.numeric(predict(mod01, grass_incidents_df, 
# MAGIC                       type="response")),
# MAGIC          uncontrolled_within_100_ha_p = as.numeric(predict(mod02, grass_incidents_df, 
# MAGIC                       type="response"))
# MAGIC   )
# MAGIC
# MAGIC forest_incidents_df = forest_incidents_df %>%
# MAGIC   mutate(uncontrolled_within_4_hrs_p = as.numeric(predict(mod03, forest_incidents_df, 
# MAGIC                       type="response")),
# MAGIC          uncontrolled_within_5_ha_p = as.numeric(predict(mod04, forest_incidents_df, 
# MAGIC                       type="response"))
# MAGIC   )

# COMMAND ----------

# MAGIC %r
# MAGIC write_csv(grass_incidents_df, GRASS_GAM_RESULTS)
# MAGIC write_csv(forest_incidents_df, FOREST_GAM_RESULTS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Diagnostic plots in Python of mgcv fits
# MAGIC Redo the plots in Python so the style matches those of the other model plots

# COMMAND ----------

forest_incidents = pd.read_csv(FOREST_GAM_RESULTS)
grass_incidents = pd.read_csv(GRASS_GAM_RESULTS)

# COMMAND ----------

get_model_diagnostics({}, grass_incidents.query('is_train_data==1'), grass_incidents.query('is_train_data==0'), 'uncontrolled_within_2_hrs', print_diagnostics=True)

# COMMAND ----------

outcome = 'uncontrolled_within_2_hrs'
get_model_diagnostics({}, grass_incidents.query('is_train_data==1'), grass_incidents.query('is_train_data==0'), outcome,
                      print_appendix_diagnostics=True)

# COMMAND ----------

get_model_diagnostics({}, grass_incidents.query('is_train_data==1'), grass_incidents.query('is_train_data==0'), 'uncontrolled_within_100_ha', print_diagnostics=True)

# COMMAND ----------

outcome = 'uncontrolled_within_100_ha'
get_model_diagnostics({}, grass_incidents.query('is_train_data==1'), grass_incidents.query('is_train_data==0'), outcome,
                      print_appendix_diagnostics=True)

# COMMAND ----------

get_model_diagnostics({}, forest_incidents.query('is_train_data==1'), forest_incidents.query('is_train_data==0'), 'uncontrolled_within_4_hrs', print_diagnostics=True)

# COMMAND ----------

outcome = 'uncontrolled_within_4_hrs'
get_model_diagnostics({}, forest_incidents.query('is_train_data==1'), forest_incidents.query('is_train_data==0'), outcome,
                      print_appendix_diagnostics=True)

# COMMAND ----------

get_model_diagnostics({}, forest_incidents.query('is_train_data==1'), forest_incidents.query('is_train_data==0'), 'uncontrolled_within_5_ha', print_diagnostics=True)

# COMMAND ----------

outcome = 'uncontrolled_within_5_ha'
get_model_diagnostics({}, forest_incidents.query('is_train_data==1'), forest_incidents.query('is_train_data==0'), outcome,
                      print_appendix_diagnostics=True)
