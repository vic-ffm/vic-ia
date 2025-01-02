# Databricks notebook source
# MAGIC %md
# MAGIC # GAMs to estimate probability of first attack failure

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load packages and data

# COMMAND ----------

# MAGIC %pip install pygam

# COMMAND ----------

# Import Python packages
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.metrics
# from sklearn.model_selection import cross_val_score, train_test_split
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.gam.api import GLMGam, BSplines

from pygam import LogisticGAM, s, f

# Import setup file
from includes.base_include import *
from get_model_data import *
from modelling_functions import *

# Refresh input files when they change
%load_ext autoreload
%autoreload 2

# COMMAND ----------

print(len(grass_incidents), len(forest_incidents), len(shrub_incidents), len(noveg_incidents))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modelling

# COMMAND ----------

# MAGIC %md
# MAGIC ### statsmodels package

# COMMAND ----------

def fit_gam(incidents_data, primary_fuel_type, outcome, model_features=model_features, interactions=None, recall_for_threshold=None, print_summary=False, print_diagnostics=False, print_confusion_matrix=False, normalise_confusion=False, transform=None):
    outcome_prob = outcome + "_p"

    incidents_train = incidents_data.sample(frac=0.8, random_state=20240124)
    incidents_test = incidents_data.drop(incidents_train.index, axis=0)

    model_features = [x for x in model_features if not x.startswith(primary_fuel_type + '_density')]

    # Create splines for all model features
    x_spline = incidents_train[model_features]
    deg = len(model_features)
    bs = BSplines(x_spline, df=[10+deg]*deg, degree=[3]*deg) # divide region into 10 splines, cubic splines

    print(incidents_data.columns)
    # TODO: How to fit splines to all variables without having T_SFC as linear term? How to decide which variables need splines? Maybe lr residuals?
    rslt = GLMGam.from_formula(formula=F"{outcome} ~ T_SFC", data=incidents_train, smoother=bs, include_intercept=True, family=sm.families.Binomial()).fit()
    if print_summary:
        print(outcome)
        print(rslt.summary())

    # incidents_test = incidents_test.assign(**{outcome_prob: rslt.predict(exog=incidents_test)})
    # incidents_train = incidents_train.assign(**{outcome_prob: rslt.predict(exog=incidents_train)})

    # return (rslt, incidents_test)
    # print(rslt.summary())
    for ii in range(len(model_features)):
        rslt.plot_partial(ii, cpr=True)


fit_gam(grass_incidents, "grass", 'uncontrolled_within_5_ha', print_diagnostics=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### pygam package

# COMMAND ----------

grass_features

# COMMAND ----------

X_train = grass_incidents.query('is_train_data==1')[grass_features].to_numpy()
y_train = grass_incidents.query('is_train_data==1')['uncontrolled_within_2_hrs'].to_numpy()
X_test = grass_incidents.query('is_train_data==0')[grass_features].to_numpy()
y_test = grass_incidents.query('is_train_data==0')['uncontrolled_within_2_hrs'].to_numpy()

gam = LogisticGAM(s(0) + f(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) + s(10) + s(11) + s(12) + s(13) + s(14) + s(15) + s(16) + s(17)).gridsearch(X_train, y_train)

# COMMAND ----------

gam.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC In R package `mgcv` you should check the significance codes and adjust the degree of the polynomial accordingly. I can't find how to do this in pygam.

# COMMAND ----------

fig, axs = plt.subplots(1, len(grass_features), figsize=(25,5))
titles = grass_features
for ii, ax in enumerate(axs):
    XX = gam.generate_X_grid(term=ii)
    pdep, confi = gam.partial_dependence(term=ii, width=.95)

    ax.plot(XX[:, ii], pdep)
    ax.plot(XX[:, ii], confi, c='r', ls='--')
    ax.set_title(titles[ii]);

# COMMAND ----------

y_proba = gam.predict_proba(X_test)

grass_incidents_test = (grass_incidents
                        .query('is_train_data==0')
                        .assign(uncontrolled_within_2_hrs_p = y_proba))

fig, axs = plt.subplots(1, 4, figsize=(20,6))
fig.suptitle('Grass: uncontrolled_within_2_hrs')
sns.histplot(data=grass_incidents_test, x='uncontrolled_within_2_hrs_p', hue='uncontrolled_within_2_hrs', ax=axs[0])

sns.kdeplot(data=grass_incidents_test, x='uncontrolled_within_2_hrs_p', hue='uncontrolled_within_2_hrs', common_norm=False, ax=axs[1])

auc = roc_auc_score(grass_incidents_test['uncontrolled_within_2_hrs'], grass_incidents_test['uncontrolled_within_2_hrs_p'])

fpr, tpr, thresholds = sklearn.metrics.roc_curve(grass_incidents_test['uncontrolled_within_2_hrs'], grass_incidents_test['uncontrolled_within_2_hrs_p'])
axs[2].plot(fpr,tpr,label=f"AUC = {auc:.4}")
axs[2].set(xlabel="False positive rate", ylabel="True positive rate")
axs[2].legend(loc=4,fontsize=18)

ap = sklearn.metrics.average_precision_score(grass_incidents_test['uncontrolled_within_2_hrs'], grass_incidents_test['uncontrolled_within_2_hrs_p'])

precision, recall, thresholds = sklearn.metrics.precision_recall_curve(grass_incidents_test['uncontrolled_within_2_hrs'], grass_incidents_test['uncontrolled_within_2_hrs_p'])
axs[3].plot(recall, precision, label=f"AP = {ap:.4}")
axs[3].set(xlabel="Recall", ylabel="Precision")
axs[3].legend(loc=8,fontsize=18)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### mgcv in R

# COMMAND ----------

# MAGIC %r
# MAGIC install.packages("PRROC")
# MAGIC install.packages("yardstick")

# COMMAND ----------

# MODEL_INPUT_CSV = INCIDENTS_FOLDER / MODEL_INPUT_VERSION / 'incidents_modelling.csv'
# incidents.to_csv(MODEL_INPUT_CSV)
GRASS_MODEL_INPUT_CSV = INCIDENTS_FOLDER / MODEL_INPUT_VERSION / 'incidents_modelling_grass.csv'
grass_incidents.to_csv(GRASS_MODEL_INPUT_CSV)
FOREST_MODEL_INPUT_CSV = INCIDENTS_FOLDER / MODEL_INPUT_VERSION / 'incidents_modelling_forest.csv'
forest_incidents.to_csv(FOREST_MODEL_INPUT_CSV)

# COMMAND ----------

# MAGIC %r
# MAGIC library(tidyverse)
# MAGIC library(mgcv)
# MAGIC library(pROC)
# MAGIC library(PRROC)
# MAGIC library(yardstick)
# MAGIC # library(DHARMa) # can't get this to work to plot residuals nicely, probably because need an updated version of tidyverse

# COMMAND ----------

# MAGIC %r
# MAGIC DATA_DIRECTORY = file.path('/dbfs/mnt/raw/suppression/incidents_clean/release_version_2.5.8')
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

# MAGIC %md
# MAGIC #### Grass, 2 hrs

# COMMAND ----------

# MAGIC %r
# MAGIC colnames(grass_incidents_df)

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
# MAGIC  
# MAGIC # # create roc curve
# MAGIC # roc_object <- roc(grass_incidents_test_df$uncontrolled_within_2_hrs, grass_incidents_test_df$uncontrolled_within_2_hrs_p)
# MAGIC
# MAGIC # # calculate area under curve
# MAGIC # auc(roc_object)

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
# MAGIC - The AUC is slightly worse but pretty similar to the `pyGAM` fit.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Grass, 100 ha

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
# MAGIC #### Forest, 4 hrs

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
# MAGIC #### Forest, 5 ha

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

# MAGIC %r
# MAGIC colnames(grass_incidents_test_df)

# COMMAND ----------

# MAGIC %r
# MAGIC str(grass_incidents_test_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Write to csv

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

# MAGIC %r
# MAGIC citation("mgcv")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Diagnostic plots in Python of mgcv fits

# COMMAND ----------



# COMMAND ----------

FOREST_GAM_RESULTS = INCIDENTS_FOLDER / MODEL_INPUT_VERSION / 'incidents_forest_test_gam_results.csv'
GRASS_GAM_RESULTS = INCIDENTS_FOLDER / MODEL_INPUT_VERSION / 'incidents_grass_test_gam_results.csv'

forest_incidents = pd.read_csv(FOREST_GAM_RESULTS)
grass_incidents = pd.read_csv(GRASS_GAM_RESULTS)

# COMMAND ----------

get_model_diagnostics({}, grass_incidents.query('is_train_data==1'), grass_incidents.query('is_train_data==0'), 'uncontrolled_within_2_hrs', print_diagnostics=True)

# COMMAND ----------

outcome = 'uncontrolled_within_2_hrs'
get_model_diagnostics({}, grass_incidents.query('is_train_data==1'), grass_incidents.query('is_train_data==0'), outcome,
                      print_appendix_diagnostics=True, save_to=DIAGNOSTICS / (outcome + '_gam_diagnostics.eps'))

# COMMAND ----------

get_model_diagnostics({}, grass_incidents.query('is_train_data==1'), grass_incidents.query('is_train_data==0'), 'uncontrolled_within_100_ha', print_diagnostics=True)

# COMMAND ----------

outcome = 'uncontrolled_within_100_ha'
get_model_diagnostics({}, grass_incidents.query('is_train_data==1'), grass_incidents.query('is_train_data==0'), outcome,
                      print_appendix_diagnostics=True, save_to=DIAGNOSTICS / (outcome + '_gam_diagnostics.eps'))

# COMMAND ----------

get_model_diagnostics({}, forest_incidents.query('is_train_data==1'), forest_incidents.query('is_train_data==0'), 'uncontrolled_within_4_hrs', print_diagnostics=True)

# COMMAND ----------

outcome = 'uncontrolled_within_4_hrs'
get_model_diagnostics({}, forest_incidents.query('is_train_data==1'), forest_incidents.query('is_train_data==0'), outcome,
                      print_appendix_diagnostics=True, save_to=DIAGNOSTICS / (outcome + '_gam_diagnostics.eps'))

# COMMAND ----------

get_model_diagnostics({}, forest_incidents.query('is_train_data==1'), forest_incidents.query('is_train_data==0'), 'uncontrolled_within_5_ha', print_diagnostics=True)

# COMMAND ----------

outcome = 'uncontrolled_within_5_ha'
get_model_diagnostics({}, forest_incidents.query('is_train_data==1'), forest_incidents.query('is_train_data==0'), outcome,
                      print_appendix_diagnostics=True, save_to=DIAGNOSTICS / (outcome + '_gam_diagnostics.eps'))
