# Databricks notebook source
# MAGIC %md
# MAGIC # Tree-based methods to classify fires as contained or uncontrolled

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

# COMMAND ----------

# MAGIC %md
# MAGIC ### Grass

# COMMAND ----------

params_grass_2_hrs_f1 = fit_tree_classifier(incidents_data=grass_incidents, outcome='uncontrolled_within_2_hrs', model_features=grass_features, classifier=BRF, fbeta=1, print_diagnostics=True, sampling_strategy="all", replacement=True, bootstrap=False)


# COMMAND ----------

params_grass_100_ha_f1 = fit_tree_classifier(incidents_data=grass_incidents, outcome='uncontrolled_within_100_ha', model_features=grass_features, classifier=BRF, fbeta=1, print_diagnostics=True, sampling_strategy="all", replacement=True, bootstrap=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Forest

# COMMAND ----------

params_forest_4_hrs_f1 = fit_tree_classifier(incidents_data=forest_incidents, outcome='uncontrolled_within_4_hrs', model_features=forest_features, classifier=BRF, fbeta=1, print_diagnostics=True, sampling_strategy="all", replacement=True, bootstrap=False)

# COMMAND ----------

params_forest_5_ha_f1 = fit_tree_classifier(incidents_data=forest_incidents, outcome='uncontrolled_within_5_ha', model_features=forest_features, classifier=BRF, fbeta=1, print_diagnostics=True, sampling_strategy="all", replacement=True, bootstrap=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Repeat (parameters saved for tuned models)

# COMMAND ----------

params_grass_2_hrs_f1 = {'max_depth': 5.002698042302658,
 'min_samples_leaf': 3.04328387916119,
 'min_samples_split': 5.569528016045849,
 'n_estimators': 105.61376095944503};
params_grass_100_ha_f1 = {'max_depth': 5.005404940436849,
 'min_samples_leaf': 2.514067478843515,
 'min_samples_split': 6.41900372256227,
 'n_estimators': 105.34323877995543};
params_forest_4_hrs_f1 =  {'max_depth': 5.789956231828946,
 'min_samples_leaf': 3.230303626768061,
 'min_samples_split': 7.181683547720534,
 'n_estimators': 104.37650530419408};
params_forest_5_ha_f1 = {'max_depth': 5.7575200134254985,
 'min_samples_leaf': 3.094806390199318,
 'min_samples_split': 5.3365247111805605,
 'n_estimators': 221.6950837584438};

# COMMAND ----------

outcome = 'uncontrolled_within_2_hrs'
fit_tree_classifier(incidents_data=grass_incidents, outcome=outcome, model_features=grass_features, classifier=BRF, fbeta=1, params_best=params_grass_2_hrs_f1, print_diagnostics=True, sampling_strategy="all", replacement=True, bootstrap=False)

# COMMAND ----------

outcome = 'uncontrolled_within_100_ha'
fit_tree_classifier(incidents_data=grass_incidents, outcome=outcome, model_features=grass_features, classifier=BRF, fbeta=1, params_best=params_grass_100_ha_f1, print_diagnostics=True, sampling_strategy="all", replacement=True, bootstrap=False)

# COMMAND ----------

outcome = 'uncontrolled_within_4_hrs'
fit_tree_classifier(incidents_data=forest_incidents, outcome=outcome, model_features=forest_features, classifier=BRF, fbeta=1, params_best=params_forest_4_hrs_f1, print_diagnostics=True, sampling_strategy="all", replacement=True, bootstrap=False)

# COMMAND ----------

outcome = 'uncontrolled_within_5_ha'
fit_tree_classifier(incidents_data=forest_incidents, outcome=outcome, model_features=forest_features, classifier=BRF, fbeta=1, params_best=params_forest_5_ha_f1, print_diagnostics=True, sampling_strategy="all", replacement=True, bootstrap=False)
