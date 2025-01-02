# Databricks notebook source
# MAGIC %md
# MAGIC # Tree-based methods to classify fires as contained or uncontrolled

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load packages and data

# COMMAND ----------

# MAGIC %pip install ISLP

# COMMAND ----------

# MAGIC %pip install -U imbalanced-learn

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Import Python packages
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.model_selection as skm 
from ISLP import confusion_table 
from ISLP.models import ModelSpec as MS

from sklearn.tree import (DecisionTreeClassifier as DTC, 
                          plot_tree , 
                          export_text) 
from sklearn.metrics import (accuracy_score, 
                             balanced_accuracy_score,
                             log_loss,
                             roc_curve,
                             roc_auc_score,
                             precision_recall_fscore_support,
                             precision_recall_curve,
                             average_precision_score) 
from sklearn.ensemble import (RandomForestClassifier as RF, 
                              GradientBoostingClassifier as GBR) 

from imblearn.ensemble import (BalancedRandomForestClassifier as BRF)

# hyperparameter tuning
from hyperopt import tpe,hp,Trials
from hyperopt.fmin import fmin
from sklearn.metrics import mean_squared_error,make_scorer

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
# MAGIC ## Modelling: hyperparameter optimisation

# COMMAND ----------

def fit_tree_classifier(incidents_data, outcome, model_features, classifier, fbeta=1, params_best=None, print_diagnostics=False, print_feature_importance=False, random_state=RANDOM_SEED, save_to=None, **args):
    model = MS(incidents_data[model_features], intercept=False)
    D = model.fit_transform(incidents_data[model_features])
    X_train = np.asarray(D[incidents_data.is_train_data == 1])
    X_test = np.asarray(D[incidents_data.is_train_data == 0])
    y_train = np.asarray(incidents_data.query('is_train_data == 1')[outcome])
    y_test = np.asarray(incidents_data.query('is_train_data == 0')[outcome])

    clf = classifier(random_state=random_state, **args).fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    y_proba=clf.predict_proba(X_test)

    # model = MS(incidents_data.columns.drop([primary_fuel_type + "_density_3km"] + ['primary_fuel_type', 'uncontrolled_within_90_mins', 'uncontrolled_within_6_hrs', 'uncontrolled_within_5_ha', 'uncontrolled_within_100_ha']), intercept=False)
    # D = model.fit_transform(incidents_data.drop([primary_fuel_type + "_density_3km"] + ['primary_fuel_type'], axis=1))
    # feature_names = list(D.columns)
    # X = np.asarray(D)

    # Split into train, test, validation: train/test split based on fire season, then split train (80% of original train), validation (20% of original train)
    # (X_train, X_test , y_train , y_test) = skm.train_test_split(X, incidents_data[outcome], test_size=0.2, random_state=random_state)
    (X_train, X_val , y_train , y_val) = skm.train_test_split(X_train, y_train, test_size=0.2, random_state=random_state)

    # Find best classifier (within a specified range of input parameters): best is measured using fbeta-score
    if params_best == None:
        trial=Trials()
        params_best=optimize(trial, classifier, X_train, y_train, X_val, y_val, fbeta, **args)
    
    # Fit the model with the best parameters
    clf_best= classifier(n_estimators=int(round(params_best['n_estimators'])),
                         max_depth=int(round(params_best['max_depth'])),
                         min_samples_leaf=int(round(params_best['min_samples_leaf'])),
                         min_samples_split=int(round(params_best['min_samples_split'])),
                         random_state=random_state,
                         **args).fit(X_train, y_train)

    # Predict on test data
    y_pred=clf_best.predict(X_test)
    y_proba=clf_best.predict_proba(X_test)

    if print_diagnostics:
        outcome_text = outcome.replace('_', ' ')
        print(f'Performance Metrics: {outcome_text}')
        precision, recall, fbetascore, _ = precision_recall_fscore_support(y_true=y_test, y_pred=y_pred, beta=fbeta)
        print(f"Precision {precision[1]:.4f}, Recall {recall[1]:.4f}, F{fbeta} score {fbetascore[1]:.4f}")
        print("------------------")
        print('Confusion table')
        print(confusion_table(predicted_labels=y_pred, true_labels=y_test))
        test_df = pd.DataFrame(data = np.transpose(np.vstack((y_test, y_proba[:, 1]))),
                               columns=['y_test', 'y_proba'])
        
        fig, axs = plt.subplots(1, 3, figsize=(20,6))
        sns.kdeplot(data=test_df, x='y_proba' , hue='y_test', common_norm=False, cut=0, ax=axs[0])
        axs[0].set_xlim([0, 1])
        axs[0].set_xlabel('Probability of being uncontrolled', fontsize=18)
        axs[0].set_ylabel('Density', fontsize=18)
        axs[0].legend(title='Ground truth', labels=['uncontrolled', 'controlled'], fontsize=16, title_fontsize=16)

        auc = roc_auc_score(test_df['y_test'], test_df['y_proba'])
        fpr, tpr, thresholds = roc_curve(test_df['y_test'], test_df['y_proba'])
        axs[1].plot(fpr,tpr,label=f"AUC = {auc:.4}")
        axs[1].set_xlabel("False positive rate", fontsize=18)
        axs[1].set_ylabel("True positive rate", fontsize=18)
        axs[1].legend(loc=4,fontsize=18, handlelength=0, handletextpad=0)

        ap = average_precision_score(test_df['y_test'], test_df['y_proba'])
        precision, recall, thresholds = precision_recall_curve(test_df['y_test'], test_df['y_proba'])
        axs[2].plot(recall, precision, label=f"AP = {ap:.4}")
        axs[2].set_xlabel("Recall", fontsize=18)
        axs[2].set_ylabel("Precision", fontsize=18)
        axs[2].legend(loc=8,fontsize=18, handlelength=0, handletextpad=0)
        if save_to is not None:
            plt.savefig(save_to)
        else:
            plt.show()

    if print_feature_importance:
        print("------------------")
        print('Feature importance')
        print(pd.DataFrame({'importance':clf_best.feature_importances_}, index=model_features).sort_values(by='importance',ascending=False))

    return params_best


def optimize(trial, classifier, X_train, y_train, X_val, y_val, fbeta, seed=2, **args):
    # https://www.kaggle.com/code/virajbagal/eda-xgb-random-forest-parameter-tuning-hyperopt
    params={'n_estimators':hp.uniform('n_estimators',100,300),
           'max_depth':hp.uniform('max_depth',5,20),
           'min_samples_leaf':hp.uniform('min_samples_leaf',1,5),
           'min_samples_split':hp.uniform('min_samples_split',2,10)}
    best=fmin(fn= lambda p: objective(p, classifier, X_train, y_train, X_val, y_val, fbeta, **args),
              space=params,algo=tpe.suggest,trials=trial,max_evals=500,rstate=np.random.default_rng(seed))
    return best


def objective(params, classifier, X_train, y_train, X_val, y_val, fbeta, **args):
    # https://www.kaggle.com/code/virajbagal/eda-xgb-random-forest-parameter-tuning-hyperopt
    est=int(params['n_estimators'])
    md=int(params['max_depth'])
    msl=int(params['min_samples_leaf'])
    mss=int(params['min_samples_split'])
    # model=RF(n_estimators=est,max_depth=md,min_samples_leaf=msl,min_samples_split=mss)
    # model=BRF(n_estimators=est,max_depth=md,min_samples_leaf=msl,min_samples_split=mss, sampling_strategy="all", replacement=True, bootstrap=False)
    model = classifier(n_estimators=est,max_depth=md,min_samples_leaf=msl,min_samples_split=mss, **args)
    model.fit(X_train, y_train)
    pred=model.predict(X_val)
    precision, recall, fbetascore, _ = precision_recall_fscore_support(y_true=y_val, y_pred=pred, beta=fbeta)
    score=fbetascore[0] # minimising 1-fbetascore for outcome
    return score

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

params_forest_5_ha_f1

# COMMAND ----------

params_grass_2_hrs_f1 = {'max_depth': 5.572583717688989,
 'min_samples_leaf': 3.475557919184253,
 'min_samples_split': 4.701436403184555,
 'n_estimators': 125.98709110408812};
params_grass_100_ha_f1 = {'max_depth': 5.467362080877832,
 'min_samples_leaf': 4.658060078535629,
 'min_samples_split': 4.441055276271106,
 'n_estimators': 100.21921330946715};
params_forest_4_hrs_f1 =  {'max_depth': 5.467362080877832,
 'min_samples_leaf': 4.658060078535629,
 'min_samples_split': 4.441055276271106,
 'n_estimators': 100.21921330946715};
params_forest_5_ha_f1 = {'max_depth': 5.250986188919112,
 'min_samples_leaf': 2.686038087799833,
 'min_samples_split': 8.237194943120665,
 'n_estimators': 146.51458948929402};

# COMMAND ----------

outcome = 'uncontrolled_within_2_hrs'
fit_tree_classifier(incidents_data=grass_incidents, outcome=outcome, model_features=grass_features, classifier=BRF, fbeta=1, params_best=params_grass_2_hrs_f1, print_diagnostics=True, sampling_strategy="all", replacement=True, bootstrap=False, 
                    save_to = DIAGNOSTICS / (outcome + '_randomforest_diagnostics.eps'))

# COMMAND ----------

outcome = 'uncontrolled_within_100_ha'
fit_tree_classifier(incidents_data=grass_incidents, outcome=outcome, model_features=grass_features, classifier=BRF, fbeta=1, params_best=params_grass_100_ha_f1, print_diagnostics=True, sampling_strategy="all", replacement=True, bootstrap=False, 
                    save_to = DIAGNOSTICS / (outcome + '_randomforest_diagnostics.eps'))

# COMMAND ----------

outcome = 'uncontrolled_within_4_hrs'
fit_tree_classifier(incidents_data=forest_incidents, outcome=outcome, model_features=forest_features, classifier=BRF, fbeta=1, params_best=params_forest_4_hrs_f1, print_diagnostics=True, sampling_strategy="all", replacement=True, bootstrap=False, 
                    save_to = DIAGNOSTICS / (outcome + '_randomforest_diagnostics.eps'))

# COMMAND ----------

outcome = 'uncontrolled_within_5_ha'
fit_tree_classifier(incidents_data=forest_incidents, outcome=outcome, model_features=forest_features, classifier=BRF, fbeta=1, params_best=params_forest_5_ha_f1, print_diagnostics=True, sampling_strategy="all", replacement=True, bootstrap=False, 
                    save_to = DIAGNOSTICS / (outcome + '_randomforest_diagnostics.eps'))
