from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# logistic regression
import sklearn.metrics
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

# random forests
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
from hyperopt import tpe,hp,Trials
from hyperopt.fmin import fmin
from sklearn.metrics import mean_squared_error,make_scorer

from base_include import *

FEATURE_NAMES = {'T_SFC': 'Temperature',
 'T_SFC_ishistorical': 'Historical indicator',
 'RH_SFC': 'Relative humidity',
 'DF_SFC': 'Drought factor',
 'WindMagKmh_SFC': 'Windspeed',
 'KBDI': 'KBDI',
 'elevation_m': 'Elevation',
 'ruggedness_average_3km': 'Average ruggedness (3km)',
 'building_density_20km': 'Building density (20km)',
 'building_density_3km': 'Building density (3km)',
 'road_density_km_in_3km': 'Road density (3km)',
 'road_distance_m': 'Distance to road',
 'grass_density_3km': 'Grass density (3km)',
 'forest_density_3km': 'Forest density (3km)',
 'shrub_density_3km': 'Scrub density (3km)',
 'noveg_density_3km': 'Noveg density (3km)',
 'distance_to_interface': 'Distance to interface',
 'soil_moisture': 'Soil moisture',
 'Curing': 'Curing'}


def fit_logistic_regression(incidents_data, outcome, model_features, interactions=None, transform=None, recall_for_threshold=None, print_summary=False, ):

    """
    Fit logistic regression to the incidents data set
    """
    
    outcome_prob = outcome + "_p"
       
    if 'is_train_data' not in incidents_data.columns:
        print('Adding training column...')
        incidents_data = add_is_train_column(incidents_data)
        
    # Split data into test and train based on is_train_data column
    incidents_train = incidents_data.query('is_train_data==1')
    incidents_test = incidents_data.query('is_train_data==0')
    print("Fraction of data used in training:", incidents_train.shape[0]/incidents_data.shape[0])
    print("Num rows in train: ", incidents_train.shape[0], "Num rows in test: ", incidents_test.shape[0])

    # model_features = [x for x in model_features if not x.startswith(primary_fuel_type + '_density')]

    if transform is not None:
        # key is the variable name, value is the function
        for key, value in transform.items():
            model_features = list(map(lambda x: x.replace(key, f"{value}({key})"), model_features))

    if interactions is not None:
        model_features = model_features + [':'.join(x) for x in interactions]

    rslt = smf.logit(formula=F"{outcome} ~ {'+'.join(model_features)}", data=incidents_train).fit()
    if print_summary:
        print(outcome)
        print(rslt.summary())

    incidents_test = incidents_test.assign(**{outcome_prob: rslt.predict(exog=incidents_test)})
    incidents_train = incidents_train.assign(**{outcome_prob: rslt.predict(exog=incidents_train)})

    return rslt, incidents_train, incidents_test, 

def get_model_diagnostics(rslt, incidents_train, incidents_test, outcome, print_diagnostics=False, print_confusion_matrix=False, print_density=False, print_appendix_diagnostics=False, normalise_confusion=False, recall_for_threshold=None, model_name_text='', save_to=None):
    outcome_prob = outcome + "_p"
    outcome_threshold_half = outcome + "_threshold_half"
    outcome_threshold_roc = outcome + "_threshold_roc"
    outcome_threshold_f1 = outcome + "_threshold_f1"
    outcome_threshold_recall = outcome + "_threshold_recall"

    # Prediction based on various thresholds
    # https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
    ## From ROC
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(incidents_train[outcome], incidents_train[outcome_prob])
    ## get the best threshold
    threshold_roc = thresholds[np.argmax(tpr-fpr)]
    ## From Precision-Recall (F1)
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(incidents_train[outcome], incidents_train[outcome_prob])
    with warnings.catch_warnings():
        # f1 score is nan when precision and recall are zero, which is the behaviour we want, so no need for a warning
        warnings.filterwarnings("ignore", message="invalid value encountered in divide")
        f1_scores = 2*recall*precision/(recall+precision)    
    threshold_f1 = thresholds[np.nanargmax(f1_scores)]
    ## From set value of recall
    if recall_for_threshold is not None:
        threshold_set_recall = thresholds[np.argmin(abs(recall_for_threshold - recall))]

    # Predictions based on the 3 thresholds
    incidents_test = incidents_test.assign(**{outcome_threshold_half: np.where(incidents_test[outcome_prob] > 0.5, 1, 0),
                                              outcome_threshold_roc: np.where(incidents_test[outcome_prob] > threshold_roc, 1, 0),
                                              outcome_threshold_f1: np.where(incidents_test[outcome_prob] > threshold_f1, 1, 0)
                                              })
    
    if recall_for_threshold is not None:
        incidents_test = incidents_test.assign(**{outcome_threshold_recall: np.where(incidents_test[outcome_prob] > threshold_set_recall, 1, 0)})

    # Diagnostic plots for model
    if print_diagnostics:
        fig, axs = plt.subplots(1, 4, figsize=(20,6))
        fig.suptitle(outcome)
        sns.histplot(data=incidents_test, x=outcome_prob , hue=outcome, ax=axs[0])
        sns.kdeplot(data=incidents_test, x=outcome_prob , hue=outcome, common_norm=False, clip=(0.0, 1.0), ax=axs[1])
        axs[1].set_xlim([0, 1])
        axs[1].set(title=outcome_prob.replace('_p', '').replace('_', ' ').capitalize(), xlabel = 'Probability of being uncontrolled')
        axs[1].legend(title='Ground truth', labels=['uncontrolled', 'controlled'])
        # sns.violinplot(data=incidents_test, x=outcome_prob , hue=outcome, ax=axs[1])

        auc = roc_auc_score(incidents_test[outcome], incidents_test[outcome_prob])

        fpr, tpr, thresholds = sklearn.metrics.roc_curve(incidents_test[outcome], incidents_test[outcome_prob])
        axs[2].plot(fpr,tpr,label=f"AUC = {auc:.4}")
        ix = (np.abs(thresholds - threshold_roc)).argmin() # index of threshold closest to best ROC threshold from training set
        axs[2].scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
        axs[2].set(xlabel="False positive rate", ylabel="True positive rate")
        axs[2].legend(loc=4,fontsize=18)

        ap = sklearn.metrics.average_precision_score(incidents_test[outcome], incidents_test[outcome_prob])

        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(incidents_test[outcome], incidents_test[outcome_prob])
        axs[3].plot(recall, precision, label=f"AP = {ap:.4}")
        ix = (np.abs(thresholds - threshold_f1)).argmin() # index of threshold closest to best F1 threshold from training set
        _, _, f1_best, _ = sklearn.metrics.precision_recall_fscore_support(incidents_test[outcome], incidents_test[outcome_threshold_roc])
        axs[3].scatter(recall[ix], precision[ix], marker='o', color='black', label=f'Best (F1 {f1_best[1]:.4f})')
        axs[3].set(xlabel="Recall", ylabel="Precision")
        axs[3].legend(loc=8,fontsize=18)
        plt.show()

    # Plot only the density plot from the diagnostics (for paper)
    if print_density:
        fig, axs = plt.subplots(1, 1, figsize=(4,4))
        sns.kdeplot(data=incidents_test, x=outcome_prob , hue=outcome, common_norm=False, clip=(0.0, 1.0), ax=axs)
        axs.set_xlim([0, 1])
        axs.set(title=(model_name_text + outcome_prob.replace('_p', '').replace('_', ' ')), xlabel = 'Probability of being uncontrolled')
        axs.legend(title='Ground truth', labels=['uncontrolled', 'controlled'])
        if save_to is not None:
            plt.savefig(save_to)
        else:
            plt.show()

    # Print diagnostics for paper appendix
    if print_appendix_diagnostics:
        fig, axs = plt.subplots(1, 3, figsize=(20,6))
        # fig.suptitle(outcome)

        sns.kdeplot(data=incidents_test, x=outcome_prob , hue=outcome, common_norm=False, clip=(0.0, 1.0), ax=axs[0])
        axs[0].set_xlim([0, 1])
        axs[0].set_xlabel('Probability of being uncontrolled', fontsize=18)
        axs[0].set_ylabel('Density', fontsize=18)
        axs[0].legend(title='Ground truth', labels=['uncontrolled', 'controlled'], fontsize=18, title_fontsize=18)
        
        auc = roc_auc_score(incidents_test[outcome], incidents_test[outcome_prob])
        fpr, tpr, _ = sklearn.metrics.roc_curve(incidents_test[outcome], incidents_test[outcome_prob])
        axs[1].plot(fpr,tpr,label=f"AUC = {auc:.4}")
        axs[1].set_xlabel("False positive rate", fontsize=18)
        axs[1].set_ylabel("True positive rate", fontsize=18)
        axs[1].legend(loc=4,fontsize=18, handlelength=0, handletextpad=0)

        ap = sklearn.metrics.average_precision_score(incidents_test[outcome], incidents_test[outcome_prob])
        precision, recall, _ = sklearn.metrics.precision_recall_curve(incidents_test[outcome], incidents_test[outcome_prob])
        axs[2].plot(recall, precision, label=f"AP = {ap:.4}")
        axs[2].set_xlabel("Recall", fontsize=18)
        axs[2].set_ylabel("Precision", fontsize=18)
        axs[2].legend(loc=8,fontsize=18, handlelength=0, handletextpad=0)
        if save_to is not None:
            plt.savefig(save_to)
        else:
            plt.show()

    
    # Confusion matrices for each of the thresholds
    if print_confusion_matrix:
        if normalise_confusion:
            print_format = '.4f'
            normalisation_type = 'all'
        else:
            print_format = 'g'
            normalisation_type = None


        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        _, recall, _, _ = sklearn.metrics.precision_recall_fscore_support(incidents_test[outcome], incidents_test[outcome_threshold_half])
        sns.heatmap(sklearn.metrics.confusion_matrix(incidents_test[outcome],
                                                     incidents_test[outcome_threshold_half], 
                                                     normalize=normalisation_type),                    
                    annot=True, cmap='Blues', fmt=print_format, annot_kws={"fontsize":20}, ax=axs[0])
        axs[0].set(xlabel="Predicted label", ylabel="True label", title=f"Standard threshold 0.5. Recall {recall[1]:.4f}")
        
        _, recall, _, _ = sklearn.metrics.precision_recall_fscore_support(incidents_test[outcome], incidents_test[outcome_threshold_roc])
        sns.heatmap(sklearn.metrics.confusion_matrix(incidents_test[outcome],
                                                     incidents_test[outcome_threshold_roc], 
                                                     normalize=normalisation_type),                       
                    annot=True, cmap='Blues', fmt=print_format, annot_kws={"fontsize":20}, ax=axs[1])
        axs[1].set(xlabel="Predicted label", ylabel="True label", title=f"AUC threshold {threshold_roc:.4f}. Recall {recall[1]:.4f}")

        _, recall, _, _ = sklearn.metrics.precision_recall_fscore_support(incidents_test[outcome], incidents_test[outcome_threshold_f1])
        sns.heatmap(sklearn.metrics.confusion_matrix(incidents_test[outcome],
                                                     incidents_test[outcome_threshold_f1],
                                                     normalize=normalisation_type),
                                                     annot=True, cmap='Blues', fmt=print_format, annot_kws={"fontsize":20}, ax=axs[2])
        axs[2].set(xlabel="Predicted label", ylabel="True label", title=f"F1 threshold {threshold_f1:.4f}. Recall {recall[1]:.4f}")

        if recall_for_threshold is not None:
            _, recall, _, _ = sklearn.metrics.precision_recall_fscore_support(incidents_test[outcome], incidents_test[outcome_threshold_recall])
            fig, axs = plt.subplots(1, 3, figsize=(15, 4))
            sns.heatmap(sklearn.metrics.confusion_matrix(incidents_test[outcome],
                                                         incidents_test[outcome_threshold_recall],
                                                         normalize=normalisation_type),                    
                        annot=True, cmap='Blues', fmt=print_format, annot_kws={"fontsize":20},
                                                         ax=axs[0]
                                                         )
            axs[0].set(xlabel="Predicted label", ylabel="True label", title=f"Recall threshold {recall[1]}")


    return None



def add_is_train_column(incidents_data, significant_fire_seasons=['2008-09', '2019-20']):
    """
    Add column which indicates which rows are in the training data set. This is done by taking 70% of the fire seasons for the training set and the rest for the test set, but of the two significant fire seasons (2008-09, 2019-20) one is assigned to each of the train and test sets.
    A test_season can be entered if you want to ensure a specific season or seasons is in the test set.
    """
    TRAINING_RATIO = 0.7
    nonsignificant_fire_seasons = [x for x in np.unique(incidents_data.season) if x not in significant_fire_seasons]
    num_samples_train = int(np.floor((len(nonsignificant_fire_seasons) + len(significant_fire_seasons))*TRAINING_RATIO))
    np.random.seed(RANDOM_SEED)
    # randomly assign 80% of the nonsignificant fire seasons to the training set
    season_train = np.random.choice(nonsignificant_fire_seasons, size=int(num_samples_train+len(significant_fire_seasons)/2), replace=False)
    # randomly add one of the two significant fire seasons to the training set.
    season_train = np.concatenate((season_train, np.random.choice(significant_fire_seasons, size=int(len(significant_fire_seasons)/2), replace=False)), axis=None)
    incidents_data = incidents_data.assign(is_train_data = np.where(incidents_data.season.isin(season_train), 1, 0))

    return incidents_data


def compute_vif(df, model_features):
    """
    Compute VIF for all the input features
    """      
    X = df.copy()[model_features]
    # the calculation of variance inflation requires a constant
    X['intercept'] = 1
    
    # create dataframe to store vif values
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif[vif['Variable']!='intercept']
    return vif.sort_values('VIF', ascending=False)


def calculate_quantile_resiuals(y_pred, y_proba):
    """
    Calculate quantile residuals (Dunn & Smyth 2018)
    https://stats.stackexchange.com/questions/1432/what-do-the-residuals-in-a-logistic-regression-mean
    """
    result = []

    for ii in range(len(y_pred)):
        a = stats.binom.cdf(y_pred.iloc[ii]-1, 1, y_proba.iloc[ii])
        b = stats.binom.cdf(y_pred.iloc[ii], 1, y_proba.iloc[ii])
        result.append(stats.norm.ppf(np.random.uniform(a, b, size=1))[0])
    
    return result


def plot_quantile_residuals(output_df, outcome, outcome_prob, model_features, log_transform=[], n_cols=4, model_name_text='', save_to=None):
    """
    Plot quantile residuals vs the outcome and each model feature, as suggested by Dunn & Smyth 2018
    """
    df = (output_df
          .assign(**{'quantile_residual': calculate_quantile_resiuals(output_df[outcome], output_df[outcome_prob]),
                     outcome_prob + '_const_inf_scale': np.arcsin(np.sqrt(output_df[outcome_prob]))
                    }
                  )
          )
    features = [outcome_prob + '_const_inf_scale'] + model_features
    feature_names = FEATURE_NAMES.copy()
    feature_names.update({outcome_prob + '_const_inf_scale': 'outcome probability on constant information scale'})

    n_rows=int(np.ceil(len(features)/n_cols))
    fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(21, 14))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.suptitle(model_name_text + outcome.replace('_', ' '), fontsize=14, y=0.9)
    for row in range(n_rows):
        if row==n_rows-1:
            col_range=len(features)%n_cols
            if col_range == 0:
                col_range = n_cols
        else:
            col_range=n_cols
        for col in range(col_range):
            feature = features[row*n_cols + col]
            axs[row,col].axhline(y=0, color='black', label='y=0')
            sns.regplot(x=feature, y='quantile_residual', data=df, scatter_kws={'alpha':0.3, 'color': 'grey'}, line_kws={'label': 'lowess fit'}, lowess=True, ax=axs[row,col])
            axs[row,col].set(xlabel = feature_names[feature], ylabel = 'Quantile residuals')
            if feature in log_transform:
                axs[row,col].set_xscale('function', functions=(np.log1p, np.expm1))
            axs[row,col].legend()
            axs[row,col].tick_params(labelrotation=90)
    
    # remove any empty subplots
    for ix in range(col, n_cols):
        fig.delaxes(axs[n_rows-1][ix])

    if save_to is None:
        plt.show()
    else:
        plt.savefig(RESIDUALS / (outcome + '.eps'))


def fit_tree_classifier(incidents_data, outcome, model_features, classifier, fbeta=1, params_best=None, print_diagnostics=False, print_feature_importance=False, random_state=RANDOM_SEED, save_to=None, **args):
    """
    Fit random forest or balanced random forest classifier
    """
    model = MS(incidents_data[model_features], intercept=False)
    D = model.fit_transform(incidents_data[model_features])
    X_train = np.asarray(D[incidents_data.is_train_data == 1])
    X_test = np.asarray(D[incidents_data.is_train_data == 0])
    y_train = np.asarray(incidents_data.query('is_train_data == 1')[outcome])
    y_test = np.asarray(incidents_data.query('is_train_data == 0')[outcome])

    clf = classifier(random_state=random_state, **args).fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    y_proba=clf.predict_proba(X_test)

    # Split into train, test, validation: train/test split based on fire season, then split train (80% of original train), validation (20% of original train)
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
    """
    Hyperparameter tuning for random forests: tries different combinations of hyperparameters and returns the best combination
    https://www.kaggle.com/code/virajbagal/eda-xgb-random-forest-parameter-tuning-hyperopt
    """
    params={'n_estimators':hp.uniform('n_estimators',100,300),
           'max_depth':hp.uniform('max_depth',5,20),
           'min_samples_leaf':hp.uniform('min_samples_leaf',1,5),
           'min_samples_split':hp.uniform('min_samples_split',2,10)}
    best=fmin(fn= lambda p: objective(p, classifier, X_train, y_train, X_val, y_val, fbeta, **args),
              space=params,algo=tpe.suggest,trials=trial,max_evals=500,rstate=np.random.default_rng(seed))
    return best


def objective(params, classifier, X_train, y_train, X_val, y_val, fbeta, **args):
    """
    Objective function for determining how well a given set of hyperparameters performs
    https://www.kaggle.com/code/virajbagal/eda-xgb-random-forest-parameter-tuning-hyperopt
    """

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
