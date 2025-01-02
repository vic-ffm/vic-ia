from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import sklearn.metrics
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

from includes import base_include
from includes.base_include import *
# from modelling.get_model_data import *

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


def check_influence(input_df, output_model, outcome, cooks_distance_level=None):
    """
    Calculate influence based on Cook's distance and studentised residual
    """

    df = add_is_train_column(input_df).query('is_train_data==1')
    df['cooks_distance'] = output_model.get_influence().cooks_distance[0]
    df['resid_studentised'] = output_model.get_influence().resid_studentized

    if cooks_distance_level is None:
        # take the 10 points with the highest cooks distance
        # by setting threshold to be the value of the 11th highest value
        cooks_distance_level = df.cooks_distance.sort_values(ascending=False).iloc[11]
    studentised_residual_level = 3

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    sns.scatterplot(x=range(df.shape[0]), y='cooks_distance', alpha=0.3, data=df, ax=axs[0])
    axs[0].axhline(y=cooks_distance_level, color='black')

    sns.scatterplot(x=range(df.shape[0]), y='resid_studentised', hue=outcome, alpha=0.3, data=df, ax=axs[1])
    axs[1].axhline(y=studentised_residual_level, color='black')
    axs[1].axhline(y=-studentised_residual_level, color='black')
    sns.move_legend(axs[1], "upper left", bbox_to_anchor=(1, 1))
    
    high_cooks_distance = df.query('cooks_distance>@cooks_distance_level').sort_values('cooks_distance', ascending=False).index
    high_studentised_residual = df.query('abs(resid_studentised)>@studentised_residual_level').sort_values('resid_studentised', ascending=False).index

    return(high_cooks_distance.to_list(), high_studentised_residual.to_list())


def plot_feature_distributions(data, fit, list_of_features, log_transform=[]):
    """
    Plot distribution of linear contribution of each feature to the input fit model. This is useful for comparing how much the different input features effect the model.
    """
    fig, ax = plt.subplots(1,1)
    for feature in list_of_features:
        if feature in log_transform:
            sns.kdeplot(data=fit.params[f'np.log1p({feature})']*np.log1p(data[[feature]]), x=feature, label=feature)
        else:    
            sns.kdeplot(data=fit.params[feature]*data[[feature]], x=feature, label=feature)
        ax.legend()
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        
    plt.xlabel('coefficient*data')
    plt.show()


def plot_summed_feature_distribution(data, fit, list_of_features):
    """
    Plot distribution of linear contribution of the sum of the input list of features to the input fit model. This is useful for comparing how much much a group of features effects the model.
    """
    df = data.copy()

    for feature in list_of_features:
        df[feature + '_sum'] = fit.params[feature]*df[feature]
    sum_dist = df[[feature + '_sum' for feature in list_of_features]].sum(axis=1).to_frame(name='summed_distribution')

    fig, ax = plt.subplots(1,1)
    sns.kdeplot(data=sum_dist, x='summed_distribution', label=' '.join(list_of_features))
    ax.legend()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
