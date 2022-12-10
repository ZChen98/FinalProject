"""
Justin Chen
Final Project - Functions
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV

PARAM_INTERESTS = ['Ladder score', 'Logged GDP per capita',
                    'Social support', 'Healthy life expectancy',
                  'Freedom to make life choices',
                  'Generosity', 'Perceptions of corruption', 'High income',
                  'Low income', 'Lower middle income', 'Upper middle income', 'IncomeGroup']

def prep_data():
    """Imports data and combines the datasets.
    Cleans the missing data and prep for further analysis.
    Returns a dataset.
    """
    # Load dataset from world happiness and gini coefficient
    df_happiness = pd.read_csv('../data/2021.csv')
    df_gini = pd.read_csv(
        '../data/Metadata_Country_API_SI.POV.GINI_DS2_en_csv_v2_4701295.csv')

    # Clean missing data
    df_gini = df_gini.drop(['Unnamed: 5', 'SpecialNotes'], axis=1)
    df_gini = df_gini[df_gini['IncomeGroup'].notna()]

    # Join two datasets on country name
    df_joined = pd.merge(df_happiness, df_gini, how='inner',
                         left_on='Country name', right_on='TableName')

    # Change the IncomeGroup column to category and get dummy values
    df_joined['IncomeGroup'] = df_joined['IncomeGroup'].astype('category')
    df_joined = pd.concat(
        (df_joined, pd.get_dummies(df_joined['IncomeGroup'])), axis=1)
    df_final = df_joined[PARAM_INTERESTS].copy()

    return df_final


def visual_data():
    """Creates data visualization. Returns plots."""
    df_visual = prep_data()
    figure_num = 0

    # Histgram of income groups
    sns.histplot(data=df_visual, x='IncomeGroup').set(
        title='World Income Groups')

    # Joint plot of the healthy life expectancy and ladder scores
    sns.jointplot(data=df_visual[['Healthy life expectancy', 'Ladder score']],
                  x='Healthy life expectancy', y='Ladder score')

    plt.figure()
    # Heat map for correlation of parameters of interests and ladder score
    sns.heatmap(data=df_visual.corr(), annot=True, cmap='Blues').set(
        title='Correlation of Parameters of Interests and Ladder Score')
    figure_num = plt.gcf().number

    plt.show()
    return figure_num

def generate_stat():
    """Generates descriptive statistics. Returns a dataframe of the results"""
    df_stat = prep_data()
    stat_result = df_stat.describe().copy()
    return stat_result


def run_models():
    """Runs the models including OLS model, ridge and lasso regression
        Returns the model's intercept and coefficients with higheset crossvalidation score
    """
    # Store all the columns except Ladder score
    # Stores medv
    df_model = prep_data()
    x_rr = df_model.loc[:, ~df_model.columns.isin(
        ['Ladder score', 'IncomeGroup'])].copy()
    y_rr = df_model['Ladder score']

    # Build linear, ridge and lasso models
    model_linear = LinearRegression().fit(x_rr, y_rr)
    model_ridge = RidgeCV(cv=8).fit(x_rr, y_rr)
    model_lasso = LassoCV(cv=8).fit(x_rr, y_rr)

    # Get the crossvalidation scores of each model
    linear_score = model_linear.score(x_rr, y_rr)

    best_ridge_alpha = model_ridge.alpha_
    best_ridge = Ridge(alpha=best_ridge_alpha).fit(x_rr, y_rr)
    ridge_score = best_ridge.score(x_rr, y_rr)  # pylint: disable=no-member

    best_lasso_alpha = model_lasso.alpha_
    best_lasso = Lasso(alpha=best_lasso_alpha).fit(x_rr, y_rr)
    lasso_score = best_lasso.score(x_rr, y_rr)

    scores = {'OLS': linear_score, 'Ridge': ridge_score, 'Lasso': lasso_score}
    models = {'OLS': {'intercept': model_linear.intercept_,
            'coefficient': model_linear.coef_},
              'Ridge': {'intercept': model_ridge.intercept_,
              'coefficient': model_ridge.coef_, 'alpha': best_ridge_alpha},
              'Lasso': {'intercept': model_lasso.intercept_,
              'coefficient': model_lasso.coef_, 'alpha': best_lasso_alpha}}
    return max(scores), max(scores.values()), models[max(scores)]
