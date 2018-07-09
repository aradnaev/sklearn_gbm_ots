"""
Purpose: provides an example of using sklearn_gbm_wrapper module
using California housing dataset in which on numerical feature (Population)
is converted to categorical feature

Data created: 2018-06-28

Author: Alex Radnaev

"""


import sklearn.datasets as sklearn_ds
import sklearn_gbm_ots as gbm_ots
import pandas as pd


def population_mapper(x):
    if x < 1000:
        return '<1k'
    elif x < 2000:
        return '1k-2k'
    else:
        return '2k+'


def prepare_california_housing_dataframe(outcome):
    """Returns california housing dataset as a dataframe

    arguments:
        outcome - column name for the outcome"""
    housing_dataset = sklearn_ds.california_housing.fetch_california_housing()
    X = housing_dataset.data
    y = housing_dataset.target

    df_X = pd.DataFrame(X, columns=housing_dataset.feature_names)
    df_y = pd.DataFrame(y, columns=[outcome])
    df = pd.concat([df_X, df_y], axis=1)
    df['Population'] = df['Population'].apply(population_mapper)

    return df


def run_gbm():
    """Prepares California housing dataset and build gbm model

    Returns:
        fitted sklearn_gbm_wrapper.GBMwrapper object"""
    outcome = 'house price, 100k'

    df = prepare_california_housing_dataframe(outcome)

    housing_gbm = gbm_ots.GBMwrapper(
        df[:100],
        outcome,
        show_plots=False,
        random_state=2018)

    gbm_params = {
        'n_estimators': 300,
        'max_depth': 3,
        'max_features': 5,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'learning_rate': 0.5,
        'loss': 'ls'}

    housing_gbm.build_model(params=gbm_params)

    return housing_gbm


def custom_plotting(gbm):
    """Demonstrates how to customize plots"""
    gbm.plot_output(
        ax_limits={'AveBedrms': {'xlim': [0.9, 1.2]}},
        figsize=(11, 11),
        absolute_yticks=False,
        absolute_yscale=False,
        file_prefix='relative_scale_relative_ticks')

    gbm.plot_output(
        ax_limits={'AveBedrms': {'xlim': [0.9, 1.2]}},
        figsize=(11, 11),
        absolute_yticks=True,
        absolute_yscale=False,
        file_prefix='relative_scale_abs_ticks')

    # gbm.plot_output(
    #     ax_limits={'AveBedrms': {'xlim': [0.9, 1.2]}},
    #     figsize=(11, 11),
    #     absolute_yticks=False,
    #     absolute_yscale=True,
    #     file_prefix='abs_scale_relative_ticks')

    # gbm.plot_output(
    #     ax_limits={'AveBedrms': {'xlim': [0.9, 1.2]}},
    #     figsize=(11, 11),
    #     absolute_yticks=True,
    #     absolute_yscale=True,
    #     file_prefix='abs_scale_abs_ticks')


def main():
    housing_gbm = run_gbm()
    custom_plotting(housing_gbm)


if __name__ == '__main__':
    main()
