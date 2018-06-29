"""Wraps sklearn Gradient Boosting Regressor to
    1) automate modeling similar to gbm library in R
    2) overlay data and descriptive statistics in data visualization
    of partial dependencies for better inference

author: Alex Radnaev

date created: 2018-06-15
"""

import sklearn_gbm_extend
import pandas as pd
import sklearn.model_selection as skl_ms
import sklearn.ensemble as skl_e
import sklearn.metrics as skl_metrics
import numpy as np
import pandas_extend as pde
import logging
import file_system as fs


class GBMwrapper():
    def __init__(
            self,
            df_dataset,
            outcome,
            weights_col_name = None,
            features_list = None,
            impute_NAs = True,
            tail_threshold = 10,
            destination_dir = './gbm_output/',
            ax_limits_per_feature = None):

        """Creates gbm object for future modeling.

        arguments:
            df_dataset - dataframe with all data:
                numeric/ categorical features, and the outcome
            outcome - column name with the outcome
            features_list - list of strings representing features columns.
                by default all columns will be used except the oucome
            impute_NAs - flag to fill NAs with median values
            tail_threshold - minimum number of observations per given category.
                Categories with smaller counts will be merged to 'Other'
            destination_dir - directory where output files will be saved
            ax_limits_per_feature - dictionary for customizing plots axes:
                {
                    'feature_name1': {'xlim': [0, 100], 'ylim': [0, 1]},
                    'feature_name2': {'ylim': [0, 0.5]},
                    'feature_name5': {'xlim': [-1, 1]},
                }
        """

        self.destination_dir = destination_dir
        fs.prepare_folder_for(self.destination_dir + 'temp.txt')
        self.df_dataset = df_dataset
        self.features_list = features_list
        self.outcome = outcome
        self.impute_NAs = impute_NAs
        self.tail_threshold = tail_threshold
        if self.features_list is not None:
            self.df_dataset = self.df_dataset[
                self.features_list + [self.outcome]]
        self.drop_na_outcomes()
        if weights_col_name is not None:
            self.weights = self.df_dataset[weights_col_name]
            self.df_dataset = self.df_dataset.drop(
                [weights_col_name], axis = 1)
        else:
            self.weights = pd.Series(np.ones(self.df_dataset.shape[0]))
        self.ax_limits_per_feature = ax_limits_per_feature
        self.default_params = {
            'n_estimators': 4000,
            'max_depth': 10,
            'max_features': 4,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'learning_rate': 0.01,
            'loss': 'ls'}

        self.get_categorical_features()
        if self.tail_threshold is not None:
            self.remove_long_tails()
        self.validate_data()
        self.prepare_data()

    def remove_long_tails(self):
        """Merges Categories with counts smaller than tail_threshold
        to 'Other'"""

        for col in self.categorical_features:
            v_counts1 = self.df_dataset[col].value_counts()
            removed = pde.remove_long_tail(
                self.df_dataset,
                col,
                self.tail_threshold,
                'Other')
            v_counts2 = self.df_dataset[col].value_counts()

            if removed:
                logging.info('removed long tail from {}'.format(col))
                logging.info('{} before: {}'.format(col, v_counts1))
                logging.info('{} after: {}'.format(col, v_counts2))

    def get_categorical_features(self):
        # Identifies categorical features in the dataset
        self.categorical_features = list(
            self.df_dataset.dtypes[self.df_dataset.dtypes == 'object'].index)
        logging.info('detected categorical_features: {}'.format(
            self.categorical_features))

    def drop_na_outcomes(self):
        # Removes observations with unknown outcome
        obs_n1 = self.df_dataset.shape[0]
        self.df_dataset = self.df_dataset.dropna(subset = [self.outcome])
        obs_n2 = self.df_dataset.shape[0]
        if obs_n2 < obs_n1:
            logging.warning('dropped {} observations without outcome'.format(
                obs_n2 - obs_n1))

    def impute_NAs_in_col(self, col):
        # Imputes NAs (only with median at the moment)
        pde.fill_na_median(self.df_dataset, col)

    def validate_data(self):
        # Ensures data does not have NAs
        na_errors = []
        for col in self.df_dataset.columns:
            if col not in self.categorical_features:
                na_count = sum(pd.isnull(self.df_dataset[col]))
                if na_count > 0:
                    if self.impute_NAs:
                        self.impute_NAs_in_col(col)
                        logging.warning('imputed {} NAs in {}'.format(
                            na_count, col))
                    else:
                        na_errors.append(
                            'Error: column "{}" has {} NA values'.format(
                                col, na_count))
        for na_error in na_errors:
            print(na_error)
        if len(na_errors) > 0:
            raise NameError('Must not have NAs for non-categorical values')

    def prepare_data(self):
        """ Prepares data for model training:
            one-hot encoding for categorical values
            setting aside test dataset """
        self.X = pd.get_dummies(self.df_dataset.drop([self.outcome], axis = 1))

        self.y = self.df_dataset[self.outcome]
        (self.train_X, self.test_X, self.train_y, self.test_y,
         self.train_weights, self.test_weights, self.df_train,
         self.df_test) = skl_ms.train_test_split(
            self.X, self.y, self.weights, self.df_dataset)

        self.gbm_tools = sklearn_gbm_extend.ToolsGBM(
            self.categorical_features,
            self.train_X,
            self.train_y,
            self.train_weights,
            outcome_label = self.outcome,
            destination_dir = self.destination_dir)

    def build_model(self, params = None, cv_n_splits = 5):
        """Builds model and stores it in the object:
            tunes number of estimators (trees) with cross validation
            evaluate performance
            plots training curves, feature importance, partial dependences

        arguments:
            params - gbm parameters dictionary (default as specified at init)
            cv_n_splits - number of splits for cross-validation (default 5)"""
        if params is None:
            params = self.default_params
        self.params = params
        logging.info('Cross-validation parameter optimization started.')
        val_scores, std_val_scores = self.gbm_tools.cv_estimate(
            params, n_splits = cv_n_splits)
        cv_n_trees, cv1se_n_trees = self.gbm_tools.cv_n_tree(
            val_scores, std_val_scores)
        vertical_lines = {}

        selected_n_trees = cv_n_trees
        if pd.notnull(cv1se_n_trees) and cv1se_n_trees != 0:
            selected_n_trees = cv1se_n_trees
            vertical_lines['selected cv min less 1 std at'] = cv1se_n_trees
            vertical_lines['cv min at '] = cv_n_trees
        else:
            vertical_lines['selected cv min at '] = cv_n_trees

        logging.info('minium cv error at {} trees'.format(cv_n_trees))
        logging.info('minimum cv error within 1 std at {} trees'.format(
            cv1se_n_trees))
        logging.info('selected n_trees: {}'.format(selected_n_trees))

        logging.info('plotting all trees training curves')
        self.gbm = skl_e.GradientBoostingRegressor(**params)
        self.gbm_fit()

        self.gbm_tools.plot_gbm_training(
            self.gbm, self.test_X, self.test_y, cv_scores = val_scores,
            cv_std_scores = std_val_scores,
            fig_params = {'figsize': (11, 11)},
            vertical_lines = vertical_lines)

        self.update_n_trees(selected_n_trees)

        self.evaluate_performance(self.gbm)

        self.plot_output()
        return self.gbm

    def gbm_fit(self):
        """fits gbm model"""
        logging.info('gbm fitting...')
        self.gbm.fit(
            self.train_X, self.train_y,
            sample_weight = self.train_weights)

    def update_n_trees(self, new_n_trees):
        """updates number of estimators, then refits and reevalute performance

        arguments:
            new_n_trees - number of trees (estimators) to be set"""

        logging.info('updating n trees with selected {}'.format(
            new_n_trees))
        self.gbm.n_estimators = new_n_trees
        self.params['n_estimators'] = new_n_trees
        self.gbm_fit()
        self.evaluate_performance(self.gbm)

    def plot_output(self, ax_limits = None, **fig_params):
        """plots feature importance and partial dependencies plots
        arguments:
            ax_limits - dictionary for customizing plots axes:
                {
                    'feature_name1': {'xlim': [0, 100], 'ylim': [0, 1]},
                    'feature_name2': {'ylim': [0, 0.5]},
                    'feature_name5': {'xlim': [-1, 1]}}
                if not provided, limits are set based on data
            **fig_params - keyword parameters to be passed to plots"""

        if ax_limits is None:
            ax_limits = self.ax_limits_per_feature
        logging.info('plotting feature importances and partial dependences')
        self.gbm_tools.feature_importances_plot(
            self.gbm, {'figsize': (11, 11)})
        self.gbm_tools.partial_dependencies_plots(
            self.gbm,
            self.params,
            ax_limits = ax_limits,
            **fig_params)

    def evaluate_performance(self, gbm):
        # evaluate performance of the provided gbm model
        logging.info('evaluating performance')
        pred_test_y = gbm.predict(self.test_X)
        mse = skl_metrics.mean_squared_error(
            self.test_y, pred_test_y,
            sample_weight = 1 / self.test_weights)
        vexp = skl_metrics.explained_variance_score(
            self.test_y, pred_test_y, sample_weight = 1 / self.test_weights)
        print('rmse: {:.2f}, variance explained: {:.0f}%'.format(
            np.sqrt(mse), vexp * 100.))
        predicted_label = 'predicted_{}'.format(self.outcome)
        self.df_test = self.df_test.copy()
        self.df_test[predicted_label] = pred_test_y
        self.df_test['diff'] = self.df_test[predicted_label]\
            - self.df_test[self.outcome]
        self.df_test['abs diff'] = abs(self.df_test['diff'])
        self.df_test[[predicted_label, self.outcome, 'diff']].hist()

        print('{} standard deviation: {:.2f},\
prediction error std: {:.2f},\
average absolute error: {:.2f}'.format(
            self.outcome,
            np.std(self.df_test[self.outcome]),
            np.std(self.df_test['diff']),
            np.mean(self.df_test['abs diff'])))

    def predict_dataset(self, df):
        """Adds column with predictions for provided data

        arguments:
            df - dataframe of the same format as training dataframe
                but without the outcome"""
        df[self.outcome + '_predicted'] = self.gbm.predict(pd.get_dummies(df))
        return df
