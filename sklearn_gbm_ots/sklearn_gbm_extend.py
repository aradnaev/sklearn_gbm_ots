"""Additional methods suplementing Gradient Boosting Regressor from
sklearn

data created: 2018-06-14

author: Alex Radnaev

# todo: fix absolute scale plotting

"""
import sklearn.model_selection as skl_ms
import sklearn.ensemble as skl_e
import numpy as np
import matplotlib.pyplot as plt
import sklearn_gbm_ots.misc_tools as mu
import sklearn.ensemble.partial_dependence as skl_e_pd
import logging
import pandas as pd
import pandas_extend as pde


class ToolsGBM():
    def __init__(
            self,
            categorical_labels,
            train_X,
            train_y,
            train_weights,
            outcome_label='outcome',
            destination_dir=None,
            show_plots=True,
            random_state=None,
            file_prefix=''):
        self.file_prefix = file_prefix
        self.random_state = random_state
        self.show_plots = show_plots
        if not self.show_plots:
            plt.ioff()
        self.destination_dir = destination_dir
        self.feature_labels = train_X.columns
        self.categorical_labels = categorical_labels
        self.train_X = train_X
        self.train_y = train_y
        self.train_weights = train_weights
        self.weights_warning()
        self.outcome_label = outcome_label
        self.categorical_features_ind = None
        self.categorical_indecies = self.get_categorical_indecies()
        print('categorical_indecies: {}'.format(self.categorical_indecies))
        self.non_categorical_indecies = \
            set([i for i in range(len(self.feature_labels))])\
            - set(self.categorical_indecies.values())
        self.consolidated_feature_importance = None
        self.feature_index_by_label = \
            dict([(self.feature_labels[i], i)
                 for i in range(len(self.feature_labels))])
        self.plot_margin = 0.1
        self.train_X_cv_subsets = []
        self.train_y_cv_subsets = []
        self.train_weights_cv_subsets = []
        self.models = []
        self.set_colors()

    def set_colors(
            self,
            pdp_color='C0',
            counts_color='C1',
            means_color='C2',
            data_color='C3'):
        self.pdp_color = pdp_color
        self.counts_color = counts_color
        self.means_color = means_color
        self.data_color = data_color

    def weights_warning(self):
        if self.train_weights is not None:
            logging.warning('please check the effective sample size \
for signal-to-noise calculations is {}. Otherwise renormalize weights\'s \
sum to the effective sample size'.format(np.sum(self.train_weights)))

    def update_random_state(self, params):
        params2 = params.copy()
        if 'random_state' not in params2:
            params2['random_state'] = self.random_state
        return params

    def cv_estimate(
            self,
            params,
            n_splits=3):
        params2 = self.update_random_state(params)
        cv = skl_ms.KFold(
            n_splits=n_splits,
            random_state=self.random_state)
        val_scores = np.zeros(
            (params['n_estimators'], n_splits),
            dtype=np.float64)
        i = 0
        for train, test in cv.split(self.train_X, self.train_y):
            # logging.debug('train indecies: {}'.format(train))
            train_X_cv_subset = self.train_X.iloc[train]
            train_y_cv_subset = self.train_y.iloc[train]
            train_weights_cv_subset = self.train_weights.iloc[train]
            self.train_X_cv_subsets.append(train_X_cv_subset)
            self.train_y_cv_subsets.append(train_y_cv_subset)
            self.train_weights_cv_subsets.append(train_weights_cv_subset)
            cv_model = skl_e.GradientBoostingRegressor(**params2)
            cv_model.fit(
                train_X_cv_subset, train_y_cv_subset,
                sample_weight=train_weights_cv_subset)

            logging.info('fitted cv model #{}/{}'.format(i + 1, n_splits))
            val_scores[:, i] += self.test_score_vs_stage(
                cv_model, self.train_X.iloc[test], self.train_y.iloc[test])
            i += 1
        mean_val_scores = np.mean(val_scores, axis=1)
        std_val_scores = np.std(val_scores, axis=1)

        return mean_val_scores, std_val_scores

    def cv_n_tree(self, mean_val_scores, std_val_scores):
        cv_n_trees = np.argmin(mean_val_scores)
        cv1se_val = mean_val_scores[cv_n_trees] + std_val_scores[cv_n_trees]
        logging.debug('cv_n_trees: {}'.format(cv_n_trees))
        logging.debug('cv1se_val: {}'.format(cv1se_val))
        if cv_n_trees > 0:
            diff = np.abs(mean_val_scores - cv1se_val)
            cv1se_n_trees = (diff[:cv_n_trees]).argmin()
            logging.debug('cv1se_n_trees: {}'.format(cv1se_n_trees))
        else:
            logging.warning('cv_n_trees is zero')
            cv1se_n_trees = np.nan
        if pd.notnull(cv1se_n_trees) and cv1se_n_trees > cv_n_trees:
            cv1se_n_trees = np.nan
        return cv_n_trees, cv1se_n_trees

    def test_score_vs_stage(self, model, test_X, test_y):
        score = np.zeros((model.n_estimators,), dtype=np.float64)
        for i, y_pred in enumerate(model.staged_predict(test_X)):
            score[i] = model.loss_(test_y, y_pred)
        return score

    def plot_gbm_training(
            self,
            gbm,
            test_X,
            test_y,
            fig_params=None,
            cv_scores=None,
            cv_std_scores=None,
            vertical_lines=None):

        if fig_params is None:
            fig_params = {}
        test_score = np.zeros(gbm.n_estimators, dtype=np.float64)
        for i, y_pred in enumerate(gbm.staged_predict(test_X)):
            test_score[i] = gbm.loss_(test_y, y_pred)

        plot_x = np.arange(gbm.n_estimators)
        fig = plt.figure(**fig_params)
        plt.title('GBM training curves')

        if cv_scores is not None:
            plt.plot(plot_x,
                     cv_scores,
                     label='CV score')
            if cv_std_scores is not None:
                # plt.errorbar(plot_x, cv_scores, yerr=cv_std_scores, fmt='o')
                plt.fill_between(
                    plot_x,
                    cv_scores - cv_std_scores,
                    cv_scores + cv_std_scores,
                    alpha=0.2)

        plt.plot(plot_x,
                 gbm.train_score_,
                 label='Training set score')
        plt.plot(plot_x,
                 test_score,
                 label='Test set score')

        plt.legend(loc='upper right')
        plt.xlabel('Number of trees')
        plt.ylabel('scores')
        ylim = plt.gca().get_ylim()
        if vertical_lines is not None:
            i = 0
            for name, n_tree in vertical_lines.items():
                plt.axvline(x=n_tree)
                plt.annotate(
                    s=' {} {}'.format(name, n_tree),
                    xy=(n_tree,
                        ylim[1] - (ylim[1] - ylim[0]) * 0.05 * (i + 1)))
                i += 1
        self.save_fig(fig, 'gbm_training_curves.png')

    def save_fig(self, fig, local_filename):
        if self.destination_dir is not None:
            fig.savefig(
                self.destination_dir + self.file_prefix + local_filename,
                bbox_inches='tight',
                dpi=150)

    def get_categorical_indecies(self):
        categorical_indecies = {}
        categorical_features_ind = {}
        for categorical_label in self.categorical_labels:
            for i, label in enumerate(self.feature_labels):
                if categorical_label + '_' in label:
                    if categorical_label not in categorical_indecies:
                        categorical_indecies[i] = categorical_label
                        if categorical_label not in categorical_features_ind:
                            categorical_features_ind[categorical_label] = set()
                        categorical_features_ind[categorical_label].add(i)
        self.categorical_features_ind = categorical_features_ind
        return categorical_indecies

    def consolidate_feature_importance(self, feature_importance_vals):
        feature_importance = {}
        categorical_indecies = self.get_categorical_indecies()
        for i, val in enumerate(feature_importance_vals):
            if i in categorical_indecies:
                feature = categorical_indecies[i]
                if feature not in feature_importance:
                    feature_importance[feature] = 0.
                feature_importance[feature] += val
            else:
                feature_importance[self.feature_labels[i]] = val
        self.consolidated_feature_importance = feature_importance
        return feature_importance

    def feature_importances_plot(self, gbm, fig_params=None):
        if fig_params is None:
            fig_params = {}
        fis = gbm.feature_importances_
        fis = (fis / fis.sum()) * 100.

        feature_labels2, fis2 = mu.dict_to_lists(
            self.consolidate_feature_importance(fis),
            target_type=np.array)
        ordered_ids = np.argsort(fis2)
        plot_x = np.arange(ordered_ids.shape[0]) + 0.5
        fig = plt.figure(**fig_params)
        plt.barh(plot_x, fis2[ordered_ids], align='center')
        plt.yticks(plot_x, feature_labels2[ordered_ids])
        plt.xlabel('Relative importance')
        plt.ylabel('Feature')
        plt.title('Feature importance ({} trees)'.format(
            gbm.n_estimators))
        if self.show_plots:
            plt.show()
        self.save_fig(fig, 'feature_importance_{}_trees.png'.format(
            gbm.n_estimators))

    def box_plots_raw_data(self, raw_data, plot_x):
        if np.sum(self.train_weights == 1) != len(self.train_weights):
            logging.warning('box plots do not take into account weights')
        bp = plt.boxplot(
            np.array(raw_data), positions=plot_x, manage_xticks=False)
        for k in bp.keys():
            [b.set_alpha(0.05) for b in bp[k]]

    def outcome_ylabel(self, absolute_yticks):
        if absolute_yticks:
            return self.outcome_label
        else:
            return self.outcome_label + ' deviation from average'

    def partial_dependency_catagorical_plot(
            self,
            gbm,
            feature_label,
            ax_limits_per_feature=None,
            overlay_box_plots=False,
            add_means=True,
            absolute_yscale=False,
            absolute_yticks=True,
            **fig_params):
        """Plots partial dependency for a categorical variable.

        arguments:
            absolute_yscale - absolute_scale (starts at zero) on y axis
            absolute_yticks - flag to subtract mean from y axis pabels
        Todo: split this methods into 2-3 smaller ones
        """
        legends = []
        legend_labels = []
        deltas = []
        deltas_unc = []
        labels = []
        means = []
        means_unc = []
        raw_data = []
        counts = []
        raw_stds = []
        width = 0.8
        if add_means:
            width = 0.35
        if ax_limits_per_feature is None:
            ax_limits_per_feature = {}
        outcome_mean = np.mean(self.train_y)
        for feature_index in self.categorical_features_ind[feature_label]:
            y, x = skl_e_pd.partial_dependence(
                gbm, [feature_index], X=self.train_X)

            stds = self.partial_dependency_uncertainty(
                [feature_index],
                grid=x[0],
                percentiles=(0.0, 1.0))
            if len(x) == 0 or len(y) == 0:
                logging.debug('no results for feature_index {}'.format(
                    feature_index))
            else:
                try:
                    delta = y[0][np.where(x[0] == 1)[0][0]]\
                            - y[0][np.where(x[0] == 0)[0][0]]
                    # if absolute_yscale or absolute_yticks:
                    #     logging.debug('original delta: {}'.format(delta))
                    #     delta += outcome_mean
                    #     logging.debug('original delta with mean: {}'.format(delta))
                    deltas.append(delta)
                    labels.append(self.feature_labels[feature_index].replace(
                        feature_label + '_', ''))
                    deltas_unc.append(np.sqrt(np.sum(stds**2)))

                    train_X_idx = self.train_X[
                        self.feature_labels[feature_index]] == 1
                    raw_data_subset = self.train_y[train_X_idx]
                    # if not absolute_yscale and not absolute_yticks:
                    #     raw_data_subset -= outcome_mean
                    raw_data_weights = self.train_weights[train_X_idx]
                    raw_data.append(raw_data_subset)
                    means.append(
                        np.average(raw_data_subset, weights=raw_data_weights))
                    means_unc.append(self.mean_uncertainty(
                        raw_data_subset,
                        weights=raw_data_weights))
                    counts.append(int(np.sum(raw_data_weights)))
                    raw_stds.append(
                        pde.std(raw_data_subset,
                                weights=raw_data_weights))
                except Exception as e:
                    logging.error('Cannot get partial dependence delta for \
feature index "{}" due to "{}"'.format(feature_index, e))
        idxs = np.argsort(deltas)
        plot_x = np.arange(idxs.shape[0]) + 0.5
        plot_deltas = np.array(deltas)[idxs]
        fig = plt.figure(**fig_params)
        ax = fig.add_subplot(1, 1, 1)
        if absolute_yscale or not absolute_yticks:
            bar_bottom = 0
        else:
            bar_bottom = outcome_mean
        logging.debug('bar bottom: {}, plot_deltas: {}'.format(
            bar_bottom, plot_deltas))
        pdp_bars = ax.bar(
            plot_x, plot_deltas,
            width=width, align='center',
            color=self.pdp_color,
            bottom=bar_bottom)

        legends.append(pdp_bars)
        legend_labels.append('Partial dependences with uncertainties')
        ax.errorbar(
            plot_x, plot_deltas + bar_bottom,
            yerr=np.array(deltas_unc)[idxs],
            color='grey',
            fmt='o')

        if add_means:
            means_y = np.array(means)[idxs]
            if not absolute_yticks:
                means_y -= outcome_mean
            logging.debug('bar bottom: {}, means_y: {}'.format(
                bar_bottom, means_y))
            means_bars = ax.bar(
                plot_x + width, means_y - bar_bottom,
                width=width, align='center',
                alpha=0.5,
                color=self.means_color,
                bottom=bar_bottom)
            ax.errorbar(
                plot_x + width, means_y,
                yerr=np.array(means_unc)[idxs],
                color='grey',
                fmt='o')
            legends.append(means_bars)
            legend_labels.append('Raw averages with uncertainties')
            # ax.legend(['Raw averages with uncertainties'])
        ax.grid(alpha=0.3)

        if overlay_box_plots:
            self.box_plots_raw_data(raw_data, plot_x)

        # store x,y limits based on primary data
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        counts_bars = self.overlay_counts_histogram(
            ax, plot_x, counts, xlim, ylim)
        if counts_bars is not None:
            legends.append(counts_bars)
            legend_labels.append('Counts')
        plt.xticks(plot_x, np.array(labels)[idxs], rotation='vertical')
        plt.xlabel('{}'.format(feature_label))
        plt.ylabel(self.outcome_ylabel(absolute_yticks))
        plt.title('Partial dependence on {} ({} trees)'.format(
            feature_label, gbm.n_estimators))
        ylim = ax_limits_per_feature.get('ylim')
        if ylim is not None:
            ax.set_ylim(tuple(ylim))
        xlim = ax_limits_per_feature.get('xlim')
        if xlim is not None:
            ax.set_ylim(tuple(xlim))
        logging.debug('creating legend with handles: "{}", labels: "{}"'.format(
            legends, legend_labels))
        try:
            ax.legend(
                handles=legends,
                labels=legend_labels,
                bbox_to_anchor=(1, 0.5))
        except Exception as e:
            logging.error('Cannot add legend for feature "{}" due to "{}"'.format(
                feature_label, e))
        # plt.legend(
        #     handles=legends[:1],
        #     loc='center left',
        #     bbox_to_anchor=(1, 0.5))

        if self.show_plots:
            plt.show()
        # plt.show()
        self.save_fig(fig, 'partial_dependence_{}.png'.format(
            feature_label))

        df = pd.DataFrame(
            [labels, deltas, deltas_unc,
             means, means_unc,
             counts, raw_stds],
            index=[
                feature_label,
                'partial dependence delta',
                'partial dependence delta uncertainty',
                'mean',
                'mean uncertainty',
                'sample size',
                'standard deviation']).T.sort_values(
                    by='partial dependence delta')
        df.to_excel(
            self.destination_dir
            + 'partial_dependence_with_stats_{}.xlsx'.format(
                feature_label))

    def add_plot_margins(self, lim):
        amin, amax = lim
        a_range = amax - amin
        return (amin - a_range * self.plot_margin,
                amax + a_range * self.plot_margin)

    def mean_uncertainty(self, data, weights=None):
        if weights is None:
            return np.std(data) / np.sqrt(len(data))
        else:
            return pde.std(data, weights=weights) / np.sqrt(np.sum(weights))

    def add_overlay_data(
            self, ax, raw_x, n_raw_datapoints, outcome_mean,
            overlay_box_plots=False,
            absolute_yscale=False,
            absolute_yticks=True):
        """Plots in given axis object the following:
            1) outcome means of buckets with mean uncertanties
            2) line connecting the means
            3) counts in each bucket
            4) optionally box plots of the buckets distributions
            """
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        locs = np.linspace(xlim[0], xlim[1], n_raw_datapoints)
        means = []
        means_unc = []
        means_x = []
        counts = []
        raw_data = []
        handles = []
        labels = []
        # means
        left = locs[0] - (locs[1] - locs[0]) / 2.
        for i in range(len(locs) - 1):
            right = locs[i] + (locs[i + 1] - locs[i]) / 2.
            idx = ((raw_x >= left) & (raw_x < right))
            left = right
            count = np.sum(idx)
            if count > 0:
                means_x.append(locs[i])
                # logging.debug(idx)
                # logging.debug(self.train_y)
                raw_y_bucket = self.train_y[idx]\
                    - outcome_mean * (1 - absolute_yticks)
                raw_data_weights = self.train_weights[idx]
                raw_data.append(raw_y_bucket)
                means.append(
                    np.average(raw_y_bucket, weights=raw_data_weights))
                means_unc.append(
                    self.mean_uncertainty(
                        raw_y_bucket,
                        weights=raw_data_weights))
                counts.append(int(np.sum(raw_data_weights)))
        plot_mean_x = np.array(means_x)
        plot_mean_y = np.array(means)
        ax.plot(
            plot_mean_x, plot_mean_y, alpha=0.1,
            color=self.means_color)
        error_bars_plot = ax.errorbar(
            plot_mean_x, plot_mean_y, yerr=np.array(means_unc), fmt='o',
            color=self.means_color)
        handles.append(error_bars_plot)
        labels.append('Raw averages with mean uncertainties')
        # box plots
        if overlay_box_plots:
            box_plots = self.box_plots_raw_data(raw_data, means_x)
            handles.append(box_plots)
            labels.append('Box plots for the buckets')
        counts_plot = self.overlay_counts_histogram(
            ax, means_x, counts, xlim, ylim)
        handles.append(counts_plot)
        labels.append('Counts')

        return handles, labels

    def overlay_counts_histogram(self, ax, means_x, counts, xlim, ylim):
        # counts
        if len(counts) == 0:
            return None
        counts_array = np.array(counts)
        max_counts = np.max(counts_array)
        histogram_height = 0.2 * (ylim[1] - ylim[0])
        counts_array = histogram_height * counts_array / max_counts
        c_max = np.argmax(counts_array)
        bar_width = 0.9 * (xlim[1] - xlim[0]) / (len(means_x) + 1)
        counts_bars = ax.bar(
            means_x, counts_array,
            alpha=0.5, bottom=ylim[0],
            width=bar_width,
            color=self.counts_color)
        ax.annotate(
            s=str(max_counts),
            xy=(means_x[c_max], ylim[0] + counts_array[c_max]),
            color='black')
        return counts_bars

    def models_sample(
            self, params):
        params2 = self.update_random_state(params)
        for i, train_X_cv_subset in enumerate(self.train_X_cv_subsets):
            cv_model = skl_e.GradientBoostingRegressor(**params2)
            # cv_model.fit(
            #     train_X_cv_subset,
            #     self.train_y_cv_subsets[i])
            cv_model.fit(
                train_X_cv_subset,
                self.train_y_cv_subsets[i],
                sample_weight=self.train_weights_cv_subsets[i])

            self.models.append(cv_model)

    def partial_dependency_uncertainty(self, features, grid, percentiles):
        # Returns standard deviation of partial dependency curve

        pdps_cv = np.zeros(
            (len(grid), len(self.models)),
            dtype=np.float64)
        # logging.debug('grid: {}; shape: {}'.format(grid, grid.shape))
        for i, cv_model in enumerate(self.models):
            pdps = skl_e_pd.partial_dependence(
                cv_model, features, grid,
                percentiles=percentiles)
            pdps_cv[:, i] = pdps[0]

        stds = np.std(pdps_cv, axis=1)
        return stds

    def plot_partial_dependence_with_unc(
            self, gbm, feature_idx, percentiles=(0.05, 0.95),
            absolute_yscale=False,
            absolute_yticks=True,
            **fig_params):
        outcome_mean = np.mean(self.train_y)
        fig = plt.figure(**fig_params)
        ax = fig.add_subplot(1, 1, 1)

        pdps, axes = skl_e_pd.partial_dependence(
            gbm, [feature_idx], X=self.train_X,
            percentiles=percentiles)

        if absolute_yscale or absolute_yticks:
            pdps = pdps + outcome_mean

                # plt.xticks(locs, labels)
        stds = self.partial_dependency_uncertainty(
            [feature_idx],
            grid=axes[0],
            percentiles=percentiles)

        pdp_uncertainty_plot = ax.fill_between(
            axes[0], pdps[0] - stds, pdps[0] + stds,
            alpha=0.2,
            color=self.pdp_color)
        pdp_plot, = ax.plot(
            axes[0], pdps[0], lw=5,
            color=self.pdp_color)
        if absolute_yscale:
            c_ylim = ax.get_ylim()
            ax.set_ylim(0, c_ylim[1])
        # if offset_mean and not offset_mean_labels:
        #     # fig.canvas.draw()
        #     ax.set_yticklabels(ax.get_yticks())
        #     labels_both = ax.get_yticklabels(which='both')

        #     for l in labels_both:
        #         l.set_text('{:.2f}'.format(outcome_mean + float(l.get_text())))

        #     ax.set_yticklabels(labels_both)

        return fig, ax, pdp_plot, pdp_uncertainty_plot

    def plot_numeric_partial_dependencies(
            self, gbm, feature_label, percentiles,
            ax_limits_per_feature,
            n_raw_datapoints,
            absolute_yscale=False,
            absolute_yticks=True,
            **fig_params):
        outcome_mean = np.mean(self.train_y)
        handles = []
        labels = []
        feature_idx = self.feature_index_by_label[feature_label]
        fig, ax, pdp_plot, pdp_uncertainty_plot = \
            self.plot_partial_dependence_with_unc(
                gbm, feature_idx, percentiles=percentiles,
                absolute_yscale=absolute_yscale,
                absolute_yticks=absolute_yticks,
                **fig_params)
        handles = handles + [pdp_plot, pdp_uncertainty_plot]
        labels = labels + ['Partial dependence', '67% confidence interval']
        ylim = ax_limits_per_feature.get(
            'ylim', self.add_plot_margins(ax.get_ylim()))
        xlim = ax_limits_per_feature.get(
            'xlim', self.add_plot_margins(ax.get_xlim()))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(alpha=0.3)

        raw_feature = self.train_X[[self.feature_labels[feature_idx]]]
        raw_outcome = self.train_y - outcome_mean
        try:
            overlay_handles, overlay_labels = self.add_overlay_data(
                ax,
                raw_feature.iloc[:, 0],
                n_raw_datapoints,
                outcome_mean,
                absolute_yscale=absolute_yscale,
                absolute_yticks=absolute_yticks)
            handles = handles + overlay_handles
            labels = labels + overlay_labels
            scatter_plot = ax.scatter(
                raw_feature, raw_outcome, alpha=0.05,
                color=self.data_color)
            handles.append(scatter_plot)
            labels.append('Raw data')
        except Exception as e:
            logging.error('cannot overlay data due to {}'.format(e))
        # locs = axs[0].get_xticks()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_ylabel(self.outcome_ylabel(absolute_yticks))
        ax.set_title('Partial dependence on {} ({} trees)'.format(
            feature_label, gbm.n_estimators))
        ax.legend(
            handles=handles,
            labels=labels,
            bbox_to_anchor=(1, 0.5))
        self.save_fig(fig, 'partial_dependence_{}.png'.format(
            feature_label))

    def partial_dependencies_plots(
            self,
            gbm,
            gbm_params,
            n_raw_datapoints=10,
            ax_limits=None,
            percentiles=(0.05, 0.95),
            overlay_box_plots=False,
            absolute_yscale=False,
            absolute_yticks=True,
            **fig_params):
        """Plots partial dependencies overlayed with other data.

        Arguments:
            ax_limits - dictionary of axes limits for each feature.
            Example: {'feature1':{'ylim':[-0.1, 0.1], xlim: [0, 100]}}"""
        if absolute_yscale:
            raise NameError('absolute_yscale=True is not implemented yet')
        feature_labels2, fis2 = mu.dict_to_lists(
            self.consolidate_feature_importance(gbm.feature_importances_),
            target_type=np.array)
        features_to_plot = np.argsort(-fis2)

        if ax_limits is None:
            ax_limits = {}
        self.models_sample(gbm_params)
        for feature in features_to_plot:
            feature_label = feature_labels2[feature]
            ax_limits_per_feature = ax_limits.get(feature_label, {})
            if feature_label in self.categorical_labels:
                self.partial_dependency_catagorical_plot(
                    gbm, feature_label,
                    ax_limits_per_feature=ax_limits_per_feature,
                    overlay_box_plots=overlay_box_plots,
                    absolute_yscale=absolute_yscale,
                    absolute_yticks=absolute_yticks,
                    **fig_params)
            else:
                self.plot_numeric_partial_dependencies(
                    gbm, feature_label, percentiles,
                    ax_limits_per_feature,
                    n_raw_datapoints,
                    absolute_yscale=absolute_yscale,
                    absolute_yticks=absolute_yticks,
                    **fig_params)
        try:
            first_feature = feature_labels2[features_to_plot[0]]
            second_feature = feature_labels2[features_to_plot[1]]
            f1 = self.feature_index_by_label.get(first_feature)
            f2 = self.feature_index_by_label.get(second_feature)
            if f1 is not None and f2 is not None:
                fig, axs = skl_e_pd.plot_partial_dependence(
                    gbm, self.train_X, [[f1, f2]],
                    feature_names=self.train_X.columns,
                    **fig_params)
                self.save_fig(fig, 'partial_dependence_{}_{}.png'.format(
                    first_feature,
                    second_feature))
        except Exception as e:
            logging.error(
                'cannot plot 2D partial dependence due to: {}'.format(e))
