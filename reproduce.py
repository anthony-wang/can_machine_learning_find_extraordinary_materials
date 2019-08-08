# import important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import os
import argparse

# import machine learning libraries
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict, GridSearchCV, KFold
from sklearn.metrics import confusion_matrix, classification_report,\
                                              precision_recall_fscore_support

# import custom functions for vectorizing & visualizing data
import utils
from process_data import get_split
plt.rcParams.update({'font.size': 12})


all_props = ['bulk_modulus',
             'thermal_conductivity',
             'shear_modulus',
             'band_gap',
             'debye_temperature',
             'thermal_expansion']

symbols = ['B', '$\\kappa$', 'G', 'E$_{\\mathrm{g}}$', 'D', '$\\alpha$']

prop_labels = ['Bulk Modulus (GPa)',
               'Log$_{10}$ Thermal Conductivity $\\left(\\dfrac{\\mathrm{W}}' +
               '{\\mathrm{m}\\cdot \\mathrm{K}}\\right)$',
               'Shear Modulus (GPa)',
               'Band Gap (eV)',
               'Log$_{10}$ Debye Temperature (K)',
               'Log$_{10}$ Thermal Expansion $(\\mathrm{K}^{-1})$']

arg2prop = {'bulk_modulus': 'ael_bulk_modulus_vrh',
            'thermal_conductivity': 'agl_log10_thermal_conductivity_300K',
            'shear_modulus': 'ael_shear_modulus_vrh',
            'band_gap': 'Egap',
            'debye_temperature': 'ael_log10_debye_temperature',
            'thermal_expansion': 'agl_log10_thermal_expansion_300K'}

prop2label = dict([[v, k] for k, v
                  in zip(prop_labels, arg2prop.values())])

parser_desc = 'Reproduce the results of this work'
parser = argparse.ArgumentParser(description=parser_desc)
group = parser.add_mutually_exclusive_group(required=True)

group.add_argument('--properties',
                   type=str,
                   nargs='+',
                   metavar='Property to reproduce',
                   choices=all_props,
                   help=('example:\n\t' +
                         'python reproduce.py --properties bulk_modulus\n\t'))

group.add_argument('--all',
                   action='store_true',
                   help='Run through each property one at a time '
                   'and generate results and figures.')

args = parser.parse_args()

if not args.all:
    mat_props = []
    for j in args.properties:
        mat_props.append(arg2prop[j])
else:
    mat_props = list(map(lambda p: arg2prop[p], all_props))
print('Reproducing results for the following data:', mat_props)


def optimize_threshold(y_train_labeled, y_train_pred):
    """Given a DataFrame of labels and predictions, return the
     optimal threshold for a high F1 score"""
    y_train_ = y_train_labeled.copy()
    y_train_pred_ = pd.Series(y_train_pred).copy()
    f1score_max = 0
    for threshold in np.arange(0.1, 1, 0.1):
        diff = (max(y_train_pred) - min(y_train_pred))
        threshold = min(y_train_pred) + threshold * diff
        y_train_pred_[y_train_pred < threshold] = 0
        y_train_pred_[y_train_pred >= threshold] = 1
        f1score = f1_score(y_train_, y_train_pred_)
        if f1score > f1score_max:
            f1score_max = f1score
            opt_thresh = threshold
    return opt_thresh


def get_performance(mat_props, seed):
    metrics_dict = {}
    for mat_prop in mat_props:
        os.makedirs('figures/'+mat_prop, exist_ok=True)
        data = get_split(mat_prop, elem_prop='oliynyk', seed_num=seed)
        X_train_scaled, X_test_scaled = data[0:2]
        y_train, y_test = data[2:4]
        y_train_labeled, y_test_labeled = data[4:6]
        formula_train, formula_test = data[6:8]

        test_threshold = y_test.iloc[-y_test_labeled.sum().astype(int)]
        train_threshold = y_train.iloc[-y_train_labeled.sum().astype(int)]

        y = pd.concat([y_train, y_test])

        plt.figure(1, figsize=(7, 7))
        ax = sns.distplot(y, bins=50, kde=False)

        rect1 = patches.Rectangle((test_threshold, 0),
                                  ax.get_xlim()[1]-test_threshold,
                                  ax.get_ylim()[1], linewidth=1,
                                  edgecolor='k',
                                  facecolor='g',
                                  alpha=0.2)
        ax.add_patch(rect1)

        text_size = 18

        ax.text(.1,
                .5,
                'Ordinary\nCompounds',
                size=text_size,
                horizontalalignment='left',
                verticalalignment='center',
                transform=ax.transAxes)
        ax.text(.98,
                .15,
                'Extraordinary\nCompounds',
                size=text_size,
                horizontalalignment='right',
                verticalalignment='center',
                transform=ax.transAxes)
        ax.tick_params(direction='in',
                       length=5,
                       bottom=True,
                       top=True,
                       left=True,
                       right=True,
                       labelsize=text_size)
        ax.set_xlabel(prop2label[mat_prop], size=text_size)
        ax.set_ylabel('number of occurrences'.title(), size=text_size)
        plt.savefig('figures/' + mat_prop + '/distplot',
                    dpi=300,
                    bbox_inches='tight')
        plt.clf()
        # ## Learn with a Ridge Regression (linear model)
        # define ridge regression object
        rr = Ridge()
        # define k-folds
        cv = KFold(n_splits=5, shuffle=True, random_state=1)
        # choose search space
        parameter_candidates = {'alpha': np.logspace(-5, 2, 10)}

        # define the grid search
        grid = GridSearchCV(estimator=rr,
                            param_grid=parameter_candidates,
                            cv=cv)
        # run grid search
        grid.fit(X_train_scaled, y_train)

        # plot grid search to ensure good values)
        utils.plot_1d_grid_search(grid, midpoint=0.75)
        print('best parameters:', grid.best_params_)
        plt.savefig('figures/' + mat_prop + '/rr_1d_search',
                    dpi=300,
                    bbox_inches='tight')
        plt.clf()
        best_params_rr = grid.best_params_

    #    best_params_rr = {'alpha': 0.0021544346900318843}
        rr = Ridge(**best_params_rr)
        rr.fit(X_train_scaled, y_train)
        y_test_predicted_rr = rr.predict(X_test_scaled)
        y_train_predicted_rr = rr.predict(X_train_scaled)
        # plot the data
        plt.figure(figsize=(6, 6))
        plt.plot(y_test,
                 y_test_predicted_rr,
                 marker='o',
                 mfc='none',
                 color='#0077be',
                 linestyle='none',
                 label='test')
        plt.plot(y_train,
                 y_train_predicted_rr,
                 marker='o',
                 mfc='none',
                 color='#e34234',
                 linestyle='none',
                 label='train')
        max_val = max(y_test.max(), y_test_predicted_rr.max())
        min_val = min(y_test.min(), y_test_predicted_rr.min())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--')
        limits = [min_val, max_val]
        plt.xlim(limits)
        plt.ylim(limits)
        plt.xlabel('actual')
        plt.ylabel('predicted')
        plt.legend(loc=4)
        plt.tick_params(direction='in',
                        length=5,
                        bottom=True,
                        top=True,
                        left=True,
                        right=True)
        plt.savefig('figures/' + mat_prop + '/rr_act_vs_pred',
                    dpi=300,
                    bbox_inches='tight')
        plt.clf()

        # ## Learn with a support vector regression (non-linear model)

        # to speed up the grid search, optimize on a subsample of data
        X_train_scaled_sampled = X_train_scaled.sample(500, random_state=1)
        y_train_sampled = y_train.loc[X_train_scaled_sampled.index.values]

        # define support vector regression object (default to rbf kernel)
        svr = SVR()
        # define k-folds
        cv = KFold(n_splits=5, shuffle=True, random_state=1)
        # choose search space
        parameter_candidates = {'C': np.logspace(2, 4, 8),
                                'gamma': np.logspace(-3, 1, 8)}

        # define the grid search
        grid = GridSearchCV(estimator=svr,
                            param_grid=parameter_candidates,
                            cv=cv)
        # run grid search
        grid.fit(X_train_scaled_sampled, y_train_sampled)

        # plot grid search to ensure good values
        utils.plot_2d_grid_search(grid, midpoint=0.7)
        plt.savefig('figures/' + mat_prop + '/svr_2d_search',
                    dpi=300, bbox_inches='tight')
        plt.clf()
        print('best parameters:', grid.best_params_)
        best_params_svr = grid.best_params_

        svr = SVR(**best_params_svr)
        svr.fit(X_train_scaled, y_train)

        y_test_predicted_svr = svr.predict(X_test_scaled)
        y_train_predicted_svr = svr.predict(X_train_scaled)

        # plot the data
        plt.figure(figsize=(6, 6))
        plt.plot(y_test,
                 y_test_predicted_svr,
                 marker='o',
                 mfc='none',
                 color='#0077be',
                 linestyle='none',
                 label='test')
        plt.plot(y_train,
                 y_train_predicted_svr,
                 marker='o',
                 mfc='none',
                 color='#e34234',
                 linestyle='none',
                 label='train')

        max_val = max(y_test.max(), y_test_predicted_svr.max())
        min_val = min(y_test.min(), y_test_predicted_svr.min())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--')
        limits = [min_val, max_val]
        plt.xlim(limits)
        plt.ylim(limits)

        plt.xlabel('actual')
        plt.ylabel('predicted')
        plt.legend(loc=4)
        plt.tick_params(direction='in',
                        length=5,
                        bottom=True,
                        top=True,
                        left=True,
                        right=True)
        plt.savefig('figures/' + mat_prop + '/svr_act_vs_pred',
                    dpi=300,
                    bbox_inches='tight')
        plt.clf()

        # # Approach the problem as a classification task
        # ## Learn with a logistic regression (linear classification)
        # define logistic regression object
        lr = LogisticRegression(solver='lbfgs')
        # define k-folds
        cv = KFold(n_splits=5, shuffle=True, random_state=1)

        # choose search space
        class_1_weight = [{0: 1, 1: weight} for weight in
                          np.linspace(1, 50, 5)]
        parameter_candidates = {'C': np.logspace(-1, 4, 5),
                                'class_weight': class_1_weight}

        # define the grid search. We use log-loss to decide which
        # parameters to use.
        grid = GridSearchCV(estimator=lr,
                            param_grid=parameter_candidates,
                            scoring='neg_log_loss',
                            cv=cv)

        # run grid search
        grid.fit(X_train_scaled, y_train_labeled)

        # plot grid search to ensure good values
        utils.plot_2d_grid_search(grid, midpoint=-0.05, vmin=-0.13, vmax=0)
        plt.savefig('figures/' + mat_prop + '/lr_2d_search',
                    dpi=300,
                    bbox_inches='tight')
        plt.clf()
        print('best parameters:', grid.best_params_)
        best_params_lr = grid.best_params_

        lr = LogisticRegression(solver='lbfgs', penalty='l2', **best_params_lr)
        lr.fit(X_train_scaled, y_train_labeled)

        # define k-folds
        cv = KFold(n_splits=5, shuffle=True, random_state=1)

        y_pred_train_lr = cross_val_predict(lr,
                                            X_train_scaled,
                                            y_train_labeled,
                                            cv=cv)
        y_prob_train_lr = cross_val_predict(lr,
                                            X_train_scaled,
                                            y_train_labeled,
                                            cv=cv,
                                            method='predict_proba')
        y_probability_train_lr = [probability[1] for probability in
                                  y_prob_train_lr]

        y_prob_test_lr = lr.predict_proba(X_test_scaled)
        y_probability_test_lr = [probability[1] for probability in
                                 y_prob_test_lr]

        df_cm = pd.DataFrame(confusion_matrix(y_train_labeled,
                                              y_pred_train_lr))

        ax = sns.heatmap(df_cm,
                         square=True,
                         annot=True,
                         annot_kws={"size": 18},
                         cbar=False,
                         linewidths=.5,
                         cmap="YlGnBu",
                         center=-10000000)

        ax.set_ylabel('actual')
        ax.set_xlabel('predicted')
        ax.xaxis.tick_top()
        plt.savefig('figures/' + mat_prop + '/lr_cm',
                    dpi=300,
                    bbox_inches='tight')
        plt.clf()

        threshold = 0.5
        utils.plot_prob(threshold,
                        y_train,
                        y_probability_train_lr,
                        threshold_x=train_threshold,
                        mat_prop=prop2label[mat_prop])
        plt.savefig('figures/' + mat_prop +
                    '/lr_train_prob_thresh={:0.2f}.png'.format(threshold),
                    dpi=300,
                    bbox_inches='tight')
        plt.clf()

        # ### Check our perfromance on the test set!

        utils.plot_prob(threshold,
                        y_test,
                        y_probability_test_lr,
                        threshold_x=test_threshold,
                        mat_prop=prop2label[mat_prop])
        plt.savefig('figures/' + mat_prop +
                    '/lr_test_prob_thresh={:0.2f}.png'.format(threshold),
                    dpi=300,
                    bbox_inches='tight')
        plt.clf()

        # ### Compare this performance to regression models
        #
        # **For the same recall, we are three times more likely that predicted
        #   compound is not actually extraordinary.**

        threshold = optimize_threshold(y_train_labeled, y_train_predicted_rr)
        utils.plot_regression(threshold,
                              y_train,
                              y_train_predicted_rr,
                              threshold_x=train_threshold,
                              mat_prop=prop2label[mat_prop])
        plt.savefig('figures/' + mat_prop +
                    '/rr_train_reg_thresh={:0.2f}.png'.format(threshold),
                    dpi=300,
                    bbox_inches='tight')
        plt.clf()

        utils.plot_regression(threshold,
                              y_test,
                              y_test_predicted_rr,
                              threshold_x=test_threshold,
                              mat_prop=prop2label[mat_prop])
        plt.savefig('figures/' + mat_prop +
                    '/rr_test_reg_thresh={:0.2f}.png'.format(threshold),
                    dpi=300,
                    bbox_inches='tight')
        plt.clf()

        threshold = optimize_threshold(y_train_labeled, y_train_predicted_svr)
        utils.plot_regression(threshold,
                              y_train,
                              y_train_predicted_svr,
                              threshold_x=train_threshold,
                              mat_prop=prop2label[mat_prop])
        plt.savefig('figures/' + mat_prop +
                    '/svr_train_reg_thresh={:0.02f}.png'.format(threshold),
                    dpi=300,
                    bbox_inches='tight')
        plt.clf()

        utils.plot_regression(threshold,
                              y_test,
                              y_test_predicted_svr,
                              threshold_x=test_threshold,
                              mat_prop=prop2label[mat_prop])
        plt.savefig('figures/' + mat_prop +
                    '/svr_test_reg_thresh={:0.02f}.png'.format(threshold),
                    dpi=300,
                    bbox_inches='tight')
        plt.clf()

        # ## Learn with a support vector classification (non-linear)
        # to speed up the grid search, optimize on a subsample of data
        index_location = X_train_scaled_sampled.index.values
        y_train_labeled_sampled = y_train_labeled.loc[index_location]

        # define suppor vector classification object
        # (need to set probability to True)
        svc = SVC(probability=True)
        # define k-folds
        cv = KFold(n_splits=5, shuffle=True, random_state=1)

        # choose search space (we will start with class_weight=1
        # as that was optimal for svc)
        parameter_candidates = {'C': np.logspace(-1, 4, 5),
                                'gamma': np.logspace(-2, 2, 5)}

        # define the grid search. We use log-loss to decide
        # which parameters to use.
        grid = GridSearchCV(estimator=svc,
                            param_grid=parameter_candidates,
                            scoring='neg_log_loss',
                            cv=cv)

        # run grid search
        grid.fit(X_train_scaled_sampled, y_train_labeled_sampled)

        # plot grid search to ensure good values
        utils.plot_2d_grid_search(grid, midpoint=-0.04, vmin=-0.13, vmax=0)
        plt.savefig('figures/' + mat_prop +
                    '/svc_2d_search.png',
                    dpi=300,
                    bbox_inches='tight')
        plt.clf()
        print('best parameters:', grid.best_params_)
        best_params_svc = grid.best_params_

        svc = SVC(probability=True, **best_params_svc)
        svc.fit(X_train_scaled, y_train_labeled)

        cv = KFold(n_splits=5, shuffle=True, random_state=1)

        y_pred_train_svc = cross_val_predict(svc,
                                             X_train_scaled,
                                             y_train_labeled,
                                             cv=cv)
        y_prob_train_svc = cross_val_predict(svc,
                                             X_train_scaled,
                                             y_train_labeled,
                                             cv=cv,
                                             method='predict_proba')
        y_probability_train_svc = [probability[1] for probability in
                                   y_prob_train_svc]

        y_prob_test_svc = svc.predict_proba(X_test_scaled)
        y_probability_test_svc = [probability[1] for probability in
                                  y_prob_test_svc]

        metrics = precision_recall_fscore_support(y_train_labeled,
                                                  y_pred_train_svc)
        precision, recall, fscore, support = metrics
        print('precision: {:0.2f}\nrecall: {:0.2f}'.format(precision[1],
              recall[1]))
        df_cm = pd.DataFrame(confusion_matrix(y_train_labeled,
                                              y_pred_train_svc))

        ax = sns.heatmap(df_cm,
                         square=True,
                         annot=True,
                         annot_kws={"size": 18},
                         cbar=False,
                         linewidths=0.5,
                         cmap="YlGnBu",
                         center=-10000000)
        ax.set_ylabel('actual')
        ax.set_xlabel('predicted')
        ax.xaxis.tick_top()
        plt.savefig('figures/' + mat_prop +
                    '/svc_cm',
                    dpi=300,
                    bbox_inches='tight')
        plt.clf()

        threshold = 0.5
        utils.plot_prob(threshold,
                        y_train,
                        y_probability_train_svc,
                        threshold_x=train_threshold,
                        mat_prop=prop2label[mat_prop])
        plt.savefig('figures/' + mat_prop +
                    '/svc_train_prob_thresh={:0.02f}.png'.format(threshold),
                    dpi=300,
                    bbox_inches='tight')
        plt.clf()

        utils.plot_prob(threshold,
                        y_test,
                        y_probability_test_svc,
                        threshold_x=test_threshold,
                        mat_prop=prop2label[mat_prop])
        plt.savefig('figures/' + mat_prop +
                    '/svc_test_prob_thresh={:0.2f}.png'.format(threshold),
                    dpi=300,
                    bbox_inches='tight')
        plt.clf()

        metrics_dict[mat_prop] = {'precision': [], 'recall': [], 'f1': []}
        threshold = optimize_threshold(y_train_labeled, y_train_predicted_rr)
        y_pred_rr = [1 if x >= threshold else 0 for x in y_test_predicted_rr]
        threshold = optimize_threshold(y_train_labeled, y_train_predicted_svr)
        y_pred_svr = [1 if x >= threshold else 0 for x in y_test_predicted_svr]
        threshold = 0.5
        y_pred_lr = [1 if x >= threshold else 0 for x in y_probability_test_lr]
        threshold = 0.5
        y_pred_svc = [1 if x >= threshold else 0 for x in
                      y_probability_test_svc]

        predictions = [y_pred_rr,
                       y_pred_svr,
                       y_pred_lr,
                       y_pred_svc]

        for prediction in predictions:
            print(classification_report(y_test_labeled, prediction))
            metrics = precision_recall_fscore_support(y_test_labeled,
                                                      prediction)
            precision, recall, f1, support = metrics
            if precision[1] == 0:
                if precision == 10:
                    pass
            metrics_dict[mat_prop]['precision'].append(precision[1])
            metrics_dict[mat_prop]['recall'].append(recall[1])
            metrics_dict[mat_prop]['f1'].append(f1[1])
    return metrics_dict


def build_metrics():
    for seed in [1]:
        metrics = get_performance(mat_props, seed)
        for prop in metrics:
            metric_csv = prop+'_metrics_seed_{:0.0f}.csv'.format(seed)
            computed_metrics = os.listdir('data/metrics/')
            if metric_csv in computed_metrics:
                continue
            else:
                df_prop_metric = pd.DataFrame(metrics[prop],
                                              index=['rr', 'svr', 'lr', 'svc'])
                df_prop_metric.to_csv('data/metrics/'+metric_csv)


def plot_metrics():
    metric_mean = {}
    metric_std = {}
    metric_mean[0] = {}
    metric_mean[1] = {}
    metric_mean[2] = {}
    metric_std[0] = {}
    metric_std[1] = {}
    metric_std[2] = {}
    for prop in mat_props:
        rr = []
        svr = []
        lr = []
        svc = []
        for seed in [1, 2, 3, 4, 5]:
            metric_csv = prop+'_metrics_seed_{:0.0f}.csv'.format(seed)
            df_prop_metric = pd.read_csv('data/metrics/'+metric_csv)
            rr.append(df_prop_metric.iloc[0, 1:].tolist())
            svr.append(df_prop_metric.iloc[1, 1:].tolist())
            lr.append(df_prop_metric.iloc[2, 1:].tolist())
            svc.append(df_prop_metric.iloc[3, 1:].tolist())
        for i in [0, 1, 2]:
            metric_mean[i][prop] = [pd.DataFrame(rr).mean()[i],
                                    pd.DataFrame(svr).mean()[i],
                                    pd.DataFrame(lr).mean()[i],
                                    pd.DataFrame(svc).mean()[i]]
            metric_std[i][prop] = [pd.DataFrame(rr).std()[i],
                                   pd.DataFrame(svr).std()[i],
                                   pd.DataFrame(lr).std()[i],
                                   pd.DataFrame(svc).std()[i]]

    df_p_mean = pd.DataFrame(metric_mean[0], index=['rr', 'svr', 'lr', 'svc'])
    df_p_std = pd.DataFrame(metric_std[0], index=['rr', 'svr', 'lr', 'svc'])
    df_r_mean = pd.DataFrame(metric_mean[1], index=['rr', 'svr', 'lr', 'svc'])
    df_r_std = pd.DataFrame(metric_std[1], index=['rr', 'svr', 'lr', 'svc'])
    df_f_mean = pd.DataFrame(metric_mean[2], index=['rr', 'svr', 'lr', 'svc'])
    df_f_std = pd.DataFrame(metric_std[2], index=['rr', 'svr', 'lr', 'svc'])

    plt.rcParams.update({'font.size': 12})
    means = [df_p_mean, df_r_mean, df_f_mean]
    stds = [df_p_std, df_r_std, df_f_std]
    metric_type = ['Precision', 'Recall', 'F1-Score']
    i = 0
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=False, figsize=(7, 9))
    f.subplots_adjust(hspace=.05, wspace=1)
    axes = [ax1, ax2, ax3]
    colors = ['#d7191c', '#fdae61', '#abdda4', '#2b83ba']

    prop_loc = [1, 2, 3, 4, 5, 6]
    for df_mean, df_std, ax in zip(means, stds, axes):
        alpha = 0.15

        ax.fill_between(prop_loc,
                        df_mean.loc['rr']+df_std.loc['rr'],
                        df_mean.loc['rr']-df_std.loc['rr'],
                        color=colors[0],
                        alpha=alpha)
        ax.fill_between(prop_loc,
                        df_mean.loc['svr']+df_std.loc['svr'],
                        df_mean.loc['svr']-df_std.loc['svr'],
                        color=colors[3],
                        alpha=alpha)
        ax.fill_between(prop_loc,
                        df_mean.loc['lr']+df_std.loc['lr'],
                        df_mean.loc['lr']-df_std.loc['lr'],
                        color=colors[2],
                        alpha=alpha)
        ax.fill_between(prop_loc,
                        df_mean.loc['svc']+df_std.loc['svc'],
                        df_mean.loc['svc']-df_std.loc['svc'],
                        color=colors[1],
                        alpha=alpha)

        ax.plot(prop_loc, df_mean.loc['rr'], '-x', color=colors[0],
                linewidth=2, label='Ridge')
        ax.plot(prop_loc, df_mean.loc['svr'], '-s', color=colors[3],
                linewidth=2, label='SVR')
        ax.plot(prop_loc, df_mean.loc['lr'], '--d', color=colors[2],
                linewidth=3, label='Logistic')
        ax.plot(prop_loc, df_mean.loc['svc'], '--*', color=colors[1],
                linewidth=3, label='SVC')
        ax.set_ylabel(metric_type[i])
        ax.tick_params(top=True, right=True, direction='in', length=6)

        if i == 0:
            ax.set_ylim(0.25, 1)
            ax.xaxis.set_ticklabels([])
        if i == 1:
            ax.set_ylim(0.25, 1)
            ax.xaxis.set_ticklabels([])
            ax.legend(loc=4, fancybox=True, framealpha=0.3)
        if i == 2:
            ax.set_ylim(0.25, 1)
            ax.xaxis.set_ticklabels([])
        i += 1

    plt.xticks(np.array(prop_loc), labels=[prop for prop in symbols],
               rotation=0, ha='center', rotation_mode='anchor')

    plt.savefig('figures/metrics.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    build_metrics()
