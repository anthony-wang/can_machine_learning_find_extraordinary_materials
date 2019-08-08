# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec


# from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve, KFold
from sklearn.metrics import confusion_matrix, auc, roc_curve,\
                            precision_recall_fscore_support


class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def plot_1d_grid_search(grid, midpoint=0.7):
    mean_test_scores = grid.cv_results_['mean_test_score']
    parameter_name = list(grid.cv_results_['params'][0].keys())[0]
    parameters = grid.cv_results_['param_'+parameter_name]

    plt.figure(figsize=(6, 6))
    plt.plot(list(parameters), list(mean_test_scores), 'k--')
    plt.tick_params(direction='in',
                    length=5,
                    bottom=True,
                    top=True,
                    left=True,
                    right=True)
    plt.xlabel(parameter_name)
    plt.ylabel('R2 score')
    plt.title('grid search')


def plot_2d_grid_search(grid, midpoint=0.7, vmin=0, vmax=1):
    cv_results = grid.cv_results_
    keys = list(cv_results.keys())
    parameters = [x[6:] for x in keys if 'param_' in x]

    param1 = list(set(cv_results['param_'+parameters[0]]))
    if parameters[1] == 'class_weight':
        param2 = list(set([d[1] for d in
                           cv_results['param_' + parameters[1]]]))
    else:
        param2 = list(set(cv_results['param_' + parameters[1]]))
    scores = cv_results['mean_test_score'].reshape(len(param1), len(param2))

    param1 = [round(param, 2) for param in param1]
    param2 = [round(param, 2) for param in param2]

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=midpoint))
    plt.xlabel(parameters[1])
    plt.ylabel(parameters[0])
    plt.tick_params(direction='in',
                    length=5,
                    bottom=True,
                    top=True,
                    left=True,
                    right=True)
    plt.colorbar()
    plt.xticks(np.arange(len(param2)), sorted(param2), rotation=90)
    plt.yticks(np.arange(len(param1)), sorted(param1))

    plt.title('grid search')


def plot_prob(threshold, y_act, y_prob, threshold_x, mat_prop='prop'):

    color = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

    y_act_labeled = [1 if x > threshold_x else 0 for x in y_act]
    y_pred_labeled = [1 if x >= threshold else 0 for x in y_prob]
    prfs = precision_recall_fscore_support(y_act_labeled, y_pred_labeled)
    precision, recall, fscore, support = prfs
    print('precision: {:0.2f}\nrecall: {:0.2f}'.format(precision[1],
                                                       recall[1]))

    tn, fp, fn, tp = confusion_matrix(y_act_labeled, y_pred_labeled).ravel() /\
        len(y_act_labeled) * 100

    plt.figure(1, figsize=(8, 8))
    left, width = 0.1, 0.65

    bottom, height = 0.1, 0.65
    bottom_h = left + width
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.15]

    ax1 = plt.axes(rect_histx)
    ax1.hist(y_act, bins=30, color='silver', edgecolor='k')
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = plt.axes(rect_scatter)

    rect1 = patches.Rectangle((threshold_x, -0.02),
                              6000,
                              threshold + 0.02,
                              linewidth=1,
                              edgecolor='k',
                              facecolor=color[3],
                              alpha=0.25,
                              label='false negative ({:0.0f}%)'.format(fn))
    rect2 = patches.Rectangle((-50, threshold),
                              threshold_x+50,
                              6000,
                              linewidth=1,
                              edgecolor='k',
                              facecolor=color[1],
                              alpha=0.25,
                              label='false postive ({:0.0f}%)'.format(fp))
    rect3 = patches.Rectangle((threshold_x, threshold),
                              6000,
                              6000,
                              linewidth=1,
                              edgecolor='k',
                              facecolor=color[2],
                              alpha=0.25,
                              label='true positive ({:0.0f}%)'.format(tp))
    rect4 = patches.Rectangle((-50, -50),
                              threshold_x+50,
                              threshold+50,
                              linewidth=1,
                              edgecolor='k',
                              facecolor='w',
                              alpha=0.25,
                              label='true negative ({:0.0f}%)'.format(tn))
    ax2.add_patch(rect1)
    ax2.add_patch(rect2)
    ax2.add_patch(rect3)
    ax2.add_patch(rect4)

    plt.tick_params(direction='in',
                    length=5,
                    bottom=True,
                    top=True,
                    left=True,
                    right=True)

    ax2.plot(y_act, y_prob, 'o', mfc='#C0C0C0', alpha=0.5,
             mec='#2F4F4F', mew=1.3)
    ax2.plot([-10, 600], [threshold, threshold], 'k--',
             label='threshold', linewidth=3)
    ax2.plot([threshold_x, threshold_x], [-1, 2], 'k:', linewidth=3)

    ax2.set_ylabel('Probability of being extraordinary'.title())
    ax2.set_xlabel(mat_prop)
    
    
    x_range = max(y_act) - min(y_act)
    ax2.set_xlim(max(y_act) - x_range*1.05, max(y_act)*1.05)
#    ax2.set_xlim(min(y_act)*1.05, max(y_act)*1.05)
    ax2.set_ylim(-.02, 1)
    ax1.set_xlim(ax2.get_xlim())
    ax1.axis('off')
    plt.legend(loc=2, framealpha=0.25)


def plot_regression(threshold, y_act, y_pred, threshold_x, mat_prop='prop'):
    color = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

    y_act_labeled = [1 if x > threshold_x else 0 for x in y_act]
    y_pred_labeled = [1 if x >= threshold else 0 for x in y_pred]
    prfs = precision_recall_fscore_support(y_act_labeled, y_pred_labeled)
    precision, recall, fscore, support = prfs

    print('precision: {:0.2f}\nrecall: {:0.2f}'.format(precision[1],
                                                       recall[1]))

    tn, fp, fn, tp = confusion_matrix(y_act_labeled, y_pred_labeled).ravel() /\
        len(y_act_labeled) * 100

    plt.figure(1, figsize=(8, 8))
    left, width = 0.1, 0.65

    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.15]
    rect_histy = [left_h, bottom, 0.15, height]

    ax1 = plt.axes(rect_histx)
    ax1.hist(y_act, bins=31, color='silver', edgecolor='k')
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax3 = plt.axes(rect_histy)
    ax3.hist(y_pred,
             bins=31,
             color='silver',
             edgecolor='k',
             orientation='horizontal')
    ax3.set_xticks([])
    ax3.set_yticks([])

    ax2 = plt.axes(rect_scatter)

    rect1 = patches.Rectangle((threshold_x, -100),
                              600,
                              threshold + 100,
                              linewidth=1,
                              edgecolor='k',
                              facecolor=color[3],
                              alpha=0.25,
                              label='false negative ({:0.0f}%)'.format(fn))
    rect2 = patches.Rectangle((-50, threshold),
                              threshold_x+50,
                              600,
                              linewidth=1,
                              edgecolor='k',
                              facecolor=color[1],
                              alpha=0.25,
                              label='false postive ({:0.0f}%)'.format(fp))
    rect3 = patches.Rectangle((threshold_x, threshold),
                              600,
                              600,
                              linewidth=1,
                              edgecolor='k',
                              facecolor=color[2],
                              alpha=0.25,
                              label='true positive ({:0.0f}%)'.format(tp))
    rect4 = patches.Rectangle((-50, -50),
                              threshold_x+50,
                              threshold+50,
                              linewidth=1,
                              edgecolor='k',
                              facecolor='w',
                              alpha=0.25,
                              label='true negative ({:0.0f}%)'.format(tn))
    ax2.add_patch(rect1)
    ax2.add_patch(rect2)
    ax2.add_patch(rect3)
    ax2.add_patch(rect4)

    ax2.tick_params(direction='in',
                    length=5,
                    bottom=True,
                    top=True,
                    left=True,
                    right=True)

    ax2.plot(y_act, y_pred, 'o', mfc='#C0C0C0', alpha=0.5, label=None,
             mec='#2F4F4F', mew=1.3)
    ax2.plot([-10, 600], [threshold, threshold], 'k--',
             label='threshold', linewidth=3)
    ax2.plot([threshold_x, threshold_x], [-100, 20000], 'k:', linewidth=3)

    ax2.set_ylabel('Predicted '+mat_prop)
    ax2.set_xlabel(mat_prop)
    x_range = max(y_act) - min(y_act)
    ax2.set_xlim(max(y_act) - x_range*1.05, max(y_act)*1.05)
    ax2.set_ylim(max(y_act) - x_range*1.05, max(y_act)*1.05)
#    if max(y_act) < 0 and min(y_act) < 0:
#        ax2.set_xlim(min(y_act)*1.1, max(y_act)/1.05)
#        ax2.set_ylim(min(y_act)*1.1, max(y_act)/1.05)
    ax1.set_xlim(ax2.get_xlim())
    ax3.set_ylim(ax2.get_ylim())
    ax1.axis('off')
    ax3.axis('off')
    ax2.legend(loc=2, framealpha=0.25)


def plot_log_reg_grid_search(parameter_candidates, roc_means, fscore_means):
    parameters = list(parameter_candidates.values())
    param1, param2 = parameters[0], parameters[1]

    param1 = [float('{:.2f}'.format(x)) for x in param1]
    param2 = [float('{:.0f}'.format(x)) for x in param2]

    gs = gridspec.GridSpec(1, 2)
    gs.update(wspace=0.25, hspace=0.15)

    plt.tick_params(direction='in',
                    length=5,
                    bottom=True,
                    top=True,
                    left=True,
                    right=True)

    data = np.array(roc_means).reshape(len(param1), len(param2))

    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(data, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=96, vmax=100, midpoint=98))
    plt.xlabel('class_weight')
    plt.ylabel('C')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(np.arange(len(param2)), sorted(param2))
    plt.yticks(np.arange(len(param1)), sorted(param1))
    plt.title('AUC ROC')

    plt.tick_params(direction='in',
                    length=5,
                    bottom=True,
                    top=True,
                    left=True,
                    right=True)

    data = np.array(fscore_means).reshape(len(param1), len(param2))
    plt.subplots_adjust(left=0, right=0.99, bottom=0.15, top=0.95)
    plt.imshow(data, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=35, vmax=55, midpoint=45))
    plt.xlabel('class_weight')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(np.arange(len(param2)), sorted(param2))
    plt.yticks(np.arange(len(param1)), sorted(param1))
    plt.title('F score')


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.
    inspired by:
        https://scikit-learn.org/stable/auto_examples/model_selection/
                plot_learning_curve.html#sphx-glr-auto-examples-model-s
                election-plot-learning-curve-py

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure(figsize=(6, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.tick_params(direction='in',
                    length=5,
                    bottom=True,
                    top=True,
                    left=True,
                    right=True)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")


def plot_act_vs_pred(y_actual, y_predicted):
    plt.figure(figsize=(6, 6))
    plt.plot(y_actual, y_predicted, marker='o', mfc='none',
             color='#0077be', linestyle='none')
    min_min = min([min(y_actual), min(y_predicted)])
    max_max = max([max(y_actual), max(y_predicted)])
    plt.plot([min_min, max_max], [min_min, max_max], 'k--')
    plt.title("actual versus predicted values")
    plt.tick_params(direction='in',
                    length=5,
                    bottom=True,
                    top=True,
                    left=True,
                    right=True)
    limits = [min_min, max_max]
    plt.xlim(limits)
    plt.ylim(limits)
    plt.xlabel('actual')
    plt.ylabel('predicted')


def get_roc_auc(actual, probability, plot=False):
    fpr, tpr, tttt = roc_curve(actual, probability, pos_label=1)
    roc_auc = auc(fpr, tpr)
    if plot is True:
        plt.figure(2, figsize=(6, 6))
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")

        plt.tick_params(direction='in',
                        length=5,
                        bottom=True,
                        top=True,
                        left=True,
                        right=True)

    return roc_auc


def get_performance_metrics(actual, predicted, probability, plot=False):
    tn, fp, fn, tp = confusion_matrix(actual, predicted).ravel()
    roc_auc = get_roc_auc(actual, probability, plot=plot) * 100
    recall = tp / (fn+tp) * 100
    precision = tp / (tp+fp) * 100
    fscore = 2 * (recall * precision) / (recall + precision)
    return fscore, roc_auc


def log_reg_grid_search(X, y, parameter_candidates, n_cv=3):
    kf = KFold(n_splits=n_cv, shuffle=True, random_state=1)
    fscore_dict = {}
    roc_dict = {}
    fscore_means = {}
    roc_means = {}
    X_gs = X.reset_index(drop=True)
    y_gs = y.reset_index(drop=True)
    i = 0

    for val_0 in parameter_candidates['C']:
        for val_1 in parameter_candidates['class_weight']:
            model = LogisticRegression(C=val_0, class_weight={0: 1, 1: val_1})
            fscore_list = []
            roc_area_list = []
            i += 1

            for train_index, test_index in kf.split(X):
                X_train, X_test = X_gs.iloc[train_index], X_gs.iloc[test_index]
                y_train, y_test = y_gs.iloc[train_index], y_gs.iloc[test_index]
                model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)
                y_test_prob_both = model.predict_proba(X_test)
                y_test_prob = [probability[1] for probability in
                               y_test_prob_both]
                fscore, roc_area = get_performance_metrics(y_test,
                                                           y_test_pred,
                                                           y_test_prob)
                fscore_list.append(fscore)
                roc_area_list.append(roc_area)
            fscore_dict[(val_0, val_1)] = fscore_list
            roc_dict[(val_0, val_1)] = roc_area_list
            fscore_means[(val_0, val_1)] = np.array(fscore_list).mean()
            roc_means[(val_0, val_1)] = np.array(roc_area_list).mean()

    return fscore_dict, roc_dict
