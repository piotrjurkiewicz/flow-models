#!/usr/bin/python3
import argparse

import pandas as pd
import sklearn.ensemble
import sklearn.metrics
import sklearn.model_selection
import sklearn.neighbors
import sklearn.neural_network
import sklearn.preprocessing
import sklearn.svm
import sklearn.tree

from flow_models.elephants.ml import prepare_decision, load_arrays, make_slice, prepare_input, my_score
from flow_models.lib.util import logmsg

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-O', '--output', default='sklearn', help='output directory and plot file name')
    parser.add_argument('files', help='directory')
    app_args = parser.parse_args()

    data = load_arrays(app_args.files)

    clf_class = sklearn.ensemble.ExtraTreesRegressor
    hparams = {
        'n_estimators': [20],
        # 'criterion': ['mse', 'mae'],
        # 'max_depth': [None],
        # 'min_samples_split': [2, 4, 8, 16, 32],
        'min_samples_leaf': [2, 3],
        # 'oob_score': [True, False],
        # 'max_features': [*range(1, 14)],
        # 'max_leaf_nodes': [1024, 2048, 4096, None],
        'bootstrap': [False, True],
        # 'warm_start': [True, False],
    }

    data_par = {'skip': 0, 'count': 2000000}
    prep_par = {'to_octets': True}

    all_data = make_slice(data, **data_par)
    all_inp, all_oc = prepare_input(all_data, **prep_par)

    scoring = sklearn.metrics.make_scorer(my_score)

    name = f"{clf_class.__name__} {prep_par} {data_par}"
    logmsg(f"Starting {name}")

    if issubclass(clf_class, sklearn.base.ClassifierMixin):
        all_decision = prepare_decision(all_oc, 0.80)
        clf = clf_class()
        # TODO: Fix scoring here
        gsc = sklearn.model_selection.GridSearchCV(clf, hparams, scoring=scoring, cv=5, n_jobs=4, verbose=5)
        gsc.fit(all_inp, all_decision)
    elif issubclass(clf_class, sklearn.base.RegressorMixin):
        clf = clf_class()
        gsc = sklearn.model_selection.GridSearchCV(clf, hparams, scoring=scoring, cv=5, n_jobs=4, verbose=5)
        gsc.fit(all_inp, all_oc)
    else:
        raise NotImplementedError

    pd.DataFrame(gsc.cv_results_).drop('params', axis=1).to_html('sklearn.html')
    name = f"{clf_class.__name__} {gsc.best_params_} {prep_par} {data_par}"
    logmsg(f"Best: {gsc.best_score_} {name}")
    logmsg(f"Finished {name}")


if __name__ == '__main__':
    main()
