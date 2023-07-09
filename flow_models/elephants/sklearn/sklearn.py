#!/usr/bin/python3
import argparse
import collections
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import sklearn.ensemble
import sklearn.neighbors
import sklearn.neural_network
import sklearn.preprocessing
import sklearn.svm
import sklearn.tree

from flow_models.elephants.ml import prepare_decision, load_arrays, make_slices, prepare_input, stats, plot, calculate_from_mixture
from flow_models.lib.util import logmsg

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fork', action='store_true', help='')
    parser.add_argument('files', help='directory')
    app_args = parser.parse_args()

    data = load_arrays(app_args.files)

    results = {}

    results['Mixture'] = np.array([*calculate_from_mixture()])
    results['Data (all)'] = stats(data[-1], data[-1])

    algos = [
        # (sklearn.tree.DecisionTreeClassifier, {}),
        # (sklearn.ensemble.RandomForestClassifier, {'n_estimators': 10}),
        # (sklearn.ensemble.ExtraTreesClassifier, {}),
        # (sklearn.ensemble.AdaBoostClassifier, {}),
        # (sklearn.ensemble.GradientBoostingClassifier, {}),
        # (sklearn.neighbors.KNeighborsClassifier, {}),
        # (sklearn.neighbors.RadiusNeighborsClassifier, {}),
        # (sklearn.svm.SVM, {}),
        # (sklearn.neural_network.MLPRegressor, {}),
        # (sklearn.neural_network.MLPRegressor, {'solver':'lbfgs'}),
        # (sklearn.tree.DecisionTreeRegressor, {}),
        # (sklearn.ensemble.RandomForestRegressor, {'n_estimators': 100}),
        # (sklearn.ensemble.RandomForestRegressor, {'n_estimators': 50}),
        # (sklearn.ensemble.RandomForestRegressor, {'n_estimators': 20}),
        # (sklearn.ensemble.RandomForestRegressor, {'n_estimators': 10}),
        # (sklearn.ensemble.RandomForestRegressor, {'n_estimators': 100, 'criterion':'mae'}),
        # (sklearn.ensemble.RandomForestRegressor, {'n_estimators': 50, 'criterion':'mae'}),
        # (sklearn.ensemble.RandomForestRegressor, {'n_estimators': 20, 'criterion':'mae'}),
        # (sklearn.ensemble.RandomForestRegressor, {'n_estimators': 10, 'criterion':'mae'}),
        (sklearn.ensemble.ExtraTreesRegressor, {'n_estimators': 10}),
        # (sklearn.ensemble.AdaBoostRegressor, {}),
        # (sklearn.ensemble.GradientBoostingRegressor, {}),
        # (sklearn.neighbors.KNeighborsRegressor, {}),
        # (sklearn.neighbors.RadiusNeighborsRegressor, {}),
        # (sklearn.svm.SVR, {}),
        # (sklearn.neural_network.MLPRegressor, {}),
        # (sklearn.neural_network.MLPRegressor, {'solver':'lbfgs'}),
    ]

    # data_params = [{'skip': 0, 'count': 1000000}, {'skip': 1000000, 'count': 1000000}, {'skip': 2000000, 'count': 1000000}]
    data_params = [{'skip': 0, 'count': 1000000}]
    prep_params = [{'to_octets': True}]
    modes = {'test'}

    for data_par in data_params:
        all_data, train_data, test_data = make_slices(data, **data_par)
        results[f'Data {data_par} (all)'] = stats(all_data[-1], all_data[-1])
        results[f'Data {data_par} (train)'] = stats(train_data[-1], train_data[-1])
        results[f'Data {data_par} (test)'] = stats(test_data[-1], test_data[-1])
        for prep_par in prep_params:
            train_inp, train_oc = prepare_input(train_data, **prep_par)
            test_inp, test_oc = prepare_input(test_data, **prep_par)
            for clf_class, clf_params in algos:
                if app_args.fork:
                    pid = os.fork()
                else:
                    pid = None
                if not pid:
                    if pid == 0:
                        results = {}
                    name = f"{clf_class.__name__} {clf_params} {prep_par} {data_par}"
                    try:
                        logmsg(f"Starting {name}")
                        if issubclass(clf_class, sklearn.base.ClassifierMixin):
                            r = collections.defaultdict(list)
                            for coverage in 1 - 1 / np.power(2, range(26))[1:]:
                                clf = clf_class(**clf_params)
                                train_decision = prepare_decision(train_oc, coverage)
                                clf.fit(train_inp, train_decision)
                                for mode in modes:
                                    inp = locals()[f'{mode}_inp']
                                    oc = locals()[f'{mode}_oc']
                                    decision_predicted = clf.predict(inp)
                                    r[mode].append([coverage, len(oc) / decision_predicted.sum(), oc[decision_predicted].sum() / oc.sum()])
                            for mode in modes:
                                results[f'{name} ({mode})'] = np.array(r[mode]).T
                        elif issubclass(clf_class, sklearn.base.RegressorMixin):
                            clf = clf_class(**clf_params)
                            clf.fit(train_inp, train_oc)
                            for mode in modes:
                                inp = locals()[f'{mode}_inp']
                                oc = locals()[f'{mode}_oc']
                                oc_predicted = clf.predict(inp)
                                results[f'{name} ({mode})'] = stats(oc, oc_predicted, 64)
                        else:
                            raise NotImplementedError
                        logmsg(f"Finished {name}")
                    except Exception as exc:
                        logmsg(f"Exception {name}")
                        raise exc

                    if pid == 0:
                        for key, val in results.items():
                            np.savetxt(f"sklearn/{key}.tsv", val)
                        exit()

    if app_args.fork:
        try:
            while os.wait():
                pass
        except ChildProcessError:
            pass
        for f in pathlib.Path('./sklearn/').glob('*.tsv'):
            results[f.stem] = np.loadtxt(str(f))

    plot(results, f'{app_args.files}')
    plt.legend(fontsize='xx-small')
    # plt.show()
    plt.savefig(f'sklearn.pdf')


if __name__ == '__main__':
    main()
