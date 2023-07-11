#!/usr/bin/python3
import argparse
import collections
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import sklearn.ensemble
import sklearn.model_selection
import sklearn.neighbors
import sklearn.neural_network
import sklearn.preprocessing
import sklearn.svm
import sklearn.tree

from flow_models.elephants.ml import prepare_decision, load_arrays, make_slice, prepare_input, stats, plot, calculate_from_mixture, interp_red
from flow_models.lib.util import logmsg

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-O', '--output', default='sklearn', help='output directory and plot file name')
    parser.add_argument('--mixture', help='')
    parser.add_argument('--fork', action='store_true', help='')
    parser.add_argument('files', help='directory')
    app_args = parser.parse_args()

    data = load_arrays(app_args.files)

    results = {}

    if app_args.mixture:
        results['Mixture'] = np.array([*calculate_from_mixture(app_args.mixture)])

    results['Data (all)'] = stats(data[-1], data[-1])

    algos = [
        # (sklearn.tree.DecisionTreeClassifier, {}),
        # (sklearn.ensemble.RandomForestClassifier, {'n_estimators': 10}),
        # (sklearn.ensemble.ExtraTreesClassifier, {'n_estimators': 10}),
        # (sklearn.ensemble.AdaBoostClassifier, {}),
        # (sklearn.ensemble.GradientBoostingClassifier, {}),
        # (sklearn.neighbors.KNeighborsClassifier, {}),
        # (sklearn.neighbors.RadiusNeighborsClassifier, {}),
        # (sklearn.svm.SVM, {}),
        # (sklearn.neural_network.MLPRegressor, {}),
        # (sklearn.neural_network.MLPRegressor, {'solver':'lbfgs'}),
        # (sklearn.tree.DecisionTreeRegressor, {}),
        # (sklearn.ensemble.RandomForestRegressor, {'bootstrap': True}),
        (sklearn.ensemble.RandomForestRegressor, {'n_estimators': 10, 'bootstrap': True}),
        # (sklearn.ensemble.RandomForestRegressor, {'n_estimators': 10, 'min_samples_leaf': 2, 'bootstrap': False}),
        # (sklearn.ensemble.RandomForestRegressor, {'n_estimators': 10, 'min_samples_leaf': 2, 'bootstrap': True}),
        # (sklearn.ensemble.RandomForestRegressor, {'n_estimators': 20, 'min_samples_leaf': 2, 'bootstrap': False}),
        # (sklearn.ensemble.RandomForestRegressor, {'n_estimators': 20, 'min_samples_leaf': 2, 'bootstrap': True}),
        # (sklearn.ensemble.RandomForestRegressor, {'n_estimators': 30, 'min_samples_leaf': 2, 'bootstrap': False}),
        # (sklearn.ensemble.RandomForestRegressor, {'n_estimators': 30, 'min_samples_leaf': 2, 'bootstrap': True}),
        # (sklearn.ensemble.ExtraTreesRegressor, {'n_estimators': 10, 'min_samples_leaf': 2, 'bootstrap': False}),
        # (sklearn.ensemble.ExtraTreesRegressor, {'n_estimators': 10, 'min_samples_leaf': 2, 'bootstrap': True}),
        # (sklearn.ensemble.ExtraTreesRegressor, {'n_estimators': 20, 'min_samples_leaf': 2, 'bootstrap': False}),
        # (sklearn.ensemble.ExtraTreesRegressor, {'n_estimators': 20, 'min_samples_leaf': 2, 'bootstrap': True}),
        # (sklearn.ensemble.ExtraTreesRegressor, {'n_estimators': 30, 'min_samples_leaf': 2, 'bootstrap': False}),
        # (sklearn.ensemble.ExtraTreesRegressor, {'n_estimators': 30, 'min_samples_leaf': 2, 'bootstrap': True}),
        # (sklearn.ensemble.AdaBoostRegressor, {}),
        # (sklearn.ensemble.GradientBoostingRegressor, {}),
        # (sklearn.neighbors.KNeighborsRegressor, {}),
        # (sklearn.neighbors.RadiusNeighborsRegressor, {}),
        # (sklearn.svm.SVR, {}),
        # (sklearn.neural_network.MLPRegressor, {}),
        # (sklearn.neural_network.MLPRegressor, {'solver':'lbfgs'}),
    ]

    # data_params = [{'skip': 0, 'count': 1000000}, {'skip': 1000000, 'count': 1000000}, {'skip': 2000000, 'count': 1000000}]
    data_params = [{}]
    prep_params = [{'to_octets': True}]
    modes = {'test'}

    for data_par in data_params:
        all_data = make_slice(data, **data_par)
        results[f'Data {data_par} (all)'] = stats(all_data[-1], all_data[-1])
        for prep_par in prep_params:
            all_inp, all_oc = prepare_input(all_data, **prep_par)
            for n, (train_index, test_index) in enumerate(sklearn.model_selection.KFold(data_par.get('folds', 5)).split(all_inp, all_oc)):
                train_inp, train_oc = all_inp[train_index], all_oc[train_index]
                # results[f'Data {data_par} {n} (train)'] = stats(train_oc, train_oc)
                test_inp, test_oc = all_inp[test_index], all_oc[test_index]
                results[f'Data {data_par} {n} (test)'] = stats(test_oc, test_oc)
                for clf_class, clf_params in algos:
                    if app_args.fork:
                        pid = os.fork()
                    else:
                        pid = None
                    if not pid:
                        if pid == 0:
                            results = {}
                        name = f"{clf_class.__name__} {clf_params} {prep_par} {data_par} {n}"
                        try:
                            logmsg(f"Starting {name}")
                            if issubclass(clf_class, sklearn.base.ClassifierMixin):
                                r = collections.defaultdict(list)
                                for coverage in 1 - 1 / np.power(2, range(26))[1:]:
                                    clf = clf_class(**clf_params)
                                    train_decision = prepare_decision(train_oc, coverage)
                                    clf.fit(train_inp, train_decision, sample_weight=train_oc)
                                    for mode in modes:
                                        inp = locals()[f'{mode}_inp']
                                        oc = locals()[f'{mode}_oc']
                                        decision_predicted = clf.predict(inp)
                                        r[mode].append([coverage, len(oc) / decision_predicted.sum(), oc[decision_predicted].sum() / oc.sum()])
                                for mode in modes:
                                    results[f'{name} ({mode})'] = np.array(r[mode]).T
                                    print(results[f'{name} ({mode})'])
                            elif issubclass(clf_class, sklearn.base.RegressorMixin):
                                clf = clf_class(**clf_params)
                                clf.fit(train_inp, train_oc)
                                logmsg(f"Fitted {name}")
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
                            pathlib.Path(f'./{app_args.output}').mkdir(parents=True, exist_ok=True)
                            for key, val in results.items():
                                np.savetxt(f'{app_args.output}/{key}.tsv', val, delimiter='\t')
                            exit()

    if app_args.fork:
        try:
            while os.wait():
                pass
        except ChildProcessError:
            pass
        for f in pathlib.Path(f'./{app_args.output}/').glob('*.tsv'):
            results[f.stem] = np.loadtxt(str(f), delimiter='\t')

    plot(results, f'{app_args.files}')
    plt.legend(fontsize=4)
    # plt.show()
    plt.savefig(f'{app_args.output}.pdf')

    with open(f'{app_args.output}.tsv', 'w') as f:
        for name, (_, red, cov) in results.items():
            x, y = interp_red(np.arange(0.8, 0.9, 0.01), red, cov)
            print(f'{y.mean():.2f}\t{y[0]:.2f}\t{name}', file=f)


if __name__ == '__main__':
    main()
