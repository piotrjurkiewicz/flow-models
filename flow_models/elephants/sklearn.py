#!/usr/bin/python3
import argparse
import collections
import os
import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import sklearn.ensemble
import sklearn.neighbors
import sklearn.neural_network
import sklearn.preprocessing
import sklearn.svm
import sklearn.tree

from flow_models.generate import load_data
from flow_models.lib import mix
from flow_models.lib.io import load_array_np
from flow_models.lib.util import logmsg

def prepare_decision(oc, coverage):

    idx = np.flip(np.argsort(oc))
    idx_back = np.argsort(idx)

    oc_cumsum = oc[idx].cumsum()
    decision = oc_cumsum < oc_cumsum[-1] * coverage
    decision = decision[idx_back]

    return decision

def load_arrays(directory):

    d = pathlib.Path(directory)
    _, _, sa = load_array_np(d / 'sa3')
    _, _, da = load_array_np(d / 'da3')
    _, _, sp = load_array_np(d / 'sp')
    _, _, dp = load_array_np(d / 'dp')
    _, _, prot = load_array_np(d / 'prot')
    _, _, oc = load_array_np(d / 'octets')

    return sa, da, sp, dp, prot, oc

def make_slices(data, skip=0, count=None, train_pct=0.80):

    sa, da, sp, dp, prot, oc = data

    if not count:
        count = len(sa)

    if isinstance(skip, float):
        skip = int(len(sa) * skip)

    if isinstance(count, float):
        count = int(len(sa) * count)

    train_count = int(count * train_pct)
    train_slice = slice(skip, skip + train_count)
    test_slice = slice(skip + train_count, skip + count)

    all_sa = sa[skip:skip + count]
    all_da = da[skip:skip + count]
    all_sp = sp[skip:skip + count]
    all_dp = dp[skip:skip + count]
    all_prot = prot[skip:skip + count]
    all_oc = oc[skip:skip + count]

    train_sa = sa[train_slice]
    train_da = da[train_slice]
    train_sp = sp[train_slice]
    train_dp = dp[train_slice]
    train_prot = prot[train_slice]
    train_oc = oc[train_slice]

    test_sa = sa[test_slice]
    test_da = da[test_slice]
    test_sp = sp[test_slice]
    test_dp = dp[test_slice]
    test_prot = prot[test_slice]
    test_oc = oc[test_slice]

    return (all_sa, all_da, all_sp, all_dp, all_prot, all_oc), \
           (train_sa, train_da, train_sp, train_dp, train_prot, train_oc), \
           (test_sa, test_da, test_sp, test_dp, test_prot, test_oc)

def prepare_input(data, shuffle=False, to_octets=False, bit_vector=False):

    sa, da, sp, dp, prot, oc = data

    if to_octets:
        sa = sa.view(np.uint8).reshape(sa.shape + (sa.dtype.itemsize,)).T
        da = da.view(np.uint8).reshape(da.shape + (da.dtype.itemsize,)).T
        sp = sp.view(np.uint8).reshape(sp.shape + (sp.dtype.itemsize,)).T
        dp = dp.view(np.uint8).reshape(dp.shape + (dp.dtype.itemsize,)).T
    else:
        sa = [sa]
        da = [da]
        sp = [sp]
        dp = [dp]

    columns = (*sp, *sa, *da, *dp, prot)

    if bit_vector:
        cols = []
        for col in columns:
            cols.append(col.view(np.uint8).reshape(col.shape + (col.dtype.itemsize,)))
        inp = np.column_stack(cols)
        inp = np.unpackbits(inp, axis=1)
    else:
        inp = np.column_stack(columns)

    if shuffle:
        idx = np.arange(len(oc))
        rng = np.random.default_rng()
        rng.shuffle(idx)
        inp = inp[idx]
        oc = oc[idx]

    print(inp.shape, inp.dtype)
    return inp, oc

def stats(oc, oc_predicted, thresholds=None):
    if thresholds is None:
        thresholds = np.power(2, range(25)) * 64
    elif isinstance(thresholds, int):
        thresholds = np.logspace(0, 24, thresholds, base=2) * 64
    else:
        thresholds = thresholds * 64

    r = []
    for threshold in thresholds:
        decision = oc_predicted > threshold
        r.append([threshold, len(oc) / decision.sum(), oc[decision].sum() / oc.sum()])
    return np.array(r).T

def plot(r, title=''):
    plt.figure(figsize=(10, 5))
    plt.yscale('log')
    xlim = (48, 102)
    ylim = (1, 10000)
    for name, (_, red, cov) in r.items():
        if name == 'Mixture':
            plt.plot(cov * 100, red, label=name, lw=1, color='k')
            xlim = plt.xlim()
        else:
            plt.plot(cov * 100, red, label=name, lw=1)
    plt.xlim(*xlim[::-1])
    plt.ylim(*ylim)
    plt.gca().get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.legend()
    plt.title(title)
    plt.xlabel('Traffic coverage [%]')
    plt.ylabel('Flow table occupancy reduction [x]')

def calculate_from_mixture():
    data = list(load_data(['/home/jurkiew/git/flow-models/data/agh_2015/mixtures/all/size/']).values())[0]
    index = 1 / np.logspace(0, 32, 1024, base=2)
    x = np.unique(np.rint(1 / np.array(index))).astype('u8')
    x *= 64

    reduction = 1 / (1 - mix.cdf(data['flows'], x))
    coverage = 1 - mix.cdf(data['octets'], x)
    mask = coverage > 0.5

    return x[mask], reduction[mask], coverage[mask]

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
