#!/usr/bin/python3
"""Trains and evaluates sklearn classifier models to classify elephant flows."""

import argparse
import collections
import itertools
import os
import pathlib

import numpy as np
import sklearn.ensemble
import sklearn.model_selection
import sklearn.neighbors
import sklearn.neural_network
import sklearn.preprocessing
import sklearn.svm
import sklearn.tree

from flow_models.elephants.simulate_data import simulate_data
from flow_models.lib.ml import load_arrays, make_slice, prepare_decision, prepare_input
from flow_models.lib.util import logmsg

PPS = None
FPS = 1810
TIMEOUT = 15
ST = 300
ET = 720

class Data:
    pass

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-O', '--output', default='sklearn', help='output directory')
    p.add_argument('--seed', type=int, default=None, help='seed')
    p.add_argument('--fork', action='store_true', help='')
    p.add_argument('--jobs', type=int, default=1, help='')
    p.add_argument('files', help='directory')
    return p

def main():
    app_args = parser().parse_args()
    jobs = set()
    data = load_arrays(app_args.files)
    results = collections.defaultdict(lambda: collections.defaultdict(list))
    decisions_true = collections.defaultdict(list)
    decisions_predicted = collections.defaultdict(list)

    algos = [
        (sklearn.tree.DecisionTreeClassifier, {}),
        # (sklearn.ensemble.RandomForestClassifier, {'n_jobs': -1, 'max_depth': 20}),
        # (sklearn.ensemble.ExtraTreesClassifier, {'n_jobs': -1, 'max_depth': 25}),
        # (sklearn.ensemble.AdaBoostClassifier, {}),
        # (sklearn.ensemble.GradientBoostingClassifier, {}),
        # (sklearn.neighbors.KNeighborsClassifier, {'n_jobs': -1}),
        # (sklearn.ensemble.HistGradientBoostingClassifier, {}),
        # (Data, {}),
    ]

    # hostname = os.uname().nodename
    # if hostname.startswith('mach'):
    #     algos = [algos[int(hostname[-1])]]

    # data_par = {'skip': 0, 'count': 1000000}
    data_par = {}
    # prep_params = [{'octets': True}]
    prep_params = [{}, {'bits': True}, {'octets': True}]
    modes = ['train', 'test']

    all_data = make_slice(data, **data_par)
    *all_inp, all_octets = all_data
    for n, (train_index, test_index) in enumerate(sklearn.model_selection.KFold(data_par.get('folds', 5)).split(all_octets)):
        logmsg(f"Folding {n}")
        train_octets, test_octets = all_octets[train_index], all_octets[test_index]
        train_flows_sum, train_octets_sum, train_flows_slots, train_octets_slots = simulate_data(app_args.files, index=train_index, mask=None, pps=PPS, fps=FPS, timeout=TIMEOUT)
        test_flows_sum, test_octets_sum, test_flows_slots, test_octets_slots = simulate_data(app_args.files, index=test_index, mask=None, pps=PPS, fps=FPS, timeout=TIMEOUT)
        for prep_par in prep_params:
            logmsg(f"Preparing {prep_par} {n}")
            prepared_inp = prepare_input(all_inp, **prep_par)
            train_inp = prepared_inp[train_index]
            test_inp = prepared_inp[test_index]
            for clf_class, clf_params in algos:
                if clf_class.__name__ == 'Data' and prep_par:
                    continue
                if clf_class.__name__ == 'KNeighborsClassifier' and 'bits' in prep_par:
                    continue
                if app_args.fork:
                    if len(jobs) >= app_args.jobs:
                        waited_pid, _ = os.wait()
                        jobs.remove(waited_pid)
                    pid = os.fork()
                else:
                    pid = None
                if pid:
                    jobs.add(pid)
                else:
                    name = f"{clf_class.__name__} {dict(**clf_params)} {prep_par} {n}"
                    logmsg(f"Starting {name}")
                    for training_coverage in 1 - 1 / np.power(1.37972966146121546, range(26))[1:]:
                        logmsg(f"Training {name} training_coverage: {training_coverage}")
                        if clf_class.__name__ == 'KNeighborsClassifier':
                            clf_kwargs = {}
                        else:
                            clf_kwargs = {'random_state': app_args.seed}
                        if clf_class.__name__ == 'Data':
                            clf = None
                        else:
                            clf = clf_class(**clf_params, **clf_kwargs)
                            train_decision = prepare_decision(train_octets, training_coverage)
                            # train_decision = train_octets > 8388608
                            idx = Ellipsis
                            # idx = top_idx(train_octets, 0.1)
                            # Balanced sample weights
                            # sample_weight = np.ones(len(train_decision))
                            # sample_weight[train_decision] *= len(train_octets[~train_decision]) / len(train_octets[train_decision])
                            # Power of octets sample weights
                            sample_weight = train_octets ** 0.5
                            if clf_class.__name__ == 'KNeighborsClassifier':
                                fit_kwargs = {}
                            else:
                                fit_kwargs = {'sample_weight': sample_weight[idx]}
                            clf.fit(train_inp[idx], train_decision[idx], **fit_kwargs)
                        for mode in modes:
                            logmsg(f"Evaluating {name} training_coverage: {training_coverage} mode: {mode}")
                            lc = locals()
                            index = lc[f'{mode}_index']
                            inp = lc[f'{mode}_inp']
                            octets = lc[f'{mode}_octets']
                            base_flows_sum, base_octets_sum = lc[f'{mode}_flows_sum'], lc[f'{mode}_octets_sum']
                            base_flows_slots, base_octets_slots = lc[f'{mode}_flows_slots'], lc[f'{mode}_octets_slots']
                            decision_true = prepare_decision(octets, training_coverage)
                            if clf_class.__name__ == 'Data':
                                decision_predicted = decision_true
                            else:
                                decision_predicted = clf.predict(inp)
                            decisions_true[f'{name} ({mode})'].append(np.packbits(decision_true))
                            decisions_predicted[f'{name} ({mode})'].append(np.packbits(decision_predicted))
                            this_flows_sum, this_octets_sum, this_flows_slots, this_octets_slots = simulate_data(app_args.files, index=index, mask=decision_predicted, pps=PPS, fps=FPS, timeout=TIMEOUT)
                            c = itertools.count()
                            results[f'{name} ({mode})'][next(c)].append(training_coverage)
                            results[f'{name} ({mode})'][next(c)].append(octets[decision_predicted].sum() / octets.sum())
                            results[f'{name} ({mode})'][next(c)].append(len(octets) / decision_predicted.sum())
                            results[f'{name} ({mode})'][next(c)].append((this_octets_slots[ST:ET] / base_octets_slots[ST:ET]).min())
                            results[f'{name} ({mode})'][next(c)].append((this_octets_slots[ST:ET] / base_octets_slots[ST:ET]).mean())
                            results[f'{name} ({mode})'][next(c)].append((this_octets_slots[ST:ET] / base_octets_slots[ST:ET]).max())
                            results[f'{name} ({mode})'][next(c)].append((base_flows_slots[ST:ET] / this_flows_slots[ST:ET]).min())
                            results[f'{name} ({mode})'][next(c)].append((base_flows_slots[ST:ET] / this_flows_slots[ST:ET]).mean())
                            results[f'{name} ({mode})'][next(c)].append((base_flows_slots[ST:ET] / this_flows_slots[ST:ET]).max())
                            for a in sklearn.metrics.confusion_matrix(decision_true, decision_predicted).ravel():
                                results[f'{name} ({mode})'][next(c)].append(a)
                            print(*(results[f'{name} ({mode})'][n][-1] for n in range(13)))
                            if clf_class.__name__ not in ['Data', 'KNeighborsClassifier', 'HistGradientBoostingClassifier']:
                                for a in clf.feature_importances_:
                                    results[f'{name} ({mode})'][next(c)].append(a)
                            if clf_class.__name__ in ['DecisionTreeClassifier', 'RandomForestClassifier', 'ExtraTreesClassifier']:
                                depths, leaves = [], []
                                if clf_class.__name__ == 'DecisionTreeClassifier':
                                    depths.append(clf.get_depth())
                                    leaves.append(clf.get_n_leaves())
                                else:
                                    for est in clf.estimators_:
                                        depths.append(est.get_depth())
                                        leaves.append(est.get_n_leaves())
                                results[f'{name} ({mode})'][next(c)].append(min(depths))
                                results[f'{name} ({mode})'][next(c)].append(sum(depths) / len(depths))
                                results[f'{name} ({mode})'][next(c)].append(max(depths))
                                results[f'{name} ({mode})'][next(c)].append(min(leaves))
                                results[f'{name} ({mode})'][next(c)].append(sum(leaves) / len(leaves))
                                results[f'{name} ({mode})'][next(c)].append(max(leaves))
                    logmsg(f"Finished {name}")

                    output = pathlib.Path(f'./{app_args.output}/classifiers')
                    if data_par:
                        output = output / f'{data_par}'
                    output.mkdir(parents=True, exist_ok=True)
                    for key, val in results.items():
                        with open(f'{output}/{key}.tsv', 'w') as ff:
                            for row in val.values():
                                row = np.array(row)
                                fmt = '%i' if row.dtype == np.int64 else '%.18e'
                                np.savetxt(ff, row.reshape(1, -1), fmt=fmt, delimiter='\t')
                    for key, val in decisions_true.items():
                        np.savez_compressed(f'{output}/{key}.dt.npz', *val)
                    for key, val in decisions_predicted.items():
                        np.savez_compressed(f'{output}/{key}.dp.npz', *val)
                    if pid == 0:
                        exit()

    if app_args.fork:
        try:
            while os.wait():
                pass
        except ChildProcessError:
            pass


if __name__ == '__main__':
    main()
