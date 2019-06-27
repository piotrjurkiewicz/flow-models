#!/usr/bin/python3

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.special
import scipy.stats

from .plot import plot
from .lib.mix import to_json
from .lib.data import detect_x_value
from .lib.util import logmsg, measure_memory

X_VALUES = ['length', 'size']
Y_VALUES = ['flows', 'packets', 'octets']
EPS = np.finfo(np.float64).eps

def fit_mix(x, mix, ws=None, max_iter=100, max_pareto_w=None):
    weights = np.zeros((x.shape[0], len(mix)))
    if ws is not None:
        avg_ws = np.average(ws)
    else:
        avg_ws = 1.0

    for i in range(max_iter):
        logmsg(i, '\n' + to_json(mix))
        for k, comp in enumerate(mix):
            dist = getattr(scipy.stats, comp[1])
            pdf = dist.pdf(x, *comp[2]) * comp[0]
            z = x[:1]
            previous_z = z - 1.0
            previous_z[z == z.min()] = 0.0
            pdf[:1] = (dist.cdf(z, *comp[2]) - dist.cdf(previous_z, *comp[2])) * comp[0]
            weights[:, k] = pdf

        w_sum = np.sum(weights, axis=1)[:, None]
        w_sum[w_sum == 0.0] = 1.0
        weights = weights / w_sum

        if ws is not None:
            weights = weights * ws[:, None] / avg_ws

        max_upper = 0.0
        for comp in mix:
            if comp[1] == 'uniform':
                max_upper = max(max_upper, comp[2][1])
            if comp[1] == 'genpareto':
                max_upper = max(max_upper, comp[2][1] + 1.0)

        logmsg(i, max_upper)

        for k, comp in enumerate(mix):
            frozen_loc = False
            if comp[1] == 'genpareto':
                frozen_loc = True
            old_params = comp[2]
            comp[2] = fit_comp(x, comp, weights, k, frozen_loc=frozen_loc)
            if comp[1] == 'lognorm' and comp[2][2] < max_upper:
                comp[2] = old_params

        for k, weight in enumerate(np.mean(weights, axis=0, keepdims=True).flat):
            mix[k][0] = weight
            if max_pareto_w and mix[k][1] == 'genpareto':
                mix[k][0] = min(weight, max_pareto_w)

    return mix

def fit_comp(x, comp, weights, k, frozen=False, frozen_loc=False):
    if frozen:
        return comp[2]

    w = weights[:, k]

    if comp[1] == 'uniform':
        return comp[2]

    elif comp[1] in ('norm', 'lognorm'):
        if comp[1] == 'lognorm':
            x = np.log(x)
            mu = np.log(comp[2][2])
        else:
            mu = comp[0]
        sumw = np.sum(w)
        if not sumw > 0.0:
            sumw = EPS
        if not frozen_loc:
            mu = np.sum(w * x) / sumw
        sigma = np.sqrt(np.sum(w * ((x - mu) ** 2)) / sumw)
        if not sigma > 0.0:
            sigma = EPS
        if comp[1] == 'norm':
            return mu, sigma
        else:
            return sigma, 0, np.exp(mu)

    elif comp[1] == 'genpareto':
        beta = comp[2][1]
        if not frozen_loc:
            max_idx = np.argmax(weights, axis=1)
            max_x = x[max_idx == k]
            if len(max_x):
                beta = np.min(max_x)
        alpha = 1 / ((np.log(x) @ w) / np.sum(w) - np.log(beta))
        return 1 / alpha, beta, beta / alpha

    elif comp[1] == 'gamma':
        bhat = np.sum(w * x) / np.sum(w) / comp[2][0]

        def func(ah):
            return np.sum(w * (-np.log(bhat) - scipy.special.psi(ah) + np.log(x)))

        ahat = scipy.optimize.fsolve(func, np.array([1.0]))
        return ahat, 0, bhat

    elif comp[1] == 'weibull_min':
        lhat = (np.sum(w * (x ** comp[2][0])) / np.sum(w)) ** (1 / comp[2][0])

        def f1(kh):
            return 1 / kh * np.sum(w) + np.sum(w * np.log(x)) - np.log(lhat) * np.sum(w) - lhat ** (-kh) * np.sum(w * x ** kh * np.log(x / lhat))

        def df1(kh):
            return -1 / (kh ** 2) * np.sum(w) - np.sum(w * (x / lhat) ** 2 * (np.log(x / lhat)) ** 2)

        khat = scipy.optimize.zeros.newton(f1, 1.0, fprime=df1, maxiter=5, disp=False)

        return khat, 0, lhat

    else:
        raise ValueError

def initial_mix(mode, x):
    mix = []
    min_x = x.min()
    uniform_number = mode.get('U', 0)
    pareto_number = mode.get('P', 0)
    lognorm_number = mode.get('L', 0)
    geom = np.geomspace(x.min() + uniform_number - 1, x.max(), pareto_number + lognorm_number)
    ng = 0
    for n in range(uniform_number):
        mix.append([0.1, 'uniform', [0, min_x + n]])
    for n in range(pareto_number):
        if n == 0:
            mix.append([0.1, 'genpareto', [1.0, min_x + uniform_number - 1, 10.0]])
        else:
            mix.append([0.1, 'genpareto', [1.0, geom[ng], 10.0]])
            ng += 1
    for n in range(lognorm_number):
        mix.append([0.1, 'lognorm', [2.0, 0, geom[ng]]])
        ng += 1

    return mix

def fit(path, x_value, y_value, size_limit=None, max_iter=100, initial=None, max_pareto_w=None):
    path = pathlib.Path(path)
    logmsg(f'Processing: {path}')

    if path.is_dir():

        raise NotImplementedError

        # TODO: to be finished
        # size = None
        # memory_maps = {}
        #
        # for name in ['packets']:
        #     try:
        #         name, dtype, in_mm = load_array_np(path / name, 'r')
        #         assert name not in memory_maps
        #         if size is None:
        #             size = in_mm.size
        #         else:
        #             assert in_mm.size == size
        #         if size_limit:
        #             in_mm = in_mm[:size_limit]
        #         memory_maps[name] = in_mm
        #         logmsg(f'Loaded array: {path / name}')
        #     except FileNotFoundError:
        #         logmsg(f'Not found array: {path / name}')
        #
        # if size_limit:
        #     size = min(size, size_limit)
        #
        # x = memory_maps[y_value]
        # weights = None

    else:

        data = pd.read_csv(path, index_col=0, sep=',', low_memory=False, usecols=lambda n: not n.endswith('_ssq'))
        x = data.index
        x = x.values
        weights = data[f'{y_value}_sum'].values

    if x_value is None:
        x_value = detect_x_value(x)

    if isinstance(initial, dict):
        mix = initial_mix(initial, x)

    else:
        mix = json.load(open(str(initial)))

    result_mix = fit_mix(x, mix, weights, max_iter=max_iter, max_pareto_w=max_pareto_w)
    print(to_json(result_mix))

    mixtures = {}
    for ff in (path.parent.parent / 'mixtures' / x_value).glob('*.json'):
        mixtures[ff.stem] = json.load(open(str(ff)))

    mixtures[y_value]['mix'] = result_mix
    mixtures[y_value]['sum'] = np.sum(weights)

    logmsg('Plotting')
    plot([path, mixtures], x_value, ext='pdf', cdf_modes={}, pdf_modes={'comp'})

    return {x_value: mixtures[y_value]}

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    parser.add_argument('file', help='input dir or file')
    parser.add_argument('-x', choices=X_VALUES, help='x axis value')
    parser.add_argument('-y', default='flows', choices=Y_VALUES, help='y axis value')
    parser.add_argument('-s', default=None, type=int, help='size limit')
    parser.add_argument('-i', default=100, type=int, help='number of iterations')
    parser.add_argument('-U', type=int, help='number of uniform distributions')
    parser.add_argument('-P', type=int, help='number of Pareto distributions')
    parser.add_argument('-L', type=int, help='number of lognorm distributions')
    parser.add_argument('--mpw', type=float, help='maximum pareto weight')
    parser.add_argument('--initial', help='initial mixture')
    parser.add_argument('--test', action='store_true', help='test fitting')
    parser.add_argument('--measure-memory', action='store_true', help='collect and print memory statistics')
    app_args = parser.parse_args()

    mode = {}
    for key, val in vars(app_args).items():
        if key.isupper() and val:
            mode[key] = val

    if mode:
        initial = mode
    else:
        initial = app_args.initial

    with measure_memory(app_args.measure_memory):
        new_mixtures = fit(app_args.file, app_args.x, app_args.y, app_args.s, app_args.i, initial=initial, max_pareto_w=app_args.mpw)
        if input('Save mixture to file?\n').startswith('y'):
            for key, val in new_mixtures.items():
                (pathlib.Path(app_args.file).parent.parent / 'mixtures' / key / (app_args.y + '.json')).write_text(to_json(**val) + '\n')
                (pathlib.Path(app_args.file).parent.parent / 'mixtures' / key / (app_args.y + '.mode')).write_text(json.dumps(vars(app_args)) + '\n')


if __name__ == '__main__':
    main()
