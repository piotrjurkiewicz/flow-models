#!/usr/bin/python3

import argparse

import numpy as np
import scipy.stats

from ..fit import fit_mix
from ..lib.mix import to_json
from ..lib.util import logmsg, measure_memory

def test(max_iter):
    logmsg("genpareto lognorm")

    a = scipy.stats.genpareto.rvs(1.450289555235508, 16, 23.204632883768134, 500000)
    b = scipy.stats.lognorm.rvs(5, 0, 20, 500000)
    vec = np.concatenate([a, b])

    mix = [
        [0.2, 'genpareto', (1.450289555235508, 16, 23.204632883768134)],
        [0.8, 'lognorm', (5, 0, 20)],
    ]

    mix = fit_mix(vec, mix, max_iter=max_iter)
    logmsg(to_json(mix))

    logmsg("gamma")

    a = scipy.stats.gamma.rvs(5.0, 0, 2.0, 500000)
    b = scipy.stats.gamma.rvs(10.0, 0, 1.0, 500000)
    vec = np.concatenate([a, b])

    mix = [
        [0.1, 'gamma', (2.0, 0, 2.0)],
        [0.1, 'gamma', (6.0, 0, 1.0)],
    ]

    mix = fit_mix(vec, mix, max_iter=max_iter)
    logmsg(to_json(mix))

    logmsg("weibull_min")

    a = scipy.stats.weibull_min.rvs(0.763166697701473, 0, 1.805880227867377e02, 500000)
    b = scipy.stats.weibull_min.rvs(0.984428347376388, 0, 9.685081880588410e04, 500000)
    vec = np.concatenate([a, b])

    mix = [
        [0.1, 'weibull_min', (0.603166697701473, 0, 1.205880227867377e02)],
        [0.1, 'weibull_min', (0.904428347376388, 0, 6.685081880588410e04)],
    ]

    mix = fit_mix(vec, mix, max_iter=max_iter)
    logmsg(to_json(mix))

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    parser.add_argument('-i', default=100, type=int, help='number of iterations')
    parser.add_argument('--measure-memory', action='store_true', help='collect and print memory statistics')
    app_args = parser.parse_args()

    with measure_memory(app_args.measure_memory):
        test(app_args.i)


if __name__ == '__main__':
    main()
