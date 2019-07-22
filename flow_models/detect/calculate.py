#!/usr/bin/python3

import argparse

import numpy as np
import pandas as pd
import scipy.stats
import scipy.interpolate

from flow_models.generate import X_VALUES, load_data
from flow_models.lib import mix
from flow_models.lib.util import logmsg

METHODS = ['first', 'treshold', 'sampling']
INTEGRATE_STEPS = 4194304

def calculate_data(data, x_probs, method):
    ad = {}
    x = np.unique(np.rint(1 / x_probs)).astype(int)
    idx = data.index.values
    idx_diff = np.concatenate([idx[:1], np.diff(idx)])

    if method == 'first':

        for what in ['flows', 'packets', 'portion', 'octets']:
            w = 'flows' if what == 'portion' else what
            cdf = data[w + '_sum'].cumsum() / data[w + '_sum'].sum()
            cdf = scipy.interpolate.interp1d(cdf.index, cdf, 'previous', bounds_error=False)(x)
            ad[what + '_mean'] = 1 - cdf

    elif method == 'treshold':

        for what in ['flows', 'packets', 'portion', 'octets']:
            w = 'flows' if what == 'portion' else what
            if what == 'flows':
                toc = data[w + '_sum']
                cdf = 1 - toc.cumsum() / data[w + '_sum'].sum()
                ad[what + '_mean'] = scipy.interpolate.interp1d(cdf.index, cdf, 'previous', bounds_error=False)(x)
            else:
                toc = (data[w + '_sum'] / idx)[::-1].cumsum()[::-1] * idx_diff
                cdf = 1 - toc.cumsum() / data[w + '_sum'].sum()
                ad[what + '_mean'] = scipy.interpolate.interp1d(cdf.index, cdf, 'linear', bounds_error=False)(x)

    else:

        pp = (1 - x_probs) ** idx[:, np.newaxis]

        for what in ['flows', 'packets', 'portion', 'octets']:
            w = 'flows' if what == 'portion' else what
            if what == 'flows':
                toc = data[w + '_sum']
            else:
                toc = (data[w + '_sum'] / idx)[::-1].cumsum()[::-1] * idx_diff
            cdf = 1 - (pp * toc[:, np.newaxis]).sum(axis=0) / data[w + '_sum'].sum()
            ad[what + '_mean'] = np.array(cdf)

    ad['add_mean'] = 1 / ad['flows_mean']
    ad['avs_mean'] = 1 / ad['portion_mean']
    for what in ['flows', 'packets', 'portion', 'octets']:
        ad[what + '_mean'] *= 100

    return pd.DataFrame(ad, x_probs if method == 'sampling' else x)

def calculate_mix(data, x_probs, method):
    ad = {}
    x = np.unique(np.rint(1 / x_probs)).astype(int)
    idx = np.geomspace(x.min(), x.max(), INTEGRATE_STEPS)
    idx = np.unique(np.rint(idx)).astype(int)
    idx_diff = np.concatenate([idx[:1], np.diff(idx)])

    if method == 'first':

        for what in ['flows', 'packets', 'portion', 'octets']:
            w = 'flows' if what == 'portion' else what
            cdf = mix.cdf(data[w], x)
            ad[what + '_mean'] = 1 - cdf

    elif method == 'treshold':

        for what in ['flows', 'packets', 'portion', 'octets']:
            w = 'flows' if what == 'portion' else what
            if what == 'flows':
                cdf = mix.cdf(data[w], x)
                ad[what + '_mean'] = 1 - cdf
            else:
                cdf = mix.cdf(data[w], idx)
                pdf = np.concatenate([cdf[:1], np.diff(cdf)])
                toc = (pdf / idx)[::-1].cumsum()[::-1] * idx_diff
                cdf = 1 - toc.cumsum()
                ad[what + '_mean'] = scipy.interpolate.interp1d(idx, cdf, 'linear', bounds_error=False)(x)

    else:

        pp = (1 - x_probs) ** idx[:, np.newaxis]

        for what in ['flows', 'packets', 'portion', 'octets']:
            w = 'flows' if what == 'portion' else what
            cdf = mix.cdf(data[w], idx)
            pdf = np.concatenate([cdf[:1], np.diff(cdf)])
            if what == 'flows':
                toc = pdf
            else:
                toc = (pdf / idx)[::-1].cumsum()[::-1] * idx_diff
            cdf = 1 - (pp * toc[:, np.newaxis]).sum(axis=0)
            ad[what + '_mean'] = np.array(cdf)

    ad['add_mean'] = 1 / ad['flows_mean']
    ad['avs_mean'] = 1 / ad['portion_mean']
    for what in ['flows', 'packets', 'portion', 'octets']:
        ad[what + '_mean'] *= 100

    return pd.DataFrame(ad, x_probs if method == 'sampling' else x)

def calculate(obj, index=None, x_val='length', methods=tuple(METHODS)):
    data = load_data(obj)

    if index is None:
        index = [1.0,
                 0.5, 0.2, 0.1,
                 0.05, 0.02, 0.01,
                 0.005, 0.002, 0.001,
                 0.0005, 0.0002, 0.0001,
                 0.00005, 0.00002, 0.00001,
                 0.000005, 0.000002, 0.000001,
                 0.0000005, 0.0000002, 0.0000001,
                 0.00000005, 0.00000002, 0.00000001]
    elif isinstance(index, int):
        index = np.geomspace(1.0, 1.0e-08, index)
    else:
        index = index

    dataframes = {}
    for method in methods:
        if isinstance(data, pd.DataFrame):
            df = calculate_data(data, np.array(index), method)
        else:
            df = calculate_mix(data, np.array(index), method)
        dataframes[method] = df

    return dataframes

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-s', type=int, default=None, help='number of index points')
    parser.add_argument('-x', default='length', choices=X_VALUES, help='x axis value')
    parser.add_argument('-m', default='all', choices=METHODS, help='method')
    parser.add_argument('--save', action='store_true', help='save to files')
    parser.add_argument('file', help='csv_hist file or mixture directory')
    app_args = parser.parse_args()

    if app_args.m == 'all':
        methods = METHODS
    else:
        methods = [app_args.m]

    resdic = calculate(app_args.file, app_args.s, app_args.x, methods)
    for method, dataframe in resdic.items():
        print(method)
        print(dataframe.info())
        print(dataframe.to_string())
        if app_args.save:
            dataframe.to_csv(method + '.csv')
            dataframe.to_pickle(method + '.df')

    logmsg('Finished')


if __name__ == '__main__':
    main()
