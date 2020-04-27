#!/usr/bin/python3
import argparse

import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.stats

from flow_models.generate import X_VALUES, load_data
from flow_models.lib import mix
from flow_models.lib.util import logmsg

METHODS = ['first', 'threshold', 'sampling']
INTEGRATE_STEPS = 262144

def calculate_data(data, x_probs, x_val, method):
    ad = {}
    x = np.unique(np.rint(1 / x_probs)).astype('u8')
    if x_val == 'size':
        x *= 64
    idx = data.index.values
    idx_diff = np.concatenate([idx[:1], np.diff(idx)])

    if method == 'first':

        for what in ['flows', 'packets', 'fraction', 'octets']:
            w = 'flows' if what == 'fraction' else what
            cdf = data[w + '_sum'].cumsum() / data[w + '_sum'].sum()
            cdf = scipy.interpolate.interp1d(cdf.index, cdf, 'previous', bounds_error=False)(x)
            ad[what + '_mean'] = 1 - cdf

    elif method == 'threshold':

        for what in ['flows', 'packets', 'fraction', 'octets']:
            w = 'flows' if what == 'fraction' else what
            if what == 'flows':
                toc = data[w + '_sum']
                cdf = 1 - toc.cumsum() / data[w + '_sum'].sum()
                ad[what + '_mean'] = scipy.interpolate.interp1d(cdf.index, cdf, 'previous', bounds_error=False)(x)
            else:
                toc = (data[w + '_sum'] / idx)[::-1].cumsum()[::-1] * idx_diff
                cdf = 1 - toc.cumsum() / data[w + '_sum'].sum()
                ad[what + '_mean'] = scipy.interpolate.interp1d(cdf.index, cdf, 'linear', bounds_error=False)(x)

    else:

        ps = []
        if x_val == 'length':
            for p in x_probs:
                ps.append((1 - p) ** idx)
        else:
            packet_size = data['octets_sum'].cumsum() / data['packets_sum'].cumsum()
            pks = np.clip(idx / packet_size, 1, np.trunc(idx / 64))
            for p in x_probs:
                ps.append((1 - np.clip(p * packet_size / 64, 0, 1)) ** (pks if x_val == 'size' else idx))

        for what in ['flows', 'packets', 'fraction', 'octets']:
            w = 'flows' if what == 'fraction' else what
            if what == 'flows':
                toc = data[w + '_sum']
            else:
                toc = (data[w + '_sum'] / idx)[::-1].cumsum()[::-1] * idx_diff
            a = []
            for p in ps:
                cdf = 1 - (p * toc).sum() / data[w + '_sum'].sum()
                a.append(cdf)
            ad[what + '_mean'] = np.array(a)

    ad['operations_mean'] = 1 / ad['flows_mean']
    ad['occupancy_mean'] = 1 / ad['fraction_mean']
    for what in ['flows', 'packets', 'fraction', 'octets']:
        ad[what + '_mean'] *= 100

    return pd.DataFrame(ad, x_probs if method == 'sampling' else x)

def calculate_mix(data, x_probs, x_val, method):
    ad = {}
    x = np.unique(np.rint(1 / x_probs)).astype('u8')
    if x_val == 'size':
        x *= 64
    idx = np.geomspace(x.min(), x.max(), INTEGRATE_STEPS)
    idx = np.unique(np.rint(idx)).astype('u8')
    idx_diff = np.concatenate([idx[:1], np.diff(idx)])

    if method == 'first':

        for what in ['flows', 'packets', 'fraction', 'octets']:
            w = 'flows' if what == 'fraction' else what
            cdf = mix.cdf(data[w], x)
            ad[what + '_mean'] = 1 - cdf

    elif method == 'threshold':

        for what in ['flows', 'packets', 'fraction', 'octets']:
            w = 'flows' if what == 'fraction' else what
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

        ps = []
        if x_val == 'length':
            for p in x_probs:
                ps.append((1 - p) ** idx)
        else:
            packet_size = (mix.cdf(data['octets'], idx) / mix.cdf(data['packets'], idx)) * (data['octets']['sum'] / data['packets']['sum'])
            pks = np.clip(idx / packet_size, 1, np.trunc(idx / 64))
            # Flows smaller than 128 bytes must be 1-packet long
            packet_size[:64] = idx[:64]
            for p in x_probs:
                ps.append((1 - np.clip(p * packet_size / 64, 0, 1)) ** (pks if x_val == 'size' else idx))

        for what in ['flows', 'packets', 'fraction', 'octets']:
            w = 'flows' if what == 'fraction' else what
            cdf = mix.cdf(data[w], idx)
            pdf = np.concatenate([cdf[:1], np.diff(cdf)])
            if what == 'flows':
                toc = pdf
            else:
                toc = (pdf / idx)[::-1].cumsum()[::-1] * idx_diff
                if x_val != 'length':
                    toc[64] += np.sum(toc[:64])
                    toc[:64] = 0
            a = []
            for p in ps:
                cdf = 1 - (p * toc).sum()
                a.append(cdf)
            ad[what + '_mean'] = np.array(a)

    ad['operations_mean'] = 1 / ad['flows_mean']
    ad['occupancy_mean'] = 1 / ad['fraction_mean']
    for what in ['flows', 'packets', 'fraction', 'octets']:
        ad[what + '_mean'] *= 100

    return pd.DataFrame(ad, x_probs if method == 'sampling' else x)

def calculate(obj, index=None, x_val='length', methods=tuple(METHODS)):
    data = load_data(obj)

    if index is None:
        index = 1 / np.power(2, range(25))
    elif isinstance(index, int):
        index = 1 / np.logspace(0, 32, index, base=2)
    else:
        index = index

    dataframes = {}
    for method in methods:
        if isinstance(data, pd.DataFrame):
            df = calculate_data(data, np.array(index), x_val, method)
        else:
            df = calculate_mix(data, np.array(index), x_val, method)
        dataframes[method] = df

    return dataframes

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-n', type=int, default=None, help='number of index points')
    parser.add_argument('-x', default='length', choices=X_VALUES, help='x axis value')
    parser.add_argument('-m', default='all', choices=METHODS, help='method')
    parser.add_argument('--save', action='store_true', help='save to files')
    parser.add_argument('file', help='csv_hist file or mixture directory')
    app_args = parser.parse_args()

    if app_args.m == 'all':
        methods = METHODS
    else:
        methods = [app_args.m]

    resdic = calculate(app_args.file, app_args.n, app_args.x, methods)
    for method, dataframe in resdic.items():
        print(method)
        print(dataframe.info())
        print(dataframe.to_string())
        if app_args.save:
            dataframe.to_string(open(method + '.txt', 'w'))
            dataframe.to_csv(method + '.csv')
            dataframe.to_pickle(method + '.df')
            dataframe.to_html(method + '.html')
            dataframe.to_latex(method + '.tex', float_format='%.2f')

    logmsg('Finished')


if __name__ == '__main__':
    main()
