#!/usr/bin/python3
"""Calculates flow reduction curves from mixture or histogram data for first, threshold and sampling methods.

The results are analytical expectations over the input flow size
distribution.  ``first`` assumes an above-threshold flow is added to the
flow table before its first packet (an oracle bound for first-packet
classifiers), ``threshold`` assumes insertion only after the threshold
count of packets or bytes has been observed, and ``sampling`` computes the
geometric expectation of packets missed before the first sampled packet —
for ``x_val='size'`` sampling operates on 64-byte units, giving the
per-packet detection probability ``1 - (1 - p) ** (packet_size / 64)``.

See :program:`simulate` for the Monte-Carlo equivalent and
:mod:`flow_models.elephants.simulate_data` for flow-level simulation on
real flow records.
"""

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

def packets_n_data(data, idx, x_val):
    """
    Estimate the conditional average number of packets per flow in each histogram bin.

    For ``x_val='length'`` the bin index is already the packet count.  For
    ``x_val='size'`` it is estimated as ``packets_sum / flows_sum`` and
    clipped to the physically possible range ``[1, size / 64]``.
    """
    if x_val == 'length':
        return idx.astype('f8')
    with np.errstate(divide='ignore', invalid='ignore'):
        n = data['packets_sum'].to_numpy() / data['flows_sum'].to_numpy()
    n = np.where(np.isfinite(n), n, 1.0)
    return np.clip(n, 1.0, np.trunc(idx / 64))

def calculate_data(data, x_probs, x_val, method, pps=None, timeout=15):
    """
    Calculate detection-method metrics from a csv_hist histogram DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        histogram indexed by flow length or size with ``flows_sum``,
        ``packets_sum`` and ``octets_sum`` columns
    x_probs : np.array
        reciprocals of the flow size thresholds or, for 'sampling', the
        sampling probabilities
    x_val : str
        x axis value, 'length' or 'size'
    method : str
        'first', 'threshold' or 'sampling'
    pps : float, optional
        assumed packets-per-second rate of each flow, makes ``occupancy_mean``
        the residency-second ratio instead of ``1/fraction``
    timeout : float, default 15
        inactive flow table timeout in seconds used by ``occupancy_mean``
        when ``pps`` is given

    Returns
    -------
    pd.DataFrame
        metrics indexed by flow size threshold or, for 'sampling', by
        sampling probability; NaN where no flow is covered
    """
    ad = {}
    x = np.unique(np.rint(1 / x_probs)).astype('u8')
    if x_val == 'size':
        x *= 64
    idx = data.index.to_numpy()
    idx_diff = np.concatenate([idx[:1], np.diff(idx)])

    if method == 'first':

        for what in ['flows', 'packets', 'fraction', 'octets']:
            w = 'flows' if what == 'fraction' else what
            cdf = data[w + '_sum'].cumsum() / data[w + '_sum'].sum()
            cdf = scipy.interpolate.interp1d(cdf.index, cdf, 'previous', bounds_error=False)(x)
            ad[what + '_mean'] = 1 - cdf

        if pps is not None:
            n = packets_n_data(data, idx, x_val)
            res = data['flows_sum'].to_numpy() * (np.floor(n / pps) + 1 + timeout)
            pos = np.searchsorted(idx, x, side='right')
            covered = np.concatenate([res[::-1].cumsum()[::-1], [0.0]])[pos]
            ad['occupancy_mean'] = res.sum() / covered

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

        if pps is not None:
            n = packets_n_data(data, idx, x_val)
            res_all = (data['flows_sum'].to_numpy() * (np.floor(n / pps) + 1 + timeout)).sum()
            missed_pk = x[:, None] if x_val == 'length' else x[:, None] * n / idx
            covered_pk = np.clip(n - missed_pk, 0, None)
            res = data['flows_sum'].to_numpy() * (idx > x[:, None]) * (np.floor(covered_pk / pps) + 1 + timeout)
            ad['occupancy_mean'] = res_all / res.sum(axis=1)

    else:

        if x_val == 'length':
            n = idx.astype('f8')
            unit = np.ones(len(idx))
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                n = data['packets_sum'].to_numpy() / data['flows_sum'].to_numpy()
            n = np.where(np.isfinite(n), n, 1.0)
            n = np.clip(n, 1.0, np.trunc(idx / 64))
            unit = idx / n / 64

        counts = {w: data[w + '_sum'].to_numpy() for w in ['flows', 'packets', 'octets']}
        if pps is not None:
            res_all = np.nansum(counts['flows'] * (np.floor(n / pps) + 1 + timeout))
        a = {what: [] for what in ['flows', 'packets', 'fraction', 'octets']}
        occ = []
        for p in x_probs:
            m = (1 - p) ** unit
            with np.errstate(divide='ignore', invalid='ignore'):
                missed = m * (1 - m ** n) / (1 - m)
            missed = np.minimum(np.where(m < 1, missed, n), n)
            det = 1 - m ** n
            cov = (n - missed) / n
            for what in a:
                w = 'flows' if what == 'fraction' else what
                share = det if what == 'flows' else cov
                a[what].append(np.nansum(counts[w] * share) / counts[w].sum())
            if pps is not None:
                res = counts['flows'] * (np.floor((n - missed) / pps) + det * (1 + timeout))
                occ.append(res_all / np.nansum(res))
        for what, v in a.items():
            ad[what + '_mean'] = np.array(v)
        if pps is not None:
            ad['occupancy_mean'] = np.array(occ)

    ad['operations_mean'] = 1 / ad['flows_mean']
    ad.setdefault('occupancy_mean', 1 / ad['fraction_mean'])
    for what in ['flows', 'packets', 'fraction', 'octets']:
        ad[what + '_mean'] *= 100

    return pd.DataFrame(ad, x_probs if method == 'sampling' else x).replace(np.inf, np.nan)

def packets_n_mix(data, idx, x_val):
    """
    Estimate the conditional average number of packets per flow at each mixture grid point.

    For ``x_val='length'`` the grid value is already the packet count.  For
    ``x_val='size'`` it is estimated with the conditional expectation
    :func:`flow_models.lib.mix.avg` and clipped to the physically possible
    range ``[1, size / 64]``.
    """
    if x_val == 'length':
        return idx.astype('f8')
    n = mix.avg(data, idx, x_val, 'packets')
    n = np.where(np.isfinite(n), n, 1.0)
    return np.clip(n, 1.0, np.trunc(idx / 64))

def pdf_mix(data, w, idx):
    """
    Mixture probability mass of field ``w`` discretized on grid ``idx``.

    Returns differences of the mixture CDF, with the probability mass
    beyond the last grid point folded into the last bin so that the result
    sums to 1.
    """
    cdf = mix.cdf(data[w], idx)
    pdf = np.concatenate([cdf[:1], np.diff(cdf)])
    pdf[-1] += 1 - cdf[-1]
    return pdf

def calculate_mix(data, x_probs, x_val, method, pps=None, timeout=15):
    """
    Calculate detection-method metrics from a fitted mixture model.

    Parameters
    ----------
    data : dict
        mixture model as loaded by :func:`flow_models.generate.load_data`
    x_probs : np.array
        reciprocals of the flow size thresholds or, for 'sampling', the
        sampling probabilities
    x_val : str
        x axis value, 'length' or 'size'
    method : str
        'first', 'threshold' or 'sampling'
    pps : float, optional
        assumed packets-per-second rate of each flow, makes ``occupancy_mean``
        the residency-second ratio instead of ``1/fraction``
    timeout : float, default 15
        inactive flow table timeout in seconds used by ``occupancy_mean``
        when ``pps`` is given

    Returns
    -------
    pd.DataFrame
        metrics indexed by flow size threshold or, for 'sampling', by
        sampling probability; NaN where no flow is covered
    """
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

        if pps is not None:
            n = packets_n_mix(data, idx, x_val)
            res = pdf_mix(data, 'flows', idx) * (np.floor(n / pps) + 1 + timeout)
            pos = np.searchsorted(idx, x, side='right')
            covered = np.concatenate([res[::-1].cumsum()[::-1], [0.0]])[pos]
            ad['occupancy_mean'] = res.sum() / covered

    elif method == 'threshold':

        for what in ['flows', 'packets', 'fraction', 'octets']:
            w = 'flows' if what == 'fraction' else what
            if what == 'flows':
                cdf = mix.cdf(data[w], x)
                ad[what + '_mean'] = 1 - cdf
            else:
                pdf = pdf_mix(data, w, idx)
                toc = (pdf / idx)[::-1].cumsum()[::-1] * idx_diff
                cdf = 1 - toc.cumsum()
                ad[what + '_mean'] = scipy.interpolate.interp1d(idx, cdf, 'linear', bounds_error=False)(x)

        if pps is not None:
            n = packets_n_mix(data, idx, x_val)
            pdf_f = pdf_mix(data, 'flows', idx)
            res_all = (pdf_f * (np.floor(n / pps) + 1 + timeout)).sum()
            missed_pk = x[:, None] if x_val == 'length' else x[:, None] * n / idx
            covered_pk = np.clip(n - missed_pk, 0, None)
            res = pdf_f * (idx > x[:, None]) * (np.floor(covered_pk / pps) + 1 + timeout)
            ad['occupancy_mean'] = res_all / res.sum(axis=1)

    else:

        if x_val == 'length':

            n = idx.astype('f8')
            unit = np.ones(len(idx))
        else:

            n = packets_n_mix(data, idx, x_val)
            unit = idx / n / 64

        pdfs = {w: pdf_mix(data, w, idx) for w in ['flows', 'packets', 'octets']}
        if pps is not None:
            res_all = (pdfs['flows'] * (np.floor(n / pps) + 1 + timeout)).sum()
        a = {what: [] for what in ['flows', 'packets', 'fraction', 'octets']}
        occ = []
        for p in x_probs:
            m = (1 - p) ** unit
            with np.errstate(divide='ignore', invalid='ignore'):
                missed = m * (1 - m ** n) / (1 - m)
            missed = np.minimum(np.where(m < 1, missed, n), n)
            det = 1 - m ** n
            cov = (n - missed) / n
            for what in a:
                w = 'flows' if what == 'fraction' else what
                share = det if what == 'flows' else cov
                a[what].append((pdfs[w] * share).sum())
            if pps is not None:
                res = pdfs['flows'] * (np.floor((n - missed) / pps) + det * (1 + timeout))
                occ.append(res_all / res.sum())
        for what, v in a.items():
            ad[what + '_mean'] = np.array(v)
        if pps is not None:
            ad['occupancy_mean'] = np.array(occ)

    ad['operations_mean'] = 1 / ad['flows_mean']
    ad.setdefault('occupancy_mean', 1 / ad['fraction_mean'])
    for what in ['flows', 'packets', 'fraction', 'octets']:
        ad[what + '_mean'] *= 100

    return pd.DataFrame(ad, x_probs if method == 'sampling' else x).replace(np.inf, np.nan)

def calculate(obj, index=None, x_val='length', methods=tuple(METHODS), pps=None, timeout=15):
    """
    Calculate flow table occupancy reduction curve for given traffic coverages.

    Parameters
    ----------
    obj : os.PathLike
        csv_hist histogram file or mixture directory
    index : int | numpy.array, optional
        flow size thresholds (or sampling probabilities for the 'sampling'
        method) to calculate upon or their number, default [2..2^24]
    x_val : str, default 'length'
        x axis value
    methods : list, ['first', 'threshold', 'sampling']
        methods to calculate
    pps : float, optional
        assumed packets-per-second rate of each flow, makes the ``occupancy``
        metric residency-based
    timeout : float, default 15
        inactive flow table timeout in seconds for the residency-based
        ``occupancy`` metric

    Returns
    -------
    dict[str, pd.DataFrame]
        calculated DataFrame for each method, indexed by flow size threshold
        or, for 'sampling', by sampling probability.  Columns are
        ``{flows, packets, fraction, octets}_mean``: percentages of flows
        added, of packets and octets transmitted after insertion (covered),
        and the mean per-flow covered packet fraction; plus the reductions
        ``operations_mean = 1/flows`` and ``occupancy_mean``.  Without
        ``pps``, ``occupancy_mean = 1/fraction`` is a time-free bound which
        assumes a flow's table residency is proportional to its covered
        packet count and models neither flow duration nor inactivity
        timeout.  When ``pps`` is given, ``occupancy_mean`` becomes the
        ratio of total to covered residency-second estimates
        ``floor(packets / pps) + 1 + timeout``, directly comparable to the
        ``flows_slots`` ratio of
        :func:`flow_models.elephants.simulate_data.simulate_data` run in
        pps mode
    """

    data = next(iter(load_data([obj]).values()))

    if index is None:
        index = 1 / np.power(2, range(25))
    elif isinstance(index, int):
        index = 1 / np.logspace(0, 32, index, base=2)

    dataframes = {}
    for method in methods:
        if isinstance(data, pd.DataFrame):
            df = calculate_data(data, np.array(index), x_val, method, pps, timeout)
        else:
            df = calculate_mix(data, np.array(index), x_val, method, pps, timeout)
        dataframes[method] = df

    return dataframes

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-n', type=int, default=None, help='number of index points')
    p.add_argument('-x', default='length', choices=X_VALUES, help='x axis value')
    p.add_argument('-m', default='all', choices=[*METHODS, 'all'], help='method')
    p.add_argument('-p', '--pps', type=float, default=None,
                   help='packets per second per flow, makes the occupancy metric residency-based')
    p.add_argument('-t', '--timeout', type=float, default=15, help='inactive flow table timeout in seconds')
    p.add_argument('--save', action='store_true', help='save to files')
    p.add_argument('file', help='csv_hist file or mixture directory')
    return p

def main():
    app_args = parser().parse_args()

    if app_args.m == 'all':
        methods = METHODS
    else:
        methods = [app_args.m]

    resdic = calculate(app_args.file, app_args.n, app_args.x, methods, app_args.pps, app_args.timeout)
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
