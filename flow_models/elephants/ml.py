import pathlib

import matplotlib.ticker
import numpy as np
from matplotlib import pyplot as plt

from flow_models.lib import mix
from flow_models.lib.data import load_data
from flow_models.lib.io import load_array_np

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

def make_slice(data, skip=0, count=None):

    sa, da, sp, dp, prot, oc = data
    if not count:
        count = len(sa)
    if isinstance(skip, float):
        skip = int(len(sa) * skip)
    if isinstance(count, float):
        count = int(len(sa) * count)
    s = slice(skip, skip + count)

    return sa[s], da[s], sp[s], dp[s], prot[s], oc[s]

def prepare_input(data, shuffle=False, to_octets=False, bit_vector=False):

    sa, da, sp, dp, prot, oc = data

    if to_octets:
        sa = sa.view(np.uint8).reshape(sa.shape + (sa.dtype.itemsize,)).T
        da = da.view(np.uint8).reshape(da.shape + (da.dtype.itemsize,)).T
        sp = sp.view(np.uint8).reshape(sp.shape + (sp.dtype.itemsize,)).T
        dp = dp.view(np.uint8).reshape(dp.shape + (dp.dtype.itemsize,)).T
        # sp = [sp]
        # dp = [dp]
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

    return inp, oc

def stats(oc, oc_predicted, thresholds=None):
    if thresholds is None:
        thresholds = np.power(2, range(25)) * 64
    elif isinstance(thresholds, int):
        thresholds = np.logspace(0, 24, thresholds, base=2) * 64
    else:
        thresholds = thresholds * 64

    r = []
    with np.errstate(divide='ignore'):
        for threshold in thresholds:
            decision = oc_predicted > threshold
            r.append([threshold, len(oc) / decision.sum(), oc[decision].sum() / oc.sum()])
    return np.array(r).T

def my_score(oc, oc_predicted):
    r = stats(oc, oc_predicted)
    red = interp_red([0.80], r[1], r[2])
    return red[1].mean() if np.isfinite(red).all() else np.nan

def interp_red(x, red, cov):
    if len(x) == 1:
        return cov, red
    else:
        y = np.interp(x, cov[::-1], red[::-1], left=np.nan, right=np.nan)
        return x, y

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
            plt.plot(cov * 100, red, label=name, lw=1, alpha=0.5)
            # x, y = interp_red(np.arange(0.8, 0.9, 0.01), red, cov)
            # plt.plot(x * 100, y, marker='.', markersize=5, linestyle='none', color='k')
    plt.xlim(*xlim[::-1])
    plt.ylim(*ylim)
    plt.gca().get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.legend()
    plt.title(title)
    plt.xlabel('Traffic coverage [%]')
    plt.ylabel('Flow table occupancy reduction [x]')

def calculate_from_mixture(path):
    data = list(load_data([path]).values())[0]
    index = 1 / np.logspace(0, 32, 1024, base=2)
    x = np.unique(np.rint(1 / np.array(index))).astype('u8')
    x *= 64
    reduction = 1 / (1 - mix.cdf(data['flows'], x))
    coverage = 1 - mix.cdf(data['octets'], x)
    mask = coverage > 0.5
    return x[mask], reduction[mask], coverage[mask]
