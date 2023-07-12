import pathlib

import matplotlib.ticker
import numpy as np
from matplotlib import pyplot as plt

from flow_models.lib import mix
from flow_models.lib.data import load_data
from flow_models.lib.io import load_array_np

def prepare_decision(oc, coverage):
    """
    Simulate mice/elephant decision to obtain a desired traffic coverage.

    Parameters
    ----------
    oc : numpy.array
        flow sizes (number of octets/bytes)
    coverage : float
        desired traffic coverage

    Returns
    -------
    np.array[bool]
        classification decision (0 for mice, 1 for elephants)
    """

    idx = np.flip(np.argsort(oc))
    idx_back = np.argsort(idx)

    oc_cumsum = oc[idx].cumsum()
    decision = oc_cumsum < oc_cumsum[-1] * coverage
    decision = decision[idx_back]

    return decision

def load_arrays(directory):
    """
    Load 5-tuple and flow sizes arrays from a directory.

    Parameters
    ----------
    directory : os.PathLike
        direcotry containing binary flow records

    Returns
    -------
    (np.array, np.array, np.array, np.array, np.array, np.array)
        sa, da, sp, dp, prot, oc
    """

    d = pathlib.Path(directory)
    _, _, sa = load_array_np(d / 'sa3')
    _, _, da = load_array_np(d / 'da3')
    _, _, sp = load_array_np(d / 'sp')
    _, _, dp = load_array_np(d / 'dp')
    _, _, prot = load_array_np(d / 'prot')
    _, _, oc = load_array_np(d / 'octets')

    return sa, da, sp, dp, prot, oc

def make_slice(data, skip=0, count=None):
    """
    Make slice of 5-tuple and flow size arrays.

    Parameters
    ----------
    data : (np.array, np.array, np.array, np.array, np.array, np.array)
        input data: (sa, da, sp, dp, prot, oc)
    skip : int, default 0
        number of flows to skip at the beggining
    count: int, optional
        number of flows to use after skipping

    Returns
    -------
    (np.array, np.array, np.array, np.array, np.array, np.array)
        sa, da, sp, dp, prot, oc
    """

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
    """
    Prepare input features array and target (flow sizes) array.

    Parameters
    ----------
    data : (np.array, np.array, np.array, np.array, np.array, np.array)
        input data: (sa, da, sp, dp, prot, oc)
    shuffle : bool, default False
        shuffle order of flows randomly
    to_octets : bool, default False
        split IP addresses to separate 1-byte octets
    bit_vector : bool, default False
        split all input fields to separate bits

    Returns
    -------
    (np.array[5], np.array)
        input features array, target value (flow sizes) array
    """

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

def calculate_reduction(oc, oc_predicted, thresholds=None):
    """
    Calculate flow reduction curve.

    Parameters
    ----------
    oc : numpy.array
        real flow sizes (number of octets/bytes)
    oc_predicted : numpy.array
        predicted flow sizes (number of octets/bytes)
    thresholds: int | numpy.array, optional
        flow size thresholds to calculate upon or their number, default [2..2^24]

    Returns
    -------
    np.array[3]
        [threshold, flow_table_reduction, traffic_coverage] array containing flow
        table size reduction and traffic coverage obtained for each flow size threshold
    """

    if thresholds is None:
        thresholds = np.power(2, range(25)) * 64
    elif isinstance(thresholds, int):
        thresholds = np.logspace(0, 24, thresholds, base=2) * 64
    else:
        thresholds = thresholds * 64

    res = []
    with np.errstate(divide='ignore'):
        for threshold in thresholds:
            decision = oc_predicted > threshold
            res.append([threshold, len(oc) / decision.sum(), oc[decision].sum() / oc.sum()])
    return np.array(res).T

def calculate_reduction_from_mixture(path):
    """
    Calculate flow reduction curve from distribution mixture JSON.

    Parameters
    ----------
    path : os.PathLike
        path to a directory with JSON mixture

    Returns
    -------
    np.array[3]
        [threshold, flow_table_reduction, traffic_coverage] array containing flow
        table size reduction and traffic coverage obtained for each flow size threshold
    """

    data = list(load_data([path]).values())[0]
    index = 1 / np.logspace(0, 32, 1024, base=2)
    x = np.unique(np.rint(1 / np.array(index))).astype('u8')
    x *= 64
    reduction = 1 / (1 - mix.cdf(data['flows'], x))
    coverage = 1 - mix.cdf(data['octets'], x)
    mask = coverage > 0.4
    return np.array([x[mask], reduction[mask], coverage[mask]])

def interp_reduction(x, flow_table_reduction, traffic_coverage):
    """
    Interpolate flow reduction curve for given traffic coverages.

    Parameters
    ----------
    x : numpy.array
        traffic coverage point to interpolate upon
    flow_table_reduction : numpy.array
        calculated flow table size reduction
    traffic_coverage : numpy.array
        traffic coverage corresponding to the flow table size reduction above

    Returns
    -------
    (numpy.array, numpy.array)
        x, traffic_coverage_for_x
    """

    y = np.interp(x, traffic_coverage[::-1], flow_table_reduction[::-1], left=np.nan, right=np.nan)
    return x, y

def score_reduction(oc, oc_predicted):
    """
    Calculate average flow table size reduction for 80% traffic coverage.

    Parameters
    ----------
    oc : numpy.array
        real flow sizes (number of octets/bytes)
    oc_predicted : numpy.array
        predicted flow sizes (number of octets/bytes)

    Returns
    -------
    float
        average flow table size reduction for 80% traffic coverage
    """

    threshold, flow_table_reduction, traffic_coverage = calculate_reduction(oc, oc_predicted)
    _, y = interp_reduction([0.80], flow_table_reduction, traffic_coverage)
    return y.mean() if np.isfinite(y).all() else np.nan

def plot(r, title=None):
    """
    Plot reduction all reduction curves in a dictionary.

    Parameters
    ----------
    r : dict[name, numpy.array[threshold, flow_table_reduction, traffic_coverage]]
        dictionary containing reduction curves and their names
    title: str, optional
        plot title
    """

    plt.figure(figsize=(10, 5))
    plt.yscale('log')
    for name, (_, red, cov) in r.items():
        color = 'k'
        if name == 'Mixture':
            plt.plot(cov * 100, red, label=name, lw=1, color=color)
        else:
            for w in name.split()[::-1]:
                try:
                    color = f'C{int(w)}'
                    break
                except ValueError:
                    pass
            plt.plot(cov * 100, red, label=name, lw=1, alpha=0.5, color=color, linestyle='-')
            # x, y = interp_red(np.arange(0.8, 0.9, 0.01), red, cov)
            # plt.plot(x * 100, y, marker='.', markersize=5, linestyle='none', color='k')
    plt.xlim(100, 50)
    plt.ylim(1, 10000)
    plt.gca().get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.title(title)
    plt.xlabel('Traffic coverage [%]')
    plt.ylabel('Flow table occupancy reduction [x]')
