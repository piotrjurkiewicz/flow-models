import pathlib

import matplotlib.ticker
import numpy as np

from matplotlib import pyplot as plt

from . import mix
from .data import load_data
from .io import load_array_np


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

def top_idx(octets, ratio, seed=None):
    """
    Get indices of the largest flows.

    Parameters
    ----------
    octets : numpy.array
        flow sizes (number of octets/bytes)
    ratio : float
        percentage of largest flows
    seed: int, default None
        seed for random generator

    Returns
    -------
    np.array[int]
        indices of the largest flows
    """

    rng = np.random.default_rng(seed)
    size = int(len(octets) * ratio / 2)
    idx = np.flip(np.argsort(octets))
    idx_rest = rng.choice(idx[size:], size, replace=False)

    return rng.permutation(np.concatenate([idx[:size], idx_rest]))

def top_split(all_input, all_octets, n_splits, ratio):
    import sklearn.model_selection
    for train_index, test_index in sklearn.model_selection.KFold(n_splits).split(all_input, all_octets):
        train_octets = all_octets[train_index]
        train_index = train_index[top_idx(train_octets, ratio)]
        yield train_index, test_index

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

def prepare_input(data, octets=False, bits=False):
    """
    Prepare input features array and target (flow sizes) array.

    Parameters
    ----------
    data : (np.array, np.array, np.array, np.array, np.array)
        input data: (sa, da, sp, dp, prot)
    octets : bool, default False
        split IP addresses to separate 1-byte octets
    bits : bool, default False
        split all input fields to separate bits

    Returns
    -------
    np.array[5]
        input features array
    """

    sa, da, sp, dp, prot = data

    if octets:
        sa = sa.view(np.uint8).reshape((*sa.shape, sa.dtype.itemsize)).T
        da = da.view(np.uint8).reshape((*da.shape, da.dtype.itemsize)).T
        sp = sp.view(np.uint8).reshape((*sp.shape, sp.dtype.itemsize)).T
        dp = dp.view(np.uint8).reshape((*dp.shape, dp.dtype.itemsize)).T
    else:
        sa = [sa]
        da = [da]
        sp = [sp]
        dp = [dp]

    columns = (*sp, *sa, *da, *dp, prot)

    if bits:
        cols = []
        for col in columns:
            cols.append(col.view(np.uint8).reshape((*col.shape, col.dtype.itemsize)))
        inp = np.column_stack(cols)
        inp = np.unpackbits(inp, axis=1)
    else:
        inp = np.column_stack(columns)

    return inp

def calculate_reduction(octets, octets_predicted, thresholds=None):
    """
    Calculate flow reduction curve.

    Parameters
    ----------
    octets : numpy.array
        real flow sizes (number of octets/bytes)
    octets_predicted : numpy.array
        predicted flow sizes (number of octets/bytes)
    thresholds: int | numpy.array, optional
        flow size thresholds to calculate upon or their number, default [2..2^24]

    Returns
    -------
    np.array[3]
        [threshold, traffic_coverage, flow_table_reduction] array containing flow
        table size reduction and traffic coverage obtained for each flow size threshold
    """

    if thresholds is None:
        thresholds = np.power(2, range(25)) * 64
    elif isinstance(thresholds, int):
        thresholds = np.logspace(0, 24, thresholds, base=2) * 64
    else:
        thresholds = thresholds * 64

    res = []
    for threshold in thresholds:
        decision = octets_predicted > threshold
        if decision.sum() > 0:
            res.append([threshold, octets[decision].sum() / octets.sum(), len(octets) / decision.sum()])
    res = np.array(res).T
    keys = np.argsort(res[1])
    return res[:, keys]

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

    data = next(iter(load_data([path]).values()))
    index = 1 / np.logspace(0, 32, 1024, base=2)
    x = np.unique(np.rint(1 / np.array(index))).astype('u8')
    x *= 64
    reduction = 1 / (1 - mix.cdf(data['flows'], x))
    coverage = 1 - mix.cdf(data['octets'], x)
    mask = coverage > 0.4
    return np.array([x[mask][::-1], coverage[mask][::-1], reduction[mask][::-1]])

def interp_reduction(x, traffic_coverage, flow_table_reduction):
    """
    Interpolate flow reduction curve for given traffic coverages.

    Parameters
    ----------
    x : numpy.array
        traffic coverage point to interpolate upon
    traffic_coverage : numpy.array
        traffic coverages corresponding to the input flow table size reductions
    flow_table_reduction : numpy.array
        input flow table size reductions

    Returns
    -------
    (numpy.array, numpy.array)
        x, flow_table_reduction_for_x
    """

    y = np.interp(x, traffic_coverage, flow_table_reduction, left=np.nan)
    return x, y

def score_reduction(octets, octets_predicted):
    """
    Calculate average flow table size reduction for 80% traffic coverage.

    Parameters
    ----------
    octets : numpy.array
        real flow sizes (number of octets/bytes)
    octets_predicted : numpy.array
        predicted flow sizes (number of octets/bytes)

    Returns
    -------
    float
        average flow table size reduction for 80% traffic coverage
    """

    threshold, traffic_coverage, flow_table_reduction = calculate_reduction(octets, octets_predicted)
    _, y = interp_reduction([0.80], traffic_coverage, flow_table_reduction)
    return y.mean() if np.isfinite(y).all() else np.nan

def plot_style():
    plt.yscale('log')
    plt.xlim(50, 100)
    plt.ylim(1, 10000)
    plt.gca().get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xlabel('Traffic coverage [%]')
    plt.ylabel('Flow table operations reduction [x]')

def dask_upload_package(client, name):
    from importlib.util import find_spec
    from pathlib import Path
    from shutil import make_archive
    from tempfile import TemporaryDirectory
    spec = find_spec('flow_models')
    path = Path(spec.origin).parent.parent
    with TemporaryDirectory() as tmpdirname:
        make_archive(f'{tmpdirname}/{name}', 'zip', path, name)
        client.upload_file(f'{tmpdirname}/{name}.zip')
