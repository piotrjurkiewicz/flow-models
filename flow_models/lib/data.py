import json
import pathlib

import numpy as np
import pandas as pd
import scipy
import scipy.interpolate
import scipy.stats

from .kde import gaussian_kde
from .util import bin_calc_one, bin_calc_log, logmsg

UNITS = {
    'length': 'packets',
    'size': 'bytes',
    'duration': 'milliseconds',
    'rate': 'bps'
}

STYLE = {
    'flows': ('r', 'Reds'),
    'packets': ('g', 'Greens'),
    'octets': ('b', 'Blues')
}

SEED = 0
LINE_NBINS = 128
HIST_NBINS = 64
KDE_NBINS = 64

def normalize_data(org_data, bin_exp=None):
    bin_widths = org_data['bin_hi'] - org_data.index.values
    if bin_widths.min() == bin_widths.max() == 1:
        bin_calc = bin_calc_one
        logmsg('Using bin_calc_one')
    else:
        bin_calc = bin_calc_log
        if bin_exp is None:
            bin_exp = int(np.log2(bin_widths[bin_widths > 1].index[0]))
        logmsg('Using bin_calc_log with exponent', bin_exp)

    data = org_data
    bna = np.diff(data.index.values)
    bin_next = np.append(bna, bna[-1]) + data.index.values
    bin_next[-1] = bin_calc(bin_next[-1], bin_exp)[0]

    bh = data['bin_hi']
    data = data.div(bin_next - data.index, axis=0)
    data['bin_hi'] = bh
    return data

def detect_x_value(x):
    if x.min() == 1:
        x_value = 'length'
    elif x.min() == 64:
        x_value = 'size'
    else:
        raise ValueError
    return x_value

def log_gaussian_kde(x, y, xmin=None, xmax=None, ymin=None, ymax=None, nbins=KDE_NBINS, weights=None, fft=False):
    xmin = np.log10(xmin if xmin else x.min())
    xmax = np.log10(xmax if xmax else x.max())

    ymin = np.log10(ymin if ymin else y.min())
    ymax = np.log10(ymax if ymax else y.max())

    x = np.log10(x)
    y = np.log10(y)

    if fft:
        kernel = gaussian_kde([x, y], weights=weights, fft=True)
    else:
        if [int(x) for x in scipy.version.version.split('.')] >= [int(x) for x in '1.3.0'.split('.')]:
            kernel = scipy.stats.gaussian_kde([x, y], weights=weights)
        else:
            kernel = gaussian_kde([x, y], weights=weights, fft=False)

    xi, yi = np.mgrid[xmin:xmax:nbins * 1j, ymin:ymax:nbins * 1j]
    zi = kernel(np.vstack([xi.flatten(), yi.flatten()]))
    return np.power(10, xi), np.power(10, yi), zi.reshape(xi.shape)

def calc_minmax(idx, *rest):
    if isinstance(idx, (pd.Series, pd.DataFrame)):
        xmin, xmax = idx.index.min(), idx.index.max()
        ymin, ymax = idx.min(), idx.max()
    else:
        xmin, xmax = idx.min(), idx.max()
        ymin, ymax = float('inf'), -float('inf')

    for r in rest:
        if r is not None:
            if isinstance(r, (pd.Series, pd.DataFrame)):
                xmin = min(xmin, r.index.min())
                xmax = max(xmax, r.index.max())
            ymin = min(ymin, r.min())
            ymax = max(ymax, r.max())

    return xmin, xmax, ymin, ymax

def avg_data(data, idx, what):
    avg_points = data[what + '_sum'] / data['flows_sum']
    scale = data[what + '_sum'].sum() / data['flows_sum'].sum()
    xmin, xmax, _, _ = calc_minmax(avg_points)
    pdf_what = pdf_from_cdf(data, idx, what)
    pdf_flows = pdf_from_cdf(data, idx, 'flows')
    avg_line = scale * pdf_what / pdf_flows
    return avg_points, avg_line

def pdf_from_cdf(data, idx, what):
    cdf = data[what + '_sum'].cumsum() / data[what + '_sum'].sum()
    cdfi = scipy.interpolate.interp1d(cdf.index, cdf, 'linear', bounds_error=False)(idx)
    pdfi = np.hstack((cdfi[0], np.diff(cdfi) / np.diff(idx)))
    return pdfi

def load_data(objects):
    data = {}
    for obj in objects:
        if isinstance(obj, (str, pathlib.Path)):
            file = pathlib.Path(obj)
            logmsg(f'Loading file {file}')
            if file.suffix == '.json':
                data[file] = json.load(open(file))
            elif file.is_dir():
                mixtures = {}
                for ff in file.glob('*.json'):
                    mixtures[ff.stem] = json.load(open(str(ff)))
                data[file] = mixtures
            else:
                data[file] = pd.read_csv(file, index_col=0, sep=',', low_memory=False,
                                         usecols=lambda col: not col.endswith('_ssq'))
            logmsg(f'Loaded file {file}')
        else:
            data[id(obj)] = obj
    return data
