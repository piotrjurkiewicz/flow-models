import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.interpolate
import scipy.stats

from .kde import gaussian_kde
from .mix import pdf, cdf, pdf_components, cdf_components
from .util import bin_calc_one, bin_calc_log, logmsg

UNITS = {
    'length': 'packets',
    'size': 'bytes'
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

def set_log_limits(xmin, xmax, ymin, ymax):
    xmar, ymar = plt.margins()
    # print(xmin, ymin)
    xbuf, ybuf = xmar * np.log10(xmax / xmin), ymar * np.log10(ymax / ymin)
    plt.xlim(xmin * 10 ** (-xbuf), xmax * 10 ** xbuf)
    plt.ylim(ymin * 10 ** (-ybuf), ymax * 10 ** ybuf)

def plot_mixture(data, idx, what, mode, fun):
    # TODO: move to mix
    if isinstance(data, dict):
        data = data[what]
    if 'mixture' in mode:
        fit = []
        label = 'mixture'
        if fun == 'cdf':
            fit = cdf(data[1], idx)
            label = what + ' mixture'
        if fun == 'pdf':
            fit = pdf(data[1], idx)

        plt.plot(idx, fit, 'k-', lw=1, ms=1, alpha=0.5, label=label)

    if 'components' in mode:
        components = {}
        if fun == 'cdf':
            components = cdf_components(data[1], idx)
        if fun == 'pdf':
            components = pdf_components(data[1], idx)
        for weight, dd in components.items():
            plt.plot(idx, dd, lw=0.1, label=str(weight))

    if 'components_stack' in mode:
        components = {}
        if fun == 'cdf':
            components = cdf_components(data[1], idx)
        if fun == 'pdf':
            components = pdf_components(data[1], idx)
        plt.stackplot(idx, *components.values(), lw=0.1, labels=components.keys(), alpha=0.5)

def plot_pdf(data, idx=None, what='flows', mode=frozenset(['points', 'mixture']), normalize=True, fft=False):
    kwargs = {'cmap': STYLE[what][1]}

    if isinstance(data, pd.DataFrame):

        if idx is None:
            idx = np.unique(np.rint(np.geomspace(data.index.min(), data.index.max(), LINE_NBINS)).astype(int))

        data = data.loc[:, 'bin_hi':what + '_sum']

        cdf = data[what + '_sum'].cumsum() / data[what + '_sum'].sum()
        cdfi = scipy.interpolate.interp1d(cdf.index, cdf, 'linear', bounds_error=False)(idx)
        pdfi = np.hstack((cdfi[0], np.diff(cdfi) / np.diff(idx)))
        xmin, xmax, ymin, ymax = calc_minmax(idx, pdfi)

        if 'line' in mode:
            plt.plot(idx, pdfi, STYLE[what][0] + '-', lw=2, ms=1, alpha=0.5,
                     label='data')

        values_sum = data[what + '_sum'].sum()
        weights = data['flows_sum'].values

        if normalize:
            logmsg('Normalizing data')
            data = normalize_data(data)
            logmsg('Normalizing data')

        pdf = data[what + '_sum'] / values_sum

        if 'points' in mode:
            logmsg('Plotting points')
            plt.plot(pdf.index, pdf,
                     STYLE[what][0] + ',', lw=1, ms=1, alpha=1,
                     label=what + ' data', rasterized=True)

        if 'hist' in mode:
            logmsg('Plotting hist')
            plt.hist2d(pdf.index, pdf, weights=weights,
                       bins=[np.geomspace(xmin, xmax, HIST_NBINS), np.geomspace(ymin, ymax, HIST_NBINS)],
                       cmin=1, norm=colors.LogNorm(),
                       alpha=1.0, label=what + ' data', rasterized=False, **kwargs)

        if 'kde' in mode:
            logmsg('Calculating KDE')
            xi, yi, zi = log_gaussian_kde(pdf.index, pdf, xmin, xmax, ymin, ymax, KDE_NBINS, weights=weights, fft=fft)
            zn = colors.PowerNorm(0.1)(zi)
            logmsg('Plotting KDE')
            # plt.pcolormesh(xi, yi, zn, shading='gouraud', rasterized=True)
            plt.contourf(xi, yi, zn, levels=16, antialiased=1, alpha=0.5, **kwargs).collections[0].set_alpha(0)
            # plt.contour(xi, yi, zn, antialiased=1, linewidths=0.25, colors='black').collections[0].set_alpha(0)

        set_log_limits(xmin, xmax, ymin, ymax)

    else:
        plot_mixture(data, idx, what, mode, 'pdf')

    plt.xscale('log')
    plt.yscale('log')
    plt.legend(frameon=False)

def plot_cdf(data, idx=None, what='flows', mode=frozenset(['mixture'])):
    if isinstance(data, pd.DataFrame):

        if idx is None:
            idx = np.unique(np.rint(np.geomspace(data.index.min(), data.index.max(), LINE_NBINS)).astype(int))

        cdf = data[what + '_sum'].cumsum() / data[what + '_sum'].sum()
        cdfi = scipy.interpolate.interp1d(cdf.index, cdf, 'linear', bounds_error=False)(idx)
        plt.plot(idx, cdfi, STYLE[what][0] + '-', lw=2, ms=1, alpha=0.5,
                 label=what + ' data')

    else:
        plot_mixture(data, idx, what, mode, 'cdf')

    plt.xscale('log')
    plt.legend(frameon=False)

def calc_avg(data, idx, what):
    avg_points = data[what + '_sum'] / data['flows_sum']
    scale = data[what + '_sum'].sum() / data['flows_sum'].sum()
    xmin, xmax, _, _ = calc_minmax(avg_points)

    cdf = data[what + '_sum'].cumsum() / data[what + '_sum'].sum()
    cdfi = scipy.interpolate.interp1d(cdf.index, cdf, 'linear', bounds_error=False)(idx)
    pdf_what = np.hstack((cdfi[0], np.diff(cdfi) / np.diff(idx)))

    cdf = data['flows_sum'].cumsum() / data['flows_sum'].sum()
    cdfi = scipy.interpolate.interp1d(cdf.index, cdf, 'linear', bounds_error=False)(idx)
    pdf_flows = np.hstack((cdfi[0], np.diff(cdfi) / np.diff(idx)))

    avg_line = scale * pdf_what / pdf_flows

    return avg_points, avg_line

def calc_avg_mix(data, idx, what):
    # TODO: move to mix
    avg_mix = (data[what][0] / data['flows'][0]) * pdf(data[what][1], idx) / pdf(data['flows'][1], idx)
    return avg_mix

def plot_avg(data, idx=None, what='packets', mode=frozenset(['mixture'])):
    if isinstance(data, pd.DataFrame):

        if idx is None:
            idx = np.unique(np.rint(np.geomspace(data.index.min(), data.index.max(), LINE_NBINS)).astype(int))

        if what in ['packets', 'octets']:
            avg_points, avg_line = calc_avg(data, idx, what)
            color = 'g' if what == 'packets' else 'b'
            plt.xscale('log')
            plt.yscale('log')
        elif what in ['packet_size', 'packet_iat']:
            if what == 'packet_size':
                num = 'octets'
            else:
                num = 'duration'
                plt.yscale('log')
            num_avg_points, num_avg_line = calc_avg(data, idx, num)
            packets_avg_points, packets_avg_line = calc_avg(data, idx, 'packets')
            avg_points = num_avg_points / packets_avg_points
            avg_line = num_avg_line / packets_avg_line
            color = 'm'
            mn, avg, mx = avg_points.min(), data[num + '_sum'].sum() / data['packets_sum'].sum(), avg_points.max()
            lab_x = 10 ** (np.log10(idx.max() - idx.min()) * 0.25 + np.log10(idx.min()))
            if what == 'packet_size':
                plt.axhline(y=mn, alpha=0.1)
                plt.text(lab_x, mn, f'min = {mn:.2f}', va='center', ha='center', backgroundcolor='w')
                plt.ylim(0, 2000)
            plt.axhline(y=avg, alpha=0.1)
            plt.text(lab_x, avg, f'avg = {avg:.2f}', va='center', ha='center', backgroundcolor='w')
            plt.axhline(y=mx, alpha=0.1)
            plt.text(lab_x, mx, f'max = {mx:.2f}', va='center', ha='center', backgroundcolor='w')
            plt.xscale('log')
        else:
            raise ValueError

        plt.plot(avg_points.index, avg_points,
                 color + ',', lw=1, ms=1, alpha=1,
                 label='data', rasterized=True)

        plt.plot(idx, avg_line, color + '-', lw=2, ms=1, alpha=0.5,
                 label='data')

    elif 'mixture' in mode:

        # TODO: move to mix
        if what in ['packets', 'octets']:
            avg_mix = calc_avg_mix(data, idx, what)
            plt.xscale('log')
            plt.yscale('log')
        elif what in ['packet_size', 'packet_iat']:
            if what == 'packet_size':
                num = 'octets'
            else:
                num = 'duration'
                plt.yscale('log')
            try:
                num_avg_mix = calc_avg_mix(data, idx, num)
            except KeyError:
                return
            packets_avg_mix = calc_avg_mix(data, idx, 'packets')
            avg_mix = num_avg_mix / packets_avg_mix
            plt.xscale('log')
        else:
            raise ValueError

        plt.plot(idx, avg_mix, 'k-', lw=2, ms=1, alpha=0.5,
                 label='infered from mixtures')

    plt.legend(frameon=False)
