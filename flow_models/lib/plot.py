import contextlib

import matplotlib
import numpy as np
import pandas as pd
import scipy
import scipy.interpolate
from matplotlib import pyplot as plt, colors as colors

from flow_models.lib.data import STYLE, LINE_NBINS, calc_minmax, normalize_data, HIST_NBINS, log_gaussian_kde, \
    KDE_NBINS, avg_data, pdf_from_cdf
from flow_models.lib.mix import cdf, pdf, cdf_comp, pdf_comp, avg
from flow_models.lib.util import logmsg

PDF_NONE = {'Creator': None, 'Producer': None, 'CreationDate': None}
MODES_MIXTURE = ['comp', 'comp_stack', 'comp_labels']
MODES_PDF = ['points', 'hist', 'kde'] + MODES_MIXTURE
MODES_CDF = MODES_MIXTURE

def save_figure(figure, fname, ext='pdf', **kwargs):
    figure.savefig(f'{fname}.{ext}', bbox_inches='tight', metadata=PDF_NONE, **kwargs)

@contextlib.contextmanager
def matplotlib_config(latex=False):
    try:
        matplotlib.rcParams['figure.dpi'] *= 2
        matplotlib.rcParams['figure.subplot.hspace'] = 0
        matplotlib.rcParams['figure.subplot.wspace'] /= 1.5
        matplotlib.rcParams['figure.subplot.left'] = 0.00
        matplotlib.rcParams['figure.subplot.bottom'] = 0.00
        matplotlib.rcParams['figure.subplot.right'] = 1.00
        matplotlib.rcParams['figure.subplot.top'] = 1.00
        matplotlib.rcParams['xtick.major.width'] = 0.25
        matplotlib.rcParams['xtick.minor.width'] = 0.25
        matplotlib.rcParams['xtick.top'] = True
        matplotlib.rcParams['xtick.direction'] = 'in'
        matplotlib.rcParams['ytick.major.width'] = 0.25
        matplotlib.rcParams['ytick.minor.width'] = 0.25
        matplotlib.rcParams['ytick.right'] = True
        matplotlib.rcParams['ytick.direction'] = 'in'
        matplotlib.rcParams['legend.frameon'] = False
        matplotlib.rcParams['errorbar.capsize'] = 2
        matplotlib.rcParams['axes.xmargin'] = 0.05
        matplotlib.rcParams['axes.ymargin'] = 0.05
        matplotlib.rcParams['axes.linewidth'] = 0.25
        matplotlib.rcParams['pdf.use14corefonts'] = True
        matplotlib.rcParams['font.family'] = 'sans'
        matplotlib.rcParams['font.size'] *= 1.15

        if latex:
            matplotlib.rcParams['text.usetex'] = True
            matplotlib.rcParams['font.family'] = 'sans-serif'
            # matplotlib.rcParams['mathtext.fontset'] = 'stix'

            # matplotlib.rcParams['text.latex.preamble'] = r'''
            # \usepackage[notextcomp]{stix}
            # '''
        yield
    finally:
        matplotlib.rcdefaults()

def set_log_limits(xmin, xmax, ymin, ymax):
    xmar, ymar = plt.margins()
    # print(xmin, ymin)
    xbuf, ybuf = xmar * np.log10(xmax / xmin), ymar * np.log10(ymax / ymin)
    plt.xlim(xmin * 10 ** (-xbuf), xmax * 10 ** xbuf)
    plt.ylim(ymin * 10 ** (-ybuf), ymax * 10 ** ybuf)

def plot_mixture(data, idx, x_val, what, mode, fun):
    plots = []
    if isinstance(data, dict):
        data = data.get(what)
        if data is None:
            return plots
    if 'mixture' in mode:
        fit = []
        label = 'model'
        if fun == 'cdf':
            fit = cdf(data, idx)
            label = what + ' (model)'
        if fun == 'pdf':
            fit = pdf(data, idx, x_val)

        plots += plt.plot(idx, fit, 'k-', lw=1, ms=1, alpha=0.5, label=label)

    if 'comp' in mode:
        comp = {}
        if fun == 'cdf':
            comp = cdf_comp(data, idx)
        if fun == 'pdf':
            comp = pdf_comp(data, idx, x_val)
        for weight, dd in comp.items():
            plots += plt.plot(idx, dd, '--', dashes=(20, 20), lw=0.1, label=str(weight) if 'comp_labels' in mode else None)

    if 'comp_stack' in mode:
        comp = {}
        if fun == 'cdf':
            comp = cdf_comp(data, idx)
        if fun == 'pdf':
            comp = pdf_comp(data, idx, x_val)
        plots += plt.stackplot(idx, *comp.values(), lw=0.1, labels=comp.keys() if 'comp_labels' in mode else [], alpha=0.5)
    return plots

def plot_pdf(data, idx=None, x_val='length', what='flows', mode=frozenset(['points', 'mixture']), normalize=True, fft=False):
    kwargs = {'cmap': STYLE[what][1]}
    plots = []

    if isinstance(data, pd.DataFrame):
        if idx is None:
            idx = np.unique(np.rint(np.geomspace(data.index.min(), data.index.max(), LINE_NBINS)).astype(int))

        data = data.loc[:, 'bin_hi':what + '_sum']

        pdfi = pdf_from_cdf(data, idx, what)
        xmin, xmax, ymin, ymax = calc_minmax(idx, pdfi)

        if 'line' in mode:
            plots += plt.plot(idx, pdfi, STYLE[what][0] + '-', lw=2, ms=1, alpha=0.5, zorder=3,
                              label='infered from data points')

        values_sum = data[what + '_sum'].sum()
        weights = data['flows_sum'].values

        if normalize:
            logmsg('Normalizing data')
            data = normalize_data(data)
            logmsg('Normalizing data')

        pdfd = data[what + '_sum'] / values_sum

        if 'points' in mode:
            logmsg('Plotting points')
            if normalize:
                plots += plt.plot(pdfd.index, pdfd,
                                  STYLE[what][0] + ',', lw=1, ms=1, alpha=1, zorder=1,
                                  label='data points (normalized)', rasterized=True)
            else:
                plots += plt.plot(pdfd.index, pdfd,
                                  'c' + ',', lw=1, ms=1, alpha=1, zorder=2,
                                  label='data points (log-binned)', rasterized=True)

        if 'hist' in mode:
            logmsg('Plotting hist')
            plt.hist2d(pdfd.index, pdfd, weights=weights,
                       bins=[np.geomspace(xmin, xmax, HIST_NBINS), np.geomspace(ymin, ymax, HIST_NBINS)],
                       cmin=1, norm=colors.LogNorm(),
                       alpha=1.0, label=what + ' data', rasterized=False, **kwargs)

        if 'kde' in mode:
            logmsg('Calculating KDE')
            xi, yi, zi = log_gaussian_kde(pdfd.index, pdfd, xmin, xmax, ymin, ymax, KDE_NBINS, weights=weights, fft=fft)
            zn = colors.PowerNorm(0.1)(zi)
            logmsg('Plotting KDE')
            # plt.pcolormesh(xi, yi, zn, shading='gouraud', rasterized=True)
            plt.contourf(xi, yi, zn, levels=16, antialiased=1, alpha=0.5, **kwargs).collections[0].set_alpha(0)
            # plt.contour(xi, yi, zn, antialiased=1, linewidths=0.25, colors='black').collections[0].set_alpha(0)

        set_log_limits(xmin, xmax, ymin, ymax)
    else:
        plots += plot_mixture(data, idx, x_val, what, mode, 'pdf')

    plt.xscale('log')
    plt.yscale('log')
    plt.legend(frameon=False)
    return plots

def plot_cdf(data, idx=None, x_val='length', what='flows', mode=frozenset(['mixture'])):
    plots = []
    if isinstance(data, pd.DataFrame):
        if idx is None:
            idx = np.unique(np.rint(np.geomspace(data.index.min(), data.index.max(), LINE_NBINS)).astype(int))
        if 'line' in mode:
            cdfd = data[what + '_sum'].cumsum() / data[what + '_sum'].sum()
            cdfi = scipy.interpolate.interp1d(cdfd.index, cdfd, 'linear', bounds_error=False)(idx)
            plots += plt.plot(idx, cdfi, STYLE[what][0] + '-', lw=2, ms=1, alpha=0.5,
                              label=what + ' (infered from data points)')
    else:
        plots += plot_mixture(data, idx, x_val, what, mode, 'cdf')

    plt.xscale('log')
    plt.legend(frameon=False)
    return plots

def plot_avg(data, idx=None, x_val='length', what='packets', mode=frozenset(['mixture'])):
    plots = []
    if isinstance(data, pd.DataFrame):
        if idx is None:
            idx = np.unique(np.rint(np.geomspace(data.index.min(), data.index.max(), LINE_NBINS)).astype(int))

        if what in ['packets', 'octets']:
            avg_points, avg_line = avg_data(data, idx, what)
            color = 'g' if what == 'packets' else 'b'
            plt.xscale('log')
            plt.yscale('log')
        elif what in ['packet_size', 'packet_iat']:
            if what == 'packet_size':
                num = 'octets'
            else:
                num = 'duration'
                plt.yscale('log')
            num_avg_points, num_avg_line = avg_data(data, idx, num)
            packets_avg_points, packets_avg_line = avg_data(data, idx, 'packets')
            avg_points = num_avg_points / packets_avg_points
            avg_line = num_avg_line / packets_avg_line
            color = 'm'
            mn, me, mx = avg_points.min(), data[num + '_sum'].sum() / data['packets_sum'].sum(), avg_points.max()
            lab_x = 10 ** (np.log10(idx.max() - idx.min()) * 0.25 + np.log10(idx.min()))
            if what == 'packet_size':
                plt.axhline(y=mn, alpha=0.1, zorder=0)
                plt.text(lab_x, mn, f'min = {mn:.2f}', va='center', ha='center', backgroundcolor='w', zorder=1)
                plt.ylim(0, 2000)
            plt.axhline(y=mx, alpha=0.1, zorder=0)
            plt.text(lab_x, mx, f'max = {mx:.2f}', va='center', ha='center', backgroundcolor='w', zorder=1)
            plt.axhline(y=me, alpha=0.1, zorder=0)
            plt.text(lab_x, me, f'avg = {me:.2f}', va='center', ha='center', backgroundcolor='w', zorder=1)
            plt.xscale('log')
        else:
            raise ValueError

        plots += plt.plot(avg_points.index.values, avg_points.values,
                          color + ',', lw=1, ms=1, alpha=1,
                          label='data points', rasterized=True)

        plots += plt.plot(idx, avg_line, color + '-', lw=2, ms=1, alpha=0.5,
                          label='infered from data points')
    elif 'mixture' in mode:
        try:
            avg_mix = avg(data, idx, x_val, what)
        except KeyError:
            return

        plt.xscale('log')
        if what in ['packets', 'octets']:
            plt.yscale('log')

        plots += plt.plot(idx, avg_mix, 'k-', lw=2, ms=1, alpha=0.5,
                          label='infered from model')

    plt.legend(frameon=False)
    return plots
