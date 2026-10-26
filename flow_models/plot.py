#!/usr/bin/python3
"""Generates plots from flow records and fitted models (requires `pandas` and `scipy`)."""

import argparse

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from flow_models.lib.data import LINE_NBINS, UNITS, load_data
from flow_models.lib.plot import MODES_CDF, MODES_PDF, matplotlib_config, plot_avg, plot_cdf, plot_pdf, save_figure
from flow_models.lib.util import logmsg

X_VALUES = ['length', 'size', 'duration', 'rate']

SIZE = 0.6
FIGSIZE = [SIZE * 11.2, SIZE * 6.8]
MARGINS = (0.10, 0.11, 0.99, 0.97)

# fixed axis limits per x value and plot kind — identical plot area across
# datasets so plots can be compared directly
LIMS = {
    'length': {
        'x': (0.3, 1e9),
        'cdf': (-0.05, 1.05),
        'pdf-flows': (1e-19, 10),
        'pdf-packets': (1e-13, 0.1),
        'pdf-octets': (1e-14, 0.1),
        'avg-flow-length': (0.2, 1e12),
        'avg-flow-size': (10, 1e15),
        'avg-packet-size': (0, 2000),
    },
    'size': {
        'x': (10, 1e12),
        'cdf': (-0.05, 1.05),
        'pdf-flows': (1e-22, 1),
        'pdf-packets': (1e-16, 0.01),
        'pdf-octets': (1e-16, 0.001),
        'avg-flow-length': (0.2, 1e12),
        'avg-flow-size': (10, 1e15),
        'avg-packet-size': (0, 2000),
    },
}

def fix_lims(ax, x_val, name):
    lim = LIMS.get(x_val)
    if not lim:
        return
    ax.set_xlim(lim['x'])
    if name in lim:
        ax.set_ylim(lim[name])
    # mpl strides major labels and drops minor ticks when a log axis spans
    # too many decades — keep decade ticks on the x axis (y spans too many
    # decades for minor ticks to stay distinct)
    lo, hi = ax.xaxis.get_view_interval()
    if ax.xaxis.get_scale() == 'log':
        ndec = int(np.ceil(np.log10(hi / lo)))
        ax.xaxis.set_major_locator(mticker.LogLocator(numticks=ndec + 2))
        ax.xaxis.set_minor_locator(mticker.LogLocator(subs=np.arange(2, 10), numticks=ndec * 10))

def plot(objects, x_val='length', ext='png', single=False, normalize=True, fft=False, cdf_modes=(), pdf_modes=(), avg_modes=()):
    data = load_data(objects)
    idx = None

    if not single:
        plt.rcParams['font.size'] *= 0.8

    if single:
        fig = plt.figure(figsize=[FIGSIZE[0] * 2.132, FIGSIZE[1] * 2])
        ax = plt.subplot(2, 2, 1)
        # hspace stays 0 so rows touch; extra room on all edges for labels
        fig.subplots_adjust(left=0.07, bottom=0.08, right=0.95, top=0.98)
    else:
        fig = plt.figure(figsize=FIGSIZE)
        ax = plt.subplot(1, 1, 1)
        plt.subplots_adjust(*MARGINS)

    for obj, df in data.items():
        if idx is None:
            idx = np.unique(np.rint(np.geomspace(df.index.min(), df.index.max(), LINE_NBINS)).astype(np.int64))
        for what in ['flows', 'packets', 'octets']:
            logmsg('Drawing CDF', obj, what)
            plot_cdf(df, idx, x_val, what, mode={'line', 'mixture', *cdf_modes})
    ax.set_xlabel(f'Flow {x_val} [{UNITS[x_val]}]')
    ax.set_ylabel('CDF (Fraction of)')
    fix_lims(ax, x_val, 'cdf')
    if single:
        # top row shares x with bottom — its labels would overlap the row below
        ax.set_xlabel('')
        plt.setp(ax.get_xticklabels(), visible=False)
    else:
        out = 'cdf'
        logmsg('Saving', out)
        save_figure(fig, out, ext=ext)
        plt.close(fig)
        logmsg('Done', out)

    for n, what in enumerate(['flows', 'packets', 'octets']):
        if single:
            ax = plt.subplot(2, 2, n + 2, sharex=ax)
        else:
            fig, ax = plt.subplots(figsize=FIGSIZE)
            plt.subplots_adjust(*MARGINS)
        for obj, df in data.items():
            logmsg('Drawing PDF', obj, what)
            plot_pdf(df, idx, x_val, what, mode={'line', 'mixture', *pdf_modes}, normalize=normalize, fft=fft)
        ax.set_xlabel(f'Flow {x_val} [{UNITS[x_val]}]')
        ax.set_ylabel(f'PDF of {what}')
        fix_lims(ax, x_val, f'pdf-{what}')
        if single and n == 0:
            ax.set_xlabel('')
            plt.setp(ax.get_xticklabels(), visible=False)
        if not single:
            out = f'pdf-{what}'
            logmsg('Saving', out)
            save_figure(fig, out, ext=ext)
            plt.close(fig)
            logmsg('Done', out)

    if single:
        out = 'single'
        logmsg('Saving', out)
        save_figure(fig, out, ext=ext)
        plt.close(fig)
        logmsg('Done', out)
        # single run kept full font size — avg plots still get the -20%
        plt.rcParams['font.size'] *= 0.8

    for what in ['packets', 'octets', 'packet_size']:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for obj, df in data.items():
            logmsg('Drawing AVG', obj, what)
            plot_avg(df, idx, x_val, what, mode={'line', 'mixture', *avg_modes})
        ax.set_xlabel(f'Flow {x_val} [{UNITS[x_val]}]')
        ax.set_ylabel(f"Average {what.replace('_', ' ')} [bytes]")
        out = {'packets': 'avg-flow-length', 'octets': 'avg-flow-size', 'packet_size': 'avg-packet-size'}[what]
        fix_lims(ax, x_val, out)
        logmsg('Saving', out)
        save_figure(fig, out, ext=ext)
        plt.close(fig)
        logmsg('Done', out)

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--format', default='png', choices=['png', 'pdf', 'svg'], help='plot file format')
    p.add_argument('--single', action='store_true', help='plot PDF and CDF in single file')
    p.add_argument('--no-normalize', action='store_false', help='do not normalize PDF datapoints')
    p.add_argument('--fft', action='store_true', help='use FFT for calculating KDE')
    p.add_argument('-P', action='append', default=[], choices=MODES_PDF, help='additional PDF plot modes (can be specified multiple times)')
    p.add_argument('-C', action='append', default=[], choices=MODES_CDF, help='additional CDF plot modes (can be specified multiple times)')
    p.add_argument('-x', default='length', choices=X_VALUES, help='x axis value')
    p.add_argument('histogram', type=str, help='csv_hist file to plot')
    p.add_argument('mixture', nargs='?', type=str, help='mixture directory to plot')
    return p

def main():
    app_args = parser().parse_args()

    files = [app_args.histogram]
    if app_args.mixture:
        files.append(app_args.mixture)
    with matplotlib_config(latex=False):
        plot(files, app_args.x, app_args.format, app_args.single, app_args.no_normalize, app_args.fft,
             app_args.C, app_args.P)


if __name__ == '__main__':
    main()
