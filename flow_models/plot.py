#!/usr/bin/python3
"""
Generates plots from flow records and fitted models (requires `pandas` and `scipy`).
"""

import argparse

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from .lib.data import UNITS, LINE_NBINS, load_data
from .lib.plot import plot_pdf, plot_cdf, plot_avg, save_figure, matplotlib_config, MODES_PDF, MODES_CDF
from .lib.util import logmsg

X_VALUES = ['length', 'size', 'duration', 'rate']

SIZE = 0.6
FIGSIZE = [SIZE * 11.2, SIZE * 6.8]

def plot(objects, x_val='length', ext='png', single=False, normalize=True, fft=False, cdf_modes=(), pdf_modes=(), avg_modes=()):
    data = load_data(objects)
    idx = None

    if single:
        fig = plt.figure(figsize=[FIGSIZE[0] * 2.132, FIGSIZE[1] * 2])
        ax = plt.subplot(2, 2, 1)
    else:
        fig = plt.figure(figsize=FIGSIZE)
        ax = plt.subplot(1, 1, 1)

    plt.subplots_adjust(0, 0, 1, 1)

    for obj, df in data.items():
        if idx is None:
            idx = np.unique(np.rint(np.geomspace(df.index.min(), df.index.max(), LINE_NBINS)).astype(int))
        for what in ['flows', 'packets', 'octets']:
            logmsg('Drawing CDF', obj, what)
            plot_cdf(df, idx, x_val, what, mode={'line', 'mixture', *cdf_modes})
    ax.set_xlabel(f'Flow {x_val} [{UNITS[x_val]}]')
    ax.set_ylabel('CDF (Fraction of)')
    if not single:
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
        plt.subplots_adjust(0, 0, 1, 1)
        for obj, df in data.items():
            logmsg('Drawing PDF', obj, what)
            plot_pdf(df, idx, x_val, what, mode={'line', 'mixture', *pdf_modes}, normalize=normalize, fft=fft)
        ax.set_xlabel(f'Flow {x_val} [{UNITS[x_val]}]')
        ax.set_ylabel(f'PDF of {what}')
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

    for what in ['packets', 'octets', 'packet_size']:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for obj, df in data.items():
            logmsg('Drawing AVG', obj, what)
            plot_avg(df, idx, x_val, what, mode={'line', 'mixture', *avg_modes})
        ax.set_xlabel(f'Flow {x_val} [{UNITS[x_val]}]')
        ax.set_ylabel(f"Average {what.replace('_', ' ')} [bytes]")
        out = f'avg-{what}'
        logmsg('Saving', out)
        save_figure(fig, out, ext=ext)
        plt.close(fig)
        logmsg('Done', out)

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--format', default='png', choices=['png', 'pdf'], help='plot file format')
    p.add_argument('--single', action='store_true', help='plot PDF and CDF in single file')
    p.add_argument('--no-normalize', action='store_false', help='do not normalize PDF datapoints')
    p.add_argument('--fft', action='store_true', help='use FFT for calculating KDE')
    p.add_argument('-P', action='append', default=[], choices=MODES_PDF, help='additional PDF plot modes (can be specified multiple times)')
    p.add_argument('-C', action='append', default=[], choices=MODES_CDF, help='additional CDF plot modes (can be specified multiple times)')
    p.add_argument('-x', default='length', choices=X_VALUES, help='x axis value')
    p.add_argument('histogram', help='csv_hist file to plot')
    p.add_argument('mixture', nargs='?', help='mixture directory to plot')
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
