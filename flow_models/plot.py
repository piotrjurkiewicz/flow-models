#!/usr/bin/python3
import argparse

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from .lib.data import UNITS, LINE_NBINS, load_data
from .lib.plot import plot_pdf, plot_cdf, plot_avg
from .lib.util import logmsg

X_VALUES = ['length', 'size', 'duration', 'rate']

SIZE = 0.6
FIGSIZE = [SIZE * 11.2, SIZE * 6.8]
PDF_NONE_METADATA = {'Creator': None, 'Producer': None, 'CreationDate': None}
matplotlib.rcParams['figure.dpi'] *= 2
matplotlib.rcParams['figure.subplot.hspace'] = 0
matplotlib.rcParams['figure.subplot.wspace'] /= 1.5
matplotlib.rcParams['figure.subplot.left'] = 0.10
matplotlib.rcParams['figure.subplot.bottom'] = 0.10
matplotlib.rcParams['figure.subplot.right'] = 0.90
matplotlib.rcParams['figure.subplot.top'] = 1.00
matplotlib.rcParams['xtick.major.width'] = 0.25
matplotlib.rcParams['xtick.minor.width'] = 0.25
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.major.width'] = 0.25
matplotlib.rcParams['ytick.minor.width'] = 0.25
matplotlib.rcParams['ytick.right'] = True
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['axes.xmargin'] = 0.05
matplotlib.rcParams['axes.ymargin'] = 0.05
matplotlib.rcParams['axes.linewidth'] = 0.25
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['font.family'] = 'sans'

# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['font.family'] = 'sans-serif'

# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'
#
# matplotlib.rcParams['text.latex.preamble'] = r'''
# \usepackage[notextcomp]{stix}
# '''

def save_figure(figure, fname, ext='png', **kwargs):
    figure.savefig(fname + f'.{ext}', bbox_inches='tight', metadata=PDF_NONE_METADATA, **kwargs)

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
            plot_cdf(df, idx, x_val, what, mode={'mixture', *cdf_modes})
    ax.set_xlabel(f'Flow {x_val} ({UNITS[x_val]})')
    ax.set_ylabel('CDF')
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
        ax.set_xlabel(f'Flow {x_val} ({UNITS[x_val]})')
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
        ax.set_xlabel(f'Flow {x_val} ({UNITS[x_val]})')
        ax.set_ylabel(f'Average {what} (bytes)')
        out = f'avg-{what}'
        logmsg('Saving', out)
        save_figure(fig, out, ext=ext)
        plt.close(fig)
        logmsg('Done', out)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--format', default='png', help='plot file format')
    parser.add_argument('--single', action='store_true', help='plot PDF and CDF in single file')
    parser.add_argument('--no-normalize', action='store_false', help='do not normalize PDF datapoints')
    parser.add_argument('--fft', action='store_true', help='use FFT for calculating KDE')
    parser.add_argument('-C', nargs='*', default=(), help='additional CDF plot modes')
    parser.add_argument('-P', nargs='*', default=(), help='additional PDF plot modes')
    parser.add_argument('-x', default='length', choices=X_VALUES, help='x axis value')
    parser.add_argument('files', nargs='+', help='csv_hist files to plot')
    app_args = parser.parse_args()

    plot(app_args.files, app_args.x, app_args.format, app_args.single, app_args.no_normalize, app_args.fft,
         app_args.C, app_args.P)


if __name__ == '__main__':
    main()
