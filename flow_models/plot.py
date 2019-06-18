#!/usr/bin/python3
import argparse
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .lib.data import plot_pdf, plot_cdf, UNITS, LINE_NBINS, plot_avg
from .lib.mix import load_mixture
from .lib.util import logmsg

X_VALUES = ['length', 'size', 'duration', 'rate']

SIZE = 0.6
FIGSIZE = [SIZE * 1.25 * 10.8, SIZE * 8.1]
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
# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'
#
# matplotlib.rcParams['text.latex.preamble'] = r'''
# \usepackage[notextcomp]{stix}
# '''

def save_figure(figure, fname, ext='png', **kwargs):
    figure.savefig(fname + f'.{ext}', bbox_inches='tight', metadata=PDF_NONE_METADATA, **kwargs)

def plot(objects, x_val='length', ext='png', one=False, normalize=True, fft=False, cdf_modes=(), pdf_modes=(), avg_modes=()):
    data = {}
    for obj in objects:
        if isinstance(obj, (str, pathlib.Path)):
            file = pathlib.Path(obj)
            logmsg(f'Loading file {file}')
            if file.suffix == '.json':
                data[file] = load_mixture(file)
            elif file.is_dir():
                mixtures = {}
                for ff in file.glob('*.json'):
                    mixtures[ff.stem] = load_mixture(ff)
                data[file] = mixtures
            else:
                data[file] = pd.read_csv(file, index_col=0, sep=',', low_memory=False, usecols=lambda col: not col.endswith('_ssq'))
            logmsg(f'Loaded file {file}')
        else:
            data[id(obj)] = obj

    idx = None

    if one:
        fig = plt.figure(figsize=[FIGSIZE[0] * 2.1, FIGSIZE[1] * 2.1])
        ax = plt.subplot(2, 2, 1)
    else:
        fig = plt.figure(figsize=FIGSIZE)
        ax = plt.subplot(1, 1, 1)

    for obj, df in data.items():
        if idx is None:
            idx = np.unique(np.rint(np.geomspace(df.index.min(), df.index.max(), LINE_NBINS)).astype(int))
        for what in ['flows', 'packets', 'octets']:
            logmsg('Drawing CDF', obj, what)
            plot_cdf(df, idx, what, mode={'mixture', *cdf_modes})
    ax.set_xlabel(f'Flow {x_val} ({UNITS[x_val]})')
    ax.set_ylabel('CDFs')
    if not one:
        out = f'cdf-{x_val}'
        logmsg('Saving', out)
        save_figure(fig, out, ext=ext)
        plt.close(fig)
        logmsg('Done', out)

    for n, what in enumerate(['flows', 'packets', 'octets']):
        if one:
            ax = plt.subplot(2, 2, n + 2, sharex=ax)
        else:
            fig, ax = plt.subplots(figsize=FIGSIZE)
        for obj, df in data.items():
            logmsg('Drawing PDF', obj, what)
            plot_pdf(df, idx, what, mode={'line', 'mixture', *pdf_modes}, normalize=normalize, fft=fft)
        ax.set_xlabel(f'Flow {x_val} ({UNITS[x_val]})')
        ax.set_ylabel(f'PDF of {what}')
        if not one:
            out = f'pdf-{x_val}-{what}'
            logmsg('Saving', out)
            save_figure(fig, out, ext=ext)
            plt.close(fig)
            logmsg('Done', out)

    if one:
        out = f'one-{x_val}'
        logmsg('Saving', out)
        save_figure(fig, out, ext=ext)
        plt.close(fig)
        logmsg('Done', out)

    for what in ['packets', 'octets', 'packet_size', 'packet_iat']:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for obj, df in data.items():
            logmsg('Drawing AVG', obj, what)
            plot_avg(df, idx, what, mode={'line', 'mixture', *avg_modes})
        ax.set_xlabel(f'Flow {x_val} ({UNITS[x_val]})')
        ax.set_ylabel(f'Average {what}')
        out = f'avg-{x_val}-{what}'
        logmsg('Saving', out)
        save_figure(fig, out, ext=ext)
        plt.close(fig)
        logmsg('Done', out)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--format', default='png', help='plot file format')
    parser.add_argument('--one', action='store_true', help='plot PDF and CDF in one file')
    parser.add_argument('--no-normalize', action='store_false', help='do not normalize PDF data')
    parser.add_argument('--fft', action='store_true', help='use FFT for calculating KDE')
    parser.add_argument('-C', nargs='*', default=(), help='additional CDF plot modes')
    parser.add_argument('-P', nargs='*', default=(), help='additional PDF plot modes')
    parser.add_argument('-x', default='length', choices=X_VALUES, help='x axis value')
    parser.add_argument('files', nargs='+', help='csv_hist files to plot')
    app_args = parser.parse_args()

    plot(app_args.files, app_args.x, app_args.format, app_args.one, app_args.no_normalize, app_args.fft, app_args.C, app_args.P)


if __name__ == '__main__':
    main()
