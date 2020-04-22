#!/usr/bin/python3

import argparse
import collections
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flow_models.detect.calculate import calculate
from flow_models.lib.data import UNITS

X_VALUES = ['length', 'size']

METHODS = {'first': '-',
           'threshold': '--',
           'sampling': ':'}

SIZE = 0.6
FIGSIZE = [SIZE * 11.2, SIZE * 6.8]
PDF_NONE = {'Creator': None, 'Producer': None, 'CreationDate': None}
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
matplotlib.rcParams["legend.frameon"] = False
matplotlib.rcParams["errorbar.capsize"] = 2
matplotlib.rcParams['axes.xmargin'] = 0.05
matplotlib.rcParams['axes.ymargin'] = 0.05
matplotlib.rcParams['axes.linewidth'] = 0.25
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['font.family'] = 'sans'

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'sans-serif'
# matplotlib.rcParams['mathtext.fontset'] = 'stix'

# matplotlib.rcParams['text.latex.preamble'] = r'''
# \usepackage[notextcomp]{stix}
# '''

def save_figure(figure, fname, **kwargs):
    figure.savefig(fname + '.pdf', bbox_inches='tight', metadata=PDF_NONE, **kwargs)

def plot_traffic(calculated):
    relative = {}
    for n, x_val in enumerate(['length', 'size']):
        nidx = 1 / pd.Float64Index(np.geomspace(1, 10000, 5000, endpoint=False))
        for method in METHODS:
            idx = 1 / calculated[method][x_val]['occupancy_mean']
            ddd = calculated[method][x_val].copy().set_index(idx)
            ddd = ddd[~ddd.index.duplicated()]
            ddd = ddd.reindex(ddd.index.union(nidx)).interpolate('slinear').reindex(nidx)
            relative.setdefault(method, {})[x_val] = ddd
    for to in ['absolute'] + list(METHODS):
        to_label = r'\%'
        fig, axes = plt.subplots(1, 2, sharey='all', figsize=[FIGSIZE[0] * 2.132, FIGSIZE[1]])
        fig.subplots_adjust(0, 0, 1, 1)
        for n, x_val in enumerate(['length', 'size']):
            ax = axes[n]
            for method in METHODS:
                d = relative[method][x_val]['octets_mean']
                if to == 'absolute':
                    r = 1
                else:
                    r = relative[to][x_val]['octets_mean']
                    to_label = f'relative to {to}'
                ax.plot(d.index, d / r, 'k' + METHODS[method], lw=2,
                        label=method)
                # plt.plot(d.index, 1 / d['operations_mean'] / r['operations_mean'], 'r' + ls[method], lw=2,
                #          label=method)
                # ax.set_xlim(ax.get_xlim()[::-1])
            ax.set_ylabel(f'Traffic coverage [{to_label}]')
            ax.set_xlabel(f'Flow table occupancy [absolute] (decision by {x_val})')
            ax.tick_params('y', labelleft=True)
            ax.set_xscale('log')
            ax.legend()
        out = f'traffic_{to}'
        save_figure(fig, out)
        save_figure(fig, out)
        plt.close(fig)

def plot_occupancy(calculated):
    relative = {}
    for n, x_val in enumerate(['length', 'size']):
        nidx = pd.Float64Index(np.linspace(50, 100, 5000, endpoint=False))
        for method in METHODS:
            idx = calculated[method][x_val]['octets_mean']
            ddd = calculated[method][x_val].copy().set_index(idx)
            ddd = ddd[~ddd.index.duplicated()]
            ddd = ddd.reindex(ddd.index.union(nidx)).interpolate('slinear').reindex(nidx)
            relative.setdefault(method, {})[x_val] = ddd
    for to in ['absolute'] + list(METHODS):
        to_label = to
        fig, axes = plt.subplots(1, 2, sharey='all', figsize=[FIGSIZE[0] * 2.132, FIGSIZE[1]])
        fig.subplots_adjust(0, 0, 1, 1)
        for n, x_val in enumerate(['length', 'size']):
            ax = axes[n]
            for method in METHODS:
                d = relative[method][x_val]['occupancy_mean']
                if to == 'absolute':
                    r = 1
                else:
                    r = relative[to][x_val]['occupancy_mean']
                    to_label = f'relative to {to}'
                ax.plot(d.index, 1 / (d / r), 'k' + METHODS[method], lw=2,
                        label=method)
                # plt.plot(d.index, 1 / d['operations_mean'] / r['operations_mean'], 'r' + ls[method], lw=2,
                #          label=method)
                ax.set_xlim(ax.get_xlim()[::-1])
            ax.set_xlabel(f'Traffic coverage [\%] (decision by {x_val})')
            ax.set_ylabel(f'Flow table occupancy [{to_label}]')
            ax.tick_params('y', labelleft=True)
            if to == 'absolute':
                ax.set_yscale('log')
            ax.legend()
        out = f'occupancy_{to}'
        save_figure(fig, out)
        plt.close(fig)

def plot_all(calculated, simulated, one):
    for method in calculated:

        if one:
            fig = plt.figure(figsize=[FIGSIZE[0] * 2.132, FIGSIZE[1] * 2])
            ax = plt.subplot(1, 2, 1)
        else:
            fig = plt.figure(figsize=FIGSIZE)
            ax = plt.subplot(1, 1, 1)

        plt.subplots_adjust(0, 0, 1, 1)

        ax2 = None

        for n, x_val in enumerate(simulated[method]):
            if one:
                ax = plt.subplot(2, 2, n + 1, sharey=ax)
            else:
                fig, ax = plt.subplots(figsize=FIGSIZE)
            plt.subplots_adjust(0, 0, 1, 1)

            pppp(ax, calculated, simulated, method, x_val, 'octets')

            if ax2 is None:
                ax2 = ax.twinx()
            else:
                nax = ax.twinx()
                ax2.get_shared_y_axes().join(ax2, nax)
                ax2 = nax

            pppp(ax2, calculated, simulated, method, x_val, 'flows')
            pppp(ax2, calculated, simulated, method, x_val, 'portion')

            ax.set_xscale("log")
            if x_val == 'length':
                ax.set_ylabel('Octet coverage (\%)')
            else:
                pass
                # ax2.set_ylabel('Flow coverage/occupancy (\%)')
            ax2.set_yscale("log")
            ax.legend(loc=3)
            ax2.legend()

            # ax.tick_params(axis='y', labelcolor='g')

            if method == 'sampling':
                ax.invert_xaxis()
                ax.set_xlabel(f'Sampling probability (sampling by {x_val})')
            else:
                ax.set_xlabel(f'Flow {x_val} threshold ({UNITS[x_val]})')
            if not one:
                out = f'detect-{method}-{x_val}'
                save_figure(fig, out)
                plt.close(fig)

        if one:
            out = f'detect-{method}'
            save_figure(fig, out)
            plt.close(fig)

def plot_prob():
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    fig.subplots_adjust(0, 0, 1, 1)
    idx = np.geomspace(1, 1000, 512)
    ax.plot(idx, 1 - (1 - 0.1) ** idx, 'k-', lw=2,
            label='p  0.1$')
    ax.plot(idx, 1 - (1 - 0.01) ** idx, 'k-', lw=2,
            label='p = 0.1$')
    ax.text(12, 0.6, '$p = 0.1$')
    ax.text(150, 0.6, '$p = 0.01$')
    ax.set_xlabel(f'Flow length [packets]')
    ax.set_ylabel(f'Total probability of being added to flow table')
    ax.set_xscale('log')
    save_figure(fig, 'probability')
    plt.close(fig)

def pppp(ax, calculated, simulated, method, x_val, w):
    sim_style = {
        'octets': 'go',
        'flows': 'ro',
        'portion': 'bo'
    }
    calc_style = {
        'octets': 'g-',
        'flows': 'r--',
        'portion': 'b:'
    }
    name = 'occupancy' if w == 'portion' else w[:-1] + ' coverage'
    d = simulated[method][x_val][w + '_mean'].iloc[:-3]
    e = simulated[method][x_val][w + '_conf'].iloc[:-3]
    ax.errorbar(d.index, d, e, None, sim_style[w], lw=1, capthick=1, ms=2, alpha=0.5,
                label=f'{name} (sim.)')
    n = calculated[method][x_val][w + '_mean']
    d = n.loc[:d.index.max() if method != 'sampling' else d.index.min()]
    ax.plot(d.index, d, calc_style[w], lw=2, alpha=0.5,
            label=f'{name} (calc.)')

def plot(dirs, one=False):
    simulated = collections.defaultdict(dict)
    calculated = collections.defaultdict(dict)
    methods = set()

    plot_prob()

    for d in dirs:
        d = pathlib.Path(d)
        x_val = d.stem.split('_')[0]
        assert x_val in X_VALUES
        for f in d.glob('*.df'):
            method = f.stem
            assert method in METHODS
            methods.add(method)
            simulated[method][x_val] = pd.read_pickle(str(f))
        for method, df in calculate('mixtures/all/' + x_val, 512, x_val=x_val, methods=methods).items():
            calculated[method][x_val] = df

    plot_all(calculated, simulated, one)
    plot_occupancy(calculated)
    plot_traffic(calculated)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--one', action='store_true', help='plot in one file')
    parser.add_argument('files', nargs='+', help='csv_hist files to plot')
    app_args = parser.parse_args()
    plot(app_args.files, app_args.one)


if __name__ == '__main__':
    main()
