#!/usr/bin/python3
import argparse
import collections
import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from flow_models.elephants.calculate import calculate
from flow_models.lib.data import UNITS
from flow_models.lib.plot import save_figure, matplotlib_config

X_VALUES = ['length', 'size']

METHODS = {'first': '-',
           'threshold': '--',
           'sampling': ':'}

SIZE = 0.6
FIGSIZE = [SIZE * 11.2, SIZE * 5.66]

def plot_traffic(calculated):
    interpolated = {}
    for n, x_val in enumerate(['length', 'size']):
        nidx = 1 / pd.Float64Index(np.geomspace(1, 10000, 5000, endpoint=False))
        for method in METHODS:
            idx = 1 / calculated[method][x_val]['occupancy_mean']
            ddd = calculated[method][x_val].copy().set_index(idx)
            ddd = ddd[~ddd.index.duplicated()]
            ddd = ddd.reindex(ddd.index.union(nidx)).interpolate('slinear').reindex(nidx)
            interpolated.setdefault(method, {})[x_val] = ddd
    for to in ['absolute'] + list(METHODS):
        to_label = '%'
        fig, axes = plt.subplots(1, 2, sharex='all', sharey='all', figsize=[FIGSIZE[0] * 2.132, FIGSIZE[1]])
        for n, x_val in enumerate(['length', 'size']):
            ax = axes[n]
            for method in METHODS:
                d = interpolated[method][x_val]['octets_mean']
                if to == 'absolute':
                    r = 1
                else:
                    r = interpolated[to][x_val]['octets_mean']
                    to_label = f'relative to {to}'
                ax.plot(d.index, d / r, 'b' + METHODS[method], lw=2,
                        label=method)
            ax.set_ylabel(f'Traffic coverage [{to_label}]')
            ax.set_xlabel(f'Flow table occupancy (decision by {x_val})')
            ax.tick_params('y', labelleft=True)
            ax.set_xscale('log')
            ax.legend()
        fig.gca().invert_xaxis()
        out = f'traffic_{to}'
        save_figure(fig, out)
        save_figure(fig, out)
        plt.close(fig)

def plot_usage(calculated, what):
    interpolated = {}
    for n, x_val in enumerate(['length', 'size']):
        nidx = pd.Float64Index(np.linspace(50, 100, 5001, endpoint=True))
        for method in METHODS:
            idx = calculated[method][x_val]['octets_mean']
            ddd = calculated[method][x_val].copy().set_index(idx)
            ddd = ddd[~ddd.index.duplicated()]
            ddd = ddd.reindex(ddd.index.union(nidx)).interpolate('slinear').reindex(nidx)
            interpolated.setdefault(x_val, {})[method] = ddd

    points = [99, 95, 90, 80, 75, 50]
    z = pd.concat({k: pd.concat(v) for k, v in interpolated.items()})
    z = z.unstack([0, 1]).swaplevel(1, 2, axis=1).sort_index(axis=1)[['occupancy_mean', 'operations_mean']]
    z = z.reindex(METHODS, axis=1, level=1)
    z.loc[points][['occupancy_mean', 'operations_mean']].to_latex(f'selected.tex',
                                                                  float_format='%.2f',
                                                                  multicolumn_format='c')

    for to in ['absolute'] + list(METHODS):
        to_label = ' reduction [x]'
        fig, axes = plt.subplots(1, 2, sharex='all', sharey='all', figsize=[FIGSIZE[0] * 2.132, FIGSIZE[1]])
        for n, x_val in enumerate(['length', 'size']):
            ax = axes[n]
            for method in METHODS:
                d = interpolated[x_val][method]
                if to == 'absolute':
                    r = d.copy()
                    for col in r.columns:
                        r[col].values[:] = 1
                    ax.plot(d.index, d[f'{what}_mean'] / r[f'{what}_mean'], 'k' + METHODS[method], lw=2,
                            label=method)
                else:
                    r = interpolated[x_val][to]
                    to_label = f' [relative to {to}]'
                    ax.plot(d.index, r[f'{what}_mean'] / d[f'{what}_mean'], 'k' + METHODS[method], lw=2,
                            label=method)
            ax.set_xlabel(f'Traffic coverage [%] (decision by {x_val})')
            ax.set_ylabel(f'Flow table {what}{to_label}')
            ax.tick_params('y', labelleft=True)
            if to == 'absolute':
                ax.set_yscale('log')
            ax.legend()
        fig.gca().invert_xaxis()
        fig.gca().get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        out = f'{what}_{to}'
        save_figure(fig, out)
        plt.close(fig)

def plot_calc_sim(ax, calculated, simulated, method, x_val, w):
    sim_style = {
        'octets': 'bo',
        'flows': 'ro',
        'fraction': 'ko'
    }
    calc_style = {
        'octets': 'b-',
        'flows': 'r-',
        'fraction': 'k-'
    }
    if w == 'fraction':
        name = 'Occupancy'
    elif w == 'octets':
        name = 'Traffic coverage'
    elif w == 'flows':
        name = 'Operations'
    else:
        name = w[:-1] + ' coverage'

    axis = 'left' if w == 'octets' else 'right'
    d = simulated[method][x_val][w + '_mean']
    try:
        e = simulated[method][x_val][w + '_conf']
    except KeyError:
        e = None
    ax.errorbar(d.index, d, e, None, sim_style[w], lw=1, capthick=1, ms=2,
                label=f'{name} (sim.) ({axis})')
    n = calculated[method][x_val][w + '_mean']
    d = n.loc[:d.index.max() if method != 'sampling' else d.index.min()]
    ax.plot(d.index, d, calc_style[w], lw=2,
            label=f'{name} (calc.) ({axis})')

def plot_all(calculated, simulated, one):
    for method in calculated:

        if one:
            fig, axes = plt.subplots(1, 2, figsize=[FIGSIZE[0] * 2.132, FIGSIZE[1]], sharey='row')
            txes = [ax.twinx() for ax in axes]
            txes[0].get_shared_y_axes().join(*txes)
        else:
            fig, ax = plt.subplots(figsize=FIGSIZE)
            tx = ax.twinx()

        for n, x_val in enumerate(simulated[method]):
            if one:
                ax = axes[n]
                tx = txes[n]

            plot_calc_sim(ax, calculated, simulated, method, x_val, 'octets')
            plot_calc_sim(tx, calculated, simulated, method, x_val, 'flows')
            plot_calc_sim(tx, calculated, simulated, method, x_val, 'fraction')

            ax.set_xscale('log')
            tx.set_yscale('log')
            ax.legend(loc=3)
            tx.legend()

            if method == 'sampling':
                ax.invert_xaxis()
                ax.set_xlabel(f'Sampling probability (sampling by {x_val})')
            else:
                ax.set_xlabel(f'Flow {x_val} threshold [{UNITS[x_val]}]')
            if not one:
                out = f'results_{method}_{x_val}'
                save_figure(fig, out)
                plt.close(fig)

        if one:
            out = f'results_{method}'
            save_figure(fig, out)
            plt.close(fig)

def plot_probability():
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
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

def plot(dirs, one=False):
    simulated = collections.defaultdict(dict)
    calculated = collections.defaultdict(dict)
    methods = set()
    for d in dirs:
        d = pathlib.Path(d)
        x_val = d.parts[-1]
        assert x_val in X_VALUES
        for f in d.glob('*.csv'):
            method = f.stem
            assert method in METHODS
            methods.add(method)
            simulated[method][x_val] = pd.read_csv(str(f), index_col=0).dropna()
        for method, df in calculate('../mixtures/all/' + x_val, 1024, x_val=x_val, methods=methods).items():
            calculated[method][x_val] = df.dropna()

    plot_all(calculated, simulated, one)
    plot_usage(calculated, 'occupancy')
    plot_usage(calculated, 'operations')
    plot_traffic(calculated)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--one', action='store_true', help='plot in one file')
    parser.add_argument('files', nargs='+', help='csv_hist files to plot')
    app_args = parser.parse_args()

    with matplotlib_config(latex=False):
        plot_probability()
        plot(app_args.files, app_args.one)


if __name__ == '__main__':
    main()
