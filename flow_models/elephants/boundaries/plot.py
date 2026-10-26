#!/usr/bin/python3
"""Plots flow reduction curves from mixture or histogram data for first, threshold and sampling methods.

Produces the publication figures (``--format pdf``) and, when ``--out`` is
given, a set of single-axes SVG figures with embedded curve metadata for the
interactive web viewer.  ``{calculate,simulate}_{mix,data}`` directories in
the input are drawn as source-labeled curve sets; ``simulate_real``
directories add the real-times curves.  The ``occupancy`` column semantics
(time-free, residency, or real-times AOR) follow the producing directory.
"""

import argparse
import collections
import contextlib
import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from flow_models.lib.data import UNITS
from flow_models.lib.plot import _fm_add, matplotlib_config, save_figure

X_VALUES = ['length', 'size']

METHODS = {'first': '-',
           'threshold': '--',
           'sampling': ':'}

MARKERS = {'first': 'o',
           'threshold': 's',
           'sampling': 'x'}

MCOLORS = {'first': 'C0',
           'threshold': 'C1',
           'sampling': 'C2'}

SIZE = 0.6
FIGSIZE = [SIZE * 11.2, SIZE * 5.66]

def save_fig(fig, out, fmt):
    for ext in (['pdf', 'svg'] if fmt == 'both' else [fmt]):
        save_figure(fig, out, ext)

def plot_traffic(calculated, fmt='pdf'):
    interpolated = {}
    for n, x_val in enumerate(['length', 'size']):
        nidx = 1 / np.geomspace(1, 10000, 5000, endpoint=False)
        for method in METHODS:
            idx = 1 / calculated[method][x_val]['occupancy_mean']
            ddd = calculated[method][x_val].copy().set_index(idx)
            ddd = ddd[~ddd.index.duplicated()]
            ddd = ddd.reindex(ddd.index.union(nidx)).interpolate('slinear').reindex(nidx)
            interpolated.setdefault(method, {})[x_val] = ddd
    for to in ['absolute', *list(METHODS)]:
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
        save_fig(fig, out, fmt)
        plt.close(fig)

def plot_usage(calculated, what, fmt='pdf'):
    interpolated = {}
    for n, x_val in enumerate(['length', 'size']):
        nidx = np.linspace(50, 100, 5001, endpoint=True)
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
    z.loc[points][['occupancy_mean', 'operations_mean']].to_latex('selected.tex',
                                                                  float_format='%.2f',
                                                                  multicolumn_format='c')

    for to in ['absolute', *list(METHODS)]:
        to_label = ' reduction [x]'
        fig, axes = plt.subplots(1, 2, sharex='all', sharey='all', figsize=[FIGSIZE[0] * 2.132, FIGSIZE[1]])
        for n, x_val in enumerate(['length', 'size']):
            ax = axes[n]
            for method in METHODS:
                d = interpolated[x_val][method]
                if to == 'absolute':
                    r = d.copy()
                    for col in r.columns:
                        r[col].to_numpy()[:] = 1
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
        fig.gca().get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        out = f'{what}_{to}'
        save_fig(fig, out, fmt)
        plt.close(fig)

def plot_calc_sim(ax, calculated, simulated, method, x_val, w):
    sim_style = {
        'octets': 'bo',
        'flows': 'ro',
        'fraction': 'ko',
    }
    calc_style = {
        'octets': 'b-',
        'flows': 'r-',
        'fraction': 'k-',
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
    d = simulated[method][x_val][w + '_mean'].dropna()
    try:
        e = simulated[method][x_val][w + '_conf'].reindex(d.index)
    except KeyError:
        e = None
    ax.errorbar(d.index, d, e, None, sim_style[w], lw=1, capthick=1, ms=2,
                label=f'{name} (sim.) ({axis})')
    n = calculated[method][x_val][w + '_mean']
    d = n.loc[:d.index.max() if method != 'sampling' else d.index.min()]
    ax.plot(d.index, d, calc_style[w], lw=2,
            label=f'{name} (calc.) ({axis})')

def plot_all(calculated, simulated, one, fmt='pdf'):
    for method in calculated:

        if one:
            fig, axes = plt.subplots(1, 2, figsize=[FIGSIZE[0] * 2.132, FIGSIZE[1]], sharey='row')
            txes = [ax.twinx() for ax in axes]
            for tx in txes[1:]:
                tx.sharey(txes[0])
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
                save_fig(fig, out, fmt)
                plt.close(fig)

        if one:
            out = f'results_{method}'
            save_fig(fig, out, fmt)
            plt.close(fig)

def print_tables(simulated):
    for x_val in X_VALUES:
        dd = pd.concat({'first': simulated['first'][x_val], 'threshold': simulated['threshold'][x_val],
                        'sampling': simulated['sampling'][x_val]})
        cols = ['octets_mean', 'operations_mean', 'occupancy_mean']
        df = dd.loc[['first', 'threshold']].unstack(0)
        if x_val == 'length':
            df = df.iloc[:22]
        df = df[cols]
        df = df.swaplevel(axis=1).sort_index(axis=1).reindex(cols, axis=1, level=1)
        df.index = df.index.astype(np.int64)
        print(df.to_latex(float_format='%.2f'))

        df = dd.loc[['sampling']].unstack(0)
        if x_val == 'length':
            df = df.iloc[:22]
        df = df[cols]
        df = df.swaplevel(axis=1).sort_index(axis=1).reindex(cols, axis=1, level=1)
        print(df.to_latex(float_format='%.2f'))

SRC_MARKERS = {'mix': 'o', 'data': 's', 'real': 'P'}
SRC_LINES = {'mix': '-', 'data': '--', 'real': '-'}
SRC_COLORS = {'mix': 'C0', 'data': 'C1', 'real': 'C2'}

def plot_web_results(calc, sim, method, x_val, out, real=None):
    """Coverage figure for one method: simulation vs calculation.

    One axes, every curve distinguished in two ways: color per metric (blue
    traffic / red operations / green occupancy share, with a light/dark shade
    per source) and marker or linestyle per source (mix 'o'/'-', data 's'/'--',
    real 'P-').
    """
    wcolor = {'octets':   ('tab:blue', 'tab:cyan'),
              'flows':    ('tab:red', 'tab:orange'),
              'fraction': ('tab:green', 'tab:olive')}
    with _skl_figure() as (fig, ax):
        for w, wname in [('octets', 'Traffic coverage'),
                         ('flows', 'Operations coverage'),
                         ('fraction', 'Occupancy share')]:
            color = dict(zip(['mix', 'data'], wcolor[w]))
            color['real'] = 'tab:purple'
            dmax = None
            for src, sdf in sim.items():
                d = sdf[w + '_mean'].dropna().sort_index()
                e = sdf.get(w + '_conf')
                e = e.reindex(d.index) if e is not None else None
                label = f'{wname} (sim. {src})'
                cont = ax.errorbar(d.index, d, e, None, marker=SRC_MARKERS[src], ls='',
                                   color=color[src], lw=1, capthick=1, ms=3,
                                   alpha=0.75, label=label)
                _fm_add('data', d.index.to_numpy(), d.to_numpy(), label=label, what=w,
                        ax=ax, artists=[cont.lines[0]])
                dmax = d.index.min() if method == 'sampling' else d.index.max()
            for src, cdf in calc.items():
                n = cdf[w + '_mean'].loc[:dmax].sort_index()
                label = f'{wname} (calc. {src})'
                (line,) = ax.plot(n.index, n, SRC_LINES[src], color=color[src],
                                  lw=1, alpha=0.5, label=label)
                _fm_add('model', n.index.to_numpy(), n.to_numpy(), label=label, what=w,
                        ax=ax, artists=[line])
            if w == 'fraction' and real is not None and 'occupancy_mean' in real:
                share = (100 / real['occupancy_mean']).dropna().sort_index()
                label = f'{wname} (real times, sim.)'
                (line,) = ax.plot(share.index, share, 'P-', ms=4, lw=1,
                                  color=color['real'], alpha=0.65, label=label)
                _fm_add('data', share.index.to_numpy(), share.to_numpy(),
                        label=label, what='occupancy', ax=ax, artists=[line])
        ax.set_xscale('log')
        ax.set_ylim(0, 100)
        ax.set_ylabel('Coverage / occupancy share [%]')
        if method == 'sampling':
            ax.invert_xaxis()
            ax.set_xlabel('Sampling probability')
        else:
            ax.set_xlabel(f'Flow {x_val} threshold [{UNITS[x_val]}]')
        ax.legend(fontsize=7, loc='best')
        save_fig(fig, str(out / f'results_{method}'), 'svg')

def _skl_figure():
    """Context manager yielding a figure styled like the sklearn plots.

    The sklearn scripts run on near-default rcParams (outward ticks, framed
    legend, default margins/fonts, ``figsize=(10, 5)``), unlike the repo's
    ``matplotlib_config`` publication style, so reset to rcdefaults first.
    """
    @contextlib.contextmanager
    def cm():
        with plt.rc_context():
            plt.rcdefaults()
            plt.rcParams['svg.fonttype'] = 'none'  # keep <text> elements for the viewer
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            yield fig, ax
    return cm()


def plot_web_operations(calc, sim, x_val, out):
    """Flow table operations reduction vs traffic coverage — skl convention."""
    with _skl_figure() as (fig, ax):
        nidx = np.linspace(50, 100, 5001, endpoint=True)
        col = 'operations_mean'
        for method, color in MCOLORS.items():
            for src, cdf in calc.items():
                if method not in cdf or col not in cdf[method]:
                    continue
                dfc = cdf[method].set_index(cdf[method]['octets_mean']).dropna()
                dfc = dfc[~dfc.index.duplicated()]
                dfc = dfc.reindex(dfc.index.union(nidx)).interpolate('slinear').reindex(nidx).dropna()
                label = f'{method} (calc. {src})'
                (line,) = ax.plot(dfc.index, dfc[col], SRC_LINES[src], lw=1,
                                  color=color, alpha=0.5, label=label)
                _fm_add('model', dfc.index.to_numpy(), dfc[col].to_numpy(),
                        label=label, what='operations', artists=[line])
            for src, sdf in sim.items():
                if col not in sdf.get(method, ()):
                    continue
                dfs = sdf[method].dropna(subset=['octets_mean', col])
                dfs = dfs.sort_values('octets_mean')
                label = f'{method} (sim. {src})'
                (pts,) = ax.plot(dfs['octets_mean'], dfs[col], SRC_MARKERS[src],
                                 color=color, ms=4, alpha=0.75, label=label)
                _fm_add('data', dfs['octets_mean'].to_numpy(), dfs[col].to_numpy(),
                        label=label, what='operations', artists=[pts])
        ax.set_yscale('log')
        ax.set_ylim(1, 10000)
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlim(50, 100)
        ax.set_xlabel('Resulting Average Traffic Coverage [%]')
        ax.set_ylabel('Flow Operations Reduction [x]')
        ax.legend(fontsize=7, loc=1)
        save_fig(fig, str(out / 'operations'), 'svg')
        plt.close(fig)

def plot_web_occupancy(calc, sim, real, x_val, out):
    """Occupancy reduction: analytical/simulated curves plus real-times curves."""
    with _skl_figure() as (fig, ax):
        nidx = np.linspace(50, 100, 5001, endpoint=True)
        col = 'occupancy_mean'
        for method, color in MCOLORS.items():
            if method in real and col in real[method]:
                dfs = real[method].dropna(subset=['octets_mean', col])
                dfs = dfs.sort_values('octets_mean')
                label = f'{method} (real times, sim.)'
                (line,) = ax.plot(dfs['octets_mean'], dfs[col], 'P-',
                                  color=color, ms=4, lw=1, alpha=0.65, label=label)
                _fm_add('data', dfs['octets_mean'].to_numpy(), dfs[col].to_numpy(),
                        label=label, what='occupancy', artists=[line])
            for src, cdf in calc.items():
                if method not in cdf or col not in cdf[method]:
                    continue
                dd = cdf[method].set_index(cdf[method]['octets_mean']).dropna()
                dd = dd[~dd.index.duplicated()]
                dd = dd.reindex(dd.index.union(nidx)).interpolate('slinear').reindex(nidx).dropna()
                label = f'{method} (calc. {src})'
                (line,) = ax.plot(dd.index, dd[col], SRC_LINES[src], lw=1,
                                  color=color, alpha=0.5, label=label)
                _fm_add('model', dd.index.to_numpy(), dd[col].to_numpy(),
                        label=label, what='occupancy', artists=[line])
            for src, sdf in sim.items():
                if col not in sdf.get(method, ()):
                    continue
                dfs = sdf[method].dropna(subset=['octets_mean', col])
                dfs = dfs.sort_values('octets_mean')
                label = f'{method} (sim. {src})'
                (pts,) = ax.plot(dfs['octets_mean'], dfs[col], SRC_MARKERS[src],
                                 color=color, ms=4, alpha=0.75, label=label)
                _fm_add('data', dfs['octets_mean'].to_numpy(), dfs[col].to_numpy(),
                        label=label, what='occupancy', artists=[pts])
        ax.set_yscale('log')
        ax.set_ylim(1, 10000)
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlim(50, 100)
        ax.set_xlabel('Resulting Average Traffic Coverage [%]')
        ax.set_ylabel('Average Occupancy Reduction [x]')
        ax.legend(fontsize=7, loc=1)
        save_fig(fig, str(out / 'occupancy'), 'svg')
        plt.close(fig)

def plot_web(x_vals, calculated, simulated, real, out):
    """Write the web figure set: single-axes SVGs with embedded curve data."""
    plt.rcParams['font.size'] *= 0.8  # figure-set convention, matches plots/{x}
    out = pathlib.Path(out)
    for x_val in x_vals:
        outdir = out / x_val
        outdir.mkdir(parents=True, exist_ok=True)
        calc = {src: {m: cdf[m][x_val] for m in cdf if x_val in cdf[m]}
                for src, cdf in calculated.items()}
        sim = {src: {m: sdf[m][x_val] for m in sdf if x_val in sdf[m]}
               for src, sdf in simulated.items()}
        reald = {m: real[m][x_val] for m in real if x_val in real[m]}
        for method in {m for sdf in sim.values() for m in sdf}:
            plot_web_results({s: cdf[method] for s, cdf in calc.items()
                              if method in cdf},
                             {s: sdf[method] for s, sdf in sim.items()
                              if method in sdf},
                             method, x_val, outdir, reald.get(method))
        plot_web_operations(calc, sim, x_val, outdir)
        plot_web_occupancy(calc, sim, reald, x_val, outdir)

def plot(dirs, one=False, fmt='pdf', out=None):
    simulated = collections.defaultdict(lambda: collections.defaultdict(dict))
    calculated = collections.defaultdict(lambda: collections.defaultdict(dict))
    real = collections.defaultdict(dict)
    methods = set()
    x_vals = []
    for d in dirs:
        d = pathlib.Path(d)
        x_val = d.parts[-1]
        assert x_val in X_VALUES
        x_vals.append(x_val)
        # result source from parent dir: {calculate,simulate}_{mix,data,real}
        kind, _, src = d.parts[-3].partition('_')
        target = calculated[src] if kind == 'calculate' else \
            real if src == 'real' else simulated[src]
        for f in d.glob('*.csv'):
            method = f.stem
            assert method in METHODS
            methods.add(method)
            target[method][x_val] = pd.read_csv(str(f), index_col=0).dropna()

    calc0 = calculated.get('mix') or next(iter(calculated.values()), {})
    sim0 = simulated.get('mix') or next(iter(simulated.values()), {})
    plot_all(calc0, sim0, one, fmt)
    plot_usage(calc0, 'occupancy', fmt)
    plot_usage(calc0, 'operations', fmt)
    plot_traffic(calc0, fmt)
    print_tables(sim0)
    if out:
        plot_web(x_vals, calculated, simulated, real, out)

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--one', action='store_true', help='plot in one file')
    p.add_argument('--format', default='pdf', choices=['pdf', 'svg', 'both'], help='output figure format')
    p.add_argument('-O', '--out', default=None, help='directory for the web SVG figure set')
    p.add_argument('files', nargs='+',
                   help='result dirs {calculate,simulate}_{mix,data,real}/all/{length,size}')
    return p

def main():
    app_args = parser().parse_args()

    with matplotlib_config(latex=False):
        plot(app_args.files, app_args.one, app_args.format, app_args.out)


if __name__ == '__main__':
    main()
