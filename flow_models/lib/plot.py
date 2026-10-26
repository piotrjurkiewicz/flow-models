import collections
import contextlib
import json
import pathlib
import re

import matplotlib
import numpy as np
import pandas as pd
import scipy
import scipy.interpolate

from matplotlib import colors
from matplotlib import pyplot as plt

from .data import HIST_NBINS, KDE_NBINS, LINE_NBINS, STYLE, avg_data, calc_minmax, log_gaussian_kde, normalize_data, \
    pdf_from_cdf
from .mix import avg, cdf, cdf_comp, pdf, pdf_comp
from .util import logmsg

PDF_NONE = {'Creator': None, 'Producer': None, 'CreationDate': None}
MODES_MIXTURE = ['comp', 'comp_stack', 'comp_labels']
MODES_PDF = ['points', 'hist', 'kde', *MODES_MIXTURE]
MODES_CDF = MODES_MIXTURE

def _fm_num(v):
    v = float(v)
    if not np.isfinite(v):
        return None
    return float(f'{v:.7g}')

def _fm_axis_data(ax=None):
    ax = ax or plt.gca()
    fig = ax.figure
    if not hasattr(fig, '_fm_axes'):
        fig._fm_axes = {}
    return ax, fig._fm_axes.setdefault(ax, {'curves': [], 'totals': {}})

def _fm_total(what, value, ax=None):
    _, info = _fm_axis_data(ax)
    info['totals'][what] = _fm_num(value)

def _fm_add(kind, x, y, label=None, what=None, component=None, ax=None, artists=()):
    _, info = _fm_axis_data(ax)
    gid = 'fm-c%d' % len(info['curves'])
    for artist in artists:
        artist.set_gid(gid)
    entry = {
        'id': gid,
        'kind': kind,
        'x': [_fm_num(v) for v in np.asarray(x, dtype=float).ravel()],
        'y': [_fm_num(v) for v in np.asarray(y, dtype=float).ravel()],
    }
    if label:
        entry['label'] = label
    if what:
        entry['what'] = what
    if component:
        entry['component'] = component
    info['curves'].append(entry)

def _fm_embed(figure, fname):
    axes = getattr(figure, '_fm_axes', None)
    if not axes:
        return
    data = {'axes': []}
    for ax, info in axes.items():
        data['axes'].append({
            'xlabel': ax.get_xlabel(),
            'ylabel': ax.get_ylabel(),
            'xscale': ax.get_xscale(),
            'yscale': ax.get_yscale(),
            'xlim': [_fm_num(v) for v in ax.get_xlim()],
            'ylim': [_fm_num(v) for v in ax.get_ylim()],
            'totals': info['totals'],
            'curves': info['curves'],
        })
    payload = json.dumps(data, separators=(',', ':')).replace('&', '&amp;').replace('<', '&lt;')
    path = pathlib.Path(fname)
    text = path.read_text()
    insert = text.find('>', text.find('<svg')) + 1
    text = text[:insert] + f'<metadata id="fm-plot-data">{payload}</metadata>' + _fm_slim(text[insert:])
    path.write_text(text)

def _fm_slim(text):
    """Collapse repeated inline text styling into a single CSS rule."""
    fams = collections.Counter(re.findall(r'font-family: ([^;"]+)', text))
    sizes = collections.Counter(re.findall(r'font-size: ([\d.]+)px', text))
    rules = ''
    if fams:
        # prefer the full fallback list over per-element resolved names
        fam = max(fams, key=lambda f: (',' in f, fams[f]))
        text = re.sub(r'font-family: [^;"]+;? ?', '', text)
        rules += f'font-family: {fam}; '
    if sizes:
        size = sizes.most_common(1)[0][0]
        text = re.sub(r'font-size: %spx;? ?' % re.escape(size), '', text)
        rules += f'font-size: {size}px; '
    if rules:
        text = text.replace(' style=""', '').replace('; "', '"')
        text = f'<style type="text/css">text {{{rules}}}</style>' + text
    return text

def save_figure(figure, fname, ext='pdf', **kwargs):
    if ext == 'pdf':
        kwargs['metadata'] = PDF_NONE
    # svg keeps the exact figure canvas so the axes rect is identical across
    # plots; pdf expands to include labels drawn outside the axes
    figure.savefig(f'{fname}.{ext}', bbox_inches=None if ext == 'svg' else 'tight', **kwargs)
    if ext == 'svg':
        _fm_embed(figure, f'{fname}.{ext}')

@contextlib.contextmanager
def matplotlib_config(latex=False):
    try:
        matplotlib.rcParams['figure.dpi'] *= 2
        matplotlib.rcParams['figure.subplot.hspace'] = 0
        matplotlib.rcParams['figure.subplot.wspace'] /= 1.5
        # fixed margins — identical axes rect across all plots
        matplotlib.rcParams['figure.subplot.left'] = 0.10
        matplotlib.rcParams['figure.subplot.bottom'] = 0.11
        matplotlib.rcParams['figure.subplot.right'] = 0.99
        matplotlib.rcParams['figure.subplot.top'] = 0.97
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
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'Liberation Sans', 'sans-serif']
        matplotlib.rcParams['svg.fonttype'] = 'none'

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
    if 'sum' in data:
        _fm_total(what, data['sum'])
    if 'mixture' in mode:
        fit = []
        label = 'Model'
        if fun == 'cdf':
            fit = cdf(data, idx)
            label = what.capitalize() + ' (model)'
        if fun == 'pdf':
            fit = pdf(data, idx, x_val)

        artists = plt.plot(idx, fit, 'k-', lw=1, ms=1, alpha=0.5, label=label)
        plots += artists
        _fm_add('model', idx, fit, label=label, what=what, artists=artists)

    components = []
    if 'comp' in mode or 'comp_stack' in mode:
        comp = {}
        if fun == 'cdf':
            comp = cdf_comp(data, idx)
        if fun == 'pdf':
            comp = pdf_comp(data, idx, x_val)
        mix = data.get('mix', [])
        weights = np.array([entry[0] for entry in mix], dtype=float)
        wsum = weights.sum()
        for entry, (label, dd), wnorm in zip(mix, comp.items(), weights / wsum if wsum else weights):
            components.append({'label': label[:1].upper() + label[1:], 'y': dd, 'weight': wnorm, 'dist': entry[1], 'params': entry[2]})

    if 'comp' in mode:
        for c in components:
            artists = plt.plot(idx, c['y'], '--', dashes=(20, 20), lw=0.1,
                               label=c['label'] if 'comp_labels' in mode else None)
            plots += artists
            _fm_add('component', idx, c['y'], label=c['label'], what=what,
                    component={'weight': _fm_num(c['weight']), 'dist': c['dist'], 'params': c['params']},
                    artists=artists)

    if 'comp_stack' in mode:
        artists = plt.stackplot(idx, *[c['y'] for c in components], lw=0.1,
                                labels=[c['label'] for c in components] if 'comp_labels' in mode else [], alpha=0.5)
        plots += artists
        for c, artist in zip(components, artists):
            _fm_add('component', idx, c['y'], label=c['label'], what=what,
                    component={'weight': _fm_num(c['weight']), 'dist': c['dist'], 'params': c['params']},
                    artists=[artist])
    return plots

def plot_pdf(data, idx=None, x_val='length', what='flows', mode=frozenset(['points', 'mixture']), normalize=True, fft=False):
    kwargs = {'cmap': STYLE[what][1]}
    plots = []

    if isinstance(data, pd.DataFrame):
        if idx is None:
            idx = np.unique(np.rint(np.geomspace(data.index.min(), data.index.max(), LINE_NBINS)).astype(np.int64))

        data = data.loc[:, 'bin_hi':what + '_sum']

        pdfi = pdf_from_cdf(data, idx, what)
        xmin, xmax, ymin, ymax = calc_minmax(idx, pdfi)

        _fm_total(what, data[what + '_sum'].sum())

        if 'line' in mode:
            label = 'Infered from data points'
            artists = plt.plot(idx, pdfi, STYLE[what][0] + '-', lw=2, ms=1, alpha=0.5, zorder=3,
                               label=label)
            plots += artists
            _fm_add('data', idx, pdfi, label=label, what=what, artists=artists)

        values_sum = data[what + '_sum'].sum()
        weights = data['flows_sum'].to_numpy()

        if normalize:
            logmsg('Normalizing data')
            data = normalize_data(data)
            logmsg('Normalizing data')

        pdfd = data[what + '_sum'] / values_sum

        if 'points' in mode:
            logmsg('Plotting points')
            if normalize:
                label = 'Data points (normalized)'
                artists = plt.plot(pdfd.index, pdfd,
                                   STYLE[what][0] + ',', lw=1, ms=1, alpha=1, zorder=1,
                                   label=label, rasterized=True)
            else:
                label = 'Data points (log-binned)'
                artists = plt.plot(pdfd.index, pdfd,
                                   'c' + ',', lw=1, ms=1, alpha=1, zorder=2,
                                   label=label, rasterized=True)
            plots += artists
            _fm_add('points', pdfd.index.to_numpy(), pdfd.to_numpy(), label=label, what=what, artists=artists)

        if 'hist' in mode:
            logmsg('Plotting hist')
            plt.hist2d(pdfd.index, pdfd, weights=weights,
                       bins=[np.geomspace(xmin, xmax, HIST_NBINS), np.geomspace(ymin, ymax, HIST_NBINS)],
                       cmin=1, norm=colors.LogNorm(),
                       alpha=1.0, label=what.capitalize() + ' data', rasterized=False, **kwargs)

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
            idx = np.unique(np.rint(np.geomspace(data.index.min(), data.index.max(), LINE_NBINS)).astype(np.int64))
        _fm_total(what, data[what + '_sum'].sum())
        if 'line' in mode:
            cdfd = data[what + '_sum'].cumsum() / data[what + '_sum'].sum()
            cdfi = scipy.interpolate.interp1d(cdfd.index, cdfd, 'linear', bounds_error=False)(idx)
            label = what.capitalize() + ' (infered from data points)'
            artists = plt.plot(idx, cdfi, STYLE[what][0] + '-', lw=2, ms=1, alpha=0.5,
                               label=label)
            plots += artists
            _fm_add('data', idx, cdfi, label=label, what=what, artists=artists)
    else:
        plots += plot_mixture(data, idx, x_val, what, mode, 'cdf')

    plt.xscale('log')
    plt.legend(frameon=False)
    return plots

def plot_avg(data, idx=None, x_val='length', what='packets', mode=frozenset(['mixture'])):
    plots = []
    if isinstance(data, pd.DataFrame):
        if idx is None:
            idx = np.unique(np.rint(np.geomspace(data.index.min(), data.index.max(), LINE_NBINS)).astype(np.int64))

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

        artists = plt.plot(avg_points.index.values, avg_points.values,
                           color + ',', lw=1, ms=1, alpha=1,
                           label='Data points', rasterized=True)
        plots += artists
        _fm_add('points', avg_points.index.values, avg_points.values, label='Data points', what=what, artists=artists)

        label = 'Infered from data points'
        artists = plt.plot(idx, avg_line, color + '-', lw=2, ms=1, alpha=0.5,
                           label=label)
        plots += artists
        _fm_add('data', idx, avg_line, label=label, what=what, artists=artists)
    elif 'mixture' in mode:
        try:
            avg_mix = avg(data, idx, x_val, what)
        except KeyError:
            return plots

        plt.xscale('log')
        if what in ['packets', 'octets']:
            plt.yscale('log')

        label = 'Infered from model'
        artists = plt.plot(idx, avg_mix, 'k-', lw=2, ms=1, alpha=0.5,
                           label=label)
        plots += artists
        _fm_add('model', idx, avg_mix, label=label, what=what, artists=artists)

    plt.legend(frameon=False)
    return plots
