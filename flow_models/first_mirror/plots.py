#!/usr/bin/python3
"""Plots first-N-packets mirroring simulation results.

Reads ``{day}.packets_mirrored``/``{day}.octets_mirrored`` per-second counter
series written by ``flow_models.first_mirror.simulate`` from mirror directories
named by N and draws mirrored packet/bit rates over the day (one curve per N)
plus a per-day totals figure.
"""

import argparse
import collections
import datetime
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from flow_models.lib.plot import matplotlib_config, save_figure

ROLL = 30          # rolling-mean window [s]
MARGIN = 600       # padding around the nonzero data span [s]
FIGSIZE = [8, 3.2]

METRICS = {'packets': ('packets_mirrored', 1, 'Mirrored packet rate [packets/s]'),
           'bits': ('octets_mirrored', 8, 'Mirrored bit rate [bits/s]')}
TRAFFIC_FILES = {'packets': 'packets', 'bits': 'octets'}

def hhmm(sec):
    return f'{int(sec // 3600):02d}:{int(sec % 3600 // 60):02d}'

def fmt_suffix(v, _):
    for f, u in ((1e9, 'G'), (1e6, 'M'), (1e3, 'k')):
        if abs(v) >= f:
            return f'{v / f:g}{u}'
    return f'{v:g}'

def day_title(day):
    try:
        return datetime.datetime.fromtimestamp(int(day) * 86400, datetime.timezone.utc).strftime('%Y-%m-%d-%a')
    except ValueError:
        return f'Day {day}'

def save_fig(fig, out, fmt):
    for ext in (['pdf', 'svg'] if fmt == 'both' else [fmt]):
        save_figure(fig, str(out), ext)

def load(dirs):
    """Load ``{n: {day: {suffix: series}}}`` from mirror directories named by N."""
    data = collections.defaultdict(dict)
    for d in dirs:
        d = pathlib.Path(d)
        for f in sorted(d.iterdir()):
            day, _, suffix = f.name.partition('.')
            if suffix in ('packets_mirrored', 'octets_mirrored'):
                data[int(d.name)].setdefault(day, {})[suffix] = \
                    pd.read_csv(f, header=None)[0].to_numpy()
    return data

def load_traffic(series, days):
    """Overall traffic per-second series per ``{day: {metric: series}}``."""
    traffic = {}
    for day in days:
        for metric, name in TRAFFIC_FILES.items():
            f = series / f'{day}.{name}'
            if f.exists():
                traffic.setdefault(day, {})[metric] = pd.read_csv(f, header=None)[0].to_numpy()
    return traffic

def plot_day(data, day, metric, out, fmt, traffic=None):
    suffix, scale, ylabel = METRICS[metric]
    series = [(n, days[day][suffix] * scale) for n, days in sorted(data.items())
              if day in days and suffix in days[day]]
    if not series:
        return
    n = len(series[0][1]) - 1
    lo, hi = n, -1
    for _, s in series:
        nz = np.nonzero(s)[0]
        if len(nz):
            lo, hi = min(lo, nz[0]), max(hi, nz[-1])
    span = hi - lo + 1 if hi >= 0 else 0
    if hi < 0:
        lo, hi = 0, n
    else:
        lo, hi = max(lo - MARGIN, 0), min(hi + MARGIN, n)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for count, s in series:
        ax.plot(np.arange(n + 1), pd.Series(s).rolling(ROLL, min_periods=1).mean(),
                lw=0.5, label=f'first {count}')
    step = next(t for t in (300, 600, 900, 1800, 3600, 7200, 14400) if (hi - lo) / t <= 10)
    ticks = np.arange(np.ceil(lo / step) * step, hi + 1, step)
    ax.set_xticks(ticks, [hhmm(t) for t in ticks])
    ax.set(xlim=(lo, hi), title=day_title(day), xlabel='Time of day [UTC]', ylabel=ylabel)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_suffix))
    ax.legend()
    if traffic is not None and span and (mean := traffic.sum() * scale / span):
        sec = ax.secondary_yaxis('right', functions=(
            lambda v: v / mean * 100, lambda p: p * mean / 100))
        sec.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}%'))
        fig.subplots_adjust(right=0.93)  # svg keeps the exact canvas — leave room for % labels
    save_fig(fig, out / f'{day}.{metric}', fmt)
    plt.close(fig)

def plot_total(data, out, fmt):
    counts = sorted(n for n, days in data.items() if days)
    if not counts:
        return
    def total(n, suffix):
        return sum(s.sum() for days in data[n].values()
                   for name, s in days.items() if name == suffix)
    fig, ax = plt.subplots(figsize=[FIGSIZE[0] * 0.66, FIGSIZE[1]])
    tx = ax.twinx()
    ax.plot(counts, [total(n, 'packets_mirrored') for n in counts], 'o-',
            color='tab:red', label='mirrored packets')
    tx.plot(counts, [total(n, 'octets_mirrored') for n in counts], 's-',
            color='tab:blue', label='mirrored octets')
    ax.set_xticks(counts)
    ax.set(xlabel='Packets mirrored per flow [N]', ylabel='Packets [per day]')
    tx.set_ylabel('Octets [per day]')
    ax.set_ylim(bottom=0)
    tx.set_ylim(bottom=0)
    fig.subplots_adjust(right=0.86)  # svg keeps the exact canvas — leave room for twin-axis labels
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_suffix))
    tx.yaxis.set_major_formatter(FuncFormatter(fmt_suffix))
    lines = ax.get_lines() + tx.get_lines()
    ax.legend(lines, [line.get_label() for line in lines])
    save_fig(fig, out / 'total', fmt)
    plt.close(fig)

def plot(dirs, fmt='pdf', out='.', series=None):
    data = load(dirs)
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    days = sorted({d for days in data.values() for d in days})
    traffic = load_traffic(pathlib.Path(series), days) if series else {}
    for day in days:
        for metric in METRICS:
            plot_day(data, day, metric, out, fmt, traffic.get(day, {}).get(metric))
    plot_total(data, out, fmt)

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--format', default='pdf', choices=['pdf', 'svg', 'both'],
                   help='output figure format')
    p.add_argument('-O', '--out', default='.', help='output directory')
    p.add_argument('--series', default=None, metavar='DIR',
                   help='directory with overall {day}.{packets,octets} series; enables a'
                        ' second y axis showing the mirrored share of mean traffic')
    p.add_argument('dirs', nargs='+', help='first_mirror/N directories')
    return p

def main():
    app_args = parser().parse_args()

    with matplotlib_config():
        plot(app_args.dirs, app_args.format, app_args.out, app_args.series)


if __name__ == '__main__':
    main()
