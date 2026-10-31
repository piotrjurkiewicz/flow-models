import datetime
import pathlib

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from flow_models.lib.plot import FM_MAX_POINTS, _fm_add, matplotlib_config, save_figure

ROLL = 30          # rolling-mean window [s]
MARGIN = 600       # padding around the nonzero data span [s]


def decimate(y, n=FM_MAX_POINTS):
    """Return ``(x, y)`` evenly decimated to at most ``n`` samples."""
    if len(y) <= n:
        return np.arange(len(y)), np.asarray(y)
    idx = np.linspace(0, len(y) - 1, n).astype(int)
    return idx, np.asarray(y)[idx]


def hhmm(sec):
    return f'{int(sec // 3600):02d}:{int(sec % 3600 // 60):02d}'


def fmt_suffix(v, _):
    for f, u in ((1e9, 'G'), (1e6, 'M'), (1e3, 'k')):
        if abs(v) >= f:
            return f'{v / f:g}{u}'
    return f'{v:g}'


def envelope(v, bins=1400):
    """Per-bin min/max envelope of raw samples, mimicking the per-pixel band."""
    v = np.asarray(v, dtype=float)
    k = max(1, len(v) // bins)
    n = len(v) // k
    v = v[:n * k].reshape(n, k)
    return np.arange(n) * k + k / 2, v.min(axis=1), v.max(axis=1)


with matplotlib_config():
    files = {}
    for suffix in ['flows', 'packets', 'octets']:
        for f in sorted(pathlib.Path().glob(f'*.{suffix}')):
            files.setdefault(f.stem, {})[suffix] = f

    for stem, by_suffix in sorted(files.items()):
        series = {}
        for suffix, f in by_suffix.items():
            s = pd.read_csv(f, header=None)[0]
            if suffix == 'octets':
                s = s * 8
            series[suffix] = s
        try:
            title = f"{datetime.datetime.fromtimestamp(int(stem) * 86400, datetime.timezone.utc).strftime('%Y-%m-%d-%a')} (Day {stem})"
        except ValueError:
            title = f"Day {stem}"

        # one x range for all metrics: union of nonzero spans, padded
        spans = {}
        for suffix, s in series.items():
            nz = np.nonzero(s.to_numpy())[0]
            if len(nz):
                spans[suffix] = (int(nz[0]), int(nz[-1]))
        n = len(next(iter(series.values())))
        lo = max(min(a for a, _ in spans.values()) - MARGIN, 0) if spans else 0
        hi = min(max(b for _, b in spans.values()) + MARGIN, n) if spans else n

        for suffix in ['flows', 'packets', 'octets']:
            if suffix not in series:
                continue
            s = series[suffix]
            what = 'bits' if suffix == 'octets' else suffix
            f = by_suffix[suffix]

            fig, ax = plt.subplots(figsize=(8, 3.2))
            dorms = f.parent / 'dorms' / f.name
            has_dorms = dorms.exists()
            ax.fill_between(*envelope(s.to_numpy()), color='k', alpha=0.15, lw=0)
            line, = ax.plot(*decimate(s.rolling(ROLL, min_periods=1).mean().to_numpy()), 'k-', lw=0.4,
                            label='all' if has_dorms else None)
            _fm_add('data', *decimate(s.rolling(ROLL, min_periods=1).mean().to_numpy()),
                    label='all' if has_dorms else None, what=what, artists=[line])
            print(title, what, sum(s))
            if has_dorms:
                sd = pd.read_csv(dorms, header=None)[0]
                if suffix == 'octets':
                    sd = sd * 8
                ax.fill_between(*envelope(sd.to_numpy()), color='tab:red', alpha=0.15, lw=0)
                line, = ax.plot(*decimate(sd.rolling(ROLL, min_periods=1).mean().to_numpy()), '-', lw=0.4,
                                color='tab:red', label='dorms')
                _fm_add('data', *decimate(sd.rolling(ROLL, min_periods=1).mean().to_numpy()),
                        label='dorms', what=what, artists=[line])
                print(' ' * len(title), f'dorms {what}', sum(sd))
                ax.legend(loc='upper left')

            step = next(t for t in (300, 600, 900, 1800, 3600, 7200, 14400) if (hi - lo) / t <= 10)
            ticks = np.arange(np.ceil(lo / step) * step, hi + 1, step)
            ax.set_xticks(ticks, [hhmm(t) for t in ticks])
            ax.set(xlim=(lo, hi),
                   xlabel='Time of day [UTC]', ylabel=f'{what.capitalize()[:-1]} rate [{what}/s]')
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_formatter(FuncFormatter(fmt_suffix))

            save_figure(fig, f'{f.stem}.{what}', 'pdf')
            save_figure(fig, f'{f.stem}.{what}', 'svg')
            plt.close(fig)
