import datetime
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from flow_models.lib.plot import PDF_NONE

matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['font.family'] = 'sans'

for what in ['flows', 'packets', 'octets']:
    for f in sorted(pathlib.Path().glob(f'*.{what}')):
        s = pd.read_csv(f, header=None)
        if what == 'octets':
            s *= 8
            what = 'bits'
        try:
            title = f"{datetime.datetime.fromtimestamp(int(f.stem) * 86400, datetime.UTC).strftime('%Y-%m-%d-%a')} (Day {f.stem})"
        except ValueError:
            title = f"Day {f.stem}"
        print(title, what, sum(s[0]))
        title = None
        ax = s.rolling(30, min_periods=1).mean().plot(kind='line', linewidth=0.2, color='k', legend=None, ylabel=f'{what.capitalize()[:-1]} rate [{what}/s]', xlabel='Time [s]', xlim=(0, 86400), title=title)
        plt.ylim(bottom=0)

        # plt.axvspan(19 * 3600, 20 * 3600, linewidth=0,  color='lightsteelblue')
        # s = pd.read_csv(f.absolute().parent.parent / 'dorms' / f.name, header=None)
        # if what == 'bits':
        #     s *= 8
        # s.rolling(30, min_periods=1).mean().plot(ax=ax, kind='line', linewidth=0.2, color='r', legend=None)

        plt.savefig(f'{f.stem}.{what}.pdf', bbox_inches='tight', metadata=PDF_NONE)
