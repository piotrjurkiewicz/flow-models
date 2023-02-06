import pathlib
import datetime

import pandas as pd
import matplotlib.pyplot as plt

for f in sorted(pathlib.Path(".").glob("*.packets")):
    try:
        title = datetime.datetime.utcfromtimestamp(int(f.stem) * 86400).strftime('%Y-%m-%d-%a')
    except ValueError:
        title = f.stem
    s = pd.read_csv(f, header=None)
    ax = s.plot(kind='area', linewidth=0, alpha=0.5, legend=None, ylabel='packets/s', xlabel='second', xlim=(0, 86400), title=title)
    # s.rolling(600, min_periods=1).mean().plot(kind='line', linewidth=0.2, color='k', legend=None, ax=ax)
    plt.show()

for f in sorted(pathlib.Path(".").glob("*.octets")):
    try:
        title = datetime.datetime.utcfromtimestamp(int(f.stem) * 86400).strftime('%Y-%m-%d-%a')
    except ValueError:
        title = f.stem
    s = pd.read_csv(f, header=None)
    s *= 8
    ax = s.plot(kind='area', linewidth=0, alpha=0.5, legend=None, ylabel='bits/s', xlabel='second', xlim=(0, 86400), title=title)
    # s.rolling(600, min_periods=1).mean().plot(kind='line', linewidth=0.2, color='k', legend=None, ax=ax)
    plt.show()
