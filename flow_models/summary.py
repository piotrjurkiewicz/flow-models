#!/usr/bin/python3
"""Produces TeX tables containing summary statistics of flow dataset (requires `scipy`)."""

import argparse
import pathlib

import numpy as np
import pandas as pd
import scipy.interpolate

from flow_models.lib.data import UNITS, detect_x_value
from flow_models.lib.util import logmsg

EPILOG = \
"""
This tool can be used to generate LaTeX summary of flow histogram statistics.

Example:

    flow_models.summary histograms/udp/length.csv
"""

X_VALUES = ['length', 'size', 'duration', 'rate']
Y_VALUES = ['flows', 'packets', 'octets']

def summary(obj, x_val=None):
    if isinstance(obj, (str, pathlib.Path)):
        file = pathlib.Path(obj)
        logmsg(f'Loading file {file}')
        data = pd.read_csv(file, index_col=0, sep=',', low_memory=False,
                           usecols=lambda col: not col.endswith('_ssq'))
        logmsg(f'Loaded file {file}')
    else:
        data = obj

    print(stats_summary(data))
    print('\n')
    print(cdf_summary(data, x_val))
    print('\n')

def stats_summary(data):
    if isinstance(data, pd.DataFrame):
        s = ['\\begin{tabular}{@{}lrr@{}}',
             '\\toprule',
             '\\textbf{Dataset name} & XXX & \\\\',
             '\\textbf{Exporter} & XXX & \\\\',
             '\\textbf{L2 technology} & XXX & \\\\',
             '\\textbf{Sampling rate} & none & \\\\',
             '\\textbf{Active timeout} & XXX & seconds \\\\',
             '\\textbf{Inactive timeout} & XXX & seconds \\\\',
             '\\midrule']
        for what in Y_VALUES:
            col = what + '_sum'
            if col in data:
                tot = data[col].sum()
                s.append(f'\\textbf{{Number of {what}}} & ' + f'{tot:,}'.replace(',', ' ') + f' & {what} \\\\')
        for what in X_VALUES:
            col = UNITS[what]
            if col not in Y_VALUES:
                col = what
            col = col + '_sum'
            if col in data:
                tot = data[col].sum() / data['flows_sum'].sum()
                s.append(f'\\textbf{{Average flow {what}}} & ' + f'{tot:.6f}' + f' & {UNITS[what]} \\\\')
        tot = data['octets_sum'].sum() / data['packets_sum'].sum()
        s.append('\\textbf{Average packet size} & ' + f'{tot:.6f}' + ' & bytes \\\\')
        s += ['\\bottomrule',
              '\\end{tabular}']
        return '\n'.join(s)

    return None

def cdf_summary(data, x_val):
    if isinstance(data, pd.DataFrame):
        if not x_val:
            x_val = detect_x_value(data.index)
        points = np.power(2, range(25))
        if x_val == 'length':
            pass
        elif x_val == 'size':
            points *= 64
        else:
            raise NotImplementedError

        vals = {}
        for what in Y_VALUES:
            cdf = data[what + '_sum'].cumsum() / data[what + '_sum'].sum()
            cdfi = scipy.interpolate.interp1d(cdf.index, cdf, 'linear', bounds_error=False)(np.array(points))
            vals[what] = cdfi * 100
        s = ['\\begin{tabular}{@{}lrrr@{}}',
             '\\toprule',
             f'\\textbf{{Flows of {x_val} up to}} & ' + '\\multicolumn{3}{c}{\\textbf{Make up \\%}} \\\\',
             '\\cmidrule(lr){2-4}',
             f'\\multicolumn{{1}}{{c}}{{({UNITS[x_val]})}} & of ' + ' & of '.join(Y_VALUES) + ' \\\\',
             '\\midrule']
        for n in range(len(points)):
            s.append(f'{points[n]} & ' + ' & '.join(f'{vals[what][n]:.4f}' for what in Y_VALUES) + ' \\\\')
        s += ['\\bottomrule',
              '\\end{tabular}']
        return '\n'.join(s)

    return None

def parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=EPILOG)
    p.add_argument('-x', choices=X_VALUES, help='x axis value')
    p.add_argument('file', help='csv_hist file to summarize')
    return p

def main():
    app_args = parser().parse_args()

    summary(app_args.file, app_args.x)


if __name__ == '__main__':
    main()
