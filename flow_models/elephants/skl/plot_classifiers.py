#!/usr/bin/python3
"""Generates plots from train_classifier results."""

import argparse
import collections
import json
import pathlib
import re

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.lines import Line2D

from flow_models.lib.ml import calculate_reduction_from_mixture, interp_reduction, plot_style
from flow_models.lib.plot import PDF_NONE
from flow_models.lib.util import logmsg

matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['font.family'] = 'sans'

TSV_COLUMNS = [
    'training_coverage', 'coverage', 'reduction',
    'min_coverage', 'avg_coverage', 'max_coverage',
    'min_occupancy', 'avg_occupancy', 'max_occupancy',
    'true_negatives', 'false_positives', 'false_negatives', 'true_positives',
]

COLUMN_NAMES = {
    'training_coverage': 'Training\nTraffic\nCoverage',
    'coverage': 'Resulting\nTotal\nTraffic\nCoverage',
    'reduction': 'Flow\nOperations\nReduction',
    'min_coverage': 'Resulting\nMin\nTraffic\nCoverage',
    'avg_coverage': 'Resulting\nAverage\nTraffic\nCoverage',
    'max_coverage': 'Resulting\nMax\nTraffic\nCoverage',
    'min_occupancy': 'Min\nOccupancy\nReduction',
    'avg_occupancy': 'Average\nOccupancy\nReduction',
    'max_occupancy': 'Max\nOccupancy\nReduction',
    'accuracy': 'Accuracy',
    'true_positives': 'True\nPositives',
    'true_negatives': 'True\nNegatives',
    'false_positives': 'False\nPositives',
    'false_negatives': 'False\nNegatives',
    'tpr': 'TPR\n(Recall)',
    'tnr': 'TNR\n(Specificity)',
    'fpr': 'FPR',
    'fnr': 'FNR',
    'true_precision': 'Precision',
    'true_recall': 'Recall',
    'true_fscore': 'FScore',
    'informedness': 'BM\n(Informed-\nness)',
    'ppv': 'PPV\n(Precision)',
    'fdr': 'FDR',
    'for': 'FOR',
    'npv': 'NPV',
    'markedness': 'MK\n(Marked-\nness)',
    'mcc': 'MCC',
    'ts': 'TS',
}

class Normalize(matplotlib.colors.Normalize):
    def __call__(self, value, clip=None):
        return 0.8 * super().__call__(value, clip)

class LogNorm(matplotlib.colors.LogNorm):
    def __call__(self, value, clip=None):
        return 0.8 * super().__call__(value, clip)

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-O', '--output', default='sklearn', help='output directory and plot file name')
    p.add_argument('--mixture', help='')
    p.add_argument('files', help='directory')
    return p

def load_data(app_args):
    results = {}

    if app_args.mixture:
        results['Mixture'] = calculate_reduction_from_mixture(app_args.mixture)

    for f in sorted(pathlib.Path(f'{app_args.files}/').glob('*.tsv')):
        logmsg(f'Loading file {f}')
        columns = []
        columns.extend(np.loadtxt(str(f), delimiter='\t', max_rows=len(TSV_COLUMNS) - 4))
        columns.extend(np.loadtxt(str(f), delimiter='\t', skiprows=len(TSV_COLUMNS) - 4, max_rows=4, dtype=np.uint64))
        if not any(a in str(f) for a in ['Data', 'KNeighborsClassifier',  'HistGradientBoostingClassifier']):
            columns.extend(np.loadtxt(str(f), delimiter='\t', skiprows=len(TSV_COLUMNS)))
        results[f.stem] = columns

    # decisions_true = {}
    # decisions_predicted = {}
    #
    # for f in sorted(pathlib.Path(f'{app_args.files}/').glob('*.dt.npz')):
    #     decisions_true[f.stem[:-3]] = np.load(str(f))
    #
    # for f in sorted(pathlib.Path(f'{app_args.files}/').glob('*.dp.npz')):
    #     decisions_predicted[f.stem[:-3]] = np.load(str(f))

    res = {}
    for part in ['Mixture', 'Data', 'DecisionTree', 'RandomForest', 'ExtraTrees']:
        for name in results:
            if part in name:
                res[name] = results[name]

    dfd = {}
    for name, columns in res.items():
        logmsg(f'Processing {name}')
        parts = re.findall(r'\w+|{[^}]*}|\d+|\(\w+\)', name)
        if len(parts) >= 4:
            algo = parts[0]
            params = parts[1]
            tt = parts[-1]
            fold = parts[-2]
            prep = 'raw'
            if not algo.startswith('Data'):
                if 'octets' in parts[-3]:
                    prep = 'octets'
                if 'bit' in parts[-3]:
                    prep = 'bits'
            if params:
                params = json.loads(params.replace("\'", "\""))
                if 'n_estimators' in params:
                    algo += f"{params['n_estimators']}t"
                if 'max_depth' in params:
                    algo += f"{params['max_depth']}d"
                if 'sw' in params:
                    algo += f"{params['sw']}"
            dfp = {}
            for n, column in enumerate(columns):
                dfp[TSV_COLUMNS[n]] = column
                if n == len(TSV_COLUMNS) - 1:
                    break
            if not algo.startswith('Data'):
                dfp['accuracy'] = (dfp['true_positives'] + dfp['true_negatives']) / (dfp['true_positives'] + dfp['true_negatives'] + dfp['false_positives'] + dfp['false_negatives'])
                dfp['true_fscore'] = (2 * dfp['true_positives']) / (2 * dfp['true_positives'] + dfp['false_positives'] + dfp['false_negatives'])
                dfp['tpr'] = dfp['true_positives'] / (dfp['true_positives'] + dfp['false_negatives'])
                dfp['fnr'] = dfp['false_negatives'] / (dfp['true_positives'] + dfp['false_negatives'])
                dfp['fpr'] = dfp['false_positives'] / (dfp['false_positives'] + dfp['true_negatives'])
                dfp['tnr'] = dfp['true_negatives'] / (dfp['false_positives'] + dfp['true_negatives'])
                dfp['informedness'] = dfp['tpr'] + dfp['tnr'] - 1.0
                dfp['ppv'] = dfp['true_positives'] / (dfp['true_positives'] + dfp['false_positives'])
                dfp['fdr'] = dfp['false_positives'] / (dfp['true_positives'] + dfp['false_positives'])
                dfp['for'] = dfp['false_negatives'] / (dfp['false_negatives'] + dfp['true_negatives'])
                dfp['npv'] = dfp['true_negatives'] / (dfp['false_negatives'] + dfp['true_negatives'])
                dfp['markedness'] = dfp['ppv'] + dfp['npv'] - 1.0
                dfp['mcc'] = np.sqrt(dfp['tpr'] * dfp['tnr'] * dfp['ppv'] * dfp['npv']) - np.sqrt(dfp['fnr'] * dfp['fpr'] * dfp['for'] * dfp['fdr'])
                dfp['ts'] = dfp['true_positives'] / (dfp['true_positives'] + dfp['false_negatives'] + dfp['false_positives'])
                if any(algo.startswith(alg) for alg in ['DecisionTreeClassifier', 'RandomForestClassifier', 'ExtraTreesClassifier']):
                    dfp['avg_leaves'] = columns[-2]
            dfd[algo, prep, tt, fold] = dfp

    new_dfd = {}
    interp_dfd = {}
    for key, dfp in dfd.items():
        algo, prep, tt, fold = key
        interp_coverage = np.arange(50, 101, 0.1)
        interp_dfp = {}
        for col in dfp:
            interp_dfp[col] = interp_reduction(interp_coverage, 100 * dfp['avg_coverage'], dfp[col])[1]
        reduction = np.append(dfp['reduction'], 1.0)
        coverage = np.append(dfp['avg_coverage'], 1.0)
        _, new_reduction = interp_reduction(interp_coverage, 100 * coverage, reduction)
        interp_dfp['reduction'] = new_reduction
        interp_dfd[algo, prep, tt, fold] = pd.DataFrame(interp_dfp, index=interp_coverage)
        if algo != 'Data':
            new_dfd[algo, prep, tt, fold] = pd.DataFrame(dfp, index=dfp['training_coverage'])

    return new_dfd, interp_dfd, res

def heatmap(ax, val, cmap, vformat, norm, xticks=None, yticks=None, std=None, title=None, xlabel=None):
    ax.pcolor(val, norm=norm, cmap=cmap)
    ax.set_frame_on(False)

    # put the major ticks at the middle of each cell
    if xticks:
        ax.set_xticks(np.arange(val.to_numpy().shape[1]) + 0.5, xticks, minor=False)
    elif xticks is None:
        ax.set_xticks(np.arange(val.to_numpy().shape[1]) + 0.5, [COLUMN_NAMES.get(k, k) for k in val.columns], minor=False)
    else:
        ax.set_xticks([], minor=False)
    if yticks:
        ax.set_yticks(np.arange(val.to_numpy().shape[0]) + 0.5, yticks, minor=False)
    else:
        ax.set_yticks([], minor=False)

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.tick_params(tick1On=False)
    ax.tick_params(tick2On=False)

    if title:
        ax.set_title(title, fontsize='medium', pad=8.0)

    if xlabel:
        ax.set_xlabel(xlabel, labelpad=8.0)

    for i in range(val.to_numpy().shape[0]):
        for j in range(val.to_numpy().shape[1]):
            v = val.to_numpy()[i, j]
            col = 'k' if norm(v) > 0.75 else 'k'
            if std is not None:
                s = std.to_numpy()[i, j]
                ax.text(j + 0.5, i + 0.5, f"{format(v, vformat)}   ± {format(s, vformat)}", ha='center', va='center', color=col)
            else:
                ax.text(j + 0.5, i + 0.5, format(v, vformat), ha='center', va='center', color=col)

def main():
    app_args = parser().parse_args()
    output = app_args.output

    dfd, new_dfd, res = load_data(app_args)

    for prep in ['raw', 'octets', 'bits']:
        for loc in [4, 9, 14]:
            tcov = {4: '80', 9: '96', 14: '99.2'}[loc]
            df = pd.concat(dfd, axis=1).reorder_levels([4, 0, 1, 2, 3], axis=1)
            df.columns.names = ['col', 'algo', 'prep', 'tt', 'fold']
            z = df.xs('(test)', 1, level='tt').iloc[loc].T.unstack(0)
            z = z.reindex([prep], level='prep')
            # z = z.reindex(['DecisionTreeClassifier', 'RandomForestClassifier', 'ExtraTreesClassifier'], level='algo')
            fig, ax = plt.subplots(ncols=21, width_ratios=[*(21 * [1])])
            fig.set_size_inches(17, 22)
            fig.tight_layout(w_pad=0.0)
            fig.subplots_adjust(wspace=0)
            iax = iter(ax)
            # heatmap(next(iax), z[['training_coverage']], 'Blues', '1.3f', Normalize(0.0, 1.4), yticks=[f"{a}, {b}, {c}" for (a, b, c) in z.index])
            # heatmap(next(iax), z[['coverage']], 'Blues', '1.3f', Normalize(0.0, 1.4))
            # heatmap(next(iax), z[['min_coverage']], 'Blues', '1.3f', Normalize(0.0, 1.4))
            heatmap(next(iax), z[['avg_coverage']], 'Blues', '1.3f', Normalize(0.0, 1.4), yticks=[f"{a}, {b}, {c}" for (a, b, c) in z.index])
            # heatmap(next(iax), z[['max_coverage']], 'Blues', '1.3f', Normalize(0.0, 1.4))
            heatmap(next(iax), z[['true_positives']], 'Greens', '0.0f', Normalize(0, 130320))
            heatmap(next(iax), z[['true_negatives']], 'Greens', '0.0f', Normalize(0, 1303200))
            heatmap(next(iax), z[['false_positives']], 'Reds', '0.0f', Normalize(0, 1303200))
            heatmap(next(iax), z[['false_negatives']], 'Reds', '0.0f', Normalize(0, 130320))
            heatmap(next(iax), z[['tpr']], 'Greens', '1.3f', Normalize(0.0, 1.4))
            heatmap(next(iax), z[['tnr']], 'Greens', '1.3f', Normalize(0.0, 1.4))
            heatmap(next(iax), z[['fpr']], 'Reds', '1.3f', Normalize(0.0, 1.4))
            heatmap(next(iax), z[['fnr']], 'Reds', '1.3f', Normalize(0.0, 1.4))
            heatmap(next(iax), z[['ppv']], 'Greens', '1.3f', Normalize(0.0, 1.4))
            heatmap(next(iax), z[['npv']], 'Greens', '1.3f', Normalize(0.0, 1.4))
            heatmap(next(iax), z[['fdr']], 'Reds','1.3f', Normalize(0.0, 1.4))
            heatmap(next(iax), z[['for']], 'Reds','1.3f', Normalize(0.0, 1.4))
            heatmap(next(iax), z[['accuracy']], 'YlGn', '1.3f', Normalize(0.7, 1.0))
            heatmap(next(iax), z[['true_fscore']], 'YlGn', '1.3f', Normalize(0.0, 0.7))
            heatmap(next(iax), z[['informedness']], 'YlGn', '1.3f', Normalize(0.0, 0.7))
            heatmap(next(iax), z[['markedness']], 'YlGn', '1.3f', Normalize(0.0, 0.7))
            heatmap(next(iax), z[['mcc']], 'YlGn', '1.3f', Normalize(0.0, 0.7))
            heatmap(next(iax), z[['ts']], 'YlGn', '1.3f', Normalize(0.0, 0.7))
            heatmap(next(iax), z[['reduction']], 'YlGn', '2.2f', LogNorm())
            # heatmap(next(iax), z[['min_occupancy']], 'YlGn', '2.2f', LogNorm())
            heatmap(next(iax), z[['avg_occupancy']], 'YlGn', '2.2f', LogNorm())
            # heatmap(next(iax), z[['max_occupancy']], 'YlGn', '2.2f', LogNorm())
            fig.suptitle(f"Training traffic coverage: {tcov}%", fontsize='medium', y=1.0345)  # 1.055
            ax[0].add_line(Line2D([0, 21], [-3.60, -3.60], lw=0.5, color='k', clip_on=False))
            fig.savefig(f'{output}-full-{prep}-{tcov}.pdf', bbox_inches='tight', metadata=PDF_NONE, pad_inches=0.02)
            plt.close(fig)

    df = pd.concat(new_dfd, axis=1).reorder_levels([4, 0, 1, 2, 3], axis=1)
    df.columns.names = ['col', 'algo', 'prep', 'tt', 'fold']
    agg = df.xs('(test)', 1, level='tt').T.groupby(level=[0, 1, 2], sort=False).agg(['mean', 'min', 'max', 'std']).unstack(0)
    agg = agg.reindex(['raw', 'octets', 'bits'], level=1)

    for what in ['DecisionTreeClassifier', 'RandomForestClassifier', 'ExtraTreesClassifier']:
        z = agg.drop(agg.index.levels[0][~agg.index.levels[0].str.contains(what)])
        for cov in [70.00000000000028, 80.00000000000043, 90.00000000000057]:
            fig, ax = plt.subplots(nrows=1, ncols=14, width_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            ax = ax.ravel()
            fig.set_size_inches(17, 6)
            fig.tight_layout(w_pad=0.0)
            fig.subplots_adjust(wspace=0, hspace=0.05)
            iax = iter(ax)

            v = z.loc(axis=1)[cov, 'mean']
            s = z.loc(axis=1)[cov, 'std']
            xticks = None

            heatmap(next(iax), v[['avg_coverage']], 'Blues', '1.4f', Normalize(0.0, 1.4), yticks=[f"{a}, {b}" for (a, b) in v.index], xticks=xticks)
            heatmap(next(iax), v[['training_coverage']], 'Blues', '1.4f', Normalize(0.95, 1.0), xticks=xticks)
            # heatmap(next(iax), v[['coverage']], 'Blues', '1.2f', Normalize(0.0, 1.4), std=s[['coverage']], xticks=xticks)
            heatmap(next(iax), v[['tpr']], 'Greens', '1.2f', Normalize(), std=s[['tpr']], xticks=xticks)
            heatmap(next(iax), v[['tnr']], 'Greens', '1.2f', Normalize(), std=s[['tnr']], xticks=xticks)
            # heatmap(next(iax), v[['fpr', 'fnr']], 'Reds', '1.2f', Normalize(0.0, 1.4), std=s[['fpr', 'fnr']], xticks=xticks)
            heatmap(next(iax), v[['ppv']], 'Greens', '1.2f', Normalize(), std=s[['ppv']], xticks=xticks)
            heatmap(next(iax), v[['npv']], 'Greens', '1.2f', Normalize(), std=s[['npv']], xticks=xticks)
            # heatmap(next(iax), v[['fdr', 'for']], 'Reds','1.2f', Normalize(0.0, 1.4), std=s[['fdr', 'for']], xticks=xticks)
            heatmap(next(iax), v[['accuracy']], 'YlGn', '1.2f', Normalize(), std=s[['accuracy']], xticks=xticks)
            heatmap(next(iax), v[['true_fscore']], 'YlGn', '1.2f', Normalize(), std=s[['true_fscore']], xticks=xticks)
            heatmap(next(iax), v[['informedness']], 'YlGn', '1.2f', Normalize(), std=s[['informedness']], xticks=xticks)
            heatmap(next(iax), v[['markedness']], 'YlGn', '1.2f', Normalize(), std=s[['markedness']], xticks=xticks)
            heatmap(next(iax), v[['mcc']], 'YlGn', '1.2f', Normalize(), std=s[['mcc']], xticks=xticks)
            heatmap(next(iax), v[['ts']], 'YlGn', '1.2f', Normalize(), std=s[['ts']], xticks=xticks)
            heatmap(next(iax), v[['reduction']], 'YlGn', '2.2f', LogNorm(), std=s[['reduction']], xticks=xticks)
            heatmap(next(iax), v[['avg_occupancy']], 'YlGn', '2.2f', LogNorm(), std=s[['avg_occupancy']], xticks=xticks)

            fig.savefig(f'{output}-res-{what}-{int(cov)}.pdf', bbox_inches='tight', metadata=PDF_NONE, pad_inches=0.02)
            plt.close(fig)

    z = agg.loc(axis=1)[[70.00000000000028, 80.00000000000043, 90.00000000000057]]
    z = z.drop('Data', errors='ignore')
    fig, ax = plt.subplots(ncols=1, nrows=1, width_ratios=[1])
    fig.set_size_inches(4, 15)
    fig.tight_layout(w_pad=0.0)
    heatmap(ax, z.loc(axis=1)[:, 'mean', 'avg_leaves'], 'Greens', '.0f', LogNorm(), xticks=['70%', '80%', '90%'], yticks=[f"{a}, {b}" for (a, b) in z.index], xlabel="Average Number of Tree Leaves (mean)", title="Resulting Average Traffic Coverage")
    fig.savefig(f'{output}-leaves.pdf', bbox_inches='tight', metadata=PDF_NONE, pad_inches=0.02)
    plt.close(fig)

    z = agg.loc(axis=1)[[70.00000000000028, 80.00000000000043, 90.00000000000057]]
    fig, ax = plt.subplots(ncols=2, nrows=1, width_ratios=[1, 1])
    ax = ax.ravel()
    fig.set_size_inches(10, 14.5)
    fig.tight_layout(w_pad=0.0)
    iax = iter(ax)
    heatmap(next(iax), z.loc(axis=1)[:, 'mean', 'reduction'], 'YlGn', '7.2f', LogNorm(), xticks=['70%', '80%', '90%'], yticks=[f"{a}, {b}" for (a, b) in z.index], std=z.loc(axis=1)[:, 'std', 'reduction'], xlabel="Flow Operations Reduction (mean ± std) [x]", title="Resulting Average Traffic Coverage")
    heatmap(next(iax), z.loc(axis=1)[:, 'mean', 'avg_occupancy'], 'YlGn', '7.2f', LogNorm(), xticks=['70%', '80%', '90%'], std=z.loc(axis=1)[:, 'std', 'avg_occupancy'], xlabel="Average Occupancy Reduction (mean ± std) [x]", title="Resulting Average Traffic Coverage")
    fig.savefig(f'{output}-combo.pdf', bbox_inches='tight', metadata=PDF_NONE, pad_inches=0.02)
    plt.close(fig)

    z = agg.loc(axis=1)[[70.00000000000028, 80.00000000000043, 90.00000000000057]]
    fig, ax = plt.subplots(ncols=2, nrows=1, width_ratios=[1, 1])
    ax = ax.ravel()
    fig.set_size_inches(10, 14.5)
    fig.tight_layout(w_pad=0.0)
    iax = iter(ax)
    z = z.drop('Data', errors='ignore')
    heatmap(next(iax), z.loc(axis=1)[:, 'mean', 'training_coverage'], 'Blues', '8.6f', LogNorm(), xticks=['70%', '80%', '90%'], yticks=[f"{a}, {b}" for (a, b) in z.index], xlabel="Training Traffic Coverage (mean)", title="Resulting Average Traffic Coverage")
    heatmap(next(iax), z.loc(axis=1)[:, 'mean', 'accuracy'], 'YlGn', '7.3f', Normalize(), xticks=['70%', '80%', '90%'], std=z.loc(axis=1)[:, 'std', 'accuracy'], xlabel="Accuracy (mean ± std)", title="Resulting Average Traffic Coverage")
    fig.savefig(f'{output}-combo2.pdf', bbox_inches='tight', metadata=PDF_NONE, pad_inches=0.02)
    plt.close(fig)

    agg = agg.reindex(['Data', 'DecisionTreeClassifier', 'DecisionTreeClassifier15d', 'DecisionTreeClassifier20d', 'RandomForestClassifier', 'RandomForestClassifier23t15d', 'RandomForestClassifier23t20d', 'ExtraTreesClassifier', 'ExtraTreesClassifier23t20d', 'ExtraTreesClassifier23t25d'], level='algo')
    zoom = False
    for chosen_prep in ['all', 'bits', 'octets', 'raw']:
        for what in ['reduction', 'avg_occupancy']:

            ls = {'raw': '-', 'octets': '--', 'bits': ':'}
            cs = {'Data': 'k'}
            plt.figure(figsize=(10, 5))

            ps = collections.defaultdict(list)
            if 'Mixture' in res:
                ps['Mixture', ''] += plt.plot(100 * res['Mixture'][1], res['Mixture'][2], lw=1, color='k', linestyle='-')
                # plt.plot(100 * res['Data (all)'][1], res['Data (all)'][2], label='Data', lw=1, color='b', linestyle='-')

            for name, prep in agg.index:
                if name == 'Data' or prep == chosen_prep or chosen_prep == 'all':
                    if name not in cs:
                        cs[name] = f'C{int(len(cs) - 1)}'
                    color = cs[name]
                    zgg = agg.loc[name, prep][:, :, what]
                    ps[name, prep] += plt.plot(zgg.loc[:, 'mean'], lw=1, alpha=0.75, color=color, linestyle=ls[prep])
                    if chosen_prep != 'all':
                        # ps[name, prep] += [plt.fill_between(agg.columns.levels[0], zgg.loc[:, 'min'], zgg.loc[:, 'max'], lw=0, alpha=0.25, color=color)]
                        ps[name, prep] += [plt.fill_between(agg.columns.levels[0], zgg.loc[:, 'mean'] - zgg.loc[:, 'std'], zgg.loc[:, 'mean'] + zgg.loc[:, 'std'], lw=0, alpha=0.25, color=color)]

            plot_style()
            plt.xlabel('Resulting Average Traffic Coverage [%]')
            plt.ylabel('Flow Operations Reduction [x]')
            if 'occupancy' in what:
                plt.ylim(1, 1000)
                plt.ylabel('Average Occupancy Reduction [x]')
            if zoom:
                plt.xlim(70, 90)
                plt.ylim(1, 1000)
                if 'occupancy' in what:
                    plt.ylim(1, 100)
                ps.pop(('Mixture', ''), None)
                ps.pop(('Data', 'raw'), None)
            plt.legend([tuple(v) for v in ps.values()], [f"{name}, {prep}" if name not in ['Mixture', 'Data'] else name for name, prep in ps], fontsize=7, loc=1)
            plt.savefig(f"{output}-{what}-{chosen_prep}{'-zoom' if zoom else ''}.pdf", bbox_inches='tight', metadata=PDF_NONE)
            plt.close()


if __name__ == '__main__':
    main()
