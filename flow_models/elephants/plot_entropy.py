#!/usr/bin/python3
"""Generates plot of features entropy and (optionally) importances."""

import argparse
import itertools

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from flow_models.lib.ml import load_arrays, prepare_input
from flow_models.lib.plot import PDF_NONE

matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['font.family'] = 'sans'

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', help='binary flow records directory')
    return p

def calculate_entropy(directory):
    """
    Calculate entropy of 5-tuple features for given flow records directory.

    Parameters
    ----------
    directory : os.PathLike
        binary flow records directory

    Returns
    -------
    dict
        entropy for subsequent bytes and bits of (sa, da, sp, dp, prot) fields
    """

    sa, da, sp, dp, prot, oc = load_arrays(directory)
    inp = prepare_input((sa, da, sp, dp, prot), octets=True)

    d = pd.DataFrame(inp)

    ent = {"bytes": [], "bits": []}

    for byte in range(13):
        vc = d[byte].value_counts()
        entropy = scipy.stats.entropy(vc, base=2)
        print('byte', byte, entropy)
        ent["bytes"].append(entropy)

    inp = prepare_input((sa, da, sp, dp, prot), bits=True)

    d = pd.DataFrame(inp)

    for bit in range(13*8):
        vc = d[bit].value_counts()
        entropy = scipy.stats.entropy(vc, base=2)
        print('bit', bit, entropy)
        ent["bits"].append(entropy)

    return ent

def main():
    app_args = parser().parse_args()

    entropy = calculate_entropy(app_args.directory)
    importances = {}

    entropy["bytes"] = [y for x in entropy["bytes"] for y in itertools.repeat(x, 8)]

    bit_start = 0
    bit_end = 104

    if importances:
        fig, ax = plt.subplots(ncols=3, width_ratios=[1, 2, 6])
        ax0, ax1, ax2 = ax
    else:
        fig, ax = plt.subplots(ncols=2, width_ratios=[1, 2])
        ax0, ax1 = ax
    fig.set_size_inches(10, (bit_end - bit_start) / 5)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0)

    if entropy:
        df = pd.DataFrame(entropy)
        normalized_df = df.copy()
        normalized_df["bytes"] = normalized_df["bytes"] / 8
        normalized_df = normalized_df ** 2

        ax0.pcolor(pd.DataFrame([0.0] * 104), vmin=0.0, vmax=2.0, cmap='Oranges')
        ax0.set_frame_on(False)

        # put the major ticks at the middle of each cell
        ax0.set_xticks(np.arange(1) + 0.5, ["Header field"], minor=False)
        ax0.set_yticks(np.arange(104) + 0.58, [f"{i:>5}" for i in range(104)], minor=False)

        # want a more natural, table-like display
        ax0.invert_yaxis()
        ax0.xaxis.tick_top()

        ax0.tick_params(tick1On=False)
        ax0.tick_params(tick2On=False)

        ax0.set_ylabel("Header feature bit number")

        ax0.hlines(y=0, xmin=0, xmax=1, colors='k', lw=2)
        ax0.hlines(y=32, xmin=0, xmax=1, colors='k', lw=2)
        ax0.hlines(y=64, xmin=0, xmax=1, colors='k', lw=2)
        ax0.hlines(y=80, xmin=0, xmax=1, colors='k', lw=2)
        ax0.hlines(y=96, xmin=0, xmax=1, colors='k', lw=2)
        ax0.hlines(y=104, xmin=0, xmax=1, colors='k', lw=2)

        ax0.text(0.5, 16, "Source IP address", rotation=90, va='center', ha='center')
        ax0.text(0.5, 48, "Destination IP address", rotation=90, va='center', ha='center')
        ax0.text(0.5, 72, "Source port", rotation=90, va='center', ha='center')
        ax0.text(0.5, 88, "Destination port", rotation=90, va='center', ha='center')
        ax0.text(0.5, 100, "Transport protocol", rotation=90, va='center', ha='center')

        ax0.set_ylim(bit_end, bit_start)

        ax1.pcolor(normalized_df.values, vmin=0.0, vmax=2.0, cmap='Oranges')
        ax1.set_frame_on(False)

        # put the major ticks at the middle of each cell
        ax1.set_xticks(np.arange(df.to_numpy().shape[1]) + 0.5, ["Octets\n[bits]", "Bit vector\n[bits]"], minor=False)
        ax1.set_yticks([], minor=False)

        # want a more natural, table-like display
        ax1.invert_yaxis()
        ax1.xaxis.tick_top()

        ax1.tick_params(tick1On=False)
        ax1.tick_params(tick2On=False)

        ax1.set_title("Entropy", fontsize='medium', pad=8.0)

        for i in range(df.to_numpy().shape[0]):
            for j in range(df.to_numpy().shape[1]):
                if bit_start <= i < bit_end:
                    val = df.to_numpy()[i, j]
                    ax1.text(j + 0.5, i + 0.58, f"{val:7.2f}", ha='center', va='center', color='w' if val > 100 else 'k')

        ax1.hlines(y=0, xmin=0, xmax=2, colors='k', lw=2)
        ax1.hlines(y=32, xmin=0, xmax=2, colors='k', lw=2)
        ax1.hlines(y=64, xmin=0, xmax=2, colors='k', lw=2)
        ax1.hlines(y=80, xmin=0, xmax=2, colors='k', lw=2)
        ax1.hlines(y=96, xmin=0, xmax=2, colors='k', lw=2)
        ax1.hlines(y=104, xmin=0, xmax=2, colors='k', lw=2)

        ax1.set_ylim(bit_end, bit_start)

    if importances:
        df = pd.DataFrame(importances)
        normalized_df = df

        ax2.pcolor(normalized_df.values, vmin=0.0, vmax=0.8, cmap='Blues')
        ax2.set_frame_on(False)

        # put the major ticks at the middle of each cell
        ax2.set_xticks(np.arange(df.to_numpy().shape[1]) + 0.5, df.columns, minor=False)
        ax2.set_yticks([], minor=False)

        # want a more natural, table-like display
        ax2.invert_yaxis()
        ax2.xaxis.tick_top()

        ax2.tick_params(tick1On=False)
        ax2.tick_params(tick2On=False)

        ax2.set_title("Feature importance", fontsize='medium', pad=8.0)

        for i in range(df.to_numpy().shape[0]):
            for j in range(df.to_numpy().shape[1]):
                if bit_start <= i < bit_end:
                    val = df.to_numpy()[i, j]
                    ax2.text(j + 0.5, i + 0.58, f"{val:7.2f}", ha='center', va='center', color='w' if val > 100 else 'k')

        ax2.hlines(y=0, xmin=0, xmax=6, colors='k', lw=2)
        ax2.hlines(y=32, xmin=0, xmax=6, colors='k', lw=2)
        ax2.hlines(y=64, xmin=0, xmax=6, colors='k', lw=2)
        ax2.hlines(y=80, xmin=0, xmax=6, colors='k', lw=2)
        ax2.hlines(y=96, xmin=0, xmax=6, colors='k', lw=2)
        ax2.hlines(y=104, xmin=0, xmax=6, colors='k', lw=2)

        ax2.set_ylim(bit_end, bit_start)

    plt.savefig('entropy.pdf', bbox_inches='tight', metadata=PDF_NONE)


if __name__ == '__main__':
    main()
