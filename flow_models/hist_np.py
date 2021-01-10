#!/usr/bin/python3
"""
Calculates histograms using multiple threads (requires `numpy`, much faster, but uses more memory).
"""

import argparse
import concurrent.futures
import functools
import os
import pathlib
import sys

import numpy as np

from .lib.io import load_array_np, write_line, write_none
from .lib.util import logmsg, bin_calc_log, measure_memory

# MAX_MEM = 64 * (1024 ** 3)
MAX_MEM = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') // 4
N_JOBS = 4

X_VALUES = ['length', 'size']
OUT_FORMATS = {'csv_hist': write_line, 'none': write_none}

def get_column_array(mm, column, start, stop):
    if column in mm:
        return mm[column][start:stop]
    elif column in ['duration', 'rate']:
        if 'duration' in mm:
            dur = mm['duration'][start:stop]
        else:
            dur = (mm['last'][start:stop] - mm['first'][start:stop]) * 1000
            dur += mm['last_ms'][start:stop]
            dur -= mm['first_ms'][start:stop]
            dur[mm['packets'][start:stop] == 1] = 0
            if column == 'duration':
                return dur
        if column == 'rate':
            rate = np.zeros(stop - start)
            np.divide(8000 * mm['octets'][start:stop], dur, out=rate, where=dur != 0)
            return rate
    else:
        raise ValueError(f'Cannot compute column: {column}')

def calc_chunk(memory_maps, key_column, start, stop, columns, algorithm='bincount'):
    sums = {}
    key_array = memory_maps[key_column][start:stop]
    max_key = key_array.max()

    if max_key * 8 < MAX_MEM:
        hist = np.bincount(key_array.view('q'))
        bins = np.nonzero(hist)[0]
        sums['flows'] = hist[bins]
    else:
        logmsg('Maximum value of key column too big, cannot use bincount, will be slow...')
        bins = np.unique(key_array)
        algorithm = 'addat'

    max_bin = bins.max()
    assert max_bin == max_key
    bins = bins.astype(np.min_scalar_type(max_bin))

    if algorithm == 'bincount':
        for column in columns:
            hist = np.bincount(key_array.view('q'), get_column_array(memory_maps, column, start, stop))[bins]
            sums[column] = hist
    elif algorithm == 'addat':
        assert len(key_array) < 2_147_483_648
        indices = np.searchsorted(bins, key_array)
        indices = indices.astype(np.min_scalar_type(len(bins)))
        for column in columns:
            hist = np.zeros(len(bins))
            np.add.at(hist, indices, get_column_array(memory_maps, column, start, stop))
            sums[column] = hist
        if 'flows' not in sums:
            hist = np.zeros(len(bins))
            np.add.at(hist, indices, 1)
            sums['flows'] = hist
    else:
        raise ValueError('Unknown algorithm', algorithm)

    return bins, sums

def calc_dir(path,  x_value, columns):

    logmsg(f'Processing directory: {path}')
    assert path.is_dir()

    key_columns = {'length': 'packets',
                   'size': 'octets'}
    key_column = key_columns[x_value]
    size = None
    memory_maps = {}

    for name in columns:

        try:
            name, dtype, in_mm = load_array_np(path / name, 'r')
            assert name not in memory_maps
            if size is None:
                size = in_mm.size
            else:
                assert in_mm.size == size
            memory_maps[name] = in_mm
            logmsg(f'Loaded array: {path / name}')
        except FileNotFoundError:
            logmsg(f'Not found array: {path / name}, will try to compute this column')

    if ('duration' in columns or 'rate' in columns) and 'duration' not in memory_maps:
        for name in ['first', 'first_ms', 'last', 'last_ms']:
            logmsg(f'Loading array: {path / name}')
            memory_maps[name] = load_array_np(path / name, 'r')[2]

    i = 0
    while True:
        i += 1
        chunk_size = size // i + 1
        if chunk_size * 8 < MAX_MEM / 2.2 and chunk_size // N_JOBS + 1 < 2_147_483_648:
            break

    starts = list(range(0, size, size // (i * N_JOBS) + 1))
    stops = starts[1:] + [size]

    logmsg(f'Calculating directory: {path} using {N_JOBS} workers in {i * N_JOBS} chunks')

    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=N_JOBS) as executor:
        futures = {}
        for start, stop in zip(starts, stops):
            future = executor.submit(calc_chunk, memory_maps, key_column, start, stop, columns)
            futures[future] = start, stop
        for future in concurrent.futures.as_completed(futures):
            start, stop = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                logmsg(f'Exception in directory: {path} chunk: [{start}:{stop}] {exc}')
                # raise
            else:
                logmsg(f'Done directory: {path} chunk: [{start}:{stop}]')

    logmsg(f'Done directory: {path} chunk: all')

    return results

def histogram(in_files, out_file, in_format='binary', out_format='csv_hist', bin_exp=0, x_value='length', additional_columns=()):

    assert in_format == 'binary'

    if bin_exp == 0:
        bin_calc_fn = None
    else:
        bin_calc_fn = bin_calc_log

    writer = OUT_FORMATS[out_format]
    columns = ['packets', 'octets'] + additional_columns

    if isinstance(in_files, list):
        dirs = [pathlib.Path(f) for f in in_files]
    else:
        dirs = [pathlib.Path(in_files)]

    results = []

    for d in dirs:
        results += calc_dir(d, x_value, columns)

    bins = functools.reduce(np.union1d, (bins for bins, _ in results))
    sums = {c: np.zeros(len(bins)) for c in ['flows'] + columns}

    for bn, data in results:
        indices = np.searchsorted(bins, bn)
        for c in sums:
            sums[c][indices] += data[c]

    header_line = 'bin_lo,bin_hi,'
    header_line += ','.join(c + '_sum' for c in sums)

    written = 0

    writer = writer(out_file, header_line)
    next(writer)

    if bin_calc_fn is None:
        for n, bin_lo in enumerate(bins):
            bin_lo = int(bin_lo)
            writer.send(f'{bin_lo},{bin_lo + 1},' + ','.join(str(int(float(s[n]))) for s in sums.values()))
            written += 1
    else:
        bin_lo, bin_hi = bin_calc_fn(bins[0], bin_exp)
        bin_sums = [0.0] * len(sums)
        for n, x in enumerate(bins):
            x = int(x)
            if x >= bin_hi:
                writer.send(f'{bin_lo},{bin_hi},' + ','.join(str(int(float(s))) for s in bin_sums))
                written += 1
                bin_lo, bin_hi = bin_calc_fn(x, bin_exp)
                bin_sums = [0.0] * len(sums)
            for i, s in enumerate(sums.values()):
                bin_sums[i] += s[n]
        writer.send(f'{bin_lo},{bin_hi},' + ','.join(str(int(float(s))) for s in bin_sums))
        written += 1

    writer.close()

    logmsg(f'Finished all directories. Flows: NA Written: {written}')

def parser():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    p.add_argument('files', nargs='+', help='input dirs')
    p.add_argument('-i', default='binary', choices=['binary'], help='format of input files')
    p.add_argument('-o', default='csv_hist', choices=['csv_hist'], help='format of output')
    p.add_argument('-O', default=sys.stdout, help='file for output')
    p.add_argument('-x', default='length', choices=X_VALUES, help='x axis value')
    p.add_argument('-b', default=0, type=int, help='bin width exponent of 2')
    p.add_argument('-c', action='append', default=[], help='additional column to sum')
    p.add_argument('--measure-memory', action='store_true', help='collect and print memory statistics')
    return p

def main():
    app_args = parser().parse_args()

    with measure_memory(app_args.measure_memory):
        histogram(app_args.files, app_args.O, app_args.i, app_args.o, app_args.b, app_args.x, app_args.c)


if __name__ == '__main__':
    main()
