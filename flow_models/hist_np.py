#!/usr/bin/python3
"""Calculates histograms using multiple threads (requires `numpy`, much faster, but uses more memory)."""

import concurrent.futures
import functools
import os
import pathlib

import numpy as np

from flow_models.lib.io import FILTER_HELP, IOArgumentParser, load_arrays, write_append, write_line, write_none
from flow_models.lib.util import bin_calc_log, logmsg, measure_memory

EPILOG = \
f"""
Use this tool to calculate histogram of flow features.

The output is a histogram of a selected feature in csv_hist format.

Feature selection is being done with -x parameter. Additionally -b parameter can be
specified, which will make histogram logarithmically binned to help reduce its size.

{FILTER_HELP}

Skipping of flow records can be done with skip_in and count_in parameters.
They specify how many flow records should be skipped (skip_in) and then read (count_in)
from input.

Example: (calculates logarithmically binned histogram of flow length from the sorted directory)

    flow_models.hist -i binary -x length -b 12 sorted
"""

# MAX_MEM = 64 * (1024 ** 3)
MAX_MEM = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') // 4
N_JOBS = 4

X_VALUES = ['length', 'size']
OUT_FORMATS = {'csv_hist': write_line, 'append': write_append, 'none': write_none}

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
        else:  # column == 'rate'
            rate = np.zeros_like(dur, dtype=np.float64)
            np.divide(8000 * mm['octets'][start:stop], dur, out=rate, where=dur != 0)
            return rate
    else:
        raise ValueError(f'Cannot compute column: {column}')

def calc_chunk(memory_maps, key_column, start, stop, columns, filtered=..., algorithm='bincount'):
    sums = {}
    key_array = memory_maps[key_column][start:stop][filtered]
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
            hist = np.bincount(key_array.view('q'), get_column_array(memory_maps, column, start, stop)[filtered])[bins]
            sums[column] = hist
    elif algorithm == 'addat':
        assert len(key_array) < 2_147_483_648
        indices = np.searchsorted(bins, key_array)
        indices = indices.astype(np.min_scalar_type(len(bins)))
        for column in columns:
            hist = np.zeros(len(bins))
            np.add.at(hist, indices, get_column_array(memory_maps, column, start, stop)[filtered])
            sums[column] = hist
        if 'flows' not in sums:
            hist = np.zeros(len(bins))
            np.add.at(hist, indices, 1)
            sums['flows'] = hist
    else:
        raise ValueError('Unknown algorithm', algorithm)

    return bins, sums

def calc_dir(path, x_value, columns, counters=None, filter_expr=None):

    if counters is None:
        counters = {'skip_in': 0, 'count_in': None, 'skip_out': 0, 'count_out': None}

    logmsg(f'Processing directory: {path}')
    assert path.is_dir()

    key_columns = {'length': 'packets',
                   'size': 'octets'}
    key_column = key_columns[x_value]

    if 'duration' in columns or 'rate' in columns:
        fields_to_load = [*columns, 'first', 'first_ms', 'last', 'last_ms']
    else:
        fields_to_load = columns

    arrays, filtered, size = load_arrays(path, fields_to_load, counters, filter_expr, require_numpy=True)

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
            future = executor.submit(calc_chunk, arrays, key_column, start, stop, columns, filtered)
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

def hist(in_files, output, in_format='binary', out_format='csv_hist', skip_in=0, count_in=None, skip_out=0, count_out=None, filter_expr=None, bin_exp=0, x_value='length', additional_columns=None):
    """
    Calculate histograms of flows length, size, duration or rate.

    Parameters
    ----------
    in_files : list[os.PathLike]
        input files paths
    output : os.PathLike | io.TextIOWrapper
        output file or directory path or stream
    in_format : str, default 'binary'
        input format
    out_format : str, default 'csv_hist'
        output format
    skip_in : int, default 0
        number of flows to skip at the beginning of input
    count_in : int, default None, meaning all flows
        number of flows to read from input
    skip_out : int, default 0
        not supported
    count_out : int, default None, meaning all flows
        not supported
    filter_expr : CodeType, optional
        filter expression
    bin_exp: int, default 0
        bin width exponent of 2
    x_value : str, default 'length'
        x axis value
    additional_columns : list[str], optional
        additional column to sum
    """

    assert in_format == 'binary'
    if count_out is not None:
        raise NotImplementedError
    if skip_out != 0:
        raise NotImplementedError

    if bin_exp == 0:
        bin_calc_fn = None
    else:
        bin_calc_fn = bin_calc_log

    writer = OUT_FORMATS[out_format]
    if additional_columns is None:
        additional_columns = []
    columns = ['packets', 'octets', *additional_columns]

    if isinstance(in_files, list):
        dirs = [pathlib.Path(f) for f in in_files]
    else:
        raise ValueError

    counters = {'skip_in': skip_in, 'count_in': count_in, 'skip_out': skip_out, 'count_out': count_out}
    results = []

    for d in dirs:
        results += calc_dir(d, x_value, columns, counters, filter_expr)

    bins = functools.reduce(np.union1d, (bins for bins, _ in results))
    sums = {c: np.zeros(len(bins)) for c in ['flows', *columns]}

    for bn, data in results:
        indices = np.searchsorted(bins, bn)
        for c in sums:
            sums[c][indices] += data[c]

    header_line = 'bin_lo,bin_hi,'
    header_line += ','.join(c + '_sum' for c in sums)

    written = 0

    writer = writer(output, header_line)
    next(writer)

    if bin_calc_fn is None:
        for n, bin_lo in enumerate(bins):
            bin_lo = int(bin_lo)
            writer.send(f'{bin_lo},{bin_lo + 1},' + ','.join(str(int(float(s[n]))) for s in sums.values()))
            written += 1
    else:
        bin_lo, bin_hi = bin_calc_fn(bins.item(0), bin_exp)
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
    p = IOArgumentParser(description=__doc__, epilog=EPILOG)
    p._option_string_actions['-i'].choices = ['binary']
    p._option_string_actions['-i'].default = 'binary'
    p._option_string_actions['-o'].choices = OUT_FORMATS
    p._option_string_actions['-o'].default = 'csv_hist'
    p.add_argument('-b', '--bin-exp', default=0, type=int, help='bin width exponent of 2')
    p.add_argument('-x', '--x-value', default='length', choices=X_VALUES, help='x axis value')
    p.add_argument('-c', '--additional-columns', action='append', default=[], help='additional column to sum')
    p.add_argument('--measure-memory', action='store_true', help='collect and print memory statistics')
    return p

def main():
    app_args = parser().parse_args()

    with measure_memory(app_args.measure_memory):
        delattr(app_args, 'measure_memory')
        hist(**vars(app_args))


if __name__ == '__main__':
    main()
