#!/usr/bin/python3
"""
Sorts flow records according to specified fields (requires `numpy`).
"""

import argparse
import pathlib

import numpy as np

from .lib.io import load_array_np
from .lib.util import logmsg, measure_memory, prepare_file_list

def create_index(key_files, index_file):
    keys = []
    size = None

    logmsg('Create index: loading key files')

    if not key_files:
        logmsg('At least one key file must be specified when creating an index.')
        raise ValueError

    for path in key_files:
        name, dtype, mm = load_array_np(path)
        if size is None:
            size = mm.size
        else:
            assert mm.size == size
        keys.append(mm)
        del mm

    if size <= 2 ** 8:
        result_dtype = 'B'
    elif size <= 2 ** 16:
        result_dtype = 'H'
    elif size <= 2 ** 32:
        result_dtype = 'I'
    else:
        result_dtype = 'Q'

    index_array = np.memmap(f'{index_file}.{result_dtype}', dtype=result_dtype, mode='w+', shape=(size,))

    logmsg('Create index: sorting keys')

    result = np.lexsort(list(reversed(keys)))

    logmsg('Create index: sorting keys completed')

    del keys

    logmsg('Create index: copying index to file')

    index_array[:] = result[:]
    del result

    logmsg('Create index: flushing index file')

    index_array.flush()

    logmsg('Create index: finished')

    del index_array

def sort_array(input_file, output_dir, index_array):
    logmsg('Sort array: loading:', input_file)

    size = index_array.size

    input_file = pathlib.Path(input_file)
    output_dir = pathlib.Path(output_dir)

    if input_file.absolute().parent == output_dir.absolute():
        mode = 'r+'
    else:
        mode = 'r'

    name, dtype, in_mm = load_array_np(input_file, mode)
    assert in_mm.size == size

    if mode == 'r':
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / f'{name}.{dtype}'
        out_mm = np.memmap(out_file, dtype=dtype, mode='w+', shape=(size,))
        logmsg('Sort array: sorting:', input_file, 'to new file:', out_file)
    else:
        out_file = input_file
        out_mm = in_mm
        logmsg('Sort array: sorting in-place:', input_file)

    result = in_mm[index_array]

    logmsg('Sort array: sorting completed:', input_file)

    del in_mm
    del index_array

    logmsg('Sort array: copying result to output file')

    out_mm[:] = result
    del result

    logmsg('Sort array: flushing:', out_file)

    out_mm.flush()

    logmsg('Sort array: finished:', input_file)

    del out_mm

def parser():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    p.add_argument('-I', default='__index', help='index file suffix')
    p.add_argument('-k', nargs='*', help='ordered key array files')
    p.add_argument('-O', help='directory for output')
    p.add_argument('--measure-memory', action='store_true', help='collect and print memory statistics')
    p.add_argument('files', nargs='+', help='array files to sort')
    return p

def main():
    app_args = parser().parse_args()

    if not app_args.O:
        logmsg('Output directory not specified. To sort in-place specify file containing directory as the output directory.')
        raise ValueError

    with measure_memory(app_args.measure_memory):
        try:
            _, _, index = load_array_np(app_args.I)
            logmsg('Using existing index file...')
        except FileNotFoundError:
            logmsg('Index file not exists, will be created...')
            create_index(app_args.k, app_args.I)
            _, _, index = load_array_np(app_args.I)

        for f in prepare_file_list(app_args.files):
            sort_array(f, app_args.O, index)


if __name__ == '__main__':
    main()
