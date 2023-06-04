#!/usr/bin/python3
"""
Sorts flow records according to specified fields (requires `numpy`).
"""

import pathlib

import numpy as np

from .lib.io import load_array_np, prepare_file_list, load_arrays, IOArgumentParser
from .lib.util import logmsg, measure_memory

def create_index(path, key_fields, index_file, counters=None, reverse=False):

    if counters is None:
        counters = {'skip_in': 0, 'count_in': None, 'skip_out': 0, 'count_out': None}

    keys = []

    logmsg('Create index: loading key files')

    if not key_fields:
        logmsg('At least one key field must be specified when creating an index.')
        raise ValueError

    arrays, filtered, size = load_arrays(path, key_fields, counters, None)

    for name in key_fields:
        keys.append(arrays[name])
        del arrays[name]
    del arrays

    logmsg('Create index: sorting keys')

    result = np.lexsort(list(reversed(keys)))

    logmsg('Create index: sorting keys completed')

    del keys

    if reverse:
        logmsg('Create index: reversing order')
        result = np.flip(result)
        logmsg('Create index: reversing order completed')

    if index_file:

        if size <= 2 ** 8:
            result_dtype = 'B'
        elif size <= 2 ** 16:
            result_dtype = 'H'
        elif size <= 2 ** 32:
            result_dtype = 'I'
        else:
            result_dtype = 'Q'

        index_array = np.memmap(f'{index_file}.{result_dtype}', dtype=result_dtype, mode='w+', shape=(size,))

        logmsg('Create index: copying index to file')

        index_array[:] = result[:]
        del result

        logmsg('Create index: flushing index file')

        index_array.flush()

        logmsg('Create index: finished')

        del index_array

        return None

    else:
        return result

def sort_array(input_file, output_dir, index_array, counters=None):

    if counters is None:
        counters = {'skip_in': 0, 'count_in': None, 'skip_out': 0, 'count_out': None}

    logmsg('Sort array: loading:', input_file)

    size = index_array.size

    input_file = pathlib.Path(input_file)
    output_dir = pathlib.Path(output_dir)

    if input_file.absolute().parent == output_dir.absolute():
        mode = 'r+'
    else:
        mode = 'r'

    name, dtype, in_mm = load_array_np(input_file, mode)
    if counters['skip_in'] > 0:
        in_mm = in_mm[counters['skip_in']:]
    if counters['count_in'] is not None and counters['count_in'] > 0:
        in_mm = in_mm[:counters['count_in']]
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

def sort(in_files, out_dir, in_format='binary', out_format='binary', key_fields=(), index_file=None, skip_in=0, count_in=None, skip_out=0, count_out=None, filter_expr=None, no_index=False, reverse=False):

    assert in_format == 'binary'
    assert out_format == 'binary'
    if count_out is not None:
        raise NotImplementedError
    if skip_out != 0:
        raise NotImplementedError
    if filter_expr is not None:
        raise NotImplementedError

    counters = {'skip_in': skip_in, 'count_in': count_in, 'skip_out': skip_out, 'count_out': count_out}

    if no_index:
        index = create_index(in_files, key_fields, None, counters, reverse)
    else:
        try:
            _, _, index = load_array_np(index_file)
            logmsg('Using existing index file...')
        except FileNotFoundError:
            logmsg('Index file not exists, will be created...')
            create_index(in_files, key_fields, index_file, counters, reverse)
            _, _, index = load_array_np(index_file)

    for f in prepare_file_list(in_files):
        sort_array(f, out_dir, index, counters)

def parser():
    p = IOArgumentParser(description=__doc__)
    p._option_string_actions['-i'].choices = ['binary']
    p._option_string_actions['-i'].default = 'binary'
    p._option_string_actions['-o'].choices = ['binary']
    p._option_string_actions['-o'].default = 'binary'
    p.add_argument('-k', '--key-files', nargs='*', help='ordered key fields names')
    p.add_argument('-I', '--index-file', default='_index', help='index file')
    p.add_argument('--no-index', action='store_true', help='do not save index into file')
    p.add_argument('--reverse', action='store_true', help='reverse order')
    p.add_argument('--measure-memory', action='store_true', help='collect and print memory statistics')
    return p

def main():
    app_args = parser().parse_args()

    if not app_args.O:
        logmsg('Output directory not specified. To sort in-place specify file containing directory as the output directory.')
        raise ValueError

    with measure_memory(app_args.measure_memory):
        delattr(app_args, 'measure_memory')
        sort(**vars(app_args))


if __name__ == '__main__':
    main()
