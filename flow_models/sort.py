#!/usr/bin/python3
"""Sorts flow records according to specified key fields (requires `numpy`)."""

import pathlib

import numpy as np

from flow_models.lib.io import FILTER_HELP, IOArgumentParser, load_array_np, load_arrays, prepare_file_list
from flow_models.lib.util import logmsg, measure_memory

EPILOG = \
f"""
This tool can be used to sort flow records in binary format.

Sorting in being done according to specified key fields. Key fields should be
specified in an order, for example '-k first first_ms' means that records
are sorted according to the first second value, and next records with the same
second value are sorted according to the millisecond.

By default records are sorted in an ascending order. To get the descending order,
use --reverse parameter.

User can specify directory for output with the -O parameter. When the output
directory is the same as the input directory, sorting will be done in-place and
overwrite input files.

Sorting of flow records can be done with skip_in and count_in parameters.
They specify how many flow records should be skipped (skip_in) and then read (count_in)
from input.

{FILTER_HELP}

Example: (sorts flows in the merged directory and saves the output to the sorted directory)

    flow_models.sort -i -k first first_ms -O sorted merged
"""

def create_index(path, key_fields, index_file, counters=None, reverse=False):
    """
    Create index array, optionally saving it to file.

    Parameters
    ----------
    path: os.PathLike
        path of a directory with key files
    key_fields: list[str]
        ordered list of key fields
    index_file : str, optional
        index file path
    counters: dict[str, int], default
        skip_in : int, default 0
            number of flows to skip at the beginning of input
        count_in : int, default None, meaning all flows
            number of flows to read from input
        skip_out : int, default 0
            not supported
        count_out : int, default None, meaning all flows
            not supported
    reverse : bool, default False
        reverse order

    Returns
    -------
    numpy.array
        index array
    """

    if counters is None:
        counters = {'skip_in': 0, 'count_in': None, 'skip_out': 0, 'count_out': None}

    keys = []
    logmsg('Create index: loading key files')
    if not key_fields:
        logmsg('At least one key field must be specified when creating an index.')
        raise ValueError

    arrays, filtered, size = load_arrays(pathlib.Path(path), key_fields, counters, None, require_numpy=True)

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
        return index_array

    else:
        return result

def sort_array(input_file, output_dir, index_array, counters=None):
    """
    Sorts flow records according to an index array.

    Parameters
    ----------
    input_file : os.PathLike
        input files path
    output_dir: os.PathLike
        output directory path
    index_array: object
        index array
    counters: dict[str, int], default
        skip_in : int, default 0
            number of flows to skip at the beginning of input
        count_in : int, default None, meaning all flows
            number of flows to read from input
        skip_out : int, default 0
            not supported
        count_out : int, default None, meaning all flows
            not supported
    """

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
        output_file = output_dir / f'{name}.{dtype}'
        out_mm = np.memmap(output_file, dtype=dtype, mode='w+', shape=(size,))
        logmsg('Sort array: sorting:', input_file, 'to new file:', output_file)
    else:
        output_file = input_file
        out_mm = in_mm
        logmsg('Sort array: sorting in-place:', input_file)

    result = in_mm[index_array]
    logmsg('Sort array: sorting completed:', input_file)

    del in_mm
    del index_array

    logmsg('Sort array: copying result to output file')
    out_mm[:] = result
    del result

    logmsg('Sort array: flushing:', output_file)
    out_mm.flush()
    logmsg('Sort array: finished:', input_file)

    del out_mm

def sort(in_files, output, key_fields, in_format='binary', out_format='binary', index_file=None, skip_in=0, count_in=None, skip_out=0, count_out=None, filter_expr=None, reverse=False):
    """
    Sorts flow records according to specified key fields.

    Parameters
    ----------
    in_files : list[str]
        input files paths
    output : os.PathLike | None
        output directory path
    key_fields : list[str]
        ordered list of key fields
    in_format : str, default 'binary'
        input format
    out_format : str, default 'binary'
        output format
    index_file : str, optional
        index file path
    skip_in : int, default 0
        number of flows to skip at the beginning of input
    count_in : int, default None, meaning all flows
        number of flows to read from input
    skip_out : int, default 0
        not supported
    count_out : int, default None, meaning all flows
        not supported
    filter_expr : CodeType, optional
        not supported
    reverse : bool, default False
        reverse order
    """

    if in_format != 'binary' or out_format != 'binary':
        raise ValueError("Both input and output formats must be binary")
    if count_out is not None:
        raise NotImplementedError
    if skip_out != 0:
        raise NotImplementedError
    if filter_expr is not None:
        raise NotImplementedError("Filter expressions are not supported")

    counters = {'skip_in': skip_in, 'count_in': count_in, 'skip_out': skip_out, 'count_out': count_out}

    if index_file is None:
        index = create_index(in_files[0], key_fields, None, counters, reverse)
    else:
        try:
            _, _, index = load_array_np(index_file)
            logmsg('Using existing index file...')
        except FileNotFoundError:
            logmsg('Index file not exists, will be created...')
            index = create_index(in_files[0], key_fields, index_file, counters, reverse)

    if output:
        for f in prepare_file_list(in_files):
            sort_array(f, output, index, counters)

def parser():
    p = IOArgumentParser(description=__doc__, epilog=EPILOG)
    p._option_string_actions['-i'].choices = ['binary']
    p._option_string_actions['-i'].default = 'binary'
    p._option_string_actions['-o'].choices = ['binary']
    p._option_string_actions['-o'].default = 'binary'
    p._option_string_actions['-O'].help = 'directory for output'
    p._option_string_actions['-O'].default = '.'
    p._optionals._remove_action(p._option_string_actions['--skip-out'])
    p._optionals._remove_action(p._option_string_actions['--count-out'])
    p._optionals._remove_action(p._option_string_actions['--filter-expr'])
    p.add_argument('-k', '--key-fields', nargs='*', help='ordered key fields names')
    p.add_argument('-I', '--index-file', default=None, help='index file')
    p.add_argument('--reverse', action='store_true', help='reverse order')
    p.add_argument('--measure-memory', action='store_true', help='collect and print memory statistics')
    return p

def main():
    app_args = parser().parse_args()

    if not app_args.output:
        logmsg('Output directory not specified. To sort in-place specify file containing directory as the output directory.')
        raise ValueError

    with measure_memory(app_args.measure_memory):
        delattr(app_args, 'measure_memory')
        sort(**vars(app_args))


if __name__ == '__main__':
    main()
