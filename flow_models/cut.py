#!/usr/bin/python3
"""Cuts binary flow records with dd."""

import array
import pathlib
import subprocess
import sys

from flow_models.lib.io import IOArgumentParser, prepare_file_list
from flow_models.lib.util import logmsg

EPILOG = \
"""
This tool can be used to cut flow records when both input and output are in
binary formats. It is more efficient to use cut than the convert tool in such case,
because flow records are not being serialized during the operation.

This tool uses the dd standard Unix command to cut binary flow record files in an
efficient way.

A single binary file can be given as an input. In such case, a single output file
should be specified with the parameter -O as an output.

In the case when multiple input files are specified, or the specified input is a
directory, the output directory should be given with the parameter -O. Cut files
will be created in that directory.

When no output is specified, the standard output (-) is being used. This is equivalent
to reading the files sequentially with dd.

This tool does not perform flow records filtering, the --filter-expr parameter should
not be specified.

Cutting of flow records can be done with skip_in, count_in parameters. They specify how
many flow records should be skipped (skip_in) and then read (count_in) from input.

Example: (skips the first 100 records and writes the next 1000)

    flow_models.cut -i binary -o binary -O sample --skip-in 100 --count-in 1000 sorted
"""

def cut(in_files, output, in_format='binary', out_format='binary', skip_in=0, count_in=None, skip_out=0, count_out=None, filter_expr=None):
    """
    Cut binary flow records with dd.

    Parameters
    ----------
    in_files : list[pathlib.Path]
        input files paths
    output : os.PathLike | io.TextIOWrapper
        output file or directory path or stream
    in_format : str, default 'binary'
        input format
    out_format : str, default 'binary'
        output format
    skip_in : int, default 0
        number of flows to skip at the beginning of input
    count_in : int, default None, meaning all flows
        number of flows to read from input
    skip_out : int, default 0
        number of flows to skip after filtering
    count_out : int, default None, meaning all flows
        number of flows to output after filtering
    filter_expr : CodeType, optional
        not supported
    """

    if in_format != 'binary' or out_format != 'binary':
        raise ValueError("Both input and output formats must be binary")
    if filter_expr is not None:
        raise NotImplementedError("Filter expressions are not supported")

    in_files = prepare_file_list(in_files)

    if output != sys.stdout:
        output = pathlib.Path(output)
        if output.is_dir() or len(in_files) > 1:
            output.mkdir(parents=True, exist_ok=True)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)

    for in_file in in_files:
        try:
            size = array.array(in_file.suffix[1:]).itemsize
            if skip_in > 0:
                skip = skip_in * size
            else:
                skip = 0
            if count_in is not None:
                if count_in > 0:
                    count = count_in * size
                else:
                    count = 0
            else:
                count = in_file.stat().st_size - skip
            if skip_out > 0:
                count = count - skip_out * size
                skip += skip_out * size
            if count_out is not None:
                if count_out > 0:
                    count = min(count, count_out * size)
                else:
                    count = 0
            count = max(count, 0)
        except TypeError:
            logmsg('Cut array:', in_file, 'Wrong type')
            return

        logmsg('Cut array:', in_file, size)
        subprocess.run(['dd', f'if={in_file}', f'of={output / in_file.name}' if output != sys.stdout else 'status=none', f'count={count}', f'skip={skip}', 'iflag=count_bytes,skip_bytes'], check=True)


def parser():
    p = IOArgumentParser(description=__doc__, epilog=EPILOG)
    p._option_string_actions['-i'].choices = ['binary']
    p._option_string_actions['-i'].default = 'binary'
    p._option_string_actions['-o'].choices = ['binary']
    p._option_string_actions['-o'].default = 'binary'
    p._optionals._remove_action(p._option_string_actions['--skip-out'])
    p._optionals._remove_action(p._option_string_actions['--count-out'])
    p._optionals._remove_action(p._option_string_actions['--filter-expr'])
    return p

def main():
    app_args = parser().parse_args()
    cut(**vars(app_args))


if __name__ == '__main__':
    main()
