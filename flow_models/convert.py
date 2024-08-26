#!/usr/bin/python3
"""Converts flow records between supported formats. Can be used for filtering and cutting flow record files."""

from flow_models.lib.io import FILTER_HELP, IN_FORMATS, IOArgumentParser, OUT_FORMATS
from flow_models.lib.util import logmsg

EPILOG = \
f"""
To convert flow records between different formats, the input and output formats should
be specified in command line. Moreover, the output file/directory should be given with -O
parameter. When not, the standard output (-) is being used. Input files or directories
should be specified as the positional argument.

Example: (reads flow records in binary format and outputs as csv lines to standard output)

    flow_models.convert -i binary -o csv_flow -O - sorted

{FILTER_HELP}

Example: (selects HTTPS protocol flows and writes them in binary format)

    flow_models.convert -i binary -o binary -O https_only --filter-expr "(prot==6) & ((sp==443) | (dp==443))" sorted

Cutting of flow records can be done with skip_in, count_in, skip_out, count_out parameters.
They specify how many flow records should be skipped (skip_in) and then read (count_in)
from input and to be skipped (skip_out) and written (count_out) after filtering.

Example: (skips the first 100 records and writes the next 1000)

    flow_models.convert -i binary -o binary -O sample --skip-in 100 --count-in 1000 sorted

When no filter is being used, usage of skip_in will give the same results as skip_out.
The same applies for count_in and count_out respectively. However, depending on input
format, usage of skip_in and count_in may result in a better performance than skip_out
and count_out.

When both input and output formats are binary and no filter expression is being used,
it will be more efficient to use the cut tool, which uses dd to cut binary record files.

Converting, filtering and cutting can be done simultaneously in a single command call.
"""

def convert(in_files, output, in_format='nfcapd', out_format='csv_flow', skip_in=0, count_in=None, skip_out=0, count_out=None, filter_expr=None):
    """
    Convert flow records between supported formats. Can also be used for filtering and cutting flow record files.

    Parameters
    ----------
    in_files : list[pathlib.Path]
        input files paths
    output : os.PathLike | io.TextIOWrapper
        output file or directory path or stream
    in_format : str, default 'nfcapd'
        input format
    out_format : str, default 'csv_flow'
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
        filter expression
    """

    reader, writer = IN_FORMATS[in_format], OUT_FORMATS[out_format]

    writer = writer(output)
    next(writer)

    counters = {'skip_in': skip_in, 'count_in': count_in, 'skip_out': skip_out, 'count_out': count_out}
    written = 0

    for file in in_files:
        for flow in reader(file, counters=counters, filter_expr=filter_expr):
            writer.send(flow)
            written += 1
        logmsg(f'Finished: {file}. Written: {written}')

    writer.close()

    logmsg(f'Finished all files. Written: {written}')

def parser():
    p = IOArgumentParser(description=__doc__, epilog=EPILOG)
    return p

def main():
    app_args = parser().parse_args()
    convert(**vars(app_args))


if __name__ == '__main__':
    main()
