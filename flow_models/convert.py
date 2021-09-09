#!/usr/bin/python3
"""
Converts flow records between supported formats.
"""

import argparse

from .lib.io import io_parser, IN_FORMATS, OUT_FORMATS
from .lib.util import logmsg, prepare_file_list

def convert(in_files, out_file, in_format='nfcapd', out_format='csv_flow', count=0, skip_input=0, skip_output=0, filter_expr=None):
    """
    Convert one flow format to another.

    Parameters
    ----------
    in_files : list[os.PathLike]
        input files paths
    out_file : Union[os.PathLike, io.TextIOWrapper]
        output file or directory path or stream
    in_format : str, optional
        input format (Default is 'nfcapd')
    out_format : str, optional
        output format (Default is 'csv_flow')
    count : int, optional
        number of flows to write (Default is 0 (all flows))
    skip_input : int, optional
        number of flows to skip at the beginning of input (Default is 0)
    skip_output : int, optional
        number of flows to skip after filtering (Default is 0)
    filter_expr : str, optional
        filter expression (Default is None)
    """

    reader, writer = IN_FORMATS[in_format], OUT_FORMATS[out_format]

    writer = writer(out_file)
    next(writer)

    written = 0

    for file in in_files:
        for flow in reader(file):
            if skip_input > 0:
                skip_input -= 1
            else:
                if filter_expr is None or eval(filter_expr):
                    if skip_output > 0:
                        skip_output -= 1
                    else:
                        writer.send(flow)
                        written += 1
                        if written == count:
                            break
        logmsg(f'Finished: {file}. Written: {written}')

    writer.close()

    logmsg(f'Finished all files. Written: {written}')

def parser():
    p = argparse.ArgumentParser(description=__doc__, parents=[io_parser])
    p.add_argument('--count', type=int, default=0, help='number of flows to copy')
    p.add_argument('--skip-input', type=int, default=0, help='number of flows to skip at the beginning of input')
    p.add_argument('--skip-output', type=int, default=0, help='number of flows to skip after filtering')
    p.add_argument('--filter', default=None, help='expression of filter')
    return p

def main():
    app_args = parser().parse_args()

    if app_args.i == 'binary':
        input_files = app_args.files
    else:
        input_files = prepare_file_list(app_args.files)

    if app_args.filter:
        filter_expr = compile(app_args.filter, '<filter>', 'eval')
    else:
        filter_expr = None

    convert(input_files, app_args.O, app_args.i, app_args.o, app_args.count, app_args.skip_input, app_args.skip_output, filter_expr)


if __name__ == '__main__':
    main()
