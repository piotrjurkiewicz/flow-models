#!/usr/bin/python3

import argparse

from .lib.io import io_parser, IN_FORMATS, OUT_FORMATS
from .lib.util import logmsg, prepare_file_list

def convert(in_files, out_file, in_format='nfcapd', out_format='csv_flow'):
    """
    Convert one flow format to another.

    :param list[os.PathLike] in_files: input files paths
    :param os.PathLike | _io.TextIOWrapper out_file: output file or directory path or stream
    :param in_format: format of input files
    :param out_format: format of output
    """

    reader, writer = IN_FORMATS[in_format], OUT_FORMATS[out_format]

    writer = writer(out_file)
    next(writer)

    written = 0

    for file in in_files:
        for flow in reader(file):
            writer.send(flow)
            written += 1
        logmsg(f'Finished: {file}. Written: {written}')

    writer.close()

    logmsg(f'Finished all files. Written: {written}')

def main():

    parser = argparse.ArgumentParser(description=__doc__, parents=[io_parser])
    app_args = parser.parse_args()

    if app_args.i == 'binary':
        input_files = app_args.files
    else:
        input_files = prepare_file_list(app_args.files)

    convert(input_files, app_args.O, app_args.i, app_args.o)


if __name__ == '__main__':
    main()
