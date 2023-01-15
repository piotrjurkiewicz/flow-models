#!/usr/bin/python3
"""
Converts flow records between supported formats.
"""

from .lib.io import IOArgumentParser, IN_FORMATS, OUT_FORMATS
from .lib.util import logmsg

def convert(in_files, out_file, in_format='nfcapd', out_format='csv_flow', count=None, skip_input=0, skip_output=0, filter_expr=None):
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
        number of flows to output (Default is 0 (all flows))
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

    counters = {'count': count, 'skip_input': skip_input, 'skip_output': skip_output}
    written = 0

    for file in in_files:
        for flow in reader(file, counters=counters, filter_expr=filter_expr):
            writer.send(flow)
            written += 1
        logmsg(f'Finished: {file}. Written: {written}')

    writer.close()

    logmsg(f'Finished all files. Written: {written}')

def parser():
    p = IOArgumentParser(description=__doc__)
    return p

def main():
    app_args = parser().parse_args()
    convert(**vars(app_args))


if __name__ == '__main__':
    main()
