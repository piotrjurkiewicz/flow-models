#!/usr/bin/python3
"""Calculates histograms of flows length, size, duration or rate."""

from flow_models.lib.io import FILTER_HELP, IN_FORMATS, IOArgumentParser, write_append, write_line, write_none
from flow_models.lib.util import bin_calc_log, bin_calc_one, logmsg

EPILOG = \
f"""
Use this tool to calculate histogram of flow features.

The output is a histogram of a selected feature in csv_hist format.

Feature selection is being done with -x parameter. Additionally -b parameter can be
specified, which will make histogram logarithmically binned to help reduce its size.

{FILTER_HELP}

Skipping of flow records can be done with skip_in, count_in, skip_out, count_out parameters.
They specify how many flow records should be skipped (skip_in) and then read (count_in)
from input and to be skipped (skip_out) and written (count_out) after filtering.

Example: (calculates logarithmically binned histogram of flow length from the sorted directory)

    flow_models.hist -i binary -x length -b 12 sorted
"""

X_VALUES = ['length', 'size', 'duration', 'rate']
OUT_FORMATS = {'csv_hist': write_line, 'append': write_append, 'none': write_none}

class FlowBin:

    __slots__ = 'bin_lo', 'bin_hi', 'flows_sum', 'packets_sum', 'octets_sum', 'duration_sum', 'rate_sum', 'aggs_sum'

    def __init__(self, bin_lo, bin_hi):
        self.bin_lo = bin_lo
        self.bin_hi = bin_hi
        self.flows_sum = 0
        self.packets_sum = 0
        self.octets_sum = 0
        self.duration_sum = 0
        self.rate_sum = 0
        self.aggs_sum = 0

    def to_line(self, fields):
        return ','.join(str(int(getattr(self, c))) for c in fields)

def hist(in_files, output, in_format='nfcapd', out_format='csv_hist', skip_in=0, count_in=None, skip_out=0, count_out=None, filter_expr=None, bin_exp=0, x_value='length', additional_columns=None):
    """
    Calculate histograms of flows length, size, duration or rate.

    Parameters
    ----------
    in_files : list[os.PathLike]
        input files paths
    output : os.PathLike | io.TextIOWrapper
        output file or directory path or stream
    in_format : str, default 'nfcapd'
        input format
    out_format : str, default 'csv_hist'
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
    bin_exp: int, default 0
        bin width exponent of 2
    x_value : str, default 'length'
        x axis value
    additional_columns : list[str], optional
        additional column to sum
    """

    if bin_exp == 0:
        bin_calc_fn = bin_calc_one
    else:
        bin_calc_fn = bin_calc_log

    bin_calc_function = {'length': lambda p, o, d, r: bin_calc_fn(p, bin_exp),
                         'size': lambda p, o, d, r: bin_calc_fn(o, bin_exp),
                         'duration': lambda p, o, d, r: bin_calc_fn(d, bin_exp),
                         'rate': lambda p, o, d, r: bin_calc_fn(r, bin_exp)}

    bin_calc = bin_calc_function[x_value]
    bins = {}

    reader, writer = IN_FORMATS[in_format], OUT_FORMATS[out_format]

    flows = 0
    written = 0

    fields = ['packets', 'octets']
    if additional_columns is None:
        additional_columns = []
    if 'duration' in additional_columns or 'rate' in additional_columns:
        fields += ['first', 'first_ms', 'last', 'last_ms']
    if 'aggs' in additional_columns:
        fields += ['aggs']

    counters = {'skip_in': skip_in, 'count_in': count_in, 'skip_out': skip_out, 'count_out': count_out}

    try:
        for file in in_files:
            for af, prot, inif, outif, sa0, sa1, sa2, sa3, da0, da1, da2, da3, sp, dp, first, first_ms, last, last_ms, packets, octets, aggs in reader(file, counters=counters, filter_expr=filter_expr, fields=fields):
                duration = 0 if packets == 1 else (last - first) * 1000 + last_ms - first_ms
                rate = 0 if duration == 0 else (8000 * octets) / duration
                bin_lo, bin_hi = bin_calc(packets, octets, duration, rate)
                if bin_lo in bins:
                    flow_bin = bins[bin_lo]
                else:
                    flow_bin = FlowBin(bin_lo, bin_hi)
                    bins[bin_lo] = flow_bin
                flow_bin.flows_sum += 1
                flow_bin.packets_sum += packets
                flow_bin.octets_sum += octets
                flow_bin.duration_sum += duration
                flow_bin.rate_sum += rate
                flow_bin.aggs_sum += aggs
                flows += 1
            logmsg(f'Finished: {file}. Flows: {flows} Bins: {len(bins)}')
    except KeyboardInterrupt:
        pass

    fields = ['bin_lo', 'bin_hi', 'flows_sum', 'packets_sum', 'octets_sum']
    fields += [c + '_sum' for c in additional_columns]

    writer = writer(output, ','.join(fields))
    next(writer)
    for bin_lo in sorted(bins):
        writer.send(bins[bin_lo].to_line(fields))
        written += 1
    writer.close()

    logmsg(f'Finished all files. Flows: {flows} Written: {written}')

def parser():
    p = IOArgumentParser(description=__doc__, epilog=EPILOG)
    p._option_string_actions['-o'].choices = OUT_FORMATS
    p._option_string_actions['-o'].default = 'csv_hist'
    p.add_argument('-b', '--bin-exp', default=0, type=int, help='bin width exponent of 2')
    p.add_argument('-x', '--x-value', default='length', choices=X_VALUES, help='x axis value')
    p.add_argument('-c', '--additional-columns', action='append', default=[], help='additional column to sum')
    return p

def main():
    app_args = parser().parse_args()
    hist(**vars(app_args))


if __name__ == '__main__':
    main()
