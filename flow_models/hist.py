#!/usr/bin/python3
"""
Calculates histograms of flows length, size, duration or rate.
"""

import argparse

from .lib.io import io_parser, write_none, write_line, IN_FORMATS
from .lib.util import logmsg, bin_calc_one, bin_calc_log, prepare_file_list

X_VALUES = ['length', 'size', 'duration', 'rate']
OUT_FORMATS = {'csv_hist': write_line, 'none': write_none}

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

def histogram(in_files, out_file, in_format='nfcapd', out_format='csv_hist', bin_exp=0, x_value='length', additional_columns=(), prot=None):

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

    val_fields = ['packets', 'octets']
    if 'duration' in additional_columns or 'rate' in additional_columns:
        val_fields += ['first', 'first_ms', 'last', 'last_ms']
    if 'aggs' in additional_columns:
        val_fields += ['aggs']

    key_fields = []
    if prot:
        key_fields.append('prot')

    try:
        for file in in_files:
            for key, first, first_ms, last, last_ms, packets, octets, aggs in reader(file, key_fields=key_fields, val_fields=val_fields):
                if prot and key[1] != prot:
                    continue
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

    writer = writer(out_file, ','.join(fields))
    next(writer)
    for bin_lo in sorted(bins):
        writer.send(bins[bin_lo].to_line(fields))
        written += 1
    writer.close()

    logmsg(f'Finished all files. Flows: {flows} Written: {written}')

def parser():
    p = argparse.ArgumentParser(description=__doc__, parents=[io_parser])
    p._option_string_actions['-o'].choices = OUT_FORMATS
    p._option_string_actions['-o'].default = 'csv_hist'
    p.add_argument('-x', default='length', choices=X_VALUES, help='x axis value')
    p.add_argument('-b', default=0, type=int, help='bin width exponent of 2')
    p.add_argument('-c', action='append', default=[], help='additional column to sum')
    p.add_argument('--prot')
    return p

def main():
    app_args = parser().parse_args()

    if app_args.i == 'binary':
        input_files = app_args.files
    else:
        input_files = prepare_file_list(app_args.files)

    if app_args.prot:
        prot = {'all': None, 'tcp': 6, 'udp': 17}[app_args.prot]
    else:
        prot = None

    histogram(input_files, app_args.O, app_args.i, app_args.o, app_args.b, app_args.x, app_args.c, prot)


if __name__ == '__main__':
    main()
