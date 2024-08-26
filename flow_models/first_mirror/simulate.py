#!/usr/bin/python3
"""Simulates first N packets of new flow mirroring feature."""
import collections
import pathlib

from flow_models.lib.io import FILTER_HELP, IN_FORMATS, IOArgumentParser
from flow_models.lib.util import logmsg

EPILOG = \
f"""
This tool can be used to simulate first N packets of new flow mirroring feature.

{FILTER_HELP}

Skipping of flow records can be done with skip_in, count_in, skip_out, count_out parameters.
They specify how many flow records should be skipped (skip_in) and then read (count_in)
from input and to be skipped (skip_out) and written (count_out) after filtering.

Example: (simulates mirroring of first 3 packets of a flow to control plane)

    flow_models.first_mirror.simulate -i binary -O mirror_3 --mirror 3 sorted
"""

def first_mirror(in_files, output, in_format='nfcapd', out_format='csv_series', skip_in=0, count_in=None, skip_out=0, count_out=None, filter_expr=None, mirror=0):
    """
    Generate mirrored packets and octets time series from flow records.

    Parameters
    ----------
    in_files : list[pathlib.Path]
        input files paths
    output : os.PathLike
        directory path
    in_format : str, default 'nfcapd'
        input format
    out_format : str, default 'csv_series'
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
    mirror : int, default 0
        mirror first N packets to switch control plane
    """

    reader = IN_FORMATS[in_format]
    assert out_format == 'csv_series'

    fields = ['first', 'first_ms', 'last', 'last_ms', 'packets', 'octets']

    counters = {'skip_in': skip_in, 'count_in': count_in, 'skip_out': skip_out, 'count_out': count_out}
    written = 0

    # [packets_mirrored, octets_mirrored]
    dd = collections.defaultdict(lambda: [[0, 0] for _ in range(86400)])

    for file in in_files:
        for af, prot, inif, outif, sa0, sa1, sa2, sa3, da0, da1, da2, da3, sp, dp, first, first_ms, last, last_ms, packets, octets, aggs in reader(file, counters=counters, filter_expr=filter_expr, fields=fields):
            day = first // 86400
            second_in_day = first % 86400
            d = dd[day]
            if packets == 1:
                if mirror > 0:
                    d[second_in_day][0] += 1
                    d[second_in_day][1] += octets
            else:
                float_second_in_day = second_in_day + first_ms / 1000
                duration = last - first + (last_ms - first_ms) / 1000
                piat = duration / packets
                avg_packet_size = octets / packets
                int_packet_size = avg_packet_size.__trunc__()
                for n in range(min(packets, mirror)):
                    packet_size = int_packet_size + (octets - int_packet_size - avg_packet_size * (packets - 1 - n)).__trunc__()
                    ds = d[float_second_in_day.__trunc__()]
                    ds[0] += 1
                    ds[1] += packet_size
                    octets -= packet_size
                    float_second_in_day += piat
                    while float_second_in_day >= 86400.0:
                        day += 1
                        float_second_in_day -= 86400.0
                        d = dd[day]
            written += 1
        logmsg(f'Finished: {file}. Written: {written}')

    output = pathlib.Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for day in dd:
        with open(output / f"{day}.packets_mirrored", 'w') as pm, open(output / f"{day}.octets_mirrored", 'w') as om:
            for second_in_day in range(86400):
                pm.write("%d\n" % dd[day][second_in_day][0])
                om.write("%d\n" % dd[day][second_in_day][1])

    logmsg(f'Finished all files. Written: {written}')

def parser():
    p = IOArgumentParser(description=__doc__, epilog=EPILOG)
    p._option_string_actions['-o'].choices = ['csv_series']
    p._option_string_actions['-o'].default = 'csv_series'
    p._option_string_actions['-O'].help = 'directory for output'
    p._option_string_actions['-O'].default = '.'
    p.add_argument('--mirror', default=0, type=int, help='mirror first N packets')
    return p

def main():
    app_args = parser().parse_args()
    first_mirror(**vars(app_args))


if __name__ == '__main__':
    main()
