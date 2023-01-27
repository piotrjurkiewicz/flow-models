#!/usr/bin/python3
"""
Generates packets and octets time series from flow records.
"""
import collections

import numpy as np

from .lib.io import IOArgumentParser, IN_FORMATS
from .lib.util import logmsg

def series(in_files, in_format='nfcapd', count=None, skip_input=0, skip_output=0, filter_expr=None):
    """
    Generate packets and octets time series from flow records.

    Parameters
    ----------
    in_files : list[os.PathLike]
        input files paths
    in_format : str, optional
        input format (Default is 'nfcapd')
    count : int, optional
        number of flows to output (Default is 0 (all flows))
    skip_input : int, optional
        number of flows to skip at the beginning of input (Default is 0)
    skip_output : int, optional
        number of flows to skip after filtering (Default is 0)
    filter_expr : str, optional
        filter expression (Default is None)
    """

    reader = IN_FORMATS[in_format]

    # TODO
    if filter_expr is None:
        key_fields = []
    else:
        key_fields = None

    val_fields = ['first', 'first_ms', 'last', 'last_ms', 'packets', 'octets']

    counters = {'count': count, 'skip_input': skip_input, 'skip_output': skip_output}
    written = 0

    day_packets = collections.defaultdict(lambda: np.zeros(86400, 'Q'))
    day_octets = collections.defaultdict(lambda: np.zeros(86400, 'd'))

    for file in in_files:
        for key, first, first_ms, last, last_ms, packets, octets, aggs in reader(file, counters=counters, filter_expr=filter_expr, key_fields=key_fields, val_fields=val_fields):
            day = first // 86400
            second_in_day = first % 86400
            if packets == 1:
                day_packets[day][second_in_day] += 1
                day_octets[day][second_in_day] += octets
            else:
                ms_in_day = float(second_in_day * 1000 + first_ms)
                duration = (last - first) * 1000 + last_ms - first_ms
                piat = duration / packets
                avg_packet_size = octets / packets
                for n in range(packets):
                    second_in_day = int(ms_in_day // 1000)
                    day_packets[day][second_in_day] += 1
                    day_octets[day][second_in_day] += avg_packet_size
                    ms_in_day += piat
                    while ms_in_day >= 86400000:
                        day += 1
                        ms_in_day -= 86400000
            written += 1
        logmsg(f'Finished: {file}. Written: {written}')

    for day in day_packets:
        day_packets[day].tofile(f"{day}.packets", '\n', "%d")
        day_octets[day].tofile(f"{day}.octets", '\n', "%d")

    logmsg(f'Finished all files. Written: {written}')

def parser():
    p = IOArgumentParser(description=__doc__)
    return p

def main():
    app_args = parser().parse_args()
    series(**vars(app_args))


if __name__ == '__main__':
    main()
