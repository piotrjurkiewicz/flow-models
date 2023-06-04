#!/usr/bin/python3
"""
Generates packets and octets time series from flow records.
"""
import collections

from .lib.io import IOArgumentParser, IN_FORMATS
from .lib.util import logmsg

def series(in_files, out_file, in_format='nfcapd', out_format='csv_series', count=None, skip_input=0, skip_output=0, filter_expr=None):
    """
    Generate packets and octets time series from flow records.

    Parameters
    ----------
    in_files : list[os.PathLike]
        input files paths
    out_file : Union[os.PathLike, io.TextIOWrapper]
        directory path
    in_format : str, optional
        input format (Default is 'nfcapd')
    out_format : str, optional
        output format (Default is 'csv_series')
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
    assert out_format == 'csv_series'

    fields = ['first', 'first_ms', 'last', 'last_ms', 'packets', 'octets']

    counters = {'count': count, 'skip_input': skip_input, 'skip_output': skip_output}
    written = 0

    dd = collections.defaultdict(lambda: [[0, 0] for _ in range(86400)])

    for file in in_files:
        for key, first, first_ms, last, last_ms, packets, octets, aggs in reader(file, counters=counters, filter_expr=filter_expr, fields=fields):
            day = first // 86400
            second_in_day = first % 86400
            d = dd[day]
            if packets == 1:
                d[second_in_day][0] += 1
                d[second_in_day][1] += octets
            else:
                float_second_in_day = second_in_day + first_ms / 1000
                duration = last - first + (last_ms - first_ms) / 1000
                piat = duration / packets
                avg_packet_size = octets / packets
                int_packet_size = avg_packet_size.__trunc__()
                for n in range(packets):
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
                assert octets == 0
            written += 1
        logmsg(f'Finished: {file}. Written: {written}')

    for day in dd:
        with open(out_file / f"{day}.packets", 'w') as p, open(out_file / f"{day}.octets", 'w') as o:
            for second_in_day in range(86400):
                p.write("%d\n" % dd[day][second_in_day][0])
                o.write("%d\n" % dd[day][second_in_day][1])

    logmsg(f'Finished all files. Written: {written}')

def parser():
    p = IOArgumentParser(description=__doc__)
    p._option_string_actions['-o'].choices = ['csv_series']
    p._option_string_actions['-o'].default = 'csv_series'
    p._option_string_actions['-O'].default = '.'
    return p

def main():
    app_args = parser().parse_args()
    series(**vars(app_args))


if __name__ == '__main__':
    main()
