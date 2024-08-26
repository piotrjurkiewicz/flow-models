#!/usr/bin/python3
"""Merges flows which were split across multiple records due to *active timeout*."""

import warnings

from flow_models.lib.io import FILTER_HELP, IN_FORMATS, IOArgumentParser, OUT_FORMATS
from flow_models.lib.util import logmsg

EPILOG = \
f"""
This tool can be used to merge flow records which were split during the collection
into multiple records due to active timeout.

User should specify active and inactive timeout values which were used during the
records collection to correctly merge flow records.

{FILTER_HELP}

Skipping of flow records can be done with skip_in, count_in, skip_out, count_out parameters.
They specify how many flow records should be skipped (skip_in) and then read (count_in)
from input and to be skipped (skip_out) and written (count_out) after filtering.

Example: (merges flows from the cleaned directory and writes output to the merged directory)

    flow_models.merge -i nfcapd -o binary -I 15 -A 300 -O merged cleaned
"""

class Flow:

    __slots__ = 'key', 'first', 'first_ms', 'last', 'last_ms', 'packets', 'octets', 'aggs'

    def __init__(self, flow_tuple):
        self.key = flow_tuple[0:14]
        self.first = flow_tuple[14]
        self.first_ms = flow_tuple[15]
        self.last = flow_tuple[16]
        self.last_ms = flow_tuple[17]
        self.packets = flow_tuple[18]
        self.octets = flow_tuple[19]
        self.aggs = flow_tuple[20] or 1

    def to_tuple(self):
        return *self.key, self.first, self.first_ms, self.last, self.last_ms, self.packets, self.octets, self.aggs

def merge(in_files, output, in_format='nfcapd', out_format='csv_flow', skip_in=0, count_in=None, skip_out=0, count_out=None, filter_expr=None, inactive_timeout=15.0, active_timeout=300.0):
    """
    Merge flows split due to timeout.

    Parameters
    ----------
    in_files : list[os.PathLike]
        input files paths
    output : os.PathLike | io.TextIOWrapper
        directory path
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
    inactive_timeout : float, default 15.0
        inactive timeout in seconds
    active_timeout : float, default 300.0
        active timeout in seconds
    """
    inactive_s, inactive_ms = divmod(inactive_timeout, 1)
    inactive_s, inactive_ms = int(inactive_s), int(inactive_ms * 1000)

    active_time = active_timeout - inactive_timeout

    cache = {}

    reader, writer = IN_FORMATS[in_format], OUT_FORMATS[out_format]

    writer = writer(output)
    next(writer)

    counters = {'skip_in': skip_in, 'count_in': count_in, 'skip_out': skip_out, 'count_out': count_out}
    written = 0
    merged = 0
    wrong = 0

    for file in in_files:
        for flow_tuple in reader(file, counters=counters, filter_expr=filter_expr):
            new_flow = Flow(flow_tuple)
            key = new_flow.key
            if key in cache:
                old_flow = cache[key]
                nfs, nfm = new_flow.first, new_flow.first_ms
                ols, olm = old_flow.last, old_flow.last_ms
                nls, nlm = new_flow.last, new_flow.last_ms
                ofs, ofm = old_flow.first, old_flow.first_ms
                if nfs > ols or nfs == ols and nfm > olm:    # new first > old last
                    # correct order
                    pass
                elif ofs > nls or ofs == nls and ofm > nlm:  # old first > new last
                    # reversed order
                    old_flow, new_flow = new_flow, old_flow
                    warnings.warn("Found a flow with the reversed order")
                else:
                    # error
                    wrong += 1
                    del cache[key]
                    continue
                delta_s = new_flow.first - old_flow.last
                delta_ms = new_flow.first_ms - old_flow.last_ms
                if delta_ms < 0:
                    delta_s -= 1
                    delta_ms = 1000 - delta_ms
                if delta_s < inactive_s or delta_s == inactive_s and delta_ms < inactive_ms:
                    # merge flows
                    merged += 1
                    old_flow.last = new_flow.last                    # update last
                    old_flow.last_ms = new_flow.last_ms              # update last
                    old_flow.aggs += 1                                   # add flow
                    old_flow.packets += new_flow.packets             # add packets
                    old_flow.octets += new_flow.octets               # add octets
                    if new_flow.last - new_flow.first < active_time:
                        # too short to merge
                        # dump it
                        del cache[key]
                        writer.send(old_flow.to_tuple())
                        written += 1
                else:
                    # dump old flow from cache
                    del cache[key]
                    writer.send(old_flow.to_tuple())
                    written += 1
                    # new flow
                    if new_flow.last - new_flow.first < active_time:
                        # too short to merge
                        # dump new flow too
                        writer.send(new_flow.to_tuple())
                        written += 1
                    else:
                        # candidate to merge
                        # add new flow to cache
                        cache[key] = new_flow
            else:
                # new flow
                if new_flow.last - new_flow.first < active_time:
                    # too short to merge
                    # dump it asap
                    writer.send(new_flow.to_tuple())
                    written += 1
                else:
                    # candidate to merge
                    # add it to cache
                    cache[key] = new_flow

        logmsg(f'Finished: {file} Cached: {len(cache)} Wrong: {wrong} Merged: {merged} Written: {written}')

    for flow in cache.values():
        # dump all remaining flows
        writer.send(flow.to_tuple())
        written += 1

    writer.close()

    logmsg(f'Finished all files. Wrong: {wrong} Merged: {merged} Written: {written}')

def parser():
    p = IOArgumentParser(description=__doc__, epilog=EPILOG)
    p.add_argument('-I', '--inactive-timeout', type=float, default=15.0, help='inactive timeout in seconds')
    p.add_argument('-A', '--active-timeout', type=float, default=300.0, help='active timeout in seconds')
    return p

def main():
    app_args = parser().parse_args()
    merge(**vars(app_args))


if __name__ == '__main__':
    main()
