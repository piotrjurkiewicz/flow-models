#!/usr/bin/python3
"""
Merges flows which were split across multiple records due to *active timeout*.
"""

import argparse
import warnings

from .lib.io import FlowValFields, io_parser, IN_FORMATS, OUT_FORMATS
from .lib.util import logmsg, prepare_file_list

class Flow:

    __slots__ = 'key', 'val'

    def __init__(self, key, first, first_ms, last, last_ms, packets, octets, aggs):
        self.key = key
        self.val = FlowValFields()
        self.val.first = first
        self.val.first_ms = first_ms
        self.val.last = last
        self.val.last_ms = last_ms
        self.val.packets = packets
        self.val.octets = octets
        self.val.aggs = aggs or 1

    def to_tuple(self):
        return self.key, self.val.first, self.val.first_ms, self.val.last, self.val.last_ms, \
               self.val.packets, self.val.octets, self.val.aggs

def merge(in_files, out_file, in_format='nfcapd', out_format='csv_flow', inactive_timeout=15.0, active_timeout=300.0):
    """
    Merge flows split due to timeout.

    :param list[os.PathLike] in_files: input files paths
    :param os.PathLike | _io.TextIOWrapper out_file: output file or directory path or stream
    :param in_format: format of input files
    :param out_format: format of output
    :param inactive_timeout: inactive timeout in seconds
    :param active_timeout: active timeout in seconds
    """
    inactive_s, inactive_ms = divmod(inactive_timeout, 1)
    inactive_s, inactive_ms = int(inactive_s), int(inactive_ms * 1000)

    active_time = active_timeout - inactive_timeout

    cache = {}

    reader, writer = IN_FORMATS[in_format], OUT_FORMATS[out_format]

    writer = writer(out_file)
    next(writer)

    written = 0
    merged = 0
    wrong = 0

    for file in in_files:
        for key, first, first_ms, last, last_ms, packets, octets, aggs in reader(file):
            new_flow = Flow(key, first, first_ms, last, last_ms, packets, octets, aggs)
            if key in cache:
                old_flow = cache[key]
                nfs, nfm = new_flow.val.first, new_flow.val.first_ms
                ols, olm = old_flow.val.last, old_flow.val.last_ms
                nls, nlm = new_flow.val.last, new_flow.val.last_ms
                ofs, ofm = old_flow.val.first, old_flow.val.first_ms
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
                delta_s = new_flow.val.first - old_flow.val.last
                delta_ms = new_flow.val.first_ms - old_flow.val.last_ms
                if delta_ms < 0:
                    delta_s -= 1
                    delta_ms = 1000 - delta_ms
                if delta_s < inactive_s or delta_s == inactive_s and delta_ms < inactive_ms:
                    # merge flows
                    merged += 1
                    old_flow.val.last = new_flow.val.last                    # update last
                    old_flow.val.last_ms = new_flow.val.last_ms              # update last
                    old_flow.val.aggs += 1                                   # add flow
                    old_flow.val.packets += new_flow.val.packets             # add packets
                    old_flow.val.octets += new_flow.val.octets               # add octets
                    if new_flow.val.last - new_flow.val.first < active_time:
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
                    if new_flow.val.last - new_flow.val.first < active_time:
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
                if new_flow.val.last - new_flow.val.first < active_time:
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
    p = argparse.ArgumentParser(description=__doc__, parents=[io_parser])
    p.add_argument('-I', type=float, default=15.0, help='inactive timeout in seconds')
    p.add_argument('-A', type=float, default=300.0, help='active timeout in seconds')
    return p

def main():
    app_args = parser().parse_args()

    if app_args.i == 'binary':
        input_files = app_args.files
    else:
        input_files = prepare_file_list(app_args.files)

    merge(input_files, app_args.O, app_args.i, app_args.o, app_args.I, app_args.A)


if __name__ == '__main__':
    main()
