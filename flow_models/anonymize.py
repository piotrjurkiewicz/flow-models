#!/usr/bin/python3
"""Anonymizes IP addresses in IPv4 flows using Crypto-PAn algorithm."""

from flow_models.lib.cryptopan import CryptoPan
from flow_models.lib.io import FILTER_HELP, IN_FORMATS, IOArgumentParser, OUT_FORMATS
from flow_models.lib.util import logmsg

EPILOG = \
f"""
This tool anonymizes IPv4 addresses using Crypto-PAn prefix-preserving algorithm.

It works only for IPv4 flows (af==2). Therefore, after processing by this tool
all flows of other address families will be filtered out. Both source (sa3) and
destination (da3) IPv4 addresses are anonymized.

{FILTER_HELP}

Skipping of flow records can be done with skip_in, count_in, skip_out, count_out parameters.
They specify how many flow records should be skipped (skip_in) and then read (count_in)
from input and to be skipped (skip_out) and written (count_out) after filtering.

Example: (encrypts flow records in binary format and outputs as csv lines to standard output)

    flow_models.anonymize -i binary -O - --count-in 1000 --key boojahyoo3vaeToong0Eijee7Ahz3yee sorted
"""

def anonymize(in_files, output, in_format='nfcapd', out_format='csv_flow', skip_in=0, count_in=None, skip_out=0, count_out=None, filter_expr=None, key=''):
    """
    Anonymizes IP addresses in IPv4 flows using Crypto-PAn algorithm.

    Parameters
    ----------
    in_files : list[pathlib.Path]
        input files paths
    output : os.PathLike | io.TextIOWrapper
        output file or directory path or stream
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
    key : str
        encryption key (32 bytes)
    """

    key = key.encode()
    if not key or len(key) != 32:
        raise ValueError("Please specify 32 bytes long encryption key")

    cp = CryptoPan(key)
    ip_cache = {}

    reader, writer = IN_FORMATS[in_format], OUT_FORMATS[out_format]

    writer = writer(output)
    next(writer)

    counters = {'skip_in': skip_in, 'count_in': count_in, 'skip_out': skip_out, 'count_out': count_out}
    written = 0

    for file in in_files:
        for af, prot, inif, outif, sa0, sa1, sa2, sa3, da0, da1, da2, da3, sp, dp, first, first_ms, last, last_ms, packets, octets, aggs in reader(file, counters=counters, filter_expr=filter_expr):
            if not af or af == 2:
                try:
                    sa3 = ip_cache[sa3]
                except KeyError:
                    ip_cache[sa3] = cp.anonymize(sa3)
                    sa3 = ip_cache[sa3]
                try:
                    da3 = ip_cache[da3]
                except KeyError:
                    ip_cache[da3] = cp.anonymize(da3)
                    da3 = ip_cache[da3]
                writer.send((af, prot, inif, outif, sa0, sa1, sa2, sa3, da0, da1, da2, da3, sp, dp, first, first_ms, last, last_ms, packets, octets, aggs))
                written += 1
        logmsg(f'Finished: {file}. Written: {written}')

    writer.close()

    logmsg(f'Finished all files. Written: {written}')

def parser():
    p = IOArgumentParser(description=__doc__, epilog=EPILOG)
    p.add_argument('--key', type=str, default='', help='32 bytes long encryption key')
    return p

def main():
    app_args = parser().parse_args()
    anonymize(**vars(app_args))


if __name__ == '__main__':
    main()
