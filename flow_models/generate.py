#!/usr/bin/python3
"""Generates flow records from histograms or mixture models."""

import argparse
import itertools
import random
import sys

import numpy as np
import pandas as pd

from flow_models.lib.data import load_data
from flow_models.lib.io import OUT_FORMATS
from flow_models.lib.mix import avg, rvs

EPILOG = \
"""
This tool can be used to generate example flow tuples from mixture or histogram file.

Generated flow stream will preserve distribution of flow features (length, size).

Example:

    flow_models.generate -s 1000 histograms/udp/length.csv
"""

X_VALUES = ['length', 'size', 'duration', 'rate']

def generate_arrays(obj, size=1, x_val='length', random_state=None):
    data = next(iter(load_data([obj]).values()))

    if isinstance(data, pd.DataFrame):
        if size:
            sample = data.sample(size, replace=True, weights='flows_sum', random_state=random_state)
            number = None
        else:
            sample = data.iloc[::-1]
            number = sample['flows_sum'].to_numpy()
        packet_size = sample['octets_sum'] / sample['packets_sum']
        sample = sample.index.to_numpy()
        packet_size = packet_size.to_numpy()
    else:
        assert size
        sample = rvs(data['flows'], x_val, size, random_state=random_state)
        number = None
        packet_size = avg(data, sample, x_val, 'packet_size')
    if x_val == 'length':
        np.clip(packet_size, 64, 1522, out=packet_size)
        packets = sample
        octets = packets * packet_size
    elif x_val == 'size':
        octets = sample
        packets = octets / packet_size
        np.trunc(packets, out=packets, where=octets / np.ceil(packets) < 64)
        np.clip(packets, 1, None, out=packets)
    else:
        raise NotImplementedError

    return packets, octets, number

def iterate_arrays(packets, octets, number):
    if number is None:
        yield from zip(packets, octets)
    else:
        for pa, oc, nu in zip(packets, octets, number):
            if nu == 1:
                yield pa, oc
            else:
                yield from itertools.repeat((pa, oc), nu)

def generate_flows(in_file, size=1, x_value='length', random_state=None):
    """
    Yield flow tuples generated from mixture or histogram file.

    Parameters
    ----------
    in_file : os.PathLike
        csv_hist file or mixture director
    size : int, default 1
        number of flows to generate
    x_value : str, default 'length'
        x axis value
    random_state : object, optional
        initial random state

    Yields
    ------
    (int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int)
        af, prot, inif, outif, sa0, sa1, sa2, sa3, da0, da1, da2, da3, sp, dp, first, first_ms, last, last_ms, packets, octets, aggs
    """

    data = next(iter(load_data([in_file]).values()))

    assert isinstance(size, int)
    assert size >= 0
    rng = random.Random(random_state)

    packets, octets, number = generate_arrays(data, size, x_value, random_state)
    for pa, oc in iterate_arrays(packets, octets, number):
        if x_value == 'length':
            pks = int(pa)
            ocs = int(oc)
            ocs = ocs if rng.random() + ocs >= oc else ocs + 1
        elif x_value == 'size':
            pks = int(pa)
            pks = pks if rng.random() + pks >= pa else pks + 1
            ocs = int(oc)
        else:
            raise NotImplementedError
        yield 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, pks, ocs, 0

def generate(in_file, output, out_format='csv_flow', size=1, x_value='length', random_state=None):
    """
    Generate flows from mixture or histogram file to output.

    Parameters
    ----------
    in_file : os.PathLike
        csv_hist file or mixture director
    output : os.PathLike | io.TextIOWrapper
        file or directory for output
    out_format : str, default 'csv_flow'
        output format
    size : int, default 1
        number of flows to generate
    x_value : str, default 'length'
        x axis value
    random_state : object, optional
        initial random state
    """

    writer = OUT_FORMATS[out_format]
    writer = writer(output)
    next(writer)

    for flow in generate_flows(in_file, size, x_value, random_state):
        writer.send(flow)

    writer.close()

def parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=EPILOG)
    p.add_argument('-s', type=int, default=1, help='number of generated flows')
    p.add_argument('--seed', type=int, default=None, help='seed')
    p.add_argument('-x', default='length', choices=X_VALUES, help='x axis value')
    p.add_argument('-o', default='csv_flow', choices=OUT_FORMATS, help='format of output')
    p.add_argument('-O', default=sys.stdout, help='file or directory for output')
    p.add_argument('file', help='csv_hist file or mixture directory')
    return p

def main():
    app_args = parser().parse_args()

    generate(app_args.file, app_args.O, app_args.o, app_args.s, app_args.x, app_args.seed)


if __name__ == '__main__':
    main()
