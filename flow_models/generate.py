#!/usr/bin/python3
"""
Generates flow records from histograms or mixture models.
"""

import argparse
import itertools
import json
import pathlib
import random
import sys

import numpy as np
import pandas as pd

from .lib.io import OUT_FORMATS
from .lib.mix import rvs, avg
from .lib.util import logmsg

X_VALUES = ['length', 'size', 'duration', 'rate']

def load_data(obj):
    # TODO: move to lib.io, deduplicate
    if isinstance(obj, (str, pathlib.Path)):
        file = pathlib.Path(obj)
        logmsg(f'Loading file {file}')
        if file.suffix == '.json':
            raise ValueError
        elif file.is_dir():
            mixtures = {}
            for ff in file.glob('*.json'):
                mixtures[ff.stem] = json.load(open(str(ff)))
            data = mixtures
        else:
            data = pd.read_csv(file, index_col=0, sep=',', low_memory=False,
                               usecols=lambda col: not col.endswith('_ssq'))
        logmsg(f'Loaded file {file}')
    else:
        data = obj
    return data

def generate_arrays(obj, size=1, x_val='length', random_state=None):
    data = load_data(obj)

    if isinstance(data, pd.DataFrame):
        if size:
            sample = data.sample(size, replace=True, weights='flows_sum', random_state=random_state)
            number = None
        else:
            sample = data.iloc[::-1]
            number = sample['flows_sum'].values
        packet_size = sample['octets_sum'] / sample['packets_sum']
        sample = sample.index.values
        packet_size = packet_size.values
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
        for packets, octets, number in zip(packets, octets, number):
            if number == 1:
                yield packets, octets
            else:
                yield from itertools.repeat((packets, octets), number)

def generate_flows(obj, size=1, x_val='length', random_state=None):
    data = load_data(obj)

    assert isinstance(size, int) and size >= 0
    rng = random.Random(random_state)

    key = 7 * (0, 0)

    packets, octets, number = generate_arrays(data, size, x_val, random_state)
    for packets, octets in iterate_arrays(packets, octets, number):
        if x_val == 'length':
            pks = int(packets)
            ocs = int(octets)
            ocs = ocs if rng.random() + ocs >= octets else ocs + 1
        elif x_val == 'size':
            pks = int(packets)
            pks = pks if rng.random() + pks >= packets else pks + 1
            ocs = int(octets)
        else:
            raise NotImplementedError
        yield key, 0, 0, 0, 0, pks, ocs, 0

def generate(obj, out_file, size=1, x_val='length', random_state=None, out_format='csv_flow'):
    writer = OUT_FORMATS[out_format]
    writer = writer(out_file)
    next(writer)

    for flow in generate_flows(obj, size, x_val, random_state):
        writer.send(flow)

    writer.close()

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-s', type=int, default=1, help='number of generated flows')
    p.add_argument('--seed', type=int, default=None, help='seed')
    p.add_argument('-x', default='length', choices=X_VALUES, help='x axis value')
    p.add_argument('-o', default='csv_flow', choices=OUT_FORMATS, help='format of output')
    p.add_argument('-O', default=sys.stdout, help='file or directory for output')
    p.add_argument('file', help='csv_hist file or mixture directory')
    return p

def main():
    app_args = parser().parse_args()

    generate(app_args.file, app_args.O, app_args.s, app_args.x, app_args.seed, out_format=app_args.o)


if __name__ == '__main__':
    main()
