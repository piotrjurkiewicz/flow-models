#!/usr/bin/python3
import argparse
import json
import pathlib
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
        sample = data.sample(size, replace=True, weights='flows_sum', random_state=random_state)
        packet_size = sample['octets_sum'] / sample['packets_sum']
        sample = sample.index.values
    else:
        sample = rvs(data['flows'], x_val, size, random_state=random_state)
        packet_size = avg(data, sample, x_val, 'packet_size')
    if x_val == 'length':
        packets = sample
        octets = np.rint(packets * packet_size).astype(int)
    elif x_val == 'size':
        octets = sample
        packets = np.rint(octets / packet_size).astype(int)
    else:
        raise NotImplementedError

    return packets, octets

def generate_flows(obj, size=1, x_val='length', random_state=None, batch=None):

    data = load_data(obj)

    assert isinstance(size, int) and size > 0
    if batch is None:
        batch = size
    else:
        assert isinstance(batch, int) and batch > 0

    key = 7 * (0, 0)
    produced = 0

    while True:
        packets, octets = generate_arrays(data, batch, x_val, random_state)
        for packets, octets in zip(packets, octets):
            yield (key, 0, 0, 0, 0, 0, packets, octets, 0)
            produced += 1
            if produced == size:
                break
        break

def generate(obj, out_file, size=1, x_val='length', random_state=None, batch=None, out_format='csv_flow'):

    writer = OUT_FORMATS[out_format]
    writer = writer(out_file)
    next(writer)

    for flow in generate_flows(obj, size, x_val, random_state, batch):
        writer.send(flow)

    writer.close()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-s', type=int, default=1, help='number of generated flows')
    parser.add_argument('-x', default='length', choices=X_VALUES, help='x axis value')
    parser.add_argument('-o', default='csv_flow', choices=OUT_FORMATS, help='format of output')
    parser.add_argument('-O', default=sys.stdout, help='file or directory for output')
    parser.add_argument('file', help='csv_hist file or mixture directory')
    app_args = parser.parse_args()

    generate(app_args.file, app_args.O, app_args.s, app_args.x, out_format=app_args.o)


if __name__ == '__main__':
    main()
