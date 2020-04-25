#!/usr/bin/python3
import argparse
import collections
import concurrent.futures
import json
import os
import pathlib
import pickle
import random
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd
import scipy.stats

from flow_models.generate import generate_flows, X_VALUES, load_data
from flow_models.lib.util import logmsg

METHODS = ['first', 'threshold', 'sampling']
CHUNK_SIZE = 262144

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def set_affinity(executor):
    n = 0
    subprocess.check_call(['renice', '19'] + [str(p) for p in executor._processes], stdout=sys.stderr)
    for pid in executor._processes:
        logmsg(f'Setting affinity {pid} to CPU{n}')
        os.sched_setaffinity(pid, [n])
        n += 1

def chunker(iterable):
    chunk = []
    for flow in iterable:
        chunk.append((flow[5], flow[6]))
        if len(chunk) == CHUNK_SIZE:
            yield pickle.dumps(chunk, protocol=pickle.HIGHEST_PROTOCOL)
            chunk.clear()
    if len(chunk) > 0:
        yield pickle.dumps(chunk, protocol=pickle.HIGHEST_PROTOCOL)
        del chunk

def conf(d):
    mean = np.nanmean(d)
    c = scipy.stats.t.interval(0.95, np.count_nonzero(~np.isnan(d)) - 1, loc=mean,
                               scale=scipy.stats.sem(d, nan_policy='omit'))
    return mean - c[0]

def simulate_chunk(data, x_val, seed, method, p, r):
    data = pickle.loads(data)
    rng = random.Random(seed)
    if method == 'sampling':
        p = p
    else:
        p = int(round(1 / p))
        if x_val == 'size':
            p *= 64
    flows_all = 0
    packets_all = 0
    octets_all = 0
    flows_added = 0
    packets_covered = 0
    fraction_covered = 0
    octets_covered = 0
    packet_size = 1
    min_packet_size = 1
    for packets, octets in data:
        flows_all += 1
        packets_all += packets
        octets_all += octets
        add_on_packet = packets
        if x_val == 'size':
            packet_size = octets / packets
            min_packet_size = 64
        if method == 'first':
            if packets > (p if x_val == 'length' else p / packet_size):
                add_on_packet = 0
        elif method == 'threshold':
            add_on_packet = p if x_val == 'length' else p / packet_size
        else:
            p_scaled = p if x_val == 'length' else p * (packet_size / min_packet_size)
            for pkt_n in range(packets):
                if rng.random() < p_scaled:
                    add_on_packet = pkt_n
                    break
        if add_on_packet < packets:
            flows_added += 1
            packets_covered += packets - add_on_packet
            fraction_of_flow_covered = (packets - add_on_packet) / packets
            fraction_covered += fraction_of_flow_covered
            octets_covered += octets * fraction_of_flow_covered
    result_u = np.array([flows_all, packets_all, octets_all, flows_added, packets_covered], dtype='u8')
    result_f = np.array([fraction_covered, octets_covered], dtype='f8')
    return method, p, r, result_u, result_f

def simulate(obj, size=1, x_val='length', seed=None, methods=tuple(METHODS), rounds=5, affinity=False):
    data = load_data(obj)

    x = 1 / np.power(2, range(25))

    running = [0]
    done = collections.defaultdict(int)
    results = {}
    lock = threading.Lock()

    def cb(future):
        with lock:
            running[0] -= 1
            try:
                result = future.result()
                mm, pp, rr, result_u, result_f = result
                done[mm] += int(result_u[0])
                if (mm, pp, rr) in results:
                    results[mm, pp, rr][0] += result_u
                    results[mm, pp, rr][1] += result_f
                else:
                    results[mm, pp, rr] = [result_u, result_f]
            except Exception as exc:
                logmsg('Exception', exc)
                raise

    with concurrent.futures.ProcessPoolExecutor(initializer=init_worker) as executor:
        try:
            for r in range(rounds):
                for chunk in chunker(generate_flows(data, size, x_val, seed + r if isinstance(seed, int) else seed)):
                    for p in x:
                        for method in methods:
                            fut = executor.submit(simulate_chunk, chunk, x_val, seed, method, p, r)
                            fut.add_done_callback(cb)
                            with lock:
                                running[0] += 1
                                queued = running[0]
                                if affinity and queued == 1 and not done:
                                    set_affinity(executor)
                            if queued >= 16384:
                                logmsg('Queued chunks', queued,
                                       'Done flows', {k: int(v / len(x)) for k, v in done.items()})
                                time.sleep(1)
        except KeyboardInterrupt:
            pass

        try:
            while running[0] > 0:
                logmsg('Remaining chunks', running[0],
                       'Done flows', {k: int(v / len(x)) for k, v in done.items()})
                time.sleep(1)
        except KeyboardInterrupt:
            os.kill(0, signal.SIGTERM)

    z = pd.DataFrame({k: (*v[0], *v[1]) for k, v in results.items()})
    z = z.transpose()
    cols = ['flows_all', 'packets_all', 'octets_all', 'flows_added',
            'packets_covered', 'fraction_covered', 'octets_covered']
    z.columns = cols.copy()
    z['flows'] = 100 * z['flows_added'] / z['flows_all']
    z['packets'] = 100 * z['packets_covered'] / z['packets_all']
    z['fraction'] = 100 * z['fraction_covered'] / z['flows_all']
    z['octets'] = 100 * z['octets_covered'] / z['octets_all']
    z['operations'] = 100 / z['flows']
    z['occupancy'] = 100 / z['fraction']
    z = z.replace(np.inf, np.nan)
    z = z.drop(columns=cols)

    df = z.groupby(level=[0, 1], axis=0).agg([np.nanmean, conf]).rename(columns={'nanmean': 'mean'})
    df.columns = ['_'.join(col).strip() for col in df.columns]
    dd = dict(iter(df.groupby(level=0)))
    for k, df in dd.items():
        df = df.droplevel(0, 0)
        if k == 'sampling':
            df.sort_index(ascending=False, inplace=True)
        else:
            df.index = df.index.astype('uint64')
        dd[k] = df

    return dd

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-n', type=int, default=1000000, help='number of generated flows')
    parser.add_argument('-r', type=int, default=10, help='rounds')
    parser.add_argument('--seed', type=int, default=None, help='seed')
    parser.add_argument('-x', default='length', choices=X_VALUES, help='x axis value')
    parser.add_argument('-m', default='all', choices=METHODS, help='method')
    parser.add_argument('--save', action='store_true', help='save to files')
    parser.add_argument('--affinity', action='store_true', help='set affinity')
    parser.add_argument('file', help='csv_hist file or mixture directory')
    app_args = parser.parse_args()

    if app_args.m == 'all':
        methods = METHODS
    else:
        methods = [app_args.m]

    if app_args.save:
        pathlib.Path(f'{app_args.m}.mode').write_text(json.dumps({k: v for k, v in vars(app_args).items() if k != 'file'}) + '\n')

    resdic = simulate(app_args.file, app_args.n, app_args.x, app_args.seed, methods, app_args.r, app_args.affinity)
    for method, dataframe in resdic.items():
        print(method)
        print(dataframe.info())
        print(dataframe.to_string())
        if app_args.save:
            dataframe.to_string(open(method + '.txt', 'w'))
            dataframe.to_csv(method + '.csv')
            dataframe.to_pickle(method + '.df')
            dataframe.to_html(method + '.html')
            dataframe.to_latex(method + '.tex', float_format='%.2f')

    logmsg('Finished')


if __name__ == '__main__':
    main()
