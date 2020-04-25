#!/usr/bin/python3
import argparse
import collections
import concurrent.futures
import os
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
                m, p, r, result_u, result_f = result
                done[m] += int(result_u[0])
                if (m, p, r) in results:
                    results[m, p, r][0] += result_u
                    results[m, p, r][1] += result_f
                else:
                    results[m, p, r] = [result_u, result_f]
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

        while running[0] > 0:
            logmsg('Remaining chunks', running[0],
                   'Done flows', {k: int(v / len(x)) for k, v in done.items()})
            time.sleep(1)

    dataframes = {}

    for method in methods:

        if method == 'sampling':
            ps = x
        else:
            ps = [int(round(1 / p)) for p in x]
            if x_val == 'size':
                ps = [p * 64 for p in ps]

        fl = collections.defaultdict(list)
        pa = collections.defaultdict(list)
        po = collections.defaultdict(list)
        oc = collections.defaultdict(list)

        for (m, p, r), result in results.items():
            if m == method:
                (flows_all, packets_all, octets_all, flows_added, packets_covered), (portion_covered, octets_covered) = result
                fl[p].append(flows_added / flows_all)
                pa[p].append(packets_covered / packets_all)
                po[p].append(portion_covered / flows_all)
                oc[p].append(octets_covered / octets_all)

        d = collections.defaultdict(list)
        for p in ps:
            ad = {
                'flows': 100 * np.array(fl[p]),
                'packets': 100 * np.array(pa[p]),
                'portion': 100 * np.array(po[p]),
                'octets': 100 * np.array(oc[p]),
                'operations': np.reciprocal(fl[p]),
                'occupancy': np.reciprocal(po[p])
            }

            for k in ad:
                ad[k][ad[k] == np.inf] = np.nan
                mean = np.nanmean(ad[k])
                conf = scipy.stats.t.interval(0.95, np.count_nonzero(~np.isnan(ad[k])) - 1, loc=mean, scale=scipy.stats.sem(ad[k], nan_policy='omit'))
                d[k + '_mean'].append(mean)
                d[k + '_conf'].append(mean - conf[0])

        dataframes[method] = pd.DataFrame(d, ps)

    return dataframes

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-n', type=int, default=1000000, help='number of generated flows')
    parser.add_argument('--seed', type=int, default=None, help='seed')
    parser.add_argument('-r', type=int, default=10, help='rounds')
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

    resdic = simulate(app_args.file, app_args.n, app_args.x, app_args.seed, methods, app_args.r, app_args.affinity)
    for method, dataframe in resdic.items():
        print(method)
        print(dataframe.info())
        print(dataframe.to_string())
        if app_args.save:
            dataframe.to_csv(method + '.csv')
            dataframe.to_pickle(method + '.df')

    logmsg('Finished')


if __name__ == '__main__':
    main()
