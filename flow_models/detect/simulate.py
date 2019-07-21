#!/usr/bin/python3

# for N in {0..39}; do taskset -cp $N $((N+16492)); done

import argparse
import collections
import concurrent.futures
import pickle
import random
import threading
import time

import numpy as np
import pandas as pd
import scipy.stats

from flow_models.generate import generate_flows, X_VALUES, load_data
from flow_models.lib.util import logmsg

METHODS = ['first', 'treshold', 'sampling']
CHUNK_SIZE = 100000

def chunker(iterator):
    chunk = []
    try:
        while True:
            for _ in range(CHUNK_SIZE):
                flow = next(iterator)
                chunk.append((flow[5], flow[6]))
            yield pickle.dumps(chunk, protocol=pickle.HIGHEST_PROTOCOL)
            chunk.clear()
    except StopIteration:
        yield pickle.dumps(chunk, protocol=pickle.HIGHEST_PROTOCOL)

def simulate_chunk(data, random_state, method, p, r):
    data = pickle.loads(data)
    rng = random.Random(random_state)
    p = p if method == 'sampling' else int(round(1 / p))
    flows_all = 0
    packets_all = 0
    octets_all = 0
    flows_added = 0
    packets_covered = 0
    portion_covered = 0
    octets_covered = 0
    for packets, octets in data:
        flows_all += 1
        packets_all += packets
        octets_all += octets
        add_on_packet = packets
        if method == 'first':
            if packets > p:
                add_on_packet = 0
        elif method == 'treshold':
            add_on_packet = p
        else:
            for pkt_n in range(0, packets):
                if rng.random() < p:
                    add_on_packet = pkt_n
                    break
        if add_on_packet < packets:
            flows_added += 1
            packets_covered += packets - add_on_packet
            portion_of_flow_covered = (packets - add_on_packet) / packets
            portion_covered += portion_of_flow_covered
            octets_covered += octets * portion_of_flow_covered
    a = np.array([flows_all, packets_all, octets_all, flows_added, packets_covered], dtype='u8')
    b = np.array([portion_covered, octets_covered], dtype='f8')
    return method, p, r, a, b

def simulate(obj, size=1, x_val='length', random_state=None, methods=tuple(METHODS), rounds=5):
    data = load_data(obj)

    x = [1.0, 0.5, 0.2, 0.1,
         0.05, 0.02, 0.01,
         0.005, 0.002, 0.001,
         0.0005, 0.0002, 0.0001,
         0.00005, 0.00002, 0.00001,
         0.000005, 0.000002, 0.000001,
         0.0000005, 0.0000002, 0.0000001]

    running = [0]
    done = collections.defaultdict(int)
    results = {}
    lock = threading.Lock()

    def cb(future):
        with lock:
            running[0] -= 1
            try:
                result = future.result()
                m, p, r, a, b = result
                done[m] += int(a[0])
                if (m, p, r) in results:
                    results[m, p, r][0] += a
                    results[m, p, r][1] += b
                else:
                    results[m, p, r] = [a, b]
            except Exception as exc:
                logmsg('Exception', exc)
                raise

    with concurrent.futures.ProcessPoolExecutor() as executor:

        for r in range(rounds):
            for chunk in chunker(generate_flows(data, size, x_val, random_state)):
                for p in x:
                    for method in methods:
                        fut = executor.submit(simulate_chunk, chunk, r, method, p, r)
                        fut.add_done_callback(cb)
                        with lock:
                            running[0] += 1
                            queued = running[0]
                        if queued >= 4096:
                            logmsg('Queued chunks', queued,
                                   'Done flows', {k: int(v / len(x)) for k, v in done.items()})
                            time.sleep(1)
            if isinstance(random_state, int):
                random_state += 1

        while running[0] > 0:
            logmsg('Remaining chunks', running[0],
                   'Done flows', {k: int(v / len(x)) for k, v in done.items()})
            time.sleep(1)

    dataframes = {}

    for method in methods:

        ps = x if method == 'sampling' else [int(round(1 / p)) for p in x]
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
                'add': np.reciprocal(fl[p]),
                'avs': np.reciprocal(po[p])
            }

            for k in ad:
                ad[k][ad[k] == np.inf] = np.nan
                mean = np.nanmean(ad[k])
                conf = scipy.stats.t.interval(0.95, np.count_nonzero(~np.isnan(ad[k])) - 1, loc=mean, scale=scipy.stats.sem(ad[k], nan_policy='omit'))
                d[k + '_mean'].append(mean)
                d[k + '_conf'].append(mean - conf[0])

            # print(f"{p:<5.2g} & {add[0]:.2f} & {add[1]:.2f} & {avs[0]:.2f} & {avs[1]:.2f} & {octets[0]:.3f} & {octets[1]:.3f} \\\\")

        dataframes[method] = pd.DataFrame(d, ps)

    return dataframes

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-s', type=int, default=1000000, help='number of generated flows')
    parser.add_argument('--seed', type=int, default=None, help='seed')
    parser.add_argument('--rounds', type=int, default=10, help='rounds')
    parser.add_argument('-x', default='length', choices=X_VALUES, help='x axis value')
    parser.add_argument('-m', default='all', choices=METHODS, help='method')
    parser.add_argument('file', help='csv_hist file or mixture directory')
    app_args = parser.parse_args()

    if app_args.m == 'all':
        methods = METHODS
    else:
        methods = [app_args.m]

    resdic = simulate(app_args.file, app_args.s, app_args.x, app_args.seed, methods, app_args.rounds)
    for method, dataframe in resdic.items():
        print(method)
        print(dataframe.info())
        print(dataframe.to_string())
        dataframe.to_csv(method + '.csv')
        dataframe.to_pickle(method + '.df')

    logmsg('Finished')


if __name__ == '__main__':
    main()
