#!/usr/bin/python3

import argparse
import collections
import concurrent.futures
import random

import numpy as np
import pandas as pd
import scipy.stats

from flow_models.generate import generate_flows, X_VALUES, load_data
from flow_models.lib.util import logmsg

METHODS = ['first', 'treshold', 'sampling']

def simulate_round(data, size, x_val, random_state, method, p):
    rng = random.Random(random_state)
    flows_all = 0
    packets_all = 0
    octets_all = 0
    flows_added = 0
    packets_covered = 0
    portion_covered = 0
    octets_covered = 0
    for flow in generate_flows(data, size, x_val, random_state):
        packets, octets = flow[6], flow[7]
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
        if flows_all % 10000000 == 0:
            logmsg(random_state, p, packets, octets, flows_all, packets_all, octets_all)
    return flows_all, packets_all, octets_all, flows_added, packets_covered, portion_covered, octets_covered

def simulate(obj, size=1, x_val='length', random_state=None, method='first', rounds=5):

    data = load_data(obj)

    ps = [1.0, 0.5, 0.2, 0.1,
          0.05, 0.02, 0.01,
          0.005, 0.002, 0.001,
          0.0005, 0.0002, 0.0001,
          0.00005, 0.00002, 0.00001,
          0.000005, 0.000002, 0.000001,
          0.0000005, 0.0000002, 0.0000001]

    if method != 'sampling':
        ps = [int(round(1 / p)) for p in ps]

    fl = {p: [] for p in ps}
    pa = {p: [] for p in ps}
    po = {p: [] for p in ps}
    oc = {p: [] for p in ps}

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {}
        for r in range(rounds):
            for p in ps:
                fut = executor.submit(simulate_round, data, size, x_val, random_state + r, method, p)
                futures[fut] = r, p
        for fut in concurrent.futures.as_completed(futures):
            try:
                r, p = futures[fut]
                result = fut.result()
                flows_all, packets_all, octets_all, flows_added, packets_covered, portion_covered, octets_covered = result
                if flows_added:
                    fl[p].append(flows_added / flows_all)
                    pa[p].append(packets_covered / packets_all)
                    po[p].append(portion_covered / flows_all)
                    oc[p].append(octets_covered / octets_all)
            except Exception as exc:
                logmsg('Exception', method, r, p, exc)
                # raise
            else:
                logmsg('Done', method, r, p)

    d = collections.defaultdict(list)

    for p in ps:
        flows = 100 * np.array(fl[p])
        packets = 100 * np.array(pa[p])
        portion = 100 * np.array(po[p])
        octets = 100 * np.array(oc[p])
        add = np.reciprocal(fl[p])
        avs = np.reciprocal(po[p])
        
        mean = np.mean(flows)
        conf = scipy.stats.t.interval(0.95, len(flows) - 1, loc=mean, scale=scipy.stats.sem(flows))
        d['flows_mean'].append(mean)
        d['flows_conf'].append(mean - conf[0])

        mean = np.mean(packets)
        conf = scipy.stats.t.interval(0.95, len(packets) - 1, loc=mean, scale=scipy.stats.sem(packets))
        d['packets_mean'].append(mean)
        d['packets_conf'].append(mean - conf[0])

        mean = np.mean(portion)
        conf = scipy.stats.t.interval(0.95, len(packets) - 1, loc=mean, scale=scipy.stats.sem(portion))
        d['portion_mean'].append(mean)
        d['portion_conf'].append(mean - conf[0])

        mean = np.mean(octets)
        conf = scipy.stats.t.interval(0.95, len(octets) - 1, loc=mean, scale=scipy.stats.sem(octets))
        d['octets_mean'].append(mean)
        d['octets_conf'].append(mean - conf[0])

        mean = np.mean(add)
        conf = scipy.stats.t.interval(0.95, len(add) - 1, loc=mean, scale=scipy.stats.sem(add))
        d['add_mean'].append(mean)
        d['add_conf'].append(mean - conf[0])

        mean = np.mean(avs)
        conf = scipy.stats.t.interval(0.95, len(avs) - 1, loc=mean, scale=scipy.stats.sem(avs))
        d['avs_mean'].append(mean)
        d['avs_conf'].append(mean - conf[0])

        # print(f"{p:<5.2g} & {add[0]:.2f} & {add[1]:.2f} & {avs[0]:.2f} & {avs[1]:.2f} & {octets[0]:.3f} & {octets[1]:.3f} \\\\")

    df = pd.concat([pd.Series(d['flows_mean']), pd.Series(d['flows_conf']),
                    pd.Series(d['packets_mean']), pd.Series(d['packets_conf']),
                    pd.Series(d['portion_mean']), pd.Series(d['portion_conf']),
                    pd.Series(d['octets_mean']), pd.Series(d['octets_conf']),
                    pd.Series(d['add_mean']), pd.Series(d['add_conf']),
                    pd.Series(d['avs_mean']), pd.Series(d['avs_conf'])],
                   axis=1)
    df.columns = d.keys()
    df.index = pd.Series(ps)
    return df

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-s', type=int, default=1000000, help='number of generated flows')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--rounds', type=int, default=1, help='rounds')
    parser.add_argument('-x', default='length', choices=X_VALUES, help='x axis value')
    parser.add_argument('-m', default='all', choices=METHODS, help='method')
    parser.add_argument('file', help='csv_hist file or mixture directory')
    app_args = parser.parse_args()

    if app_args.m == 'all':
        methods = METHODS
    else:
        methods = [app_args.m]

    for method in methods:
        dataframe = simulate(app_args.file, app_args.s, app_args.x, app_args.seed, method, app_args.rounds)
        print(method)
        print(dataframe.info())
        print(dataframe.to_string())
        dataframe.to_csv(method + '.csv')
        dataframe.to_pickle(method + '.df')


if __name__ == '__main__':
    main()
