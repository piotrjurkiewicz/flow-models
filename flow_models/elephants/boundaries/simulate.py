#!/usr/bin/python3
"""Simulates first, threshold and sampling elephant detection methods on packet-level from mixture or histogram data.

Flows are generated with :func:`flow_models.generate.generate_flows` and each
flow is processed packet by packet.  Every (method, x grid point) pair is
repeated ``rounds`` times and results are aggregated into means and 95%
confidence intervals.  The simulation models packet-level detection delay:
``first`` inserts an above-threshold flow before its first packet (an oracle
bound for first-packet classifiers), ``threshold`` inserts a flow only after
the threshold count of packets or bytes has been observed, and ``sampling``
inserts a flow at its first sampled packet.  For ``x_val='size'`` sampling
operates on 64-byte units, so the per-packet detection probability is
``1 - (1 - p) ** (packet_size / 64)``.

See :program:`calculate` for the analytical equivalent and
:mod:`flow_models.elephants.simulate_data` for flow-level simulation on real
flow records.
"""

import argparse
import collections
import concurrent.futures
import json
import os
import pathlib
import pickle
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd
import scipy.stats

from flow_models.generate import X_VALUES, generate_flows, load_data
from flow_models.lib.util import logmsg

METHODS = ['first', 'threshold', 'sampling']
CHUNK_SIZE = 262144

# real-times experiment window, matching elephants.skl training scripts
REAL_FPS = 1810
REAL_ST, REAL_ET = 300, 720

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def set_affinity(executor):
    subprocess.check_call(['renice', '19'] + [str(p) for p in executor._processes], stdout=sys.stderr)
    for n, pid in enumerate(executor._processes):
        logmsg(f'Setting affinity {pid} to CPU{n}')
        os.sched_setaffinity(pid, [n])

def chunker(iterable):
    """
    Group generated flow records into pickled chunks of ``(packets, octets)``.

    Parameters
    ----------
    iterable : iterable
        flow records in the binary array format (packets at index 18,
        octets at index 19)

    Yields
    ------
    bytes
        pickled list of ``(packets, octets)`` tuples of at most CHUNK_SIZE
    """
    chunk = []
    for flow in iterable:
        chunk.append((flow[18], flow[19]))
        if len(chunk) == CHUNK_SIZE:
            yield pickle.dumps(chunk, protocol=pickle.HIGHEST_PROTOCOL)
            chunk.clear()
    if len(chunk) > 0:
        yield pickle.dumps(chunk, protocol=pickle.HIGHEST_PROTOCOL)
        del chunk

def conf(d):
    """Half-width of the 95% t-distribution confidence interval of ``d``."""
    mean = np.nanmean(d)
    c = scipy.stats.t.interval(0.95, np.count_nonzero(~np.isnan(d)) - 1, loc=mean,
                               scale=scipy.stats.sem(d, nan_policy='omit'))
    return mean - c[0]

def simulate_chunk(data, x_val, seed, methods, ps, r, pps=None, timeout=15):
    """
    Simulate all detection methods on a single chunk of generated flows.

    Parameters
    ----------
    data : bytes
        pickled list of ``(packets, octets)`` tuples, as produced by
        :func:`chunker`
    x_val : str
        x axis value, 'length' or 'size'
    seed : int, optional
        seed of the sampling random generator
    methods : list of str
        'first', 'threshold' and/or 'sampling'
    ps : array_like
        reciprocals of the detection thresholds or, for 'sampling', the
        sampling probabilities
    r : int
        simulation round number (returned for results indexing)
    pps : float, optional
        assumed packets-per-second rate of each flow, enables residency
        accounting: every flow is assumed to occupy the flow table for
        ``floor(packets / pps) + 1 + timeout`` seconds, a detected flow
        for ``floor(covered_packets / pps) + 1 + timeout`` seconds
    timeout : float, default 15
        inactive flow table timeout in seconds

    Returns
    -------
    list of tuple
        ``(method, p, r, result_u, result_f)`` for every method and p, where
        ``result_u`` holds the integer counters ``[flows_all, packets_all,
        octets_all, flows_added, packets_covered]`` and ``result_f`` holds
        the float accumulators ``[fraction_covered, octets_covered]``
        followed by ``[residency_all, residency_covered]`` when ``pps`` is
        given
    """
    data = np.asarray(pickle.loads(data), dtype='u8')
    packets = data[:, 0].astype('f8')
    octets = data[:, 1].astype('f8')
    flows_all = len(data)
    packets_all = packets.sum()
    octets_all = octets.sum()
    residency_all = (np.floor(packets / pps) + 1 + timeout).sum() if pps is not None else 0.0
    packet_size = octets / packets if x_val == 'size' else None
    rng = np.random.default_rng(seed)

    results = []
    for method in methods:
        for p in ps:
            if method == 'sampling':
                p_ret = p
            else:
                p_ret = int(round(1 / p)) * (64 if x_val == 'size' else 1)
            if method == 'first':
                t = int(round(1 / p))
                detected = packets > t if x_val == 'length' else octets > 64 * t
                covered = np.where(detected, packets, 0.0)
            elif method == 'threshold':
                t = int(round(1 / p))
                add_on = np.full(flows_all, t, dtype='f8') if x_val == 'length' \
                    else 64 * t / packet_size
                detected = add_on < packets
                covered = np.where(detected, packets - add_on, 0.0)
            else:
                p_scaled = p if x_val == 'length' else 1 - (1 - p) ** (packet_size / 64)
                g = rng.geometric(p_scaled, size=flows_all) - 1
                detected = g < packets
                covered = np.where(detected, packets - np.minimum(g, packets), 0.0)
            fraction = covered / packets
            if pps is not None:
                residency_covered = np.where(
                    detected, np.floor(covered / pps) + 1 + timeout, 0.0).sum()
            else:
                residency_covered = 0.0
            result_u = np.array([flows_all, packets_all, octets_all,
                                 detected.sum(), covered.sum()], dtype='u8')
            result_f = np.array([fraction.sum(), (octets * fraction).sum()] +
                                ([residency_all, residency_covered] if pps is not None else []),
                                dtype='f8')
            results.append((method, p_ret, r, result_u, result_f))
    return results

def simulate(obj, size=1, x_val='length', seed=None, methods=tuple(METHODS), rounds=5, affinity=False,
             pps=None, timeout=15, max_x=None):
    """
    Simulate flow table occupancy reduction curve for given traffic coverages.

    Parameters
    ----------
    obj : os.PathLike
        csv_hist histogram file or mixture directory
    size : int, default 1
        number of flows to generate
    x_val : str, default 'length'
        x axis value
    seed : int, optional
        seed for the random generator
    methods : list, ['first', 'threshold', 'sampling']
        methods to calculate
    rounds : int, default 5
        number of repetitions of simulation on generated flows
    affinity: bool, default False
        set per-cpu affinity for all processes
    pps : float, optional
        assumed packets-per-second rate of each flow, makes the ``occupancy``
        metric residency-based
    timeout : float, default 15
        inactive flow table timeout in seconds for the residency-based
        ``occupancy`` metric

    Returns
    -------
    dict[str, pd.DataFrame]
        simulated DataFrame for each method, indexed by detection threshold
        (``x = 1/p``) or, for 'sampling', by sampling probability.  Columns
        are ``{flows, packets, fraction, octets, operations,
        occupancy}_{mean,conf}``: percentages of flows added, of packets and
        octets transmitted after insertion (covered), the mean per-flow
        covered packet fraction, and the reductions ``operations = 1/flows``
        and ``occupancy``.  Without ``pps``, ``occupancy = 100/fraction`` is
        a time-free bound which assumes a flow's table residency is
        proportional to its covered packet count and models neither flow
        duration nor inactivity timeout.  When ``pps`` is given,
        ``occupancy`` becomes the ratio of total to covered residency-second
        estimates ``floor(packets / pps) + 1 + timeout``, directly
        comparable to the ``flows_slots`` ratio of
        :func:`flow_models.elephants.simulate_data.simulate_data` run in
        pps mode
    """

    data = next(iter(load_data([obj]).values()))

    x = 1 / np.power(2, range(25))

    running = [0]
    done = collections.defaultdict(int)
    results = {}
    lock = threading.Lock()

    def cb(future):
        with lock:
            running[0] -= 1
            try:
                for mm, pp, rr, result_u, result_f in future.result():
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
                for cn, chunk in enumerate(chunker(generate_flows(data, size, x_val, seed + r if isinstance(seed, int) else seed, max_x))):
                    chunk_seed = seed + r + cn * rounds if isinstance(seed, int) else seed
                    fut = executor.submit(simulate_chunk, chunk, x_val, chunk_seed, methods, x, r, pps, timeout)
                    fut.add_done_callback(cb)
                    with lock:
                        running[0] += 1
                        queued = running[0]
                        if affinity and queued == 1 and not done:
                            set_affinity(executor)
                    if queued >= 16384:
                        logmsg('Queued chunks', queued,
                               'Done flows', {k: int(v / len(x) / rounds) for k, v in done.items()})
                        time.sleep(1)
        except KeyboardInterrupt:
            pass

        try:
            while running[0] > 0:
                logmsg('Remaining chunks', running[0],
                       'Done flows', {k: int(v / len(x) / rounds) for k, v in done.items()})
                time.sleep(1)
        except KeyboardInterrupt:
            os.kill(0, signal.SIGTERM)

    z = pd.DataFrame({k: (*v[0], *v[1]) for k, v in results.items()})
    z = z.transpose()
    cols = ['flows_all', 'packets_all', 'octets_all', 'flows_added',
            'packets_covered', 'fraction_covered', 'octets_covered']
    if pps is not None:
        cols += ['residency_all', 'residency_covered']
    z.columns = cols.copy()
    z['flows'] = 100 * z['flows_added'] / z['flows_all']
    z['packets'] = 100 * z['packets_covered'] / z['packets_all']
    z['fraction'] = 100 * z['fraction_covered'] / z['flows_all']
    z['octets'] = 100 * z['octets_covered'] / z['octets_all']
    z['operations'] = 100 / z['flows']
    if pps is not None:
        z['occupancy'] = z['residency_all'] / z['residency_covered']
    else:
        z['occupancy'] = 100 / z['fraction']
    z = z.replace(np.inf, np.nan)
    z = z.drop(columns=cols)

    df = z.groupby(level=[0, 1]).agg([np.nanmean, conf]).rename(columns={'nanmean': 'mean'})
    df.columns = ['_'.join(col).strip() for col in df.columns]
    dd = dict(iter(df.groupby(level=0)))
    for k, df in dd.items():
        df = df.droplevel(0, 0)
        if k == 'sampling':
            df.sort_index(ascending=False, inplace=True)  # noqa: PD002
        else:
            df.index = df.index.astype('uint64')
        dd[k] = df

    return dd

def simulate_real(directory, x_val, seed=None, methods=tuple(METHODS), rounds=3,
                  timeout=15, fps=REAL_FPS, st=REAL_ST, et=REAL_ET, stride=1):
    """
    Real-times occupancy metrics on sorted binary flow records.

    Follows the protocol of the RF elephant-detection experiments (see
    :mod:`flow_models.elephants.skl.train_classifiers`):
    :func:`flow_models.elephants.simulate_data.simulate_data` runs over all
    flow records with ``fps``-paced uniform arrivals and real flow
    durations; traffic coverage and occupancy reduction are the means of
    the per-second ratios over the ``[st:et]`` window, i.e.
    ``mean(flow_entries_all[t] / flow_entries_elephants[t])``.

    Detection delay for the delayed methods is modeled with the uniform
    intra-flow rate assumption: ``threshold`` inserts at the moment the
    threshold count of packets/bytes elapsed, ``sampling`` at its first
    sampled unit (``64 * geometric(p)`` bytes for ``x_val='size'``).

    Parameters
    ----------
    directory : os.PathLike
        sorted binary flow records directory
    x_val : str
        x axis value, 'length' or 'size'
    seed : int, optional
        seed of the sampling random generator
    methods : list of str
        'first', 'threshold' and/or 'sampling'
    rounds : int, default 3
        repetitions; only ``sampling`` varies across rounds, ``first`` and
        ``threshold`` are deterministic and run once
    timeout : float, default 15
        inactive flow table timeout in seconds
    fps : float, default REAL_FPS
        flows-per-second arrival pace
    st, et : int
        measurement window in seconds
    stride : int, default 1
        subsample every ``stride``-th record — records are sorted by start
        time, so a strided subset preserves the per-second coverage and
        occupancy ratios while making very large captures tractable;
        ``fps`` still describes the real arrival rate (pacing is applied
        to the subsampled positions)

    Returns
    -------
    dict[str, pd.DataFrame]
        DataFrame for each method, indexed by detection threshold (or, for
        'sampling', by sampling probability) with columns
        ``octets_mean`` (average per-second traffic coverage [%]),
        ``occupancy_mean`` and ``occupancy_conf`` (average real-times
        occupancy reduction [x] and its 95% confidence interval)
    """
    from flow_models.elephants.simulate_data import simulate_data
    from flow_models.lib.io import load_array_np

    directory = pathlib.Path(directory)
    arrays = {n: np.array(load_array_np(directory / n)[2][::stride])
              for n in ('packets', 'octets', 'first', 'last')}
    packets, octets = arrays['packets'], arrays['octets']
    elapsed = (arrays['last'].astype('i8') - arrays['first'] + 1)
    units = octets if x_val == 'size' else packets
    rng = np.random.default_rng(seed)

    _, _, fta, oca = simulate_data(arrays, fps=fps, timeout=timeout, max_seconds=et)
    fta, oca = fta[st:et], oca[st:et]
    m_c = oca > 0

    resdic = {}
    for method in methods:
        rs, covs, reds = [], [], []
        for r in range(rounds if method == 'sampling' else 1):
            for p in 1 / np.power(2, range(25)):
                if method == 'sampling':
                    p_ret = p
                    g = rng.geometric(p, size=len(units)) * (64 if x_val == 'size' else 1)
                    mask = g <= units
                    delay = np.floor(elapsed * np.minimum(g / units, 1.0))
                else:
                    t = int(round(1 / p)) * (64 if x_val == 'size' else 1)
                    p_ret = t
                    mask = units > t
                    delay = np.floor(elapsed * np.minimum(t / units, 1.0)) \
                        if method == 'threshold' else 0
                _, _, ftr, ocr = simulate_data(arrays, mask=mask, delay=delay,
                                               fps=fps, timeout=timeout, max_seconds=et)
                ftr, ocr = ftr[st:et], ocr[st:et]
                m = (ftr > 0) & (fta > 0)
                if m.any():
                    rs.append(p_ret)
                    covs.append(100 * (ocr[m_c] / oca[m_c]).mean())
                    reds.append((fta[m] / ftr[m]).mean())
        df = pd.DataFrame({'octets_mean': covs, 'occupancy_mean': reds},
                          index=pd.Index(rs, name=None)).groupby(level=0).agg(
                              {'octets_mean': 'mean',
                               'occupancy_mean': ['mean', conf]})
        df.columns = ['octets_mean', 'occupancy_mean', 'occupancy_conf']
        if df['occupancy_conf'].isna().all():
            df = df.drop(columns='occupancy_conf')  # deterministic method, no variance
        else:
            df['occupancy_conf'] = df['occupancy_conf'].fillna(0)
        if method == 'sampling':
            df.sort_index(ascending=False, inplace=True)  # noqa: PD002
        else:
            df.index = df.index.astype('uint64')
        resdic[method] = df
    return resdic

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-n', type=int, default=1000000, help='number of generated flows')
    p.add_argument('-r', type=int, default=10, help='rounds')
    p.add_argument('--seed', type=int, default=None, help='seed')
    p.add_argument('-x', default='length', choices=X_VALUES, help='x axis value')
    p.add_argument('-m', default='all', choices=[*METHODS, 'all'], help='method')
    p.add_argument('-p', '--pps', type=float, default=None,
                   help='packets per second per flow, makes the occupancy metric residency-based')
    p.add_argument('-t', '--timeout', type=float, default=15, help='inactive flow table timeout in seconds')
    p.add_argument('--save', action='store_true', help='save to files')
    p.add_argument('--affinity', action='store_true', help='set affinity')
    p.add_argument('--real', metavar='DIR',
                   help='sorted flow records directory: measure the ``real`` (real-times) '
                        'occupancy metric instead of generating flows')
    p.add_argument('--real-fps', type=float, default=REAL_FPS,
                   help='flows-per-second arrival pace for --real')
    p.add_argument('--real-stride', type=int, default=1,
                   help='subsample every Nth flow record for --real')
    p.add_argument('--max-x', type=float, default=None,
                   help='truncate generated x quantity at this value (caps unbounded mixture tails)')
    p.add_argument('file', nargs='?', help='csv_hist file or mixture directory')
    return p

def main():
    app_args = parser().parse_args()

    if app_args.m == 'all':
        methods = METHODS
    else:
        methods = [app_args.m]

    if app_args.save:
        pathlib.Path(f'{app_args.m}.mode').write_text(json.dumps({k: v for k, v in vars(app_args).items() if k != 'file'}) + '\n')

    if app_args.real:
        resdic = simulate_real(app_args.real, app_args.x, app_args.seed, methods,
                               app_args.r, app_args.timeout, app_args.real_fps,
                               stride=app_args.real_stride)
    else:
        assert app_args.file, 'csv_hist file or mixture directory required'
        resdic = simulate(app_args.file, app_args.n, app_args.x, app_args.seed, methods,
                          app_args.r, app_args.affinity, app_args.pps, app_args.timeout,
                          app_args.max_x)
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
