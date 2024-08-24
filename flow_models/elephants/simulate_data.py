#!/usr/bin/python3
"""Simulates flow table for elephant flow classification using flow records data directory."""

import argparse
import pathlib

import numpy as np
import sklearn.model_selection

from flow_models.lib.io import load_array_np
from flow_models.lib.util import logmsg


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-O', '--output', default='sklearn', help='output directory and plot file name')
    p.add_argument('--seed', type=int, default=None, help='seed')
    p.add_argument('--mixture', help='')
    p.add_argument('--fork', action='store_true', help='')
    p.add_argument('files', help='directory')
    return p

def simulate_data(directory, index=Ellipsis, mask=None, pps=None, fps=None, timeout=15, max_seconds=3600):
    """
    Simulate flow table occupancy reduction curve for given traffic coverages.

    Parameters
    ----------
    directory : os.PathLike
        binary flow records directory
    index : np.array, default Ellipsis
        index array of flows to use for simulating data
    mask : np.array[bool], default None
        flows to add to flow table (elephants)
    pps : float, default None
        packets per second, when None flow records times are used for calculation
    fps : float, default None
        flows per second, when None flow records times are used for calculation
    timeout : float, default 15.0
        inactive flow table timeout in seconds
    max_seconds: int, default 3600
        total seconds number of simulation

    Returns
    -------
    int, int, np.array, np.array
        flows_sum, octets_sum, flows_slots, octets_slots
    """

    d = pathlib.Path(directory)
    _, _, packets = load_array_np(d / 'packets')
    _, _, octets = load_array_np(d / 'octets')
    packets = packets[index]
    octets = octets[index]

    if fps is None or pps is None:
        _, _, first = load_array_np(d / 'first')
        _, _, last = load_array_np(d / 'last')
        first = first[index]
        last = last[index]
    else:
        first, last = None, None

    if pps is None:
        assert (last >= first[0]).all()
        elapsed = last - first + 1
    else:
        elapsed = np.floor_divide(packets, pps, dtype=np.int64) + 1

    if fps is None:
        assert first[0] == first.min()
        start = first - first[0]
        end = start + elapsed
    else:
        flow_numbers = np.arange(len(packets))
        start = np.floor_divide(flow_numbers, fps)
        end = start + elapsed

    avg_packet_size = octets / packets
    avg_pps = packets / elapsed
    avg_bps = avg_packet_size * avg_pps

    flows_slots = np.zeros(max_seconds, dtype=np.uint64)
    octets_slots = np.zeros(max_seconds, dtype=np.float64)

    if mask is None:
        iterator = range(len(packets))
        flows_sum = len(packets)
        octets_sum = octets.sum()
    else:
        iterator = np.flatnonzero(mask)
        flows_sum = mask.sum()
        octets_sum = octets[mask].sum()

    # max_flow_start = 0
    for n in iterator:
        flow_start = start[n]
        flow_end = end[n]
        flows_slots[flow_start:flow_end + timeout] += 1
        octets_slots[flow_start:flow_end] += avg_bps[n]
        # max_flow_start = max(max_flow_start, flow_start)

    # logmsg(f"Flows number {len(packets)} Max flow start {max_flow_start}")

    return flows_sum, octets_sum, flows_slots, octets_slots

def old(directory, threshold=0, timeout=15, max_seconds=3600):
    d = pathlib.Path(directory)
    _, _, first = load_array_np(d / 'first')
    _, _, last = load_array_np(d / 'last')
    _, _, packets = load_array_np(d / 'packets')
    _, _, octets = load_array_np(d / 'octets')
    # first = first[FPS * 3600 * 2:FPS * 3600 * 22]
    # last = last[FPS * 3600 * 2:FPS * 3600 * 22]
    # packets = packets[FPS * 3600 * 2:FPS * 3600 * 22]
    # octets = octets[FPS * 3600 * 2:FPS * 3600 * 22]
    avg_packet_size = octets / packets
    # elapsed = np.floor_divide(packets, PPS)
    elapsed = last - first + 1
    avg_pps = packets / elapsed
    avg_bps = avg_packet_size * avg_pps

    assert first[0] == first.min()
    assert (last >= first[0]).all()

    start = first - first[0]
    end = start + elapsed

    max_seconds = end.max() + timeout

    fta = np.zeros(max_seconds, dtype=np.uint64)
    ftr = np.zeros(max_seconds, dtype=np.uint64)
    oca = np.zeros(max_seconds, dtype=np.float64)
    ocr = np.zeros(max_seconds, dtype=np.float64)
    all_fl = 0
    cov_fl = 0
    all_oc = 0
    cov_oc = 0

    for n in range(len(packets)):
        flow_size = octets[n]
        # flow_avg_packet_size = avg_packet_size[n]
        # flow_avg_pps = avg_pps[n]
        all_fl += 1
        all_oc += flow_size
        # start = n // FPS
        # end = start + int(elapsed[n]) + 1
        flow_start = start[n]
        flow_end = end[n]
        fta[flow_start:flow_end + timeout] += 1
        oca[flow_start:flow_end] += avg_bps[n]
        if flow_size > threshold:
            cov_fl += 1
            cov_oc += flow_size
            ftr[flow_start:flow_end + timeout] += 1
            ocr[flow_start:flow_end] += avg_bps[n]

    return all_fl, all_oc, fta, oca, cov_fl, cov_oc, ftr, ocr

def main():
    # app_args = parser().parse_args()

    FPS = 1810
    # PPS = 480
    PPS = None
    THRESHOLD = 10000000
    TIMEOUT = 15

    d = pathlib.Path('data/agh_2015061019_IPv4_anon/sorted')
    _, _, octets = load_array_np(d / 'octets')
    # d = pathlib.Path('data/agh_2015/sorted')

    # all_fl, all_oc, fta, oca, cov_fl, cov_oc, ftr, ocr = old(d, threshold=THRESHOLD, timeout=TIMEOUT)

    for fold, (train_index, test_index) in enumerate(sklearn.model_selection.KFold(5).split(octets)):
        logmsg(f"Fold {fold}")

        all_fl, all_oc, fta, oca = simulate_data(d, index=test_index, mask=None, pps=PPS, fps=FPS, timeout=TIMEOUT)
        cov_fl, cov_oc, ftr, ocr = simulate_data(d, index=test_index, mask=octets[test_index] > THRESHOLD, pps=PPS, fps=FPS, timeout=TIMEOUT)

        print(all_fl / cov_fl)
        print(cov_oc / all_oc)
        print()
        print(fta / ftr)
        print(fta.sum() / ftr.sum())
        print((fta / ftr)[300:720].min())
        print((fta / ftr)[300:720].mean())
        print((fta / ftr)[300:720].max())
        print()
        print(ocr / oca)
        print(ocr.sum() / oca.sum())
        print((ocr / oca)[300:720].min())
        print((ocr / oca)[300:720].mean())
        print((ocr / oca)[300:720].max())

    logmsg("Done")


if __name__ == '__main__':
    main()
