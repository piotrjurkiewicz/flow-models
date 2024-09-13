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
    p.add_argument('--pps', type=float, default=None, help='packets per second, when absent flow records times are used for calculation')
    p.add_argument('--fps', type=float, default=None, help='flows per second, when absent flow records times are used for calculation')
    p.add_argument('--timeout', type=float, default=15.0, help='inactive flow table timeout in seconds')
    p.add_argument('--max-seconds', type=int, default=3600, help='total seconds number of simulation')
    p.add_argument('--threshold', type=int, default=10000000, help='elephant flow size threshold')
    p.add_argument('directory', help='binary flow records directory')
    return p

def simulate_data(directory, index=Ellipsis, mask=None, pps=None, fps=None, timeout=15, max_seconds=3600):
    """
    Simulate flow table occupancy reduction curve for given flow records.

    Parameters
    ----------
    directory : os.PathLike
        binary flow records directory
    index : np.array, default Ellipsis
        index array of flows to use for simulating data
    mask : np.array[bool], optional
        flows to add to flow table (elephants)
    pps : float, optional
        packets per second, when None flow records times are used for calculation
    fps : float, optional
        flows per second, when None flow records times are used for calculation
    timeout : float, default 15.0
        inactive flow table timeout in seconds
    max_seconds: int, default 3600
        total seconds number of simulation

    Returns
    -------
    flows_sum : int
        sum of flows added to flow table
    octets_sum : int
        sum of octets transmitted by flows while being in flow table
    flows_slots : np.array
        number of flows present in flow table in each second
    octets_slots : np.array
        amount of octets trasmitted by flows in flow table in each second
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

def main():
    app_args = parser().parse_args()

    pps = app_args.pps
    fps = app_args.fps
    timeout = app_args.timeout
    max_seconds = app_args.max_seconds
    threshold = app_args.threshold

    directory = pathlib.Path(app_args.directory)
    _, _, octets = load_array_np(directory / 'octets')

    for fold, (train_index, test_index) in enumerate(sklearn.model_selection.KFold(5).split(octets)):
        logmsg(f"Fold {fold}")

        all_fl, all_oc, fta, oca = simulate_data(directory, index=test_index, mask=None, pps=pps, fps=fps, timeout=timeout, max_seconds=max_seconds)
        cov_fl, cov_oc, ftr, ocr = simulate_data(directory, index=test_index, mask=octets[test_index] > threshold, pps=pps, fps=fps, timeout=timeout, max_seconds=max_seconds)

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
