import contextlib
import sys
import threading
import time

start_time = time.time()

def logmsg(*msg):
    print(f'{time.time() - start_time:.2f}', *msg, file=sys.stderr)

def bin_calc_one(x, _):
    """
    Calculate bin size of 1 width.

    Parameters
    ----------
    x : int
        value
    _ : int
        not used

    Returns
    -------
    (int, int)
        bin_lo, bin_hi
    """
    return x, x + 1

def bin_calc_log(x, b):
    """
    Calculate logarithmic bin size.

    Parameters
    ----------
    x : int
        value
    b : int
        bin width exponent of 2

    Returns
    -------
    (int, int)
        bin_lo, bin_hi
    """
    bin_width = 1 << max(0, int(x).bit_length() - b)
    bin_lo = (x // bin_width) * bin_width
    bin_hi = bin_lo + bin_width
    return bin_lo, bin_hi

@contextlib.contextmanager
def measure_memory(on=False):
    """
    Measure and print minimum, average and maximum memory usage.

    Parameters
    ----------
    on : default False
        run measurement
    """

    memstats = []
    running = True
    thread = None
    if on:
        def collect():
            while running:
                memstats.extend(int(line.split()[1]) for line in open('/proc/self/status') if 'RssAnon' in line)
                time.sleep(0.1)

        thread = threading.Thread(target=collect, daemon=True)
        thread.start()
    try:
        yield
    finally:
        if on:
            running = False
            thread.join()
            print('Memory min/avg/max:', min(memstats), sum(memstats) / len(memstats), max(memstats))
