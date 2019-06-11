import contextlib
import sys
import threading
import time

def logmsg(*msg):
    print(f'{time.time() - start_ts:.2f}', *msg, file=sys.stderr)

def bin_calc_one(x, _):
    return x, x + 1

def bin_calc_log(x, b):
    bin_width = 1 << max(0, int(x).bit_length() - b)
    bin_lo = (x // bin_width) * bin_width
    bin_hi = bin_lo + bin_width
    return bin_lo, bin_hi

start_ts = time.time()

@contextlib.contextmanager
def measure_memory(on=False):
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