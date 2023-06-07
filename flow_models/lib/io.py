import argparse
import array
import io
import mmap
import pathlib
import subprocess
import sys
import warnings

class ZeroArray:
    def __getitem__(self, item):
        return 0

class Flows:
    fields = {
        'af': 'B',
        'prot': 'B',
        'inif': 'H',
        'outif': 'H',
        'sa0': 'I',
        'sa1': 'I',
        'sa2': 'I',
        'sa3': 'I',
        'da0': 'I',
        'da1': 'I',
        'da2': 'I',
        'da3': 'I',
        'sp': 'H',
        'dp': 'H',
        'first': 'I',
        'first_ms': 'H',
        'last': 'I',
        'last_ms': 'H',
        'packets': 'Q',
        'octets': 'Q',
        'aggs': 'I'
    }

    __slots__ = ['af', 'prot', 'inif', 'outif', 'sa0', 'sa1', 'sa2', 'sa3', 'da0', 'da1', 'da2', 'da3', 'sp', 'dp',
                 'first', 'first_ms', 'last', 'last_ms', 'packets', 'octets', 'aggs']

def flow_to_csv_line(flow):
    return f'{str(flow)[1:-1]}'

def flow_append(flow, fields):
    fields.af.append(flow[0])
    fields.prot.append(flow[1])
    fields.inif.append(flow[2])
    fields.outif.append(flow[3])
    fields.sa0.append(flow[4])
    fields.sa1.append(flow[5])
    fields.sa2.append(flow[6])
    fields.sa3.append(flow[7])
    fields.da0.append(flow[8])
    fields.da1.append(flow[9])
    fields.da2.append(flow[10])
    fields.da3.append(flow[11])
    fields.sp.append(flow[12])
    fields.dp.append(flow[13])
    fields.first.append(flow[14])
    fields.first_ms.append(flow[15])
    fields.last.append(flow[16])
    fields.last_ms.append(flow[17])
    fields.packets.append(flow[18])
    fields.octets.append(flow[19])
    fields.aggs.append(flow[20])

def read_flow_csv(in_file, counters=None, filter_expr=None, fields=None):
    """
    Read and yield all flows in a csv_flow file/stream.

    :param os.PathLike | _io.IOWrapper in_file: csv_flow file or stream to read
    :param counters: {'skip_in': int, 'count_in': int, 'skip_out': int, 'count_out': int}
    :param filter_expr: filter expression
    :param fields: read only these fields, other can be zeros

    :return: af, prot, inif, outif, sa0, sa1, sa2, sa3, da0, da1, da2, da3, sp, dp, first, first_ms, last, last_ms, packets, octets, aggs
    :rtype: (int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int)
    """

    if counters is None:
        counters = {'skip_in': 0, 'count_in': None, 'skip_out': 0, 'count_out': None}

    if isinstance(in_file, io.IOBase):
        stream = in_file
    else:
        stream = open(str(in_file), 'r')

    for line in stream:
        if counters['skip_in'] > 0:
            counters['skip_in'] -= 1
        else:
            if counters['count_in'] is not None:
                if counters['count_in'] > 0:
                    counters['count_in'] -= 1
                else:
                    break
            af, prot, inif, outif, \
            sa0, sa1, sa2, sa3, \
            da0, da1, da2, da3, \
            sp, dp, first, first_ms, last, last_ms, \
            packets, octets, aggs = line.split(',')
            if filter_expr is None or eval(filter_expr):
                if counters['skip_out'] > 0:
                    counters['skip_out'] -= 1
                else:
                    if counters['count_out'] is not None:
                        if counters['count_out'] > 0:
                            counters['count_out'] -= 1
                        else:
                            break
                    yield int(af), int(prot), int(inif), int(outif), \
                          int(sa0), int(sa1), int(sa2), int(sa3), \
                          int(da0), int(da1), int(da2), int(da3), \
                          int(sp), int(dp), \
                          int(first), int(first_ms), int(last), int(last_ms), int(packets), int(octets), int(aggs)

    if not isinstance(in_file, io.IOBase):
        stream.close()

def read_pipe(in_file, counters=None, filter_expr=None, fields=None):
    """
    Read and yield all flows in a nfdump pipe file/stream.

    This function calls nfdump program to parse nfdump file.

    :param os.PathLike | _io.IOWrapper in_file: nfdump pipe file or stream to read
    :param counters: {'skip_in': int, 'count_in': int, 'skip_out': int, 'count_out': int}
    :param filter_expr: filter expression
    :param fields: read only these fields, other can be zeros

    :return: af, prot, inif, outif, sa0, sa1, sa2, sa3, da0, da1, da2, da3, sp, dp, first, first_ms, last, last_ms, packets, octets, aggs
    :rtype: (int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int)
    """

    if counters is None:
        counters = {'skip_in': 0, 'count_in': None, 'skip_out': 0, 'count_out': None}

    if isinstance(in_file, io.IOBase):
        stream = in_file
    else:
        stream = open(str(in_file), 'r')

    for line in stream:
        if counters['skip_in'] > 0:
            counters['skip_in'] -= 1
        else:
            if counters['count_in'] is not None:
                if counters['count_in'] > 0:
                    counters['count_in'] -= 1
                else:
                    break
            af, first, last, prot, \
            sa0, sa1, sa2, sa3, sp, da0, da1, da2, da3, dp, \
            srcas, dstas, inif, outif, \
            tcp_flags, tos, packets, octets = line.split(b'|')
            first, first_ms = first[:-3], first[-3:]
            last, last_ms = last[:-3], last[-3:]
            if filter_expr is None or eval(filter_expr):
                if counters['skip_out'] > 0:
                    counters['skip_out'] -= 1
                else:
                    if counters['count_out'] is not None:
                        if counters['count_out'] > 0:
                            counters['count_out'] -= 1
                        else:
                            break
                    yield int(af), int(prot), int(inif), int(outif), \
                          int(sa0), int(sa1), int(sa2), int(sa3), \
                          int(da0), int(da1), int(da2), int(da3), \
                          int(sp), int(dp), \
                          int(first), int(first_ms), int(last), int(last_ms), int(packets), int(octets), 0

    if not isinstance(in_file, io.IOBase):
        stream.close()

def read_nfcapd(in_file, counters=None, filter_expr=None, fields=None):
    """
    Read and yield all flows in a nfdump nfpcapd file.

    This function calls nfdump program to parse nfpcapd file.

    :param os.PathLike in_file: nfdump nfpcapd file to read
    :param counters: {'skip_in': int, 'count_in': int, 'skip_out': int, 'count_out': int}
    :param filter_expr: filter expression
    :param fields: read only these key fields, other can be zeros

    :return: af, prot, inif, outif, sa0, sa1, sa2, sa3, da0, da1, da2, da3, sp, dp, first, first_ms, last, last_ms, packets, octets, aggs
    :rtype: (int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int)
    """

    nfdump_process = subprocess.Popen(['nfdump', '-r', str(in_file), '-q', '-o', 'pipe'], stdout=subprocess.PIPE)
    stream = nfdump_process.stdout

    yield from read_pipe(stream, counters, filter_expr, fields)

    if nfdump_process is not None:
        rc = nfdump_process.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(nfdump_process.returncode, nfdump_process.args)

def read_flow_binary(in_dir, counters=None, filter_expr=None, fields=None):
    """
    Read and yield all flows in a directory containing array files.

    :param in_dir: directory to read from
    :param counters: {'skip_in': int, 'count_in': int, 'skip_out': int, 'count_out': int}
    :param filter_expr: filter expression
    :param fields: read only these key fields, other can be zeros

    :return: key, first, first_ms, last, last_ms, packets, octets, aggs
    :rtype: (tuple, int, int, int, int, int, int, int)
    """

    if counters is None:
        counters = {'skip_in': 0, 'count_in': None, 'skip_out': 0, 'count_out': None}

    path = pathlib.Path(in_dir)
    assert path.exists() and path.is_dir()
    flows = Flows()
    fields_to_load = [name for name in flows.fields if fields is None or name in fields]

    arrays, filtered, size = load_arrays(path, fields_to_load, counters, filter_expr)
    for name in flows.fields:
        if name in arrays:
            setattr(flows, name, arrays[name])
        else:
            setattr(flows, name, ZeroArray())

    for n in range(size):
        if filtered is ... or filtered[n]:
            if counters['skip_out'] > 0:
                counters['skip_out'] -= 1
            else:
                if counters['count_out'] is not None:
                    if counters['count_out'] > 0:
                        counters['count_out'] -= 1
                    else:
                        break
                yield flows.af[n], flows.prot[n], flows.inif[n], flows.outif[n], \
                      flows.sa0[n], flows.sa1[n], flows.sa2[n], flows.sa3[n], \
                      flows.da0[n], flows.da1[n], flows.da2[n], flows.da3[n], \
                      flows.sp[n], flows.dp[n], \
                      flows.first[n], flows.first_ms[n], flows.last[n], flows.last_ms[n], \
                      flows.packets[n], flows.octets[n], flows.aggs[n]

def write_none(_):
    while True:
        _ = yield

def write_line(output, header_line=None):
    if isinstance(output, io.IOBase):
        stream = output
    else:
        stream = open(str(output), 'w')
    if header_line:
        print(header_line, file=stream)
    try:
        while True:
            line = yield
            print(line, file=stream)
    except GeneratorExit:
        pass
    if not isinstance(output, io.IOBase):
        stream.close()

def write_flow_csv(output):
    if isinstance(output, io.IOBase):
        stream = output
    else:
        stream = open(str(output), 'w')
    try:
        while True:
            flow = yield
            print(flow_to_csv_line(flow), file=stream)
    except GeneratorExit:
        pass
    if not isinstance(output, io.IOBase):
        stream.close()

def write_flow_binary(output_dir):
    d = pathlib.Path(output_dir)
    d.mkdir(parents=True, exist_ok=True)
    assert d.is_dir()
    fields = Flows()
    files = {}
    for name, typecode in fields.fields.items():
        setattr(fields, name, array.array(typecode))
        files[name] = open(str(d / f'{name}.{typecode}'), 'wb')
    n = 0
    try:
        while True:
            flow = yield
            flow_append(flow, fields)
            n += 1
            if n == 32768:
                dump_binary(fields, files)
                n = 0
    except GeneratorExit:
        pass
    dump_binary(fields, files)
    for f in files.values():
        f.close()

def dump_binary(fields, files):
    for name in fields.fields:
        arr = getattr(fields, name)
        arr.tofile(files[name])
        del arr[:]

def find_array_path(path):
    p = pathlib.Path(path)
    if not p.suffix:
        candidates = list(p.parent.glob(f'{p.stem}.*'))
        if not candidates:
            raise FileNotFoundError(0, p.parent, f'{p.stem}.*')
        elif len(candidates) > 1:
            raise FileExistsError('More than one file matching to pattern: ', candidates)
        else:
            p = candidates[0]
    name, dtype = p.stem, p.suffix[1:]
    return name, dtype, p

def load_array_mv(path, mode='r'):
    name, dtype, path = find_array_path(path)
    flags = mmap.MAP_SHARED if mode == 'c' else mmap.MAP_SHARED
    prot = mmap.PROT_READ
    if mode != 'r':
        prot |= mmap.PROT_WRITE
    with open(str(path)) as f:
        mm = mmap.mmap(f.fileno(), 0, flags=flags, prot=prot)
    mv = memoryview(mm).cast(dtype)
    return name, dtype, mv

def load_array_np(path, mode='r'):
    import numpy as np
    name, dtype, path = find_array_path(path)
    mm = np.memmap(str(path), dtype=dtype, mode=mode)
    return name, dtype, mm

def load_arrays(path, fields, counters, filter_expr):
    ars = {}
    fields_to_load = set(fields) | set(filter_expr.co_names if filter_expr else ())
    size = None
    for name in fields_to_load:
        try:
            name, dtype, ar = load_array_np(path / name, 'r')
            if size is None:
                size = ar.size
            else:
                assert ar.size == size
            if counters['skip_in'] > 0:
                ar = ar[counters['skip_in']:]
            if counters['count_in'] is not None and counters['count_in'] > 0:
                ar = ar[:counters['count_in']]
            ars[name] = ar
        except FileNotFoundError:
            warnings.warn(f"Array file for flow field '{name}' not found in directory {path}."
                          " Assuming zero as value of this field")
            ars[name] = ZeroArray()

    if size is not None:
        if counters['skip_in'] > 0:
            size = max(size - counters['skip_in'], 0)
            counters['skip_in'] -= min(counters['skip_in'], size)
        if counters['count_in'] is not None and counters['count_in'] > 0:
            size = min(size, counters['count_in'])
            counters['count_in'] -= min(counters['count_in'], size)

    for ar in ars.values():
        if not isinstance(ar, ZeroArray):
            assert len(ar) == size

    filtered = ...
    if filter_expr is not None:
        for name in filter_expr.co_names:
            if isinstance(ars[name], ZeroArray):
                raise ValueError(f"Filter is using flow field '{name}' which is not present in input files")
        filtered = eval(filter_expr, ars)

    arrays = {}
    for name in fields:
        arrays[name] = ars[name]

    return arrays, filtered, size

def prepare_file_list(file_paths):
    files = []
    for file_path in file_paths:
        if file_path == '-':
            files.append(sys.stdin)
        else:
            path = pathlib.Path(file_path)
            if not path.exists():
                raise ValueError(f'File {path} does not exist')
            if path.is_dir():
                for path in sorted(path.glob('**/*')):
                    if path.is_file():
                        files.append(path)
            else:
                if path.is_file():
                    files.append(path)
                else:
                    raise ValueError(f'File {path} is not file')
    return files

class IOArgumentParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=True)
        self.add_argument('in_files', nargs='+', help='input files or directories')
        self.add_argument('-i', '--in-format', default='nfcapd', choices=IN_FORMATS, help='format of input files')
        self.add_argument('-o', '--out-format', default='csv_flow', choices=OUT_FORMATS, help='format of output')
        self.add_argument('-O', '--output', default=sys.stdout, help='file or directory for output')
        self.add_argument('--skip-in', type=int, default=0, help='number of flows to skip at the beginning of input')
        self.add_argument('--count-in', type=int, default=None, help='number of flows to read from input')
        self.add_argument('--skip-out', type=int, default=0, help='number of flows to skip after filtering')
        self.add_argument('--count-out', type=int, default=None, help='number of flows to output after filtering')
        self.add_argument('--filter-expr', default=None, help='expression of filter')

    def parse_args(self, *args):
        namespace = super().parse_args(*args)
        if namespace.in_format != 'binary':
            namespace.in_files = prepare_file_list(namespace.in_files)
        if not isinstance(namespace.output, io.IOBase):
            namespace.output = pathlib.Path(namespace.output)
        if namespace.filter_expr:
            namespace.filter_expr = compile(namespace.filter_expr, f'FILTER: {namespace.filter_expr}', 'eval')
        return namespace

IN_FORMATS = {'csv_flow': read_flow_csv, 'pipe': read_pipe, 'nfcapd': read_nfcapd, 'binary': read_flow_binary}
OUT_FORMATS = {'csv_flow': write_flow_csv, 'binary': write_flow_binary, 'none': write_none}
