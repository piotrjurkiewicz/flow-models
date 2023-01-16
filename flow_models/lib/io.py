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

class FlowKeyFields:
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
        'dp': 'H'
    }

    __slots__ = ['af', 'prot', 'inif', 'outif', 'sa0', 'sa1', 'sa2', 'sa3', 'da0', 'da1', 'da2', 'da3', 'sp', 'dp']

class FlowValFields:
    fields = {
        'first': 'I',
        'first_ms': 'H',
        'last': 'I',
        'last_ms': 'H',
        'packets': 'Q',
        'octets': 'Q',
        'aggs': 'I'
    }

    __slots__ = ['first', 'first_ms', 'last', 'last_ms', 'packets', 'octets', 'aggs']

def flow_to_csv_line(flow):
    return f'{str(flow[0])[1:-1]}, {str(flow[1:])[1:-1]}'

def flow_append(flow, keys, vals):
    keys.af.append(flow[0][0])
    keys.prot.append(flow[0][1])
    keys.inif.append(flow[0][2])
    keys.outif.append(flow[0][3])
    keys.sa0.append(flow[0][4])
    keys.sa1.append(flow[0][5])
    keys.sa2.append(flow[0][6])
    keys.sa3.append(flow[0][7])
    keys.da0.append(flow[0][8])
    keys.da1.append(flow[0][9])
    keys.da2.append(flow[0][10])
    keys.da3.append(flow[0][11])
    keys.sp.append(flow[0][12])
    keys.dp.append(flow[0][13])
    vals.first.append(flow[1])
    vals.first_ms.append(flow[2])
    vals.last.append(flow[3])
    vals.last_ms.append(flow[4])
    vals.packets.append(flow[5])
    vals.octets.append(flow[6])
    vals.aggs.append(flow[7])

def read_flow_csv(in_file, counters=None, filter_expr=None, key_fields=None, val_fields=None):
    """
    Read and yield all flows in a csv_flow file/stream.

    :param os.PathLike | _io.IOWrapper in_file: csv_flow file or stream to read
    :param counters: {'count': int, 'skip_input': int, 'skip_output': int}
    :param filter_expr: filter expression
    :param key_fields: read only these key fields, other can be zeros
    :param val_fields: read only these val fields, other can be zeros

    :return: key, first, first_ms, last, last_ms, packets, octets, aggs
    :rtype: (tuple, int, int, int, int, int, int, int)
    """

    if counters is None:
        counters = {'count': None, 'skip_input': 0, 'skip_output': 0}

    if isinstance(in_file, io.IOBase):
        stream = in_file
    else:
        stream = open(str(in_file), 'r')

    if key_fields is None or key_fields:
        key = None
    else:
        key = ()

    for line in stream:
        if counters['skip_input'] > 0:
            counters['skip_input'] -= 1
        else:
            af, prot, inif, outif, \
                sa0, sa1, sa2, sa3, \
                da0, da1, da2, da3, \
                sp, dp, first, first_ms, last, last_ms, \
                packets, octets, aggs = line.split(',')
            if filter_expr is None or eval(filter_expr):
                if counters['skip_output'] > 0:
                    counters['skip_output'] -= 1
                else:
                    if counters['count'] is not None:
                        if counters['count'] > 0:
                            counters['count'] -= 1
                        else:
                            break
                    if key is None:
                        key = (int(af), int(prot), int(inif), int(outif),
                               int(sa0), int(sa1), int(sa2), int(sa3),
                               int(da0), int(da1), int(da2), int(da3),
                               int(sp), int(dp))
                    yield key, int(first), int(first_ms), int(last), int(last_ms), int(packets), int(octets), int(aggs)

    if not isinstance(in_file, io.IOBase):
        stream.close()

def read_pipe(in_file, counters=None, filter_expr=None, key_fields=None, val_fields=None):
    """
    Read and yield all flows in a nfdump pipe file/stream.

    This function calls nfdump program to parse nfdump file.

    :param os.PathLike | _io.IOWrapper in_file: nfdump pipe file or stream to read
    :param counters: {'count': int, 'skip_input': int, 'skip_output': int}
    :param filter_expr: filter expression
    :param key_fields: read only these key fields, other can be zeros
    :param val_fields: read only these val fields, other can be zeros

    :return: key, first, first_ms, last, last_ms, packets, octets, aggs
    :rtype: (tuple, int, int, int, int, int, int, int)
    """

    if counters is None:
        counters = {'count': None, 'skip_input': 0, 'skip_output': 0}

    if isinstance(in_file, io.IOBase):
        stream = in_file
    else:
        stream = open(str(in_file), 'r')

    if key_fields is None or key_fields:
        key = None
    else:
        key = ()

    for line in stream:
        if counters['skip_input'] > 0:
            counters['skip_input'] -= 1
        else:
            af, first, first_ms, last, last_ms, prot, \
                sa0, sa1, sa2, sa3, sp, da0, da1, da2, da3, dp, \
                srcas, dstas, inif, outif, \
                tcp_flags, tos, packets, octets = line.split(b'|')
            if filter_expr is None or eval(filter_expr):
                if counters['skip_output'] > 0:
                    counters['skip_output'] -= 1
                else:
                    if counters['count'] is not None:
                        if counters['count'] > 0:
                            counters['count'] -= 1
                        else:
                            break
                    if key is None:
                        key = (int(af), int(prot), int(inif), int(outif),
                               int(sa0), int(sa1), int(sa2), int(sa3),
                               int(da0), int(da1), int(da2), int(da3),
                               int(sp), int(dp))
                    yield key, int(first), int(first_ms), int(last), int(last_ms), int(packets), int(octets), 0

    if not isinstance(in_file, io.IOBase):
        stream.close()

def read_nfcapd(in_file, counters=None, filter_expr=None, key_fields=None, val_fields=None):
    """
    Read and yield all flows in a nfdump nfpcapd file.

    This function calls nfdump program to parse nfpcapd file.

    :param os.PathLike in_file: nfdump nfpcapd file to read
    :param counters: {'count': int, 'skip_input': int, 'skip_output': int}
    :param filter_expr: filter expression
    :param key_fields: read only these key fields, other can be zeros
    :param val_fields: read only these val fields, other can be zeros

    :return: key, first, first_ms, last, last_ms, packets, octets, aggs
    :rtype: (tuple, int, int, int, int, int, int, int)
    """

    nfdump_process = subprocess.Popen(['nfdump', '-r', str(in_file), '-q', '-o', 'pipe'], stdout=subprocess.PIPE)
    stream = nfdump_process.stdout

    yield from read_pipe(stream, counters, filter_expr, key_fields, val_fields)

    if nfdump_process is not None:
        rc = nfdump_process.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(nfdump_process.returncode, nfdump_process.args)

def read_flow_binary(in_dir, counters=None, filter_expr=None, key_fields=None, val_fields=None):
    """
    Read and yield all flows in a directory containing array files.

    :param in_dir: directory to read from
    :param counters: {'count': int, 'skip_input': int, 'skip_output': int}
    :param filter_expr: filter expression
    :param key_fields: read only these key fields, other can be zeros
    :param val_fields: read only these val fields, other can be zeros

    :return: key, first, first_ms, last, last_ms, packets, octets, aggs
    :rtype: (tuple, int, int, int, int, int, int, int)
    """

    if counters is None:
        counters = {'count': None, 'skip_input': 0, 'skip_output': 0}

    d = pathlib.Path(in_dir)
    assert d.exists() and d.is_dir()
    keys = FlowKeyFields()
    vals = FlowValFields()
    size = None
    for mvs, interesting_fields in [(keys, key_fields), (vals, val_fields)]:
        for name in mvs.fields.keys():
            if interesting_fields is None or name in interesting_fields:
                try:
                    _, dtype, mv = load_array_mv(d / name)
                    if size is None:
                        size = len(mv)
                    else:
                        assert len(mv) == size
                    if counters['skip_input'] > 0:
                        mv = mv[counters['skip_input']:]
                    setattr(mvs, name, mv)
                except FileNotFoundError:
                    warnings.warn(f"Array file for flow field '{name}' not found in directory {d}."
                                  " Assuming None as value of this field")
                    setattr(mvs, name, ZeroArray())
            else:
                setattr(mvs, name, ZeroArray())

    # TODO: lazy reading of fields
    arrays = {}
    filtered = ZeroArray()
    if filter_expr is not None:
        import numpy as np
        for name in keys.fields:
            mv = getattr(keys, name)
            if not isinstance(mv, ZeroArray):
                arrays[name] = np.asarray(mv)
        for name in vals.fields:
            mv = getattr(vals, name)
            if not isinstance(mv, ZeroArray):
                arrays[name] = np.asarray(mv)
        filtered = eval(filter_expr, arrays)

    if all(isinstance(getattr(keys, name), ZeroArray) for name in keys.fields):
        key = ()
    else:
        key = None

    if counters['skip_input'] > 0:
        size = max(size - counters['skip_input'], 0)
        counters['skip_input'] -= min(counters['skip_input'], size)

    for n in range(size):
        if filter_expr is None or filtered[n]:
            if counters['skip_output'] > 0:
                counters['skip_output'] -= 1
            else:
                if counters['count'] is not None:
                    if counters['count'] > 0:
                        counters['count'] -= 1
                    else:
                        break
                if key is None:
                    key = (keys.af[n], keys.prot[n], keys.inif[n], keys.outif[n],
                           keys.sa0[n], keys.sa1[n], keys.sa2[n], keys.sa3[n],
                           keys.da0[n], keys.da1[n], keys.da2[n], keys.da3[n],
                           keys.sp[n], keys.dp[n])
                yield key, vals.first[n], vals.first_ms[n], vals.last[n], vals.last_ms[n], \
                           vals.packets[n], vals.octets[n], vals.aggs[n]

    try:
        del arrays
        for name in keys.fields:
            mv = getattr(keys, name)
            if not isinstance(mv, ZeroArray):
                obj = mv.obj
                mv.release()
                obj.close()
        for name in vals.fields:
            mv = getattr(vals, name)
            if not isinstance(mv, ZeroArray):
                obj = mv.obj
                mv.release()
                obj.close()
    except AttributeError:
        # Bug in PyPy: https://bitbucket.org/pypy/pypy/issues/3016/attributeerror-memoryview-object-has-no
        pass

def write_none(_):
    while True:
        _ = yield

def write_line(out_file, header_line=None):
    if isinstance(out_file, io.IOBase):
        stream = out_file
    else:
        stream = open(str(out_file), 'w')
    if header_line:
        print(header_line, file=stream)
    try:
        while True:
            line = yield
            print(line, file=stream)
    except GeneratorExit:
        pass
    if not isinstance(out_file, io.IOBase):
        stream.close()

def write_flow_csv(out_file):
    if isinstance(out_file, io.IOBase):
        stream = out_file
    else:
        stream = open(str(out_file), 'w')
    try:
        while True:
            flow = yield
            print(flow_to_csv_line(flow), file=stream)
    except GeneratorExit:
        pass
    if not isinstance(out_file, io.IOBase):
        stream.close()

def write_flow_binary(out_dir):
    d = pathlib.Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    assert d.is_dir()
    keys = FlowKeyFields()
    vals = FlowValFields()
    files = {}
    for name, typecode in keys.fields.items():
        setattr(keys, name, array.array(typecode))
        files[name] = open(str(d / f'_{name}.{typecode}'), 'wb')
    for name, typecode in vals.fields.items():
        setattr(vals, name, array.array(typecode))
        files[name] = open(str(d / f'{name}.{typecode}'), 'wb')
    n = 0
    try:
        while True:
            flow = yield
            flow_append(flow, keys, vals)
            n += 1
            if n == 32768:
                dump_binary(keys, vals, files)
                n = 0
    except GeneratorExit:
        pass
    dump_binary(keys, vals, files)
    for f in files.values():
        f.close()

def dump_binary(keys, vals, files):
    for name in keys.fields:
        arr = getattr(keys, name)
        arr.tofile(files[name])
        del arr[:]
    for name in vals.fields:
        arr = getattr(vals, name)
        arr.tofile(files[name])
        del arr[:]

def find_array_path(path):
    p = pathlib.Path(path)
    if not p.suffix:
        candidates = list(p.parent.glob(f'{p.stem}.*')) + list(p.parent.glob(f'_{p.stem}.*'))
        if not candidates:
            raise FileNotFoundError(0, p.parent, f'{p.stem}.*')
        elif len(candidates) > 1:
            raise FileExistsError('More than one file matching to pattern: ', candidates)
        else:
            p = candidates[0]
    name, dtype = p.stem, p.suffix[1:]
    return name, dtype, p

def load_array_np(path, mode='r'):
    import numpy as np
    name, dtype, path = find_array_path(path)
    mm = np.memmap(str(path), dtype=dtype, mode=mode)
    return name, dtype, mm

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
        super().__init__(**kwargs, formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
        self.add_argument('in_files', nargs='+', help='input files or directories')
        self.add_argument('-i', '--in-format', default='nfcapd', choices=IN_FORMATS, help='format of input files')
        self.add_argument('-o', '--out-format', default='csv_flow', choices=OUT_FORMATS, help='format of output')
        self.add_argument('-O', '--out-file', default=sys.stdout, help='file or directory for output')
        self.add_argument('--count', type=int, default=None, help='number of flows to output')
        self.add_argument('--skip-input', type=int, default=0, help='number of flows to skip at the beginning of input')
        self.add_argument('--skip-output', type=int, default=0, help='number of flows to skip after filtering')
        self.add_argument('--filter-expr', default=None, help='expression of filter')

    def parse_args(self, *args):
        namespace = super().parse_args(*args)
        if namespace.in_format != 'binary':
            namespace.in_files = prepare_file_list(namespace.in_files)
        if namespace.filter_expr:
            namespace.filter_expr = compile(namespace.filter_expr, '<filter_expr>', 'eval')
        return namespace


IN_FORMATS = {'csv_flow': read_flow_csv, 'pipe': read_pipe, 'nfcapd': read_nfcapd, 'binary': read_flow_binary}
OUT_FORMATS = {'csv_flow': write_flow_csv, 'binary': write_flow_binary, 'none': write_none}
