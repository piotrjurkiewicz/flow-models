import argparse
import array
import io
import pathlib
import subprocess
import sys
import warnings

import mmap

from .util import logmsg

FILTER_HELP = \
"""
To filter flow records, the filter expressions should be specified. Filter expression should use
the Python syntax. Bitwise (&, |, ~) operators should be used instead logical ones (and, or, not).
The following fields are available:

    af, prot, inif, outif, sa0, sa1, sa2, sa3, da0, da1, da2, da3, sp, dp,
    first, first_ms, last, last_ms, packets, octets, aggs
"""

class ZeroArray:
    def __getitem__(self, item):
        return 0
    def item(self, _):
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
        'aggs': 'I',
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

def flow_to_int(flow):
    return int(flow[0]), int(flow[1]), int(flow[2]), int(flow[3]), \
           int(flow[4]), int(flow[5]), int(flow[6]), int(flow[7]), \
           int(flow[8]), int(flow[9]), int(flow[10]), int(flow[11]), \
           int(flow[12]), int(flow[13]), int(flow[14]), int(flow[15]), int(flow[16]), int(flow[17]), \
           int(flow[18]), int(flow[19]), int(flow[20])

def read_flow_csv(in_file, counters=None, filter_expr=None, fields=None):  # noqa: ARG001
    """
    Read and yield all flows in a csv_flow file/stream.

    Parameters
    ----------
    in_file : os.PathLike | io.TextIOWrapper
        csv_flow file or stream to read
    counters: dict[str, int], default
        skip_in : int, default 0
            number of flows to skip at the beginning of input
        count_in : int, default None, meaning all flows
            number of flows to read from input
        skip_out : int, default 0
            number of flows to skip after filtering
        count_out : int, default None, meaning all flows
            number of flows to output after filtering
    filter_expr : CodeType, optional
        filter expression
    fields : list[str], optional
        read only these fields, other can be zeros

    Returns
    -------
    (int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int)
        af, prot, inif, outif, sa0, sa1, sa2, sa3, da0, da1, da2, da3, sp, dp, first, first_ms, last, last_ms, packets, octets, aggs
    """

    if counters is None:
        counters = {'skip_in': 0, 'count_in': None, 'skip_out': 0, 'count_out': None}

    if isinstance(in_file, io.IOBase):
        stream = in_file
    else:
        stream = open(str(in_file))

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
            packets, octets, aggs = flow_to_int(line.split(','))
            if filter_expr is None or eval(filter_expr):
                if counters['skip_out'] > 0:
                    counters['skip_out'] -= 1
                else:
                    if counters['count_out'] is not None:
                        if counters['count_out'] > 0:
                            counters['count_out'] -= 1
                        else:
                            break
                    yield af, prot, inif, outif, \
                          sa0, sa1, sa2, sa3, \
                          da0, da1, da2, da3, \
                          sp, dp, first, first_ms, last, last_ms, \
                          packets, octets, aggs

    if not isinstance(in_file, io.IOBase):
        stream.close()

def read_pipe(in_file, counters=None, filter_expr=None, fields=None):  # noqa: ARG001
    """
    Read and yield all flows in a nfdump pipe file/stream.

    This function calls nfdump program to parse nfdump file.

    Parameters
    ----------
    in_file : os.PathLike | io.TextIOWrapper
        nfdump pipe file or stream to read
    counters: dict[str, int], default
        skip_in : int, default 0
            number of flows to skip at the beginning of input
        count_in : int, default None, meaning all flows
            number of flows to read from input
        skip_out : int, default 0
            number of flows to skip after filtering
        count_out : int, default None, meaning all flows
            number of flows to output after filtering
    filter_expr : CodeType, optional
        filter expression
    fields : list[str], optional
        read only these fields, other can be zeros

    Returns
    -------
    (int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int)
        af, prot, inif, outif, sa0, sa1, sa2, sa3, da0, da1, da2, da3, sp, dp, first, first_ms, last, last_ms, packets, octets, aggs
    """

    if counters is None:
        counters = {'skip_in': 0, 'count_in': None, 'skip_out': 0, 'count_out': None}

    if isinstance(in_file, io.IOBase):
        stream = in_file
    else:
        stream = open(str(in_file))

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
            first, first_ms = int(first[:-3]), int(first[-3:])
            last, last_ms = int(last[:-3]), int(last[-3:])
            af, prot, inif, outif, \
            sa0, sa1, sa2, sa3, \
            da0, da1, da2, da3, \
            sp, dp, packets, octets, aggs = int(af), int(prot), int(inif), int(outif), \
                                            int(sa0), int(sa1), int(sa2), int(sa3), \
                                            int(da0), int(da1), int(da2), int(da3), \
                                            int(sp), int(dp), \
                                            int(packets), int(octets), 0
            if filter_expr is None or eval(filter_expr):
                if counters['skip_out'] > 0:
                    counters['skip_out'] -= 1
                else:
                    if counters['count_out'] is not None:
                        if counters['count_out'] > 0:
                            counters['count_out'] -= 1
                        else:
                            break
                    yield af, prot, inif, outif, \
                          sa0, sa1, sa2, sa3, \
                          da0, da1, da2, da3, \
                          sp, dp, first, first_ms, last, last_ms, \
                          packets, octets, aggs

    if not isinstance(in_file, io.IOBase):
        stream.close()

def read_nfcapd(in_file, counters=None, filter_expr=None, fields=None):
    """
    Read and yield all flows in a nfdump nfpcapd file.

    This function calls nfdump program to parse nfpcapd file.

    Parameters
    ----------
    in_file : os.PathLike
        nfdump nfpcapd file to read
    counters: dict[str, int], default
        skip_in : int, default 0
            number of flows to skip at the beginning of input
        count_in : int, default None, meaning all flows
            number of flows to read from input
        skip_out : int, default 0
            number of flows to skip after filtering
        count_out : int, default None, meaning all flows
            number of flows to output after filtering
    filter_expr : CodeType, optional
        filter expression
    fields : list[str], optional
        read only these fields, other can be zeros

    Returns
    -------
    (int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int)
        af, prot, inif, outif, sa0, sa1, sa2, sa3, da0, da1, da2, da3, sp, dp, first, first_ms, last, last_ms, packets, octets, aggs
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

    Parameters
    ----------
    in_dir : os.PathLike
        directory to read from
    counters: dict[str, int], default
        skip_in : int, default 0
            number of flows to skip at the beginning of input
        count_in : int, default None, meaning all flows
            number of flows to read from input
        skip_out : int, default 0
            number of flows to skip after filtering
        count_out : int, default None, meaning all flows
            number of flows to output after filtering
    filter_expr : CodeType, optional
        filter expression
    fields : list[str], optional
        read only these fields, other can be zeros

    Returns
    -------
    (int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int)
        af, prot, inif, outif, sa0, sa1, sa2, sa3, da0, da1, da2, da3, sp, dp, first, first_ms, last, last_ms, packets, octets, aggs
    """

    if counters is None:
        counters = {'skip_in': 0, 'count_in': None, 'skip_out': 0, 'count_out': None}

    path = pathlib.Path(in_dir)
    assert path.exists()
    assert path.is_dir()
    flows = Flows()
    fields_to_load = [name for name in flows.fields if fields is None or name in fields]
    use_numpy = True

    arrays, filtered, size = load_arrays(path, fields_to_load, counters, filter_expr)
    if not size:
        return
    for name in flows.fields:
        if name in arrays:
            setattr(flows, name, arrays[name])
            if isinstance(arrays[name], memoryview):
                use_numpy = False
        else:
            setattr(flows, name, ZeroArray())

    if use_numpy:
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
                    yield flows.af.item(n), flows.prot.item(n), flows.inif.item(n), flows.outif.item(n), \
                          flows.sa0.item(n), flows.sa1.item(n), flows.sa2.item(n), flows.sa3.item(n), \
                          flows.da0.item(n), flows.da1.item(n), flows.da2.item(n), flows.da3.item(n), \
                          flows.sp.item(n), flows.dp.item(n), \
                          flows.first.item(n), flows.first_ms.item(n), flows.last.item(n), flows.last_ms.item(n), \
                          flows.packets.item(n), flows.octets.item(n), flows.aggs.item(n)
    else:
        for n in range(size):
            af = flows.af[n]
            prot = flows.prot[n]
            inif = flows.inif[n]
            outif = flows.outif[n]
            sa0 = flows.sa0[n]
            sa1 = flows.sa1[n]
            sa2 = flows.sa2[n]
            sa3 = flows.sa3[n]
            da0 = flows.da0[n]
            da1 = flows.da1[n]
            da2 = flows.da2[n]
            da3 = flows.da3[n]
            sp = flows.sp[n]
            dp = flows.dp[n]
            first = flows.first[n]
            first_ms = flows.first_ms[n]
            last = flows.last[n]
            last_ms = flows.last_ms[n]
            packets = flows.packets[n]
            octets = flows.octets[n]
            aggs = flows.aggs[n]
            if filter_expr is None or eval(filter_expr):
                if counters['skip_out'] > 0:
                    counters['skip_out'] -= 1
                else:
                    if counters['count_out'] is not None:
                        if counters['count_out'] > 0:
                            counters['count_out'] -= 1
                        else:
                            break
                    yield af, prot, inif, outif, \
                          sa0, sa1, sa2, sa3, \
                          da0, da1, da2, da3, \
                          sp, dp, \
                          first, first_ms, last, last_ms, packets, octets, aggs
def write_none(_):
    while True:
        _ = yield

def write_append(output, *_):
    while True:
        flow = yield
        output.append(flow)

def write_extend(output, *_):
    while True:
        flow = yield
        output.extend(flow)

def write_line(output, header_line=None):
    """
    Write lines to the output.

    Parameters
    ----------
    output : os.PathLike | io.TextIOWrapper
        file to write or stream
    header_line: str, optional
        header line to write at the beggining of the file
    """

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
    """
    Write flow tuples to the output.

    Parameters
    ----------
    output : os.PathLike | io.TextIOWrapper
        file to write or stream
    """

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
    """
    Write flow tuples to binary flow files.

    Parameters
    ----------
    output_dir : os.PathLike
        directory to write to
    """

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
    """
    Find an exact numpy array path and type.

    Parameters
    ----------
    path : os.PathLike
        array file path

    Returns
    -------
    (str, str, pathlib.Path)
        name, dtype, path
    """

    path = pathlib.Path(path)
    if not path.suffix:
        candidates = list(path.parent.glob(f'{path.stem}.*'))
        if not candidates:
            raise FileNotFoundError(0, path.parent, f'{path.stem}.*')
        elif len(candidates) > 1:
            raise FileExistsError('More than one file matching to pattern: ', candidates)
        else:
            path = candidates[0]
    name, dtype = path.stem, path.suffix[1:]
    return name, dtype, path

def load_array_mv(path, mode='r'):
    """
    Load a numpy array as memoryview.

    Parameters
    ----------
    path : os.PathLike
        array file path
    mode : str, default 'r'
        file open mode
    """

    name, dtype, path = find_array_path(path)
    path = pathlib.Path(path)
    flags = mmap.MAP_SHARED if mode == 'c' else mmap.MAP_SHARED
    prot = mmap.PROT_READ
    if mode != 'r':
        prot |= mmap.PROT_WRITE
    if path.stat().st_size > 0:
        with open(str(path)) as f:
            mm = mmap.mmap(f.fileno(), 0, flags=flags, prot=prot)
        mv = memoryview(mm).cast(dtype)
    else:
        mv = memoryview(b'').cast(dtype)
    return name, dtype, mv

def load_array_np(path, mode='r'):
    """
    Load a numpy array as numpy.memmap.

    Parameters
    ----------
    path : os.PathLike
        array file path
    mode : str, default 'r'
        file open mode
    """

    import numpy as np
    name, dtype, path = find_array_path(path)
    mm = np.memmap(str(path), dtype=dtype, mode=mode)
    return name, dtype, mm

def load_arrays(path, fields, counters, filter_expr, require_numpy=False):
    """
    Load all binary flow arrays from a directory.

    Parameters
    ----------
    path : os.PathLike
        directory path
    fields : list[str]
        fields to load
    counters: dict[str, int], default
        skip_in : int, default 0
            number of flows to skip at the beginning of input
        count_in : int, default None, meaning all flows
            number of flows to read from input
        skip_out : int, default 0
            not supported
        count_out : int, default None, meaning all flows
            not supported
    filter_expr : CodeType, optional
            filter expression
    require_numpy : bool, default False
            require to load arrays as numpy arrays

    Returns
    -------
    (dict[str, memoryview | numpy.array], numpy.array, int)
        arrays, filtered, size
    """

    use_numpy = False
    if require_numpy or filter_expr is not None:
        try:
            import numpy as np
            use_numpy = True
        except ImportError:
            if require_numpy:
                raise

    ars = {}
    fields_to_load = set(fields) | set(filter_expr.co_names if filter_expr else ())
    size = None
    for name in fields_to_load:
        try:
            if use_numpy:
                name, dtype, ar = load_array_np(path / name, 'r')
            else:
                name, dtype, ar = load_array_mv(path / name, 'r')
            if size is None:
                size = len(ar)
            else:
                assert len(ar) == size
            if counters['skip_in'] > 0:
                ar = ar[counters['skip_in']:]
            if counters['count_in'] is not None:
                if counters['count_in'] > 0:
                    ar = ar[:counters['count_in']]
                else:
                    ar = ar[0:0]
            ars[name] = ar
        except FileNotFoundError:  # noqa: PERF203
            warnings.warn(f"Array file for flow field '{name}' not found in directory {path}."
                          " Assuming zero as value of this field")
            ars[name] = ZeroArray()

    if size is not None:
        if counters['skip_in'] > 0:
            size = max(size - counters['skip_in'], 0)
            counters['skip_in'] -= min(counters['skip_in'], size)
        if counters['count_in'] is not None:
            if counters['count_in'] > 0:
                size = min(size, counters['count_in'])
                counters['count_in'] -= min(counters['count_in'], size)
            else:
                size = 0

    for ar in ars.values():
        if not isinstance(ar, ZeroArray):
            assert len(ar) == size

    filtered = ...
    if use_numpy and filter_expr is not None:
        for name in filter_expr.co_names:
            if isinstance(ars[name], ZeroArray):
                raise ValueError(f"Filter is using flow field '{name}' which is not present in input files")
        logmsg(f"Starting filtering: {filter_expr.co_filename}")
        filtered = eval(filter_expr, ars)
        filtered_count = np.count_nonzero(filtered)
        logmsg(f"Finished filtering: {filtered_count}/{size} ({100 * filtered_count/size if size else 0} %)")

    arrays = {}
    for name in (fields if use_numpy else fields_to_load):
        arrays[name] = ars[name]

    return arrays, filtered, size

def prepare_file_list(file_paths):
    """
    Prepare files list from file list or directory.

    Parameters
    ----------
    file_paths : list[str | os.PathLike | io.IOBase]
        list of file paths

    Returns
    -------
    list[pathlib.Path | io.IOBase]
        prepared file list
    """

    files = []
    for file_path in file_paths:
        if isinstance(file_path, io.IOBase):
            files.append(file_path)
        elif file_path == '-':
            files.append(sys.stdin)
        else:
            path = pathlib.Path(file_path)
            if not path.exists():
                raise ValueError(f'File {path} does not exist')
            if path.is_dir():
                for subpath in sorted(path.glob('**/*')):
                    if subpath.is_file():
                        files.append(subpath)
            else:
                if path.is_file():
                    files.append(path)
                else:
                    raise ValueError(f'File {path} is not file')
    return files

class Formatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

class IOArgumentParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, formatter_class=Formatter, add_help=True)
        self.add_argument('in_files', nargs='+', help='input files or directories')
        self.add_argument('-i', '--in-format', default='nfcapd', choices=IN_FORMATS, help='format of input files')
        self.add_argument('-o', '--out-format', default='csv_flow', choices=OUT_FORMATS, help='format of output')
        self.add_argument('-O', '--output', default='-', help='file or directory for output')
        self.add_argument('--skip-in', type=int, default=0, help='number of flows to skip at the beginning of input')
        self.add_argument('--count-in', type=int, default=None, help='limit for number of flows to read from input')
        self.add_argument('--skip-out', type=int, default=0, help='number of flows to skip after filtering')
        self.add_argument('--count-out', type=int, default=None, help='limit for number of flows to output after filtering')
        self.add_argument('--filter-expr', default=None, help='expression of filter')

    def parse_args(self, *args):
        namespace = super().parse_args(*args)
        if namespace.in_format != 'binary':
            namespace.in_files = prepare_file_list(namespace.in_files)
        if not isinstance(namespace.output, io.IOBase):
            if namespace.output == '-':
                namespace.output = sys.stdout
            else:
                namespace.output = pathlib.Path(namespace.output)
        if 'filter_expr' in namespace and namespace.filter_expr:
            namespace.filter_expr = compile(namespace.filter_expr, f'FILTER: {namespace.filter_expr}', 'eval')
        return namespace

IN_FORMATS = {'csv_flow': read_flow_csv, 'pipe': read_pipe, 'nfcapd': read_nfcapd, 'binary': read_flow_binary}
OUT_FORMATS = {'csv_flow': write_flow_csv, 'binary': write_flow_binary, 'append': write_append, 'extend': write_extend, 'none': write_none}
