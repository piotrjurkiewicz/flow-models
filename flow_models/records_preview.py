"""
Write first N records of a binary flow trace directory as preview.json.

Run in a directory containing {field}.{dtype} files (see docs/formats.rst).
"""

import array
import json
import pathlib

from flow_models.lib.io import CODECS

N = 100

fields = {}
for f in sorted(pathlib.Path().iterdir()):
    name, dot, code = f.name.rpartition('.')
    if not dot or code not in CODECS or not name:
        continue
    a = array.array(code)
    a.frombytes(f.open('rb').read(N * a.itemsize))
    fields[name] = {
        'dtype': code,
        'size': a.itemsize,
        'count': f.stat().st_size // a.itemsize,
        'values': list(a),
    }

preview = {
    'fields': [{'name': n, 'dtype': v['dtype'], 'size': v['size']} for n, v in fields.items()],
    'count': max((v['count'] for v in fields.values()), default=0),
    'records': [list(r) for r in zip(*(v['values'] for v in fields.values()))],
}
pathlib.Path('preview.json').write_text(json.dumps(preview, separators=(',', ':')))
print(f'{len(fields)} fields, {preview["count"]} records, {len(preview["records"])} shown')
