import hashlib
import pathlib

import flow_models.series

IN_FILES = 'data/agh_2015061019_IPv4_anon/sorted'

def test_series():
    flow_models.series.series([IN_FILES], IN_FILES + '_4096_series_test', in_format='binary', out_format='csv_series', count_in=4096)

    for f in pathlib.Path(IN_FILES + '_4096_series_test').glob('*'):
        assert hashlib.md5(f.read_bytes()).hexdigest() == {'16596.octets': 'b618027f22a8cc2d1f5c9f5a48867933',
                                                           '16596.flows': '0a19274d0155cba07289f4090ad851af',
                                                           '16596.packets': 'fd1b58df84b5005b11ea5e0705c3ca8c'}[f.name]


if __name__ == '__main__':
    test_series()
