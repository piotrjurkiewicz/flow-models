import hashlib

import flow_models.hist
import flow_models.hist_np

IN_FILES = 'data/agh_2015061019_IPv4_anon/sorted'

def test_hist():
    result = []
    flow_models.hist.hist([IN_FILES], result, in_format='binary', out_format='append', x_value='length')
    assert hashlib.md5(''.join(result).encode()).hexdigest() == '36252b49cb9d34a01d071f97361326be'

def test_hist_np():
    result = []
    flow_models.hist_np.hist([IN_FILES], result, in_format='binary', out_format='append', x_value='length')
    assert hashlib.md5(''.join(result).encode()).hexdigest() == '36252b49cb9d34a01d071f97361326be'


if __name__ == '__main__':
    test_hist()
    test_hist_np()
