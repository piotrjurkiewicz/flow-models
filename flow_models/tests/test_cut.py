import array
import hashlib
import pathlib

import flow_models.convert
import flow_models.cut
import flow_models.tests.test_convert

IN_FILES = 'data/agh_2015061019_IPv4_anon/sorted'

def test_cut():
    flow_models.cut.cut([IN_FILES], IN_FILES + '_4096_cut', in_format='binary', out_format='binary', count_in=4096)

    result = array.array('Q')
    flow_models.convert.convert([IN_FILES + '_4096_cut'], result, in_format='binary', out_format='extend')
    assert len(result) == 4096 * 21
    assert hashlib.md5(result).hexdigest() == '24d26df2a70d2533c82661bab440e5b2'

    matrix = {
        'skip_in': [-1, 0, 1, 4097],
        'count_in': [-1, 0, None, 1, 4097],
        'skip_out': [-1, 0, 1, 4097],
        'count_out': [-1, 0, None, 1, 4097]
    }

    for params in flow_models.tests.test_convert.product_dict(**matrix):
        for f in pathlib.Path(IN_FILES + '_4096_cut_test').glob('*'):
            f.unlink()
        flow_models.cut.cut([IN_FILES + '_4096_cut'], IN_FILES + '_4096_cut_test', in_format='binary', out_format='binary', **params)
        result = array.array('Q')
        flow_models.convert.convert([IN_FILES + '_4096_cut_test'], result, in_format='binary', out_format='extend')
        if params['skip_in'] > 4096 or params['skip_out'] > 4096 or params['count_in'] in (-1, 0) or params['count_out'] in (-1, 0):
            assert len(result) == 0
        else:
            ha = hashlib.md5(result).hexdigest()[:16]
            assert params in flow_models.tests.test_convert.RESULTS['all'][ha]


if __name__ == '__main__':
    test_cut()
