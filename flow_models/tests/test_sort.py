import array
import hashlib

import flow_models.convert
import flow_models.sort

IN_FILES = 'data/agh_2015061019_IPv4_anon/sorted'

def test_sort():
    flow_models.sort.sort([IN_FILES], IN_FILES + '_test_sort', ['first', 'first_ms'], in_format='binary', out_format='binary')

    result = array.array('Q')
    flow_models.convert.convert([IN_FILES + '_test_sort'], result, in_format='binary', out_format='extend')
    assert len(result) == 6517484 * 21
    assert hashlib.md5(result).hexdigest() == '13d9b8b7117ee6b0ab37ce14ff90075e'

    flow_models.sort.sort([IN_FILES], IN_FILES + '_test_sort', ['first', 'first_ms'], in_format='binary', out_format='binary', reverse=True)

    result = array.array('Q')
    flow_models.convert.convert([IN_FILES + '_test_sort'], result, in_format='binary', out_format='extend')
    assert len(result) == 6517484 * 21
    assert hashlib.md5(result).hexdigest() == '21174932c7a7a67f4e74d7a752389ae0'

    flow_models.sort.sort([IN_FILES], IN_FILES + '_test_sort', ['last', 'last_ms'], in_format='binary', out_format='binary')

    result = array.array('Q')
    flow_models.convert.convert([IN_FILES + '_test_sort'], result, in_format='binary', out_format='extend')
    assert len(result) == 6517484 * 21
    assert hashlib.md5(result).hexdigest() == '7a1684a451735e26dc79da496d774e89'


if __name__ == '__main__':
    test_sort()
