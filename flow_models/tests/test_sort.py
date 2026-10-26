import array
import hashlib

import flow_models.convert
import flow_models.sort

IN_FILES = 'data/agh_2015061019_IPv4_anon/ALL/sorted'

def test_sort():
    flow_models.sort.sort([IN_FILES], IN_FILES + '_test_sort', ['first', 'first_ms'], in_format='binary', out_format='binary')

    result = array.array('Q')
    flow_models.convert.convert([IN_FILES + '_test_sort'], result, in_format='binary', out_format='extend')
    assert len(result) == 6517484 * 21
    assert hashlib.md5(result).hexdigest() == '5fc79a2478e75d50120a5515dfea7da6'

    flow_models.sort.sort([IN_FILES], IN_FILES + '_test_sort', ['first', 'first_ms'], in_format='binary', out_format='binary', reverse=True)

    result = array.array('Q')
    flow_models.convert.convert([IN_FILES + '_test_sort'], result, in_format='binary', out_format='extend')
    assert len(result) == 6517484 * 21
    assert hashlib.md5(result).hexdigest() == 'f288ba1072cb1343fd9811c89521bc21'

    flow_models.sort.sort([IN_FILES], IN_FILES + '_test_sort', ['last', 'last_ms'], in_format='binary', out_format='binary')

    result = array.array('Q')
    flow_models.convert.convert([IN_FILES + '_test_sort'], result, in_format='binary', out_format='extend')
    assert len(result) == 6517484 * 21
    assert hashlib.md5(result).hexdigest() == '6fc5a707ac5b9d57b52bf9e847d95da5'


if __name__ == '__main__':
    test_sort()
