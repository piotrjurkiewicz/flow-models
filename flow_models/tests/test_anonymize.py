import flow_models.anonymize
import flow_models.convert

IN_FILES = 'data/agh_2015061019_IPv4_anon/sorted'
KEY = "boojahyoo3vaeToong0Eijee7Ahz3yee"

def test_anonymize():
    result = []
    flow_models.convert.convert([IN_FILES], result, in_format='binary', out_format='append', count_in=3)
    assert len(result) == 3

    result_anon = []
    flow_models.anonymize.anonymize([IN_FILES], result_anon, in_format='binary', out_format='append', count_in=3, key=KEY)
    assert len(result_anon) == 3

    for n in range(3):
        for p in range(21):
            if p in [7, 11]:
                assert result[n][p] != result_anon[n][p]
            else:
                assert result[n][p] == result_anon[n][p]


if __name__ == '__main__':
    test_anonymize()
