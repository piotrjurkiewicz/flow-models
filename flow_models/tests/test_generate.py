import flow_models.generate

IN_FILES = 'data/agh_2015061019_IPv4_anon/histograms/all/length.csv'

def test_generate():
    result = []
    flow_models.generate.generate(IN_FILES, result, out_format='append', size=3, x_value='length', random_state=0)
    assert result == [(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 261, 0),
                      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 709, 0),
                      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 261, 0)]


if __name__ == '__main__':
    test_generate()
