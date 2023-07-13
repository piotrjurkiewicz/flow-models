import flow_models.fit

IN_FILES = 'data/agh_2015061019_IPv4_anon/histograms/all/length.csv'

def test_fit():
    result = flow_models.fit.fit(IN_FILES, 'flows', 400, initial={'U': 6, 'L': 4})
    assert result == {'mix': [[0.2667839375411922, 'uniform', [0, 1]], [0.2498173937170916, 'uniform', [0, 2]],
                              [0.07891291919980845, 'uniform', [0, 3]], [0.04500096756101436, 'uniform', [0, 4]],
                              [0.01744045367227924, 'uniform', [0, 5]], [0.08315270229031899, 'uniform', [0, 6]],
                              [0.1332450209694342, 'lognorm', (0.49364076506762783, 0, 7.379776617881778)],
                              [0.07856960684414944, 'lognorm', (0.7294315551184803, 0, 20.034092933439634)],
                              [0.0376312022303655, 'lognorm', (1.1885847327306296, 0, 107.16017390959571)],
                              [0.009445795974345318, 'lognorm', (2.0387101598164863, 0, 1373.9749450320094)]],
                      'sum': 6517484}


if __name__ == '__main__':
    test_fit()
