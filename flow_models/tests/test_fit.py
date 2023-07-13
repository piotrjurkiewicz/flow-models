import pytest

import flow_models.fit

IN_FILES = 'data/agh_2015061019_IPv4_anon/histograms/all/length.csv'

def test_fit():
    result = flow_models.fit.fit(IN_FILES, 'flows', 400, initial={'U': 6, 'L': 4})
    weights = []
    for ds in result['mix']:
        weights.append(ds[0])
    assert weights == pytest.approx(
        [0.2667839375411922, 0.2498173937170916, 0.07891291919980845, 0.04500096756101436, 0.01744045367227924,
         0.08315270229031899, 0.1332450209694342, 0.07856960684414944, 0.0376312022303655, 0.009445795974345318])
    assert result['sum'] == 6517484


if __name__ == '__main__':
    test_fit()
