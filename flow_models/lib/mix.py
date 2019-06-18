import json

import numpy as np
import scipy.stats

def mix_rvs(mix, size):
    weights = np.array([mx[0] for mx in mix])
    weights /= weights.sum()  # in case these did not add up to 1
    data = np.zeros((size, len(mix)))
    for n, (weight, distr) in enumerate(mix):
        data[:, n] = distr.rvs(size=size)
    random_n = np.random.choice(np.arange(len(mix)), size=[size], p=weights)
    sample = data[np.arange(size), random_n]
    return sample

def mix_cdf(mix, x):
    weights = np.array([mx[0] for mx in mix])
    weights /= weights.sum()  # in case these did not add up to 1
    data = np.zeros(len(x))
    for n, (weight, distr) in enumerate(mix):
        data += weights[n] * distr.cdf(x)
    return data

def mix_pdf(mix, x):
    data = mix_cdf(mix, x)
    return np.hstack((data[0], np.diff(data) / np.diff(x)))

def mix_cdf_components(mix, x):
    weights = np.array([mx[0] for mx in mix])
    weights /= weights.sum()  # in case these did not add up to 1
    data = {}
    for n, (weight, distr) in enumerate(mix):
        data[str(weight) + ' ' + distr.dist.name] = weights[n] * distr.cdf(x)
    return data

def mix_pdf_components(mix, x):
    ccc = mix_cdf_components(mix, x)
    data = {}
    for k, v in ccc.items():
        data[k] = np.hstack((v[0], np.diff(v) / np.diff(x)))
    return data

def load_mixture(file):
    obj = json.load(open(file))
    mixture = []
    for row in obj['mix']:
        mixture.append((row[0], getattr(scipy.stats, row[1])(*row[2])))
    return obj['sum'], mixture
