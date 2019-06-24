import json

import numpy as np
import scipy.stats

def rvs(mix, size, x_val):
    weights = np.array([mx[0] for mx in mix])
    weights /= weights.sum()  # in case these did not add up to 1
    data = np.zeros((size, len(mix)))
    for n, (weight, distr) in enumerate(mix):
        data[:, n] = distr.rvs(size=size)
    random_n = np.random.choice(np.arange(len(mix)), size=[size], p=weights)
    sample = data[np.arange(size), random_n]
    if x_val in ['length', 'size']:
        sample = np.trunc(sample) + 1
        if x_val == 'size':
            sample[sample < 64] = 64
    return sample

def cdf(mix, x):
    weights = np.array([mx[0] for mx in mix])
    weights /= weights.sum()  # in case these did not add up to 1
    data = np.zeros(len(x))
    for n, (weight, distr) in enumerate(mix):
        data += weights[n] * distr.cdf(x)
    return data

def pdf(mix, x):
    data = cdf(mix, x)
    return np.hstack((data[0], np.diff(data) / np.diff(x)))

def cdf_components(mix, x):
    weights = np.array([mx[0] for mx in mix])
    weights /= weights.sum()  # in case these did not add up to 1
    data = {}
    for n, (weight, distr) in enumerate(mix):
        data[str(weight) + ' ' + distr.dist.name] = weights[n] * distr.cdf(x)
    return data

def pdf_components(mix, x):
    ccc = cdf_components(mix, x)
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
