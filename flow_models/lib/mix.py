import json

import numpy as np
import scipy.stats

def rvs(mix, x_val, size=1, random_state=None):
    weights = np.array([mx[0] for mx in mix])
    weights /= weights.sum()  # in case these did not add up to 1
    data = np.zeros((size, len(mix)))
    for n, (weight, dist, args) in enumerate(mix):
        data[:, n] = getattr(scipy.stats, dist).rvs(*args, size=size, random_state=random_state)
    random_n = np.random.choice(np.arange(len(mix)), size=[size], p=weights)
    sample = data[np.arange(size), random_n]
    if x_val in ['length', 'size']:
        sample = np.trunc(sample) + 1
        if x_val == 'size':
            sample[sample < 64] = 64
    return sample

def cdf(mix, x):
    if isinstance(mix, dict):
        mix = mix['mix']
    weights = np.array([mx[0] for mx in mix])
    weights /= weights.sum()  # in case these did not add up to 1
    data = np.zeros(len(x))
    for n, (weight, dist, args) in enumerate(mix):
        data += weights[n] * getattr(scipy.stats, dist).cdf(x, *args)
    return data

def pdf(mix, x):
    data = cdf(mix, x)
    return np.hstack((data[0], np.diff(data) / np.diff(x)))

def cdf_comp(mix, x):
    if isinstance(mix, dict):
        mix = mix['mix']
    weights = np.array([mx[0] for mx in mix])
    weights /= weights.sum()  # in case these did not add up to 1
    data = {}
    for n, (weight, dist, args) in enumerate(mix):
        data[f'{weight} {dist}'] = weights[n] * getattr(scipy.stats, dist).cdf(x, *args)
    return data

def pdf_comp(mix, x):
    cdf_components = cdf_comp(mix, x)
    data = {}
    for k, v in cdf_components.items():
        data[k] = np.hstack((v[0], np.diff(v) / np.diff(x)))
    return data

def to_json(mix, sum=None):
    s = ''
    if sum:
        s += '{\n' + f'  "sum": {sum},\n' + '  "mix": '
    s += '[\n    ' + ',\n    '.join(json.dumps(c, default=lambda v: v.item()) for c in mix) + '\n  ]'
    if sum:
        s += '\n}'
    return s
