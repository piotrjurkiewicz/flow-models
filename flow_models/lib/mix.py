import json

import numpy as np
import scipy.stats
import scipy._lib._util

def mask_values(x, x_val):
    # Due to numerical constrains PDF for large x values needs to
    # be calculated using .pdf functions instead CDF difference
    x_min = 1
    if x_val == 'size':
        x_min = 64
    mask = x < 1024
    masked_x = x[mask]
    previous_x = masked_x - 1.0
    previous_x[masked_x == x_min] = 0.0
    return mask, masked_x, previous_x

def rvs(mix, x_val, size=1, random_state=None):
    if isinstance(mix, dict):
        mix = mix['mix']
    weights = np.array([mx[0] for mx in mix])
    weights /= weights.sum()  # in case these did not add up to 1
    data = np.zeros((size, len(mix)))
    for n, (weight, dist, args) in enumerate(mix):
        data[:, n] = getattr(scipy.stats, dist).rvs(*args, size=size, random_state=random_state)
    rng = scipy._lib._util.check_random_state(random_state)
    random_n = rng.choice(np.arange(len(mix)), size=[size], p=weights)
    sample = data[np.arange(size), random_n]
    if x_val in ['length', 'size']:
        sample = sample.astype(int) + 1
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

def pdf(mix, x, x_val):
    if isinstance(mix, dict):
        mix = mix['mix']
    weights = np.array([mx[0] for mx in mix])
    weights /= weights.sum()  # in case these did not add up to 1
    data = np.zeros(len(x))
    mask, masked_x, previous_x = mask_values(x, x_val)
    for n, (weight, name, args) in enumerate(mix):
        dist = getattr(scipy.stats, name)
        data[mask] += weights[n] * (dist.cdf(masked_x, *args) - dist.cdf(previous_x, *args))
        data[~mask] += weights[n] * dist.pdf(x[~mask], *args)
    return data

def cdf_comp(mix, x):
    if isinstance(mix, dict):
        mix = mix['mix']
    weights = np.array([mx[0] for mx in mix])
    weights /= weights.sum()  # in case these did not add up to 1
    data = {}
    for n, (weight, dist, args) in enumerate(mix):
        data[f'{weight} {dist}'] = weights[n] * getattr(scipy.stats, dist).cdf(x, *args)
    return data

def pdf_comp(mix, x, x_val):
    if isinstance(mix, dict):
        mix = mix['mix']
    weights = np.array([mx[0] for mx in mix])
    weights /= weights.sum()  # in case these did not add up to 1
    components = {}
    mask, masked_x, previous_x = mask_values(x, x_val)
    for n, (weight, name, args) in enumerate(mix):
        data = np.zeros(len(x))
        dist = getattr(scipy.stats, name)
        data[mask] = dist.cdf(masked_x, *args) - dist.cdf(previous_x, *args)
        data[~mask] = dist.pdf(x[~mask], *args)
        data *= weights[n]
        components[f'{weight} {name} {args}'] = data
    return components

def avg(data, x, x_val, what):
    if what in ['packets', 'octets']:
        avg_mix = (data[what]['sum'] / data['flows']['sum']) * pdf(data[what], x, x_val) / pdf(data['flows'], x, x_val)
    elif what in ['packet_size', 'packet_iat']:
        if what == 'packet_size':
            num = 'octets'
        else:
            num = 'duration'
        avg_mix = pdf(data[num], x, x_val) / pdf(data['packets'], x, x_val)
        avg_mix *= data[num]['sum'] / data['packets']['sum']
    else:
        raise ValueError

    return avg_mix

def to_json(mix, sum=None):
    s = ''
    if sum:
        s += '{\n' + f'  "sum": {sum},\n' + '  "mix": '
    s += '[\n    ' + ',\n    '.join(json.dumps(c, default=lambda v: v.item()) for c in mix) + '\n  ]'
    if sum:
        s += '\n}'
    return s
