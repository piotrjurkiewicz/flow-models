import json

import numpy as np
import scipy._lib._util
import scipy.stats


def mask_values(x, x_val):
    # Due to numerical constrains PDF for large x values needs to
    # be calculated using .pdf functions instead CDF difference
    x_min = 1
    if x_val == 'size':
        x_min = 64
    mask = x < 1024
    mu, mui = np.unique(x[mask], return_inverse=True)
    pu = mu - 1.0
    pu[mu == x_min] = 0.0
    nu, nui = np.unique(x[~mask], return_inverse=True)
    return mask, mu, pu, mui, nu, nui

def rvs(mix, x_val, size=1, random_state=None):
    """
    Return a sample from a distribution mixture.

    Parameters
    ----------
    mix : dict | list[list]
        distribution mixture
    x_val: str
        flow feature being sampled
    size : int, default 1
        number of elements in sample
    random_state : object, optional
        random state to use

    Returns
    -------
    numpy.array
        sample
    """

    if isinstance(mix, dict):
        mix = mix['mix']
    weights = np.array([mx[0] for mx in mix])
    weights /= weights.sum()  # in case these did not add up to 1
    sample = np.zeros(size)
    rng = scipy._lib._util.check_random_state(random_state)
    random_n = rng.choice(np.arange(len(mix)), size=size, p=weights)
    for n, (weight, dist, args) in enumerate(mix):
        mask = random_n == n
        data = getattr(scipy.stats, dist).rvs(*args, size=np.count_nonzero(mask), random_state=random_state)
        sample[mask] = data
    if x_val in ['length', 'size']:
        sample = sample.astype('u8') + 1
        if x_val == 'size':
            np.clip(sample, 64, None, out=sample)
    return sample

def cdf(mix, x):
    """
    Return a CDF from a distribution mixture.

    Parameters
    ----------
    mix : dict | list[list]
        distribution mixture
    x: numpy.array
        input vector

    Returns
    -------
    numpy.array
        CDF vector
    """

    if isinstance(mix, dict):
        mix = mix['mix']
    weights = np.array([mx[0] for mx in mix])
    weights /= weights.sum()  # in case these did not add up to 1
    data = np.zeros(len(x))
    for n, (weight, dist, args) in enumerate(mix):
        data += weights[n] * getattr(scipy.stats, dist).cdf(x, *args)
    return data

def pdf(mix, x, x_val):
    """
    Return a PDF from a distribution mixture.

    Parameters
    ----------
    mix : dict | list[list]
        distribution mixture
    x: numpy.array
        input vector
    x_val: str
        flow feature being sampled

    Returns
    -------
    numpy.array
        PDF vector
    """

    if isinstance(mix, dict):
        mix = mix['mix']
    weights = np.array([mx[0] for mx in mix])
    weights /= weights.sum()  # in case these did not add up to 1
    data = np.zeros(len(x))
    mask, mu, pu, mui, nu, nui = mask_values(x, x_val)
    for n, (weight, name, args) in enumerate(mix):
        dist = getattr(scipy.stats, name)
        data[mask] += weights[n] * (dist.cdf(mu, *args) - dist.cdf(pu, *args))[mui]
        data[~mask] += weights[n] * dist.pdf(nu, *args)[nui]
    return data

def cdf_comp(mix, x):
    """
    Return CDF components from a distribution mixture.

    Parameters
    ----------
    mix : dict | list[list]
        distribution mixture
    x: numpy.array
        input vector

    Returns
    -------
    dict[str, numpy.array]
        dict of named CDF components
    """

    if isinstance(mix, dict):
        mix = mix['mix']
    weights = np.array([mx[0] for mx in mix])
    weights /= weights.sum()  # in case these did not add up to 1
    data = {}
    for n, (weight, dist, args) in enumerate(mix):
        data[f'{weight} {dist}'] = weights[n] * getattr(scipy.stats, dist).cdf(x, *args)
    return data

def pdf_comp(mix, x, x_val):
    """
    Return PDF components from a distribution mixture.

    Parameters
    ----------
    mix : dict | list[list]
        distribution mixture
    x: numpy.array
        input vector
    x_val: str
        flow feature being sampled

    Returns
    -------
    dict[str, numpy.array]
        dict of named PDF components
    """

    if isinstance(mix, dict):
        mix = mix['mix']
    weights = np.array([mx[0] for mx in mix])
    weights /= weights.sum()  # in case these did not add up to 1
    components = {}
    mask, mu, pu, mui, nu, nui = mask_values(x, x_val)
    for n, (weight, name, args) in enumerate(mix):
        data = np.zeros(len(x))
        dist = getattr(scipy.stats, name)
        data[mask] = dist.cdf(mu, *args) - dist.cdf(pu, *args)[mui]
        data[~mask] = dist.pdf(nu, *args)[nui]
        data *= weights[n]
        components[f'{weight} {name} {args}'] = data
    return components

def avg(data, x, x_val, what):
    """
    Calculate average value of a feature basing on distribution mixtures.

    Parameters
    ----------
    data : dict[dict[list[list]]
        container with all distribution mixtures
    x: numpy.array
        input vector
    x_val: str
        flow feature being sampled
    what : str
        average value to calculate

    Returns
    -------
    numpy.array
        vector of average values
    """

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

def to_json(mix, suma=None):
    s = ''
    if suma:
        s += '{\n' + f'  "sum": {suma},\n' + '  "mix": '
    s += '[\n    ' + ',\n    '.join(json.dumps(c, default=lambda v: v.item()) for c in mix) + '\n  ]'
    if suma:
        s += '\n}'
    return s
