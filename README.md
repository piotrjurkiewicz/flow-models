# flow-models: A framework for analysis and modeling of IP network flows

Packages like `flow-tools` or `nfdump` provide tools for filtering and calculating simple summary/top-N statistics
from network flow records. They lack, however, any capabilities for analysis and modeling of flow features (length,
size, duration, rate, etc.) distributions. The goal of this framework is to fill this gap.

`flow-models` is a software framework for creating precise and reproducible statistical flow models from
NetFlow/IPFIX flow records. It can be used to merge split records, calculate histograms of flow features and create
General Mixture Models fitting them. Created models can be used both as an input in analytical calculations and to
generate realistic traffic in simulations.

The framework can be installed from [Python Package Index (PyPI)](https://pypi.org/project/flow-models/) using the
following command:

    pip install flow-models

A detailed documentation, including usage examples, is available at: https://flow-models.readthedocs.io

Apart from the framework, the Git repository also contains a library of flow models created with it, including
histograms and fitted mixture models.

## Provided tools

The framework currently includes the following tools:

- `merge` -- merges flows which were split across multiple records due to *active timeout*
- `sort` -- sorts flow records according to specified fields (requires `numpy`)
- `hist` -- calculates histograms of flows length, size, duration or rate
- `hist_np` -- calculates histograms using multiple threads (requires `numpy`, much faster, but uses more memory)
- `fit` -- creates General Mixture Models (GMM) fitted to flow records (requires `scipy`)
- `plot` -- generates plots from flow records and fitted models (requires `pandas` and `scipy`)
- `generate` -- generates flow records from histograms or mixture models
- `summary` -- produces TeX tables containing summary statistics of flow dataset (requires `scipy`)
- `convert` -- converts flow records between supported formats

Following the Unix philosophy, each tool is a separate Python program aimed at a single purpose. Features provided
by the tools are orthogonal and they are tailored to be used sequentially in data-processing pipelines.

## Models library

The `data` directory contains a library of flow models. They consist of histogram CSV files, fitted mixture JSON
files and plots. Full flow records are not included. The following flow models are currently available:

### agh_2015

    Piotr Jurkiewicz, Grzegorz Rzym and Piotr Boryło
    Flow length and size distributions in campus Internet traffic
    Computer Communications 167, 15-30
    DOI: 10.1016/j.comcom.2020.12.016

Paper available at: http://arxiv.org/abs/1809.03486

Based on NetFlow records collected on the Internet-facing interface of the AGH University of Science and Technology
network during the consecutive period of 30 days.

Dormitories, populated with nearly 8000 students, generated 69% of the traffic. The rest of the university (over
4000 employees) generated 31%. In the case of dormitories, 91% of traffic was downstream traffic (from the
Internet). In the case of rest of the university, downstream traffic made up 73% of the total traffic. Therefore,
this model can also be considered as representative of residential traffic.

| Parameter | Value | Unit |
| - | -: | -: |
| Dataset name | agh_2015 | |
| Flow definition | 5-tuple | |
| Exporter | Cisco router | (NetFlow) |
| L2 technology | Ethernet | |
| Sampling rate | none | |
| Active timeout | 300 | seconds |
| Inactive timeout | 15 | seconds|
| | | |
| Number of flows | 4 032 376 751 | flows |
| Number of packets | 316 857 594 090 | packets |
| Number of bytes | 275 858 498 994 998 | bytes |
| Average flow length | 78.578370 | packets |
| Average flow size | 68410.894128 | octets |
| Average packet size | 870.607188 | bytes |


|    | TCP | UDP | Other |
| :- | -:  | -:  | -:    |
| Flows | 53.85% | 43.09% | 3.06% |
| Packets | 83.51% | 16.01% | 0.48% |
| Octets | 88.57% | 11.27% | 0.1% |
