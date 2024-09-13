# flow-models: A framework for analysis and modeling of IP network flows

Packages like `flow-tools` or `nfdump` provide tools for filtering and calculating simple summary/top-N statistics
from network flow records. They lack, however, any capabilities for analysis and modeling of flow features (length,
size, duration, rate, etc.) distributions. The goal of this framework is to fill this gap.

`flow-models` is a software framework for creating precise and reproducible statistical flow models from
NetFlow/IPFIX flow records. It can be used to merge split records, calculate histograms of flow features and create
General Mixture Models fitting them. Created models can be used both as an input in analytical calculations and to
generate realistic traffic in simulations.

You can cite the following paper if you use `flow-models` in your research:

    @article{flow-models,
        title = {flow-models: A framework for analysis and modeling of IP network flows},
        journal = {SoftwareX},
        volume = {17},
        pages = {100929},
        year = {2022},
        issn = {2352-7110},
        doi = {10.1016/j.softx.2021.100929},
        author = {Piotr Jurkiewicz}
    }

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
files and plots. Full flow records are also included in smaller models. The following models are currently available:

### agh_2015

Please cite as:

    @article{flows-agh,
        title = {Flow length and size distributions in campus Internet traffic},
        journal = {Computer Communications},
        volume = {167},
        pages = {15-30},
        year = {2021},
        issn = {0140-3664},
        doi = {10.1016/j.comcom.2020.12.016},
        author = {Piotr Jurkiewicz and Grzegorz Rzym and Piotr Boryło}
    }

Available at: http://arxiv.org/abs/1809.03486

Based on NetFlow records collected on the Internet-facing interface of the AGH University of Krakow
network during the consecutive period of 30 days.

Dormitories, populated with nearly 8000 students, generated 69% of the traffic. The rest of the university (over
4000 employees) generated 31%. In the case of dormitories, 91% of traffic was downstream traffic (from the
Internet). In the case of rest of the university, downstream traffic made up 73% of the total traffic. Therefore,
this model can also be considered as representative of residential traffic.

| Parameter           |               Value |      Unit |
|---------------------|--------------------:|----------:|
| Dataset name        |            agh_2015 |           |
| Flow definition     |             5-tuple |           |
| Collection period   |                  30 |      days |
| Exporter            |        Cisco router | (NetFlow) |
| L2 technology       |            Ethernet |           |
| Sampling rate       |                none |           |
| Active timeout      |                 300 |   seconds |
| Inactive timeout    |                  15 |   seconds |
| Flow records        |        not included |           |
| IP Anonymization    |                   - |           |
|                     |                     |           |
| All traffic         |                     |           |
| Number of flows     |       4 032 376 751 |     flows |
| Number of packets   |     316 857 594 090 |   packets |
| Number of bytes     | 275 858 498 994 998 |     bytes |
| Average flow length |           78.578370 |   packets |
| Average flow size   |        68410.894128 |    octets |
| Average packet size |          870.607188 |     bytes |


|         |    TCP |    UDP | Other |
|:--------|-------:|-------:|------:|
| Flows   | 53.85% | 43.09% | 3.06% |
| Packets | 83.51% | 16.01% | 0.48% |
| Octets  | 88.57% | 11.27% |  0.1% |

### agh_2015061019_IPv4_anon

Please cite as:

    @article{flow-models2,
        title = {flow-models 2.0: Elephant flows modeling and detection with machine learning},
        journal = {SoftwareX},
        volume = {24},
        pages = {101506},
        year = {2023},
        issn = {2352-7110},
        doi = {10.1016/j.softx.2023.101506},
        author = {Piotr Jurkiewicz}
    }

Available at: https://doi.org/10.1016/j.softx.2023.101506

This model is a subset of flows derived from the larger `agh_2015` dataset. It corresponds to a one-hour period between 19:00-20:00 UTC on Wednesday, June 10th. These flows were collected from the Internet-facing interface of the AGH University of Krakow network over a period of 30 consecutive days. The selected hour represents a typical working day with normal university operations and the presence of students in dormitories, which contributes to the majority of traffic.

We have carefully examined the network traffic during this hour and confirmed that it is free from anomalies or irregularities that may indicate unusual network activity. Furthermore, the calculated theoretical reduction rate curves of a perfect elephant detection algorithm utilizing only the first packet for this specific hour closely resemble those of the entire `agh_2015` dataset.

This model includes full flow records in a binary format. To protect privacy, the IP addresses have been anonymized using the prefix-preserving Crypto-PAn algorithm. It is worth noting that this anonymization process does not adversely affect the performance of machine learning algorithms trained on these addresses, as demonstrated in the aforementioned paper.

Dormitories, populated with nearly 8000 students, generated 86% of the traffic. The rest of the university (over 4000 employees) generated 14%. Therefore, this model can also be considered as representative of residential traffic.

| Parameter           |                    Value |      Unit |
|---------------------|-------------------------:|----------:|
| Dataset name        | agh_2015061019_IPv4_anon |           |
| Flow definition     |                  5-tuple |           |
| Collection period   |                        1 |      hour |
| Exporter            |             Cisco router | (NetFlow) |
| L2 technology       |                 Ethernet |           |
| Sampling rate       |                     none |           |
| Active timeout      |                      300 |   seconds |
| Inactive timeout    |                       15 |   seconds |
| Flow records        |        included (binary) |           |
| IP Anonymization    |               Crypto-PAn |           |
|                     |                          |           |
| All traffic         |                          |           |
| Number of flows     |                6 517 484 |     flows |
| Number of packets   |              680 883 573 |   packets |
| Number of bytes     |          547 382 325 881 |     bytes |
| Average flow length |               104.470310 |   packets |
| Average flow size   |             83986.754073 |    octets |
| Average packet size |               803.929405 |     bytes |

|         |    TCP |    UDP | Other |
|:--------|-------:|-------:|------:|
| Flows   | 52.52% | 45.76% | 1.72% |
| Packets | 81.09% | 18.60% | 0.31% |
| Octets  | 86.76% | 13.13% | 0.11% |
