.. flow-models documentation master file, created by
   sphinx-quickstart on Thu Dec 24 01:48:45 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to flow-models's documentation!
***************************************

`flow-models`_ is a software framework for creating precise and reproducible statistical flow models from
NetFlow/IPFIX flow records. It can be used to merge split records, calculate histograms of flow features and create
General Mixture Models fitting them. Created models can be used both as an input in analytical calculations and to
generate realistic traffic in simulations.

First mirror subpackage allows to simulate first N packet mirroring feature in switches, which sends copies for first packets of a new flow to switch's CPU or controller to perform inspection and flow identification.

Elephants subpackage provides functionalities to simulate and analyze elephant flow related mechanisms. Elephant flows (also called heavy-hitters) are flows which are responsible for the vast majority of traffic in the Internet. Keeping focused on such flows allows to utilise advanced traffic engineering (TE) mechanisms in the network without the need to maintain individual entries for all flows.

Provided tools
==============

The framework currently includes the following tools:

- `tools/merge` -- merges flows which were split across multiple records due to *active timeout*
- `tools/sort` -- sorts flow records according to specified fields (requires `numpy`)
- `tools/hist` -- calculates histograms of flows length, size, duration or rate
- `tools/hist_np` -- calculates histograms using multiple threads (requires `numpy`, much faster, but uses more memory)
- `tools/fit` -- creates General Mixture Models (GMM) fitted to flow records (requires `scipy`)
- `tools/plot` -- generates plots from flow records and fitted models (requires `pandas` and `scipy`)
- `tools/generate` -- generates example flow records from histograms or mixture models
- `tools/summary` -- produces TeX tables containing summary statistics of flow dataset (requires `scipy`)
- `tools/convert` -- coverts flow records between different formats, can also cut and filter them
- `tools/cut` -- cuts binary flow record files using `dd`
- `tools/anonymize` -- anonymizes IPv4 addreses in flow records with prefix-preserving Crypto-PAn algorithm
- `tools/series` -- generates time series of link's bit or packet rate from flow records


Following the Unix philosophy, each tool is a separate Python program aimed at a single purpose. Features provided
by the tools are orthogonal and they are tailored to be used sequentially in data-processing pipelines.

Contents
========

.. toctree::
   :hidden:

   self

.. toctree::
   :maxdepth: 2

   tutorial
   workflow
   tools
   first_mirror
   elephants

Reference
=========

.. toctree::
   :maxdepth: 3

   formats
   flow_models

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
