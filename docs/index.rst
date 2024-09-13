.. flow-models documentation master file, created by
   sphinx-quickstart on Thu Dec 24 01:48:45 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to flow-models's documentation!
***************************************

`flow-models`_ is a software framework for creating precise and reproducible statistical flow models from NetFlow/IPFIX flow records. It offers features such as merging split records, calculating histograms of flow features, and creating General Mixture Models to fit the data. These models can be used for analytical calculations and simulations to generate realistic traffic.

`first_mirror` subpackage allows for the simulation of the first N packet mirroring feature in switches. This feature involves sending copies of the initial packets of a new flow to the switch's CPU or controller for inspection and flow identification.

`elephants` subpackage provides functionalities for simulating and analyzing mechanisms related to elephant flows. Elephant flows (also called heavy-hitters) are flows which are responsible for the vast majority of traffic in the Internet. By focusing on these flows, advanced traffic engineering (TE) mechanisms can be leveraged in the network without the requirement of maintaining individual entries for every flow.

You can cite the following paper if you use `flow-models` in your research:

.. code-block:: bibtex

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

Models library
==============

The GitHub repository contains a library of flow models. They consist of histogram CSV files, fitted mixture JSON
files and plots. Full flow records are also included in smaller models. Available models can be explored here: https://github.com/piotrjurkiewicz/flow-models/tree/master/data

Contents
========

.. toctree::
   :hidden:

   self

.. toctree::
   :maxdepth: 2

   workflow
   tools
   tutorial
   first_mirror
   elephants
   elephants_tutorial

Reference
=========

.. toctree::
   :maxdepth: 3

   formats
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
