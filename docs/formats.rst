File formats
************

.. toctree::
   :maxdepth: 2

The framework currently supports the following flow records formats:

- ``pipe`` -- `nfdump`_ pipe format
- ``nfcapd`` -- `nfdump`_ binary format
- ``csv_flow`` -- comma-separated values text format (see below)
- ``binary`` -- separate binary array file for each field (see below)

Additionally, the framework currently supports the following formats:

- ``csv_hist`` -- comma-separated values flow histogram text format (see below)
- ``csv_series`` -- line-separated time series of packets and bytes (see below)

.. _csv_flow:

``csv_flow``
============

File contains the following fields: ::

    af, prot, inif, outif, sa0, sa1, sa2, sa3, da0, da1, da2, da3, sp, dp, first, first_ms, last, last_ms, packets, octets, aggs

- `af` -- address family
- `prot` -- IP protocol number
- `inif` -- input interface number
- `outif` -- output interface number
- `sa0` - `sa3` -- consecutive 32-bit words forming source IP address
- `da0` - `da3` -- consecutive 32-bit words forming destination IP address
- `sp` -- source transport layer port
- `dp` -- destination transport layer port
- `first` -- timestamp of first packet (seconds component)
- `first_ms` -- timestamp of first packet (milliseconds component)
- `last` -- timestamp of last packet (seconds component)
- `last_ms` -- timestamp of last packet (milliseconds component)
- `packets` -- number of packets (flow length)
- `octets` -- number of octets (bytes) (flow size)
- `aggs` -- number of aggregated flow records forming this record

``binary``
==========

The ``binary`` file format is used as an effective internal on-disk format to exchange data between tools included in the framework.
Each flows trace is a directory, which contains several binary files. Each binary file stores one
field as an array of binary values with a specified type.

File naming schema is the following: ::

    {field_name}.{dtype}

Suffix ``dtype`` specifies the type of binary object stored in the file (using `array` type codes):

.. csv-table::
   :header: Type code,C Type
   :widths: 15, 10

    ``b``,signed char
    ``B``,unsigned char
    ``h``,signed short
    ``H``,unsigned short
    ``i``,signed int
    ``I``,unsigned int
    ``l``,signed long
    ``L``,unsigned long
    ``q``,signed long long
    ``Q``,unsigned long long
    ``f``,float
    ``d``,double

Such a storage schema has several advantages:

- fields can be distributed independently (for example one can share flow records without ``sa*`` and ``da*`` address fields for privacy reasons)
- fields can be compressed/uncompressed selectively (important when processing data which barely fits on disks)
- additional or custom fields can be trivially added or removed
- supports storage of any field using any object type (signedness, precision)
- files can be memory-mapped as numerical arrays (unlike IPFIX, nfcapd or any other structured/TLV format)
- the format is so simple that files can be memory-mapped into any big data processing software and is future-proof
- memory-mapping is IO and cache efficient (columnar memory layout allows applications to avoid unnecessary IO and accelerate analytical processing performance on modern CPUs and GPUs)

Example:

.. code-block:: none

    agh_2015/
    └── day-01
        ├── af.B             ─┐
        ├── da0.I             │
        ├── da1.I             │
        ├── da2.I             │
        ├── da3.I             │
        ├── dp.H              │
        ├── inif.H            │ key
        ├── outif.H           │ fields
        ├── prot.B            │
        ├── sa0.I             │
        ├── sa1.I             │
        ├── sa2.I             │
        ├── sa3.I             │
        ├── sp.H             ─┘
        ├── first.I          ─┐
        ├── first_ms.H        │
        ├── last.I            │ value
        ├── last_ms.H         │ fields
        ├── octets.Q          │
        └── packets.Q        ─┘

``csv_hist``
============

File contains the following fields: ::

    bin_lo, bin_hi, flows_sum, packets_sum, octets_sum, duration_sum, rate_sum, aggs_sum

- `bin_lo` -- lower edge of a bin (inclusive)
- `bin_hi` -- upper edge of a bin (exclusive)
- `flows_sum` -- number of flows within a particular bin
- `packets_sum` -- sum of packets of all flows within a bin
- `octets_sum` -- sum of octets of all flows within a bin
- `duration_sum` -- sum of duration of all flows within a bin (in milliseconds)
- `rate_sum` -- sum of rates of all flows within a bin (in bps)
- `aggs_sum` -- sum of aggregated flows of all flows within a bin

Histograms can be calculated using `hist` or `hist_np` modules. The former is a pure Python implementation which can take advantage of unlimited width integer support in Python in order to perform more accurate calculations. The latter uses the `numpy` package to perform binning, which can utilise SIMD instructions and multiple threads and is therefore many orders of magnitude faster but requires more memory and can introduce rounding errors due to the operation on doubles having limited precision. Both tools output a CSV file which can be directly used to plot a histogram, CDF or PDF of a particular flow feature.

The framework user can specify a parameter *b*, which is a power-of-two defining starting point for logarithmic binning. For example, *b = 12* means that bin widths will start increasing for values *> 4096* (for lower values bin width will be equal to one). Therefore, values between 4096-8192 would be binned into bins of width 2, between 8192-16384 into bins of width 4, etc.

``csv_series``
==============
