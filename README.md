# flow-models: A framework for analysis and modeling of IP network flows

Packages like `flow-tools` or `nfdump` provide tools for filtering and calculating simple summary/top-N statistics
from network flow records. They lack, however, any capabilities for analysis and modeling of flow features (length,
size, duration, rate, etc.) distributions. The goal of this framework is to fill this gap.


## Provided tools

The framework currently includes the following tools:

- `merge` -- performs merge of flows which were split across multiple records due to *active timeout*
- `convert` -- converts supported flow records formats between each other
- `sort_np` -- sorts flow records according to specified key fields (requires `numpy`)
- `hist` -- calculates histograms of flows length, size, duration or rate
- `hist_np` -- calculates histograms using multiple threads and `numpy` (much faster, but requires more memory)
- `fit` -- creates General Mixture Models (GMMs) fitted to flow records
- `plot` -- generates PDF and CDF plots from flow records and fitted models

Each tool is a separate Python program. Features they provide are orthogonal and they are tailored to be used together in data processing pipelines.

## File formats

The framework currently supports the following flow records formats:

- `pipe` -- `nfdump` pipe format
- `nfcapd` -- `nfdump` binary format
- `csv_flow` -- comma-separated values text format (see below)
- `binary` -- separate binary array file for each field (see below)

Additionally, the framework currently supports the following flow histograms formats:

- `csv_hist` -- comma-separated values text format (see below)

### `csv_flow` file format

    af, prot, inif, outif, sa0, sa1, sa2, sa3, da0, da1, da2, da3, sp, dp, first, first_ms, last, last_ms, packets, octets, aggs

### `csv_hist` file format

    bin_lo, bin_hi, flows_sum, packets_sum, octets_sum, duration_sum, rate_sum, aggs_sum

### `binary` file format

The `binary` file format is also used as an internal on-disk format to exchange data between tools included in the framework.
Each flows trace is a directory, which contains several binary files. Each binary file stores one
field as an array of binary values with a specified type.

File naming schema is the following:

    _{field_name}.{dtype}  -- key fields
    {field_name}.{dtype}   -- value fields

Suffix `dtype` specifies the type of binary object stored in file:

| Type code | C Type |
| - | - |
| `b`   | signed char |
| `B`   | unsigned char |
| `h`   | signed short |
| `H`   | unsigned short |
| `i`   | signed int |
| `I`   | unsigned int |
| `l`   | signed long |
| `L`   | unsigned long |
| `q`   | signed long long |
| `Q`   | unsigned long long |
| `f`   | float |
| `d`   | double |

Such a storage schema has several advantages:

- fields can be distributed independently (for example one can share flow records without `sa*` and `da*` address fields for privacy reasons)
- fields can be compressed/uncompressed selectively (important when processing data which barely fits on disks)
- additional or custom fields can be trivially added or removed
- supports storage of any field using any object type (signedness, precision)
- files can be memory-mapped as numerical arrays (unlike `IPFIX`, `nfcapd` or any other structured/TLV format)
- the format is so simple that files can be memory-mapped into any big data processing software and is future-proof
- memory-mapping is IO and cache efficient (columnar memory layout allows applications to avoid unnecessary IO and accelerate analytical processing performance on modern CPUs and GPUs)

Example:

    agh_201506/
    └── day-01
        ├── _af.B            ─┐
        ├── _da0.I            │
        ├── _da1.I            │
        ├── _da2.I            │
        ├── _da3.I            │
        ├── _dp.H             │
        ├── _inif.H           │ key
        ├── _outif.H          │ fields
        ├── _prot.B           │
        ├── _sa0.I            │
        ├── _sa1.I            │
        ├── _sa2.I            │
        ├── _sa3.I            │
        ├── _sp.H            ─┘
        ├── first.I          ─┐
        ├── first_ms.H        │
        ├── last.I            │ value
        ├── last_ms.H         │ fields
        ├── octets.Q          │
        └── packets.Q        ─┘

