Dataset naming convention
*************************

.. toctree::
   :maxdepth: 2

Each dataset is stored in a directory named according to the following
scheme: ::

    <dataset>[_<selection>][_<transform>...]/<selector>/<selector>/...

- ``dataset`` -- name of the base dataset (e.g. ``agh_2015``)
- ``selection`` -- optional time selection, folded into the dataset identifier
  as lowercase digits (e.g. ``agh_2015061019`` selects the hour between
  19:00-20:00 UTC on June 10th)
- ``transform`` -- optional trailing lowercase tokens describing
  transformations applied to the dataset records (e.g. ``anon``)
- ``selector`` -- a filter or datetime selection represented by one directory;
  selector directories may be nested to any depth

Every model node stores its generated artifacts in an ``ALL`` child directory.
The repository-level unfiltered model uses the top-level ``ALL`` directory;
``IPv4/ALL`` contains artifacts after applying ``IPv4`` with no additional
child selector, and ``IPv4/TCP/ALL`` contains artifacts after applying both
selectors. ``ALL`` is an artifact container and never contains
``selector.json``, ``Makefile``, or ``README.md``.

Each selector applies in addition to every selector above it. For example,
``IPv4/TCP/noDNS`` combines the predicates declared by ``IPv4``, ``TCP``, and
``noDNS``. A hierarchy can mix datetime and filter selectors, allowing paths
such as ``2025/2025-06/2025-06-10/IPv4/TCP``. The selector node contains its
metadata and build instructions, while its ``ALL`` child contains the resulting
artifacts.

Only protocol names are written in UPPERCASE. All other tokens (filters and
transforms) are written in lowercase.

Repository metadata
===================

Every repository root must contain ``metadata.json``. The web interface uses
this file to discover datasets and does not contain a hardcoded dataset list.
The supported fields are:

.. code-block:: json

    {
      "name": "agh_2015061019_IPv4_anon",
      "description": "One-hour anonymized AGH dataset.",
      "records": true
    }

``name`` and ``description`` identify the repository. ``records`` states
whether binary flow records are published. Repository metadata is descriptive;
all selection fields belong in ``selector.json``.

Selector metadata
=================

Every repository root and selector directory must contain a ``selector.json``
file. The root file declares any filter or datetime selection applied to the
whole repository and is empty only when no such selection exists. The file is
included in ``files.json`` and may declare either or both of these fields:

.. code-block:: json

    {
      "filter": "prot == 6",
      "datetime_start": "2015-06-10T19:00:00Z",
      "datetime_end": "2015-06-10T20:00:00Z"
    }

``filter`` is the predicate introduced by that directory only, not the complete
predicate inherited from its parents. The effective filter expression is the
logical AND of all non-empty ``filter`` values from the dataset and selector
hierarchy.

``datetime_start`` and ``datetime_end`` define a half-open ISO 8601 interval.
They must be specified together. Nested datetime selectors must describe
intervals contained by their parents. When several datetime selectors are
active, user interfaces display the start and end of the most specific
interval, defined as the interval with the shortest duration. Directories may
contain filter metadata, datetime metadata, or both. There is no limit on
selector nesting depth.

Both metadata files must be included in ``files.json``. The web interface
builds cascading selectors from these files and hides the selector control when
the empty root selector is the repository's only model.

Each model node follows this layout: ::

    selector.json
    Makefile
    README.md
    ALL/
        plots/
        mixtures/
        histograms/
        summary/
    <child-selector>/

``Makefile`` runs directly from the model node with ``OUT := ALL`` and writes
generated content under ``ALL`` without recursive Make invocations. Public
targets such as ``plots`` and ``summary`` are aliases for their corresponding
``ALL`` targets. ``README.md`` documents the effective model. Selector
directories and generated artifact directories therefore never occupy the same
directory.

Filter tokens
=============

Each filter token denotes a single predicate, equivalent to a
``--filter-expr`` expression. The following tokens are defined:

.. list-table::
   :header-rows: 1

   * - Token
     - Meaning
     - Equivalent ``--filter-expr``
   * - ``IPv4``
     - IPv4 flows only
     - ``af == 2``
   * - ``IPv6``
     - IPv6 flows only
     - ``af == 10``
   * - ``TCP``
     - TCP flows only
     - ``prot == 6``
   * - ``UDP``
     - UDP flows only
     - ``prot == 17``
   * - ``ICMP``
     - ICMP flows only
     - ``prot == 1``
   * - ``DNS``
     - DNS flows only (queries and responses)
     - ``(sp == 53) | (dp == 53)``
   * - ``HTTP3``
     - HTTP/3 (QUIC) flows only
     - ``(prot == 17) & ((sp == 443) | (dp == 443))``
   * - ``ge<N>p``
     - flows of at least N packets
     - ``packets >= N``
   * - ``lt<N>p``
     - flows of fewer than N packets
     - ``packets < N``
   * - ``ge<N>o``
     - flows of at least N octets
     - ``octets >= N``
   * - ``lt<N>o``
     - flows of fewer than N octets
     - ``octets < N``
   * - ``ge<N>s``
     - flows lasting at least N seconds
     - ``(last - first) >= N``
   * - ``lt<N>s``
     - flows lasting fewer than N seconds
     - ``(last - first) < N``
   * - ``wellknown``
     - flows involving a well-known port
     - ``(sp < 1024) | (dp < 1024)``
   * - ``up`` / ``down``
     - upstream / downstream flows (dataset-specific expression)
     - e.g. source/destination subnet check

Address family tokens (``IPv4``, ``IPv6``) and protocol tokens (``TCP``,
``UDP``, ``ICMP``, ``HTTP3``) are uppercase; all remaining filter tokens are
lowercase. Port-based tokens match either endpoint (``sp`` or ``dp``), since flow
records are unidirectional and a service exchange produces separate query
and response flows. If only client-to-server flows are wanted, the
dataset's Makefile can narrow the expression to ``dp`` only and this must
be documented there.

When multiple filters are used, their nested directories should appear in the
order: address family, protocol, size, duration, direction, port.

Negative filters
----------------

A filter token may be negated by prefixing it with ``no``. The ``no``
modifier is lowercase; the protocol name retains its uppercase form.

.. list-table::
   :header-rows: 1

   * - Token
     - Meaning
     - Equivalent ``--filter-expr``
   * - ``noTCP``
     - non-TCP flows only
     - ``prot != 6``
   * - ``noUDP``
     - non-UDP flows only
     - ``prot != 17``
   * - ``noDNS``
     - non-DNS flows only
     - ``(sp != 53) & (dp != 53)``
   * - ``noHTTP3``
     - non-HTTP/3 flows only
     - ``(prot != 17) | ((sp != 443) & (dp != 443))``
   * - ``nowellknown``
     - flows not involving any well-known port
     - ``(sp >= 1024) & (dp >= 1024)``

Negative tokens should only be used where the complement is not already
expressible by a positive token: ``noIPv4`` is simply ``IPv6``, ``noup`` is
``down``, and ``noge5p`` is ``lt5p``. In such cases the positive form must be
used.

Transform tokens
================

Trailing lowercase tokens describe transformations applied to the records
after filtering:

.. list-table::
   :header-rows: 1

   * - Token
     - Meaning
   * - ``anon``
     - IP addresses anonymized with the Crypto-PAn algorithm

Examples
========

.. list-table::
   :header-rows: 1

   * - Directory name
     - Contents
   * - ``agh_2015061019_IPv4_anon/ALL``
     - one-hour selection of ``agh_2015``, IPv4 flows only, anonymized
   * - ``agh_2015061019_IPv4_anon/TCP``
     - same selection, TCP flows only, anonymized
   * - ``agh_2015061019_IPv4_anon/HTTP3``
     - same selection, HTTP/3 flows only, anonymized
   * - ``agh_2015061019_IPv4_anon/ge5p``
     - same selection, flows with at least 5 packets, anonymized
   * - ``agh_2015061019_IPv4_anon/noTCP``
     - same selection, non-TCP flows only, anonymized
   * - ``agh_2015061019_IPv4_anon/UDP/noDNS``
     - same selection, UDP flows excluding DNS; ``noDNS`` is additive to ``UDP``
   * - ``agh_2015/IPv6``
     - full ``agh_2015`` dataset, IPv6 flows only
   * - ``agh_2015/2015-06/2015-06-10/IPv4/TCP``
     - TCP/IPv4 flows from June 10, selected through nested datetime intervals

Each selector's exact ``--filter-expr`` fragment and datetime interval are
stored in its ``selector.json``. A Makefile may repeat the effective expression
for reproducibility, but metadata must contain only the fragment introduced by
that directory.
