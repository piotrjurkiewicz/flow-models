Dataset naming convention
*************************

.. toctree::
   :maxdepth: 2

Each dataset is stored in a directory named according to the following
scheme: ::

    <dataset>[_<selection>][_<transform>...]/<filters>

- ``dataset`` -- name of the base dataset (e.g. ``agh_2015``)
- ``selection`` -- optional time selection, folded into the dataset identifier
  as lowercase digits (e.g. ``agh_2015061019`` selects the hour between
  19:00-20:00 UTC on June 10th)
- ``transform`` -- optional trailing lowercase tokens describing
  transformations applied to the dataset records (e.g. ``anon``)
- ``filters`` -- subdirectory inside the dataset directory, named by the
  filter tokens joined with ``_``, applied in the order they appear;
  ``ALL`` denotes the unfiltered dataset

Only protocol names are written in UPPERCASE. All other tokens (filters and
transforms) are written in lowercase.

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

When multiple filters are used, they should appear in the order:
address family, protocol, size, duration, direction, port.

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
   * - ``agh_2015061019_IPv4_anon/UDP_noDNS``
     - same selection, UDP flows excluding DNS, anonymized
   * - ``agh_2015/IPv6``
     - full ``agh_2015`` dataset, IPv6 flows only

The subdirectory name encodes only *which* filters were applied. The exact
``--filter-expr`` expressions used to produce the records should be
documented in the subdirectory's Makefile, as in
``data/agh_2015061019_IPv4_anon/TCP/Makefile``.
