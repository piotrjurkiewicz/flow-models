Detection methods
=================

Three elephant detection methods are evaluated:

``first``
    Idealized first-packet classification: a flow is added to the flow table
    before its first packet when its size exceeds the threshold.  This is an
    oracle bound for algorithms making the decision on the first packet, such
    as the machine learning classifiers evaluated with
    :program:`simulate_data`.

``threshold``
    Counter-based detection: a flow is added only after the threshold number
    of packets (``x_val='length'``) or bytes (``x_val='size'``) has been
    observed.  Traffic preceding the threshold is not covered.

``sampling``
    Packet sampling: packets are sampled independently with probability *p*
    and a flow is added at its first sampled packet.  For ``x_val='size'``
    sampling operates on 64-byte units, hence the per-packet detection
    probability is ``1 - (1 - p) ** (packet_size / 64)``.

Result metrics
==============

``flows``, ``packets``, ``octets``
    Percentage of flows added to the flow table, and percentages of packets
    and octets transmitted after insertion (i.e. covered).

``fraction``
    Mean per-flow fraction of packets covered, in percent.

``operations``
    Flow table insertion reduction ``1 / flows``.

``occupancy``
    Flow table occupancy reduction; its exact meaning follows the mode of
    the run:

    *default* (neither ``--pps`` nor ``--real``)
        Time-free bound ``1 / fraction`` (``100 / fraction`` in
        :program:`simulate` output).  It assumes that a flow's table
        residency is proportional to its covered packet count and models
        neither flow duration nor the inactivity timeout.  For the
        ``first`` method it reduces to the inverse of the detected flow
        fraction.

    ``--pps``
        Residency model: every flow is assumed to occupy the table for
        ``floor(packets / pps) + 1 + timeout`` seconds, a detected flow for
        ``floor(covered_packets / pps) + 1 + timeout`` seconds, and
        ``occupancy`` is the ratio of total to covered residency.  This
        approximates the ``flows_slots`` ratio of :program:`simulate_data`
        running in ``--pps`` mode, matching it on the same flow population
        when the simulation window (``--max-seconds``) is large enough to
        avoid truncating flow residency.  Note that ``pps`` is a constant
        packet-rate model: real flow durations vary with the actual packet
        rate, which histograms and mixtures do not carry, so it remains an
        estimate rather than an exact simulation.  The metric is sensitive
        to both parameters — at realistic burst rates (``pps`` ≳ 100) every
        entry's residency approaches the ``1 + timeout`` floor and
        ``occupancy`` converges toward the ``operations`` entry-count
        reduction; at low ``pps`` detected elephants hold entries much
        longer than mice and the reduction is dragged down.

    ``--real``
        Real-times measurement on sorted binary flow records, following
        the protocol of the :mod:`flow_models.elephants.skl` experiments:
        :program:`simulate_data` runs over the records with ``fps``-paced
        uniform arrivals and real flow durations; ``octets_mean`` is the
        mean per-second traffic coverage and ``occupancy_mean`` the mean
        per-second flow-table occupancy reduction
        ``flow_entries_all[t] / flow_entries_elephants[t]`` over the
        ``[300:720]`` window.  ``threshold`` and ``sampling`` insertions
        are delayed by the modeled detection time under the uniform
        intra-flow rate assumption (threshold count elapsed, or first
        sampled unit).  These real-times curves are the realistic
        occupancy reduction shown in the combined occupancy figure.
