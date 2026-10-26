:program:`simulate`
********************

Monte-Carlo simulation of elephant detection method boundaries on synthetic
traffic.  The input is a ``csv_hist`` histogram file or a fitted mixture
directory; flows are generated with :func:`flow_models.generate.generate_flows`
and each flow is processed packet by packet.  Every (method, x grid point)
combination is repeated for a configurable number of rounds and results are
reported as means with 95% confidence intervals.

Coverage results (``flows``, ``packets``, ``octets``) for the ``first``
method are equivalent to a :program:`simulate_data` mask run with the same
size threshold.  See :program:`calculate` for the analytical equivalent of
this simulation.

With ``--real DIR`` the program instead reads sorted binary flow records
and computes the real-times ``occupancy`` metric for all methods:
:program:`simulate_data` is invoked per detection threshold with uniform
``fps``-paced arrivals and real durations, and delayed methods shift each
detected flow's insertion by its modeled detection time.

.. include:: metrics.inc

.. argparse::
   :ref: flow_models.elephants.boundaries.simulate.parser
   :prog: python3 -m flow_models.elephants.boundaries.simulate
