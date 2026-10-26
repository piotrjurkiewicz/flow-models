:program:`calculate`
*********************

Analytical calculation of elephant detection method boundaries.  The input
is a ``csv_hist`` histogram file or a fitted mixture directory; metrics are
computed as expectations over the flow size distribution on an integration
grid (mixture probability mass beyond the last grid point is folded into the
last bin).  Sampling coverage uses the conditional average packet count per
size bin and the geometric expectation of packets missed before the first
sampled packet.

Results agree with the :program:`simulate` Monte-Carlo simulation within
confidence intervals for all methods and x axis values.

.. include:: metrics.inc

.. argparse::
   :ref: flow_models.elephants.boundaries.calculate.parser
   :prog: python3 -m flow_models.elephants.boundaries.calculate
