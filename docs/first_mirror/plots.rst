:program:`plots`
****************

Plots the first N packets mirroring simulation results: per-second counter
series are loaded from ``{day}.packets_mirrored``/``{day}.octets_mirrored``
files produced by :program:`simulate`.  Generates time-of-day mirrored
packet/bit rate figures (one curve per mirror count N) and a totals figure.

Each positional argument is a mirror directory ``first_mirror/N`` named by
the number of mirrored packets N.  With ``--format svg`` or ``--format both``
the publication figures are also written as SVG.

.. argparse::
   :ref: flow_models.first_mirror.plots.parser
   :prog: python3 -m flow_models.first_mirror.plots
