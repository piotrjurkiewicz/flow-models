:program:`plot`
****************

Plots the detection method boundaries: curves are loaded from result CSVs
produced by :program:`calculate`, :program:`simulate` (histogram ``_data``
and mixture ``_mix`` variants) and :program:`simulate --real`.  Generates
per-method three-panel coverage figures (traffic, operations, occupancy
share), flow table operations and occupancy reduction figures, and LaTeX
tables of selected results.

Each positional argument is a results directory
``{calculate,simulate}_{mix,data,real}/all/{length,size}`` containing
per-method CSV files.  In the web figures every curve is labeled with its
result source: ``(calc. mix)`` / ``(calc. data)`` for analytical curves,
``(sim. mix)`` / ``(sim. data)`` for simulations, and ``(real times, sim.)``
for the real-times mode.

With ``--format svg`` or ``--format both`` the publication figures are also
written as SVG.  The ``--out DIR`` option additionally writes a per-x-value
set of SVG figures (``DIR/{length,size}/``) with embedded curve metadata for
the interactive web viewer.

.. argparse::
   :ref: flow_models.elephants.boundaries.plot.parser
   :prog: python3 -m flow_models.elephants.boundaries.plot
