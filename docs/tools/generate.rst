`generate`
**********

.. toctree::
   :maxdepth: 2

.. argparse::
   :ref: flow_models.generate.parser
   :prog: flow-models-generate

In order to be used for benchmarking network mechanisms, models must enable the generation of traffic matching the mixtures. The tool `generate` provides a reference for how to properly generate flows from distribution mixtures. It takes a path to the directory containing JSON mixture models as input and outputs flow records. Additionally, a CSV histogram file can be used instead of a mixture model as an input, to generate flows exactly matching the particular dataset.
