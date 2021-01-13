:program:`plot`
***************

.. toctree::
   :maxdepth: 2

.. argparse::
   :ref: flow_models.plot.parser
   :prog: flow-models-plot

An important part of any modeling task is the visualization of both input data and resulting models. The ``plot`` tool can be used for that purpose. It can generate probability density (PDF), cumulative distribution function (CDF) and average packet size and packet interarrival time plots. It takes CSV histogram files and mixture model JSON files as input. The input histogram data can be visualized on PDF plot as points, 2-dimensional histogram or kernel density estimation (KDE) contour plot. Model mixtures are presented as lines. Additionally, components of a mixture can be plotted, both separately and in stacked mode. The tool automatically normalizes data points in the case of logarithmically-binned histograms. Moreover, the framework contains a custom fast Fourier transform (FFT) based implementation of weighted KDE computation.
