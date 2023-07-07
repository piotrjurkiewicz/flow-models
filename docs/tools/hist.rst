:program:`hist`
***************

.. argparse::
   :ref: flow_models.hist.parser
   :prog: python3 -m flow_models.hist

Fitting of mixture models does not have to be performed on complete flow records. Instead, it can be performed on histograms, calculated by binning flow records into buckets according to the selected parameter (e.g. flow length or size). Histogram files can also be easily published as they are many orders of magnitude smaller and, unlike flow records, do not contain private information such as IP addresses.

The tool takes flow records in any supported format as an input and outputs a histogram file in a CSV format. A user should specify the parameter to be binned (flow length, size, duration or rate) and additional columns to be summed in a histogram (by default packets and octets are counted, additional fields can be rate and duration). The user can also specify a parameter, which is a power-of-two defining starting point for logarithmic binning. Logarithmic binning significantly reduces the size of histogram files without affecting the quality of the fitting process noticeably.

Two implementations of the tool are available: ``hist`` and :doc:`/tools/hist_np`. The former is a pure Python implementation that takes advantage of unlimited width integer support in Python in order to perform more accurate calculations. The latter uses the NumPy package to perform binning, which can utilize SIMD instructions and multiple threads and is many orders of magnitude faster, but requires more memory and can introduce rounding errors due to the operation on doubles having limited precision.
