:program:`sort`
***************

.. argparse::
   :ref: flow_models.sort.parser
   :prog: flow-models-sort

During the merging process, flow records may become reordered. This applies especially to long flows, which in some circumstances may stay cached until the end of merging process. Such flows are dumped at the end of output files. The purpose of ``sort`` tool is to reorder flow records in a file according to specified keys, usually flow start or flow end times. This step is unnecessary when further operations will be performed on the whole file. However, in a case when only a part of a record file will be used, sorting is necessary.
