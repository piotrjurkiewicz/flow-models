:program:`merge`
****************

.. argparse::
   :ref: flow_models.merge.parser
   :prog: flow-models-merge

In all hardware and many software exporters, long-lasting flows may become split due to *active timeout* and reported as multiple flow records. Such flow records have to be found and merged back in order to obtain accurate flow length, size or duration values. The ``merge`` tool available in our framework can be used for that purpose. Additionally, it filters out erroneously split records. The tool processes all flow records sequentially and performs all calculations using only integers to ensure precision and reproducibility. This is possible thanks to Python's unlimited width integer support.

The tool takes flow records in any supported format as an input and outputs merged flow records in binary or CSV format. Each merged flow record contains *aggs* field, which tells how many flow records were merged back into that particular aggregate flow record. A user should specify both *active* and *inactive* timeouts used in the collection process when calling the command to ensure the correctness of merge operation.
