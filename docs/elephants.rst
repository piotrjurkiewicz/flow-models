Elephant flow classification and detection
******************************************

.. toctree::
   :glob:
   :maxdepth: 2

The ``elephants`` subpackage provides functionalities related to elephant flows modeling. It offers tools for simulating flow table occupancy and operations reduction for a desired fraction of traffic, which can be achieved using various elephant detection methods. These methods include classifying flows based on the first packet header (including machine learning-based algorithms), reaching a predefined counter threshold (including inexact counters like sketches or Bloom filters), or detecting elephants through sampling.

The `plot_entropy` tool read flow records in binary format and generates plots of features entropy and importances. These information can be useful for determining packet header portions to be used for training machine learning models.

The `simulate_data` tool performs simulations at the flow level. It reads flow records in binary format and simulate flow table behavior according to given mouse/elephant classification mask array.

Programs
========

.. toctree::
   :glob:
   :maxdepth: 1

   elephants/*

Example scripts for scikit-learn
================================

Additionally, we provide the `elephants.skl` subpackage, which offers examples on how to use the scikit-learn library to train machine learning algorithms for detecting elephant flows based on the first packet. None of previous works analyze metrics such as flow table reduction or the amount of traffic transmitted after flow classification, which we believe are crucial from the perspective of traffic engineering and QoS. These studies primarily focus on classification accuracy, measured by parameters like true positive rate, true negative rate, and accuracy of flow size and duration prediction. They provide limited insight into the effectiveness of the analyzed algorithms in our specific application. For example, misclassifying the largest flow in the network has a much greater impact on the change in traffic coverage than misclassifying a small flow. The metrics presented so far do not account for this difference. Our proposal is to use novel metrics for evaluating ML algorithms in the context of elephant flow detection, specifically flow table occupancy reduction and fraction of traffic covered. There is a tradeoff between these two metrics: increasing the elephant detection threshold leads to greater flow table reduction but decreases the fraction of covered traffic.

The examples in the `elephants.skl` subpackage demonstrate how to train and validate classifiers and regressors from the scikit-learn library to obtain reduction curves and how to tune the hyperparameters of these algorithms. Additionally, we provide the `lib.ml` utility module, which contains functions useful in machine learning applications, such as data preparation and calculation of flow table reduction scores.

.. toctree::
   :glob:
   :maxdepth: 1

   elephants/skl/*
