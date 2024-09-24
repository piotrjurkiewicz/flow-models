Tutorial: Elephant flow classification with scikit-learn
********************************************************

.. toctree::
   :maxdepth: 2

This tutorial provides a step-by-step guide through the training and evaluation of a mice/elephant classification model using scikit-learn.

Prerequisites
=============

1. Ensure that the required system packages are installed. We will work in a virtual environment.

   In some distributions, the virtual environment ``venv`` package is not installed by default with the Python binaries. Also, the ``git`` code versioning system should be installed.
   On Debian-like systems, they can be installed using the following command:

   .. code-block:: shell-session

       $ sudo apt-get install git python3-venv

2. Clone the ``flow-models`` framework, which provides examples of scripts for training and evaluating elephant flow classifiers, as well as plotting the results:

   .. code-block:: shell-session

       $ git clone https://github.com/piotrjurkiewicz/flow-models.git

3. Initialize and clone the flow datasets, including binary flow records needed for training, as they are stored in separate git submodules:

   .. code-block:: shell-session

       $ cd flow-models
       $ git submodule update --init

4. Create and activate a virtual environment named ``venv`` in the project directory:

   .. code-block:: shell-session

       $ python3 -m venv venv
       $ source venv/bin/activate

5. Install all required Python packages into the virtual environment:

   .. code-block:: shell-session

       (venv) user@host:~/flow-models$ pip install -r requirements.txt

6. Export the path to the repository root directory to the ``PYTHONPATH`` environment variable to ensure access to the ``flow_models`` package:

   .. code-block:: shell-session

       (venv) user@host:~/flow-models$ export PYTHONPATH=.

Dataset
=======

In this tutorial, we will use flow records from the dataset ``agh_2015061019_IPv4_anon``. This dataset is a subset of flows derived from the larger ``agh_2015`` dataset, corresponding to a one-hour period between 19:00-20:00 UTC on Wednesday, June 10th.

These flows were collected from the Internet-facing interface of the AGH University of Krakow network over a period of 30 consecutive days. The selected hour represents a typical working day with normal university operations and the presence of students in dormitories, which contributes to the majority of traffic. The network traffic during this hour has been carefully examined and confirmed to be free from anomalies or irregularities that may indicate unusual network activity.

To protect privacy, the IP addresses have been anonymized using the prefix-preserving Crypto-PAn algorithm. It is worth noting that this anonymization process does not adversely affect the performance of machine learning algorithms trained on these addresses.

The anonymized flow records in binary format are stored in the ``data/agh_2015061019_IPv4_anon/sorted`` directory. Flows were sorted according to their start time (first packet timestamp), allowing the simulation of a network switch observing appearing flows. The format of binary flow records is described in detail on the `formats` page.

Script command line parameters
==============================

The `skl.train_classifiers` module provides an example script for training classifier models. This script can be executed with necessary input provided as command line parameters:

.. code-block:: shell-session

    (venv) user@host:~/flow-models$ python3 flow_models/elephants/skl/train_classifiers.py --help
    usage: train_classifiers.py [-h] [-O OUTPUT] [--seed SEED] [--fork] [--jobs JOBS] directory

    Trains and evaluates sklearn classifier models to classify elephant flows.

    positional arguments:
      directory             binary flow records directory

    options:
      -h, --help            show this help message and exit
      -O OUTPUT, --output OUTPUT
                            results output directory
      --seed SEED           seed
      --fork                fork to subprocess for each simulation
      --jobs JOBS           maximum number of simultaneous subprocesses

The compulsory argument ``directory`` is a path to a directory containing binary flow records. In our case, this will be ``data/agh_2015061019_IPv4_anon/sorted``.

The ``output`` parameter can be used to specify the directory for files with results. The ``seed`` parameter allows to control the random generator seed to ensure repeatability of experiments.

Parameters ``fork`` and ``jobs`` control the parallel training of multiple models. Even when a single algorithm is analyzed, models are trained for multiple folds (default 5) to perform cross-validation. Moreover, each model can be trained with different input data preparation parameters.
The ``fork`` option allows to fork separate subprocesses to parallelize training of multiple models. The ``jobs`` parameter control how many subprocesses can be forked simultaneously.
When ``fork`` parameter is not specified, all models will be trained and evaluated sequentially within the original process.

.. note::
    The ``fork`` option has to be used carefully, especially in environments with limited memory. Some models can use a significant amount of memory during training. Therefore, it needs to be ensured that the memory requirements of parallel-running multiple jobs will not exceed the available memory.

    Many ``sklearn`` algorithms have built-in internal parallelization and are able to utilize all cores on the machine anyway. This can be enabled by providing ``{'n_jobs': -1}`` parameter into the model parameters. In such cases, the gain from using ``fork`` is limited only to periods of evaluation, which is performed on a single core.

Exploring the script code
=========================

Within the `train_classifiers` script, the machine learning algorithms used for training and evaluation are specified in the ``algos`` list:

.. code-block:: python

    algos = [
        (sklearn.tree.DecisionTreeClassifier, {}),
        (sklearn.ensemble.RandomForestClassifier, {'n_jobs': -1, 'max_depth': 20}),
        (sklearn.ensemble.ExtraTreesClassifier, {'n_jobs': -1, 'max_depth': 25}),
        (sklearn.ensemble.AdaBoostClassifier, {}),
        (sklearn.ensemble.GradientBoostingClassifier, {}),
        (sklearn.neighbors.KNeighborsClassifier, {'n_jobs': -1}),
        (sklearn.ensemble.HistGradientBoostingClassifier, {}),
        (Data, {}),
    ]

Each element in the ``algos`` list is a 2-tuple containing:

- A scikit-learn ``Classifier`` class.
- A dictionary of parameters to be passed to the classifier during initialization.

The algorithms are trained and evaluated sequentially, in the order they appear in the list.

Flow record control: ``data_par``
---------------------------------

The ``data_par`` dictionary controls how many flow records are used for training and evaluation.

.. code-block:: python

    # data_par = {'skip': 0, 'count': 1000000}
    data_par = {}

- ``skip`` - Defines the number of initial flow records to skip from the dataset.
- ``count`` - Specifies the number of flow records to process after the skipped ones.

By adjusting these parameters, you can focus the training on a specific subset of the data.

Data preparation: ``prep_params``
---------------------------------

The ``prep_params`` list defines different combinations of input data preparation options:

.. code-block:: python

    # prep_params = [{'octets': True}]
    prep_params = [{}, {'bits': True}, {'octets': True}]

Each dictionary in ``prep_params`` specifies a unique way to preprocess the input data. Training and evaluation are performed for all the combinations listed. The options include:

- ``bits`` - Transforms each component of the 5-tuple (source IP, destination IP, source port, destination port, protocol) into individual bits, treating them as separate features.
- ``octets`` - Splits any 5-tuple field longer than 8 bits into separate byte features.

When no flags are set, the 5-tuple fields are used as 32-bit integers corresponding to the features ``(source IP, destination IP, source port, destination port, protocol)``.

Evaluation modes: ``modes``
---------------------------

The ``modes`` list controls which evaluation modes are activated:

.. code-block:: python

    # modes = ['train', 'test']
    modes = ['test']

- ``test`` mode - Evaluates the model on a test dataset that does not overlap with the training data.
- ``train`` mode - Evaluates the model on the same data used for training. This is useful for diagnosing whether the model is learning effectively or merely memorizing the training data (i.e., overfitting).

Flow labeling: ``train_decision``
---------------------------------

In binary classification, the model's output is a binary decision (0/1). In our context, this decision determines whether a network flow is classified as a *mouse* or an *elephant*. This classification informs whether the flow is added to the table. Before starting the training phase, it is necessary to define a threshold for elephant flow size to appropriately label the training dataset.

.. code-block:: python

    train_decision = prepare_decision(train_octets, training_coverage)
    # train_decision = train_octets > 8388608

The ``train_decision`` array holds the labels for each flow in the training dataset. The ``prepare_decision`` function generates these labels based on the flow sizes, aiming to achieve the desired traffic coverage in the training dataset. It does so by sorting the flows in descending order of size and labeling the largest flows as elephants until the specified traffic coverage is reached. Alternatively, training labels can be generated using a size-threshold by applying a boolean comparison directly to the flow size array.

Dataset shrinking: ``idx``
--------------------------

.. code-block:: python

    # idx = Ellipsis
    idx = top_idx(train_octets, 0.1)

The ``idx`` variable allows further limiting the dataset used for training. The ``top_idx`` function retrieves the indices of the largest flows within the dataset. This function can be used to shrink the training dataset, for instance, by selecting 5% of the largest flows and an additional 5% of randomly selected smaller flows. Such a reduction can greatly decrease training time with only a minor impact on the model's accuracy. To use all flows in the training dataset, simply pass ``Ellipsis`` as ``idx``.

Handling class imbalance: ``sample_weight``
-------------------------------------------

.. code-block:: python

    # Balanced sample weights
    # sample_weight = np.ones(len(train_decision))
    # sample_weight[train_decision] *= len(train_octets[~train_decision]) / len(train_octets[train_decision])
    # Power of octets sample weights
    sample_weight = train_octets ** 0.5

Class imbalance between elephant and mouse flows poses a challenge for machine learning models, often leading to reduced classification accuracy. The ``sample_weight`` parameter can help address this imbalance during model training.

One approach, outlined in the commented section *Balanced sample weights*, involves normalizing the sample weights so that the sum of the weights for both classes is the same. This balances the number of samples from each class used for training.

However, in practice, we found that using the square root of the flow size (in bytes) as the sample weight provides better results. This method adjusts the weight of each sample according to the flow size, favoring larger flows while still considering the smaller ones.

Running the training process
============================

To train and evaluate the selected models, execute the following command:

.. code-block:: shell-session

    (venv) user@host:~/flow-models$ python3 flow_models/elephants/skl/train_classifiers.py -O results --seed 0 data/agh_2015061019_IPv4_anon/sorted

This command will create a new directory named ``results`` in the current working directory. Inside this directory, tab-separated values (TSV) files will be generated containing the results for each model.

For the purpose of this tutorial, we will run the experiment with the following settings:

.. code-block:: python

    algos = [
        (sklearn.ensemble.RandomForestClassifier, {'n_jobs': -1, 'max_depth': 20}),
        (sklearn.ensemble.ExtraTreesClassifier, {'n_jobs': -1, 'max_depth': 25}),
        (sklearn.ensemble.HistGradientBoostingClassifier, {}),
        (Data, {}),
    ]

    data_par = {}
    prep_params = [{}, {'bits': True}, {'octets': True}]
    modes = ['test']

To speed up the process, we will use the following setting: ``idx = top_idx(train_octets, 0.1)``

On a server equipped with 2x Intel Xeon Silver 4114 CPUs running at 2.20GHz (40 logical cores), completing the training and evaluation with the selected models takes approximately 5 hours when running in single-process mode (without the ``--fork`` option). During this time, the peak memory usage by the process is about 8 GB.

If you are running this on a machine with fewer cores, the memory requirements might be slightly lower, though execution time may increase.
