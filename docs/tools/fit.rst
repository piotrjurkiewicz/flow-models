:program:`fit`
**************

.. toctree::
   :maxdepth: 2

.. argparse::
   :ref: flow_models.fit.parser
   :prog: flow-models-fit

The ``fit`` tool is the key component of the framework. Its purpose is to find a mixture of distributions (along with their parameters) matching accurately the selected flow feature. We have implemented the Expectation-Maximisation (EM) algorithm to estimate the parameters of a statistical model composed of mixture components.

The tool takes a flow histogram CSV file as an input and performs distribution mixture fitting. JSON file, describing shares of separate distributions in the mixture and their parameters, is an output. In order to start the EM algorithm, an initial distribution mixture has to be provided. Its parameters are then iteratively refined in order to find the local optimum. The tool can receive an initial distribution mixture from a user, but it can also generate an initial mixture for a particular dataset on its own, which means that the user has to only provide the number and types of distributions used in a mixture.

Currently, *uniform*, *normal*, *lognormal*, *Pareto*, *Weibull* and *gamma* distributions can be used in mixtures fitted by our tool. However, we have discovered, that *uniform* and *lognormal* distributions are usually sufficient to provide an accurate mixture model of flow lengths and sizes. They have an advantage of being fast to fit, since their maximization steps have analytical solutions, whereas some other distribution parameters (*Weibull* or *gamma*) must be calculated using numerical optimization methods. Another advantage is that they are widely implemented, so distribution mixtures composed of them can be usable in various network simulators and traffic generators.

The ``fit`` tool can operate in command line mode and graphical interactive mode (GUI). In the case of batch operation, fitting is performed according to provided command line parameters and the result is saved in a JSON file in the working directory. In the case of interactive operation, the user can observe the fitting process in real-time on a GUI. After its completion, he can examine the model quality on plots and, if necessary, refine the number of distributions and their initial parameters and repeat the fitting. The video showing the interactive fitting process is provided in :doc:`/tutorial`.
