Time-series classification
==========================

.. currentmodule:: prosemble.models

Time series of the same class often show the same pattern at different
times. This tutorial classifies such series with :class:`SoftDTWRSLVQ`,
whose prototypes are series themselves and whose distance aligns the
series before comparing them.

The data
--------

Trace, from the UCR time series archive, has four classes of simulated
instrument readings: 100 training and 100 test series of length 275. The
loader of the time-series benchmark (in a clone of the repository)
downloads it and z-normalises every series:

.. code-block:: python

   import sys
   sys.path.insert(0, "benchmarks/time_series")
   from run import load

   X_train, y_train, X_test, y_test = load("Trace")

Train
-----

.. code-block:: python

   import numpy as np
   from prosemble.models import SoftDTWRSLVQ

   model = SoftDTWRSLVQ(gamma=1.0, n_prototypes_per_class=1, max_iter=150,
                        lr=0.05, batch_size=64, random_seed=0)
   model.fit(X_train, y_train)

   np.mean(model.predict(X_test) == y_test)     # 1.0

The model classifies all 100 test series correctly. A 1-nearest-neighbour
classifier with the Euclidean distance gets 76 % on this dataset, because
the classes differ in patterns that do not occur at the same time in every
series.

The prototypes
--------------

.. figure:: /_static/figures/tutorial-timeseries-light.png
   :class: only-light pr-figure

.. figure:: /_static/figures/tutorial-timeseries-dark.png
   :class: only-dark pr-figure

   For each class, 25 training series (grey), their point-wise mean
   (dashed), and the learned soft-DTW prototype (colour).

``model.prototypes_`` holds one series per class, which can be plotted next
to the data. The point-wise mean of each class blurs the sharp features,
because they occur at different times: the spike of class 1 and the drop
of class 2 are smeared out in the mean but kept in the prototypes. The
prototypes are trained to separate the classes rather than to average
them, so they also exaggerate some differences, such as the level after
the drop in class 2.

Settings and cost
-----------------

* ``gamma`` smooths the alignment. On the UCR benchmark, ``gamma=1``
  was more accurate than the default ``gamma=0.1``; see
  :doc:`/benchmarks/index`.
* Training compares every series with every prototype through an
  alignment whose cost grows with the product of the lengths. For long
  series or large training sets, set ``batch_size``, and use ``band`` to
  restrict alignments to a band around the diagonal.

See :doc:`/user_guide/time_series` for the model and its options.
