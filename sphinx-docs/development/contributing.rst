Contributing
============

Development environment
-----------------------

.. tab-set::

   .. tab-item:: uv

      .. code-block:: bash

         git clone https://github.com/naotoo1/prosemble.git
         cd prosemble
         uv sync --all-extras

   .. tab-item:: pip

      .. code-block:: bash

         git clone https://github.com/naotoo1/prosemble.git
         cd prosemble
         pip install -e ".[all,docs]"

   .. tab-item:: devenv (Nix)

      .. code-block:: bash

         git clone https://github.com/naotoo1/prosemble.git
         cd prosemble
         devenv shell
         uv sync --extra jax --extra dev

Running the tests
-----------------

.. code-block:: bash

   pytest tests/ -n auto -x -q

``-n auto`` runs the tests in parallel, one worker per CPU core
(pytest-xdist, installed with the ``all`` and ``dev`` extras); drop it to
run them in one process. Continuous integration runs the full suite on
Python 3.11, 3.12 and 3.13 for every pull request. The suite trains many
models and takes a while;
while working on one area, run its test file, for example
``pytest tests/test_evaluation.py``.

Building the documentation
--------------------------

.. code-block:: bash

   pip install -e ".[docs]"
   cd sphinx-docs
   make html SPHINXOPTS="-W --keep-going"   # open _build/html/index.html

Continuous integration builds the documentation for every pull request
and fails on any warning, so build with ``-W`` before opening one.

The API pages are generated from the docstrings at build time. The figures
in the tutorials are committed images; after changing a tutorial, redraw
them with

.. code-block:: bash

   python sphinx-docs/_scripts/make_figures.py            # all figures
   python sphinx-docs/_scripts/make_figures.py wine       # one figure

Docstrings follow the NumPy style. Write mathematics with the ``:math:``
role and ``.. math::`` directive, so that it renders with MathJax.

Adding a model
--------------

Supervised models subclass
:class:`~prosemble.models.SupervisedPrototypeModel`, which provides
``fit``, the training loops, prediction, rejection, persistence and export.
A new model implements:

* ``_init_state(X, y, key)``: the initial parameters, a dictionary with
  the prototypes and any other learned quantity (relevances, a matrix
  :math:`\Omega`, widths).
* ``_compute_loss(params, X, y, proto_labels)``: the cost as a JAX
  function of the parameters. Gradients, optimizers and the compiled
  training loop come from the base class.
* ``_post_update(params)``, optionally: a projection applied after every
  update, for constraints such as normalisation.
* ``_extract_results(...)``, when the model learns more than prototypes:
  sets the fitted attributes from the final parameters.

:class:`~prosemble.models.GRLVQ` in ``prosemble/models/relevance_lvq.py``
is a compact example. Then:

#. Export the model from ``prosemble/models/__init__.py`` and add it to
   ``_MODEL_REGISTRY`` there; ``load`` looks classes up in the registry.
#. Add tests in ``tests/``, covering at least fitting, prediction, saving
   and loading.
#. Document it: a section in the user guide page of its family, and an
   entry in ``sphinx-docs/api/models.rst``.

Releasing
---------

#. Update ``version`` in ``pyproject.toml``. The documentation reads the
   version from the installed package.
#. Merge to ``main`` and push a tag ``vX.Y.Z``. The release workflow runs
   the tests, publishes the package to PyPI and creates a GitHub release.
#. Read the Docs builds the documentation for the new tag.
