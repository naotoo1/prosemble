Deep, Siamese and image models
==============================

.. currentmodule:: prosemble.models

These models put a neural network in front of the prototype layer. They
can separate classes that no metric on the raw features separates, at a
cost in interpretability: the decision is still a comparison with
prototypes, but in a learned feature space. Where the prototypes live
matters for explanation. :class:`LVQMLN` and :class:`PLVQ` keep prototypes
in the latent space; the Siamese and image models keep them in the input
space, where they can be inspected like the data.

Deep Variants (LVQMLN, PLVQ)
----------------------------

Add a trainable MLP backbone for nonlinear feature extraction.
Prototypes live in the latent space (not the input space).

.. code-block:: python

   from prosemble.models import LVQMLN

   model = LVQMLN(
       n_prototypes_per_class=2,
       hidden_sizes=[64, 32],  # MLP architecture
       max_iter=100,
       lr=0.001,
   )
   model.fit(X_train, y_train)

Siamese Variants
----------------

Like LVQMLN, but prototypes are in the **input space** and pass through
the same backbone as input data. This makes prototypes interpretable.

.. code-block:: python

   from prosemble.models import SiameseGLVQ, SiameseGMLVQ, SiameseGTLVQ

   model = SiameseGLVQ(
       n_prototypes_per_class=2,
       hidden_sizes=[64, 32],
       max_iter=100,
       lr=0.001,
   )
   model.fit(X_train, y_train)
   # model.prototypes_ is in the original input space

Image LVQ
---------

Siamese architecture with a CNN backbone for image classification.

.. code-block:: python

   from prosemble.models import ImageGLVQ, ImageGMLVQ, ImageGTLVQ

   model = ImageGLVQ(
       n_prototypes_per_class=2,
       max_iter=50,
       lr=0.001,
   )
   # X_train should be image data: (n_samples, height, width, channels)
   model.fit(X_train, y_train)

CBC — Classification-By-Components
----------------------------------

Uses classless components with learned reasoning matrices for explainable
classification.

.. code-block:: python

   from prosemble.models import CBC

   model = CBC(
       n_components=6,
       n_classes=3,
       max_iter=100,
       lr=0.001,
   )
   model.fit(X_train, y_train)
