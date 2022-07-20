KERMIT V2.0
==================================

.. automodule:: kerMIT

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Welcome to the KERMIT code documentation.
KERMIT has been introduced in  `our EMNLP2020 paper`_, although it has a longer history.

.. _our EMNLP2020 paper: https://aclanthology.org/2020.emnlp-main.18/

KERMIT aims to  collaborate with other Neural Networks (for example, with Transformers) for exploiting structures in decision making and


.. figure:: /_static/_img/KermitPlusTranformer.png
   :scale: 50 %
   :name: fig-kermit
   :target: ../../_static/_img/KermitPlusTranformer.png
   :align: center
   :alt: KERMIT architecture

   KERMIT architecture

and for explaining how decisions are made using structures:

.. figure:: /_static/_img/KermitInterpretationPass.png
   :scale: 50 %
   :name: fig-kermit-explaination
   :target: ../../_static/_img/Teaser.png
   :align: center
   :alt: KERMIT architecture

   KERMIT Explanation Pass


kerMIT structure encoders
==================================

This package contains the Distributed Structure Encoders.
These are responsible for taking structured data and producing vectors
that are representing these structured data as their substructures in a reduced space (see the following figure).

.. figure:: /_static/_img/KermitDSE.png
   :scale: 50 %
   :name: fig-kermit_dse
   :target: ../../_static/_img/KermitDSE.png
   :align: center
   :alt: KERMIT's Generic Distributed Structure Encoder

   KERMIT's Generic Distributed Structure Encoder (or Embedder, what do you prefer?)


Distributed Structure Embedder (DSE)
------------------------------------

.. automodule:: kerMIT.structenc.dse
   :members:

Distributed Tree Embedder (DTE)
------------------------------------
.. automodule:: kerMIT.structenc.dte
   :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`