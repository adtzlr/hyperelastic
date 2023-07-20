Math
~~~~

..  warning::
    Shear terms are not doubled for strain-like tensors, instead all math operations take care of the reduced vector storage.

..  admonition:: Symmetric properties of dyadic products

    The minor **and** major-symmetric property indices whether the fourth-order tensor as a result of a dyadic product of two symmetric second-order tensors may be stored in reduced matrix storage. Special cases of minor but not major-symmetry and vice versa exist but are not shown here.

    +------------------+-------------------------------------------+-----------------------------------------+
    |     Function     | :math:`\boldsymbol{A} \ne \boldsymbol{B}` | :math:`\boldsymbol{A} = \boldsymbol{B}` |
    +==================+===========================================+=========================================+
    | :func:`.dya`     |                    ❌                     |                    ✔️                   |
    +------------------+-------------------------------------------+-----------------------------------------+
    | :func:`.cdya`    |                    ✔️                     |                    ✔️                   |
    +------------------+-------------------------------------------+-----------------------------------------+
    | :func:`.cdya_ik` |                    ❌                     |                    ❌                   |
    +------------------+-------------------------------------------+-----------------------------------------+
    | :func:`.cdya_il` |                    ❌                     |                    ❌                   |
    +------------------+-------------------------------------------+-----------------------------------------+

.. automodule:: hyperelastic.math
   :members:
   :undoc-members:
   :inherited-members:
