.. currentmodule:: oggm

Mass balance models
===================

OGGM allows to mix and compare many different implementations
of climatic mass balance (MB) models. These models all follow a predefined
"interface" so that they can communicate with the :doc:`geometry-evolution`.

**In fact, what OGGM calls a "mass balance model" is any python function that
is able to provide annual surface mass balance information to the geometry evolution
model for the duration of the run**.

Therefore, while some mass balance models can be quite complex and take
many physical processes into account, other model classes can implement idealized
concepts, random samples or simplifications of the more realistic ones.

.. admonition:: **Check out these recent developments!**

    PyGEM is a standalone model that has been applied to High Mountain Asia
    (Rounce et al.,
    `2020a <https://doi.org/10.1017/jog.2019.91>`_,
    `2020b <https://doi.org/10.3389/feart.2019.00331>`_). David and the
    OGGM team have worked extensively to make PyGEM's MB model
    compatible with the OGGM workflow, resulting in this
    `global PyGEM-OGGM study published in Science <https://doi.org/10.1126/science.abo1324>`_.
    The capacity to run OGGM with PyGEM as mass balance model is now available
    on the PyGEM repository.

    `OGGM's massbalance-sandbox <https://github.com/OGGM/massbalance-sandbox>`_
    is the future generation of OGGM's climatic MB models. They are currently
    in the development and testing phase, but they can readily be used with
    a recent OGGM version.

Check out the following resources for more information:

* :doc:`mass-balance-monthly`
* :doc:`mass-balance-16guide`
* :doc:`mass-balance-toys`
* The `massbalance-sandbox <https://github.com/OGGM/massbalance-sandbox>`_ repository


.. toctree::
    :maxdepth: 1
    :hidden:

    mass-balance-monthly.rst
    mass-balance-16guide.rst
    mass-balance-toys.rst
    massbalance-sandbox <https://github.com/OGGM/massbalance-sandbox>