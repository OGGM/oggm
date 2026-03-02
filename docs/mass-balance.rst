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
    The capacity to run OGGM with PyGEM as mass balance model and calibration tool is now available
    on the `PyGEM repository <https://github.com/PyGEM-Community/PyGEM>`_ and described in the
    `PyGEM documentation <https://pygem.readthedocs.io/en/latest/index.html>`_.

    OGGM v1.7 will include different climatic MB models. Experienced OGGM users can already test them via the `OGGM dev branch <https://github.com/OGGM/oggm/tree/dev>`_ (some example use cases are available at the 
    `digital twin component for glaciers notebooks <https://notebooks.dtcglaciers.org/welcome.html>`_). 
    The OGGM dev branch MB model options are based on the `OGGM's massbalance-sandbox <https://github.com/OGGM/massbalance-sandbox>`_. 

Check out the following resources for more information:

* :doc:`mass-balance-monthly`
* :doc:`mass-balance-16guide`
* :doc:`mass-balance-toys`



.. toctree::
    :maxdepth: 1
    :hidden:

    mass-balance-monthly.rst
    mass-balance-16guide.rst
    mass-balance-toys.rst

