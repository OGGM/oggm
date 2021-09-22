.. currentmodule:: oggm

.. _mass-balance:

Mass-balance models
===================

Per design, OGGM allows to mix and compare many different implementations
of climatic mass-balance (MB) models. These models all follow a predefined
"interface" (programmin jargon for *naming conventions*) so that they can
communicate with the :ref:`geometry-evolution`.

In fact, what OGGM calls a "mass-balance model" is any object that provides
surface mass-balance information to the geometry evolution model for the
run. Therefore, while some of these MB models can be quite complex and take
many processes into account, other model classes can implement idealized
concepts, averages or simplifications of the more realistic ones.

Here are some of the options available to our users to compute the mass-balance:

.. toctree::
    :maxdepth: 1

    mass-balance-2012.rst
    mass-balance-2012-pergla.rst
    mass-balance-toys.rst
    PyGEM <https://github.com/drounce/PyGEM>
    OGGM's massbalance-sandbox <https://github.com/OGGM/massbalance-sandbox>

.. admonition:: **Out of the oven!**

    PyGEM is a standalone model that has been applied to High Mountain Asia (Rounce et al.,
    `2020a <https://doi.org/10.1017/jog.2019.91>`_,
    `2020b <https://doi.org/10.3389/feart.2019.00331>`_). David and the
    OGGM team have worked extensively to make PyGEM's MB model
    compatible with the OGGM workflow. This feature is now available on the
    PyGEM repository.

    `OGGM's massbalance-sandbox <https://github.com/OGGM/massbalance-sandbox>`_
    is the future generation of OGGM's climatic MB models. They are currently
    in the development and testing phase, but they can readily be used with
    a recent OGGM version.
