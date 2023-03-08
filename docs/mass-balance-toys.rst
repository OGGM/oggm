.. currentmodule:: oggm

"Toy models" for experimentation
================================

OGGM comes with a suite of simplified models allowing to run idealized or
simplified experiments very useful for testing or teaching. Like all
other mass balance models in OGGM, they follow the
:py:class:`~oggm.MassBalanceModel` interface.

.. autosummary::
    :toctree: generated/
    :nosignatures:

    core.massbalance.ScalarMassBalance
    core.massbalance.LinearMassBalance
    core.massbalance.ConstantMassBalance
    core.massbalance.RandomMassBalance
    core.massbalance.UncertainMassBalance
