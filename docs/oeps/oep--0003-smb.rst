.. _oep0003:

===========================================
OEP-0003: Surface mass-balance enhancements
===========================================

:Authors: Fabien Maussion
:Status: Draft - not implemented
:Created: 28.08.2019


Abstract
--------

We present a list of possible enhancements to the OGGM mass-balance model(s).
Each of them can be tackled separately, but it could make sense to address
some of them together, since it is quite an involved endeavor.

Motivation
----------

OGGM's mass-balance (MB) model is a temperature index model first developed by
`Marzeion et al. (2012) <https://www.the-cryosphere.net/6/1295/2012/>`_
and adapted for OGGM (e.g. to be distributed according to elevation along
the flowlines). The important aspect of our MB model is the calibration of the
temperature sensitivity, which is... peculiar to say the least.
See :ref:`mass-balance` for an  illustration of the method.

This method is powerful, but also has caveats (some are listed below).
Furthermore, it has not changed since 2012, and could make much better use of
newly available data: mostly, geodetic mass-balance for a much larger number of
glaciers.

Proposed improvements
---------------------

Varying temperature sensitivities for snow and ice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Rationale**

Currently, the temperature sensitivity :math:`\mu^{*}` (or "melt factor", units
mm w.e. yr-1 K-1) is the same all over the glacier. There are good reasons
to assume that this melt factor should be different for different surface
conditions.

One relatively simple way to deal with it woud be to define a new model
parameter, ``snow_melt_factor``, which defines a temperature
sensitivity for snow as :math:`\mu^{*}_{Snow} = f \, \mu^{*}_{Ice}` with
:math:`f` constant and somewhere between 0 and 1 (1 would be the current
default).

**Implementation**

The implementation is not as straightforward as it sounds, but should be
feasible. The main culprits are:

- one will need to track snow cover and snow age with time, and transform
  snow to ice after some years.
- the calibration procedure will become a chicken and egg problem, since
  snow cover evolution will depend on :math:`\mu^{*}`, which will itself depend
  on snow cover evolution. Possibly, this will need to a relatively costly
  iterative procedure.

**Calibration / Validation**

This will introduce a new parameter, which should be constrained. Ideally,
it would be fit to observations of MB profiles from the WGMS.


Find a sensible algorithm to avoid the interpolation of t*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO

Make use of available geodetic MB data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO

Use Bayes
~~~~~~~~~

TODO
