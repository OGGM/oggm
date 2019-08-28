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
some them together as well.

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

Find a sensible algorithm to avoid the interpolation of t*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Make use of available geodetic MB data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calibration and validation
~~~~~~~~~~~~~~~~~~~~~~~~~~

