.. currentmodule:: oggm

.. _mass-balance-2012-pergla:

Temperature index model calibrated on geodetic MB data
======================================================

As of OGGM v1.5.3, we implemented a simple mass-balance model that can be
calibrated on the geodetic mass-balance data from
`Hugonnet et al., 2021 <https://www.nature.com/articles/s41586-021-03436-z>_`
(see also: :ref:`shop-geod`).

This can be seen as a temporary improvement of the :ref:`mass-balance-2012`
until the `sandbox models <https://github.com/OGGM/massbalance-sandbox>`_ become
fully operational.

Like for the previous model, the monthly mass-balance :math:`B_i` at elevation :math:`z` is computed as:

.. math::

    B_i(z) = P_i^{Solid}(z) - \mu ^{*} \, max \left( T_i(z) - T_{Melt}, 0 \right)

where :math:`P_i^{Solid}` is the monthly solid precipitation, :math:`T_i`
the monthly temperature and :math:`T_{Melt}` is the monthly mean air
temperature above which ice melt is assumed to occur (-1°C per default).
Solid precipitation is computed out of the total precipitation. The fraction of
solid precipitation is based on the monthly mean temperature: all solid below
``temp_all_solid`` (default: 0°C) and all liquid above ``temp_all_liq``
(default: 2°C), linear change in between.

The difference with the previous model is that :math:`\epsilon` isn't needed
anymore, now that we can calibrate on each glacier individually.

Calibration
-----------

The glacier per glacier calibration is done for two parameters:

- the parameter :math:`\mu ^{*}` indicates the temperature sensitivity of the
  glacier, and is calibrated to match the geodetic mass-balance data over the
  reference period.
- if the resulting :math:`\mu ^{*}` is outside of predefined bounds
  (for example [20, 600] :math:`mm w.e. K^{-1} mth^{-1}`), then the temperature
  is bias corrected until a physically reasonable :math:`\mu ^{*}` is found.

Like for the previous model, several parameters ("hyper parameters") are
calibrated globally: ``temp_all_liq``, ``temp_all_solid``, :math:`T_{Melt}`,
and the precipitation correction factor :math:`P_f`.

How can I use this model instead of the old one?
------------------------------------------------

Since the model equations are the same, it can be used readily within OGGM.
Only the calibration steps are different (and simpler!). Have a look at our
tutorials for more information, or use our :ref:`preprodir` with precalibrated
parameters.

Notes
-----

Although this model is a clear improvement to the previous one (mostly,
for better matching observations and for getting rid of the residual :math:`\epsilon`),
more sensible approaches are possible. Importantly, we need to take the uncertainty
estimates into account, and we need to tackle the issue of the precipitation
correction.
More exigent users might have a look
at `PyGEM <https://github.com/drounce/PyGEM>`_ or
`OGGM sandbox <https://github.com/OGGM/massbalance-sandbox>`_, which are offering
ways to deal with these issues.
