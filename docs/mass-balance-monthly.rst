.. currentmodule:: oggm

Monthly temperature index model calibrated on geodetic MB data
==============================================================

As of OGGM v1.6, the simplest and standard mass balance model available
in OGGM is a monthly temperature index model that can be calibrated on
any mass balance product (the default is `Hugonnet et al., 2021`_,
see :ref:`shop-geod`).

.. _Hugonnet et al., 2021: https://www.nature.com/articles/s41586-021-03436-z

The monthly mass balance :math:`B_i` at elevation :math:`z` is computed as:

.. math::

    B_i(z) = P_i^{Solid}(z) - d_f \, max \left( T_i(z) - T_{Melt}, 0 \right)

where :math:`P_i^{Solid}` is the monthly solid precipitation, :math:`T_i`
the monthly temperature and :math:`T_{Melt}` is the monthly mean air
temperature above which ice melt is assumed to occur (-1°C per default).
Solid precipitation is computed out of the total precipitation. The fraction of
solid precipitation is based on the monthly mean temperature: all solid below
``temp_all_solid`` (default: 0°C) and all liquid above ``temp_all_liq``
(default: 2°C), linear change in between. Total precipitation is obtained from
the climate dataset, multiplied by a precipitation correction factor :math:`P_f`.
The parameter :math:`d_f` indicates the temperature sensitivity of the
glacier, and it needs to be calibrated. The model needs to compute the
temperature and precipitation at the altitude :math:`z` of the glacier
grid points. The default is to use a fixed lapse rate of
-6.5K km :math:`^{-1}` and no gradient for precipitation.

.. _mb-calib:

Calibration
-----------

.. admonition:: **New in version 1.6!**

   A major change from previous OGGM versions, the calibration is now much
   more flexible, explicit, and adaptable. We recommend all users to
   spend some time in getting familiar with the calibration procedure in
   order to adapt it for their own purposes.

Visit the new `mass balance calibration tutorial <https://tutorials.oggm.org/master/notebooks/tutorials/massbalance_calibration.html>`_
for an overview.

Notes
-----

Although this mass balance model is a clear improvement to previous OGGM versions
(mostly, for better using observations and for getting rid of the
residual parameter :math:`\epsilon`), more physical approaches are possible.
Importantly, we need to take the uncertainty estimates into account, and
we need to tackle the issue of daily data for hydrological models.

More exigent users might have a look at `PyGEM <https://github.com/drounce/PyGEM>`_
or `OGGM sandbox <https://github.com/OGGM/massbalance-sandbox>`_, which are
offering clever ways to deal with these issues.
