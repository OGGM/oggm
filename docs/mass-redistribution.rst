Mass redistribution curve model
===============================

Since v1.5.3, OGGM comes with a parameterized version of the standard ice dynamics
model. Instead of solving for the shallow-ice approximation along the flowline, the
:py:class:`core.flowline.MassRedistributionCurveModel` uses empirically derived
mass-redistribution curves as described in
`Huss & Hock (2015) <https://www.frontiersin.org/articles/10.3389/feart.2015.00054/full>`_ or
`Rounce et al. (2020) <https://www.frontiersin.org/articles/10.3389/feart.2019.00331/full>`_.

Each year, the total glacier mass balance is redistributed along the glacier
so that the surface elevation change is zero at the glacier top and
maximal at the glacier terminus.

This simple model works remarkably well for retreating glacier situations over short periods
of time. However, the parameterization cannot realistically reproduce the terminus behavior
of glaciers that are in equilibrium or advancing.

The ``MassRedistributionCurveModel`` class is available for
sensitivity analyses and testing, but we still recommend to use the standard
flowline model in most cases.
