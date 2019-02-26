.. _faq:

***********************
FAQ and Troubleshooting
***********************

We list here some of the questions we get most often, either on the
`OGGM Users <https://mailman.zfn.uni-bremen.de/cgi-bin/mailman/listinfo/oggm-users>`_
mailing list or on `github <https://github.com/OGGM/oggm>`_.

General questions
=================

What is the difference between OGGM and other glacier models?
-------------------------------------------------------------

There are plenty of established ice dynamics models, and some of them are
**open-source** (e.g. `PISM <http://www.pism-docs.org/wiki/doku.php>`_,
`Elmer/Ice <http://elmerice.elmerfem.org/>`_).

The purpose of OGGM is to be an *easy to use*, *fully automated*
**global glacier model**, i.e. applicable to any glacier in the
world without specific tuning or tweaking. Therefore, it does not attempt to
replace (and even less compete with) these established ice dynamics models:
it can be seen as a "framework", a
set of unified tools with eases the process of working with many mountain
glaciers at once.

There is a standard modelling chain in OGGM (with a mass-balance model
and a multiple flowline model) but there is no obligation to use all
of these tools. For example, we can easily picture a workflow where people will
use OGGM to create homogenized initial conditions (topography, climate) but
use a higher order dynamical model like PISM instead of the simplified OGGM
dynamics. For these kind of workflows, we created the
`OGGM-contrib <https://github.com/OGGM/oggmcontrib>`_ example package which
should help OGGM users to implement their own physics in OGGM.


Can I use OGGM to simulate long term glacier evolution?
-------------------------------------------------------

It depends what you mean by "long-term": at centenial time scales, yes. At
millenial time scales, maybe. At glacial time scales, probably not. The major
issue we have to face with OGGM is that it uses a "glacier-centric" approach:
it can simulate the mountain glaciers and ice-caps we know from contemporary
inventories, but it cannot simulate glaciers which existed before but have
disappeared until today.
Also, if glaciers grow into large ice complexes and ice caps, the
flowline assumption becomes much less valid than for typical valley glaciers
found today. For these situations, fully distributed models like PISM
are more appropriate.

Troubleshooting
===============

Some glaciers exit with errors. What should I do?
-------------------------------------------------

Many things can go wrong when simulating all the world glaciers with a single
model. We've tried our best, but still many glaciers cannot be simulated
automatically. Possible reasons include complex glacier geometries that cannot
be simulated by flowlines, very cold climates which don't allow melting to
occur, or numerical instabilities during the simulation. Altogether, 4218
glaciers (3.6% of the total area worldwide) could not be modelled by
OGGM in the
`standard global simulations <https://www.geosci-model-dev-discuss.net/gmd-2018-9/>`_.
Some regions experience more errors than others (see the paper).

When you experience errors, you have to decide if they are due to an error
in your code or a problem in OGGM itself. The number and type of errors
might help you out to decide if you want to act and troubleshoot them
(see below). Also, always keep in mind that the *number* of errors is less
important than the *glacier area* they represent. Plenty or errors on
small glaciers is not as bad as one large glacier missing.

Then, you have to carefully consider how to deal with missing glaciers. Most
studies will upscale diagnostic quantities using power laws or interpolation:
for example, use volume-area-scaling to compute the volume of glaciers that
are missing after an OGGM run. Importantly, you have to always be aware that
these quantities will be missing from the compiled run outputs, and should
be accounted for in quantitative analyses.


What does the "`Glacier exceeds domain boundaries`" error mean?
---------------------------------------------------------------

This happens when a glacier grows larger than the original map boundaries.
We recommend to increase the glacier map in this case, by setting
`cfg.PARAMS['border']` to a larger value, e.g. 100 or 200. The larger this
value, the larger the glacier can grow (the drawback is that simulations
become slowier and hungrier in memory because the number of grid points
increases as well). We do not recommend to go larger than 250, however:
for these cases it is likely that something else is wrong in your workflow
or OGGM itself.

What does the "`NaN in numerical solution`" error mean?
-------------------------------------------------------

This happens when the ice dynamics simulation is unstable. In OGGM we use an
adaptive time stepping scheme (which should avoid these kind of situations),
but we also implemented thresholds for small time steps: i.e. if a simulation
requires very small time steps we still use a larger one to avoid extremely
slow runs. These thresholds are "bad practice" but required for operational
reasons: when this happens, it is likely that the simulations blow up with
a numerical error. There is not much you can do here, unless maybe set your
own thresholds for small time steps (at the cost of computation time).
