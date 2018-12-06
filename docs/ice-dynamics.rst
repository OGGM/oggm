Ice dynamics
============

The glaciers in OGGM are represented by a depth integrated flowline
model. The equations for the isothermal shallow ice are solved along
the glacier centerline, computed to represent best the flow of ice
along the glacier (see for example `antarcticglaciers.org`_ for a general
introduction about the various type of glacier models).

.. _antarcticglaciers.org: http://www.antarcticglaciers.org/glaciers-and-climate/numerical-ice-sheet-models/hierarchy-ice-sheet-models-introduction/

.. _ice-flow:

Ice flow
--------

Let :math:`S` be the area of a cross-section perpendicular to the
flowline. It has a width :math:`w` and a thickness :math:`h` and, in this
example, a parabolic bed shape.

    .. figure:: _static/hef_flowline.jpg
        :width: 80%

        Example of a cross-section along the glacier flowline. Background
        image from
        http://www.swisseduc.ch/glaciers/alps/hintereisferner/index-de.html

Volume conservation for this discrete element implies:

.. math::

    \frac{\partial S}{\partial t} = w \, \dot{m} - \nabla \cdot q

where :math:`\dot{m}` is the mass-balance, :math:`q = u S` the flux of ice, and
:math:`u` the depth-integrated ice velocity ([Cuffey_Paterson_2010]_, p 310).
This velocity can be computed from Glen's flow law as a function of the
basal shear stress :math:`\tau`:

.. math::

    u = u_d + u_s = f_d h \tau^n + f_s \frac{\tau^n}{h}

The second term is to account for basal sliding, see e.g. [Oerlemans_1997]_ or
[Golledge_Levy_2011]_. It introduces an additional free parameter :math:`f_s`
and will therefore be ignored in a first approach. The deformation parameter
:math:`f_d` is better constrained and relates to Glen's
temperature‐dependent creep parameter :math:`A`:

.. math::

    f_d = \frac{2 A}{n + 2}

The basal shear stress :math:`\tau` depends e.g. on the geometry of the bed
[Cuffey_Paterson_2010]_. Currently it is assumed to be
equal to the driving stress :math:`\tau_d`:

.. math::

    \tau_d = \alpha \rho g h

where :math:`\alpha` is the slope of the flowline and :math:`\rho` the density
of ice. Both the ``FluxBasedModel`` and the ``MUSCLSuperBeeModel`` solve
for these equations, but with different numerical schemes.


Bed shapes
----------

OGGM implements a number of possible bed-shapes. Currently the shape has no
direct influence on the shear stress (i.e. Cuffey and Paterson's "shape factor"
is not considered), but the shape will still have a considerable influence
on glacier dynamics:

- the width change as a result of mass transport will be different for
  each shape, thus influencing the mass balance :math:`w \, \dot{m}`
- with all other things held constant, a change in section area
  :math:`\partial S / \partial t` due to mass convergence/divergence
  will result in a different :math:`\partial h / \partial t` and thus in
  different shear stress computation at the next time step.


Rectangular
~~~~~~~~~~~


    .. figure:: _static/bed_vertical.png
        :width: 40%


The simplest shape. The glacier width does not change with ice thickness.


Trapezoidal
~~~~~~~~~~~


    .. figure:: _static/bed_trapezoidal.png
        :width: 40%


Trapezoidal shape with two degrees of freedom. The width change with thickness
depends on :math:`\lambda`. [Golledge_Levy_2011]_ uses :math:`\lambda = 1`
(a 45° wall angle).


Parabolic
~~~~~~~~~


    .. figure:: _static/bed_parabolic.png
        :width: 40%


Parabolic shape with one degree of freedom, which makes it particularly
useful for the bed inversion: if :math:`S` and :math:`w` are known:

.. math::

    h = \frac{3}{2} \frac{S}{w}

The parabola is defined by the bed-shape parameter
:math:`P_s = 4 h / w^2` [1]_. Very small values of this parameter imply very
`flat` shapes, unrealistically sensitive to changes in :math:`h`. For this
reason, the default in OGGM is to use the mixed flowline model.

.. [1] the local thickness :math:`y`  of the parabolic bed can be described by
    :math:`y = h − P_s x^2`. At :math:`x = w / 2`, :math:`y = 0` and
    therefore :math:`P_s = 4 h / w^2`.


Mixed
~~~~~

A combination of rectangular, trapezoidal and parabolic shapes.
The default is parabolic, but can be adapted in two cases:

- if the glacier section touches an ice-divide or a neighboring tributary
  catchment outline, the bed is considered to be rectangular;
- if the parabolic shape parameter :math:`P_s` is below a certain threshold,
  a trapezoidal shape is used. Indeed, flat parabolas tend to be very
  sensitive to small changes in :math:`h`, which is undesired.


Numerics
--------

"Flux based" model
~~~~~~~~~~~~~~~~~~

Most flowline models treat the volume conservation equation as a
diffusion problem, taking advantage of the robust numerical solutions
available for this type of equations. The problem with this approach is that
it implies developing the :math:`\partial S / \partial t` term to solve for
ice thickness :math:`h` directly, thus implying different diffusion equations
for different bed geometries (e.g. [Oerlemans_1997]_ with a trapezoidal bed).

The OGGM "flux based" model solves for the :math:`\nabla \cdot q` term
(hence the name). The strong advantage of this method is that
the numerical equations are the same for *any* bed shape, considerably
simplifying the implementation. Similar to the "diffusion approach", the
model looses mass-conservation in very steep slopes ([Jarosch_etal_2013]_).

The numerical scheme implemented in OGGM is tested against A. Jarosch's
MUSCLSuperBee Model (see below) and Hans Oerleman's diffusion model for
various idealized cases. For all cases but the steep slope one the model
performs very well.

In order to increase the stability and speed of the computations, we solve the
numerical equations on a forward staggered grid and we use an adaptive time
stepping scheme.


MUSCLSuperBeeModel
~~~~~~~~~~~~~~~~~~

A shallow ice model with improved numerics ensuring mass-conservation in
very steep walls [Jarosch_etal_2013]_. The model is currently in
development to account for various bed shapes and tributaries and will
likely become the default in OGGM.


Glacier tributaries
-------------------

Glaciers in OGGM have a main centerline and, sometimes, one or more
tributaries (which can themselves also have tributaries, see
:ref:`flowlines`). The number of these tributaries depends on many
factors, but most of the time the algorithm works well.

The main flowline and its tributaries are all modelled individually.
At the end of a time step, the tributaries will transport mass to the branch
they are flowing to. Numerically, this mass transport is
handled by adding an element at the end of the flowline with the same
properties (with, thickness...) as the last grid point, with the difference
that the slope :math:`\alpha` is computed with respect to the altitude of
the point they are flowing to. The ice flux is then computed as usual and
transferred to the downlying branch.

The computation of the ice flux is always done first from the lowest order
branches (without tributaries) to the highest ones, ensuring a correct
mass-redistribution. The use of the slope between the tributary and main branch
ensures that the former is not dynamical coupled to the latter. If the angle is
positive or if no ice is present at the end of the tributary,
no mass exchange occurs.


References
----------

.. [Cuffey_Paterson_2010] Cuffey, K., and W. S. B. Paterson (2010).
    The Physics of Glaciers, Butterworth‐Heinemann, Oxford, U.K.

.. [Farinotti_etal_2009] Farinotti, D., Huss, M., Bauder, A., Funk, M., &
    Truffer, M. (2009). A method to estimate the ice volume and
    ice-thickness distribution of alpine glaciers. Journal of Glaciology, 55
    (191), 422–430.

.. [Golledge_Levy_2011] Golledge, N. R., and Levy, R. H. (2011).
    Geometry and dynamics of an East Antarctic Ice Sheet outlet glacier, under
    past and present climates. Journal of Geophysical Research:
    Earth Surface, 116(3), 1–13.

.. [Jarosch_etal_2013] Jarosch, a. H., Schoof, C. G., & Anslow, F. S. (2013).
    Restoring mass conservation to shallow ice flow models over complex
    terrain. Cryosphere, 7(1), 229–240. http://doi.org/10.5194/tc-7-229-2013

.. [Oerlemans_1997] Oerlemans, J. (1997).
    A flowline model for Nigardsbreen, Norway:
    projection of future glacier length based on dynamic calibration with the
    historic record. Journal of Glaciology, 24, 382–389.
