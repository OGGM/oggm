Introduction
============


We illustrate with an example how the OGGM workflow is applied to the
Tasman Glacier in New Zealand (see figure below).
Here we describe shortly the purpose of each processing step, while more
details are provided in other sections:

Preprocessing
  The glacier outlines are extracted from a reference dataset (`RGI`_)
  and projected onto a local
  gridded map of the glacier (**Fig. a**). Depending on the
  glacier location, a suitable source for the topographical data is
  downloaded automatically and interpolated to the local grid.
  The spatial resolution of the map depends on the size of the glacier.

Flowlines
  The glacier centerlines are computed using a geometrical routing algorithm
  (**Fig. b**),
  then filtered and slightly modified to become glacier "flowlines"
  with a fixed grid spacing (**Fig. c**).

Catchment areas and widths
  The geometrical widths along the flowlines are obtained by intersecting the
  normals at each grid point with the glacier outlines and the tributaries'
  catchment areas. Each tributary and the main flowline has a catchment area,
  which is then used to correct the geometrical widths so that the flowline
  representation of the glacier is in close accordance with the actual
  altitude-area distribution of the glacier (**Fig. d**).

Climate data and mass-balance
  Gridded climate data (monthly temperature and precipitation) are interpolated
  to the glacier location and corrected for altitude at each flowline's grid
  point. A carefully calibrated temperature-index model is used to compute the
  mass-balance for any month in the past.

Ice thickness inversion
  Using the mass-balance data computed above and relying on mass-conservation
  considerations, an estimate of the ice flux along each glacier grid point cross-section
  is computed by making assumptions about the shape of the cross-section
  (parabolic, rectangular or trapezoid). Using the physics of ice flow and the shallow ice approximation, 
  the model then computes the thickness of the glacier along the flowlines and the total
  volume of the glacier (**Fig. e**).

Glacier evolution
  A dynamical flowline model is used to simulate the advance and retreat of the
  glacier under preselected climate time series. Here (**Fig. f**), a 120-yrs
  long random climate sequence leads to a glacier advance.

.. _RGI: https://www.glims.org/RGI/


.. admonition:: **New in version 1.4!**

   Since v1.4, OGGM now has another way to compute flowlines via
   **binned elevation bands** [Huss_Farinotti_2012]_. See
   :ref:`flowlines` for more details.

.. figure:: _static/ex_workflow.png
    :width: 100%
