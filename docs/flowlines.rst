.. currentmodule:: oggm

.. _flowlines:

Glacier flowlines
=================


.. ipython:: python
   :suppress:

    fpath = "_code/prepare_flowlines.py"
    with open(fpath) as f:
        code = compile(f.read(), fpath, 'exec')
        exec(code)

.. ipython:: python
   :suppress:

    from oggm import graphics

Centerlines
-----------

Computing the centerlines is the first task to run after the definition
of the local map and topography.

Our algorithm is an implementation of the procedure described by
`Kienholz et al., (2014)`_. Appart from some minor changes (mostly the choice
of certain parameters), we stayed close to the original algorithm.

.. _Kienholz et al., (2014): http://www.the-cryosphere.net/8/503/2014/


The basic idea is to find the terminus of the glacier (its lowest point) and
a series of flowline "heads" (local elevation maxima). The centerlines are then
computed with a least cost routing algorithm minimizing both (i) the total
elevation gain and (ii) the distance from the glacier outline:

.. ipython:: python

    @savefig plot_fls_centerlines.png width=80%
    graphics.plot_centerlines(gdir)

The glacier has a major centerline (the longest one), and
tributary branches (in this case: two).

At this stage, the centerlines are still not fully suitable
for modelling. Therefore, a rather simple
procedure converts them to "flowlines", which
now have a regular coordinate spacing (which they will
keep for the rest of the workflow). The tail of the tributaries are cut
according to a distance threshold rule:

.. ipython:: python

    @savefig plot_fls_flowlines.png width=80%
    graphics.plot_centerlines(gdir, use_flowlines=True)


Downstream lines
----------------

For the glacier to be able to grow we need to determine the flowlines
downstream of the current glacier geometry:

.. ipython:: python

    @savefig plot_fls_downstream.png width=80%
    graphics.plot_centerlines(gdir, use_flowlines=True, add_downstream=True)

The downsteam lines area also computed using a routing algorithm minimizing
the distance between the glacier terminus and the border of the map as well
as the total elevation gain, therefore following the valley floor.


Catchment areas
---------------

Each flowline has its own "catchment area". These areas are computed
using similar flow routing methods as the one used for determining the
flowlines. Their purpose is to attribute each glacier pixel to the right
tributory in order to compute mass gain and loss for each tributary.
This will also influence the later computation of the glacier widths).

.. ipython:: python

    tasks.catchment_area(gdir)
    @savefig plot_fls_catchments.png width=80%
    graphics.plot_catchment_areas(gdir)


Flowline widths
---------------

Finally, the flowline widths are computed in two steps.

First, we compute the geometrical width at each grid point. The width is drawn
from the intersection of a line perpendicular to the flowline and either (i)
the glacier outlines or (ii) the catchment boundaries:

.. ipython:: python

    tasks.catchment_width_geom(gdir)
    @savefig plot_fls_width.png width=80%
    graphics.plot_catchment_width(gdir)

Then, these geometrical widths are corrected so that the altitude-area
distribution of the "flowline-glacier" is as close as possible as the actual
distribution of the glacier using its full 2D geometry:

.. ipython:: python

    tasks.catchment_width_correction(gdir)
    @savefig plot_fls_width_cor.png width=80%
    graphics.plot_catchment_width(gdir, corrected=True)

Note that a perfect match is not possible since the sample size is
not the same between the "1.5D" and the 2D representation of the glacier.


Implementation details
----------------------

Shared setup for these examples:

.. literalinclude:: _code/prepare_flowlines.py
