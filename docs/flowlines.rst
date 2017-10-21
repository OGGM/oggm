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

Centerlines
-----------

Computing the centerlines is the first task to run after the initialisation
of the local glacier directories and of the local topography.

Our algorithm is an implementation of the procedure described by
`Kienholz et al., (2014)`_. Appart from some minor changes (mostly the choice
of certain parameters), we stayed close to the original algorithm.

.. _Kienholz et al., (2014): http://www.the-cryosphere.net/8/503/2014/

The relevant task is :py:func:`tasks.compute_centerlines`:

.. ipython:: python

    @savefig plot_fls_centerlines.png width=80%
    graphics.plot_centerlines(gdir)

The glacier has a major centerline (the longest one), and
tributaries (in this case two ). The :py:class:`Centerline` objects are stored
as a list, the last one being
the major one. Navigation between inflows (can be more than one) and
outflow (only one or none) is facilitated by the ``inflows`` and
``flows_to`` attributes:

.. ipython:: python

    fls = gdir.read_pickle('centerlines')
    fls[0]  # a Centerline object
    # make sure the first flowline realy flows into the major one:
    assert fls[0].flows_to is fls[-1]

At this stage, the centerline coordinates are still defined on the original
grid, and they are not considered as "flowlines" by OGGM. A rather simple task
(:py:func:`tasks.initialize_flowlines`) converts them to flowlines which
now have a regular coordinate spacing along the flowline (which they will
keep for the rest of the workflow). The tail of the tributaries are cut
according to a distance threshold rule:

.. ipython:: python

    @savefig plot_fls_flowlines.png width=80%
    graphics.plot_centerlines(gdir, use_flowlines=True)


Downstream lines
----------------

For the glacier to be able to grow we need to determine the flowlines
downstream of the current glacier geometry. This is done by the
:py:func:`tasks.compute_downstream_line` task:


.. ipython:: python

    @savefig plot_fls_downstream.png width=80%
    graphics.plot_centerlines(gdir, use_flowlines=True, add_downstream=True)

The downsteam lines area also computed using a routing algorithm minimizing
the distance to cover and upward slopes.


Catchment areas
---------------

Each flowline has it's own "catchment area". These areas are computed
using similar flow routing methods as the one used for determining the
flowlines. Their role is to attribute each glacier pixel to the right
tributory (this will also influence the later computation of the glacier
widths).

.. ipython:: python

    tasks.catchment_area(gdir)
    @savefig plot_fls_catchments.png width=80%
    graphics.plot_catchment_areas(gdir)


Flowline widths
---------------

Finally, the glacier widths are computed in two steps.

First, we compute the geometrical width at each grid point. The width is drawn
from the intersection of a line normal to the flowline and either the glacier
or the catchment outlines (when there are tributaries):

.. ipython:: python

    tasks.catchment_width_geom(gdir)
    @savefig plot_fls_width.png width=80%
    graphics.plot_catchment_width(gdir)

Then, these geometrical widths are corrected so that the altitude-area
distribution of the "flowline-glacier" is as close as possible as the actual
distribution of the glacier using its full 2D geometry. This job is done by
the :py:func:`tasks.catchment_width_correction` task:

.. ipython:: python

    tasks.catchment_width_correction(gdir)
    @savefig plot_fls_width_cor.png width=80%
    graphics.plot_catchment_width(gdir, corrected=True)

Note that a perfect distribution is not possible since the sample size is
not the same between the "1.5D" and the 2D representation of the glacier.
OGGM deals with this by iteratively search for an altidute bin size which
ensures that both representations have at least one element for each bin.


Implementation details
----------------------

Shared setup for these examples:

.. literalinclude:: _code/prepare_flowlines.py
