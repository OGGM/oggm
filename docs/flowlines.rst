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
of certain parameters), we stayed very close to the original paper.

.. _Kienholz et al., (2014): http://www.the-cryosphere.net/8/503/2014/

The relevant task is :py:func:`tasks.compute_centerlines`:

.. ipython:: python

    tasks.compute_centerlines(gdir)
    @savefig plot_fls_centerlines.png width=80%
    graphics.plot_centerlines(gdir)

Each of the three divides has a major centerline (always the longest one), and
sometimes tributaries (in this case only one for the largest divide). The
:py:class:`Centerline` objects are stored as a list, the last one being
the major one. Navigation between inflows (can be more than one) and
outflow (only one or none) is facilitated by the ``inflows`` and
``flows_to`` attributes:

.. ipython:: python

    fls = gdir.read_pickle('centerlines', div_id=2)
    fls[0]  # a Centerline object
    # make sure the first flowline flows into the major one:
    assert fls[0].flows_to is fls[-1]

At this stage, the centerlines coordinates are still defined on the original
grid, and they are not considered as "flowlines" by OGGM. A rather simple task
(:py:func:`tasks.initialize_flowlines`) converts them to
:py:class:`InversionFlowline` objects. These flowlines now have a regular
corrdinate spacing along the flowline (which they will keep for the rest of
the workflow), and the tail of the tributaries are cut
according to a simple rule:

.. ipython:: python

    tasks.initialize_flowlines(gdir)
    @savefig plot_fls_flowlines.png width=80%
    graphics.plot_centerlines(gdir, use_flowlines=True)


Downstream lines
----------------

For the glacier to be able to grow we need to determine the flowlines
downstream of the current glacier geometry. This is done by the
:py:func:`tasks.compute_downstream_lines` task:


.. ipython:: python

    tasks.compute_downstream_lines(gdir)
    @savefig plot_fls_downstream.png width=80%
    graphics.plot_centerlines(gdir, add_downstream=True)

Note that the task also determines the new tributaries originating from the
glacier divides (while the concept of divides is necessary for the
preprocessing, all divides are then merged to make one glacier for the
actual run).


Catchment areas
---------------

Each tributary flowline has it's own "catchment area". These areas are computed
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
