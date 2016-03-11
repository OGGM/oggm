ITMIX Experiment 2016
=====================

Author: F. Maussion

Date: 11.03.2016

OGGM participates to the `ITMIX`_ experiment organised by the IACS
`Working Group`_ on Glacier Ice Thickness Estimation.

.. _ITMIX: http://people.ee.ethz.ch/~danielfa/IACS/register.html
.. _Working Group: http://www.cryosphericsciences.org/wg_glacierIceThickEst.html

The idea is to compare models able to provide a distributed estimate of
ice thickness for 18 glaciers worldwide.

.. figure:: https://dl.dropboxusercontent.com/u/20930277/itmix_public/0_globmap.png?raw=1
    :width: 100%

The deadline for the experiment was February 29th. Definitely too early
for OGGM, with which we had performed the inversion for the European Alps
only. We still didn't want to miss this opportunity and started an
intense `development phase`_ to make OGGM applicable globally. After two
weeks of intense work we are now able to provide an estimate for all
glaciers except Antarctica. While this surely marks an important step in the
development of OGGM, this project again raised many questions and research
ideas, since (as you will see below), nothing is easy when doing global
scale distributed modelling.

.. _development phase: https://github.com/OGGM/oggm/pull/36


ITMIX preprocessing
-------------------

The glaciers are heterogeneous: valley glaciers, ice-caps,
marine-terminating fronts... In addition, the input data was not (entirely)
standardized.

.. image:: https://dl.dropboxusercontent.com/u/20930277/itmix_public/Devon.png?raw=1
    :width: 49%
.. image:: https://dl.dropboxusercontent.com/u/20930277/itmix_public/Elbrus.png?raw=1
    :width: 49%
.. image:: https://dl.dropboxusercontent.com/u/20930277/itmix_public/Unteraar.png?raw=1
    :width: 49%
.. image:: https://dl.dropboxusercontent.com/u/20930277/itmix_public/Aqqutikitsoq.png?raw=1
    :width: 49%
.. image:: https://dl.dropboxusercontent.com/u/20930277/itmix_public/Austfonna.png?raw=1
    :width: 49%
.. image:: https://dl.dropboxusercontent.com/u/20930277/itmix_public/Urumqi.png?raw=1
    :width: 49%

*Blue: ITMIX outlines, Black: RGI outlines. White means that ITMIX didn't
provide topography for these points.*

The first step that we needed to do is to formalize all this data so that
OGGM can deal with it. This turned out to be a bit complicated but we found
a compromise:

- we updated the RGI outlines with the ITMIX ones where possible
- for ice-caps, we kept the RGI outlines because OGGM currently needs the
  "pieces of cake" to compute the centerlines, and ITMIX provided only the
  whole ice-cap
- with still use the OGGM working maps to do the inversion. This is
  important because OGGM makes decisions about the grid spacing it uses, and
  also the entire workflow is depending on these standardized local maps.
  We introduced a new routine in the pipeline in order to update the
  SRTM/ASTER topography with the ITMIX one. This also was not trivial because
  some ITMIX topographies are stopping directly at the glacier boundary.


OGGM preprocessing
------------------

Topography
~~~~~~~~~~

Before ITMIX, OGGM was not able to make a local glacier map automatically.
Now, OGGM searches for the DEM data automatically:

- SRTM V4.1 for [60S, 60N] (http://srtm.csi.cgiar.org/)
- GIMP DEM for Greenland (https://bpcrc.osu.edu/gdg/data/gimpdem)
- Corrected DEMs for Svalbard, Iceland, and North Canada
  (http://viewfinderpanoramas.org/dem3.html)
- ASTER V2 elsewhere

The corrected DEMs where necessary because ASTER data has many issues over
glaciers. Take for example the DEM for two glaciers in Iceland:

.. image:: https://dl.dropboxusercontent.com/u/20930277/itmix_public/wgms_dyngjujoekull_rgi50-06.00477_dom.png?raw=1
    :width: 49%
.. image:: https://dl.dropboxusercontent.com/u/20930277/itmix_public/rgi50-07.01394.png?raw=1
    :width: 49%

Note that the hypsometry provided in RGI V5 also contains these errors.
While the problems with the right plot are obvious, the glacier on the left
(*Dyngjujoekull*) is practically impossible to filter automatically. On the
plot below, I show the hypsometry that OGGM computed and the one by Mathias
Huss:

.. image:: https://dl.dropboxusercontent.com/u/20930277/itmix_public/hypso_rgi50-06.00477.png?raw=1
    :width: 60%

Up to a few discrepancies due to projection issues, we both have the problem
of non-zero bins below 750 m a.s.l. Fortunately, thanks to the work by
`Jonathan de Ferranti`_, these problems are now resolved in OGGM:

.. _Jonathan de Ferranti: http://viewfinderpanoramas.org/dem3.html

.. image:: https://dl.dropboxusercontent.com/u/20930277/itmix_public/w_dyngjujoekull_rgi50-06.00477_cls.png?raw=1
    :width: 60%

There is potential for even better coverage of corrected DEM, but this would
require a bit more work (J. de Ferranti's data is not always logically
structured).


Calibration data
~~~~~~~~~~~~~~~~

Another obstacle to global coverage was that the databases required for
calibration (WGMS FoG and GlaThiDa) are not "linked" to RGI, i.e. there is
no way to know which RGI entity corresponds to each database entry. Thanks
to the work of `Johannes Landmann`_ at UIBK we now have comprehensive links
with global coverage.

.. _`Johannes Landmann`: https://github.com/OGGM/databases-links

Some of the ITMIX glaciers were in the database, I removed them manually for
the sake of the experiment ("blind run" - in a second run I will use the
mass-balance data since it is authorized by ITMIX):

- WGMS: Kesselwandferner, Brewster, Devon, Elbrus, Freya, Hellstugubreen,
  Urumqi
- GlaThiDa: Kesselwandferner, Unteraar

These leaves us with 196 WGMS glaciers with at least 5 years of mass-balance
data available for calibration of the mass-balance:

.. figure:: https://dl.dropboxusercontent.com/u/20930277/itmix_public/globmap_wgms.png?raw=1
    :width: 100%

And 143 GlaThiDa glaciers with glacier-wide average thickness estimates. The
coverage of GlaThiDa is not very good, and this might become an issue:

.. figure:: https://dl.dropboxusercontent.com/u/20930277/itmix_public/globmap_glathida.png?raw=1
    :width: 100%


Inversion procedure
-------------------

Refer to the general `documentation`_ for details about the inversion
procedure.

.. _documentation: http://oggm.org

Here we go directly to the calibration results of the ice-thickness
inversion. OGGM currently has only one free parameter to tune for the ice
thickness inversion: Glen's creep parameter*A*. This is very similar to
[Farinotti_etal_2009]_, with the difference that we are not calibrating *A*
for each glacier individually: we tried to do that, but didn't manage (yet).

Land-terminating glaciers
~~~~~~~~~~~~~~~~~~~~~~~~~

Currently, *A* is varied until the glacier-wide thickness computed by OGGM
minimizes the RMSD with GlaThiDa. We removed all ice-caps and
marine-terminating glaciers from the dataset in
order to avoid these specific cases (135 glaciers left). Here are the
results of the calibration (left: volume-area scaling, right: OGGM):

.. figure:: https://dl.dropboxusercontent.com/u/20930277/itmix_public/scatter_all_glaciers.png?raw=1
    :width: 100%

We can see that OGGM has a slghtly lower score than volume-area scaling
(VAS). This is due to the presence of a couple of outliers. In particular,
the thickest glacier in GlaThiDa and VAS is strinkingly thin in OGGM: this
is the  Black Rapids glacier in Canada, which is quite well mentione in the
literature because it is a surging glacier. Closer inspection in OGGM
reveals that this glacier has one of the lowest mass-balance gradient. CRU
precipitation is 849 mm yr :math:`^{1}` (after application of the 2.5
correction factor!), which I assume is too low.

We recall here that one central difference between our approach and that of
[HussFarinotti_2012]_ is that we use real climate data to compute the
apparent mass-balance, and thus have glacier specific mass-balance gradients.
This is a strength but can also become a burden. The mass-balance gradient
depends mostly on precipitation, but also on temperature and its seasonal
cycle. Here I show the apparent mass-balance gradient in the ablation area
of all GlaThiDa glaciers:

.. figure:: https://dl.dropboxusercontent.com/u/20930277/itmix_public/mb_grad.png?raw=1
    :width: 100%

Is the MB gradient related to the error OGGM makes in comparison to GlaThiDa?

.. figure:: https://dl.dropboxusercontent.com/u/20930277/itmix_public/rel_error.png?raw=1
    :width: 100%

No not really. And this is similar with all other glacier characteristics I
could look at until now. The error that OGGM makes is not easily
attributable to specific causes... It it would, that would be great! Indeed,
this would allow maybe to define a rule for the calibration factor *A*. If
you have ideas at which parameter to look at, let me know!

VAS vs OGGM
^^^^^^^^^^^

I don't want the calibration to be too altered by the Black Rapids outlier,
so I removed it from the calibration set. The plot now looks better, and
after this little hack OGGM is even *slightly* better than VAS:

.. figure:: https://dl.dropboxusercontent.com/u/20930277/itmix_public/scatter_all_glaciers_no_rapids.png?raw=1
    :width: 100%

What is really interesting however is that OGGM and VAS are incredibly
similar in their dissimilarity with GlaThiDa. So similar that if we plot the
two approaches together on a scatter, one could argue that the thousands of
lines of code of OGGM really aren't worth the effort ;). I think that I have
to talk to David Bahr about this, he will surely be pleased.

.. figure:: https://dl.dropboxusercontent.com/u/20930277/itmix_public/vas_vs_oggm.png?raw=1
    :width: 50%


Conclusions?
^^^^^^^^^^^^

We find an optimised factor for *A* of 3.03 (i.e. the inversion *A* is three
times larger than the standard of 2.4e-24 [Cuffey_Paterson_2010]_). This
makes sense since we do not consider the sliding velocity in the inversion,
which means that we need an ice which is less stiff to compensate.


Marine-terminating glaciers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

What happens if we apply this optimized *A* to the 6 marine-terminating
glaciers we have in GlaThiDa? They are all in Svalbard, but this still can
give an idea:

.. figure:: https://dl.dropboxusercontent.com/u/20930277/itmix_public/scatter_marine.png?raw=1
    :width: 100%

As expected, using the same method for calving glaciers leads to
overestimated thicknesses. OGGM however has not such a clear signal like
VAS: either the glaciers are not calving too much, or OGGM thicknesses are
underestimated (which is probable, maybe OGGM underestimates very large
glaciers as a whole).


References
----------

.. [Cuffey_Paterson_2010] Cuffey, K., and W. S. B. Paterson (2010).
    The Physics of Glaciers, Butterworth‐Heinemann, Oxford, U.K.

.. [Farinotti_etal_2009] Farinotti, D., Huss, M., Bauder, A., Funk, M., &
    Truffer, M. (2009). A method to estimate the ice volume and
    ice-thickness distribution of alpine glaciers. Journal of Glaciology, 55
    (191), 422–430.

.. [HussFarinotti_2012] Huss, M., & Farinotti, D. (2012). Distributed ice
   thickness and volume of all glaciers around the globe. Journal of
   Geophysical Research: Earth Surface, 117(4), F04010.