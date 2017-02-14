.. currentmodule:: oggm

.. _mass-balance:

Mass-balance
============

The mass-balance (MB) model implemented in OGGM is an extended version of the
temperature index model presented by `Marzeion et al., (2012)`_.
While the equation governing the mass-balance is that of a traditional
temperature index model, our approach to calibration requires that we spend
some time describing it.

.. _Marzeion et al., (2012): http://www.the-cryosphere.net/6/1295/2012/tc-6-1295-2012.html

.. ipython:: python
   :suppress:

    fpath = "_code/prepare_climate.py"
    with open(fpath) as f:
        code = compile(f.read(), fpath, 'exec')
        exec(code)



Climate data
------------

The MB model implemented in OGGM needs monthly time series of temperature and
precipitation. The current default is to download and use the `CRU TS v3.24`_
data provided by the Climatic Research Unit of the University of East Anglia.

.. _CRU TS v3.24: https://crudata.uea.ac.uk/cru/data/hrg/


CRU (default)
~~~~~~~~~~~~~

If not specified otherwise, OGGM will automatically download and unpack the
latest dataset from the CRU servers.

.. warning::

    While the downloaded zip files are ~370mb in size, they are ~5.6Gb large
    after decompression!

The raw, coarse (0.5°) dataset is then downscaled to a higher resolution grid
(CRU CL v2.0 at 10' resolution) following the anomaly mapping approach
described by Tim Mitchell in his `CRU faq`_ (Q25). Note that we don't expect
this downscaling to add any new information than already available at the
original resolution, but this allows us to have an elevation-dependent dataset,
from which we can compute the temperature at the elevation of the glacier.

.. _CRU faq: https://crudata.uea.ac.uk/~timm/grid/faq.html


User-provided dataset
~~~~~~~~~~~~~~~~~~~~~

You can provide any other dataset to OGGM by setting the ``climate_file``
parameter in ``params.cfg``. The format of the file is not (yet) very flexible:
see the HISTALP data file in the `sample-data`_ folder for an example.

.. _sample-data: https://github.com/OGGM/oggm-sample-data/tree/master/test-workflow


.. ipython:: python

    @savefig plot_temp_ts.png width=100%
    example_plot_temp_ts()  # the code for these examples is posted below


Elevation dependency
~~~~~~~~~~~~~~~~~~~~

OGGM finally needs to compute the temperature and precipitation at the altitude
of the glacier grid points. The default is to use a fixed gradient of
-6.5K km :math:`^{-1}` and no gradient for precipitation. However, OGGM
implements a module which computes the local gradient by linear
regression of the 9 surrounding grid points. This implies that you trust the
near-surface temperature lapse-rates provided by the climate dataset (which
is probably not always a good idea). The relevant parameters are
``temp_use_local_gradient``, ``temp_default_gradient``, and
``temp_local_gradient_bounds``.


Temperature index model
-----------------------

The monthly mass-balance :math:`B_i` at elevation :math:`z`
is computed as:

.. math::

    B_i(z) = P_i^{Solid}(z) - \mu ^{*} \, max \left( T_i(z) - T_{Melt}, 0 \right)

where :math:`P_i^{Solid}` is the monthly solid precipitation, :math:`T_i`
the monthly temperature and :math:`T_{Melt}` is the monthly mean air
temperature above which ice melt is assumed to occur (0°C per default).
Solid precipitation is computed out of the total precipitation. The fraction of
solid precipitation is based on the monthly mean temperature: all solid below
``temp_all_solid`` (default: 0°C) all liquid above ``temp_all_liq``
(default: 2°C), linear in between.

The parameter :math:`\mu ^{*}` indicates the temperature sensitivity of the
glacier, and it needs to be calibrated.

Calibration
-----------

We will start by making two observations:

- the sensitivity parameter :math:`\mu ^{*}` is depending on many parameters,
  most of them being glacier-specific (e.g. avalanches, topographical shading,
  cloudiness...).
- the sensitivity parameter :math:`\mu ^{*}` will be affected by uncertainties
  and systematic biases in the input climate data.

As a result, :math:`\mu ^{*}` can vary greatly between neighboring glaciers.
The calibration procedure introduced by `Marzeion et al., (2012)`_ and
implemented in OGGM makes full use of these two observations by turning these
apparent handicaps into assets.

The calibration procedure starts with glaciers for which we have direct
observations of the annual specific mass-balance SMB. We use the `WGMS FoG`_
(shipped with OGGM) for this purpose.

.. _WGMS FoG: http://wgms.ch/data_databaseversions/

For each of these glaciers, time-dependent "candidate" temperature sensitivities
:math:`\mu (t)` are estimated by requiring that the average specific
mass-balance :math:`B_{31}` is equal to zero. :math:`B_{31}` is computed
for a 31 yr period centered around the year :math:`t` **and for a constant
glacier geometry fixed at the RGI date** (e.g. 2003 for most glaciers in the
European Alps).

.. ipython:: python

    @savefig plot_mu_ts.png width=100%
    example_plot_mu_ts()  # the code for these examples is posted below

Around 1900, the climate was cold and wet. As a consequence, the
temperature sensitivity required to maintain the 2003 glacier geometry is high.
Inversely, the recent climate is warm and the glacier must have a small
temperature sensitivity in order to preserve its geometry.

Note that these :math:`\mu (t)` are just
hypothetical sensitivities necessary to maintain the glacier in equilibrium in
an average climate at the year :math:`t`. We call them "candidates", as one
of these is likely to be close to the "real" sensitivity of the glacier.

This is when the mass-balance observations come into play: each of these
candidates can be used to compute the mass-balance during the period
were we have observations. We then compare the model output
with the expected mass-balance and compute the model bias:

.. ipython:: python

    @savefig plot_bias_ts.png width=100%
    example_plot_bias_ts()  # the code for these examples is posted below

The bias is positive when :math:`\mu` is too low, and negative when :math:`\mu`
is too high. The vertical dashed lines mark the times where the bias is
closest to zero. They all correspond to approximately the same :math:`\mu` (but
not exactly, as precipitation and temperature both influence :math:`\mu`).
These dates at which the :math:`\mu` candidates
are close to the real :math:`\mu` are called :math:`t^*` (the associated sensitivities
:math:`\mu (t^*)` are called :math:`\mu^*`).

At the glaciers where observations are available, this detour via the :math:`\mu`
candidates is not necessary to find the correct :math:`\mu^*`. Indeed, the goal
of these computations are in fact to find :math:`t^*`, **which is the actual
value to be interpolated to glaciers where no observations are available**.

The benefit of this approach is best shown with the results of a cross-validation
study realized by `Marzeion et al., (2012)`_ (and confirmed by OGGM):

.. figure:: _static/marzeion_mus.png
    :width: 100%

    Benefit of spatially interpolating :math:`t^*` instead of :math:`\mu^*`;
    (a) error distribution of :math:`\mu^*` if determined as the mean of
    :math:`\mu^*` of all other glaciers with mass balance measurements in the
    respective region; (b) error distribution of :math:`\mu^*` if determined
    by interpolation of :math:`t^*`. Source: `Marzeion et al., (2012)`_.

This substantial improvement in model performance is due to several factors:

- the equilibrium constraint applied on :math:`\mu` implies that the
  sensitivity cannot vary much during the last century.
  In fact, :math:`\mu` at one glacier varies far
  less in one century than between neighboring glaciers,
  because of all the factors mentioned above.
  In particular, it will vary comparatively little around a given year
  :math:`t` : errors in :math:`t^*` (even large) will result in small errors in
  :math:`\mu^*`.
- the equilibrium constraint will also imply that systematic biases in
  temperature and precipitation (no matter how large) will automatically be
  compensated by all :math:`\mu (t)`, and therefore also by :math:`\mu^*`.
  In that sense, the calibration procedure can be seen as a empirically driven
  downscaling strategy: if a glacier is here, then the local climate (or the
  glacier temperature sensitivity) *must* allow a glacier to be there. For
  example, the effect of avalanches or a negative bias in precipitation input
  will have the same impact on calibration: :math:`\mu^*` should be reduced to
  take these effects into account, even though they are not resolved by
  the mass-balance model.

The most important drawback of this calibration method is that it assumes that
two neighboring glaciers should have a similar :math:`t^*`. This is not
necessarily the case, as other factors than climate (such as the glacier size)
will influence :math:`t^*` too. Our results (and the arguments listed above)
show however that this is an approximation we can cope with.

In a final note, it is important to mention that the :math:`\mu^*` and
:math:`t^*` should not be over-interpreted in terms of "real"
temperature sensitivities or "real" response time of the glacier.
This procedure is primarily a calibration method, and as such it can be
statistically scrutinized (for example with cross-validation).
It can also be noted that the MB observations play a
relatively minor role in the calibration: they could be entirely avoided by
fixing a :math:`t^*` for all glaciers in a region (or even worldwide). The
resulting changes in calibrated :math:`\mu^*` will be comparatively small
(again, because of the local constraints on :math:`\mu`). The MB observations,
however, play a major role for the assessment of model uncertainty.


Implementation details
----------------------

If you had the courage to read until here, it means that you have concrete
questions about the implementation of the mass-balance model in OGGM.
Here are some more details:

- the mass-balance in OGGM is computed from the altitudes and widths
  of the flowlines grid points (see :ref:`flowlines`). The easiest way to let
  OGGM compute the mass-balance for you is to use the
  :py:class:`core.models.massbalance.PastMassBalanceModel`.
- the interpolation of :math:`t^*` is done with an inverse distance weighting
  algorithm (see :py:func:`tasks.distribute_t_stars`)
- if more than one :math:`t^*` is found for some reference glaciers, than the
  glaciers with only one :math:`t^*` will determine the most likely :math:`t^*`
  for the other glaciers (see :py:func:`tasks.compute_ref_t_stars`)
- yes, the temperature gradients and the precipitation scaling factor will have
  an influence on the results, but it is small since any change will automatically
  be compensated by :math:`\mu^*`. We are currently quantifying these effects
  more precisely.
- the cross-validation procedure is done systematically
  (:py:func:`tasks.crossval_t_stars`). Currently, this cross-validation is done
  with fixed geometry (it will be extended to a full validation
  with the dynamical model at a later stage).


Code used to generate these examples:

.. literalinclude:: _code/prepare_climate.py
