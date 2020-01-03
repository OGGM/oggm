.. _pitfalls:

.. currentmodule:: oggm

************************
Pitfalls and limitations
************************

As the OGGM project is gaining visibility and momentum, we also see an increase
of potential misuse or misunderstandings about what OGGM can and cannot do.
Hefer to our :ref:`faq` for a general introduction. Here, we discuss
specific pitfalls in more details.

The default ice dynamics parameter "Glen A" is not calibrated
=============================================================

Out-of-the box OGGM will uses fixed values for the creep parameter
:math:`A` and the sliding parameter :math:`f_s`:

.. ipython:: python

   from oggm import cfg
   cfg.initialize()
   cfg.PARAMS['glen_a']
   cfg.PARAMS['fs']

That is, :math:`A` is set to the standard value for temperate ice as given in
[Cuffey_Paterson_2010]_, and sliding is set to zero. While these values are
reasonable, they are unlikely to be the ones yielding the best results at the
global scale, and even more unlikely at regional or local scales. In particular,
in the absence of sliding parameter, it is recommended to set :math:`A` to a
larger value to compensate for this missing process.

There is a way to calibrate :math:`A` for the ice thickness inversion
procedure based on observations of ice thickness. This does not mean that this
:math:`A` can be applied unchanged to the forward model, unfortunately.
At the global scale, a value in the range of [1.1-1.5] times the default value
gives estimates close to [Farinotti_etal_2019]_. At regional scale, these
values can differ, with a value closer to a factor 3 e.g. for the Alps. Note
that this depends on other variables as well, such as precipitation estimates
(which affect the mass turnover).

Finally, note that a change in :math:`A` has a very strong influence
for values close to the default value, but this influences reduces to the
power of 1/5 for large values of A (in other worlds, there is a big
difference between values of 1 to 1.3 times the default :math:`A`, but a
comparatively small difference for values between 3 to 5 times the
default :math:`A`). This is best shown by this figure from
[Maussion_etal_2019]_:

.. figure:: _static/global_volume_mau2019.png
    :width: 100%

    Global volume estimates as a function of the multiplication factor
    applied to the ice creep parameter A, with five different setups:
    defaults, with sliding velocity, with lateral drag, and with rectangular
    and parabolic bed shapes only (instead of the default mixed
    parabolic/rectangular). In addition, we plotted the estimates from
    standard volume–area scaling (VAS, :math:`V = 0.034 S^{1.375}`),
    Huss and Farinotti (2012) (HF2012) and Grinsted (2013) (G2013).
    The latter two estimates are provided for indication only as they
    are based on a different glacier inventory

**How to choose the "best A" for my application?**
Sorry, but we don't know yet. We are working on it though!

.. _pitfalls.numerics:

The numerical model in OGGM is numerically unstable in some conditions
======================================================================

See `this github issue <https://github.com/OGGM/oggm/issues/909>`_ for an
ongoing discussion. We will post and update here soon!


The mass-balance model of OGGM is not calibrated with remote sensing data
=========================================================================

Currently, the values for the mass-balance parameters such as the
temperature sensitivity, the precipitation correction factor, etc. are
calibrated based on the in-situ measurements provided by the WGMS
(traditional mass-balance data). For more information about the procedure,
see [Maussion_etal_2019]_ and our
`performance monitoring website <https://cluster.klima.uni-bremen.de/~github/crossval/>`_.

This, however, is not really "state of the art" anymore. Other recent
studies by e.g. [Huss_Hock_2015]_ and [Zekollari_etal_2019]_
also use geodetic mass-balance estimates
to calibrate their model.

We are looking for people to help us with this task: join us! See
e.g. :ref:`oep0003` for a discussion document.


References
==========

.. [Farinotti_etal_2019] Farinotti, D., Huss, M., Fürst, J. J., Landmann, J.,
    Machguth, H., Maussion, F. and Pandit, A.: A consensus estimate for the ice
    thickness distribution of all glaciers on Earth, Nat. Geosci., 12(3),
    168–173, doi:10.1038/s41561-019-0300-3, 2019.

.. [Maussion_etal_2019] Maussion, F., Butenko, A., Champollion, N., Dusch, M.,
    Eis, J., Fourteau, K., Gregor, P., Jarosch, A. H., Landmann, J., Oesterle,
    F., Recinos, B., Rothenpieler, T., Vlug, A., Wild, C. T. and Marzeion, B.:
    The Open Global Glacier Model (OGGM) v1.1, Geosci. Model Dev., 12(3),
    909–931, doi:10.5194/gmd-12-909-2019, 2019.

.. [Huss_Hock_2015] Huss, M. and Hock, R.: A new model for global glacier
    change and sea-level rise, Front. Earth Sci., 3(September), 1–22,
    doi:10.3389/feart.2015.00054, 2015.

.. [Zekollari_etal_2019] Zekollari, H., Huss, M. and Farinotti, D.: Modelling
    the future evolution of glaciers in the European Alps under the EURO-CORDEX
    RCM ensemble, Cryosphere, 13(4), 1125–1146, doi:10.5194/tc-13-1125-2019, 2019.