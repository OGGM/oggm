Geometry evolution models
=========================

**Geometry evolution models** are responsible to compute the change in
glacier geometry as a response to the climatic mass balance
forcing computed with the :doc:`mass-balance`. They are also in charge of reporting
diagnostic variables such as length, area, volume. Depending on the model's complexity,
they can also report about ice velocity, ice thickness, etc.

Currently, OGGM has three operational geometry evolution models:

.. toctree::
    :maxdepth: 1

    ice-dynamics.rst
    mass-redistribution.rst
    OGGM-VAS (volume-area scaling) <https://github.com/OGGM/oggm-vas>

`OGGM-VAS <https://github.com/OGGM/oggm-vas>`_ is a complete python re-write of
`Ben Marzeion's 2012 model <https://tc.copernicus.org/articles/6/1295/2012/>`_.
It should be equivalent to the original matlab model,
but follows the OGGM syntax very closely (and uses OGGM for data pre- and
post-processing). See Moritz Oberrauch's
`thesis <https://diglib.uibk.ac.at/ulbtirolhs/content/titleinfo/5878449>`_
for more information.

The :doc:`mass-redistribution` is an implementation of the "delta-h" model
by `Huss & Hock (2015) <https://www.frontiersin.org/articles/10.3389/feart.2015.00054/full>`_
and `Rounce et al., (2020) <https://doi.org/10.3389/feart.2019.00331>`_.
It works quite well for short simulations of the glacier retreat phase.

The default geometry evolution model in OGGM is the :doc:`ice-dynamics`.
We also have a `distributed SIA model <https://github.com/OGGM/oggm/blob/master/oggm/core/sia2d.py>`_
to play around, but nothing operational yet.
