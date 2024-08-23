Geometry evolution models
=========================

**Geometry evolution models** are responsible to compute the change in
glacier geometry as a response to the climatic mass balance
forcing computed with the :doc:`mass-balance`. They are also in charge of reporting
diagnostic variables such as length, area, volume. Depending on the model's complexity,
they can also report about ice velocity, ice thickness, etc.

Currently, OGGM has four geometry evolution models:

.. toctree::
    :maxdepth: 1
    :hidden:

    ice-dynamics.rst
    igm.rst
    mass-redistribution.rst
    OGGM-VAS (volume-area scaling) <https://github.com/OGGM/oggm-vas>


* :doc:`ice-dynamics`: the default!
* :doc:`igm`: the future!
* :doc:`mass-redistribution`: an implementation of the "delta-h" model by `Huss & Hock (2015) <https://www.frontiersin.org/articles/10.3389/feart.2015.00054/full>`_. It works quite well for short simulations of the glacier retreat phase.
* `OGGM-VAS (volume-area scaling) <https://github.com/OGGM/oggm-vas>`_: a python re-write of `Ben Marzeion's 2012 model <https://tc.copernicus.org/articles/6/1295/2012/>`_. See Moritz Oberrauch's `thesis <https://diglib.uibk.ac.at/ulbtirolhs/content/titleinfo/5878449>`_ for more information.
