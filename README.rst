.. image:: docs/_static/logo.png

|


The model builds upon `Marzeion et al., (2012)`_ and intends to become a
global scale, modular, and open source model for glacier dynamics. The model
accounts for glacier geometry (including contributory branches) and includes
a simple (yet explicit) ice dynamics module. It can simulate past and
future mass-balance, volume and geometry of any glacier in a fully
automated workflow. We rely exclusively on publicly available data for
calibration and validation.

The project is currently in intense development. `Get in touch`_ with us if
you want to contribute.

.. _Marzeion et al., (2012): http://www.the-cryosphere.net/6/1295/2012/tc-6-1295-2012.html


Example
-------

We use the `Hintereisferner`_ as benchmark:

.. image:: ./oggm/tests/baseline_images/test_graphics/test_googlestatic_1.5+.png
   :width: 200 px

We first define a local grid and compute the centerlines (`Kienholz et al., 2014`_) as well as the downstream flowlines:

.. image:: ./oggm/tests/baseline_images/test_graphics/test_downstream_cls_1.5+.png
   :width: 200 px

The glacier is then represented as several flowlines of varying width:

.. image:: ./oggm/tests/baseline_images/test_graphics/test_width_corrected_1.5+.png
   :width: 200 px

Finally, following an inversion algorithm based on mass-balance (`Marzeion et al., 2012`_) and ice-flow dynamics (`Farinotti et al., 2009`_), we derive the ice thickness of the glacier:

.. image:: ./oggm/tests/baseline_images/test_graphics/test_inversion_1.5+.png
   :width: 200 px

.. _Hintereisferner: http://acinn.uibk.ac.at/research/ice-and-climate/projects/hef
.. _Marzeion et al., 2012: http://www.the-cryosphere.net/6/1295/2012/tc-6-1295-2012.html
.. _Kienholz et al., 2014 : http://www.the-cryosphere.net/8/503/2014/tc-8-503-2014.html
.. _Farinotti et al., 2009 : http://www.igsoc.org/journal/55/191/


Installation, documentation
---------------------------

A documentation draft is hosted on ReadTheDocs: http://oggm.org


Get in touch
------------

- To ask questions or discuss OGGM, send us an `e-mail`_.
- Report bugs, share your ideas or view the source code `on GitHub`_.

.. _e-mail: http://www.fabienmaussion.info/
.. _on GitHub: https://github.com/OGGM/oggm


About
-----

:Status:
    Experimental - in development

:Tests:
    .. image:: https://coveralls.io/repos/OGGM/oggm/badge.svg?branch=master&service=github
        :target: https://coveralls.io/github/OGGM/oggm?branch=master
        :alt: Code coverage

    .. image:: https://travis-ci.org/OGGM/oggm.svg?branch=master
        :target: https://travis-ci.org/OGGM/oggm
        :alt: Linux build status

    .. image:: https://ci.appveyor.com/api/projects/status/alealh9rxmqgd3nm/branch/master?svg=true
        :target: https://ci.appveyor.com/project/fmaussion/oggm
        :alt: Windows-conda build status

    .. image:: https://readthedocs.org/projects/oggm/badge/?version=latest
        :target: http://oggm.readthedocs.org/en/latest/?badge=latest
        :alt: Documentation status

:License:

    OGGM is available under the open source `GNU GPLv3 license`_.

    .. _GNU GPLv3 license: http://www.gnu.org/licenses/gpl-3.0.en.html

:Authors:

    See `whats-new`_ for a list of all contributors.

    .. _whats-new: http://oggm.readthedocs.org/en/latest/whats-new.html

:Funding:
    Austrian Research Foundation FWF, Projects P22443-N21 and P25362-N26

    .. image:: http://acinn.uibk.ac.at/sites/all/themes/imgi/images/acinn_logo.png

    .. image:: http://www.uni-bremen.de/fileadmin/images/logo-uni-bremen-EXZELLENT.png
        :align: right
