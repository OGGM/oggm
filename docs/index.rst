.. image:: _static/logo.png

|

An open source glacier model in Python
--------------------------------------

The model builds upon `Marzeion et al., (2012)`_ and intends to become a
global scale, modular, and open source model for glacier dynamics. The model
accounts for glacier geometry (including contributory branches) and includes
a simple (yet explicit) ice dynamics module. It will be able to simulate
past and future mass-balance, volume and geometry of any glacier in a fully
automated workflow. We rely exclusively on publicly available data for
calibration and validation.

The project is currently in development. `Get in touch`_ with us if
you want to contribute.

.. _Marzeion et al., (2012): http://www.the-cryosphere.net/6/1295/2012/tc-6-1295-2012.html


News
----

- **30.11.2016**: the paper summarizing the ITMIX experiment is online for
  review! Visit `this link <http://www.the-cryosphere-discuss.net/tc-2016-250/>`_
  to read and discuss the manuscript.
- **24.10.2016**: Registration to the :ref:`workshop` is now open!
- **20.04.2016**: Fabien presented OGGM at the
  `European Geosciences Union General Assembly <http://meetingorganizer.copernicus.org/EGU2016/orals/20092>`_
- **29.03.2016**: OGGM participated to the
  `Ice Thickness Models Intercomparison eXperiment <http://fabienmaussion.info/2016/06/18/itmix-experiment-phase1/>`_
- **11.02.2016**: The 1st OGGM workshop took place in Obergurgl, Austrian Alps


Index
-----

Principles
^^^^^^^^^^

* :doc:`introduction`
* :doc:`flowlines`
* :doc:`mass-balance`
* :doc:`ice-dynamics`
* :doc:`inversion`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Principles

    introduction.rst
    flowlines.rst
    mass-balance.rst
    ice-dynamics.rst
    inversion.rst

Using OGGM
^^^^^^^^^^

* :doc:`installing-oggm`
* :doc:`getting-started`
* :doc:`glacierdir-gen`
* :doc:`input-data`
* :doc:`mpi`
* :doc:`api`
* :doc:`contributing`
* :doc:`whats-new`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Using OGGM

    installing-oggm.rst
    getting-started.rst
    glacierdir-gen.rst
    input-data.rst
    mpi.rst
    api.rst
    whats-new.rst

Contributing
^^^^^^^^^^^^

* :doc:`contributing`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Contributing

    contributing.rst


Get in touch
------------

- To ask questions or discuss OGGM, send us an `e-mail`_.
- Report bugs, share your ideas or view the source code `on GitHub`_.

.. _e-mail: http://www.fabienmaussion.info/
.. _on GitHub: https://github.com/OGGM/oggm


License
-------

.. image:: _static/gpl.png
   :width: 140 px

OGGM is available under the open source `GNU GPLv3 license`_.

.. _GNU GPLv3 license: http://www.gnu.org/licenses/gpl-3.0.en.html

OGGM is free software. This implies that you are free to use the model and
copy or modify its code at your wish, under certain conditions:

1. When using this software, please acknowledge the original authors of this
   contribution. Currently, we recommend to use the `Zenodo citation`_ for this
   purpose.

   An example BibTeX entry::

        @misc{OGGM_v0.1.1,
          author       = {Fabien Maussion and
                          Timo Rothenpieler and
                          Ben Marzeion and
                          Johannes Landmann and
                          Felix Oesterle and
                          Alexander Jarosch and
                          Beatriz Recinos and
                          Anouk Vlug},
          title        = {OGGM/oggm: v0.1.1},
          month        = feb,
          year         = 2017,
          doi          = {10.5281/zenodo.292630},
          url          = {https://doi.org/10.5281/zenodo.292630}
        }


2. Your modifications to the code belong to you, but if you decide
   to share these modifications with others you'll have to do so under the same
   license as OGGM (the GNU General Public License as published by the Free
   Software Foundation).

See the `wikipedia page about GPL`_ and the `OGGM license`_ for more
information.

.. _Zenodo citation: https://zenodo.org/badge/latestdoi/43965645

.. _wikipedia page about GPL: https://en.wikipedia.org/wiki/GNU_General_Public_License

.. _OGGM license: https://github.com/OGGM/oggm/blob/master/LICENSE.rst

About
-----

:Status:
    Experimental - in development

:License:
    GNU GPLv3

:Authors:

    See :ref:`whats-new` for a list of all contributors.

:Funding:
    Austrian Research Foundation FWF, Projects P22443-N21 and P25362-N26

    .. image:: http://acinn.uibk.ac.at/sites/all/themes/imgi/images/acinn_logo.png

    .. image:: http://www.uni-bremen.de/fileadmin/images/logo-uni-bremen-EXZELLENT.png
        :align: right
