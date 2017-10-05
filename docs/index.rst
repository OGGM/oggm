.. image:: _static/logo.png

|

An open source glacier model in Python
--------------------------------------

Extending `Marzeion et al., (2012)`_, the model accounts for glacier geometry
(including contributory branches) and includes an explicit ice dynamics module.
It can simulate past and future mass-balance, volume and geometry of (almost)
any glacier in the world in a fully automated workflow. We rely exclusively on
publicly available data for calibration and validation.

**This is the model's documentation for users and for developers**. For more
information about the project and for the latest news, visit
`oggm.org <http://oggm.org>`_.

.. _Marzeion et al., (2012): http://www.the-cryosphere.net/6/1295/2012/tc-6-1295-2012.html


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
* :doc:`run`
* :doc:`api`
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
    run.rst
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

- View the source code `on GitHub`_.
- Report bugs or share your ideas on the `issue tracker`_.
- Improve the model by submitting a `pull request`_.
- Or you can always send us an `e-mail`_ the good old way.

.. _e-mail: info@oggm.org
.. _on GitHub: https://github.com/OGGM/oggm
.. _issue tracker: https://github.com/OGGM/oggm/issues
.. _pull request: https://github.com/OGGM/oggm/pulls


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

    See the :ref:`whats-new` for a list of all contributors.
