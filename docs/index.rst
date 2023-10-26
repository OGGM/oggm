
A modular and open source glacier model
---------------------------------------

**OGGM is a cutting-edge open source modelling framework** designed to simulate
the past and future mass balance, volume, and geometry of glaciers worldwide.

The model features several glacier evolution models, including an explicit ice
dynamics module accounting for glacier geometry and frontal ablation.
**With an unwavering commitment to using publicly available data for calibration
and validation, OGGM is a reliable and readily applicable tool for studying glaciers**.

OGGM is also a modular platform that supports novel modelling workflows,
**encouraging researchers to create unique models and analyses for their research**.
Our framework is designed to be flexible and adaptable, making it an
ideal tool for a wide range of applications in glaciology and related fields.

.. warning::

   OGGM v1.6 is a substantial change to v1.5.3. Among other developments,
   the mass balance calibration has substantially improved. In order to
   allow easier and faster developments from the community in the future,
   several variable names have changed and older workflows are not available
   anymore. **We recommend our users to switch to 1.6 as soon as
   possible, but only if they are at the development stage of a study.**
   Older versions of OGGM will always be available via github and Zenodo,
   and older documentation pages can be accessed via the interface below.

**This webpage is for the software documentation: for general information about the
OGGM project and related news, visit** `oggm.org <http://oggm.org>`_.

Video presentation
^^^^^^^^^^^^^^^^^^

If you are new to OGGM and would like a short introduction, here is a 15'
presentation from April 2020:

.. raw:: html

    <iframe width="672" height="378" src="https://www.youtube.com/embed/ttJMxcwXUjw?start=1270" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

Disclaimer: If you don't have access to youtube from your location, you won't be able to see this video.
*Slides available* `here <https://oggm.org/framework_talk>`_


Overview
^^^^^^^^

Core principles and structure of the OGGM modelling framework.

* :doc:`introduction`
* :doc:`structure`
* :doc:`flowlines`
* :doc:`mass-balance`
* :doc:`geometry-evolution`
* :doc:`inversion`
* :doc:`frontal-ablation`
* :doc:`dynamic-spinup`
* :doc:`whats-new`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Overview

    introduction.rst
    structure.rst
    flowlines.rst
    mass-balance.rst
    geometry-evolution.rst
    inversion.rst
    frontal-ablation.rst
    dynamic-spinup.rst
    whats-new.rst

Using OGGM
^^^^^^^^^^

How to use the model, with concrete code examples and links to the tutorials.

* :doc:`cloud`
* :doc:`installing-oggm`
* :doc:`getting-started`
* `Tutorials <https://oggm.org/tutorials/stable>`_
* :doc:`api`
* :doc:`practicalities`
* :doc:`faq`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Using OGGM

    cloud.rst
    installing-oggm.rst
    getting-started.rst
    Tutorials <https://oggm.org/tutorials/stable>
    api.rst
    practicalities.rst
    faq.rst

Datasets and downloads
^^^^^^^^^^^^^^^^^^^^^^

All the things that OGGM has on offer.

* :doc:`shop`
* :doc:`climate-data`
* :doc:`reference-mass-balance-data`
* :doc:`rgitopo`
* :doc:`download-projections`
* :doc:`assets`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Datasets and downloads

    shop.rst
    climate-data.rst
    reference-mass-balance-data.rst
    rgitopo.rst
    download-projections.rst
    assets.rst

Contributing
^^^^^^^^^^^^

Do you want to contribute to the model? This is the right place to start.

* :doc:`citing-oggm`
* :doc:`add-module`
* :doc:`contributing`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Contributing

    citing-oggm.rst
    add-module.rst
    contributing.rst

.. _contact:

Get in touch
------------

- View the source code `on GitHub`_.
- Report bugs or share your ideas on the `issue tracker`_, and improve
  the model by submitting a `pull request`_.
- Chat with us on `Slack`_! (just send us an `e-mail`_ so we can add you)
- Follow us on `Twitter`_.
- Participate to our regular `meeting`_. (`reach out`_  if you want to join in)
- Or you can always send us an `e-mail`_ the good old way.

.. _e-mail: info@oggm.org
.. _Slack: https://slack.com
.. _on GitHub: https://github.com/OGGM/oggm
.. _issue tracker: https://github.com/OGGM/oggm/issues
.. _pull request: https://github.com/OGGM/oggm/pulls
.. _Twitter: https://twitter.com/OGGM1
.. _meeting: https://oggm.org/meetings/
.. _reach out: info@oggm.org


License and citation
--------------------

OGGM is available under the open source `3-Clause BSD License`_.

.. _3-Clause BSD License: https://opensource.org/licenses/BSD-3-Clause

OGGM is a free software. This implies that you are free to use the model and
copy, modify or redistribute its code at your wish, under certain conditions:

1. When using this software, please acknowledge the original authors of this
   contribution by using our logo, referring to our website or using an
   appropriate citation. See :doc:`citing-oggm` for how to do that.

2. Redistributions of any substantial portion of the OGGM source code must
   meet the conditions listed in the `OGGM license`_

3. Neither OGGM e.V. nor the names of OGGM contributors may be used to endorse
   or promote products derived from this software without specific prior
   written permission. This does not mean that you need our written permission
   to work with OGGM or publish results based on OGGM: it simply means that
   the OGGM developers are not accountable for what you do with the tool
   (`more info <https://opensource.stackexchange.com/a/9137>`_).

See the `OGGM license`_ for more information.

.. _OGGM license: https://github.com/OGGM/oggm/blob/master/LICENSE.txt

About
-----

:Version:
    .. image:: https://img.shields.io/pypi/v/oggm.svg
        :target: https://pypi.python.org/pypi/oggm
        :alt: Pypi version

    .. image:: https://img.shields.io/pypi/pyversions/oggm.svg
        :target: https://pypi.python.org/pypi/oggm
        :alt: Supported python versions

:Citation:
    .. image:: https://img.shields.io/badge/Citation-GMD%20paper-orange.svg
        :target: https://www.geosci-model-dev.net/12/909/2019/
        :alt: GMD Paper

    .. image:: https://zenodo.org/badge/43965645.svg
        :target: https://zenodo.org/badge/latestdoi/43965645
        :alt: Zenodo

:Tests:
    .. image:: https://coveralls.io/repos/github/OGGM/oggm/badge.svg?branch=master
        :target: https://coveralls.io/github/OGGM/oggm?branch=master
        :alt: Code coverage

    .. image:: https://github.com/OGGM/oggm/actions/workflows/run-tests.yml/badge.svg?branch=master
        :target: https://github.com/OGGM/oggm/actions/workflows/run-tests.yml
        :alt: Linux build status

    .. image:: https://img.shields.io/badge/Cross-validation-blue.svg
        :target: https://cluster.klima.uni-bremen.de/~oggm/ref_mb_params/oggm_v1.4/crossval.html
        :alt: Mass balance cross validation

    .. image:: https://readthedocs.org/projects/oggm/badge/?version=latest
        :target: http://docs.oggm.org/en/latest
        :alt: Documentation status

    .. image:: https://img.shields.io/badge/benchmarked%20by-asv-green.svg?style=flat
        :target: https://cluster.klima.uni-bremen.de/~github/asv/
        :alt: Benchmark status

:License:
    .. image:: https://img.shields.io/pypi/l/oggm.svg
        :target: https://github.com/OGGM/oggm/blob/master/LICENSE.txt
        :alt: BSD-3-Clause License

:Authors:

    See the `version history`_ for a list of all contributors.

    .. _version history: http://docs.oggm.org/en/latest/whats-new.html
