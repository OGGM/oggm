
A modular and open source glacier model
---------------------------------------

**OGGM is an open source modelling framework** designed to simulate
the past and future mass balance, volume, and geometry of all glaciers worldwide.

The model framework features several glacier evolution models, including an flowline ice
dynamics module accounting for frontal ablation, and several mass-balance models, including
a pre-calibrated temperature-index model.

OGGM is above all a modular platform that supports novel modelling workflows,
**encouraging researchers to create unique model chains for their research**.
Our framework is designed to be flexible and adaptable, making it an
ideal tool for a wide range of applications in glaciology and related fields.

**This webpage is for the software documentation for version 1.6.2**:

- check-out :doc:`whats-new` to read about the updates in this version
- visit `oggm.org <http://oggm.org>`_ for general news
- visit `tutorials.oggm.org <http://tutorials.oggm.org>`_ for the interactive tutorials
- visit `edu.oggm.org <http://edu.oggm.org>`_ for the educational platform

.. admonition:: A note for new users

    OGGM is well established and well documented, but it is a complex model requiring
    background knowledge in glacier modelling and programming.
    In our experience, many research projects can actually be completed *without*
    running OGGM yourself: we provide a range of pre-computed glacier change projections
    covering a wide range of scenarios and use cases. Checkout :doc:`download-projections`
    and see if this fits your needs before diving into the model itself.


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
* `Tutorials <https://tutorials.oggm.org/stable>`_
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
    Tutorials <https://tutorials.oggm.org/stable>
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
- Participate to our regular `meeting`_. (`reach out`_  if you want to join in)
- Or you can always send us an `e-mail`_ the good old way.

.. _e-mail: info@oggm.org
.. _Slack: https://slack.com
.. _on GitHub: https://github.com/OGGM/oggm
.. _issue tracker: https://github.com/OGGM/oggm/issues
.. _pull request: https://github.com/OGGM/oggm/pulls
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

    .. image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.597193-blue.svg
        :target: https://zenodo.org/doi/10.5281/zenodo.597193
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
