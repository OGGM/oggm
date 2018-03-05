.. currentmodule:: oggm
.. _whats-new:

Version history
===============

v1.X (unreleased)
-----------------

Breaking changes
~~~~~~~~~~~~~~~~

- Accumulation Area Ratio (AAR) is now correctly computed (:issue:`361`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- The method used to apply CRU and GCM anomalies to the climatology has
  changed for precipitation: we now use scaled anomalies instead of the
  standard anomalies (:pull:`393`). The previous method might have lead to
  negative values in some cases. The corresponding reference t* have also
  been updated (:pull:`407`). This change has some consequences on the
  the model results: cross-validation indicates very similar scores, but
  the influence on global model output has not been assessed yet.
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- It is now possible to run a simulation with spinup in the standard
  workflow (:pull:`411`). For this to happen it was necessary to clean up
  the many `*filesuffix` options. The new names are more explicit
  but not backwards compatible. The previous `filesuffix` is now
  called `output_filesuffix`. The previous `input_filesuffix` is now
  called `climate_input_filesuffix`. The `random_glacier_evolution` task
  is now called `run_random_climate` for consistency with the other tasks
  See the PR linked above for more info.
  By `Fabien Maussion <https://github.com/fmaussion>`_.

Enhancements
~~~~~~~~~~~~

- Added a utility function to easily get intersects files (:pull:`402`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- The old GlaThiDa file linking the total volume of glaciers (T database) to
  RGI has been updated to RGI Version 6 (:pull:`403`).
  Generally, we do not recommend to use these data for calibration or
  validation because of largely unknown uncertainties.
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- The computing efficiency of the 2D shallow ice model has been increased
  by a factor 2 (:pull:`415`), by avoiding useless repetitions of indexing
  operations. The results shouldn't change at all.
  By `Fabien Maussion <https://github.com/fmaussion>`_.


Bug fixes
~~~~~~~~~

- Remove dependency to deprecated matplotlib._cntr module (:issue:`418`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.


Internals
~~~~~~~~~

- We now use a dedicated server for input data such as modified RGI files
  (:pull:`408`). By `Fabien Maussion <https://github.com/fmaussion>`_.
- Test fix for googlemaps.
  By `Fabien Maussion <https://github.com/fmaussion>`_.
- Added a utility function (:py:func:`~utils.idealized_gdir`) useful
  to dow flowline experiments without have to create a local map (:pull:`413`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.


.. _whats-new.1.0:

v1.0 (16 January 2018)
----------------------

This is the first official major release of OGGM. It is concomitant to the
submission of a manuscript to
`Geoscientific Model Development (GMD) <https://www.geoscientific-model-development.net>`_.

This marks the stabilization of the codebase (hopefully) and implies that
future changes will have to be documented carefully to ensure traceability.

New contributors to the project:

- **Anouk Vlug** (PhD student, University of Bremen), added the CESM
  climate data tools.
- **Anton Butenko** (PhD student, University of Bremen), developed the
  downstream bedshape algorithm
- **Beatriz Recinos** (PhD student, University of Bremen), participated to the
  development of the calving parametrization
- **Julia Eis** (PhD student, University of Bremen), developed the glacier
  partitioning algorithm
- **Schmitty Smith** (PhD student,  Northand College, Wisconsin US), added
  optional parameters to the mass-balance models


.. _whats-new.0.1.1:

v0.1.1 (16 February 2017)
-------------------------

Minor release: changes in ITMIX to handle the synthetic glacier cases.

It was tagged only recently for long term documentation purposes and storage
on `Zenodo <https://zenodo.org/record/292630#.WMAwelcX77g>`_.

.. _whats-new.0.1.0:

v0.1 (29 March 2016)
--------------------

Initial release, used to prepare the data submitted to ITMIX (see
`here <http://www.fabienmaussion.info/2016/06/18/itmix-experiment-phase1/>`_).


This release is the result of several months of development (outside of GitHub
for a large part). Several people have contributed to this release:

- **Michael Adamer** (intern, UIBK), participated to the development of the
  centerline determination algorithm (2014)
- **Kévin Fourteau** (intern, UIBK, ENS Cachan), participated to the
  development of the inversion and the flowline modelling algorithms
  (2014-2015)
- **Alexander H. Jarosch** (Associate Professor, University of Iceland),
  developed the MUSCL-SuperBee model (:pull:`23`)
- **Johannes Landmann** (intern, UIBK), participated to the
  `links between databases`_ project (2015)
- **Ben Marzeion** (project leader, University of Bremen)
- **Fabien Maussion** (project leader, UIBK)
- **Felix Oesterle** (Post-Doc, UIBK), develops `OGGR`_ and provided the
  AWS deployment script (:pull:`25`)
- **Timo Rothenpieler** (programmer, University of Bremen), participated to the
  OGGM deployment script (e.g. :pull:`34`, :pull:`48`), and developed OGGM
  `installation`_ tools
- **Christian Wild** (master student, UIBK), participated to the development of
  the centerline determination algorithm (2014)

.. _OGGR: http://oggr.org/
.. _links between databases: https://github.com/OGGM/databases-links
.. _installation: https://github.com/OGGM/OGGM-Anaconda
