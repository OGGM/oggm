.. -*- rst -*- -*- restructuredtext -*-
.. This file should be written using restructured text conventions
.. default-role:: math

.. image:: ./files/logo.png

Ongoing work. The model is based on `Marzeion et al., (2012) <http://www.the-cryosphere.net/6/1295/2012/tc-6-1295-2012.html>`_
and implements several improvements to account e.g. for glacier geometry and ice dynamics.

Example
-------

We use the `Hintereisferner <http://acinn.uibk.ac.at/research/ice-and-climate/projects/hef>`_ as benchmark:

.. image:: ./oggm/tests/baseline_images/test_graphics/test_googlestatic_1.5.0.png

We first define a local grid and compute the centerlines (automatically, `Kienholz et al., (2014) <http://www.the-cryosphere.net/8/503/2014/tc-8-503-2014.html>`_) as well as the downstream flowlines:

.. image:: ./oggm/tests/baseline_images/test_graphics/test_downstream_cls_1.5.0.png

The glacier is then represented as several flowlines of varying width:

.. image:: ./oggm/tests/baseline_images/test_graphics/test_width_corrected_1.5.0.png

Finally, following an inversion algorithm based on mass-balance (`Marzeion et al., (2012) <http://www.the-cryosphere.net/6/1295/2012/tc-6-1295-2012.html>`_) and ice-flow dynamics (`Farinotti et al., (2009) <http://www.igsoc.org/journal/55/191/>`_), we derive the ice thickness of the glacier:

.. image:: ./oggm/tests/baseline_images/test_graphics/test_inversion_1.5.0.png


Installation
------------

OGGM is compatible with Python 2.7 and Python 3+.

To work with OGGM we recommend to clone the repository:

   $ git clone git@github.com:OGGM/oggm.git

OGGM has several dependencies, some of them are unfortunately not always
trivial to install (GDAL, Fiona). See `INSTALL.rst <./docs/INSTALL.rst>`_
for more information.

If you use the code we strongly recommend to contact us (adress below) since
the code base is likely to evolve in the near future, with probable
backwards incompatible changes.


About
-----

:Status:
    Experimental - in development

:Tests:
    .. image:: https://coveralls.io/repos/OGGM/oggm/badge.svg?branch=master&service=github
      :target: https://coveralls.io/github/OGGM/oggm?branch=master

    .. image:: https://travis-ci.org/OGGM/oggm.svg?branch=master
        :target: https://travis-ci.org/OGGM/oggm
    
:License:
    GNU GPLv3

:Authors:
    - Fabien Maussion - fabien.maussion@uibk.ac.at
    - Ben Marzeion
    - Kévin Fourteau
    - Christian Wild
    - Michael Adamer

:Funding:
    Austrian Research Foundation FWF, Projects P22443-N21 and P25362-N26

    .. image:: http://acinn.uibk.ac.at/sites/all/themes/imgi/images/acinn_logo.png
    
    .. image:: http://www.uni-bremen.de/fileadmin/images/logo-uni-bremen-EXZELLENT.png
