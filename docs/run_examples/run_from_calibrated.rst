.. currentmodule:: oggm

.. _run-from-calibrated:

2. Complete run for a list of glaciers
======================================

This example shows how to run the OGGM  model for a list of selected glaciers.

Note that the default in OGGM is to use a previously calibrated list of
:math:`t^*` for the run, which means that we don't have to calibrate the
mass balance model ourselves (thankfully, otherwise you would have to add
all the calibration glaciers to your list too).

Note that in order to be correct, the automated calibration can only be used
if the model parameters don't change between the calibration and the run.
After testing, it appears that changing the 'border' parameter won't affect
the results much (as expected), so it's ok to change this parameter.
Some other parameters (e.g. topo smoothing, dx, precip factor, alternative
climate data...) will probably need a re-calibration step
(see :ref:`run-calibration`).


Script
------

This script decomposes the steps in a bit more detail than the previous
example:

.. literalinclude:: _code/run_from_calibrated.py

If everything went well, you should see an output similar to::


    2018-05-01 17:51:43: oggm.cfg: Parameter file: /home/mowglie/Documents/git/oggm-fork/oggm/params.cfg
    2018-05-01 17:51:47: __main__: Starting OGGM run
    2018-05-01 17:51:47: __main__: Number of glaciers: 3
    2018-05-01 17:51:47: oggm.workflow: Multiprocessing: using all available processors (N=8)
    2018-05-01 17:51:47: oggm.core.gis: (RGI60-11.00897) define_glacier_region
    2018-05-01 17:51:47: oggm.core.gis: (RGI60-01.10299) define_glacier_region
    (...)
    2018-05-01 17:52:14: oggm.core.flowline: (RGI60-01.10299) default time stepping was successful!
    2018-05-01 17:52:24: oggm.core.flowline: (RGI60-18.02342) default time stepping was successful!
    2018-05-01 17:52:24: __main__: Compiling output
    2018-05-01 17:52:25: __main__: OGGM is done! Time needed: 0:00:41

.. note::

   During the ``run_random_climate`` task some numerical warnings might
   occur. These are expected to happen and are caught by the solver, which then
   tries a more conservative time stepping scheme.

.. note::

    The ``run_random_climate`` task can be replaced by any climate
    scenario built by the user. For this you'll have to develop your own task,
    which will be the topic of another example script.


Starting from a preprocessed state
----------------------------------

Now that we've gone through all the preprocessing steps once and that their
output is stored on disk, it isn't necessary to re-run everything to make
a new experiment. The code can be simplified to:

.. literalinclude:: _code/run_from_calibrated_and_prepro.py

.. note::

    Note the use of the ``output_filesuffix`` keyword argument. This allows to
    store the output of different runs in different files,
    useful for later analyses.


Some analyses
-------------

The output directory contains the compiled output files from the run. The
``glacier_characteristics.csv`` file contains various information about each
glacier after the preprocessing, either from the RGI directly
(location, name, glacier type...) or from the model itself (hypsometry,
inversion model output...).

Here is an example of how to read the file:

.. code-block:: python

    from os import path
    import pandas as pd
    WORKING_DIR = path.join(path.expanduser('~'), 'tmp', 'OGGM_precalibrated_run')
    df = pd.read_csv(path.join(WORKING_DIR, 'glacier_characteristics.csv'))
    print(df)

Output (reduced for clarity):

==============  ===============  ==============  ==============  =================
rgi_id          name               dem_max_elev    dem_min_elev    inv_thickness_m
==============  ===============  ==============  ==============  =================
RGI50-18.02342  Tasman Glacier             3662             715             186.32
RGI50-01.10299  Coxe Glacier               1840               6             145.85
RGI50-11.00897  Hintereisferner            3684            2447             107.65
==============  ===============  ==============  ==============  =================


The run outpout is stored in netCDF files, and it can therefore be read with
any tool able to read those (Matlab, R, ...).

I myself am familiar with python, and I use the xarray package to
read the data:

.. literalinclude:: _code/example_analysis_ts.py

This code snippet should produce the following plot:

.. figure:: ../_static/run_example.png
    :width: 100%

.. warning::

    In the script above we have to "smooth" our length data for nicer plots.
    This is necessary because of annual snow cover: indeed, OGGM cannot
    differentiate between snow and ice. At the end of a cold mass-balance year,
    it can happen that some snow remains at the tongue and below: for the
    model, this looks like a longer glacier... (this cover is very thin, so
    that it doesn't influence the volume much).

More analyses
-------------

Here is a more complex example to demonstrate how to plot the glacier
geometries after the run:

.. literalinclude:: _code/example_analysis_maps.py

Which should produce the following plot:

.. figure:: ../_static/example_run_map_plot.png
    :width: 100%
