.. currentmodule:: oggm

.. _run-calibration:

5. Run the mass-balance calibration
===================================

Sometimes you will need to do the mass-balance calibration yourself. For
example if you use alternate climate data, or if you change the global
parameters of the model (such as precipitation scaling factor or melting
threshold, for example). Here we show how to run the calibration
for all available reference glaciers. We also use this example to illustrate
how to do a "full run", i.e. without relying on :ref:`preprodir`.

The output of this script is the ``ref_tstars.csv`` file, which is found in
the working directory. The ``ref_tstars.csv`` file can then be used for further
runs, simply by copying it in the corresponding working directory before the
run.


Script
------

.. literalinclude:: _code/run_reference_mb_glaciers.py


Cross-validation
----------------

The results of the cross-validation are found in the ``crossval_tstars.csv``
file. Let's replicate Figure 3 in  `Marzeion et al., (2012)`_ :

.. literalinclude:: _code/mb_crossval.py

This should generate an output similar to::

    Median bias: 18.78
    Mean bias: 15.34
    RMS: 518.83
    Sigma bias: 0.87


.. figure:: ../_static/mb_crossval.png
    :width: 100%

    Error (bias) distribution of the mass-balance computed with a
    leave-one-out cross-validation.

.. _Marzeion et al., (2012): http://www.the-cryosphere.net/6/1295/2012/tc-6-1295-2012.html
