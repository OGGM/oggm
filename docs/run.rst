.. _run-set-up:

Set-up an OGGM run
==================

These examples will show you some example scripts to realise regional or
global runs with OGGM. The examples can be run on a laptop within a
few minutes. The mass balance calibration example requires more resources but
with a reasonably powerful personal computer this should be OK.

For region-wide or global simulations, we recommend to use a cluster
environment. OGGM should work well on cloud computing services like Amazon
or Google Cloud, and we are in the process of testing such a deployment.

**Before you start, make sure you had a look at the** :ref:`input-data` **section.**

.. toctree::
    :maxdepth: 1

    run_examples/run_rgi_region.rst
    run_examples/run_inversion.rst
    run_examples/run_errors.rst
    run_examples/run_with_spinup.rst
    run_examples/run_mb_calibration.rst
