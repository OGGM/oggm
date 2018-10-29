Parallel computations
=====================

.. currentmodule:: oggm

OGGM is designed to use the available resources as well as possible. For single
nodes machines but with more than one processor (frequent case for personal
computers) OGGM ships with a multiprocessing approach which is fairly simple to
use. For cluster environments, use `MPI`_.


Multiprocessing
---------------

Most OGGM computations are `embarrassingly parallel`_: they
are standalone operations to be realized on one single glacier entity and
therefore independent from each other
(they are called **entity tasks**, as opposed to the non-parallelizable
**global tasks**).

.. _embarrassingly parallel: https://en.wikipedia.org/wiki/Embarrassingly_parallel

When given a list of :ref:`glacierdir` on which to apply a given task,
the :py:func:`workflow.execute_entity_task` will distribute the operations on
the available processors using Python's `multiprocessing`_ module.
You can control this behavior with the ``use_multiprocessing`` config
parameter and the number of processors with ``mp_processes``.
The default in OGGM is:

.. ipython:: python

    from oggm import cfg
    cfg.initialize()
    cfg.PARAMS['use_multiprocessing']  # whether to use multiprocessing
    cfg.PARAMS['mp_processes']  # number of processors to use

``-1`` means that all available processors will be used.

.. _multiprocessing: https://docs.python.org/3.6/library/multiprocessing.html

MPI
---

OGGM can be run in a clustered environment, using standard mpi features.
OGGM depends on mpi4py in that case, which can be installed via either conda::

    conda install -c conda-forge mpi4py

or pip::

    pip install mpi4py


mpi4py itself depends on a working mpi environment, which is usually supplied by the maintainers of your cluster.
On conda, it comes with its own copy of mpich, which is nice and easy for quick testing, but likely undesireable for the performance of actual runs.

For an actual run, invoke any script using oggm via mpiexec, and pass the ``--mpi`` parameter to the script itself::

    mpiexec -n 10 python ./run_rgi_region.py --mpi

Be aware that the first process with rank 0 is the manager process, that by itself does not do any calculations and is only used to distribute tasks.
So the actual number of working processes is one lower than the number passed to mpiexec/your clusters scheduler.
