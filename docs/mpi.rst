Using OGGM with MPI
===================

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
