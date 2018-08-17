"""MPI process management helpers and setup"""
from mpi4py import MPI
from oggm import cfg
import atexit
import argparse
import sys

OGGM_MPI_SIZE = 0
OGGM_MPI_COMM = None
OGGM_MPI_ROOT = 0


def _imprint(s):
    print(s)
    sys.stdout.flush()


def _shutdown_slaves():
    global OGGM_MPI_COMM
    if OGGM_MPI_COMM is not None and OGGM_MPI_COMM != MPI.COMM_NULL:
        msgs = [StopIteration] * OGGM_MPI_SIZE
        status = MPI.Status()
        OGGM_MPI_COMM.bcast((None, None), root=OGGM_MPI_ROOT)
        for msg in msgs:
            OGGM_MPI_COMM.recv(source=MPI.ANY_SOURCE, status=status)
            OGGM_MPI_COMM.send(obj=msg, dest=status.Get_source())
        OGGM_MPI_COMM.gather(sendobj=None, root=OGGM_MPI_ROOT)
    OGGM_MPI_COMM = None


def _init_oggm_mpi():
    global OGGM_MPI_COMM, OGGM_MPI_SIZE

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mpi-help', action='help',
                        help='Prints this help text')
    parser.add_argument('--mpi', action='store_true',
                        help='Run OGGM in mpi mode')
    args, unkn = parser.parse_known_args()

    if not args.mpi:
        return

    OGGM_MPI_COMM = MPI.COMM_WORLD
    OGGM_MPI_SIZE = OGGM_MPI_COMM.Get_size() - 1
    rank = OGGM_MPI_COMM.Get_rank()

    if OGGM_MPI_SIZE <= 0:
        _imprint("Error: MPI world size is too small, at least one worker "
                 "process is required.")
        sys.exit(1)

    if rank != OGGM_MPI_ROOT:
        _mpi_slave()
        sys.exit(0)

    if OGGM_MPI_SIZE < 2:
        _imprint("Warning: MPI world size is small, this is pointless and "
                 "has no benefit.")

    atexit.register(_shutdown_slaves)

    _imprint("MPI initialized with a worker count of %s" % OGGM_MPI_SIZE)


def mpi_master_spin_tasks(task, gdirs):
    comm = OGGM_MPI_COMM
    cfg_store = cfg.pack_config()
    msg_list = ([gdir for gdir in gdirs if gdir is not None] +
                [None] * OGGM_MPI_SIZE)

    _imprint("Starting MPI task distribution...")

    comm.bcast((cfg_store, task), root=OGGM_MPI_ROOT)

    status = MPI.Status()
    for msg in msg_list:
        comm.recv(source=MPI.ANY_SOURCE, status=status)
        comm.send(obj=msg, dest=status.Get_source())

    _imprint("MPI task distribution done, collecting results...")

    comm.gather(sendobj=None, root=OGGM_MPI_ROOT)

    _imprint("MPI task results gotten!")


def _mpi_slave_bcast(comm):
    cfg_store, task_func = comm.bcast(None, root=OGGM_MPI_ROOT)
    if cfg_store is not None:
        cfg.unpack_config(cfg_store)
    return task_func


def _mpi_slave_sendrecv(comm):
    try:
        bufsize = int(cfg.PARAMS['mpi_recv_buf_size'])
    except BaseException:
        bufsize = None

    sreq = comm.isend(1, dest=OGGM_MPI_ROOT)
    rreq = comm.irecv(source=OGGM_MPI_ROOT, buf=bufsize)
    return sreq, rreq


def _mpi_slave():
    comm = OGGM_MPI_COMM
    rank = comm.Get_rank()

    _imprint("MPI worker %s ready!" % rank)

    task_func = _mpi_slave_bcast(comm)
    sreq, rreq = _mpi_slave_sendrecv(comm)

    while True:
        sreq.wait()
        task = rreq.wait()
        if task is None:
            comm.gather(sendobj="TASK_DONE", root=OGGM_MPI_ROOT)
            task_func = _mpi_slave_bcast(comm)
            sreq, rreq = _mpi_slave_sendrecv(comm)
            continue
        elif task is StopIteration:
            break
        sreq, rreq = _mpi_slave_sendrecv(comm)
        task_func(task)
    comm.gather(sendobj="WORKER_SHUTDOWN", root=OGGM_MPI_ROOT)

    _imprint("MPI Worker %s exiting" % rank)
    sys.exit(0)
