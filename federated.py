#!/usr/bin/env python

"""
Usage: 
    
    mpirun -n <num_procs> ./federated.py 

if not executed with the right permission, you'll get an error similar to this:

    [proxy:0:0@<machine-id>] HYDU_create_process (utils/launch/launch.c:74): execvp error on file ./federated.py (Permission denied)

To fix this, run `chmod +x federated.py`

"""
import logging
import sys
from typing import Optional
import json
from tabulate import tabulate
import mpi4py
import pandas as pd
from _base import Config, Worker, plot_fed_result
from keras_models import KerasModel, KerasServer
from utils import timed_op

logging.basicConfig(
    format="[%(asctime)s]: %(message)s (%(filename)s:%(lineno)d)",
    level=logging.INFO,
    datefmt="%H.%M.%S",
)

LOG = logging.getLogger()

try:
    comm = mpi4py.MPI.Intercomm()
    # Could raise an exception with the following message:
    #
    # mpi4py.MPI.Exception: Invalid communicator, error stack:
    #     PMPI_Comm_rank(111): MPI_Comm_rank(MPI_COMM_NULL, rank=0x7ffe8218b174) failed
    #     PMPI_Comm_rank(68).: Null communicator
    #
    # This could suggest that script is run on a single machine, not a cluster
    rank = comm.Get_rank()
except mpi4py.MPI.Exception:
    comm = mpi4py.MPI.COMM_WORLD
finally:
    rank = comm.Get_rank()
    size = comm.Get_size()


def check_context():
    """Ensure that script is run with mpirun"""
    if size == 1:
        LOG.warning(
            "The MPI COMM_WORLD only has access to one process. "
            "Ensure this script is executed with mpirun. "
            f"e.g. mpirun -n <numprocs> ./{__file__}"
        )
        sys.exit("Script must be invoked using mpirun command")


def create_client(config: Optional[Config] = None):
    config = config or Config()
    model = KerasModel()
    params = dict(config=config, comm=comm, model=model)
    return KerasServer(size=size, **params) if rank == 0 else Worker(**params)


def summarize_events(events):
    df = pd.DataFrame(
        events,
        columns=["worker_id", "node", "msg", "duration", "iteration", "tag"],
    )
    num_nodes = df["node"].nunique()
    node_list = df["node"].unique().tolist()
    print("Trained on {} nodes ({})".format(num_nodes, ", ".join(sorted(node_list))))


def main(iterations=10):
    check_context()
    client = create_client()

    LOG.debug("Preparing client at rank %d", rank)
    client.prepare()

    for iteration in range(iterations):
        with timed_op("Training Worker %02d (Iteration %d)" % (rank, iteration), LOG):
            client.train(iteration=iteration)

    client.finalise()

    if rank == 0:
        # Save events to disk
        with open("events.json", "w") as fp:
            json.dump(client.events, fp=fp)

        summarize_events(client.events)
        plot_fed_result(client)
        LOG.info("Finished")


if __name__ == "__main__":
    with timed_op("federated learning (on rank %d)" % rank, LOG):
        main()
