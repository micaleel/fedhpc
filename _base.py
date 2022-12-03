#!/usr/bin/env python
import multiprocessing

# Crucial for training concurrent Tensorflow models, phew!!
multiprocessing.set_start_method("spawn", force=True)

import logging
import platform
import time
from abc import abstractmethod
from contextlib import contextmanager
from datetime import timedelta
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from attr import dataclass
from mpi4py import MPI
from mpi4py.MPI import Intercomm, Intracomm

from datautils import ML100K

logging.basicConfig(
    format="[%(asctime)s]: %(message)s (%(filename)s:%(lineno)d)",
    level=logging.INFO,
    datefmt="%H.%M.%S",
)

LOG = logging.getLogger()

# hyper-parameters
ITERATIONS = 30
USERS_PER_ROUND = 0.1  # type: Union[float, int]
EPOCHS_PER_ROUND = 3

# tags for MPI communication
T_MODEL_FROM_SERVER = 10
T_MODEL_FROM_WORKER = 20
T_MODEL_USERS = 30
T_MODEL_JSON = 40
T_TRAIN_DF = 50
T_MODEL_INIT_KWARGS = 60
T_TRAINING = 70
T_FINALIZE = 80


# types
comm = MPI.COMM_WORLD
Communicator = Union[Intercomm, Intracomm]


@dataclass
class Config:
    """Persistable configuration"""

    batch_size: Union[float, int] = 0.1
    early_stopping: bool = True
    embedding_size: int = 8  # 10
    epochs_per_round: int = 6
    num_gpus: Optional[int] = 0
    regs: Optional[Tuple[float, float]] = None
    use_best_weights: bool = True
    users_per_round: Union[float, int] = 0.1
    verbose: bool = False
    optimizer: str = "adam"
    loss: str = "binary_crossentropy"
    # metrics: Optional[List[str]] = ["mae"]
    full_evaluation: bool = True

    def get_users_per_round(self, num_users: int):
        if isinstance(self.users_per_round, int):
            return self.users_per_round
        else:
            return int(USERS_PER_ROUND * num_users)


class Model:
    """Wraps a PyTorch or Keras model"""

    @abstractmethod
    def initialize(config: Config, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train_user(self, profile_df, config: Config) -> List[float]:
        raise NotImplementedError

    @abstractmethod
    def get_weights(self) -> List[np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def set_weights(self, weights: List[np.ndarray]):
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        data: pd.DataFrame,
        full: bool = True,
        **kwargs,
    ) -> Dict[str, float]:
        raise NotImplementedError


class PyTorchModel(Model):
    def __init__(self) -> None:
        self.model = None

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, weights):
        self.model.load_state_dict(weights)


@dataclass
class Update:
    """Result of a training operation"""

    history: List[float] = None
    train_size: int = None
    user_id: int = None
    weights: Optional[List[Any]] = None
    iteration: int = None


class Event(NamedTuple):
    """Record of a timed event"""

    worker_id: int
    node: str
    msg: str
    duration: float
    iteration: Optional[int]
    tag: str


class _ProcessBase:
    """Base class for all MPI processes"""

    def __init__(self, config: Config, comm: Communicator):
        self.comm = comm
        self.config = config
        self.events = []  # type: List[Dict[str, Any]]
        self.rank = comm.Get_rank()  # type: int
        self.worker_id = comm.Get_rank()  # type: int

    def log_event(
        self,
        msg: str,
        duration: float,
        iteration: Optional[int] = None,
        tag: str = None,
    ):
        self.events.append(
            Event(
                duration=duration,
                iteration=iteration,
                msg=msg,
                node=platform.node(),
                tag=tag,
                worker_id=self.rank,
            )
        )

    @contextmanager
    def logged_event(
        self,
        msg: str,
        iteration: Optional[Dict] = None,
        tag: Optional[str] = None,
        logger=None,
    ):
        t0 = time.time()
        try:
            if logger:
                logger.info("Started %s", msg)
            yield
        finally:
            interval = time.time() - t0
            self.log_event(msg=msg, duration=interval, iteration=iteration, tag=tag)
            if logger:
                logger.info("Finished %s in %s", msg, str(timedelta(interval)))


class Server(_ProcessBase):
    def __init__(
        self, size: int, config: Config, comm: Communicator, model: Model
    ) -> None:
        super(Server, self).__init__(config=config, comm=comm)
        LOG.info("Initializing Server in federated_base")
        self.model = model
        self.worker_ids = [x for x in range(size) if x != 0]

        # Data for model training and testing
        self.train_df = None  # type: Optional[pd.DataFrame]
        self.test_df = None  # type: Optional[pd.DataFrame]

        self.user_ids = None  # type: List[int]
        self.item_ids = None  # type: List[int]

        # For tracking model performance across communication rounds
        self.train_losses = dict()  # type: Dict[int, float]
        self.test_metrics = []  # type: List[Dict]

        self.users_per_round: int = 0.1

    def load_data(self):
        self.train_df, self.test_df = ML100K.load(loo=True, chrono=True)

        self.user_ids = self.train_df["user_id"].unique()
        self.item_ids = self.train_df["item_id"].unique()

    def broadcast(self, data: Any, tag: str):
        for worker_id in self.worker_ids:
            self.comm.send(data, dest=worker_id, tag=tag)

    def prepare(self) -> None:
        with self.logged_event(msg="Loading data from disk on server"):
            self.load_data()

        num_users, num_items = len(self.user_ids), len(self.item_ids)
        self.users_per_round = self.config.get_users_per_round(num_users)

        # Create model on server
        model_init_kwds = dict(
            embed_size=self.config.embedding_size,
            num_items=num_items,
            num_users=num_users,
            regs=self.config.regs,
        )
        LOG.info("About to initialise model")
        self.model.initialize(self.config, **model_init_kwds)

        # Send model initialisation params to all workers
        with self.logged_event(
            msg="Broadcasting model initiliasation parameters",
            tag=T_MODEL_INIT_KWARGS,
            logger=LOG,
        ):
            self.broadcast(model_init_kwds, T_MODEL_INIT_KWARGS)

        # Send training data
        with self.logged_event(
            msg="Broadcasting training data", tag=T_TRAIN_DF, logger=LOG
        ):
            self.broadcast(self.train_df.to_dict(), T_TRAIN_DF)

    @abstractmethod
    def average_models(self, updates: List[Update], prev_weights: List[np.ndarray]):
        raise NotImplementedError

    def train(self, iteration) -> None:
        """Called at the start of a communication round"""
        sampled_user_ids = np.random.choice(self.user_ids, self.users_per_round)

        # Send model weights to all workers
        with self.logged_event(
            f"Broadcasting model weights for iteration {iteration}",
            iteration=iteration,
            tag=T_MODEL_FROM_SERVER,
            logger=LOG,
        ):
            weights = self.model.get_weights()
            self.broadcast(weights, T_MODEL_FROM_SERVER)

        # Split users across different workers
        chunks = np.array_split(sampled_user_ids, len(self.worker_ids))
        with self.logged_event(
            f"Splitting users across workers {iteration}",
            iteration=iteration,
            tag=T_MODEL_FROM_SERVER,
        ):
            for worker_id, chunk in zip(self.worker_ids, chunks):
                self.comm.send(list(chunk), dest=worker_id, tag=T_MODEL_USERS)

        # Wait for response from clients
        updated_weights = []
        for worker_id in self.worker_ids:
            new_weights = self.comm.recv(source=worker_id, tag=T_MODEL_FROM_WORKER)
            updated_weights.extend(new_weights)

        # Aggregate local models to update global model
        new_state = self.average_models(updated_weights, self.model.get_weights())
        self.model.set_weights(new_state)

        losses = [u.history[-1] for u in updated_weights]
        self.train_losses[iteration] = np.mean(losses)

        # Evaluate on test set
        test_metrics = self.model.evaluate(
            data=self.test_df,
            full=self.config.full_evaluation,
        )
        test_metrics["iteration"] = iteration
        self.test_metrics.append(test_metrics)

    def finalise(self) -> None:
        for worker_id in self.worker_ids:
            worker_events = self.comm.recv(source=worker_id, tag=T_FINALIZE)
            self.events.extend(worker_events)


class Worker(_ProcessBase):
    def __init__(self, comm: Communicator, config: Config, model: Model) -> None:
        super(Worker, self).__init__(config=config, comm=comm)
        self.model = model
        self.train_df = None  # type: pd.DataFrame

    def read_from_server(self, tag):
        return self.comm.recv(source=0, tag=tag)

    def prepare(self):
        model_init_kwds = self.read_from_server(T_MODEL_INIT_KWARGS)

        # Initialise model
        LOG.info("Model initialization in worker")
        self.model.initialize(self.config, **model_init_kwds)

        # Reconstruct model
        self.train_df = pd.DataFrame.from_dict(self.read_from_server(T_TRAIN_DF))

    def train(self, iteration) -> None:
        prefix = "[Worker %02d | Iteration %d]" % (self.rank, iteration)

        # Get model from server
        weights = self.read_from_server(T_MODEL_FROM_SERVER)
        LOG.debug("Weights received from server")

        # Get list of users from server
        users = self.read_from_server(tag=T_MODEL_USERS)

        # Train user models
        with self.logged_event(
            "Training user models", tag=T_TRAINING, iteration=iteration
        ):
            updates = []  # type: List[Update]
            for user in users:
                # Restore model to state identical to server
                LOG.debug("Restoring weights")
                self.model.set_weights(weights)

                # Use current user's data to train the model
                # TODO replace expensive query() call
                profile_df = self.train_df.query("user_id == @user")

                LOG.debug(
                    "%s Training user %4d [size %d]", prefix, profile_df.shape[0], user
                )
                train_losses = self.model.train_user(profile_df, self.config)

                updates.append(
                    Update(
                        history=train_losses,
                        train_size=profile_df.shape[0],
                        user_id=user,
                        weights=np.copy(self.model.get_weights()),
                        iteration=iteration,
                    )
                )
        # Send updated model back to server
        with self.logged_event(
            "Sending model to server",
            iteration=iteration,
            tag=T_MODEL_FROM_WORKER,
        ):
            self.comm.send(updates, dest=0, tag=T_MODEL_FROM_WORKER)

    def finalise(self) -> None:
        # Send events to server
        self.comm.send(self.events, dest=0, tag=T_FINALIZE)


def plot_fed_result(client: Server, fpath="history-fed.png"):
    df = pd.DataFrame(client.test_metrics).set_index("iteration")
    columns = [c for c in df.columns.tolist() if c != "iteration"]

    ncols = len(columns) + 1
    fig, axes = plt.subplots(ncols=ncols, figsize=(3.5 * ncols, 4))
    axes = axes.flatten()

    ax = axes[-1]
    x = np.arange(len(client.train_losses.keys()))
    y = np.array(list(client.train_losses.values()))

    ax.plot(x, y)
    ax.set_title("Train losses")
    ax.set_xlabel("Iteration")

    titles = {"hit_ratio": "Test HR@10", "ndcg": "Test NDCG@10", "loss": "Test Loss"}
    for column, ax in zip(columns, axes[:-1]):
        df.sort_index()[column].plot(ax=ax)
        ax.set_title(titles.get(column, column))

    fig.tight_layout()
    plt.savefig(fpath, dpi=600)
