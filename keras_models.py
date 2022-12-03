import logging
from typing import Dict, List

import numpy as np

from utils import evaluate_metrics
from _base import Communicator, Config, Model, Server, Update
from keras_gmf import GMF
from utils import timed_op
import pandas as pd
import numpy as np
from typing import Tuple

logging.basicConfig(
    format="[%(asctime)s]: %(message)s (%(filename)s:%(lineno)d)",
    level=logging.INFO,
    datefmt="%H.%M.%S",
)

LOG = logging.getLogger()


def calculate_metrics(
    recommendations: np.ndarray, test_matrix: np.ndarray
) -> Tuple[Dict[str, float], List[float]]:
    max_hits = np.sum(test_matrix > 0)
    rateable = np.count_nonzero(np.sum(test_matrix, axis=1))
    top_n = recommendations.shape[1]
    tp_fn = rateable * top_n
    true_positives = 0
    __ux = []
    hr = []
    for user in range(recommendations.shape[0]):
        _ux = 0
        found = False
        for item in recommendations[user]:
            if test_matrix[user, item] > 0:
                true_positives += 1
                _ux += 1
                found = True
        __ux.append(_ux / recommendations.shape[1])
        if found:
            hr.append(1)
        else:
            hr.append(0)
    precision = true_positives / tp_fn
    recall = true_positives / max_hits
    result = {
        "precision": precision,
        "recall": recall,
        "hit_ratio": np.mean(hr),
    }
    return result, __ux


class KerasModel(Model):
    def initialize(self, config: Config, **kwargs: dict):
        self.model = GMF(
            n_users=kwargs["num_users"],
            n_items=kwargs["num_items"],
            regs=kwargs["regs"],
            embedding_size=kwargs["embed_size"],
            implicit=True,
        )  # type: tensorflow.keras.Model
        self.model.compile(
            optimizer=config.optimizer,
            loss=config.loss,
            # metrics=config.metrics,
        )

    def train_user(self, profile_df, config: Config) -> List[float]:
        if config.num_gpus > 1:

            from tensorflow.keras.utils import multi_gpu_model

            LOG.debug("Using %d GPUs" % config.num_gpus)

            # self.model must be recompiled if n_gpus > 1
            _metrics = self.model.metrics
            _optimizer = self.model.optimizer
            _loss = self.model.loss

            self.model = multi_gpu_model(self.model, gpus=config.n_gpus)
            self.model.compile(optimizer=_optimizer, loss=_loss, metrics=_metrics)

        assert profile_df["user_id"].nunique() == 1
        n_items = profile_df["item_id"].shape[0]

        # Training values
        params = {}
        if isinstance(config.batch_size, float):
            params["batch_size"] = max(1, int(n_items * config.batch_size))
        elif isinstance(config.batch_size, int):
            params["batch_size"] = min(n_items, config.batch_size)
        else:
            msg = "Expected batch_size to be int or float, but got {}"
            raise ValueError(msg.format(type(config.batch_size)))

        callbacks = []
        if config.early_stopping:
            import os

            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["USE_SIMPLE_THREADED_LEVEL3"] = "1"

            import tensorflow as tf

            es = tf.keras.callbacks.EarlyStopping(
                monitor="loss",
                verbose=config.verbose,
                restore_best_weights=config.use_best_weights,
            )
            callbacks.append(es)

        msg = (
            f"training model using batch_size {params['batch_size']:,} (of {n_items:,})"
        )
        with timed_op(msg):
            x = [profile_df["user_id"].values, profile_df["item_id"].values]
            y = profile_df["rating"].values
            history = self.model.fit(
                x=x,
                y=y,
                epochs=config.epochs_per_round,
                verbose=config.verbose,
                batch_size=params["batch_size"],
                callbacks=callbacks or None,
            )

        return list(history.history["loss"])

    def get_weights(self) -> List[np.ndarray]:
        return self.model.get_weights()

    def set_weights(self, weights: List[np.ndarray]):
        assert not np.array_equal(weights, self.model.get_weights())
        self.model.set_weights(weights)

    def evaluate(
        self,
        data: pd.DataFrame,
        full: bool = True,
        **kwargs,
    ) -> Dict[str, float]:
        batch_size = kwargs.get("batch_size", data.shape[0])
        item_ids = data["item_id"].values
        ratings = data["rating"].values  # np.ones(data.shape[0])
        user_ids = data["user_id"].values
        verbose = 0  # 0 = silent, 1 = verbose, 2 = one log line per epoch

        results = self.model.evaluate(
            [user_ids, item_ids], ratings, batch_size, verbose=verbose
        )
        test_loss = results if isinstance(results, float) else results[0]
        predictions = self.model.predict(
            [data["user_id"].values, data["item_id"].values],
            batch_size=batch_size,
            verbose=verbose,
        )
        data["predictions"] = predictions

        metrics = evaluate_metrics(data=data) if full else {}
        metrics["loss"] = test_loss
        return metrics


class KerasServer(Server):
    def __init__(
        self, size: int, config: Config, comm: Communicator, model: Model
    ) -> None:
        super(KerasServer, self).__init__(
            size=size, config=config, comm=comm, model=model
        )

    def average_models(
        self, updates: List[Update], prev_weights: List[np.ndarray]
    ) -> List[np.ndarray]:
        # Placeholder for aggregated weights from local updates
        curr_weights = [np.zeros(w.shape) for w in prev_weights]
        weights, train_sizes, user_ids = [], [], []
        for update in updates:
            weights.append(update.weights)
            train_sizes.append(update.train_size)
            user_ids.append(update.user_id)
        total_sizes = np.sum(train_sizes)
        LOG.info(train_sizes)
        LOG.info("num_unique_users.average_models %d", len(set(user_ids)))
        for user_idx in range(len(weights)):
            for layer_idx in range(len(curr_weights)):
                curr_weights[layer_idx] += (
                    weights[user_idx][layer_idx] * train_sizes[user_idx] / total_sizes
                )
        return curr_weights
