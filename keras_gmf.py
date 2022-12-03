#!/usr/bin/env python
import logging
import os
import random
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from attr import dataclass
from mpi4py import MPI

from datautils import ML100K
from utils import plot_history, timed_op

LOG = logging.getLogger("fedfast")
EPOCHS = 10

__all__ = ["GMF", "user_embeddings_from_weights"]

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Turn off TensorFlow deprecation warnings
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


@dataclass
class Update:
    """Result of a training operation"""

    history: Dict[str, Union[np.ndarray, list]] = None
    train_size: int = None
    user_id: int = None
    weights: Optional[List[Any]] = None
    iteration: int = None

    # Results from client-side testing
    test_score: Dict[str, Optional[float]] = None

    @property
    def final(self):
        d = {metric: values[-1] for metric, values in self.history.items()}
        d["iteration"] = self.iteration
        d["user_id"] = self.user_id
        d["train_size"] = self.train_size
        return d

    @property
    def metrics(self) -> List[str]:
        return list(set(m.replace("val_", "") for m in self.metric_names()))

    def metric_names(self) -> List[str]:
        return list(self.history.keys())

    def get_metric(self, m) -> List[float]:
        return self.history.get(m, [])

    @property
    def loss(self) -> List[float]:
        return self.history.get("loss", [])

    @property
    def val_loss(self) -> List[float]:
        return self.history.get("val_loss", [])


def set_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def _create_gmf(n_users, n_items, regs=None, embedding_size=10, implicit=False):
    """Create a GMF model

    Args:
        n_users (int): number of user embeddings
        n_items (int): number of item embeddings
        regs[Optional[List[int]]]: regularization parameters for embedding layer
        embedding_size(int): size of each user and item embedding

    Returns:
        "keras.models.Model": GMF model
    """
    # To control where and how TensorFlow sessions are created, we
    # do imports here make keras work well with multiprocessing.
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import Dense, Embedding, Flatten, multiply
    from tensorflow.keras.regularizers import l2

    if regs:
        assert len(regs) == 2

    regs = regs or [0, 0]
    user_input = Input(shape=(1,), dtype="int32", name="user_input")
    item_input = Input(shape=(1,), dtype="int32", name="item_input")

    mf_embedding_user = Embedding(
        input_dim=n_users,
        output_dim=embedding_size,
        name="user_embedding",
        embeddings_initializer="normal",
        embeddings_regularizer=l2(regs[0]),
        input_length=1,
    )
    mf_embedding_item = Embedding(
        input_dim=n_items,
        output_dim=embedding_size,
        name="item_embedding",
        embeddings_initializer="normal",
        embeddings_regularizer=l2(regs[1]),
        input_length=1,
    )

    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(mf_embedding_user(user_input))
    item_latent = Flatten()(mf_embedding_item(item_input))

    # Element-wise product of user and item embeddings
    predict_vector = multiply([user_latent, item_latent])
    # TODO apply Dropout after the mutliply operation?

    # Final prediction layer
    if implicit:
        LOG.debug("GMF model to predict binary implicit preference")
        # prediction = Lambda(lambda x: K.sigmoid(K.sum(x)), output_shape=(1,))(predict_vector)
        prediction = Dense(
            1,
            activation="sigmoid",
            kernel_initializer="lecun_uniform",
            name="prediction",
        )(predict_vector)
    else:
        LOG.debug("Prediction layer configured for explicit feedback")
        prediction = Dense(
            1,
            activation="linear",
            kernel_initializer="lecun_uniform",
            name="prediction",
        )(predict_vector)
    model = Model(inputs=[user_input, item_input], outputs=prediction)

    return model


GMF = _create_gmf


def compile_model(
    model,  # type: tf.keras.models.Model
    optimizer,  # type: tf.keras.optimizers.Optimizer
    loss=None,  # type: Callable[[Any, Any], Any]
    metrics=None,  # type: Union[List[str], str]
    summary=False,  # type: bool
    **kwargs,
):
    """Compiles a keras model"""
    if isinstance(metrics, str):
        metrics = [metrics]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)
    if summary:
        model.summary()
    return model


# noinspection PyUnresolvedReferences
def fit_model(
    keras_model: "tf.keras.models.Model",
    data: pd.DataFrame,
    epochs=1,
    verbose=0,
    n_gpus=0,
    batch_size=None,
    early_stopping=True,
    # validation_split: Optional[float] = None,  # 0.05,
) -> "tf.keras.callbacks.History":
    from tensorflow.keras.optimizers import Adam

    if validation_split:
        assert 0 <= validation_split <= 1, "validation_split must be in range [0,1]"
    LOG.debug("Fitting model")

    if n_gpus > 1:
        from tensorflow.keras.utils import multi_gpu_model

        LOG.debug("fit() will use %d GPUs" % n_gpus)

        # keras_model must be recompiled in n_gpus > 1
        _metrics = keras_model.metrics
        _optimizer = keras_model.optimizer
        _loss = keras_model.loss

        keras_model = multi_gpu_model(keras_model, gpus=n_gpus)
        keras_model.compile(optimizer=_optimizer, loss=_loss, metrics=_metrics)

    x, y, n_items = [], [], 0

    x = [data["user_id"].values, data["item_id"].values]
    y = data["rating"].values
    n_items = data["item_id"].shape[0]

    # Training values
    params = {}
    if batch_size is None or batch_size <= 0:
        if "batch_size" in params:
            del params["batch_size"]
    elif isinstance(batch_size, float):
        params["batch_size"] = max(1, int(n_items * batch_size))
    elif isinstance(batch_size, int):
        params["batch_size"] = min(n_items, batch_size)
    else:
        raise ValueError(
            "Invalid batch size. Expected int or float but got {}".format(
                type(batch_size)
            )
        )

    LOG.debug(f"fit() using batch_size {params['batch_size']:,} (of {n_items:,})")
    LOG.debug("Fitting the model")
    start = time.time()
    callbacks = []
    if early_stopping:
        from tensorflow.keras.callbacks import EarlyStopping

        es = EarlyStopping(monitor="loss", verbose=verbose, restore_best_weights=True)
        callbacks.append(es)
    history = keras_model.fit(
        x=x,
        y=y,
        epochs=epochs,
        verbose=int(verbose),
        # validation_split=validation_split,
        batch_size=params["batch_size"],
        callbacks=callbacks,
    )
    duration = time.time() - start
    LOG.debug(f"Done fitting the model in {duration:,.4f} seconds")
    return history


# noinspection PyUnresolvedReferences
def _evaluate_model(
    keras_model: "tf.keras.models.Model",
    data: pd.DataFrame,
    verbose: bool = False,
):
    LOG.debug(f"Evaluating model on test set")
    item_ids, user_ids, ratings = None, None, None

    item_ids = data["item_id"]
    user_ids = data["user_id"]
    ratings = data["rating"]

    score = keras_model.evaluate([user_ids, item_ids], ratings, verbose=verbose)
    return score


def evaluate_model(history, model, loss, test):
    for is_trainset in (True, False):
        is_validset = not is_trainset
        if is_trainset:
            LOG.info("\nTrain Errors:")
        else:
            LOG.info("\nValidation Errors:")

        for metric, metric_values in history.history.items():
            if is_trainset and metric.startswith("val_"):
                continue
            if is_validset and not metric.startswith("val_"):
                continue
            metric_value = metric_values[-1]
            if metric.lower() == loss and loss.lower() in ("mse", "mean_squared_error"):
                metric_value = np.sqrt(metric_value)

            metric_name = metric.upper()
            if metric.lower() == "loss":
                metric.upper() + f"({loss.upper()})"
            LOG.info(f"  {metric_name:<40} = {metric_value:.4f}")

    LOG.info("\nTest Errors:")
    test_result = dict(zip(model.metrics_names, _evaluate_model(model, data=test)))
    if loss.lower() in ("mse" or "mean_squared_error"):
        test_result["loss"] = np.sqrt(test_result["loss"])
    for metric, metric_value in test_result.items():
        metric_name = metric.upper()
        if metric.lower() == "loss":
            metric.upper() + f"({loss.upper()})"
        LOG.info(f"  {metric_name:<40} = {metric_value:.4f}")
    return test_result


def main(
    seed=42,
    regs=(0, 0),
    embedding_size=8,
    loss="binary_crossentropy",
    optimizer="adam",
    metrics="mae",
    verbose=True,
    batch_size=32,
    n_gpus=2,
    epochs=10,
):
    # Set random seeds for reproducibility
    set_seeds(seed)

    train_df = ML100K.load(subset="train")
    test_df = ML100K.load(subset="test")
    train = train_df
    test = test_df

    # Initialise and compile model
    model = GMF(
        n_users=train_df["user_id"].nunique(),
        regs=regs,
        n_items=train_df["item_id"].nunique(),
        embedding_size=embedding_size,
        implicit=True,
    )
    loss_fn = loss
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
    LOG.debug("Model compiled")

    history = fit_model(
        keras_model=model,
        verbose=verbose,
        batch_size=batch_size,
        n_gpus=n_gpus,
        epochs=epochs,
        data=train,
    )
    plot_history(history, ylabel="binary_crossentropy")
    plt.savefig("history.png")


if __name__ == "__main__":
    with timed_op("Training GMF with TF"):
        main(epochs=EPOCHS)
