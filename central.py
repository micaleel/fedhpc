"""
This module test the performance of the Keras GMF model during centralised training
"""
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["USE_SIMPLE_THREADED_LEVEL3"] = "1"
import logging
import os
from pprint import pprint as pp

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import multi_gpu_model

from datautils import ML100K
from keras_gmf import GMF
from utils import evaluate_metrics, timed_op

# tf.config.threading.set_intra_op_parallelism_threads(1)
# tf.config.threading.set_inter_op_parallelism_threads(1)
logging.basicConfig(
    format="[%(asctime)s]: %(message)s (%(filename)s:%(lineno)d)",
    level=logging.INFO,
    datefmt="%H.%M.%S",
)

LOG = logging.getLogger()


def main():
    # load data
    train_df, test_df = ML100K.load(loo=True, chrono=True)
    LOG.info("n_users %d", train_df["user_id"].nunique())
    LOG.info("n_items %d", train_df["item_id"].nunique())
    # create model
    model = GMF(
        n_users=train_df["user_id"].nunique(),
        n_items=train_df["item_id"].nunique(),
        embedding_size=10,
        regs=[0, 0],
    )
    model.compile(optimizer="adam", loss="binary_crossentropy")

    # train model

    history = model.fit(
        x=[train_df["user_id"].to_numpy(), train_df["item_id"].to_numpy()],
        y=train_df["rating"].to_numpy(),
        batch_size=32,
        epochs=10,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="loss", verbose=True, restore_best_weights=True
            )
        ],
    )

    # plot loss
    plt.plot(history.history["loss"])
    plt.title("Train Loss")
    plt.savefig("history-cen.png", dpi=600)

    # evaluate model
    results = model.evaluate(
        x=[test_df["user_id"].values, test_df["item_id"].values],
        y=test_df["rating"].values,
        batch_size=test_df.shape[0],
    )
    LOG.info("Test result")
    pp(results)

    test_df["predictions"] = model.predict(
        [test_df["user_id"].values, test_df["item_id"].values]
    )
    results_metrics = evaluate_metrics(
        data=test_df, data_grouped=test_df.groupby("user_id")
    )
    pp(results_metrics)


if __name__ == "__main__":
    with timed_op():
        main()
