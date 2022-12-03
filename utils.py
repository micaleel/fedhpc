import heapq
import math
import time
from contextlib import contextmanager
from datetime import timedelta
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@contextmanager
def timed_op(name: str = None, logger=None):
    t0 = time.time()
    try:
        yield
    finally:
        duration = time.time() - t0
        duration_str = str(timedelta(seconds=duration))
        if logger is not None:
            logger.info("Finished %s in %s", name, duration_str)


def plot_history(
    history: Union[dict, "tensorflow.keras.callbacks.History"],
    loss=None,
    figsize: Tuple[float, float] = (12, 4),
    ylabel: Optional[str] = None,
):
    loss = loss or "loss"
    _history = history if isinstance(history, dict) else history.history
    metrics = {m.replace("val_", "") for m in _history.keys()}
    xpad = 0.2
    fig, axes = plt.subplots(ncols=len(metrics), figsize=figsize)
    for metric, ax in zip(metrics, np.atleast_1d(axes)):
        sub_metrics = {"training": metric, "validation": "val_{}".format(metric)}
        for m in sub_metrics.values():
            if m not in _history:
                continue
            Y = np.array(_history[m])
            X = np.arange(Y.shape[0])
            ax.plot(X, Y)
            ax.set_xlim(-xpad, Y.shape[0] - (1 - xpad))
            ax.set_xticks(np.arange(Y.shape[0]))
        ax.legend([x.replace("loss", loss) for x in sub_metrics.keys()])
        ax.set_title(
            "{metric} (Training: {train:.4f})".format(
                metric=metric.replace("loss", loss), train=_history[metric][-1]
            )
        )
        ax.set_xlabel("Epochs")
        if ylabel is not None:
            ax.set_ylabel(ylabel)


def get_ndcg_binary(ranked_list, item_id):
    """
    Compute the NDCG of a ranked with with binary relevance

    Args:
        ranked_list (List[int]): recommendations

    Todo:
        Change implementation to allow for multiple item IDs. Current
            implementation only works for leave-one-out protocol.

    Return:
        float: NDCG
    """
    try:
        ndcg = math.log(2) / math.log(ranked_list.index(item_id) + 2)
    except ValueError:
        ndcg = 0
    return ndcg


def evaluate_metrics(
    data: pd.DataFrame,
    data_grouped: Optional[pd.core.groupby.generic.DataFrameGroupBy] = None,
    topk: int = 10,
) -> Dict[str, float]:
    num_users = data["user_id"].nunique()
    hits, ndcgs = np.zeros(num_users), np.zeros(num_users)

    if data_grouped is None:
        data_grouped = data.groupby("user_id")
    for idx, (user_id, user_samples) in enumerate(data.groupby("user_id")):
        pos_samples = np.atleast_1d(user_samples.iloc[0]["pos_item_id"])
        ranked_list = (
            user_samples.drop(columns=["pos_item_id"])
            .sort_values("predictions", ascending=False)
            .head(topk)
        )
        ranked_list_items = ranked_list["item_id"].tolist()
        hit = len(np.intersect1d(ranked_list_items, pos_samples))
        try:
            ndcg = math.log(2) / math.log(ranked_list_items.index(pos_samples) + 2)
        except ValueError:
            ndcg = 0

        hits[idx] = hit
        ndcgs[idx] = ndcg
    hr = hits.mean()
    ndcg = ndcgs.mean()
    return {"hit_ratio": hr, "ndcg": ndcg}
