import os
import random
from pathlib import Path
from typing import List, Optional, Union, Tuple

import logging
import numpy as np
import pandas as pd
from scipy.sparse import dok_matrix
from sklearn.preprocessing import LabelEncoder

PROJECT_DIR = Path(__file__).parent.resolve().as_posix()
DATA_DIR = os.path.join(PROJECT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

RandomState = Optional[Union[int, np.random.RandomState]]
logging.basicConfig(
    format="[%(asctime)s]: %(message)s (%(filename)s:%(lineno)d)",
    level=logging.INFO,
    datefmt="%H.%M.%S",
)

LOG = logging.getLogger()


def train_test_split(
    df: pd.DataFrame,
    train_size=0.8,
    chrono=False,
    random_state: RandomState = None,
    loo=False,
):
    """Split a ratings DataFrame into a training and test data"""
    if chrono:
        assert "timestamp" in df.columns, "Missing timestamp column"
        df = df.sort_values("timestamp")
    else:
        df = df.sample(n=df.shape[0], random_state=random_state)

    train_indices, test_indices = set(), set()
    for user_id, user_df in df.groupby("user_id"):
        ratings_idxs = user_df.index.to_numpy()  # indices for this user's ratings
        if loo:
            train_idxs = ratings_idxs[:-1]
            test_idxs = ratings_idxs[-1]
            assert 1 + train_idxs.shape[0] == ratings_idxs.shape[0]
        else:
            n_train = int(ratings_idxs.shape[0] * train_size)
            train_idxs = ratings_idxs[:n_train]
            test_idxs = np.setdiff1d(ratings_idxs, train_idxs)
            assert test_idxs.shape[0] + train_idxs.shape[0] == ratings_idxs.shape[0]

        train_indices.update(train_idxs)
        test_indices.update(test_idxs) if isinstance(
            test_idxs, np.ndarray
        ) else test_indices.add(test_idxs)
    return df.loc[list(train_indices)], df.loc[list(test_indices)]


class NegativeSampler:
    def __init__(self, item_ids: Optional[List[int]] = None, use_numpy=True):
        self.train = None  # type: Optional[pd.DataFrame]
        self.test = None  # type: Optional[pd.DataFrame]
        self.item_ids = set(item_ids)
        self.use_numpy = use_numpy

    @staticmethod
    def get_user_interactions(df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby("user_id")["item_id"].apply(set).reset_index()

    @staticmethod
    def random_sample(
        row,
        col,
        size=10,
        rng: np.random._generator.Generator = None,
        seed=42,
        use_numpy=True,
    ):
        rng = np.random.default_rng(seed) if rng is None else rng
        candidates = set(row[col]) - {row["item_id"]}
        if size < len(candidates):
            if use_numpy:
                neg_samples = rng.choice(a=list(candidates), size=size)
            else:
                neg_samples = random.sample(population=candidates, k=size)
        else:
            neg_samples = candidates
        return neg_samples

    def fit(self, train: pd.DataFrame, test: pd.DataFrame):
        self.train, self.test = train, test
        if len(self.item_ids) == 0:
            self.item_ids = frozenset(
                np.concatenate([train["item_id"].unique(), test["item_id"].unique()])
            )

        train_profiles = self.get_user_interactions(train).rename(
            columns={"item_id": "item_id_pos_train"}
        )
        # Create pool of negative items per user
        train_profiles["item_id_neg_train"] = train_profiles["item_id_pos_train"].apply(
            lambda x: self.item_ids - x
        )
        test_interactions = self.get_user_interactions(test).rename(
            columns={"item_id": "item_id_pos_test"}
        )
        assert len(train_profiles) == len(test_interactions)
        self.interactions = pd.merge(
            train_profiles, test_interactions, on="user_id", how="inner"
        )
        del train_profiles, test_interactions
        self.interactions["item_id_neg_test"] = self.interactions.apply(
            lambda x: x["item_id_neg_train"] - x["item_id_pos_test"], axis="columns"
        )
        self.interactions = self.interactions.drop(
            columns=["item_id_pos_test", "item_id_pos_train"]
        )
        return self

    def transform(self, df: pd.DataFrame, train: bool = True, size=10):
        # Attach negative samples to test DataFrame
        col = "item_id_neg_train" if train else "item_id_neg_test"
        merged = pd.merge(
            df, self.interactions[["user_id", col]], on="user_id", how="inner"
        )
        merged["neg_samples"] = merged.apply(
            lambda x: self.random_sample(
                row=x, col=col, size=size, use_numpy=self.use_numpy
            ),
            axis="columns",
        )
        merged = merged.drop(columns=[col])

        cols = ["item_id", "user_id", "neg_samples"]
        merged = merged[cols]

        def _f():
            item_id_idx = cols.index("item_id") + 1  # cater for DataFrame index
            user_id_idx = cols.index("user_id") + 1  # cater for DataFrame index
            neg_samples_idx = cols.index("neg_samples") + 1  # cater for DataFrame index

            for r in merged.to_records():
                item_id = r[item_id_idx]
                user_id = r[user_id_idx]
                yield {
                    "user_id": user_id,
                    "item_id": item_id,
                    "rating": 1,
                    "pos_item_id": item_id,
                }
                for sample in r[neg_samples_idx]:
                    yield {
                        "user_id": user_id,
                        "item_id": sample,
                        "rating": 0,
                        "pos_item_id": item_id,
                    }

        concat_df = pd.DataFrame(_f())
        return concat_df


class ML100K:
    @classmethod
    def read_source(
        cls,
        path: Optional[str] = None,
        binary: bool = True,
    ) -> pd.DataFrame:
        path = os.path.join(DATA_DIR, "ml-100k/u.data")
        ratings = pd.read_csv(
            path,
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"],
        )
        if binary:
            ratings["rating"] = 1
        for c in ["user_id", "item_id"]:
            ratings[c] = LabelEncoder().fit(ratings[c]).transform(ratings[c])
        return ratings

    @classmethod
    def load(
        cls,
        path: Optional[str] = None,
        chrono=False,
        loo=False,
        random_state: RandomState = None,
        train_size=0.8,
        binary: bool = True,
        n_train: int = 5,
        n_test: int = 50,
        use_numpy: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = cls.read_source(path=path, binary=binary)

        LOG.info("Splitting data")
        trn_df, tst_df = train_test_split(
            df=df,
            train_size=train_size,
            chrono=chrono,
            random_state=random_state,
            loo=loo,
        )

        LOG.info("Negative sampling")
        sampler = NegativeSampler(
            item_ids=df["item_id"].unique(), use_numpy=use_numpy
        ).fit(
            train=trn_df,
            test=tst_df,
        )
        trn_df = sampler.transform(trn_df, train=True, size=n_train)
        tst_df = sampler.transform(tst_df, train=False, size=n_test)

        return trn_df, tst_df

    def load_matrix(
        cls,
        path: Optional[str] = None,
        subset: Optional[str] = None,
        chrono=False,
        loo=False,
        random_state: RandomState = None,
        train_size=0.8,
        binary: bool = True,
        cache: bool = True,
        n_train: int = 5,
        n_test: int = 50,
        use_numpy: bool = False,
        sparse: bool = False,
    ) -> Union[dok_matrix, np.ndarray]:
        df = cls.load(
            path=path,
            subset=subset,
            chrono=chrono,
            loo=loo,
            random_state=random_state,
            train_size=train_size,
            binary=binary,
            cache=cache,
            n_train=n_train,
            n_test=n_test,
            use_numpy=use_numpy,
        )
        matrix = np.zeros(shape=(df["user_id"].nunique(), df["item_id"].nunique()))
        matrix[df["user_id"], df["item_id"]] = df["rating"]
        return dok_matrix(matrix) if sparse else matrix
