# src/app/models/item_based_cf.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from src.app.models.recommender import Recommender

@dataclass(frozen=True)
class IndexMaps:
    """
    user_enc: maps userID <-> integer index
    item_enc: maps recipeID <-> integer index
    """
    user_enc: LabelEncoder
    item_enc: LabelEncoder
    idx2user: List[str]
    idx2item: List[str]

@dataclass
class RatingsMatrix:
    """
    X_csr: mean-centered ratings, shape (U, I)
    user_means: shape (U,...)
    col_norms: L2 norms of item columns in X, shape (I,...)
    """
    X_csr: sparse.csr_matrix
    user_means: np.ndarray
    col_norms: np.ndarray

@dataclass
class SimilarityGraph:
    """
    For each recipe j: list of (i, score)
    """
    neighbors: List[List[Tuple[int, float]]]

class ItemBasedCF(Recommender):

    def __init__(
            self,
            json_path: str,
            min_user_ratings: int = 1,
            min_item_ratings: int = 1,
            topk_neighbors: int = 100,
            shrinkage_beta: float = 50.0,
            min_co_count: int = 2,
            block_size: int = 1000,
    ):
        """
        :param json_path: cleaned data
        :param min_user_ratings: minimum num user's ratings, preset to 1 -> filter our ppl less than threshold
        :param min_item_ratings: minimum num item's ratings, preset to 1 -> filter our ppl less than threshold
        :param topk_neighbors: num of neighbors to keep per recipe
        :param shrinkage_beta: shrinkage applied as n / (n + beta) where n is co-rating count
        :param min_co_count: min co-rating count required to consider an item pair
        :param block_size: num of item columns to process per block when computing similarities
        """
        super()
        self.df = ItemBasedCF._load_json_to_df(self, json_path)
        self.df = ItemBasedCF._filter_data(self, min_user_ratings, min_item_ratings)
        self.maps, self.R = ItemBasedCF._build_matrix(self)
        self.rm = ItemBasedCF._build_mean_centered_matrix(self)
        self.graph = self._item_item_topk(
            x=self.rm.X_csr,
            col_norms=self.rm.col_norms,
            topk=topk_neighbors,
            beta=shrinkage_beta,
            min_co_count=min_co_count,
            block_size=block_size,
        )


    """
    Data loading
    """
    def _load_json_to_df(self, path: str) -> pd.DataFrame:
        """
        Reads a JSON of the form:
        {
          "recipeA": [["user1", 4.0], ["user7", 5.0]],
          "recipeB": [["user3", 2.0]]
        }
        @:param path: path to json file
        @:returns df: a DataFrame with columns [userID, recipeID, rating].
        """
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        rows: List[Tuple[str, str, float]] = []
        for recipe_id, pairs in raw.items():
            for user_id, rating in pairs:
                rows.append((str(user_id), str(recipe_id), float(rating)))

        df = pd.DataFrame(rows, columns=["userID", "recipeID", "rating"])
        return df


    def _filter_data(self, min_user: int, min_item: int) -> pd.DataFrame:
        """
        Removes users and items with interactions lower than the set threshold
        """
        if min_user > 1:
            user_counts = self.df["userID"].value_counts()
            keep_users = set(user_counts[user_counts >= min_user].index)
            self.df = self.df[self.df["userID"].isin(keep_users)]
        if min_item > 1:
            item_counts = self.df["recipeID"].value_counts()
            keep_items = set(item_counts[item_counts >= min_item].index)
            self.df = self.df[self.df["recipeID"].isin(keep_items)]
        return self.df.reset_index(drop=True)


    """
    Making the matrix
    """
    def _build_matrix(self) -> Tuple[IndexMaps, sparse.csr_matrix]:
        """
        Encodes string IDs to integer indices and builds a CSR matrix R of shape (U, I).
        :returns: maps, R: sparse matrix of shape (U, I),
        """
        user_enc = LabelEncoder()
        item_enc = LabelEncoder()

        u_idx = user_enc.fit_transform(self.df["userID"])
        i_idx = item_enc.fit_transform(self.df["recipeID"])

        U = user_enc.classes_.size
        I = item_enc.classes_.size

        R = sparse.coo_matrix(
            (self.df["rating"].astype(np.float32), (u_idx, i_idx)),
            shape=(U, I),
        ).tocsr()

        maps = IndexMaps(
            user_enc=user_enc,
            item_enc=item_enc,
            idx2user=list(user_enc.classes_),
            idx2item=list(item_enc.classes_),
        )
        return maps, R

    def _build_mean_centered_matrix(self) -> RatingsMatrix:
        """
        Adjusted cosine: for each user row, subtract the user's mean from all their nonzero ratings.
        Produces X and precomputed item column norms.
        """
        U, I = self.R.shape

        user_nnz = np.diff(self.R.indptr)
        user_sums = np.asarray(self.R.sum(axis=1)).ravel()
        user_means = np.zeros(U, dtype=np.float32)
        mask = user_nnz > 0
        user_means[mask] = (user_sums[mask] / user_nnz[mask]).astype(np.float32)

        X = self.R.astype(np.float32).copy()
        for u in range(U):
            start, end = X.indptr[u], X.indptr[u + 1]
            if start < end:
                X.data[start:end] -= user_means[u]

        col_norms = np.sqrt(np.asarray(X.power(2).sum(axis=0)).ravel()).astype(np.float32)
        return RatingsMatrix(X_csr=X.tocsr(), user_means=user_means, col_norms=col_norms)

    """
    Similarity computation
    """
    def _item_item_topk(
            self,
            x: sparse.csr_matrix,
            col_norms: np.ndarray,
            topk: int,
            beta: float,
            min_co_count: int,
            block_size: int,
    ) -> SimilarityGraph:
        """
        Computes top-K neighbors for each item using:
          dot products: d = x.T @ x_block
          cosine: dot / (||i|| * ||j||)
          shrinkage: (n / (n + beta)) * cosine, with n the co-rating count
        Uses a block loop to keep memory bounded.
        :param x: sparse.csr_matrix
        :param col_norms: np.ndarray
        :param topk: int
        :param beta: float
        :param min_co_count: int
        :param block_size: int
        :return: SimilarityGraph
        """
        U, I = x.shape
        neighbors: List[List[Tuple[int, float]]] = [[] for _ in range(I)]

        x_csc = x.tocsc()
        x_bin = x.copy()
        x_bin.data[:] = 1.0

        for j0 in tqdm(range(0, I, block_size), desc="Item blocks"):
            j1 = min(j0 + block_size, I)

            x_block = x_csc[:, j0:j1]
            d = (x.T @ x_block).tocsr()
            nb = (x_bin.T @ x_bin[:, j0:j1]).tocsr()

            d_indptr, d_indices, d_data = d.indptr, d.indices, d.data

            for local_col in range(d.shape[1]):
                j = j0 + local_col
                start, end = d_indptr[local_col], d_indptr[local_col + 1]
                if start == end:
                    continue

                idx = d_indices[start:end]
                dots = d_data[start:end].astype(np.float32)

                denom = col_norms[idx] * max(col_norms[j], 1e-8)
                sims = np.zeros_like(dots, dtype=np.float32)
                nz = denom > 0
                sims[nz] = dots[nz] / denom[nz]

                n_row = nb.getcol(local_col)
                n_counts = np.zeros_like(sims)
                if n_row.nnz:
                    ni = n_row.indices
                    nv = n_row.data
                    cdict = dict(zip(ni.tolist(), nv.tolist()))
                    for t, it in enumerate(idx):
                        n_counts[t] = cdict.get(int(it), 0.0)

                keep = (n_counts >= float(min_co_count)) & (idx != j)
                idx = idx[keep]
                sims = sims[keep]
                n_counts = n_counts[keep]

                if idx.size == 0:
                    continue

                sims = (n_counts / (n_counts + beta)) * sims

                if idx.size > topk:
                    part = np.argpartition(-sims, topk)[:topk]
                    idx = idx[part]
                    sims = sims[part]

                order = np.argsort(-sims)
                neighbors[j] = [(int(idx[t]), float(sims[t])) for t in order]

        return SimilarityGraph(neighbors=neighbors)

    """
    Public Methods
    """
    def similar_items(self, recipe_id: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Returns up to k similar recipeIds with scores for the given recipe_id
        :param recipe_id
        :param k
        :return:
        """
        try:
            j = int(self.maps.item_enc.transform([recipe_id])[0])
        except Exception:
            return []
        neigh = self.graph.neighbors[j][:k]
        return [(self.maps.idx2item[i], s) for i, s in neigh]

    def recommend(self, user_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Very simple user-to-item recommendation via neighbor voting:
          1) fetch items the user rated and their mean-centered scores
          2) sum neighbor similarities weighted by the user's scores
          3) exclude items already rated
          4) return top_k new items
        """
        try:
            u = int(self.maps.user_enc.transform([user_id])[0])
        except Exception:
            return []

        X = self.rm.X_csr

        start, end = X.indptr[u], X.indptr[u + 1]
        rated_items = X.indices[start:end]
        rated_scores = X.data[start:end]

        if rated_items.size == 0:
            return []

        scores: Dict[int, float] = {}
        seen = set(rated_items.tolist())

        for i, r_ui in zip(rated_items, rated_scores):
            for j, s_ij in self.graph.neighbors[i]:
                if j in seen:
                    continue
                scores[j] = scores.get(j, 0.0) + (s_ij * float(r_ui))

        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [(self.maps.idx2item[j], float(score)) for j, score in ranked]