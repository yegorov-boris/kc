import math
import pickle
import random
from typing import List, Tuple

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from tqdm.auto import tqdm
import time


class Solution:
    def __init__(self, n_estimators: int = 100, lr: float = 0.5, ndcg_top_k: int = 10,
                 subsample: float = 0.6, colsample_bytree: float = 0.9,
                 max_depth: int = 5, min_samples_leaf: int = 8):
        self._prepare_data()

        self.ndcg_top_k = ndcg_top_k
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

        # допишите ваш код здесь
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.trees = []
        self.feature_ixs = []
        self.scores = []

    def _get_data(self) -> List[np.ndarray]:
        train_df, test_df = msrank_10k()

        X_train = train_df.drop([0, 1], axis=1).values
        y_train = train_df[0].values
        query_ids_train = train_df[1].values.astype(int)

        X_test = test_df.drop([0, 1], axis=1).values
        y_test = test_df[0].values
        query_ids_test = test_df[1].values.astype(int)

        return [X_train, y_train, query_ids_train, X_test, y_test, query_ids_test]

    def _prepare_data(self) -> None:
        (X_train, y_train, self.query_ids_train,
            X_test, y_test, self.query_ids_test) = self._get_data()
        # допишите ваш код здесь
        self.X_train, self.X_test, self.ys_train, self.ys_test = map(torch.FloatTensor, [
            self._scale_features_in_query_groups(X_train, self.query_ids_train),
            self._scale_features_in_query_groups(X_test, self.query_ids_train),
            y_train.reshape(-1, 1),
            y_test.reshape(-1, 1)
        ])

    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> np.ndarray:
        # допишите ваш код здесь 
        scaler = StandardScaler()
        groups = {}

        for query_id, feat in zip(inp_query_ids, inp_feat_array):
            if query_id in groups:
                groups[query_id].append(feat)
            else:
                groups[query_id] = [feat]

        for query_id in groups:
            groups[query_id] = scaler.fit_transform(groups[query_id])

        result = []

        for query_id in inp_query_ids:
            if query_id in groups and groups[query_id].size:
                result.append(groups[query_id][0])
                groups[query_id] = groups[query_id][1:]

        return np.array(result)

    def _train_one_tree(self, cur_tree_idx: int,
                        train_preds: torch.FloatTensor
                        ) -> Tuple[DecisionTreeRegressor, np.ndarray]:
        # допишите ваш код здесь
        np.random.seed(cur_tree_idx)
        feature_ixs = np.random.choice(range(self.X_train.shape[1]), int(np.floor(self.colsample_bytree * self.X_train.shape[1])), replace=False)
        objects_ixs = np.random.choice(range(self.X_train.shape[0]), int(np.floor(self.subsample * self.X_train.shape[0])), replace=False)

        lambdas = torch.zeros(self.ys_train.shape[0], 1)

        for query_id in np.unique(self.query_ids_train):
            mask = self.query_ids_train == query_id
            lambdas[mask] = self._compute_lambdas(self.ys_train[mask], train_preds[mask])

        dtr = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
        x = self.X_train[objects_ixs].t()[feature_ixs].t()
        y = lambdas[objects_ixs]
        dtr.fit(x, y)

        return dtr, feature_ixs

    def _calc_data_ndcg(self, queries_list: np.ndarray,
                        true_labels: torch.FloatTensor, preds: torch.FloatTensor) -> float:
        # допишите ваш код здесь
        ndcgs = []
        for query_id in np.unique(queries_list):
            mask = queries_list == query_id
            ys_true = true_labels[mask]
            ys_pred = preds[mask]
            ndcg = self._ndcg_k(ys_true.reshape(ys_true.shape[0]), ys_pred.reshape(ys_pred.shape[0]), self.ndcg_top_k)
            ndcgs.append(ndcg)

        return float(np.mean(ndcgs))

    def fit(self):
        np.random.seed(0)
        # допишите ваш код здесь
        train_preds = torch.zeros(self.X_train.shape[0], 1).float()

        for i in range(self.n_estimators):
            dtr, feature_ixs = self._train_one_tree(i, train_preds)
            self.trees.append(dtr)
            self.feature_ixs.append(feature_ixs)

            for query_id in np.unique(self.query_ids_train):
                mask = self.query_ids_train == query_id
                cur_preds = dtr.predict(self.X_train[mask].t()[feature_ixs].t())
                train_preds[mask] += self.lr * torch.Tensor(cur_preds).reshape(-1, 1).float()

            # test_preds = torch.zeros(self.X_test.shape[0], 1).float()
            #
            # for query_id in np.unique(self.query_ids_test):
            #     mask = self.query_ids_test == query_id
            #     cur_test_preds = self.predict(self.X_test[mask])
            #     test_preds[mask] += self.lr * cur_test_preds

            self.scores.append(self._calc_data_ndcg(self.query_ids_train, self.ys_test, train_preds))
            # self.scores.append(self._calc_data_ndcg(self.query_ids_test, self.ys_test, test_preds))

        self.trees = self.trees[:1+np.argmax(self.scores)]

    def predict(self, data: torch.FloatTensor) -> torch.FloatTensor:
        # допишите ваш код здесь
        preds = torch.zeros(data.shape[0], 1).float()

        for i, dtr in enumerate(self.trees):
            cur_preds = dtr.predict(data.t()[self.feature_ixs[i]].t())
            preds += self.lr * torch.Tensor(cur_preds).reshape(-1, 1).float()

        return preds

    def _compute_lambdas(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        # допишите ваш код здесь
        # рассчитаем нормировку, IdealDCG
        ideal_dcg = dcg(y_true, y_true, 'exp2', y_true.shape[0])
        N = 1 / ideal_dcg if ideal_dcg else 0

        # рассчитаем порядок документов согласно оценкам релевантности
        _, rank_order = torch.sort(y_true, descending=True, dim=0)
        rank_order += 1

        with torch.no_grad():
            # получаем все попарные разницы скоров в батче
            pos_pairs_score_diff = 1.0 + torch.exp((y_pred - y_pred.t()))

            # поставим разметку для пар, 1 если первый документ релевантнее
            # -1 если второй документ релевантнее
            Sij = compute_labels_in_batch(y_true)
            # посчитаем изменение gain из-за перестановок
            gain_diff = compute_gain_diff(y_true, "exp2")

            # посчитаем изменение знаменателей-дискаунтеров
            decay_diff = (1.0 / torch.log2(rank_order + 1.0)) - (1.0 / torch.log2(rank_order.t() + 1.0))
            # посчитаем непосредственное изменение nDCG
            delta_ndcg = torch.abs(N * gain_diff * decay_diff)
            # посчитаем лямбды
            lambda_update = (0.5 * (1 - Sij) - 1 / pos_pairs_score_diff) * delta_ndcg
            lambda_update = torch.sum(lambda_update, dim=1, keepdim=True)

            return lambda_update.float()

    def _ndcg_k(self, ys_true, ys_pred, ndcg_top_k) -> float:
        # допишите ваш код здесь
        ideal = dcg(ys_true, ys_true, 'exp2', ndcg_top_k)
        return dcg(ys_true, ys_pred, 'exp2', ndcg_top_k) / ideal if ideal else 0

    def save_model(self, path: str):
        # допишите ваш код здесь
        model = {
            'trees': self.trees,
            'feature_ixs': self.feature_ixs,
            'lr': self.lr,
        }
        pickle.dump(model, open('%s.lmart' % (path), "wb"), protocol=2)

    def load_model(self, path: str):
        # допишите ваш код здесь
        model = pickle.load(open(path, "rb"))
        self.trees = model['trees']
        self.feature_ixs = model['feature_ixs']
        self.lr = model['lr']


def compute_gain(y_value: float, gain_scheme: str) -> float:
    if gain_scheme == 'const':
        return y_value

    if gain_scheme == 'exp2':
        return pow(2.0, y_value) - 1.0

    raise ValueError('incorrect gain_scheme')


def dcg(ys_true: torch.Tensor, ys_pred: torch.Tensor, gain_scheme: str, ndcg_top_k: int) -> float:
    _, args = torch.sort(ys_pred, descending=True, dim=0)
    return sum(map(lambda p: compute_gain(p[1].item(), gain_scheme) / np.log2(p[0]), enumerate(ys_true[args[:ndcg_top_k]], 2)))


def compute_labels_in_batch(y_true):
    # разница релевантностей каждого с каждым объектом
    rel_diff = y_true - y_true.t()

    # 1 в этой матрице - объект более релевантен
    pos_pairs = (rel_diff > 0).type(torch.float32)

    # 1 тут - объект менее релевантен
    neg_pairs = (rel_diff < 0).type(torch.float32)
    Sij = pos_pairs - neg_pairs
    return Sij


def compute_gain_diff(y_true, gain_scheme):
    if gain_scheme == "exp2":
        gain_diff = torch.pow(2.0, y_true) - torch.pow(2.0, y_true.t())
    elif gain_scheme == "diff":
        gain_diff = y_true - y_true.t()
    else:
        raise ValueError(f"{gain_scheme} method not supported")
    return gain_diff


s = Solution(n_estimators=30)

ts = time.time()
s.fit()
print((time.time() - ts)*1000)

for score in s.scores:
    print(score)
