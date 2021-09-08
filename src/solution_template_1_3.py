import math

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler

from typing import List


class ListNet(torch.nn.Module):
    def __init__(self, num_input_features: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # укажите архитектуру простой модели здесь
        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_input_features, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, input_1: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_1)
        return logits


class Solution:
    def __init__(self, n_epochs: int = 5, listnet_hidden_dim: int = 30,
                 lr: float = 0.001, ndcg_top_k: int = 10):
        self._prepare_data()
        self.num_input_features = self.X_train.shape[1]
        self.ndcg_top_k = ndcg_top_k
        self.n_epochs = n_epochs

        self.model = self._create_model(
            self.num_input_features, listnet_hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

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

        self.X_train, self.X_test, self.ys_train, self.ys_test = map(torch.FloatTensor, [
            self._scale_features_in_query_groups(X_train, self.query_ids_train),
            self._scale_features_in_query_groups(X_test, self.query_ids_train),
            y_train,
            y_test
        ])

    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> np.ndarray:

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

    def _create_model(self, listnet_num_input_features: int,
                      listnet_hidden_dim: int) -> torch.nn.Module:
        torch.manual_seed(0)
        # допишите ваш код здесь
        net = ListNet(num_input_features=listnet_num_input_features, hidden_dim=listnet_hidden_dim)
        return net

    def fit(self) -> List[float]:
        result = []
        for _ in range(self.n_epochs):
            self._train_one_epoch()
            result.append(self._eval_test_set())
        return result

    def _calc_loss(self, batch_ys: torch.FloatTensor,
                   batch_pred: torch.FloatTensor) -> torch.FloatTensor:
        P_y_i = torch.softmax(batch_ys, dim=0)
        P_z_i = torch.softmax(batch_pred, dim=0)

        return -torch.sum(P_y_i * torch.log(P_z_i / P_y_i))

    def _train_one_epoch(self) -> None:
        self.model.train()
        # допишите ваш код здесь
        for query_id in np.unique(self.query_ids_train):
            batch_X = self.X_train[self.query_ids_train == query_id]
            batch_ys = self.ys_train[self.query_ids_train == query_id]

            self.optimizer.zero_grad()
            batch_pred = self.model(batch_X).reshape(-1, )
            batch_loss = self._calc_loss(batch_ys, batch_pred)
            batch_loss.backward(retain_graph=True)
            self.optimizer.step()

    def _eval_test_set(self) -> float:
        with torch.no_grad():
            self.model.eval()
            ndcgs = []
            # допишите ваш код здесь
            for query_id in np.unique(self.query_ids_test):
                ys_pred = self.model(self.X_test[self.query_ids_test == query_id])
                ys_true = self.ys_test[self.query_ids_test == query_id]
                ndcg = self._ndcg_k(ys_true, ys_pred, self.ndcg_top_k)
                ndcgs.append(ndcg)

            return float(np.mean(ndcgs))

    def _ndcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor,
                ndcg_top_k: int) -> float:
        # допишите ваш код здесь
        ideal = dcg(ys_true, ys_true, 'exp2', ndcg_top_k)
        return dcg(ys_true, ys_pred, 'exp2', ndcg_top_k) / ideal if ideal else 0


def compute_gain(y_value: float, gain_scheme: str) -> float:
    if gain_scheme == 'const':
        return y_value

    if gain_scheme == 'exp2':
        return pow(2.0, y_value) - 1.0

    raise ValueError('incorrect gain_scheme')


def dcg(ys_true: torch.Tensor, ys_pred: torch.Tensor, gain_scheme: str, ndcg_top_k: int) -> float:
    _, args = torch.sort(ys_pred, descending=True, dim=0)
    return sum(map(lambda p: compute_gain(p[1].item(), gain_scheme) / np.log2(p[0]), enumerate(ys_true[args[:ndcg_top_k]], 2)))


s = Solution()

print(s)
