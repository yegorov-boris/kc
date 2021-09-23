from collections import OrderedDict, defaultdict
from typing import Callable, Tuple, Dict, List

import numpy as np
from tqdm.auto import tqdm

# from datetime import datetime, timedelta


def distance(pointA: np.ndarray, documents: np.ndarray) -> np.ndarray:
    # допишите ваш код здесь 
    return np.linalg.norm(documents - pointA, axis=1)[..., None]


def create_sw_graph(
        data: np.ndarray,
        num_candidates_for_choice_long: int = 10,
        num_edges_long: int = 5,
        num_candidates_for_choice_short: int = 10,
        num_edges_short: int = 5,
        use_sampling: bool = False,
        sampling_share: float = 0.05,
        dist_f: Callable = distance
    ) -> Dict[int, List[int]]:
    # допишите ваш код здесь 
    d = {}

    for i, p in enumerate(data):
        ds = dist_f(p, data).flatten()
        short_candidates = np.argpartition(ds, 1+num_candidates_for_choice_short)[:1+num_candidates_for_choice_short]
        short_candidates = short_candidates[short_candidates != i]
        long_candidates = np.argpartition(ds, -num_candidates_for_choice_long)[-num_candidates_for_choice_long:]
        d[i] = np.random.choice(long_candidates, num_edges_long, replace=False).tolist() + np.random.choice(short_candidates, num_edges_short, replace=False).tolist()

    return d


class PQ():
    def __init__(self, search_k):
        self.pq = [-1] * search_k
        self.result = [-1] * search_k
        self.k = search_k
    p = 0

    def push(self, n, d):
        if self.p != self.k:
            self.pq[self.p] = d
            self.result[self.p] = n
            self.p = self.p + 1
            return

        if n not in self.result:
            for i, v in enumerate(self.pq):
                if d < v or v == -1:
                    self.pq[i] = d
                    self.result[i] = n
                    break


def nsw(query_point: np.ndarray, all_documents: np.ndarray, 
        graph_edges: Dict[int, List[int]],
        search_k: int = 10, num_start_points: int = 5,
        dist_f: Callable = distance) -> np.ndarray:
    # допишите ваш код здесь 
    pq = PQ(search_k)
    # total = [timedelta(0)]

    def search(cur_point, cur_dist):
        # pq.push(cur_point, cur_dist)
        neighbors = graph_edges[cur_point]
        dd = all_documents[neighbors]
        # tmp = datetime.now()
        neighbor_dists = dist_f(query_point, dd)
        # print(neighbor_dists.flatten().round())
        # print(neighbors)
        # total[0] += datetime.now() - tmp
        i_min = neighbor_dists.argmin(axis=0)[0]
        d_min = neighbor_dists[i_min][0]

        if d_min < cur_dist:
            search(neighbors[i_min], d_min)
        else:
            print('end', cur_point)
            for i, d in enumerate(neighbor_dists):
                pq.push(neighbors[i], d[0])

    # for start in np.random.randint(0, len(all_documents) - 1, 1):
    for start in np.random.randint(0, len(all_documents) - 1, num_start_points):
        start_dist = dist_f(query_point, all_documents[start:start + 1])[0]
        print('start', start)
        search(start, start_dist)

    # print(total[0])
    return np.array(pq.result)
