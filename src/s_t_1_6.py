from collections import OrderedDict, defaultdict
from typing import Callable, Tuple, Dict, List

import numpy as np
from tqdm.auto import tqdm


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


class PQ:
    def __init__(self, search_k):
        self.pq = [-1] * search_k
        self.result = [-1] * search_k
        self.k = search_k
        self.s = set()

    def push(self, n, d):
        if n not in self.result:
            for i, v in enumerate(self.pq):
                if d < v or v == -1:
                    self.result.insert(i, n)
                    self.result.pop(-1)
                    self.pq.insert(i, d)
                    self.pq.pop(-1)
                    self.s.add(n)
                    break


def nsw(query_point: np.ndarray, all_documents: np.ndarray, 
        graph_edges: Dict[int, List[int]],
        search_k: int = 10, num_start_points: int = 5,
        dist_f: Callable = distance) -> np.ndarray:
    # допишите ваш код здесь 
    pq = PQ(search_k)
    roots = set()
    visited = set()

    def search(cur_point, cur_dist, epoch=1, max_epochs=0):
        roots.add(cur_point)
        neighbors = list(set(graph_edges[cur_point]) - visited)

        if not len(neighbors):
            return

        dd = all_documents[neighbors]
        neighbor_dists = dist_f(query_point, dd)

        for i, d in enumerate(neighbor_dists):
            n = neighbors[i]
            pq.push(n, d[0])
            visited.add(n)

        i_min = neighbor_dists.argmin(axis=0)[0]
        d_min = neighbor_dists[i_min][0]

        if d_min < cur_dist and (epoch < max_epochs or not max_epochs):
            return search(neighbors[i_min], d_min, epoch+1, max_epochs)

    for _ in range(num_start_points):
        start = np.random.randint(0, len(all_documents) - 1)
        start_dist = dist_f(query_point, all_documents[start:start+1])[0][0]
        search(start, start_dist, 1, 2)

    # for _ in range(num_start_points):
    for i, n in enumerate(pq.result):
        if i and n not in roots:
            search(n, pq.pq[i])
                # break

    return np.array(pq.result[:search_k])
