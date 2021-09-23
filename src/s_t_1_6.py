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

def nsw(query_point: np.ndarray, all_documents: np.ndarray, 
        graph_edges: Dict[int, List[int]],
        search_k: int = 10, num_start_points: int = 5,
        dist_f: Callable = distance) -> np.ndarray:
    # допишите ваш код здесь 
    class PQ():
        pq = [-1] * search_k
        result = [-1] * search_k
        p = 0
        
        def push(self, n, d):
            if self.p != search_k:
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
                        
    pq = PQ()

    def search(cur_point, cur_dist):
        pq.push(cur_point, cur_dist)
        neighbors = graph_edges[cur_point]
        neighbor_dists = dist_f(query_point, all_documents[neighbors]).flatten()
        i_min = neighbor_dists.argmin()
        d_min = neighbor_dists[i_min]

        if d_min < cur_dist:
            search(neighbors[i_min], d_min)

    for start in np.random.choice(list(graph_edges.keys()), num_start_points, replace=False):
        start_dist = dist_f(query_point, all_documents[start:start + 1])[0]
        search(start, start_dist)

    return np.array(pq.result)
