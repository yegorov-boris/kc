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
    pass

