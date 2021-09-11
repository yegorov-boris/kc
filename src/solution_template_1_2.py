from math import log2

from torch import Tensor, sort


def num_swapped_pairs(ys_true: Tensor, ys_pred: Tensor) -> int:
    s_pred, args = sort(ys_pred, descending=True, dim=0)
    s_true = ys_true[args]
    c = 0
    for i in range(1, s_true.shape[0]):
        c += s_true[i:][s_true[i:] > s_true[i-1]].shape[0]
    return c


def compute_gain(y_value: float, gain_scheme: str) -> float:
    if gain_scheme == 'const':
        return y_value

    if gain_scheme == 'exp2':
        return pow(2.0, y_value) - 1.0

    raise ValueError('incorrect gain_scheme')


def dcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str) -> float:
    _, args = sort(ys_pred, descending=True, dim=0)
    return sum(map(lambda p: compute_gain(p[1].item(), gain_scheme) / log2(p[0]), enumerate(ys_true[args], 2)))


def ndcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str = 'const') -> float:
    return dcg(ys_true, ys_pred, gain_scheme) / dcg(ys_true, ys_true, gain_scheme)


def precission_at_k(ys_true: Tensor, ys_pred: Tensor, k: int) -> float:
    if ys_true.sum() == 0:
        return float(-1)

    _, args = sort(ys_pred, descending=True, dim=0)

    return float(ys_true[args][:k].sum() / sort(ys_true)[0][-k:].sum())


def reciprocal_rank(ys_true: Tensor, ys_pred: Tensor) -> float:
    s_true = ys_true[sort(ys_pred, descending=True, dim=0)[1]]

    if ys_true.sum() == 0:
        return float(0)

    i = 1 + s_true.nonzero(as_tuple=True)[0].item()

    return float(1 / i)


def p_found(ys_true: Tensor, ys_pred: Tensor, p_break: float = 0.15 ) -> float:
    look = 1
    found = 0

    for y in ys_true[sort(ys_pred, descending=True, dim=0)[1]]:
        found += look * float(y)
        look *= (1 - float(y)) * (1 - p_break)

    return found


def average_precision(ys_true: Tensor, ys_pred: Tensor) -> float:
    if ys_true.sum() == 0:
        return float(-1)

    s_true = ys_true[sort(ys_pred, descending=True, dim=0)[1]]
    idx = s_true.nonzero(as_tuple=True)[0].add(1).to(float)
    n = idx.shape[0]
    cumsum = Tensor(range(1, n+1)).to(float)

    return float(sum(cumsum.div(idx)) / n if n else 0)
