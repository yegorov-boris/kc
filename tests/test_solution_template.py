import unittest
import torch
from math import log2
import src.solution_template_1_2 as s_t


class MyTestCase(unittest.TestCase):
    n = 20

    def rand_ys(self):
        return torch.rand(self.n), torch.rand(self.n)

    def test_num_swapped_pairs(self):
        y_true, y_pred = self.rand_ys()
        self.assertEqual(num_swapped_pairs(y_true, y_pred), s_t.num_swapped_pairs(y_true, y_pred))

    def test_dcg(self):
        y_true, y_pred = self.rand_ys()
        self.assertEqual(dcg(y_true, y_pred, 'const'), s_t.dcg(y_true, y_pred, 'const'))
        self.assertEqual(dcg(y_true, y_pred, 'exp2'), s_t.dcg(y_true, y_pred, 'exp2'))

    def test_ndcg(self):
        y_true, y_pred = self.rand_ys()
        self.assertEqual(ndcg(y_true, y_pred, 'const'), s_t.ndcg(y_true, y_pred, 'const'))
        self.assertEqual(ndcg(y_true, y_pred, 'exp2'), s_t.ndcg(y_true, y_pred, 'exp2'))

    def test_precission_at_k(self):
        _, y_pred = self.rand_ys()
        k = torch.randint(1, self.n, (1, 1))[0][0]
        m = torch.randint(1, self.n, (1, 1))[0][0]
        y_true = torch.Tensor([1]*m + [0]*(self.n-m))
        self.assertEqual(precission_at_k(y_true, y_pred, k), s_t.precission_at_k(y_true, y_pred, k))

    def test_reciprocal_rank(self):
        _, y_pred = self.rand_ys()
        k = torch.randint(self.n - 1, (1, 1))[0][0]
        y_true = torch.zeros(self.n)

        self.assertEqual(0, s_t.reciprocal_rank(y_true, y_pred))

        y_true[k] = 1
        expected = reciprocal_rank(y_true, y_pred)
        actual = s_t.reciprocal_rank(y_true, y_pred)
        self.assertEqual(expected, actual)

    def test_average_precision(self):
        ys_true, ys_pred = self.rand_ys()
        ys_true = ys_true.round()
        self.assertEqual(average_precision(ys_true, ys_pred), s_t.average_precision(ys_true, ys_pred))

    def test_p_found(self):
        ys_true, ys_pred = self.rand_ys()
        p_break = torch.rand(1).item()
        self.assertEqual(p_found(ys_true, ys_pred, p_break), s_t.p_found(ys_true, ys_pred, p_break))


def num_swapped_pairs(ys_true, ys_pred):
    s_pred, args = torch.sort(ys_pred, descending=True, dim=0)
    s_true = ys_true[args]
    n = s_true.shape[0]
    c = 0
    for i in range(n-1):
        for j in range(i+1, n):
            if (s_true[i] < s_true[j] and s_pred[i] > s_pred[j]) or (s_true[i] > s_true[j] and s_pred[i] < s_pred[j]):
                c += 1
    return c


def dcg(ys_true, ys_pred, gain_scheme):
    _, args = torch.sort(ys_pred, descending=True, dim=0)
    s_true = ys_true[args]
    r = 0
    for i, y in enumerate(s_true, 1):
        r += s_t.compute_gain(y.item(), gain_scheme) / log2(i+1)
    return r


def ndcg(ys_true, ys_pred, gain_scheme):
    return dcg(ys_true, ys_pred, gain_scheme) / dcg(ys_true, ys_true, gain_scheme)


def precission_at_k(ys_true, ys_pred, k):
    if ys_true.sum() == 0:
        return float(-1)

    _, args = torch.sort(ys_pred, descending=True, dim=0)
    s_true = ys_true[args]

    return float(s_true[:k].sum() / min(k, ys_true.sum()))


def reciprocal_rank(ys_true, ys_pred):
    s_true = ys_true[torch.sort(ys_pred, descending=True, dim=0)[1]]

    for i, y in enumerate(s_true, 1):
        if y == 1:
            return float(1 / i)

    return float(0)


def average_precision(ys_true, ys_pred):
    if ys_true.sum() == 0:
        return float(-1)

    s_true = ys_true[torch.sort(ys_pred, descending=True, dim=0)[1]]
    num_corr_ans = 0
    rolling_sum = 0

    for i, y in enumerate(s_true, 1):
        if y == 1:
            num_corr_ans += 1
            rolling_sum += num_corr_ans / i

    return float(0 if num_corr_ans == 0 else rolling_sum / num_corr_ans)


def p_found(ys_true, ys_pred, p_break=0.15):
    s_true = ys_true[torch.sort(ys_pred, descending=True, dim=0)[1]]
    look = 1
    found = 0

    for y in s_true:
        found += look*float(y)
        look *= (1-float(y))*(1-p_break)
    return found


if __name__ == '__main__':
    unittest.main()
