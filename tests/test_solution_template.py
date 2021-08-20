import unittest
import torch
import src.solution_template as s_t


class MyTestCase(unittest.TestCase):
    def test_num_swapped_pairs(self):
        n = 10
        y_true = torch.rand(n)
        perm = torch.randperm(n)
        y_pred = y_true[perm]
        print(y_true, y_pred)
        self.assertEqual(get_inv_count(perm), s_t.num_swapped_pairs(y_true, y_pred))


def get_inv_count(arr):
    n = len(arr)
    inv_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] < arr[j]:
                inv_count += 1

    return inv_count


if __name__ == '__main__':
    unittest.main()
