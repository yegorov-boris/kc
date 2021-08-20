from math import log2

from torch import Tensor, sort


def num_swapped_pairs(ys_true: Tensor, ys_pred: Tensor) -> int:
    def mergeSort(arr):
        # A temp_arr is created to store
        # sorted array in merge function
        n = len(arr)
        temp_arr = [0] * n
        return _mergeSort(arr, temp_arr, 0, n - 1)

    # This Function will use MergeSort to count inversions

    def _mergeSort(arr, temp_arr, left, right):

        # A variable inv_count is used to store
        # inversion counts in each recursive call

        inv_count = 0

        # We will make a recursive call if and only if
        # we have more than one elements

        if left < right:
            # mid is calculated to divide the array into two subarrays
            # Floor division is must in case of python

            mid = (left + right) // 2

            # It will calculate inversion
            # counts in the left subarray

            inv_count += _mergeSort(arr, temp_arr,
                                    left, mid)

            # It will calculate inversion
            # counts in right subarray

            inv_count += _mergeSort(arr, temp_arr,
                                    mid + 1, right)

            # It will merge two subarrays in
            # a sorted subarray

            inv_count += merge(arr, temp_arr, left, mid, right)
        return inv_count

    # This function will merge two subarrays
    # in a single sorted subarray
    def merge(arr, temp_arr, left, mid, right):
        i = left  # Starting index of left subarray
        j = mid + 1  # Starting index of right subarray
        k = left  # Starting index of to be sorted subarray
        inv_count = 0

        # Conditions are checked to make sure that
        # i and j don't exceed their
        # subarray limits.

        while i <= mid and j <= right:

            # There will be no inversion if arr[i] <= arr[j]

            if arr[i] <= arr[j]:
                temp_arr[k] = arr[i]
                k += 1
                i += 1
            else:
                # Inversion will occur.
                temp_arr[k] = arr[j]
                inv_count += (mid - i + 1)
                k += 1
                j += 1

        # Copy the remaining elements of left
        # subarray into temporary array
        while i <= mid:
            temp_arr[k] = arr[i]
            k += 1
            i += 1

        # Copy the remaining elements of right
        # subarray into temporary array
        while j <= right:
            temp_arr[k] = arr[j]
            k += 1
            j += 1

        # Copy the sorted subarray into Original array
        for loop_var in range(left, right + 1):
            arr[loop_var] = temp_arr[loop_var]

        return inv_count

    index = [(ys_true == v).nonzero().item() for v in ys_pred]

    return mergeSort(list(reversed(index)))


# print(num_swapped_pairs(Tensor([1, 2]), Tensor([2, 1])))

def compute_gain(y_value: float, gain_scheme: str) -> float:
    # допишите ваш код здесь
    pass


def dcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str) -> float:
    # допишите ваш код здесь
    pass


def ndcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str = 'const') -> float:
    # допишите ваш код здесь
    pass


def precission_at_k(ys_true: Tensor, ys_pred: Tensor, k: int) -> float:
    # допишите ваш код здесь
    pass


def reciprocal_rank(ys_true: Tensor, ys_pred: Tensor) -> float:
    # допишите ваш код здесь
    pass


def p_found(ys_true: Tensor, ys_pred: Tensor, p_break: float = 0.15 ) -> float:
    # допишите ваш код здесь
    pass


def average_precision(ys_true: Tensor, ys_pred: Tensor) -> float:
    # допишите ваш код здесь
    pass
