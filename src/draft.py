import numpy as np
from s_t_1_6 import create_sw_graph, distance

data = np.array([[1, 9], [2, 3], [4, 1], [3, 7], [5, 4]])#, [6, 8], [7, 2], [8, 8], [7, 9], [9, 6]])

print(create_sw_graph(data, 2, 2, 2, 2))

# print(distance(data[0], data[1:]))
