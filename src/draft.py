import numpy as np
from datetime import datetime
from s_t_1_6 import create_sw_graph, nsw

# data = np.array([[1, 9], [2, 3], [4, 1], [3, 7], [5, 4]])
# graph_edges = create_sw_graph(data, 2, 2, 2, 2)
# query = np.array([[1.5, 2.5]])
# result = nsw(query, data, graph_edges, 2, 2)

data = np.random.rand(10000, 100)
graph_edges = create_sw_graph(data)
query = np.random.rand(1, 100)
t = datetime.now()
result = nsw(query, data, graph_edges)
print(datetime.now() - t)

print(result)
