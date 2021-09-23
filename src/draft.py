import numpy as np
from datetime import datetime
from s_t_1_6 import create_sw_graph, nsw, PQ, distance

# data = np.array([[1, 9], [2, 3], [4, 1], [3, 7], [5, 4]])
# graph_edges = create_sw_graph(data, 2, 2, 2, 2)
# query = np.array([[1.5, 2.5]])
# result = nsw(query, data, graph_edges, 2, 2)

n = 1000
d = 100
k = 10
data = np.random.rand(n, d)
for i in range(n):
    data[i][np.random.randint(0, d - 1)] *= 10
graph_edges = create_sw_graph(data, 5, 5, 5, 5)
# fake_graph_edges = {}
# for i in range(n):
#     fake_graph_edges[i] = np.random.randint(0, n - 1, k).tolist()
i = np.random.randint(0, n - 1)
query = data[i]
# tmp = zip(np.random.randint(0, n, n), np.random.rand(n))
# pq = PQ(10)
t = datetime.now()
# for n, d in tmp:
#     pq.push(n, d)
# result = nsw(query, data, fake_graph_edges)
result = nsw(query, data, graph_edges, search_k=5, num_start_points=20)
print(datetime.now() - t)

print(
    result,
    distance(query, data[result]).flatten(),
    graph_edges[i][5:],
    distance(query, data[graph_edges[i][5:]]).flatten()
)
