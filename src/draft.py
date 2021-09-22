from datetime import datetime
from s_t_1_5 import Solution
s = Solution(glue_qqp_dir='../data/QQP/', glove_vectors_path='../data/glove.6B.50d.txt')
t = datetime.now()
s.train(1)
print(datetime.now() - t)
