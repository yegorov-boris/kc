from solution_template_1_5 import Solution

s = Solution(glue_qqp_dir='../data/QQP/', glove_vectors_path='../data/glove.6B.50d.txt')
s.train(1)
ndcg = s.valid(s.model, s.val_dataloader)
print(ndcg)
