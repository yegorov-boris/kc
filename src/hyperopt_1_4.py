from hyperopt import fmin, hp, tpe
from solution_template_1_4 import Solution

space = [hp.uniform('lr', 0.05, 0.8), hp.uniform('subsample', 0.1, 0.5)]


# укажем objective-функцию
def f(args):
    lr, subsample = args
    s = Solution(lr=lr, colsample_bytree=0.5, subsample=subsample)
    s.fit()
    return 1 - max(s.scores)


best = fmin(f, space, algo=tpe.suggest, max_evals=10)
print ('TPE result: ', best)
