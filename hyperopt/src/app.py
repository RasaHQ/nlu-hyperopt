import math
from hyperopt import fmin, tpe, hp, space_eval
from hyperopt.mongoexp import MongoTrials


space = hp.uniform('x', -2, 2)

trials = MongoTrials('mongo://mongodb:27017/foo_db/jobs', exp_key='exp1')
best = fmin(math.sin, space, trials=trials, algo=tpe.suggest, max_evals=10)

print best
print space_eval(space, best