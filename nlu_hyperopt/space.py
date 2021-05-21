from hyperopt import hp

# Define the search space here, e.g.
# from hyperopt.pyll.base import scope

# search_space = {
#     'epochs': hp.qloguniform('epochs', 0, 4, 2),
#     'max_df': hp.uniform('max_df', 1, 2),
#     'max_ngrams': scope.int(hp.quniform('max_ngram', 3, 9, 1))
#     }

# Default search space: Try different numbers of training epochs.
search_space = {"epochs": hp.qloguniform("epochs", 0, 4, 2)}
