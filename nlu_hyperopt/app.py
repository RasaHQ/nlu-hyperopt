from hyperopt import fmin, tpe, space_eval
from hyperopt.mongoexp import MongoTrials
import os
import logging

from nlu_hyperopt.space import search_space


logger = logging.getLogger(__name__)


def objective(space):
    """The objective function is pickled and transferred to the workers.
       Hence, this function has to contain all the imports we need.
    """
    from nlu_hyperopt.optimization import optimize

    return optimize(space)


if __name__ == "__main__":
    # This function is run by the `nlu_hyperopt-master` and coordinates the
    # hyperparameter search.
    exp_key = os.environ.get("EXP_KEY", "default")
    mongo_url = os.environ.get("MONGO_URL", None)

    max_evals = int(os.environ.get("MAX_EVALS", 100))

    logger.info("Starting up\n"
                "Running experiment : {}\n"
                "Max evaluations: {}\n"
                "Search space: {}".format(exp_key, max_evals, search_space))

    if mongo_url:
        logger.debug("Running async via mongo.")
        if not mongo_url.startswith("mongo://"):
            mongo_url = "mongo://" + mongo_url
        if not mongo_url.endswith("/jobs"):
            mongo_url = mongo_url + "/jobs"

        trials = MongoTrials(mongo_url, exp_key=exp_key)
    else:
        logger.debug("No mongo database set. Running in memory.")
        trials = None

    best = fmin(objective, search_space, trials=trials, algo=tpe.suggest, 
                max_evals=max_evals)

    logger.info("Hyperparameter search complete!")
    best_config = space_eval(search_space, best)
    logger.info("The best configuration is: {}".format(best_config))
