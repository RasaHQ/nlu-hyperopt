from hyperopt import fmin, tpe, space_eval
from hyperopt.mongoexp import MongoTrials
import os
import logging
import sys


running_as_action = os.environ.get("RUNNING_AS_ACTION", False)

def import_space(input_search_space):
    """ Imports search_space from an absolute path.
        This is used when running as a Github Action, where space.py
        can't be copied into the directory before building the docker image
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("nlu_hyperopt.space", input_search_space)
    space = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(space)
    return space.search_space

input_search_space = os.environ.get("INPUT_SEARCH_SPACE")
if input_search_space:
    search_space = import_space(input_search_space)
else:
    from nlu_hyperopt.space import search_space

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

logger = logging.getLogger(__name__)


def worker_function(space):
    """This function is pickled and transferred to the workers.
       Hence, this function has to contain all the imports we need.
    """
    from nlu_hyperopt.optimization import run_trial

    return run_trial(space)


if __name__ == "__main__":
    # This function is run by the `nlu_hyperopt-master` and coordinates the
    # hyperparameter search.
    exp_key = os.environ.get("INPUT_EXP_KEY", "default")
    mongo_url = os.environ.get("INPUT_MONGO_URL")

    max_evals = int(os.environ.get("INPUT_MAX_EVALS", 100))

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

    best = fmin(worker_function, search_space, trials=trials, algo=tpe.suggest,
                max_evals=max_evals)

    logger.info("Hyperparameter search complete!")

    best_config = space_eval(search_space, best)
    logger.debug("The best values are: {}".format(best_config))

    data_dir = os.environ.get("INPUT_DATA_DIRECTORY", "./data")
    with open(os.path.join(data_dir, "template_config.yml")) as f:
        config_yml = f.read().format(**best_config)
        logger.info("The best configuration is: \n{}\n".format(config_yml))

        if running_as_action:
            config_yml=config_yml.replace('%','%25') ## github actions does not handle multiline outputs properly
            config_yml=config_yml.replace('\n','%0A') ## https://github.community/t5/GitHub-Actions/set-output-Truncates-Multiline-Strings/td-p/37870
            config_yml=config_yml.replace('\r','%0D')

            print(f'::set-output name=best_config::"{config_yml}"')
