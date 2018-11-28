import math
from hyperopt import fmin, tpe, hp, space_eval
from hyperopt.mongoexp import MongoTrials
from space import search_space
import os

def objective(space):
    """Objective function has to be picklable, hence
    contain all the imports we need.
    """
    from hyperopt import STATUS_OK
    from rasa_nlu.training_data import load_data
    from rasa_nlu.config import RasaNLUModelConfig
    from rasa_nlu.utils import read_yaml
    from rasa_nlu.evaluate import run_evaluation, get_intent_targets, get_intent_predictions, evaluate_intents
    from rasa_nlu.model import Trainer, Interpreter, Metadata
    import os
    import numpy as np

    hyper_dir = os.environ.get("HYPEROPT_DIR", "./hyperopt")
    data_dir = os.path.join(hyper_dir, "data")
    print(space)

    with open(os.path.join(data_dir, "template_config.yml")) as f:
        config_yml = f.read().format(**space)
        config = read_yaml(config_yml)
        config = RasaNLUModelConfig(config)

    trainer = Trainer(config)
    training_data = load_data(os.path.join(data_dir, 'train.md'))
    test_data = load_data(os.path.join(data_dir, 'test.md'))
    # wrap in train and eval in try/except in case
    # hyperopt proposes invalid conbination of params
    try:
        model = trainer.train(training_data)
        model_path = trainer.persist(os.path.join(hyper_dir, 'models/'))
        intent_targets = get_intent_targets(test_data)
        intent_results = get_intent_predictions(
            intent_targets, model, test_data)
        asymmetry = 1.
        incorrect_below = 0
        correct_above = 0
        cutoff = space["cutoff"]
        for x in intent_results:
            if x.target == x.prediction and x.confidence > cutoff:
                correct_above += 1
            elif x.target != x.prediction and x.confidence < cutoff:
                incorrect_below += 1

        correct_above /= len(intent_results)
        incorrect_below /= len(intent_results)

        intent_evaluation = evaluate_intents(intent_results, None, None, None)
        intent_f1 = intent_evaluation['f1_score']

        loss = - (intent_f1 + asymmetry * incorrect_below + correct_above)
        print("intent f1: {}, loss: {}".format(intent_f1, loss))
        return {'loss': loss, 'status': STATUS_OK }
    except:
        raise
        return {'loss': 100, 'status': STATUS_OK }


if __name__ == "__main__":

    exp_key = os.environ.get("EXP_KEY", "default")
    mongo_database = os.environ.get("MONGO_DATABASE")
    max_evals = int(os.environ.get("MAX_EVALS", 100))

    print("starting up\n"
          "running experiment : {}\n"
          "max evaluations: {} ".format(exp_key, max_evals))

    print("search space:")
    print(search_space)

    if mongo_database:
        print("running async via mongo")
        url = 'mongo://mongodb:27017/{}/jobs'.format(mongo_database)
        trials = MongoTrials(url, exp_key=exp_key)
    else:
        print("running in memory")
        trials = None

    best = fmin(objective, search_space, trials=trials, algo=tpe.suggest, max_evals=max_evals)

    print("search complete!")
    print(best)
    print(space_eval(search_space, best))
