import math
from hyperopt import fmin, tpe, hp, space_eval
from hyperopt.mongoexp import MongoTrials

def objective(x):
    from hyperopt import STATUS_OK
    from rasa_nlu.training_data import load_data
    from rasa_nlu.config import RasaNLUModelConfig
    from rasa_nlu.utils import read_yaml
    from rasa_nlu.evaluate import run_evaluation
    from rasa_nlu.model import Trainer, Interpreter, Metadata

    config_yml = """
language: en
pipeline:
- name: "intent_featurizer_count_vectors"
- name: "intent_classifier_tensorflow_embedding"
  "epochs": {}
""".format(round(x))
    config = read_yaml(config_yml)
    config = RasaNLUModelConfig(config)
    trainer = Trainer(config)
    # temporary hack around nlu bug
    trainer.pipeline[1].epochs = round(x)
    training_data = load_data('/hyperopt/data/train.md')
    model = trainer.train(training_data)
    model_path = trainer.persist('/hyperopt/models')
    evaluation = run_evaluation('/hyperopt/data/test.md', model_path, confmat_filename=None)
    intent_f1 = evaluation['intent_evaluation']['f1_score']
    return {'loss': 1-intent_f1, 'status': STATUS_OK }

print("starting process")
space = hp.qloguniform('epochs', 0, 4, 2)
trials = MongoTrials('mongo://mongodb:27017/foo_db/jobs', exp_key='exp3')
best = fmin(objective, space, trials=trials, algo=tpe.suggest, max_evals=100)

print(best)
print(space_eval(space, best))
