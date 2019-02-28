from hyperopt import STATUS_OK, STATUS_FAIL
from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.utils import read_yaml
from rasa_nlu.evaluate import run_evaluation
from rasa_nlu.model import Trainer
import os
import logging

logger = logging.getLogger(__name__)

AVAILABLE_METRICS = ["f1_score", "accuracy", "precision", "threshold_loss"]


def run_trial(space):
    """The objective function is pickled and transferred to the workers.
       Hence, this function has to contain all the imports we need.
    """

    data_dir = os.environ.get("DATA_DIRECTORY", "./data")
    model_dir = os.environ.get("MODEL_DIRECTORY", "./models")
    target_metric = os.environ.get("TARGET_METRIC", "f1_score")

    if target_metric not in AVAILABLE_METRICS:
        logger.error("The metric '{}' is not in the available metrics. "
                     "Please use one of the available metrics: {}."
                     "".format(target_metric, AVAILABLE_METRICS))

        return {"loss": 1, "status": STATUS_FAIL}

    logger.debug("Search space: {}".format(space))

    # The epoch has to be an int since `tqdm` otherwise will cause an exception.
    if "epochs" in space:
        space["epochs"] = int(space["epochs"])

    with open(os.path.join(data_dir, "template_config.yml")) as f:
        config_yml = f.read().format(**space)
        config = read_yaml(config_yml)
        config = RasaNLUModelConfig(config)

    trainer = Trainer(config)
    training_data = load_data(os.path.join(data_dir, "train.md"))
    test_data_path = os.path.join(data_dir, "test.md")

    # wrap in train and eval in try/except in case
    # nlu_hyperopt proposes invalid combination of params

    try:
        model = trainer.train(training_data)
        model_path = trainer.persist(model_dir)

        if target_metric is None or target_metric == "threshold_loss":
            loss = _get_threshold_loss(model, test_data_path)
        else:
            loss = _get_nlu_evaluation_loss(model_path,
                                            target_metric,
                                            test_data_path)
        return {"loss": loss, "status": STATUS_OK}
    except Exception as e:
        logger.error(e)
        return {"loss": 1, "status": STATUS_FAIL}


def _get_nlu_evaluation_loss(model_path, metric, data_path):
    logger.info("Calculating '{}' loss.".format(metric))

    evaluation_result = run_evaluation(data_path, model_path)
    metric_result = evaluation_result['intent_evaluation'][metric]
    logger.info("{}: {}".format(metric, metric_result))
    
    return 1 - metric_result


def _get_threshold_loss(model, data_path):
    logger.info("Calculating threshold loss.")

    data = load_data(data_path)
    threshold = float(os.environ.get("THRESHOLD", 0.8))
    margin_weight = float(os.environ.get("ABOVE_BELOW_WEIGHT", 0.5))

    correct_below = 0
    incorrect_above = 0

    for example in data.intent_examples:
        prediction = model.parse(example.text)
        label = example.data["intent"]
        predicted_intent = prediction["intent"]["name"]
        
        is_correct = label == predicted_intent
        above_threshold = prediction["intent"]["confidence"] > threshold
    
        if is_correct and not above_threshold:
            correct_below += 1
        elif not is_correct and above_threshold:
            incorrect_above += 1

    number_examples = len(data.intent_examples)
    correct_below /= number_examples
    incorrect_above /= number_examples

    loss = margin_weight * incorrect_above + (1 - margin_weight) * correct_below
    logger.info("Threshold loss: {} (incorrect above: {}, correct below: {})"
                "".format(loss, incorrect_above, correct_below))

    return loss
