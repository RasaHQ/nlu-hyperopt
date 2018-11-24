## Hyperparameter Search for Rasa NLU

This repo provides a setup for doing hyperparamter search, locally or on a cloud instance.

It uses [hyperopt](https://github.com/hyperopt/hyperopt) to do the actual work.
This is based on a template [here](https://github.com/erdiolmezogullari/docker-parallel-hyperopt).

For local development, you can run this without docker or mongodb for fast debugging.


## Installation

### Development

`pip install -r requirements.txt`


### On a brand new VPS (e.g. on google cloud)

1. clone this repo
2. `sudo bash install.sh`


## Quickstart

To run hyperparameter search, you have to define a template Rasa NLU config file and a search space.

### Template config
Here is an example. Just replace the parameters you want to search over with variable names:

```
language: en
pipeline:
- name: "intent_featurizer_count_vectors"
  analyzer: char_wb
  max_df: {max_df}
  min_ngrams: 2
  max_ngrams: {max_ngrams}
- name: "intent_classifier_tensorflow_embedding"
  epochs: {epochs}
```

Save this at `hyperopt/data/template_config.yml`

### Search Space

You need to define a search space in the `hyperopt/src/space.py` file.
This is mounted into docker rather than copied into the container so 
you don't have to rebuild when you change something.

```
from hyperopt import hp

search_space = {
    'epochs': hp.qloguniform('epochs', 0, 4, 2),
    'max_df': hp.uniform('max_df', 0.01, 1.0),
    'max_ngrams': hp.quniform('max_ngram', 3, 9, 1)
}
```

Check the hyperopt docs for details on how to define a space.


### Training and test data

Put your training and test data in `hyperopt/data/{train, test}.md`
You can do a train-test split in rasa nlu with:

```
from rasa_nlu.training_data import load_data
data = load_data('all_my_data.md')
train, test = data.train_test_split(train_frac=0.7)
```

and you can write markdown by writing the output of `train.as_markdown()` to a file.


## Start your experiment

Set the experiment name and max evaluations in your `.env` file

Here is an example:

```
EXPERIMENT_KEY=default-experiment
MAX_EVALS=100
MONGO_DATABASE=foo_db
HYPEROPT_DIR=/hyperopt
```

To run:

`docker-compose up -d --scale hyperopt-mongo-worker=4`

It's up to you how many workers you want to run.
A good first guess is to set it to the numer of CPUs your machine has.

## Seeing results

All evaluations are stored in mongodb. By default, the loss is defined
as `1 - f`, where `f` is the f1 score of the intent evaluation on your test data.