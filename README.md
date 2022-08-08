## Hyperparameter Search for [Rasa NLU](https://rasa.com/docs/nlu/)

This repo provides a setup for doing hyperparameter search for the best
configuration of the [pipeline components](https://rasa.com/docs/nlu/components/).
This can either be done locally or on a cluster. It uses [hyperopt](https://github.com/hyperopt/hyperopt) to do the actual work.
This is based on a template 
[here](https://github.com/erdiolmezogullari/docker-parallel-hyperopt).

For local development, you can run this without docker or mongodb for fast 
debugging.

This repo also includes a [Github action](#github-action) for running nlu-hyperopt in a workflow.  

## Installation

### Development

`pip install -r requirements.txt`


### On a brand new ubuntu 16 VPS (e.g. on google cloud)

1. clone this repo
2. `sudo bash install/install.sh`

This will install Docker and docker-compose.

### Quickstart

To run a quick test whether everything works, run `docker-compose up`.
This will run a default experiment with the provided sample configuration and
data.

## How to Use

### Step 1: Write a Template Configuration
Here is an example. Replace the parameters you want to search over with 
variable names:

```yaml
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

Save this at `data/template_config.yml`

### Step 2: Define a Search Space

You need to define a search space in the `nlu_hyperopt/space.py` file.

```python
from hyperopt import hp
from hyperopt.pyll.base import scope

search_space = {
    'epochs': hp.qloguniform('epochs', 0, 4, 2),
    'max_df': hp.uniform('max_df', 1, 2),
    'max_ngrams': scope.int(hp.quniform('max_ngram', 3, 9, 1))
    }
}
```

Check the [hyperopt docs](https://github.com/hyperopt/hyperopt/wiki/FMin#2-defining-a-search-space) 
for details on how to define a space.


### Step 3: Provide Your Training and Test data

Put your training and test data in `train_test_split/{training_data, test_data}.yml`
You can do a train-test split in Rasa NLU with:

```sh
rasa data split nlu
```

You can specify a non-default `--training-fraction` as a decimal; the default is `0.8`.


## Step 4: Configure your experiment

This table lists all the options you can configure through
environment variables:

| Environment Variable | Description                                                                                                                                                                                                                                                                                                                                                                                                |
|----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| INPUT_MAX_EVALS            | Maximum number of evaluations which are run during the hyperparameter search                                                                                                                                                                                                                                                                                                                               |
| INPUT_DATA_DIRECTORY       | Directory which contains the files `training_data.yml`,`test_data.yml`, and `template_config.yml` (default: `./train_test_split`)                                                                                                                                                                                                                                                                                                     |
| INPUT_MODEL_DIRECTORY      | Directory which contains the trained models (default: `./models`)                                                                                                                                                                                                                                                                                                                                           |
| INPUT_TARGET_METRIC        | Target metric for the evaluation. You can choose between `f1_score`, `accuracy`, `precision`, and `threshold_loss`.                                                                                                                                                                                                                                                                                        |
| INPUT_THRESHOLD            | Only used by `threshold_loss`. Sets the threshold which the confidence of the correct intent has to be above or wrong predictions have to be below (default: 0.8).                                                                                                                                                                                                                                         |
| INPUT_ABOVE\_BELOW\_WEIGHT | Only used by `threshold_loss` (default: 0.5). This loss function penalizes incorrect predictions above the given threshold and correct predictions below a certain threshold. With the `ABOVE_BELOW_WEIGHT` you can configure the balance between these penalties. A larger value means that incorrect predictions above the threshold are penalized more heavily than correct predictions below the threshold. |

## Step 5: Start your experiment

### Run it locally

To quickly test on your local machine without docker or mongodb:
 
```
python -m nlu_hyperopt.app
```

### Running on a Server

Set the experiment name and max evaluations in your `.env` file

Here is an example:

```
INPUT_EXPERIMENT_KEY=default-experiment
INPUT_MAX_EVALS=100
INPUT_MONGO_URL=mongodb:27017/nlu-hyperopt
```

To run:

`docker-compose up -d --scale hyperopt-worker=4`

It's up to you how many workers you want to run.
A good first guess is to set it to the numer of CPUs your machine has.

## Experiment Results

The best configuration is printed by the hyperopt-master at the end of the 
the hyperparameter search.
All evaluation results are stored in the mongodb immediately after they run.
To see the results while the optimization is running, open a mongo shell session in the mongo container:

Run this command to see the experiment with the lowest value of the loss so far


```
use nlu-hyperopt
db.jobs.find({"exp_key" : "default-experiment", "result.loss":{$exists: 1}}).sort({"result.loss": 1}).limit(1).pretty()
```

replacing the value of the `exp_key` with your experiment name.


# Loss Functions

## f1_score
This loss is defined as `1 - f`, where `f` is the f1 score of the intent 
evaluation on your test data.

## accuracy
This loss is defined as `1 - f`, where `f` is the accuracy score of the 
intent evaluation on your test data.

## precision
This loss is defined as `1 - f`, where `f` is the precision score of the 
intent evaluation on your test data.

## threshold_loss
This loss is defined as 
```
l * incorrect_above + (1-l) * correct_below
```

 where
`incorrect_above` describes the fraction of incorrect predictions above a 
certain threshold and `correct_below` describes the fraction of correct 
predictions below a certain threshold. Threshold and `l` can be configured 
through environment variables.

# Github Action

> Take note of Github Action's [usage limit](https://help.github.com/en/actions/getting-started-with-github-actions/about-github-actions#usage-limits)
of 360 minutes per job. Keep this in mind when choosing `max_evals`.

## Inputs

- `search_space`: **Required** Path to your search space definition (`space.py`)
- `data_directory`:  **Required** see `INPUT_DATA_DIRECTORY`
- `max_evals`: see `INPUT_MAX_EVALS`
- `target_metric`: see `INPUT_TARGET_METRIC`
- `threshold`: see `INPUT_THRESHOLD`
- `above_below_weight`: see `INPUT_ABOVE_BELOW_WEIGHT`

## Example usage

```yaml
jobs:
  nlu-hyperopt:
    name: NLU hyperparameter optimization
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - uses: RasaHQ/nlu-hyperopt@v1
      name: Run NLU Hyperoptimization
      with:
        max_evals: 50
        target_metric: f1_score
        data_directory: ${{ github.workspace }}/train_test_split
        search_space: ${{ github.workspace }}/nlu_hyperopt/space.py

```
