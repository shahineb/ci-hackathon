# Climate Informatics 2020 Data Challenge

In the context of the [Climate Informatics 2020 conference](https://ci2020.web.ox.ac.uk/home) data challenge, we attempt to predict visible imagery at night using thermal infra-red observation.

## Getting started

__From command line__

Setup a configuration file specifying experiment hyperparameters, templates provided here.
Then run

```
$ python run_training.py --cfg=path_to_config_file --o=path_to_logs_directory --device=gpu_id
```

__From a notebook__

See [demonstration notebook](https://github.com/shahineb/ci-hackathon/blob/master/notebooks/demo.ipynb) on how to run an experiment.


## Installation

Code implemented in Python 3.8

#### Setting up environment

Clone and go to repository
```bash
$ git clone https://github.com/shahineb/ci-hackathon.git
$ cd ci-hackathon
```

Create and activate environment
```bash
$ pyenv virtualenv 3.8.2 hackathon
$ pyenv activate hackathon
$ (hackathon)
```

Install dependencies
```bash
$ (hackathon) pip install -r requirements.txt
```
