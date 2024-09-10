# ORTHRUS: Achieving High Quality of Attribution in Provenance-based Intrusion Detection Systems

This repo contains the official code of Orthrus.

## Setup

- install the environment and requirements ([guidelines](settings/environment-settings.md)).
- download files from DARPA datasets, parse bin files into json files, put all json
files in a folder and set `raw_dir` value of dict `DATASET_DEFAULT_CONFIG` in `src/config.py` 
as the json file folder path
- create postgre databases ([guidelines](settings/database.md), replace `database_name` with the name of the dataset)
- optionaly, the `ROOT_ARTIFACT_DIR` within `src/config.py` can be changed. All preprocessed files and model weights will be stored there when the code runs
- run scripts to fill the database from the downloaded files with the following command:

```shell
python src/create_database.py [dataset]
```

**Note:** Large storage capacity is needed to download, parse and save datasets and databases.

**Note:** Large storage capacity is needed to run experiments. A single run can generate more than 15GB of artifact files on E3 datasets, and much more with larger E5 datasets.

## Run experiments

### Reproduce results from the paper

Launching Orthrus is as simple as running:

```shell
python src/orthrus.py [dataset] [config args...]
```

Running `orthrus.py` will run by default the `preprocessing`, `featurization`, `detection` and `triage` tasks configured within the `config/orthrus.yml` file. This configuration can be updated directly in the YML file or from the CLI, as shown above.

To reproduce the experimental results of Orthrus on node detection:


**CADETS_E3**
```
python src/orthrus.py CADETS_E3 --detection.gnn_training.num_epochs=20 --detection.gnn_training.encoder.graph_attention.dropout=0.25 --detection.evaluation.node_evaluation.kmeans_top_K=30
```

**THEIA_E3**
```
python src/orthrus.py THEIA_E3
```

**CLEARSCOPE_E3**
```
python src/orthrus.py CLEARSCOPE_E3 --preprocessing.build_graphs.time_window_size=1.0 --detection.gnn_training.encoder.graph_attention.dropout=0.1
```

**CADETS_E5**
```
python src/orthrus.py
```

**THEIA_E5**
```
python src/orthrus.py THEIA_E5 --detection.gnn_training.lr=0.000005
```

**CLEARSCOPE_E5**
```
python src/orthrus.py CLEARSCOPE_E5 --detection.gnn_training.num_epochs=10 --detection.gnn_training.lr=0.0001 --detection.gnn_training.encoder.graph_attention.dropout=0.25
```

### Subsequent runs

When run once, datasets are preprocessed and stored in the `ROOT_ARTIFACT_DIR` path within `config.py`. There is thus no need to recompute them. To avoid re-computing the `preprocessing` and `featurization` tasks, Orthrus can be run directly from the `detection` task using the arg `--run_from_training`.

```shell
python src/orthrus.py CADETS_E3 --run_from_training
```

### Weights & Biases interface

W&B is used as the default interface to visualize and historize experiments. First log into your account from the CLI using:

```shell
wandb login
```

Set your API key, which can be found on the website. Then you can push the logs and results of experiments to the interface using the `--wandb` arg.
The preferred solution is to run the `run.sh` script, which directly logs the experiments to the W&B interface.

```shell
python src/orthrus.py THEIA_E3 --wandb
```
