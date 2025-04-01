[![DOI](https://zenodo.org/badge/852328574.svg)](https://doi.org/10.5281/zenodo.14641605)

# ORTHRUS: Achieving High Quality of Attribution in Provenance-based Intrusion Detection Systems

This repo contains the official code of Orthrus.

## Citing our work

```
@inproceedings{jian2025,
	title={{ORTHRUS: Achieving High Quality of Attribution in Provenance-based Intrusion
	Detection Systems}},
	author={Jiang, Baoxiang and Bilot, Tristan  and El Madhoun, Nour and Al Agha, Khaldoun  and Zouaoui, Anis and Iqbal, Shahrear and Han, Xueyuan and Pasquier, Thomas},
	booktitle={Security Symposium (USENIX Sec'25)},
	year={2025},
	organization={USENIX}
}
```

You can find the paper preprint [here](https://tfjmp.org/publications/2025-usenixsec.pdf).

## Setup

### Clone the repo with submodules
```
git clone --recurse-submodules https://github.com/ubc-provenance/orthrus.git
```

### Download files
1. create a new folder (referred to as the *data folder*) and download all `.gz` files from a specific DARPA dataset (follow the link provided for DARPA E3 [here](https://drive.google.com/drive/folders/1fOCY3ERsEmXmvDekG-LUUSjfWs6TRdp-) and DARPA E5 [here](https://drive.google.com/drive/folders/1GVlHQwjJte3yz0n1a1y4H4TfSe8cu6WJ)). If using CLI, [use gdown](https://stackoverflow.com/a/50670037/10183259), by taking the ID of the document directly from the URL. In some cases, the downloading of a file may stop, in this case, simply ctrl+C and re-run the same gdown command with `--continue` until the file is fully downloaded. 
**NOTE:** Old files should be deleted before  downloading a new dataset.

2. in the data folder, download the java binary (ta3-java-consumer.tar.gz) to build the avro files for DARPA [E3](https://drive.google.com/drive/folders/1kCRC5CPI8MvTKQFvPO4hWIRHeuUXLrr1) and [E5](https://drive.google.com/drive/folders/1YDxodpEmwu4VTlczsrLGkZMnh_o70lUh).

3. in the data folder, download the schema files (i.e. files with filename extension '.avdl' and '.avsc') for DARPA [E3](https://drive.google.com/drive/folders/1gwm2gAlKHQnFvETgPA8kJXLLm3L-Z3H1) and [E5](https://drive.google.com/drive/folders/1fCdYCIMBCm7gmBpmqDTuoMhbOoIY6Wzq).

### Docker Container
#### Docker Install

First install Docker following the [steps from the official site](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository).

#### Building Image
For a quick environment setup using docker, under orthrus/:
1. In ```orthrus/compose.yml```, set ```/path/of/data/folder``` as the data folder

2. Build the local image:
    ```
    sudo docker compose build
    ```
3. Start the container up:
    ```
    sudo docker compose up -d
    ```
4. In a first terminal, start the `orthrus container`, where the python env is installed and where experiments will be conducted:
    ```
    sudo docker compose exec orthrus bash
    ```
5. In a second terminal, fire the `postgres container`, where the database will be installed:
    ```
    sudo docker compose exec postgres bash
    ```

These two containers will be used in the following steps.

### Convert bin files to JSON

At this stage, the data folder should contain the downloaded dataset files (.gz), the java client (tar.gz) and the schema files (.avdl, .avsc).

Then go back to the `orthrus container` within the ```/home``` folder and run the following command to convert files:

```shell
./settings/scripts/uncompress_darpa_files.sh /data/
```

> This may take multiple hours based on the dataset.

### Create the database
Within `postgres container`'s shell, simply run:

```shell
./scripts/create_database.sh dataset_name
```
where `dataset_name` is one of: `[clearscope_e3 | cadets_e3 | theia_e3 | clearscope_e5 | cadets_e5 | theia_e5]`

### Optional configurations
- optionally, if using a specific postgres database instead of the postgres docker, update the connection config by setting `DATABASE_DEFAULT_CONFIG` within `src/config.py`.

- optionaly, the `ROOT_ARTIFACT_DIR` within `src/config.py` can be changed. All preprocessed files and model weights will be stored there when the code runs.

- optionaly, if using a manually-set environment instead of the orthrus docker, go to `src/config.py` and search for `DATASET_DEFAULT_CONFIG` and set the path to the uncompressed JSON files folder in the `raw_dir` variable of your downloaded dataset.

### Fill the database

Within `orthrus container`'s shell, fill the database for the corresponding dataset by running this command:

```shell
python src/create_database.py [CLEARSCOPE_E3 | CADETS_E3 | THEIA_E3 | CLEARSCOPE_E5 | CADETS_E5 | THEIA_E5]
```

**Note:** Large storage capacity is needed to download, parse and save datasets and databases, as well as to run experiments. A single run can generate more than 15GB of artifact files on E3 datasets, and much more with larger E5 datasets.

## Run experiments

The following commands should be executed within the `orthrus container`.

### Reproduce results from the paper

Launching Orthrus is as simple as running:

```shell
python src/orthrus.py [dataset] [config args...]
```

Running `orthrus.py` will run by default the `graph_construction`, `edge_featurization`, `detection` and `attack_reconstruction` tasks configured within the `config/orthrus.yml` file. This configuration can be updated directly in the YML file or from the CLI, as shown above.

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
python src/orthrus.py CLEARSCOPE_E3 --graph_construction.build_graphs.time_window_size=1.0 --detection.gnn_training.encoder.graph_attention.dropout=0.1
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

When run once, datasets are preprocessed and stored in the `ROOT_ARTIFACT_DIR` path within `config.py`. There is thus no need to recompute them. To avoid re-computing the `graph_construction` and `edge_featurization` tasks, Orthrus can be run directly from the `detection` task using the arg `--run_from_training`.

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

## License

See [licence](LICENSE).
