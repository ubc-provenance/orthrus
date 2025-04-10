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
You can download all required files directly by running:

```shell
pip install gdown
```
```shell
./settings/scripts/download_{dataset}.sh {data_folder}
```
where `{dataset}` can be either `clearscope_e3`, `cadets_e3`, `theia_e3`, `clearscope_e5`, `cadets_e5` or `theia_e5` and `{data_folder}` is the absolute path to the output folder where all raw files will be downloaded.

Alternatively, you can [download the files manually](settings/download-files.md) by selecting download URLs from Google Drive.

### Docker Container
#### Docker Install

1. First install Docker following the [steps from the official site](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository).

2. Then, install dependencies for CUDA support with Docker:

```shell
# Add the NVIDIA package repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update and install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart services
sudo systemctl restart docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

#### Building Image
We aim to use two containers: one for the python env, where experiments will be conducted, and one running the postgres database.

1. In ```orthrus/compose.yml```, set ```/path/of/data/folder``` as the data folder

2. Build the local image (under `orthrus/`):
    ```
    sudo docker compose build
    ```
3. Start the container up:
    ```
    sudo docker compose up -d --build
    ```
4. In a terminal, get a shell into the `orthrus container`, where the python env is installed and where experiments will be conducted:
    ```
    sudo docker compose exec orthrus bash
    ```
5. (optional) You can get a shell to the `postgres container` for running specific commands on the postgres database.
    ```
    sudo docker compose exec postgres bash
    ```

In the remaining steps, only `orthrus container` will be used.

### Convert bin files to JSON

At this stage, the data folder should contain the downloaded dataset files (.gz), the java client (tar.gz) and the schema files (.avdl, .avsc).

Go to the `orthrus container` within the ```/home``` folder and run the following command to convert files:

```shell
./settings/scripts/uncompress_darpa_files.sh /data/
```

> This may take multiple hours depending on the dataset.

### Optional configurations
- optionally, if using a specific postgres database instead of the postgres docker, update the connection config by setting `DATABASE_DEFAULT_CONFIG` within `src/config.py`.

- optionaly, if you want to change the output folder where generated files are stored, update accordingly the volume by uncommenting `./artifacts:/home/artifacts` in `compose.yml`.

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

> [!NOTE]
> Due to missing PYTHONHASHSEED when seeding Gensim’s Word2Vec, we were unable to exactly reproduce the original results, though the following experiments produce closely aligned outcomes.

#### Expected results
| Name             | TP  | FP  | TN       | FN  | Precision | MCC       |
|------------------|-----|-----|----------|-----|-----------|-----------|
| CADETS_E3_full  | 21  | 13  | 268,072   | 47  | 0.61   | 0.43   |
| CADETS_E3_ano   | 9   | 0   | 268,085   | 59  | 1.00   | 0.36   |
| THEIA_E3_full  | 36  | 10  | 699,167   | 82  | 0.78   | 0.48   |
| THEIA_E3_ano    | 6   | 0   | 699,177   | 112 | 1.00   | 0.22   |
| CADETS_E5_full  | 1   | 11  | 3111,244  | 122 | 0.08   | 0.02   |
| CADETS_E5_ano   | 1   | 4   | 3111,251  | 122 | 0.20   | 0.04   |
| THEIA_E5_full  | 13  | 2   | 747,381   | 56  | 0.86   | 0.40   |
| THEIA_E5_ano    | 2   | 0   | 747,383   | 67  | 1.00   | 0.17   |
| CLEARSCOPE_E5_full  | 3   | 5   | 150,669   | 48  | 0.37   | 0.14   |
| CLEARSCOPE_E5_ano | 1   | 4   | 150,670 | 50  | 0.20   | 0.06   |

#### Experiments

**CADETS_E3**
```
PYTHONHASHSEED=0 python src/orthrus.py CADETS_E3 --detection.gnn_training.encoder.graph_attention.dropout=0.25 --detection.gnn_training.node_hid_dim=256 --detection.gnn_training.node_out_dim=256 --detection.gnn_training.lr=0.001 --detection.gnn_training.num_epochs=20
```

**THEIA_E3**
```
PYTHONHASHSEED=0 python src/orthrus.py THEIA_E3
```

**CLEARSCOPE_E3**
```
PYTHONHASHSEED=0 python src/orthrus.py CLEARSCOPE_E3 --graph_construction.build_graphs.time_window_size=1.0 --detection.gnn_training.encoder.graph_attention.dropout=0.1
```

**CADETS_E5**
```
PYTHONHASHSEED=0 python src/orthrus.py CADETS_E5 --detection.gnn_training.lr=0.0001
```

**THEIA_E5**
```
PYTHONHASHSEED=0 python src/orthrus.py THEIA_E5
```

**CLEARSCOPE_E5**
```
PYTHONHASHSEED=0 python src/orthrus.py CLEARSCOPE_E5 --detection.gnn_training.lr=0.0001 --detection.gnn_training.encoder.graph_attention.dropout=0.1 --detection.gnn_training.node_out_dim=128 
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
