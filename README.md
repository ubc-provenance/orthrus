[![DOI](https://zenodo.org/badge/852328574.svg)](https://doi.org/10.5281/zenodo.14641605)

# ORTHRUS: Achieving High Quality of Attribution in Provenance-based Intrusion Detection Systems

This repo contains the official code of the [Orthrus paper](https://www.usenix.org/system/files/conference/usenixsecurity25/sec25cycle1-prepub-103-jiang-baoxiang.pdf).

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

## Updates

[2025.06.04] setup guidelines are now simplified. The DARPA TC databases can be directly downloaded and installed locally. No need to fill them locally anymore.

## Setup

### Clone the repo with submodules
```
git clone --recurse-submodules https://github.com/ubc-provenance/orthrus.git
```

### 10-min install of Docker and Datasets

We have made the installation of DARPA TC/OpTC easy and fast, simply follow [these guidelines](https://github.com/ubc-provenance/PIDSMaker/blob/velox/settings/ten-minute-install.md).

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
| CADETS_E3_full  | 29  | 22  | 268,063   | 39  | 0.57   | 0.49   |
| CADETS_E3_ano   | 17   | 1   | 268,084   | 51  | 0.94   | 0.49   |
| THEIA_E3_full  | 22  | 0  | 699,177   | 96  | 1.00   | 0.43   |
| THEIA_E3_ano    | 2   | 0   | 699,177   | 116 | 1.00   | 0.13   |
| CADETS_E5_full  | 2   | 10  | 3111,245  | 121 | 0.17   | 0.05   |
| CADETS_E5_ano   | 1   | 5   | 3111,250  | 122 | 0.17   | 0.04   |
| THEIA_E5_full  | 13  | 2   | 747,381   | 56  | 0.86   | 0.40   |
| THEIA_E5_ano    | 2   | 0   | 747,383   | 67  | 1.00   | 0.17   |
| CLEARSCOPE_E3_full  | 1   | 504   | 110,858   | 40 | 0.00  | 0.00 |
| CLEARSCOPE_E3_ano | 1 | 6 | 111,356 | 40  | 0.14  | 0.06  |
| CLEARSCOPE_E5_full  | 4  | 201   | 150,473 | 47  | 0.20   | 0.04   |
| CLEARSCOPE_E5_ano | 2   | 7   | 150,667 | 49  | 0.22   | 0.09   |


#### Experiments

**CADETS_E3**
```
PYTHONHASHSEED=0 python src/orthrus.py CADETS_E3 --detection.gnn_training.encoder.graph_attention.dropout=0.25 --detection.gnn_training.node_hid_dim=256 --detection.gnn_training.node_out_dim=256 --detection.gnn_training.lr=0.001 --detection.gnn_training.num_epochs=20 --seed=4
```

**THEIA_E3**
```
PYTHONHASHSEED=0 python src/orthrus.py THEIA_E3 --detection.gnn_training.encoder.graph_attention.dropout=0.1
```

**CLEARSCOPE_E3**
```
PYTHONHASHSEED=0 python src/orthrus.py CLEARSCOPE_E3 --graph_construction.build_graphs.time_window_size=1.0 --detection.gnn_training.encoder.graph_attention.dropout=0.1 --seed=2
```

**CADETS_E5**
```
PYTHONHASHSEED=0 python src/orthrus.py CADETS_E5 --detection.gnn_training.node_out_dim=128 --detection.gnn_training.lr=0.0001
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
