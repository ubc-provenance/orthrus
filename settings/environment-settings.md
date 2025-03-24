# Environment settings

## Docker
For a quick environment setup using docker, under orthrus/:
1. In ```orthrus/compose.yml```, set ```/path/of/data/folder``` as the data folder

2. Then, use the following command to build the local image:
    ```
    docker-compose build
    ```
3. Then, you can launch a temporary shell with the environment:
    ```
    docker-compose run --user $(id -u) --rm orthrus
    ```
4. Once inside the container, start the conda shell:
    ```
    source /opt/conda/bin/activate orthrus
    ```

Or you can set up the environment manually as following:

## Prerequisites
We use the following settings to run the experiments reported in the paper:
1. OS Version: 5.19.0-46-generic #47~22.04.1-Ubuntu
2. Anaconda: 23.3.1
3. PostgresSQL: Version 15.3, Ubuntu 15.3-1.pgdg22.04+1 ([installation guide](https://www.cherryservers.com/blog/how-to-install-and-setup-postgresql-server-on-ubuntu-20-04))
4. GraphViz: 2.43.0 
5. GPU (Driver Version: 530.41.03): CUDA Version 12.1

## Python Libraries
Install the following libraries:

```commandline
conda create -n orthrus python=3.9
conda activate orthrus
# Note: using "pip install psycopg2" to install may fail
conda install psycopg2
conda install tqdm
# We encountered a problem in feature hashing functions with version 1.2.2
pip install scikit-learn==1.2.0
pip install networkx==2.8.7
pip install xxhash==3.2.0
pip install graphviz==0.20.1
pip install psutil
pip install scipy==1.10.1
pip install matplotlib==3.8.4
pip install wandb==0.16.6
pip install chardet==5.2.0
pip install nltk==3.8.1
pip install igraph==0.11.5
pip install cairocffi==1.7.0

# Replace your CUDA version here
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install torch_geometric==2.5.3 --no-cache-dir
pip install pyg_lib==0.2.0 torch_scatter==2.1.1 torch_sparse==0.6.17 torch_cluster==1.6.1 torch_spline_conv==1.2.2 -f https://data.pyg.org/whl/torch-1.13.0+cu117.html --no-cache-dir

# Gensim Word2Vec library 
pip install gensim==4.3.1

pip install pytz==2024.1 pandas==2.2.2 yacs==0.1.8

# Reinstall numpy to solve C API problem
conda install numpy=1.26.4

```

## Troubleshooting

**Issue**: When running `psycopg2.connect()`, I received this error:
```
OperationalError: connection to server on socket "/var/run/postgresql/.s.PGSQL.5432" failed: FATAL:  Peer authentication failed for user "postgres".
```

**Solution**: Follow the solution in [this](https://stackoverflow.com/questions/18664074/getting-error-peer-authentication-failed-for-user-postgres-when-trying-to-ge) Stack Overflow post.

**Issue**: When running `psycopg2.connect()`, I received this error:
```
OperationalError: could not connect to server: No such file or directory the server running locally and accepting connections on Unix domain socket "/XXX/.s.PGSQL.5432"?
```

**Solution**:
* Check if `postgres` is running. If not, start it, re-run the code, and see if the problem still exists.
* If the problem still exists when `postgres` is running, identify the location of the file `.s.PGSQL.5432`. 
Then set the `host` parameter in `psycopg2.connect()` to be `/the/location/of/the/file/`. The problem should be fixed then.