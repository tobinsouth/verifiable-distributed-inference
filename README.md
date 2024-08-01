# Distributed Verifiable Model Inference 
This is a project that leverages zkSNARKs to enable partial-privacy in a distributed set of (untrusted) nodes that collectively run an ML model. 
Each node/worker is assigned a shard to run and can later be prompted to generate a zkSNARK, proving that the model (shard) M was run correctly by taking an input x and generating an input y using M.
To create said zkSNARKs for ML inference runs, we leverage the [ezkl](https://github.com/zkonduit/ezkl) toolkit.

## Project Structure
Main files
- `config.py` contains important configuration values. As noted in the file, feel free to adjust the first few variables, but we recommend to keep certain values as configured.
- `worker.py` contains the logic for the worker nodes that run individual shards.
- `coordinator.py` contains the logic for the coordinator that assigns the shards to the workers and orchestrates the entire process.

## Installation
The use of a Python [venv](https://docs.python.org/3/library/venv.html) is recommended.

Install dependencies: 
```shell
pip install -r requirements.txt
```

Optionally, if you want to use ezkl's CLI version (v12.0.1), install as follows: 
```shell
curl https://raw.githubusercontent.com/zkonduit/ezkl/main/install_ezkl_cli.sh | bash -s -- v12.0.1
```
If you wish to use ezkl's CLI version, keep `USE_EZKL_CLI = True` in [config.py](config.py), otherwise set it to `False`.

## Run the Project

First spawn a (1) coordinator, then spawn (N) workers **sequentially**.

There are four (4) models available in [model_training.py](modules%2Fmodel_training.py):
1. mlp: A simple 711 parameter MLP-style model
2. cnn: A Convolutional Neural Network with 26K parameters
3. mlp2: A large 548K parameter MLP-style model
4. attention: A large 1.19M parameter attention-style model

When running the system, use one of the values above to set the model that's going to be used.

**Warning:** mlp2 and attention require multiple hundred GB of free RAM space and take a significant time to run. 
Use with caution! 

### Sample Usage:
First spawn a coordinator.
The coordinator accepts connections on localhost:8000, and expects a 4-shard setup with the mlp model.
```shell
python coordinator.py localhost 8000 4 mlp
```
All following workers must be spawned in order.

Spawn the first worker. This worker takes on the role FIRST.
```shell
python worker.py localhost 8001 localhost 8000 FIRST
```

Spawn the second worker
```shell
python worker.py localhost 8002 localhost 8000 MIDDLE
```

Spawn the third worker
```shell
python worker.py localhost 8003 localhost 8000 MIDDLE
```

Spawn the fourth worker. This worker takes on the role LAST.
```shell
python worker.py localhost 8004 localhost 8000 LAST
```

If you want to spawn the setup with a single worker, make sure to give the worker the role SOLO.

## Run Benchmarks

There are two benchmarking scripts in the `/benchmarking` directory: 
- [accuracy_benchmark.py](benchmarking%2Faccuracy_benchmark.py): Benchmarking for ezkl's quantization of values
- [system_benchmark.py](benchmarking%2Fsystem_benchmark.py): Benchmarking for the entire system and determining ezkl proof artifact sizes and generation times

We recommend executing the respective benchmarking scripts from within the `/benchmarking` directory.
```shell
cd ./benchmarking
```

### Sample Usage:

Example for running the accuracy benchmark for the mlp model and storing the results in the `./tmp-mlp` directory.
When running this benchmark in parallel, make sure to use a different directory for each run as files get deleted.
```shell
python accuracy_benchmark.py mlp ./tmp-mlp
```

Example for running the system benchmark for the cnn model and defaults to storing the results in the `./tmp-system-benchmark` directory.
```shell
python system_benchmark.py cnn
```


## Additional Notes

This codebase is part of a research project. The corresponding paper(s) can be found here: [TODO]()