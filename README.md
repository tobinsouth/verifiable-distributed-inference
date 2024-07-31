# Distributed Verifiable Model Inference 
This is a project that leverages zkSNARKs to enable partial-privacy in a distributed set of (untrusted) nodes that collectively run an ML model. 
Each node/worker is assigned a shard to run and can later be prompted to generate a zkSNARK, proving that the model (shard) M was run correctly by taking an input x and generating an input y using M.
To create said zkSNARKs for ML inference runs, we leverage the [ezkl](https://github.com/zkonduit/ezkl) toolkit.

## Project Structure
Main files
- `config.py` contains important configuration values. As noted in the file, feel free to adjust the first few variables, but we recommend to keep certain values as configured.
- `worker.py`
- `coordinator.py`

## Run the Project



First spawn a (1) coordinator
Then spawn (N) workers

There are four (4) models available in `modules/model_training.py`

---
Sample Usage:

## Run Benchmarks

There are two benchmarking scripts in the `/benchmarking` directory: 
- `accuracy_benchmark.py`
- `system_benchmark.py`

---
Sample Usage:

## Additional Notes

This codebase is part of a research project. The corresponding paper(s) can be found here: [TODO]()