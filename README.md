# Distributed Verifiable Model Inference 
This is a project to see if we can run distributed inference of ML models in a verifiable way by running intermediate chunks via zk-SNARKs. To make this easy in prototyping, we will use the [ezkl](https://github.com/zkonduit/ezkl) toolkit.


## Repository Structure
- `model.py`: We build a model and train it on simple data; we then split this model into chunks and export each sub-chunk to an onnx file.
- `distributed_run.py`: This is where we will test how models are being run in a distributed manner (currently just using onnxruntime but could build this out.)
- `proven_run.py`: This is where we're going to proof setups for each sub-model and write code to verify across them.