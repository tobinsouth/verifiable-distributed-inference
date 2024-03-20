# Imports
import json
import re
import ezkl
import os
import onnxruntime as ort
import torchvision
import numpy as np

# Constants
PY_RUN_ARGS = ezkl.PyRunArgs()
PY_RUN_ARGS.input_visibility = "public"
PY_RUN_ARGS.output_visibility = "public"
PY_RUN_ARGS.param_visibility = "fixed"
MODEL_DIR = "./model"
DATA_DIR = "./data"
PROOF_DIR = "./proof"
# Choose between "resources" and "accuracy"
EZKL_OPTIMIZATION_GOAL = "resources"
MODEL_ID = "model_2"

# Setup & Helper Functions
# Create `PROOF_DIR` if it doesn't exist already.
os.makedirs(PROOF_DIR, exist_ok=True)


# Fetches input data for the first shard.
def fetch_mnist_input_data() -> np.ndarray:
    test_data = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=False,
                                           transform=torchvision.transforms.ToTensor())
    return fetch_first_image(test_data)


# Helper function to convert first MNIST element to a NumPy array.
def fetch_first_image(data) -> np.ndarray:
    image, _ = data[0]
    np_image = image.numpy()[np.newaxis, :]
    return np_image


# Fetch random input data for the first shard.
def fetch_random_input_data(shape: tuple):
    return np.random.rand(*shape)


# Set input_data to the output of the previous shard. Doesn't need to be set for the first shard.
def generate_example_model_output(model_path, data_path, input_data=None):
    ort_session = ort.InferenceSession(model_path)
    if input_data is None:
        # This can be swapped out with `fetch_random_input_data()`
        input_data = fetch_mnist_input_data()
    output_data = ort_session.run(None, {'input': input_data})
    witness_data = dict(input_shapes=[input_data.shape],
                        input_data=[input_data.reshape([-1]).tolist()],
                        output_data=[o.reshape([-1]).tolist() for o in output_data])
    with open(data_path, 'w') as f:
        json.dump(witness_data, f)
    # The first element in output_data is the actual NumPy ndarray we need to pass to the function when we call it for
    # the next shard.
    return output_data[0]


# Retrieves the number of Shards that were created for a corresponding `model_id`.
def get_number_of_shards(model_id):
    pattern = re.compile(f"{model_id}_shard_(\d+).onnx")
    highest_shard_id = -1
    for file in os.listdir(MODEL_DIR):
        match = pattern.match(file)
        if match:
            match_shard_id = int(match.group(1))
            if match_shard_id > highest_shard_id:
                highest_shard_id = match_shard_id
    return highest_shard_id


# Generates ezkl proof for shard `shard_id`. Requires the output of the previous shard as input for subsequent one.
# `previous_shard_output` is not required for the first shard.
def generate_proof(shard_id, previous_shard_output=None):
    model_path = f"{MODEL_DIR}/{MODEL_ID}_shard_{shard_id}.onnx"
    settings_path = f"{PROOF_DIR}/{MODEL_ID}_shard_{shard_id}_settings.json"
    data_path = f"{PROOF_DIR}/{MODEL_ID}_shard_{shard_id}_data.json"
    compiled_model_path = f"{PROOF_DIR}/{MODEL_ID}_shard_{shard_id}_network.compiled"
    srs_path = f"{PROOF_DIR}/{MODEL_ID}_shard_{shard_id}_kzg.srs"
    vk_path = f"{PROOF_DIR}/{MODEL_ID}_shard_{shard_id}_vk.key"
    pk_path = f"{PROOF_DIR}/{MODEL_ID}_shard_{shard_id}_pk.key"
    witness_path = f"{PROOF_DIR}/{MODEL_ID}_shard_{shard_id}_witness.json"
    proof_path = f"{PROOF_DIR}/{MODEL_ID}_shard_{shard_id}_proof.pf"

    # Generate Settings File
    res = ezkl.gen_settings(
        model=model_path,
        output=settings_path,
        py_run_args=PY_RUN_ARGS
    )
    assert res == True

    # Calibrate Settings
    # We only need to save the intermediate output here, as the witness data is placed in `DATA_PATH`.
    data_output = generate_example_model_output(model_path, data_path, previous_shard_output)
    res = ezkl.calibrate_settings(
        data=data_path,
        model=model_path,
        settings=settings_path,
        target=EZKL_OPTIMIZATION_GOAL
    )
    assert res == True

    # Compile model to a circuit
    res = ezkl.compile_circuit(
        model=model_path,
        compiled_circuit=compiled_model_path,
        settings_path=settings_path
    )
    assert res == True

    # (Down)load an SRS String
    res = ezkl.get_srs(
        settings_path=settings_path,
        logrows=None,
        srs_path=srs_path
    )
    assert res == True
    assert os.path.isfile(srs_path)

    # Setup Proof
    res = ezkl.setup(
        model=compiled_model_path,
        vk_path=vk_path,
        pk_path=pk_path,
        srs_path=srs_path
    )
    assert res == True
    assert os.path.isfile(vk_path)
    assert os.path.isfile(pk_path)

    # Generate witness file
    res = ezkl.gen_witness(
        data=data_path,
        model=compiled_model_path,
        output=witness_path
    )
    assert os.path.isfile(witness_path)

    # Create ZK-SNARK for the execution of the model
    res = ezkl.prove(
        witness=witness_path,
        model=compiled_model_path,
        pk_path=pk_path,
        proof_path=proof_path,
        srs_path=srs_path
    )
    assert os.path.isfile(proof_path)

    # Return the intermediate output of the shard, as this is needed for the subsequent proof.
    return data_output, proof_path, settings_path, vk_path, srs_path


# Verifies ezkl proof.
def verify_proof(proof_path, settings_path, vk_path, srs_path):
    # Verify proof
    res = ezkl.verify(
        proof_path=proof_path,
        settings_path=settings_path,
        vk_path=vk_path,
        srs_path=srs_path
    )
    assert res == True


# Prove Shard(s)
if __name__ == '__main__':
    num_shards = get_number_of_shards(MODEL_ID)
    print(f"Identified {num_shards} shards for model {MODEL_ID}")
    previous_shard_output = None
    for shard_id in range(num_shards + 1):
        previous_shard_output, shard_proof_path, shard_settings_path, shard_vk_path, shard_srs_path = generate_proof(
            shard_id=shard_id,
            previous_shard_output=previous_shard_output
        )
        print(f"Proof of Shard {shard_id} generated at {shard_proof_path}")

        # Added to ensure proofs are correct
        verify_proof(shard_proof_path, shard_settings_path, shard_vk_path, shard_srs_path)
        print(f"Proof of Shard {shard_id} verified")
        num_shards += 1

    print(f"Completed processing for {num_shards} shards")
