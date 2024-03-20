# Imports
import json
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
# Either resources or accuracy
EZKL_OPTIMIZATION_GOAL = "resources"
MODEL_ID = "model"

# Setup & Helper Functions
def fetch_test_data():
    test_data = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=False,
                                           transform=torchvision.transforms.ToTensor())
    return fetch_first_image(test_data)


def fetch_first_image(data):
    image, _ = data[0]
    np_image = image.numpy()[np.newaxis, :]
    return np_image


def generate_example_model_output(model_path, data_path):
    ort_session = ort.InferenceSession(model_path)
    input_data = fetch_test_data()
    output_data = ort_session.run(None, {'input': input_data})
    witness_data = dict(input_shapes=[input_data.shape],
                        input_data=[input_data.reshape([-1]).tolist()],
                        output_data=[o.reshape([-1]).tolist() for o in output_data])
    with open(data_path, 'w') as f:
        json.dump(witness_data, f)
    return data_path

def generate_proof():
    model_path = f"{MODEL_DIR}/{MODEL_ID}.onnx"
    settings_path = f"{PROOF_DIR}/{MODEL_ID}_settings.json"
    data_path = f"{PROOF_DIR}/{MODEL_ID}_data.json"
    compiled_model_path = f"{PROOF_DIR}/{MODEL_ID}_network.compiled"
    srs_path = f"{PROOF_DIR}/{MODEL_ID}_kzg.srs"
    vk_path = f"{PROOF_DIR}/{MODEL_ID}_vk.key"
    pk_path = f"{PROOF_DIR}/{MODEL_ID}_pk.key"
    witness_path = f"{PROOF_DIR}/{MODEL_ID}_witness.json"
    proof_path = f"{PROOF_DIR}/{MODEL_ID}_proof.pf"

    # Generate Settings File
    res = ezkl.gen_settings(
        model=model_path,
        output=settings_path,
        py_run_args=PY_RUN_ARGS
    )
    assert res == True

    # Calibrate Settings
    data_path = generate_example_model_output(model_path, data_path)
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

    return proof_path, settings_path, vk_path, srs_path


def verify_proof(proof_path, settings_path, vk_path, srs_path):
    # Verify proof
    res = ezkl.verify(
        proof_path=proof_path,
        settings_path=settings_path,
        vk_path=vk_path,
        srs_path=srs_path
    )
    assert res == True


if __name__ == "__main__":
    os.makedirs(PROOF_DIR, exist_ok=True)
    # Generate Proof
    proof_path, settings_path, vk_path, srs_path = generate_proof()
    # Verify Proof
    verify_proof(proof_path, settings_path, vk_path, srs_path)
