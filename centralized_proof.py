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

MODEL_PATH = f"{MODEL_DIR}/{MODEL_ID}.onnx"
SETTINGS_PATH = f"{PROOF_DIR}/{MODEL_ID}_settings.json"
DATA_PATH = f"{PROOF_DIR}/{MODEL_ID}_data.json"
COMPILED_MODEL_PATH = f"{PROOF_DIR}/{MODEL_ID}_network.compiled"
SRS_PATH = f"{PROOF_DIR}/{MODEL_ID}_kzg.srs"
VK_PATH = f"{PROOF_DIR}/{MODEL_ID}_vk.key"
PK_PATH = f"{PROOF_DIR}/{MODEL_ID}_pk.key"
WITNESS_PATH = f"{PROOF_DIR}/{MODEL_ID}_witness.json"
PROOF_PATH = f"{PROOF_DIR}/{MODEL_ID}_proof.pf"

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
    # Generate Settings File
    res = ezkl.gen_settings(
        model=MODEL_PATH,
        output=SETTINGS_PATH,
        py_run_args=PY_RUN_ARGS
    )
    assert res == True

    # Calibrate Settings
    DATA_PATH = generate_example_model_output(MODEL_PATH, DATA_PATH)
    res = ezkl.calibrate_settings(
        data=DATA_PATH,
        model=MODEL_PATH,
        settings=SETTINGS_PATH,
        target=EZKL_OPTIMIZATION_GOAL
    )
    assert res == True

    # Compile model to a circuit
    res = ezkl.compile_circuit(
        model=MODEL_PATH,
        compiled_circuit=COMPILED_MODEL_PATH,
        settings_path=SETTINGS_PATH
    )
    assert res == True

    # (Down)load an SRS String
    res = ezkl.get_srs(
        settings_path=SETTINGS_PATH,
        logrows=None,
        srs_path=SRS_PATH
    )
    assert res == True
    assert os.path.isfile(SRS_PATH)

    # Setup Proof
    res = ezkl.setup(
        model=COMPILED_MODEL_PATH,
        vk_path=VK_PATH,
        pk_path=PK_PATH,
        srs_path=SRS_PATH
    )
    assert res == True
    assert os.path.isfile(VK_PATH)
    assert os.path.isfile(PK_PATH)

    # Generate witness file
    res = ezkl.gen_witness(
        data=DATA_PATH,
        model=COMPILED_MODEL_PATH,
        output=WITNESS_PATH
    )
    assert os.path.isfile(WITNESS_PATH)

    # Create ZK-SNARK for the execution of the model
    res = ezkl.prove(
        witness=WITNESS_PATH,
        model=COMPILED_MODEL_PATH,
        pk_path=PK_PATH,
        proof_path=PROOF_PATH,
        srs_path=SRS_PATH
    )
    assert os.path.isfile(PROOF_PATH)

    return PROOF_PATH, SETTINGS_PATH, VK_PATH, SRS_PATH


def verify_proof():
    # Verify proof
    res = ezkl.verify(
        proof_path=PROOF_PATH,
        settings_path=SETTINGS_PATH,
        vk_path=VK_PATH,
        srs_path=SRS_PATH
    )
    assert res == True


if __name__ == "__main__":
    os.makedirs(PROOF_DIR, exist_ok=True)
    # Generate Proof
    generate_proof()
    # Verify Proof
    verify_proof()
