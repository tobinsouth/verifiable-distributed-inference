import ezkl, os, torch, json, numpy as np
import onnxruntime # This is what helps us run models

py_run_args = ezkl.PyRunArgs()
py_run_args.input_visibility = "public"
py_run_args.output_visibility = "public"
py_run_args.param_visibility = "fixed" # "fixed" for params means that the committed to params are used for all proofs


model1 = "proofs/model_part1.onnx"
model2 = "proofs/model_part2.onnx"

# %% Functions to help us set up the proof

def setup_proof(model_path, postfix, example_data):
    """
    Sets up the proof for verifiable distributed inference.

    Args:
        model_path (str): The path to the model file.
        postfix (str): The postfix to be added to the file names.
        example_data (numpy.ndarray): The example input data.

    Returns:
        tuple: A tuple containing the following elements:
            - circuit_path (str): The path to the compiled circuit file.
            - vk_path (str): The path to the verification key file.
            - pk_path (str): The path to the proving key file.
            - example_output (numpy.ndarray): The output of the model for the example input.
            - settings_path (str): The path to the settings file.
    """
    settings_path= f"proofs/settings{postfix}.json"
    data_path = f"proofs/example_input{postfix}.json"
    circuit_path = f"proofs/compiled_circuit{postfix}.ezkl"
    vk_path = f"proofs/vk{postfix}.key"
    pk_path = f"proofs/pk{postfix}.key"

    ezkl.gen_settings(model1, settings_path, py_run_args=py_run_args)

    session = onnxruntime.InferenceSession(model_path)
    example_output = session.run(None, {"input.1": example_data})[0]

    witness_data = dict(input_shapes=[example_data.shape],
                input_data=[example_data.reshape([-1]).tolist()],
                output_data=[o.reshape([-1]).tolist() for o in example_output])
    json.dump(witness_data, open(data_path, 'w'))

    ezkl.calibrate_settings(data_path, model_path, settings_path, "resources")
    ezkl.get_srs(settings_path)
    ezkl.compile_circuit(model_path, circuit_path, settings_path)
    ezkl.setup(circuit_path, vk_path, pk_path)

    return circuit_path, vk_path, pk_path, example_output, settings_path


def cli_setup_proof(model_path, postfix, example_data):
    """Same as setup_proof, but using the command line interface."""
    settings_path= f"proofs/settings{postfix}.json"
    data_path = f"proofs/example_input{postfix}.json"
    circuit_path = f"proofs/compiled_circuit{postfix}.ezkl"
    vk_path = f"proofs/vk{postfix}.key"
    pk_path = f"proofs/pk{postfix}.key"

    os.system(f"ezkl gen-settings -M {model_path} -O {settings_path}")

    session = onnxruntime.InferenceSession(model_path)
    example_output = session.run(None, {"input.1": example_data})[0]

    witness_data = dict(input_shapes=[example_data.shape],
                input_data=[example_data.reshape([-1]).tolist()],
                output_data=[o.reshape([-1]).tolist() for o in example_output])
    json.dump(witness_data, open(data_path, 'w'))

    os.system(f"ezkl calibrate-settings -D {data_path} -M {model_path} -O {settings_path} --target=accuracy")
    os.system(f"ezkl get-srs -S {settings_path}")
    os.system(f"ezkl compile-circuit -M {model_path} --compiled-circuit {circuit_path} -S {settings_path}")
    os.system(f"ezkl setup --compiled-circuit {circuit_path} --vk-path {vk_path} --pk-path {pk_path}")

    return circuit_path, vk_path, pk_path, example_output, settings_path


# Now we can run the proof

def run_proof(circuit_path, pk_path, input_data, prefix):
    """
    Runs the proof generation process for a given circuit.

    Args:
        circuit_path (str): The path to the circuit file.
        pk_path (str): The path to the public key file.
        input_data (numpy.ndarray): The input data for the circuit.
        prefix (str): The prefix used for naming the proof and witness files.

    Returns:
        str: The path to the generated proof file.
    """
    proof_path = f"proofs/proof{prefix}.json"
    witness_path = f"proofs/witness{prefix}.json"

    witness_data = dict(input_shapes=[input_data.shape],
                input_data=[input_data.reshape([-1]).tolist()])
                # output_data=[o.reshape([-1]).tolist() for o in example_output])
    json.dump(witness_data, open(witness_path, 'w'))

    ezkl.gen_witness(witness_path, circuit_path, witness_path)
    ezkl.prove(witness_path, circuit_path, pk_path, proof_path)

    return proof_path

def cli_run_proof(circuit_path, pk_path, input_data, prefix):
    """Same as run_proof, but using the command line interface."""
    proof_path = f"proofs/proof{prefix}.json"
    witness_path = f"proofs/witness{prefix}.json"

    witness_data = dict(input_shapes=[input_data.shape],
                input_data=[input_data.reshape([-1]).tolist()])
    json.dump(witness_data, open(witness_path, 'w'))

    os.system(f"ezkl gen-witness -D {witness_path} -M {circuit_path} -O {witness_path}")
    os.system(f"ezkl prove -W {witness_path} -M {circuit_path} --pk-path {pk_path} --proof-path {proof_path}")

    return proof_path


def verify_run(vk_path, proof_path, settings_path):
    return ezkl.verify(proof_path, settings_path, vk_path)

def cli_verify_run(vk_path, proof_path, settings_path):
    return os.system(f"ezkl verify --vk-path {vk_path} --proof-path {proof_path} -S {settings_path}")    




# %% Now we can actually perform inference, proof generation and verification

# Loading in test data
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=1, shuffle=False)

example_data = next(iter(testloader))[0].numpy()


# Prove first part
circuit_path_1, vk_path_1, pk_path_1, example_output, settings_path_1 = cli_setup_proof(model1, 1, example_data)
proof_path_1 = cli_run_proof(circuit_path_1, pk_path_1, example_data, 1)
assert cli_verify_run(vk_path_1, proof_path_1, settings_path_1) == 0


# See what the output is (we can look directly into the proof)
proof_content = json.load(open(proof_path_1))
proof_content = np.array(proof_content['pretty_public_inputs']['rescaled_outputs']).astype(np.float32)

# Prove second part
example_output_flat = torch.flatten(torch.tensor(example_output),1 ).numpy()
circuit_path_2, vk_path_2, pk_path_2, _, settings_path_2 = cli_setup_proof(model2, 2, example_output_flat)

proof_path_2 = cli_run_proof(circuit_path_2, pk_path_2, proof_content, 2)


