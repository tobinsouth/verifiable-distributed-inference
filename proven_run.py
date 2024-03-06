import ezkl, os, torch, json, numpy as np
import onnxruntime

py_run_args = ezkl.PyRunArgs()
py_run_args.input_visibility = "public"
py_run_args.output_visibility = "public"
py_run_args.param_visibility = "fixed" # "fixed" for params means that the committed to params are used for all proofs


model1 = "proofs/model_part1.onnx"
model2 = "proofs/model_part2.onnx"

def setup_proof(model_path, postfix, example_data):
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

    return circuit_path, vk_path, pk_path, example_output


def setup_proof(model_path, postfix, example_data):
    settings_path= f"proofs/settings{postfix}.json"
    data_path = f"proofs/example_input{postfix}.json"
    circuit_path = f"proofs/compiled_circuit{postfix}.ezkl"
    vk_path = f"proofs/vk{postfix}.key"
    pk_path = f"proofs/pk{postfix}.key"

    ezkl.gen_settings(model_path, settings_path, py_run_args=py_run_args)

    session = onnxruntime.InferenceSession(model_path)
    example_output = session.run(None, {"input.1": example_data})[0]

    witness_data = dict(input_shapes=[example_data.shape],
                input_data=[example_data.reshape([-1]).tolist()],
                output_data=[o.reshape([-1]).tolist() for o in example_output])
    json.dump(witness_data, open(data_path, 'w'))

    ezkl.calibrate_settings(data_path, model_path, settings_path, "accuracy")
    ezkl.get_srs(settings_path)
    ezkl.compile_circuit(model_path, circuit_path, settings_path)
    ezkl.setup(circuit_path, vk_path, pk_path)

    return circuit_path, vk_path, pk_path, example_output


# Now we can run the proof

def cli_run_proof(circuit_path, pk_path, input_data, prefix):
    proof_path = f"proofs/proof{prefix}.json"
    witness_path = f"proofs/witness{prefix}.json"

    witness_data = dict(input_shapes=[input_data.shape],
                input_data=[input_data.reshape([-1]).tolist()])
    json.dump(witness_data, open(witness_path, 'w'))

    os.system(f"ezkl gen-witness -D {witness_path} -M {circuit_path} -O {witness_path}")
    os.system(f"ezkl prove -W {witness_path} -M {circuit_path} --pk-path {pk_path} --proof-path {proof_path}")

    return proof_path


def run_proof(circuit_path, pk_path, input_data, prefix):
    proof_path = f"proofs/proof{prefix}.json"
    witness_path = f"proofs/witness{prefix}.json"

    witness_data = dict(input_shapes=[input_data.shape],
                input_data=[input_data.reshape([-1]).tolist()])
                # output_data=[o.reshape([-1]).tolist() for o in example_output])
    json.dump(witness_data, open(witness_path, 'w'))

    ezkl.gen_witness(witness_path, circuit_path, witness_path)
    ezkl.prove(witness_path, circuit_path, pk_path, proof_path)

    return proof_path




from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
])

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=1, shuffle=False)



# If that fails, try the cli version
example_data = next(iter(testloader))[0].numpy()

circuit_path_1, vk_path_1, pk_path_1, example_output = cli_setup_proof(model1, 1, example_data)
example_output_flat = torch.flatten(torch.tensor(example_output),1 ).numpy()
circuit_path_2, vk_path_2, pk_path_2, _ = cli_setup_proof(model2, 2, example_output_flat)

proof_path_1 = cli_run_proof(circuit_path_1, pk_path_1, example_data, 1)
proof_content = json.load(open(proof_path_1))
proof_content = np.array(proof_content['pretty_public_inputs']['rescaled_outputs']).astype(np.float32)

proof_path_2 = cli_run_proof(circuit_path_2, pk_path_2, proof_content, 2)

proof_content.shape