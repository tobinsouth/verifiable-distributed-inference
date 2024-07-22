# Imports
import asyncio
import json
import os.path
import sys

import ezkl
import numpy as np
import onnxruntime as ort
import torch
from torch import nn

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Constants
DEVICE = "cpu"
STORAGE_DIR = "./model_dir"
MODEL_ID = "model"


class Model(nn.Module):
    name = 'testing'

    # Input dimensions to each layer
    model_dimensions = [
        (1, 10),
        (1, 20),
        (1, 20)
    ]

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        return x


def main():
    os.makedirs(STORAGE_DIR, exist_ok=True)
    # Approach: pre-processing with random data
    model_path = f"{STORAGE_DIR}/{MODEL_ID}.onnx"
    settings_path = f"{STORAGE_DIR}/settings.json"
    calibration_data_path = f"{STORAGE_DIR}/calibration_data.json"
    data_path = f"{STORAGE_DIR}/data.json"
    compiled_model_path = f"{STORAGE_DIR}/network.compiled"
    srs_path = f"{STORAGE_DIR}/kzg.srs"
    vk_path = f"{STORAGE_DIR}/vk.key"
    pk_path = f"{STORAGE_DIR}/pk.key"
    witness_path = f"{STORAGE_DIR}/witness.json"
    proof_path = f"{STORAGE_DIR}/proof.pf"

    py_run_args = ezkl.PyRunArgs()
    py_run_args.input_visibility = "public"
    py_run_args.output_visibility = "public"
    py_run_args.param_visibility = "hashed"
    ezkl_optimization_goal = "resources"

    model = Model().to(DEVICE)

    # Generate sample tensor with dims (1, 10)
    sample_input_tensor = torch.randn(*(model.model_dimensions[0])).to(DEVICE)

    torch.onnx.export(
        model=model,
        args=sample_input_tensor,
        f=model_path,
        verbose=True,
        input_names=['input'],
        output_names=['output']
    )

    return_val = os.system(f'ezkl gen-settings '
                           f'-M {model_path} '
                           f'-O {settings_path} '
                           f'--input-visibility public '
                           f'--output-visibility public '
                           f'--param-visibility hashed')
    assert return_val == 0

    # Generate sample tensor with dims (1, 10)
    input_tensor = torch.randn(*(model.model_dimensions[0])).to(DEVICE)
    input_data = input_tensor.numpy().astype(np.float32)

    # Purposefully increasing the range of calibration data points
    calibration_data = dict(
        input_data=[
            input_data.reshape([-1]).tolist(),
            (input_data + 1).reshape([-1]).tolist(),
            (input_data - 1).reshape([-1]).tolist(),
            (input_data + 5).reshape([-1]).tolist(),
            (input_data - 5).reshape([-1]).tolist(),
        ]
    )
    with open(calibration_data_path, 'w') as f:
        json.dump(calibration_data, f)

    return_val = os.system(f'ezkl calibrate-settings '
                           f'-D {calibration_data_path} '
                           f'-M {model_path} '
                           f'--settings-path {settings_path} '
                           f'--target {ezkl_optimization_goal}')
    assert return_val == 0

    return_val = os.system(f'ezkl compile-circuit '
                           f'--model {model_path} '
                           f'--compiled-circuit {compiled_model_path} '
                           f'--settings-path {settings_path}')
    assert return_val == 0

    return_val = os.system(f'ezkl get-srs '
                           f'--srs-path {srs_path} '
                           f'--settings-path {settings_path} ')
    assert return_val == 0
    assert os.path.isfile(srs_path)

    return_val = os.system(f'ezkl setup '
                           f'--compiled-circuit {compiled_model_path} '
                           f'--srs-path {srs_path} '
                           f'--vk-path {vk_path} '
                           f'--pk-path {pk_path} ')
    assert return_val == 0
    assert os.path.isfile(vk_path)
    assert os.path.isfile(pk_path)

    # Generate witness file
    ort_session = ort.InferenceSession(model_path)
    input_tensor = torch.randn(*(model.model_dimensions[0])).to(DEVICE)
    input_data = input_tensor.numpy().astype(np.float32)
    output_data = ort_session.run(None, {'input': input_data})
    witness_data = dict(input_shapes=[input_data.shape],
                        input_data=[input_data.reshape([-1]).tolist()],
                        output_data=[o.reshape([-1]).tolist() for o in output_data])
    with open(data_path, 'w') as f:
        json.dump(witness_data, f)

    return_val = os.system(f'ezkl gen-witness '
                           f'-D {data_path} '
                           f'-O {witness_path} '
                           f'-M {compiled_model_path}')
    assert return_val == 0
    assert os.path.isfile(witness_path)

    return_val = os.system(f'RUST_BACKTRACE=full ezkl prove '
                           f'-W {witness_path} '
                           f'-M {compiled_model_path} '
                           f'--proof-path {proof_path} '
                           f'--srs-path={srs_path}  '
                           f'--pk-path={pk_path}')
    assert return_val == 0
    assert os.path.isfile(proof_path)

    os.system(f'ezkl verify '
              f'--proof-path {proof_path} '
              f'-S {settings_path} '
              f'--vk-path={vk_path}  '
              f'--srs-path={srs_path}')


if __name__ == "__main__":
    main()
