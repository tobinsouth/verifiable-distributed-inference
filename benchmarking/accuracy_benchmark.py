# Benchmark to test accuracy loss.
# Variables:
# ezkl optimization goal (resources/accuracy)
# No. of nodes we pass through (1, 2, 3, 4)
import asyncio
import json
import os
import sys
import shutil
import time
import uuid
from typing import Tuple

import ezkl
import numpy as np
import pandas as pd
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules.model_processing import Processor
from modules.model_training import Trainer, AVAILABLE_MODELS
from modules.seed import set_seed
from config import USE_EZKL_CLI, DEVICE
from utils.helpers import rmse, rmspe

STORAGE_DIR = './tmp'
RESULTS_DIR = './results'

async def setup(model_path: str,
                settings_path: str,
                calibration_data_path: str,
                compiled_model_path: str,
                srs_path: str,
                vk_path: str,
                pk_path: str,
                ezkl_optimization_goal: str) -> None:
    py_run_args = ezkl.PyRunArgs()
    py_run_args.input_visibility = "public"
    py_run_args.output_visibility = "public"
    py_run_args.param_visibility = "hashed"

    # gen_settings
    if USE_EZKL_CLI:
        os.system(f'ezkl gen-settings '
                  f'-M {model_path} '
                  f'-O {settings_path} '
                  f'--input-visibility {py_run_args.input_visibility} '
                  f'--output-visibility {py_run_args.output_visibility} '
                  f'--param-visibility {py_run_args.param_visibility} ')
    else:
        result_gen_settings = ezkl.gen_settings(
            model=model_path,
            output=settings_path,
            py_run_args=py_run_args
        )

    # calibrate_settings
    if USE_EZKL_CLI:
        os.system(f'ezkl calibrate-settings '
                  f'-D {calibration_data_path} '
                  f'-M {model_path} '
                  f'--settings-path {settings_path} '
                  f'--target {ezkl_optimization_goal} ')
    else:
        result_calibrate_settings = await ezkl.calibrate_settings(
            data=calibration_data_path,
            model=model_path,
            settings=settings_path,
            target=ezkl_optimization_goal
        )

    # compile_circuit
    if USE_EZKL_CLI:
        os.system(f'ezkl compile-circuit '
                  f'--model {model_path} '
                  f'--compiled-circuit {compiled_model_path} '
                  f'--settings-path {settings_path}')
    else:
        result_compile = ezkl.compile_circuit(
            model=model_path,
            compiled_circuit=compiled_model_path,
            settings_path=settings_path
        )

    # get_srs
    if USE_EZKL_CLI:
        os.system(f'ezkl get-srs '
                  f'--srs-path {srs_path} '
                  f'--settings-path {settings_path} ')
    else:
        result_srs = await ezkl.get_srs(
            settings_path=settings_path,
            srs_path=srs_path
        )

    # setup
    if USE_EZKL_CLI:
        os.system(f'ezkl setup '
                  f'--compiled-circuit {compiled_model_path} '
                  f'--srs-path {srs_path} '
                  f'--vk-path {vk_path} '
                  f'--pk-path {pk_path} ')
    else:
        result_setup = ezkl.setup(
            model=compiled_model_path,
            vk_path=vk_path,
            pk_path=pk_path,
            srs_path=srs_path
        )


async def gen_witness(witness_path: str, raw_witness_path: str, compiled_model_path: str) -> None:
    # gen-witness
    if USE_EZKL_CLI:
        os.system(f'ezkl gen-witness '
                  f'-D {raw_witness_path} '
                  f'-O {witness_path} '
                  f'-M {compiled_model_path}')
    else:
        witness_result = await ezkl.gen_witness(
            data=raw_witness_path,
            model=compiled_model_path,
            output=witness_path
        )


def run_benchmark(ezkl_optimization_goal: str, num_nodes: int, model_name: str) -> float:
    model_id: str = 'model'
    # clear out storage dir
    shutil.rmtree(STORAGE_DIR)
    os.mkdir(STORAGE_DIR)

    # Setup trainer
    trainer = Trainer(
        load_training_data=False,
        model_name=model_name
    )
    model = trainer.model

    # Setup processor
    processor: Processor = Processor(
        model=model,
        sample_input=torch.randn(*model.model_dimensions[0]).to(DEVICE)
    )
    # Shard model into `num_nodes` shards
    processor.shard(num_nodes)

    processor.save(
        model_id=model_id,
        storage_dir=f'{STORAGE_DIR}/shards',
    )

    shards: list = processor.shards
    shard_paths: list = processor.shard_paths
    shard_dimensions: list = processor.shard_dimensions

    prev_output = torch.randn(*shard_dimensions[0]).to(DEVICE)

    # Generate a 5 random calibration datapoints
    # calibration_data: list[np.ndarray] = [np.random.rand(*shard_dimensions[0]).astype(np.float32).flatten().tolist()
    #                                       for _ in range(5)]
    calibration_data: list[torch.Tensor] = [torch.rand(*shard_dimensions[0]) for _ in range(5)]

    total_loss: float = 0
    for i in range(num_nodes):
        # Define filepaths
        model_path = shard_paths[i]
        settings_path = f"{STORAGE_DIR}/{model_id}_shard_{i}_settings.json"
        calibration_data_path = f"{STORAGE_DIR}/{model_id}_shard_{i}_calibration_data.json"
        raw_witness_path = f"{STORAGE_DIR}/{model_id}_shard_{i}_witness.json"
        compiled_model_path = f"{STORAGE_DIR}/{model_id}_shard_{i}_network.compiled"
        srs_path = f"{STORAGE_DIR}/{model_id}_shard_{i}_kzg.srs"
        vk_path = f"{STORAGE_DIR}/{model_id}_shard_{i}_vk.key"
        pk_path = f"{STORAGE_DIR}/{model_id}_shard_{i}_pk.key"
        witness_path = f"{STORAGE_DIR}/{model_id}_shard_{i}_witness.json"

        calibration_file_data = []
        for j in range(len(calibration_data)):
            try:
                np_arr = calibration_data[j].to('cpu').numpy().astype(np.float32)
            except Exception:
                np_arr = calibration_data[j].to('cpu').detach().numpy().astype(np.float32)
            calibration_file_data.append(np_arr.flatten().tolist())
            calibration_data[j] = shards[i](calibration_data[j])

        cal_data = dict(
            input_data=calibration_file_data
        )
        json.dump(cal_data, open(calibration_data_path, 'w'))

        # Generate witness datapoint

        # feed input data through model shard
        tensor_output = shards[i](prev_output).to(DEVICE)
        # convert tensor to numpy array
        try:
            target = tensor_output.to(DEVICE).numpy().astype(np.float32)
            witness_input = prev_output.numpy().astype(np.float32)
        except RuntimeError as e:
            target = tensor_output.to(DEVICE).detach().numpy().astype(np.float32)
            witness_input = prev_output.detach().numpy().astype(np.float32)

        # Write out (raw) witness data
        witness_data = dict(
            input_shapes=[witness_input.shape],
            input_data=[witness_input.reshape([-1]).tolist()],
            output_data=[o.reshape([-1]).tolist() for o in target]
        )
        json.dump(witness_data, open(raw_witness_path, 'w'))

        asyncio.run(setup(
            model_path=model_path,
            settings_path=settings_path,
            calibration_data_path=calibration_data_path,
            compiled_model_path=compiled_model_path,
            srs_path=srs_path,
            vk_path=vk_path,
            pk_path=pk_path,
            ezkl_optimization_goal=ezkl_optimization_goal
        ))

        asyncio.run(gen_witness(
            witness_path=witness_path,
            raw_witness_path=raw_witness_path,
            compiled_model_path=compiled_model_path,
        ))

        witness_data = open(witness_path, 'r')
        json_data = json.load(witness_data)
        output = json_data['pretty_elements']['rescaled_outputs'][0]
        output = np.asarray(output)

        # compare output and target
        loss: float = rmse(output, target)
        print(f'Shard results: RMSE: {rmse(output, target)}, RMSPE: {rmspe(output, target)}')
        # add to loss, as we're calculating the cumulative loss
        total_loss += loss

        # Update tensor for following shard
        prev_output = tensor_output

    return total_loss


def run_benchmark_with_existing_artifacts(num_nodes: int, model_name: str) -> float:
    # TODO: figure out why this generates very high loss values!
    setup_id: str = f'{model_name}-{num_nodes}'
    if not os.path.exists(f'{STORAGE_DIR}/{setup_id}'):
        print(f'PATH {STORAGE_DIR}/{setup_id} doesn\'t exist!, skipping configuration!')

    total_loss: float = 0

    # Setup trainer
    trainer = Trainer(
        load_training_data=False,
        model_name=model_name
    )
    model = trainer.model

    # Setup processor
    processor: Processor = Processor(
        model=model,
        sample_input=torch.randn(*model.model_dimensions[0]).to(DEVICE)
    )
    # Shard model into `num_nodes` shards
    processor.shard(num_nodes)

    shards: list = processor.shards
    shard_dimensions: list = processor.shard_dimensions

    prev_output = torch.randn(*shard_dimensions[0]).to(DEVICE)
    for i in range(num_nodes):
        witness_id: str = f"acc_bench_{uuid.uuid4().hex}"

        # Define filepaths
        raw_witness_path = f"{STORAGE_DIR}/{setup_id}/shard_{i}/{witness_id}_raw.json"
        witness_path = f"{STORAGE_DIR}/{setup_id}/shard_{i}/{witness_id}.json"
        compiled_model_path = f"{STORAGE_DIR}/{setup_id}/shard_{i}/{model_name}_network.compiled"

        tensor_output = shards[i](prev_output).to(DEVICE)
        # convert tensor to numpy array
        try:
            target = tensor_output.to(DEVICE).numpy().astype(np.float32)
            witness_input = prev_output.numpy().astype(np.float32)
        except RuntimeError as e:
            target = tensor_output.to(DEVICE).detach().numpy().astype(np.float32)
            witness_input = prev_output.detach().numpy().astype(np.float32)

        # Write out (raw) witness data
        witness_data = dict(
            input_shapes=[witness_input.shape],
            input_data=[witness_input.reshape([-1]).tolist()],
            output_data=[o.reshape([-1]).tolist() for o in target]
        )
        json.dump(witness_data, open(raw_witness_path, 'w'))

        # Generate witness with existing artifacts
        asyncio.run(gen_witness(
            witness_path=witness_path,
            raw_witness_path=raw_witness_path,
            compiled_model_path=compiled_model_path,
        ))

        witness_data = open(witness_path, 'r')
        json_data = json.load(witness_data)
        output = json_data['pretty_elements']['rescaled_outputs'][0]
        output = np.asarray(output)

        # compare output and target
        loss: float = rmse(output, target)
        print(f'Shard results: RMSE: {rmse(output, target)}, RMSPE: {rmspe(output, target)}')
        # add to loss, as we're calculating the cumulative loss
        total_loss += loss

        # Update tensor for following shard
        prev_output = tensor_output

    return total_loss


if __name__ == '__main__':
    # Example usage:
    # python accuracy_benchmark.py mlp ./tmp-mlp
    # python accuracy_benchmark.py cnn ./tmp-cnn
    # python accuracy_benchmark.py mlp2 ./tmp-system-benchmark true
    # python accuracy_benchmark.py mlp2 ./tmp-system-benchmark true 4

    if len(sys.argv) < 2:
        print("Invalid usage!")
        print(f'Usage: accuracy_benchmark <model> [storage_dir] [use_existing_artifacts] [num_nodes]')
        print(f'Available models are: {", ".join(AVAILABLE_MODELS)}')
        sys.exit(1)

    model_name: str = sys.argv[1]
    if model_name not in AVAILABLE_MODELS:
        print(f'Incorrect model value! Available models are: {", ".join(AVAILABLE_MODELS)}')
        sys.exit(1)

    if len(sys.argv) == 3:
        STORAGE_DIR = sys.argv[2]

    use_existing_artifacts: bool = False
    if len(sys.argv) == 4:
        STORAGE_DIR = sys.argv[2]
        use_existing_artifacts = sys.argv[3] == 'true'

    num_node_list: list[int] = []
    default_num_node_list: list[int] = [1, 2, 3, 4, 6, 12]
    if len(sys.argv) == 5:
        STORAGE_DIR = sys.argv[2]
        use_existing_artifacts = sys.argv[3] == 'true'
        num_nodes = int(sys.argv[4])
        if num_nodes not in default_num_node_list:
            print(f'Invalid value for num_nodes: {num_nodes} (must be between any one of: {default_num_node_list})')
            sys.exit(1)
        else:
            num_node_list.append(num_nodes)
    else:
        num_node_list = default_num_node_list

    set_seed()
    os.makedirs(STORAGE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Open the CSV file for appending
    csv_file_path = f'{RESULTS_DIR}/accuracy_benchmark_{time.time_ns()}.csv'
    with open(csv_file_path, 'a') as csv_file:
        csv_file.write('ezkl_optimization_goal,num_nodes,accuracy_loss,reference_accuracy_loss,model\n')

    # There's an option here to add 'accuracy' as optimization goal. Runtimes increase DRASTICALLY.
    for optimization_goal in ['resources']:
        for num_nodes in num_node_list:
            print(f'Running config for: Model {model_name} with {optimization_goal} goal and {num_nodes} nodes')
            if use_existing_artifacts:
                accuracy_loss = run_benchmark_with_existing_artifacts(
                    num_nodes=num_nodes,
                    model_name=model_name
                )
            else:
                accuracy_loss = run_benchmark(
                    ezkl_optimization_goal=optimization_goal,
                    num_nodes=num_nodes,
                    model_name=model_name
                )
            print(f'Completed benchmarking for: {optimization_goal} with {num_nodes} nodes -> {accuracy_loss}')
            row = {
                'ezkl_optimization_goal': optimization_goal,
                'num_nodes': num_nodes,
                'accuracy_loss': accuracy_loss,
                'reference_accuracy_loss': 0,
                'model': model_name
            }
            print(row)

            # Write the row to the CSV file immediately
            with open(csv_file_path, 'a') as csv_file:
                csv_file.write(f"{optimization_goal},{num_nodes},{accuracy_loss},0,{model_name}\n")

    print('Saved ALL benchmarking results')
