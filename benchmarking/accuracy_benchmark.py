# Benchmark to test accuracy loss.
# Variables:
# ezkl optimization goal (resources/accuracy)
# No. of nodes we pass through (1, 2, 3, 4)
import asyncio
import json
import os
import shutil

import ezkl
import numpy as np
import pandas as pd
import torch

from modules.model_processing import Processor
from modules.model_training import Trainer
from modules.seed import set_seed

DEVICE = "cpu"

STORAGE_DIR = './tmp'
RESULTS_DIR = './results'

def rmse(y_pred, y_true):
    y_pred = y_pred.astype(np.float32).flatten()
    y_true = y_true.astype(np.float32).flatten()
    return np.sqrt(np.mean(np.square(y_true - y_pred)))


def rmspe(y_pred, y_true, epsilon = 1e-10) -> float:
    y_pred = y_pred.astype(np.float32).flatten()
    y_true = y_true.astype(np.float32).flatten()

    errors = (y_true - y_pred) / (y_true + epsilon)
    return np.sqrt(np.mean(np.square(errors)))



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
    result_gen_settings = ezkl.gen_settings(
        model=model_path,
        output=settings_path,
        py_run_args=py_run_args
    )

    # calibrate_settings
    result_calibrate_settings = await ezkl.calibrate_settings(
        data=calibration_data_path,
        model=model_path,
        settings=settings_path,
        target=ezkl_optimization_goal
    )

    # compile_circuit
    result_compile = ezkl.compile_circuit(
        model=model_path,
        compiled_circuit=compiled_model_path,
        settings_path=settings_path
    )

    # get_srs
    result_srs = await ezkl.get_srs(
        settings_path=settings_path,
        srs_path=srs_path
    )

    # setup
    result_setup = ezkl.setup(
        model=compiled_model_path,
        vk_path=vk_path,
        pk_path=pk_path,
        srs_path=srs_path
    )


async def gen_witness(witness_path: str, raw_witness_path: str, compiled_model_path: str) -> None:
    # gen-witness
    witness_result = await ezkl.gen_witness(
        data=raw_witness_path,
        model=compiled_model_path,
        output=witness_path
    )


def run_benchmark(ezkl_optimization_goal: str, num_nodes: int) -> float:
    model_id: str = 'model'
    # clear out storage dir
    shutil.rmtree(STORAGE_DIR)
    os.mkdir(STORAGE_DIR)

    # Setup trainer
    trainer = Trainer(
        load_training_data=False
    )

    # Setup processor
    processor: Processor = Processor(
        model=trainer.model,
        sample_input=torch.randn(1, 1, 5, 5).to(DEVICE)
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

        # Generate a 5 random calibration datapoints
        calibration_data: list[np.ndarray] = [np.random.rand(*shard_dimensions[i]).astype(np.float32).flatten().tolist()
                                              for _ in range(5)]

        cal_data = dict(
            input_data=calibration_data
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
        print(f'!!!!!RMSE: {rmse(output, target)}, RMSPE: {rmspe(output, target)}')
        # add to loss, as we're calculating the cumulative loss
        total_loss += loss

        # Update tensor for following shard
        prev_output = tensor_output

    return total_loss


if __name__ == '__main__':
    set_seed()

    rows = []
    # TODO: add 'accuracy' back in
    for optimization_goal in ['resources']:
        # TODO: add 4 back in when model is adjusted
        for num_nodes in [1, 2, 3]:
            accuracy_loss = run_benchmark(optimization_goal, num_nodes)
            print(f'Completed benchmarking for: {optimization_goal} with {num_nodes} nodes -> {accuracy_loss}')
            row = {
                'ezkl_optimization_goal': optimization_goal,
                'num_nodes': num_nodes,
                'accuracy_loss': accuracy_loss,
                'reference_accuracy_loss': 0,  # Reference value for the 'ideal' loss value
            }
            rows.append(row)
            print(row)

    df = pd.DataFrame(rows)
    df.to_csv(f'{RESULTS_DIR}/accuracy_benchmark.csv')
    print('Saved benchmarking results')

