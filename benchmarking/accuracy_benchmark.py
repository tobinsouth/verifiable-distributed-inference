# Benchmark to test accuracy loss.
# Variables:
# ezkl optimization goal (resources/accuracy)
# No. of nodes we pass through (1, 2, 3, 4)
import numpy as np
import pandas as pd
from torch import nn
import torch
from modules.model_processing import Processor
from modules.model_training import Trainer
from modules.seed import set_seed
import shutil

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

STORAGE_DIR = './tmp'

def mse_loss(output, target) -> float:
    return np.sum(np.square(output - target)) / len(target)


def run_benchmark(ezkl_optimization_goal: str, num_nodes: int) -> float:
    # clear out storage dir


    # Setup trainer
    trainer = Trainer()

    # Setup processor
    processor: Processor = Processor(
        model=trainer.model,
        sample_input=torch.randn(1, 1, 28, 28).to(DEVICE)
    )
    # Shard model into `num_nodes` shards
    processor.shard(num_nodes)
    processor.save(
        model_id='model',
        storage_dir=f'{STORAGE_DIR}/shards',
    )

    shards: list = processor.shards
    shard_paths: list = processor.shard_paths

    # TODO: initialize this with input to the first layer
    prev_output = 0

    total_loss: float = 0
    for i in range(num_nodes):
        # use numpy arrays to calcuate
        # TODO feed prev_output through onnx model
        target = 0

        # TODO ezkl setup
        # use optimization goal!

        # TODO generate witness with ezkl
        output = 0

        # compare output and target
        loss: float = mse_loss(output, target)
        # add to loss, as we're calculating the cumulative loss
        total_loss += loss


    pass


if __name__ == '__main__':
    set_seed()

    df = pd.DataFrame(columns=['ezkl_optimization_goal', 'num_nodes', 'accuracy'])
    for optimization_goal in ['resources', 'accuracy']:
        for num_nodes in [1, 2, 3, 4]:
            accuracy = run_benchmark(optimization_goal, num_nodes)
            print(f'Completed benchmarking for: {optimization_goal} with {num_nodes} nodes -> {accuracy}')
            df.append(
                {
                    'ezkl_optimization_goal': optimization_goal,
                    'num_nodes': num_nodes,
                    'accuracy': accuracy
                },
                ignore_index=True
            )
    df.to_csv('./results/accuracy_benchmark.csv')
    print('Saved benchmarking results')
