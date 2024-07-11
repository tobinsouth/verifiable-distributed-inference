import math
import os

from torch import nn
import torch
from utils.helpers import conditional_print

VERBOSE = True


class Processor:
    # Prefix used in filepath after Model ID, but before Shard ID.
    SHARD_PATH_PREFIX = "_shard"

    # Constructor takes in (pre-trained) PyTorch NN model.
    def __init__(self, model: nn.Module, sample_input: torch.Tensor):
        self.model = model
        self.sample_input = sample_input
        self.shards = []
        # Mapping of paths to model (shards)
        self.shard_paths = []

    # Shards model based on NN layers.
    def shard(self, num_shards: int) -> None:
        children_list: list = list(self.model.children())
        num_layers: int = len(children_list)
        if num_layers % num_shards != 0:
            print(f'[WARNING] Invalid number of shards: {num_layers} is not a multiple of {num_shards}!')
            return
        group_size: int = math.ceil(num_layers / num_shards)
        for i in range(0, num_layers, group_size):
            self.shards.append(
                nn.Sequential(*children_list[i:i + group_size])
            )
        conditional_print(f'[PREPROCESSING] Split model into {len(self.shards)} shards.', VERBOSE)

    # Saves Model / Model Shards.
    def save(self, model_id: str, storage_dir: str) -> None:
        os.makedirs(storage_dir, exist_ok=True)
        # If model wasn't sharded, save it as one file.
        if len(self.shards) == 0:
            model_path: str = f"{storage_dir}/{model_id}.onnx"
            self.shard_paths.append(model_path)
            sample_input_tensor = self.sample_input
            torch.onnx.export(
                model=self.model,
                args=sample_input_tensor,
                f=model_path,
                verbose=VERBOSE,
                input_names=['input'],
                output_names=['output']
            )
            conditional_print(f'[PREPROCESSING] Saved model at {model_path}.', VERBOSE)
        # Save individual shards to individual files.
        else:
            sample_input_tensor = self.sample_input
            for i in range(len(self.shards)):
                shard_path: str = f"{storage_dir}/{model_id}{self.SHARD_PATH_PREFIX}_{i}.onnx"
                model_shard: nn.Module = self.shards[i]
                self.shard_paths.append(shard_path)
                torch.onnx.export(
                    model=model_shard,
                    args=sample_input_tensor,
                    f=shard_path,
                    verbose=VERBOSE,
                    input_names=['input'],
                    output_names=['output']
                )
                conditional_print(f'[PREPROCESSING] Saved shard at {shard_path}.', VERBOSE)
                # Update sample input tensor for next layer/stack of layers
                model_shard.eval()
                sample_input_tensor = model_shard(sample_input_tensor)
