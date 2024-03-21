from torch import nn
import torch


class Processor:
    # Prefix used in filepath after Model ID, but before Shard ID.
    SHARD_PATH_PREFIX = "_shard"

    # Constructor takes in (pre-trained) PyTorch NN model.
    def __init__(self, model: nn.Module):
        self.model = model
        self.shards = []

    # Shards model based on NN layers.
    def shard(self) -> None:
        num_splits = len(list(self.model.children()))
        for i in range(num_splits):
            self.shards.append(nn.Sequential(list(self.model.children())[i]))

    # Saves Model / Model Shards.
    def save(self, model_id: str, storage_dir: str) -> None:
        # If model wasn't sharded, save it as one file.
        if len(self.shards) == 0:
            model_path = f"{storage_dir}/{model_id}.onnx"
            # TODO: figure this out.
            sample_input_tensor = None
            torch.onnx.export(
                self.model,
                sample_input_tensor,
                model_path,
                verbose=False,
                input_names=['input'],
                output_names=['output']
            )
            pass
        # Save individual shards to individual files.
        else:
            # TODO: figure this out
            sample_input_tensor = None
            for i in range(len(self.shards)):
                shard_path = f"{storage_dir}/{model_id}{self.SHARD_PATH_PREFIX}_{i}.onnx"
                model_shard = self.shards[i]
                torch.onnx.export(
                    model=model_shard,
                    args=sample_input_tensor,
                    f=shard_path,
                    verbose=True,
                    input_names=['input'],
                    output_names=['output']
                )

                # Update sample input tensor for next layer/stack of layers
                model_shard.eval()
                sample_input_tensor = model_shard(sample_input_tensor)
