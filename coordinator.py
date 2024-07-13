import json
import sys
import socket
import threading
import time
from typing import Tuple

import numpy as np
import torch

from modules.connection_handler import CoordinatorConnectionHandler
from modules.witness_manager import WitnessManager
from modules.model_processing import Processor
from modules.model_training import Trainer
from modules.model_proving import Prover
from modules.file_manager import FileManager
from torch import nn
from utils.helpers import conditional_print, decode_b64_to_np_array, encode_np_array_to_b64

VERBOSE = True
MODEL_ID = "model_0"
MODEL_STORAGE_DIR = "./shared-storage/shards"
RUN_LOGIC = True


class Coordinator:
    def __init__(self,
                 address: Tuple[str, int],
                 num_shards: int,
                 benchmarking_mode: bool = False):
        self.connections: dict[str, threading.Thread] = {}
        self.handlers: dict[str, CoordinatorConnectionHandler] = {}
        self.connection_to_inbound_address: dict = {}
        self.address: Tuple[str, int] = address
        self.socket: socket = None
        self.witness_manager: WitnessManager = WitnessManager()
        self.num_shards: int = num_shards
        self.num_ready_nodes: int = 0
        self.benchmarking_mode: bool = benchmarking_mode

    def run(self) -> None:
        self.open_socket()
        trainer = Trainer()
        # Optionally train the model here.
        # trainer.train()
        # trainer.eval()

        model: nn.Module = trainer.model

        # Generate a dummy input tensor. Needed for the processing of the model.
        dummy_input: torch.Tensor = trainer.get_dummy_input()

        conditional_print("[PREPROCESSING] Loaded model", VERBOSE)

        # Process the model and turn it into shards.
        model_processor = Processor(
            model=model,
            sample_input=dummy_input
        )
        model_processor.shard(self.num_shards)

        conditional_print("[PREPROCESSING] Sharded model", VERBOSE)

        model_processor.save(
            model_id=MODEL_ID,
            storage_dir=MODEL_STORAGE_DIR
        )

        conditional_print("[PREPROCESSING] Saved model (shards)", VERBOSE)

        connection_counter: int = 0

        conditional_print("[LOGIC] Starting connection loop", VERBOSE)
        try:

            while connection_counter < self.num_shards:
                socket_connection, socket_address = self.socket.accept()
                conditional_print(f"[LOGIC] Accepted connection from: {socket_address}", VERBOSE)
                if connection_counter > self.num_shards - 1:
                    print(f"[ISSUE] There are more connections than available shards: {self.num_shards} shards are taken."
                          f" Shutting down connection.")
                    socket_connection.close()
                    continue
                # Not needed anymore, as we make file retrieval based on `shard_id` and `model_id`.
                # shard_path: str = model_processor.shard_paths[connection_counter]
                conn_handler = CoordinatorConnectionHandler(
                    connection=socket_connection,
                    address=socket_address,
                    initiating_node=self,
                    shard_id=connection_counter,
                    model_id=MODEL_ID
                )
                conn_thread = threading.Thread(
                    target=conn_handler.run
                )
                conn_key = f"{socket_address[0]}:{socket_address[1]}"
                # Store the connection thread
                self.connections[conn_key] = conn_thread
                # Store connection handler. Required to later send messages to each node using its .send() method.
                self.handlers[conn_key] = conn_handler
                conn_thread.start()
                connection_counter += 1

            # Loops until all nodes have sent a ready message
            while self.num_ready_nodes != self.num_shards:
                conditional_print(f'[LOGIC] Not all nodes ready to process inference request(s). '
                                  f'Retrying in 5s.', VERBOSE)
                time.sleep(5)

            print('[LOGIC] Setup completed. Inference runs can now be served.')
            first_node_handler: CoordinatorConnectionHandler = list(self.handlers.values())[0]
            second_node_handler: CoordinatorConnectionHandler = list(self.handlers.values())[1]
            np_arr = Trainer.get_dummy_input().cpu().numpy()
            encoded_np_arr = encode_np_array_to_b64(np_arr)
            message: bytes = b'run_inference|' + encoded_np_arr
            first_node_handler.send_bytes(message)

            time.sleep(10)

            first_node_handler.send(f'get_proof|1ca6dd0091304ca9b985b615a4998eaa_0')
            second_node_handler.send(f'get_proof|3f1f607fdd6f485588c9ea7533c2c136_0')

            if self.benchmarking_mode:
                for handler in list(self.handlers.values()):
                    handler.send('save_benchmarking_results')

        except KeyboardInterrupt:
            self.socket.close()
            for thread in self.connections.values():
                thread.join()
            sys.exit(0)
        finally:
            self.socket.close()
            for thread in self.connections.values():
                thread.join()

    def open_socket(self) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        self.socket.bind(address)
        self.socket.listen()

    # Intermediate function that adds a witness_id to the witness_manager.
    def store_witness(self, witness_id: str, shard_id: int) -> None:
        self.witness_manager.add_witness(witness_id, shard_id)

    # Requests a proof for a specific witness_id.
    def request_single_proof(self, witness_id: str) -> None:
        shard_id = self.witness_manager.get_shard_by_witness_id(witness_id)
        for handler in self.handlers.values():
            if handler.shard_id == shard_id:
                handler.send(f'get_proof|{witness_id}')
                pass

    # Requests proofs from all nodes for one specific run_id.
    # To be used when we can't pinpoint which node is running the model incorrectly.
    def request_inference_run_proof(self, run_id: int) -> None:
        # Pull in all witness_ids for a specific run
        witness_id_list: list[str] = self.witness_manager.get_witnesses_by_run_id(run_id)

        # Request proof from all nodes
        for witness_id in witness_id_list:
            witness_shard_id = self.witness_manager.get_shard_by_witness_id(witness_id)
            for handler in self.handlers.values():
                if handler.shard_id == witness_shard_id:
                    handler.send(f'get_proof|{witness_id}')

    # Intermediate function that verifies the proof
    def verify_proof(self, proof_path: str, shard_id: int, model_id: str) -> None:
        # Only the proof_path is passed, and the other two paths are derived, as they stay static
        # for all future proofs.
        Prover.verify_proof(
            proof_path=proof_path,
            settings_path=FileManager.get_settings_path_static(shard_id, model_id),
            vk_path=FileManager.get_vk_path_static(shard_id, model_id)
        )

    # Saves the final inference output to a file
    def save_final_inference_output(self, raw_output_data: bytes, run_id: int) -> None:
        file_path: str = FileManager.get_final_output_path(MODEL_ID, run_id)
        output_array: np.ndarray = decode_b64_to_np_array(raw_output_data)
        raw_output_file_data: dict = dict(
            output_shapes=[output_array.shape],
            output_data=[o.reshape([-1]).tolist() for o in output_array]
        )
        with open(file_path, 'w') as f:
            json.dump(raw_output_file_data, f)
            f.close()
        conditional_print(f'Wrote final inference output to: {file_path}', VERBOSE)

    # Adds 1 to the number of ready nodes.
    def register_ready_node(self) -> None:
        self.num_ready_nodes += 1


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f'Usage: coordinator.py <host> <port> <num_shards>')
        sys.exit(0)

    address = (sys.argv[1], int(sys.argv[2]))
    num_shards = int(sys.argv[3])
    coordinator = Coordinator(
        address=address,
        num_shards=num_shards
    )
    coordinator.run()
