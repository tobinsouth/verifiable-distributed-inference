import json
import sys
import socket
import threading
import time
from typing import Tuple
import numpy as np
import pandas as pd
import torch
from torch import nn

from modules.connection_handler import CoordinatorConnectionHandler
from modules.witness_manager import WitnessManager
from modules.model_processing import Processor
from modules.model_training import Trainer, AVAILABLE_MODELS
from modules.model_proving import Prover
from modules.file_manager import FileManager
from utils.helpers import conditional_print, decode_b64_to_np_array, encode_np_array_to_b64, save_dataframe
from config import STORAGE_DIR, VERBOSE, NUM_CALIBRATION_DATAPOINTS, DEVICE


class Coordinator:
    def __init__(self,
                 address: Tuple[str, int],
                 num_shards: int,
                 model_name: str,
                 benchmarking_mode: bool = False,
                 storage_dir: str = STORAGE_DIR):
        self.trainer: Trainer = None
        self.model_processor: Processor = None
        self.connections: dict[str, threading.Thread] = {}
        self.handlers: dict[str, CoordinatorConnectionHandler] = {}
        self.connection_to_inbound_address: dict = {}
        self.address: Tuple[str, int] = address
        self.socket: socket = None
        self.witness_manager: WitnessManager = WitnessManager()
        self.num_shards: int = num_shards
        self.num_ready_nodes: int = 0
        self.model_name: str = model_name
        self.benchmarking_mode: bool = benchmarking_mode
        self.storage_dir: str = storage_dir
        self.verification_data: list = []

    def run(self) -> None:
        self.open_socket()
        self.trainer = Trainer(
            load_training_data=False,
            model_name=self.model_name
        )

        model: nn.Module = self.trainer.model

        # Generate a dummy input tensor. Needed for the processing of the model.
        dummy_input: torch.Tensor = self.trainer.get_dummy_input()

        conditional_print(f"[PREPROCESSING] Loaded model {self.model_name}", VERBOSE)

        # Process the model and turn it into shards.
        self.model_processor = Processor(
            model=model,
            sample_input=dummy_input
        )
        self.model_processor.shard(self.num_shards)

        conditional_print("[PREPROCESSING] Sharded model", VERBOSE)

        self.model_processor.save(
            model_id=self.model_name,
            storage_dir=FileManager.get_model_storage_dir_static(self.storage_dir)
        )

        conditional_print("[PREPROCESSING] Saved model (shards)", VERBOSE)

        # Generate calibration data
        dummy_values = [self.trainer.get_dummy_input() for _ in range(NUM_CALIBRATION_DATAPOINTS)]
        model_shards = self.model_processor.shards

        for i in range(num_shards):
            cal_data = dict(
                input_data=[]
            )
            file_path: str = FileManager.get_calibration_data_path_static(
                shard_id=i,
                model_id=self.model_name,
                storage_dir=self.storage_dir
            )

            for j in range(len(dummy_values)):
                try:
                    np_arr = dummy_values[j].to(DEVICE).numpy().astype(np.float32)
                except Exception:
                    np_arr = dummy_values[j].to(DEVICE).detach().numpy().astype(np.float32)
                # Add np arrays to dict
                cal_data['input_data'].append(np_arr.flatten().tolist())
                # Update tensors
                dummy_values[j] = model_shards[i](dummy_values[j])

            with open(file_path, 'w') as f:
                json.dump(cal_data, f)

        connection_counter: int = 0

        conditional_print("[LOGIC] Starting connection loop", VERBOSE)
        try:
            while connection_counter < self.num_shards:
                socket_connection, socket_address = self.socket.accept()
                conditional_print(f"[LOGIC] Accepted connection from: {socket_address}", VERBOSE)
                if connection_counter > self.num_shards - 1:
                    print(f"[ERROR] There are more connections than available shards: {self.num_shards} shards are taken."
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
                    model_id=self.model_name
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
                                  f'Retrying in 60s.', VERBOSE)
                time.sleep(60)

            print('[LOGIC] Setup completed. Inference runs can now be served.')

            if self.benchmarking_mode:
                sleep_time_seconds: int = 10 if self.model_name == 'mlp' else 60*3
                handler_list: list[CoordinatorConnectionHandler] = list(self.handlers.values())
                # The first node will receive all inference requests from the coordinator
                first_node_handler: CoordinatorConnectionHandler = handler_list[0]

                # Get random input to send to first worker
                np_arr = self.trainer.get_dummy_input().cpu().numpy()
                encoded_np_arr = encode_np_array_to_b64(np_arr)
                message: bytes = b'run_inference|' + encoded_np_arr
                first_node_handler.send_bytes(message)

                while len(self.witness_manager.witness_to_shard_map) < self.num_shards:
                    conditional_print(
                        f'[LOGIC] Still waiting for all witnesses to be generated. Retrying in {sleep_time_seconds}s!',
                        VERBOSE)
                    time.sleep(sleep_time_seconds)

                all_witnesses: list = list(self.witness_manager.witness_to_shard_map.keys())
                # Get all proofs
                for witness in all_witnesses:
                    shard_id: int = self.witness_manager.witness_to_shard_map[witness]
                    connection_handler: CoordinatorConnectionHandler = handler_list[shard_id]
                    connection_handler.send(f'get_proof|{witness}')

                while set(self.witness_manager.verified_witnesses) != set(all_witnesses):
                    conditional_print(f'[LOGIC] Still waiting for all proofs to be received. Retrying in {sleep_time_seconds}s!', VERBOSE)
                    time.sleep(sleep_time_seconds)

                # Triggers all workers to save their benchmarking results
                for handler in handler_list:
                    handler.send('save_benchmarking_results')
                    time.sleep(sleep_time_seconds)
                    # After nodes have been saved, trigger shutdown
                    # (this is important, as it properly frees up ports etc.)
                    handler.send('shutdown')
                    time.sleep(sleep_time_seconds)

                self.save_benchmarking_results()
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
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
            self.socket.bind(address)
            self.socket.listen()
            conditional_print(f'[LOGIC] Coordinator socket is listening')
            time.sleep(5)
        except Exception as e:
            print('ERROR OPENING SOCKET', e)


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

        # Starts measuring time
        start_time: float = 0
        if self.benchmarking_mode:
            start_time = time.perf_counter()

        # Check if proof is a valid proof
        Prover.verify_proof(
            proof_path=proof_path,
            settings_path=FileManager.get_settings_path_static(shard_id, model_id, self.storage_dir),
            vk_path=FileManager.get_vk_path_static(shard_id, model_id, self.storage_dir),
            srs_path=FileManager.get_srs_path_static(shard_id, model_id, self.storage_dir)
        )

        # Stops measuring time & logs time difference
        end_time: float = 0
        if self.benchmarking_mode:
            end_time = time.perf_counter()
            difference: float = end_time - start_time
            self.verification_data.append(
                {
                    'shard_id': shard_id,
                    'model_id': model_id,
                    'verification_time': difference
                }
            )

        tokens = proof_path.split('/')
        tokens = tokens[-1].split('_proof')
        witness_id_of_proof: str = tokens[0]
        self.witness_manager.verified_witnesses.append(witness_id_of_proof)


    # Saves the final inference output to a file
    def save_final_inference_output(self, raw_output_data: bytes, run_id: int) -> None:
        file_path: str = FileManager.get_final_output_path(self.model_name, run_id, self.storage_dir)
        output_array: np.ndarray = decode_b64_to_np_array(raw_output_data)
        raw_output_file_data: dict = dict(
            output_shapes=[output_array.shape],
            output_data=[o.reshape([-1]).tolist() for o in output_array]
        )
        with open(file_path, 'w') as f:
            json.dump(raw_output_file_data, f)
            f.close()
        conditional_print(f'[LOGIC] Wrote final inference output to: {file_path}', VERBOSE)

    # Adds 1 to the number of ready nodes.
    def register_ready_node(self) -> None:
        self.num_ready_nodes += 1

    # Persists logged benchmarking results
    def save_benchmarking_results(self) -> None:
        if not self.benchmarking_mode:
            return

        df_verification = pd.DataFrame(self.verification_data)

        data_dir: str = FileManager.get_benchmarking_results_dir_static(self.storage_dir)
        verification_file: str = f'{data_dir}/verification_data.csv'

        save_dataframe(
            file_path=verification_file,
            df=df_verification
        )


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(f'Usage: coordinator.py <host> <port> <num_shards> <model> [benchmarking_mode] [storage_dir]')
        sys.exit(1)

    address = (sys.argv[1], int(sys.argv[2]))
    num_shards = int(sys.argv[3])
    model_name = sys.argv[4]

    if model_name not in AVAILABLE_MODELS:
        print(f'Incorrect model value! Available models are: {", ".join(AVAILABLE_MODELS)}')
        sys.exit(1)

    coordinator = None

    if len(sys.argv) == 5:
        coordinator = Coordinator(
            address=address,
            num_shards=num_shards,
            model_name=model_name
        )

    benchmarking_mode = False
    if len(sys.argv) == 6:
        if sys.argv[5] == 'true':
            benchmarking_mode = True
        elif sys.argv[5] == 'false':
            benchmarking_mode = False
        else:
            print(f'Incorrect benchmarking_mode value! Options are: true, false')
            sys.exit(1)

        coordinator = Coordinator(
            address=address,
            num_shards=num_shards,
            model_name=model_name,
            benchmarking_mode=benchmarking_mode
        )

    storage_dir: str = sys.argv[6]
    if len(sys.argv) == 7:
        if sys.argv[5] == 'true':
            benchmarking_mode = True
        elif sys.argv[5] == 'false':
            benchmarking_mode = False
        else:
            print(f'Incorrect benchmarking_mode value! Options are: true, false')
            sys.exit(1)

        if storage_dir == "":
            print(f'Incorrect storage_dir value!')
            sys.exit(1)

        coordinator = Coordinator(
            address=address,
            num_shards=num_shards,
            model_name=model_name,
            benchmarking_mode=benchmarking_mode,
            storage_dir=storage_dir
        )

    if len(sys.argv) > 7:
        print(f'Too many arguments!')
        sys.exit(1)

    coordinator.run()
