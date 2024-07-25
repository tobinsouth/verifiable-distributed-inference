import asyncio
import os
import sys
import socket
import threading
import time
import uuid
import json

import numpy as np
import pandas as pd
import onnxruntime as ort
from typing import Tuple
from modules.connection_handler import CoordinatorConnectionHandler, WorkerConnectionHandler
from modules.model_proving import Prover
from modules.file_manager import FileManager
from utils.helpers import conditional_print, decode_b64_to_np_array, encode_np_array_to_b64, save_dataframe
from config import STORAGE_DIR, VERBOSE, INPUT_VISIBILITY, OUTPUT_VISIBILITY, PARAM_VISIBILITY, OPTIMIZATION_GOAL, \
    INPUT_SCALE, PARAM_SCALE


class Worker:
    def __init__(self,
                 address: Tuple[str, int],
                 coordinator_address: Tuple[str, int],
                 node_role: str,
                 benchmarking_mode: bool = False,
                 storage_dir: str = STORAGE_DIR):
        self.shard_id: int = None
        self.model_id: str = None
        self.storage_dir: str = storage_dir

        # Switch to log/time results to be used for benchmarking results
        self.benchmarking_mode: bool = benchmarking_mode
        self.witness_data: list = []
        self.proving_data: list = []
        self.setup_data: list = []

        # Address other nodes use to connect to worker
        self.address: Tuple[str, int] = address

        # "Outbound" connection to coordinator
        self.coordinator_address: Tuple[str, int] = coordinator_address
        self.coordinator_socket: socket = None
        self.coordinator_conn_handler = None
        self.conn_coordinator_thread = None

        # "Outbound" connection to subsequent worker
        self.outbound_worker_address: Tuple[str, int] = None
        self.outbound_worker_socket: socket = None
        self.outbound_worker_conn_handler = None
        self.outbound_conn_worker_thread = None

        # "Inbound" connection to previous worker
        self.inbound_worker_address: Tuple[str, int] = None
        self.inbound_worker_socket: socket = None
        self.inbound_worker_conn_handler = None
        self.inbound_conn_worker_thread = None

        # Can take values FIRST, MIDDLE, LAST, SOLO
        self.node_role: str = node_role

        self.model_runtime_session: ort.InferenceSession = None

        self.file_manager: FileManager = None

        self.prover: Prover = None

        # Counts the number of inference runs. Counter gets appended to uuid generated during inference run.
        self.inference_run_counter: int = 0

    def run(self):
        try:
            # Open socket to accept connection from previous worker.
            # First node / Solo node don't need to open a socket.
            if not (self.node_role == 'FIRST' or self.node_role == 'SOLO'):
                self.open_worker_socket()

            self.connect_to_coordinator()
            self.coordinator_conn_handler = CoordinatorConnectionHandler(
                connection=self.coordinator_socket,
                address=coordinator_address,
                initiating_node=self
            )

            # Spawn first thread to handle connection from worker to coordinator
            self.conn_coordinator_thread = threading.Thread(
                target=self.coordinator_conn_handler.run
            )
            self.conn_coordinator_thread.start()

            # Connecting to neighbouring nodes is only required when there's more than one node present.
            if not self.node_role == "SOLO":
                # Report the address that a worker can accept connections from. Important for the previous worker
                if not self.node_role == "FIRST":
                    self.coordinator_conn_handler.send(f"set_inbound_connection_address|"
                                                       f"{self.address[0]}|{self.address[1]}")

                # The last node will not need to obtain and/or connect to another node.
                # All other worker nodes should obtain the address of their subsequent node to connect to.
                if not self.node_role == "LAST":
                    self.coordinator_conn_handler.send("get_subsequent_worker_address")

                if not self.node_role == "FIRST":
                    self.inbound_worker_socket, self.inbound_worker_address = self.inbound_worker_socket.accept()
                    self.inbound_worker_conn_handler = WorkerConnectionHandler(
                        connection=self.inbound_worker_socket,
                        address=self.inbound_worker_address,
                        initiating_node=self
                    )
                    self.inbound_conn_worker_thread = threading.Thread(
                        target=self.inbound_worker_conn_handler.run
                    )
                    self.inbound_conn_worker_thread.start()
                    conditional_print(f"[LOGIC] Accepted connection from {self.inbound_worker_address}", VERBOSE)

                # The last node in the inference chain doesn't have a subsequent node to connect to.
                if not self.node_role == "LAST":
                    # Spawn second thread to handle connection from worker to next worker (in the inference chain)
                    self.connect_to_worker()
                    self.outbound_worker_conn_handler = WorkerConnectionHandler(
                        self.outbound_worker_socket,
                        self.outbound_worker_address,
                        initiating_node=self
                    )
                    self.outbound_conn_worker_thread = threading.Thread(
                        target=self.outbound_worker_conn_handler.run
                    )
                    self.outbound_conn_worker_thread.start()

                conditional_print(f"[LOGIC] Connections between all nodes established.", VERBOSE)

            while self.shard_id is None and self.model_id is None:
                conditional_print(f'[LOGIC] IDs not set yet. Retrying in 5s', VERBOSE)
                time.sleep(5)
            # ONLY after all connections have been created, do the ezkl setup!
            self.load_model()
            # Report to coordinator that ezkl setup has been completed
            self.coordinator_conn_handler.send("report_setup_complete")

        except KeyboardInterrupt:
            self.shutdown()

    def connect_to_coordinator(self):
        try:
            self.coordinator_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
            self.coordinator_socket.connect(self.coordinator_address)
            conditional_print(f"[LOGIC] Connected to Coordinator @ {self.coordinator_address}", VERBOSE)
        except Exception as e:
            print(e)
            sys.exit(0)

    def connect_to_worker(self):
        try:
            # Waits until this node has received the address it needs to connect to.
            while self.outbound_worker_address is None:
                if VERBOSE:
                    print("Next worker address not set yet. Sleeping for 1s, then trying again.")
                    time.sleep(1)
            self.outbound_worker_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
            self.outbound_worker_socket.connect(self.outbound_worker_address)
            conditional_print(f"[LOGIC] Connected to Worker @ {self.outbound_worker_address}", VERBOSE)

        except Exception as e:
            print(e)
            sys.exit(0)

    def open_worker_socket(self) -> None:
        self.inbound_worker_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        self.inbound_worker_socket.bind(self.address)
        self.inbound_worker_socket.listen()

    def set_ids(self, shard_id: int, model_id: str):
        self.shard_id = shard_id
        self.model_id = model_id
        # The FileManager has to be initialized here, as this is the earliest time that the worker has access to both
        # IDs.
        self.file_manager = FileManager(
            model_id=model_id,
            shard_id=shard_id,
            storage_dir=self.storage_dir
        )
        # The Prover also has to be initialized here, as it relies on the file_manager
        self.prover = Prover(
            file_manager=self.file_manager,
            input_visibility=INPUT_VISIBILITY,
            output_visibility=OUTPUT_VISIBILITY,
            param_visibility=PARAM_VISIBILITY,
            ezkl_optimization_goal=OPTIMIZATION_GOAL,
            input_scale=INPUT_SCALE,  # Optional param, can be removed.
            param_scale=PARAM_SCALE  # Optional param, can be removed
        )

    # Loads model (shard) so that inference can be made on model (shard)
    def load_model(self) -> None:
        # Initialize ONNX Runtime to run inference on model (shard)
        self.model_runtime_session = ort.InferenceSession(self.file_manager.get_model_path())

        conditional_print("[PREPROCESSING] Model (shard) successfully loaded", VERBOSE)

        conditional_print("[PREPROCESSING] Starting ezkl setup", VERBOSE)

        # Starts measuring time
        start_time: float = 0
        if self.benchmarking_mode:
            start_time = time.perf_counter()

        # Set up ezkl proof artefacts needed for future inference step(s).
        asyncio.run(self.prover.setup())
        # conditional_print("Skipping ezkl setup for now.", VERBOSE)

        # Stops measuring time & logs time difference
        end_time: float = 0
        if self.benchmarking_mode:
            end_time = time.perf_counter()
            difference: float = end_time - start_time

            vk_file_size: int = 0
            vk_path: str = self.file_manager.get_vk_path()
            if os.path.exists(vk_path):
                vk_file_size = os.path.getsize(vk_path)

            pk_file_size: int = 0
            pk_path: str = self.file_manager.get_pk_path()
            if os.path.exists(vk_path):
                pk_file_size = os.path.getsize(pk_path)

            self.setup_data.append(
                {
                    'shard_id': self.shard_id,
                    'model_id': self.model_id,
                    'setup_time': difference,
                    'vk_size': vk_file_size,
                    'pk_size': pk_file_size
                }
            )

        conditional_print("[PREPROCESSING] ezkl setup completed", VERBOSE)

    # Runs onnx model (shard) with input data, creates and writes witness file.
    #  Takes in (base64) raw_input_data and parses into numpy array prior to further steps.
    def run_inference(self, raw_input_data: bytes) -> None:
        input_array: np.ndarray = decode_b64_to_np_array(raw_input_data)
        # Run input_array (x) through model(shard) to obtain output (y).
        output = self.model_runtime_session.run(
            None,
            {'input': input_array}
        )

        # Starts measuring time
        start_time: float = 0
        if self.benchmarking_mode:
            start_time = time.perf_counter()

        # The inference_run_counter, indicates in which inference "run" we currently are. Can later be used to
        # obtain proofs of all nodes in the chain for e.g. run 55.
        witness_id: str = f"{uuid.uuid4().hex}_{self.inference_run_counter}"

        # Registers witness_id with file manager. Required before retrieving the witness path for the witness_id.
        # Crucial step to keep track of witness_ids for post-hoc proofs of specific witness (that corresponds to
        # specific witness_id)
        self.file_manager.add_witness_id(witness_id)

        # Generate the "raw" input-output data that is fed in the ezkl generate_witness() function.
        raw_witness_data: dict = dict(
            input_shapes=[input_array.shape],
            input_data=[input_array.reshape([-1]).tolist()],
            output_data=[o.reshape([-1]).tolist() for o in output]
        )
        with open(self.file_manager.get_raw_witness_path(witness_id), 'w') as f:
            json.dump(raw_witness_data, f)
            f.close()

        # Generate witness file using ezkl.gen_witness(). Pulls in raw witness file data based on witness_id.
        asyncio.run(self.prover.generate_witness(witness_id))

        # Stops measuring time & logs time difference
        end_time: float = 0
        if self.benchmarking_mode:
            end_time = time.perf_counter()
            difference: float = end_time - start_time

            witness_file_size: int = 0
            witness_path: str = self.file_manager.get_witness_path(witness_id)
            if os.path.exists(witness_path):
                witness_file_size = os.path.getsize(witness_path)
            self.witness_data.append(
                {
                    'witness_id': witness_id,
                    'shard_id': self.shard_id,
                    'model_id': self.model_id,
                    'witness_generation_time': difference,
                    'witness_size': witness_file_size
                }
            )

        # The output data gets formatted in a list, and the actual ndarray is at index 0.
        output_data = output[0]
        # Encode the output data (ndarray) as base64 bytes
        encoded_output_data: bytes = encode_np_array_to_b64(output_data)

        # The last node send the final result to the coordinator. Also applies when there's only one node.
        if self.node_role == "LAST" or self.node_role == "SOLO":
            message: bytes = (b'report_final_inference_output|' +
                              self.inference_run_counter.to_bytes(8, byteorder='big') +
                              b'|' +
                              encoded_output_data)
            self.coordinator_conn_handler.send_bytes(message)
        # All other nodes send their result to their neighbouring node (outbound_worker_connection).
        else:
            message: bytes = (b'run_inference|' +
                              encoded_output_data)
            self.outbound_worker_conn_handler.send_bytes(message)

        self.coordinator_conn_handler.send(f"report_witness|{witness_id}")

    # Intermediate function that calls the prover to generate a proof.
    def generate_proof(self, witness_id: str) -> None:
        # Starts measuring time
        start_time: float = 0
        if self.benchmarking_mode:
            start_time = time.perf_counter()

        # Trigger proof generation
        self.prover.generate_proof_for_witness(witness_id)

        # Stops measuring time & logs time difference
        end_time: float = 0
        if self.benchmarking_mode:
            end_time = time.perf_counter()
            difference: float = end_time - start_time

        proof_path: str = self.file_manager.get_proof_path(witness_id)
        if self.benchmarking_mode:
            file_size: int = 0
            if os.path.exists(proof_path):
                file_size = os.path.getsize(proof_path)
            self.proving_data.append(
                {
                    'witness_id': witness_id,
                    'shard_id': self.shard_id,
                    'model_id': self.model_id,
                    'proof_generation_time': difference,
                    'proof_size': file_size
                }
            )
        self.coordinator_conn_handler.send(f'report_proof|{proof_path}')

    # Persists logged benchmarking results
    def save_benchmarking_results(self) -> None:
        if not self.benchmarking_mode:
            return

        df_witness_times = pd.DataFrame(self.witness_data)
        df_proving_times = pd.DataFrame(self.proving_data)
        df_setup_times = pd.DataFrame(self.setup_data)

        data_dir: str = self.file_manager.get_benchmarking_results_dir()
        witness_file: str = f'{data_dir}/witness_data.csv'
        proving_file: str = f'{data_dir}/proving_data.csv'
        setup_file: str = f'{data_dir}/setup_data.csv'

        save_dataframe(
            file_path=witness_file,
            df=df_witness_times
        )
        save_dataframe(
            file_path=proving_file,
            df=df_proving_times
        )
        save_dataframe(
            file_path=setup_file,
            df=df_setup_times
        )

    # Triggers shutdown logic where connection handlers are stopped, sockets are closed and threads shutdown.
    def shutdown(self):
        conditional_print('[LOGIC] Starting shutdown process')
        if self.coordinator_socket is not None:
            self.coordinator_conn_handler.RUNNING = False
            self.coordinator_socket.close()
            if threading.current_thread() != self.conn_coordinator_thread:
                conditional_print('[LOGIC] Joining coordinator thread')
                self.conn_coordinator_thread.join()

        if self.outbound_worker_socket is not None:
            self.outbound_worker_conn_handler.RUNNING = False
            self.outbound_worker_socket.close()
            if threading.current_thread() != self.outbound_conn_worker_thread:
                conditional_print('[LOGIC] Joining outbound worker thread')
                self.outbound_conn_worker_thread.join()

        if self.inbound_worker_socket is not None:
            self.inbound_worker_conn_handler.RUNNING = False
            self.inbound_worker_socket.close()
            if threading.current_thread() != self.inbound_conn_worker_thread:
                conditional_print('[LOGIC] Joining inbound worker thread')
                self.inbound_conn_worker_thread.join()

        sys.exit(0)


if __name__ == "__main__":
    # Workers need to be started in order: FIRST ... ... ... LAST
    if len(sys.argv) < 6:
        print(f'Usage: worker.py <worker_host> <worker_port> <coordinator_host> <coordinator_port> '
              f'<node_role> [benchmarking_mode] [storage_dir]')
        sys.exit(1)

    worker_address = (sys.argv[1], int(sys.argv[2]))
    coordinator_address = (sys.argv[3], int(sys.argv[4]))

    node_role = sys.argv[5]
    if node_role not in ['FIRST', 'MIDDLE', 'LAST', 'SOLO']:
        print('[node_role] has 3 options: FIRST, MIDDLE, LAST, or SOLO. \n'
              '>1 node: One node has to identify as the FIRST/LAST, the others as MIDDLE\n'
              '=1 node: Node identifies as SOLO')
        sys.exit(1)

    worker = None

    if len(sys.argv) == 6:
        worker = Worker(
            address=worker_address,
            coordinator_address=coordinator_address,
            node_role=node_role
        )

    benchmarking_mode = False
    if len(sys.argv) == 7:
        if sys.argv[6] == 'true':
            benchmarking_mode = True
        elif sys.argv[6] == 'false':
            benchmarking_mode = False
        else:
            print(f'Incorrect benchmarking_mode value! Options are: true, false')
            sys.exit(1)
        worker = Worker(
            address=worker_address,
            coordinator_address=coordinator_address,
            node_role=node_role,
            benchmarking_mode=benchmarking_mode
        )

    storage_dir: str = sys.argv[7]
    if len(sys.argv) == 8:
        if sys.argv[6] == 'true':
            benchmarking_mode = True
        elif sys.argv[6] == 'false':
            benchmarking_mode = False
        else:
            print(f'Incorrect benchmarking_mode value! Options are: true, false')
            sys.exit(1)

        if storage_dir == '':
            print(f'Incorrect storage_dir value!')
            sys.exit(1)

        worker = Worker(
            address=worker_address,
            coordinator_address=coordinator_address,
            node_role=node_role,
            benchmarking_mode=benchmarking_mode,
            storage_dir=storage_dir
        )

    worker.run()
