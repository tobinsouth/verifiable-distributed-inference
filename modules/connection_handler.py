import time
from socket import socket
from typing import Tuple
from abc import ABC, abstractmethod
from utils.helpers import conditional_print
from config import VERBOSE, BUF_SIZE


# Makes this class abstract.
class ConnectionHandler(ABC):

    def __init__(self, connection: socket, address: Tuple[str, int]) -> None:
        self.connection = connection
        self.address = address
        self.RUNNING = True

    @property
    @abstractmethod
    def run(self) -> None:
        pass

    @property
    @abstractmethod
    def execute_request(self, request: bytes) -> None:
        pass

    @property
    @abstractmethod
    def execute_request_str(self, request: str) -> None:
        pass

    @property
    @abstractmethod
    def execute_request_bytes(self, request: bytes) -> None:
        pass

    # Sends message over socket connection.
    # Appends \r\n (message delimiter, to ensure messages larger then 4096 bytes get processed as one)
    def send(self, request: str) -> None:
        conditional_print(f'[MESSAGING] Sent: {request}', VERBOSE)
        # Required to correctly identify message boundaries.
        request = request + "\r\n"
        self.connection.send(request.encode("utf-8"))

    def send_bytes(self, request: bytes) -> None:
        conditional_print(f'[MESSAGING] Sent: {request}', VERBOSE)
        # Required to correctly identify message boundaries.
        request = request + b'\r\n'
        self.connection.send(request)


class WorkerConnectionHandler(ConnectionHandler):
    def __init__(self,
                 connection: socket,
                 address: Tuple[str, int],
                 initiating_node: object) -> None:
        super().__init__(connection, address)
        self.initiating_node = initiating_node

    def run(self) -> None:
        # Buffer to store "excess" data that needs to be carried between received messages.
        buffer = b''

        while self.RUNNING:
            # Holds the current state of the received message. Will be appended to if needed.
            message = b''
            data = b''
            delimiter_index = buffer.find(b'\r\n')
            # Before pulling in new data, check if we have an entire message remaining in the buffer
            # \r\n exists in buffer
            if delimiter_index != -1:
                # Empty out buffer and continue processing.
                data = buffer
                buffer = b''
            # \r\n doesn't exist in buffer, we need to pull in more data
            else:
                data = self.connection.recv(BUF_SIZE)
                message += buffer
            # Empty message -> we assume the connection was shut down.
            if not data:
                print(f"[MESSAGING] Connection was shut down by {self.address}")
                self.RUNNING = False
                return

            while True:
                # Check if message delimiter exists in latest data, i.e. we've found the end of the message
                delimiter_index = data.find(b'\r\n')
                # Message end was not found
                if delimiter_index == -1:
                    # Save current data to message variable
                    message += data
                    # Pull in new data
                    data = self.connection.recv(BUF_SIZE)
                    # Restart the loop
                    continue
                    # Message end was found
                else:
                    # Split data into end of message and
                    message_tail = data[:delimiter_index]
                    message += message_tail
                    remainder = data[delimiter_index + 2:]
                    buffer += remainder
                    # Break out of loop to initiate further message processing.
                    break
            self.execute_request(message)

    def execute_request(self, request: bytes) -> None:
        # run_inference command needs to be processed in bytes, otherwise important data is lost through decoding it.
        if request.startswith(b'run_inference'):
            self.execute_request_bytes(request)
        else:
            self.execute_request_str(request.decode("utf-8"))

    def execute_request_str(self, request: str) -> None:
        if len(request) == 0:
            return
        print(f"[MESSAGING] Received: {request}")
        # Messages are sent in the format `command|message`
        tokens = request.split('|')
        if tokens[0] == "":
            pass

    def execute_request_bytes(self, request: bytes) -> None:
        if len(request) == 0:
            return
        print(f"[MESSAGING] Received: {request}")
        # Messages are sent in the format `command|message`
        tokens: list = request.split(b'|')

        # Worker receives an intermediate output from its predecessor and runs it's shard on that input.
        # Sends intermediate output to the next node.
        if tokens[0] == b'run_inference':
            raw_data: bytes = tokens[1]
            self.initiating_node.run_inference(raw_data)


class CoordinatorConnectionHandler(ConnectionHandler):
    def __init__(self,
                 connection: socket,
                 address: Tuple[str, int],
                 initiating_node: object,
                 shard_id: int = -1,
                 model_id: str = ""
                 ) -> None:
        super().__init__(connection, address)
        # Deriving the mode variable based on the type of the `initiating_node` object.
        self.mode = type(initiating_node).__name__.upper()
        self.initiating_node = initiating_node
        self.model_id = model_id
        self.shard_id = shard_id

    def run(self) -> None:
        # Buffer to store "excess" data that needs to be carried between received messages.
        buffer = b''

        if self.mode == "COORDINATOR":
            self.send(f"set_shard_and_model|{self.shard_id}|{self.model_id}")
        while self.RUNNING:
            # Holds the current state of the received message. Will be appended to if needed.
            message = b''
            data = b''
            delimiter_index = buffer.find(b'\r\n')
            # Before pulling in new data, check if we have an entire message remaining in the buffer
            # \r\n exists in buffer
            if delimiter_index != -1:
                # Empty out buffer and continue processing.
                data = buffer
                buffer = b''
            # \r\n doesn't exist in buffer, we need to pull in more data
            else:
                data = self.connection.recv(BUF_SIZE)
                message += buffer
            # Empty message -> we assume the connection was shut down.
            if not data:
                print(f"Connection was shut down by {self.address}")
                self.RUNNING = False
                return

            while True:
                # Check if message delimiter exists in latest data, i.e. we've found the end of the message
                delimiter_index = data.find(b'\r\n')
                # Message end was not found
                if delimiter_index == -1:
                    # Save current data to message variable
                    message += data
                    # Pull in new data
                    data = self.connection.recv(BUF_SIZE)
                    # Restart the loop
                    continue
                    # Message end was found
                else:
                    # Split data into end of message and
                    message_tail = data[:delimiter_index]
                    message += message_tail
                    remainder = data[delimiter_index + 2:]
                    buffer += remainder
                    # Break out of loop to initiate further message processing.
                    break
            self.execute_request(message)

    def execute_request(self, request: bytes) -> None:
        # run_inference command needs to be processed in bytes, otherwise important data is lost through decoding it.
        if request.startswith(b'run_inference') or request.startswith(b'report_final_inference_output'):
            self.execute_request_bytes(request)
        else:
            self.execute_request_str(request.decode("utf-8"))

    def execute_request_str(self, request: str) -> None:
        if len(request) == 0:
            return
        print(f"[MESSAGING] Received: {request}")
        # Messages are sent in the format `command|message`
        tokens: list = request.split('|')
        # For the sake of readability, split the coordinator and worker commands into different blocks.
        if self.mode == "COORDINATOR":
            # Worker requests the ip:port string of its subsequent worker, which it has to connect to
            if tokens[0] == "get_subsequent_worker_address":
                # Check if there's at least one more active connection to the coordinator
                connection_keys: list = list(self.initiating_node.connections.keys())
                ip_port_str: str = f"{self.address[0]}:{self.address[1]}"
                connection_index: int = connection_keys.index(ip_port_str)
                num_connections: int = len(connection_keys)
                message: str = ""
                if connection_index < num_connections - 1:
                    subsequent_node_address: str = connection_keys[connection_index + 1]
                    # Wait until the address is set!
                    while True:
                        try:
                            inbound_address: str = self.initiating_node.connection_to_inbound_address[subsequent_node_address]
                            break
                        except Exception as e:
                            print(e)
                            conditional_print(f'[MESSAGING] Inbound address for {subsequent_node_address} not set yet. '
                                              f'Trying again in 10s.', VERBOSE)
                            time.sleep(10)
                    address_tokens: list = inbound_address.split(":")
                    host: int = address_tokens[0]
                    port: int = address_tokens[1]
                    message = f"set_subsequent_worker_address|{host}|{port}"
                else:
                    message = f"set_subsequent_worker_address|wait"
                self.send(message)
            # Worker reports the ip:port strong that it wants to receive an inbound (worker, not coordinator)
            # connection from
            elif tokens[0] == "set_inbound_connection_address":
                host: str = tokens[1]
                port: int = int(tokens[2])
                ip_port_str: str = f"{self.address[0]}:{self.address[1]}"
                self.initiating_node.connection_to_inbound_address[ip_port_str] = f"{host}:{port}"
            # Worker reports a generated witness file, which later must be used in proof by worker
            elif tokens[0] == "report_witness":
                # Reported witness_id. Format: "{uuid}_{inference run count}"
                witness_id: str = tokens[1]
                self.initiating_node.store_witness(witness_id, self.shard_id)
            # AA
            elif tokens[0] == "report_proof":
                proof_path: str = tokens[1]
                self.initiating_node.verify_proof(
                    proof_path=proof_path,
                    shard_id=self.shard_id,
                    model_id=self.model_id
                )
            # Registers that another node has completed it's ezkl setup procedures and can now process inference
            # requests.
            elif tokens[0] == "report_setup_complete":
                self.initiating_node.register_ready_node()
        elif self.mode == "WORKER":
            # Sets model shard that the worker should run
            if tokens[0] == "set_shard_and_model":
                shard_id: int = int(tokens[1])
                model_id: str = tokens[2]
                # These two IDs have to be set, because they're only set/known by the coordinator upfront and sent to
                # the worker using this message.
                self.shard_id = shard_id
                self.model_id = model_id
                self.initiating_node.set_ids(shard_id, model_id)
                # IMPORTANT: this had to be done later down the line, as communication deteriorated due to ezkl
                # computations.
                # self.initiating_node.load_model()
            # Sets the address of the subsequent worker, which receives the inference output
            elif tokens[0] == "set_subsequent_worker_address":
                # Handle case where the subsequent node wasn't spawned yet and coordinator is waiting for it connect.
                if tokens[1] == "wait":
                    time.sleep(10)
                    self.send("get_subsequent_worker_address")
                else:
                    host = tokens[1]
                    port = int(tokens[2])
                    address = (host, port)
                    self.initiating_node.outbound_worker_address = address
            # Coordinator requests a proof to be generated for a specific witness_id.
            elif tokens[0] == "get_proof":
                witness_id: str = tokens[1]
                self.initiating_node.generate_proof(witness_id)
            # Coordinator triggers that benchmarking results are persisted.
            elif tokens[0] == "save_benchmarking_results":
                # Setup data is already saved at this point
                self.initiating_node.save_witness_data()
                self.initiating_node.save_proving_data()
            # Coordinator triggers node shutdown.
            elif tokens[0] == "shutdown":
                self.initiating_node.shutdown()

    # Executes requests that NEED to be processed in byte form
    # (and not decoded to a string like in execute_request_str).
    def execute_request_bytes(self, request: bytes) -> None:
        if len(request) == 0:
            return
        print(f"[MESSAGING] Received: {request}")
        # Messages are sent in the format `command|message`
        tokens: list = request.split(b'|')
        if self.mode == "COORDINATOR":
            if tokens[0] == b'report_final_inference_output':
                inference_run : int = int(int.from_bytes(tokens[1], byteorder='big'))
                raw_data: bytes = tokens[2]
                self.initiating_node.save_final_inference_output(raw_data, inference_run)
        elif self.mode == "WORKER":
            # This handles the initial request (for inference) sent from the coordinator to the first node in the chain
            if tokens[0] == b"run_inference":
                raw_data: bytes = tokens[1]
                self.initiating_node.run_inference(raw_data)
