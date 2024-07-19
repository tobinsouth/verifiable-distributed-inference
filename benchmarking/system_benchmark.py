import subprocess
import sys
import time
import os

from modules.model_training import AVAILABLE_MODELS

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

RESULTS_DIR = './results'
STORAGE_DIR = './tmp-system-benchmark'


def start_coordinator(host, port, num_shards):
    # TODO: fix parameters on this
    return subprocess.Popen(
        ['python', '../coordinator.py', host, str(port), str(num_shards)]
    )


def start_worker(host, port, coordinator_host, coordinator_port, node_role):
    # TODO: fix parameters on this
    return subprocess.Popen(
        ['python', '../worker.py', host, str(port), coordinator_host, str(coordinator_port), node_role]
    )


# TODO: does the port_offset avoid port collisions??
def run_setup(num_workers: int, model_name: str, port_offset: int):
    # Coordinator params
    coordinator_host: str = '127.0.0.1'
    coordinator_port: int = 8000 + port_offset
    coordinator_process = start_coordinator(coordinator_host, coordinator_port, num_workers)

    # Wait for coordinator to initialize
    time.sleep(10)

    # Worker setup
    worker_processes = []
    for i in range(num_workers):
        worker_host: str = '127.0.0.1'
        worker_port: int = 8001 + port_offset + i
        if num_workers == 1:
            node_role: str = 'SOLO'
        elif i == 0:
            node_role: str = 'FIRST'
        elif i == num_workers - 1:
            node_role: str = 'LAST'
        else:
            node_role: str = ''

        worker_processes.append(start_worker(worker_host, worker_port, coordinator_host, coordinator_port, node_role))

        # Wait for worker `i` to initialize
        time.sleep(10)

    # Wait for processes to complete
    coordinator_process.wait()
    for p in worker_processes:
        p.wait()


if __name__ == "__main__":
    # TODO: add model support (we need to run these seperately for each model!)
    # TODO: add storage path support (we need to run these concurrently!)
    # Example usage:
    # python system_benchmark.py linear_relu ./tmp
    # python system_benchmark.py cnn ./tmp2
    # python system_benchmark.py attention ./tmp3

    if len(sys.argv) < 2:
        print("Invalid usage!")
        print(f'Usage: accuracy_benchmark <model> [storage_dir]')
        print(f'Available models are: {", ".join(AVAILABLE_MODELS)}')
        sys.exit(1)

    model_name: str = sys.argv[1]
    if model_name not in AVAILABLE_MODELS:
        print(f'Incorrect model value! Available models are: {", ".join(AVAILABLE_MODELS)}')
        sys.exit(1)

    # Option to set a custom path.
    if len(sys.argv) == 3:
        STORAGE_DIR = sys.argv[2]

    os.makedirs(STORAGE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # We need this port offset so that all three model benchmarks can run concurrently and don't block the
    # ports for each other.
    port_offset_table = {}
    for i, model_name_ in enumerate(AVAILABLE_MODELS):
        port_offset_table[model_name_] = (i - 1) * 1000

    # TODO: change back to [1, 2, 3, 4, 6, 12]
    # temporary for testing purposes
    for num_workers in [1]:
        print(f"Running setup with {num_workers} workers")
        run_setup(num_workers, model_name, port_offset_table[model_name])
        print(f"Setup with {num_workers} workers completed")
