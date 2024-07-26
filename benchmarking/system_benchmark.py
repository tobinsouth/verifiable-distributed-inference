import subprocess
import sys
import time
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules.model_training import AVAILABLE_MODELS


def start_coordinator(host: str, port: int, num_shards: int, model_name: str, storage_dir: str):
    print(f'Spawning coordinator on {host}:{port}')
    return subprocess.Popen(
        ['python', '../coordinator.py', host, str(port), str(num_shards), model_name, 'true', storage_dir],
        stdout=sys.stdout,
        stderr=sys.stderr
    )


def start_worker(host: str, port: int, coordinator_host: str, coordinator_port: int, node_role: str, storage_dir: str):
    print(f'Spawning worker on {host}:{port}')
    return subprocess.Popen(
        ['python', '../worker.py', host, str(port), coordinator_host, str(coordinator_port), node_role, 'true',
         storage_dir],
        stdout=sys.stdout,
        stderr=sys.stderr
    )


def run_setup(num_workers: int, model_name: str, storage_dir: str, port_offset: int):
    # Coordinator params
    coordinator_host: str = '127.0.0.1'
    coordinator_port: int = 8000 + port_offset
    coordinator_process = start_coordinator(
        host=coordinator_host,
        port=coordinator_port,
        num_shards=num_workers,
        model_name=model_name,
        storage_dir=storage_dir
    )

    # Wait for coordinator to initialize
    time.sleep(10 if model_name == 'mlp' else 30)

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
            node_role: str = 'MIDDLE'

        worker_processes.append(
            start_worker(
                host=worker_host,
                port=worker_port,
                coordinator_host=coordinator_host,
                coordinator_port=coordinator_port,
                node_role=node_role,
                storage_dir=storage_dir
            )
        )

        # Wait for worker `i` to initialize
        time.sleep(10 if model_name == 'mlp' else 30)

    # Wait for processes to complete
    coordinator_process.wait()
    for p in worker_processes:
        p.wait()


if __name__ == "__main__":
    # Example usage:
    # python system_benchmark.py mlp
    # python system_benchmark.py cnn
    # python system_benchmark.py attention

    if len(sys.argv) < 2:
        print("Invalid usage!")
        print(f'Usage: system_benchmark <model> [num_shards]')
        print(f'Available models are: {", ".join(AVAILABLE_MODELS)}')
        sys.exit(1)

    model_name: str = sys.argv[1]
    if model_name not in AVAILABLE_MODELS:
        print(f'Incorrect model value! Available models are: {", ".join(AVAILABLE_MODELS)}')
        sys.exit(1)

    num_node_list: list[int] = []
    default_num_node_list: list[int] = [1, 2, 3, 4, 6, 12]
    if len(sys.argv) == 3:
        num_nodes = int(sys.argv[2])
        if num_nodes not in default_num_node_list:
            print(f'Invalid value for num_nodes: {num_nodes} (must be between any one of: {default_num_node_list})')
            sys.exit(1)
        else:
            num_node_list.append(num_nodes)
    else:
        num_node_list = default_num_node_list

    # We need this port offset so that all three model benchmarks can run concurrently and don't block the
    # ports for each other.
    port_offset_table = {}
    for i, model_name_ in enumerate(AVAILABLE_MODELS):
        port_offset_table[model_name_] = i * 1000

    for idx, num_workers in enumerate(num_node_list):
        print(f"Running setup with {num_workers} workers")

        # This makes sure the coordinator and workers never spawn on the same port
        additional_offset: int = 10 * num_workers

        storage_dir: str = f'./tmp-system-benchmark/{model_name}-{num_workers}'
        os.makedirs(storage_dir, exist_ok=True)
        run_setup(
            num_workers=num_workers,
            model_name=model_name,
            port_offset=port_offset_table[model_name] + additional_offset,
            storage_dir=storage_dir
        )
        print(f"Setup with {num_workers} workers completed")
