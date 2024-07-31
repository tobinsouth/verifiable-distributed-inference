"""
Reference file structure:
STORAGE_DIR/
    COORDINATOR_SUB_DIR/
        final inference outputs...
    BENCHMARKING_RESULTS_SUB_DIR/
        benchmarking results...
    MODEL_SUB_DIR/
        model_shards...

    shard_0/
    ...
    shard_n/
"""
# Defines file structure
STORAGE_DIR: str = "./shared-storage"
# Folder structure within the STORAGE_DIR
MODEL_SUB_DIR: str = "/shards"
COORDINATOR_SUB_DIR: str = "/coordinator"
BENCHMARKING_RESULTS_SUB_DIR: str = "/benchmarking_results"

# Toggle for debug level prints
VERBOSE: bool = True

# PyTorch configuration
DEVICE = "cpu"

# ----- We recommend to leave the following values as is -----

# Buffer size for socket message receiving
BUF_SIZE = 4096

# ezkl configuration
# Toggles whether CLI or Py bindings are used
USE_EZKL_CLI = True
# We want the input and outputs to be "publicly" visible in the
NUM_CALIBRATION_DATAPOINTS = 50
INPUT_VISIBILITY: str = "public"
OUTPUT_VISIBILITY: str = "public"
PARAM_VISIBILITY: str = "hashed"
OPTIMIZATION_GOAL: str = "resources"
INPUT_SCALE = None
PARAM_SCALE = None
