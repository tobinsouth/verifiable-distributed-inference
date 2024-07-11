import os

MAIN_DIR = "./shared-storage"
MODEL_SUB_DIR = "/shards"
COORDINATOR_SUB_DIR = "/coordinator"


# Class that streamlines the retrieval of proof-related artefacts.
# Designed in a way so that it can be easily moved to cloud-storage (currently local storage).
class FileManager:
    def __init__(self, model_id: str, shard_id: int):
        self.model_id: str = model_id
        self.shard_id: int = shard_id

        # Create (sub)directories if they don't exist yet
        os.makedirs(MAIN_DIR, exist_ok=True)
        os.makedirs(f"{MAIN_DIR}/shard_{self.shard_id}", exist_ok=True)
        os.makedirs(f"{MAIN_DIR}{COORDINATOR_SUB_DIR}", exist_ok=True)

        # List to store all existing witness file IDs in
        self.witness_list: list[str] = []

    # Returns the ONNX model path
    def get_model_path(self) -> str:
        return f"{MAIN_DIR}{MODEL_SUB_DIR}/{self.model_id}_shard_{self.shard_id}.onnx"

    # Returns the ezkl settings.json path
    def get_settings_path(self) -> str:
        return f"{MAIN_DIR}/shard_{self.shard_id}/{self.model_id}_settings.json"

    # Returns the ezkl settings.json path
    @staticmethod
    def get_settings_path_static(shard_id: int, model_id: str) -> str:
        os.makedirs(f"{MAIN_DIR}/shard_{shard_id}", exist_ok=True)
        return f"{MAIN_DIR}/shard_{shard_id}/{model_id}_settings.json"

    # Returns the calibration data path
    def get_calibration_data_path(self) -> str:
        return f"{MAIN_DIR}/shard_{self.shard_id}/{self.model_id}_data.json"

    # Returns the compiled circuit (of onnx model) path
    def get_compiled_circuit_path(self) -> str:
        return f"{MAIN_DIR}/shard_{self.shard_id}/{self.model_id}_network.compiled"

    # Returns the SRS file path
    def get_srs_path(self) -> str:
        return f"{MAIN_DIR}/shard_{self.shard_id}/{self.model_id}_kzg.srs"

    # Returns the verification key path
    def get_vk_path(self) -> str:
        return f"{MAIN_DIR}/shard_{self.shard_id}/{self.model_id}_vk.key"

    # Returns the verification key path
    @staticmethod
    def get_vk_path_static(shard_id: int, model_id: str) -> str:
        os.makedirs(f"{MAIN_DIR}/shard_{shard_id}", exist_ok=True)
        return f"{MAIN_DIR}/shard_{shard_id}/{model_id}_vk.key"

    # Returns the proving key path
    def get_pk_path(self) -> str:
        return f"{MAIN_DIR}/shard_{self.shard_id}/{self.model_id}_pk.key"

    # Stores witness_id and returns path for witness.
    # This is called prior to storing the raw/clear-text witness data (which is NOT the real witness file yet).
    def add_witness_id(self, witness_id: str) -> bool:
        # Check if witness with that id already exists
        if witness_id in self.witness_list:
            return False
        # Add witness to list
        self.witness_list.append(witness_id)
        return True

    # Returns the path of a raw witness data file based on witness_id. This data is fed into the witness generator.
    # This method should only be called once the witness has been registered using add_witness()
    def get_raw_witness_path(self, witness_id: str) -> str:
        # Check if witness with `witness_id` exists for this shard/node
        if witness_id not in self.witness_list:
            return ""
        return f"{MAIN_DIR}/shard_{self.shard_id}/{witness_id}_raw.json"

    # Returns the path of a witness file based on witness_id.
    # This method should only be called once the witness has been registered using add_witness()
    def get_witness_path(self, witness_id: str) -> str:
        # Check if witness with `witness_id` exists for this shard/node
        if witness_id not in self.witness_list:
            return ""

        return f"{MAIN_DIR}/shard_{self.shard_id}/{witness_id}.json"

    # Returns the proof file path based on the witness_id.
    def get_proof_path(self, witness_id: str) -> str:
        if witness_id not in self.witness_list:
            return ""

        return f"{MAIN_DIR}/shard_{self.shard_id}/{witness_id}_proof.pf"

    # Returns the path that should hold the final inference output data after it has passed through all nodes/
    @staticmethod
    def get_final_output_path(model_id: str, run_id: int) -> str:
        os.makedirs(f"{MAIN_DIR}{COORDINATOR_SUB_DIR}", exist_ok=True)
        return f'{MAIN_DIR}{COORDINATOR_SUB_DIR}/{model_id}_final_output_{run_id}.json'
