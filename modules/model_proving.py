import re
import os
import ezkl
import time
import numpy as np
import onnxruntime as ort
import json

from modules.file_manager import FileManager
from utils.helpers import conditional_print
from config import VERBOSE, USE_EZKL_CLI


class Prover:
    def __init__(
        self,
        file_manager: FileManager,
        input_visibility: str,
        output_visibility: str,
        param_visibility: str,
        ezkl_optimization_goal: str,
        input_scale: int = None,
        param_scale: int = None
    ):
        self.file_manager = file_manager
        self.py_run_args = ezkl.PyRunArgs()
        self.py_run_args.input_visibility = input_visibility
        self.py_run_args.output_visibility = output_visibility
        self.py_run_args.param_visibility = param_visibility
        if input_scale is not None:
            self.py_run_args.input_scale = input_scale
        if param_scale is not None:
            self.py_run_args.param_scale = param_scale
        self.ezkl_optimization_goal = ezkl_optimization_goal

    @staticmethod
    def get_number_of_shards(model_id: str, model_dir: str) -> int:
        pattern = re.compile(f"{model_id}_shard_(\d+).onnx")
        highest_shard_id = -1
        for file in os.listdir(model_dir):
            match = pattern.match(file)
            if match:
                match_shard_id = int(match.group(1))
                if match_shard_id > highest_shard_id:
                    highest_shard_id = match_shard_id
        return highest_shard_id + 1

    @staticmethod
    def fetch_random_input_data(shape: tuple):
        return np.random.rand(*shape).astype(np.float32)

    # Set input_data to the output of the previous shard. Doesn't need to be set for the first shard.
    @staticmethod
    def generate_example_model_output(model_path, data_path, input_data=None):
        ort_session = ort.InferenceSession(model_path)
        if input_data is None:
            # This can be swapped out to actually fetch real data from the training dataset
            # Required dimensions for MNIST:
            input_data = Prover.fetch_random_input_data((1, 1, 28, 28))
        # Required dimensions for model_training Model:
        # input_data = np.random.rand(*(32, 128, 4096)).astype(np.float16)
        output_data = ort_session.run(None, {'input': input_data})
        witness_data = dict(input_shapes=[input_data.shape],
                            input_data=[input_data.reshape([-1]).tolist()],
                            output_data=[o.reshape([-1]).tolist() for o in output_data])
        with open(data_path, 'w') as f:
            json.dump(witness_data, f)
        # The first element in output_data is the actual NumPy ndarray we need to pass to the function when we call
        # it for the next shard.
        return output_data[0]

    # Function runs initial setup for a model.
    async def setup(self) -> None:
        # Do not repeat settings steps, if they've been completed before. (calibrating settings takes a while!)
        if not os.path.isfile(self.file_manager.get_settings_path()):
            # Generate settings file
            if USE_EZKL_CLI:
                return_code_gen_settings = os.system(f'ezkl gen-settings '
                                                     f'-M {self.file_manager.get_model_path()} '
                                                     f'-O {self.file_manager.get_settings_path()} '
                                                     f'--input-visibility {self.py_run_args.input_visibility} '
                                                     f'--output-visibility {self.py_run_args.output_visibility} '
                                                     f'--param-visibility {self.py_run_args.param_visibility}')
                if return_code_gen_settings != 0:
                    conditional_print(f"[ERROR] Unable to generate ezkl settings", VERBOSE)

                return_code_calibrate_settings = os.system(f'ezkl calibrate-settings '
                                                           f'-D {self.file_manager.get_calibration_data_path()} '
                                                           f'-M {self.file_manager.get_model_path()} '
                                                           f'--settings-path {self.file_manager.get_settings_path()} '
                                                           f'--target {self.ezkl_optimization_goal}')
                if return_code_calibrate_settings != 0:
                    conditional_print(f"[ERROR] Unable to calibrate ezkl settings", VERBOSE)
            else:
                result_gen_settings = ezkl.gen_settings(
                    model=self.file_manager.get_model_path(),
                    output=self.file_manager.get_settings_path(),
                    py_run_args=self.py_run_args
                )
                if not result_gen_settings:
                    conditional_print(f"[ERROR] Unable to generate ezkl settings", VERBOSE)

                # Calibrate settings file with data
                result_calibrate_settings = await ezkl.calibrate_settings(
                    data=self.file_manager.get_calibration_data_path(),
                    model=self.file_manager.get_model_path(),
                    settings=self.file_manager.get_settings_path(),
                    target=self.ezkl_optimization_goal
                )
                if not result_calibrate_settings:
                    conditional_print(f"[ERROR] Unable to calibrate ezkl settings", VERBOSE)
        else:
            conditional_print(f"[PREPROCESSING] ezkl settings already generated and calibrated", VERBOSE)

        # Do not repeat circuit compilation steps, if they've been completed before.
        if not os.path.isfile(self.file_manager.get_compiled_circuit_path()):
            # Pre-compile model (shard) with ezkl.compile_circuit. This compiled circuit is later used to generate a
            # witness file for every input-output pair created during inference.
            if USE_EZKL_CLI:
                return_code_compile_circuit = os.system(f'ezkl compile-circuit '
                                                        f'--model {self.file_manager.get_model_path()} '
                                                        f'--compiled-circuit {self.file_manager.get_compiled_circuit_path()} '
                                                        f'--settings-path {self.file_manager.get_settings_path()}')
                if return_code_compile_circuit != 0:
                    conditional_print(f"[ERROR] Unable to compile model to ezkl circuit", VERBOSE)
            else:
                result_compile = ezkl.compile_circuit(
                    model=self.file_manager.get_model_path(),
                    compiled_circuit=self.file_manager.get_compiled_circuit_path(),
                    settings_path=self.file_manager.get_settings_path()
                )
                if not result_compile:
                    conditional_print(f"[ERROR] Unable to compile model to ezkl circuit", VERBOSE)
        else:
            conditional_print(f"[PREPROCESSING] ezkl circuit already compiled", VERBOSE)

        # Do not fetch the srs, if it has been pulled in already (downloading takes a while!)
        if not os.path.isfile(self.file_manager.get_srs_path()):
            if USE_EZKL_CLI:
                return_code_srs = os.system(f'ezkl get-srs '
                                            f'--settings-path {self.file_manager.get_settings_path()} '
                                            f'--srs-path {self.file_manager.get_srs_path()}')
                if return_code_srs != 0:
                    conditional_print(f"[ERROR] Unable to get SRS", VERBOSE)
            else:
                result_srs = await ezkl.get_srs(
                    settings_path=self.file_manager.get_settings_path(),
                    srs_path=self.file_manager.get_srs_path()
                )
                if not result_srs:
                    conditional_print(f"[ERROR] Unable to get SRS", VERBOSE)
        else:
            conditional_print(f"[PREPROCESSING] ezkl SRS already downloaded", VERBOSE)

        if USE_EZKL_CLI:
            return_code_setup = os.system(f'ezkl setup '
                                          f'--compiled-circuit {self.file_manager.get_compiled_circuit_path()} '
                                          f'--vk-path {self.file_manager.get_vk_path()} '
                                          f'--pk-path {self.file_manager.get_pk_path()} '
                                          f'--srs-path {self.file_manager.get_srs_path()}')
            if return_code_setup != 0:
                conditional_print(f"[ERROR] Unable to complete final ezkl setup", VERBOSE)
        else:
            result_setup = ezkl.setup(
                model=self.file_manager.get_compiled_circuit_path(),
                vk_path=self.file_manager.get_vk_path(),
                pk_path=self.file_manager.get_pk_path(),
                srs_path=self.file_manager.get_srs_path()
            )
            if not result_setup:
                conditional_print(f"[ERROR] Unable to complete final ezkl setup", VERBOSE)

    # Generate witness file
    async def generate_witness(self, witness_id: str) -> None:
        # Fetch raw witness data path (clear-text json data)
        raw_witness_path: str = self.file_manager.get_raw_witness_path(witness_id)
        # Wait for the witness to be available if not yet available.
        while raw_witness_path == "":
            conditional_print(f"[ERROR] Raw witness file data not available at {raw_witness_path}", VERBOSE)
            time.sleep(0.5)
            raw_witness_path = self.file_manager.get_raw_witness_path(witness_id)

        # Fetch real witness data path
        witness_path: str = self.file_manager.get_witness_path(witness_id)
        if USE_EZKL_CLI:
            return_code_witness = os.system(f'ezkl gen-witness '
                                            f'--data {raw_witness_path} '
                                            f'--compiled-circuit {self.file_manager.get_compiled_circuit_path()} '
                                            f'--output {witness_path}')
            file_exists: bool = os.path.isfile(witness_path)
            if (return_code_witness != 0) or not file_exists:
                conditional_print(f"[ERROR] Unable to generate ezkl witness at {witness_path}", VERBOSE)
            else:
                conditional_print(f'[LOGIC] Witness successfully generated at {witness_path}', VERBOSE)
        else:
            witness_result = await ezkl.gen_witness(
                data=raw_witness_path,
                model=self.file_manager.get_compiled_circuit_path(),
                output=witness_path
            )
            file_exists: bool = os.path.isfile(witness_path)
            if not witness_result or not file_exists:
                conditional_print(f"[ERROR] Unable to generate ezkl witness at {witness_path}", VERBOSE)
            else:
                conditional_print(f'[LOGIC] Witness successfully generated at {witness_path}', VERBOSE)

    # Generates ezkl proof given a previously generated witness file, compiled circuit, etc.
    def generate_proof_for_witness(self, witness_id: str) -> None:
        proof_path: str = self.file_manager.get_proof_path(witness_id)
        if USE_EZKL_CLI:
            return_code_prove = os.system(f'ezkl prove '
                                          f'--witness {self.file_manager.get_witness_path(witness_id)} '
                                          f'--compiled-circuit {self.file_manager.get_compiled_circuit_path()} '
                                          f'--pk-path {self.file_manager.get_pk_path()} '
                                          f'--proof-path {proof_path} '
                                          f'--srs-path {self.file_manager.get_srs_path()}')
            file_exists: bool = os.path.isfile(proof_path)
            if (return_code_prove != 0) or not file_exists:
                conditional_print(f"[ERROR] Unable to generate ezkl proof for witness {witness_id} at {proof_path}",
                                  VERBOSE)
            else:
                conditional_print(f'[LOGIC] Proof successfully generated at {proof_path}', VERBOSE)
        else:
            prove_result: bool = ezkl.prove(
                witness=self.file_manager.get_witness_path(witness_id),
                model=self.file_manager.get_compiled_circuit_path(),
                pk_path=self.file_manager.get_pk_path(),
                proof_path=proof_path,
                srs_path=self.file_manager.get_srs_path()
            )
            file_exists: bool = os.path.isfile(proof_path)
            if not prove_result or not file_exists:
                conditional_print(f"[ERROR] Unable to generate ezkl proof for witness {witness_id} at {proof_path}",
                                  VERBOSE)
            else:
                conditional_print(f'[LOGIC] Proof successfully generated at {proof_path}', VERBOSE)

    # Verifies ezkl proof.
    @staticmethod
    def verify_proof(proof_path: str, settings_path: str, vk_path: str, srs_path: str) -> None:

        # Verify proof
        if USE_EZKL_CLI:
            return_code_verify = os.system(f'ezkl verify '
                                           f'--proof-path {proof_path} '
                                           f'--settings-path {settings_path} '
                                           f'--vk-path {vk_path} '
                                           f'--srs-path {srs_path}')
            if return_code_verify != 0:
                conditional_print(f'Proof at {proof_path} NOT valid', VERBOSE)
            else:
                conditional_print(f'Proof at {proof_path} valid', VERBOSE)
        else:
            res = ezkl.verify(
                proof_path=proof_path,
                settings_path=settings_path,
                vk_path=vk_path
            )
            if not res:
                conditional_print(f'Proof at {proof_path} NOT valid', VERBOSE)
            else:
                conditional_print(f'Proof at {proof_path} valid', VERBOSE)

    """ OUTDATED CODE:
    def old_generate_proof(self, shard_id: int = None, previous_shard_output=None):
        model_path, settings_path, data_path, compiled_model_path, srs_path, vk_path, pk_path, witness_path, proof_path = (
                                                                                                                              "",) * 9
        # Model wasn't sharded -> no shard_ids
        if shard_id is None:
            model_path = f"{self.model_dir}/{self.model_id}.onnx"
            settings_path = f"{self.proof_dir}/{self.model_id}_settings.json"
            data_path = f"{self.proof_dir}/{self.model_id}_data.json"
            compiled_model_path = f"{self.proof_dir}/{self.model_id}_network.compiled"
            srs_path = f"{self.proof_dir}/{self.model_id}_kzg.srs"
            vk_path = f"{self.proof_dir}/{self.model_id}_vk.key"
            pk_path = f"{self.proof_dir}/{self.model_id}_pk.key"
            witness_path = f"{self.proof_dir}/{self.model_id}_witness.json"
            proof_path = f"{self.proof_dir}/{self.model_id}_proof.pf"
        # Model was sharded
        else:
            model_path = f"{self.model_dir}/{self.model_id}_shard_{shard_id}.onnx"
            settings_path = f"{self.proof_dir}/{self.model_id}_shard_{shard_id}_settings.json"
            data_path = f"{self.proof_dir}/{self.model_id}_shard_{shard_id}_data.json"
            compiled_model_path = f"{self.proof_dir}/{self.model_id}_shard_{shard_id}_network.compiled"
            srs_path = f"{self.proof_dir}/{self.model_id}_shard_{shard_id}_kzg.srs"
            vk_path = f"{self.proof_dir}/{self.model_id}_shard_{shard_id}_vk.key"
            pk_path = f"{self.proof_dir}/{self.model_id}_shard_{shard_id}_pk.key"
            witness_path = f"{self.proof_dir}/{self.model_id}_shard_{shard_id}_witness.json"
            proof_path = f"{self.proof_dir}/{self.model_id}_shard_{shard_id}_proof.pf"

        # Generate Settings File
        res = ezkl.gen_settings(
            model=model_path,
            output=settings_path,
            py_run_args=self.py_run_args
        )
        assert res == True

        # Calibrate Settings
        # We only need to save the intermediate output here, as the witness data is placed in `DATA_PATH`.
        data_output = Prover.generate_example_model_output(model_path, data_path, previous_shard_output)
        res = ezkl.calibrate_settings(
            data=data_path,
            model=model_path,
            settings=settings_path,
            target=self.ezkl_optimization_goal
        )
        assert res == True

        # Compile model to a circuit
        res = ezkl.compile_circuit(
            model=model_path,
            compiled_circuit=compiled_model_path,
            settings_path=settings_path
        )
        assert res == True

        # (Down)load an SRS String
        res = ezkl.get_srs(
            settings_path=settings_path,
            logrows=None,
            srs_path=srs_path
        )
        assert res == True
        assert os.path.isfile(srs_path)

        # Setup Proof
        res = ezkl.setup(
            model=compiled_model_path,
            vk_path=vk_path,
            pk_path=pk_path,
            srs_path=srs_path
        )
        assert res == True
        assert os.path.isfile(vk_path)
        assert os.path.isfile(pk_path)

        # Generate witness file
        res = ezkl.gen_witness(
            data=data_path,
            model=compiled_model_path,
            output=witness_path
        )
        assert os.path.isfile(witness_path)

        # Create ZK-SNARK for the execution of the model
        res = ezkl.prove(
            witness=witness_path,
            model=compiled_model_path,
            pk_path=pk_path,
            proof_path=proof_path,
            srs_path=srs_path
        )
        assert os.path.isfile(proof_path)

        if shard_id is None:
            # No need to return intermediate output if only one proof is generated.
            return proof_path, settings_path, vk_path, srs_path
        else:
            # Return the intermediate output of the shard, as this is needed for the subsequent proof.
            return data_output, proof_path, settings_path, vk_path, srs_path

    def old_prove(self):
        num_shards = self.get_number_of_shards(self.model_id, self.model_dir)
        # Handle case where model wasn't sharded.
        if num_shards <= 0:
            start = time.time()
            proof_path, settings_path, vk_path, srs_path = self.old_generate_proof()
            print(f"Proof of Shard {self.model_id} generated at {proof_path}")
            end = time.time()
            print(f"Generating proof took {end - start} s")

        # Model was sharded.
        else:
            print(f"Identified {num_shards} shards for model {self.model_id}")
            previous_shard_output = None
            start = time.time()
            for shard_id in range(num_shards):
                previous_shard_output, shard_proof_path, shard_settings_path, shard_vk_path, shard_srs_path = self.old_generate_proof(
                    shard_id=shard_id,
                    previous_shard_output=previous_shard_output
                )
                print(f"Proof of Shard {shard_id} generated at {shard_proof_path}")

                # Added to ensure proofs are correct
                # verify_proof(shard_proof_path, shard_settings_path, shard_vk_path, shard_srs_path)
                # print(f"Proof of Shard {shard_id} verified")
            end = time.time()
            print(f"Completed processing for {num_shards} shards")
            print(f"Generating proofs took {end - start} s")
    """
