import re
import os
import ezkl
import time
import numpy as np
import onnxruntime as ort
import json


class Prover:
    def __init__(
        self,
        model_id: str,
        model_dir: str,
        proof_dir: str,
        input_visibility: str,
        output_visibility: str,
        param_visibility: str,
        ezkl_optimization_goal: str
    ):
        self.model_id = model_id
        self.model_dir = model_dir
        self.proof_dir = proof_dir
        self.py_run_args = ezkl.PyRunArgs()
        self.py_run_args.input_visibility = input_visibility
        self.py_run_args.output_visibility = output_visibility
        self.py_run_args.param_visibility = param_visibility
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
            input_data = Prover.fetch_random_input_data((1, 1, 28, 28))
        output_data = ort_session.run(None, {'input': input_data})
        witness_data = dict(input_shapes=[input_data.shape],
                            input_data=[input_data.reshape([-1]).tolist()],
                            output_data=[o.reshape([-1]).tolist() for o in output_data])
        with open(data_path, 'w') as f:
            json.dump(witness_data, f)
        # The first element in output_data is the actual NumPy ndarray we need to pass to the function when we call
        # it for the next shard.
        return output_data[0]

    # Generates ezkl proof for shard `shard_id`. Requires the output of the previous shard as input for subsequent one.
    # `previous_shard_output` is not required for the first shard.
    def generate_proof(self, shard_id: int = None, previous_shard_output=None):
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

    # Verifies ezkl proof.
    @staticmethod
    def verify_proof(proof_path, settings_path, vk_path, srs_path):
        # Verify proof
        res = ezkl.verify(
            proof_path=proof_path,
            settings_path=settings_path,
            vk_path=vk_path,
            srs_path=srs_path
        )
        assert res == True

    def prove(self):
        num_shards = self.get_number_of_shards(self.model_id, self.model_dir)
        # Handle case where model wasn't sharded.
        if num_shards <= 0:
            start = time.time()
            proof_path, settings_path, vk_path, srs_path = self.generate_proof()
            print(f"Proof of Shard {self.model_id} generated at {proof_path}")
            end = time.time()
            print(f"Generating proof took {end - start} s")

        # Model was sharded.
        else:
            print(f"Identified {num_shards} shards for model {self.model_id}")
            previous_shard_output = None
            start = time.time()
            for shard_id in range(num_shards):
                previous_shard_output, shard_proof_path, shard_settings_path, shard_vk_path, shard_srs_path = self.generate_proof(
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
