
class WitnessManager:
    def __init__(self):
        self.shard_to_witnesses_map: dict[int, list[str]] = {}
        self.witness_to_shard_map: dict[str, int] = {}
        self.run_to_witnesses_map: dict[int, list[str]] = {}

    # Adds witness_id, shard_id, run_id to the relevant datastructures.
    def add_witness(self, witness_id: str, shard_id: int) -> None:
        tokens = witness_id.split('_')
        run_id = int(tokens[1])

        # Check if list of witnesses exists for shard_id
        if shard_id not in self.shard_to_witnesses_map:
            self.shard_to_witnesses_map[shard_id] = []
        # Map shard_id -> [witness_id, ..]
        self.shard_to_witnesses_map[shard_id].append(witness_id)

        # Map witness_id -> shard_id
        self.witness_to_shard_map[witness_id] = shard_id

        # Check if list of witnesses exists for run_id
        if run_id not in self.run_to_witnesses_map:
            self.run_to_witnesses_map[run_id] = []
        # Map shard_id -> [witness_id, ..]
        self.run_to_witnesses_map[run_id].append(witness_id)

    # Returns the corresponding shard that generated witness with id `witness_id`
    def get_shard_by_witness_id(self, witness_id: str) -> int:
        return self.witness_to_shard_map.get(witness_id, -1)

    # Returns the corresponding list of witnesses that generated witness with id `witness_id`
    def get_witnesses_by_run_id(self, run_id: int) -> list[str]:
        return self.run_to_witnesses_map.get(run_id, [])
