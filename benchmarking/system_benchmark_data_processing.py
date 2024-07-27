import os
import sys

import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import BENCHMARKING_RESULTS_SUB_DIR
from modules.model_training import AVAILABLE_MODELS

SYSTEM_BENCHMARK_DATA_PATH = './tmp-system-benchmark'
RESULTS_DIR = './results'

if __name__ == '__main__':
    df_setup_data = pd.DataFrame(
        columns=['setup_id', 'shard_id', 'model_id', 'setup_time', 'vk_size', 'pk_size']
    )
    df_witness_data = pd.DataFrame(
        columns=['setup_id', 'witness_id', 'shard_id', 'model_id', 'witness_generation_time', 'witness_size']
    )
    df_proving_data = pd.DataFrame(
        columns=['setup_id', 'witness_id', 'shard_id', 'model_id', 'proof_generation_time', 'proof_size']
    )
    df_verification_data = pd.DataFrame(
        columns=['shard_id', 'model_id', 'verification_time']
    )

    for model in AVAILABLE_MODELS:
        for num_shards in [1, 2, 3, 4, 6, 12]:
            setup_id: str = f'{model}-{num_shards}'
            print(f'Handling {setup_id}')
            folder_path: str = f'{SYSTEM_BENCHMARK_DATA_PATH}/{setup_id}{BENCHMARKING_RESULTS_SUB_DIR}'
            if not os.path.exists(folder_path):
                print(f'Folder {folder_path} does not exist!')
                continue

            tmp_setup_data = pd.read_csv(f'{folder_path}/setup_data.csv')
            # add setup_id
            tmp_setup_data['setup_id'] = setup_id
            # reorder cols
            tmp_setup_data = tmp_setup_data[['setup_id', 'shard_id', 'model_id', 'setup_time', 'vk_size', 'pk_size']]
            df_setup_data = pd.concat([df_setup_data, tmp_setup_data], ignore_index=True)

            tmp_witness_data = pd.read_csv(f'{folder_path}/witness_data.csv')
            # add setup_id
            tmp_witness_data['setup_id'] = setup_id
            # reorder cols
            tmp_witness_data = tmp_witness_data[
                ['setup_id', 'witness_id', 'shard_id', 'model_id', 'witness_generation_time', 'witness_size']]
            df_witness_data = pd.concat([df_witness_data, tmp_witness_data], ignore_index=True)

            tmp_proving_data = pd.read_csv(f'{folder_path}/proving_data.csv')
            # add setup_id
            tmp_proving_data['setup_id'] = setup_id
            # reorder cols
            tmp_proving_data = tmp_proving_data[
                ['setup_id', 'witness_id', 'shard_id', 'model_id', 'proof_generation_time', 'proof_size']]
            df_proving_data = pd.concat([df_proving_data, tmp_proving_data], ignore_index=True)

            tmp_verification_data = pd.read_csv(f'{folder_path}/verification_data.csv')
            df_verification_data = pd.concat([df_verification_data, tmp_verification_data], ignore_index=True)

    df_cumulative_setup_time = df_setup_data.groupby(['setup_id']).agg({
        'setup_time': 'sum',
        'shard_id': 'count',
        'model_id': 'first',
    }).reset_index()

    df_cumulative_setup_time.columns = ['setup_id', 'total_setup_time', 'num_shards', 'model_id']
    df_cumulative_setup_time = df_cumulative_setup_time.drop(columns=['setup_id'])
    df_cumulative_setup_time = df_cumulative_setup_time[['model_id', 'num_shards', 'total_setup_time']]

    # Aggregate data
    df_cumulative_proving_time = df_proving_data.groupby(['setup_id']).agg({
        'proof_generation_time': 'sum',
        'shard_id': 'count',
        'model_id': 'first',
    }).reset_index()
    # Rename columns
    df_cumulative_proving_time.columns = ['setup_id', 'total_proof_generation_time', 'num_shards', 'model_id']
    # Drop setup_id
    df_cumulative_proving_time = df_cumulative_proving_time.drop(columns=['setup_id'])
    # Reorder columns
    df_cumulative_proving_time = df_cumulative_proving_time[['model_id', 'num_shards', 'total_proof_generation_time']]

    merged_df = pd.merge(df_witness_data, df_proving_data, on=['shard_id', 'model_id', 'setup_id', 'witness_id'])
    df_cumulative_proof_and_witness_size = merged_df.groupby(['setup_id']).agg({
        'proof_size': 'sum',
        'witness_size': 'sum',
        'model_id': 'first',
        'shard_id': 'count'
    }).reset_index()

    df_cumulative_pk_and_vk_size = df_setup_data.groupby(['setup_id']).agg({
        'vk_size': 'sum',
        'pk_size': 'sum',
        'model_id': 'first',
        'shard_id': 'count'
    }).reset_index()

    df_file_sizes = pd.merge(df_cumulative_proof_and_witness_size, df_cumulative_pk_and_vk_size,
                             on=['shard_id', 'model_id', 'setup_id'], how='inner')
    # Rename columns
    df_file_sizes.columns = ['setup_id', 'total_proof_size', 'total_witness_size', 'model_id', 'num_shards',
                             'total_vk_size', 'total_pk_size']
    # Drop setup_id
    df_file_sizes = df_file_sizes.drop(columns=['setup_id'])
    # Reorder columns
    df_file_sizes = df_file_sizes[
        ['model_id', 'num_shards', 'total_proof_size', 'total_witness_size', 'total_vk_size', 'total_pk_size']]

    df_cumulative_witness_time = df_witness_data.groupby('setup_id').agg({
        'witness_generation_time': 'sum',
        'shard_id': 'count',
        'model_id': 'first'
    }).reset_index()
    # Rename columns
    df_cumulative_witness_time.columns = ['setup_id', 'total_witness_generation_time', 'num_shards', 'model_id']
    # Drop setup_id
    df_cumulative_witness_time = df_cumulative_witness_time.drop(columns=['setup_id'])
    # Reorder columns
    df_cumulative_witness_time = df_cumulative_witness_time[['model_id', 'num_shards', 'total_witness_generation_time']]

    # Save data
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_cumulative_setup_time.to_csv(f'{RESULTS_DIR}/cumulative_setup_time.csv', index=False)
    df_cumulative_proving_time.to_csv(f'{RESULTS_DIR}/cumulative_proving_time.csv', index=False)
    df_file_sizes.to_csv(f'{RESULTS_DIR}/file_sizes.csv', index=False)
    df_cumulative_witness_time.to_csv(f'{RESULTS_DIR}/cumulative_witness_time.csv', index=False)
