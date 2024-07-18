import os

import pandas as pd

from modules.model_training import MLPModel, CNNModel, AttentionModel

OUTPUT_DIR = './results'

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = []

    linear_relu_model = MLPModel()
    total_params_linear_relu = sum(p.numel() for p in linear_relu_model.parameters())
    rows.append({
        'model': linear_relu_model.name,
        'num_params': total_params_linear_relu
    })

    cnn_model = CNNModel()
    total_params_cnn = sum(p.numel() for p in cnn_model.parameters())
    rows.append({
        'model': cnn_model.name,
        'num_params': total_params_cnn
    })

    attention_model = AttentionModel()
    total_params_attention = sum(p.numel() for p in attention_model.parameters())
    rows.append({
        'model': attention_model.name,
        'num_params': total_params_attention
    })

    print(rows)

    df = pd.DataFrame(rows)
    file_path: str = f'{OUTPUT_DIR}/model_params.csv'
    df.to_csv(file_path)
    print(f'Saved parameters to {file_path}')
