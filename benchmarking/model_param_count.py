import os
import sys

import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules.model_training import MLPModel, CNNModel, AttentionModel, MLP2Model

OUTPUT_DIR = './results'

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = []

    mlp_model = MLPModel()
    total_params_mlp = sum(p.numel() for p in mlp_model.parameters())
    rows.append({
        'model': mlp_model.name,
        'num_params': total_params_mlp
    })

    mlp2_model = MLP2Model()
    total_params_mlp2 = sum(p.numel() for p in mlp2_model.parameters())
    rows.append({
        'model': mlp2_model.name,
        'num_params': total_params_mlp2
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
