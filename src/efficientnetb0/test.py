import sys
import os
import json
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, fbeta_score, confusion_matrix
from model import get_embedding_model
from metric import evaluate

current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'src'))
sys.path.append(src_dir)

# Load configuration
with open(os.path.join(current_dir, 'exp_config.json')) as config_file:
    config = json.load(config_file)

with open(os.path.join(current_dir, 'hyperParameters.yaml')) as hp_file:
    hyperparams = yaml.load(hp_file, Loader=yaml.FullLoader)

# Load test data
test_data = torch.load(os.path.join(current_dir, 'test_data_best_params.pt'))
test_dataset = TensorDataset(test_data['anchors'], test_data['positives'], test_data['negatives'])
test_loader = DataLoader(test_dataset, batch_size=hyperparams['batch_size'], shuffle=False)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_embedding_model(config['num_classes']).to(device)
model.load_state_dict(torch.load(config['model_save_path'], map_location=device))
model.eval()

beta = config['beta']
eer, eer_threshold, fbeta, ks_statistic, optimal_threshold, tp, fp, fn, tn = evaluate(test_loader, model, beta)

print(f"EER: {eer:.4f} at threshold: {eer_threshold:.4f}")
print(f"F-beta: {fbeta:.4f}")
print(f"KS Statistic: {ks_statistic:.4f} at threshold: {optimal_threshold:.4f}")
print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")


# update metrics in config
config['test_EER'] = eer
config['test_ks'] = ks_statistic
config['test_TP'] = int(tp)
config['test_FP'] = int(fp)
config['test_FN'] = int(fn)
config['test_TN'] = int(tn)

with open(os.path.join(current_dir, 'exp_config.json'), 'w') as config_file:
    json.dump(config, config_file, indent=4)