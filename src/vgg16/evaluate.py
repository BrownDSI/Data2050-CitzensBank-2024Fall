import sys
import os
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'src'))
sys.path.append(src_dir)
import json
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, fbeta_score
from model import get_embedding_model  # may show "Import "model" could not be resolved", but its actually fine

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

# Function to evaluate the model
def evaluate(loader, model, beta):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    all_labels = []
    all_scores = []
    with torch.no_grad():
        for anchors, positives, negatives in loader:
            anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
            
            # Model now receives all three arguments
            anchor_embeddings, positive_embeddings, negative_embeddings = model(anchors, positives, negatives)

            # Calculate distances
            positive_distances = torch.nn.functional.pairwise_distance(anchor_embeddings, positive_embeddings)
            negative_distances = torch.nn.functional.pairwise_distance(anchor_embeddings, negative_embeddings)
            
            # Labels: 1 for positive pair, 0 for negative pair
            labels = np.concatenate([np.ones(len(positive_distances)), np.zeros(len(negative_distances))])
            scores = np.concatenate([positive_distances.cpu().numpy(), negative_distances.cpu().numpy()])
            
            all_labels.extend(labels)
            all_scores.extend(scores)
    
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    # Calculate EER
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    frr = 1 - tpr
    try:
        eer_threshold = thresholds[np.nanargmin(np.absolute((fpr - frr)))]
        eer = fpr[np.nanargmin(np.absolute((fpr - frr)))]
    except ValueError:
        eer_threshold = None
        eer = None

    # Calculate F-beta Score
    predictions = all_scores <= eer_threshold  # Assuming threshold-based classification
    fbeta = fbeta_score(all_labels, predictions, beta=beta)
    
    # Calculate KS Statistic
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    ks_statistic = np.max(tpr - fpr)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    
    return eer, eer_threshold, fbeta, ks_statistic, optimal_threshold

# Evaluate the model
beta = config['beta']
eer, eer_threshold, fbeta, ks_statistic, optimal_threshold = evaluate(test_loader, model, beta)

print(f'EER: {eer:.4f} at threshold: {eer_threshold:.4f}')
print(f'F-beta: {fbeta:.4f}')
print(f'KS Statistic: {ks_statistic:.4f} at threshold: {optimal_threshold:.4f}')

# update metrics in config
config['test_EER'] = eer
config['test_ks'] = ks_statistic
with open(os.path.join(current_dir, 'exp_config.json'), 'w') as config_file:
    json.dump(config, config_file, indent=4)
