import sys
import os
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'src'))
sys.path.append(src_dir)
import json
import torch
import yaml
import optuna
import joblib
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam, SGD
from sklearn.metrics import roc_curve, fbeta_score
from sklearn.model_selection import train_test_split
from model import get_embedding_model # may show "Import "model" could not be resolved", but its actually fine


# Load exp_config
with open(os.path.join(current_dir, 'exp_config.json')) as config_file:
    config = json.load(config_file)

# Load preprocessed triplets
triplets_path = os.path.abspath("../preprocessing/preprocess_dataset/preprocessed/EfficientNet/preprocessed_triplets.npy")
# triplets_path = os.path.abspath(os.path.join(current_dir, '../data-preprocessing', '/EfficientNetb0', 'preprocessed_triplets.npy'))
triplets = np.load(triplets_path, allow_pickle=True)

# Split triplets into train, validation, and test sets
train_triplets, temp_triplets = train_test_split(triplets, train_size=config['train_size'], random_state=config['random_state'])
val_triplets, test_triplets = train_test_split(temp_triplets, test_size=config['test_size'] / (config['val_size'] + config['test_size']), random_state=config['random_state'])

# Function to convert triplets to DataLoader
def triplets_to_dataloader(triplets, batch_size):
    anchors = np.stack([triplet[0] for triplet in triplets]).astype(np.float32)
    positives = np.stack([triplet[1] for triplet in triplets]).astype(np.float32)
    negatives = np.stack([triplet[2] for triplet in triplets]).astype(np.float32)
    
    anchors = anchors.transpose((0, 3, 1, 2))  # Ensure images have shape (batch_size, 3, 224, 224)
    positives = positives.transpose((0, 3, 1, 2))
    negatives = negatives.transpose((0, 3, 1, 2))
    
    dataset = TensorDataset(
        torch.tensor(anchors, dtype=torch.float32),
        torch.tensor(positives, dtype=torch.float32),
        torch.tensor(negatives, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create DataLoaders
train_loader = triplets_to_dataloader(train_triplets, config['batch_size'])
val_loader = triplets_to_dataloader(val_triplets, config['batch_size'])
test_loader = triplets_to_dataloader(test_triplets, config['batch_size'])

# Custom loss function for triplet loss
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

# Function to calculate EER
def calculate_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    frr = 1 - tpr
    try:
        eer_threshold = thresholds[np.nanargmin(np.absolute((fpr - frr)))]
        eer = fpr[np.nanargmin(np.absolute((fpr - frr)))]
    except ValueError:
        eer_threshold = None
        eer = None
    return eer, eer_threshold

# Function to evaluate the model
def evaluate(loader, model, beta):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    all_labels = []
    all_scores = []
    with torch.no_grad():
        for anchor, positive, negative in loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anchor_embedding, positive_embedding, negative_embedding = model(anchor, positive, negative)
            
            # Calculate distances
            distance_positive = torch.nn.functional.pairwise_distance(anchor_embedding, positive_embedding).cpu().numpy()
            distance_negative = torch.nn.functional.pairwise_distance(anchor_embedding, negative_embedding).cpu().numpy()
            
            # Labels: 1 for positive pair, 0 for negative pair
            labels = np.concatenate([np.ones(len(distance_positive)), np.zeros(len(distance_negative))])
            scores = np.concatenate([distance_positive, distance_negative])
            
            all_labels.extend(labels)
            all_scores.extend(scores)
    
    # Calculate EER
    eer, eer_threshold = calculate_eer(np.array(all_labels), np.array(all_scores))
    
    # Calculate F-beta Score
    predictions = np.array(all_scores) <= eer_threshold  # Assuming threshold-based classification
    fbeta = fbeta_score(np.array(all_labels), predictions, beta=beta)

    # Calculate KS Statistic
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    ks_statistic = np.max(tpr - fpr)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    
    return eer, eer_threshold, fbeta, ks_statistic, optimal_threshold

# Initialize an empty DataFrame to store results
history_df = pd.DataFrame()

# Objective function for Optuna
def objective(trial):
    global history_df

    batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    momentum = trial.suggest_float('momentum', 0.5, 0.9) if optimizer_name == 'SGD' else None
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
    num_epochs = trial.suggest_int('num_epochs', 10, 50)
    #num_epochs = trial.suggest_int('num_epochs', 1, 5) # testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_embedding_model(config['num_classes']).to(device)
    criterion = TripletLoss().to(device)

    if optimizer_name == 'Adam':
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        model.train()
        print(f'Starting epoch {epoch}')
        for i, (anchor, positive, negative) in enumerate(train_loader):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            optimizer.zero_grad()
            anchor_embedding, positive_embedding, negative_embedding = model(anchor, positive, negative)
            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
            loss.backward()
            optimizer.step()

    eer, eer_threshold, fbeta, ks_statistic_val, optimal_threshold_val = evaluate(val_loader, model, beta=config['beta'])

    # Prune the trial if EER calculation is not possible
    if eer is None:
        raise optuna.TrialPruned()

    #combined_metric = 0.5 * eer + 0.5 * (1 - fbeta)  # Note: EER should be minimized, F-beta should be maximized
    
    result = {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'optimizer': optimizer_name,
        'momentum': momentum if optimizer_name == 'SGD' else None,
        'weight_decay': weight_decay,
        'num_epochs': num_epochs,
        'eer': eer,
        'fbeta': fbeta,
        'ks_statistic_val': ks_statistic_val,
        'optimal_threshold_val': optimal_threshold_val
    }
    
    result_df = pd.DataFrame([result])
    history_df = pd.concat([history_df, result_df], ignore_index=True)
    
    return eer

# Create and optimize study
study = optuna.create_study(sampler=optuna.samplers.GPSampler(), direction='minimize')
print('Start training to find best hyperparameters')
study.optimize(objective, n_trials=25)
#study.optimize(objective, n_trials=1) # testing

# Print best hyperparameters
print(f'Best hyperparameters based on eer: {study.best_params}')

joblib.dump(study, os.path.join(current_dir, 'optuna_study.pkl'))

# Save the accumulated history to CSV
history_path = os.path.join(current_dir, 'history.csv')
history_df.to_csv(history_path, index=False)
print(f'history log saved to {history_path}')

# Extract the best parameters for further use
best_params = study.best_params
with open(os.path.join(current_dir, 'hyperParameters.yaml'), 'w') as hp_file:
    yaml.dump(best_params, hp_file)

config['batch_size'] = best_params['batch_size']
config['learning_rate'] = best_params['learning_rate']
config['num_epochs'] = best_params['num_epochs']

with open(os.path.join(current_dir, 'exp_config.json'), 'w') as config_file:
    json.dump(config, config_file, indent=4)

# Recreate DataLoaders with best parameters
train_loader = triplets_to_dataloader(train_triplets, best_params['batch_size'])
val_loader = triplets_to_dataloader(val_triplets, best_params['batch_size'])
test_loader = triplets_to_dataloader(test_triplets, best_params['batch_size'])

# Save the test_loader data with the best parameters
anchors, positives, negatives = [], [], []
# Extract data from DataLoader
for anchor, positive, negative in test_loader:
    anchors.append(anchor)
    positives.append(positive)
    negatives.append(negative)
# Stack all parts of the triplets for saving
anchor_stack = torch.cat(anchors, dim=0)
positive_stack = torch.cat(positives, dim=0)
negative_stack = torch.cat(negatives, dim=0)
# Save the dataset
torch.save({
    'anchors': anchor_stack,
    'positives': positive_stack,
    'negatives': negative_stack
}, os.path.join(current_dir, 'test_data_best_params.pt'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_embedding_model(config['num_classes']).to(device)
criterion = TripletLoss().to(device)

if best_params['optimizer'] == 'Adam':
    optimizer = Adam(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
elif best_params['optimizer'] == 'SGD':
    optimizer = SGD(model.parameters(), lr=best_params['learning_rate'], momentum=best_params['momentum'], weight_decay=best_params['weight_decay'])

print('Start training with best hyperparameters obtained from Optuna')
for epoch in range(best_params['num_epochs']):
    model.train()
    print(f'Starting epoch {epoch}')
    for i, (anchor, positive, negative) in enumerate(train_loader):
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        optimizer.zero_grad()
        anchor_embedding, positive_embedding, negative_embedding = model(anchor, positive, negative)
        loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), config['model_save_path'])
print("model saved")

