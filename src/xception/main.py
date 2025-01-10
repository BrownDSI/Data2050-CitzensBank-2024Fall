import os
import sys
import yaml
import torch
import wandb
import numpy as np
import pandas as pd
from torch.optim import Adam, SGD
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from metric import evaluate
from model import get_embedding_model, TripletLoss
from load_data import triplets_to_dataloader


# ———————————————————————————————————————— CONFIGURATION SETUP ——————————————————————————————————————————

# Load YAML configuration
with open("FixedParameters.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load YAML configuration
with open("BestParameters.yaml", "r") as file:
    best_params = yaml.safe_load(file)

# Load preprocessed triplets
triplets_path = os.path.abspath("../preprocessing/preprocessed_dataset/preprocessed/Xception/preprocessed_triplets.npy")
triplets = np.load(triplets_path, allow_pickle=True)

# # Split triplets into train, validation, and test sets
# train_triplets, temp_triplets = train_test_split(triplets, train_size=config['train_size'], random_state=config['random_state'])
# val_triplets, test_triplets = train_test_split(temp_triplets, test_size=config['test_size'] / (config['val_size'] + config['test_size']), random_state=config['random_state'])

# # Recreate DataLoaders with best parameters
# train_loader = triplets_to_dataloader(train_triplets, best_params['batch_size'])
# val_loader = triplets_to_dataloader(val_triplets, best_params['batch_size'])
# test_loader = triplets_to_dataloader(test_triplets, best_params['batch_size'])

# Convert to DataFrame for easier manipulation
triplet_df = pd.DataFrame(triplets, columns=['group_id', 'anchor', 'positive', 'negative'])

# Group-based splitting
gss = GroupShuffleSplit(n_splits=1, train_size=config['train_size'], random_state=config['random_state'])
for train_idx, temp_idx in gss.split(triplet_df, groups=triplet_df['group_id']):
    train_triplets = triplet_df.iloc[train_idx]
    temp_triplets = triplet_df.iloc[temp_idx]

# Further split temp_triplets into validation and test
gss_temp = GroupShuffleSplit(n_splits=1, test_size=config['test_size'] / (config['val_size'] + config['test_size']), random_state=42)
for val_idx, test_idx in gss_temp.split(temp_triplets, groups=temp_triplets['group_id']):
    val_triplets = temp_triplets.iloc[val_idx]
    test_triplets = temp_triplets.iloc[test_idx]

# Convert back to list format if needed
train_triplets_list = train_triplets[['anchor', 'positive', 'negative']].values.tolist()
val_triplets_list = val_triplets[['anchor', 'positive', 'negative']].values.tolist()
test_triplets_list = test_triplets[['anchor', 'positive', 'negative']].values.tolist()

train_loader = triplets_to_dataloader(train_triplets_list, batch_size=32)
val_loader = triplets_to_dataloader(val_triplets_list, batch_size=32)
test_loader = triplets_to_dataloader(test_triplets_list, batch_size=32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ———————————————————————————————————————— W&B INITIALIZATION ——————————————————————————————————————————

wandb.init(project="Xception-Basic-Split-Train-Val-Test", config={
    "random_state": config['random_state'],
    "train_size": config['train_size'],
    "val_size": config['val_size'],
    "test_size": config['test_size'],
    "beta": config['beta'],
    "optimizer": best_params['optimizer'],
    "batch_size": best_params['batch_size'],
    "learning_rate": best_params['learning_rate'],
    "weight_decay": best_params['weight_decay'],
    "momentum": best_params.get('momentum', None),  # Optional for Adam
    "num_epochs": best_params['num_epochs']
})

# ———————————————————————————————————————— MODEL AND OPTIMIZER SETUP ——————————————————————————————————————————

model = get_embedding_model(config['num_classes']).to(device)
criterion = TripletLoss().to(device)

if best_params['optimizer'] == 'Adam':
    optimizer = Adam(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
elif best_params['optimizer'] == 'SGD':
    optimizer = SGD(model.parameters(), lr=best_params['learning_rate'], momentum=best_params['momentum'], weight_decay=best_params['weight_decay'])

# ———————————————————————————————————————— TRAINING LOOP ——————————————————————————————————————————

print('Start training with best hyperparameters obtained from wandb')
for epoch in range(best_params['num_epochs']):
    model.train()
    print(f'Starting epoch {epoch}')
    train_loss = 0.0

    for i, (anchor, positive, negative) in enumerate(train_loader):
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        optimizer.zero_grad()
        anchor_embedding, positive_embedding, negative_embedding = model(anchor, positive, negative)
        loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Calculate average train loss
    avg_train_loss = train_loss / len(train_loader)

    # Switch to evaluation mode
    model.eval()

    # Evaluate on training set
    train_eer, train_eer_threshold, train_fbeta, train_ks_statistic, train_optimal_threshold, train_tp, train_fp, train_fn, train_tn = evaluate(train_loader, model, config['beta'])

    # Evaluate on validation set
    val_loss = 0.0
    with torch.no_grad():
        for anchor, positive, negative in val_loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_embedding, positive_embedding, negative_embedding = model(anchor, positive, negative)
            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
            val_loss += loss.item()

    # Calculate validation metrics (e.g., EER, F-beta)
    val_eer, val_eer_threshold, val_fbeta, val_ks_statistic, val_optimal_threshold, val_tp, val_fp, val_fn, val_tn = evaluate(val_loader, model, config['beta'])

    # Calculate average validation loss
    avg_val_loss = val_loss / len(val_loader)

    # Log train and validation results to W&B
    wandb.log({
        "epoch": epoch,

        "train_loss": avg_train_loss,
        "train_eer": train_eer,
        "train_eer_threshold": train_eer_threshold,
        "train_fbeta": train_fbeta,
        "train_ks_statistic": train_ks_statistic,
        "train_optimal_threshold": train_optimal_threshold,
        "train_tp": train_tp,
        "train_fp": train_fp,
        "train_fn": train_fn,
        "train_tn": train_tn,

        "val_loss": avg_val_loss,
        "val_eer": val_eer,
        "val_eer_threshold": val_eer_threshold,
        "val_fbeta": val_fbeta,
        "val_ks_statistic": val_ks_statistic,
        "val_optimal_threshold": val_optimal_threshold,
        "val_tp": val_tp,
        "val_fp": val_fp,
        "val_fn": val_fn,
        "val_tn": val_tn
    })

    # Switch back to training mode
    model.train()


# Save the model
directory = os.path.dirname(config['model_save_path'])
if not os.path.exists(directory):
    os.makedirs(directory)

torch.save(model.state_dict(), config['model_save_path'])
print("Model saved")

# ———————————————————————————————————————— EVALUATION ON TEST DATA ——————————————————————————————————————————

model.eval()

test_eer, test_eer_threshold, test_fbeta, test_ks_statistic, test_optimal_threshold, test_tp, test_fp, test_fn, test_tn = evaluate(test_loader, model, config['beta'])

print(f"EER: {test_eer:.4f} at threshold: {test_eer_threshold:.4f}")
print(f"F-beta: {test_fbeta:.4f}")
print(f"KS Statistic: {test_ks_statistic:.4f} at threshold: {test_optimal_threshold:.4f}")
print(f"TP: {test_tp}, FP: {test_fp}, FN: {test_fn}, TN: {test_tn}")

# Log final test metrics to W&B
wandb.log({
    "test_eer": test_eer,
    "test_eer_threshold": test_eer_threshold,
    "test_fbeta": test_fbeta,
    "test_ks_statistic": test_ks_statistic,
    "test_optimal_threshold": test_optimal_threshold,
    "test_tp": test_tp,
    "test_fp": test_fp,
    "test_fn": test_fn,
    "test_tn": test_tn
})

# Save evaluation results to a CSV file
results = pd.DataFrame({
    "eer": [test_eer],
    "eer_threshold": [test_eer_threshold],
    "fbeta": [test_fbeta],
    "ks_statistic": [test_ks_statistic],
    "optimal_threshold": [test_optimal_threshold],
    "TP": [int(test_tp)],
    "FP": [int(test_fp)],
    "FN": [int(test_fn)],
    "TN": [int(test_tn)]
})

results.to_csv('results.csv', index=False)
print("Results saved to results.csv")

# Finish W&B run
wandb.finish()
