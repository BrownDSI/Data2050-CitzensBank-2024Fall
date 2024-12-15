import sys
import os
# import json
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


def objective():
    # Initialize W&B run
    wandb.init()  # W&B automatically reads the configuration from the sweep
    config = wandb.config  # Access sweep-configured hyperparameters

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model, criterion, and optimizer setup
    model = get_embedding_model(fixed_config['num_classes']).to(device)
    criterion = TripletLoss().to(device)
    if config.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        optimizer = SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)

    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0
        print(f'Starting epoch {epoch}')
        for i, (anchor, positive, negative) in enumerate(train_loader):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            optimizer.zero_grad()
            anchor_embedding, positive_embedding, negative_embedding = model(anchor, positive, negative)
            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        wandb.log({"epoch": epoch, "train_loss": avg_loss})

    # Validation
    eer, eer_threshold, fbeta, ks_statistic_val, optimal_threshold_val, tp, fp, fn, tn = evaluate(val_loader, model, beta=fixed_config['beta'])

    # if eer is None:
    #     wandb.log({"error": "EER calculation failed"})

    # Log metrics and hyperparameters to W&B
    result = {
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "optimizer": config.optimizer,
        "momentum": config.momentum,
        "weight_decay": config.weight_decay,
        "num_epochs": config.num_epochs,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "fbeta": fbeta,
        "ks_statistic_val": ks_statistic_val,
        "optimal_threshold_val": optimal_threshold_val,
        "True_Positives": tp,
        "False_Positives": fp,
        "False_Negatives": fn,
        "True_Negatives": tn,
        "final_loss": avg_loss
    }
    wandb.log(result)

    # Optionally save to CSV for local tracking (if needed)
    result_df = pd.DataFrame([result])
    history_path = os.path.join(current_dir, 'history.csv')
    if os.path.exists(history_path):
        history_df = pd.read_csv(history_path)
        history_df = pd.concat([history_df, result_df], ignore_index=True)
    else:
        history_df = result_df
    history_df.to_csv(history_path, index=False)
    print(f"History log saved to {history_path}")

    return eer  # Metric to minimize


if __name__ == "__main__":
    # Set up paths and data
    current_dir = os.path.dirname(__file__)
    src_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'src'))
    sys.path.append(src_dir)

    # Load fixed configurations
    with open("FixedParameters.yaml", "r") as file:
        fixed_config = yaml.safe_load(file)

    # Load preprocessed triplets
    triplets_path = os.path.abspath("../preprocessing/preprocessed_dataset/preprocessed/EfficientNet/preprocessed_triplets.npy")
    triplets = np.load(triplets_path, allow_pickle=True)

    # # Split triplets into train, validation, and test sets
    # train_triplets, temp_triplets = train_test_split(triplets, train_size=fixed_config['train_size'], random_state=fixed_config['random_state'])
    # val_triplets, test_triplets = train_test_split(temp_triplets, test_size=fixed_config['test_size'] / (fixed_config['val_size'] + fixed_config['test_size']), random_state=fixed_config['random_state'])

    # global train_loader, val_loader, test_loader
    # train_loader = triplets_to_dataloader(train_triplets, batch_size=32)
    # val_loader = triplets_to_dataloader(val_triplets, batch_size=32)
    # test_loader = triplets_to_dataloader(test_triplets, batch_size=32)

    
    # Convert to DataFrame for easier manipulation
    triplet_df = pd.DataFrame(triplets, columns=['group_id', 'anchor', 'positive', 'negative'])

    # Group-based splitting
    gss = GroupShuffleSplit(n_splits=1, train_size=fixed_config['train_size'], random_state=fixed_config['random_state'])
    for train_idx, temp_idx in gss.split(triplet_df, groups=triplet_df['group_id']):
        train_triplets = triplet_df.iloc[train_idx]
        temp_triplets = triplet_df.iloc[temp_idx]

    # Further split temp_triplets into validation and test
    gss_temp = GroupShuffleSplit(n_splits=1, test_size=fixed_config['test_size'] / (fixed_config['val_size'] + fixed_config['test_size']), random_state=42)
    for val_idx, test_idx in gss_temp.split(temp_triplets, groups=temp_triplets['group_id']):
        val_triplets = temp_triplets.iloc[val_idx]
        test_triplets = temp_triplets.iloc[test_idx]

    # Convert back to list format if needed
    train_triplets_list = train_triplets[['anchor', 'positive', 'negative']].values.tolist()
    val_triplets_list = val_triplets[['anchor', 'positive', 'negative']].values.tolist()
    test_triplets_list = test_triplets[['anchor', 'positive', 'negative']].values.tolist()

    global train_loader, val_loader, test_loader
    train_loader = triplets_to_dataloader(train_triplets_list, batch_size=32)
    val_loader = triplets_to_dataloader(val_triplets_list, batch_size=32)
    test_loader = triplets_to_dataloader(test_triplets_list, batch_size=32)

    # Initialize and run W&B Sweep
    def sweep_objective():
        objective()

    # Start the sweep using the configuration from sweep_config.yaml
    with open("sweep_config.yaml", "r") as file:
        sweep_config = yaml.safe_load(file)

    # Pass the dictionary to wandb.sweep()
    sweep_id = wandb.sweep(sweep_config, project="EfficientNetb0-Finalized-param")
    wandb.agent(sweep_id, sweep_objective)
