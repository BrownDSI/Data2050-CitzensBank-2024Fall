import sys
import os
import json
import torch
import yaml
import optuna
import joblib
import pandas as pd
import numpy as np
from torch.optim import Adam, SGD
from sklearn.model_selection import train_test_split
from metric import evaluate
from model import get_embedding_model, TripletLoss
from load_data import triplets_to_dataloader


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

    
    # Initialize a list to log the loss for each epoch
    train_loss_log = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0  # To accumulate loss over the epoch
        print(f'Starting epoch {epoch}')
        for i, (anchor, positive, negative) in enumerate(train_loader):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            optimizer.zero_grad()
            anchor_embedding, positive_embedding, negative_embedding = model(anchor, positive, negative)
            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
            loss.backward()
            optimizer.step()
        
            # Accumulate the loss for this epoch
            epoch_loss += loss.item()
        # Calculate and log the average loss for the epoch
        avg_loss = epoch_loss / len(train_loader)
        train_loss_log.append(avg_loss)
        print(f'Epoch {epoch}: Average Loss = {avg_loss:.4f}')

    eer, eer_threshold, fbeta, ks_statistic_val, optimal_threshold_val, tp, fp, fn, tn = evaluate(val_loader, model, beta=config['beta'])

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
        'eer_threshold': eer_threshold,
        'fbeta': fbeta,
        'ks_statistic_val': ks_statistic_val,
        'optimal_threshold_val': optimal_threshold_val,
        'loss': avg_loss
    }
    
    result_df = pd.DataFrame([result])
    history_df = pd.concat([history_df, result_df], ignore_index=True)
    
    return eer

if __name__=="__main__":

    # ———————————————————————————————————————— START TO SEARCH THE BEST HYPERPARAMETER COMBINATION ——————————————————————————————————————————
    current_dir = os.path.dirname(__file__)
    src_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'src'))
    sys.path.append(src_dir)

    # Load exp_config
    with open(os.path.join(current_dir, 'exp_config.json')) as config_file:
        config = json.load(config_file)

    # Load preprocessed triplets
    triplets_path = os.path.abspath("../preprocessing/preprocessed_dataset/preprocessed/EfficientNet/preprocessed_triplets.npy")
    # triplets_path = os.path.abspath(os.path.join(current_dir, '../data-preprocessing', '/EfficientNetb0', 'preprocessed_triplets.npy'))
    triplets = np.load(triplets_path, allow_pickle=True)

    # Split triplets into train, validation, and test sets
    train_triplets, temp_triplets = train_test_split(triplets, train_size=config['train_size'], random_state=config['random_state'])
    val_triplets, test_triplets = train_test_split(temp_triplets, test_size=config['test_size'] / (config['val_size'] + config['test_size']), random_state=config['random_state'])


    # Create DataLoaders
    train_loader = triplets_to_dataloader(train_triplets, config['batch_size'])
    val_loader = triplets_to_dataloader(val_triplets, config['batch_size'])
    test_loader = triplets_to_dataloader(test_triplets, config['batch_size'])

    # Initialize an empty DataFrame to store results
    history_df = pd.DataFrame()

    # Create and optimize study
    study = optuna.create_study(sampler=optuna.samplers.GPSampler(), direction='minimize')
    print('Start training to find best hyperparameters')
    study.optimize(objective, n_trials=20)

    # Print best hyperparameters
    print(f'Best hyperparameters based on eer: {study.best_params}')

    joblib.dump(study, os.path.join(current_dir, 'optuna_study.pkl'))

    # Save the accumulated history to CSV
    history_path = os.path.join(current_dir, 'history.csv')
    history_df.to_csv(history_path, index=False)
    print(f'history log saved to {history_path}')


    # ———————————————————————————————————————— USE THE BEST HYPERPARAMETER TO LOAD DATA ——————————————————————————————————————————

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

    # ———————————————————————————————————————— START TO RETRAIN THE MODEL ——————————————————————————————————————————

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

    # Get the directory part of the path
    directory = os.path.dirname(config['model_save_path'])

    # Check if the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory) # Create the directory (and any intermediate directories if they don't exist)

    torch.save(model.state_dict(), config['model_save_path'])
    print("model saved")