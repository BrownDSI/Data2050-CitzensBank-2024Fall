import os
import pandas as pd
import wandb

# Get the absolute path to the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
history_path = os.path.join(script_dir, 'history.csv')

# Load the history DataFrame
history_df = pd.read_csv(history_path)

# Iterate through each row and log as a separate run
for _, row in history_df.iterrows():
    # Initialize wandb for each row
    wandb.init(project='project_1_exp_1_241108', reinit=True)
    
    # Prepare the configuration
    config = {
        'batch_size': row['batch_size'],
        'learning_rate': row['learning_rate'],
        'optimizer': row['optimizer'],
        'momentum': row['momentum'] if 'momentum' in row else None,
        'weight_decay': row['weight_decay'],
        'num_epochs': row['num_epochs']
    }
    
    # Log the configuration
    wandb.config.update(config)
    
    # Log the metrics
    wandb.log({"eer": row['eer'], "fbeta": row['fbeta']})

    # Finish the run
    wandb.finish()
