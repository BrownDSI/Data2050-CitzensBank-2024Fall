import wandb
import yaml

# Define the project and sweep details
entity = "zhaoxiner1129-brown-university"  # Replace with your W&B username or team name
project_name = "Xception-LargeDataSet" # Replace with your project name
sweep_id = "8md863zb"  # Replace with your actual sweep ID

# Initialize W&B API
api = wandb.Api()

# Get the sweep
sweep_path = f"{entity}/{project_name}/{sweep_id}"
sweep = api.sweep(sweep_path)

# Retrieve all runs in the sweep
runs = sweep.runs

# Sort runs by the target metric (EER) in ascending order
best_run = sorted(runs, key=lambda run: run.summary.get("eer", float("inf")))[0]

# Retrieve the best parameters
best_params = best_run.config

print("Best parameters:")
print(best_params)

# Save the best parameters to a YAML file
output_file = "BestParameters.yaml"
with open(output_file, "w") as yaml_file:
    yaml.dump(best_params, yaml_file, default_flow_style=False)
print(f"Best parameters saved to {output_file}")