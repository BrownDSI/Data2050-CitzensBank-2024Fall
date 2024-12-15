# EfficientNet-b0 Triplet Loss Model

This repository contains an implementation of the EfficientNet-b0 model trained with triplet loss. It provides a modular approach for loading data, defining the model architecture, performing hyperparameter optimization, and evaluating performance metrics. The repository is designed for easy replication, experimentation, and customization.



## Features

- **EfficientNet-b0 Architecture**: Leverages a lightweight and efficient convolutional neural network for image-based tasks.
- **Triplet Loss**: Optimized for comparing triplets of data (anchor, positive, and negative) to enhance feature learning.
- **Hyperparameter Optimization**: Uses **Weights & Biases (wandb)** for parameter tuning and experiment tracking.
- **Comprehensive Metrics**: Supports evaluation using metrics like Equal Error Rate (EER), F-beta score, and KS statistics.


## Table of Contents

1. [File Descriptions](#file-descriptions)
    - [FixedParameters.yaml](#1-fixedparametersyaml)
    - [load_data.py](#2-load_datapy)
    - [model.py](#3-modelpy)
    - [metric.py](#4-metricpy)
    - [sweep_config.yaml](#5-sweep_configyaml)
    - [parameter_search.py](#6-parameter_searchpy)
    - [retrieve_best_params.py](#7-retrieve_best_paramspy)
    - [BestParameters.yaml](#8-bestparametersyaml)
    - [main.py](#9-mainpy)
2. [How to Train the EfficientNet-b0 Model](#how-to-train-the-efficientNet-b0-Model)
3. [Dependencies](#dependencies)
4. [Acknowledgments](#acknowledgments)


<!-- ## File Descriptions -->
<!-- 
### 1. **FixedParameters.yaml**
- Contains all fixed hyperparameters for the EfficientNet-b0 model.
- Includes parameters such as:
  - `random_state`
  - `train_size`
  - `beta`
  - `num_classes`

---

### 2. **load_data.py**
- Loads triplet data (anchor, positive, negative) for train, validation, and test datasets.
- Converts triplets into a `DataLoader` format for batch processing. -->


## File Descriptions

#### 1. **`FixedParameters.yaml`**
- **Purpose**: Contains all fixed configuration parameters for the EfficientNet-b0 model.
- **Examples of Parameters**: 
  - beta
  - train_size
  - num_classes
  - random_state

---

#### 2. **`load_data.py`**
- **Purpose**: Loads triplet data (anchor, positive, negative) for train, validation, and test datasets.
- **Functionality**: Converts triplets into PyTorch `DataLoader` format for efficient batch processing.

---

#### 3. **`model.py`**
- **Purpose**: Defines the EfficientNet-b0 model architecture.
- **Additional Feature**: Includes the implementation of **triplet loss** for optimizing the model during training.

---

#### 4. **`metric.py`**
- **Purpose**: Computes evaluation metrics to assess model performance.
- **Supported Metrics**:
  - Equal Error Rate (EER) and EER Threshold
  - F-beta Score
  - KS Statistics
  - True Positives (TP), False Positives (FP), False Negatives (FN), True Negatives (TN)

---

#### 5. **`sweep_config.yaml`**
- **Purpose**: Specifies the parameter ranges for hyperparameter optimization.
- **Example Parameters**:
  - Batch size
  - Learning rate
  - Optimizer
  - Num epochs
  - Any additional tunable parameters
- **Usage**: Used by `parameter_search.py` to create a wandb sweep for exploring the parameter space.
---

#### 6. **`parameter_search.py`**
- **Purpose**: Automates hyperparameter tuning using wandb.
- **How It Works**:
  - Initializes a wandb sweep based on `sweep_config.yaml`.
  - Iteratively tests combinations of parameters like batch size, learning rate, and others.
  - Tracks and logs results, including metrics such as EER, KS statistics, F-beta score, and training loss.
- **Output**:
  - Saves a CSV file named `history.csv` in the current directory, which includes:
    - Parameter combinations for each trial.
    - Results for metrics and losses associated with each trial.

---

#### 7. **`retrieve_best_params.py`**
- **Purpose**: Retrieves the best hyperparameters from wandb API.
- **Usage**:
  - Connects to the wandb API to fetch the optimal parameter combination.
  - Saves the parameters in `BestParameters.yaml`.

---

#### 8. **`BestParameters.yaml`**
- **Purpose**: Stores the best hyperparameter combination identified from the wandb sweep.
- **Usage**: Used during the model training phase (`main.py`) to load optimal parameter values.

---

#### 9. **`main.py`**
- **Purpose**: Core script for model training and evaluation.
- **Functionality**:
  - Loads the best parameters from `BestParameters.yaml`.
  - Trains the EfficientNet-b0 model on the training dataset.
  - Evaluates the model on the test dataset using metrics from `metric.py`.
  - Logs the results and metrics.
- **Output**:
  - Saves a file named `results.csv` in the current directory, containing evaluation metrics for the test dataset.


## How to Train the EfficientNet-b0 Model

### 1. Set Up wandb
- Create an account or log in at [wandb.ai](https://wandb.ai).
- Authenticate wandb in your terminal:

  ``
  wandb login
  ``

---

### 2. Navigate to the Project Directory
- Use the `cd` command to navigate to the folder:

  ``
  cd path/to/EfficientNet
  ``

---

### 3. Run Hyperparameter Search
- Execute the `parameter_search.py` script:

  ``
  python parameter_search.py
  ``
- This will initialize a wandb sweep.

---

### 4. Find and Set the Sweep ID
- Log in to your wandb account.
- Navigate to the project containing your sweep and locate the Sweep ID.
- Replace the placeholder Sweep ID in `retrieve_best_params.py`:
  ```python
  sweep_id = "your_sweep_id"  # Replace with your Sweep ID
  ```

---

### 5. Retrieve Best Parameters
- Run the script to fetch the optimal parameters:

  ``
  python retrieve_best_params.py
  ``

---

### 6. Train and Evaluate the Model
- Run the training script:

  ``
  python main.py
  ``
- This will train the model and save evaluation results to `results.csv`.

---

## Outputs

1. **`history.csv`**: Logs hyperparameter sweep trials, including parameters and metrics.
2. **`BestParameters.yaml`**: Stores the optimal hyperparameters for training.
3. **`results.csv`**: Final evaluation metrics from the test dataset.

---

## Dependencies

### Install Required Libraries
Run the following command to install all required dependencies:

``
pip install -r requirements.txt
``

<!-- ## Common Use Cases

- Training the EfficientNet-b0 model for tasks involving triplet-based learning.
- Experimenting with different parameter combinations for optimized performance.
- Analyzing and comparing evaluation metrics for improved decision-making.


## Contributing

We welcome contributions! If you'd like to improve this project, please open a pull request or submit an issue for discussion.



## License

This project is licensed under the MIT License. See the `LICENSE` file for details. -->



## Acknowledgments

This project leverages:
- **EfficientNet-b0**: A scalable and efficient deep learning architecture.
- **Weights & Biases (wandb)**: For experiment tracking and hyperparameter optimization.
- **PyTorch**: A powerful deep learning framework.

Special thanks to the open-source community for tools and resources that made this project possible.
