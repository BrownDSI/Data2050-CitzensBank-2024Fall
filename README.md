# Fraudulent Signature Verification

This repository contains the source code and data for the "Fraudulent Signature Verification" project conducted at Citizens Bank in collaboration with Brown University. The project aims to address the challenges of forged signature detection by leveraging advanced machine learning techniques.


## Project Structure
```
CitizensBank-Fraud-Signature-Detection
|-- data/                                # Datasets used for training and testing
|   |-- BHSig260/                        # Bengali and Hindi signatures
|   |-- Cedar/                           # English signatures
|   |-- Hansig/                          # Chinese signatures
|   |-- Real_Fake_Signature/             # Turkish signatures
|   |-- Signature_Verification_Dataset/  # Dutch signatures
|
|-- src/                                 # Source code for the project
|   |-- preprocessing/                   # Data preprocessing methods
|   |-- resnet50/                        # ResNet50 implementation
|   |-- vgg16/                           # VGG-16 implementation
|   |-- efficientnet_b0/                 # EfficientNet-b0 implementation
|   |-- inceptionv3/                     # Inception-v3 implementation
|   |-- xception/                        # Xception implementation
|   |-- demo/                            # demo for predicting an image of signature
```


## Collaborators
**Project Members:**
- Xiner Zhao
- Ruize Ma
- Hongming Fu
- Yi Sun

**Supervisors:**
- Dr. Shekhar Pradhan
- Ryan Murray

**Authors**
- Xiner Zhao

**Date:** December 18, 2024


## Problem Statement
Signatures are a trusted method for validating transactions, yet fraudulent signatures pose significant risks to financial institutions. This project addresses these risks by developing an automated pipeline for fraudulent signature verification. The system focuses on:

- **Financial Security:** Mitigating monetary losses due to fraudulent signatures.
- **Customer Trust:** Enhancing confidence in secure transactions.
- **Operational Efficiency:** Automating signature verification to reduce human error and workload.



## Features

- **Support for Multiple Architectures**: EfficientNet-b0, VGG-16, ResNet-50, Inception-v3, and Xception for diverse and robust feature learning.
- **Triplet Loss**: Optimized for comparing triplets of data (anchor, positive, and negative) to enhance feature learning.
- **Hyperparameter Optimization**: Uses **Weights & Biases (wandb)** for parameter tuning and experiment tracking.
- **Comprehensive Metrics**: Supports evaluation using metrics like Equal Error Rate (EER), F-beta score, and KS statistics.


## Table of Contents

1. [Dependencies](#dependencies)
2. [Datasets](#datasets)
3. [Data Preprocessing](#data-preprocessing)
4. [File Descriptions in Each Model Folder](#file-descriptions)
    - [FixedParameters.yaml](#1-fixedparametersyaml)
    - [load_data.py](#2-load_datapy)
    - [model.py](#3-modelpy)
    - [metric.py](#4-metricpy)
    - [sweep_config.yaml](#5-sweep_configyaml)
    - [parameter_search.py](#6-parameter_searchpy)
    - [retrieve_best_params.py](#7-retrieve_best_paramspy)
    - [BestParameters.yaml](#8-bestparametersyaml)
    - [main.py](#9-mainpy)
5. [How to Train the Model](#how-to-train-the-model)
6. [Outputs](#outputs)
7. [Future Improvements](#future-improvements)
8. [Contacts](#contact)


## Dependencies

### Install Required Libraries
Run the following command to install all required dependencies:

``
pip install -r requirements.txt
``


## Datasets

Our project uses 5 datasets from 6 languages to ensure robust model training and evaluation:

1. **BHSig260**:
   - **Languages**: Bengali, Hindi
   - **Samples**: 11,000 total signatures (genuine and forged)
   - **Source**: Public dataset for handwritten signature verification.

2. **Cedar**:
   - **Language**: English
   - **Samples**: 2,640 total signatures
   - **Source**: Commonly used dataset for signature verification research.

3. **HanSig**:
   - **Language**: Chinese
   - **Samples**: 10,200 total signatures
   - **Source**: Dataset for Chinese signature verification tasks.

4. **Real Fake Signatures**:
   - **Language**: Turkish
   - **Samples**: 2,812 total signatures
   - **Source**: Kaggle dataset for evaluating signature verification.

5. **Signature Verification Dataset**:
   - **Language**: Dutch
   - **Samples**: 2,064 total signatures
   - **Source**: Research dataset for signature verification.


## Data Preprocessing

Our preprocessing pipeline transforms raw signature images into a format suitable for deep learning models. Key steps include:

1. **Grayscale Conversion**:
   - Converts RGB images to grayscale to reduce computational complexity and focus on essential features.

2. **Binary Conversion**:
   - Applies Gaussian filtering for noise reduction and Otsu's thresholding for clear segmentation of the signature.

3. **Cropping to Margins**:
   - Removes unnecessary image margins, ensuring the focus is solely on the signature.

4. **Contrast Enhancement**:
   - **CLAHE** (Contrast Limited Adaptive Histogram Equalization) improves local contrast, particularly in uneven lighting conditions.

5. **Adaptive Thresholding**:
   - Dynamically adjusts pixel thresholding based on the surrounding region, effectively segmenting signatures in varying illumination.

6. **Denoising**:
   - Smooths out small artifacts and irregularities in binary images to enhance signature clarity.

7. **Skeletonization**:
   - Reduces binary images to their essential structure, emphasizing the core patterns of each signature.

8. **Data Augmentation**:
   - Enhances dataset diversity with techniques such as rotation, scaling, flipping, noise addition, and elastic distortions.

9. **Model-Specific Preprocessing**:
   - Resizes images and applies normalization specific to each model architecture (e.g., EfficientNet: 224x224, normalized to [0,1]).

10. **Switchable Preprocessing Steps**:
    - Configurable pipeline allows selective activation of steps like CLAHE, Gaussian blur, and skeletonization, making it adaptable to various datasets.


## File Descriptions in Each Model Folder

#### 1. **`FixedParameters.yaml`**
- **Purpose**: Contains all fixed configuration parameters for training any model.
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
- **Purpose**: General model architecture file, replaced with specific implementations (EfficientNet, VGG-16, ResNet-50, Inception-v3, Xception).
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
  - Trains the selected model on the training dataset.
  - Evaluates the model on the test dataset using metrics from `metric.py`.
  - Logs the results and metrics.
- **Output**:
  - Saves a file named `results.csv` in the current directory, containing evaluation metrics for the test dataset.




## How to Train the Model

### 1. Set Up wandb
- Create an account or log in at [wandb.ai](https://wandb.ai).
- Authenticate wandb in your terminal:

  ```bash
  wandb login
  ```

---

### 2. Navigate to the Project Directory
- Use the `cd` command to navigate to the folder:

  ```bash
  cd path/to/project
  ```
- For example:
  ```bash
  cd src/efficientnetb0
  ```

---

### 3. Run Hyperparameter Search
- Execute the `parameter_search.py` script:

  ```bash
  python parameter_search.py
  ```
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

  ```bash
  python retrieve_best_params.py
  ```

---

### 6. Train and Evaluate the Model
- Run the training script:

  ```bash
  python main.py
  ```
- This will train the model and save evaluation results to `results.csv`.



## Outputs

1. **`history.csv`**: Logs hyperparameter sweep trials, including parameters and metrics.
2. **`BestParameters.yaml`**: Stores the optimal hyperparameters for training.
3. **`results.csv`**: Final evaluation metrics from the test dataset.



## Future Improvements
- Expand datasets to include more languages.
- Enhance hardware for faster training and preprocessing.
- Experiment with additional deep learning architectures.

## Contact
For inquiries or feedback, please contact:

**Xiner Zhao**
- Email: xiner_zhao@brown.edu or zhaoxiner1129@gmail.com
- GitHub: [XXXXiner](https://github.com/XXXXiner)

