# %%
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Import sampling methods
from BHSIG260_Bengali_Sampling import sample_dataset_BHSig260_Bengali
from BHSIG260_Hindi_Sampling import sample_dataset_BHSig260_Hindi
from Cedar_sampling import sample_dataset_cedar
from Real_Fake_Data_Sampling import sample_dataset_Real_Fake_Data
from Signature_Verification_sampling import sample_signature_verification_dataset

# Set project root directory and add to Python path
project_root = Path.cwd()
sys.path.append(str(project_root))

# Import utility functions for image preprocessing
import img_preprocessing_util_functions as img_utils

def create_output_folder(base_output_dir, seed, switches, hyperparams):
    """
    Creates a structured output folder based on preprocessing switches and hyperparameters.
    """
    seed_dir = os.path.join(base_output_dir, f"seed_{seed}")
    for method in switches:
        method_dir = f"{method}_enabled_{'_'.join(f'{k}{v}' for k, v in hyperparams[method].items())}" if switches[method] else f"{method}_disabled"
        seed_dir = os.path.join(seed_dir, method_dir)
    os.makedirs(seed_dir, exist_ok=True)
    return seed_dir

def preprocess_image(image_path, steps, switches, hyperparams):
    """
    Applies a sequence of preprocessing steps to an image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    for step in steps:
        if switches.get(step, False):
            if step == 'clahe':
                clahe = cv2.createCLAHE(clipLimit=hyperparams[step]['clipLimit'], tileGridSize=(hyperparams[step]['tileGridSize'], hyperparams[step]['tileGridSize']))
                image = clahe.apply(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            elif step == 'gaussian_blur':
                image = cv2.GaussianBlur(image, (5, 5), hyperparams[step]['sigma'])
            elif step == 'adaptive_threshold':
                image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                              blockSize=hyperparams[step]['blockSize'], C=hyperparams[step]['C'])
            elif step == 'skeletonize':
                image = img_utils.skeletonize_image(image)
            elif step == 'augment':
                image = img_utils.augment_image(image)
    return image

def preprocess_data_from_seed(seed_path, base_output_dir, seed, steps, switches, hyperparams):
    """
    Processes all images in a dataset seed folder and stores them in a structured output directory.
    """
    preprocessed_data = []
    label_mapping = {'true': 0, 'forge': 1}
    image_data = img_utils.load_images_from_directory(seed_path)

    for idx, (img_path, label, person_id) in enumerate(image_data):
        img_type = 'true' if label == 0 else 'forge'
        output_folder = create_output_folder(base_output_dir, seed, switches, hyperparams)
        processed_image = preprocess_image(img_path, steps, switches, hyperparams)
        save_path = os.path.join(output_folder, f'{person_id}_{img_type}_{idx+1}.png')
        cv2.imwrite(save_path, processed_image)
        preprocessed_data.append((processed_image, label, person_id))

    # Return preprocessed data for further use
    return preprocessed_data

def run_sampling_methods_with_preprocessing(methods, params, base_output_dir, steps, switches, hyperparams, save_triplets=False):
    """
    Executes sampling methods with parameters, applies preprocessing, and saves outputs in structured format.
    """
    all_preprocessed_data = []
    for method, method_name in methods:
        method_params = params.get(method_name)
        if method_params:
            print(f"Running {method_name} with parameters: {method_params}")
            method_output_dir = os.path.join(base_output_dir, f"{method_name}_Dataset")
            os.makedirs(method_output_dir, exist_ok=True)
            method(**method_params)
            print(f"{method_name} sampling completed. Proceeding with preprocessing...\n")

            dataset_path = os.path.join(base_output_dir, f"{method_name}_Dataset")
            all_seeds = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
            for seed in all_seeds:
                seed_path = os.path.join(dataset_path, seed)
                preprocessed_data = preprocess_data_from_seed(seed_path, base_output_dir, seed, steps, switches, hyperparams)
                all_preprocessed_data.extend(preprocessed_data)
        else:
            print(f"No parameters provided for {method_name}. Skipping...\n")

    # Convert accumulated data to a DataFrame similar to Xiner's output
    images = np.array([item[0] for item in all_preprocessed_data])
    labels = np.array([item[1] for item in all_preprocessed_data])
    person_ids = np.array([item[2] for item in all_preprocessed_data])
    preprocessed_df = pd.DataFrame({'person_id': person_ids, 'image': list(images), 'label': labels})

    # Save the DataFrame and optionally save triplets
    preprocessed_pickle_path = Path(base_output_dir) / 'preprocessed_signature_df.pkl'
    preprocessed_df.to_pickle(preprocessed_pickle_path)
    print(f'Saved preprocessed_df to {preprocessed_pickle_path}')

    if save_triplets:
        triplets_save_path = Path(base_output_dir) / 'preprocessed_triplets.npy'
        img_utils.save_triplets(preprocessed_df, triplets_save_path)
        print(f'Saved preprocessed triplets to {triplets_save_path}')

# Define methods, parameters, and other configurations
methods = [
    (sample_dataset_cedar, 'CEDAR'),
    (sample_signature_verification_dataset, 'Signature_Verification'),
    (sample_dataset_BHSig260_Bengali, 'BHSig260_Bengali'), 
    (sample_dataset_BHSig260_Hindi, 'BHSig260_Hindi'), 
    (sample_dataset_Real_Fake_Data, 'Real_Fake_Data')
]

params = {
    'CEDAR': {
        'path': '/Users/hongmingfu/Desktop/Brown University/DATA2050/Cedar',
        'num_samples': 5,
        'random_seeds': [123, 456, 789]
    },
    'Signature_Verification': {
        'path': '/Users/hongmingfu/Desktop/Brown University/DATA2050/Signature_Verification_Dataset',
        'num_samples': 5,
        'random_seeds': [123, 456, 789]
    },
    'BHSig260_Bengali': {
        'path': '/Users/hongmingfu/Desktop/Brown University/DATA2050/BHSig260/Bengali',
        'num_samples': 10,           # Adjust based on available data and requirements
        'random_seeds': [101, 202, 303]
    },
    'BHSig260_Hindi': {
        'path': '/Users/hongmingfu/Desktop/Brown University/DATA2050/BHSig260/Hindi',
        'num_samples': 8,            # Adjust based on available data and requirements
        'random_seeds': [404, 505, 606]
    },
    'Real_Fake_Data': {
        'path': '/Users/hongmingfu/Desktop/Brown University/DATA2050/Real_Fake_Signature',
        'num_samples': 6,            # Adjust based on available data and requirements
        'random_seeds': [707, 808, 909]
    }
}

# Define preprocessing steps order, switches, and hyperparameters
steps = ['clahe', 'gaussian_blur', 'adaptive_threshold', 'skeletonize', 'augment']  # Ordered steps
switches = {
    'clahe': True,
    'adaptive_threshold': True,
    'gaussian_blur': False,
    'skeletonize': True,
    'augment': False
}
hyperparams = {
    'clahe': {'clipLimit': 2.0, 'tileGridSize': 8},
    'adaptive_threshold': {'blockSize': 11, 'C': 2},
    'gaussian_blur': {'sigma': 1.5}
}

if __name__ == "__main__":
    base_output_dir = "./processed_data/"
    run_sampling_methods_with_preprocessing(methods, params, base_output_dir, steps, switches, hyperparams, save_triplets=True)
    
    # Load and check preprocessed data
    preprocessed_pickle_path = Path(base_output_dir) / 'preprocessed_signature_df.pkl'
    preprocessed_df = pd.read_pickle(preprocessed_pickle_path)
    print("Preprocessed DataFrame loaded successfully. Verifying data...")

    # Validate data before triplet generation
    if preprocessed_df.empty or preprocessed_df['person_id'].nunique() < 2:
        print("Error: Not enough data to create triplets. Ensure that the sampling methods produced valid data.")
    else:
        triplets_save_path = Path(base_output_dir) / 'preprocessed_triplets.npy'
        img_utils.save_triplets(preprocessed_df, triplets_save_path)
        print(f'Saved preprocessed triplets to {triplets_save_path}')

    print("All sampling and preprocessing tasks executed successfully!")



# %%



