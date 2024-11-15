import os
import cv2 
import numpy as np
import pandas as pd
from sampling_BHSIG260_Bengali import sample_dataset_bhsig260_bengali
from sampling_BHSIG260_Hindi import sample_dataset_bhsig260_hindi
from sampling_Cedar import sample_dataset_cedar
from sampling_Real_Fake_Data import sample_dataset_real_fake
from sampling_Signature_Verification import sample_signature_verification_dataset
from sampling_Hansig import sample_dataset_hansig
from pathlib import Path
from sampling import run_sampling_methods
from preprocessing import preprocess_sampling_data
import img_preprocessing_util_functions as img_utils
from img_preprocessing_util_functions import create_preprocessed_signature_df




# Define sampling methods with corresponding method names
methods = [
    (sample_dataset_cedar, 'CEDAR'),
    (sample_signature_verification_dataset, 'Signature_Verification'),
    (sample_dataset_bhsig260_bengali, 'BHSig260_Bengali'), 
    (sample_dataset_bhsig260_hindi, 'BHSig260_Hindi'), 
    (sample_dataset_real_fake, 'Real_Fake_Data'),
    (sample_dataset_hansig, 'Hansig')
]

# Define preprocessing steps, switches, and hyperparameters
steps = ['clahe', 'gaussian_blur', 'adaptive_threshold', 'skeletonize', 'augment']
switches = {
    'grayscale': True,
    'grey_to_binary': True,
    'clahe': True,
    'gaussian_blur': False,
    'adaptive_threshold': True,
    'skeletonize': True,
    'augment': False
}
hyperparams = {
    'clahe': {'clipLimit': 2.0, 'tileGridSize': 8},
    'adaptive_threshold': {'blockSize': 11, 'C': 2},
    'gaussian_blur': {'sigma': 1.5}
}

if __name__ == "__main__":

    # Where to extract five data folders 
    base_data_path = "../../data"
    
   # Where to store sampled images, preprocessed images and pkl,npy file
    
    main_output_dir = '../preprocessing/preprocessed_dataset'

    sampling_output_dir = os.path.join(main_output_dir, 'sampled')
    preprocessing_output_dir = os.path.join(main_output_dir, 'preprocessed')

    # Define parameters for each dataset with sampling output in sampled folder
    params = {
        'CEDAR': {
            'data_path': f"{base_data_path}/Cedar",
            'destination_path': f"{sampling_output_dir}/Cedar_Sampled",
            'num_individuals': 10,
            'seeds': [123],
            'number_of_signatures': 5,
            'language': 'English'
        },
        'Signature_Verification': {
            'data_path': f"{base_data_path}/Signature_Verification_Dataset",
            'destination_path': f"{sampling_output_dir}/Signature_Verification_Sampled",
            'num_individuals': 10,
            'seed': 123,
            'number_of_signatures': 5,
            'language': 'English'
        },
        'BHSig260_Bengali': {
            'data_path': f"{base_data_path}/BHSig260/Bengali",
            'destination_path': f"{sampling_output_dir}/BHSig260_Bengali_Sampled",
            'num_individuals': 10,
            'seed': 101,
            'number_of_signatures': 5,
            'language': 'Bengali'
        },
        'BHSig260_Hindi': {
            'data_path': f"{base_data_path}/BHSig260/Hindi",
            'destination_path': f"{sampling_output_dir}/BHSig260_Hindi_Sampled",
            'num_individuals': 10,
            'seed': 404,
            'number_of_signatures': 5,
            'language': 'Hindi'
        },
        'Real_Fake_Data': {
            'data_path': f"{base_data_path}/Real_Fake_Signature/Signature Images",
            'destination_path': f"{sampling_output_dir}/Real_Fake_Signature_Sampled",
            'num_individuals': 9,
            'seed': 707,
            'number_of_signatures': 5,
            'language': 'Turkish'
        },
        'Hansig': {
            'data_path': f"{base_data_path}/Hansig",
            'destination_path': f"{sampling_output_dir}/Hansig_Sampled",
            'num_individuals': 10,
            'seed': 555,
            'number_of_signatures': 5,
            'language': 'Chinese'
        }
    }

    
    sampled_df = run_sampling_methods(methods, params, sampling_output_dir)

    preprocessed_df = preprocess_sampling_data(
        sampling_output_dir, 
        preprocessing_output_dir, 
        steps, 
        switches, 
        hyperparams
    )
    # model can be in ['EfficientNet', 'VGG16', 'ResNet']
    models = ['EfficientNet', 'VGG16']
    for model in models:
        model_df = create_preprocessed_signature_df(preprocessed_df, model, preprocessing_output_dir)
        # Generate and save triplets
        num_triplets = 100  # Set number of triplets to generate
        triplets = img_utils.create_triplets(model_df, num_triplets)
        triplet_save_path = Path(preprocessing_output_dir) / model / "preprocessed_triplets.npy"
        np.save(triplet_save_path, np.array(triplets))
        print(f"Saved triplets for {model} to {triplet_save_path}")

    
