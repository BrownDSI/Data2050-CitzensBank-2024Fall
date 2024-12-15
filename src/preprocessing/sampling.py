import os
import pandas as pd
from pathlib import Path
import sys

# Import sampling methods
from sampling_BHSIG260_Bengali import sample_dataset_bhsig260_bengali
from sampling_BHSIG260_Hindi import sample_dataset_bhsig260_hindi
from sampling_Cedar import sample_dataset_cedar
from sampling_Real_Fake_Data import sample_dataset_real_fake
from sampling_Signature_Verification import sample_signature_verification_dataset
from sampling_Hansig import sample_dataset_hansig

# Set project root directory and add to Python path
project_root = Path.cwd()
sys.path.append(str(project_root))


def run_sampling_methods(methods, params, base_output_dir):
    all_sampling_data = []

    for method, method_name in methods:
        method_params = params.get(method_name)
        
        if method_params:
            print(f"Running {method_name} with parameters: {method_params}")
            method_output_dir = os.path.join(base_output_dir, f"{method_name}_Sampled")
            os.makedirs(method_output_dir, exist_ok=True)

            # Filter parameters for each method
            if method_name == 'CEDAR':
                filtered_params = {
                    'data_path': method_params['data_path'],
                    'destination_path': method_output_dir,
                    'num_individuals': method_params['num_individuals'],
                    'seed': method_params['seed'],  # Use the first seed in the list
                    'number_of_signatures': method_params['number_of_signatures']
                }
            elif method_name == 'Signature_Verification':
                filtered_params = {
                    'data_path': method_params['data_path'],
                    'destination_path': method_output_dir,
                    'num_individuals': method_params['num_individuals'],
                    'seed': method_params['seed'],
                    'number_of_signatures': method_params['number_of_signatures']
                }
            else:
                filtered_params = {
                    'data_path': method_params['data_path'],
                    'destination_path': method_output_dir,
                    'num_individuals': method_params['num_individuals'],
                    'seed': method_params['seed'],
                    'number_of_signatures': method_params['number_of_signatures']
                }
            
            # Check if data path exists
            if not os.path.exists(method_params['data_path']):
                raise FileNotFoundError(f"Data path does not exist: {method_params['data_path']}")

            method(**filtered_params)
            print(f"{method_name} sampling completed.\n")

            # Collect sampling information
            for root, _, files in os.walk(method_output_dir):
                for file in files:
                    if file.endswith(('.jpeg', '.jpg', '.png')):
                        path_parts = root.split(os.sep)
                        seed_folder = path_parts[-3]  # Assume seed folder is one level above person folder
                        person_folder = path_parts[-2]

                        label = 'TRUE' if 'true' in root.lower() else 'FORGED'
                        image_id = os.path.splitext(file)[0].split('_')[-1]

                        all_sampling_data.append({
                            "Data Source": method_name,
                            "Language": method_params.get("language", ""),
                            "Seed": seed_folder,
                            "Person ID/Name": person_folder,
                            "Class": label,
                            "Image ID": image_id,
                            "Image File": os.path.join(root, file)
                        })

    sampling_df = pd.DataFrame(all_sampling_data)
    sampling_csv_path = Path(base_output_dir) / 'sampling_info.csv'
    sampling_df.to_csv(sampling_csv_path, index=False)
    print(f'Saved sampling data to {sampling_csv_path}')
    
    return sampling_df


if __name__ == "__main__":
    base_data_path = "../data"
    base_output_path = "/users/fhongmin/CitizensBank-Fraud-Signature-Detection/src/preprocessing/sampled_data"
    
    params = {
        'CEDAR': {
            'data_path': f"{base_data_path}/Cedar",
            'destination_path': f"{base_output_path}/Cedar_Sampled",
            'num_individuals': 10,
            'seed': 42,
            'number_of_signatures': 10,
            'language': 'English'
        },
        'Signature_Verification': {
            'data_path': f"{base_data_path}/Signature_Verification_Dataset",
            'destination_path': f"{base_output_path}/Signature_Verification_Sampled",
            'num_individuals': 10,
            'seed': 42,
            'number_of_signatures': 10,
            'language': 'English'
        },
        'BHSig260_Bengali': {
            'data_path': f"{base_data_path}/BHSig260/Bengali",
            'destination_path': f"{base_output_path}/BHSig260_Bengali_Sampled",
            'num_individuals': 10,
            'seed': 42,
            'number_of_signatures': 10,
            'language': 'Bengali'
        },
        'BHSig260_Hindi': {
            'data_path': f"{base_data_path}/BHSig260/Hindi",
            'destination_path': f"{base_output_path}/BHSig260_Hindi_Sampled",
            'num_individuals': 10,
            'seed': 42,
            'number_of_signatures': 10,
            'language': 'Hindi'
        },
        'Real_Fake_Data': {
            'data_path': f"{base_data_path}/Real_Fake_Signature/Signature Images",
            'destination_path': f"{base_output_path}/Real_Fake_Signature_Sampled",
            'num_individuals': 10,
            'seed': 42,
            'number_of_signatures': 10,
            'language': 'Turkish'
        },
        'Hansig': {
            'data_path': f"{base_data_path}/Hansig",
            'destination_path': f"{base_output_path}/Hansig_Sampled",
            'num_individuals': 10,
            'seed': 42,
            'number_of_signatures': 10,
            'language': 'Chinese'
        }
    }
    methods = [
    (sample_dataset_cedar, 'CEDAR'),
    (sample_signature_verification_dataset, 'Signature_Verification'),
    (sample_dataset_bhsig260_bengali, 'BHSig260_Bengali'), 
    (sample_dataset_bhsig260_hindi, 'BHSig260_Hindi'), 
    (sample_dataset_real_fake, 'Real_Fake_Data'),
    (sample_dataset_hansig, 'Hansig')
]

    sampling_df = run_sampling_methods(methods, params, base_output_path)
