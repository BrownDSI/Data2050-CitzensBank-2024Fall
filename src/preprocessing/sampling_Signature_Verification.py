import os
import random
import shutil

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def sample_signature_verification_dataset(data_path, destination_path, num_individuals, seed, number_of_signatures):
    """
    Samples individuals from the signature verification dataset and reorganizes their genuine and forgery signatures 
    in a structured format similar to the Cedar dataset.
    
    Parameters:
        data_path (str): Path to the root dataset containing 'train' and 'test' folders.
        destination_path (str): Path to the destination directory for the reorganized structure.
        num_individuals (int): Number of individuals to sample.
        seed (int): Random seed for reproducibility.
        number_of_signatures (int): Number of signatures to include per individual for both genuine and forged.
    """
    # Set random seed for reproducibility
    print('number of num individuals for svd is',num_individuals)
    random.seed(seed)

    # Define export directory structure based on seed
    seed_folder = os.path.join(destination_path, f"random_seeds_{seed}")
    create_directory(seed_folder)

    # Paths for train and test datasets
    dataset_folders = ['train', 'test']

    # Collect all unique person IDs in the dataset (e.g., '001', '002') and check their eligibility
    eligible_individuals = []
    for dataset in dataset_folders:
        dataset_path = os.path.join(data_path, dataset)
        subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        for subdir in subdirs:
            person = subdir.split('_')[0]  # Get the person ID
            true_path = os.path.join(dataset_path, person)
            forge_path = os.path.join(dataset_path, f'{person}_forg')

            # Check if the person has enough genuine and forged signatures
            if os.path.exists(true_path) and os.path.exists(forge_path):
                genuine_files = os.listdir(true_path)
                forged_files = os.listdir(forge_path)
                if len(genuine_files) >= number_of_signatures and len(forged_files) >= number_of_signatures:
                    eligible_individuals.append(person)

    # Check if there are enough eligible individuals
    if num_individuals > len(eligible_individuals):
        raise ValueError(f"Cannot select {num_individuals} individuals. Only {len(eligible_individuals)} have enough genuine and forged signatures.")

    # Randomly sample the specified number of eligible individuals
    selected_individuals = random.sample(eligible_individuals, num_individuals)
    print('the selected individuals are , for svd, are,',selected_individuals)
    # Copy files into the new structure for each sampled person
    for person in selected_individuals:
        person_folder = os.path.join(seed_folder, f'person_{person}')
        true_folder = os.path.join(person_folder, 'true')
        forge_folder = os.path.join(person_folder, 'forge')
        create_directory(true_folder)
        create_directory(forge_folder)

        # Loop over train and test datasets to copy files
        for dataset in dataset_folders:
            dataset_path = os.path.join(data_path, dataset)
            true_path = os.path.join(dataset_path, person)
            forge_path = os.path.join(dataset_path, f'{person}_forg')

            # Copy genuine signatures
            if os.path.exists(true_path):
                genuine_files = os.listdir(true_path)
                selected_true_files = random.sample(genuine_files, number_of_signatures)
                for idx, img_file in enumerate(selected_true_files, 1):
                    src = os.path.join(true_path, img_file)
                    dest = os.path.join(true_folder, f'Signature_Verification_person{person}_true_{idx}.png')
                    shutil.copy(src, dest)

            # Copy forgery signatures
            if os.path.exists(forge_path):
                forged_files = os.listdir(forge_path)
                selected_forge_files = random.sample(forged_files, number_of_signatures)
                for idx, img_file in enumerate(selected_forge_files, 1):
                    src = os.path.join(forge_path, img_file)
                    dest = os.path.join(forge_folder, f'Signature_Verification_person{person}_forge_{idx}.png')
                    shutil.copy(src, dest)

    print(f"Dataset reorganized and exported for seed {seed} to: {seed_folder}")

# Example usage
if __name__ == "__main__":
    data_path = './data/Signature_Verification_Dataset'
    destination_path = './preprocessed_dataset/sampled/Signature_Verification_Sampled'
    num_individuals = 5
    seed = 123
    number_of_signatures = 5

    sample_signature_verification_dataset(data_path, destination_path, num_individuals, seed, number_of_signatures)
    print("Sampling Complete!")
