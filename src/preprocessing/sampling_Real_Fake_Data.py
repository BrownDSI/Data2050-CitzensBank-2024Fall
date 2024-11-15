import os
import random
import shutil

# Function to create directories if they don't exist
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def sample_dataset_real_fake(data_path, destination_path, num_individuals, seed, number_of_signatures):
    """
    Samples individuals from the dataset and organizes their genuine and forged signatures
    into a structured Cedar-like format.
    
    Parameters:
        data_path (str): Path to the source directory containing signature images.
        destination_path (str): Path to the destination directory for the reorganized structure.
        num_individuals (int): Number of individuals to sample.
        seed (int): Random seed for reproducibility.
        number_of_signatures (int): Number of signatures to include per individual for both genuine and forged.
    """
    # Set the random seed for reproducibility
    random.seed(seed)
    
    # Define export directory structure based on seed
    seed_folder = os.path.join(destination_path, f"random_seeds_{seed}")
    create_directory(seed_folder)

    # Collect all individual identifiers (e.g., '01', '02', etc.) based on filenames
    all_files = [f for f in os.listdir(data_path) if f.endswith(".jpeg")]
    # print(f"Found files: {all_files}")  # Debug: list all files found
    individuals = sorted(set(f[:2] for f in all_files if len(f) > 3))
    print(f"Identified individuals: {individuals}")  # Debug: show identified individuals

    # Raise error if requested sample exceeds available individuals
    if num_individuals > len(individuals):
        raise ValueError(f"Cannot select {num_individuals} individuals, only {len(individuals)} available.")

    # Randomly select individuals
    selected_individuals = random.sample(individuals, num_individuals)

    # Organize files for each selected individual
    for individual_id in selected_individuals:
        # Create base directories for each individual and category
        person_folder = os.path.join(seed_folder, f"person_{individual_id}")
        true_folder = os.path.join(person_folder, "true")
        forge_folder = os.path.join(person_folder, "forge")
        create_directory(true_folder)
        create_directory(forge_folder)

        # Filter and sample genuine and forged files for the selected individual
        individual_true_files = [f for f in all_files if f[:2] == individual_id and 'T' in f]
        individual_forge_files = [f for f in all_files if f[:2] == individual_id and 'F' in f]

        # Sample files based on the specified number of signatures
        selected_true_files = random.sample(individual_true_files, min(number_of_signatures, len(individual_true_files)))
        selected_forge_files = random.sample(individual_forge_files, min(number_of_signatures, len(individual_forge_files)))

        # Copy genuine files to the true folder
        for file in selected_true_files:
            signature_number = file[3:-5]  # Extract the part of filename after identifier (excluding '.jpeg')
            new_filename = f"RealFake_person_{individual_id}_true_{signature_number}.jpeg"
            src_file = os.path.join(data_path, file)
            dest_file = os.path.join(true_folder, new_filename)
            shutil.copy(src_file, dest_file)

        # Copy forged files to the forge folder
        for file in selected_forge_files:
            signature_number = file[3:-5]  # Extract the part of filename after identifier (excluding '.jpeg')
            new_filename = f"RealFake_person_{individual_id}_forge_{signature_number}.jpeg"
            src_file = os.path.join(data_path, file)
            dest_file = os.path.join(forge_folder, new_filename)
            shutil.copy(src_file, dest_file)

    print(f"Dataset reorganized and exported for seed {seed} to: {seed_folder}")


# Example usage
if __name__ == "__main__":
    data_path = './data/Real_Fake_Dataset'
    destination_path = './preprocessed_dataset/sampled/Real_Fake_Sampled'
    num_individuals = 5
    seed = 123
    number_of_signatures = 5

    sample_dataset_real_fake(data_path, destination_path, num_individuals, seed, number_of_signatures)
    print("Sampling Complete!")
