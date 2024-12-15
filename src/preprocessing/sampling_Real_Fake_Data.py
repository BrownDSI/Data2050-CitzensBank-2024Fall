import os
import shutil
import random
import re

# Function to create directories if they don't exist
def create_directory(path):
    """Creates directories if they don't exist."""
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

    # Regular expression to match file names in the format "XXTYY" or "XXFYY"
    pattern = re.compile(r"(\d{2})([TF])(\d+)")
    
    # Collect all individual identifiers based on filenames
    all_files = [f for f in os.listdir(data_path) if f.lower().endswith(".jpeg")]
    
    # Identify unique individuals and check their eligibility
    individuals = sorted(set(match.group(1) for f in all_files if (match := pattern.match(f)) and len(match.group(1)) == 2))
    eligible_individuals = []
    for individual_id in individuals:
        genuine_files = [f for f in all_files if pattern.match(f) and pattern.match(f).group(1) == individual_id and 'T' in pattern.match(f).group(2)]
        forged_files = [f for f in all_files if pattern.match(f) and pattern.match(f).group(1) == individual_id and 'F' in pattern.match(f).group(2)]
        if len(genuine_files) >= number_of_signatures and len(forged_files) >= number_of_signatures:
            eligible_individuals.append(individual_id)

    # Check if there are enough eligible individuals
    if num_individuals > len(eligible_individuals):
        raise ValueError(f"Cannot select {num_individuals} individuals. Only {len(eligible_individuals)} individuals have enough genuine and forged signatures.")

    # Randomly select individuals from the eligible list
    selected_individuals = random.sample(eligible_individuals, num_individuals)

    # Organize files for each selected individual
    for individual_id in selected_individuals:
        # Create base directories for each individual and category
        person_folder = os.path.join(seed_folder, f"person_{individual_id}")
        true_folder = os.path.join(person_folder, "true")
        forge_folder = os.path.join(person_folder, "forge")
        create_directory(true_folder)
        create_directory(forge_folder)

        # Filter files for the selected individual and identify true (T) or forged (F)
        genuine_files = [f for f in all_files if pattern.match(f) and pattern.match(f).group(1) == individual_id and 'T' in pattern.match(f).group(2)]
        forged_files = [f for f in all_files if pattern.match(f) and pattern.match(f).group(1) == individual_id and 'F' in pattern.match(f).group(2)]

        # Sample files based on the specified number of signatures
        selected_true_files = random.sample(genuine_files, number_of_signatures)
        selected_forge_files = random.sample(forged_files, number_of_signatures)

        # Copy genuine files to the true folder
        for file in selected_true_files:
            match = pattern.match(file)
            signature_number = match.group(3)  # Extract replicate number
            new_filename = f"RealFake_person_{individual_id}_true_{signature_number}.jpeg"
            src_file = os.path.join(data_path, file)
            dest_file = os.path.join(true_folder, new_filename)
            shutil.copy(src_file, dest_file)

        # Copy forged files to the forge folder
        for file in selected_forge_files:
            match = pattern.match(file)
            signature_number = match.group(3)  # Extract replicate number
            new_filename = f"RealFake_person_{individual_id}_forge_{signature_number}.jpeg"
            src_file = os.path.join(data_path, file)
            dest_file = os.path.join(forge_folder, new_filename)
            shutil.copy(src_file, dest_file)

    print(f"Dataset reorganized and exported for seed {seed} to: {seed_folder}")


# Example usage
if __name__ == "__main__":
    data_path = './data/Real_Fake_Dataset'
    destination_path = './preprocessed_dataset/sampled/Real_Fake_Sampled'
    num_individuals = 15  # Adjusted to test with more IDs
    seed = 123
    number_of_signatures = 10

    sample_dataset_real_fake(data_path, destination_path, num_individuals, seed, number_of_signatures)
    print("Sampling Complete!")
