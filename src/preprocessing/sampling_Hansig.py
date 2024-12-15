import os
import shutil
import random

def sample_dataset_hansig(data_path, destination_path, num_individuals, seed, number_of_signatures):
    """
    Samples individuals from the HanSig dataset and restructures it to follow a Cedar-like structure.
    
    Parameters:
        data_path (str): Path to the source directory containing genuine and forged signatures.
        destination_path (str): Path to the destination directory for the reorganized structure.
        num_individuals (int): Number of individuals to sample.
        seed (int): Random seed for reproducibility.
        number_of_signatures (int): Number of signatures to include per individual for both genuine and forged.
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Create the seed parent folder within the destination path
    seed_folder = os.path.join(destination_path, f"random_seeds_{seed}")
    os.makedirs(seed_folder, exist_ok=True)
    
    # Paths for the genuine and forged signature folders
    genuine_path = os.path.join(data_path, "genuine")
    forged_path = os.path.join(data_path, "forged")
    
    # Get all genuine and forged files separately
    genuine_files = os.listdir(genuine_path)
    forged_files = os.listdir(forged_path)
    
    # Extract unique individual identifiers based on the second numerical value in the naming pattern
    individuals = sorted(set(f.split('_')[2] for f in genuine_files + forged_files))

    # Filter individuals who have both enough genuine and forged signatures
    eligible_individuals = []
    for individual in individuals:
        genuine_individual_files = [f for f in genuine_files if f.split('_')[2] == individual]
        forged_individual_files = [f for f in forged_files if f.split('_')[2] == individual]
        if len(genuine_individual_files) >= number_of_signatures and len(forged_individual_files) >= number_of_signatures:
            eligible_individuals.append(individual)

    # Check if the requested number of individuals is available
    if num_individuals > len(eligible_individuals):
        raise ValueError(f"Cannot select {num_individuals} individuals, only {len(eligible_individuals)} are eligible with enough genuine and forged signatures.")
    
    # Select random individuals from the eligible ones
    selected_individuals = random.sample(eligible_individuals, num_individuals)
    
    # Create individual folder structure under the seed folder
    for individual in selected_individuals:
        person_folder = os.path.join(seed_folder, f"person_{individual}")
        true_folder = os.path.join(person_folder, "true")
        forge_folder = os.path.join(person_folder, "forge")
        
        os.makedirs(true_folder, exist_ok=True)
        os.makedirs(forge_folder, exist_ok=True)
        
        # Copy corresponding genuine files to the true folder
        genuine_individual_files = [f for f in genuine_files if f.split('_')[2] == individual]
        selected_genuine_files = random.sample(genuine_individual_files, number_of_signatures)
        
        for i, file in enumerate(selected_genuine_files):
            new_filename = f"Hansig_person_{individual}_true_{i + 1}.jpg"
            shutil.copy(
                os.path.join(genuine_path, file),
                os.path.join(true_folder, new_filename)
            )

        # Copy corresponding forged files to the forge folder
        forged_individual_files = [f for f in forged_files if f.split('_')[2] == individual]
        selected_forged_files = random.sample(forged_individual_files, number_of_signatures)
        
        for i, file in enumerate(selected_forged_files):
            new_filename = f"Hansig_person_{individual}_forge_{i + 1}.jpg"
            shutil.copy(
                os.path.join(forged_path, file),
                os.path.join(forge_folder, new_filename)
            )

    print(f"Restructuring complete! Data for {num_individuals} individuals with {number_of_signatures} genuine and forged signatures each has been organized in '{seed_folder}'.")

# Example usage
if __name__ == "__main__":
    data_path = './data/Hansig'
    dest_path = './preprocessed_dataset/sampled/Hansig_Sampled'
    num_individuals = 2
    seed = 156
    num_signatures = 5

    sample_dataset_hansig(data_path, dest_path, num_individuals, seed, num_signatures)
