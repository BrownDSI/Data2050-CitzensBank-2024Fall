import os
import random
import shutil

def sample_dataset_cedar(data_path, destination_path, num_individuals, seed, number_of_signatures):
    """
    Samples individuals from the CEDAR dataset and restructures it to follow a Cedar-like structure.
    
    Parameters:
        data_path (str): Path to the root dataset (where 'full_forg' and 'full_org' folders are located).
        destination_path (str): Path to the destination directory for the reorganized structure.
        num_individuals (int): Number of individuals to sample.
        seeds (list): List of random seeds for which to create datasets.
        number_of_signatures (int): Number of signatures to include per individual for both genuine and forged.
    """
    # Paths to the forgery and genuine signature folders
    print('the number of num for cedar is',num_individuals)
    forg_path = os.path.join(data_path, 'full_forg')
    org_path = os.path.join(data_path, 'full_org')

    # Get all signature files from forgery and genuine folders
    forgeries = [f for f in os.listdir(forg_path) if f.endswith('.png')]
    originals = [f for f in os.listdir(org_path) if f.endswith('.png')]

    # Extract all individual IDs from file names
    all_persons = list(set(f.split('_')[1] for f in forgeries + originals))

    # Filter individuals with sufficient genuine and forged signatures
    eligible_persons = []
    for person in all_persons:
        # Count the number of genuine and forged signatures for the person
        genuine_signatures = [f for f in originals if f.split('_')[1] == person]
        forged_signatures = [f for f in forgeries if f.split('_')[1] == person]
        
        # Check if both have enough signatures
        if len(genuine_signatures) >= number_of_signatures and len(forged_signatures) >= number_of_signatures:
            eligible_persons.append(person)

    # Check if the requested number of individuals is available
    if num_individuals > len(eligible_persons):
        raise ValueError(f"Cannot select {num_individuals} individuals, only {len(eligible_persons)} are eligible with enough signatures.")

    # Loop over each seed and create a dataset for it
    # Set random seed for reproducibility
    random.seed(seed)

    # Randomly sample the required number of individuals from eligible persons
    sampled_persons = random.sample(eligible_persons, num_individuals)
    print('the sampled persons are for cedar')
    print('--------------------------')
    print(sampled_persons)
    # Output directory structure for this seed
    seed_folder = os.path.join(destination_path, f'random_seeds_{seed}')
    os.makedirs(seed_folder, exist_ok=True)

    # Copy files into the new structure for each sampled person
    for person in sampled_persons:
        # Format the person ID to always have 3 digits (e.g., '001', '023')
        person_folder = os.path.join(seed_folder, f'person_{int(person):03d}')
        true_folder = os.path.join(person_folder, 'true')
        forge_folder = os.path.join(person_folder, 'forge')
        os.makedirs(true_folder, exist_ok=True)
        os.makedirs(forge_folder, exist_ok=True)

        # Copy corresponding genuine files to the true folder
        genuine_signatures = [f for f in originals if f.split('_')[1] == person]
        selected_true_files = random.sample(genuine_signatures, number_of_signatures)
        for idx, filename in enumerate(selected_true_files, 1):
            src = os.path.join(org_path, filename)
            dest = os.path.join(true_folder, f'Cedar_person{int(person):03d}_true_{idx}.png')
            shutil.copy(src, dest)

        # Copy corresponding forged files to the forge folder
        forged_signatures = [f for f in forgeries if f.split('_')[1] == person]
        selected_forge_files = random.sample(forged_signatures, number_of_signatures)
        for idx, filename in enumerate(selected_forge_files, 1):
            src = os.path.join(forg_path, filename)
            dest = os.path.join(forge_folder, f'Cedar_person{int(person):03d}_forge_{idx}.png')
            shutil.copy(src, dest)

    print(f"Dataset reorganized and exported for seed {seed} to: {seed_folder}")

# Example usage
 
