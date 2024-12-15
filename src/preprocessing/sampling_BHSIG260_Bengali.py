import os
import shutil
import random

# Function to create directories if they don't exist
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def sample_dataset_bhsig260_bengali(data_path, destination_path, num_individuals, seed, number_of_signatures):
    """
    Samples individuals from the BHSig260 Bengali dataset and restructures it to follow a Cedar-like structure.
    
    Parameters:
        data_path (str): Path to the source directory containing individual user folders.
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
    
    # Get a list of all user directories in the dataset
    user_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    
    # Filter out individuals who do not have enough genuine and forged signatures
    eligible_users = []
    for user_id in user_dirs:
        user_path = os.path.join(data_path, user_id)
        image_files = [f for f in os.listdir(user_path) if f.endswith(".tif")]
        
        # Separate genuine and forged images based on 'G' or 'F' in the filename
        genuine_images = [img for img in image_files if '-G-' in img]
        forged_images = [img for img in image_files if '-F-' in img]
        
        # Check if the user has enough signatures
        if len(genuine_images) >= number_of_signatures and len(forged_images) >= number_of_signatures:
            eligible_users.append(user_id)
    
    # Check if the requested number of individuals is available
    if num_individuals > len(eligible_users):
        raise ValueError(f"Cannot select {num_individuals} individuals, only {len(eligible_users)} available with sufficient signatures.")
    
    # Select random individuals
    sampled_users = random.sample(eligible_users, num_individuals)
    
    # Process each sampled user
    for user_id in sampled_users:
        user_folder = os.path.join(seed_folder, f"user_{user_id}")
        true_folder = os.path.join(user_folder, "true")
        forge_folder = os.path.join(user_folder, "forge")
        
        os.makedirs(true_folder, exist_ok=True)
        os.makedirs(forge_folder, exist_ok=True)
        
        # Path to the individual’s folder
        user_path = os.path.join(data_path, user_id)
        image_files = [f for f in os.listdir(user_path) if f.endswith(".tif")]
        
        # Separate genuine and forged images based on 'G' or 'F' in the filename
        genuine_images = [img for img in image_files if '-G-' in img]
        forged_images = [img for img in image_files if '-F-' in img]
        
        # Select up to number_of_signatures for each category
        selected_genuine_images = random.sample(genuine_images, number_of_signatures)
        selected_forged_images = random.sample(forged_images, number_of_signatures)
        
        # Copy genuine images
        for i, img_file in enumerate(selected_genuine_images, 1):
            # Construct new filename
            new_filename = f"BHSig260_user_{user_id}_true_{i}.png"
            src = os.path.join(user_path, img_file)
            dest = os.path.join(true_folder, new_filename)
            shutil.copyfile(src, dest)
        
        # Copy forged images
        for i, img_file in enumerate(selected_forged_images, 1):
            # Construct new filename
            new_filename = f"BHSig260_user_{user_id}_forge_{i}.png"
            src = os.path.join(user_path, img_file)
            dest = os.path.join(forge_folder, new_filename)
            shutil.copyfile(src, dest)

    print(f"Restructuring complete! Data for {num_individuals} individuals with {number_of_signatures} genuine and forged signatures each has been organized in '{seed_folder}'.")

# Example usage
if __name__ == "__main__":
    data_path = "./data/BHSig260/Bengali/"
    dest_path = "./preprocessed_dataset/sampled/BHSig260_Dataset_Bengali/"
    num_individuals = 5
    seed = 123
    number_of_signatures = 10

    sample_dataset_bhsig260_bengali(data_path, dest_path, num_individuals, seed, number_of_signatures) 
