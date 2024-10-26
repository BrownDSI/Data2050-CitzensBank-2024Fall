import os
import random
import shutil
import pandas as pd


# Function to create directories if they don't exist
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)



def sample_dataset_BHSig260_Bengali(path, num_samples, random_seed):
    """
    The function samples individuals from the BHSig260 Bengali dataset.
    base_path: The base directory of the dataset.
    num_samples: The number of individuals to sample.
    random_seed: The random seed for reproducibility.
    """

    data_source = "BHSig260"
    language = "Bengali"
    export_base_dir = "./sample_data/BHSig260_Dataset_Bengali/"

    # Initialize an empty list to hold the dataset rows
    dataset_rows = []

    # Get a list of all user directories (e.g., '001', '100', etc.)
    user_dirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    # Set the random seed for reproducibility
    random.seed(random_seed)

    # Sample from the list of user directories
    sampled_users = random.sample(user_dirs, num_samples)

    # Output the sampled user directories and list images in each
    for user_dir in sampled_users:

        # Get all image files (e.g., .tif) under this user directory
        image_files = [f for f in os.listdir(user_dir) if f.endswith(".tif")]
        # print(image_files)

        # Separate genuine and forged images based on 'G' or 'F' in the filename
        genuine_images = [os.path.join(user_dir, img) for img in image_files if '-G-' in img]
        forged_images = [os.path.join(user_dir, img) for img in image_files if '-F-' in img]

        num_genuine_img = len(genuine_images)
        num_forged_img = len(forged_images)

        num_samples_img = min(num_genuine_img, num_forged_img)
        sampled_genuine_img = random.sample(genuine_images, num_samples_img)
        sampled_forged_img = random.sample(forged_images, num_samples_img)
        all_sampled_images = sampled_genuine_img + sampled_forged_img
        
        for img_file in all_sampled_images:
            # Split the file name to get relevant parts
            parts = img_file.split('-')
            user_id = parts[2]  # e.g, '61' from 'B-S-61-F-04.tif'
            image_num = parts[4].split('.')[0]  # e.g, '04' from 'B-S-61-F-04.tif'

            # Determine if the image is forged or genuine
            if 'F' in img_file:
                new_filename = f"BHSig260_Bengali_person_{user_id}_forge_{image_num}.png"
                category = "forge"
            elif 'G' in img_file:
                new_filename = f"BHSig260_Bengali_person_{user_id}_true_{image_num}.png"
                category = "true"

            # Create directory for the specific user and category (forge or true)
            seed_dir = os.path.join(export_base_dir, f"seed_{random_seed}")
            user_dir = os.path.join(seed_dir, f"person_{user_id}", category)
            create_directory(user_dir)

            # Define old and new paths
            old_path = os.path.join(img_file)  # Assuming you have the source image path here
            new_path = os.path.join(user_dir, new_filename)

            # Copy the image to the new directory and rename it
            shutil.copyfile(old_path, new_path)

            # Store this image's data in the dataset
            dataset_rows.append({
                "Data Source": data_source,
                "Language": language,
                "Seed": random_seed,
                "Person ID/Name": f"person_{user_id}",
                "Class": category,
                "Image ID": image_num,
                "Image File": new_path
            })
    return dataset_rows


def sample_dataset_BHSig260_Hindi(path, num_samples, random_seed):
    """
    The function samples individuals from the BHSig260 Hindi dataset.
    base_path: The base directory of the dataset.
    num_samples: The number of individuals to sample.
    random_seed: The random seed for reproducibility.
    """

    data_source = "BHSig260"
    language = "Hindi"
    export_base_dir = "./sample_data/BHSig260_Dataset_Hindi/"

    # Initialize an empty list to hold the dataset rows
    dataset_rows = []

    # Get a list of all user directories (e.g., '001', '100', etc.)
    user_dirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    # Set the random seed for reproducibility
    random.seed(random_seed)

    # Sample from the list of user directories
    sampled_users = random.sample(user_dirs, num_samples)

    # Output the sampled user directories and list images in each
    for user_dir in sampled_users:

        # Get all image files (e.g., .tif) under this user directory
        image_files = [f for f in os.listdir(user_dir) if f.endswith(".tif")]
        # print(image_files)

        # Separate genuine and forged images based on 'G' or 'F' in the filename
        genuine_images = [os.path.join(user_dir, img) for img in image_files if '-G-' in img]
        forged_images = [os.path.join(user_dir, img) for img in image_files if '-F-' in img]

        num_genuine_img = len(genuine_images)
        num_forged_img = len(forged_images)

        num_samples_img = min(num_genuine_img, num_forged_img)
        sampled_genuine_img = random.sample(genuine_images, num_samples_img)
        sampled_forged_img = random.sample(forged_images, num_samples_img)
        all_sampled_images = sampled_genuine_img + sampled_forged_img
        
        for img_file in all_sampled_images:
            # Split the file name to get relevant parts
            parts = img_file.split('-')
            user_id = parts[2]  # e.g, '61' from 'B-S-61-F-04.tif'
            image_num = parts[4].split('.')[0]  # e.g, '04' from 'B-S-61-F-04.tif'

            # Determine if the image is forged or genuine
            if 'F' in img_file:
                new_filename = f"BHSig260_Hindi_person_{user_id}_forge_{image_num}.png"
                category = "forge"
            elif 'G' in img_file:
                new_filename = f"BHSig260_Hindi_person_{user_id}_true_{image_num}.png"
                category = "true"

            # Create directory for the specific user and category (forge or true)
            seed_dir = os.path.join(export_base_dir, f"seed_{random_seed}")
            user_dir = os.path.join(seed_dir, f"person_{user_id}", category)
            create_directory(user_dir)

            # Define old and new paths
            old_path = os.path.join(img_file)  # Assuming you have the source image path here
            new_path = os.path.join(user_dir, new_filename)

            # Copy the image to the new directory and rename it
            shutil.copyfile(old_path, new_path)

                        # Store this image's data in the dataset
            dataset_rows.append({
                "Data Source": data_source,
                "Language": language,
                "Seed": random_seed,
                "Person ID/Name": f"person_{user_id}",
                "Class": category,
                "Image ID": image_num,
                "Image File": new_path
            })
    return dataset_rows
        

if __name__=="__main__":

    # Sampling BHSig260 Bengali Dataset
    data_path = "../../data/BHSig260/Bengali/"
    num_samples = 3
    seed = 123

    dataset_Bengali = sample_dataset_BHSig260_Bengali(data_path, num_samples, seed)
    print("Sample Successfully!")

    df_Bengali = pd.DataFrame(dataset_Bengali)
    df_Bengali.to_csv("sample_dataset_BHSig260_Bengali.csv", index=False)
    # df_Bengali.to_parquet("sample_dataset_BHSig260_Bengali.parquet", index=False)
    print("Sample BHSig260/Bengali dataset generated and saved to CSV/Parquet!")



    # Sampling BHSig260 Hindi Dataset
    data_path = "../../data/BHSig260/Hindi/"
    num_samples = 3
    seed = 456

    dataset_Hindi = sample_dataset_BHSig260_Hindi(data_path, num_samples, seed)
    print("Sample Successfully!")

    # Create the DataFrame from the collected data
    df_Hindi = pd.DataFrame(dataset_Hindi)
    df_Hindi.to_csv("sample_dataset_BHSig260_Hindi.csv", index=False)
    # df_Hindi.to_parquet("sample_dataset_BHSig260_Hindi.parquet", index=False)
    print("Sample BHSig260/Hindi dataset generated and saved to CSV/Parquet!")