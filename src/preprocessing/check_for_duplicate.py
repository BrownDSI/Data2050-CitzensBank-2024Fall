import os
import hashlib
from collections import defaultdict

def calculate_hash(file_path):
    """
    Calculate the MD5 hash of a file.
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def find_and_remove_duplicates(folder_path):
    """
    Find and remove duplicate files in a folder based on their content.
    Retain only one copy of each duplicate.
    """
    hashes = defaultdict(list)
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".png"):
                file_path = os.path.join(root, file)
                file_hash = calculate_hash(file_path)
                hashes[file_hash].append(file_path)

    # Process duplicates
    duplicate_files = []
    for file_paths in hashes.values():
        if len(file_paths) > 1:
            # Retain the first file and mark others as duplicates
            duplicate_files.extend(file_paths[1:])

    # Remove duplicate files
    for dup_file in duplicate_files:
        os.remove(dup_file)

    return len(duplicate_files)

def check_and_remove_duplicates(base_path):
    """
    Check and remove duplicates in each dataset under the base path.
    This function looks into the `forge` and `true` folders of each dataset.
    
    Parameters:
        base_path (str): Path to the main preprocessed dataset folder.
    """
    datasets = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    total_duplicates_removed = 0

    for dataset in datasets:
        dataset_path = os.path.join(base_path, dataset, "random_seeds_42")
        if not os.path.exists(dataset_path):
            print(f"{dataset_path} does not exist. Skipping dataset {dataset}.")
            continue

        print(f"Checking dataset: {dataset}")
        user_folders = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path)
                        if os.path.isdir(os.path.join(dataset_path, d))]

        for user_folder in user_folders:
            forge_folder = os.path.join(user_folder, "forge")
            true_folder = os.path.join(user_folder, "true")

            for folder in [forge_folder, true_folder]:
                if os.path.exists(folder):
                    num_removed = find_and_remove_duplicates(folder)
                    total_duplicates_removed += num_removed
                    if num_removed > 0:
                        print(f"Removed {num_removed} duplicate(s) from {folder}.")
                else:
                    print(f"{folder} does not exist. Skipping.")

    print(f"Total duplicates removed: {total_duplicates_removed}")

