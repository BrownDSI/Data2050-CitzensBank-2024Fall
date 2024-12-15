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


def find_duplicates_within_folder(folder_path):
    """
    Find duplicate files within a single folder based on their content.
    """
    hashes = defaultdict(list)
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.png'):
                file_path = os.path.join(root, file)
                file_hash = calculate_hash(file_path)
                hashes[file_hash].append(file_path)

    # Return only hashes with more than one file (duplicates)
    return [file_paths for file_paths in hashes.values() if len(file_paths) > 1]


def find_duplicates_across_folders(folder_paths):
    """
    Find duplicate files across multiple folders based on their content.
    """
    hashes = defaultdict(list)
    for folder_path in folder_paths:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.png'):
                    file_path = os.path.join(root, file)
                    file_hash = calculate_hash(file_path)
                    hashes[file_hash].append(file_path)

    # Return only hashes with more than one file (duplicates)
    return [file_paths for file_paths in hashes.values() if len(file_paths) > 1]


def check_for_duplicates_within_and_across(base_folder, log_file, dataset_names):
    """
    Check for duplicate pictures both within individual datasets and across datasets.
    Log the results to the same log file.
    """
    dataset_paths = {
        dataset: os.path.join(base_folder, f"{dataset}_Sampled", "random_seeds_42")
        for dataset in dataset_names
    }

    # Check duplicates within each dataset
    for dataset, path in dataset_paths.items():
        if not os.path.exists(path):
            log_file.write(f"{path} does not exist.\n")
            continue

        log_file.write(f"\nChecking duplicates within dataset: {dataset}\n")
        forge_folders = []
        true_folders = []

        # Collect all forge and true folders
        user_folders = [
            os.path.join(path, d)
            for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))
        ]
        for user_folder in user_folders:
            forge_folder = os.path.join(user_folder, "forge")
            true_folder = os.path.join(user_folder, "true")
            if os.path.exists(forge_folder):
                forge_folders.append(forge_folder)
            if os.path.exists(true_folder):
                true_folders.append(true_folder)

        # Find duplicates within forge and true folders
        forge_duplicates_within = []
        true_duplicates_within = []

        for folder in forge_folders:
            forge_duplicates_within.extend(find_duplicates_within_folder(folder))

        for folder in true_folders:
            true_duplicates_within.extend(find_duplicates_within_folder(folder))

        # Log results for this dataset
        if forge_duplicates_within:
            log_file.write("Duplicate pictures found within forge folders:\n")
            for dup_group in forge_duplicates_within:
                log_file.write(f"- {dup_group}\n")
        else:
            log_file.write("No duplicates found within forge folders.\n")

        if true_duplicates_within:
            log_file.write("Duplicate pictures found within true folders:\n")
            for dup_group in true_duplicates_within:
                log_file.write(f"- {dup_group}\n")
        else:
            log_file.write("No duplicates found within true folders.\n")

    # Check duplicates across datasets
    log_file.write("\nChecking duplicates across all datasets...\n")
    all_forge_folders = []
    all_true_folders = []

    # Collect all forge and true folders from all datasets
    for dataset, path in dataset_paths.items():
        if os.path.exists(path):
            user_folders = [
                os.path.join(path, d)
                for d in os.listdir(path)
                if os.path.isdir(os.path.join(path, d))
            ]
            for user_folder in user_folders:
                forge_folder = os.path.join(user_folder, "forge")
                true_folder = os.path.join(user_folder, "true")
                if os.path.exists(forge_folder):
                    all_forge_folders.append(forge_folder)
                if os.path.exists(true_folder):
                    all_true_folders.append(true_folder)

    # Find duplicates across forge and true folders
    forge_duplicates_across = find_duplicates_across_folders(all_forge_folders)
    true_duplicates_across = find_duplicates_across_folders(all_true_folders)

    # Log results across datasets
    if forge_duplicates_across:
        log_file.write("Duplicate pictures found across forge folders in all datasets:\n")
        for dup_group in forge_duplicates_across:
            log_file.write(f"- {dup_group}\n")
    else:
        log_file.write("No duplicates found across forge folders in all datasets.\n")

    if true_duplicates_across:
        log_file.write("Duplicate pictures found across true folders in all datasets:\n")
        for dup_group in true_duplicates_across:
            log_file.write(f"- {dup_group}\n")
    else:
        log_file.write("No duplicates found across true folders in all datasets.\n")


# List of datasets
dataset_names = [
    "CEDAR",
    "Signature_Verification",
    "BHSig260_Bengali",
    "BHSig260_Hindi",
    "Real_Fake_Data",
    "Hansig"
]

# Base directory path
base_directory = "/users/fhongmin/CitizensBank-Fraud-Signature-Detection/src/preprocessing/preprocessed_dataset/preprocessed"
log_file_path = os.path.join(base_directory, "combined_duplicate_check_log.txt")

# Write all logs to the same file
with open(log_file_path, "w") as log_file:
    print(f"Checking for duplicates within and across datasets...")
    check_for_duplicates_within_and_across(base_directory, log_file, dataset_names)

print(f"Duplicate check completed. Results saved to {log_file_path}")
