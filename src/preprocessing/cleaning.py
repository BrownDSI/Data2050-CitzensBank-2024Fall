import os
import shutil

def clear_directory_content(base_dir):
    """
    Clears all content under each subdirectory in the given base directory.
    
    Parameters:
        base_dir (str): The base directory containing subdirectories whose content should be cleared.
    """
    subdirectories = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for subdir in subdirectories:
        for item in os.listdir(subdir):
            item_path = os.path.join(subdir, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)  # Remove file or symbolic link
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)  # Remove directory
            except Exception as e:
                print(f"Error deleting {item_path}: {e}")
        print(f"Cleared all content under: {subdir}")

# Example usage
if __name__ == "__main__":
    base_directory = "./preprocessed_dataset/sampled"
    clear_directory_content(base_directory)
    second_base_directory = './preprocessed_dataset'
    clear_directory_content(second_base_directory)
    print("All directories cleared.")
