import pandas as pd
import numpy as np
from pathlib import Path
import cv2
import sys
import img_preprocessing_util_functions as img_utils


# # Add the Project directory to the Python path
# project_root = Path(__file__).resolve().parents[2]
# sys.path.append(str(project_root))

def load_image_paths(csv_file_path):
    """
    Load the dataset from a CSV file that contains image paths.
    """
    print("Loading the data...")
    df_csv = pd.read_csv(csv_file_path)
    return df_csv


def preprocess_image(image_path):
    """
    Preprocess the image data including skeletonization and augmentation.
    """
    # Load the image from file
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Convert the image if it's in color
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply preprocessing steps from utility functions
    gray_image = img_utils.rgb_to_grey(image)
    binary_image = img_utils.grey_to_binary(gray_image)
    skeleton = img_utils.skeletonize_image(binary_image)
    augmented_image = img_utils.augment_image(skeleton)
    preprocessed_image = img_utils.preprocess_for_efficientnet(augmented_image)  # Or EfficientNet, ResNet, etc.

    return preprocessed_image

def preprocess_data(df):
    """
    Preprocess the image data by applying preprocessing steps to each image.
    """
    print("Preprocessing the data...")
    label_mapping = {'true': 0, 'forge': 1}  # Assuming 'true' is genuine and 'forge' is forged
    preprocessed_data = []

    for i, row in df.iterrows():
        image_path = row['Image File']  # Assuming this column contains the image paths
        label = label_mapping[row['Class']]  # Assuming 'Class' column has 'true' or 'forge'
        person_id = row['Person ID/Name']  # Assuming 'Person ID/Name' column exists

        # Preprocess the image
        image_path = image_path.replace('./sample_data/', '../data-sampling/sample_data/')
        processed_image = preprocess_image(image_path)

        # Append the preprocessed image, label, and person ID to the list
        preprocessed_data.append((processed_image, label, person_id))

    # Extract separate lists of images, labels, and person IDs
    images = np.array([item[0] for item in preprocessed_data])
    labels = np.array([item[1] for item in preprocessed_data])
    person_ids = np.array([item[2] for item in preprocessed_data])

    # Create a new DataFrame with the preprocessed data
    preprocessed_df = pd.DataFrame({'person_id': person_ids, 'image': list(images), 'label': labels})

    return preprocessed_df

# Main function to execute the preprocessing
if __name__ == "__main__":
    # Define file paths
    csv_file_path = '../data-sampling/sample_dataset_BHSig260_Bengali.csv'

    project_name = 'EfficientNetb0'
    output_dir = Path("./" + project_name)

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the image paths from the CSV file
    df = load_image_paths(csv_file_path)

    # Preprocess the image data
    preprocessed_df = preprocess_data(df)
    print("Preprocessed data successfully!")

    # Save the preprocessed DataFrame to a pickle file
    preprocessed_pickle_path = output_dir / 'preprocessed_signature_df.pkl'
    # preprocessed_pickle_path = './preprocessed_signature_df.pkl'
    preprocessed_df.to_pickle(preprocessed_pickle_path)
    print(f'Saved preprocessed_df to {preprocessed_pickle_path}')

    # Save the triplets (optional) using utility function
    triplets_save_path = output_dir / 'preprocessed_triplets.npy'
    img_utils.save_triplets(preprocessed_df, triplets_save_path)
    print(f'Saved preprocessed triplets to {triplets_save_path}')
