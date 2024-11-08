# preprocessing.py
import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import img_preprocessing_util_functions as img_utils

import pandas as pd
import os
import cv2
import numpy as np
from pathlib import Path
import img_preprocessing_util_functions as img_utils

def preprocess_image(image_path, steps, switches, hyperparams):
    """
    Applies a sequence of preprocessing steps to an image, including optional grayscale and binary conversion.
    """
    # Read in the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Perform grayscale first if enabled, followed by optional binary conversion
    if switches.get('grayscale', False):
        image = img_utils.rgb_to_grey(image)
    if switches.get('grey_to_binary', False):
        image = img_utils.grey_to_binary(image)
    
    # Ensure the image is in uint8 format after binary processing
    image = (image * 255).astype(np.uint8) if image.dtype == bool else image

    # Apply other preprocessing steps
    for step in steps:
        if switches.get(step, False):
            if step == 'clahe':
                clahe = cv2.createCLAHE(clipLimit=hyperparams[step]['clipLimit'], 
                                        tileGridSize=(hyperparams[step]['tileGridSize'], hyperparams[step]['tileGridSize']))
                image = clahe.apply(image)
            elif step == 'gaussian_blur':
                image = cv2.GaussianBlur(image, (5, 5), hyperparams[step]['sigma'])
            elif step == 'adaptive_threshold':
                image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                              blockSize=hyperparams[step]['blockSize'], C=hyperparams[step]['C'])
    
    # Additional binary-only steps
    for step in steps:
        if switches.get(step, False):
            if step == 'skeletonize':
                image = img_utils.skeletonize_image(image)
            elif step == 'augment':
                image = img_utils.augment_image(image)

    # Ensure image is in uint8 format before returning
    image = (image * 255).astype(np.uint8) if image.dtype == bool else image
    return image

def preprocess_sampling_data(sampling_dir, output_dir, steps, switches, hyperparams):
    preprocessed_data = []

    # Walk through the sampling directory to process each image file
    for root, _, files in os.walk(sampling_dir):
        for file in files:
            if file.endswith(('.jpeg', '.jpg', '.png')):
                # Set class label based on directory structure
                label = "TRUE" if "true" in root.lower() else "FORGED"
                
                # Create output path preserving directory structure within output_dir
                relative_path = Path(root).relative_to(sampling_dir)
                output_path = Path(output_dir) / relative_path / file

                # Ensure output directory exists
                os.makedirs(output_path.parent, exist_ok=True)
                
                # Process and save the image
                img_path = os.path.join(root, file)
                processed_img = preprocess_image(img_path, steps, switches, hyperparams)
                cv2.imwrite(str(output_path), processed_img)

                # Extract metadata from directory and file structure
                path_parts = relative_path.parts  # Parts of the relative path
                data_source = path_parts[0] if len(path_parts) > 0 else ""  # First part as Data Source
                language = data_source.split('_')[-1] if '_' in data_source else ""
                seed = path_parts[1] if len(path_parts) > 1 else ""
                person_id = path_parts[2] if len(path_parts) > 2 else ""
                image_id = os.path.splitext(file)[0].split('_')[-1]  # Extract the last part of the filename

                # Append data to preprocessed_data list
                preprocessed_data.append({
                    "Data Source": data_source,
                    "Language": language,
                    "Seed": seed,
                    "Person ID/Name": person_id,
                    "Class": label,
                    "Image ID": image_id,
                    "Image File": str(output_path)
                })

    # Save preprocessing details to a CSV
    preprocessing_df = pd.DataFrame(preprocessed_data)
    preprocessing_csv_path = Path(output_dir) / 'preprocessed_info.csv'
    preprocessing_df.to_csv(preprocessing_csv_path, index=False)
    print(f'Saved preprocessed data to {preprocessing_csv_path}')
    
    return preprocessing_df


# Define preprocessing parameters
steps = ['skeletonize', 'augment']
switches = {
    'grayscale': True,
    'grey_to_binary': True,
    'clahe': False,
    'gaussian_blur': False,
    'adaptive_threshold': False,
    'skeletonize': True,
    'augment': True
}
hyperparams = {
    'clahe': {'clipLimit': 2.0, 'tileGridSize': 8},
    'adaptive_threshold': {'blockSize': 11, 'C': 2},
    'gaussian_blur': {'sigma': 1.5}
}

if __name__ == "__main__":
    sampling_dir = "../preprocessing/sampled_data"
    output_dir = "../preprocessing/preprocessed_data"
    preprocess_sampling_data(sampling_dir, output_dir, steps, switches, hyperparams)
