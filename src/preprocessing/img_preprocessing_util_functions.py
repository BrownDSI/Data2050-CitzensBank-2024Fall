import pandas as pd
import numpy as np
import cv2
import base64
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
import imgaug.augmenters as iaa
import random
from sklearn.model_selection import train_test_split
from pathlib import Path
from torchvision import transforms
from PIL import Image


def decode_image(base64_str, shape):
    """
    Decode the image data from base64 or bytes.
    """
    image_bytes = base64.b64decode(base64_str)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8).reshape(shape)
    return image_array

# def rgb_to_grey(img):
#     """
#     Converts RGB image to grayscale.
#     """
#     grey_img = np.zeros((img.shape[0], img.shape[1]))
#     for row in range(len(img)):
#         for col in range(len(img[row])):
#             grey_img[row][col] = np.average(img[row][col])
#     return grey_img

def rgb_to_grey(img):
    """
    Converts RGB image to grayscale using OpenCV.
    """
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grey_img


def grey_to_binary(img):
    """
    Converts grayscale image to binary using Gaussian filter and Otsu's thresholding.
    """
    blur_radius = 0.8
    img = ndimage.gaussian_filter(img, blur_radius)  # to remove small components or noise
    thres = threshold_otsu(img)
    bin_img = img > thres
    bin_img = np.logical_not(bin_img)
    return bin_img

def cut_out_size_to_margin(img):
    """
    Crop the image to the bounding box of the signature.
    """
    r, c = np.where(img == 1)
    cropped_img = img[r.min(): r.max(), c.min(): c.max()]
    cropped_img = 255 * cropped_img
    cropped_img = cropped_img.astype('uint8')
    return cropped_img

def augment_image(image):
    """
    Apply a series of augmentations to an image.
    """
    aug = iaa.Sequential([
        iaa.Affine(rotate=(-10, 10), translate_percent=(-0.1, 0.1), scale=(0.9, 1.1)),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.2),
        iaa.Multiply((0.8, 1.2)),
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
        iaa.ElasticTransformation(alpha=50, sigma=5),
        iaa.Cutout(nb_iterations=2, size=0.2, squared=False)
    ])
    return aug(image=image)

def skeletonize_image(binary_image):
    """
    Skeletonize the binary image.
    """
    skeleton = skeletonize(binary_image // 255)
    skeleton = img_as_ubyte(skeleton)
    return skeleton

# def preprocess_for_efficientnet(image):
#     """
#     Preprocess the image for EfficientNet.
#     """
#     resized_image = cv2.resize(image, (224, 224))
#     normalized_image = resized_image / 255.0
#     final_image = np.stack((normalized_image,) * 3, axis=-1)
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     standardized_image = (final_image - mean) / std
#     return standardized_image

# def preprocess_for_vgg16(image):
#     """
#     Preprocess the image for VGG16.
#     """
#     resized_image = cv2.resize(image, (224, 224))
#     final_image = np.stack((resized_image,) * 3, axis=-1) if resized_image.ndim == 2 else resized_image
#     mean = np.array([123.68, 116.779, 103.939])
#     standardized_image = final_image - mean
#     return standardized_image

# def preprocess_for_resnet(image):
#     """
#     Preprocess the image for ResNet-50.
#     """
#     resized_image = cv2.resize(image, (224, 224))
#     final_image = np.stack((resized_image,) * 3, axis=-1) if resized_image.ndim == 2 else resized_image
#     mean = np.array([123.675, 116.28, 103.53])
#     std = np.array([58.395, 57.12, 57.375])
#     standardized_image = (final_image - mean) / std
#     return standardized_image



def preprocess_for_efficientnet(image):
    """
    Preprocess the image for EfficientNet.
    """
    resized_image = cv2.resize(image, (224, 224))
    normalized_image = resized_image / 255.0
    # Ensure we have 3 channels; if the image is grayscale, duplicate the single channel
    if normalized_image.ndim == 2 or normalized_image.shape[-1] == 1:
        final_image = np.stack((normalized_image,) * 3, axis=-1)
    else:
        final_image = normalized_image
    
    # Apply mean and std normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    standardized_image = (final_image - mean) / std
    return standardized_image

def preprocess_for_vgg16(image):
    """
    Preprocess the image for VGG16.
    """
    resized_image = cv2.resize(image, (224, 224))
    # Ensure we have 3 channels; if the image is grayscale, duplicate the single channel
    if resized_image.ndim == 2 or resized_image.shape[-1] == 1:
        final_image = np.stack((resized_image,) * 3, axis=-1)
    else:
        final_image = resized_image
    
    # Subtract mean values
    mean = np.array([123.68, 116.779, 103.939])
    standardized_image = final_image - mean
    return standardized_image

def preprocess_for_resnet(image):
    """
    Preprocess the image for ResNet-50.
    """
    resized_image = cv2.resize(image, (224, 224))
    # Ensure we have 3 channels; if the image is grayscale, duplicate the single channel
    if resized_image.ndim == 2 or resized_image.shape[-1] == 1:
        final_image = np.stack((resized_image,) * 3, axis=-1)
    else:
        final_image = resized_image
    
    # Apply mean and std normalization
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    standardized_image = (final_image - mean) / std
    return standardized_image

def preprocess_for_inception_v3(image):
    # Convert numpy array to PIL Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),          # Resize to 299x299 pixels
        transforms.ToTensor(),                  # Convert the image to a PyTorch tensor
        transforms.Normalize(                   # Normalize with mean and std for Inception V3
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return preprocess(image)

def preprocess_for_xception(image):
    """
    Preprocess the image for Xception.
    """
    # Resize the image to 299x299 pixels (Xception's input size)
    resized_image = cv2.resize(image, (299, 299))
    
    # Ensure the image has 3 channels; if the image is grayscale, duplicate the single channel
    if resized_image.ndim == 2 or resized_image.shape[-1] == 1:
        final_image = np.stack((resized_image,) * 3, axis=-1)
    else:
        final_image = resized_image
    
    # Normalize pixel values to [0, 1]
    normalized_image = final_image / 255.0
    
    # Apply mean and standard deviation normalization
    mean = np.array([0.5, 0.5, 0.5])  # Xception typically uses [0.5, 0.5, 0.5]
    std = np.array([0.5, 0.5, 0.5])   # Standard deviation normalization
    standardized_image = (normalized_image - mean) / std
    
    return standardized_image

def split_data(df, train_size, val_size, test_size, random_state):
    """
    Split the data into training, validation, and test sets.
    """
    assert train_size + val_size + test_size == 1.0, "Train, validation, and test sizes must sum to 1.0"
    train_df, temp_df = train_test_split(df, train_size=train_size, random_state=random_state)
    val_df, test_df = train_test_split(temp_df, test_size=test_size / (val_size + test_size), random_state=random_state)
    return train_df, val_df, test_df

def create_triplets(df, num_triplets):
    """
    Create triplets with mixed negative sampling.
    :param df: DataFrame with columns 'person_id', 'image', 'label', 'Real or Forged'
    :param num_triplets: Number of triplets to generate
    :return: List of triplets
    """
    triplets = []
    person_ids = df['person_id'].unique()
    
    for _ in range(num_triplets):
        # Select anchor and positive from the same person (genuine signatures)
        positive_person_id = random.choice(person_ids)
        positives = df[(df['person_id'] == positive_person_id) & (df['label'] == 1)]  # assuming '1' is genuine

        if len(positives) < 2:
            continue  # Skip if not enough genuine samples

        anchor_idx, positive_idx = np.random.choice(len(positives), 2, replace=False)
        anchor = positives.iloc[anchor_idx]['image']
        positive = positives.iloc[positive_idx]['image']

        # Mixed negative sampling
        if random.random() < 0.5:
            # Negative from a different person
            negative_person_id = random.choice([pid for pid in person_ids if pid != positive_person_id])
            negatives = df[df['person_id'] == negative_person_id]
        else:
            # Negative as forged from the same person
            negatives = df[(df['person_id'] == positive_person_id) & (df['label'] == 0)]  # assuming '0' is forged

        if negatives.empty:
            continue  # Skip if no suitable negative found

        negative = negatives.sample(1).iloc[0]['image']
        triplets.append((anchor, positive, negative))
    
    return triplets

def save_triplets(df, file_path, num_triplets=1000):
    """
    Generate and save triplets to a file.
    """
    triplets = create_triplets(df, num_triplets)
    # Saving triplets as a list of tuples
    with open(file_path, 'wb') as f:
        np.save(f, triplets)
    print(f"Saved {len(triplets)} triplets to {file_path}")


def create_preprocessed_signature_df(preprocessed_df, model_type, preprocessing_output_dir):
    """
    Creates a DataFrame with references to the preprocessed images as arrays, applying model-specific preprocessing.
    """
    # Define a mapping for model-specific preprocessing functions
    model_preprocess_map = {
        'EfficientNet': preprocess_for_efficientnet,
        'VGG16': preprocess_for_vgg16,
        'ResNet': preprocess_for_resnet,
        'InceptionV3': preprocess_for_inception_v3,
        'Xception': preprocess_for_xception
    }

    # Ensure the model type is supported
    if model_type not in model_preprocess_map:
        raise ValueError(f"Unsupported model type: {model_type}")

    preprocess_func = model_preprocess_map[model_type]
    preprocessed_data = []

    # Process each image in the DataFrame
    for _, row in preprocessed_df.iterrows():
        image_path = row['Image File']
        label = row['Class']  # Assuming the label column is 'Class'
        person_id = row['Person ID/Name']

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image at {image_path}. Skipping.")
            continue
        
        # Apply the model-specific preprocessing function
        processed_image = preprocess_func(image)
        
        # Append processed image as array, label, and person ID to the list
        preprocessed_data.append({
            'person_id': person_id,
            'image': processed_image,  # Save the array directly here
            'label': 1 if label.lower() == "true" else 0  # Convert 'TRUE'/'FORGED' to 1/0: true means positive (genuine signature), false means negative(forged siganture)
        })

    # Create DataFrame with preprocessed data
    preprocessed_signature_df = pd.DataFrame(preprocessed_data)

    # Save the DataFrame as a pickle file
    model_output_dir = Path(preprocessing_output_dir) / model_type
    model_output_dir.mkdir(parents=True, exist_ok=True)
    output_pickle_path = model_output_dir / f"preprocessed_signature_df_{model_type}.pkl"
    preprocessed_signature_df.to_pickle(output_pickle_path)
    print(f"Saved preprocessed DataFrame for {model_type} to {output_pickle_path}")

    return preprocessed_signature_df
