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

def decode_image(base64_str, shape):
    """
    Decode the image data from base64 or bytes.
    """
    image_bytes = base64.b64decode(base64_str)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8).reshape(shape)
    return image_array

def rgb_to_grey(img):
    """
    Converts RGB image to grayscale.
    """
    grey_img = np.zeros((img.shape[0], img.shape[1]))
    for row in range(len(img)):
        for col in range(len(img[row])):
            grey_img[row][col] = np.average(img[row][col])
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

def preprocess_for_efficientnet(image):
    """
    Preprocess the image for EfficientNet.
    """
    resized_image = cv2.resize(image, (224, 224))
    normalized_image = resized_image / 255.0
    final_image = np.stack((normalized_image,) * 3, axis=-1)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    standardized_image = (final_image - mean) / std
    return standardized_image

def preprocess_for_vgg16(image):
    """
    Preprocess the image for VGG16.
    """
    resized_image = cv2.resize(image, (224, 224))
    final_image = np.stack((resized_image,) * 3, axis=-1) if resized_image.ndim == 2 else resized_image
    mean = np.array([123.68, 116.779, 103.939])
    standardized_image = final_image - mean
    return standardized_image

def preprocess_for_resnet(image):
    """
    Preprocess the image for ResNet-50.
    """
    resized_image = cv2.resize(image, (224, 224))
    final_image = np.stack((resized_image,) * 3, axis=-1) if resized_image.ndim == 2 else resized_image
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    standardized_image = (final_image - mean) / std
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
    :param df: DataFrame with columns 'person_id', 'image', 'label'
    :param num_triplets: Number of triplets to generate
    :return: List of triplets
    """
    print("Creating triplets...")
    triplets = []
    person_ids = df['person_id'].unique()
    
    for _ in range(num_triplets):
        # Select anchor and positive from the same person (genuine signatures)
        person_id = random.choice(person_ids)
        positives = df[(df['person_id'] == person_id) & (df['label'] == 0)]  # assuming '0' is genuine
        negatives = df[(df['person_id'] == person_id) & (df['label'] == 1)]  # assuming '1' is forged

        if len(positives) < 2:
            continue  # Skip if not enough genuine samples

        anchor_idx, positive_idx = np.random.choice(len(positives), 2, replace=False)
        negative_idx = np.random.choice(len(negatives), 1, replace=False)[0]
        anchor = positives.iloc[anchor_idx]['image']
        positive = positives.iloc[positive_idx]['image']
        negative = negatives.iloc[negative_idx]['image']

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
