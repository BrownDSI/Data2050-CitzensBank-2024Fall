import numpy as np
from tqdm import tqdm

def stream_and_check_duplicates(triplet_file_path, chunk_size=100):
    duplicate_count = 0
    unique_triplets = set()

    with open(triplet_file_path, "rb") as f:
        while True:
            try:
                batch = np.load(f, allow_pickle=True)
            except EOFError:
                break

            for triplet in tqdm(batch, desc="Processing triplets"):
                hashable_triplet = tuple(tuple(element.ravel().tolist()) for element in triplet)
                if hashable_triplet in unique_triplets:
                    duplicate_count += 1
                else:
                    unique_triplets.add(hashable_triplet)

    print(f"Duplicate triplets: {duplicate_count}")
    print(f"Unique triplets: {len(unique_triplets)}")
    return len(unique_triplets), duplicate_count

triplet_file_path = "/users/fhongmin/CitizensBank-Fraud-Signature-Detection/src/preprocessing/preprocessed_dataset/preprocessed/InceptionV3/preprocessed_triplets.npy"
stream_and_check_duplicates(triplet_file_path)
