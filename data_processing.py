import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DEFAULT_DATA_PATH = "mini_gm_public_v0.1.p"


def load_and_process_data(data_path=DEFAULT_DATA_PATH):
    """Carrega o pickle hierárquico (síndrome -> sujeito -> imagem -> embedding)
    e o achata em X, y e image_ids."""
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    embeddings, syndrome_ids, image_ids = [], [], []

    for syndrome_id, subjects_data in data.items():
        for subject_id, images_data in subjects_data.items():
            for image_id, embedding_vector in images_data.items():
                embeddings.append(embedding_vector)
                syndrome_ids.append(syndrome_id)
                image_ids.append(image_id)

    X = np.array(embeddings)
    y = np.array(syndrome_ids)
    image_ids = np.array(image_ids)

    print(f"Total number of images: {len(X)}")
    print(f"Shape of embeddings (X): {X.shape}")
    print(f"Number of unique syndromes: {len(np.unique(y))}")

    return X, y, image_ids


def encode_labels(y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("Syndrome IDs successfully encoded.")
    print(f"Shape of y: {y_encoded.shape}")
    print(f"Encoded classes: {np.unique(y_encoded)}")
    print(f"Original classes: {label_encoder.inverse_transform(np.unique(y_encoded))}")

    return y_encoded, label_encoder


def run_eda(y):
    syndrome_distribution = pd.Series(y).value_counts()
    print("\nDistribution of Syndrome IDs:")
    print(syndrome_distribution)
    return syndrome_distribution
