import numpy as np
from model.preprocessing.descriptorsExtraction import extract_color_hystogram, extract_texture_hog, \
    extract_haralick_features


def extract_combined_features(image):
    # Estrai tutte le caratteristiche
    hist = extract_color_hystogram(image)
    hog_features = extract_texture_hog(image)
    #sift_features = extract_sift_features(image)
    haralick_features = extract_haralick_features(image)

    # Combina tutte le caratteristiche in un unico vettore
    combined_features = np.concatenate([hist, hog_features, haralick_features])

    return combined_features
