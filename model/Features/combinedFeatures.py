import numpy as np
from model.Features.featuresExtraction import extract_color_hystogram, extract_texture_hog, \
     extract_sift_features
from model.Features.paddingFeatures import padding_features

def extract_combined_features(image):

    # Estrai tutte le caratteristiche
    hist = extract_color_hystogram(image)
    hist = padding_features(hist, 5000)

    hog_features = extract_texture_hog(image)
    hog_features = padding_features(hog_features, 5000)

    sift_features = extract_sift_features(image)
    sift_features = padding_features(sift_features, 5000)

    # Combina tutte le caratteristiche in un unico vettore
    combined_features = np.concatenate([hist, hog_features, sift_features])
    combined_features = padding_features(combined_features, 15000)

    return combined_features
