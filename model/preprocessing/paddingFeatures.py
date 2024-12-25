import numpy as np

def padding_features(feature):
    target_size = 103731  # La dimensione da forzare per ogni feature

    if feature.shape[0] < target_size:
        # Aggiungi zeri (padding) se la caratteristica è più piccola
        padding = np.zeros(target_size - feature.shape[0])
        feature = np.concatenate((feature, padding))
    elif feature.shape[0] > target_size:
        # Troncamento se la caratteristica è più grande
        feature = feature[:target_size]

    return feature