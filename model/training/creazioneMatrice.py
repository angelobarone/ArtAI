import numpy as np
import logging
import joblib
from sklearn.decomposition import PCA

from model.preprocessing.combinedFeatures import extract_combined_features
from model.training.ImageLoaderTrain import image_loader_train
from model.preprocessing.paddingFeatures import padding_features

logging.basicConfig(
    filename ="../training/loadingDataset.log",
    filemode = "w",
    level = logging.INFO,
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

def load_dataset_from_folder(folder_path, n):
    features_list = []
    image_list = []

    #il valore n indica quante immagini includere nel caricamento
    for i in range(n):
        image = image_loader_train(i, folder_path)
        if image is not None:
            features = extract_combined_features(image)
            features_list.append(features)
            image_list.append(i)
            logging.info("Folder "+ str(i) + " loaded")
        else:
            logging.error("Folder "+ str(i) + " not loaded")

    # Riduzione della dimensionalità
    pca = PCA(n_components=100)
    features_list = pca.fit_transform(features_list)
    joblib.dump(pca, "trained\\pca_model.pkl")

    # Converti la lista in un array NumPy
    X = np.vstack(features_list)

    return X, image_list
