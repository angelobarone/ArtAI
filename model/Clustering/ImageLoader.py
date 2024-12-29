import os

import cv2
import numpy as np
import logging
from model.Features.CNN import extract_features_CNNauto

logging.basicConfig(
    filename ="/loadingDataset.log",
    filemode = "w",
    level = logging.INFO,
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

def load_dataset_from_folder(folder_path, n, type):
    features_list = []
    image_list = []

    #il valore n indica quante immagini includere nel caricamento
    for i in range(n):
        image_path = None
        if type == "MET":
            image_path = image_path_MET(i, folder_path)
        else:
            image_path = image_path_mixed(i, folder_path)
        if image_path is not None:
            #features_extractor = keras.models.load_model("../Features/Trained/CNN_features_extractor.keras")
            features = extract_features_CNNauto(image_path)#CNN.extract_features(image_path, features_extractor)
            features_list.append(features)
            image_list.append(i)
            logging.info("Folder "+ str(i) + " loaded")
        else:
            logging.error("Folder "+ str(i) + " not loaded")

    #Riduzione della dimensionalità
    #pca = PCA(n_components=100)
    #features_list = pca.fit_transform(features_list)
    #joblib.dump(pca, "trained\\pca_model.pkl")

    #Converti la lista in un array NumPy
    X = np.vstack(features_list)

    return X, image_list


def image_loader_MET(i, folder_path):
    image_folder_path = folder_path + "\\" + str(i)

    # Verifica se la cartella esiste
    if os.path.exists(image_folder_path):
        files = os.listdir(image_folder_path)

        if files:
            image_path = os.path.join(image_folder_path, files[0])
            image = cv2.imread(image_path)
            if image is not None:
                return image
            else:
                print(f"Impossibile caricare l'immagine {image_path}")
        else:
            print(f"La cartella {image_folder_path} è vuota.")
    else:
        print(f"La cartella {image_folder_path} non esiste.")

    return None

def image_path_MET(i, folder_path):
    image_folder_path = folder_path + "\\" + str(i)
    if os.path.exists(image_folder_path):
        files = os.listdir(image_folder_path)
        if files:
            image_path = os.path.join(image_folder_path, files[0])
        else:
            image_path = None
    else:
        image_path = None

    return image_path

def image_path_mixed(i, folder_path):
    files = os.listdir(folder_path)
    if files:
        image_path = os.path.join(folder_path, files[i])
    else:
        return None

    return image_path
