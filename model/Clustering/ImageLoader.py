import os
import cv2
import numpy as np
import logging
from PIL import Image
from tqdm import tqdm

from model.Features.CNN import extract_features_CNNauto

def load_dataset_from_folder(folder_path, n, tipo):
    features_list = []
    image_list = []
    with open("log.txt", "w") as log:
        # il valore n indica quante immagini includere nel caricamento
        for i in tqdm(range(n), desc="Loading dataset"):
            image_path = None
            if tipo == "MET":
                image_path = image_path_MET(i, folder_path)
            else:
                image_path = image_path_mixed(i, folder_path)
            if image_path is not None and is_image_valid(image_path):
                if tipo == "MET":
                    image_list.append(i)
                else:
                    image_name = os.path.basename(image_path)
                    image_list.append(image_name)
                    log.write("Image: " + str(image_name) + " loaded" +"\n")

                # features_extractor = keras.models.load_model("../Features/Trained/CNN_features_extractor.keras")
                features = extract_features_CNNauto(image_path)  # CNN.extract_features(image_path, features_extractor)
                features_list.append(features)
            else:
                log.write("Image" + str(image_path) + " not loaded" + "\n")

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
    image_path = None
    files = os.listdir(folder_path)
    if files:
        try:
            image_path = os.path.join(folder_path, files[i])
        except IndexError:
            print(f"Indice fuori dai limiti: {i}")
        except Exception as e:
            print(f"Errore durante il caricamento del file: {e}")
    else:
        return None

    return image_path

def is_image_valid(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verifica l'integrità dell'immagine
        return True
    except (IOError, SyntaxError):
        return False