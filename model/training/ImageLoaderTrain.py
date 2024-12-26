import os
import cv2

def image_loader_train(i, folder_path):

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
