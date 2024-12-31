import os
import cv2


def image_loader_MET_test(folder_path):
    images = []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.jpg'):
            image = cv2.imread(file_path)
            images.append([file_name, image])

    return images