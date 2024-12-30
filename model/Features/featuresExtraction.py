import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure


def extract_color_hystogram(image):

    #Converte l'immagine in formato RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #Calcola l'istogramma per i canali RGB
    hist = cv2.calcHist([image], [0,1,2], None, (8,8,8), [0, 256, 0, 256, 0, 256])
    #cv2.calcHist calcola l'istogramma di un'immagine.
    #I parametri 0,1,2 indicano che stiamo calcolando l'istogramma per tutti e tre i canali (R,G,B)
    #bins è il numero di suddivisioni in cui dividere l'intervallo [0, 256] per ciascun canale.

    #Normalizza l'istogramma
    hist /= hist.sum()
    #Dividiamo l'istogramma per la somma di tutti i suoi valori in modo che la somma totale delle
    #sequenze sia pari a 1 -> l'istogramma sarà scale-invariant

    #Appiattisci l'istogramma in un vettore unidimensionale
    return hist.flatten()


def extract_color_moments(image):
    # Converti in spazio colore Lab
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    channels = cv2.split(image_lab)

    # Calcola i momenti di colore per ogni canale
    moments = []
    for channel in channels:
        mean = np.mean(channel)
        var = np.var(channel)
        skew = np.mean((channel - mean) ** 3)
        moments.extend([mean, var, skew])

    return np.array(moments)


def extract_texture_hog(image):
    #Trasforma l'immagine in scala di grigi
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Calcola le caratteristiche HOG
    features, hog_image =  hog(image, orientations = 9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)

    #Migliora la visualizzazione dell'immagine HOG
    hog_image_rescaled = exposure.rescale_intensity(hog_image, out_range=(0, 10))

    return features

def extract_sift_features(image):
    sift = cv2.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(image, None)

    if descriptors is None:
        return np.zeros(128)
    else:
        return descriptors.flatten()

def detect_orb_features(image):
    orb = cv2.ORB_create()

    keypoints, descriptors = orb.detectAndCompute(image, None)

    if descriptors is None:
        return np.zeros(128)
    else:
        return descriptors.flatten()


def extract_haralick_features(image):
    # Trasforma l'immagine in scala di grigi
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calcola i momenti di Hu
    moments = cv2.moments(image)
    hu_moments = cv2.HuMoments(moments)

    return hu_moments.flatten()

