import cv2
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt

def extract_color_hystogram(image):

    #Converte l'immagine in formato RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #Calcola l'istogramma per i canali RGB
    hist = cv2.calcHist([image], [0,1,2], None, (8,8,8), [0, 256, 0, 256, 0, 256])
    #cv2.calcHist calcila l'istogramma di un'immagine.
    #I parametri 0,1,2 indicano che stiamo calcolando l'istogramma per tutti e tre i canali (R,G,B)
    #bins è il numero di suddivisioni in cui dividere l'intervallo [0, 256] per ciascun canale.

    #Normalizza l'istogramma
    hist /= hist.sum()
    #Dividiamo l'istogramma per la somma di tutti i suoi valori in modo che la somma totale delle
    #sequenze sia pari a 1 -> l'istogramma sarà scale-invariant

    #Appiattisci l'istogramma in un vettore unidimensionale
    return hist.flatten()
    #Sarà usato come caratteristica dell'immagine

def extract_texture_hog(image):
    #Trasforma l'immagine in scala di grigi
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Calcola le caratteristiche HOG
    features, hog_image =  hog(image, orientations = 9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)

    #Migliora la visualizzazione dell'immagine HOG
    hog_image_rescaled = exposure.rescale_intensity(hog_image, out_range=(0, 10))

    # Visualizza l'immagine HOG
    plt.imshow(hog_image_rescaled, cmap=plt.gray())
    plt.title('HOG Features')
    plt.show()

    return features

#def extract_sift_features(image):


def extract_haralick_features(image):
    # Trasforma l'immagine in scala di grigi
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calcola i momenti di Hu
    moments = cv2.moments(image)
    hu_moments = cv2.HuMoments(moments)

    return hu_moments.flatten()

