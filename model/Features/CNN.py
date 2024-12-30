import numpy as np
from PIL.ImageFile import ImageFile
from keras.src.applications.resnet import ResNet50
from keras.src.layers import BatchNormalization
from keras.src.utils import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications.resnet50 import preprocess_input

keras_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features_CNNauto(image_path):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        features = keras_model.predict(image)
        return features.flatten()
    except Exception as e:
        print(e)
        image = load_img("F:\\universit\\A.A.2024.2025\\FIA\\ArtAIPy\\model\\Gui\\line-drawing-of-an-empty-square-frame-on-a-white_534611_wh860.png", target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        features = keras_model.predict(image)
        return features.flatten()

def model(train_generator):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(train_generator.num_classes, activation='softmax')
    ])

    model.summary()
    return model

def extract_features(image_path, feature_extractor):
    #Carica l'immagine e preprocessa
    img = load_img(image_path, target_size=(128,128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    #Estrazione delle Features con il modello CNN
    features = feature_extractor.predict(img_array)
    return features
