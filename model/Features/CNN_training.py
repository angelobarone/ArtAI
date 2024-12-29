import numpy as np
from keras import Model
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from model.Features import CNN

folder_path = "F:\\universit\\A.A.2024.2025\\FIA\\ArtAIPy\\dataset\\dataset2\\00.classi_CNN"
img_size = (128, 128)
batch_size = 32

datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    folder_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
)

val_generator = datagen.flow_from_directory(
    folder_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

model = CNN.model(train_generator)

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

feature_extractor = Model(inputs=model.inputs, outputs=model.layers[-3].output)
feature_extractor.save("Trained/CNN_features_extractor.keras")