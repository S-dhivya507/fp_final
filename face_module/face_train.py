import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = "../dataset/face/emotion detection for face/train"
test_dir = "../dataset/face/emotion detection for face/test"

datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(train_dir, target_size=(48,48),
                                          color_mode='grayscale', class_mode='categorical')

test_data = datagen.flow_from_directory(test_dir, target_size=(48,48),
                                         color_mode='grayscale', class_mode='categorical')

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=30, validation_data=test_data)

model.save("../models/face_emotion_model.h5")
