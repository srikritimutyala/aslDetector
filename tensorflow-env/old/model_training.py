import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2

# Load your dataset
dataset_path = '../images/asl_alphabet_test'
img_size = 64

data = []
labels = []
categories = os.listdir(dataset_path)

for category in categories:
    class_num = categories.index(category)
    category_path = os.path.join(dataset_path, category)
    img_array = cv2.imread(os.path.join(dataset_path, category), cv2.IMREAD_GRAYSCALE)
    resized_array = cv2.resize(img_array, (img_size, img_size))
    data.append(resized_array)
    labels.append(class_num)

data = np.array(data)
labels = np.array(labels)

# Normalize the data
data = data / 255.0
data = data.reshape(-1, img_size, img_size, 1)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define a more complex model
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),  # Add dropout to reduce overfitting
    layers.Dense(128, activation='relu'),
    layers.Dense(len(categories), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))

# Save the model
model.save('enhanced_asl_model.h5')
