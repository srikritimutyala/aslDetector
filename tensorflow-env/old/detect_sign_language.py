import tensorflow as tf
import numpy as np
import cv2

# Load the trained model
model = tf.keras.models.load_model('enhanced_asl_model.h5')

# Load class labels (same as categories from data_preprocessing.py)
categories = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
              "V", "W", "X", "Y", "Z"]

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    flipped_frame = cv2.flip(frame, 1)

    # Preprocess the frame
    gray_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (64, 64))
    normalized_frame = resized_frame / 255.0
    reshaped_frame = np.reshape(normalized_frame, (1, 64, 64, 1))

    # Predict the gesture
    prediction = model.predict(reshaped_frame)
    gesture = categories[np.argmax(prediction)]

    # Display the prediction on the screen
    cv2.putText(flipped_frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Sign Language Detection', flipped_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
