import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 100

cap = cv2.VideoCapture(0)  # Replace 0 with the correct index if necessary
if not cap.isOpened():
    print("Error: Camera not accessible. Please check the camera index or connection.")
    exit()

flipped = True

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    # Initial prompt to the user
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Failed to capture image. Please check the camera.")
            break

        if flipped:
            frame = cv2.flip(frame, 1)  # Flip the frame horizontally (mirror image)

        cv2.putText(frame, 'Ready? Press "Q" to start or "K" to quit and flip', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(25) & 0xFF

        if key == ord('q'):  # Start data collection when "Q" is pressed
            break
        elif key == ord('k'):  # Flip the camera and end the program when "K" is pressed
            flipped = not flipped  # Toggle the flipped state
            cap.release()
            cv2.destroyAllWindows()
            print("Program terminated by user.")
            exit()

    # Collect images for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Failed to capture image. Skipping this frame.")
            continue

        if flipped:
            frame = cv2.flip(frame, 1)  # Flip the frame horizontally (mirror image)

        cv2.imshow('frame', frame)
        key = cv2.waitKey(25) & 0xFF

        if key == ord('k'):  # Flip the camera and end the program when "K" is pressed
            flipped = not flipped  # Toggle the flipped state
            cap.release()
            cv2.destroyAllWindows()
            print("Program terminated by user.")
            exit()

        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
