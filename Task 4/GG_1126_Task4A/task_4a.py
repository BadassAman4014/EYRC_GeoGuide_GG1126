import numpy as np
import cv2 as cv
import os
from keras.models import load_model
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

arena_path = "arena/arenacap.png"
#arena_path = "arena/arenacap.png"
event_list = []

military_vehicles = "0 militaryvehicles"
combat = "1 combat"
destroyed_building = "2 destroyedbuilding"
fire = "3 fire"
rehab = "4 humanitarianaid"

class Transition:
    def __init__(self):
        self.from_choice = ''
        self.to_choice = ''
        self.path = ''
        self.angle_change = 0

def set_camera_resolution(cap, width, height):
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)


def arena_image(arena_path):
    arena = cv.imread(arena_path)
    return arena


def event_identification(arena):
    event_list = []
    points = [(800, 940),
              (1265, 730),
              (1270, 530),
              (785, 535),
              (805, 215)]

    for img_index, (x, y) in enumerate(points, start=1):
        size = 40

        # Check if the ROI coordinates are within the valid range
        if 0 <= y - size < arena.shape[0] and 0 <= x - size < arena.shape[1]:
            roi = arena[y - size:y + size, x - size:x + size]
            event_list.append(roi)

            # Save the extracted image as a PNG file (optional)
            image_filename = f'events/event_{img_index}.png'
            cv.imwrite(image_filename, roi, [cv.IMWRITE_PNG_COMPRESSION, 0])
            print(f'Event {img_index} saved as {image_filename}')

            # Display the extracted image (optional)
            plt.subplot(1, 5, img_index)
            plt.imshow(cv.cvtColor(roi, cv.COLOR_BGR2RGB))
            plt.title(f'Event {img_index}')
            plt.axis('off')

        else:
            print(f"Invalid ROI coordinates for Event {img_index}")

    plt.show()

    return event_list

def draw_squares(image, points, detected_labels):
    colors = [(0, 225, 0), (0, 225, 0), (0, 225, 0), (0, 225, 0), (0, 225, 0)]  # Set different colors
    for (x, y), color, (key, label) in zip(points, colors, detected_labels.items()):
        size = 40
        cv.rectangle(image, (x - size, y - size), (x + size, y + size), color, 2)  # Draw bounding box
        label_size = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        label_x = x - size + (size * 2 - label_size[0]) // 2
        label_y = y - size - 10
        cv.putText(image, label, (label_x, label_y), cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)  # Add label

def classify_event(image, model_path="Models/keras_model5.h5", labels_path="Models/labels.txt"):
    eventpath = "events/"
    model = load_model(model_path, compile=False)
    class_names = open(labels_path, "r").readlines()
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Predict the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
    print(confidence_score)
    return class_name


def classify_images_in_folder():
    folder_path = "events/"
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png'))]

    detected_labels = {}

    for index, image_file in enumerate(image_files, start=1):
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path).convert("RGB")
        predicted_label = classify_event(image)
        print(f"Image: {image_file}, Predicted Class: {predicted_label[2:]}")

        label_key = chr(ord('A') + index - 1)

        if predicted_label == combat:
            detected_labels[label_key] = "combat"
        elif predicted_label == rehab:
            detected_labels[label_key] = "humanitarianaid"
        elif predicted_label == military_vehicles:
            detected_labels[label_key] = "militaryvehicles"
        elif predicted_label == fire:
            detected_labels[label_key] = "fire"
        elif predicted_label == destroyed_building:
            detected_labels[label_key] = "destroyedbuilding"

    return detected_labels


def classification(event_list):
    detected_labels = classify_images_in_folder()
    return detected_labels


def generate_path_array(detected_labels):
    priority_order = ['fire', 'destroyedbuilding', 'humanitarianaid', 'militaryvehicles', 'combat']

    sorted_labels = sorted(detected_labels.items(), key=lambda x: priority_order.index(x[1]))

    path_array = [key for key, _ in sorted_labels]

    return ''.join(path_array)

def task_4a_return():
    points = [(800, 940),
            (1265, 730),
            (1270, 530),
            (785, 535),
            (805, 215)]
    
    top_margin = 1
    bottom_margin = 1
    left_margin = 520
    right_margin = 300 
    
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        return

    set_camera_resolution(cap, 1920, 1080)
    print("Press 's' to capture an image.")
    
    detected_labels = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame2 = frame.copy()
        frame2 = frame2[top_margin:-bottom_margin, left_margin:-right_margin]
        #cv.resize(frame2, (800, 800))
        cv.imshow("Arena ", frame2)
        key = cv.waitKey(1)

        if key == ord('s'):
            filename = "arena/arenacap.png"
            cv.imwrite(filename, frame, [cv.IMWRITE_PNG_COMPRESSION, 0])
            print(f"Image captured and saved as {filename}")

            arena = arena_image(filename)
            event_list = event_identification(arena)
            detected_labels = classification(event_list)

            draw_squares(arena, points, detected_labels)
            cropped_arena = arena[top_margin:-bottom_margin, left_margin:-right_margin]
            resized_arena = cv.resize(cropped_arena, (700, 700))
            cv.imshow('Resized and Cropped Image with Labeled Squares', resized_arena)

        if key == 27:
            break

    cap.release()
    cv.destroyAllWindows()
    return detected_labels

if __name__ == '__main__':
    identified_labels = task_4a_return()
    print(identified_labels)

