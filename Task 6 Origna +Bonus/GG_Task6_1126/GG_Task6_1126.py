"""
Team Id: 1126
Author List: Aman Raut, Viranchi Dakhare, Vishakha Fulare, Gagan Loya
Filename: GG_Task6_1126.py
Theme: Geo Guide Theme

Functions:
- transmit_path(str)
- set_camera_resolution(cv.VideoCapture, int, int)
- capture_image()
- show(numpy.ndarray)
- arena_image(str)
- event_identification(numpy.ndarray)
- classify_event(PIL.Image.Image, str, str)
- draw_squares(numpy.ndarray, list, dict)
- classify_images_in_folder()
- generate_path_array(dict)
- detected_list_processing(dict)
- handle_transition(str, str, list, Transition)
- task_4a_return()
- process_webcam(tuple, set, str, int)
- calculate_marker_angle(list of tuples)
- calculate_distance_with_orientation(tuple, tuple, float, float)
- detect_ArUco_details(numpy.ndarray)
- mark_ArUco_image(numpy.ndarray, dict, dict)
- find_closest_aruco_id(dict, int)
- read_csv(str)
- write_csv(list, str)
- tracker(int, dict, set)

Global Variables:
- arena_path
- event_list
- threshold
- ip
- points
"""


import numpy as np
import cv2 as cv
import os
from keras.models import load_model
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import socket
from time import sleep
import signal		
import sys		
import numpy as np
import cv2
from cv2 import aruco
import math
import csv

# Variable Name: arena_path: Path to the image of the arena
# Type: String
arena_path = "arena/arenacap.png"

# Variable Name: event_list: List to store event types
# Type: List
event_list = []

combat = "0 Combat"
destroyed_building = "1 Destroyedbuildings"
fire = "2 Fire"
rehab = "3 Humanitarianaid"
military_vehicles = "4 Militaryvehicles"
empty = "5 Empty"

# Variable Name: threshold: Threshold value for aruco detection
# Type: Integer
threshold = 33

size = 42
# Variable Name: ip: IP address for communication
ip = "192.168.70.139"     # Mobile
#ip = "192.168.1.35"        # Wifi

# Variable Name: points: List of points
# Type: List of tuples
points = [(800, 965),
          (1265, 750),      
          (1267, 548),
          (784, 558),
          (800, 239)]

# Class Name: Transition
# Input: None
# Output: None
# Logic: Represents a transition between states
# Example Call: transition = Transition()
class Transition:
    def __init__(self):
        self.from_choice = ''  # Previous choice in the transition
        self.to_choice = ''    # Current choice in the transition
        self.path = ''         # Path followed during the transition
        self.angle_change = 0  # Angle change during the transition


'''
    * Function Name: transmit_path
    * Input: instructions (string): Instructions to transmit
    * Output: None
    * Logic: Transmits instructions over a socket connection
    * Example Call: transmit_path(path)
'''    
def transmit_path(instructions):
    def signal_handler(sig, frame):
        print('Clean-up !')
        cleanup()
        sys.exit(0)


    def cleanup():
        s.close()
        print("cleanup done")


    # To understand the working of the code, visit https://docs.python.org/3/library/socket.html
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((ip, 8002))
        s.listen()
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            data = conn.recv(1024)
            print(data.decode())  # Decode the received data
            conn.sendall(str.encode(instructions))
            sleep(1)

    # Cleanup after sending instructions
    cleanup()


'''
    * Function Name: set_camera_resolution
    * Input: cap (cv.VideoCapture): VideoCapture object, width (int): Width of the camera resolution, height (int): Height of the camera resolution
    * Output: None
    * Logic: Sets the camera resolution using OpenCV's VideoCapture object
    * Example Call: set_camera_resolution(cap, 1920, 1080)
'''
def set_camera_resolution(cap, width, height):
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)


'''
    * Function Name: capture_image
    * Input: None
    * Output: None
    * Logic: Captures an image from the camera and saves it to a file
    * Example Call: capture_image()
'''   
def capture_image():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        return
    set_camera_resolution(cap, 1920, 1080)
    print("Press 's' to capture an image.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv.imshow("Capture Image", frame)
        if cv.waitKey(1) & 0xFF == ord('s'):
            filename = "arena/arenacap.png"
            cv.imwrite(filename, frame, [cv.IMWRITE_PNG_COMPRESSION, 0])
            print(f"Image captured and saved as {filename}")
            cap.release()
            cv.destroyAllWindows()
            break
    cap.release()

def show(image):
    cv.imshow("Image", np.array(image))
    cv.waitKey(0)
    cv.destroyAllWindows()

def arena_image(arena_path):
    arena = cv.imread(arena_path)
    return arena


'''
    * Function Name: event_identification
    * Input: arena (numpy.ndarray): Image data of the arena
    * Output: event_list (list): List of event images
    * Logic: Identifies events within the arena image, saves them as separate images, and displays them using matplotlib
    * Example Call: event_identification(arena)
'''
def event_identification(arena):
    event_list = []

    for img_index, (x, y) in enumerate(points, start=1):
        roi = arena[y - size:y + size, x - size:x + size]
        if not roi.size == 0:  # Check if roi is not empty
            event_list.append(roi)
            image_filename = f'events/event_{img_index}.png'
            cv.imwrite(image_filename, roi, [cv.IMWRITE_PNG_COMPRESSION, 0])
            print(f'Event {img_index} saved as {image_filename}')

            plt.subplot(1, 5, img_index)
            plt.imshow(cv.cvtColor(roi, cv.COLOR_BGR2RGB))
            plt.title(f'Event {img_index}')
            plt.axis('off')

        else:
            print(f"Event {img_index} is empty!")
    plt.show()

    return event_list


'''
    * Function Name: classify_event
    * Input: image (PIL.Image.Image): Input image to classify, model_path (str): Path to the trained model, labels_path (str): Path to the labels file
    * Output: class_name (str): Predicted class name
    * Logic: Classifies the given image using a trained model and returns the predicted class name
    * Example Call: classify_event(image)
'''
def classify_event(image, model_path="model_and_labels/task_6.h5", labels_path="model_and_labels/labels.txt"):
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


'''
    * Function Name: draw_squares
    * Input: image (numpy.ndarray): Input image to draw bounding boxes on, points (list): List of points to draw bounding boxes around events, detected_labels (dict): Dictionary containing detected labels for events
    * Output: None
    * Logic: Draws bounding boxes and labels around detected events on the input image
    * Example Call: draw_squares(image, points, detected_labels)
'''
def draw_squares(image, points, detected_labels):
    colors = [(0, 225, 0), (0, 225, 0), (0, 225, 0), (0, 225, 0), (0, 225, 0)]  # Set different colors
    for (x, y), color, (key, label) in zip(points, colors, detected_labels.items()):
        size = 40
        cv.rectangle(image, (x - size, y - size), (x + size, y + size), color, 2)  # Draw bounding box
        label_size = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        label_x = x - size + (size * 2 - label_size[0]) // 2
        label_y = y - size - 10
        cv.putText(image, label, (label_x, label_y), cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)  # Add label


'''
    * Function Name: classify_images_in_folder
    * Input: None
    * Output: detected_labels (dict): Dictionary containing detected labels for each image in the folder
    * Logic: Classifies all images in a folder and returns the detected labels for each image
    * Example Call: classify_images_in_folder()
'''
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
        if predicted_label == empty:
            detected_labels[label_key] = "empty"
        elif predicted_label == combat:
            detected_labels[label_key] = "combat"
        elif predicted_label == rehab:
            detected_labels[label_key] = "humanitarian_aid"
        elif predicted_label == military_vehicles:
            detected_labels[label_key] = "military_vehicles"
        elif predicted_label == fire:
            detected_labels[label_key] = "fire"
        elif predicted_label == destroyed_building:
            detected_labels[label_key] = "destroyed_buildings"
        
    return detected_labels


'''
    * Function Name: generate_path_array
    * Input: detected_labels (dict): Dictionary containing detected labels for events
    * Output: path_array (str): String representing the path array of detected events
    * Logic: Generates a path array based on the detected labels and their priority order
    * Example Call: generate_path_array(detected_labels)
'''
def generate_path_array(detected_labels):
    # Remove 'empty' label if present
    detected_labels = {key: value for key, value in detected_labels.items() if value != 'empty'}

    priority_order = ['fire', 'destroyed_buildings', 'humanitarian_aid', 'military_vehicles', 'combat']
    
    sorted_labels = sorted(detected_labels.items(), key=lambda x: priority_order.index(x[1]))

    path_array = [key for key, _ in sorted_labels]

    # Ensure the path array always ends with 'S'
    path_array.append('S')

    return ''.join(path_array)


'''
    * Function Name: detected_list_processing
    * Input: detected_labels (dict): Dictionary containing detected labels for events
    * Output: None
    * Logic: Processes the detected labels and writes them to a file
    * Example Call: detected_list_processing(detected_labels)
'''
def detected_list_processing(detected_labels):
    try:
        with open("detected_labels.txt", "w") as detected_labels_file:
            detected_labels_file.write(str(detected_labels))
    except Exception as e:
        print("Error:", e)


'''
    * Function Name: handle_transition
    * Input: prev_choice (str): Previous choice made, current_choice (str): Current choice made, prev_angle (list): List containing the previous angle, transition (Transition): Transition object to store transition information
    * Output: None
    * Logic: Handles transitions between choices and updates transition information
    * Example Call: handle_transition(prev_choice, current_choice, prev_angle, transition)
'''
def handle_transition(prev_choice, current_choice, prev_angle, transition):
    def print_transition(message, path, angle_change):
        #print(f"{message} {path} +{angle_change}")
        pass

    if prev_choice == 'S':
        if current_choice == 'A':
            print_transition("Transition from Start to A:", "R", 90)
            transition.path = "RO"
            prev_angle[0] = 90
        elif current_choice == 'B':
            print_transition("Transition from Start to B:", "(RFL)", 0)
            transition.path = "RFLU"
            prev_angle[0] = 0
        elif current_choice == 'C':
            print_transition("Transition from Start to C:", "(RFFL)", 0)
            transition.path = "RFFLU"
            prev_angle[0] = 0
        elif current_choice == 'D':
            print_transition("Transition from Start to D:", "(FFR)", 90)
            transition.path = "FFRO"
            prev_angle[0] = 90
        elif current_choice == 'E':
            print_transition("Transition from Start to E:", "(F,F,F,F)", 90)
            transition.path = "FFFFX"
            prev_angle[0] = 90
        else:
            print("Invalid transition from Start")
    elif prev_choice == 'A':
        if prev_angle[0] == 90:
            if current_choice == 'B':
                print_transition("Transition from A to B:", "(FL)", 0)
                transition.path += "FLU"
                prev_angle[0] = 0
            elif current_choice == 'C':
                print_transition("Transition from A to C:", "(FFL)", 0)
                transition.path += "FFLU"
                prev_angle[0] = 0
            elif current_choice == 'D':
                print_transition("Transition from A to D:", "(LLRR)", 90)    #LFLW , 0
                transition.path += "LLRRO"    
                prev_angle[0] = 90
            elif current_choice == 'E':
                print_transition("Transition from A to E:", "(LFFLR)", 90)
                transition.path += "LFFLRX"
                prev_angle[0] = 90
            elif current_choice == 'S':
                print_transition("Transition from A to S:", "(LLLF)", 90)
                transition.path += "LLLF"
                prev_angle[0] = 90
            else:
                print("Invalid transition from A")
        elif prev_angle[0] == 0:
            if current_choice == 'B':
                print_transition("Transition from A to B:", "RRF", 90)
                transition.path += "RRFV"
                prev_angle[0] = 90
            elif current_choice == 'C':
                print_transition("Transition from A to C:", "(RRLR)", 90)
                transition.path += "RRLRV"
                prev_angle[0] = 90
            elif current_choice == 'D':
                print_transition("Transition from A to D:", "(RFR)", 90)
                transition.path += "RFRO"
                prev_angle[0] = 90
            elif current_choice == 'E':
                print_transition("Transition from A to E:", "(RFFF)", 90)
                transition.path += "RFFFX"
                prev_angle[0] = 90
            elif current_choice == 'S':
                print_transition("Transition from A to S:", "(L)", 90)
                transition.path += "L"
                prev_angle[0] = 90
            else:
                print("Invalid transition from A")
        else:
            print("Invalid transition from A")
    elif prev_choice == 'B':
        if prev_angle[0] == 90:
            if current_choice == 'A':
                print_transition("Transition from B to A:", "(RF)", 0)
                transition.path += "RFW"
                prev_angle[0] = 0
            elif current_choice == 'C':
                print_transition("Transition from B to C:", "(LL)", 0)
                transition.path += "LLU"
                prev_angle[0] = 0
            elif current_choice == 'D':
                print_transition("Transition from B to D:", "(RRFL)", 0)
                transition.path += "RRFLW"
                prev_angle[0] = 0
            elif current_choice == 'E':
                print_transition("Transition from B to E:", "(LFF)", 90)
                transition.path += "LFLFRX"
                prev_angle[0] = 90
            elif current_choice == 'S':
                print_transition("Transition from A to S:", "(RFL)", 90)
                transition.path += "RFL"
                prev_angle[0] = 90
            else:
                print("Invalid transition from B")
        elif prev_angle[0] == 0:
            if current_choice == 'A':
                print_transition("Transition from B to A:", "(LR)", 0)
                transition.path += "LRW"
                prev_angle[0] = 0
            elif current_choice == 'C':
                print_transition("Transition from B to C:", "(RR)", 90)
                transition.path += "RRV"
                prev_angle[0] = 90
            elif current_choice == 'D':
                print_transition("Transition from B to D:", "(RL)", 0)
                transition.path += "RLW"
                prev_angle[0] = 0
            elif current_choice == 'E':
                print_transition("Transition from B to E:", "(RFLR)", 90)
                transition.path += "RFLRX"
                prev_angle[0] = 90
            elif current_choice == 'S':
                print_transition("Transition from A to S:", "(LRL)", 90)
                transition.path += "LRL"
                prev_angle[0] = 90
            else:
                print("Invalid transition from B")
        else:
            print("Invalid transition from B")
    elif prev_choice == 'C':
        if prev_angle[0] == 90:
            if current_choice == 'A':
                print_transition("Transition from C to A:", "(RFF)", 0)
                transition.path += "RFFW"
                prev_angle[0] = 0
            elif current_choice == 'B':
                print_transition("Transition from C to B:", "(RR)", 0)
                transition.path += "RRU"
                prev_angle[0] = 0
            elif current_choice == 'D':
                print_transition("Transition from C to D:", "(LLLR)", 0)
                transition.path += "LLLRW"
                prev_angle[0] = 0
            elif current_choice == 'E':
                print_transition("Transition from C to E:", "(LLFR)", 90)
                transition.path += "LLFRX"
                prev_angle[0] = 90
            elif current_choice == 'S':
                print_transition("Transition from A to S:", "(RFFL)", 90)
                transition.path += "RFFL"
                prev_angle[0] = 90
            else:
                print("Invalid transition from C")
        elif prev_angle[0] == 0:
            if current_choice == 'A':
                print_transition("Transition from C to A:", "(LFR)", 0)
                transition.path += "LFRW"
                prev_angle[0] = 0
            elif current_choice == 'B':
                print_transition("Transition from C to B:", "(LL)", 90)
                transition.path += "LLV"
                prev_angle[0] = 90
            elif current_choice == 'D':
                print_transition("Transition from C to D:", "(F)", 0)
                transition.path += "FW"
                prev_angle[0] = 0
            elif current_choice == 'E':
                print_transition("Transition from C to E:", "(RLR)", 90)
                transition.path += "RLRX"
                prev_angle[0] = 90
            elif current_choice == 'S':
                print_transition("Transition from A to S:", "(LFRL)", 90)
                transition.path += "LFRL"
                prev_angle[0] = 90
            else:
                print("Invalid transition from C")
        else:
            print("Invalid transition from C")
    elif prev_choice == 'D':
        if prev_angle[0] == 90:
            if current_choice == 'A':
                print_transition("Transition from D to A:", "(RFR)", 0)
                transition.path += "RFRW"
                prev_angle[0] = 0
            elif current_choice == 'B':
                print_transition("Transition from D to B:", "(RL)", 90)
                transition.path += "RLV"
                prev_angle[0] = 90
            elif current_choice == 'C':
                print_transition("Transition from D to C:", "(F)", 90)
                transition.path += "FV"
                prev_angle[0] = 90
            elif current_choice == 'E':
                print_transition("Transition from D to E:", "(LLR)", 90)
                transition.path += "LLRX"
                prev_angle[0] = 90
            elif current_choice == 'S':
                print_transition("Transition from A to S:", "(RFRL)", 90)
                transition.path += "RFRL"
                prev_angle[0] = 90
            else:
                print("Invalid transition from D")
        elif prev_angle[0] == 0:
            if current_choice == 'A':
                print_transition("Transition from D to A:", "LFL", 90)
                transition.path += "LFLO"
                prev_angle[0] = 90
            elif current_choice == 'B':
                print_transition("Transition from D to B:", "(LLF)", 90)
                transition.path += "LLFV"
                prev_angle[0] = 90
            elif current_choice == 'C':
                print_transition("Transition from D to C:", "(LLLR)", 90)
                transition.path += "LLLRV"
                prev_angle[0] = 90
            elif current_choice == 'E':
                print_transition("Transition from D to E:", "(RF)", 90)
                transition.path += "RFX"
                prev_angle[0] = 90
            elif current_choice == 'S':
                print_transition("Transition from A to S:", "(LFF)", 90)
                transition.path += "LFF"
                prev_angle[0] = 90
            else:
                print("Invalid transition from D")
        else:
            print("Invalid transition from D")
    elif prev_choice == 'E':
        if prev_angle[0] == 90:
            if current_choice == 'A':
                print_transition("Transition from E to A:", "(FFFF)", 0)
                transition.path += "FFFFW"
                prev_angle[0] = 0
            elif current_choice == 'B':
                print_transition("Transition from E to B:", "(F,F,R)", 0)
                transition.path += "FFRU"
                prev_angle[0] = 0
            elif current_choice == 'C':
                print_transition("Transition from E to C:", "(F,R)", 0)
                transition.path += "FRU"
                prev_angle[0] = 0
            elif current_choice == 'D':
                print_transition("Transition from E to D:", "(RLR)", 0)
                transition.path += "RLRW"
                prev_angle[0] = 0
            elif current_choice == 'S':
                print_transition("Transition from A to S:", "(FFFL)", 90)
                transition.path += "FFFFL"
                prev_angle[0] = 90
            else:
                print("Invalid transition from E")
        elif prev_angle[0] == 0:
            if current_choice == 'A':
                print_transition("Transition from E to A:", "FFFL", 90)
                transition.path += "FFFLO"
                prev_angle[0] = 90
            elif current_choice == 'B':
                print_transition("Transition from E to B:", "(FFLF)", 90)
                transition.path += "FFLFV"
                prev_angle[0] = 90
            elif current_choice == 'C':
                print_transition("Transition from E to C:", "(LRL)", 90)
                transition.path += "LRLV"
                prev_angle[0] = 90
            elif current_choice == 'D':
                print_transition("Transition from E to D:", "(FL)", 90)
                transition.path += "FLO"
                prev_angle[0] = 90
            elif current_choice == 'S':
                print_transition("Transition from A to S:", "(FFFF)", 90)
                transition.path += "FFFF"
                prev_angle[0] = 90
            else:
                print("Invalid transition from E")
        else:
            print("Invalid transition from E")

    else:
        print("Invalid previous choice")

    # Update transition information
    transition.from_choice = prev_choice
    transition.to_choice = current_choice
    transition.angle_change = prev_angle[0]


'''
* Function Name: task_4a_return
* Input: None
* Output: Tuple (detected_labels, path_array, Final_Path)
* Logic: Captures an image from the camera, identifies events in the arena, classifies the events, processes the detected labels,
          generates a path array based on the detected labels, draws squares around the detected events, handles transitions between
          choices in the path array, and displays the resized and cropped arena image with labeled squares.
* Example Call: detected_labels, path_array, Final_Path = task_4a_return()
'''
def task_4a_return():    
    top_margin = 1
    bottom_margin = 1
    left_margin = 520
    right_margin = 300 
    
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        return

    set_camera_resolution(cap, 1920, 1080)
    print("Press 's' to capture an image.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame2 = frame.copy()
        frame2 = frame2[top_margin:-bottom_margin, left_margin:-right_margin]
        cv.resize(frame2, (800, 800))
        cv.imshow("Arena Image", frame2)
        key = cv.waitKey(1)

        if key == ord('s'):
            filename = "arena/arenacap.png"
            alt="arena/WIN_20240126_17_42_28_Pro.jpg"
            cv.imwrite(filename, frame, [cv.IMWRITE_PNG_COMPRESSION, 0])
            print(f"Image captured and saved as {filename}")

            #arena = arena_image(alt)
            arena = arena_image(filename)
            event_list = event_identification(arena)
            detected_labels = classify_images_in_folder()
            detected_list_processing(detected_labels)
            path_array = generate_path_array(detected_labels)
            draw_squares(arena, points, detected_labels)
            cropped_arena = arena[top_margin:-bottom_margin, left_margin:-right_margin]
            resized_arena = cv.resize(cropped_arena, (700, 700))

            choices_sequence = path_array.strip()
            choices = ['S', 'A', 'B', 'C', 'D', 'E']
            prev_angle = [90]
            transition = Transition()
            prev_choice = 'S'
            for current_choice in choices_sequence:
                handle_transition(prev_choice, current_choice, prev_angle, transition)
                prev_choice = current_choice
            Final_Path = transition.path
            cv.imshow('Resized and Cropped Image with Labeled Squares', resized_arena)

        if key == 27:
            break

    cap.release()
    cv.destroyAllWindows()
    return detected_labels, path_array, Final_Path


'''
* Function Name: process_webcam
* Input: lat_lon (tuple): Latitude and Longitude coordinates
         written_ids (set): Set of written IDs
         fpath (str): File path
         flagg (int): Flag indicating if the path should be transmitted
* Output: None
* Logic: Processes the webcam feed, detects ArUco markers, finds the closest marker, tracks its movement, and writes the coordinates
         to a file.
* Example Call: process_webcam(lat_lon, written_ids, fpath, flagg)
'''
def process_webcam(lat_lon, written_ids, fpath ,flagg):
    reset_count=15
    cap = cv2.VideoCapture(0)  

    if not cap.isOpened():
        print("Error opening webcam.")
        return
    
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    wait_time = int(1000 / frame_rate) 

    # Manually crop the frame
    crop_top = 1
    crop_bottom = 1
    crop_left = 110
    crop_right = 20

    write_count = 0

    if flagg ==0:
        transmit_path(str(fpath))
        flagg+=1

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        cropped_frame = frame[crop_top:-crop_bottom, crop_left:-crop_right]

        ArUco_details_dict, ArUco_corners = detect_ArUco_details(cropped_frame)

        origin_id = 100
        closest_id, closest_distance = find_closest_aruco_id(ArUco_details_dict, origin_id)

        if closest_id is not None and closest_distance < threshold:
            tracker_result = tracker(closest_id, lat_lon, written_ids)
            if tracker_result is not None:
                print(f"Coordinates written to GG_1126_task_5b.csv: {tracker_result}")
                write_count += 1

                # Check if the write_count has reached the reset_count
                if write_count >= reset_count:
                    written_ids.clear()  # Reset the set
                    write_count = 0
        # else:
        #     print("Invalid ArUco ID or distance greater than threshold.")

        cropped_frame = cv2.resize(cropped_frame, (800, 800))
        cv2.imshow("Webcam Frame", cropped_frame)


        # Break the loop if 'q' key is pressed
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


'''
* Function Name: calculate_marker_angle
* Input: corners (list of tuples): List of tuples containing corner coordinates of the marker
* Output: angle_deg (float): Angle of the marker in degrees
* Logic: Calculates the angle of the marker based on its corner coordinates
* Example Call: angle = calculate_marker_angle(corners)
'''
def calculate_marker_angle(corners):
    tl, tr, br, bl = corners
    x1, y1 = tl[0], tl[1]
    x2, y2 = tr[0], tr[1]
    angle_rad = np.arctan2(y2 - y1, x2 - x1)
    angle_deg = np.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360
    return angle_deg


'''
* Function Name: calculate_distance_with_orientation
* Input: point1 (tuple): Coordinates of the first point (x1, y1)
         point2 (tuple): Coordinates of the second point (x2, y2)
         angle1 (float): Angle of the first marker in degrees
         angle2 (float): Angle of the second marker in degrees
* Output: distance (float): Euclidean distance between the two points
* Logic: Calculates the Euclidean distance between two points considering their orientations
* Example Call: dist = calculate_distance_with_orientation(point1, point2, angle1, angle2)
'''
def calculate_distance_with_orientation(point1, point2, angle1, angle2):
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]

    # Convert angles to radians
    angle1_rad = np.radians(angle1)
    angle2_rad = np.radians(angle2)

    # Rotate the coordinates to align with the orientation of the first marker
    rotated_dx = dx * np.cos(angle1_rad) + dy * np.sin(angle1_rad)
    rotated_dy = -dx * np.sin(angle1_rad) + dy * np.cos(angle1_rad)

    # Calculate the Euclidean distance in the rotated coordinates
    distance = math.sqrt(rotated_dx**2 + rotated_dy**2)

    return distance


'''
* Function Name: detect_ArUco_details
* Input: image (numpy.ndarray): Image containing ArUco markers
* Output: ArUco_details_dict (dict), ArUco_corners (dict)
* Logic: Detects ArUco markers in the input image, calculates their centers and angles, and returns the details as dictionaries
* Example Call: details, corners = detect_ArUco_details(image)
'''
def detect_ArUco_details(image):
    ArUco_details_dict = {}
    ArUco_corners = {}

    dictionary_type = aruco.DICT_4X4_1000
    aruco_dict = aruco.getPredefinedDictionary(dictionary_type)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict)
    if ids is not None:
        for i in range(len(ids)):
            marker_id = int(ids[i][0])  # Convert to int
            marker_corners = corners[i][0]
            center = [int(marker_corners[:, 0].mean()), int(marker_corners[:, 1].mean())]
            angle = int(calculate_marker_angle(marker_corners))
            ArUco_details_dict[marker_id] = [center, angle]
            ArUco_corners[marker_id] = marker_corners
    # print("Detected details of ArUco as a dictionary:")
    # print(ArUco_details_dict)
            
    return ArUco_details_dict, ArUco_corners


'''
* Function Name: mark_ArUco_image
* Input: image (numpy.ndarray): Input image containing ArUco markers
         ArUco_details_dict (dict): Dictionary containing ArUco marker details
         ArUco_corners (dict): Dictionary containing ArUco marker corner coordinates
* Output: image (numpy.ndarray): Image with ArUco marker IDs marked
* Logic: Marks ArUco marker IDs on the input image based on the provided details and corner coordinates.
* Example Call: marked_image = mark_ArUco_image(image, ArUco_details_dict, ArUco_corners)
'''
def mark_ArUco_image(image, ArUco_details_dict, ArUco_corners):
    for ids, details in ArUco_details_dict.items():
        center = details[0]
        cv2.putText(image, str(ids), center, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    return image


'''
* Function Name: find_closest_aruco_id
* Input: ArUco_details_dict (dict): Dictionary containing ArUco marker details
         origin_id (int): ID of the origin ArUco marker for distance calculation
* Output: closest_id (int): ID of the closest ArUco marker to the origin
          closest_distance (float): Euclidean distance between the origin and the closest marker
* Logic: Finds the closest ArUco marker to the origin marker based on their center coordinates and angles.
* Example Call: closest_id, closest_distance = find_closest_aruco_id(ArUco_details_dict, origin_id)
'''
def find_closest_aruco_id(ArUco_details_dict, origin_id):
    if origin_id in ArUco_details_dict:
        origin_center = np.array(ArUco_details_dict[origin_id][0])
        origin_angle = ArUco_details_dict[origin_id][1]
        closest_id = None
        closest_distance = float('inf')

        for ids, details in ArUco_details_dict.items():
            if ids != origin_id:
                current_center = np.array(details[0])
                current_angle = details[1]

                distance = calculate_distance_with_orientation(origin_center, current_center, origin_angle, current_angle)

                if distance < closest_distance:
                    closest_distance = distance
                    closest_id = ids

        return closest_id, closest_distance
    else:
        return None, None
    

'''
* Function Name: read_csv
* Input: csv_name (str): Name of the CSV file to read
* Output: lat_lon (dict): Dictionary containing AR marker IDs and their corresponding latitude and longitude
* Logic: Reads data from the specified CSV file and stores AR marker IDs along with their latitude and longitude coordinates in a dictionary.
* Example Call: lat_lon_data = read_csv('data.csv')
'''    
def read_csv(csv_name):
    lat_lon = {}

    with open(csv_name, mode='r') as file:

        csv_reader = csv.reader(file)

        header = next(csv_reader, None)

        if header:
            header_dict = {
                'lat': header[1],
                'lon': header[2]
            }
            lat_lon['id'] = header_dict
            
        for row in csv_reader:
            aruco_id = int(row[0]) 
            lat = float(row[1])     
            lon = float(row[2])     
            lat_lon[aruco_id] = [lat, lon]
        #print(lat_lon)

    return lat_lon


'''
* Function Name: write_csv
* Input: loc (list): List containing location data to be written to the CSV file
         csv_name (str): Name of the CSV file to write
* Output: None
* Logic: Writes location data to the specified CSV file.
* Example Call: write_csv(location_data, 'data.csv')
'''
def write_csv(loc, csv_name):

    with open(csv_name, mode='w', newline='') as file:
  
        csv_writer = csv.writer(file)

        csv_writer.writerow(loc)


'''
* Function Name: tracker
* Input: ar_id (int): AR marker ID
         lat_lon (dict): Dictionary containing AR marker IDs and their corresponding latitude and longitude
         written_ids (set): Set containing AR marker IDs that have already been written to the CSV file
* Output: new_coordinate (list): New coordinate associated with the AR marker ID
* Logic: Tracks the movement of AR markers, updates their coordinates in a CSV file, and returns the new coordinates.
* Example Call: new_coords = tracker(ar_marker_id, lat_lon_data, written_ids)
'''
def tracker(ar_id, lat_lon, written_ids):
    # find the lat, lon associated with ar_id (aruco id)

    # Check if ar_id exists in lat_lon dictionary
    if ar_id in lat_lon:
        new_coordinate = lat_lon[ar_id]

        # Check if the ar_id has already been written to the CSV file
        if ar_id not in written_ids:
            # Read existing data from CSV file
            with open('GG_1126_task_4b.csv', mode='r', newline='') as file:
                csv_reader = csv.reader(file)
                rows = list(csv_reader)

            # Check if there are already rows in the CSV file
            if len(rows) > 1:
                # Update the row in the CSV file
                rows[1] = [str(new_coordinate[0]), str(new_coordinate[1])]
            else:
                # If no rows exist, append a new row
                rows.append(["lat", "lon"])
                rows.append([str(new_coordinate[0]), str(new_coordinate[1])])

            # Write the new data to the CSV file
            with open('GG_1126_task_4b.csv', mode='w', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerows(rows)

            # Mark the ar_id as written
            written_ids.add(ar_id)

            # Return the new coordinate
            return new_coordinate
        else:
            #print(f"ArUco ID {ar_id} already written to data.csv.")
            return None
    else:
        print(f"ArUco ID {ar_id} not found in lat_lon dictionary.")
        return None


"""
* Function Name: main
* Input: None
* Output: None
* Logic: Entry point of the program. Reads CSV data, retrieves identified labels and sorted path, and processes webcam feed.
* Example Call: main()
"""
if __name__ == '__main__':
    # Set the path to the image
    path = 'arena/a2.jpg'
    
    # Initialize flag variable
    flagg = 0
    
    # Read latitude and longitude data from CSV file
    lat_lon = read_csv('lat_long.csv')
    
    # Initialize an empty set to track written IDs
    written_ids = set()  
    
    # Call task_4a_return function to get labels, sorted path, and final path
    identified_labels, sorted_path, fpath = task_4a_return()
    
    # Define replacements for identified labels
    replacements = {
        'humanitarian_aid': 'Humanitarian Aid and rehabilitation',
        'combat': 'Combat',
        'fire': 'Fire',
        'empty': 'Empty',
        'destroyed_buildings': 'Destroyed buildings',
        'military_vehicles': 'Military Vehicles'
    }
    
    # Replace labels with more descriptive ones
    for key, value in identified_labels.items():
        if value in replacements:
            identified_labels[key] = replacements[value]

    # Print the identified labels after replacement
    print(identified_labels)
    
    # Print the sorted sequence of path
    print(f"Sorted Sequence of Path: {sorted_path}")
    
    # Call process_webcam function with the obtained data
    process_webcam(lat_lon, written_ids, fpath, flagg)

