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

arena_path = "arena/arenacap.png"
event_list = []
combat = "0 Combat"
destroyed_building = "1 Destroyedbuildings"
fire = "2 Fire"
rehab = "3 Humanitarianaid"
military_vehicles = "4 Militaryvehicles"
empty = "5 Empty"

threshold = 35
ip = "192.168.189.139"     # Mobile
#ip = "192.168.1.35"        # Wifi


points = [(800, 943),
          (1265, 733),
          (1265, 530),
          (782, 540),
          (800, 222)]

class Transition:
    def __init__(self):
        self.from_choice = ''
        self.to_choice = ''
        self.path = ''
        self.angle_change = 0

def transmit_path(instructions):
    def signal_handler(sig, frame):
        print('Clean-up !')
        cleanup()
        sys.exit(0)


    def cleanup():
        s.close()
        print("cleanup done")

    ip = "192.168.189.139"     # Mobile

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
            #instructions = "FFROLRRFROFLF"
            conn.sendall(str.encode(instructions))
            sleep(1)

    # Cleanup after sending instructions
    cleanup()

def set_camera_resolution(cap, width, height):
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

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

def event_identification(arena):
    event_list = []

    for img_index, (x, y) in enumerate(points, start=1):
        size = 40
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

def classify_event(image, model_path="model_and_labels/task_5.h5", labels_path="model_and_labels/labels.txt"):
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

def draw_squares(image, points, detected_labels):
    colors = [(0, 225, 0), (0, 225, 0), (0, 225, 0), (0, 225, 0), (0, 225, 0)]  # Set different colors
    for (x, y), color, (key, label) in zip(points, colors, detected_labels.items()):
        size = 40
        cv.rectangle(image, (x - size, y - size), (x + size, y + size), color, 2)  # Draw bounding box
        label_size = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        label_x = x - size + (size * 2 - label_size[0]) // 2
        label_y = y - size - 10
        cv.putText(image, label, (label_x, label_y), cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)  # Add label

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
            detected_labels[label_key] = "Empty"
        elif predicted_label == combat:
            detected_labels[label_key] = "Combat"
        elif predicted_label == rehab:
            detected_labels[label_key] = "Humanitarian_Aid_and_rehabilitation"
        elif predicted_label == military_vehicles:
            detected_labels[label_key] = "Military_Vehicles"
        elif predicted_label == fire:
            detected_labels[label_key] = "Fire"
        elif predicted_label == destroyed_building:
            detected_labels[label_key] = "Destroyed_Buildings"
        
    return detected_labels

def generate_path_array(detected_labels):
    # Remove 'empty' label if present
    detected_labels = {key: value for key, value in detected_labels.items() if value != 'Empty'}

    priority_order = ['Fire', 'Destroyed_Buildings', 'Humanitarian_Aid_and_rehabilitation', 'Military_Vehicles', 'Combat']
    
    sorted_labels = sorted(detected_labels.items(), key=lambda x: priority_order.index(x[1]))

    path_array = [key for key, _ in sorted_labels]

    # Ensure the path array always ends with 'S'
    path_array.append('S')

    return ''.join(path_array)

def detected_list_processing(detected_labels):
    try:
        with open("detected_labels.txt", "w") as detected_labels_file:
            detected_labels_file.write(str(detected_labels))
    except Exception as e:
        print("Error:", e)

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
                print(f"Coordinates written to data.csv: {tracker_result}")
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

def calculate_marker_angle(corners):
    tl, tr, br, bl = corners
    x1, y1 = tl[0], tl[1]
    x2, y2 = tr[0], tr[1]
    angle_rad = np.arctan2(y2 - y1, x2 - x1)
    angle_deg = np.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360
    return angle_deg

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

def mark_ArUco_image(image, ArUco_details_dict, ArUco_corners):
    for ids, details in ArUco_details_dict.items():
        center = details[0]
        cv2.putText(image, str(ids), center, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    return image

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
        print(lat_lon)

    return lat_lon

def write_csv(loc, csv_name):

    with open(csv_name, mode='w', newline='') as file:
  
        csv_writer = csv.writer(file)

        csv_writer.writerow(loc)

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

if __name__ == '__main__':
    flagg = 0
    lat_lon = read_csv('lat_long.csv')
    written_ids = set()  # Initialize an empty set to track written IDs
    identified_labels , sorted_path ,fpath = task_4a_return()
    print(identified_labels)
    print(f"Sorted Sequence of Path : {sorted_path}")
    print(f"Final Path: {fpath}")
    process_webcam(lat_lon, written_ids, fpath, flagg)
