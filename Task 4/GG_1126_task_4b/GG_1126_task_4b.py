import numpy as np
import cv2
from cv2 import aruco
import math
import csv

threshold = 35

def process_webcam(lat_lon, written_ids, reset_count=20):
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

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        cropped_frame = frame[crop_top:-crop_bottom, crop_left:-crop_right]

        ArUco_details_dict, ArUco_corners = detect_ArUco_details(cropped_frame)

        origin_id = 85
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


if __name__ == "__main__":
    lat_lon = read_csv('lat_long.csv')
    written_ids = set()  # Initialize an empty set to track written IDs
    process_webcam(lat_lon, written_ids)



