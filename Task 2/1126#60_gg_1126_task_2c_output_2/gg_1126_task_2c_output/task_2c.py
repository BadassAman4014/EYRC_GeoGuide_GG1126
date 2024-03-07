'''
*****************************************************************************************
*
*        		===============================================
*           		GeoGuide(GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 2C of GeoGuide(GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''
############################## FILL THE MANDATORY INFORMATION BELOW ###############################

# Team ID:			[ Team-ID ]
# Author List:		[ Names of team members worked on this file separated by Comma: Name1, Name2, ... ]
# Filename:			task_2c.py
# Functions:	    [`classify_event(image)` ]
###################################################################################################

# IMPORTS (DO NOT CHANGE/REMOVE THESE IMPORTS)
from sys import platform
import numpy as np
import subprocess
import cv2 as cv       # OpenCV Library
import shutil
import ast
import sys
import os

# Additional Imports
'''
You can import your required libraries here
'''
import torch
from torchvision import transforms
from PIL import Image

# DECLARING VARIABLES (DO NOT CHANGE/REMOVE THESE VARIABLES)
arena_path = "arena.png"            # Path of generated arena image
event_list = []
detected_list = []

# Declaring Variables
'''
You can delare the necessary variables here
'''

# EVENT NAMES
'''
We have already specified the event names that you should train your model with.
DO NOT CHANGE THE BELOW EVENT NAMES IN ANY CASE
If you have accidently created a different name for the event, you can create another 
function to use the below shared event names wherever your event names are used.
(Remember, the 'classify_event()' should always return the predefined event names)  
'''
combat = "combat"
rehab = "humanitarianaid"
military_vehicles = "militaryvehicles"
fire = "fire"
destroyed_building = "destroyedbuilding"

# Extracting Events from Arena
def arena_image(arena_path):            # NOTE: This function has already been done for you, don't make any changes in it.
    ''' 
    Purpose:
    ---
    This function will take the path of the generated image as input and 
    read the image specified by the path.
    
    Input Arguments:
    ---
    `arena_path`: Generated image path i.e. arena_path (declared above) 	
    
    Returns:
    ---
    `arena` : [ Numpy Array ]

    Example call:
    ---
    arena = arena_image(arena_path)
    '''
    '''
    ADD YOUR CODE HERE
    '''
  
    frame = cv.imread(arena_path)
    arena = cv.resize(frame, (700, 700))
    return arena 

    

def event_identification(arena):        # NOTE: You can tweak this function in case you need to give more inputs 
    ''' 
    Purpose:
    ---
    This function will select the events on arena image and extract them as
    separate images.
    
    Input Arguments:
    ---
    `arena`: Image of arena detected by arena_image() 	
    
    Returns:
    ---
    `event_list`,  : [ List ]
                            event_list will store the extracted event images

    Example call:
    ---
    event_list = event_identification(arena)
    '''
    event_list = []

    # Define the coordinates and colors for points
    points = [(182, 625), 
              (487, 496), 
              (492, 362),  
              (173, 362),  
              (185, 147)]  

    for x, y in points:
        size = 30  # Size of the square

        # Extract the region of interest (ROI) from the arena
        roi = arena[y - size:y + size, x - size:x + size]

        # Convert the NumPy array (ROI) to a PIL image
        pil_image = Image.fromarray(roi)

        # Classify the ROI using classify_event function
        event = pil_image
        event_list.append(event)
    return event_list

# Event Detection
def classify_event(image):
    ''' 
    Purpose:
    ---
    This function will load your trained model and classify the event from an image which is 
    sent as an input.
    
    Input Arguments:
    ---
    `image`: Image path sent by input file 	
    
    Returns:
    ---
    `event` : [ String ]
                          Detected event is returned in the form of a string

    Example call:
    ---
    event = classify_event(image_path)
    '''
    '''
    ADD YOUR CODE HERE
    '''
    model = torch.load('Task2C_ModelV11.pth', map_location=torch.device('cpu'))
    class_names = ["combat", "destroyedbuilding", "fire", "humanitarianaid", "militaryvehicles"]
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # You don't need to open the image, as it's already a PIL image
    image = transform(image)
    image = image.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    event = class_names[predicted.item()]

    return event

# ADDITIONAL FUNCTIONS
'''
Although not required but if there are any additonal functions that you're using, you shall add them here. 
'''
###################################################################################################
########################### DO NOT MAKE ANY CHANGES IN THE SCRIPT BELOW ###########################
def classification(event_list):
    for img_index in range(0,5):
        img = event_list[img_index]
        detected_event = classify_event(img)
        print((img_index + 1), detected_event)
        if detected_event == combat:
            detected_list.append("combat")
        if detected_event == rehab:
            detected_list.append("rehab")
        if detected_event == military_vehicles:
            detected_list.append("militaryvehicles")
        if detected_event == fire:
            detected_list.append("fire")
        if detected_event == destroyed_building:
            detected_list.append("destroyedbuilding")
    #os.remove('arena.png')
    return detected_list

def detected_list_processing(detected_list):
    try:
        detected_events = open("detected_events.txt", "w")
        detected_events.writelines(str(detected_list))
    except Exception as e:
        print("Error: ", e)

def input_function():
    if platform == "win32":
        try:
            subprocess.run("input.exe")
        except Exception as e:
            print("Error: ", e)
    if platform == "linux":
        try:
            subprocess.run("./input")
        except Exception as e:
            print("Error: ", e)

def output_function():
    if platform == "win32":
        try:
            subprocess.run("output.exe")
        except Exception as e:
            print("Error: ", e)
    if platform == "linux":
        try:
            subprocess.run("./output")
        except Exception as e:
            print("Error: ", e)

###################################################################################################
def main():
    ##### Input #####
    input_function()
    #################

    ##### Process #####
    arena = arena_image(arena_path)
    event_list = event_identification(arena)
    detected_list = classification(event_list)
    detected_list_processing(detected_list)
    ###################

    ##### Output #####
    output_function()
    ##################

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        if os.path.exists('arena.png'):
            #os.remove('arena.png')
            print("done")
        if os.path.exists('detected_events.txt'):
            os.remove('detected_events.txt')
        sys.exit()
