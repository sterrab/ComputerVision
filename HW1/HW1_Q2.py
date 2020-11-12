# Soraya Terrab - CSCI 507 - Computer Vision - Fall 2020
# Homework 1
# Question 2

import numpy as np
import sys

sys.stdout = open("Homework1_Q2_output.txt", "w")
# Billiard Table/Ball Parameters, dimensions in meters
table_width = 88 * 2.54e-2 # inches converted to meters
table_height = 44 * 2.54e-2
table_distance_from_camera = 2

ball_size = 57 * 1e-3 # mm to m
ball_location_accuracy = 1e-2

# Image Requirements
num_of_pixels_per_dim = 10
pixel_precision = 1

# Camera Lens Field of View (FOV)
FOV = np.array([30, 60, 90])
Resolution  = np.array ([256, 512, 1024])

for k in range(len(Resolution)):
    print( "\n #### Camera Resolution: " + str(Resolution[k]) + "x" +  str(Resolution[k]) + " ####" )
    for i in range(len(FOV)):
        print("\n Results for Lens FOV of " +  str(FOV[i])+ ":")
        # Calculating Focal length for given FOV and camera resolution, converting degrees to radians
        focal_length = Resolution[k]/(2*np.tan((np.pi/180 *FOV[i])/2))

        # Pixels Needed to Image Pool Table
        pixels_for_table_width = focal_length * table_width / table_distance_from_camera
        pixels_for_table_height = focal_length * table_height / table_distance_from_camera
        if pixels_for_table_width <= Resolution[k] and pixels_for_table_height <= Resolution[k]:
            Output = "Meets Table Imaging Requirements"
        else:
            Output = "Does Not Meet Table Imaging Requirements"
        print("~Pixels Needed to Image the Billiard Table : " + str(pixels_for_table_height.astype(int)) + "x" + str(pixels_for_table_width.astype(int)))
        print("~~~~"+ str(Output))

        # Pixels Needed to Image Billiard Ball
        pixels_for_ball = focal_length * ball_size / table_distance_from_camera
        if pixels_for_ball >= num_of_pixels_per_dim:
            Output = "Meets Ball Imaging Requirements"
        else:
            Output = "Does Not Meet Ball Imaging Requirements"
        print("~Pixels Needed to Image the Billiard Ball : " + str(round(pixels_for_ball,2)) + "x" + str(round(pixels_for_ball,2)))
        print("~~~~"+ str(Output))

        # Pixels Needed to Achieve Location Accuracy
        pixels_for_center_location = focal_length * ball_location_accuracy / table_distance_from_camera
        if pixels_for_center_location >= pixel_precision:
            Output = "Meets Center Precision for Ball Location Accuracy"
        else:
            Output = "Does Not Meet Center Precision for Ball Location Accuracy"
        print("~Pixels Needed for Location Accuracy: " + str(round(pixels_for_center_location,2)))
        print("~~~~" + str(Output))

sys.stdout.close()


