# S. Terrab - CSCI 507 Computer Vision - Fall 2020
# Homework 2 - Question 2
#
# Using the method of normalized cross-correlation, find all instances of the letter ``a"
# in the image ``textsample.tif". To do this, first extract a template subimage $w(x,y)$
# of an example of the letter ``a". Then match this template to the image
# (you can use OpenCV's ``matchTemplate" function).
# Find local peaks in the correlation scores image.
# Then threshold these peaks (you will have to experiment with the threshold) so that
# you get 1's where there are ``a"s and nowhere else. You can use OpenCV's
# ``connectedComponentsWithStats" function to extract the centroid of the peaks.
# Take the locations found and draw a box (or some type of marker) overlay on the original
# image showing the locations of the ``a"s. Your program should also count the number of detected ``a"s.

import cv2
import sys
import numpy as np


# Mouse callback function. Appends the x,y location of mouse click to a list.
def get_xy(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        #print(x, y)
        Position.append((x,y))

    return Position

def main():
    # Read image file
    image = cv2.imread('textsample.tif')
    width = image.shape[1]
    height = image.shape[0]
    scores_image = np.zeros((height, width))

    # Code below used to determine center of template
    # Position = []
    # cv2.imshow('Click on the center of a letter "a" ', image)
    # cv2.setMouseCallback(window_name='Click on the center of a letter "a" ', on_mouse=get_xy, param=image)
    # cv2.waitKey(0)
    # print(Position)

    Position = np.array([685, 141])
    halfwidth_of_template = 8
    template = image[Position[1]-halfwidth_of_template:Position[1]+halfwidth_of_template, Position[0]-halfwidth_of_template:Position[0]+halfwidth_of_template]
    cv2.imshow("Template", template)
    cv2.imwrite("HW2_Q2_" + "template.jpg", template)

    # Template Matching for multiple Objects, referenced from:
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
    C = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    x_locations, y_locations = np.where(C >= threshold)
    for i in range(len(x_locations)):
        cv2.rectangle(image, (y_locations[i], x_locations[i]),
                      (y_locations[i] + 2*halfwidth_of_template, x_locations[i] + 2* halfwidth_of_template),
                      [0, 255, 0], thickness=2)
        scores_image[x_locations[i],y_locations[i]]= 255
    cv2.imshow("Image with Detected Location of Letter 'a'", image)
    cv2.imwrite("HW2_Q2_" + "letter_a.jpg", image)
    cv2.imshow("Locations of Letter 'a' in Image", scores_image)
    cv2.imwrite("HW2_Q2_" + "scores.jpg", scores_image)
    cv2.waitKey(0)

    # Using Connected Components to Count the number of "a"s:
    _, binary_image = cv2.threshold(scores_image, thresh=0, maxval=255, type=cv2.THRESH_BINARY)
    num_labels, labels = cv2.connectedComponents(binary_image.astype(np.uint8))
    print('The number of detected "a"s is: ' + str(num_labels))

if __name__ == "__main__":
    main()