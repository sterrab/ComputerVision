# S. Terrab - CSCI 507 Computer Vision - Fall 2020
# Homework 2 - Question 2

# The video ``fiveCCC.wmv" or ``fiveCCC.avi" shows a target composed of five concentric contrasting circle (CCC) features.
# Read and display each image of the video, and print the frame number on the displayed image.
# Write a program that finds the five CCCs in each image, and marks each one with a cross hair or rectangle.
# Create an output video of your results and post it on YouTube.




import cv2
import sys
import numpy as np


def main():
    # Reading  images from the video file
    video_capture = cv2.VideoCapture("fiveCCC.avi")     # Open video capture object
    #video_capture = cv2.VideoCapture(index=0)     # Open video capture object
    got_image, frame = video_capture.read()       # Make sure we can read video
    if not got_image:
        print("Cannot read video source")
        sys.exit()

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter("HW4_Q4_video.avi", fourcc=fourcc, fps=30.0,
                                  frameSize=(np.shape(frame)[1], np.shape(frame)[0]))

    # Read/label images and mark CCCs until end of video is reached.
    frame_number = 0
    while True:
        got_image, frame = video_capture.read()
        if not got_image:
            break       # End of video; exit the while loop

        # Converting image to grayscale and then binary using threshold as defined in lecture
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(gray_img, thresh=127, maxval=255, type=cv2.THRESH_BINARY)

        # Thresholding the image to limit the connected components to the CCCs
        ksize = 2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
        # cv2.imshow("Image after Opening and Closing", binary_img)

        # Find connected component labels and Centroids for all white blobs.
        num_white_labels, labels_white_img, stats_white, centroids_white = cv2.connectedComponentsWithStats(binary_img)
        # labels_display = cv2.normalize(src=labels_white_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # cv2.imshow("White Components image", labels_display)
        # Find connected component labels for all black blobs.
        num_black_labels, labels_black_img, stats_black, centroids_black = cv2.connectedComponentsWithStats(cv2.bitwise_not(binary_img))
        # labels_black_display = cv2.normalize(src=labels_black_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,  dtype=cv2.CV_8U)
        # cv2.imshow("Black Components image", labels_black_display)

        # Checking which white centroids are within black centroids and drawing markers at CCCs
        count = 0
        tolerance = 1.3
        CCC_centroids = np.zeros((10,2))
        for i in range(num_black_labels):
            xc_b, yc_b = centroids_black[i]
            for j in range(num_white_labels):
                xc_w, yc_w = centroids_white[j]
                if np.absolute(xc_b-xc_w) < tolerance and np.absolute(yc_b-yc_w) < tolerance :
                    CCC_centroids[count, :] = np.array([xc_w, yc_w])
                    cv2.drawMarker(frame, position=(int(xc_w), int(yc_w)), color=(0, 0, 255),
                                   markerType=cv2.MARKER_CROSS)
                    count = count + 1

        # Writing the Frame Number
        cv2.putText(frame, text=str(frame_number), org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        color=(255, 0, 0), fontScale=1.2)
        cv2.imshow("Video with marked CCCs", frame)

        frame_output = frame.copy()
        videoWriter.write(frame_output)

        # Wait for xx msec (0 = wait till keypress).
        key_pressed = cv2.waitKey(1) & 0xFF

        if key_pressed == 27 or key_pressed == ord('q'):
            break  # Quit on ESC or q

        frame_number = frame_number + 1
        
    videoWriter.release()

if __name__ == "__main__":
    main()
