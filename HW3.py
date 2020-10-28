# S. Terrab - CSCI 507 Computer Vision - Fall 2020
# Homework 3

# The video ``fiveCCC.mp4"  shows a target composed of five concentric contrasting circle (CCC) features.

# 1. Read and display each image of the video, and print the frame number on the displayed image.
# 2. Write a program that finds the five CCCs in each image, and also finds the correspondence between the
# image features and the model features.
# 3. Find a picture that you like and map it onto the target plane in each image, using a homography
# (projective transform). The picture should look like it is attached to the plane of the target.
# 4. Find the pose of the target with respect to the camera in each frame of the video. Draw the XYZ coordinate
# axes of the target as an overlay on the image, and also print the pose values on the image, in terms of translation (in inches) and the rotation vector.


import cv2
import numpy as np

## Image to Overlay in Frame, Taken from https://www.mines.edu/webcentral/logos/
logo = cv2.imread('CO-Mines-logo-stacked-4C.jpg')
h = logo.shape[0]
w = logo.shape[1]
logoPoints = np.array([[0, 0],
                       [int(w/2),0],
                       [w, 0],
                       [0, h],
                       [w, h]])

def main():
    f = 531.0
    cx = 320
    cy = 240
    K = np.array(((f, 0, cx), (0, f, cy), (0, 0, 1)))

    TargetPoints = np.array([[-3.7, -2.275, 0],
                       [0, -2.275, 0],
                       [3.7, -2.275, 0],
                       [-3.7, 2.275, 0],
                       [3.7, 2.275, 0]])

    # Reading  images from the video file
    video_capture = cv2.VideoCapture("fiveCCC.mp4")  # Open video capture object
    got_image, frame = video_capture.read()  # Make sure we can read video
    input_image_height = frame.shape[0]
    input_image_width = frame.shape[1]
    if not got_image:
        print("Cannot read video source")
        sys.exit()

    # File to Save Video
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter("HW3_video.avi", fourcc=fourcc, fps=30.0,
                                  frameSize=(np.shape(frame)[1], np.shape(frame)[0]))

    # Read/label images and mark CCCs until end of video is reached.
    frame_number = 0
    while True:
        got_image, frame = video_capture.read()
        if not got_image:
            break  # End of video; exit the while loop

        # Converting image to grayscale and then binary using threshold as defined in lecture
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(gray_img, thresh=120, maxval=255, type=cv2.THRESH_BINARY)

        # Thresholding the image to limit the connected components to the CCCs
        ksize = 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

        # Find connected component labels and Centroids for all white blobs.
        num_white_labels, labels_white_img, stats_white, centroids_white = cv2.connectedComponentsWithStats(binary_img)
        # Find connected component labels for all black blobs.
        num_black_labels, labels_black_img, stats_black, centroids_black = cv2.connectedComponentsWithStats(
            cv2.bitwise_not(binary_img))

        # Checking which white centroids are within black centroids and drawing markers at CCCs
        count = 0
        tolerance = 1.3
        CCC_centroids = np.zeros((20, 2))
        for i in range(num_black_labels):
            xc_b, yc_b = centroids_black[i]
            for j in range(num_white_labels):
                xc_w, yc_w = centroids_white[j]
                if np.absolute(xc_b - xc_w) < tolerance and np.absolute(yc_b - yc_w) < tolerance:
                    CCC_centroids[count, :] = np.array([xc_w, yc_w])
                    count = count + 1

        # Identifying the correspondence to the 0, 1, 2, 3, 4 points in target image.
        # For frame 0, the first 5 centroids happened to correspond to the respective 0-4 points in target image.
        #    (This was verified by drawing the number at each point and verifying the correspondence.)
        tol = 10
        if frame_number == 0:
            OrderedPoints = np.zeros((5, 2))
            OrderedPoints[0,:] = CCC_centroids[0,:]
            OrderedPoints[1, :] = CCC_centroids[1, :]
            OrderedPoints[2, :] = CCC_centroids[2, :]
            OrderedPoints[3, :] = CCC_centroids[3, :]
            OrderedPoints[4, :] = CCC_centroids[4, :]

            # The position of the points is OrderedPoints and is checked for any pixel change for subsequent frames
            # for a motion of up to tol pixels in x and y.
            # (This was verified by drawing the number by each point for correspondence; commented below .)
        elif frame_number != 0:
            for i in range(5):
                for j in range(20):
                    if np.abs(CCC_centroids[j, 0] - OrderedPoints[i, 0]) < tol and np.abs(CCC_centroids[j, 1] - OrderedPoints[i, 1]) < tol:
                        OrderedPoints[i, :] = CCC_centroids[j, :]
                        # cv2.putText(frame, text=str(i), org=(int(OrderedPoints[i, 0]) - 10,
                        #                                      int(OrderedPoints[i, 1]) - 10),
                        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, thickness=2,
                        #             color=(0, 0, 255), fontScale=0.5)

        # Mapping image onto the points using a homography
        H_0_1, _ = cv2.findHomography(logoPoints, OrderedPoints)
        bgr_output0 = cv2.warpPerspective(logo, H_0_1, (input_image_width, input_image_height))
        frame[np.nonzero(bgr_output0)] = bgr_output0[np.nonzero(bgr_output0)]
        cv2.imshow("scaled and placed", bgr_output0)

        # Taken from Alignment Nonlinear Slides --------
        isPoseFound, rvec, tvec = cv2.solvePnP(objectPoints=TargetPoints, imagePoints=OrderedPoints, cameraMatrix=K,
                                               distCoeffs=None)

        # Draw coordinate axes onto the image.  Scale the length of the axes
        # according to the size of the model, so that the axes are visible.
        W = np.amax(TargetPoints, axis=0) - np.amin(TargetPoints, axis=0)  # Size of model in X,Y,Z
        L = np.linalg.norm(W)  # Length of the diagonal of the bounding box
        d = L / 5  # This will be the length of the coordinate axes
        pAxes = np.float32([[0, 0, 0],  # origin
                            [d, 0, 0],  # x axis
                            [0, d, 0],  # y axis
                            [0, 0, d]  # z axis
                            ])
        pImg, J = cv2.projectPoints(objectPoints=pAxes, rvec=rvec, tvec=tvec, cameraMatrix=K, distCoeffs=None)
        pImg = pImg.reshape(-1, 2)  # reshape from size (N,1,2) to (N,2)
        cv2.line(frame, tuple(np.int32(pImg[0])), tuple(np.int32(pImg[1])), (0, 0, 255), 3)  # x
        cv2.line(frame, tuple(np.int32(pImg[0])), tuple(np.int32(pImg[2])), (0, 255, 0), 3)  # y
        cv2.line(frame, tuple(np.int32(pImg[0])), tuple(np.int32(pImg[3])), (255, 0, 0), 3)  # z

        # Writing rotation and translation vectors on frame
        cv2.putText(frame, text='rvec=(%.2f,%.2f,%.2f)' % (rvec[0], rvec[1], rvec[2]),
                    org=(cx - 250, cy + 100),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, thickness=2,
                    color=(255, 255, 255), fontScale=0.9)
        cv2.putText(frame, text='tvec=(%.2f,%.2f,%.2f)' % (tvec[0], tvec[1], tvec[2]),
                    org=(cx - 250, cy + 130),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, thickness=2,
                    color=(255, 255, 255), fontScale=0.9)

        # Writing the Frame Number
        cv2.putText(frame, text=str(frame_number), org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, thickness=2,
                    color=(255, 255, 255), fontScale=1.2)

        # Viewing the frame after all added changes
        cv2.imshow("Video with Pose of Target and overlayed Mines Logo", frame)

        # Writing to Video
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