# S. Terrab - CSCI 507 Computer Vision - Fall 2020
# Homework 4

# The video hw4.avi is a video of a CD player with two ArUco tags attached to it.
# Your mission is to use computer vision to recognize and identify the tags, and
# display the location of the ``on/off" switch to the user.
# The markers are $4\times4$ bit patterns, size is 2 inches on a side, and the
# markers were created by specifying 100 markers in the dictionary.
# The marker on the left is id=0; the marker on the right is id=1.
#
# Here is the location of the switch (in inches):
#  With respect to marker 0: $(X, Y, Z) = (2.5;\ -2.0; \ -1.0)$.
#  With respect to marker 1: $(X, Y, Z) = (-2.5;\ -2.0; \ -5.0)$.
#
# The camera's intrinsic parameters are $f_x = 675,\ f_y = 675,\ c_x = 320,\ c_y = 240$,
# assume no distortion.
#
# Your program should recognize a marker, determine its pose, and
# overlay a pyramid on the image showing the location of the on/off switch as an overlay.


import cv2
import numpy as np

def main():
    f = 675.0
    cx = 320.0
    cy = 240.0
    K = np.array(((f, 0, cx), (0, f, cy), (0, 0, 1)))

    # Reading  images from the video file
    video_capture = cv2.VideoCapture("hw4.avi")  # Open video capture object
    got_image, frame = video_capture.read()  # Make sure we can read video
    if not got_image:
        print("Cannot read video source")
        sys.exit()

    # File to Save Video
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter("HW4_ARvideo.avi", fourcc=fourcc, fps=30.0,
                                  frameSize=(np.shape(frame)[1], np.shape(frame)[0]))

    # Find ArUco tag and include pyramid to identify switch location
    while True:
        got_image, frame = video_capture.read()
        if not got_image:
            break  # End of video; exit the while loop

        # Coordinates of Pyramid
        pyramid = np.array([[-0.5, 0.5, 0.5, -0.5, 0],
                            [-0.5, -0.5, 0.5, 0.5, 0],
                            [2, 2, 2, 2, 0],
                             np.ones(5)])

        # Get the pattern dictionary for 4x4 markers, with ids 0 through 99.
        arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        corners, ids, _ = cv2.aruco.detectMarkers(image=frame, dictionary=arucoDict)
        mLength = 2

        if ids is not None:
            # Draw square boundary of marker
            cv2.aruco.drawDetectedMarkers(image=frame, corners=corners, ids=ids, borderColor=(0, 0, 255))
            # Finding Pose of Marker
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners=corners, markerLength=mLength,
                                                              cameraMatrix=K, distCoeffs=None)

            # Get the pose of the first detected marker with respect to the camera.
            rm_c = rvecs[0]                 # This is a 1x3 rotation vector
            tm_c = tvecs[0]                 # This is a 1x3 translation vector

            # Draw XYZ coordinate axes of Pose of Marker
            cv2.aruco.drawAxis(image=frame, cameraMatrix=K, distCoeffs=None, rvec=rm_c, tvec=tm_c, length=int(mLength/2))

            # Homogeneous Transformations ----------
            #MARKER {M} wrt CAMERA {C}
            R = cv2.Rodrigues(rm_c)[0]
            #   H_M_C means transform M to C
            H_M_C = np.block([[R, tm_c.T], [0, 0, 0, 1]])

            #SWITCH {S} wrt MARKER {M}
            if ids == 0:
                # Translation: origin of P in M.
                tSorg_M = np.array([[2.5, -2.0, -1.0]]).T
                # tSorg_M = np.zeros((3,1))
                ay = np.deg2rad(90)
            elif ids == 1:
                tSorg_M = np.array([[-2.5, -2.0, -5.0]]).T
                # tSorg_M = np.zeros((3, 1))
                ay = np.deg2rad(-90)
            # Rotation of S wrt M
            sy = np.sin(ay)
            cy = np.cos(ay)
            Ry = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))
            # H_S_M means transform S to M.
            H_S_M = np.block([[Ry, tSorg_M], [0, 0, 0, 1]])

            #SWITCH {S} wrt CAMERA {C} - and -  Projecting Pyramid onto Image
            H_S_C = H_M_C @ H_S_M
            # Extrinsic Matrix
            Mext = H_S_C[0:3, :]
            # Pyramid In Camera Coordinates
            p_C = K @ Mext @ pyramid
            p_C = p_C / p_C[2, :]

            # Adding Wireframe to adjacent points of projected Pyramid
            for i in range(4):
                cv2.line(frame, (p_C[0, i].astype(int), p_C[1, i].astype(int)),
                         (p_C[0, i + 1].astype(int), p_C[1, i + 1].astype(int)), (0,0, 255), thickness=2)

            # Adding Wireframe to other 3 base points to vertex of pyramid
            cv2.line(frame, (p_C[0, 0].astype(int), p_C[1, 0].astype(int)),
                     (p_C[0, 4].astype(int), p_C[1, 4].astype(int)), (0,0, 255), thickness=2)
            cv2.line(frame, (p_C[0, 1].astype(int), p_C[1, 1].astype(int)),
                     (p_C[0, 4].astype(int), p_C[1, 4].astype(int)), (0,0, 255), thickness=2)
            cv2.line(frame, (p_C[0, 2].astype(int), p_C[1, 2].astype(int)),
                     (p_C[0, 4].astype(int), p_C[1, 4].astype(int)), (0,0, 255), thickness=2)
            # Connecting First and Fourth base corners
            cv2.line(frame, (p_C[0, 0].astype(int), p_C[1, 0].astype(int)),
                     (p_C[0, 3].astype(int), p_C[1, 3].astype(int)), (0,0, 255), thickness=2)

        # Viewing the frame after adding pose and pyramid pointing onto switch
        cv2.imshow("Switch Location of CD player", frame)

        # Writing to Video
        frame_output = frame.copy()
        videoWriter.write(frame_output)

        # Wait for xx msec (0 = wait till keypress).
        key_pressed = cv2.waitKey(1) & 0xFF

        if key_pressed == 27 or key_pressed == ord('q'):
            break  # Quit on ESC or q

    videoWriter.release()

if __name__ == "__main__":
    main()