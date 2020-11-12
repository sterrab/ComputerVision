import numpy as np
import cv2


# Utility function to create an image window.
def create_named_window(window_name, image):
    # WINDOW_NORMAL allows resize; use WINDOW_AUTOSIZE for no resize.
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    h = image.shape[0]  # image height
    w = image.shape[1]  # image width

    # Shrink the window if it is too big (exceeds some maximum size).
    WIN_MAX_SIZE = 1000
    if max(w, h) > WIN_MAX_SIZE:
        scale = WIN_MAX_SIZE / max(w, h)
    else:
        scale = 1
    cv2.resizeWindow(winname=window_name, width=int(w * scale), height=int(h * scale))


def get_xy(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        window_name, image, point_list = param  # Unpack parameters
        cv2.rectangle(image, pt1=(x-15, y-15), pt2=(x+15, y+15), color=(0,0,255),
                      thickness=3)
        cv2.putText(image, str(len(point_list)), org=(x,y-15), color=(0,0,255),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, thickness=2)
        cv2.imshow(window_name, image)
        point_list.append((x,y))





def main():


    oldSign = cv2.imread('US_Cellular.jpg')
    blaster = cv2.imread('blaster-logos.jpg')


    # Create two lists.  The (x,y) points go in these lists.
    oldSignPts = []
    blasterPts = []

    # Display images.
    displayA = oldSign.copy()
    displayB = blaster.copy()
    create_named_window("Old Sign", displayA)
    create_named_window("Blaster", displayB)
    cv2.imshow("Old Sign", displayA)
    cv2.imshow("Blaster", displayB)

    # Assign the mouse callback function, which collects (x,y) points.
    cv2.setMouseCallback("Old Sign", on_mouse=get_xy, param=("Old Sign", displayA, oldSignPts))
    cv2.setMouseCallback("Blaster", on_mouse=get_xy, param=("Blaster", displayB, blasterPts))



    # Loop until user hits the ESC key.
    print("Click on points.  Hit ESC to exit.")
    while True:
        if cv2.waitKey(100) == 27:  # ESC is ASCII code 27
            if not len(oldSignPts) == len(blasterPts):
                print("Error: you need same number of points in both images!")
            else:
                break
    print("oldSignPts:", oldSignPts)  # Print points to the console
    print("blasterPts:", blasterPts)


    input_image_height = oldSign.shape[0]
    input_image_width = oldSign.shape[1]

    oldSignPts = np.asarray(oldSignPts)
    blasterPts  = np.asarray(blasterPts)

    # Find    homography    between    image    0 and image    1.
    H_0_1, _ = cv2.findHomography(blasterPts,oldSignPts)


    bgr_output0 = cv2.warpPerspective(blaster, H_0_1, (input_image_width,
                                                       input_image_height))

    oldSign[np.nonzero(bgr_output0)] = bgr_output0[np.nonzero(bgr_output0)]

    newSign = oldSign
    # cv2.imwrite('newSign.jpg',newSign)
    create_named_window('New Sign',newSign)
    cv2.imshow('New Sign', newSign)


    create_named_window("scaled and placed", bgr_output0)
    cv2.imshow("scaled and placed", bgr_output0)

    cv2.waitKey(0)


if __name__ == "__main__":
    main()