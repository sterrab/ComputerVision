# S. Terrab - CSCI 507 Computer Vision - Homework 5
# Image Stitching a Mural


import cv2
import numpy as np


def main():

    # bgr_image1 = cv2.imread('mural01.jpg')
    bgr_image1 = cv2.imread('desk01.png')

    # # RAN ONCE TO FIND PTS:
    # # Manually pick the four (x,y) corners of the first panel (clockwise from top left corner).
    # cv2.imshow('Click on the 4 corners', bgr_image1)
    # cv2.setMouseCallback(window_name='Click on the 4 corners', on_mouse=get_xy, param=bgr_image1)
    # cv2.waitKey(0)

    # Mural Points
    # pts1 = np.array([[155, 116], [382, 64], [304, 500], [26, 478]])

    # Desk Points
    pts1 = np.array([[157, 301], [249, 275], [282, 384], [184, 415]])

    # Specify the corresponding points in the output orthophoto -  (clockwise from top left corner).
    # Mural Points
    # width = 159 cm * (1pix/0.5cm) = 318; height = 218cm * (1pix/0.5cm) = 436 pix
    # pts1_ortho = np.array([[0, 0], [318, 0], [318, 436], [0, 436]])
    # pts1_ortho[:, 0] += 50  # Add x offset
    # pts1_ortho[:, 1] += 100  # Add y offset

    # Desk Points
    # width = 10.5 cm * (1pix/0.5cm) = 21; height = 14cm * (1pix/0.5cm) = 28 pix
    pts1_ortho = np.array([[0, 0], [21, 0], [21, 28], [0, 28]])
    pts1_ortho[:, 0] += 50  # Add x offset
    pts1_ortho[:, 1] += 120  # Add y offset

    # Find the homography to map the input image to the orthophoto image.
    H1, _ = cv2.findHomography(srcPoints=pts1, dstPoints=pts1_ortho)

    # Mural Dimensions
    # output_width = 5200
    # output_height = 750

    # Desk Dimensions
    output_width = 500
    output_height = 600

    # Warp the image to the orthophoto.
    image_mosaic = cv2.warpPerspective(bgr_image1, H1, (output_width, output_height))
    cv2.imshow('Orthophoto', image_mosaic)
    cv2.waitKey(0)

    H_prev_mosaic = H1
    image_previous = bgr_image1

    count = 1
    num_of_images = 6 #12
    for image in range(num_of_images - 1):
        # Defining Current Image
        count += 1
        if count < 10:
            num = '0' + str(count)
        else:
            num = str(count)

        # filename = 'mural' + num + '.jpg'
        filename = 'desk' + num + '.png'
        image_current = cv2.imread(filename)

        #Compute Homography between current and previous image
        # ---- Extract keypoints and descriptors.
        kp_train, desc_train = detect_features(image_current, show_features=False)
        kp_query, desc_query = detect_features(image_previous, show_features=False)

        # --- Match query image descriptors to the training image.
        # ------Use k nearest neighbor matching and apply ratio test.
        matcher = cv2.BFMatcher.create(cv2.NORM_L2)
        matches = matcher.knnMatch(desc_query, desc_train, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good.append(m)
        matches = good

        src_pts = np.float32([kp_train[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_query[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Calculating homography
        H_current_prev, _ = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC)

        # New homography
        H_current_mosaic = H_prev_mosaic @ H_current_prev

        # Warp the image to the orthophoto.
        image_warped = cv2.warpPerspective(image_current, H_current_mosaic, (output_width, output_height))
        cv2.imshow('Warped Image', image_warped)
        cv2.waitKey(0)

        # Combine the images.
        mosaic_output = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        mosaic_output = fuse_color_images(mosaic_output, image_warped)
        mosaic_output = fuse_color_images(mosaic_output, image_mosaic)
        cv2.imshow("Mosaic in Progress", mosaic_output)
        cv2.waitKey(0)

        # Updating Images and Homography transformations
        image_previous = image_current
        H_prev_mosaic = H_current_mosaic
        image_mosaic = mosaic_output

    cv2.imshow("Final Mosaic", image_mosaic)
    cv2.waitKey(0)
    # cv2.imwrite('HW5_mural_mosaic.jpg', image_mosaic)
    cv2.imwrite('HW5_desk_mosaic.jpg', image_mosaic)

# Mouse callback function. Appends the x,y location of mouse click to a list.
def get_xy(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)


# Detect features in the image and return the keypoints and descriptors.
def detect_features(bgr_img, show_features=False):
    detector = cv2.xfeatures2d.SURF_create(
        hessianThreshold=100,  # default = 100
        nOctaves=4,  # default = 4
        nOctaveLayers=3,  # default = 3
        extended=False,  # default = False
        upright=False  # default = False
    )

    # Extract keypoints and descriptors from image.
    gray_image = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray_image, mask=None)

    # Optionally draw detected keypoints.
    if show_features:
        # Possible flags: DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, DRAW_MATCHES_FLAGS_DEFAULT
        bgr_display = bgr_img.copy()
        cv2.drawKeypoints(image=bgr_display, keypoints=keypoints,
                          outImage=bgr_display,
                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Features", bgr_display)
        print("Number of keypoints: ", len(keypoints))
        cv2.waitKey(0)

    return keypoints, descriptors

# Fuse two color images.  Assume that zero indicates an unknown value.
# At pixels where both values are known, the output is the average of the two.
# At pixels where only one is known, the output uses that value.
def fuse_color_images(A, B):
    assert(A.ndim == 3 and B.ndim == 3)
    assert(A.shape == B.shape)

    # Allocate result image.
    C = np.zeros(A.shape, dtype=np.uint8)

    # Create masks for pixels that are not zero.
    A_mask = np.sum(A, axis=2) > 0
    B_mask = np.sum(B, axis=2) > 0

    # Compute regions of overlap.
    A_only = A_mask & ~B_mask
    B_only = B_mask & ~A_mask
    A_and_B = A_mask & B_mask
    C[A_only] = A[A_only]
    C[B_only] = B[B_only]
    C[A_and_B] = 0.5 * A[A_and_B] + 0.5 * B[A_and_B]
    return C

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


if __name__ == "__main__":
    main()
