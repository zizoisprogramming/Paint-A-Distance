import cv2
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('webcam_skin_detection')

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def display(title, img, max_size=200000):
    assert isinstance(img, np.ndarray), 'img must be a np array'
    assert isinstance(title, str), 'title must be a string'
    scale = np.sqrt(min(1.0, float(max_size) / (img.shape[0] * img.shape[1])))
    shape = (int(scale * img.shape[1]), int(scale * img.shape[0]))
    img = cv2.resize(img, shape)
    cv2.imshow(title, img)

def get_hsv_mask(img, debug=False):
    assert isinstance(img, np.ndarray), 'image must be a np array'
    assert img.ndim == 3, 'skin detection can only work on color images'
    # logger.debug('getting hsv mask')

    lower_thresh = np.array([20, 0, 0], dtype=np.uint8)
    upper_thresh = np.array([120, 255, 255], dtype=np.uint8)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    msk_hsv = cv2.inRange(img_hsv, lower_thresh, upper_thresh)

    msk_hsv[msk_hsv < 128] = 0
    msk_hsv[msk_hsv >= 128] = 1

    if debug:
        display('input', img)
        display('mask_hsv', msk_hsv)

    return msk_hsv.astype(float)


def get_rgb_mask(img, debug=False):
    assert isinstance(img, np.ndarray), 'image must be a np array'
    assert img.ndim == 3, 'skin detection can only work on color images'
    # logger.debug('getting rgb mask')

    # Define thresholds for RGB
    lower_thresh = np.array([45, 52, 108], dtype=np.uint8)
    upper_thresh = np.array([255, 255, 255], dtype=np.uint8)

    # Mask for thresholding RGB range
    mask_a = cv2.inRange(img, lower_thresh, upper_thresh)

    # Additional conditions
    red_greater_green = img[:, :, 2] > img[:, :, 1]  # Red > Green
    green_greater_blue = img[:, :, 1] > img[:, :, 0]  # Green > Blue

    # Convert boolean conditions to uint8 mask
    mask_red_green = red_greater_green.astype(np.uint8) * 255
    mask_green_blue = green_greater_blue.astype(np.uint8) * 255

    # Combine RGB range mask with additional conditions
    mask_combined = np.bitwise_and(mask_a, mask_red_green)
    mask_combined = np.bitwise_and(mask_combined, mask_green_blue)

    # Contrast and color difference conditions
    mask_b = 255 * ((img[:, :, 2] - img[:, :, 1]) / 20)  # Red - Green scaled
    mask_c = 255 * ((np.max(img, axis=2) - np.min(img, axis=2)) / 20)  # Color contrast

    # Combine all masks
    mask_d = np.bitwise_and(np.uint64(mask_combined), np.uint64(mask_b))
    msk_rgb = np.bitwise_and(np.uint64(mask_c), np.uint64(mask_d))

    # Binarize the final mask
    msk_rgb[msk_rgb < 128] = 0
    msk_rgb[msk_rgb >= 128] = 1

    # Debug visualization
    if debug:
        display('input', img)
        display('mask_rgb', msk_rgb)

    return msk_rgb.astype(float)



def get_ycrcb_mask(img, debug=False):
    assert isinstance(img, np.ndarray), 'image must be a np array'
    assert img.ndim == 3, 'skin detection can only work on color images'
    # logger.debug('getting ycrcb mask')

    lower_thresh = np.array([90, 100, 130], dtype=np.uint8)
    upper_thresh = np.array([230, 120, 180], dtype=np.uint8)

    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    msk_ycrcb = cv2.inRange(img_ycrcb, lower_thresh, upper_thresh)

    msk_ycrcb[msk_ycrcb < 128] = 0
    msk_ycrcb[msk_ycrcb >= 128] = 1

    if debug:
        display('input', img)
        display('mask_ycrcb', msk_ycrcb)

    return msk_ycrcb.astype(float)


def remove_face_region(frame, mask):
    """
    Detects faces in the frame using Haar Cascade and removes the face region from the skin mask.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        # Remove the face region from the mask
        cv2.rectangle(mask, (x, y), (x + w, y + h), (0, 0, 0), thickness=-1)

    return mask

def closing(mask):
    assert isinstance(mask, np.ndarray), 'mask must be a np array'
    assert mask.ndim == 2, 'mask must be a greyscale image'
    # logger.debug("closing mask of shape {0}".format(mask.shape))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    return mask

def process(img, thresh=0.5, debug=False):
    """
    Detect skin regions in the image using multiple color spaces.
    """
    assert isinstance(img, np.ndarray), 'image must be a np array'
    assert img.ndim == 3, 'skin detection can only work on color images'

    # Generate masks using different color spaces
    mask_hsv = get_hsv_mask(img, debug=debug)
    mask_rgb = get_rgb_mask(img, debug=debug)
    mask_ycrcb = get_ycrcb_mask(img, debug=debug)

    # Combine masks
    n_masks = 3.0
    mask = (mask_hsv + mask_rgb + mask_ycrcb) / n_masks
    mask[mask < thresh] = 0.0
    mask[mask >= thresh] = 255.0

    # Convert to binary and refine the mask
    mask = mask.astype(np.uint8)
    mask = closing(mask)

    # Remove the face region from the mask
    # mask = remove_face_region(img, mask)

    return mask

def remove_noise(mask):
    """
    Apply noise removal techniques to the binary mask.
    """
    assert isinstance(mask, np.ndarray), 'mask must be a numpy array'
    assert mask.ndim == 2, 'mask must be a greyscale image'
    # logger.debug("Removing noise from mask")

    # Define a kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Apply morphological opening (remove small objects from the foreground)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Apply morphological closing (fill small holes in the foreground)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Optionally, apply a Gaussian blur to smooth the edges
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    return mask

def isolate_hand(mask, debug=False):
    """
    Isolate the hand by finding the largest contour in the skin mask.
    """
    assert isinstance(mask, np.ndarray), 'mask must be a numpy array'
    assert mask.ndim == 2, 'mask must be a greyscale image'

    # logger.debug("Isolating hand using contours")

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour (assuming it's the hand)
        largest_contour = max(contours, key=cv2.contourArea)

        # Create a new mask for the largest contour
        hand_mask = np.zeros_like(mask)
        cv2.drawContours(hand_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        if debug:
            display('Hand Mask', hand_mask)

        return hand_mask
    else:
        logger.warning("No contours found in the mask")
        return np.zeros_like(mask)

import cv2
import numpy as np
import logging

# Add skeletonization and finger counting
def skeletonize(mask):
    """
    Skeletonize the binary hand mask.
    """
    assert isinstance(mask, np.ndarray), 'mask must be a numpy array'
    assert mask.ndim == 2, 'mask must be a grayscale image'

    # logger.debug("Skeletonizing hand mask")

    # Use OpenCV's thinning function (skeletonization)
    skeleton = cv2.ximgproc.thinning(mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    return skeleton

import numpy as np
import cv2

import numpy as np
import cv2
import os
def save_unique_image(base_name, image):
    count = 1
    name, ext = os.path.splitext(base_name)
    while os.path.exists(base_name):
        base_name = f"{name}_{count}{ext}"
        count += 1
    cv2.imwrite(base_name, image)

def calculate_defects(contour, hull_indices, debug_image=None):
    defects = []
    print("h")
    for i in range(len(hull_indices)):
        start_idx = hull_indices[i][0]
        end_idx = hull_indices[(i + 1) % len(hull_indices)][0]

        start_point = contour[start_idx][0]
        end_point = contour[end_idx][0]
        print("h")

        between_indices = (
            range(start_idx + 1, end_idx)
            if start_idx < end_idx
            else list(range(start_idx + 1, len(contour))) + list(range(0, end_idx))
        )

        min_theta = np.pi  # Start with the largest possible angle
        min_point = None
        max_depth = 0

        for idx in between_indices:
            point = contour[idx][0]

            # Vectors from the point to the hull start and end points
            vector1 = np.array(start_point) - np.array(point)
            vector2 = np.array(end_point) - np.array(point)

            dot_product = np.dot(vector1, vector2)
            magnitude1 = np.linalg.norm(vector1)
            magnitude2 = np.linalg.norm(vector2)
            print(debug_image)
            if debug_image is not None:
                cv2.line(debug_image, tuple(start_point), tuple(point), (255, 0, 0), 2)  # vector1 in blue
                cv2.line(debug_image, tuple(end_point), tuple(point), (0, 255, 0), 2)  # vector2 in green
                cv2.circle(debug_image, tuple(point), 5, (0, 0, 255), -1)  # defect point in red
                save_unique_image("vectors.jpg", debug_image)

            if magnitude1 > 0 and magnitude2 > 0:
                theta = np.arccos(dot_product / (magnitude1 * magnitude2))  # Angle between vectors

                # Depth: Perpendicular distance from point to the line segment
                depth = np.linalg.norm(np.cross(vector2, vector1)) / magnitude1

                # Update the minimum angle and corresponding point
                if theta < min_theta and depth > max_depth:  # Smallest angle and significant depth
                    min_theta = theta
                    min_point = (idx, point)
                    max_depth = depth

        if min_point:
            # Draw debug vectors if an image is provided

            defects.append((start_idx, end_idx, min_point[0]))

    return defects



def count_fingers2(mask, debug=True):
    import numpy as np
    import cv2
    import os

    def save_unique_image(base_name, image):
        count = 1
        name, ext = os.path.splitext(base_name)
        while os.path.exists(base_name):
            base_name = f"{name}_{count}{ext}"
            count += 1
        cv2.imwrite(base_name, image)

    if mask.dtype == np.bool_:
        mask = mask.astype(np.uint8) * 255

    # Convert mask to color for visualization
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if debug:
        print(f"Number of contours found: {len(contours)}")

    if not contours:
        if debug:
            print("No contours found.")
        return 0

    largest_contour = max(contours, key=cv2.contourArea)
    if len(largest_contour) < 3:
        if debug:
            print("Largest contour has less than 3 points.")
        return 0

    # Step 3: Compute bounding rectangle
    rect = cv2.minAreaRect(largest_contour)
    box_center = np.array(rect[0])  # Center of the box (x, y)
    box_height = rect[1][1]  # Height of the box
    if debug:
        print(f"Box center: {box_center}, Box height: {box_height}")

    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    if debug:
        print(f"Simplified contour points: {len(simplified_contour)}")

    hull = cv2.convexHull(simplified_contour, returnPoints=False)
    if hull is None or len(hull) < 3:
        if debug:
            print("Convex hull is None or has less than 3 points.")
        return 0
    if debug:
        print(f"Convex hull indices: {hull.flatten()}")

    # Draw convex hull points on the mask
    for i in hull.flatten():
        cv2.circle(mask_color, tuple(simplified_contour[i][0]), 5, (0, 255, 255), -1)  # Yellow points
    # save_unique_image("convex_hull_points.jpg", mask_color)

    # Validate hull indices
    try:
        defects = cv2.convexityDefects(simplified_contour, hull)
        return 0
    except cv2.error as e:
        if debug:
            print(f"Error in cv2.convexityDefects: {e}")

    if defects is None:
        if debug:
            print("No convexity defects found.")
        return 0

    # Draw box center on the mask
    cv2.circle(mask_color, (int(box_center[0]), int(box_center[1])), 5, (0, 255, 255), -1)  # Yellow point
    # save_unique_image("box_center.jpg", mask_color)

    finger_count = 0
    for start_idx, end_idx, far_idx in defects:
        start_point = tuple(simplified_contour[start_idx][0])
        depth_point = tuple(simplified_contour[far_idx][0])
        cv2.circle(mask_color, depth_point, 5, (0, 255, 255), -1)  # Yellow point for depth
        cv2.line(mask_color, start_point, depth_point, (0, 255, 255), 2)
        save_unique_image("defects_points.jpg", mask_color)

        if (
            (start_point[1] < box_center[1] and depth_point[1] < box_center[1]) and
            start_point[1] < depth_point[1] and
            np.linalg.norm(np.array(start_point) - np.array(depth_point)) > box_height / 6.5
        ):
            finger_count += 1

            if debug:
                print(f"Finger detected: Start point {start_point}, Depth point {depth_point}")
                cv2.circle(mask_color, depth_point, 5, (0, 255, 0), -1)  # Yellow point for depth
                cv2.line(mask_color, start_point, depth_point, (0, 255, 0), 2)  # Yellow line
    
    save_unique_image("defects_points s.jpg", mask_color)

    if debug:
        print(f"Total fingers counted: {finger_count}")

    return finger_count



def count_fingers(mask, debug=False):
    """
    Count fingers in the binary hand mask.
    """
    assert isinstance(mask, np.ndarray), 'mask must be a numpy array'
    assert mask.ndim == 2, 'mask must be a grayscale image'

    # Find contours of the hand
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.warning("No contours found in the mask")
        return 0

    # Use the largest contour (assuming it’s the hand)
    largest_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest_contour, returnPoints=False)
    defects = cv2.convexityDefects(largest_contour, hull)

    if defects is None:
        logger.warning("No convexity defects found")
        return 0

    # Count fingers based on convexity defects
    finger_count = 0
    for i in range(defects.shape[0]):
        start_idx, end_idx, far_idx, depth = defects[i, 0]
        start_point = tuple(largest_contour[start_idx][0])
        end_point = tuple(largest_contour[end_idx][0])
        far_point = tuple(largest_contour[far_idx][0])

        # Calculate the angles to detect fingers
        a = np.linalg.norm(np.array(start_point) - np.array(far_point))
        b = np.linalg.norm(np.array(end_point) - np.array(far_point))
        c = np.linalg.norm(np.array(start_point) - np.array(end_point))
        angle = np.arccos((a**2 + b**2 - c**2) / (2 * a * b))

        # Angle threshold to filter fingers (about 90 degrees)
        if angle < np.pi / 2 and depth > 10000:  # Depth threshold to avoid noise
            finger_count += 1

            # Optional: Draw on the image for debugging
            if debug:
                cv2.circle(mask, far_point, 5, (255, 0, 0), -1)
                cv2.line(mask, start_point, far_point, (0, 255, 0), 2)
                cv2.line(mask, end_point, far_point, (0, 255, 0), 2)

    # Return the number of fingers (add 1 for the thumb)
    return finger_count + 1
from scipy.signal import convolve
def detect_fingertips_gray(hand_mask, skeleton, threshold=150, debug=False):
    """
    Detects fingertips from a grayscale skeletonized hand image.

    Args:
        hand_mask (numpy.ndarray): The original hand mask frame.
        skeleton (numpy.ndarray): The grayscale skeletonized hand image.
        threshold (int): Threshold for binarizing the skeleton. Default is 50.
        debug (bool): If True, saves or prints intermediate matrices for debugging.

    Returns:
        list: List of (x, y) coordinates of detected fingertips.
    """
    # Binarize the skeleton
    _, skeleton_binary = cv2.threshold(skeleton, 1, 255, cv2.THRESH_BINARY)
    skeleton_binary = skeleton_binary // 255  # Normalize to 0 and 1

    if debug:
        # Save the binary skeleton as an image for inspection
        cv2.imwrite('skeleton_binary_debug.png', (skeleton_binary * 255).astype(np.uint8))
        print("Binary skeleton saved as 'skeleton_binary_debug.png'")

        # Optionally, print the matrix (only practical for small matrices)
        np.set_printoptions(threshold=np.inf)  # Allow full array print
        print("Skeleton Binary Matrix:")
        print(skeleton_binary)

        # Save as text if the matrix is too large
        with open('skeleton_binary_debug.txt', 'w') as f:
            np.savetxt(f, skeleton_binary, fmt='%d')
            print("Binary skeleton matrix saved as 'skeleton_binary_debug.txt'")

    # Define kernel for endpoint detection
    kernel = np.array([[1, 1, 1],
                       [1, 5, 1],
                       [1, 5, 1]])

    # Convolve to find endpoints
    convolved = convolve(skeleton_binary, kernel)

    if debug:
        print("Convolved Matrix:")
        print(convolved)
        # Save the convolved matrix as text for debugging
        with open('convolved_matrix_debug.txt', 'w') as f:
            np.savetxt(f, convolved, fmt='%d')
            print("Convolved matrix saved as 'convolved_matrix_debug.txt'")

    # Endpoints are where the result equals 20
    fingertip_coords = np.argwhere(convolved == 11)
    print(np.unique(convolved))

    # Convert to (x, y) format for OpenCV compatibility
    fingertips = [(y, x) for x, y in fingertip_coords]

    if not fingertips:
        return None

    # Find the highest fingertip (min y value)
    highest_fingertip = min(fingertips, key=lambda p: p[1])  # (x, y) -> compare by y (height)

    return highest_fingertip