import cv2
import numpy as np
from skimage.morphology import skeletonize

def process_hand_image(frame):
    skin_ycrcb_mint = np.array((0, 133, 77))
    skin_ycrcb_maxt = np.array((255, 173, 127))

    # Convert the frame to YCrCb color space
    frame_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    
    # Create a binary mask where skin color is detected
    skin_mask = cv2.inRange(frame_ycrcb, skin_ycrcb_mint, skin_ycrcb_maxt)

    return skin_mask
def preprocess_hand_image(image):
        """
        Preprocess the input image to create a binary hand blob
        
        Args:
            image (np.ndarray): Input image
        
        Returns:
            np.ndarray: Binary hand blob
        """
        # Apply skin color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Typical skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create skin mask
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations to clean the mask
        kernel = np.ones((8,8), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        return skin_mask


def skeletonize_image(image):
    """Skeletonizes a binary image."""
    
    skeleton = skeletonize(image)

    skeleton_8bit = (skeleton * 255).astype(np.uint8)

    kernel = np.ones((10, 10), np.uint8)

    dilated = cv2.dilate(skeleton_8bit, kernel, iterations=1)

    kernel_2 = np.ones((8, 8), np.uint8)

    eroded = cv2.erode(dilated, kernel_2)

    #return dilated
    return eroded

cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture frame.")
        break

    skin_mask = process_hand_image(frame)

    kernel_erosion = np.ones((8, 8), np.uint8)

    skin_mask_eroded = cv2.erode(skin_mask, kernel_erosion, iterations=1)

    kernel_dialation = np.ones((30, 30), np.uint8)

    skin_mask_dilated = cv2.dilate(skin_mask_eroded, kernel_dialation, iterations=1)

    kernel_erosion_2 = np.ones((3, 3), np.uint8)

    skin_mask_eroded_2 = cv2.erode(skin_mask_dilated, kernel_erosion_2, iterations=1)

    skeleton = skeletonize_image(skin_mask_eroded_2)

    cv2.imshow("skeleton", skeleton)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()