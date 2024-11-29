import cv2
import numpy as np

# Open the default camera (index 0)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Define skin color range in YCrCb color space
skin_ycrcb_mint = np.array((0, 133, 77))
skin_ycrcb_maxt = np.array((255, 173, 127))

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert the frame to YCrCb color space
    frame_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    
    # Create a binary mask where skin color is detected
    skin_mask = cv2.inRange(frame_ycrcb, skin_ycrcb_mint, skin_ycrcb_maxt)

    # Find contours in the skin mask
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the original frame if the contour area is large enough
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > 1000:  # Only consider large contours
            cv2.drawContours(frame, contours, i, (255, 0, 0), 3)

    # Display the original frame with contours
    cv2.imshow('Skin Detection with Contours', frame)
    
    # Display the skin mask (for debugging purposes)
    cv2.imshow('Skin Mask', skin_mask)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
