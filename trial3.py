import cv2
import numpy as np
import logging
from utils import *

def main():
    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam")
        return

    logger.info("Webcam opened successfully")

    # Create a blank canvas for drawing
    drawing_canvas = None

    try:
        while True:
            # Read a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from webcam")
                break

            # Initialize the drawing canvas if not already done
            if drawing_canvas is None:
                drawing_canvas = np.zeros_like(frame)

            # Process the frame
            try:
                # Process skin mask
                skin_mask = process(frame, thresh=0.5, debug=False)
                skin_mask = remove_face_region(frame,skin_mask)
                cleaned_mask = remove_noise(skin_mask)
                hand_mask = isolate_hand(cleaned_mask, debug=False)
                skeleton = skeletonize(hand_mask)
                finger_count = count_fingers(hand_mask)

                logger.info(f"Fingers detected: {finger_count}")

                # Draw or clear canvas based on finger count
                if finger_count == 1:  # Draw on the canvas
                    tips = detect_fingertips_gray(hand_mask, skeleton)
                    if tips:
                        x, y = tips
                        cv2.circle(drawing_canvas, (x, y), 5, (255, 0, 0), -1)

                elif finger_count == 5:  # Clear the canvas
                    drawing_canvas = np.zeros_like(frame)

                # Combine the drawing canvas with the original frame
                combined_frame = cv2.addWeighted(frame, 0.7, drawing_canvas, 0.3, 0)

                # Display the results
                cv2.imshow('Original + Drawing', combined_frame)
                cv2.imshow('Hand Skeleton', skeleton)
                cv2.imshow('Hand Mask', cleaned_mask)

            except Exception as e:
                logger.error(f"Error during processing: {e}")
                break

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Release the webcam and close windows
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Webcam released and windows closed")

if __name__ == "__main__":
    main()
