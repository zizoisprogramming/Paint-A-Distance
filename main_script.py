import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, thin
from skimage.io import imshow
import time
from utils import *

def RGBtoYCbCr (R, G, B):
    R = int(R)
    G = int(G)
    B = int(B)
    R /= 255.0
    G /= 255.0
    B /= 255.0
    Y = 16 + (65.481 * R + 128.553 * G + 24.966 * B)
    Cb = 128 + (-37.797 * R - 74.203 * G + 112.0 * B)
    Cr = 128 + (112.0 * R - 93.786 * G - 18.214 * B)
    return Y, Cb, Cr

path = './Skin_NonSkin.txt'
content = ""
with open(path, 'r') as file:
    content = file.read()
entries = content.split('\n')
dataset = dict()
for line in entries:
    if line:
        R, G, B, label = line.split()
        label = int(label)
        if(label not in dataset):
            dataset[label] = []
        Y, Cb, Cr = RGBtoYCbCr(R, G, B)
        dataset[label].append([Cb, Cr])

def get_mean_cov(dataset):
    mean = dict()
    cov = dict()
    for label in dataset:
        data = np.array(dataset[label])
        mean[label] = np.mean(data, axis=0)
        cov[label] = np.cov(data, rowvar=False)
    return mean, cov
mean, cov = get_mean_cov(dataset)
skinMean = mean[1]
skinCov = cov[1]
nonSkinMean = mean[2]
nonSkinCov = cov[2]

def prob_c_label(C, mean, cov):
    C = np.array(C)
    mean = np.array(mean)
    cov = np.array(cov)
    
    C_diff = C - mean
    inv_cov = np.linalg.inv(cov)
    
    # Use log determinant to avoid overflow
    log_det = np.log(np.linalg.det(cov))
    
    # Compute in log space
    log_norm_factor = 0.5 * (log_det + C.shape[1] * np.log(2 * np.pi))
    
    # Rest of the computation remains similar
    x = np.einsum('ijk,kl,ijl->ij', C_diff, inv_cov, C_diff)
    
    # Compute log-probability first to avoid overflow
    log_prob = -0.5 * x - log_norm_factor
    
    prob = np.exp(log_prob)
    
    return prob

def prob_skin_c(C, skinMean, skinCov, nonSkinMean, nonSkinCov):
    probCskin = prob_c_label(C, skinMean, skinCov)
    probCnonSkin = prob_c_label(C, nonSkinMean, nonSkinCov)

    return probCskin / (probCskin + probCnonSkin)

max_face = None
max_area_face = 0
def appendFaceContour(frame, resizedSize, contour_areas, smallest_contours):
    original_height, original_width = frame.shape[:2]
    scale_x = resizedSize[0] / original_width
    scale_y = resizedSize[1] / original_height
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    global max_face
    global max_area_face
    if len(faces) > 0:
        max_area = 0
        # max_face = None
        for (x, y, w, h) in faces:
            area = w * h
            if area > max_area and (max_area_face == 0 or abs(area - max_area_face) < 0.3 * max_area_face):
                max_area = area
                max_face = (x, y, w, h)
                max_area_face = max_area
    if max_face:
        (x, y, w, h) = max_face
        face = frame.copy()
        cv2.rectangle(face, (x, y), (x + w, y + h), (0, 255, 0), 2)
        radius = min(w, h) // 2 
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.circle(face, (center_x, center_y), radius, (0, 255, 0), 2)
        cv2.imshow('face', face)
        x_resized = int(x * scale_x)
        y_resized = int(y * scale_y)
        w_resized = int(w * scale_x)
        h_resized = int(h * scale_y)
        middle_x = x_resized + w_resized // 2
        middle_y = y_resized + h_resized // 2
        for _, contour in contour_areas[-min(2,len(contour_areas)):]: 
            temp = contour.reshape(-1, 2)
            min_x = min(temp[:, 0])
            max_x = max(temp[:, 0])
            min_y = min(temp[:, 1])
            max_y = max(temp[:, 1])
            print(f"middle point {[middle_x, middle_y]}")
            print(f"min,max {[min_x,max_x,min_y, max_y]}" )
            if (middle_x >= min_x and middle_x <= max_x) and (middle_y >= min_y and middle_y <= max_y):
                smallest_contours.append(contour)
                print("face shall be removed")
    elif len(contour_areas) > 1:
        smallest_contours.append(contour_areas[-2][1])

def removeContours(mask, frame, resizedSize):
    closedMask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((20, 5)), iterations=3)
    contours = cv2.findContours(closedMask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    contour_areas = [(cv2.contourArea(c), c) for c in contours]
    contour_areas.sort(key=lambda x: x[0]) 
    smallest_contours = [c[1] for c in contour_areas[:-min(2, len(contour_areas))]]
    ### can be removed
    if len(contour_areas) > 1 and contour_areas[-2][0] < contour_areas[-1][0] / 2:
        smallest_contours.append(contour_areas[-2][1])
    ### can be removed
    maybe_face = contour_areas[-min(2, len(contour_areas)):]
    
    appendFaceContour(frame, resizedSize, maybe_face, smallest_contours)
    if len(smallest_contours) > 0:
        cv2.drawContours(mask, smallest_contours, -1, 0, -1)
    image = mask.copy()
    for x in contours[-min(2,len(contours)):]:
        cv2.drawContours(image, [x], -1, (0, 0, 255), 2)
    cv2.imshow('contours', image)

def cleanMask(skinMask, frame, resizedSize):
    binaryMask = cv2.morphologyEx(skinMask, cv2.MORPH_CLOSE, np.ones((3, 3)), iterations=3)
    binaryMask = cv2.threshold(binaryMask, 0.005, 1, cv2.THRESH_BINARY)[1]
    binaryMask = cv2.morphologyEx(binaryMask, cv2.MORPH_CLOSE, np.ones((3, 3)), iterations=3)
    removeContours(binaryMask, frame, resizedSize)
    return binaryMask

def removeFaceMask(image, face_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    Y, X = np.ogrid[:image.shape[0], :image.shape[1]]  # Create grid of indices
    for (x, y, w, h) in faces:
        center_x, center_y = x + w // 2, y + h // 2  
        radius = 1.1 * (max(w, h) // 2 ) 
        
        dist_from_center = (X - center_x)**2 + (Y - center_y)**2
        circular_mask = dist_from_center <= radius**2
        
        mask[circular_mask] = 0
    
    return mask

drawing_color = (0, 0, 255)

color_squares = {
    (10, 10, 50, 50): (255, 0, 0), 
    (60, 10, 110, 50): (0, 255, 0), 
    (120, 10, 170, 50): (0, 0, 255), 
}

def select_color_from_fingertip(x, y):
    global drawing_color
    for (x1, y1, x2, y2), color in color_squares.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            drawing_color = color
            break
def main():
    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam")
        return

    logger.info("Webcam opened successfully")
    global drawing_color
    # Create a blank canvas for drawing
    drawing_canvas = None
    currTime = -1
    counter_pause = 0
    try:
        x_canvas, y_canvas = -1, -1
        while True:
            # Read a frame from the webcam
            ret, frame = cap.read()
            resizingSizeX = 512
            resizingSizeY = 512
            resizingSize = (resizingSizeX, resizingSizeY)
            if not ret:
                logger.error("Failed to read frame from webcam")
                break
            frame = cv2.flip(frame, 1)
            # Initialize the drawing canvas if not already done
            if drawing_canvas is None:
                drawing_canvas = np.zeros((resizingSizeX,resizingSizeY, 3), dtype=np.float64)

            # Process the frame
            try:
                # Process skin mask
                faceMask = np.ones((frame.shape[0], frame.shape[1]))
                # faceMask = removeFaceMask(frame, face_cascade)
                original = frame.copy()
                frame = np.multiply(frame, faceMask[:, :, np.newaxis])
                frame = frame.astype(np.uint8)
                frame = cv2.resize(frame, resizingSize)
                cv2.imshow('Original Frame', frame)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                YCC = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
                frame = frame.astype(np.float64) / 255
                C = YCC[:, :, 1:]

                skinMask = prob_skin_c(C, skinMean, skinCov, nonSkinMean, nonSkinCov)
                cv2.imshow('Skin Mask before cleaning wiz threshold',  cv2.threshold(skinMask, 0.75, 1, cv2.THRESH_BINARY)[1])

                hand_mask = cleanMask(skinMask, original, resizingSize)
                if hand_mask.dtype != np.uint8:
                    hand_mask = (hand_mask * 255).astype(np.uint8)
                skeleton = skeletonize(hand_mask)
                try:
                    finger_count = count_fingers(hand_mask)
                except:
                    finger_count = 0
                    print("Error in count_fingers")
                

                logger.info(f"Fingers detected: {finger_count}")

                # Draw or clear canvas based on finger count
                if (time.time() - currTime > 2):
                    currTime = -1
                if finger_count != 2:
                    counter_pause = 0
                if finger_count == 1 :  # Draw on the canvas
                    tips = detect_fingertips_gray(hand_mask, skeleton)
                    if tips:
                        x, y = tips
                        select_color_from_fingertip(x, y)
                        if (currTime == -1):
                            cv2.circle(drawing_canvas, (x, y), 7, drawing_color, -1)
                            if x_canvas != -1 and y_canvas != -1 :
                                if (x_canvas - x) ** 2 + (y_canvas - y) ** 2 < 10000:
                                    cv2.line(drawing_canvas, (x, y), (x_canvas, y_canvas), drawing_color, 7)
                            x_canvas, y_canvas = x, y
                elif finger_count == 2:
                    if (counter_pause >= 5):
                        x_canvas, y_canvas = -1, -1
                        currTime = time.time()
                    else: 
                        counter_pause += 1
                elif finger_count >= 4:  # Clear the canvas
                    drawing_canvas = np.zeros_like(frame)
                elif finger_count != 2:  
                    x_canvas, y_canvas = -1, -1
                # Combine the drawing canvas with the original frame
 
                combined_frame = cv2.addWeighted(frame, 0.7, drawing_canvas, 0.3, 0)
                combined_frame = (combined_frame * 255).astype(np.uint8)
                for (x1, y1, x2, y2), color in color_squares.items():
                    cv2.rectangle(combined_frame, (x1, y1), (x2, y2), color, -1)
                    cv2.rectangle(combined_frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)
                # Display the results
                cv2.imshow('Original + Drawing', combined_frame)
                cv2.imshow('Hand Skeleton', skeleton)
                cv2.imshow('Hand Mask', hand_mask)

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
