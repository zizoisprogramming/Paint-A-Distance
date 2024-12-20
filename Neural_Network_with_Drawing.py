import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time

detector = HandDetector(maxHands=1, detectionCon=0.8)
video = cv2.VideoCapture(0)

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

x_canvas = -1
y_canvas = -1
currTime = -1
counter_pause = 0
img_draw = None

while True:
    _, img = video.read()
    img = cv2.flip(img, 1)
    original = img.copy()

    if img_draw is None:
            img_draw = np.zeros_like(img)

    hands = detector.findHands(img, draw=False)  
    if hands:  
        hand = hands[0]
        if hand:
            lmlist = hand[0]['lmList']
            fingerup = detector.fingersUp(hand[0])  
            finger_count = sum(fingerup)  
            real_tips = []

            if fingerup[0] == 1:  
                thumb_tip = lmlist[4]
                cv2.circle(img, (thumb_tip[0], thumb_tip[1]), 10, (0, 255, 0), cv2.FILLED)
                real_tips.append(thumb_tip)
            if fingerup[1] == 1:  
                index_tip = lmlist[8]
                cv2.circle(img, (index_tip[0], index_tip[1]), 10, (0, 255, 0), cv2.FILLED)
                real_tips.append(index_tip)
            if fingerup[2] == 1:  
                middle_tip = lmlist[12]
                cv2.circle(img, (middle_tip[0], middle_tip[1]), 10, (0, 255, 0), cv2.FILLED)
                real_tips.append(middle_tip)
            if fingerup[3] == 1:  
                ring_tip = lmlist[16]
                cv2.circle(img, (ring_tip[0], ring_tip[1]), 10, (0, 255, 0), cv2.FILLED)
                real_tips.append(ring_tip)
            if fingerup[4] == 1:  
                pinky_tip = lmlist[20]
                cv2.circle(img, (pinky_tip[0], pinky_tip[1]), 10, (0, 255, 0), cv2.FILLED)
                real_tips.append(pinky_tip)

            if (time.time() - currTime > 2):
                currTime = -1
            if finger_count != 2:
                counter_pause = 0

            if finger_count == 1 and fingerup[1] == 1:
                x , y = index_tip[0], index_tip[1]
                select_color_from_fingertip(index_tip[0], index_tip[1])
                if (currTime == -1):
                    cv2.circle(img_draw, (index_tip[0], index_tip[1]), 7, drawing_color, -1)
                    if x_canvas != -1 and y_canvas != -1 :
                        if (x_canvas - x) ** 2 + (y_canvas - y) ** 2 < 10000:
                            cv2.line(img_draw, (x, y), (x_canvas, y_canvas), drawing_color, 7)
                    x_canvas, y_canvas = x, y
            elif finger_count == 2:
                    if (counter_pause >= 5):
                        x_canvas, y_canvas = -1, -1
                        currTime = time.time()
                    else: 
                        counter_pause += 1
            if finger_count == 5:
                img_draw = np.zeros_like(img)
                x_canvas, y_canvas = -1, -1

            combined_frame = cv2.addWeighted(original, 0.7, img_draw, 0.3, 0)
            for (x1, y1, x2, y2), color in color_squares.items():
                    cv2.rectangle(combined_frame, (x1, y1), (x2, y2), color, -1)
                    cv2.rectangle(combined_frame, (x1, y1), (x2, y2), (0, 0, 0), 2)  

            cv2.imshow("Video with Drawing", combined_frame)
            
            cv2.putText(img, f'Fingers: {finger_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            

    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
