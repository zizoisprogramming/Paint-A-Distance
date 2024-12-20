import cv2
from cvzone.HandTrackingModule import HandDetector

detector = HandDetector(maxHands=1, detectionCon=0.8)
video = cv2.VideoCapture(0)

while True:
    _, img = video.read()
    img = cv2.flip(img, 1)  

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
            
            cv2.putText(img, f'Fingers: {finger_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
