import cv2
import numpy as np
import time
import PoseEstimationModule as pm

cap = cv2.VideoCapture(0)

left = True

wCam, hCam = 640, 480

cap.set(3, wCam)
cap.set(4, hCam)

currTime = 0
prevTime = 0

detector = pm.PoseEstimator()
count = 0
direction = 0

while True:
    success, img = cap.read()

    img = cv2.resize(img, (1280,720))

    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)

    if lmList != []:
        # Find left arm angle

        if left:
            angle = detector.findAngle(img, 11, 13, 15, draw=True)
        else:
            # Find right arm angle
            angle = detector.findAngle(img, 12, 14, 16, draw=True)

        per = np.interp(angle, (210, 310), (0,100))
        
        # Check for the curls
        if per == 100:
            if direction == 0:
                count += 0.5
                direction = 1
        
        if per == 0:
            if direction == 1:
                count += 0.5
                direction = 0

        cv2.putText(img, f"Count: {str(count)}", (20, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.putText(img, f"Percentage: {int(per)}%", (20, 140), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    cv2.putText(img, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.imshow("Image", img)

    cv2.waitKey(1)