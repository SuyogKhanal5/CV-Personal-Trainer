import cv2
import numpy as np
import time
import PoseEstimationModule as pm

cap = cv2.VideoCapture("curl.mp4")

wCam, hCam = 640, 480

cap.set(3, wCam)
cap.set(4, hCam)

currTime = 0
prevTime = 0

detector = pm.PoseEstimator()

while True:
    success, img = cap.read()

    img = cv2.resize(img, (1280,720))

    img = detector.findPose(img)
    lmList = detector.findPosition(img, False)

    if lmList != []:
        # Find left arm angle
        detector.findAngle(img, 11, 13, 15)

        # Find right arm angle
        detector.findAngle(img, 12, 14, 16)

    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    cv2.putText(img, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.imshow("Image", img)

    cv2.waitKey(1)