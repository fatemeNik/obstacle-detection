import cv2
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from object_detection import ObjectDetection
from playsound import playsound
from threading import Thread


print("SELECT MODE")
print("1-object detection")
print("2-obstacle detection")
mode = int(input())

# Point current frame
center_points_cur_frame = []

if mode == 1:
    od = ObjectDetection()
    cap = cv2.VideoCapture("test.mp4")

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (480,640), interpolation = cv2.INTER_AREA)
   
        region = frame[280:480,0:480]
        if not ret:
            break
        # Detect objects on frame
        (class_ids, scores, boxes) = od.detect(region)

        for box in boxes:
            (x, y, w, h) = box
            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)
            center_points_cur_frame.append((cx, cy))

            cv2.rectangle(region, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow("roi",region)
        cv2.imshow("frame",frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

elif mode == 2:
    
    cap = cv2.VideoCapture("walking.mp4")
    # cap = cv2.VideoCapture(0)
    object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
    areas = []
    while True:
         
        ret, frame = cap.read()
        region = frame[280:480,0:200]
        # region = frame[250:480,100:590]
  

       

       
        blur = cv2.bilateralFilter(region,20,100,100)
        median = cv2.medianBlur(blur,9)
        laplacian = cv2.Laplacian(median,cv2.CV_64F)
        mask = object_detector.apply(laplacian)
        countours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        for cnt in countours: 
            area =cv2.contourArea(cnt)
            if area > 100 :
                    print("unknown obstacle")
                    playsound("beep.wav")
                    x, y, w, h = cv2.boundingRect(cnt)
                    cx = int((x + x + w) / 2)
                    cy = int((y + y + h) / 2)
                    center_points_cur_frame.append((cx, cy))
                    cv2.circle(region,(cx,cy),5,(0,0,255),-1)
                    cv2.rectangle(region, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("roi",region)
        cv2.imshow("frame",frame)
        cv2.imshow("mask",mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




cap.release()
cv2.destroyAllWindows()
