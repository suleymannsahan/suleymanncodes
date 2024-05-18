import cv2
import numpy as np 
import math

cap = cv2.VideoCapture(0)
cv2.namedWindow("AUTO EDGE")

def nothing(x):
    pass

cv2.createTrackbar("CED", "AUTO EDGE", 0, 255, nothing)
cv2.createTrackbar("DEC", "AUTO EDGE", 0, 255, nothing)

while(1):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1025, 725))
    
    trackbar1 = cv2.getTrackbarPos("CED", "AUTO EDGE")
    trackbar2 = cv2.getTrackbarPos("DEC", "AUTO EDGE")
    
    v = np.median(frame)
    sigma = 200
    
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    
    edge = cv2.Canny(frame, lower, upper)

    cv2.imshow("ORIGINAL", frame)
    cv2.imshow("AUTO EDGE", edge)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()