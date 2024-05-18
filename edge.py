import cv2
import numpy as np
from PIL import Image

cap = cv2.VideoCapture(0)
cv2.namedWindow("EDGE IMAGE")

def nothing(x):
    pass

cv2.createTrackbar("CED", "EDGE IMAGE", 0, 255, nothing)
cv2.createTrackbar("DEC", "EDGE IMAGE", 0, 255, nothing)


while(True):

    ret, frame = cap.read()
    frame = cv2.resize(frame, (1000, 725))
    
    trackbar1 = cv2.getTrackbarPos("CED", "EDGE IMAGE")
    trackbar2 = cv2.getTrackbarPos("DEC", "EDGE IMAGE")    
    
    edge = cv2.Canny(frame, trackbar2, trackbar1)
    
    cv2.imshow("ORIGINAL IMAGE", frame)
    cv2.imshow("EDGE IMAGE", edge)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()