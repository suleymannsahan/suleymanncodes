import cv2 
import numpy as np

cap = cv2.VideoCapture(0)

lower_t = 50
upper_t = 150

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1025, 725))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray_frame,5)
    
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 20, param1=40, param2=40, minRadius=1, maxRadius=100)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        

    edge = cv2.Canny(frame, lower_t, upper_t)

    cv2.imshow("edge", edge)
    cv2.imshow("frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
