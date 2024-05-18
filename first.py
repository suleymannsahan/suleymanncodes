import cv2
import numpy as np
from PIL import Image
import math

a, b, c = 10, 5, 0


cap = cv2.VideoCapture(0)

while(1):
    ret, frame = cap.read()

    hsv_cap = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    
    lower_blue = np.array([75, 100, 100])
    upper_blue = np.array([130, 255, 255])

    blue_mask = cv2.inRange(hsv_cap, lower_blue, upper_blue)
    blue_pixels = cv2.bitwise_and(frame, frame, mask = blue_mask)

    mask_ = Image.fromarray(blue_mask)

    bbox = mask_.getbbox()

    if bbox is not None:
        x1, y1, x2, y2 = bbox

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
        cv2.circle(frame, (x1, y1), 5, (0, 0, 255), -1)
        cv2.circle(frame, (x2, y2), 5, (0, 0, 255), -1)

        rect1_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        rect2_center = (frame.shape[1] // 2, frame.shape[0] // 2)



        if (x2 - x1) > a:
            cv2.line(frame, rect1_center, rect2_center, (50, 255, 50), 5)
        elif b < (x2 - x1) <= a:
            cv2.line(frame, rect1_center, rect2_center, (100, 255, 100), 5)
        elif c < (x2 - x1) <= b:
            cv2.line(frame, rect1_center, rect2_center, (200, 255, 200), 5)


    cv2.imshow("camera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()