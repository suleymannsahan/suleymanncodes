import cv2
import numpy as np

image = np.ones((512, 512, 3), np.uint8)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.namedWindow("GRAY IMAGE")


while(1):
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
   
    cv2.imshow("GRAY IMAGE", gray_image)


cv2.destroyAllWindows()