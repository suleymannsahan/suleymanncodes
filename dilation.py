import cv2
import numpy as np


image = cv2.imread("/home/suleymann/Pictures/KGG.jpeg")

resize_image = cv2.resize(image, (512, 512))
cv2.namedWindow("DILATION IMAGE")

def nothing(x):
    pass

cv2.createTrackbar("DILATION", "DILATION IMAGE", 1, 255, nothing)

cv2.imshow("IMAGE", resize_image)
while(1):
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
    dilation = cv2.getTrackbarPos("DILATION", "DILATION IMAGE")
    
    kernel = np.ones((dilation, dilation), np.uint8)
    dilation_image = cv2.dilate(resize_image, kernel, iterations=1)
    print(dilation)

    cv2.imshow("DILATION IMAGE", dilation_image)

cv2.destroyAllWindows()