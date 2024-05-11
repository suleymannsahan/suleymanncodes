import cv2
import numpy as np


image = cv2.imread("/home/suleymann/Pictures/KGG.jpeg")

resize_image = cv2.resize(image, (512, 512))
cv2.namedWindow("EROSION IMAGE")

def nothing(x):
    pass

cv2.createTrackbar("EROSION", "EROSION IMAGE", 1, 255, nothing)

cv2.imshow("IMAGE", resize_image)
while(1):
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
    erosion = cv2.getTrackbarPos("EROSION", "EROSION IMAGE")
    
    kernel = np.ones((erosion, erosion), np.uint8)
    erosion_image = cv2.erode(resize_image, kernel, iterations=1)
    print(erosion)

    cv2.imshow("EROSION IMAGE", erosion_image)

cv2.destroyAllWindows()