import cv2
import numpy as np

image = cv2.imread("/home/suleymann/Downloads/IMG_4630.jpg")
resize_image = cv2.resize(image, (512, 512))

hsv_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2HSV)

lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])

blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

blue_pixels = cv2.bitwise_and(resize_image, resize_image, mask=blue_mask)

while True:
    cv2.imshow("Original Image", resize_image)
    cv2.imshow("Blue Pixels", blue_pixels)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
