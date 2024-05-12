import cv2

image = cv2.resize(cv2.imread("/home/suleymann/Downloads/IMG_4630.jpg"), (512, 512))

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_blue = (100, 50, 50)
upper_blue = (130, 255, 255)

blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

cv2.imshow("Blue Contours", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
