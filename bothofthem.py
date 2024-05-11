import cv2
import numpy as np

image = cv2.imread("/home/suleymann/Pictures/KGG.jpeg")
resize_image = cv2.resize(image, (512, 512))
cv2.namedWindow("MASKED IMAGE")

def nothing(x):
    pass

blur = 0
dilation = 0
erosion = 0

cv2.createTrackbar("BLUR", "MASKED IMAGE", 0, 255, nothing)
cv2.createTrackbar("DILATION", "MASKED IMAGE", 1, 255, nothing)
cv2.createTrackbar("EROSION", "MASKED IMAGE", 1, 255, nothing)


while(1):
    

    blur = cv2.getTrackbarPos("BLUR", "MASKED IMAGE")
    erosion = cv2.getTrackbarPos("EROSION", "MASKED IMAGE")
    dilation = cv2.getTrackbarPos("DILATION", "MASKED IMAGE")
    
    blur = max(1, blur)  
    kernel = np.ones((blur, blur), np.uint8)
    blur_image = cv2.blur(resize_image, (blur, blur))

    kernel = np.ones((erosion, erosion), np.uint8)
    erosion_image = cv2.erode(blur_image, kernel, iterations=1)

    kernel = np.ones((dilation, dilation), np.uint8)
    image_dilated = cv2.dilate(erosion_image, kernel, iterations=1)
    
    cv2.imshow("MASKED IMAGE", image_dilated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    

cv2.destroyAllWindows()
