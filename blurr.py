import cv2
import numpy as np

image = cv2.imread("/home/suleymann/Pictures/KGG.jpeg")
resize_image = cv2.resize(image, (512,512))

cv2.namedWindow("RESIZE IMAGE")

def nothing(x):
    pass


blur = 0
cv2.createTrackbar("BLUR", "RESIZE IMAGE", blur, 255, nothing)

while(1):
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
   
    blur = cv2.getTrackbarPos("BLUR", "RESIZE IMAGE")

    kernel_size = max(1, blur)  
    if kernel_size % 2 == 0:  
        kernel_size += 1
    ksize = (kernel_size, kernel_size)

    
    resized_blurred_image = cv2.GaussianBlur(resize_image, ksize, 0)
    
    cv2.imshow("RESIZE IMAGE", resized_blurred_image)

cv2.destroyAllWindows()



