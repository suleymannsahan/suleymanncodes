import cv2
import numpy as np

img = np.ones((512, 512), np.uint8)
image = cv2.imread("/home/suleymann/Pictures/KGG.jpeg")
resize_image = cv2.resize(image, (512, 512))

cv2.namedWindow("IMAGE")

def nothing(x):
    pass

cv2.createTrackbar("BELIRGINLIK", "IMAGE", 0, 255, nothing)

while(1):
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
    belirginlik = cv2.getTrackbarPos("BELIRGINLIK", "IMAGE")
    
    doluluk_orani = belirginlik / 255.0
    
    rectangle = cv2.rectangle(np.copy(resize_image), (65, 65), (427, 427), (255, 255, 255), -1)
    
    blended_image = cv2.addWeighted(resize_image, 1 - doluluk_orani, rectangle, doluluk_orani, 0)
    
    cv2.imshow("IMAGE", blended_image)

cv2.destroyAllWindows()
